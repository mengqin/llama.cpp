#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"
#include "llama-moq.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <clocale>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

struct results_perplexity {
    std::vector<llama_token> tokens;
    double                   ppl_value;
    std::vector<float>       logits;
    std::vector<float>       probs;
};

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

static inline int nearest_int(float fval) {
    //assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static double log_softmax(int n_vocab, const float * logits, uint16_t * log_prob, int tok) {
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale = (max_logit - min_logit)/65535.f;
    float * d = (float *)log_prob;
    d[0] = scale;
    d[1] = min_log_prob;
    log_prob += 4;
    if (scale) {
        const float inv_scale = 1/scale;
        for (int i = 0; i < n_vocab; ++i) {
            log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale*(logits[i] - min_logit)) : 0;
        }
    } else {
        std::memset(log_prob, 0, n_vocab*sizeof(uint16_t));
    }
    return max_logit + log_sum_exp - logits[tok];
}

static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history
) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + size_t(i)*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

static void process_logits(std::ostream& out, int n_vocab, const float * logits, const int * tokens, int n_token,
        std::vector<std::thread> & workers, std::vector<uint16_t> & log_probs, double & nll, double & nll2) {
    std::mutex mutex;
    const int nv = 2*((n_vocab + 1)/2) + 4;
    int counter = 0;
    auto compute = [&mutex, &counter, &log_probs, &nll, &nll2, n_vocab, logits, tokens, n_token, nv] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const double v = log_softmax(n_vocab, logits + size_t(i)*n_vocab, log_probs.data() + size_t(i)*nv, tokens[i+1]);
            local_nll += v;
            local_nll2 += v*v;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
    out.write((const char *)log_probs.data(), size_t(n_token)*nv*sizeof(uint16_t));
}

struct kl_divergence_result {
    double sum_nll          = 0.0;
    double sum_nll2         = 0.0;
    double sum_nll_base     = 0.0;
    double sum_nll_base2    = 0.0;
    double sum_nll_nll_base = 0.0;
    double sum_kld          = 0.0;
    double sum_kld2         = 0.0;
    double sum_p_diff       = 0.0;
    double sum_p_diff2      = 0.0;
    double sum_p_diff4      = 0.0;
    float  max_p_diff       = 0.0f;
    size_t n_same_top       = 0.0;
    size_t count            = 0.0;
};

static std::pair<double, float> log_softmax(int n_vocab, const float * logits, const uint16_t * base_log_prob, int tok, kl_divergence_result & kld) {
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax = i;
        }
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float * d = (const float *)base_log_prob;
    const float scale = d[0];
    const float min_log_prob = d[1];
    base_log_prob += 4;

    const float nll = max_logit + log_sum_exp - logits[tok];
    kld.sum_nll  += nll;
    kld.sum_nll2 += nll*nll;

    const float nll_base = -(scale*base_log_prob[tok] + min_log_prob);
    kld.sum_nll_base  += nll_base;
    kld.sum_nll_base2 += nll_base*nll_base;

    kld.sum_nll_nll_base += nll*nll_base;

    max_logit += log_sum_exp;
    double sum = 0;
    int imax_base = -1;
    float p_log_base_max = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale*base_log_prob[i] + min_log_prob;
        if (i == 0 || p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            imax_base = i;
        }
        if (p_log_base > -16.f) {
            const float p_base = expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    kld.sum_kld  += sum;
    kld.sum_kld2 += sum*sum;
    ++kld.count;
    if (imax == imax_base) {
        ++kld.n_same_top;
    }

    const float p_base = expf(-nll_base);
    const float p = expf(-nll);
    const float p_diff = p - p_base;
    kld.sum_p_diff  += p_diff;
    const double p_diff2 = p_diff*p_diff;
    kld.sum_p_diff2 += p_diff2;
    kld.sum_p_diff4 += p_diff2*p_diff2;
    kld.max_p_diff = std::max(kld.max_p_diff, std::fabs(p_diff));

    return std::make_pair(sum, p_diff);
}

static void moq_kld_accumulate(kl_divergence_result & dst, const kl_divergence_result & src) {
    dst.sum_nll          += src.sum_nll;
    dst.sum_nll2         += src.sum_nll2;
    dst.sum_nll_base     += src.sum_nll_base;
    dst.sum_nll_base2    += src.sum_nll_base2;
    dst.sum_nll_nll_base += src.sum_nll_nll_base;
    dst.sum_kld          += src.sum_kld;
    dst.sum_kld2         += src.sum_kld2;
    dst.sum_p_diff       += src.sum_p_diff;
    dst.sum_p_diff2      += src.sum_p_diff2;
    dst.sum_p_diff4      += src.sum_p_diff4;
    dst.n_same_top       += src.n_same_top;
    dst.max_p_diff        = std::max(dst.max_p_diff, src.max_p_diff);
    dst.count            += src.count;
}

static void process_logits_range(int n_vocab, const float * logits, const int * tokens, int begin, int end,
        int nv, const uint16_t * base_log_probs, kl_divergence_result & kld,
        float * kld_values, float * p_diff_values) {
    for (int i = begin; i < end; ++i) {
        std::pair<double, float> v = log_softmax(n_vocab, logits + size_t(i)*n_vocab,
                base_log_probs + size_t(i)*nv, tokens[i+1], kld);
        kld_values[i]    = (float)v.first;
        p_diff_values[i] = v.second;
    }
}

static void process_logits(int n_vocab, const float * logits, const int * tokens, int n_token,
        std::vector<std::thread> & workers, const uint16_t * base_log_probs, kl_divergence_result & kld,
        float * kld_values, float * p_diff_values) {
    const int nv = 2*((n_vocab + 1)/2) + 4;
    const size_t n_workers = workers.size() + 1;
    std::vector<kl_divergence_result> local_results(n_workers);

    auto compute = [=, &local_results](size_t worker_index) {
        const int begin = (int) (worker_index * (size_t) n_token / n_workers);
        const int end   = (int) ((worker_index + 1) * (size_t) n_token / n_workers);
        kl_divergence_result & local_kld = local_results[worker_index];
        process_logits_range(n_vocab, logits, tokens, begin, end, nv, base_log_probs, local_kld, kld_values, p_diff_values);
    };
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread(compute, i + 1);
    }
    compute(0);
    for (auto & w : workers) {
        w.join();
    }
    for (const auto & local : local_results) {
        moq_kld_accumulate(kld, local);
    }
}

static results_perplexity perplexity_v2(llama_context * ctx, const common_params & params) {
    // Download: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    // Run `./perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    LOG_INF("%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens to evaluate perplexity with a context of %d\n",__func__,2*n_ctx,
                n_ctx);
        LOG_ERR("%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return {std::move(tokens), 0., {}, {}};
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    logit_history.resize(tokens.size());
    prob_history.resize(tokens.size());

    if (params.ppl_stride <= 0) {
        LOG_ERR("%s: stride is %d but must be greater than zero!\n",__func__,params.ppl_stride);
        return {tokens, -1, logit_history, prob_history};
    }

    const int calc_chunk = n_ctx;

    LOG_INF("%s: have %zu tokens. Calculation chunk = %d\n", __func__, tokens.size(), calc_chunk);

    if (int(tokens.size()) <= calc_chunk) {
        LOG_ERR("%s: there are only %zu tokens, this is not enough for a context size of %d and stride %d\n",__func__,
                tokens.size(), n_ctx, params.ppl_stride);
        return {tokens, -1, logit_history, prob_history};
    }

    const int n_chunk_max = (tokens.size() - calc_chunk + params.ppl_stride - 1)  / params.ppl_stride;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    int count = 0;
    double nll = 0.0;

    const int n_seq = std::max(1, n_batch / n_ctx);
    LOG_INF("%s: computing over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d\n", __func__, n_chunk, n_ctx, n_batch, n_seq);

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * params.ppl_stride;
        const int end   = start + calc_chunk;

        const int num_batches = (calc_chunk + n_batch - 1) / n_batch;
        //LOG_DBG("%s: evaluating %d...%d using %d batches\n", __func__, start, end, num_batches);

        std::vector<float> logits;

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(n_batch, 0, 1);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            common_batch_clear(batch);
            for (int i = 0; i < batch_size; i++) {
                common_batch_add(batch, tokens[batch_start + i], j*n_batch + i, {0}, true);
            }

            //LOG_DBG("    Batch %d: starts at %d, size is %d, n_past is %d\n",j,batch_start,batch_size,j * n_batch);
            if (llama_decode(ctx, batch)) {
                //LOG_ERR("%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return {tokens, -1, logit_history, prob_history};
            }

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (add_bos && j == 0) {
                tokens[batch_start] = llama_vocab_bos(vocab);
            }

            const auto * batch_logits = llama_get_logits(ctx);
            logits.insert(logits.end(), batch_logits, batch_logits + size_t(batch_size) * n_vocab);

            if (j == 0) {
                tokens[batch_start] = token_org;
            }
        }

        llama_batch_free(batch);

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        //LOG_DBG("%s: using tokens %d...%d\n",__func__,params.n_ctx - params.ppl_stride + start, params.n_ctx + start);
        for (int j = n_ctx - params.ppl_stride - 1; j < n_ctx - 1; ++j) {
            // Calculate probability of next token, given the previous ones.
            const std::vector<float> tok_logits(
                logits.begin() + size_t(j + 0) * n_vocab,
                logits.begin() + size_t(j + 1) * n_vocab);

            const float prob = softmax(tok_logits)[tokens[start + j + 1]];
            logit_history[start + j + 1] = tok_logits[tokens[start + j + 1]];
            prob_history[start + j + 1]  = prob;

            nll += -std::log(prob);
            ++count;
        }
        // perplexity is e^(average negative log-likelihood)
        if (params.ppl_output_type == 0) {
            LOG("[%d]%.4lf,", i + 1, std::exp(nll / count));
        } else {
            LOG("%8d  %.4lf\n", i*params.ppl_stride, std::exp(nll / count));
        }
    }
    LOG("\n");

    return {tokens, std::exp(nll / count), logit_history, prob_history};
}

static results_perplexity perplexity(llama_context * ctx, const common_params & params, const int32_t n_ctx) {
    if (params.ppl_stride > 0) {
        return perplexity_v2(ctx, params);
    }

    // Download: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    // Run `./llama-perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    std::ofstream logits_stream;
    if (!params.logits_file.empty()) {
        logits_stream.open(params.logits_file.c_str(), std::ios::binary);
        if (!logits_stream.is_open()) {
            LOG_ERR("%s: failed to open %s for writing\n", __func__, params.logits_file.c_str());
            return {};
        }
        LOG_INF("%s: saving all logits to %s\n", __func__, params.logits_file.c_str());
        logits_stream.write("_logits_", 8);
        logits_stream.write(reinterpret_cast<const char *>(&n_ctx), sizeof(n_ctx));
    }

    auto tim1 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

    auto tim2 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens to evaluate perplexity with a context of %d\n",__func__,2*n_ctx,
                n_ctx);
        LOG_ERR("%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return {std::move(tokens), 0., {}, {}};
    }

    std::vector<float> logit_history;
    logit_history.resize(tokens.size());

    std::vector<float> prob_history;
    prob_history.resize(tokens.size());

    const int n_chunk_max = tokens.size() / n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;
    const int n_seq = std::max(1, n_batch / n_ctx);

    GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0);
    GGML_ASSERT(params.n_ctx == n_seq * n_ctx);

    llama_batch batch = llama_batch_init(std::min(n_batch, n_ctx*n_seq), 0, 1);

    std::vector<float> logits;
    if (num_batches > 1) {
        logits.reserve(size_t(n_ctx) * n_vocab);
    }

    LOG_INF("%s: calculating perplexity over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d\n", __func__, n_chunk, n_ctx, n_batch, n_seq);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    std::vector<uint16_t> log_probs;
    if (!params.logits_file.empty()) {
        logits_stream.write((const char *)&n_vocab, sizeof(n_vocab));
        logits_stream.write((const char *)&n_chunk, sizeof(n_chunk));
        logits_stream.write((const char *)tokens.data(), n_chunk*n_ctx*sizeof(tokens[0]));
        const int nv = 2*((n_vocab + 1)/2) + 4;
        log_probs.resize(size_t(n_ctx) * nv);
    }

    // We get the logits for all the tokens in the context window (params.n_ctx)
    // from llama_decode below.  Now, based on https://huggingface.co/docs/transformers/perplexity,
    // calculate the perplexity over the last half of the window (so the model always has
    // some context to predict the token).
    //
    // We rely on the fact that attention in the forward pass only looks at previous
    // tokens here, so the logits returned for each token are an accurate representation
    // of what the model would have predicted at that point.
    //
    // Example, we have a context window of 512, we will compute perplexity for each of the
    // last 256 tokens.  Then, we split the input up into context window size chunks to
    // process the entire prompt.
    const int first = n_ctx/2;

    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk - i);

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            int n_outputs = 0;

            batch.n_tokens = 0;
            for (int seq = 0; seq < n_seq_batch; seq++) {
                int seq_start = batch_start + seq*n_ctx;

                // save original token and restore it after decode
                const auto token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }

                for (int k = 0; k < batch_size; ++k) {
                    const int idx = seq*n_ctx + k;
                    batch.token   [idx]    = tokens[seq_start + k];
                    batch.pos     [idx]    = j*n_batch + k;
                    batch.n_seq_id[idx]    = 1;
                    batch.seq_id  [idx][0] = seq;
                    batch.logits  [idx]    = batch.pos[idx] >= first ? 1 : 0;

                    n_outputs += batch.logits[idx] != 0;
                }
                batch.n_tokens += batch_size;

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (llama_decode(ctx, batch)) {
                LOG_INF("%s : failed to decode\n", __func__);
                return {tokens, -1, logit_history, prob_history};
            }

            if (num_batches > 1 && n_outputs > 0) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + size_t(n_outputs) * n_vocab);
            }
        }


        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total*n_chunk/n_seq);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        for (int seq = 0; seq < n_seq_batch; seq++) {
            const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx + first);

            llama_token * tokens_data = tokens.data() + start + seq*n_ctx + first;
            if (!params.logits_file.empty()) {
                process_logits(logits_stream, n_vocab, all_logits,
                        tokens_data, n_ctx - 1 - first,
                        workers, log_probs, nll, nll2);
            } else {
                process_logits(n_vocab, all_logits,
                        tokens_data, n_ctx - 1 - first,
                        workers, nll, nll2,
                        logit_history.data() + start + seq*n_ctx + first,
                        prob_history.data()  + start + seq*n_ctx + first);
            }
            count += n_ctx - first - 1;

            // perplexity is e^(average negative log-likelihood)
            if (params.ppl_output_type == 0) {
                LOG("[%d]%.4lf,", i + seq + 1, std::exp(nll / count));
            } else {
                double av = nll/count;
                double av2 = nll2/count - av*av;
                if (av2 > 0) {
                    av2 = sqrt(av2/(count-1));
                }
                LOG("%8d  %.4lf  %4lf  %4lf\n", i*n_ctx, std::exp(nll / count), av, av2);
            }
        }

        logits.clear();
    }
    LOG("\n");

    nll2 /= count;
    nll /= count;
    const double ppl = exp(nll);
    nll2 -= nll * nll;
    if (nll2 > 0) {
        nll2 = sqrt(nll2/(count-1));
        LOG_INF("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
    } else {
        LOG_ERR("Unexpected negative standard deviation of log(prob)\n");
    }

    llama_batch_free(batch);

    return {tokens, ppl, logit_history, prob_history};
}

static bool decode_helper(llama_context * ctx, llama_batch & batch, std::vector<float> & batch_logits, int n_batch, int n_vocab) {
    int prev_outputs = 0;
    for (int i = 0; i < (int) batch.n_tokens; i += n_batch) {
        const int n_tokens = std::min<int>(n_batch, batch.n_tokens - i);

        llama_batch batch_view = {
            n_tokens,
            batch.token    + i,
            nullptr,
            batch.pos      + i,
            batch.n_seq_id + i,
            batch.seq_id   + i,
            batch.logits   + i,
        };

        const int ret = llama_decode(ctx, batch_view);
        if (ret != 0) {
            LOG_ERR("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
            return false;
        }

        int n_outputs = 0;
        for (int i = 0; i < n_tokens; ++i) {
            n_outputs += batch_view.logits[i] != 0;
        }

        memcpy(batch_logits.data() + size_t(prev_outputs)*n_vocab, llama_get_logits(ctx), size_t(n_outputs)*n_vocab*sizeof(float));

        prev_outputs += n_outputs;
    }

    return true;
}

#define K_TOKEN_CHUNK 4

static void compute_logprobs(const float * batch_logits, int n_vocab, std::vector<std::thread>& workers,
        const std::vector<std::pair<size_t, llama_token>>& eval_pairs, std::vector<float>& eval_results) {
    if (eval_results.size() != eval_pairs.size()) {
        eval_results.resize(eval_pairs.size());
    }
    if (eval_pairs.empty()) {
        return;
    }

    size_t max_threads = std::min((eval_pairs.size() + K_TOKEN_CHUNK - 1)/K_TOKEN_CHUNK, workers.size());

    std::atomic<int> counter(0);
    auto compute = [&counter, &eval_pairs, &eval_results, batch_logits, n_vocab] () {
        float local_logprobs[K_TOKEN_CHUNK];
        while (true) {
            const size_t first = counter.fetch_add(K_TOKEN_CHUNK, std::memory_order_relaxed);
            if (first >= eval_results.size()) {
                break;
            }
            const size_t last = std::min(first + K_TOKEN_CHUNK, eval_results.size());
            for (size_t i = first; i < last; ++i) {
                const auto * logits = batch_logits + eval_pairs[i].first * n_vocab;
                float max_logit = logits[0];
                for (int j = 1; j < n_vocab; ++j) {
                    max_logit = std::max(max_logit, logits[j]);
                }
                float sum_p = 0.f;
                for (int j = 0; j < n_vocab; ++j) {
                    sum_p += expf(logits[j] - max_logit);
                }
                local_logprobs[i - first] = logits[eval_pairs[i].second] - max_logit - std::log(sum_p);
            }
            std::memcpy(eval_results.data() + first, local_logprobs, (last - first)*sizeof(float));
        }
    };

    for (size_t it = 0; it < max_threads; ++it) {
        workers[it] = std::thread(compute);
    }
    for (size_t it = 0; it < max_threads; ++it) {
        workers[it].join();
    }
}

static void hellaswag_score(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Calculates hellaswag score (acc_norm) from prompt
    //
    // Data extracted from the HellaSwag validation dataset (MIT license) https://github.com/rowanz/hellaswag/blob/master/data/hellaswag_val.jsonl
    // All used data fields are preprocessed as in https://github.com/EleutherAI/lm-evaluation-harness/blob/df3da98c5405deafd519c2ddca52bb7c3fe36bef/lm_eval/tasks/hellaswag.py#L62-L68
    //
    // All 10042 tasks should be extracted to keep the results standardized like other implementations.
    //
    // Datafile layout:
    // ['??'] denotes json fields
    // 6 lines per task:
    // ['activity_label'] + ": " +['ctx']  - The first part of the query, the context
    // ['label'] - The index the best common sense ending aka gold ending
    // ['endings'][0] - Endings added to the first part of the query
    // ['endings'][1]
    // ['endings'][2]
    // ['endings'][3]

    std::vector<std::string> prompt_lines;
    std::istringstream strstream(params.prompt);
    std::string line;

    while (std::getline(strstream,line,'\n')) {
        prompt_lines.push_back(line);
    }

    if (prompt_lines.size() % 6 != 0) {
        LOG_ERR("%s : number of lines in prompt not a multiple of 6.\n", __func__);
        return;
    }

    size_t hs_task_count = prompt_lines.size()/6;
    LOG_INF("%s : loaded %zu tasks from prompt.\n", __func__, hs_task_count);

    const bool is_spm = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM;
    LOG_INF("================================= is_spm = %d\n", is_spm);

    // The tasks should be randomized so the score stabilizes quickly.
    bool randomize_tasks = true;

    // Number of tasks to use when computing the score
    if (params.hellaswag_tasks < hs_task_count) {
        hs_task_count = params.hellaswag_tasks;
    }

    // The random seed should not impact the final result if the computation is done over enough tasks, so kept hardcoded for now
    std::mt19937 rng(1);

    // Dataholder for hellaswag tasks
    struct hs_data_t {
        std::string context;
        size_t gold_ending_idx;
        std::string ending[4];
        size_t ending_logprob_count[4];
        double ending_logprob[4];

        size_t i_logits;        // starting index of logits in the llama_batch
        size_t common_prefix;   // max number of initial tokens that are the same in all sentences
        size_t required_tokens; // needed number of tokens to evaluate all 4 endings
        std::vector<llama_token> seq_tokens[4];
    };

    LOG_INF("%s : selecting %zu %s tasks.\n", __func__, hs_task_count, (randomize_tasks?"randomized":"the first")  );

    // Select and read data from prompt lines
    std::vector<hs_data_t> hs_data(hs_task_count);
    for (size_t i = 0; i < hs_task_count; i++) {
        size_t idx = i;

        auto & hs_cur = hs_data[i];

        // Select a random example of those left in the prompt
        if (randomize_tasks) {
            std::uniform_int_distribution<size_t> dist(0, prompt_lines.size()/6-1 ) ;
            idx = dist(rng);
        }

        hs_cur.context = prompt_lines[idx*6];
        hs_cur.gold_ending_idx = std::stoi( prompt_lines[idx*6+1] );
        for (size_t j = 0; j < 4; j++) {
            hs_cur.ending[j] = prompt_lines[idx*6+2+j];
            hs_cur.seq_tokens[j] = common_tokenize(ctx, hs_cur.context + " " + hs_cur.ending[j], true);
        }

        // determine the common prefix of the endings
        hs_cur.common_prefix = 0;
        for (size_t k = 0; k < hs_cur.seq_tokens[0].size(); k++) {
            if (hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[1][k] ||
                hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[2][k] ||
                hs_cur.seq_tokens[0][k] != hs_cur.seq_tokens[3][k]) {
                break;
            }
            hs_cur.common_prefix++;
        }
        hs_cur.required_tokens = hs_cur.common_prefix +
            hs_cur.seq_tokens[0].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[1].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[2].size() - hs_cur.common_prefix +
            hs_cur.seq_tokens[3].size() - hs_cur.common_prefix;

        //GGML_ASSERT(hs_cur.common_prefix >= ::llama_tokenize(ctx, hs_cur.context, true).size());

        // Delete the selected random example from the prompt
        if (randomize_tasks) {
            prompt_lines.erase( std::next(prompt_lines.begin(),idx*6)  , std::next(prompt_lines.begin(),idx*6+6) );
        }
    }

    LOG_INF("%s : calculating hellaswag score over selected tasks.\n", __func__);

    LOG("\ntask\tacc_norm\t95%% confidence interval\n");

    double acc = 0.0f;

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int max_tasks_per_batch = 32;
    const int max_seq = std::min(4*max_tasks_per_batch, (int) llama_n_seq_max(ctx));

    llama_batch batch = llama_batch_init(n_ctx, 0, 4);

    std::vector<float> tok_logits(n_vocab);
    // TODO: this could be made smaller; it's currently the worst-case size
    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());

    for (size_t i0 = 0; i0 < hs_task_count; i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_logits = 0; // this tells us how many logits were needed before this point in the batch

        common_batch_clear(batch);

        // batch as much tasks as possible into the available context
        // each task has 4 unique sequence ids - one for each ending
        // the common prefix is shared among the 4 sequences to save tokens
        // we extract logits only from the last common token and from all ending tokens of each sequence
        while (n_cur + (int) hs_data[i1].required_tokens <= n_ctx) {
            auto & hs_cur = hs_data[i1];
            int n_logits = 0;

            const int s0 = 4*(i1 - i0);
            if (s0 + 4 > max_seq) {
                break;
            }

            for (size_t i = 0; i < hs_cur.common_prefix; ++i) {
                common_batch_add(batch, hs_cur.seq_tokens[0][i], i, { s0 + 0, s0 + 1, s0 + 2, s0 + 3 }, false);
            }
            batch.logits[batch.n_tokens - 1] = true; // we need logits for the last token of the common prefix
            n_logits += 1;

            for (int s = 0; s < 4; ++s) {
                const size_t seq_tokens_size = hs_cur.seq_tokens[s].size();
                // TODO: don't evaluate the last token of each sequence
                for (size_t i = hs_cur.common_prefix; i < seq_tokens_size; ++i) {
                    const bool needs_logits = i < seq_tokens_size - 1;
                    common_batch_add(batch, hs_cur.seq_tokens[s][i], i, { s0 + s }, needs_logits);
                    n_logits += needs_logits;
                }
            }

            hs_cur.i_logits = i_logits;
            i_logits += n_logits;

            n_cur += hs_data[i1].required_tokens;
            if (++i1 == hs_task_count) {
                break;
            }
        }

        if (i0 == i1) {
            LOG_ERR("%s : task %zu does not fit in the context window (requires %zu tokens)\n", __func__, i0, hs_data[i0].required_tokens);
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return;
        }

        // Compute log-probs in parallel
        // First we collect all tasks
        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto & hs_cur = hs_data[i];
            size_t li = 1; // skip the last logit of the common prefix (computed separately below)
            for (int s = 0; s < 4; ++s) {
                for (size_t j = hs_cur.common_prefix; j < hs_cur.seq_tokens[s].size() - 1; j++) {
                    eval_pairs.emplace_back(hs_cur.i_logits + li++, hs_cur.seq_tokens[s][j + 1]);
                }
            }
        }
        // Then we do the actual calculation
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;

        // compute the logprobs for each ending of the decoded tasks
        for (size_t i = i0; i < i1; ++i) {
            auto & hs_cur = hs_data[i];

            // get the logits of the last token of the common prefix
            std::memcpy(tok_logits.data(), batch_logits.data() + hs_cur.i_logits*n_vocab, n_vocab*sizeof(float));

            const auto first_probs = softmax(tok_logits);

            for (int s = 0; s < 4; ++s) {
                hs_cur.ending_logprob_count[s] = 1;
                hs_cur.ending_logprob[s] = std::log(first_probs[hs_cur.seq_tokens[s][hs_cur.common_prefix]]);
                for (size_t j = hs_cur.common_prefix; j < hs_cur.seq_tokens[s].size() - 1; j++) {
                    hs_cur.ending_logprob[s] += eval_results[ir++];
                    hs_cur.ending_logprob_count[s]++;
                }
                hs_cur.ending_logprob[s] /= hs_cur.ending_logprob_count[s];
            }

            // Find the ending with maximum logprob
            size_t ending_logprob_max_idx = 0;
            double ending_logprob_max_val = hs_cur.ending_logprob[0];
            for (size_t s = 1; s < 4; s++) {
                if (hs_cur.ending_logprob[s] > ending_logprob_max_val) {
                    ending_logprob_max_idx = s;
                    ending_logprob_max_val =  hs_cur.ending_logprob[s];
                }
            }

            //LOG("max logprob ending idx %lu, gold ending idx %lu\n", ending_logprob_max_idx, hs_cur.gold_ending_idx);

            // If the gold ending got the maximum logprobe add one accuracy point
            if (ending_logprob_max_idx == hs_cur.gold_ending_idx) {
                acc += 1.0;
            }

            double freq = acc / double(i + 1);

            const double za = 1.95996398454;

            // // Wald normal approx
            // double conf =za*sqrt(freq*(1-freq)/double(i + 1));
            // LOG("%zu\t%.8lf +/- %.8lf\n", i + 1, freq*100.0, conf*100.0);

            // Wilson score interval, more accurate
            double z   = za * za / double(i + 1);
            double cnf = z * sqrt(double(i + 1) * (4.0 * freq * (1 - freq) + z)) / (za + za);
            double a   = (freq + z * 0.5 - cnf) / (1.0 + z);
            double b   = (freq + z * 0.5 + cnf) / (1.0 + z);

            // Print the accumulated accuracy mean x 100 and confidence interval
            LOG("%zu\t%3.8lf%%\t[%3.4lf%%, %3.4lf%%]\n", i + 1, freq * 100.0, a * 100.0, b * 100.0);
        }

        i0 = i1 - 1;
    }

    llama_batch_free(batch);

    LOG("\n");
}

struct winogrande_entry {
    std::string first;
    std::string second;
    std::array<std::string, 2> choices;
    int answer;

    size_t i_logits;
    size_t common_prefix;
    size_t required_tokens;
    size_t n_base1; // number of tokens for context + choice 1
    size_t n_base2; // number of tokens for context + choice 2
    std::vector<llama_token> seq_tokens[2];
};

static std::vector<winogrande_entry> load_winogrande_from_csv(const std::string & prompt) {
    std::vector<winogrande_entry> result;
    std::istringstream in(prompt);
    std::string line;
    std::array<int, 4> comma_pos;
    while (true) {
        std::getline(in, line);
        if (in.fail() || in.eof()) break;
        int ipos = 0;
        bool quote_open = false;
        for (int i = 0; i < int(line.size()); ++i) {
            if (!quote_open) {
                if (line[i] == ',') {
                    comma_pos[ipos++] = i;
                    if (ipos == 4) break;
                }
                else if (line[i] == '"') {
                    quote_open = true;
                }
            }
            else {
                if (line[i] == '"') {
                    quote_open = false;
                }
            }
        }
        if (ipos != 4) {
            LOG_ERR("%s: failed to find comma separators in <%s>\n", __func__, line.c_str());
            continue;
        }
        auto sentence = line[comma_pos[0]+1] == '"' ? line.substr(comma_pos[0]+2, comma_pos[1] - comma_pos[0] - 3)
                                                    : line.substr(comma_pos[0]+1, comma_pos[1] - comma_pos[0] - 1);
        auto choice1 = line.substr(comma_pos[1]+1, comma_pos[2] - comma_pos[1] - 1);
        auto choice2 = line.substr(comma_pos[2]+1, comma_pos[3] - comma_pos[2] - 1);
        auto answer  = line.substr(comma_pos[3]+1, line.size() - comma_pos[3] - 1);
        auto index = line.substr(0, comma_pos[0]);
        int where = 0;
        for ( ; where < int(sentence.size()); ++where) {
            if (sentence[where] == '_') break;
        }
        if (where == int(sentence.size())) {
            LOG_ERR("%s: no _ in <%s>\n", __func__, sentence.c_str());
            continue;
        }
        std::istringstream stream(answer.c_str());
        int i_answer; stream >> i_answer;
        if (stream.fail() || i_answer < 1 || i_answer > 2) {
            LOG_ERR("%s: failed to parse answer <%s>\n", __func__, answer.c_str());
            continue;
        }
        result.emplace_back();
        auto& wg = result.back();
        wg.first = sentence.substr(0, where);
        wg.second = sentence.substr(where + 1, sentence.size() - where - 1);
        wg.choices[0] = std::move(choice1);
        wg.choices[1] = std::move(choice2);
        wg.answer = i_answer;
    }
    return result;
}

/*
 * Evaluates the Winogrande score.
 * Uses a CSV containing task index, dentence, choice 1, choice 2, answer (1 or 2)
 * You can get one such dataset from e.g. https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp
 * As an example, the 1st row in the above dataset is
 *
 *    0,Sarah was a much better surgeon than Maria so _ always got the easier cases.,Sarah,Maria,2
 *
 */
static void winogrande_score(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    constexpr int k_min_trailing_ctx = 3;

    auto data = load_winogrande_from_csv(params.prompt);
    if (data.empty()) {
        LOG_ERR("%s: no tasks\n", __func__);
        return;
    }

    LOG_INF("%s : loaded %zu tasks from prompt.\n", __func__, data.size());

    if (params.winogrande_tasks > 0 && params.winogrande_tasks < data.size()) {
        LOG_INF("%s : selecting %zu random tasks\n", __func__, params.winogrande_tasks);
        std::mt19937 rng(1);
        std::vector<int> aux(data.size());
        for (int i = 0; i < int(data.size()); ++i) {
            aux[i] = i;
        }
        float scale = 1/(1.f + (float)rng.max());
        std::vector<winogrande_entry> selected;
        selected.resize(params.winogrande_tasks);
        for (int i = 0; i < int(params.winogrande_tasks); ++i) {
            int j = int(scale*rng()*aux.size());
            selected[i] = std::move(data[aux[j]]);
            aux[j] = aux.back();
            aux.pop_back();
        }
        data = std::move(selected);
    }

    LOG_INF("%s : tokenizing selected tasks\n", __func__);

    for (auto & task : data) {
        task.seq_tokens[0] = common_tokenize(ctx, task.first + task.choices[0] + task.second, true);
        task.seq_tokens[1] = common_tokenize(ctx, task.first + task.choices[1] + task.second, true);

        task.common_prefix = 0;
        for (size_t k = 0; k < task.seq_tokens[0].size(); k++) {
            if (task.seq_tokens[0][k] != task.seq_tokens[1][k]) {
                break;
            }
            task.common_prefix++;
        }

        // TODO: the last token of each of the sequences don't need to be evaluated
        task.required_tokens = task.common_prefix +
            task.seq_tokens[0].size() - task.common_prefix +
            task.seq_tokens[1].size() - task.common_prefix;

        task.n_base1 = common_tokenize(ctx, task.first + task.choices[0], true).size();
        task.n_base2 = common_tokenize(ctx, task.first + task.choices[1], true).size();
    }

    LOG_INF("%s : calculating winogrande score over selected tasks.\n", __func__);

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int max_tasks_per_batch = 128;
    const int max_seq = std::min(2*max_tasks_per_batch, (int) llama_n_seq_max(ctx));

    llama_batch batch = llama_batch_init(n_ctx, 0, 2);

    std::vector<float> tok_logits(n_vocab);
    // TODO: this could be made smaller; it's currently the worst-case size
    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());

    int n_correct = 0;
    int n_done    = 0;

    for (size_t i0 = 0; i0 < data.size(); i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_logits = 0;

        common_batch_clear(batch);

        while (n_cur + (int) data[i1].required_tokens <= n_ctx) {
            int n_logits = 0;
            const int s0 = 2*(i1 - i0);
            if (s0 + 2 > max_seq) {
                break;
            }

            for (size_t i = 0; i < data[i1].common_prefix; ++i) {
                common_batch_add(batch, data[i1].seq_tokens[0][i], i, { s0 + 0, s0 + 1 }, false);
            }
            batch.logits[batch.n_tokens - 1] = true;
            n_logits += 1;

            for (int s = 0; s < 2; ++s) {
                // TODO: end before the last token, no need to predict past the end of the sequences
                for (size_t i = data[i1].common_prefix; i < data[i1].seq_tokens[s].size(); ++i) {
                    common_batch_add(batch, data[i1].seq_tokens[s][i], i, { s0 + s }, true);
                    n_logits += 1;
                }
            }

            data[i1].i_logits = i_logits;
            i_logits += n_logits;

            n_cur += data[i1].required_tokens;
            if (++i1 == data.size()) {
                break;
            }
        }

        if (i0 == i1) {
            LOG_ERR("%s : task %zu does not fit in the context window (requires %zu tokens)\n", __func__, i0, data[i0].required_tokens);
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return;
        }

        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto & task = data[i];

            const bool skip_choice =
                task.seq_tokens[0].size() - task.common_prefix > k_min_trailing_ctx &&
                task.seq_tokens[1].size() - task.common_prefix > k_min_trailing_ctx;

            const auto& n_base1 = skip_choice ? task.n_base1 : task.common_prefix;
            const int last_1st = task.seq_tokens[0].size() - n_base1 > 1 ? 1 : 0;
            size_t li = n_base1 - task.common_prefix;
            for (size_t j = n_base1-1; j < task.seq_tokens[0].size()-1-last_1st; ++j) {
                eval_pairs.emplace_back(task.i_logits + li++, task.seq_tokens[0][j+1]);
            }
            const auto& n_base2 = skip_choice ? task.n_base2 : task.common_prefix;
            const int last_2nd = task.seq_tokens[1].size() - n_base2 > 1 ? 1 : 0;
            // FIXME: this uses the wrong first logits when not skipping the choice word
            li = task.seq_tokens[0].size() - task.common_prefix + n_base2 - task.common_prefix;
            for (size_t j = n_base2-1; j < task.seq_tokens[1].size()-1-last_2nd; ++j) {
                eval_pairs.emplace_back(task.i_logits + li++, task.seq_tokens[1][j+1]);
            }
        }
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;
        for (size_t i = i0; i < i1; ++i) {
            auto & task = data[i];

            const bool skip_choice =
                task.seq_tokens[0].size() - task.common_prefix > k_min_trailing_ctx &&
                task.seq_tokens[1].size() - task.common_prefix > k_min_trailing_ctx;

            float score_1st = 0;
            const auto& n_base1 = skip_choice ? task.n_base1 : task.common_prefix;
            const int last_1st = task.seq_tokens[0].size() - n_base1 > 1 ? 1 : 0;
            for (size_t j = n_base1-1; j < task.seq_tokens[0].size()-1-last_1st; ++j) {
                score_1st += eval_results[ir++];
            }
            score_1st /= (task.seq_tokens[0].size() - n_base1 - last_1st);

            float score_2nd = 0;
            const auto& n_base2 = skip_choice ? task.n_base2 : task.common_prefix;
            const int last_2nd = task.seq_tokens[1].size() - n_base2 > 1 ? 1 : 0;
            for (size_t j = n_base2-1; j < task.seq_tokens[1].size()-1-last_2nd; ++j) {
                score_2nd += eval_results[ir++];
            }
            score_2nd /= (task.seq_tokens[1].size() - n_base2 - last_2nd);

            int result = score_1st > score_2nd ? 1 : 2;

            if (result == task.answer) {
                ++n_correct;
            }
            ++n_done;

            // print the accumulated accuracy mean x 100
            LOG("%zu\t%.4lf\t%10.6f  %10.6f  %d  %d\n", i+1, 100.0 * n_correct/n_done, score_1st, score_2nd, result, task.answer);
        }

        i0 = i1 - 1;
    }

    LOG("\n");

    if (n_done < 100) return;

    const float p = 1.f*n_correct/n_done;
    const float sigma = 100.f*sqrt(p*(1-p)/(n_done-1));

    LOG_INF("Final Winogrande score(%d tasks): %.4lf +/- %.4lf\n", n_done, 100*p, sigma);
}

static bool deserialize_string(std::istream & in, std::string & str) {
    uint32_t size;
    if (!in.read((char *)&size, sizeof(size)).fail()) {
        str.resize(size);
        if (!in.read((char *)&str[0], size).fail()) return true;
    }
    return false;
}

struct multiple_choice_answers {
    std::vector<std::string> answers;
    std::vector<int>         labels;
    bool deserialize(std::istream& in) {
        uint32_t n;
        in.read((char *)&n, sizeof(n));
        if (in.fail() || n > 100) return false; // 100 as max. number of answers should be good enough for any practical purpose
        answers.resize(n);
        labels.resize(n);
        for (auto& a : answers) {
            if (!deserialize_string(in, a)) return false;
        }
        in.read((char *)labels.data(), n*sizeof(int));
        return !in.fail();
    }
};

struct multiple_choice_task {
    std::string question;         // the question (or context that needs to be continued)
    multiple_choice_answers mc1;  // possible answers (continuations) with a single correct answer
    multiple_choice_answers mc2;  // possible answers (continuations) with multiple correct answers - not handled yet
    bool deserialize(std::istream& in) {
        if (!deserialize_string(in, question)) return false;
        return mc1.deserialize(in) && mc2.deserialize(in);
    }

    // For evaluation
    size_t i_logits;        // starting index of logits in the llama_batch
    size_t common_prefix;   // max number of initial tokens that are the same in all sentences
    size_t required_tokens; // needed number of tokens to evaluate all answers
    std::vector<std::vector<llama_token>> seq_tokens;
    std::vector<float> log_probs;
};

static bool multiple_choice_prepare_one_task(llama_context * ctx, multiple_choice_task& task, bool log_error) {
    if (task.question.empty() || task.mc1.answers.empty()) {
        if (log_error) {
            LOG_ERR("%s: found bad task with empty question and/or answers\n", __func__);
        }
        return false;
    }
    task.seq_tokens.reserve(task.mc1.answers.size());
    for (auto& answer : task.mc1.answers) {
        if (answer.empty()) {
            if (log_error) {
                LOG_ERR("%s: found empty answer\n", __func__);
            }
            return false;
        }
        task.seq_tokens.emplace_back(::common_tokenize(ctx, task.question + " " + answer, true));
    }
    auto min_len = task.seq_tokens.front().size();
    for (auto& seq : task.seq_tokens) {
        min_len = std::min(min_len, seq.size());
    }
    task.common_prefix = 0;
    for (size_t k = 0; k < min_len; ++k) {
        auto token = task.seq_tokens[0][k];
        bool all_same = true;
        for (size_t i = 1; i < task.seq_tokens.size(); ++i) {
            if (task.seq_tokens[i][k] != token) {
                all_same = false;
                break;
            }
        }
        if (!all_same) {
            break;
        }
        ++task.common_prefix;
    }
    task.required_tokens = task.common_prefix;
    for (auto& seq : task.seq_tokens) {
        task.required_tokens += seq.size() - task.common_prefix;
    }
    return true;
}

//
// Calculates score for multiple choice tasks with single correct answer from prompt.
// Commonly used LLM evaluation metrics of this type are
//   * ARC
//   * HellaSwag
//   * MMLU
//   * TruthfulQA
//
// Validation datasets for these 4 tests can be found at
//     https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp
// The data for these datasets was extracted from
//     git@hf.co:datasets/allenai/ai2_arc
//     https://github.com/rowanz/hellaswag/blob/master/data/hellaswag_val.jsonl
//     git@hf.co:datasets/Stevross/mmlu
//     https://huggingface.co/datasets/truthful_qa
//
static void multiple_choice_score(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::istringstream strstream(params.prompt);
    uint32_t n_task;
    strstream.read((char *)&n_task, sizeof(n_task));
    if (strstream.fail() || n_task == 0) {
        LOG_ERR("%s: no tasks\n", __func__);
        return;
    }
    LOG_INF("%s: there are %u tasks in prompt\n", __func__, n_task);
    std::vector<uint32_t> task_pos(n_task);
    strstream.read((char *)task_pos.data(), task_pos.size()*sizeof(uint32_t));
    if (strstream.fail()) {
        LOG_ERR("%s: failed to read task positions from prompt\n", __func__);
        return;
    }

    std::vector<multiple_choice_task> tasks;
    if (params.multiple_choice_tasks == 0 || params.multiple_choice_tasks >= (size_t)n_task) {
        // Use all tasks
        tasks.resize(n_task);
        LOG_INF("%s: reading tasks", __func__);
        int n_dot = std::max((int) n_task/100, 1);
        int i = 0;
        for (auto& task : tasks) {
            ++i;
            if (!task.deserialize(strstream)) {
                LOG_ERR("%s: failed to read task %d of %u\n", __func__, i, n_task);
                return;
            }
            if (i%n_dot == 0) LOG(".");
        }
        LOG("done\n");
    }
    else {
        LOG_INF("%s: selecting %zu random tasks from %u tasks available\n", __func__, params.multiple_choice_tasks, n_task);
        std::mt19937 rng(1);
        std::vector<int> aux(n_task);
        for (uint32_t i = 0; i < n_task; ++i) aux[i] = i;
        float scale = 1.f/(1.f + (float)std::mt19937::max());
        tasks.resize(params.multiple_choice_tasks);
        for (auto& task : tasks) {
            int j = (int)(scale * rng() * aux.size());
            int idx = aux[j];
            aux[j] = aux.back();
            aux.pop_back();
            strstream.seekg(task_pos[idx], std::ios::beg);
            if (!task.deserialize(strstream)) {
                LOG_ERR("%s: failed to read task %d at position %u\n", __func__, idx, task_pos[idx]);
                return;
            }
        }
        n_task = params.multiple_choice_tasks;
    }

    LOG_INF("%s: preparing task data", __func__);
    if (n_task > 500) {
        LOG("...");
        std::atomic<int> counter(0);
        std::atomic<int> n_bad(0);
        auto prepare = [&counter, &n_bad, &tasks, ctx] () {
            int num_tasks = tasks.size();
            int n_bad_local = 0;
            while (true) {
                int first = counter.fetch_add(K_TOKEN_CHUNK);
                if (first >= num_tasks) {
                    if (n_bad_local > 0) n_bad += n_bad_local;
                    break;
                }
                int last = std::min(first + K_TOKEN_CHUNK, num_tasks);
                for (int i = first; i < last; ++i) {
                    if (!multiple_choice_prepare_one_task(ctx, tasks[i], false)) ++n_bad_local;
                }
            }
        };
        size_t max_thread = std::thread::hardware_concurrency();
        max_thread = std::min(max_thread, (tasks.size() + K_TOKEN_CHUNK - 1)/K_TOKEN_CHUNK);
        std::vector<std::thread> workers(max_thread-1);
        for (auto& w : workers) w = std::thread(prepare);
        prepare();
        for (auto& w : workers) w.join();
        LOG("done\n");
        int nbad = n_bad;
        if (nbad > 0) {
            LOG_ERR("%s: found %d malformed tasks\n", __func__, nbad);
            return;
        }
    } else {
        int n_dot = std::max((int) n_task/100, 1);
        int i_task = 0;
        for (auto& task : tasks) {
            ++i_task;
            if (!multiple_choice_prepare_one_task(ctx, task, true)) {
                return;
            }
            if (i_task%n_dot == 0) {
                LOG(".");
            }
        }
        LOG("done\n");
    }

    LOG_INF("%s : calculating TruthfulQA score over %zu tasks.\n", __func__, tasks.size());

    LOG("\ntask\tacc_norm\n");

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int max_tasks_per_batch = 32;
    const int max_seq = std::min(4*max_tasks_per_batch, (int) llama_n_seq_max(ctx));

    llama_batch batch = llama_batch_init(n_ctx, 0, max_seq);

    std::vector<float> tok_logits(n_vocab);
    std::vector<float> batch_logits(size_t(n_ctx)*n_vocab);

    std::vector<std::pair<size_t, llama_token>> eval_pairs;
    std::vector<float> eval_results;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());
    std::vector<int> batch_indeces;

    int n_done = 0;
    int n_correct = 0;
    int n_tot_answers = 0;

    for (size_t i0 = 0; i0 < tasks.size(); i0++) {
        int n_cur = 0;

        size_t i1 = i0;
        size_t i_logits = 0; // this tells us how many logits were needed before this point in the batch

        common_batch_clear(batch);

        // batch as much tasks as possible into the available context
        // each task has 4 unique sequence ids - one for each ending
        // the common prefix is shared among the 4 sequences to save tokens
        // we extract logits only from the last common token and from all ending tokens of each sequence
        int s0 = 0;
        while (n_cur + (int) tasks[i1].required_tokens <= n_ctx) {
            auto& cur_task = tasks[i1];
            int n_logits = 0;

            int num_answers = cur_task.seq_tokens.size();
            if (s0 + num_answers > max_seq) {
                if (s0 == 0) {
                    LOG_ERR("%s : task %zu requires a higher -np|--parallel value (at least %d)\n", __func__, i0, num_answers);
                    return;
                }
                break;
            }

            if (int(batch_indeces.size()) != num_answers) {
                batch_indeces.resize(num_answers);
            }

            for (int s = 0; s < num_answers; ++s) {
                batch_indeces[s] = s0 + s;
            }

            for (size_t i = 0; i < cur_task.common_prefix; ++i) {
                //llama_batch_add(batch, cur_task.seq_tokens[0][i], i, { s0 + 0, s0 + 1, s0 + 2, s0 + 3}, false);
                common_batch_add(batch, cur_task.seq_tokens[0][i], i, batch_indeces, false);
            }
            batch.logits[batch.n_tokens - 1] = true; // we need logits for the last token of the common prefix
            n_logits += 1;

            for (int s = 0; s < int(cur_task.seq_tokens.size()); ++s) {
                const size_t seq_tokens_size = cur_task.seq_tokens[s].size();
                // TODO: don't evaluate the last token of each sequence
                for (size_t i = cur_task.common_prefix; i < seq_tokens_size; ++i) {
                    const bool needs_logits = i < seq_tokens_size - 1;
                    common_batch_add(batch, cur_task.seq_tokens[s][i], i, { s0 + s }, needs_logits);
                    n_logits += needs_logits;
                }
            }

            s0 += num_answers;

            cur_task.i_logits = i_logits;
            i_logits += n_logits;

            n_cur += cur_task.required_tokens;
            if (++i1 == tasks.size()) {
                break;
            }
        }

        if (i0 == i1) {
            LOG_ERR("%s : task %zu does not fit in the context window (requires %zu tokens)\n", __func__, i0, tasks[i0].required_tokens);
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        // decode all tasks [i0, i1)
        if (!decode_helper(ctx, batch, batch_logits, n_batch, n_vocab)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return;
        }

        // Compute log-probs in parallel
        // First we collect all tasks
        eval_pairs.clear();
        for (size_t i = i0; i < i1; ++i) {
            auto& cur_task = tasks[i];
            size_t li = 1; // skip the last logit of the common prefix (computed separately below)
            for (int s = 0; s < int(cur_task.seq_tokens.size()); ++s) {
                for (size_t j = cur_task.common_prefix; j < cur_task.seq_tokens[s].size() - 1; j++) {
                    eval_pairs.emplace_back(cur_task.i_logits + li++, cur_task.seq_tokens[s][j + 1]);
                }
            }
        }
        // Then we do the actual calculation
        compute_logprobs(batch_logits.data(), n_vocab, workers, eval_pairs, eval_results);

        size_t ir = 0;

        // compute the logprobs for each ending of the decoded tasks
        for (size_t i = i0; i < i1; ++i) {
            auto & cur_task = tasks[i];
            //LOG("==== Evaluating <%s> with correct answer ", cur_task.question.c_str());
            //for (int j = 0; j < int(cur_task.mc1.labels.size()); ++j) {
            //    if (cur_task.mc1.labels[j] == 1) {
            //        LOG("%d", j+1);
            //    }
            //}
            //LOG("\n    common_prefix: %zu\n", cur_task.common_prefix);

            // get the logits of the last token of the common prefix
            std::memcpy(tok_logits.data(), batch_logits.data() + cur_task.i_logits*n_vocab, n_vocab*sizeof(float));

            const auto first_probs = softmax(tok_logits);

            cur_task.log_probs.resize(cur_task.seq_tokens.size());
            for (int s = 0; s < int(cur_task.seq_tokens.size()); ++s) {
                size_t count = 1;
                float  log_prob  = std::log(first_probs[cur_task.seq_tokens[s][cur_task.common_prefix]]);
                for (size_t j = cur_task.common_prefix; j < cur_task.seq_tokens[s].size() - 1; j++) {
                    //LOG("        %zu  %g\n", ir, eval_results[ir]);
                    ++count;
                    log_prob += eval_results[ir++];
                }
                cur_task.log_probs[s] = log_prob / count;
                //LOG("        Final: %g\n", log_prob / count);
                //LOG("    <%s> : %g\n", cur_task.mc1.answers[s].c_str(), log_prob/count);
            }

            // Find the ending with maximum logprob
            size_t logprob_max_idx = 0;
            float  logprob_max_val = cur_task.log_probs[0];
            for (size_t s = 1; s < cur_task.log_probs.size(); s++) {
                if (cur_task.log_probs[s] > logprob_max_val) {
                    logprob_max_val = cur_task.log_probs[s];
                    logprob_max_idx = s;
                }
            }

            n_tot_answers += cur_task.log_probs.size();
            if (cur_task.mc1.labels[logprob_max_idx] == 1) {
                ++n_correct;
            }
            ++n_done;

            // Print the accumulated accuracy mean x 100
            LOG("%d\t%.8lf\n", n_done, 100.*n_correct/n_done);
        }

        i0 = i1 - 1;
    }

    llama_batch_free(batch);

    if (n_done < 100 && (params.multiple_choice_tasks != 0 && params.multiple_choice_tasks < (size_t)n_task)) return;

    float p = 1.f*n_correct/n_done;
    float sigma = sqrt(p*(1-p)/(n_done-1));
    LOG("\n");
    LOG_INF("Final result: %.4f +/- %.4f\n", 100.f*p, 100.f*sigma);
    p = 1.f*n_done/n_tot_answers;
    sigma = sqrt(p*(1-p)/(n_done-1));
    LOG_INF("Random chance: %.4f +/- %.4f\n", 100.f*p, 100.f*sigma);

    LOG_INF("\n");
}

struct moq_eval_result {
    bool ok = false;
    std::string error;

    int n_ctx = 0;
    int n_vocab = 0;
    int n_chunks = 0;
    size_t count = 0;

    double ppl = 0.0;
    double ppl_base = 0.0;
    double mean_kld = 0.0;
    double max_kld = 0.0;
    double p99_kld = 0.0;
    double p999_kld = 0.0;

    double eval_ms = 0.0;
    double base_read_ms = 0.0;
    double batch_build_ms = 0.0;
    double decode_ms = 0.0;
    double llama_synchronize_ms = 0.0;
    double logits_ms = 0.0;
    double logits_copy_ms = 0.0;
    double process_logits_ms = 0.0;
    double kld_queue_wait_ms = 0.0;
    double kld_worker_ms = 0.0;
    double kld_join_ms = 0.0;
    double kld_ring_wait_ms = 0.0;
    double sort_ms = 0.0;
    bool logits_ring_pinned_enabled = false;
    std::string logits_alloc_mode;
};

struct moq_base_logits_store {
    std::string path;
    std::string mode = "mmap";
    uint32_t n_ctx = 0;
    int n_vocab = 0;
    int n_chunk = 0;
    int nv = 0;
    int n_eval_per_chunk = 0;
    size_t file_size = 0;
    size_t log_probs_offset = 0;
    size_t log_probs_stride_bytes = 0;

    double open_ms = 0.0;
    double header_read_ms = 0.0;
    double tokens_read_ms = 0.0;
    double preload_ms = 0.0;
    double mmap_ms = 0.0;

    std::vector<llama_token> tokens;
    std::vector<uint8_t> preload_data;
    std::ifstream stream;

    const uint8_t * mapped_data = nullptr;
    bool mapped = false;

#if defined(_WIN32)
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle = nullptr;
#else
    int fd = -1;
#endif

    ~moq_base_logits_store() {
        close();
    }

    void close() {
        stream.close();
        preload_data.clear();
#if defined(_WIN32)
        if (mapped && mapped_data != nullptr) {
                UnmapViewOfFile(mapped_data);
        }
        if (mapping_handle != nullptr) {
            CloseHandle(mapping_handle);
        }
        if (file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle);
        }
        mapping_handle = nullptr;
        file_handle = INVALID_HANDLE_VALUE;
#else
        if (mapped && mapped_data != nullptr && file_size > 0) {
            munmap((void *) mapped_data, file_size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
        fd = -1;
#endif
        mapped_data = nullptr;
        mapped = false;
    }

    bool open(const std::string & p, const std::string & requested_mode, std::string & error) {
        close();
        path = p;
        mode = requested_mode;
        if (mode != "stream" && mode != "mmap" && mode != "preload") {
            error = "invalid base logits mode: " + mode;
            return false;
        }

        const int64_t t_open = llama_time_us();
        std::ifstream in(path.c_str(), std::ios::binary | std::ios::ate);
        if (!in) {
            error = "failed to open base logits file: " + path;
            return false;
        }
        file_size = (size_t) in.tellg();
        in.seekg(0, std::ios::beg);
        open_ms = (llama_time_us() - t_open) / 1000.0;

        const int64_t t_header = llama_time_us();
        char check[9];
        check[8] = 0;
        in.read(check, 8);
        if (in.fail() || strncmp("_logits_", check, 8) != 0) {
            error = path + " does not look like a log-probability file";
            return false;
        }
        in.read((char *) &n_ctx, sizeof(n_ctx));
        in.read((char *) &n_vocab, sizeof(n_vocab));
        in.read((char *) &n_chunk, sizeof(n_chunk));
        if (in.fail()) {
            error = "failed reading base logits header: " + path;
            return false;
        }
        if (n_ctx == 0 || n_vocab <= 0 || n_chunk <= 0) {
            error = "invalid base logits header: " + path;
            return false;
        }
        header_read_ms = (llama_time_us() - t_header) / 1000.0;

        const int first = (int) n_ctx / 2;
        n_eval_per_chunk = (int) n_ctx - 1 - first;
        nv = 2*((n_vocab + 1)/2) + 4;
        const size_t tokens_bytes = size_t(n_ctx) * size_t(n_chunk) * sizeof(llama_token);
        log_probs_offset = 8 + sizeof(n_ctx) + sizeof(n_vocab) + sizeof(n_chunk) + tokens_bytes;
        log_probs_stride_bytes = size_t(n_eval_per_chunk) * size_t(nv) * sizeof(uint16_t);
        const size_t expected_size = log_probs_offset + size_t(n_chunk) * log_probs_stride_bytes;
        if (expected_size > file_size) {
            error = string_format("base logits file is truncated: expected at least %zu bytes, got %zu", expected_size, file_size);
            return false;
        }

        const int64_t t_tokens = llama_time_us();
        tokens.resize(size_t(n_ctx) * size_t(n_chunk));
        if (in.read((char *) tokens.data(), tokens_bytes).fail()) {
            error = "failed reading evaluation tokens from " + path;
            return false;
        }
        tokens_read_ms = (llama_time_us() - t_tokens) / 1000.0;
        in.close();

        if (mode == "stream") {
            stream.open(path.c_str(), std::ios::binary);
            if (!stream) {
                error = "failed to open base logits stream: " + path;
                return false;
            }
            return true;
        }

        if (mode == "preload") {
            const int64_t t_preload = llama_time_us();
            std::ifstream pin(path.c_str(), std::ios::binary | std::ios::ate);
            if (!pin) {
                error = "failed to preload base logits file: " + path;
                return false;
            }
            const size_t sz = (size_t) pin.tellg();
            pin.seekg(0, std::ios::beg);
            preload_data.resize(sz);
            if (pin.read((char *) preload_data.data(), preload_data.size()).fail()) {
                error = "failed reading base logits file into memory: " + path;
                return false;
            }
            mapped_data = preload_data.data();
            preload_ms = (llama_time_us() - t_preload) / 1000.0;
            return true;
        }

        const int64_t t_mmap = llama_time_us();
#if defined(_WIN32)
        file_handle = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle == INVALID_HANDLE_VALUE) {
            error = "failed to open base logits file for mmap: " + path;
            return false;
        }
        LARGE_INTEGER li_size;
        if (!GetFileSizeEx(file_handle, &li_size) || li_size.QuadPart <= 0) {
            error = "failed to stat base logits file for mmap: " + path;
            return false;
        }
        file_size = (size_t) li_size.QuadPart;
        mapping_handle = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle == nullptr) {
            error = "failed to create base logits file mapping: " + path;
            return false;
        }
        mapped_data = (const uint8_t *) MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
        if (mapped_data == nullptr) {
            error = "failed to map base logits file: " + path;
            return false;
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            error = "failed to open base logits file for mmap: " + path;
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            error = "failed to stat base logits file for mmap: " + path;
            return false;
        }
        file_size = (size_t) st.st_size;
        mapped_data = (const uint8_t *) mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_data == MAP_FAILED) {
            mapped_data = nullptr;
            error = "failed to mmap base logits file: " + path;
            return false;
        }
#endif
        mapped = true;
        mmap_ms = (llama_time_us() - t_mmap) / 1000.0;
        return true;
    }

    bool get_log_probs(int chunk, std::vector<uint16_t> & scratch, const uint16_t *& ptr, double & read_ms, std::string & error) {
        if (chunk < 0 || chunk >= n_chunk) {
            error = string_format("base logits chunk %d out of range [0,%d)", chunk, n_chunk);
            return false;
        }

        const size_t offset = log_probs_offset + size_t(chunk) * log_probs_stride_bytes;
        const int64_t t_start = llama_time_us();
        if (mode == "stream") {
            scratch.resize(log_probs_stride_bytes / sizeof(uint16_t));
            stream.clear();
            stream.seekg((std::streamoff) offset, std::ios::beg);
            if (stream.read((char *) scratch.data(), log_probs_stride_bytes).fail()) {
                error = string_format("failed reading log-probs for chunk %d", chunk);
                return false;
            }
            ptr = scratch.data();
        } else {
            if (mapped_data == nullptr || offset + log_probs_stride_bytes > file_size) {
                error = string_format("base logits mapped/preloaded data missing for chunk %d", chunk);
                return false;
            }
            ptr = (const uint16_t *) (mapped_data + offset);
        }
        read_ms = (llama_time_us() - t_start) / 1000.0;
        return true;
    }
};

struct moq_eval_label {
    int candidate = -1;
    std::string group = "base";
    std::string qtype = "base";
    std::string mode = "";
};

struct moq_eval_profile_totals {
    int chunks = 0;
    size_t n_outputs = 0;
    double base_read_ms = 0.0;
    double batch_build_ms = 0.0;
    double decode_ms = 0.0;
    double logits_ms = 0.0;
    double logits_copy_ms = 0.0;
    double process_logits_ms = 0.0;
    double kld_queue_wait_ms = 0.0;
    double kld_worker_ms = 0.0;
    double kld_join_ms = 0.0;
    double kld_ring_wait_ms = 0.0;
    double sort_ms = 0.0;
    double total_ms = 0.0;
};

struct moq_eval_profile_candidate {
    moq_eval_label label;
    moq_eval_profile_totals totals;
    double ppl = 0.0;
    double mean_kld = 0.0;
    double p999_kld = 0.0;
    bool ok = false;
    bool logits_ring_pinned_enabled = false;
    std::string logits_alloc_mode;
    std::string error;
};

struct moq_eval_chunk_timing {
    moq_eval_label label;
    int chunk = 0;
    int n_outputs = 0;
    double base_read_ms = 0.0;
    double batch_build_ms = 0.0;
    double decode_ms = 0.0;
    double logits_ms = 0.0;
    double logits_copy_ms = 0.0;
    double process_logits_ms = 0.0;
    double kld_queue_wait_ms = 0.0;
    double kld_worker_ms = 0.0;
    double kld_join_ms = 0.0;
    double kld_ring_wait_ms = 0.0;
    double sort_ms = 0.0;
    double total_ms = 0.0;
    bool logits_ring_pinned_enabled = false;
    std::string logits_alloc_mode;
};

static void moq_profile_csv_escaped(std::ostream & out, const std::string & s) {
    const bool quote = s.find_first_of(",\"\n\r") != std::string::npos;
    if (!quote) {
        out << s;
        return;
    }
    out << '"';
    for (char c : s) {
        out << (c == '"' ? "\"\"" : std::string(1, c));
    }
    out << '"';
}

struct moq_eval_profiler {
    int level = 0;
    fs::path output_dir;
    bool chunk_header_written = false;
    std::ofstream chunk_out;
    std::ofstream chunk_overlap_out;
    std::ofstream chunk_ring_out;
    std::vector<moq_eval_profile_candidate> candidates;
    moq_eval_profile_totals totals;

    bool open(const fs::path & dir, int profile_level, std::string & error) {
        level = profile_level;
        output_dir = dir;
        if (level <= 0) {
            return true;
        }
        std::error_code ec;
        fs::create_directories(output_dir, ec);
        if (ec) {
            error = "failed to create MoQ profile output directory: " + output_dir.string();
            return false;
        }
        if (level >= 2) {
            chunk_out.open(output_dir / "chunk_timing.csv");
            if (!chunk_out) {
                error = "failed to open chunk_timing.csv for writing";
                return false;
            }
            chunk_overlap_out.open(output_dir / "chunk_timing_overlap.csv");
            if (!chunk_overlap_out) {
                error = "failed to open chunk_timing_overlap.csv for writing";
                return false;
            }
            chunk_ring_out.open(output_dir / "chunk_timing_ring.csv");
            if (!chunk_ring_out) {
                error = "failed to open chunk_timing_ring.csv for writing";
                return false;
            }
            const char * header = "candidate,group,qtype,mode,chunk,n_outputs,logits_ring_pinned_enabled,logits_alloc_mode,base_read_ms,batch_build_ms,decode_ms,logits_ms,logits_copy_ms,process_logits_ms,kld_queue_wait_ms,kld_worker_ms,kld_join_ms,kld_ring_wait_ms,sort_ms,total_ms\n";
            chunk_out << header;
            chunk_overlap_out << header;
            chunk_ring_out << header;
            chunk_header_written = true;
        }
        return true;
    }

    void add_chunk(const moq_eval_chunk_timing & c) {
        if (level <= 0) {
            return;
        }
        totals.chunks++;
        totals.n_outputs += c.n_outputs;
        totals.base_read_ms += c.base_read_ms;
        totals.batch_build_ms += c.batch_build_ms;
        totals.decode_ms += c.decode_ms;
        totals.logits_ms += c.logits_ms;
        totals.logits_copy_ms += c.logits_copy_ms;
        totals.process_logits_ms += c.process_logits_ms;
        totals.kld_queue_wait_ms += c.kld_queue_wait_ms;
        totals.kld_worker_ms += c.kld_worker_ms;
        totals.kld_join_ms += c.kld_join_ms;
        totals.kld_ring_wait_ms += c.kld_ring_wait_ms;

        if (level >= 2 && chunk_out) {
            auto write_row = [&](std::ostream & out) {
                out << c.label.candidate << ',';
                moq_profile_csv_escaped(out, c.label.group); out << ',';
                moq_profile_csv_escaped(out, c.label.qtype); out << ',';
                moq_profile_csv_escaped(out, c.label.mode); out << ',';
                out << c.chunk << ','
                    << c.n_outputs << ','
                    << (c.logits_ring_pinned_enabled ? "true" : "false") << ',';
                moq_profile_csv_escaped(out, c.logits_alloc_mode); out << ',';
                out
                    << c.base_read_ms << ','
                    << c.batch_build_ms << ','
                    << c.decode_ms << ','
                    << c.logits_ms << ','
                    << c.logits_copy_ms << ','
                    << c.process_logits_ms << ','
                    << c.kld_queue_wait_ms << ','
                    << c.kld_worker_ms << ','
                    << c.kld_join_ms << ','
                    << c.kld_ring_wait_ms << ','
                    << c.sort_ms << ','
                    << c.total_ms << '\n';
            };
            write_row(chunk_out);
            if (chunk_overlap_out) {
                write_row(chunk_overlap_out);
            }
            if (chunk_ring_out) {
                write_row(chunk_ring_out);
            }
        }
    }

    void add_candidate(const moq_eval_profile_candidate & c) {
        if (level <= 0) {
            return;
        }
        totals.sort_ms += c.totals.sort_ms;
        totals.kld_join_ms += c.totals.kld_join_ms;
        totals.kld_ring_wait_ms += c.totals.kld_ring_wait_ms;
        totals.total_ms += c.totals.total_ms;
        candidates.push_back(c);
    }

    void write_summary(const moq_base_logits_store & base_logits) {
        if (level <= 0) {
            return;
        }
        std::ofstream out(output_dir / "eval_profile_summary.txt");
        const int n_eval = std::max<int>(1, (int) candidates.size());
        const double total = std::max(1e-9, totals.total_ms);
        auto pct = [&](double v) { return 100.0 * v / total; };

        out << "MoQ eval profile summary\n\n";
        out << "Base logits: " << base_logits.path << "\n";
        out << "Base logits mode: " << base_logits.mode << "\n";
        out << "Base logits open_ms: " << base_logits.open_ms << "\n";
        out << "Base header read_ms: " << base_logits.header_read_ms << "\n";
        out << "Base tokens read_ms: " << base_logits.tokens_read_ms << "\n";
        out << "Base preload_ms: " << base_logits.preload_ms << "\n";
        out << "Base mmap_ms: " << base_logits.mmap_ms << "\n";
        out << "n_ctx: " << base_logits.n_ctx << "\n";
        out << "n_vocab: " << base_logits.n_vocab << "\n";
        out << "n_chunk_file: " << base_logits.n_chunk << "\n\n";

        out << "Evaluations: " << candidates.size() << "\n";
        out << "Chunks profiled: " << totals.chunks << "\n";
        out << "Outputs profiled: " << totals.n_outputs << "\n\n";

        out << "Average decode_ms: " << totals.decode_ms / n_eval << "\n";
        out << "Average base_read_ms: " << totals.base_read_ms / n_eval << "\n";
        out << "Average logits_ms: " << totals.logits_ms / n_eval << "\n";
        out << "Average logits_copy_ms: " << totals.logits_copy_ms / n_eval << "\n";
        out << "Average process_logits_ms: " << totals.process_logits_ms / n_eval << "\n";
        out << "Average kld_queue_wait_ms: " << totals.kld_queue_wait_ms / n_eval << "\n";
        out << "Average kld_worker_ms: " << totals.kld_worker_ms / n_eval << "\n";
        out << "Average kld_join_ms: " << totals.kld_join_ms / n_eval << "\n";
        out << "Average kld_ring_wait_ms: " << totals.kld_ring_wait_ms / n_eval << "\n";
        out << "Average sort_ms: " << totals.sort_ms / n_eval << "\n";
        out << "Average batch_build_ms: " << totals.batch_build_ms / n_eval << "\n";
        out << "Average total_eval_ms: " << totals.total_ms / n_eval << "\n\n";

        out << "decode percentage: " << pct(totals.decode_ms) << "\n";
        out << "base read percentage: " << pct(totals.base_read_ms) << "\n";
        out << "logits percentage: " << pct(totals.logits_ms) << "\n";
        out << "logits copy percentage: " << pct(totals.logits_copy_ms) << "\n";
        out << "process_logits percentage: " << pct(totals.process_logits_ms) << "\n";
        out << "kld queue wait percentage: " << pct(totals.kld_queue_wait_ms) << "\n";
        out << "kld worker percentage: " << pct(totals.kld_worker_ms) << "\n";
        out << "kld join percentage: " << pct(totals.kld_join_ms) << "\n";
        out << "kld ring wait percentage: " << pct(totals.kld_ring_wait_ms) << "\n";
        out << "sort percentage: " << pct(totals.sort_ms) << "\n";
        out << "batch build percentage: " << pct(totals.batch_build_ms) << "\n\n";

        out << "Candidate summaries:\n";
        out << "candidate,group,qtype,mode,ok,chunks,n_outputs,logits_ring_pinned_enabled,logits_alloc_mode,total_ms,decode_ms,base_read_ms,logits_ms,logits_copy_ms,process_logits_ms,kld_queue_wait_ms,kld_worker_ms,kld_join_ms,kld_ring_wait_ms,sort_ms,ppl,mean_kld,p999_kld,error\n";
        for (const auto & c : candidates) {
            out << c.label.candidate << ',';
            moq_profile_csv_escaped(out, c.label.group); out << ',';
            moq_profile_csv_escaped(out, c.label.qtype); out << ',';
            moq_profile_csv_escaped(out, c.label.mode); out << ',';
            out << (c.ok ? "true" : "false") << ','
                << c.totals.chunks << ','
                << c.totals.n_outputs << ','
                << (c.logits_ring_pinned_enabled ? "true" : "false") << ',';
            moq_profile_csv_escaped(out, c.logits_alloc_mode); out << ',';
            out
                << c.totals.total_ms << ','
                << c.totals.decode_ms << ','
                << c.totals.base_read_ms << ','
                << c.totals.logits_ms << ','
                << c.totals.logits_copy_ms << ','
                << c.totals.process_logits_ms << ','
                << c.totals.kld_queue_wait_ms << ','
                << c.totals.kld_worker_ms << ','
                << c.totals.kld_join_ms << ','
                << c.totals.kld_ring_wait_ms << ','
                << c.totals.sort_ms << ','
                << c.ppl << ','
                << c.mean_kld << ','
                << c.p999_kld << ',';
            moq_profile_csv_escaped(out, c.error);
            out << '\n';
        }
    }
};

enum class moq_logits_slot_state {
    free,
    decoding,
    ready_for_kld,
    processing_kld,
};

struct moq_logits_ring_slot {
    int id = -1;
    std::vector<float> logits_host;
    ggml_backend_buffer_ptr logits_backend;
    float * logits = nullptr;
    size_t logits_capacity = 0;
    bool pinned_enabled = false;
    std::string alloc_mode = "host";
    int chunk_id = -1;
    int n_outputs = 0;
    moq_logits_slot_state state = moq_logits_slot_state::free;
    std::atomic<int> pending_chunks{0};

    bool ensure_capacity(size_t capacity_floats, ggml_backend_buffer_type_t pinned_buft, bool request_pinned) {
        if (logits_capacity >= capacity_floats && logits != nullptr &&
                (!request_pinned || pinned_enabled || pinned_buft == nullptr)) {
            return true;
        }

        logits = nullptr;
        logits_capacity = 0;
        pinned_enabled = false;
        alloc_mode = "host";
        logits_backend.reset();
        logits_host.clear();

        if (request_pinned && pinned_buft != nullptr) {
            const size_t bytes = capacity_floats * sizeof(float);
            ggml_backend_buffer_t raw = ggml_backend_buft_alloc_buffer(pinned_buft, bytes);
            if (raw != nullptr) {
                logits_backend.reset(raw);
                logits = (float *) ggml_backend_buffer_get_base(logits_backend.get());
                logits_capacity = capacity_floats;
                ggml_backend_buffer_type_t actual_buft = ggml_backend_buffer_get_type(logits_backend.get());
                alloc_mode = ggml_backend_buft_name(actual_buft);
                pinned_enabled = actual_buft == pinned_buft;
                if (logits != nullptr) {
                    return true;
                }
                logits_backend.reset();
                logits_capacity = 0;
                alloc_mode = "host";
            }
        }

        try {
            logits_host.resize(capacity_floats);
        } catch (const std::bad_alloc &) {
            return false;
        }
        logits = logits_host.data();
        logits_capacity = logits_host.size();
        pinned_enabled = false;
        alloc_mode = "std_vector";
        return logits != nullptr;
    }
};

class moq_logits_ring_pool {
public:
    explicit moq_logits_ring_pool(int n_slots, bool request_pinned, ggml_backend_buffer_type_t pinned_buft) :
        request_pinned(request_pinned), pinned_buft(pinned_buft) {
        const int n = std::max(1, n_slots);
        slots.reserve(n);
        for (int i = 0; i < n; ++i) {
            auto slot = std::make_shared<moq_logits_ring_slot>();
            slot->id = i;
            slots.push_back(std::move(slot));
        }
    }

    std::shared_ptr<moq_logits_ring_slot> acquire(size_t capacity_floats, int chunk_id, int n_outputs, double & wait_ms) {
        const int64_t t0 = llama_time_us();
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() {
            return std::any_of(slots.begin(), slots.end(), [](const std::shared_ptr<moq_logits_ring_slot> & slot) {
                return slot->state == moq_logits_slot_state::free;
            });
        });

        auto it = std::find_if(slots.begin(), slots.end(), [](const std::shared_ptr<moq_logits_ring_slot> & slot) {
            return slot->state == moq_logits_slot_state::free;
        });
        GGML_ASSERT(it != slots.end());

        auto slot = *it;
        slot->state = moq_logits_slot_state::decoding;
        slot->chunk_id = chunk_id;
        slot->n_outputs = n_outputs;
        slot->pending_chunks.store(0);
        if (!slot->ensure_capacity(capacity_floats, pinned_buft, request_pinned)) {
            slot->state = moq_logits_slot_state::free;
            cv.notify_all();
            wait_ms = (llama_time_us() - t0) / 1000.0;
            return nullptr;
        }
        wait_ms = (llama_time_us() - t0) / 1000.0;
        return slot;
    }

    void mark_processing(const std::shared_ptr<moq_logits_ring_slot> & slot, int pending_chunks) {
        std::lock_guard<std::mutex> lock(mutex);
        slot->state = moq_logits_slot_state::processing_kld;
        slot->pending_chunks.store(std::max(0, pending_chunks));
        if (pending_chunks <= 0) {
            slot->state = moq_logits_slot_state::free;
            cv.notify_all();
        }
    }

    void release_chunk(const std::shared_ptr<moq_logits_ring_slot> & slot) {
        std::lock_guard<std::mutex> lock(mutex);
        if (slot->pending_chunks.fetch_sub(1) == 1) {
            slot->state = moq_logits_slot_state::free;
            slot->chunk_id = -1;
            slot->n_outputs = 0;
            cv.notify_all();
        }
    }

private:
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::shared_ptr<moq_logits_ring_slot>> slots;
    bool request_pinned = false;
    ggml_backend_buffer_type_t pinned_buft = nullptr;
};

struct moq_kld_chunk_state {
    moq_eval_chunk_timing timing;
    std::shared_ptr<float> logits_owner;
    const float * logits = nullptr;
    size_t logits_count = 0;
    std::shared_ptr<moq_logits_ring_slot> ring_slot;
    moq_logits_ring_pool * ring_pool = nullptr;
    std::vector<llama_token> tokens;
    const uint16_t * base_log_probs = nullptr;
    std::shared_ptr<std::vector<uint16_t>> base_log_probs_owner;
    float * kld_values = nullptr;
    float * p_diff_values = nullptr;
    int n_vocab = 0;
    int n_token = 0;
    int nv = 0;
    int n_ranges = 0;
    std::vector<kl_divergence_result> partials;
    std::vector<double> range_queue_wait_ms;
    std::vector<double> range_worker_ms;
    std::atomic<int> remaining{0};
    std::mutex timing_mutex;
    int64_t first_worker_start_us = 0;
    int64_t last_worker_end_us = 0;
};

struct moq_kld_range_job {
    std::shared_ptr<moq_kld_chunk_state> state;
    int range_index = 0;
    int begin = 0;
    int end = 0;
    int64_t submit_us = 0;
};

class moq_kld_overlap_executor {
public:
    moq_kld_overlap_executor(int n_workers, int ring_chunks) :
        ring_chunks(std::max(1, ring_chunks)) {
        const int n = std::max(1, n_workers);
        workers.reserve(n);
        for (int i = 0; i < n; ++i) {
            workers.emplace_back([this]() { worker_loop(); });
        }
    }

    ~moq_kld_overlap_executor() {
        shutdown();
    }

    double wait_for_slot() {
        const int64_t t0 = llama_time_us();
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait(lock, [&]() { return stopping || in_flight_chunks < ring_chunks; });
        return (llama_time_us() - t0) / 1000.0;
    }

    void submit_chunk(const std::shared_ptr<moq_kld_chunk_state> & state) {
        std::lock_guard<std::mutex> lock(mutex);
        in_flight_chunks++;
        const int64_t submit_us = llama_time_us();
        for (int ir = 0; ir < state->n_ranges; ++ir) {
            const int begin = (int) ((size_t) ir * (size_t) state->n_token / (size_t) state->n_ranges);
            const int end   = (int) ((size_t) (ir + 1) * (size_t) state->n_token / (size_t) state->n_ranges);
            jobs.push_back({state, ir, begin, end, submit_us});
        }
        cv_job.notify_all();
    }

    double wait_all() {
        const int64_t t0 = llama_time_us();
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait(lock, [&]() { return in_flight_chunks == 0 && jobs.empty() && active_jobs == 0; });
        return (llama_time_us() - t0) / 1000.0;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping) {
                return;
            }
            stopping = true;
        }
        cv_job.notify_all();
        for (auto & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::mutex mutex;
    std::condition_variable cv_job;
    std::condition_variable cv_done;
    std::deque<moq_kld_range_job> jobs;
    std::vector<std::thread> workers;
    int ring_chunks = 1;
    int in_flight_chunks = 0;
    int active_jobs = 0;
    bool stopping = false;

    void worker_loop() {
        while (true) {
            moq_kld_range_job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_job.wait(lock, [&]() { return stopping || !jobs.empty(); });
                if (stopping && jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
                active_jobs++;
            }

            const int64_t t_worker_start = llama_time_us();
            const double queue_wait_ms = (t_worker_start - job.submit_us) / 1000.0;
            kl_divergence_result local;
            process_logits_range(job.state->n_vocab, job.state->logits, job.state->tokens.data(),
                    job.begin, job.end, job.state->nv, job.state->base_log_probs, local,
                    job.state->kld_values, job.state->p_diff_values);
            const int64_t t_worker_end = llama_time_us();

            {
                std::lock_guard<std::mutex> state_lock(job.state->timing_mutex);
                job.state->partials[job.range_index] = local;
                job.state->range_queue_wait_ms[job.range_index] = queue_wait_ms;
                job.state->range_worker_ms[job.range_index] = (t_worker_end - t_worker_start) / 1000.0;
                if (job.state->first_worker_start_us == 0 || t_worker_start < job.state->first_worker_start_us) {
                    job.state->first_worker_start_us = t_worker_start;
                }
                job.state->last_worker_end_us = std::max(job.state->last_worker_end_us, t_worker_end);
            }

            const bool chunk_done = job.state->remaining.fetch_sub(1) == 1;
            moq_logits_ring_pool * ring_pool = nullptr;
            std::shared_ptr<moq_logits_ring_slot> ring_slot;
            {
                std::lock_guard<std::mutex> lock(mutex);
                active_jobs--;
                if (chunk_done) {
                    in_flight_chunks--;
                    ring_pool = job.state->ring_pool;
                    ring_slot = job.state->ring_slot;
                }
            }
            if (ring_pool != nullptr && ring_slot) {
                ring_pool->release_chunk(ring_slot);
            }
            cv_done.notify_all();
        }
    }
};

static int moq_kld_worker_count(const common_params & params) {
    if (params.moq_kld_workers > 0) {
        return params.moq_kld_workers;
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 1;
    }
    return (int) std::min<unsigned>(hw, 16);
}

static ggml_backend_buffer_type_t moq_select_logits_pinned_buft(std::string & desc) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (props.type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        if (!props.caps.host_buffer) {
            continue;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
        if (buft != nullptr) {
            desc = string_format("%s:%s", ggml_backend_dev_name(dev), ggml_backend_buft_name(buft));
            return buft;
        }
    }

    for (const auto type : {GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU, GGML_BACKEND_DEVICE_TYPE_ACCEL}) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
        if (dev != nullptr) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
            if (buft != nullptr) {
                desc = string_format("%s:%s", ggml_backend_dev_name(dev), ggml_backend_buft_name(buft));
                return buft;
            }
        }
    }

    desc = "none";
    return nullptr;
}

struct moq_logits_output_buffer_guard {
    llama_context * ctx = nullptr;
    bool active = false;

    ~moq_logits_output_buffer_guard() {
        if (active && ctx != nullptr) {
            llama_clear_logits_output_buffer(ctx);
        }
    }
};

static std::pair<double, double> moq_mean_and_uncertainty(double sum, double sum2, size_t count) {
    if (count < 1) {
        return std::make_pair(0.0, 0.0);
    }
    double f = sum/count;
    double df = sum2/count - f*f;
    df = df > 0 && count > 10 ? sqrt(df/(count-1)) : 0.0;
    return std::make_pair(f, df);
}

static double moq_percentile_sorted(const std::vector<float> & values, float fraction) {
    if (values.empty()) {
        return 0.0;
    }
    if (fraction <= 0.0f) {
        return values.front();
    }
    if (fraction >= 1.0f) {
        return values.back();
    }
    float p = fraction*(values.size() - 1);
    size_t ip = size_t(p);
    p -= ip;
    return (1.0f - p)*values[ip] + p*values[std::min(ip + 1, values.size() - 1)];
}

static moq_eval_result kl_divergence_eval_once(
        llama_context * ctx,
        const common_params & params,
        moq_base_logits_store & base_logits,
        int chunk_limit,
        bool log_progress,
        moq_eval_profiler * profiler = nullptr,
        const moq_eval_label & label = {}) {
    moq_eval_result result;
    const int64_t t_start_us = llama_time_us();
    moq_eval_profile_candidate candidate_profile;
    candidate_profile.label = label;

    auto fail = [&result](const std::string & msg) {
        result.error = msg;
        return result;
    };

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (base_logits.path.empty()) {
        return fail("missing base logits file");
    }
    uint32_t n_ctx = base_logits.n_ctx;
    if (n_ctx > llama_n_ctx(ctx)) {
        return fail(string_format("base logits n_ctx=%u exceeds current context n_ctx=%d", n_ctx, llama_n_ctx(ctx)));
    }
    int n_vocab = base_logits.n_vocab;
    int n_chunk_file = base_logits.n_chunk;
    if (n_vocab != llama_vocab_n_tokens(vocab)) {
        return fail(string_format("inconsistent vocabulary: logits=%d current=%d", n_vocab, llama_vocab_n_tokens(vocab)));
    }
    if (n_chunk_file <= 0) {
        return fail("base logits file contains no chunks");
    }

    const int n_chunk_eval = chunk_limit > 0 ? std::min(chunk_limit, n_chunk_file) : n_chunk_file;

    std::vector<llama_token> tokens = base_logits.tokens;

    const int n_batch = params.n_batch;
    const int num_batches = (static_cast<int>(n_ctx) + n_batch - 1) / n_batch;
    const int n_seq_max = llama_n_seq_max(ctx);
    int n_seq = std::max(1, n_batch / static_cast<int>(n_ctx));
    if (n_seq > n_seq_max) {
        n_seq = n_seq_max;
    }

    const int nv = 2*((n_vocab + 1)/2) + 4;
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    llama_batch batch = llama_batch_init(std::min(n_batch, static_cast<int>(n_ctx)*n_seq), 0, 1);

    const int first = n_ctx/2;
    const int n_eval_per_chunk = n_ctx - 1 - first;
    std::vector<uint16_t> log_probs_uint16(size_t(n_eval_per_chunk) * nv);
    std::vector<float> kld_values(size_t(n_eval_per_chunk) * n_chunk_eval);
    std::vector<float> p_diff_values(size_t(n_eval_per_chunk) * n_chunk_eval);
    std::vector<float> logits;
    if (num_batches > 1) {
        logits.reserve(size_t(n_ctx) * n_vocab);
    }

    if (log_progress) {
        LOG_INF("%s: computing over %d chunks, n_ctx=%u, batch_size=%d, n_seq=%d\n",
                __func__, n_chunk_eval, n_ctx, n_batch, n_seq);
    }

    bool overlap_enabled = params.moq_kld_overlap == "on";
    std::string logits_buffer_mode = params.moq_logits_buffer_mode;
    if (logits_buffer_mode != "context" && logits_buffer_mode != "copy" && logits_buffer_mode != "ring") {
        return fail("invalid logits buffer mode: " + logits_buffer_mode);
    }
    if (params.moq_logits_ring_pinned != "on" && params.moq_logits_ring_pinned != "off") {
        return fail("invalid logits ring pinned mode: " + params.moq_logits_ring_pinned);
    }
    if (!overlap_enabled && logits_buffer_mode != "context") {
        LOG_WRN("%s: --moq-logits-buffer-mode %s requires --moq-kld-overlap on; using context mode\n",
                __func__, logits_buffer_mode.c_str());
        logits_buffer_mode = "context";
    }
    if (overlap_enabled && logits_buffer_mode == "context") {
        LOG_WRN("%s: --moq-kld-overlap on with context logits is unsafe; using copy mode\n", __func__);
        logits_buffer_mode = "copy";
    }
    if (overlap_enabled && logits_buffer_mode == "ring" && num_batches > 1) {
        LOG_WRN("%s: ring logits buffer mode currently requires one decode batch per chunk group; using copy mode\n", __func__);
        logits_buffer_mode = "copy";
    }
    const bool overlap_ring = overlap_enabled && logits_buffer_mode == "ring";
    const bool overlap_copy = overlap_enabled && logits_buffer_mode == "copy";
    candidate_profile.label.mode = params.moq_kld_overlap + "/" + logits_buffer_mode;

    moq_logits_output_buffer_guard logits_output_guard{ctx, overlap_ring};
    const bool request_pinned_ring = overlap_ring && params.moq_logits_ring_pinned == "on";
    std::string pinned_buft_desc;
    ggml_backend_buffer_type_t pinned_buft = request_pinned_ring ? moq_select_logits_pinned_buft(pinned_buft_desc) : nullptr;
    if (request_pinned_ring && pinned_buft == nullptr) {
        LOG_WRN("%s: --moq-logits-ring-pinned on requested, but no accelerator host buffer type is available; falling back to ordinary host memory\n", __func__);
    }
    const int overlap_workers = moq_kld_worker_count(params);
    const unsigned hw_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(!overlap_enabled && hw_threads > 1 ? hw_threads - 1 : 0);
    std::unique_ptr<moq_logits_ring_pool> logits_ring_pool;
    std::unique_ptr<moq_kld_overlap_executor> overlap_executor;
    std::vector<std::shared_ptr<moq_kld_chunk_state>> overlap_states;
    if (overlap_enabled) {
        overlap_executor = std::make_unique<moq_kld_overlap_executor>(overlap_workers, params.moq_kld_ring);
        overlap_states.reserve(n_chunk_eval);
    }
    if (overlap_ring) {
        logits_ring_pool = std::make_unique<moq_logits_ring_pool>(params.moq_logits_ring, request_pinned_ring, pinned_buft);
    }

    kl_divergence_result kld;
    auto kld_ptr = kld_values.data();
    auto p_diff_ptr = p_diff_values.data();
    const bool profiling = profiler != nullptr && profiler->level > 0;

    for (int i = 0; i < n_chunk_eval; i += n_seq) {
        const int start = i * n_ctx;
        const int end = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk_eval - i);
        std::vector<moq_eval_chunk_timing> chunk_timings(static_cast<size_t>(n_seq_batch));
        for (int seq = 0; seq < n_seq_batch; ++seq) {
            chunk_timings[seq].label = label;
            chunk_timings[seq].label.mode = candidate_profile.label.mode;
            chunk_timings[seq].chunk = i + seq;
            chunk_timings[seq].n_outputs = n_eval_per_chunk;
        }

        const int64_t t_clear = llama_time_us();
        llama_memory_clear(llama_get_memory(ctx), true);
        double shared_batch_build_ms = (llama_time_us() - t_clear) / 1000.0;
        double shared_decode_ms = 0.0;
        double shared_sync_ms = 0.0;
        double shared_logits_ms = 0.0;
        double shared_ring_wait_ms = 0.0;
        std::shared_ptr<moq_logits_ring_slot> logits_ring_slot;

        if (overlap_ring) {
            const size_t n_outputs_ring = size_t(n_seq_batch) * size_t(n_ctx - first);
            const size_t capacity_floats = std::max<size_t>(n_outputs_ring, llama_n_seq_max(ctx)) * size_t(n_vocab);
            logits_ring_slot = logits_ring_pool->acquire(capacity_floats, i, (int) n_outputs_ring, shared_ring_wait_ms);
            if (!logits_ring_slot || logits_ring_slot->logits == nullptr) {
                llama_batch_free(batch);
                return fail("failed to allocate logits ring slot");
            }
            if (!llama_set_logits_output_buffer(ctx, logits_ring_slot->logits, logits_ring_slot->logits_capacity)) {
                llama_batch_free(batch);
                return fail("failed to set external logits output buffer for ring mode");
            }
        }

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            int n_outputs = 0;

            const int64_t t_batch_build = llama_time_us();
            common_batch_clear(batch);
            for (int seq = 0; seq < n_seq_batch; seq++) {
                int seq_start = batch_start + seq*n_ctx;

                const auto token_org = tokens[seq_start];

                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }

                for (int k = 0; k < batch_size; ++k) {
                    const int pos = j*n_batch + k;
                    const bool need_logits = pos >= first;
                    common_batch_add(batch, tokens[seq_start + k], pos, { seq }, need_logits);
                    n_outputs += need_logits;
                }

                tokens[seq_start] = token_org;
            }
            shared_batch_build_ms += (llama_time_us() - t_batch_build) / 1000.0;

            const int64_t t_decode = llama_time_us();
            if (llama_decode(ctx, batch)) {
                llama_batch_free(batch);
                return fail("llama_decode failed during KLD evaluation");
            }
            const int64_t t_after_decode_call = llama_time_us();
            if (profiling) {
                llama_synchronize(ctx);
            }
            const int64_t t_after_sync = llama_time_us();
            shared_decode_ms += (t_after_sync - t_decode) / 1000.0;
            shared_sync_ms += profiling ? (t_after_sync - t_after_decode_call) / 1000.0 : 0.0;

            if (num_batches > 1 && n_outputs > 0) {
                const int64_t t_logits = llama_time_us();
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + size_t(n_outputs) * n_vocab);
                shared_logits_ms += (llama_time_us() - t_logits) / 1000.0;
            }
        }

        if (overlap_ring && logits_ring_slot) {
            logits_ring_pool->mark_processing(logits_ring_slot, n_seq_batch);
        }

        const double inv_n_seq_batch = n_seq_batch > 0 ? 1.0 / n_seq_batch : 1.0;
        for (int seq = 0; seq < n_seq_batch; seq++) {
            moq_eval_chunk_timing & chunk_timing = chunk_timings[seq];
            chunk_timing.batch_build_ms += shared_batch_build_ms * inv_n_seq_batch;
            chunk_timing.decode_ms += shared_decode_ms * inv_n_seq_batch;
            chunk_timing.logits_ms += shared_logits_ms * inv_n_seq_batch;
            result.batch_build_ms += chunk_timing.batch_build_ms;
            result.decode_ms += chunk_timing.decode_ms;
            result.llama_synchronize_ms += shared_sync_ms * inv_n_seq_batch;
            chunk_timing.kld_ring_wait_ms += shared_ring_wait_ms * inv_n_seq_batch;
            result.kld_ring_wait_ms += shared_ring_wait_ms * inv_n_seq_batch;
            if (overlap_ring && logits_ring_slot) {
                chunk_timing.logits_ring_pinned_enabled = logits_ring_slot->pinned_enabled;
                chunk_timing.logits_alloc_mode = logits_ring_slot->alloc_mode;
                result.logits_ring_pinned_enabled = result.logits_ring_pinned_enabled || logits_ring_slot->pinned_enabled;
                if (result.logits_alloc_mode.empty()) {
                    result.logits_alloc_mode = logits_ring_slot->alloc_mode;
                } else if (result.logits_alloc_mode.find(logits_ring_slot->alloc_mode) == std::string::npos) {
                    result.logits_alloc_mode += "+" + logits_ring_slot->alloc_mode;
                }
            } else if (chunk_timing.logits_alloc_mode.empty()) {
                chunk_timing.logits_alloc_mode = overlap_copy ? "copy_owned" : "context";
            }

            const uint16_t * base_log_probs = nullptr;
            std::string base_error;
            double base_read_ms = 0.0;
            if (!base_logits.get_log_probs(i + seq, log_probs_uint16, base_log_probs, base_read_ms, base_error)) {
                llama_batch_free(batch);
                return fail(base_error);
            }
            chunk_timing.base_read_ms += base_read_ms;
            result.base_read_ms += base_read_ms;

            const int64_t t_logits_access = llama_time_us();
            const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx + first);
            const double logits_access_ms = (llama_time_us() - t_logits_access) / 1000.0;
            chunk_timing.logits_ms += logits_access_ms;
            result.logits_ms += logits_access_ms + shared_logits_ms * inv_n_seq_batch;

            if (overlap_enabled) {
                const double ring_wait_ms = overlap_executor->wait_for_slot();
                chunk_timing.kld_ring_wait_ms += ring_wait_ms;
                result.kld_ring_wait_ms += ring_wait_ms;

                auto state = std::make_shared<moq_kld_chunk_state>();
                state->timing = chunk_timing;
                state->n_vocab = n_vocab;
                state->n_token = n_eval_per_chunk;
                state->nv = nv;
                state->n_ranges = std::max(1, std::min(overlap_workers, n_eval_per_chunk));
                state->remaining.store(state->n_ranges);
                state->partials.resize(state->n_ranges);
                state->range_queue_wait_ms.resize(state->n_ranges, 0.0);
                state->range_worker_ms.resize(state->n_ranges, 0.0);
                state->kld_values = kld_values.data() + size_t(i + seq) * n_eval_per_chunk;
                state->p_diff_values = p_diff_values.data() + size_t(i + seq) * n_eval_per_chunk;

                const llama_token * token_ptr = tokens.data() + start + seq*n_ctx + first;
                state->tokens.assign(token_ptr, token_ptr + n_eval_per_chunk + 1);

                if (base_logits.mode == "stream") {
                    state->base_log_probs_owner = std::make_shared<std::vector<uint16_t>>(
                            base_log_probs, base_log_probs + size_t(n_eval_per_chunk) * nv);
                    state->base_log_probs = state->base_log_probs_owner->data();
                } else {
                    state->base_log_probs = base_log_probs;
                }

                state->logits_count = size_t(n_eval_per_chunk) * n_vocab;
                if (overlap_copy) {
                    const int64_t t_copy = llama_time_us();
                    state->logits_owner = std::shared_ptr<float>(new float[state->logits_count], std::default_delete<float[]>());
                    std::memcpy(state->logits_owner.get(), all_logits, state->logits_count * sizeof(float));
                    state->logits = state->logits_owner.get();
                    const double copy_ms = (llama_time_us() - t_copy) / 1000.0;
                    state->timing.logits_copy_ms += copy_ms;
                    result.logits_copy_ms += copy_ms;
                } else if (overlap_ring) {
                    state->logits = all_logits;
                    state->ring_slot = logits_ring_slot;
                    state->ring_pool = logits_ring_pool.get();
                } else {
                    llama_batch_free(batch);
                    return fail("unsupported overlap logits buffer mode");
                }

                overlap_executor->submit_chunk(state);
                overlap_states.push_back(std::move(state));
            } else {
                const int64_t t_process = llama_time_us();
                process_logits(n_vocab, all_logits, tokens.data() + start + seq*n_ctx + first, n_eval_per_chunk,
                        workers, base_log_probs, kld, kld_ptr, p_diff_ptr);
                const double process_ms = (llama_time_us() - t_process) / 1000.0;
                chunk_timing.process_logits_ms += process_ms;
                chunk_timing.kld_worker_ms += process_ms;
                result.process_logits_ms += process_ms;
                result.kld_worker_ms += process_ms;
                p_diff_ptr += n_eval_per_chunk;
                kld_ptr += n_eval_per_chunk;
                chunk_timing.total_ms = chunk_timing.base_read_ms + chunk_timing.batch_build_ms +
                    chunk_timing.decode_ms + chunk_timing.logits_ms + chunk_timing.process_logits_ms;
                candidate_profile.totals.chunks++;
                candidate_profile.totals.n_outputs += chunk_timing.n_outputs;
                candidate_profile.totals.base_read_ms += chunk_timing.base_read_ms;
                candidate_profile.totals.batch_build_ms += chunk_timing.batch_build_ms;
                candidate_profile.totals.decode_ms += chunk_timing.decode_ms;
                candidate_profile.totals.logits_ms += chunk_timing.logits_ms;
                candidate_profile.totals.process_logits_ms += chunk_timing.process_logits_ms;
                candidate_profile.totals.kld_worker_ms += chunk_timing.kld_worker_ms;
                if (profiler != nullptr) {
                    profiler->add_chunk(chunk_timing);
                }
            }

            if (log_progress && !overlap_enabled) {
                auto log_ppl = moq_mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
                auto kl_div = moq_mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
                LOG_INF("%s: chunk %d/%d PPL=%.6lf mean_KLD=%.6lf\n",
                        __func__, i + seq + 1, n_chunk_eval, exp(log_ppl.first), kl_div.first);
            }
        }

        logits.clear();
    }

    if (overlap_enabled) {
        const double join_ms = overlap_executor->wait_all();
        result.kld_join_ms += join_ms;
        overlap_executor->shutdown();

        const double join_per_chunk = overlap_states.empty() ? 0.0 : join_ms / overlap_states.size();
        for (const auto & state : overlap_states) {
            double queue_wait_ms = 0.0;
            double worker_sum_ms = 0.0;
            for (int ir = 0; ir < state->n_ranges; ++ir) {
                moq_kld_accumulate(kld, state->partials[ir]);
                queue_wait_ms = std::max(queue_wait_ms, state->range_queue_wait_ms[ir]);
                worker_sum_ms += state->range_worker_ms[ir];
            }

            moq_eval_chunk_timing chunk_timing = state->timing;
            const double worker_wall_ms = state->last_worker_end_us > state->first_worker_start_us ?
                (state->last_worker_end_us - state->first_worker_start_us) / 1000.0 : worker_sum_ms;
            chunk_timing.kld_queue_wait_ms += queue_wait_ms;
            chunk_timing.kld_worker_ms += worker_wall_ms;
            chunk_timing.process_logits_ms += worker_wall_ms;
            chunk_timing.kld_join_ms += join_per_chunk;
            chunk_timing.total_ms = chunk_timing.base_read_ms + chunk_timing.batch_build_ms +
                chunk_timing.decode_ms + chunk_timing.logits_ms + chunk_timing.logits_copy_ms +
                chunk_timing.kld_ring_wait_ms + chunk_timing.kld_join_ms;

            result.kld_queue_wait_ms += queue_wait_ms;
            result.kld_worker_ms += worker_wall_ms;
            result.process_logits_ms += worker_wall_ms;

            candidate_profile.totals.chunks++;
            candidate_profile.totals.n_outputs += chunk_timing.n_outputs;
            candidate_profile.totals.base_read_ms += chunk_timing.base_read_ms;
            candidate_profile.totals.batch_build_ms += chunk_timing.batch_build_ms;
            candidate_profile.totals.decode_ms += chunk_timing.decode_ms;
            candidate_profile.totals.logits_ms += chunk_timing.logits_ms;
            candidate_profile.totals.logits_copy_ms += chunk_timing.logits_copy_ms;
            candidate_profile.totals.process_logits_ms += chunk_timing.process_logits_ms;
            candidate_profile.totals.kld_queue_wait_ms += chunk_timing.kld_queue_wait_ms;
            candidate_profile.totals.kld_worker_ms += chunk_timing.kld_worker_ms;
            candidate_profile.totals.kld_join_ms += chunk_timing.kld_join_ms;
            candidate_profile.totals.kld_ring_wait_ms += chunk_timing.kld_ring_wait_ms;
            if (profiler != nullptr) {
                profiler->add_chunk(chunk_timing);
            }
        }
    }

    llama_batch_free(batch);

    if (kld.count == 0) {
        return fail("no tokens were evaluated");
    }

    const int64_t t_sort = llama_time_us();
    kld_values.resize(kld.count);
    std::sort(kld_values.begin(), kld_values.end());
    result.sort_ms = (llama_time_us() - t_sort) / 1000.0;

    auto log_ppl = moq_mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
    auto log_ppl_base = moq_mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    auto kl_div = moq_mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);

    result.ok = true;
    result.n_ctx = (int) n_ctx;
    result.n_vocab = n_vocab;
    result.n_chunks = n_chunk_eval;
    result.count = kld.count;
    result.ppl = exp(log_ppl.first);
    result.ppl_base = exp(log_ppl_base.first);
    result.mean_kld = kl_div.first;
    result.max_kld = kld_values.empty() ? 0.0 : kld_values.back();
    result.p99_kld = moq_percentile_sorted(kld_values, 0.990f);
    result.p999_kld = moq_percentile_sorted(kld_values, 0.999f);
    result.eval_ms = (llama_time_us() - t_start_us) / 1000.0;
    candidate_profile.ok = true;
    candidate_profile.totals.sort_ms = result.sort_ms;
    candidate_profile.totals.total_ms = result.eval_ms;
    candidate_profile.ppl = result.ppl;
    candidate_profile.mean_kld = result.mean_kld;
    candidate_profile.p999_kld = result.p999_kld;
    candidate_profile.logits_ring_pinned_enabled = result.logits_ring_pinned_enabled;
    candidate_profile.logits_alloc_mode = result.logits_alloc_mode.empty() ?
        (overlap_ring ? (request_pinned_ring ? "fallback" : "std_vector") : (overlap_copy ? "copy_owned" : "context")) :
        result.logits_alloc_mode;
    if (profiler != nullptr) {
        profiler->add_candidate(candidate_profile);
    }
    return result;
}

struct moq_source_tensor {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<int64_t> ne;
    size_t nbytes = 0;
    size_t file_offset = 0;
};

static uint64_t moq_fnv1a64(const std::string & s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

static std::string moq_hex_u64(uint64_t v) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << v;
    return ss.str();
}

static std::string moq_file_identity_hash(const std::string & path) {
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    const auto mtime = fs::last_write_time(path, ec);
    const auto ticks = mtime.time_since_epoch().count();
    return moq_hex_u64(moq_fnv1a64(path + "|" + std::to_string(size) + "|" + std::to_string(ticks)));
}

struct moq_source_store {
    std::string path;
    std::string source_hash;
    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_meta;
    std::unordered_map<std::string, moq_source_tensor> tensors;

    bool open(const std::string & p, std::string & error) {
        path = p;
        source_hash = moq_file_identity_hash(path);

        ggml_context * raw_meta = nullptr;
        gguf_init_params params = {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ &raw_meta,
        };

        ctx_gguf.reset(gguf_init_from_file(path.c_str(), params));
        ctx_meta.reset(raw_meta);
        if (!ctx_gguf || !ctx_meta) {
            error = string_format("failed to open source GGUF: %s", path.c_str());
            return false;
        }

        tensors.clear();
        const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf.get());
        const size_t data_offset = gguf_get_data_offset(ctx_gguf.get());
        tensors.reserve((size_t) n_tensors);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            ggml_tensor * t = ggml_get_tensor(ctx_meta.get(), name);
            if (t == nullptr) {
                continue;
            }

            moq_source_tensor st;
            st.name = name;
            st.type = t->type;
            st.ne.assign(t->ne, t->ne + GGML_MAX_DIMS);
            st.nbytes = ggml_nbytes(t);
            st.file_offset = data_offset + gguf_get_tensor_offset(ctx_gguf.get(), i);
            tensors.emplace(st.name, std::move(st));
        }

        return true;
    }

    const moq_source_tensor * get(const std::string & name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }

    bool read_data(const moq_source_tensor & tensor, std::vector<uint8_t> & data, std::string & error) const {
        std::ifstream in(path.c_str(), std::ios::binary);
        if (!in) {
            error = string_format("failed to open source GGUF for tensor read: %s", path.c_str());
            return false;
        }

        data.resize(tensor.nbytes);
        in.seekg((std::streamoff) tensor.file_offset, std::ios::beg);
        if (in.fail()) {
            error = string_format("failed to seek source tensor %s", tensor.name.c_str());
            return false;
        }
        if (in.read((char *) data.data(), data.size()).fail()) {
            error = string_format("failed to read source tensor %s", tensor.name.c_str());
            return false;
        }
        return true;
    }
};

struct moq_imatrix_store {
    std::string path;
    std::string imatrix_hash = "none";
    std::vector<std::string> datasets;
    int chunk_count = -1;
    uint32_t chunk_size = 0;
    std::unordered_map<std::string, std::vector<float>> data;

    bool empty() const {
        return data.empty();
    }

    const std::vector<float> * get(const std::string & name) const {
        auto it = data.find(name);
        return it == data.end() ? nullptr : &it->second;
    }

    bool open_legacy(const std::string & p, std::string & error) {
        std::ifstream in(p.c_str(), std::ios::binary);
        if (!in) {
            error = string_format("failed to open imatrix file: %s", p.c_str());
            return false;
        }

        int n_entries = 0;
        in.read((char *) &n_entries, sizeof(n_entries));
        if (in.fail() || n_entries < 1) {
            error = string_format("no imatrix data in file: %s", p.c_str());
            return false;
        }

        data.clear();
        data.reserve((size_t) n_entries);
        for (int i = 0; i < n_entries; ++i) {
            int len = 0;
            in.read((char *) &len, sizeof(len));
            if (in.fail() || len <= 0) {
                error = string_format("failed reading imatrix entry name length at entry %d", i);
                return false;
            }

            std::vector<char> name_buf((size_t) len + 1);
            in.read(name_buf.data(), len);
            if (in.fail()) {
                error = string_format("failed reading imatrix entry name at entry %d", i);
                return false;
            }
            name_buf[len] = 0;

            int ncall = 0;
            int nval = 0;
            in.read((char *) &ncall, sizeof(ncall));
            in.read((char *) &nval, sizeof(nval));
            if (in.fail() || nval < 1) {
                error = string_format("failed reading imatrix entry header for %s", name_buf.data());
                return false;
            }

            auto & values = data[name_buf.data()];
            values.resize((size_t) nval);
            in.read((char *) values.data(), (size_t) nval * sizeof(float));
            if (in.fail()) {
                error = string_format("failed reading imatrix values for %s", name_buf.data());
                return false;
            }

            if (ncall > 0) {
                for (float & v : values) {
                    v /= ncall;
                }
            }
        }

        if (in.peek() != EOF) {
            int last_call = 0;
            int dataset_len = 0;
            in.read((char *) &last_call, sizeof(last_call));
            in.read((char *) &dataset_len, sizeof(dataset_len));
            if (!in.fail() && dataset_len > 0) {
                std::vector<char> dataset_buf((size_t) dataset_len);
                in.read(dataset_buf.data(), dataset_len);
                if (!in.fail()) {
                    datasets.emplace_back(dataset_buf.begin(), dataset_buf.end());
                    chunk_count = last_call;
                }
            }
        }

        return true;
    }

    bool open(const std::string & p, std::string & error) {
        path = p;
        imatrix_hash = moq_file_identity_hash(path);
        datasets.clear();
        data.clear();
        chunk_count = -1;
        chunk_size = 0;

        ggml_context * raw_ctx = nullptr;
        gguf_init_params params = {
            /*.no_alloc =*/ false,
            /*.ctx      =*/ &raw_ctx,
        };
        gguf_context_ptr ctx_gguf(gguf_init_from_file(path.c_str(), params));
        ggml_context_ptr ctx(raw_ctx);

        if (!ctx_gguf || !ctx) {
            return open_legacy(path, error);
        }

        const int64_t dataset_idx     = gguf_find_key(ctx_gguf.get(), "imatrix.datasets");
        const int64_t chunk_count_idx = gguf_find_key(ctx_gguf.get(), "imatrix.chunk_count");
        const int64_t chunk_size_idx  = gguf_find_key(ctx_gguf.get(), "imatrix.chunk_size");
        if (dataset_idx < 0 || chunk_count_idx < 0 || chunk_size_idx < 0) {
            error = string_format("missing imatrix metadata in file: %s", path.c_str());
            return false;
        }

        chunk_count = (int) gguf_get_val_u32(ctx_gguf.get(), chunk_count_idx);
        chunk_size  = gguf_get_val_u32(ctx_gguf.get(), chunk_size_idx);
        const size_t n_datasets = gguf_get_arr_n(ctx_gguf.get(), dataset_idx);
        datasets.reserve(n_datasets);
        for (size_t i = 0; i < n_datasets; ++i) {
            datasets.push_back(gguf_get_arr_str(ctx_gguf.get(), dataset_idx, i));
        }

        const std::string sums_suffix{ ".in_sum2" };
        const std::string counts_suffix{ ".counts" };
        std::map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> sums_counts_for;

        for (ggml_tensor * cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
            std::string name = cur->name;
            if (name.empty()) {
                continue;
            }
            if (string_remove_suffix(name, sums_suffix)) {
                sums_counts_for[std::move(name)].first = cur;
            } else if (string_remove_suffix(name, counts_suffix)) {
                sums_counts_for[std::move(name)].second = cur;
            }
        }

        for (const auto & kv : sums_counts_for) {
            const std::string & name = kv.first;
            const ggml_tensor * sums = kv.second.first;
            const ggml_tensor * counts = kv.second.second;
            if (sums == nullptr || counts == nullptr) {
                error = string_format("mismatched imatrix sums/counts for %s", name.c_str());
                return false;
            }
            if (sums->type != GGML_TYPE_F32 || counts->type != GGML_TYPE_F32) {
                error = string_format("imatrix tensors for %s must be F32", name.c_str());
                return false;
            }

            const int64_t ne0 = sums->ne[0];
            const int64_t ne1 = sums->ne[1];
            if (counts->ne[0] < ne1) {
                error = string_format("imatrix counts shape mismatch for %s", name.c_str());
                return false;
            }

            auto & values = data[name];
            values.resize((size_t) ggml_nelements(sums));
            const float * sums_data = (const float *) sums->data;
            const float * counts_data = (const float *) counts->data;
            for (int64_t j = 0; j < ne1; ++j) {
                const float count = counts_data[j];
                for (int64_t i = 0; i < ne0; ++i) {
                    values[(size_t) j * ne0 + i] = count > 0.0f ? sums_data[(size_t) j * ne0 + i] / count : 1.0f;
                }
            }

            for (float v : values) {
                if (!std::isfinite(v)) {
                    error = string_format("imatrix contains non-finite value for %s", name.c_str());
                    return false;
                }
            }
        }

        if (data.empty()) {
            error = string_format("no imatrix tensor entries in file: %s", path.c_str());
            return false;
        }
        return true;
    }
};

struct moq_group {
    std::string name;
    std::vector<std::string> tensors;
};

static bool moq_parse_groups(const std::string & path, std::vector<moq_group> & groups, std::string & error) {
    std::ifstream in(path.c_str());
    if (!in) {
        error = string_format("failed to open MoQ groups file: %s", path.c_str());
        return false;
    }

    try {
        json doc = json::parse(in);
        if (!doc.contains("groups") || !doc["groups"].is_array()) {
            error = "MoQ groups JSON must contain an array field named 'groups'";
            return false;
        }

        groups.clear();
        int idx = 0;
        for (const auto & g : doc["groups"]) {
            moq_group group;
            group.name = g.value("name", string_format("group_%d", idx++));
            if (!g.contains("tensors") || !g["tensors"].is_array()) {
                error = string_format("MoQ group '%s' is missing tensor array", group.name.c_str());
                return false;
            }
            for (const auto & t : g["tensors"]) {
                if (!t.is_string()) {
                    error = string_format("MoQ group '%s' contains a non-string tensor name", group.name.c_str());
                    return false;
                }
                group.tensors.push_back(t.get<std::string>());
            }
            groups.push_back(std::move(group));
        }
    } catch (const std::exception & e) {
        error = string_format("failed to parse MoQ groups JSON: %s", e.what());
        return false;
    }

    if (groups.empty()) {
        error = "MoQ groups JSON contains no groups";
        return false;
    }
    return true;
}

static std::string moq_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::toupper(c); });
    return s;
}

struct moq_qtype_candidate {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    bool supported = false;
    bool requires_imatrix = false;
    std::string unsupported_reason;
};

static const std::vector<moq_qtype_candidate> & moq_qtype_registry() {
    static const std::vector<moq_qtype_candidate> registry = {
        { "IQ1_S",    GGML_TYPE_IQ1_S,   true,  true,  "" },
        { "IQ1_M",    GGML_TYPE_IQ1_M,   true,  false, "" },
        { "IQ2_XXS",  GGML_TYPE_IQ2_XXS, true,  true,  "" },
        { "IQ2_XS",   GGML_TYPE_IQ2_XS,  true,  true,  "" },
        { "IQ2_S",    GGML_TYPE_IQ2_S,   true,  false, "" },
        { "IQ2_M",    GGML_TYPE_COUNT,   false, false, "mixture ftype, no single GGML tensor type" },
        { "Q2_K",     GGML_TYPE_Q2_K,    true,  false, "" },
        { "IQ3_XXS",  GGML_TYPE_IQ3_XXS, true,  false, "" },
        { "IQ3_S",    GGML_TYPE_IQ3_S,   true,  false, "" },
        { "IQ3_M",    GGML_TYPE_COUNT,   false, false, "mixture ftype, no single GGML tensor type" },
        { "Q3_K",     GGML_TYPE_Q3_K,    true,  false, "" },
        { "IQ4_NL",   GGML_TYPE_IQ4_NL,  true,  false, "" },
        { "IQ4_XS",   GGML_TYPE_IQ4_XS,  true,  false, "" },
        { "Q4_0",     GGML_TYPE_Q4_0,    true,  false, "" },
        { "Q4_1",     GGML_TYPE_Q4_1,    true,  false, "" },
        { "Q4_K",     GGML_TYPE_Q4_K,    true,  false, "" },
        { "Q5_0",     GGML_TYPE_Q5_0,    true,  false, "" },
        { "Q5_1",     GGML_TYPE_Q5_1,    true,  false, "" },
        { "Q5_K",     GGML_TYPE_Q5_K,    true,  false, "" },
        { "Q6_K",     GGML_TYPE_Q6_K,    true,  false, "" },
        { "Q8_0",     GGML_TYPE_Q8_0,    true,  false, "" },
        { "F16",      GGML_TYPE_F16,     true,  false, "" },
        { "BF16",     GGML_TYPE_BF16,    true,  false, "" },
        { "F32",      GGML_TYPE_F32,     true,  false, "" },
    };
    return registry;
}

static bool moq_parse_qtype(const std::string & s, moq_qtype_candidate & candidate) {
    const std::string q = moq_upper(string_strip(s));

    const std::string canonical =
        (q == "Q4_K_M" || q == "Q4_K_S") ? "Q4_K" :
        (q == "Q5_K_M" || q == "Q5_K_S") ? "Q5_K" :
        q;

    for (const auto & info : moq_qtype_registry()) {
        if (info.name == canonical) {
            candidate = info;
            return true;
        }
    }
    return false;
}

static bool moq_parse_candidates(const std::string & list, std::vector<moq_qtype_candidate> & candidates, std::string & error) {
    candidates.clear();
    for (const auto & item : string_split<std::string>(list, ',')) {
        if (string_strip(item).empty()) {
            continue;
        }
        moq_qtype_candidate candidate;
        if (!moq_parse_qtype(item, candidate)) {
            error = string_format("unknown MoQ candidate qtype: %s", item.c_str());
            return false;
        }
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty()) {
        error = "MoQ candidate list is empty";
        return false;
    }
    return true;
}

static void moq_print_qtypes() {
    for (const auto & info : moq_qtype_registry()) {
        if (info.supported) {
            printf("%-10s supported%s\n", info.name.c_str(), info.requires_imatrix ? " (requires imatrix)" : "");
        } else {
            printf("%-10s unsupported: %s\n", info.name.c_str(), info.unsupported_reason.c_str());
        }
    }
}

static double moq_qtype_sort_bpw(const moq_qtype_candidate & c) {
    if (!c.supported) {
        return 1e9;
    }
    switch (c.type) {
        case GGML_TYPE_IQ1_S:   return 1.56;
        case GGML_TYPE_IQ1_M:   return 1.75;
        case GGML_TYPE_IQ2_XXS: return 2.06;
        case GGML_TYPE_IQ2_XS:  return 2.31;
        case GGML_TYPE_IQ2_S:   return 2.50;
        case GGML_TYPE_Q2_K:    return 2.96;
        case GGML_TYPE_IQ3_XXS: return 3.06;
        case GGML_TYPE_IQ3_S:   return 3.44;
        case GGML_TYPE_Q3_K:    return 3.55;
        case GGML_TYPE_IQ4_XS:  return 4.25;
        case GGML_TYPE_Q4_0:    return 4.34;
        case GGML_TYPE_Q4_1:    return 4.78;
        case GGML_TYPE_IQ4_NL:  return 4.50;
        case GGML_TYPE_Q4_K:    return 4.50;
        case GGML_TYPE_Q5_0:    return 5.20;
        case GGML_TYPE_Q5_1:    return 5.60;
        case GGML_TYPE_Q5_K:    return 5.50;
        case GGML_TYPE_Q6_K:    return 6.56;
        case GGML_TYPE_Q8_0:    return 8.50;
        case GGML_TYPE_F16:     return 16.0;
        case GGML_TYPE_BF16:    return 16.0;
        case GGML_TYPE_F32:     return 32.0;
        default:                return 100.0;
    }
}

static std::string moq_safe_component(std::string s) {
    for (char & c : s) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            c = '_';
        }
    }
    return s;
}

static bool moq_same_shape(const std::vector<int64_t> & ne, const ggml_tensor * t) {
    if (t == nullptr || ne.size() < GGML_MAX_DIMS) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (ne[i] != t->ne[i]) {
            return false;
        }
    }
    return true;
}

static int64_t moq_nelements(const std::vector<int64_t> & ne) {
    int64_t n = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        n *= ne[i];
    }
    return n;
}

static size_t moq_quant_nbytes(ggml_type type, const std::vector<int64_t> & ne) {
    return (size_t) ne[1] * (size_t) ne[2] * (size_t) ne[3] * ggml_row_size(type, ne[0]);
}

static bool moq_source_to_f32(
        const moq_source_tensor & tensor,
        const std::vector<uint8_t> & raw,
        std::vector<float> & f32,
        std::string & error) {
    const int64_t n = moq_nelements(tensor.ne);
    if (n <= 0) {
        error = string_format("source tensor %s has invalid shape", tensor.name.c_str());
        return false;
    }

    f32.resize((size_t) n);
    if (tensor.type == GGML_TYPE_F32) {
        if (raw.size() != (size_t) n * sizeof(float)) {
            error = string_format("source tensor %s F32 byte size mismatch", tensor.name.c_str());
            return false;
        }
        memcpy(f32.data(), raw.data(), raw.size());
        return true;
    }

    if (tensor.type != GGML_TYPE_F16 && tensor.type != GGML_TYPE_BF16) {
        error = string_format("source tensor %s has unsupported source type %s", tensor.name.c_str(), ggml_type_name(tensor.type));
        return false;
    }

    const auto * traits = ggml_get_type_traits(tensor.type);
    if (traits == nullptr || traits->to_float == nullptr) {
        error = string_format("source tensor %s type %s cannot convert to F32", tensor.name.c_str(), ggml_type_name(tensor.type));
        return false;
    }
    traits->to_float(raw.data(), f32.data(), n);
    return true;
}

static bool moq_quantize_f32(
        ggml_type type,
        const std::vector<int64_t> & ne,
        const std::vector<float> & f32,
        const std::vector<float> * imatrix,
        std::vector<uint8_t> & out,
        std::string & error) {
    const int64_t blck = ggml_blck_size(type);
    if (ne[0] % blck != 0) {
        error = string_format("ne[0]=%lld is not divisible by block size %lld for %s",
                (long long) ne[0], (long long) blck, ggml_type_name(type));
        return false;
    }

    const int64_t n_per_row = ne[0];
    const int64_t nrows = ne[1];
    const int64_t nslices = ne[2] * ne[3];
    const int64_t nelements_matrix = n_per_row * nrows;
    const size_t row_size = ggml_row_size(type, n_per_row);
    out.assign(moq_quant_nbytes(type, ne), 0);

    const float * imatrix_data = imatrix == nullptr || imatrix->empty() ? nullptr : imatrix->data();
    if (imatrix_data != nullptr && imatrix->size() != (size_t) n_per_row * (size_t) nslices) {
        error = string_format("imatrix size mismatch: tensor expects %lld values, got %zu",
                (long long) n_per_row * nslices, imatrix->size());
        return false;
    }
    if (ggml_quantize_requires_imatrix(type) && imatrix_data == nullptr) {
        error = string_format("%s requires imatrix data", ggml_type_name(type));
        return false;
    }

    size_t written_total = 0;
    for (int64_t is = 0; is < nslices; ++is) {
        const float * src = f32.data() + is * nelements_matrix;
        void * dst = out.data() + (size_t) is * (size_t) nrows * row_size;
        const float * imatrix_slice = imatrix_data == nullptr ? nullptr : imatrix_data + is * n_per_row;
        const size_t written = ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, imatrix_slice);
        written_total += written;
    }

    if (written_total != out.size()) {
        error = string_format("quantized byte size mismatch for %s: wrote %zu expected %zu",
                ggml_type_name(type), written_total, out.size());
        return false;
    }
    if (!ggml_validate_row_data(type, out.data(), out.size())) {
        error = string_format("quantized data validation failed for %s", ggml_type_name(type));
        return false;
    }
    return true;
}

static const std::vector<float> * moq_select_imatrix(
        const moq_imatrix_store * imatrix,
        const std::string & tensor_name,
        const moq_source_tensor & src_tensor,
        ggml_type type,
        std::string * warning) {
    if (warning != nullptr) {
        warning->clear();
    }
    if (imatrix == nullptr || imatrix->empty() || !ggml_is_quantized(type)) {
        return nullptr;
    }

    const std::vector<float> * data = imatrix->get(tensor_name);
    if (data == nullptr) {
        if (warning != nullptr) {
            *warning = string_format("imatrix entry not found for tensor %s", tensor_name.c_str());
        }
        return nullptr;
    }

    const size_t expected = (size_t) src_tensor.ne[0] * (size_t) src_tensor.ne[2] * (size_t) src_tensor.ne[3];
    if (data->size() != expected) {
        if (warning != nullptr) {
            *warning = string_format("imatrix size mismatch for tensor %s: expected %zu got %zu",
                    tensor_name.c_str(), expected, data->size());
        }
        return nullptr;
    }

    return data;
}

struct moq_owned_tensor {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * tensor = nullptr;
    size_t nbytes = 0;
};

struct moq_cache_stats {
    size_t requests = 0;
    size_t hits = 0;
    size_t misses = 0;
    size_t memory_hits = 0;
    size_t memory_misses = 0;
    size_t disk_hits = 0;
    size_t disk_misses = 0;
    size_t memory_evictions = 0;
    size_t bytes_read = 0;
    size_t bytes_written = 0;
    size_t memory_current_bytes = 0;
    size_t memory_peak_bytes = 0;
    size_t memory_limit_bytes = 0;
    size_t prequant_tasks = 0;
    size_t prequant_ready = 0;
    size_t prequant_built = 0;
    size_t prequant_failed = 0;
    size_t prequant_bytes_written = 0;
    double prequant_ms = 0.0;
    double prequant_quantize_ms = 0.0;
    int prequant_threads = 0;
};

struct moq_tensor_build_result {
    std::shared_ptr<moq_owned_tensor> owned;
    bool cache_hit = false;
    bool mem_cache_hit = false;
    bool disk_cache_hit = false;
    bool cache_miss = false;
    bool imatrix_used = false;
    size_t source_bytes = 0;
    size_t quant_bytes = 0;
    double quantize_ms = 0.0;
    double cache_load_ms = 0.0;
    double mem_cache_load_ms = 0.0;
    double disk_cache_load_ms = 0.0;
    double upload_ms = 0.0;
    std::string error;
};

struct moq_mem_cache_entry {
    std::vector<uint8_t> data;
    size_t bytes = 0;
    std::string label;
    std::list<std::string>::iterator lru_it;
};

struct moq_tensor_cache {
    std::string source_hash;
    std::string imatrix_hash = "none";
    std::string quantizer_version = "moq_dynamic_v1";
    bool disk_enabled = true;
    bool mem_enabled = true;
    std::list<std::string> lru;
    std::unordered_map<std::string, moq_mem_cache_entry> memory;
    moq_cache_stats stats;
};

struct moq_cache_paths {
    std::string qtype_name;
    std::string effective_imatrix_hash;
    std::string key_hash;
    std::string disk_stem;
    std::string cache_key;
    fs::path bin_path;
    fs::path meta_path;
    bool imatrix_used = false;
    size_t quant_bytes = 0;
};

static bool moq_mem_cache_get(moq_tensor_cache & cache, const std::string & key, std::vector<uint8_t> & data) {
    if (!cache.mem_enabled || cache.stats.memory_limit_bytes == 0) {
        cache.stats.memory_misses++;
        return false;
    }

    auto it = cache.memory.find(key);
    if (it == cache.memory.end()) {
        cache.stats.memory_misses++;
        return false;
    }

    cache.lru.erase(it->second.lru_it);
    cache.lru.push_front(key);
    it->second.lru_it = cache.lru.begin();
    data = it->second.data;
    cache.stats.memory_hits++;
    return true;
}

static void moq_mem_cache_insert(
        moq_tensor_cache & cache,
        const std::string & key,
        const std::string & label,
        const std::vector<uint8_t> & data) {
    if (!cache.mem_enabled || cache.stats.memory_limit_bytes == 0 || data.empty()) {
        return;
    }
    if (data.size() > cache.stats.memory_limit_bytes) {
        return;
    }

    auto it_existing = cache.memory.find(key);
    if (it_existing != cache.memory.end()) {
        cache.stats.memory_current_bytes -= it_existing->second.bytes;
        cache.lru.erase(it_existing->second.lru_it);
        cache.memory.erase(it_existing);
    }

    while (cache.stats.memory_current_bytes + data.size() > cache.stats.memory_limit_bytes && !cache.lru.empty()) {
        const std::string victim = cache.lru.back();
        cache.lru.pop_back();
        auto it = cache.memory.find(victim);
        if (it != cache.memory.end()) {
            cache.stats.memory_current_bytes -= it->second.bytes;
            cache.memory.erase(it);
            cache.stats.memory_evictions++;
        }
    }

    cache.lru.push_front(key);
    moq_mem_cache_entry entry;
    entry.data = data;
    entry.bytes = data.size();
    entry.label = label;
    entry.lru_it = cache.lru.begin();
    cache.stats.memory_current_bytes += entry.bytes;
    cache.stats.memory_peak_bytes = std::max(cache.stats.memory_peak_bytes, cache.stats.memory_current_bytes);
    cache.memory.emplace(key, std::move(entry));
}

static moq_cache_paths moq_make_cache_paths(
        const moq_source_tensor & tensor,
        ggml_type type,
        const moq_tensor_cache & cache,
        const std::vector<float> * imatrix,
        const fs::path & cache_dir) {
    moq_cache_paths paths;
    paths.qtype_name = ggml_type_name(type);
    paths.quant_bytes = moq_quant_nbytes(type, tensor.ne);
    paths.imatrix_used = imatrix != nullptr && !imatrix->empty() && ggml_is_quantized(type);
    paths.effective_imatrix_hash = paths.imatrix_used ? cache.imatrix_hash : "none";

    const std::string key_material = cache.source_hash + "\n" + tensor.name + "\n" + paths.qtype_name + "\n" +
        paths.effective_imatrix_hash + "\n" + (paths.imatrix_used ? "imatrix=true" : "imatrix=false") + "\n" +
        cache.quantizer_version;
    paths.key_hash = moq_hex_u64(moq_fnv1a64(key_material));
    paths.disk_stem = moq_safe_component(tensor.name) + "." + paths.qtype_name + "." + paths.key_hash;
    paths.cache_key = paths.disk_stem;
    paths.bin_path = cache_dir / (paths.disk_stem + ".bin");
    paths.meta_path = cache_dir / (paths.disk_stem + ".json");
    return paths;
}

static json moq_cache_metadata_json(
        const moq_source_tensor & tensor,
        ggml_type type,
        size_t nbytes,
        const moq_tensor_cache & cache,
        bool imatrix_used,
        const std::string & effective_imatrix_hash) {
    json j;
    j["tensor_name"] = tensor.name;
    j["qtype"] = ggml_type_name(type);
    j["shape"] = tensor.ne;
    j["nbytes"] = nbytes;
    j["source_hash"] = cache.source_hash;
    j["imatrix_hash"] = effective_imatrix_hash;
    j["imatrix_used"] = imatrix_used;
    j["quantizer_version"] = cache.quantizer_version;
    return j;
}

static bool moq_cache_metadata_matches(
        const fs::path & meta_path,
        const moq_source_tensor & tensor,
        ggml_type type,
        const moq_tensor_cache & cache,
        size_t expected_nbytes,
        bool imatrix_used,
        const std::string & effective_imatrix_hash) {
    std::ifstream in(meta_path);
    if (!in) {
        return false;
    }
    try {
        json j = json::parse(in);
        return j.value("tensor_name", "") == tensor.name &&
               j.value("qtype", "") == ggml_type_name(type) &&
               j.value("nbytes", (size_t) 0) == expected_nbytes &&
               j.value("source_hash", "") == cache.source_hash &&
               j.value("imatrix_hash", "") == effective_imatrix_hash &&
               j.value("imatrix_used", false) == imatrix_used &&
               j.value("quantizer_version", "") == cache.quantizer_version;
    } catch (...) {
        return false;
    }
}

static bool moq_disk_cache_ready(
        const moq_source_tensor & tensor,
        ggml_type type,
        const moq_tensor_cache & cache,
        const moq_cache_paths & paths) {
    if (!cache.disk_enabled || !fs::exists(paths.bin_path)) {
        return false;
    }
    return moq_cache_metadata_matches(paths.meta_path, tensor, type, cache, paths.quant_bytes,
            paths.imatrix_used, paths.effective_imatrix_hash);
}

static bool moq_alloc_runtime_tensor(
        const std::string & name,
        ggml_type type,
        const std::vector<int64_t> & ne,
        ggml_backend_buffer_type_t buft,
        const std::vector<uint8_t> & data,
        std::shared_ptr<moq_owned_tensor> & owned,
        std::string & error) {
    ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    owned = std::make_shared<moq_owned_tensor>();
    owned->ctx.reset(ggml_init(params));
    if (!owned->ctx) {
        error = "failed to allocate ggml context for dynamic tensor";
        return false;
    }

    int64_t ne_arr[GGML_MAX_DIMS] = { ne[0], ne[1], ne[2], ne[3] };
    owned->tensor = ggml_new_tensor(owned->ctx.get(), type, GGML_MAX_DIMS, ne_arr);
    if (owned->tensor == nullptr) {
        error = "failed to create dynamic tensor";
        return false;
    }
    ggml_set_name(owned->tensor, name.c_str());

    if (ggml_nbytes(owned->tensor) != data.size()) {
        error = string_format("dynamic tensor byte size mismatch: tensor=%zu data=%zu", ggml_nbytes(owned->tensor), data.size());
        return false;
    }

    owned->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(owned->ctx.get(), buft));
    if (!owned->buffer) {
        error = string_format("failed to allocate dynamic tensor backend buffer: %s", ggml_backend_buft_name(buft));
        return false;
    }
    ggml_backend_buffer_set_usage(owned->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(owned->tensor, data.data(), 0, data.size());
    owned->nbytes = data.size();
    return true;
}

static moq_tensor_build_result moq_get_or_build_tensor(
        const moq_source_store & source,
        moq_tensor_cache & cache,
        const moq_source_tensor & src_tensor,
        ggml_type target_type,
        const std::vector<float> * imatrix,
        ggml_backend_buffer_type_t buft,
        const fs::path & cache_dir,
        bool allow_quantize_on_miss) {
    moq_tensor_build_result result;
    result.source_bytes = src_tensor.nbytes;
    const moq_cache_paths paths = moq_make_cache_paths(src_tensor, target_type, cache, imatrix, cache_dir);
    result.quant_bytes = paths.quant_bytes;
    result.imatrix_used = paths.imatrix_used;

    cache.stats.requests++;

    std::vector<uint8_t> quant_data;
    {
        const int64_t t0 = llama_time_us();
        if (moq_mem_cache_get(cache, paths.cache_key, quant_data)) {
            result.mem_cache_hit = true;
            result.cache_hit = true;
            result.mem_cache_load_ms += (llama_time_us() - t0) / 1000.0;
            result.cache_load_ms += result.mem_cache_load_ms;
            cache.stats.hits++;
        }
    }

    if (result.mem_cache_hit) {
        const int64_t t_upload = llama_time_us();
        std::string error;
        if (!moq_alloc_runtime_tensor(src_tensor.name, target_type, src_tensor.ne, buft, quant_data, result.owned, error)) {
            result.error = error;
            return result;
        }
        result.upload_ms += (llama_time_us() - t_upload) / 1000.0;
        return result;
    }

    if (moq_disk_cache_ready(src_tensor, target_type, cache, paths)) {
        const int64_t t0 = llama_time_us();
        std::ifstream in(paths.bin_path, std::ios::binary);
        quant_data.resize(result.quant_bytes);
        if (in && !in.read((char *) quant_data.data(), quant_data.size()).fail()) {
            result.cache_hit = true;
            result.disk_cache_hit = true;
            result.disk_cache_load_ms += (llama_time_us() - t0) / 1000.0;
            result.cache_load_ms += result.disk_cache_load_ms;
            cache.stats.hits++;
            cache.stats.disk_hits++;
            cache.stats.bytes_read += quant_data.size();
            moq_mem_cache_insert(cache, paths.cache_key, paths.disk_stem, quant_data);
        } else {
            quant_data.clear();
        }
    } else if (cache.disk_enabled) {
        cache.stats.disk_misses++;
    }

    if (quant_data.empty()) {
        result.cache_miss = true;
        cache.stats.misses++;
        if (!allow_quantize_on_miss) {
            result.error = string_format("disk cache miss after MoQ prequantization: %s", paths.disk_stem.c_str());
            return result;
        }

        const int64_t t0 = llama_time_us();

        std::vector<uint8_t> src_data;
        std::string error;
        if (!source.read_data(src_tensor, src_data, error)) {
            result.error = error;
            return result;
        }

        std::vector<float> f32;
        if (!moq_source_to_f32(src_tensor, src_data, f32, error)) {
            result.error = error;
            return result;
        }
        if (!moq_quantize_f32(target_type, src_tensor.ne, f32, result.imatrix_used ? imatrix : nullptr, quant_data, error)) {
            result.error = error;
            return result;
        }

        result.quantize_ms += (llama_time_us() - t0) / 1000.0;

        if (cache.disk_enabled) {
            std::error_code ec;
            fs::create_directories(cache_dir, ec);
            {
                std::ofstream out(paths.bin_path, std::ios::binary);
                if (out) {
                    out.write((const char *) quant_data.data(), quant_data.size());
                    cache.stats.bytes_written += quant_data.size();
                }
            }
            {
                std::ofstream out(paths.meta_path);
                if (out) {
                    out << moq_cache_metadata_json(src_tensor, target_type, quant_data.size(), cache,
                            result.imatrix_used, paths.effective_imatrix_hash).dump(2) << "\n";
                }
            }
        }
        moq_mem_cache_insert(cache, paths.cache_key, paths.disk_stem, quant_data);
    }

    const int64_t t_upload = llama_time_us();
    std::string error;
    if (!moq_alloc_runtime_tensor(src_tensor.name, target_type, src_tensor.ne, buft, quant_data, result.owned, error)) {
        result.error = error;
        return result;
    }
    result.upload_ms += (llama_time_us() - t_upload) / 1000.0;

    return result;
}

struct moq_prequant_task {
    std::string tensor_name;
    ggml_type type = GGML_TYPE_COUNT;
    const moq_source_tensor * src_tensor = nullptr;
    const std::vector<float> * imatrix = nullptr;
    moq_cache_paths paths;
};

struct moq_prequant_result {
    size_t tasks = 0;
    size_t ready = 0;
    size_t built = 0;
    size_t failed = 0;
    size_t bytes_written = 0;
    double total_ms = 0.0;
    double quantize_ms = 0.0;
    std::vector<std::string> failures;
};

static int moq_prequant_thread_count(const common_params & params) {
    int n_threads = params.cpuparams_batch.n_threads > 0 ? params.cpuparams_batch.n_threads : params.cpuparams.n_threads;
    if (n_threads <= 0) {
        n_threads = (int) std::thread::hardware_concurrency();
    }
    if (n_threads <= 0) {
        n_threads = 1;
    }
    return std::max(1, std::min(n_threads, 16));
}

static bool moq_write_disk_cache_blob(
        const moq_source_tensor & tensor,
        ggml_type type,
        const std::vector<uint8_t> & data,
        const moq_tensor_cache & cache,
        const moq_cache_paths & paths,
        const fs::path & cache_dir,
        std::string & error) {
    if (data.size() != paths.quant_bytes) {
        error = string_format("cache blob byte size mismatch for %s/%s: got %zu expected %zu",
                tensor.name.c_str(), ggml_type_name(type), data.size(), paths.quant_bytes);
        return false;
    }

    std::error_code ec;
    fs::create_directories(cache_dir, ec);
    if (ec) {
        error = string_format("failed to create cache directory %s: %s", cache_dir.string().c_str(), ec.message().c_str());
        return false;
    }

    {
        std::ofstream out(paths.bin_path, std::ios::binary);
        if (!out) {
            error = string_format("failed to write cache blob %s", paths.bin_path.string().c_str());
            return false;
        }
        out.write((const char *) data.data(), data.size());
        if (!out) {
            error = string_format("failed while writing cache blob %s", paths.bin_path.string().c_str());
            return false;
        }
    }

    {
        std::ofstream out(paths.meta_path);
        if (!out) {
            error = string_format("failed to write cache metadata %s", paths.meta_path.string().c_str());
            return false;
        }
        out << moq_cache_metadata_json(tensor, type, data.size(), cache,
                paths.imatrix_used, paths.effective_imatrix_hash).dump(2) << "\n";
        if (!out) {
            error = string_format("failed while writing cache metadata %s", paths.meta_path.string().c_str());
            return false;
        }
    }

    return true;
}

static moq_prequant_result moq_prequantize_disk_cache(
        const moq_source_store & source,
        const moq_tensor_cache & cache,
        const std::vector<moq_prequant_task> & tasks,
        const fs::path & cache_dir,
        int n_threads) {
    moq_prequant_result result;
    result.tasks = tasks.size();
    if (tasks.empty()) {
        return result;
    }

    std::set<ggml_type> init_types;
    for (const auto & task : tasks) {
        if (task.src_tensor != nullptr && ggml_is_quantized(task.type)) {
            init_types.insert(task.type);
        }
    }
    for (ggml_type type : init_types) {
        ggml_quantize_init(type);
    }

    const int64_t t_total = llama_time_us();
    std::atomic<size_t> next{0};
    std::mutex merge_mutex;
    n_threads = std::max(1, std::min<int>(n_threads, (int) tasks.size()));

    auto worker = [&]() {
        size_t local_ready = 0;
        size_t local_built = 0;
        size_t local_failed = 0;
        size_t local_bytes_written = 0;
        double local_quantize_ms = 0.0;
        std::vector<std::string> local_failures;

        for (;;) {
            const size_t index = next.fetch_add(1);
            if (index >= tasks.size()) {
                break;
            }

            const moq_prequant_task & task = tasks[index];
            if (task.src_tensor == nullptr) {
                local_failed++;
                local_failures.push_back(task.tensor_name + ": missing source tensor");
                continue;
            }

            if (moq_disk_cache_ready(*task.src_tensor, task.type, cache, task.paths)) {
                local_ready++;
                continue;
            }

            std::vector<uint8_t> src_data;
            std::vector<float> f32;
            std::vector<uint8_t> quant_data;
            std::string error;
            if (!source.read_data(*task.src_tensor, src_data, error)) {
                local_failed++;
                local_failures.push_back(task.tensor_name + "/" + ggml_type_name(task.type) + ": " + error);
                continue;
            }
            if (!moq_source_to_f32(*task.src_tensor, src_data, f32, error)) {
                local_failed++;
                local_failures.push_back(task.tensor_name + "/" + ggml_type_name(task.type) + ": " + error);
                continue;
            }

            const int64_t t_quant = llama_time_us();
            if (!moq_quantize_f32(task.type, task.src_tensor->ne, f32,
                        task.paths.imatrix_used ? task.imatrix : nullptr, quant_data, error)) {
                local_failed++;
                local_failures.push_back(task.tensor_name + "/" + ggml_type_name(task.type) + ": " + error);
                continue;
            }
            local_quantize_ms += (llama_time_us() - t_quant) / 1000.0;

            if (!moq_write_disk_cache_blob(*task.src_tensor, task.type, quant_data, cache, task.paths, cache_dir, error)) {
                local_failed++;
                local_failures.push_back(task.tensor_name + "/" + ggml_type_name(task.type) + ": " + error);
                continue;
            }

            local_built++;
            local_bytes_written += quant_data.size();
        }

        std::lock_guard<std::mutex> lock(merge_mutex);
        result.ready += local_ready;
        result.built += local_built;
        result.failed += local_failed;
        result.bytes_written += local_bytes_written;
        result.quantize_ms += local_quantize_ms;
        result.failures.insert(result.failures.end(), local_failures.begin(), local_failures.end());
    };

    std::vector<std::thread> workers;
    workers.reserve((size_t) n_threads);
    for (int i = 0; i < n_threads; ++i) {
        workers.emplace_back(worker);
    }
    for (auto & thread : workers) {
        thread.join();
    }

    result.total_ms = (llama_time_us() - t_total) / 1000.0;
    return result;
}

struct moq_sweep_record {
    std::string group;
    std::string qtype;
    std::string status = "ok";
    std::string error;
    int n_tensors = 0;
    int unchanged_tensors = 0;
    int replaced_tensors = 0;
    int restored_tensors = 0;
    int newly_quantized_tensors = 0;
    int cache_hit_tensors = 0;
    int mem_cache_hits = 0;
    int disk_cache_hits = 0;
    int cache_misses = 0;
    size_t source_bytes = 0;
    size_t source_elements = 0;
    size_t quant_bytes = 0;
    double cache_hit_ratio = 0.0;
    double ppl = 0.0;
    double mean_kld = 0.0;
    double max_kld = 0.0;
    double p99_kld = 0.0;
    double p999_kld = 0.0;
    double quantize_ms = 0.0;
    double cache_load_ms = 0.0;
    double mem_cache_load_ms = 0.0;
    double disk_cache_load_ms = 0.0;
    double upload_ms = 0.0;
    double replace_ms = 0.0;
    double eval_ms = 0.0;
    double restore_ms = 0.0;
    double batch_replace_ms = 0.0;
    double batch_restore_ms = 0.0;
    double total_ms = 0.0;
    int diff_saved_replace_count = 0;
    double diff_saved_ms_estimate = 0.0;
    bool imatrix_used = false;
};

static json moq_record_to_json(const moq_sweep_record & r) {
    json j;
    j["group"] = r.group;
    j["qtype"] = r.qtype;
    j["status"] = r.status;
    j["error"] = r.error;
    j["n_tensors"] = r.n_tensors;
    j["unchanged_tensors"] = r.unchanged_tensors;
    j["replaced_tensors"] = r.replaced_tensors;
    j["restored_tensors"] = r.restored_tensors;
    j["newly_quantized_tensors"] = r.newly_quantized_tensors;
    j["cache_hit_tensors"] = r.cache_hit_tensors;
    j["mem_cache_hits"] = r.mem_cache_hits;
    j["disk_cache_hits"] = r.disk_cache_hits;
    j["cache_misses"] = r.cache_misses;
    j["source_bytes"] = r.source_bytes;
    j["source_elements"] = r.source_elements;
    j["quant_bytes"] = r.quant_bytes;
    j["cache_hit_ratio"] = r.cache_hit_ratio;
    j["PPL"] = r.ppl;
    j["mean_KLD"] = r.mean_kld;
    j["max_KLD"] = r.max_kld;
    j["p99_KLD"] = r.p99_kld;
    j["p99_9_KLD"] = r.p999_kld;
    j["quantize_ms"] = r.quantize_ms;
    j["cache_load_ms"] = r.cache_load_ms;
    j["mem_cache_load_ms"] = r.mem_cache_load_ms;
    j["disk_cache_load_ms"] = r.disk_cache_load_ms;
    j["upload_ms"] = r.upload_ms;
    j["replace_ms"] = r.replace_ms;
    j["batch_replace_ms"] = r.batch_replace_ms;
    j["eval_ms"] = r.eval_ms;
    j["restore_ms"] = r.restore_ms;
    j["batch_restore_ms"] = r.batch_restore_ms;
    j["total_ms"] = r.total_ms;
    j["diff_saved_replace_count"] = r.diff_saved_replace_count;
    j["diff_saved_ms_estimate"] = r.diff_saved_ms_estimate;
    j["imatrix_used"] = r.imatrix_used;
    return j;
}

static void moq_write_csv_escaped(std::ostream & out, const std::string & s) {
    const bool quote = s.find_first_of(",\"\n\r") != std::string::npos;
    if (!quote) {
        out << s;
        return;
    }
    out << '"';
    for (char c : s) {
        if (c == '"') {
            out << "\"\"";
        } else {
            out << c;
        }
    }
    out << '"';
}

static void moq_write_results_csv(const fs::path & path, const std::vector<moq_sweep_record> & records) {
    std::ofstream out(path);
    out << "group,qtype,n_tensors,unchanged_tensors,replaced_tensors,restored_tensors,newly_quantized_tensors,cache_hit_tensors,"
        << "mem_cache_hits,disk_cache_hits,cache_misses,source_bytes,source_elements,quant_bytes,cache_hit_ratio,"
        << "PPL,mean KLD,max KLD,p99 KLD,p99.9 KLD,"
        << "quantize_ms,cache_load_ms,mem_cache_load_ms,disk_cache_load_ms,upload_ms,replace_ms,batch_replace_ms,"
        << "eval_ms,restore_ms,batch_restore_ms,total_ms,diff_saved_replace_count,diff_saved_ms_estimate,imatrix_used,status,error\n";
    for (const auto & r : records) {
        moq_write_csv_escaped(out, r.group); out << ',';
        moq_write_csv_escaped(out, r.qtype); out << ',';
        out << r.n_tensors << ','
            << r.unchanged_tensors << ','
            << r.replaced_tensors << ','
            << r.restored_tensors << ','
            << r.newly_quantized_tensors << ','
            << r.cache_hit_tensors << ','
            << r.mem_cache_hits << ','
            << r.disk_cache_hits << ','
            << r.cache_misses << ','
            << r.source_bytes << ','
            << r.source_elements << ','
            << r.quant_bytes << ','
            << r.cache_hit_ratio << ','
            << r.ppl << ','
            << r.mean_kld << ','
            << r.max_kld << ','
            << r.p99_kld << ','
            << r.p999_kld << ','
            << r.quantize_ms << ','
            << r.cache_load_ms << ','
            << r.mem_cache_load_ms << ','
            << r.disk_cache_load_ms << ','
            << r.upload_ms << ','
            << r.replace_ms << ','
            << r.batch_replace_ms << ','
            << r.eval_ms << ','
            << r.restore_ms << ','
            << r.batch_restore_ms << ','
            << r.total_ms << ','
            << r.diff_saved_replace_count << ','
            << r.diff_saved_ms_estimate << ','
            << (r.imatrix_used ? "true" : "false") << ',';
        moq_write_csv_escaped(out, r.status); out << ',';
        moq_write_csv_escaped(out, r.error); out << '\n';
    }
}

static void moq_write_results_json(
        const fs::path & path,
        const std::vector<moq_sweep_record> & records,
        const moq_eval_result & base_before,
        const moq_eval_result & base_after,
        const std::vector<std::string> & warnings) {
    json j;
    j["base_before"] = {
        {"ok", base_before.ok},
        {"PPL", base_before.ppl},
        {"mean_KLD", base_before.mean_kld},
        {"count", base_before.count},
        {"eval_ms", base_before.eval_ms},
    };
    j["base_after"] = {
        {"ok", base_after.ok},
        {"PPL", base_after.ppl},
        {"mean_KLD", base_after.mean_kld},
        {"count", base_after.count},
        {"eval_ms", base_after.eval_ms},
        {"error", base_after.error},
    };
    j["warnings"] = warnings;
    j["results"] = json::array();
    for (const auto & r : records) {
        j["results"].push_back(moq_record_to_json(r));
    }

    std::ofstream out(path);
    out << j.dump(2) << "\n";
}

static void moq_write_timing_csv(const fs::path & path, const std::vector<moq_sweep_record> & records) {
    std::ofstream out(path);
    out << "group,qtype,quantize_ms,cache_load_ms,mem_cache_load_ms,disk_cache_load_ms,upload_ms,replace_ms,batch_replace_ms,eval_ms,restore_ms,batch_restore_ms,total_ms,cache_hit_ratio,status\n";
    for (const auto & r : records) {
        moq_write_csv_escaped(out, r.group); out << ',';
        moq_write_csv_escaped(out, r.qtype); out << ',';
        out << r.quantize_ms << ','
            << r.cache_load_ms << ','
            << r.mem_cache_load_ms << ','
            << r.disk_cache_load_ms << ','
            << r.upload_ms << ','
            << r.replace_ms << ','
            << r.batch_replace_ms << ','
            << r.eval_ms << ','
            << r.restore_ms << ','
            << r.batch_restore_ms << ','
            << r.total_ms << ','
            << r.cache_hit_ratio << ',';
        moq_write_csv_escaped(out, r.status);
        out << '\n';
    }
}

static double moq_average_if(const std::vector<moq_sweep_record> & records, bool warm, double moq_sweep_record::*field) {
    double sum = 0.0;
    int count = 0;
    for (const auto & r : records) {
        if (r.status != "ok") {
            continue;
        }
        const bool is_warm = r.cache_hit_ratio >= 0.999;
        if (is_warm == warm) {
            sum += r.*field;
            count++;
        }
    }
    return count ? sum / count : 0.0;
}

static double moq_average_ok(const std::vector<moq_sweep_record> & records, double moq_sweep_record::*field) {
    double sum = 0.0;
    int count = 0;
    for (const auto & r : records) {
        if (r.status == "ok") {
            sum += r.*field;
            count++;
        }
    }
    return count ? sum / count : 0.0;
}

static double moq_sum_ok(const std::vector<moq_sweep_record> & records, double moq_sweep_record::*field) {
    double sum = 0.0;
    for (const auto & r : records) {
        if (r.status == "ok") {
            sum += r.*field;
        }
    }
    return sum;
}

static int moq_sum_int_ok(const std::vector<moq_sweep_record> & records, int moq_sweep_record::*field) {
    int sum = 0;
    for (const auto & r : records) {
        if (r.status == "ok") {
            sum += r.*field;
        }
    }
    return sum;
}

static double moq_time_per_tensor_ok(const std::vector<moq_sweep_record> & records, int moq_sweep_record::*count_field, double moq_sweep_record::*time_field) {
    double time_ms = 0.0;
    int count = 0;
    for (const auto & r : records) {
        if (r.status != "ok") {
            continue;
        }
        time_ms += r.*time_field;
        count += r.*count_field;
    }
    return count > 0 ? time_ms / count : 0.0;
}

static void moq_write_summary(
        const fs::path & path,
        const common_params & params,
        const moq_source_store & source,
        const moq_imatrix_store * imatrix,
        const moq_tensor_cache & cache,
        const std::vector<moq_qtype_candidate> & candidates,
        const std::vector<moq_sweep_record> & records,
        const moq_eval_result & base_before,
        const moq_eval_result & base_after,
        const std::vector<std::string> & warnings,
        const std::vector<std::string> & failures) {
    std::ofstream out(path);

    const double avg_candidate = moq_average_ok(records, &moq_sweep_record::total_ms);
    const double avg_cold = moq_average_if(records, false, &moq_sweep_record::total_ms);
    const double avg_warm = moq_average_if(records, true,  &moq_sweep_record::total_ms);
    const double avg_eval = moq_average_ok(records, &moq_sweep_record::eval_ms);
    const double avg_quantize = moq_average_ok(records, &moq_sweep_record::quantize_ms);
    const double avg_disk_load = moq_average_ok(records, &moq_sweep_record::disk_cache_load_ms);
    const double avg_mem_load = moq_average_ok(records, &moq_sweep_record::mem_cache_load_ms);
    const double avg_upload = moq_average_ok(records, &moq_sweep_record::upload_ms);
    const double avg_replace = moq_average_ok(records, &moq_sweep_record::replace_ms);
    const double avg_restore = moq_average_ok(records, &moq_sweep_record::restore_ms);
    const double mem_hit_rate = (cache.stats.memory_hits + cache.stats.memory_misses) > 0 ?
        (double) cache.stats.memory_hits / (double) (cache.stats.memory_hits + cache.stats.memory_misses) : 0.0;
    const double disk_hit_rate = (cache.stats.disk_hits + cache.stats.disk_misses) > 0 ?
        (double) cache.stats.disk_hits / (double) (cache.stats.disk_hits + cache.stats.disk_misses) : 0.0;
    const double mem_per_hit = moq_time_per_tensor_ok(records, &moq_sweep_record::mem_cache_hits, &moq_sweep_record::mem_cache_load_ms);
    const double disk_per_hit = moq_time_per_tensor_ok(records, &moq_sweep_record::disk_cache_hits, &moq_sweep_record::disk_cache_load_ms);
    const double memory_speedup = mem_per_hit > 0.0 ? disk_per_hit / mem_per_hit : 0.0;
    const double total_ms = moq_sum_ok(records, &moq_sweep_record::total_ms);
    const double diff_saved_ms = moq_sum_ok(records, &moq_sweep_record::diff_saved_ms_estimate);
    const double diff_speedup = total_ms > 0.0 ? (total_ms + diff_saved_ms) / total_ms : 1.0;

    double cold_quant_once = cache.stats.prequant_ms;
    for (const auto & r : records) {
        if (cache.stats.prequant_ms == 0.0 && r.status == "ok") {
            cold_quant_once += r.quantize_ms;
        }
    }
    const double avg_replace_restore = moq_average_ok(records, &moq_sweep_record::cache_load_ms) +
        moq_average_ok(records, &moq_sweep_record::upload_ms) +
        moq_average_ok(records, &moq_sweep_record::replace_ms) +
        moq_average_ok(records, &moq_sweep_record::restore_ms);
    auto estimate_ms = [&](int candidates) {
        return cold_quant_once + candidates * (avg_eval + avg_replace_restore);
    };

    out << "Model: " << params.model.path << "\n";
    out << "Source BF16: " << source.path << "\n";
    out << "Chunks: " << (params.moq_chunks > 0 ? params.moq_chunks : base_before.n_chunks) << "\n";
    out << "GPU/offload settings: n_gpu_layers=" << params.n_gpu_layers << ", split_mode=" << params.split_mode << "\n";
    out << "Flash Attention: " << llama_flash_attn_type_name(params.flash_attn_type) << "\n";
    out << "Dynamic backend mode: " << params.moq_dynamic_backend << "\n";
    out << "Replace mode: " << params.moq_replace_mode << "\n";
    out << "Sweep order: " << params.moq_sweep_order << "\n";
    out << "CUDA graphs policy: " << params.moq_cuda_graphs << "\n";
    out << "Base logits mode: " << params.moq_base_logits_mode << "\n";
    out << "Profile level: " << params.moq_profile_level << "\n";
    out << "KLD overlap: " << params.moq_kld_overlap << "\n";
    out << "KLD ring: " << params.moq_kld_ring << "\n";
    out << "Logits buffer mode: " << params.moq_logits_buffer_mode << "\n";
    out << "Logits ring slots: " << params.moq_logits_ring << "\n";
    out << "Logits ring pinned: " << params.moq_logits_ring_pinned << "\n";
    out << "KLD workers: " << (params.moq_kld_workers > 0 ? std::to_string(params.moq_kld_workers) : "auto") << "\n";
    out << "Base logits reused: " << (params.logits_file.empty() ? "false" : "true") << "\n";
    out << "Imatrix used: " << (imatrix != nullptr && !imatrix->empty() ? "true" : "false") << "\n";
    out << "Imatrix: " << (imatrix != nullptr && !imatrix->empty() ? imatrix->path : "none") << "\n";
    out << "Imatrix hash: " << (imatrix != nullptr && !imatrix->empty() ? imatrix->imatrix_hash : "none") << "\n";
    out << "Imatrix entries: " << (imatrix != nullptr ? imatrix->data.size() : 0) << "\n\n";

    out << "Base before: ok=" << (base_before.ok ? "true" : "false") << ", PPL=" << base_before.ppl
        << ", mean_KLD=" << base_before.mean_kld << ", eval_ms=" << base_before.eval_ms << "\n";
    out << "Base after restore: ok=" << (base_after.ok ? "true" : "false") << ", PPL=" << base_after.ppl
        << ", mean_KLD=" << base_after.mean_kld << ", eval_ms=" << base_after.eval_ms << "\n\n";

    out << "Average candidate time: " << avg_candidate << " ms\n";
    out << "Average cold candidate time: " << avg_cold << " ms\n";
    out << "Average warm candidate time: " << avg_warm << " ms\n";
    out << "Average eval time: " << avg_eval << " ms\n";
    out << "Average eval-only time: " << avg_eval << " ms\n";
    out << "Average quantize time: " << avg_quantize << " ms\n";
    out << "Average disk cache load time: " << avg_disk_load << " ms\n";
    out << "Average memory cache load time: " << avg_mem_load << " ms\n";
    out << "Average upload time: " << avg_upload << " ms\n";
    out << "Average replace time: " << avg_replace << " ms\n";
    out << "Average restore time: " << avg_restore << " ms\n\n";

    out << "CPU prequant threads: " << cache.stats.prequant_threads << "\n";
    out << "CPU prequant tasks: " << cache.stats.prequant_tasks
        << ", ready=" << cache.stats.prequant_ready
        << ", built=" << cache.stats.prequant_built
        << ", failed=" << cache.stats.prequant_failed << "\n";
    out << "CPU prequant total time: " << cache.stats.prequant_ms << " ms\n";
    out << "CPU prequant quantize time: " << cache.stats.prequant_quantize_ms << " ms\n";
    out << "CPU prequant bytes written: " << cache.stats.prequant_bytes_written << " bytes\n\n";

    out << "Memory cache limit: " << cache.stats.memory_limit_bytes << " bytes\n";
    out << "Memory cache peak: " << cache.stats.memory_peak_bytes << " bytes\n";
    out << "Memory cache hit rate: " << mem_hit_rate << "\n";
    out << "Disk cache hit rate: " << disk_hit_rate << "\n";
    out << "Evictions: " << cache.stats.memory_evictions << "\n";
    out << "Speedup from memory cache: " << (memory_speedup > 0.0 ? string_format("%.3fx", memory_speedup) : "n/a") << "\n";
    out << "Speedup from diff replace: " << diff_speedup << "x, saved_replace_count="
        << moq_sum_int_ok(records, &moq_sweep_record::diff_saved_replace_count) << ", saved_ms_estimate=" << diff_saved_ms << "\n";
    out << "Speedup from batch replace: batch path active; compare batch_replace_ms with older per-tensor runs\n\n";

    out << "Estimated 100-candidate sweep: " << estimate_ms(100)/1000.0 << " s\n";
    out << "Estimated 250-candidate sweep: " << estimate_ms(250)/1000.0 << " s\n";
    out << "Estimated 500-candidate sweep: " << estimate_ms(500)/1000.0 << " s\n\n";

    out << "Estimated 100 candidates: " << estimate_ms(100)/1000.0 << " s\n";
    out << "Estimated 250 candidates: " << estimate_ms(250)/1000.0 << " s\n";
    out << "Estimated 500 candidates: " << estimate_ms(500)/1000.0 << " s\n\n";

    std::vector<moq_sweep_record> slow = records;
    std::sort(slow.begin(), slow.end(), [](const auto & a, const auto & b) { return a.total_ms > b.total_ms; });
    out << "Top slowest groups:\n";
    for (size_t i = 0; i < std::min<size_t>(5, slow.size()); ++i) {
        out << "  " << slow[i].group << " " << slow[i].qtype << " total_ms=" << slow[i].total_ms << "\n";
    }

    std::vector<moq_sweep_record> largest = records;
    std::sort(largest.begin(), largest.end(), [](const auto & a, const auto & b) { return a.source_bytes > b.source_bytes; });
    out << "\nTop largest groups:\n";
    for (size_t i = 0; i < std::min<size_t>(5, largest.size()); ++i) {
        out << "  " << largest[i].group << " " << largest[i].qtype
            << " source_bytes=" << largest[i].source_bytes
            << " quant_bytes=" << largest[i].quant_bytes << "\n";
    }

    std::vector<const moq_mem_cache_entry *> cached;
    cached.reserve(cache.memory.size());
    for (const auto & kv : cache.memory) {
        cached.push_back(&kv.second);
    }
    std::sort(cached.begin(), cached.end(), [](const auto * a, const auto * b) { return a->bytes > b->bytes; });
    out << "\nTop largest cached tensors:\n";
    if (cached.empty()) {
        out << "  none\n";
    } else {
        for (size_t i = 0; i < std::min<size_t>(5, cached.size()); ++i) {
            out << "  " << cached[i]->label << " bytes=" << cached[i]->bytes << "\n";
        }
    }

    out << "\nSupported qtypes:\n";
    for (const auto & info : moq_qtype_registry()) {
        if (info.supported) {
            out << "  " << info.name << (info.requires_imatrix ? " (requires imatrix)" : "") << "\n";
        }
    }

    out << "\nUnsupported qtypes:\n";
    bool any_unsupported = false;
    for (const auto & info : moq_qtype_registry()) {
        if (!info.supported) {
            any_unsupported = true;
            out << "  " << info.name << ": " << info.unsupported_reason << "\n";
        }
    }
    if (!any_unsupported) {
        out << "  none\n";
    }

    out << "\nRequested unsupported qtypes:\n";
    bool any_requested_unsupported = false;
    for (const auto & info : candidates) {
        if (!info.supported) {
            any_requested_unsupported = true;
            out << "  " << info.name << ": " << info.unsupported_reason << "\n";
        }
    }
    if (!any_requested_unsupported) {
        out << "  none\n";
    }

    out << "\nFailures:\n";
    if (failures.empty()) {
        out << "  none\n";
    } else {
        for (const auto & f : failures) {
            out << "  " << f << "\n";
        }
    }

    out << "\nWarnings:\n";
    if (warnings.empty()) {
        out << "  none\n";
    } else {
        for (const auto & w : warnings) {
            out << "  " << w << "\n";
        }
    }

    out << "\nCurrent limitations:\n";
    out << "  Supported dynamic qtypes are the single GGML tensor types listed above; mixture ftypes such as IQ2_M/IQ3_M are recognized but not represented by one tensor type\n";
    out << "  imatrix support: enabled with --moq-imatrix for matching tensor entries; missing entries fall back to unweighted quantization with warnings\n";
    out << "  IQ support: enabled through ggml_quantize_chunk; IQ1_S, IQ2_XXS, and IQ2_XS require imatrix\n";
    out << "  GPU same-backend support: enabled via --moq-dynamic-backend same when the original tensor backend buffer can allocate the replacement tensor\n";
    out << "  Tensor slots: token_embd.weight, output.weight, blk.N.attn_qkv.weight, blk.N.attn_q.weight, blk.N.attn_k.weight, blk.N.attn_v.weight, blk.N.attn_gate.weight, blk.N.attn_output.weight, blk.N.ffn_gate.weight, blk.N.ffn_up.weight, blk.N.ffn_down.weight, blk.N.ssm_alpha.weight, blk.N.ssm_beta.weight, blk.N.ssm_out.weight\n";
}

static void moq_write_cache_stats(const fs::path & path, const moq_tensor_cache & cache) {
    const moq_cache_stats & stats = cache.stats;
    json j;
    j["requests"] = stats.requests;
    j["hits"] = stats.hits;
    j["misses"] = stats.misses;
    j["memory_hits"] = stats.memory_hits;
    j["memory_misses"] = stats.memory_misses;
    j["disk_hits"] = stats.disk_hits;
    j["disk_misses"] = stats.disk_misses;
    j["memory_evictions"] = stats.memory_evictions;
    j["bytes_read"] = stats.bytes_read;
    j["bytes_written"] = stats.bytes_written;
    j["memory_current_bytes"] = stats.memory_current_bytes;
    j["memory_peak_bytes"] = stats.memory_peak_bytes;
    j["memory_limit_bytes"] = stats.memory_limit_bytes;
    j["prequant_tasks"] = stats.prequant_tasks;
    j["prequant_ready"] = stats.prequant_ready;
    j["prequant_built"] = stats.prequant_built;
    j["prequant_failed"] = stats.prequant_failed;
    j["prequant_bytes_written"] = stats.prequant_bytes_written;
    j["prequant_ms"] = stats.prequant_ms;
    j["prequant_quantize_ms"] = stats.prequant_quantize_ms;
    j["prequant_threads"] = stats.prequant_threads;
    j["entries"] = json::array();
    std::vector<const moq_mem_cache_entry *> entries;
    entries.reserve(cache.memory.size());
    for (const auto & kv : cache.memory) {
        entries.push_back(&kv.second);
    }
    std::sort(entries.begin(), entries.end(), [](const auto * a, const auto * b) { return a->bytes > b->bytes; });
    for (const auto * entry : entries) {
        j["entries"].push_back({
            {"label", entry->label},
            {"bytes", entry->bytes},
        });
    }
    std::ofstream out(path);
    out << j.dump(2) << "\n";
}

static void moq_write_failed_candidates_csv(const fs::path & path, const std::vector<moq_sweep_record> & records) {
    std::ofstream out(path);
    out << "group,qtype,status,error\n";
    for (const auto & r : records) {
        if (r.status == "ok") {
            continue;
        }
        moq_write_csv_escaped(out, r.group); out << ',';
        moq_write_csv_escaped(out, r.qtype); out << ',';
        moq_write_csv_escaped(out, r.status); out << ',';
        moq_write_csv_escaped(out, r.error); out << '\n';
    }
}

static double moq_elasticity_loss(const common_params & params, const moq_sweep_record & r) {
    return params.moq_loss_mean_weight * r.mean_kld +
        params.moq_loss_p999_weight * r.p999_kld +
        params.moq_loss_p99_weight * r.p99_kld +
        params.moq_loss_ppl_weight * r.ppl +
        params.moq_loss_max_weight * r.max_kld;
}

static bool moq_qtype_is_high_precision(const std::string & qtype) {
    return qtype == "F32" || qtype == "BF16" || qtype == "F16";
}

static bool moq_qtype_is_low_bit(const std::string & qtype) {
    return qtype.rfind("IQ", 0) == 0 || qtype.rfind("Q2", 0) == 0 || qtype.rfind("Q3", 0) == 0;
}

struct moq_elasticity_row {
    const moq_sweep_record * rec = nullptr;
    double bpw_delta = 0.0;
    double loss = 0.0;
    double loss_vs_best = 0.0;
    double loss_vs_worst = 0.0;
    int64_t extra_bytes_vs_smallest = 0;
    int64_t saved_bytes_vs_largest = 0;
    double gain_vs_smallest = 0.0;
    double gain_per_gib = 0.0;
    double loss_per_gib_saved = 0.0;
    int quality_rank = 0;
    int size_rank = 0;
    bool pareto_dominated = false;
    bool tail_risky = false;
};

struct moq_group_ranking {
    std::string group;
    std::string best_qtype;
    std::string smallest_qtype;
    std::string best_gain_qtype;
    double best_loss = 0.0;
    double smallest_loss = 0.0;
    double p999_best = 0.0;
    double p999_smallest = 0.0;
    size_t total_bytes = 0;
    double elasticity_score = 0.0;
    double tail_risk_score = 0.0;
    double best_gain_per_gib = 0.0;
};

static void moq_write_elasticity_reports(
        const fs::path & out_dir,
        const common_params & params,
        const std::vector<moq_sweep_record> & records,
        const std::vector<moq_qtype_candidate> & candidates) {
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;

    std::vector<moq_elasticity_row> rows;
    rows.reserve(records.size());
    for (const auto & r : records) {
        if (r.status != "ok") {
            continue;
        }
        moq_elasticity_row row;
        row.rec = &r;
        row.bpw_delta = r.source_elements > 0 ? 8.0 * (double) r.quant_bytes / (double) r.source_elements : 0.0;
        row.loss = moq_elasticity_loss(params, r);
        rows.push_back(row);
    }

    std::map<std::string, std::vector<size_t>> by_group;
    std::map<std::string, std::vector<size_t>> by_qtype;
    for (size_t i = 0; i < rows.size(); ++i) {
        by_group[rows[i].rec->group].push_back(i);
        by_qtype[rows[i].rec->qtype].push_back(i);
    }

    std::vector<moq_group_ranking> group_rankings;
    group_rankings.reserve(by_group.size());

    for (const auto & kv : by_group) {
        const std::vector<size_t> & idxs = kv.second;
        if (idxs.empty()) {
            continue;
        }

        size_t best_idx = idxs.front();
        size_t worst_idx = idxs.front();
        size_t smallest_idx = idxs.front();
        size_t largest_idx = idxs.front();
        for (size_t idx : idxs) {
            if (rows[idx].loss < rows[best_idx].loss) {
                best_idx = idx;
            }
            if (rows[idx].loss > rows[worst_idx].loss) {
                worst_idx = idx;
            }
            if (rows[idx].rec->quant_bytes < rows[smallest_idx].rec->quant_bytes) {
                smallest_idx = idx;
            }
            if (rows[idx].rec->quant_bytes > rows[largest_idx].rec->quant_bytes) {
                largest_idx = idx;
            }
        }

        const double best_loss = rows[best_idx].loss;
        const double worst_loss = rows[worst_idx].loss;
        const double smallest_loss = rows[smallest_idx].loss;
        const double largest_loss = rows[largest_idx].loss;
        const double best_p999 = rows[best_idx].rec->p999_kld;
        const size_t smallest_bytes = rows[smallest_idx].rec->quant_bytes;
        const size_t largest_bytes = rows[largest_idx].rec->quant_bytes;

        std::vector<size_t> quality = idxs;
        std::sort(quality.begin(), quality.end(), [&](size_t a, size_t b) {
            return rows[a].loss < rows[b].loss;
        });
        for (size_t rank = 0; rank < quality.size(); ++rank) {
            rows[quality[rank]].quality_rank = (int) rank + 1;
        }

        std::vector<size_t> size_rank = idxs;
        std::sort(size_rank.begin(), size_rank.end(), [&](size_t a, size_t b) {
            return rows[a].rec->quant_bytes < rows[b].rec->quant_bytes;
        });
        for (size_t rank = 0; rank < size_rank.size(); ++rank) {
            rows[size_rank[rank]].size_rank = (int) rank + 1;
        }

        for (size_t idx : idxs) {
            auto & row = rows[idx];
            row.loss_vs_best = row.loss - best_loss;
            row.loss_vs_worst = row.loss - worst_loss;
            row.extra_bytes_vs_smallest = (int64_t) row.rec->quant_bytes - (int64_t) smallest_bytes;
            row.saved_bytes_vs_largest = (int64_t) largest_bytes - (int64_t) row.rec->quant_bytes;
            row.gain_vs_smallest = smallest_loss - row.loss;
            row.gain_per_gib = row.extra_bytes_vs_smallest > 0 ?
                row.gain_vs_smallest / ((double) row.extra_bytes_vs_smallest / gib) : 0.0;
            row.loss_per_gib_saved = row.saved_bytes_vs_largest > 0 ?
                (row.loss - largest_loss) / ((double) row.saved_bytes_vs_largest / gib) : 0.0;
            row.tail_risky = row.rec->p999_kld > std::max(0.01, 2.0 * best_p999);

            for (size_t other_idx : idxs) {
                if (other_idx == idx) {
                    continue;
                }
                const auto & other = rows[other_idx];
                const bool no_larger = other.rec->quant_bytes <= row.rec->quant_bytes;
                const bool no_worse = other.loss <= row.loss;
                const bool strictly_better = other.rec->quant_bytes < row.rec->quant_bytes || other.loss < row.loss;
                if (no_larger && no_worse && strictly_better) {
                    row.pareto_dominated = true;
                    break;
                }
            }
        }

        size_t best_gain_idx = smallest_idx;
        double best_gain_per_gib = -std::numeric_limits<double>::infinity();
        for (size_t idx : idxs) {
            if (rows[idx].extra_bytes_vs_smallest > 0 && rows[idx].gain_per_gib > best_gain_per_gib) {
                best_gain_per_gib = rows[idx].gain_per_gib;
                best_gain_idx = idx;
            }
        }
        if (!std::isfinite(best_gain_per_gib)) {
            best_gain_per_gib = 0.0;
        }

        moq_group_ranking gr;
        gr.group = kv.first;
        gr.best_qtype = rows[best_idx].rec->qtype;
        gr.smallest_qtype = rows[smallest_idx].rec->qtype;
        gr.best_gain_qtype = rows[best_gain_idx].rec->qtype;
        gr.best_loss = best_loss;
        gr.smallest_loss = smallest_loss;
        gr.p999_best = rows[best_idx].rec->p999_kld;
        gr.p999_smallest = rows[smallest_idx].rec->p999_kld;
        gr.total_bytes = rows[largest_idx].rec->source_bytes;
        gr.elasticity_score = smallest_loss - best_loss;
        gr.tail_risk_score = best_p999 > 0.0 ? rows[worst_idx].rec->p999_kld / best_p999 : 0.0;
        gr.best_gain_per_gib = best_gain_per_gib;
        group_rankings.push_back(gr);
    }

    {
        std::ofstream out(out_dir / "elasticity_table.csv");
        out << "group,qtype,n_tensors,source_bytes,quant_bytes,bpw_delta,"
            << "ppl,mean_kld,max_kld,p99_kld,p999_kld,"
            << "loss,loss_vs_best,loss_vs_worst,"
            << "extra_bytes_vs_smallest,saved_bytes_vs_largest,"
            << "gain_vs_smallest,gain_per_gib,loss_per_gib_saved,"
            << "quality_rank,size_rank,pareto_dominated\n";
        for (const auto & row : rows) {
            const auto & r = *row.rec;
            moq_write_csv_escaped(out, r.group); out << ',';
            moq_write_csv_escaped(out, r.qtype); out << ',';
            out << r.n_tensors << ','
                << r.source_bytes << ','
                << r.quant_bytes << ','
                << row.bpw_delta << ','
                << r.ppl << ','
                << r.mean_kld << ','
                << r.max_kld << ','
                << r.p99_kld << ','
                << r.p999_kld << ','
                << row.loss << ','
                << row.loss_vs_best << ','
                << row.loss_vs_worst << ','
                << row.extra_bytes_vs_smallest << ','
                << row.saved_bytes_vs_largest << ','
                << row.gain_vs_smallest << ','
                << row.gain_per_gib << ','
                << row.loss_per_gib_saved << ','
                << row.quality_rank << ','
                << row.size_rank << ','
                << (row.pareto_dominated ? "true" : "false") << '\n';
        }
    }

    {
        json j = json::array();
        for (const auto & row : rows) {
            const auto & r = *row.rec;
            j.push_back({
                {"group", r.group},
                {"qtype", r.qtype},
                {"n_tensors", r.n_tensors},
                {"source_bytes", r.source_bytes},
                {"quant_bytes", r.quant_bytes},
                {"bpw_delta", row.bpw_delta},
                {"ppl", r.ppl},
                {"mean_kld", r.mean_kld},
                {"max_kld", r.max_kld},
                {"p99_kld", r.p99_kld},
                {"p999_kld", r.p999_kld},
                {"loss", row.loss},
                {"loss_vs_best", row.loss_vs_best},
                {"loss_vs_worst", row.loss_vs_worst},
                {"extra_bytes_vs_smallest", row.extra_bytes_vs_smallest},
                {"saved_bytes_vs_largest", row.saved_bytes_vs_largest},
                {"gain_vs_smallest", row.gain_vs_smallest},
                {"gain_per_gib", row.gain_per_gib},
                {"loss_per_gib_saved", row.loss_per_gib_saved},
                {"quality_rank", row.quality_rank},
                {"size_rank", row.size_rank},
                {"pareto_dominated", row.pareto_dominated},
                {"tail_risky", row.tail_risky},
            });
        }
        std::ofstream out(out_dir / "elasticity_table.json");
        out << j.dump(2) << "\n";
    }

    {
        std::ofstream out(out_dir / "group_rankings.csv");
        out << "group,best_qtype,smallest_qtype,best_gain_qtype,"
            << "best_loss,smallest_loss,p999_best,p999_smallest,"
            << "total_bytes,elasticity_score,tail_risk_score\n";
        for (const auto & gr : group_rankings) {
            moq_write_csv_escaped(out, gr.group); out << ',';
            moq_write_csv_escaped(out, gr.best_qtype); out << ',';
            moq_write_csv_escaped(out, gr.smallest_qtype); out << ',';
            moq_write_csv_escaped(out, gr.best_gain_qtype); out << ',';
            out << gr.best_loss << ','
                << gr.smallest_loss << ','
                << gr.p999_best << ','
                << gr.p999_smallest << ','
                << gr.total_bytes << ','
                << gr.elasticity_score << ','
                << gr.tail_risk_score << '\n';
        }
    }

    {
        std::ofstream out(out_dir / "qtype_rankings.csv");
        out << "qtype,avg_loss,avg_p999,avg_ppl,total_quant_bytes,groups_tested,dominated_count\n";
        for (const auto & kv : by_qtype) {
            double loss = 0.0;
            double p999 = 0.0;
            double ppl = 0.0;
            size_t bytes = 0;
            int dominated = 0;
            for (size_t idx : kv.second) {
                loss += rows[idx].loss;
                p999 += rows[idx].rec->p999_kld;
                ppl += rows[idx].rec->ppl;
                bytes += rows[idx].rec->quant_bytes;
                dominated += rows[idx].pareto_dominated ? 1 : 0;
            }
            const double n = (double) kv.second.size();
            moq_write_csv_escaped(out, kv.first); out << ',';
            out << (n > 0.0 ? loss / n : 0.0) << ','
                << (n > 0.0 ? p999 / n : 0.0) << ','
                << (n > 0.0 ? ppl / n : 0.0) << ','
                << bytes << ','
                << kv.second.size() << ','
                << dominated << '\n';
        }
    }

    {
        std::ofstream out(out_dir / "elasticity_summary.txt");
        out << "Loss formula: " << params.moq_loss_mean_weight << "*mean_kld + "
            << params.moq_loss_p999_weight << "*p999_kld + "
            << params.moq_loss_p99_weight << "*p99_kld + "
            << params.moq_loss_ppl_weight << "*ppl + "
            << params.moq_loss_max_weight << "*max_kld\n\n";

        std::vector<moq_group_ranking> sensitive = group_rankings;
        std::sort(sensitive.begin(), sensitive.end(), [](const auto & a, const auto & b) {
            return a.elasticity_score > b.elasticity_score;
        });
        out << "Most sensitive groups:\n";
        for (size_t i = 0; i < std::min<size_t>(5, sensitive.size()); ++i) {
            out << "  " << sensitive[i].group << " score=" << sensitive[i].elasticity_score
                << " best=" << sensitive[i].best_qtype << " smallest=" << sensitive[i].smallest_qtype << "\n";
        }

        std::vector<moq_group_ranking> insensitive = group_rankings;
        std::sort(insensitive.begin(), insensitive.end(), [](const auto & a, const auto & b) {
            return a.elasticity_score < b.elasticity_score;
        });
        out << "\nLeast sensitive groups:\n";
        for (size_t i = 0; i < std::min<size_t>(5, insensitive.size()); ++i) {
            out << "  " << insensitive[i].group << " score=" << insensitive[i].elasticity_score
                << " best=" << insensitive[i].best_qtype << " smallest=" << insensitive[i].smallest_qtype << "\n";
        }

        std::vector<moq_group_ranking> gain = group_rankings;
        std::sort(gain.begin(), gain.end(), [](const auto & a, const auto & b) {
            return a.best_gain_per_gib > b.best_gain_per_gib;
        });
        out << "\nBest gain-per-byte upgrades:\n";
        for (size_t i = 0; i < std::min<size_t>(5, gain.size()); ++i) {
            out << "  " << gain[i].group << " -> " << gain[i].best_gain_qtype
                << " gain_per_gib=" << gain[i].best_gain_per_gib << "\n";
        }

        out << "\nGroups where BF16/F16 is still needed:\n";
        bool any = false;
        for (const auto & gr : group_rankings) {
            if (moq_qtype_is_high_precision(gr.best_qtype)) {
                any = true;
                out << "  " << gr.group << " best=" << gr.best_qtype << " best_loss=" << gr.best_loss << "\n";
            }
        }
        if (!any) {
            out << "  none\n";
        }

        out << "\nGroups where IQ*/Q2/Q3 is acceptable:\n";
        any = false;
        for (const auto & kv : by_group) {
            const auto & idxs = kv.second;
            auto it_smallest = std::min_element(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
                return rows[a].rec->quant_bytes < rows[b].rec->quant_bytes;
            });
            if (it_smallest != idxs.end()) {
                const auto & row = rows[*it_smallest];
                const double threshold = std::max(1e-5, std::abs(row.loss - row.loss_vs_best) * 0.10);
                if (moq_qtype_is_low_bit(row.rec->qtype) && row.loss_vs_best <= threshold) {
                    any = true;
                    out << "  " << row.rec->group << " smallest=" << row.rec->qtype
                        << " loss_vs_best=" << row.loss_vs_best << "\n";
                }
            }
        }
        if (!any) {
            out << "  none\n";
        }

        out << "\nGroups with tail risk:\n";
        any = false;
        for (const auto & row : rows) {
            if (row.tail_risky) {
                any = true;
                out << "  " << row.rec->group << " " << row.rec->qtype
                    << " p999=" << row.rec->p999_kld << " loss=" << row.loss << "\n";
            }
        }
        if (!any) {
            out << "  none\n";
        }

        out << "\nDominated qtypes:\n";
        any = false;
        for (const auto & row : rows) {
            if (row.pareto_dominated) {
                any = true;
                out << "  " << row.rec->group << " " << row.rec->qtype
                    << " bytes=" << row.rec->quant_bytes << " loss=" << row.loss << "\n";
            }
        }
        if (!any) {
            out << "  none\n";
        }

        out << "\nUnsupported qtypes:\n";
        any = false;
        for (const auto & c : candidates) {
            if (!c.supported) {
                any = true;
                out << "  " << c.name << ": " << c.unsupported_reason << "\n";
            }
        }
        if (!any) {
            out << "  none\n";
        }
    }
}

struct moq_sweep_job {
    size_t group_index = 0;
    size_t candidate_index = 0;
};

static std::vector<moq_sweep_job> moq_make_sweep_jobs(
        const std::vector<moq_group> & groups,
        const std::vector<moq_qtype_candidate> & candidates,
        const std::string & order) {
    std::vector<moq_sweep_job> jobs;
    jobs.reserve(groups.size() * candidates.size());

    if (order == "qtype_major") {
        for (size_t ic = 0; ic < candidates.size(); ++ic) {
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                jobs.push_back({ig, ic});
            }
        }
    } else if (order == "size_ascending") {
        std::vector<size_t> idx(candidates.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return moq_qtype_sort_bpw(candidates[a]) < moq_qtype_sort_bpw(candidates[b]);
        });
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            for (size_t ic : idx) {
                jobs.push_back({ig, ic});
            }
        }
    } else {
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            for (size_t ic = 0; ic < candidates.size(); ++ic) {
                jobs.push_back({ig, ic});
            }
        }
    }

    return jobs;
}

struct moq_recipe_def {
    std::string path;
    std::string name;
    std::string solver;
    double target_bpw = 0.0;
    double estimated_bpw = 0.0;
    double absolute_model_bpw = 0.0;
    double predicted_loss = 0.0;
    std::map<std::string, std::string> groups;
    bool diagnostic_only = false;
};

struct moq_recipe_validation_record {
    std::string recipe;
    std::string path;
    std::string solver;
    std::string status = "ok";
    std::string error;
    double target_bpw = 0.0;
    double estimated_bpw = 0.0;
    double absolute_model_bpw = 0.0;
    double predicted_loss = 0.0;
    double actual_loss = 0.0;
    double actual_ppl = 0.0;
    double actual_mean_kld = 0.0;
    double actual_p99_kld = 0.0;
    double actual_p999_kld = 0.0;
    double actual_max_kld = 0.0;
    int n_groups = 0;
    int n_tensors = 0;
    size_t total_source_bytes = 0;
    size_t total_quant_bytes = 0;
    double cache_load_ms = 0.0;
    double mem_cache_load_ms = 0.0;
    double disk_cache_load_ms = 0.0;
    double quantize_ms = 0.0;
    double upload_ms = 0.0;
    double replace_ms = 0.0;
    double eval_ms = 0.0;
    double restore_ms = 0.0;
    double total_ms = 0.0;
};

static std::vector<std::string> moq_read_recipe_paths(const common_params & params) {
    std::vector<std::string> paths;
    if (!params.moq_recipe.empty()) {
        paths.push_back(params.moq_recipe);
    }
    if (!params.moq_recipe_list.empty()) {
        std::ifstream in(params.moq_recipe_list.c_str());
        if (!in) {
            throw std::runtime_error(string_format("failed to open MoQ recipe list: %s", params.moq_recipe_list.c_str()));
        }
        const fs::path base_dir = fs::path(params.moq_recipe_list).parent_path();
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            line = string_strip(line);
            if (line.empty() || line[0] == '#') {
                continue;
            }
            fs::path p(line);
            if (p.is_relative() && !fs::exists(p) && !base_dir.empty()) {
                p = base_dir / p;
            }
            paths.push_back(p.string());
        }
    }
    return paths;
}

static bool moq_parse_recipe(const std::string & path, moq_recipe_def & recipe, std::string & error) {
    std::ifstream in(path.c_str());
    if (!in) {
        error = string_format("failed to open MoQ recipe JSON: %s", path.c_str());
        return false;
    }
    try {
        json doc = json::parse(in);
        recipe = {};
        recipe.path = path;
        const std::string stem = fs::path(path).stem().string();
        recipe.name = doc.value("name", stem);
        if (stem.find("_rank_") != std::string::npos) {
            recipe.name = stem;
        }
        recipe.solver = doc.value("solver", "");
        recipe.target_bpw = doc.value("target_bpw", 0.0);
        recipe.estimated_bpw = doc.value("estimated_bpw", 0.0);
        recipe.absolute_model_bpw = doc.value("absolute_model_bpw", recipe.estimated_bpw);
        recipe.predicted_loss = doc.value("loss", 0.0);
        recipe.diagnostic_only = doc.value("diagnostic_only", false);
        if (!doc.contains("groups") || !doc["groups"].is_object()) {
            error = string_format("recipe %s missing object field 'groups'", path.c_str());
            return false;
        }
        for (auto it = doc["groups"].begin(); it != doc["groups"].end(); ++it) {
            if (!it.value().is_string()) {
                error = string_format("recipe %s group %s qtype is not a string", path.c_str(), it.key().c_str());
                return false;
            }
            recipe.groups[it.key()] = it.value().get<std::string>();
        }
        if (recipe.groups.empty()) {
            error = string_format("recipe %s contains no groups", path.c_str());
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        error = string_format("failed to parse MoQ recipe JSON %s: %s", path.c_str(), e.what());
        return false;
    }
}

static std::map<std::string, const moq_group *> moq_make_group_lookup(const std::vector<moq_group> & groups) {
    std::map<std::string, const moq_group *> out;
    for (const auto & group : groups) {
        out[group.name] = &group;
    }
    return out;
}

static void moq_write_recipe_validation_csv(const fs::path & path, const std::vector<moq_recipe_validation_record> & records) {
    std::ofstream out(path);
    out << "recipe,solver,target_bpw,estimated_bpw,absolute_model_bpw,"
        << "actual_ppl,actual_mean_kld,actual_max_kld,actual_p999_kld,actual_loss,predicted_loss,"
        << "n_groups,n_tensors,total_quant_bytes,status,error,"
        << "actual_p99_kld,total_source_bytes,cache_load_ms,mem_cache_load_ms,disk_cache_load_ms,"
        << "quantize_ms,upload_ms,replace_ms,eval_ms,restore_ms,total_ms\n";
    for (const auto & r : records) {
        moq_write_csv_escaped(out, r.recipe); out << ',';
        moq_write_csv_escaped(out, r.solver); out << ',';
        out << r.target_bpw << ','
            << r.estimated_bpw << ','
            << r.absolute_model_bpw << ','
            << r.actual_ppl << ','
            << r.actual_mean_kld << ','
            << r.actual_max_kld << ','
            << r.actual_p999_kld << ','
            << r.actual_loss << ','
            << r.predicted_loss << ','
            << r.n_groups << ','
            << r.n_tensors << ','
            << r.total_quant_bytes << ',';
        moq_write_csv_escaped(out, r.status); out << ',';
        moq_write_csv_escaped(out, r.error); out << ',';
        out << r.actual_p99_kld << ','
            << r.total_source_bytes << ','
            << r.cache_load_ms << ','
            << r.mem_cache_load_ms << ','
            << r.disk_cache_load_ms << ','
            << r.quantize_ms << ','
            << r.upload_ms << ','
            << r.replace_ms << ','
            << r.eval_ms << ','
            << r.restore_ms << ','
            << r.total_ms << "\n";
    }
}

static void moq_write_recipe_validation_json(
        const fs::path & path,
        const std::vector<moq_recipe_validation_record> & records,
        const moq_eval_result & base_before,
        const moq_eval_result & base_after) {
    json j;
    j["base_before"] = {
        {"ok", base_before.ok},
        {"ppl", base_before.ppl},
        {"mean_kld", base_before.mean_kld},
        {"p999_kld", base_before.p999_kld},
    };
    j["base_after"] = {
        {"ok", base_after.ok},
        {"ppl", base_after.ppl},
        {"mean_kld", base_after.mean_kld},
        {"p999_kld", base_after.p999_kld},
    };
    j["recipes"] = json::array();
    for (const auto & r : records) {
        j["recipes"].push_back({
            {"recipe", r.recipe},
            {"path", r.path},
            {"solver", r.solver},
            {"target_bpw", r.target_bpw},
            {"estimated_bpw", r.estimated_bpw},
            {"absolute_model_bpw", r.absolute_model_bpw},
            {"predicted_loss", r.predicted_loss},
            {"actual_loss", r.actual_loss},
            {"actual_ppl", r.actual_ppl},
            {"actual_mean_kld", r.actual_mean_kld},
            {"actual_p99_kld", r.actual_p99_kld},
            {"actual_p999_kld", r.actual_p999_kld},
            {"actual_max_kld", r.actual_max_kld},
            {"n_groups", r.n_groups},
            {"n_tensors", r.n_tensors},
            {"total_quant_bytes", r.total_quant_bytes},
            {"status", r.status},
            {"error", r.error},
        });
    }
    std::ofstream out(path);
    out << std::setw(2) << j << "\n";
}

static void moq_write_recipe_actual_best_by_bpw(const fs::path & path, std::vector<moq_recipe_validation_record> records) {
    records.erase(std::remove_if(records.begin(), records.end(), [](const moq_recipe_validation_record & r) {
        return r.status != "ok";
    }), records.end());

    std::map<double, std::vector<moq_recipe_validation_record>> by_target;
    for (const auto & r : records) {
        by_target[r.target_bpw].push_back(r);
    }

    std::ofstream out(path);
    out << "target_bpw,recipe,solver,estimated_bpw,absolute_model_bpw,predicted_loss,actual_loss,actual_ppl,actual_mean_kld,actual_p99_kld,actual_p999_kld,actual_max_kld,n_groups,n_tensors,total_quant_bytes\n";
    for (const auto & kv : by_target) {
        if (kv.second.empty()) {
            continue;
        }
        const auto & best = *std::min_element(kv.second.begin(), kv.second.end(), [](const auto & a, const auto & b) {
            if (a.actual_loss != b.actual_loss) {
                return a.actual_loss < b.actual_loss;
            }
            return a.predicted_loss < b.predicted_loss;
        });
        out << kv.first << ',';
        moq_write_csv_escaped(out, best.recipe); out << ',';
        moq_write_csv_escaped(out, best.solver); out << ',';
        out << best.estimated_bpw << ','
            << best.absolute_model_bpw << ','
            << best.predicted_loss << ','
            << best.actual_loss << ','
            << best.actual_ppl << ','
            << best.actual_mean_kld << ','
            << best.actual_p99_kld << ','
            << best.actual_p999_kld << ','
            << best.actual_max_kld << ','
            << best.n_groups << ','
            << best.n_tensors << ','
            << best.total_quant_bytes << "\n";
    }
}

static double moq_recipe_actual_loss(const common_params & params, const moq_recipe_validation_record & r) {
    return params.moq_loss_mean_weight * r.actual_mean_kld +
        params.moq_loss_p999_weight * r.actual_p999_kld +
        params.moq_loss_p99_weight * r.actual_p99_kld +
        params.moq_loss_ppl_weight * r.actual_ppl +
        params.moq_loss_max_weight * r.actual_max_kld;
}

static void moq_write_recipe_prediction_compare(const fs::path & path, const common_params & params, std::vector<moq_recipe_validation_record> records) {
    records.erase(std::remove_if(records.begin(), records.end(), [](const moq_recipe_validation_record & r) {
        return r.status != "ok";
    }), records.end());

    std::ofstream out(path);
    out << "MoQ recipe prediction compare\n\n";
    out << "Loss formula: " << params.moq_loss_mean_weight << "*mean_kld + "
        << params.moq_loss_p999_weight << "*p999_kld + "
        << params.moq_loss_p99_weight << "*p99_kld + "
        << params.moq_loss_ppl_weight << "*ppl + "
        << params.moq_loss_max_weight << "*max_kld\n\n";
    if (records.empty()) {
        out << "No successful recipe validation records.\n";
        return;
    }

    for (auto & r : records) {
        r.actual_loss = moq_recipe_actual_loss(params, r);
    }
    std::vector<moq_recipe_validation_record> by_pred = records;
    std::vector<moq_recipe_validation_record> by_actual = records;
    std::sort(by_pred.begin(), by_pred.end(), [](const auto & a, const auto & b) {
        return a.predicted_loss < b.predicted_loss;
    });
    std::sort(by_actual.begin(), by_actual.end(), [](const auto & a, const auto & b) {
        return a.actual_loss < b.actual_loss;
    });

    int same_rank = 0;
    const size_t n = std::min(by_pred.size(), by_actual.size());
    for (size_t i = 0; i < n; ++i) {
        same_rank += by_pred[i].recipe == by_actual[i].recipe ? 1 : 0;
    }

    out << "1. Predicted vs actual ranking:\n";
    out << "  exact_same_rank_positions=" << same_rank << "/" << n << "\n";
    if (n >= 2) {
        std::map<std::string, double> pred_rank;
        std::map<std::string, double> actual_rank;
        for (size_t i = 0; i < n; ++i) {
            pred_rank[by_pred[i].recipe] = (double) i + 1.0;
            actual_rank[by_actual[i].recipe] = (double) i + 1.0;
        }
        double sum_d2 = 0.0;
        for (const auto & r : records) {
            const double d = pred_rank[r.recipe] - actual_rank[r.recipe];
            sum_d2 += d * d;
        }
        const double spearman = 1.0 - 6.0 * sum_d2 / ((double) n * ((double) n * (double) n - 1.0));
        out << "  spearman_rank_correlation=" << spearman << "\n";
    }
    out << "  best_predicted=" << by_pred.front().recipe << " predicted_loss=" << by_pred.front().predicted_loss
        << " actual_loss=" << by_pred.front().actual_loss << "\n";
    out << "  best_actual=" << by_actual.front().recipe << " predicted_loss=" << by_actual.front().predicted_loss
        << " actual_loss=" << by_actual.front().actual_loss << "\n\n";

    out << "2. Lambda vs greedy by target:\n";
    std::map<double, std::vector<moq_recipe_validation_record>> by_target;
    for (const auto & r : records) {
        by_target[r.target_bpw].push_back(r);
    }
    for (const auto & kv : by_target) {
        if (kv.second.empty()) {
            continue;
        }
        const auto & best = *std::min_element(kv.second.begin(), kv.second.end(), [](const auto & a, const auto & b) {
            return a.actual_loss < b.actual_loss;
        });
        out << "  target_bpw=" << kv.first << " best_actual=" << best.recipe
            << " solver=" << best.solver << " actual_loss=" << best.actual_loss
            << " p999=" << best.actual_p999_kld << "\n";
        std::vector<moq_recipe_validation_record> lambda_rows;
        std::vector<moq_recipe_validation_record> greedy_rows;
        for (const auto & r : kv.second) {
            if (r.solver == "lambda") {
                lambda_rows.push_back(r);
            } else if (r.solver == "greedy") {
                greedy_rows.push_back(r);
            }
        }
        if (!lambda_rows.empty() && !greedy_rows.empty()) {
            const auto & lambda = *std::min_element(lambda_rows.begin(), lambda_rows.end(), [](const auto & a, const auto & b) {
                return a.actual_loss < b.actual_loss;
            });
            const auto & greedy = *std::min_element(greedy_rows.begin(), greedy_rows.end(), [](const auto & a, const auto & b) {
                return a.actual_loss < b.actual_loss;
            });
            const double delta = greedy.actual_loss - lambda.actual_loss;
            if (std::abs(delta) < 1e-9) {
                out << "    lambda_vs_greedy=tie\n";
            } else {
                out << "    lambda_vs_greedy=" << (delta < 0 ? "greedy_better" : "lambda_better")
                    << " actual_loss_delta_greedy_best_minus_lambda_best=" << delta
                    << " lambda_best=" << lambda.recipe
                    << " greedy_best=" << greedy.recipe << "\n";
            }
        }
    }

    out << "\n3. Best value BPW step:\n";
    std::vector<moq_recipe_validation_record> target_best;
    for (const auto & kv : by_target) {
        target_best.push_back(*std::min_element(kv.second.begin(), kv.second.end(), [](const auto & a, const auto & b) {
            return a.actual_loss < b.actual_loss;
        }));
    }
    std::sort(target_best.begin(), target_best.end(), [](const auto & a, const auto & b) {
        return a.target_bpw < b.target_bpw;
    });
    double best_gain = -1.0;
    std::string best_step = "n/a";
    for (size_t i = 1; i < target_best.size(); ++i) {
        const double db = target_best[i].target_bpw - target_best[i - 1].target_bpw;
        const double gain = target_best[i - 1].actual_loss - target_best[i].actual_loss;
        if (db > 1e-9 && gain / db > best_gain) {
            best_gain = gain / db;
            best_step = target_best[i - 1].recipe + " -> " + target_best[i].recipe;
        }
    }
    out << "  best_actual_loss_reduction_per_target_bpw=" << best_gain << " step=" << best_step << "\n\n";

    out << "4. Predicted good but actual tail risky:\n";
    std::vector<double> tails;
    for (const auto & r : records) {
        tails.push_back(r.actual_p999_kld);
    }
    std::sort(tails.begin(), tails.end());
    const double median_tail = tails[tails.size() / 2];
    int risky = 0;
    for (const auto & r : by_pred) {
        if (r.actual_p999_kld > median_tail * 2.0 && r.actual_p999_kld > 1e-4) {
            out << "  " << r.recipe << " predicted_loss=" << r.predicted_loss
                << " actual_p999=" << r.actual_p999_kld << "\n";
            risky++;
        }
    }
    if (risky == 0) {
        out << "  none\n";
    }

    out << "\n5. Guard recommendation:\n";
    if (risky > 0 || same_rank < (int) n / 2) {
        out << "  Add a p999 guard or group interaction penalty before using solver output as final recipe.\n";
    } else {
        out << "  Current predicted ordering is usable for prescreening; still validate final recipes with recipe sweep.\n";
    }
}

static void moq_write_recipe_validation_summary(
        const fs::path & path,
        const common_params & params,
        const std::vector<moq_recipe_validation_record> & records,
        const std::vector<std::string> & warnings) {
    std::ofstream out(path);
    out << "MoQ recipe validation summary\n\n";
    out << "Recipes tested: " << records.size() << "\n";
    int ok = 0;
    for (const auto & r : records) {
        ok += r.status == "ok" ? 1 : 0;
    }
    out << "Successful: " << ok << "\n";
    out << "Failed: " << (records.size() - ok) << "\n";
    out << "Groups file: " << params.moq_groups << "\n";
    out << "Base logits mode: " << params.moq_base_logits_mode << "\n";
    out << "Dynamic backend: " << params.moq_dynamic_backend << "\n";
    out << "KLD overlap: " << params.moq_kld_overlap << "\n";
    out << "Logits buffer mode: " << params.moq_logits_buffer_mode << "\n";
    out << "Logits ring: " << params.moq_logits_ring << "\n";
    out << "Logits ring pinned: " << params.moq_logits_ring_pinned << "\n\n";
    out << "Records:\n";
    for (const auto & r : records) {
        out << "  " << r.recipe << " status=" << r.status
            << " target_bpw=" << r.target_bpw
            << " estimated_bpw=" << r.estimated_bpw
            << " predicted_loss=" << r.predicted_loss
            << " actual_mean_kld=" << r.actual_mean_kld
            << " actual_p999=" << r.actual_p999_kld
            << " tensors=" << r.n_tensors
            << " total_ms=" << r.total_ms << "\n";
        if (!r.error.empty()) {
            out << "    error=" << r.error << "\n";
        }
    }
    out << "\nWarnings:\n";
    if (warnings.empty()) {
        out << "  none\n";
    } else {
        for (const auto & w : warnings) {
            out << "  " << w << "\n";
        }
    }
}

static int moq_recipe_validation(llama_context * ctx, llama_model * model, const common_params & params) {
    std::error_code ec;
    fs::create_directories(params.moq_output, ec);
    fs::create_directories(params.moq_cache_dir, ec);

    std::string error;
    moq_source_store source;
    if (!source.open(params.moq_source_bf16, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }

    std::vector<moq_group> groups;
    if (!moq_parse_groups(params.moq_groups, groups, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }
    const std::map<std::string, const moq_group *> group_lookup = moq_make_group_lookup(groups);

    std::vector<moq_recipe_def> recipes;
    try {
        for (const auto & path : moq_read_recipe_paths(params)) {
            moq_recipe_def recipe;
            if (!moq_parse_recipe(path, recipe, error)) {
                LOG_ERR("%s: %s\n", __func__, error.c_str());
                return 1;
            }
            recipes.push_back(std::move(recipe));
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return 1;
    }
    if (recipes.empty()) {
        LOG_ERR("%s: no recipes were provided\n", __func__);
        return 1;
    }

    std::unique_ptr<moq_imatrix_store> imatrix;
    if (!params.moq_imatrix.empty()) {
        imatrix = std::make_unique<moq_imatrix_store>();
        if (!imatrix->open(params.moq_imatrix, error)) {
            LOG_ERR("%s: %s\n", __func__, error.c_str());
            return 1;
        }
    }

    moq_slot_registry registry;
    if (!moq_register_tensor_slots(*model, registry)) {
        LOG_ERR("%s: failed to register MoQ tensor slots\n", __func__);
        return 1;
    }

    moq_tensor_cache cache;
    cache.source_hash = source.source_hash;
    cache.disk_enabled = !params.moq_disable_disk_cache;
    cache.mem_enabled = !params.moq_disable_mem_cache && params.moq_mem_cache_mb > 0;
    cache.stats.memory_limit_bytes = cache.mem_enabled ? (size_t) params.moq_mem_cache_mb * 1024ull * 1024ull : 0;
    if (imatrix && !imatrix->empty()) {
        cache.imatrix_hash = imatrix->imatrix_hash;
    }

    std::vector<std::string> warnings;
    std::vector<std::string> failures;

    std::map<std::string, moq_qtype_candidate> qtype_by_name;
    std::vector<moq_prequant_task> prequant_tasks;
    std::set<std::string> seen_prequant_keys;

    for (const auto & recipe : recipes) {
        for (const auto & gq : recipe.groups) {
            auto git = group_lookup.find(gq.first);
            if (git == group_lookup.end()) {
                failures.push_back(recipe.name + ": recipe group not found in --moq-groups: " + gq.first);
                continue;
            }
            moq_qtype_candidate candidate;
            if (!moq_parse_qtype(gq.second, candidate) || !candidate.supported) {
                failures.push_back(recipe.name + "/" + gq.first + ": unsupported qtype " + gq.second);
                continue;
            }
            qtype_by_name[candidate.name] = candidate;
            if (!cache.disk_enabled || !params.moq_prebuild_cache) {
                continue;
            }
            for (const auto & tensor_name : git->second->tensors) {
                ggml_tensor * base_tensor = moq_get_base_tensor(registry, tensor_name);
                const moq_source_tensor * src_tensor = source.get(tensor_name);
                if (base_tensor == nullptr || src_tensor == nullptr || !moq_same_shape(src_tensor->ne, base_tensor)) {
                    continue;
                }
                const std::vector<float> * imatrix_data = moq_select_imatrix(
                        imatrix.get(), tensor_name, *src_tensor, candidate.type, nullptr);
                moq_cache_paths paths = moq_make_cache_paths(*src_tensor, candidate.type, cache, imatrix_data, params.moq_cache_dir);
                if (!seen_prequant_keys.insert(paths.cache_key).second) {
                    continue;
                }
                moq_prequant_task task;
                task.tensor_name = tensor_name;
                task.type = candidate.type;
                task.src_tensor = src_tensor;
                task.imatrix = imatrix_data;
                task.paths = std::move(paths);
                prequant_tasks.push_back(std::move(task));
            }
        }
    }

    if (!failures.empty()) {
        for (const auto & f : failures) {
            LOG_ERR("%s: %s\n", __func__, f.c_str());
        }
        return 1;
    }

    if (cache.disk_enabled && params.moq_prebuild_cache) {
        const int prequant_threads = moq_prequant_thread_count(params);
        cache.stats.prequant_threads = prequant_threads;
        LOG_INF("%s: CPU recipe prequant phase start: tasks=%zu threads=%d cache_dir=%s\n",
                __func__, prequant_tasks.size(), prequant_threads, params.moq_cache_dir.c_str());
        const moq_prequant_result prequant = moq_prequantize_disk_cache(
                source, cache, prequant_tasks, params.moq_cache_dir, prequant_threads);
        cache.stats.prequant_tasks = prequant.tasks;
        cache.stats.prequant_ready = prequant.ready;
        cache.stats.prequant_built = prequant.built;
        cache.stats.prequant_failed = prequant.failed;
        cache.stats.prequant_bytes_written = prequant.bytes_written;
        cache.stats.prequant_ms = prequant.total_ms;
        cache.stats.prequant_quantize_ms = prequant.quantize_ms;
        cache.stats.bytes_written += prequant.bytes_written;
        for (size_t i = 0; i < std::min<size_t>(20, prequant.failures.size()); ++i) {
            warnings.push_back("prequant failed: " + prequant.failures[i]);
        }
    }

    const bool allow_runtime_quantize = !cache.disk_enabled;
    moq_base_logits_store base_logits;
    if (!base_logits.open(params.logits_file, params.moq_base_logits_mode, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }
    if ((int) base_logits.n_vocab != llama_vocab_n_tokens(llama_model_get_vocab(model))) {
        LOG_ERR("%s: base logits vocabulary mismatch: logits=%d model=%d\n",
                __func__, base_logits.n_vocab, llama_vocab_n_tokens(llama_model_get_vocab(model)));
        return 1;
    }

    moq_eval_profiler profiler;
    if (!profiler.open(params.moq_output, params.moq_profile_level, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }

    moq_restore_all(registry);
    moq_invalidate_graph(ctx);
    moq_eval_result base_before = kl_divergence_eval_once(
            ctx, params, base_logits, params.moq_chunks, false, &profiler, {-1, "recipe_base_before", "base"});
    if (!base_before.ok) {
        LOG_ERR("%s: base KLD evaluation failed: %s\n", __func__, base_before.error.c_str());
        return 1;
    }

    std::vector<moq_recipe_validation_record> records;
    for (size_t irecipe = 0; irecipe < recipes.size(); ++irecipe) {
        const moq_recipe_def & recipe = recipes[irecipe];
        const int64_t t_total = llama_time_us();
        moq_recipe_validation_record rec;
        rec.recipe = recipe.name;
        rec.path = recipe.path;
        rec.solver = recipe.solver;
        rec.target_bpw = recipe.target_bpw;
        rec.estimated_bpw = recipe.estimated_bpw;
        rec.absolute_model_bpw = recipe.absolute_model_bpw;
        rec.predicted_loss = recipe.predicted_loss;
        rec.n_groups = (int) recipe.groups.size();

        std::map<std::string, std::string> tensor_assignment;
        for (const auto & gq : recipe.groups) {
            auto git = group_lookup.find(gq.first);
            if (git == group_lookup.end()) {
                rec.status = "failed";
                rec.error = "recipe group not found in --moq-groups: " + gq.first;
                break;
            }
            for (const auto & tensor_name : git->second->tensors) {
                auto inserted = tensor_assignment.emplace(tensor_name, gq.second);
                if (!inserted.second && inserted.first->second != gq.second) {
                    rec.status = "failed";
                    rec.error = string_format("overlapping tensor %s assigned both %s and %s",
                            tensor_name.c_str(), inserted.first->second.c_str(), gq.second.c_str());
                    break;
                }
                if (!inserted.second && inserted.first->second == gq.second) {
                    rec.status = "failed";
                    rec.error = string_format("overlapping tensor %s appears more than once in recipe", tensor_name.c_str());
                    break;
                }
            }
            if (rec.status != "ok") {
                break;
            }
        }

        std::vector<moq_replacement> replacements;
        std::vector<std::shared_ptr<moq_owned_tensor>> owned_tensors;
        if (rec.status == "ok") {
            for (const auto & tq : tensor_assignment) {
                moq_qtype_candidate candidate;
                if (!moq_parse_qtype(tq.second, candidate) || !candidate.supported) {
                    rec.status = "failed";
                    rec.error = "unsupported qtype: " + tq.second;
                    break;
                }
                ggml_tensor * base_tensor = moq_get_base_tensor(registry, tq.first);
                const moq_source_tensor * src_tensor = source.get(tq.first);
                if (base_tensor == nullptr || src_tensor == nullptr || !moq_same_shape(src_tensor->ne, base_tensor)) {
                    rec.status = "failed";
                    rec.error = "tensor missing or shape mismatch: " + tq.first;
                    break;
                }

                ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
                if (params.moq_dynamic_backend == "same") {
                    buft = moq_get_base_tensor_buft(registry, tq.first);
                }
                std::string imatrix_warning;
                const std::vector<float> * imatrix_data = moq_select_imatrix(
                        imatrix.get(), tq.first, *src_tensor, candidate.type, &imatrix_warning);
                if (!imatrix_warning.empty()) {
                    warnings.push_back(recipe.name + "/" + tq.first + ": " + imatrix_warning);
                }
                moq_tensor_build_result built = moq_get_or_build_tensor(source, cache, *src_tensor, candidate.type,
                        imatrix_data, buft, params.moq_cache_dir, allow_runtime_quantize);
                rec.quantize_ms += built.quantize_ms;
                rec.cache_load_ms += built.cache_load_ms;
                rec.mem_cache_load_ms += built.mem_cache_load_ms;
                rec.disk_cache_load_ms += built.disk_cache_load_ms;
                rec.upload_ms += built.upload_ms;
                rec.total_source_bytes += src_tensor->nbytes;
                rec.total_quant_bytes += moq_quant_nbytes(candidate.type, src_tensor->ne);
                if (!built.error.empty() || !built.owned || built.owned->tensor == nullptr) {
                    rec.status = "failed";
                    rec.error = tq.first + ": " + (built.error.empty() ? "failed to build tensor" : built.error);
                    break;
                }
                replacements.push_back({tq.first, built.owned->tensor});
                owned_tensors.push_back(std::move(built.owned));
            }
        }

        rec.n_tensors = (int) replacements.size();
        if (rec.status == "ok") {
            const int64_t t_replace = llama_time_us();
            if (!moq_replace_tensor_batch(registry, replacements)) {
                rec.status = "failed";
                rec.error = "batch replace failed";
            }
            rec.replace_ms = (llama_time_us() - t_replace) / 1000.0;
            if (rec.status == "ok") {
                moq_invalidate_graph(ctx);
                moq_eval_result eval = kl_divergence_eval_once(
                        ctx, params, base_logits, params.moq_chunks, false, &profiler,
                        {(int) irecipe, recipe.name, recipe.solver});
                rec.eval_ms = eval.eval_ms;
                if (!eval.ok) {
                    rec.status = "failed";
                    rec.error = eval.error;
                } else {
                    rec.actual_ppl = eval.ppl;
                    rec.actual_mean_kld = eval.mean_kld;
                    rec.actual_p99_kld = eval.p99_kld;
                    rec.actual_p999_kld = eval.p999_kld;
                    rec.actual_max_kld = eval.max_kld;
                    rec.actual_loss = moq_recipe_actual_loss(params, rec);
                    LOG_INF("%s: recipe=%s PPL=%.6lf mean_KLD=%.6lf p999=%.6lf tensors=%d\n",
                            __func__, recipe.name.c_str(), rec.actual_ppl, rec.actual_mean_kld,
                            rec.actual_p999_kld, rec.n_tensors);
                }
            }
        }

        const int64_t t_restore = llama_time_us();
        moq_restore_all(registry);
        moq_invalidate_graph(ctx);
        rec.restore_ms = (llama_time_us() - t_restore) / 1000.0;
        owned_tensors.clear();
        rec.total_ms = (llama_time_us() - t_total) / 1000.0;
        if (rec.status != "ok") {
            failures.push_back(recipe.name + ": " + rec.error);
        }
        records.push_back(rec);
    }

    moq_restore_all(registry);
    moq_invalidate_graph(ctx);
    moq_eval_result base_after = kl_divergence_eval_once(
            ctx, params, base_logits, params.moq_chunks, false, &profiler, {-2, "recipe_base_after", "base"});
    if (!base_after.ok) {
        warnings.push_back("base-after-restore evaluation failed: " + base_after.error);
    }

    const fs::path out_dir(params.moq_output);
    moq_write_recipe_validation_csv(out_dir / "recipe_validation.csv", records);
    moq_write_recipe_validation_csv(out_dir / "recipe_validation_topk.csv", records);
    moq_write_recipe_validation_json(out_dir / "recipe_validation.json", records, base_before, base_after);
    moq_write_recipe_validation_summary(out_dir / "recipe_validation_summary.txt", params, records, warnings);
    moq_write_recipe_prediction_compare(out_dir / "recipe_prediction_compare.txt", params, records);
    moq_write_recipe_prediction_compare(out_dir / "recipe_prediction_compare_topk.txt", params, records);
    moq_write_recipe_actual_best_by_bpw(out_dir / "recipe_actual_best_by_bpw.csv", records);
    moq_write_cache_stats(out_dir / "cache_stats.json", cache);
    profiler.write_summary(base_logits);

    LOG_INF("%s: wrote MoQ recipe validation outputs to %s\n", __func__, params.moq_output.c_str());
    return failures.empty() ? 0 : 1;
}

static int moq_dynamic_sweep(llama_context * ctx, llama_model * model, const common_params & params) {
    std::error_code ec;
    fs::create_directories(params.moq_output, ec);
    fs::create_directories(params.moq_cache_dir, ec);

    std::string error;
    moq_source_store source;
    if (!source.open(params.moq_source_bf16, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }

    std::vector<moq_group> groups;
    if (!moq_parse_groups(params.moq_groups, groups, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }

    std::vector<moq_qtype_candidate> candidates;
    if (!moq_parse_candidates(params.moq_candidates, candidates, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }

    std::unique_ptr<moq_imatrix_store> imatrix;
    if (!params.moq_imatrix.empty()) {
        imatrix = std::make_unique<moq_imatrix_store>();
        if (!imatrix->open(params.moq_imatrix, error)) {
            LOG_ERR("%s: %s\n", __func__, error.c_str());
            return 1;
        }
    }

    moq_slot_registry registry;
    if (!moq_register_tensor_slots(*model, registry)) {
        LOG_ERR("%s: failed to register MoQ tensor slots\n", __func__);
        return 1;
    }

    moq_tensor_cache cache;
    cache.source_hash = source.source_hash;
    cache.disk_enabled = !params.moq_disable_disk_cache;
    cache.mem_enabled = !params.moq_disable_mem_cache && params.moq_mem_cache_mb > 0;
    cache.stats.memory_limit_bytes = cache.mem_enabled ? (size_t) params.moq_mem_cache_mb * 1024ull * 1024ull : 0;
    if (imatrix && !imatrix->empty()) {
        cache.imatrix_hash = imatrix->imatrix_hash;
    }

    std::vector<moq_sweep_record> records;
    std::vector<std::string> warnings;
    std::vector<std::string> failures;

    LOG_INF("%s: source=%s, groups=%zu, candidates=%zu, cache=%s, output=%s, backend=%s, source_hash=%s, imatrix=%s\n",
            __func__, source.path.c_str(), groups.size(), candidates.size(), params.moq_cache_dir.c_str(),
            params.moq_output.c_str(), params.moq_dynamic_backend.c_str(), source.source_hash.c_str(),
            imatrix && !imatrix->empty() ? imatrix->path.c_str() : "none");

    const std::vector<moq_sweep_job> jobs = moq_make_sweep_jobs(groups, candidates, params.moq_sweep_order);
    std::unordered_map<std::string, std::string> active_state;
    std::unordered_map<std::string, std::shared_ptr<moq_owned_tensor>> active_owned;
    double replace_ms_per_tensor_est = 0.0;
    bool fatal_failure = false;

    if (cache.disk_enabled && params.moq_prebuild_cache) {
        std::vector<moq_prequant_task> prequant_tasks;
        std::set<std::string> seen_prequant_keys;
        for (const auto & job : jobs) {
            const moq_group & group = groups[job.group_index];
            const moq_qtype_candidate & candidate = candidates[job.candidate_index];
            if (!candidate.supported) {
                continue;
            }

            for (const std::string & tensor_name : group.tensors) {
                ggml_tensor * base_tensor = moq_get_base_tensor(registry, tensor_name);
                const moq_source_tensor * src_tensor = source.get(tensor_name);
                if (base_tensor == nullptr || src_tensor == nullptr || !moq_same_shape(src_tensor->ne, base_tensor)) {
                    continue;
                }

                const std::vector<float> * imatrix_data = moq_select_imatrix(
                        imatrix.get(), tensor_name, *src_tensor, candidate.type, nullptr);
                moq_cache_paths paths = moq_make_cache_paths(*src_tensor, candidate.type, cache, imatrix_data, params.moq_cache_dir);
                if (!seen_prequant_keys.insert(paths.cache_key).second) {
                    continue;
                }

                moq_prequant_task task;
                task.tensor_name = tensor_name;
                task.type = candidate.type;
                task.src_tensor = src_tensor;
                task.imatrix = imatrix_data;
                task.paths = std::move(paths);
                prequant_tasks.push_back(std::move(task));
            }
        }

        const int prequant_threads = moq_prequant_thread_count(params);
        cache.stats.prequant_threads = prequant_threads;
        LOG_INF("%s: CPU prequant phase start: tasks=%zu threads=%d cache_dir=%s\n",
                __func__, prequant_tasks.size(), prequant_threads, params.moq_cache_dir.c_str());
        const moq_prequant_result prequant = moq_prequantize_disk_cache(
                source, cache, prequant_tasks, params.moq_cache_dir, prequant_threads);
        cache.stats.prequant_tasks = prequant.tasks;
        cache.stats.prequant_ready = prequant.ready;
        cache.stats.prequant_built = prequant.built;
        cache.stats.prequant_failed = prequant.failed;
        cache.stats.prequant_bytes_written = prequant.bytes_written;
        cache.stats.prequant_ms = prequant.total_ms;
        cache.stats.prequant_quantize_ms = prequant.quantize_ms;
        cache.stats.bytes_written += prequant.bytes_written;

        LOG_INF("%s: CPU prequant phase done: tasks=%zu ready=%zu built=%zu failed=%zu bytes=%zu total=%.2f ms quantize=%.2f ms\n",
                __func__, prequant.tasks, prequant.ready, prequant.built, prequant.failed,
                prequant.bytes_written, prequant.total_ms, prequant.quantize_ms);
        for (size_t i = 0; i < std::min<size_t>(20, prequant.failures.size()); ++i) {
            warnings.push_back("prequant failed: " + prequant.failures[i]);
        }
        if (prequant.failures.size() > 20) {
            warnings.push_back(string_format("prequant failed: %zu additional failures suppressed",
                    prequant.failures.size() - 20));
        }
    } else {
        LOG_INF("%s: CPU prequant phase skipped because disk cache is %s and prebuild_cache=%s\n",
                __func__, cache.disk_enabled ? "enabled" : "disabled", params.moq_prebuild_cache ? "true" : "false");
    }

    const bool allow_runtime_quantize = !cache.disk_enabled;
    moq_base_logits_store base_logits;
    if (!base_logits.open(params.logits_file, params.moq_base_logits_mode, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }
    if ((int) base_logits.n_vocab != llama_vocab_n_tokens(llama_model_get_vocab(model))) {
        LOG_ERR("%s: base logits vocabulary mismatch: logits=%d model=%d\n",
                __func__, base_logits.n_vocab, llama_vocab_n_tokens(llama_model_get_vocab(model)));
        return 1;
    }

    moq_eval_profiler profiler;
    if (!profiler.open(params.moq_output, params.moq_profile_level, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }

    moq_restore_all(registry);
    moq_invalidate_graph(ctx);
    LOG_INF("%s: evaluating restored base before sweep\n", __func__);
    moq_eval_result base_before = kl_divergence_eval_once(
            ctx, params, base_logits, params.moq_chunks, false, &profiler, {-1, "base_before", "base"});
    if (!base_before.ok) {
        LOG_ERR("%s: base KLD evaluation failed: %s\n", __func__, base_before.error.c_str());
        return 1;
    }

    for (size_t ijob = 0; ijob < jobs.size(); ++ijob) {
        const moq_group & group = groups[jobs[ijob].group_index];
        const moq_qtype_candidate & candidate = candidates[jobs[ijob].candidate_index];

        const int64_t t_total = llama_time_us();
        moq_sweep_record rec;
        rec.group = group.name;
        rec.qtype = candidate.name;

        LOG_INF("%s: group=%s qtype=%s tensors=%zu\n", __func__, rec.group.c_str(), rec.qtype.c_str(), group.tensors.size());

        if (!candidate.supported) {
            rec.status = "failed";
            rec.error = "unsupported qtype: " + candidate.unsupported_reason;
            rec.total_ms = (llama_time_us() - t_total) / 1000.0;
            warnings.push_back(rec.group + "/" + rec.qtype + ": " + rec.error);
            failures.push_back(rec.group + "/" + rec.qtype + ": " + rec.error);
            records.push_back(rec);
            continue;
        }

        auto restore_names = [&](const std::vector<std::string> & names) {
            if (names.empty()) {
                return true;
            }
            const int64_t t_restore = llama_time_us();
            const bool ok = moq_restore_tensor_batch(registry, names);
            const double elapsed_ms = (llama_time_us() - t_restore) / 1000.0;
            rec.batch_restore_ms += elapsed_ms;
            rec.restore_ms += elapsed_ms;
            if (ok) {
                moq_invalidate_graph(ctx);
                rec.restored_tensors += (int) names.size();
                for (const auto & name : names) {
                    active_state.erase(name);
                    active_owned.erase(name);
                }
            }
            return ok;
        };

        if (params.moq_replace_mode == "restore_each" && !active_state.empty()) {
            std::vector<std::string> names;
            names.reserve(active_state.size());
            for (const auto & kv : active_state) {
                names.push_back(kv.first);
            }
            if (!restore_names(names)) {
                rec.status = "failed";
                rec.error = "batch restore failed before candidate";
                failures.push_back(rec.group + "/" + rec.qtype + ": " + rec.error);
                moq_restore_all(registry);
                moq_invalidate_graph(ctx);
                active_state.clear();
                active_owned.clear();
                rec.total_ms = (llama_time_us() - t_total) / 1000.0;
                records.push_back(rec);
                continue;
            }
        }

        std::vector<moq_replacement> replacements;
        std::vector<std::pair<std::string, std::shared_ptr<moq_owned_tensor>>> replacement_owned;
        std::vector<std::string> desired_names;
        std::unordered_map<std::string, std::string> desired_state;
        bool candidate_failed = false;

        for (const std::string & tensor_name : group.tensors) {
            ggml_tensor * base_tensor = moq_get_base_tensor(registry, tensor_name);
            const moq_source_tensor * src_tensor = source.get(tensor_name);
            std::string local_error;
            if (base_tensor == nullptr) {
                local_error = "slot not found";
            } else if (src_tensor == nullptr) {
                local_error = "source tensor not found";
            } else if (!moq_same_shape(src_tensor->ne, base_tensor)) {
                local_error = "shape mismatch";
            }

            if (!local_error.empty()) {
                const std::string msg = string_format("%s/%s/%s: %s",
                        rec.group.c_str(), rec.qtype.c_str(), tensor_name.c_str(), local_error.c_str());
                if (params.moq_skip_missing_tensors) {
                    LOG_WRN("%s\n", msg.c_str());
                    warnings.push_back(msg);
                    continue;
                }
                rec.status = "failed";
                rec.error = msg;
                failures.push_back(msg);
                candidate_failed = true;
                break;
            }

            desired_names.push_back(tensor_name);
            desired_state[tensor_name] = candidate.name;
            rec.source_bytes += src_tensor->nbytes;
            rec.source_elements += (size_t) moq_nelements(src_tensor->ne);
            rec.quant_bytes += moq_quant_nbytes(candidate.type, src_tensor->ne);

            auto it_active = active_state.find(tensor_name);
            if (params.moq_replace_mode == "diff" && it_active != active_state.end() && it_active->second == candidate.name) {
                rec.unchanged_tensors++;
                continue;
            }
            if (params.moq_replace_mode == "diff" && it_active != active_state.end()) {
                rec.diff_saved_replace_count++;
            }

            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
            if (params.moq_dynamic_backend == "same") {
                buft = moq_get_base_tensor_buft(registry, tensor_name);
            }

            std::string imatrix_warning;
            const std::vector<float> * imatrix_data = moq_select_imatrix(
                    imatrix.get(), tensor_name, *src_tensor, candidate.type, &imatrix_warning);
            if (!imatrix_warning.empty()) {
                const std::string warning = string_format("%s/%s: %s",
                        rec.group.c_str(), rec.qtype.c_str(), imatrix_warning.c_str());
                LOG_WRN("%s\n", warning.c_str());
                warnings.push_back(warning);
            }

            moq_tensor_build_result built = moq_get_or_build_tensor(source, cache, *src_tensor, candidate.type,
                    imatrix_data, buft, params.moq_cache_dir, allow_runtime_quantize);
            rec.quantize_ms += built.quantize_ms;
            rec.cache_load_ms += built.cache_load_ms;
            rec.mem_cache_load_ms += built.mem_cache_load_ms;
            rec.disk_cache_load_ms += built.disk_cache_load_ms;
            rec.upload_ms += built.upload_ms;
            rec.imatrix_used = rec.imatrix_used || built.imatrix_used;
            rec.cache_hit_tensors += built.cache_hit ? 1 : 0;
            rec.mem_cache_hits += built.mem_cache_hit ? 1 : 0;
            rec.disk_cache_hits += built.disk_cache_hit ? 1 : 0;
            rec.cache_misses += built.cache_miss ? 1 : 0;
            rec.newly_quantized_tensors += built.cache_miss ? 1 : 0;

            if (!built.error.empty() || !built.owned || built.owned->tensor == nullptr) {
                rec.status = "failed";
                rec.error = string_format("%s/%s/%s: %s",
                        rec.group.c_str(), rec.qtype.c_str(), tensor_name.c_str(),
                        built.error.empty() ? "failed to build dynamic tensor" : built.error.c_str());
                failures.push_back(rec.error);
                candidate_failed = true;
                break;
            }

            replacements.push_back({tensor_name, built.owned->tensor});
            replacement_owned.push_back({tensor_name, built.owned});
        }

        rec.n_tensors = (int) desired_names.size();
        rec.cache_hit_ratio = rec.n_tensors > 0 ? (double) rec.cache_hit_tensors / rec.n_tensors : 0.0;
        rec.diff_saved_replace_count += rec.unchanged_tensors;
        rec.diff_saved_ms_estimate = replace_ms_per_tensor_est * rec.diff_saved_replace_count;

        if (!candidate_failed && params.moq_replace_mode == "diff") {
            std::vector<std::string> restore_diff;
            for (const auto & kv : active_state) {
                if (desired_state.find(kv.first) == desired_state.end()) {
                    restore_diff.push_back(kv.first);
                }
            }
            if (!restore_names(restore_diff)) {
                rec.status = "failed";
                rec.error = "batch restore failed in diff mode";
                failures.push_back(rec.group + "/" + rec.qtype + ": " + rec.error);
                candidate_failed = true;
            }
        }

        if (!candidate_failed && !replacements.empty()) {
            const int64_t t_replace = llama_time_us();
            if (moq_replace_tensor_batch(registry, replacements)) {
                rec.batch_replace_ms += (llama_time_us() - t_replace) / 1000.0;
                rec.replace_ms += rec.batch_replace_ms;
                rec.replaced_tensors += (int) replacements.size();
                for (const auto & repl : replacements) {
                    active_state[repl.name] = candidate.name;
                }
                for (auto & owned : replacement_owned) {
                    active_owned[owned.first] = std::move(owned.second);
                }
                moq_invalidate_graph(ctx);
                if (rec.replaced_tensors > 0) {
                    const double sample = rec.batch_replace_ms / rec.replaced_tensors;
                    replace_ms_per_tensor_est = replace_ms_per_tensor_est == 0.0 ? sample : 0.8 * replace_ms_per_tensor_est + 0.2 * sample;
                }
            } else {
                rec.status = "failed";
                rec.error = "batch replace failed";
                failures.push_back(rec.group + "/" + rec.qtype + ": " + rec.error);
                candidate_failed = true;
            }
        }

        if (rec.n_tensors == 0 && !candidate_failed && !group.tensors.empty()) {
            rec.status = "failed";
            rec.error = "no tensors were selected";
            failures.push_back(rec.group + "/" + rec.qtype + ": " + rec.error);
            candidate_failed = true;
        }

        if (!candidate_failed) {
            moq_eval_result eval = kl_divergence_eval_once(
                    ctx, params, base_logits, params.moq_chunks, false, &profiler,
                    {(int) ijob, rec.group, rec.qtype});
            rec.eval_ms = eval.eval_ms;
            if (!eval.ok) {
                rec.status = "failed";
                rec.error = eval.error;
                failures.push_back(rec.group + "/" + rec.qtype + ": " + eval.error);
                candidate_failed = true;
            } else {
                rec.ppl = eval.ppl;
                rec.mean_kld = eval.mean_kld;
                rec.max_kld = eval.max_kld;
                rec.p99_kld = eval.p99_kld;
                rec.p999_kld = eval.p999_kld;
                LOG_INF("%s: result group=%s qtype=%s PPL=%.6lf mean_KLD=%.6lf p99=%.6lf unchanged=%d replaced=%d restored=%d total_so_far=%.2lf ms\n",
                        __func__, rec.group.c_str(), rec.qtype.c_str(), rec.ppl, rec.mean_kld, rec.p99_kld,
                        rec.unchanged_tensors, rec.replaced_tensors, rec.restored_tensors,
                        (llama_time_us() - t_total) / 1000.0);
            }
        }

        if (params.moq_replace_mode == "restore_each") {
            std::vector<std::string> names;
            names.reserve(active_state.size());
            for (const auto & kv : active_state) {
                names.push_back(kv.first);
            }
            if (!restore_names(names) && rec.status == "ok") {
                rec.status = "failed";
                rec.error = "batch restore failed after candidate";
                failures.push_back(rec.group + "/" + rec.qtype + ": " + rec.error);
            }
        } else if (candidate_failed) {
            moq_restore_all(registry);
            moq_invalidate_graph(ctx);
            active_state.clear();
            active_owned.clear();
        }

        rec.total_ms = (llama_time_us() - t_total) / 1000.0;
        records.push_back(rec);

        if (params.moq_replace_mode == "diff" && params.moq_diff_check_interval > 0 && (ijob + 1) % params.moq_diff_check_interval == 0) {
            moq_restore_all(registry);
            moq_invalidate_graph(ctx);
            active_state.clear();
            active_owned.clear();
            moq_eval_result check = kl_divergence_eval_once(
                    ctx, params, base_logits, params.moq_chunks, false, &profiler,
                    {(int) ijob, "diff_base_check", "base"});
            if (!check.ok ||
                    std::abs(check.ppl - base_before.ppl) > 1e-9 ||
                    std::abs(check.mean_kld - base_before.mean_kld) > 1e-12) {
                const std::string failure = string_format("diff base check failed after candidate %zu: PPL %.12lf vs %.12lf, mean_KLD %.12lf vs %.12lf",
                        ijob + 1, check.ppl, base_before.ppl, check.mean_kld, base_before.mean_kld);
                LOG_ERR("%s\n", failure.c_str());
                failures.push_back(failure);
                fatal_failure = true;
                break;
            }
        }
    }

    moq_restore_all(registry);
    moq_invalidate_graph(ctx);
    active_state.clear();
    active_owned.clear();
    LOG_INF("%s: evaluating restored base after sweep\n", __func__);
    moq_eval_result base_after = kl_divergence_eval_once(
            ctx, params, base_logits, params.moq_chunks, false, &profiler, {-2, "base_after", "base"});
    if (!base_after.ok) {
        warnings.push_back("base-after-restore evaluation failed: " + base_after.error);
    }

    const fs::path out_dir(params.moq_output);
    moq_write_results_csv(out_dir / "sweep_results.csv", records);
    moq_write_results_json(out_dir / "sweep_results.json", records, base_before, base_after, warnings);
    moq_write_timing_csv(out_dir / "sweep_timing.csv", records);
    moq_write_elasticity_reports(out_dir, params, records, candidates);
    moq_write_summary(out_dir / "sweep_summary.txt", params, source, imatrix.get(), cache, candidates, records, base_before, base_after, warnings, failures);
    moq_write_cache_stats(out_dir / "cache_stats.json", cache);
    moq_write_failed_candidates_csv(out_dir / "failed_candidates.csv", records);
    profiler.write_summary(base_logits);

    {
        std::ofstream out(out_dir / "failed_candidates.txt");
        for (const auto & f : failures) {
            out << f << "\n";
        }
    }

    LOG_INF("%s: wrote MoQ sweep outputs to %s\n", __func__, params.moq_output.c_str());
    return failures.empty() ? 0 : 1;
}

static int moq_profiled_kl_divergence(llama_context * ctx, const common_params & params) {
    if (params.logits_file.empty()) {
        LOG_ERR("%s: --kl-divergence with --moq-profile-level requires --kl-divergence-base/--moq-base-logits\n", __func__);
        return 1;
    }

    std::string error;
    moq_base_logits_store base_logits;
    if (!base_logits.open(params.logits_file, params.moq_base_logits_mode, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }

    moq_eval_profiler profiler;
    if (!profiler.open(params.moq_output, params.moq_profile_level, error)) {
        LOG_ERR("%s: %s\n", __func__, error.c_str());
        return 1;
    }

    moq_eval_result eval = kl_divergence_eval_once(
            ctx, params, base_logits, params.moq_chunks, true, &profiler, {0, "normal_kld", "base"});
    profiler.write_summary(base_logits);

    if (!eval.ok) {
        LOG_ERR("%s: profiled KLD evaluation failed: %s\n", __func__, eval.error.c_str());
        return 1;
    }

    LOG_INF("%s: PPL=%.6lf PPL_base=%.6lf mean_KLD=%.6lf max_KLD=%.6lf p99=%.6lf p999=%.6lf eval_ms=%.3lf\n",
            __func__, eval.ppl, eval.ppl_base, eval.mean_kld, eval.max_kld, eval.p99_kld, eval.p999_kld, eval.eval_ms);
    LOG_INF("%s: profile output written to %s\n", __func__, params.moq_output.c_str());
    return 0;
}

static void kl_divergence(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.logits_file.empty()) {
        LOG_ERR("%s: you must provide a name of a file containing the log probabilities of the base model\n", __func__);
        return;
    }
    std::ifstream in(params.logits_file.c_str(), std::ios::binary);
    if (!in) {
        LOG_ERR("%s: failed to open %s\n", __func__, params.logits_file.c_str());
        return;
    }
    {
        char check[9]; check[8] = 0;
        in.read(check, 8);
        if (in.fail() || strncmp("_logits_", check, 8) != 0) {
            LOG_ERR("%s: %s does not look like a file containing log-probabilities\n", __func__, params.logits_file.c_str());
            return;
        }
    }

    uint32_t n_ctx;
    in.read((char *)&n_ctx, sizeof(n_ctx));
    if (n_ctx > llama_n_ctx(ctx)) {
        LOG_ERR("%s: %s has been computed with %u, while the current context is %d. Increase it with -c and retry\n",
                __func__, params.logits_file.c_str(), n_ctx, params.n_ctx);
    }

    int n_vocab;
    int n_chunk;
    in.read((char *)&n_vocab, sizeof(n_vocab));
    in.read((char *)&n_chunk, sizeof(n_chunk));
    if (in.fail()) {
        LOG_ERR("%s: failed reading n_vocab, n_chunk from %s\n", __func__, params.logits_file.c_str());
        return;
    }
    if (n_vocab != llama_vocab_n_tokens(vocab)) {
        LOG_ERR("%s: inconsistent vocabulary (%d vs %d)\n", __func__, n_vocab, llama_vocab_n_tokens(vocab));
    }

    std::vector<llama_token> tokens(size_t(n_ctx) * n_chunk);
    if (in.read((char *)tokens.data(), tokens.size()*sizeof(tokens[0])).fail()) {
        LOG_ERR("%s: failed reading evaluation tokens from %s\n", __func__, params.logits_file.c_str());
        return;
    }

    const int n_batch = params.n_batch;
    const int num_batches = (static_cast<int>(n_ctx) + n_batch - 1) / n_batch;
    // Calculate n_seq based on the logits file's n_ctx, but cap it at what the context supports
    const int n_seq_max = llama_n_seq_max(ctx);
    int n_seq = std::max(1, n_batch / static_cast<int>(n_ctx));
    if (n_seq > n_seq_max) {
        LOG_WRN("%s: calculated n_seq=%d exceeds context's n_seq_max=%d, capping at %d\n",
                __func__, n_seq, n_seq_max, n_seq_max);
        n_seq = n_seq_max;
    }
    const int nv = 2*((n_vocab + 1)/2) + 4;
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    llama_batch batch = llama_batch_init(std::min(n_batch, static_cast<int>(n_ctx)*n_seq), 0, 1);

    std::vector<uint16_t> log_probs_uint16(size_t(n_ctx - 1 - n_ctx/2) * nv);
    std::vector<float>    kld_values(size_t(n_ctx - 1 - n_ctx/2)*n_chunk);
    std::vector<float> p_diff_values(size_t(n_ctx - 1 - n_ctx/2)*n_chunk);
    std::vector<float> logits;
    if (num_batches > 1) {
        logits.reserve(size_t(n_ctx) * n_vocab);
    }

    LOG_INF("%s: computing over %d chunks, n_ctx=%u, batch_size=%d, n_seq=%d\n", __func__, n_chunk, n_ctx, n_batch, n_seq);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    auto mean_and_uncertainty = [] (double sum, double sum2, size_t count) {
        if (count < 1) {
            return std::make_pair(0., 0.);
        }
        double f = sum/count;
        double df = sum2/count - f*f;
        df = df > 0 && count > 10 ? sqrt(df/(count-1)) : 0.;
        return std::make_pair(f, df);
    };
    auto covariance = [] (double suma, double sumb, double sumab, size_t count) {
        if (count < 10) {
            return 0.0;
        }
        double var = sumab/count - (suma/count)*(sumb/count);
        var /= count - 1;
        return var;
    };

    kl_divergence_result kld;
    auto    kld_ptr =    kld_values.data();
    auto p_diff_ptr = p_diff_values.data();

    const int first = n_ctx/2;

    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk - i);

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            int n_outputs = 0;

            common_batch_clear(batch);
            for (int seq = 0; seq < n_seq_batch; seq++) {
                int seq_start = batch_start + seq*n_ctx;

                // save original token and restore it after eval
                const auto token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }

                for (int k = 0; k < batch_size; ++k) {
                    const int pos = j*n_batch + k;
                    const bool need_logits = pos >= first;
                    common_batch_add(batch, tokens[seq_start + k], pos, { seq }, need_logits);
                    n_outputs += need_logits;
                }

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s : failed to decode\n", __func__);
                llama_batch_free(batch);
                return;
            }

            if (num_batches > 1 && n_outputs > 0) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + size_t(n_outputs) * n_vocab);
            }
        }

        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk / n_seq);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
            LOG("\n");
            LOG("chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p\n");
        }

        // Read log probs for each sequence in the batch
        for (int seq = 0; seq < n_seq_batch; seq++) {
            if (in.read((char *)log_probs_uint16.data(), log_probs_uint16.size()*sizeof(uint16_t)).fail()) {
                LOG_ERR("%s: failed reading log-probs for chunk %d\n", __func__, i + seq);
                llama_batch_free(batch);
                return;
            }

            const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx + first);

            process_logits(n_vocab, all_logits, tokens.data() + start + seq*n_ctx + first, n_ctx - 1 - first,
                    workers, log_probs_uint16.data(), kld, kld_ptr, p_diff_ptr);
            p_diff_ptr += n_ctx - 1 - first;
            kld_ptr    += n_ctx - 1 - first;

            LOG("%4d", i + seq + 1);

            auto log_ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
            const double ppl_val = exp(log_ppl.first);
            const double ppl_unc = ppl_val * log_ppl.second;
            LOG("    %9.4lf ± %9.4lf", ppl_val, ppl_unc);

            auto log_ppl_base = mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
            const double log_ppl_cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
            const double log_ppl_ratio_val = log_ppl.first - log_ppl_base.first;
            const double log_ppl_ratio_unc = sqrt(log_ppl.second*log_ppl.second + log_ppl_base.second*log_ppl_base.second - 2.0*log_ppl_cov);
            LOG("    %10.5lf ± %10.5lf", log_ppl_ratio_val, log_ppl_ratio_unc);

            auto kl_div = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
            LOG("    %10.5lf ± %10.5lf", kl_div.first, kl_div.second);

            auto p_diff_mse   = mean_and_uncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
            const double p_diff_rms_val = sqrt(p_diff_mse.first);
            const double p_diff_rms_unc = 0.5/p_diff_rms_val * p_diff_mse.second;
            LOG("    %6.3lf ± %6.3lf %%", 100.0*p_diff_rms_val, 100.0*p_diff_rms_unc);

            double p_top_val = 1.*kld.n_same_top/kld.count;
            double p_top_unc = sqrt(p_top_val*(1 - p_top_val)/(kld.count - 1));
            LOG("    %6.3lf ± %6.3lf %%", 100.0*p_top_val, 100.0*p_top_unc);

            LOG("\n");
        }

        logits.clear();
    }

    llama_batch_free(batch);
    LOG("\n");

    if (kld.count < 100) return; // we do not wish to do statistics on so few values

    std::sort(kld_values.begin(), kld_values.end());
    std::sort(p_diff_values.begin(), p_diff_values.end());

    LOG("====== Perplexity statistics ======\n");

    auto log_ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
    const double ppl_val = exp(log_ppl.first);
    const double ppl_unc = ppl_val * log_ppl.second; // ppl_unc = sqrt( (dexp(x) / dx) ** 2 * log_ppl.second ** 2 )
    LOG("Mean PPL(Q)                   : %10.6lf ± %10.6lf\n", ppl_val, ppl_unc);

    auto log_ppl_base = mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    const double ppl_base_val = exp(log_ppl_base.first);
    const double ppl_base_unc = ppl_base_val * log_ppl_base.second; // ppl_base_unc = sqrt( (dexp(x) / dx) ** 2 * log_ppl_base.second ** 2 )
    LOG("Mean PPL(base)                : %10.6lf ± %10.6lf\n", ppl_base_val, ppl_base_unc);

    const double log_ppl_cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
    // LOG("Cov(ln(PPL(Q)), ln(PPL(base))): %10.6lf\n", log_ppl_cov);
    const double log_ppl_cor = log_ppl_cov / (log_ppl.second*log_ppl_base.second);
    LOG("Cor(ln(PPL(Q)), ln(PPL(base))): %6.2lf%%\n", 100.0*log_ppl_cor);

    const double log_ppl_ratio_val = log_ppl.first - log_ppl_base.first;
    const double log_ppl_ratio_unc = sqrt(log_ppl.second*log_ppl.second + log_ppl_base.second*log_ppl_base.second - 2.0*log_ppl_cov);
    LOG("Mean ln(PPL(Q)/PPL(base))     : %10.6lf ± %10.6lf\n", log_ppl_ratio_val, log_ppl_ratio_unc);

    const double ppl_ratio_val = exp(log_ppl_ratio_val);
    const double ppl_ratio_unc = ppl_ratio_val * log_ppl_ratio_unc; // ppl_ratio_unc = sqrt( (dexp(x) / dx) ** 2 * log_ppl_ratio.second ** 2 )
    LOG("Mean PPL(Q)/PPL(base)         : %10.6lf ± %10.6lf\n", ppl_ratio_val, ppl_ratio_unc);

    const double ppl_cov = ppl_val * ppl_base_val * log_ppl_cov;
    const double ppl_diff_val = ppl_val - ppl_base_val;
    const double ppl_diff_unc = sqrt(ppl_unc*ppl_unc + ppl_base_unc*ppl_base_unc - 2.0*ppl_cov);
    LOG("Mean PPL(Q)-PPL(base)         : %10.6lf ± %10.6lf\n", ppl_diff_val, ppl_diff_unc);

    LOG("\n");

    LOG("====== KL divergence statistics ======\n");
    auto kl_div = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
    LOG("Mean    KLD: %10.6lf ± %10.6lf\n", kl_div.first, kl_div.second);
    auto kld_median = kld_values.size()%2 == 0 ? 0.5f*(kld_values[kld_values.size()/2] + kld_values[kld_values.size()/2-1])
                                               : kld_values[kld_values.size()/2];

    auto percentile = [] (std::vector<float> values, float fraction) {
        if (fraction <= 0) return values.front();
        if (fraction >= 1) return values.back();
        float p = fraction*(values.size() - 1);
        size_t ip = size_t(p); p -= ip;
        return (1 - p)*values[ip] + p*values[std::min(ip+1, values.size()-1)];
    };

    LOG("Maximum KLD: %10.6f\n", kld_values.back());
    LOG("99.9%%   KLD: %10.6f\n", percentile(kld_values, 0.999f));
    LOG("99.0%%   KLD: %10.6f\n", percentile(kld_values, 0.990f));
    LOG("95.0%%   KLD: %10.6f\n", percentile(kld_values, 0.950f));
    LOG("90.0%%   KLD: %10.6f\n", percentile(kld_values, 0.900f));
    LOG("Median  KLD: %10.6f\n", kld_median);
    LOG("10.0%%   KLD: %10.6f\n", percentile(kld_values, 0.100f));
    LOG(" 5.0%%   KLD: %10.6f\n", percentile(kld_values, 0.050f));
    LOG(" 1.0%%   KLD: %10.6f\n", percentile(kld_values, 0.010f));
    LOG(" 0.1%%   KLD: %10.6f\n", percentile(kld_values, 0.001f));
    LOG("Minimum KLD: %10.6f\n", kld_values.front());

    LOG("\n");

    LOG("====== Token probability statistics ======\n");

    auto p_diff = mean_and_uncertainty(kld.sum_p_diff, kld.sum_p_diff2, kld.count);
    LOG("Mean    Δp: %6.3lf ± %5.3lf %%\n",  100.0*p_diff.first, 100.0*p_diff.second);

    auto p_diff_median = p_diff_values.size()%2 == 0 ? 0.5f*(p_diff_values[p_diff_values.size()/2] + p_diff_values[p_diff_values.size()/2-1])
                                               : p_diff_values[p_diff_values.size()/2];

    LOG("Maximum Δp: %6.3lf%%\n",  100.0*p_diff_values.back());
    LOG("99.9%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.999f));
    LOG("99.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.990f));
    LOG("95.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.950f));
    LOG("90.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.900f));
    LOG("75.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.750f));
    LOG("Median  Δp: %6.3lf%%\n",  100.0*p_diff_median);
    LOG("25.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.250f));
    LOG("10.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.100f));
    LOG(" 5.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.050f));
    LOG(" 1.0%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.010f));
    LOG(" 0.1%%   Δp: %6.3lf%%\n", 100.0*percentile(p_diff_values, 0.001f));
    LOG("Minimum Δp: %6.3lf%%\n",  100.0*p_diff_values.front());

    auto p_diff_mse = mean_and_uncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
    // LOG("MSE Δp    : %10.6lf ± %10.6lf\n", p_diff_mse.first, p_diff_mse.second);

    const double p_diff_rms_val = sqrt(p_diff_mse.first);
    const double p_diff_rms_unc = 0.5/p_diff_rms_val * p_diff_mse.second;
    LOG("RMS Δp    : %6.3lf ± %5.3lf %%\n", 100.0*p_diff_rms_val, 100.0*p_diff_rms_unc);

    const double same_top_p = 1.0*kld.n_same_top/kld.count;
    LOG("Same top p: %6.3lf ± %5.3lf %%\n", 100.0*same_top_p, 100.0*sqrt(same_top_p*(1.0 - same_top_p)/(kld.count - 1)));
}

// satisfies -Wmissing-declarations
int llama_perplexity(int argc, char ** argv);

int llama_perplexity(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    params.n_ctx = 512;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    if (params.moq_list_qtypes) {
        moq_print_qtypes();
        return 0;
    }

    if (params.moq_dynamic_sweep) {
        const bool moq_recipe_mode = !params.moq_recipe.empty() || !params.moq_recipe_list.empty();
        if (params.moq_source_bf16.empty()) {
            LOG_ERR("%s: --moq-dynamic-sweep requires --moq-source-bf16\n", __func__);
            return 1;
        }
        if (params.logits_file.empty()) {
            LOG_ERR("%s: --moq-dynamic-sweep requires --moq-base-logits\n", __func__);
            return 1;
        }
        if (params.moq_groups.empty()) {
            LOG_ERR("%s: --moq-dynamic-sweep requires --moq-groups\n", __func__);
            return 1;
        }
        if (!moq_recipe_mode && params.moq_candidates.empty()) {
            LOG_ERR("%s: --moq-dynamic-sweep requires --moq-candidates\n", __func__);
            return 1;
        }
        if (params.ppl_stride > 0) {
            LOG_ERR("%s: --moq-dynamic-sweep currently requires non-strided perplexity/KLD logits\n", __func__);
            return 1;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LOG_WRN("%s: MoQ dynamic sweep forces Flash Attention on for the high-speed path\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.moq_cuda_graphs == "off") {
#if defined(_WIN32)
            _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
#else
            setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
#endif
        } else {
#if defined(_WIN32)
            _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "");
#else
            unsetenv("GGML_CUDA_DISABLE_GRAPHS");
#endif
        }
        LOG_INF("%s: MoQ dynamic sweep enabled; mode=%s cuda_graphs=%s flash_attn=%s\n",
                __func__, moq_recipe_mode ? "recipe_validation" : "sweep",
                params.moq_cuda_graphs.c_str(), llama_flash_attn_type_name(params.flash_attn_type));
    }

    const int32_t n_ctx = params.n_ctx;

    if (n_ctx <= 0) {
        LOG_ERR("%s: perplexity tool requires '--ctx-size' > 0\n", __func__);
        return 1;
    }

    if (params.hellaswag || params.winogrande || params.multiple_choice) {
        params.n_parallel = std::max(4, params.n_parallel);
        params.kv_unified = true;
    } else { // Perplexity & KL divergence
        params.n_parallel = std::max(1, params.n_batch / n_ctx);
    }
    params.n_ctx = params.n_parallel * n_ctx;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    if (params.ppl_stride > 0) {
        LOG_INF("Will perform strided perplexity calculation -> adjusting context size from %d to %d\n",
                params.n_ctx, params.n_ctx + params.ppl_stride/2);
        params.n_ctx += params.ppl_stride/2;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model and apply lora adapter, if any
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);

    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    int ret = 0;
    struct results_perplexity results;
    if (params.hellaswag) {
        hellaswag_score(ctx, params);
    } else if (params.winogrande) {
        winogrande_score(ctx, params);
    } else if (params.multiple_choice) {
        multiple_choice_score(ctx, params);
    } else if (params.moq_dynamic_sweep) {
        if (!params.moq_recipe.empty() || !params.moq_recipe_list.empty()) {
            ret = moq_recipe_validation(ctx, model, params);
        } else {
            ret = moq_dynamic_sweep(ctx, model, params);
        }
    } else if (params.kl_divergence) {
        if (params.moq_profile_level > 0) {
            ret = moq_profiled_kl_divergence(ctx, params);
        } else {
            kl_divergence(ctx, params);
        }
    } else {
        results = perplexity(ctx, params, n_ctx);
    }

    LOG("\n");
    llama_perf_context_print(ctx);
    common_memory_breakdown_print(ctx);

    llama_backend_free();

    return ret;
}
