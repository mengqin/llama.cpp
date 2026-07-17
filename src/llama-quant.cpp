#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-ext.h"
#include "ggml-quants.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <regex>
#include <thread>
#include <unordered_map>
#include <vector>

// result of parsing --tensor-type option
// (changes to this struct must be reflected in tools/quantize/quantize.cpp)
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
};

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix(const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

//
// helper functions for tensor name matching
//

static bool tensor_name_match_token_embd(const char * tensor_name) {
    return std::strcmp(tensor_name, "token_embd.weight") == 0 ||
           std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0;
}

static bool tensor_name_match_output_weight(const char * tensor_name) {
    return std::strcmp(tensor_name, "output.weight") == 0;
}

static constexpr int LLAMA_QUANT_WHT_DIM = 256;
static constexpr const char * LLAMA_QUANT_WHT_SCHEME = "pqk_rht_v1";
static constexpr const char * LLAMA_QUANT_WHT_DEFAULT_SKIP_TYPES = "Q3_K,IQ2_XXS,IQ2_XS,IQ2_S,IQ3_XXS,IQ3_S";
static constexpr uint32_t LLAMA_QUANT_WHT_VERSION = 1;

static const char * llama_nvfp4_scale_mode_name(llama_nvfp4_scale_mode mode) {
    switch (mode) {
        case LLAMA_NVFP4_SCALE_MODE_M6:       return "m6";
        case LLAMA_NVFP4_SCALE_MODE_M4:       return "m4";
        case LLAMA_NVFP4_SCALE_MODE_ADAPTIVE: return "adaptive";
        default:                              return "m6";
    }
}

static ggml_nvfp4_scale_mode llama_nvfp4_scale_mode_to_ggml(llama_nvfp4_scale_mode mode) {
    switch (mode) {
        case LLAMA_NVFP4_SCALE_MODE_M4:       return GGML_NVFP4_SCALE_MODE_M4;
        case LLAMA_NVFP4_SCALE_MODE_ADAPTIVE: return GGML_NVFP4_SCALE_MODE_ADAPTIVE;
        case LLAMA_NVFP4_SCALE_MODE_M6:
        default:                              return GGML_NVFP4_SCALE_MODE_M6;
    }
}

static const char * llama_env_or_unset(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] ? value : "(unset)";
}

static bool llama_nvfp4_debug_tensor_match(const char * tensor_name) {
    const char * env = std::getenv("GGML_NVFP4_DEBUG_TENSOR");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }

    const std::string list(env);
    size_t start = 0;
    while (start <= list.size()) {
        const size_t comma = list.find(',', start);
        const size_t end = comma == std::string::npos ? list.size() : comma;
        size_t left = start;
        size_t right = end;
        while (left < right && std::isspace((unsigned char) list[left])) {
            ++left;
        }
        while (right > left && std::isspace((unsigned char) list[right - 1])) {
            --right;
        }
        if (right > left && list.compare(left, right - left, tensor_name) == 0) {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    return false;
}

static ggml_nvfp4_scale_stats llama_nvfp4_scale_stats_subtract(
        const ggml_nvfp4_scale_stats & after,
        const ggml_nvfp4_scale_stats & before) {
    ggml_nvfp4_scale_stats out = {};
#define LLAMA_NVFP4_DIFF_U(field) out.field = after.field >= before.field ? after.field - before.field : 0
#define LLAMA_NVFP4_DIFF_D(field) out.field = after.field - before.field
    LLAMA_NVFP4_DIFF_U(n_subblocks);
    LLAMA_NVFP4_DIFF_U(n_weighted_subblocks);
    LLAMA_NVFP4_DIFF_U(n_ordinary_best_selected);
    LLAMA_NVFP4_DIFF_U(n_weighted_best_selected);
    LLAMA_NVFP4_DIFF_U(n_knee_selected);
    LLAMA_NVFP4_DIFF_U(n_weighted_guard_accepted);
    LLAMA_NVFP4_DIFF_U(n_weighted_guard_fallback);
    LLAMA_NVFP4_DIFF_U(n_m4);
    LLAMA_NVFP4_DIFF_U(n_m5);
    LLAMA_NVFP4_DIFF_U(n_m6);
    LLAMA_NVFP4_DIFF_D(mse_m6);
    LLAMA_NVFP4_DIFF_D(mse_selected);
    LLAMA_NVFP4_DIFF_D(weighted_mse_m6);
    LLAMA_NVFP4_DIFF_D(weighted_mse_selected);
    LLAMA_NVFP4_DIFF_U(candidate_count_total);
    LLAMA_NVFP4_DIFF_U(n_anchor_top1);
    LLAMA_NVFP4_DIFF_U(n_slot1);
    LLAMA_NVFP4_DIFF_U(n_slot15);
    LLAMA_NVFP4_DIFF_U(n_slot2);
    LLAMA_NVFP4_DIFF_U(n_slot3);
    LLAMA_NVFP4_DIFF_U(n_slot_other);
    LLAMA_NVFP4_DIFF_U(n_source_rich);
    LLAMA_NVFP4_DIFF_U(n_source_cjso_ordinary);
    LLAMA_NVFP4_DIFF_U(n_source_cjso_weighted);
    LLAMA_NVFP4_DIFF_U(n_source_both);
    LLAMA_NVFP4_DIFF_U(n_source_fallback);
    LLAMA_NVFP4_DIFF_U(n_cjso_ordinary_candidates);
    LLAMA_NVFP4_DIFF_U(n_cjso_weighted_candidates);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_m4);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_m3);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_m2);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_m1);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_0);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_p1);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_p2);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_p3);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_p4);
    LLAMA_NVFP4_DIFF_U(n_cjso_delta_other);
    LLAMA_NVFP4_DIFF_U(n_rsf_tensors);
    LLAMA_NVFP4_DIFF_U(n_rsf_0875);
    LLAMA_NVFP4_DIFF_U(n_rsf_09375);
    LLAMA_NVFP4_DIFF_U(n_rsf_1000);
    LLAMA_NVFP4_DIFF_U(n_rsf_10625);
    LLAMA_NVFP4_DIFF_U(n_rsf_1125);
    LLAMA_NVFP4_DIFF_D(rsf_sample_error_baseline);
    LLAMA_NVFP4_DIFF_D(rsf_sample_error_selected);
#undef LLAMA_NVFP4_DIFF_U
#undef LLAMA_NVFP4_DIFF_D
    return out;
}

static void llama_nvfp4_log_runtime_config(const char * prefix, bool relevant) {
    if (!relevant) {
        return;
    }
    LLAMA_LOG_INFO("%s: NVFP4 env/config:\n", prefix);
    LLAMA_LOG_INFO("%s:   GGML_NVFP4_SEARCH_ALGO=%s -> %s\n", prefix,
            llama_env_or_unset("GGML_NVFP4_SEARCH_ALGO"),
            ggml_nvfp4_search_algo_name(ggml_nvfp4_get_search_algo()));
    LLAMA_LOG_INFO("%s:   GGML_NVFP4_CJSO_RADIUS=%s -> %d\n", prefix,
            llama_env_or_unset("GGML_NVFP4_CJSO_RADIUS"), ggml_nvfp4_cjso_radius());
    LLAMA_LOG_INFO("%s:   GGML_NVFP4_RICH_SCALE_SEARCH=%s -> %d\n", prefix,
            llama_env_or_unset("GGML_NVFP4_RICH_SCALE_SEARCH"), ggml_nvfp4_rich_scale_search_enabled());
    LLAMA_LOG_INFO("%s:   GGML_NVFP4_RSF_LITE=%s -> %d\n", prefix,
            llama_env_or_unset("GGML_NVFP4_RSF_LITE"), ggml_nvfp4_rsf_lite_enabled());
    LLAMA_LOG_INFO("%s:   GGML_NVFP4_IMATRIX_SELECT_MODE=%s -> %s\n", prefix,
            llama_env_or_unset("GGML_NVFP4_IMATRIX_SELECT_MODE"),
            ggml_nvfp4_imatrix_select_mode_name(ggml_nvfp4_get_imatrix_select_mode()));
    LLAMA_LOG_INFO("%s:   GGML_NVFP4_IMATRIX_ORDINARY_MSE_GUARD=%s -> %.6g\n", prefix,
            llama_env_or_unset("GGML_NVFP4_IMATRIX_ORDINARY_MSE_GUARD"),
            (double) ggml_nvfp4_imatrix_ordinary_mse_guard());
    LLAMA_LOG_INFO("%s:   GGML_CUDA_NVFP4_ACTIVITY_ADAPTIVE=%s\n", prefix,
            llama_env_or_unset("GGML_CUDA_NVFP4_ACTIVITY_ADAPTIVE"));
    LLAMA_LOG_INFO("%s:   candidates: %s\n", prefix, ggml_nvfp4_adaptive_candidates());
}

static void llama_nvfp4_log_tensor_debug(
        const char * prefix,
        const char * tensor_name,
        int64_t slice,
        const void * quantized_data,
        int64_t nrows,
        int64_t n_per_row,
        float rsf_multiplier,
        const ggml_nvfp4_scale_stats & stats) {
    std::array<uint64_t, 256> scale_hist = {};
    std::array<uint64_t, 16> q_hist = {};
    uint8_t scale_min = 255;
    uint8_t scale_max = 0;

    const int64_t blocks_per_row = n_per_row / QK_NVFP4;
    const int64_t n_blocks = nrows * blocks_per_row;
    const block_nvfp4 * blocks = (const block_nvfp4 *) quantized_data;
    uint64_t q_nonzero = 0;
    uint64_t q_saturated = 0;

    for (int64_t bi = 0; bi < n_blocks; ++bi) {
        const block_nvfp4 & block = blocks[bi];
        for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s) {
            const uint8_t sc = block.d[s];
            scale_hist[sc]++;
            scale_min = std::min(scale_min, sc);
            scale_max = std::max(scale_max, sc);
        }
        for (int j = 0; j < QK_NVFP4 / 2; ++j) {
            const uint8_t q0 = block.qs[j] & 0x0f;
            const uint8_t q1 = block.qs[j] >> 4;
            q_hist[q0]++;
            q_hist[q1]++;
            q_nonzero += (q0 != 0 && q0 != 8) ? 1 : 0;
            q_nonzero += (q1 != 0 && q1 != 8) ? 1 : 0;
            q_saturated += (q0 == 7 || q0 == 15) ? 1 : 0;
            q_saturated += (q1 == 7 || q1 == 15) ? 1 : 0;
        }
    }

    std::vector<std::pair<int, uint64_t>> top_scales;
    top_scales.reserve(256);
    for (int i = 0; i < 256; ++i) {
        if (scale_hist[i] != 0) {
            top_scales.push_back({ i, scale_hist[i] });
        }
    }
    std::sort(top_scales.begin(), top_scales.end(), [](const auto & a, const auto & b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });

    const double total_subblocks = (double) std::max<uint64_t>(stats.n_subblocks, 1);
    const double total_q = (double) std::max<int64_t>(n_blocks * QK_NVFP4, 1);
    const double avg_candidates = stats.n_subblocks > 0 ?
            (double) stats.candidate_count_total / (double) stats.n_subblocks : 0.0;
    const double p6 = 100.0 * (double) stats.n_m6 / total_subblocks;
    const double p5 = 100.0 * (double) stats.n_m5 / total_subblocks;
    const double p4 = 100.0 * (double) stats.n_m4 / total_subblocks;
    const double p3 = 100.0 * (double) stats.n_slot3 / total_subblocks;
    const double p2 = 100.0 * (double) stats.n_slot2 / total_subblocks;
    const double p15 = 100.0 * (double) stats.n_slot15 / total_subblocks;
    const double p1 = 100.0 * (double) stats.n_slot1 / total_subblocks;
    const double p_other = 100.0 * (double) stats.n_slot_other / total_subblocks;
    const double source_rich_pct = 100.0 * (double) stats.n_source_rich / total_subblocks;
    const double source_cjso_ord_pct = 100.0 * (double) stats.n_source_cjso_ordinary / total_subblocks;
    const double source_cjso_wgt_pct = 100.0 * (double) stats.n_source_cjso_weighted / total_subblocks;
    const double source_both_pct = 100.0 * (double) stats.n_source_both / total_subblocks;
    const double source_fallback_pct = 100.0 * (double) stats.n_source_fallback / total_subblocks;
    const double ordinary_ratio = stats.mse_m6 > 0.0 ? stats.mse_selected / stats.mse_m6 : 1.0;
    const double weighted_ratio = stats.weighted_mse_m6 > 0.0 ? stats.weighted_mse_selected / stats.weighted_mse_m6 : 1.0;
    const double weighted_total = (double) std::max<uint64_t>(stats.n_weighted_subblocks, 1);
    const double ordinary_best_pct = 100.0 * (double) stats.n_ordinary_best_selected / weighted_total;
    const double weighted_best_pct = 100.0 * (double) stats.n_weighted_best_selected / weighted_total;
    const double knee_pct = 100.0 * (double) stats.n_knee_selected / weighted_total;

    LLAMA_LOG_INFO("%s: NVFP4 debug tensor: %s slice=%lld\n", prefix, tensor_name, (long long) slice);
    LLAMA_LOG_INFO("%s:   nrow=%lld n_per_row=%lld row_size=%zu total_subblocks=%llu\n", prefix,
            (long long) nrows, (long long) n_per_row, ggml_row_size(GGML_TYPE_NVFP4, n_per_row),
            (unsigned long long) stats.n_subblocks);
    LLAMA_LOG_INFO("%s:   RSF multiplier=%.6g search_algo=%s cjso_radius=%d select_mode=%s\n", prefix,
            (double) rsf_multiplier,
            ggml_nvfp4_search_algo_name(ggml_nvfp4_get_search_algo()),
            ggml_nvfp4_cjso_radius(),
            ggml_nvfp4_imatrix_select_mode_name(ggml_nvfp4_get_imatrix_select_mode()));
    LLAMA_LOG_INFO("%s:   candidates: %s\n", prefix, ggml_nvfp4_adaptive_candidates());
    LLAMA_LOG_INFO("%s:   slot distribution: 6=%.3f%%, 5=%.3f%%, 4=%.3f%%, 3=%.3f%%, 2=%.3f%%, 1.5=%.3f%%, 1=%.3f%%, other=%.3f%%\n",
            prefix, p6, p5, p4, p3, p2, p15, p1, p_other);
    LLAMA_LOG_INFO("%s:   source distribution: rich=%.3f%%, cjso_ord=%.3f%%, cjso_weighted=%.3f%%, both=%.3f%%, fallback=%.3f%%\n",
            prefix, source_rich_pct, source_cjso_ord_pct, source_cjso_wgt_pct, source_both_pct, source_fallback_pct);
    LLAMA_LOG_INFO("%s:   avg candidates=%.3f selected ordinary ratio=%.9g weighted ratio=%.9g\n",
            prefix, avg_candidates, ordinary_ratio, weighted_ratio);
    if (stats.n_weighted_subblocks > 0) {
        LLAMA_LOG_INFO("%s:   two-objective selected: ordinary_best=%.3f%% weighted_best=%.3f%% knee=%.3f%%\n",
                prefix, ordinary_best_pct, weighted_best_pct, knee_pct);
    }
    LLAMA_LOG_INFO("%s:   scale code min=%u max=%u q_nonzero=%.3f%% q_saturated=%.3f%%\n", prefix,
            (unsigned) scale_min, (unsigned) scale_max,
            100.0 * (double) q_nonzero / total_q,
            100.0 * (double) q_saturated / total_q);

    std::string top_line;
    const int n_top = std::min<int>(20, (int) top_scales.size());
    for (int i = 0; i < n_top; ++i) {
        char item[64];
        snprintf(item, sizeof(item), "%s%d:%llu", i == 0 ? "" : ",",
                top_scales[i].first, (unsigned long long) top_scales[i].second);
        top_line += item;
    }
    LLAMA_LOG_INFO("%s:   top scale codes: %s\n", prefix, top_line.empty() ? "(none)" : top_line.c_str());

    std::string q_line;
    for (int i = 0; i < 16; ++i) {
        char item[64];
        snprintf(item, sizeof(item), "%s%d:%llu", i == 0 ? "" : ",", i, (unsigned long long) q_hist[i]);
        q_line += item;
    }
    LLAMA_LOG_INFO("%s:   q nibble histogram: %s\n", prefix, q_line.c_str());
}

static const uint32_t LLAMA_QUANT_WHT_SIGNS1_256_BITS[8] = {
    0xc3284666u, 0xce93b542u, 0x79141579u, 0x9aa89715u,
    0x9b0404dau, 0x0af8ae67u, 0xef41f700u, 0xd712a44au
};

static const uint32_t LLAMA_QUANT_WHT_SIGNS2_256_BITS[8] = {
    0x6e2e718eu, 0x82fc60a0u, 0xb7719342u, 0x67487f5au,
    0xbfd09d07u, 0xaeadc1c4u, 0xd5c0b687u, 0x6c1b19a0u
};

static float llama_quant_wht_apply_sign_256(float x, const uint32_t * signs, int idx) {
    const uint32_t sign = ((signs[idx >> 5] >> (idx & 31)) & 1u) << 31;
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits ^= sign;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static void llama_quant_fwht_256(float * data) {
    for (int len = 1; len < LLAMA_QUANT_WHT_DIM; len <<= 1) {
        for (int base = 0; base < LLAMA_QUANT_WHT_DIM; base += 2 * len) {
            for (int i = 0; i < len; ++i) {
                const float a = data[base + i + 0];
                const float b = data[base + i + len];
                data[base + i + 0] = a + b;
                data[base + i + len] = a - b;
            }
        }
    }
}

static void llama_quant_wht_forward_256(float * data) {
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        data[i] = llama_quant_wht_apply_sign_256(data[i], LLAMA_QUANT_WHT_SIGNS1_256_BITS, i);
    }
    llama_quant_fwht_256(data);
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        data[i] = llama_quant_wht_apply_sign_256(data[i] * 0.0625f, LLAMA_QUANT_WHT_SIGNS2_256_BITS, i);
    }
}

static void llama_quant_wht_inverse_256(float * data) {
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        data[i] = llama_quant_wht_apply_sign_256(data[i], LLAMA_QUANT_WHT_SIGNS2_256_BITS, i);
    }
    llama_quant_fwht_256(data);
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        data[i] = llama_quant_wht_apply_sign_256(data[i] * 0.0625f, LLAMA_QUANT_WHT_SIGNS1_256_BITS, i);
    }
}

static bool llama_quant_wht_type_supported(ggml_type type) {
    return type == GGML_TYPE_Q2_K ||
           type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_IQ1_S ||
           type == GGML_TYPE_IQ1_M ||
           type == GGML_TYPE_IQ2_XXS ||
           type == GGML_TYPE_IQ2_XS ||
           type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ3_XXS ||
           type == GGML_TYPE_IQ3_S ||
           type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_IQ4_XS;
}

static const char * llama_quant_wht_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:    return "Q2_K";
        case GGML_TYPE_Q3_K:    return "Q3_K";
        case GGML_TYPE_Q4_K:    return "Q4_K";
        case GGML_TYPE_Q5_K:    return "Q5_K";
        case GGML_TYPE_Q6_K:    return "Q6_K";
        case GGML_TYPE_Q8_0:    return "Q8_0";
        case GGML_TYPE_IQ1_S:   return "IQ1_S";
        case GGML_TYPE_IQ1_M:   return "IQ1_M";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS:  return "IQ2_XS";
        case GGML_TYPE_IQ2_S:   return "IQ2_S";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ3_S:   return "IQ3_S";
        case GGML_TYPE_IQ4_NL:  return "IQ4_NL";
        case GGML_TYPE_IQ4_XS:  return "IQ4_XS";
        default:                return nullptr;
    }
}

static std::string llama_quant_wht_normalize_type_token(std::string token) {
    token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) { return std::isspace(c) != 0; }), token.end());
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) { return (char) std::toupper(c); });
    return token;
}

static ggml_type llama_quant_wht_parse_type_token(const std::string & token) {
    const std::string name = llama_quant_wht_normalize_type_token(token);
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type type = (ggml_type) i;
        const char * type_name = llama_quant_wht_type_name(type);
        if (type_name != nullptr && name == type_name) {
            return type;
        }
    }
    return GGML_TYPE_COUNT;
}

static bool llama_quant_wht_skip_list_has(const std::string & skip_types, ggml_type type) {
    size_t start = 0;
    while (start <= skip_types.size()) {
        const size_t end = skip_types.find(',', start);
        const std::string token = skip_types.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!llama_quant_wht_normalize_type_token(token).empty()) {
            const ggml_type parsed = llama_quant_wht_parse_type_token(token);
            if (parsed == GGML_TYPE_COUNT || !llama_quant_wht_type_supported(parsed)) {
                throw std::runtime_error(format("unsupported quant_wht skip type: %s", token.c_str()));
            }
            if (parsed == type) {
                return true;
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
}

static std::string llama_quant_wht_normalize_skip_types(const char * skip_types) {
    const std::string input = skip_types == nullptr ? LLAMA_QUANT_WHT_DEFAULT_SKIP_TYPES : skip_types;
    std::string result;
    size_t start = 0;
    while (start <= input.size()) {
        const size_t end = input.find(',', start);
        const std::string token = input.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!llama_quant_wht_normalize_type_token(token).empty()) {
            const ggml_type parsed = llama_quant_wht_parse_type_token(token);
            if (parsed == GGML_TYPE_COUNT || !llama_quant_wht_type_supported(parsed)) {
                throw std::runtime_error(format("unsupported quant_wht skip type: %s", token.c_str()));
            }
            if (!llama_quant_wht_skip_list_has(result, parsed)) {
                if (!result.empty()) {
                    result += ",";
                }
                result += llama_quant_wht_type_name(parsed);
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return result;
}

static bool llama_quant_wht_type_enabled(ggml_type type, const std::string & skip_types) {
    return llama_quant_wht_type_supported(type) &&
           !llama_quant_wht_skip_list_has(skip_types, type);
}

static bool llama_quant_wht_name_supported(const std::string & name, tensor_category category) {
    if (category == tensor_category::TOKEN_EMBD) {
        return false;
    }
    if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0) {
        return false;
    }
    if (name.find("norm") != std::string::npos ||
        name.find("rope") != std::string::npos ||
        name.find("conv") != std::string::npos ||
        name.find("ssm_alpha") != std::string::npos ||
        name.find("ssm_beta") != std::string::npos ||
        name.find("ssm_ba") != std::string::npos ||
        name.find("ssm_dt") != std::string::npos) {
        return false;
    }
    return true;
}

static void llama_quant_wht_inverse_rows_256(float * data, int64_t nrows, int64_t n_per_row, std::vector<std::thread> & workers, int nthread) {
    GGML_ASSERT(n_per_row % LLAMA_QUANT_WHT_DIM == 0);

    std::mutex mutex;
    int64_t next_row = 0;

    auto compute = [&]() {
        while (true) {
            int64_t row = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                row = next_row++;
            }
            if (row >= nrows) {
                break;
            }

            float * row_data = data + row * n_per_row;
            for (int64_t col = 0; col < n_per_row; col += LLAMA_QUANT_WHT_DIM) {
                llama_quant_wht_inverse_256(row_data + col);
            }
        }
    };

    const int nthread_use = nthread > 1 ? nthread : 1;
    for (int it = 0; it < nthread_use - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
    workers.clear();
}

//
// tensor categorization for quantization
//
// (this is different from LLM_TN - we want broad categories, not specific tensor names per arch)
//

static tensor_category tensor_get_category(const std::string & tensor_name) {
    if (tensor_name_match_output_weight(tensor_name.c_str())) {
        return tensor_category::OUTPUT;
    }
    if (tensor_name_match_token_embd(tensor_name.c_str())) {
        return tensor_category::TOKEN_EMBD;
    }
    if (tensor_name.find("attn_qkv.weight") != std::string::npos) {
        return tensor_category::ATTENTION_QKV;
    }
    if (tensor_name.find("attn_kv_b.weight") != std::string::npos) {
        return tensor_category::ATTENTION_KV_B;
    }
    if (tensor_name.find("attn_v.weight") != std::string::npos) {
        return tensor_category::ATTENTION_V;
    }
    if (tensor_name.find("attn_k.weight") != std::string::npos) {
        return tensor_category::ATTENTION_K;
    }
    if (tensor_name.find("attn_q.weight") != std::string::npos) {
        return tensor_category::ATTENTION_Q;
    }
    if (tensor_name.find("attn_output.weight") != std::string::npos) {
        return tensor_category::ATTENTION_OUTPUT;
    }
    if (tensor_name.find("ffn_up") != std::string::npos) {
        return tensor_category::FFN_UP;
    }
    if (tensor_name.find("ffn_gate") != std::string::npos) {
        return tensor_category::FFN_GATE;
    }
    if (tensor_name.find("ffn_down") != std::string::npos) {
        return tensor_category::FFN_DOWN;
    }
    return tensor_category::OTHER;
}

// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool category_is_attn_v(tensor_category cat) {
    return cat == tensor_category::ATTENTION_V     ||
           cat == tensor_category::ATTENTION_QKV   ||
           cat == tensor_category::ATTENTION_KV_B;
}

static int tensor_layer_from_name_or(const std::string & name, int fallback, int n_layers) {
    int i_layer = fallback;
    if (sscanf(name.c_str(), "blk.%d.", &i_layer) == 1 && i_layer >= 0 && i_layer < n_layers) {
        return i_layer;
    }
    return fallback;
}

//
// quantization state
//

struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
    bool has_tied_embeddings = true; // assume tied until we see output.weight

    // tensor type override patterns (compiled once, used twice)
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params):
        model(model), params(params)
    {
        // compile regex patterns once - they are expensive
        if (params->tt_overrides) {
            for (const auto * p = params->tt_overrides; p->pattern != nullptr; p++) {
                tensor_type_patterns.emplace_back(std::regex(p->pattern), p->type);
            }
        }
    }
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    std::string     name;
    ggml_type       target_type;
    tensor_category category;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
    bool            quant_wht_rotate;
};

struct nvfp4_m_policy_bucket {
    uint64_t count = 0;
    size_t bytes = 0;
};

struct nvfp4_m_policy_summary {
    nvfp4_m_policy_bucket nvfp4;
    nvfp4_m_policy_bucket q5_k;
    nvfp4_m_policy_bucket q6_k;
    nvfp4_m_policy_bucket q8_0;
    nvfp4_m_policy_bucket other;
};

static size_t tensor_size_as_type(const ggml_tensor * tensor, ggml_type type) {
    return (size_t) ggml_nrows(tensor) * ggml_row_size(type, tensor->ne[0]);
}

static void nvfp4_m_policy_summary_add(nvfp4_m_policy_summary & summary, const ggml_tensor * tensor, ggml_type type) {
    nvfp4_m_policy_bucket * bucket = nullptr;
    switch (type) {
        case GGML_TYPE_NVFP4: bucket = &summary.nvfp4; break;
        case GGML_TYPE_Q5_K:  bucket = &summary.q5_k;  break;
        case GGML_TYPE_Q6_K:  bucket = &summary.q6_k;  break;
        case GGML_TYPE_Q8_0:  bucket = &summary.q8_0;  break;
        default:              bucket = &summary.other; break;
    }

    bucket->count++;
    bucket->bytes += tensor_size_as_type(tensor, type);
}

static void nvfp4_m_policy_summary_print(const nvfp4_m_policy_summary & summary) {
    const char * prefix = __func__;
    auto print_bucket = [prefix](const char * name, const nvfp4_m_policy_bucket & bucket) {
        LLAMA_LOG_INFO("%s:   %-5s count=%4llu bytes=%10.2f MiB\n", prefix, name,
                (unsigned long long) bucket.count, bucket.bytes / 1024.0 / 1024.0);
    };

    LLAMA_LOG_INFO("%s: NVFP4_M policy enabled\n", prefix);
    LLAMA_LOG_INFO("%s:   default type: NVFP4\n", prefix);
    LLAMA_LOG_INFO("%s:   overrides / target type summary:\n", prefix);
    print_bucket("NVFP4", summary.nvfp4);
    print_bucket("Q5_K",  summary.q5_k);
    print_bucket("Q6_K",  summary.q6_k);
    print_bucket("Q8_0",  summary.q8_0);
    print_bucket("other", summary.other);
}

//
// dequantization
//

static void llama_tensor_dequantize_impl(
    ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    const bool quant_wht = (tensor->flags & GGML_TENSOR_FLAG_QUANT_WHT) != 0;
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    auto apply_quant_wht_inverse = [&]() {
        if (!quant_wht) {
            return;
        }
        const int64_t n_per_row = tensor->ne[0];
        if (n_per_row % LLAMA_QUANT_WHT_DIM != 0 || nelements % (size_t) n_per_row != 0) {
            throw std::runtime_error(format("cannot inverse quant_wht tensor %s: reduction dimension %" PRId64 " is not divisible by %d",
                        tensor->name, n_per_row, LLAMA_QUANT_WHT_DIM));
        }
        llama_quant_wht_inverse_rows_256(f32_output, (int64_t) (nelements / (size_t) n_per_row), n_per_row, workers, nthread);
    };

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype->to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        apply_quant_wht_inverse();
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int64_t nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype->to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, (int64_t) thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
    apply_quant_wht_inverse();
}

//
// do we allow this tensor to be quantized?
//

static bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor) {
    // trivial checks first -- no string ops needed
    if (params->only_copy)       return false;

    // quantize only 2D and 3D tensors (experts)
    if (ggml_n_dims(tensor) < 2) return false;

    const std::string name = ggml_get_name(tensor);

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

    // do not quantize norm tensors
    quantize &= name.find("_norm.weight") == std::string::npos;

    quantize &= params->quantize_output_tensor || name != "output.weight";

    // do not quantize expert gating tensors
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

    // these are very small (e.g. 4x4)
    quantize &= name.find("altup")  == std::string::npos;
    quantize &= name.find("laurel") == std::string::npos;

    // these are not too big so keep them as it is
    quantize &= name.find("per_layer_model_proj") == std::string::npos;

    // do not quantize positional embeddings and token types (BERT)
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_POS_EMBD,    "weight");
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

    // do not quantize Mamba/Kimi's small conv1d weights
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ssm_conv1d") == std::string::npos;
    quantize &= name.find("shortconv.conv.weight") == std::string::npos;

    // do not quantize RWKV's small yet 2D weights
    quantize &= name.find("time_mix_first.weight") == std::string::npos;
    quantize &= name.find("time_mix_w0.weight") == std::string::npos;
    quantize &= name.find("time_mix_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_v0.weight") == std::string::npos;
    quantize &= name.find("time_mix_v1.weight") == std::string::npos;
    quantize &= name.find("time_mix_v2.weight") == std::string::npos;
    quantize &= name.find("time_mix_a0.weight") == std::string::npos;
    quantize &= name.find("time_mix_a1.weight") == std::string::npos;
    quantize &= name.find("time_mix_a2.weight") == std::string::npos;
    quantize &= name.find("time_mix_g1.weight") == std::string::npos;
    quantize &= name.find("time_mix_g2.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

    // do not quantize relative position bias (T5)
    quantize &= name.find("attn_rel_b.weight") == std::string::npos;

    // do not quantize specific multimodal tensors
    quantize &= name.find(".position_embd") == std::string::npos;
    quantize &= name.find("sam.pos_embd")   == std::string::npos;
    quantize &= name.find("sam.neck.")      == std::string::npos;
    quantize &= name.find("sam.net_")       == std::string::npos;
    quantize &= name.find(".rel_pos")       == std::string::npos;
    quantize &= name.find(".patch_embd")    == std::string::npos;
    quantize &= name.find(".patch_merger")  == std::string::npos;

    return quantize;
}

//
// tensor type selection
//

static bool ftype_is_pq_0(const llama_ftype ftype) {
    return ftype == LLAMA_FTYPE_MOSTLY_PQ2_0 ||
           ftype == LLAMA_FTYPE_MOSTLY_PQ3_0 ||
           ftype == LLAMA_FTYPE_MOSTLY_PQ4_0;
}

static bool ftype_is_pq_k(const llama_ftype ftype) {
    return ftype == LLAMA_FTYPE_MOSTLY_PQ2_K ||
           ftype == LLAMA_FTYPE_MOSTLY_PQ3_K ||
           ftype == LLAMA_FTYPE_MOSTLY_PQ4_K;
}

static ggml_type pq0_reference_default_type(const llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_PQ2_0:
        case LLAMA_FTYPE_MOSTLY_PQ3_0:
        case LLAMA_FTYPE_MOSTLY_PQ4_0:
            return GGML_TYPE_Q4_0;
        default:
            return GGML_TYPE_COUNT;
    }
}

static ggml_type pq0_map_reference_type(const llama_ftype ftype, const ggml_type type) {
    if (type != GGML_TYPE_Q4_0) {
        return type;
    }

    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_PQ2_0: return GGML_TYPE_PQ2_0;
        case LLAMA_FTYPE_MOSTLY_PQ3_0: return GGML_TYPE_PQ3_0;
        case LLAMA_FTYPE_MOSTLY_PQ4_0: return GGML_TYPE_PQ4_0;
        default:                       return type;
    }
}

static llama_ftype pqk_reference_ftype(const llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_PQ2_K: return LLAMA_FTYPE_MOSTLY_Q2_K;
        case LLAMA_FTYPE_MOSTLY_PQ3_K: return LLAMA_FTYPE_MOSTLY_Q3_K_M;
        case LLAMA_FTYPE_MOSTLY_PQ4_K: return LLAMA_FTYPE_MOSTLY_Q4_K_M;
        default:                       return ftype;
    }
}

static ggml_type pqk_reference_default_type(const llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_PQ2_K: return GGML_TYPE_Q2_K;
        case LLAMA_FTYPE_MOSTLY_PQ3_K: return GGML_TYPE_Q3_K;
        case LLAMA_FTYPE_MOSTLY_PQ4_K: return GGML_TYPE_Q4_K;
        default:                       return GGML_TYPE_COUNT;
    }
}

static ggml_type pqk_map_reference_type(const llama_ftype ftype, const ggml_type type) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_PQ2_K:
            if (type == GGML_TYPE_Q2_K) return GGML_TYPE_PQ2_K;
            if (type == GGML_TYPE_Q3_K) return GGML_TYPE_PQ3_K;
            if (type == GGML_TYPE_Q4_K) return GGML_TYPE_PQ4_K;
            return type;
        case LLAMA_FTYPE_MOSTLY_PQ3_K:
            if (type == GGML_TYPE_Q3_K) return GGML_TYPE_PQ3_K;
            if (type == GGML_TYPE_Q4_K) return GGML_TYPE_PQ4_K;
            return type;
        case LLAMA_FTYPE_MOSTLY_PQ4_K:
            if (type == GGML_TYPE_Q4_K) return GGML_TYPE_PQ4_K;
            return type;
        default:
            return type;
    }
}

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantize_state_impl & qs, const ggml_tensor * t, const ggml_type target_type) {
    ggml_type return_type = target_type;

    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);
    int64_t required_cols = qk_k;

    switch (target_type) {
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_PQ3_0:
        case GGML_TYPE_PQ4_0:
            required_cols = 128;
            break;
        default: break;
    }

    if (ncols % required_cols != 0) { // this tensor's shape is incompatible with this quant
        LLAMA_LOG_WARN("warning: %-36s - ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
                        t->name, ncols, required_cols, ggml_type_name(target_type));
        ++qs.n_fallback;

        switch (target_type) {
            // types on the left: block size 256
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:   // types on the right: block size 32
            case GGML_TYPE_IQ4_XS:  return_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:   return_type = GGML_TYPE_Q4_0;   break;
            case GGML_TYPE_PQ2_0:
            case GGML_TYPE_PQ3_0:
            case GGML_TYPE_PQ4_0:
            case GGML_TYPE_PQ2_K:
            case GGML_TYPE_PQ3_K:
            case GGML_TYPE_PQ4_K:   return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;
            default:
                throw std::runtime_error(format("no tensor type fallback is defined for type %s",
                                                ggml_type_name(target_type)));
        }
        if (ncols % ggml_blck_size(return_type) != 0) {
            //
            // the fallback return type is still not compatible for this tensor!
            //
            // most likely, this tensor's first dimension is not divisible by 32.
            // this is very rare. we can either abort the quantization, or
            // fallback to F16 / F32.
            //
            LLAMA_LOG_WARN("(WARNING: must use F16 due to unusual shape) ");
            return_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN("-> falling back to %7s\n", ggml_type_name(return_type));
    }
    return return_type;
}

// internal standard logic for selecting the target tensor type based on tensor category, ftype, and model arch
static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype, tensor_category category) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (category == tensor_category::OUTPUT || (qs.has_tied_embeddings && category == tensor_category::TOKEN_EMBD)) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else {
            const int64_t nx = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);

            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (category == tensor_category::TOKEN_EMBD) {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype_is_pq_k(ftype)) {
                const llama_ftype ref_ftype = pqk_reference_ftype(ftype);
                new_type = llama_tensor_get_type_impl(qs, pqk_reference_default_type(ftype), tensor, ref_ftype, category);
                new_type = pqk_map_reference_type(ftype, new_type);
            }
            else if (ftype_is_pq_0(ftype)) {
                new_type = llama_tensor_get_type_impl(qs, pq0_reference_default_type(ftype), tensor, LLAMA_FTYPE_MOSTLY_Q4_0, category);
                new_type = pq0_map_reference_type(ftype, new_type);
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0 || ftype == LLAMA_FTYPE_MOSTLY_Q2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_NVFP4_M) {
        const int n_layers = (int) qs.model.hparams.n_layer();

        if (name.find("ssm_alpha.weight") != std::string::npos ||
            name.find("ssm_beta.weight")  != std::string::npos) {
            new_type = GGML_TYPE_Q8_0;
        } else if (name.find("ssm_out.weight")  != std::string::npos ||
                   name.find("attn_gate.weight") != std::string::npos) {
            new_type = GGML_TYPE_Q5_K;
        } else if (category == tensor_category::ATTENTION_QKV ||
                   category == tensor_category::ATTENTION_K   ||
                   category == tensor_category::ATTENTION_V   ||
                   category == tensor_category::ATTENTION_OUTPUT) {
            const int i_layer = tensor_layer_from_name_or(name, qs.i_attention_wv, n_layers);
            if (use_more_bits(i_layer, n_layers)) {
                new_type = GGML_TYPE_Q5_K;
            }
        } else if (category == tensor_category::FFN_DOWN) {
            const int i_layer = tensor_layer_from_name_or(name, qs.i_ffn_down, n_layers);
            if (use_more_bits(i_layer, n_layers)) {
                new_type = GGML_TYPE_Q5_K;
            }
        }

        if (category_is_attn_v(category)) {
            ++qs.i_attention_wv;
        } else if (category == tensor_category::FFN_DOWN) {
            ++qs.i_ffn_down;
        } else if (category == tensor_category::FFN_GATE) {
            ++qs.i_ffn_gate;
        } else if (category == tensor_category::FFN_UP) {
            ++qs.i_ffn_up;
        }
    } else if (ftype_is_pq_k(ftype)) {
        const llama_ftype ref_ftype = pqk_reference_ftype(ftype);
        new_type = llama_tensor_get_type_impl(qs, pqk_reference_default_type(ftype), tensor, ref_ftype, category);
        new_type = pqk_map_reference_type(ftype, new_type);
    } else if (ftype_is_pq_0(ftype)) {
        new_type = llama_tensor_get_type_impl(qs, pq0_reference_default_type(ftype), tensor, LLAMA_FTYPE_MOSTLY_Q4_0, category);
        new_type = pq0_map_reference_type(ftype, new_type);
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (category_is_attn_v(category)) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && category == tensor_category::ATTENTION_K) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (category == tensor_category::FFN_DOWN) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (category == tensor_category::ATTENTION_OUTPUT) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (category_is_attn_v(category)) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (category == tensor_category::ATTENTION_K) {
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::ATTENTION_Q) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::FFN_DOWN) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (category == tensor_category::ATTENTION_OUTPUT) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (category == tensor_category::ATTENTION_QKV) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (category == tensor_category::FFN_GATE) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    }
    else if (category == tensor_category::FFN_UP) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    return new_type;
}

// outer wrapper: determine the ggml_type that this tensor should be quantized to
static ggml_type llama_tensor_get_type(quantize_state_impl & qs, const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type default_type, const tensor_metadata & tm) {
    if (!tensor_allows_quantization(params, qs.model.arch, tensor)) {
        return tensor->type;
    }
    if (params->token_embedding_type < GGML_TYPE_COUNT && tm.category == tensor_category::TOKEN_EMBD) {
        return params->token_embedding_type;
    }
    if (params->output_tensor_type < GGML_TYPE_COUNT && tm.category == tensor_category::OUTPUT) {
        return params->output_tensor_type;
    }

    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (ggml_is_quantized(default_type)) {
        // if the user provided tensor types - use those
        bool manual = false;
        if (!qs.tensor_type_patterns.empty()) {
            const std::string tensor_name(tensor->name);
            for (const auto & [pattern, qtype] : qs.tensor_type_patterns) {
                if (std::regex_search(tensor_name, pattern)) {
                    if (qtype != new_type) {
                        LLAMA_LOG_WARN("%s: %-36s - applying manual override: %s -> %s\n",
                                       __func__, tensor_name.c_str(), ggml_type_name(new_type), ggml_type_name(qtype));
                        new_type = qtype;
                    }
                    manual = true;
                    break;
                }
            }
        }

        // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
        if (!manual && !params->pure) {
            new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category);
        }

        // incompatible tensor shapes are handled here - fallback to a compatible type
        new_type = tensor_type_fallback(qs, tensor, new_type);
    }

    return new_type;
}

//
// quantization implementation
//

struct llama_nvfp4_rsf_override_scope {
    bool active = false;

    void set(float multiplier) {
        ggml_nvfp4_set_rsf_multiplier_override(multiplier);
        active = true;
    }

    ~llama_nvfp4_rsf_override_scope() {
        if (active) {
            ggml_nvfp4_clear_rsf_multiplier_override();
        }
    }
};

static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix);
            local_size += this_size;

            // validate the quantized data
            const size_t row_size  = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}

static void llama_quant_wht_rotate_rows_256(const float * src, float * dst, int64_t nrows, int64_t n_per_row, std::vector<std::thread> & workers, int nthread) {
    GGML_ASSERT(n_per_row % LLAMA_QUANT_WHT_DIM == 0);

    auto compute = [src, dst, n_per_row](int64_t row_begin, int64_t row_end) {
        for (int64_t row = row_begin; row < row_end; ++row) {
            const float * src_row = src + row * n_per_row;
            float * dst_row = dst + row * n_per_row;
            for (int64_t col = 0; col < n_per_row; col += LLAMA_QUANT_WHT_DIM) {
                memcpy(dst_row + col, src_row + col, sizeof(float) * LLAMA_QUANT_WHT_DIM);
                llama_quant_wht_forward_256(dst_row + col);
            }
        }
    };

    if (nthread < 2 || nrows < 2) {
        compute(0, nrows);
        return;
    }

    const int64_t rows_per_thread = (nrows + nthread - 1) / nthread;
    for (int ith = 0; ith < nthread - 1; ++ith) {
        const int64_t row_begin = ith * rows_per_thread;
        const int64_t row_end = std::min<int64_t>(nrows, row_begin + rows_per_thread);
        if (row_begin < row_end) {
            workers.emplace_back(compute, row_begin, row_end);
        }
    }
    const int64_t row_begin = (nthread - 1) * rows_per_thread;
    if (row_begin < nrows) {
        compute(row_begin, nrows);
    }
    for (auto & w : workers) {
        w.join();
    }
    workers.clear();
}

static float llama_quant_wht_exact_weighted_loss_256(
        ggml_type type, const float * rotated, const void * qdata, const float * imatrix,
        float * dequant, float * residual) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    GGML_ASSERT(traits->to_float != nullptr);

    traits->to_float(qdata, dequant, LLAMA_QUANT_WHT_DIM);
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        residual[i] = dequant[i] - rotated[i];
    }
    llama_quant_wht_inverse_256(residual);

    float loss = 0.0f;
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        const float r = residual[i];
        loss += imatrix[i] * r * r;
    }
    return loss;
}

static void llama_quant_wht_make_residual_weights_256(
        ggml_type type, const float * rotated, const void * qdata, const float * imatrix,
        float * dequant, float * residual, float * proposal) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    GGML_ASSERT(traits->to_float != nullptr);

    float mean_w = 0.0f;
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        mean_w += imatrix[i];
    }
    mean_w /= LLAMA_QUANT_WHT_DIM;
    const float floor_w = fmaxf(mean_w * 1e-6f, 1e-12f);

    traits->to_float(qdata, dequant, LLAMA_QUANT_WHT_DIM);
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        residual[i] = dequant[i] - rotated[i];
    }

    float weighted_original[LLAMA_QUANT_WHT_DIM];
    memcpy(weighted_original, residual, sizeof(weighted_original));
    llama_quant_wht_inverse_256(weighted_original);
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        weighted_original[i] *= imatrix[i];
    }
    llama_quant_wht_forward_256(weighted_original);

    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        const float denom = fabsf(residual[i]) + 1e-6f;
        proposal[i] = fmaxf(fabsf(weighted_original[i]) / denom, floor_w);
    }
}

static void llama_quant_wht_make_transformed_weights_256(const float * imatrix, float * proposal) {
    float mean_w = 0.0f;
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        mean_w += imatrix[i];
        proposal[i] = imatrix[i];
    }
    mean_w /= LLAMA_QUANT_WHT_DIM;
    const float floor_w = fmaxf(mean_w * 1e-6f, 1e-12f);

    llama_quant_wht_forward_256(proposal);
    for (int i = 0; i < LLAMA_QUANT_WHT_DIM; ++i) {
        proposal[i] = fmaxf(fabsf(proposal[i]), floor_w);
    }
}

static bool llama_quant_wht_quantize_candidate_256(
        ggml_type type, const float * rotated, const float * proposal_weights, uint8_t * qdata, size_t qsize) {
    const size_t wrote = ggml_quantize_chunk(type, rotated, qdata, 0, 1, LLAMA_QUANT_WHT_DIM, proposal_weights);
    GGML_ASSERT(wrote == qsize);
    return ggml_validate_row_data(type, qdata, qsize);
}

static size_t llama_tensor_quantize_wht_imatrix_impl(
        ggml_type type, const float * rotated_data, void * new_data, int64_t nrows, int64_t n_per_row,
        const float * imatrix, std::vector<std::thread> & workers, int nthread) {
    GGML_ASSERT(imatrix != nullptr);
    GGML_ASSERT(n_per_row % LLAMA_QUANT_WHT_DIM == 0);

    const size_t row_size = ggml_row_size(type, n_per_row);
    const size_t qsize = ggml_row_size(type, LLAMA_QUANT_WHT_DIM);
    const int64_t blck_size = ggml_blck_size(type);
    const size_t type_size = ggml_type_size(type);
    const bool requires_imatrix = ggml_quantize_requires_imatrix(type);

    bool valid = true;
    std::mutex mutex;
    int64_t next_row = 0;

    auto compute = [&]() {
        std::vector<uint8_t> best(qsize);
        std::vector<uint8_t> trial(qsize);
        std::vector<float> dequant(LLAMA_QUANT_WHT_DIM);
        std::vector<float> residual(LLAMA_QUANT_WHT_DIM);
        std::vector<float> proposal(LLAMA_QUANT_WHT_DIM);

        while (true) {
            int64_t row = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                row = next_row++;
            }
            if (row >= nrows) {
                break;
            }

            const float * row_rotated = rotated_data + row * n_per_row;
            char * row_dst = (char *) new_data + row * row_size;

            for (int64_t col = 0; col < n_per_row; col += LLAMA_QUANT_WHT_DIM) {
                const float * block_rotated = row_rotated + col;
                const float * block_imatrix = imatrix + col;
                char * block_dst = row_dst + (col / blck_size) * type_size;

                if (!llama_quant_wht_quantize_candidate_256(type, block_rotated, requires_imatrix ? block_imatrix : nullptr, best.data(), qsize)) {
                    std::unique_lock<std::mutex> lock(mutex);
                    valid = false;
                    return;
                }
                float best_loss = llama_quant_wht_exact_weighted_loss_256(
                        type, block_rotated, best.data(), block_imatrix, dequant.data(), residual.data());

                if (type != GGML_TYPE_Q8_0) {
                    if (!requires_imatrix) {
                        if (!llama_quant_wht_quantize_candidate_256(type, block_rotated, block_imatrix, trial.data(), qsize)) {
                            std::unique_lock<std::mutex> lock(mutex);
                            valid = false;
                            return;
                        }
                        float loss = llama_quant_wht_exact_weighted_loss_256(
                                type, block_rotated, trial.data(), block_imatrix, dequant.data(), residual.data());
                        if (loss < best_loss) {
                            best_loss = loss;
                            std::swap(best, trial);
                        }
                    }

                    llama_quant_wht_make_transformed_weights_256(block_imatrix, proposal.data());
                    if (!llama_quant_wht_quantize_candidate_256(type, block_rotated, proposal.data(), trial.data(), qsize)) {
                        std::unique_lock<std::mutex> lock(mutex);
                        valid = false;
                        return;
                    }
                    float loss = llama_quant_wht_exact_weighted_loss_256(
                            type, block_rotated, trial.data(), block_imatrix, dequant.data(), residual.data());
                    if (loss < best_loss) {
                        best_loss = loss;
                        std::swap(best, trial);
                    }

                    llama_quant_wht_make_residual_weights_256(
                            type, block_rotated, best.data(), block_imatrix, dequant.data(), residual.data(), proposal.data());
                    if (!llama_quant_wht_quantize_candidate_256(type, block_rotated, proposal.data(), trial.data(), qsize)) {
                        std::unique_lock<std::mutex> lock(mutex);
                        valid = false;
                        return;
                    }
                    loss = llama_quant_wht_exact_weighted_loss_256(
                            type, block_rotated, trial.data(), block_imatrix, dequant.data(), residual.data());
                    if (loss < best_loss) {
                        std::swap(best, trial);
                    }
                }

                memcpy(block_dst, best.data(), qsize);
            }
        }
    };

    const int nthread_use = nthread > 1 ? nthread : 1;
    for (int it = 0; it < nthread_use - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
    workers.clear();

    if (!valid || !ggml_validate_row_data(type, new_data, nrows * row_size)) {
        throw std::runtime_error("quant_wht imatrix quantized data validation failed");
    }
    return nrows * row_size;
}

//
// imatrix requirement check
//

static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) {
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}

//
// given a file type, get the default tensor type
//

ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return GGML_TYPE_Q5_0;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return GGML_TYPE_Q5_1;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_F16:  return GGML_TYPE_F16;
        case LLAMA_FTYPE_MOSTLY_BF16: return GGML_TYPE_BF16;
        case LLAMA_FTYPE_ALL_F32:     return GGML_TYPE_F32;
        case LLAMA_FTYPE_MOSTLY_Q1_0: return GGML_TYPE_Q1_0;
        case LLAMA_FTYPE_MOSTLY_Q2_0: return GGML_TYPE_Q2_0;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return GGML_TYPE_MXFP4;
        case LLAMA_FTYPE_MOSTLY_NVFP4:     return GGML_TYPE_NVFP4;
        case LLAMA_FTYPE_MOSTLY_NVFP4_M:   return GGML_TYPE_NVFP4;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    return GGML_TYPE_Q2_K;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  return GGML_TYPE_IQ3_S;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  return GGML_TYPE_Q3_K;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  return GGML_TYPE_Q4_K;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  return GGML_TYPE_Q5_K;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    return GGML_TYPE_Q6_K;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   return GGML_TYPE_TQ1_0;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;
        case LLAMA_FTYPE_MOSTLY_PQ2_0:   return GGML_TYPE_PQ2_0;
        case LLAMA_FTYPE_MOSTLY_PQ3_0:   return GGML_TYPE_PQ3_0;
        case LLAMA_FTYPE_MOSTLY_PQ4_0:   return GGML_TYPE_PQ4_0;
        case LLAMA_FTYPE_MOSTLY_PQ2_K:   return GGML_TYPE_PQ2_K;
        case LLAMA_FTYPE_MOSTLY_PQ3_K:   return GGML_TYPE_PQ3_K;
        case LLAMA_FTYPE_MOSTLY_PQ4_K:   return GGML_TYPE_PQ4_K;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return GGML_TYPE_IQ2_S;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return GGML_TYPE_IQ1_S;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return GGML_TYPE_IQ3_S;

        default: return GGML_TYPE_COUNT;
    }
}


static void init_quantize_state_counters(quantize_state_impl & qs, std::vector<tensor_metadata> & metadata) {
    for (auto & tm : metadata) {
        tensor_category cat = tensor_get_category(tm.name);
        tm.category = cat;

        if (category_is_attn_v(cat)) {
            ++qs.n_attention_wv;
        }

        if (cat == tensor_category::OUTPUT) {
            qs.has_tied_embeddings = false;
        }
    }
    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer_all;
}

//
// main quantization driver
//

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    llama_ftype ftype = params->ftype;

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    ggml_type default_type = llama_ftype_get_default_type(ftype);
    if (default_type == GGML_TYPE_COUNT) {
        throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    const bool nvfp4_mode_relevant = default_type == GGML_TYPE_NVFP4 || params->nvfp4_scale_mode != LLAMA_NVFP4_SCALE_MODE_M6;
    ggml_nvfp4_set_scale_mode(llama_nvfp4_scale_mode_to_ggml(params->nvfp4_scale_mode));
    ggml_nvfp4_reset_scale_stats();
    if (nvfp4_mode_relevant) {
        LLAMA_LOG_INFO("%s: NVFP4 scale mode: %s\n", __func__, llama_nvfp4_scale_mode_name(params->nvfp4_scale_mode));
        llama_nvfp4_log_runtime_config(__func__, true);
    }

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    const llama_model_kv_override * kv_overrides = params->kv_overrides;
    std::vector<std::string> splits = {};
    llama_model_loader ml(/*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr,
        fname_inp, splits, /*file*/ nullptr, use_mmap, /*use_direct_io*/ false, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    auto mparams = llama_model_default_params();
    std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, mparams));

    auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
    if (model == nullptr) {
        GGML_ABORT("fatal error: model does not implement llama_model_base");
    }

    model->load_hparams(ml);
    model->load_stats  (ml);

    quantize_state_impl qs(*model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    const bool quant_wht_enabled = params->quant_wht || params->quant_wht_full || params->quant_wht_skip_types != nullptr;
    const std::string quant_wht_skip_types =
        quant_wht_enabled && !params->quant_wht_full ? llama_quant_wht_normalize_skip_types(params->quant_wht_skip_types) : "";
    if (quant_wht_enabled && params->quant_wht_dim != LLAMA_QUANT_WHT_DIM) {
        throw std::runtime_error(format("quant_wht_dim %u is unsupported; only %d is supported", params->quant_wht_dim, LLAMA_QUANT_WHT_DIM));
    }
    std::unordered_map<std::string, std::vector<float>> i_data;
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        for (const llama_model_imatrix_data * p = params->imatrix; p->name != nullptr; p++) {
            i_data.emplace(p->name, std::vector<float>(p->data, p->data + p->size));
        }
        imatrix_data = & i_data;
        if (imatrix_data) {
            LLAMA_LOG_INFO("\n%s: have importance matrix data with %d entries\n",
                           __func__, (int)imatrix_data->size());
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        for (const int32_t * p = params->prune_layers; * p != -1; p++) {
            prune_list.push_back(* p);
        }
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.metadata);
    gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_GENERAL_QUANTIZATION_VERSION).c_str(), GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_GENERAL_FILE_TYPE).c_str(), ftype);
    if (!params->only_copy && default_type == GGML_TYPE_NVFP4) {
        gguf_set_val_str(ctx_out.get(), "general.quantization.nvfp4_scale_mode", llama_nvfp4_scale_mode_name(params->nvfp4_scale_mode));
        if (params->nvfp4_scale_mode == LLAMA_NVFP4_SCALE_MODE_ADAPTIVE) {
            gguf_set_val_str(ctx_out.get(), "general.quantization.nvfp4_adaptive_candidates", ggml_nvfp4_adaptive_candidates());
            gguf_set_val_str(ctx_out.get(), "general.quantization.nvfp4_search_algo", ggml_nvfp4_search_algo_name(ggml_nvfp4_get_search_algo()));
            gguf_set_val_u32(ctx_out.get(), "general.quantization.nvfp4_cjso_radius", ggml_nvfp4_cjso_radius());
            if (imatrix_data) {
                gguf_set_val_str(ctx_out.get(), "general.quantization.nvfp4_adaptive_criterion", ggml_nvfp4_imatrix_select_criterion());
                if (ggml_nvfp4_get_imatrix_select_mode() == GGML_NVFP4_IMATRIX_SELECT_WEIGHTED_GUARD) {
                    gguf_set_val_f32(ctx_out.get(), "general.quantization.nvfp4_imatrix_guard", ggml_nvfp4_imatrix_ordinary_mse_guard());
                } else {
                    gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_imatrix_guard");
                }
            } else {
                gguf_set_val_str(ctx_out.get(), "general.quantization.nvfp4_adaptive_criterion", "mse");
                gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_imatrix_guard");
            }
        } else {
            gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_adaptive_candidates");
            gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_search_algo");
            gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_cjso_radius");
            gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_adaptive_criterion");
            gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_imatrix_guard");
        }
    } else {
        gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_scale_mode");
        gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_adaptive_candidates");
        gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_search_algo");
        gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_cjso_radius");
        gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_adaptive_criterion");
        gguf_remove_key(ctx_out.get(), "general.quantization.nvfp4_imatrix_guard");
    }
    if (!params->only_copy) {
        gguf_set_val_bool(ctx_out.get(), "general.quant_wht.enabled", quant_wht_enabled);
        if (quant_wht_enabled) {
            gguf_set_val_u32(ctx_out.get(), "general.quant_wht.dim", LLAMA_QUANT_WHT_DIM);
            gguf_set_val_str(ctx_out.get(), "general.quant_wht.scheme", LLAMA_QUANT_WHT_SCHEME);
            gguf_set_val_u32(ctx_out.get(), "general.quant_wht.version", LLAMA_QUANT_WHT_VERSION);
            gguf_remove_key(ctx_out.get(), "general.quant_wht.mode");
            if (quant_wht_skip_types.empty()) {
                gguf_remove_key(ctx_out.get(), "general.quant_wht.skip_types");
            } else {
                gguf_set_val_str(ctx_out.get(), "general.quant_wht.skip_types", quant_wht_skip_types.c_str());
            }
            LLAMA_LOG_WARN("%s: WARNING: writing experimental WHT-rotated Q_K/Q8_0/IQ GGUF (skip_types=%s); incompatible with builds that do not support general.quant_wht\n",
                    __func__, quant_wht_skip_types.empty() ? "<none>" : quant_wht_skip_types.c_str());
        } else {
            gguf_remove_key(ctx_out.get(), "general.quant_wht.dim");
            gguf_remove_key(ctx_out.get(), "general.quant_wht.scheme");
            gguf_remove_key(ctx_out.get(), "general.quant_wht.version");
            gguf_remove_key(ctx_out.get(), "general.quant_wht.mode");
            gguf_remove_key(ctx_out.get(), "general.quant_wht.skip_types");
        }
    }

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        for (const llama_model_kv_override * o = params->kv_overrides; o->key[0] != 0; ++o) {
            if (o->tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o->key, o->val_f64);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                // Setting type to UINT32. See https://github.com/ggml-org/llama.cpp/pull/14182 for context
                gguf_set_val_u32(ctx_out.get(), o->key, (uint32_t)std::abs(o->val_i64));
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o->key, o->val_bool);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o->key, o->val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o->key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }

        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n", __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }
        tensors.push_back(&it.second);
    }
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    // compute tensor metadata once and cache it
    std::vector<tensor_metadata> metadata(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        metadata[i].name = ggml_get_name(tensors[i]->tensor);
    }

    // initialize quantization state counters and metadata categories
    init_quantize_state_counters(qs, metadata);

    int idx = 0;
    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // flag for --dry-run
    bool will_require_imatrix = false;
    int quant_wht_n_tensors = 0;
    int quant_wht_n_skipped_tensors = 0;

    //
    // preliminary iteration over all weights
    //

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * it = tensors[i];
        const struct ggml_tensor * tensor = it->tensor;

        uint16_t i_split = params->keep_split ? it->idx : 0;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);

        metadata[i].allows_quantization = tensor_allows_quantization(params, model->arch, tensor);

        if (metadata[i].allows_quantization) {
            metadata[i].target_type = llama_tensor_get_type(qs, params, tensor, default_type, metadata[i]);
        } else {
            metadata[i].target_type = tensor->type;
        }
        metadata[i].quant_wht_rotate = false;
        if (quant_wht_enabled &&
                metadata[i].allows_quantization &&
                llama_quant_wht_type_supported(metadata[i].target_type) &&
                llama_quant_wht_name_supported(metadata[i].name, metadata[i].category)) {
            if (!llama_quant_wht_type_enabled(metadata[i].target_type, quant_wht_skip_types)) {
                ++quant_wht_n_skipped_tensors;
            } else if (tensor->ne[0] % LLAMA_QUANT_WHT_DIM != 0) {
                throw std::runtime_error(format("cannot use --quant-wht for tensor %s: reduction dimension %" PRId64 " is not divisible by %d",
                            metadata[i].name.c_str(), tensor->ne[0], LLAMA_QUANT_WHT_DIM));
            } else {
                metadata[i].quant_wht_rotate = true;
                ++quant_wht_n_tensors;
            }
        }

        metadata[i].requires_imatrix = tensor_requires_imatrix(tensor->name, metadata[i].target_type, ftype);

        if (params->imatrix) {
            metadata[i].remapped_imatrix_name = remap_imatrix(tensor->name, mapped);
        } else if (metadata[i].allows_quantization && metadata[i].requires_imatrix) {
            if (params->dry_run) {
                will_require_imatrix = true;
            } else {
                LLAMA_LOG_ERROR("\n============================================================================\n"
                                " ERROR: this quantization requires an importance matrix!\n"
                                "        - offending tensor: %s\n"
                                "        - target type: %s\n"
                                "============================================================================\n\n",
                                metadata[i].name.c_str(), ggml_type_name(metadata[i].target_type));
                throw std::runtime_error("this quantization requires an imatrix!");
            }
        }
    }

    if (ftype == LLAMA_FTYPE_MOSTLY_NVFP4_M) {
        nvfp4_m_policy_summary summary;
        for (size_t i = 0; i < tensors.size(); ++i) {
            nvfp4_m_policy_summary_add(summary, tensors[i]->tensor, metadata[i].target_type);
        }
        nvfp4_m_policy_summary_print(summary);
    }

    if (quant_wht_enabled) {
        LLAMA_LOG_WARN("%s: WARNING: --quant-wht enabled, rotating %d eligible Q_K/Q8_0/IQ tensor(s) with %dD %s",
                __func__, quant_wht_n_tensors, LLAMA_QUANT_WHT_DIM, LLAMA_QUANT_WHT_SCHEME);
        if (!quant_wht_skip_types.empty()) {
            LLAMA_LOG_WARN(", skipped %d tensor(s) via skip_types=%s", quant_wht_n_skipped_tensors, quant_wht_skip_types.c_str());
        }
        LLAMA_LOG_WARN("\n");
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)tensors.size());
        }
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;
    std::vector<no_init<float>> quant_wht_buf;

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path.data());
        }

        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    // no output file for --dry-run
    if (!params->dry_run) {
        new_ofstream(0);
    }

    //
    // main loop: iterate over all weights
    //

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & weight = *tensors[i];
        const auto & tm = metadata[i];
        ggml_tensor * tensor = weight.tensor;

        if (!params->dry_run && (weight.idx != cur_split && params->keep_split)) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const size_t tensor_size = ggml_nbytes(tensor);

        if (!params->dry_run) {
            if (!ml.use_mmap) {
                if (read_data.size() < tensor_size) {
                    read_data.resize(tensor_size);
                }
                tensor->data = read_data.data();
            }
            ml.load_data_for(tensor);
        }

        LLAMA_LOG_INFO("[%4d/%4d] %-36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        const ggml_type cur_type = tensor->type;
        const ggml_type new_type = tm.target_type;

        // If we've decided to quantize to the same type the tensor is already
        // in then there's nothing to do.
        bool quantize = cur_type != new_type || tm.quant_wht_rotate;

        void * new_data;
        size_t new_size;

        if (params->dry_run) {
            // the --dry-run option calculates the final quantization size without quantizing
            if (quantize) {
                new_size = ggml_nrows(tensor) * ggml_row_size(new_type, tensor->ne[0]);
                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB (%s)\n",
                               tensor_size/1024.0/1024.0,
                               new_size/1024.0/1024.0,
                               ggml_type_name(new_type));
                if (!will_require_imatrix && tm.requires_imatrix) {
                    will_require_imatrix = true;
                }
            } else {
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;
            continue;
        } else {
            // no --dry-run, perform quantization
            if (!quantize) {
                new_data = tensor->data;
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", tensor_size/1024.0/1024.0);
            } else {
                const int64_t nelements = ggml_nelements(tensor);

                const float * imatrix = nullptr;
                if (imatrix_data) {
                    auto it = imatrix_data->find(tm.remapped_imatrix_name);
                    if (it == imatrix_data->end()) {
                        LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                    } else {
                        if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                            imatrix = it->second.data();
                        } else {
                            LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                    int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                            // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                            // this is a significant error and it may be good idea to abort the process if this happens,
                            // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                            // tok_embd should be ignored in this case, since it always causes this warning
                            if (!tensor_name_match_token_embd(tensor->name)) {
                                throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                        int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                            }
                        }
                    }
                }
                if (!imatrix && tm.requires_imatrix) {
                    LLAMA_LOG_ERROR("\n\n============================================================\n");
                    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                    LLAMA_LOG_ERROR("============================================================\n\n");
                    throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
                }

                float * f32_data;

                if (tensor->type == GGML_TYPE_F32) {
                    f32_data = (float *) tensor->data;
                } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                    throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
                } else {
                    llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();
                }

                LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
                fflush(stdout);

                if (work.size() < (size_t)nelements * 4) {
                    work.resize(nelements * 4); // upper bound on size
                }
                new_data = work.data();

                const int64_t n_per_row = tensor->ne[0];
                const int64_t nrows = tensor->ne[1];

                static const int64_t min_chunk_size = 32 * 512;
                const int64_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row));

                const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
                const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
                const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;

                // quantize each expert separately since they have different importance matrices
                new_size = 0;
                for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
                    const float * f32_data_03 = f32_data + i03 * nelements_matrix;
                    void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
                    const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;
                    const float * quant_data_03 = f32_data_03;

                    if (tm.quant_wht_rotate) {
                        if (quant_wht_buf.size() < (size_t) nelements_matrix) {
                            quant_wht_buf.resize(nelements_matrix);
                        }
                        llama_quant_wht_rotate_rows_256(f32_data_03, (float *) quant_wht_buf.data(), nrows, n_per_row, workers, nthread_use);
                        quant_data_03 = (const float *) quant_wht_buf.data();
                    }

                    llama_nvfp4_rsf_override_scope nvfp4_rsf_scope;
                    float nvfp4_rsf_multiplier = 1.0f;
                    if (new_type == GGML_TYPE_NVFP4 &&
                            params->nvfp4_scale_mode == LLAMA_NVFP4_SCALE_MODE_ADAPTIVE &&
                            ggml_nvfp4_rsf_lite_enabled()) {
                        nvfp4_rsf_multiplier = ggml_nvfp4_choose_rsf_multiplier(quant_data_03, nrows, n_per_row, imatrix_03);
                        nvfp4_rsf_scope.set(nvfp4_rsf_multiplier);
                    }

                    const bool nvfp4_debug_tensor =
                            new_type == GGML_TYPE_NVFP4 &&
                            llama_nvfp4_debug_tensor_match(metadata[i].name.c_str());
                    ggml_nvfp4_scale_stats nvfp4_debug_stats_before = {};
                    if (nvfp4_debug_tensor) {
                        ggml_nvfp4_get_scale_stats(&nvfp4_debug_stats_before);
                    }

                    if (new_type == GGML_TYPE_NVFP4) {
                        ggml_nvfp4_set_debug_context(metadata[i].name.c_str(), quant_data_03, n_per_row);
                    }

                    if (tm.quant_wht_rotate && imatrix_03 && new_type != GGML_TYPE_Q8_0) {
                        static std::once_flag once;
                        std::call_once(once, [] {
                            LLAMA_LOG_WARN("%s: WARNING: --quant-wht with --imatrix uses experimental exact original-domain scoring over rotated quantization candidates\n", __func__);
                        });
                        new_size += llama_tensor_quantize_wht_imatrix_impl(new_type, quant_data_03, new_data_03, nrows, n_per_row, imatrix_03, workers, nthread_use);
                    } else {
                        new_size += llama_tensor_quantize_impl(new_type, quant_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, workers, nthread_use);
                    }

                    if (new_type == GGML_TYPE_NVFP4) {
                        ggml_nvfp4_set_debug_context(nullptr, nullptr, 0);
                    }
                    if (nvfp4_debug_tensor) {
                        ggml_nvfp4_scale_stats nvfp4_debug_stats_after = {};
                        ggml_nvfp4_get_scale_stats(&nvfp4_debug_stats_after);
                        const ggml_nvfp4_scale_stats nvfp4_debug_stats =
                                llama_nvfp4_scale_stats_subtract(nvfp4_debug_stats_after, nvfp4_debug_stats_before);
                        llama_nvfp4_log_tensor_debug(
                                __func__, metadata[i].name.c_str(), i03, new_data_03, nrows, n_per_row,
                                nvfp4_rsf_multiplier, nvfp4_debug_stats);
                    }
                }
                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", tensor_size/1024.0/1024.0, new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;

            // update the gguf meta data as we go
            gguf_set_tensor_type(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), metadata[i].name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_data);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
        } // no --dry-run
    } // main loop

    if (!params->dry_run) {
        close_ofstream();
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (params->nvfp4_scale_mode == LLAMA_NVFP4_SCALE_MODE_ADAPTIVE) {
        ggml_nvfp4_scale_stats stats;
        if (ggml_nvfp4_get_scale_stats(&stats)) {
            const double p4 = 100.0 * (double) stats.n_m4 / (double) stats.n_subblocks;
            const double p5 = 100.0 * (double) stats.n_m5 / (double) stats.n_subblocks;
            const double p6 = 100.0 * (double) stats.n_m6 / (double) stats.n_subblocks;
            const double p_other = 100.0 * (double) stats.n_slot_other / (double) stats.n_subblocks;
            const double ratio = stats.mse_m6 > 0.0 ? stats.mse_selected / stats.mse_m6 : 1.0;
            const bool weighted = stats.n_weighted_subblocks > 0;
            const double weighted_ratio = stats.weighted_mse_m6 > 0.0 ? stats.weighted_mse_selected / stats.weighted_mse_m6 : 1.0;
            const double ordinary_best_pct = weighted ? 100.0 * (double) stats.n_ordinary_best_selected / (double) stats.n_weighted_subblocks : 0.0;
            const double weighted_best_pct = weighted ? 100.0 * (double) stats.n_weighted_best_selected / (double) stats.n_weighted_subblocks : 0.0;
            const double knee_pct = weighted ? 100.0 * (double) stats.n_knee_selected / (double) stats.n_weighted_subblocks : 0.0;
            const double guard_accepted_pct = weighted ? 100.0 * (double) stats.n_weighted_guard_accepted / (double) stats.n_weighted_subblocks : 0.0;
            const double guard_fallback_pct = weighted ? 100.0 * (double) stats.n_weighted_guard_fallback / (double) stats.n_weighted_subblocks : 0.0;
            const bool rich = ggml_nvfp4_rich_scale_search_enabled() != 0;
            const bool rsf = ggml_nvfp4_rsf_lite_enabled() != 0;
            const auto search_algo = ggml_nvfp4_get_search_algo();
            const bool has_cjso = search_algo == GGML_NVFP4_SEARCH_CJSO || search_algo == GGML_NVFP4_SEARCH_RICH_CJSO;
            const ggml_nvfp4_imatrix_select_mode select_mode = weighted ?
                    ggml_nvfp4_get_imatrix_select_mode() : GGML_NVFP4_IMATRIX_SELECT_TWO_OBJECTIVE;
            LLAMA_LOG_INFO("%s: NVFP4 adaptive:\n", __func__);
            LLAMA_LOG_INFO("%s:   candidates: %s\n", __func__, ggml_nvfp4_adaptive_candidates());
            LLAMA_LOG_INFO("%s:   search algo: %s\n", __func__, ggml_nvfp4_search_algo_name(search_algo));
            LLAMA_LOG_INFO("%s:   CJSO radius: %d\n", __func__, ggml_nvfp4_cjso_radius());
            LLAMA_LOG_INFO("%s:   rich scale search: %s\n", __func__, rich ? "enabled" : "disabled");
            const double avg_candidates = stats.n_subblocks > 0 ?
                    (double) stats.candidate_count_total / (double) stats.n_subblocks : 0.0;
            LLAMA_LOG_INFO("%s:   avg candidates after dedup: %.3f\n", __func__, avg_candidates);
            if (rich) {
                const double p_top1 = stats.n_subblocks > 0 ? 100.0 * (double) stats.n_anchor_top1 / (double) stats.n_subblocks : 0.0;
                const double p_slot1 = stats.n_subblocks > 0 ? 100.0 * (double) stats.n_slot1 / (double) stats.n_subblocks : 0.0;
                const double p_slot15 = stats.n_subblocks > 0 ? 100.0 * (double) stats.n_slot15 / (double) stats.n_subblocks : 0.0;
                const double p_slot2 = stats.n_subblocks > 0 ? 100.0 * (double) stats.n_slot2 / (double) stats.n_subblocks : 0.0;
                const double p_slot3 = stats.n_subblocks > 0 ? 100.0 * (double) stats.n_slot3 / (double) stats.n_subblocks : 0.0;
                LLAMA_LOG_INFO("%s:   rich anchor: top1 maxabs\n", __func__);
                LLAMA_LOG_INFO("%s:   rich slots: 6,5,4,3,2,1.5,1\n", __func__);
                LLAMA_LOG_INFO("%s:   rich slot code_radius: %d\n", __func__, ggml_nvfp4_rich_scale_code_radius());
                LLAMA_LOG_INFO("%s:   rich selected anchor top1: %.3f%%\n", __func__, p_top1);
                LLAMA_LOG_INFO("%s:   slot distribution: 6=%.3f%%, 5=%.3f%%, 4=%.3f%%, 3=%.3f%%, 2=%.3f%%, 1.5=%.3f%%, 1=%.3f%%, other=%.3f%%\n",
                        __func__, p6, p5, p4, p_slot3, p_slot2, p_slot15, p_slot1, p_other);
            }
            if (has_cjso) {
                const double source_rich_pct = 100.0 * (double) stats.n_source_rich / (double) stats.n_subblocks;
                const double source_cjso_ord_pct = 100.0 * (double) stats.n_source_cjso_ordinary / (double) stats.n_subblocks;
                const double source_cjso_wgt_pct = 100.0 * (double) stats.n_source_cjso_weighted / (double) stats.n_subblocks;
                const double source_both_pct = 100.0 * (double) stats.n_source_both / (double) stats.n_subblocks;
                const double source_fallback_pct = 100.0 * (double) stats.n_source_fallback / (double) stats.n_subblocks;
                const uint64_t total_cjso_selected =
                        stats.n_source_cjso_ordinary + stats.n_source_cjso_weighted + stats.n_source_both;
                const double cjso_denom = (double) std::max<uint64_t>(total_cjso_selected, 1);
                LLAMA_LOG_INFO("%s:   CJSO ordinary anchor candidate count: %llu\n",
                        __func__, (unsigned long long) stats.n_cjso_ordinary_candidates);
                LLAMA_LOG_INFO("%s:   CJSO weighted anchor candidate count: %llu\n",
                        __func__, (unsigned long long) stats.n_cjso_weighted_candidates);
                LLAMA_LOG_INFO("%s:   source distribution: rich=%.3f%%, cjso_ord=%.3f%%, cjso_weighted=%.3f%%, both=%.3f%%, fallback=%.3f%%\n",
                        __func__, source_rich_pct, source_cjso_ord_pct, source_cjso_wgt_pct, source_both_pct, source_fallback_pct);
                LLAMA_LOG_INFO("%s:   CJSO selected delta distribution: -4=%.3f%%, -3=%.3f%%, -2=%.3f%%, -1=%.3f%%, 0=%.3f%%, +1=%.3f%%, +2=%.3f%%, +3=%.3f%%, +4=%.3f%%, other=%.3f%%\n",
                        __func__,
                        100.0 * (double) stats.n_cjso_delta_m4 / cjso_denom,
                        100.0 * (double) stats.n_cjso_delta_m3 / cjso_denom,
                        100.0 * (double) stats.n_cjso_delta_m2 / cjso_denom,
                        100.0 * (double) stats.n_cjso_delta_m1 / cjso_denom,
                        100.0 * (double) stats.n_cjso_delta_0  / cjso_denom,
                        100.0 * (double) stats.n_cjso_delta_p1 / cjso_denom,
                        100.0 * (double) stats.n_cjso_delta_p2 / cjso_denom,
                        100.0 * (double) stats.n_cjso_delta_p3 / cjso_denom,
                        100.0 * (double) stats.n_cjso_delta_p4 / cjso_denom,
                        100.0 * (double) stats.n_cjso_delta_other / cjso_denom);
            }
            LLAMA_LOG_INFO("%s:   RSF-lite: %s\n", __func__, rsf ? "enabled" : "disabled");
            if (rsf) {
                const double rsf_ratio = stats.rsf_sample_error_baseline > 0.0 ?
                        stats.rsf_sample_error_selected / stats.rsf_sample_error_baseline : 1.0;
                LLAMA_LOG_INFO("%s:   RSF multipliers: 0.875,0.9375,1.0,1.0625,1.125\n", __func__);
                LLAMA_LOG_INFO("%s:   RSF tensors: %llu\n", __func__, (unsigned long long) stats.n_rsf_tensors);
                LLAMA_LOG_INFO("%s:   RSF multiplier counts: 0.875=%llu, 0.9375=%llu, 1.0=%llu, 1.0625=%llu, 1.125=%llu\n",
                        __func__,
                        (unsigned long long) stats.n_rsf_0875,
                        (unsigned long long) stats.n_rsf_09375,
                        (unsigned long long) stats.n_rsf_1000,
                        (unsigned long long) stats.n_rsf_10625,
                        (unsigned long long) stats.n_rsf_1125);
                LLAMA_LOG_INFO("%s:   RSF sampled error ratio vs r=1.0: %.9g\n", __func__, rsf_ratio);
            }
            if (weighted) {
                switch (select_mode) {
                    case GGML_NVFP4_IMATRIX_SELECT_ORDINARY:
                        LLAMA_LOG_INFO("%s:   criterion: imatrix ordinary MSE\n", __func__);
                        break;
                    case GGML_NVFP4_IMATRIX_SELECT_WEIGHTED_GUARD:
                        LLAMA_LOG_INFO("%s:   criterion: imatrix weighted guard, guard=%.6g\n", __func__,
                                (double) ggml_nvfp4_imatrix_ordinary_mse_guard());
                        break;
                    case GGML_NVFP4_IMATRIX_SELECT_WEIGHTED_RAW:
                        LLAMA_LOG_INFO("%s:   criterion: imatrix weighted raw\n", __func__);
                        LLAMA_LOG_WARN("%s:   warning: NVFP4 imatrix weighted_raw may produce NaN\n", __func__);
                        break;
                    case GGML_NVFP4_IMATRIX_SELECT_TWO_OBJECTIVE:
                    default:
                        LLAMA_LOG_INFO("%s:   criterion: imatrix two-objective log-square\n", __func__);
                        break;
                }
            } else {
                LLAMA_LOG_INFO("%s:   criterion: ordinary MSE\n", __func__);
            }
            LLAMA_LOG_INFO("%s:   subblocks: %llu\n", __func__, (unsigned long long) stats.n_subblocks);
            LLAMA_LOG_INFO("%s:   M=4 selected: %llu (%.3f%%)\n", __func__, (unsigned long long) stats.n_m4, p4);
            LLAMA_LOG_INFO("%s:   M=5 selected: %llu (%.3f%%)\n", __func__, (unsigned long long) stats.n_m5, p5);
            LLAMA_LOG_INFO("%s:   M=6 selected: %llu (%.3f%%)\n", __func__, (unsigned long long) stats.n_m6, p6);
            LLAMA_LOG_INFO("%s:   non-slot selected: %llu (%.3f%%)\n", __func__, (unsigned long long) stats.n_slot_other, p_other);
            if (weighted) {
                if (select_mode == GGML_NVFP4_IMATRIX_SELECT_WEIGHTED_GUARD) {
                    LLAMA_LOG_INFO("%s:   weighted accepted: %llu (%.3f%%)\n", __func__,
                            (unsigned long long) stats.n_weighted_guard_accepted, guard_accepted_pct);
                    LLAMA_LOG_INFO("%s:   fallback to ordinary: %llu (%.3f%%)\n", __func__,
                            (unsigned long long) stats.n_weighted_guard_fallback, guard_fallback_pct);
                } else {
                    LLAMA_LOG_INFO("%s:   ordinary-best selected: %llu (%.3f%%)\n", __func__,
                            (unsigned long long) stats.n_ordinary_best_selected, ordinary_best_pct);
                    LLAMA_LOG_INFO("%s:   weighted-best selected: %llu (%.3f%%)\n", __func__,
                            (unsigned long long) stats.n_weighted_best_selected, weighted_best_pct);
                    LLAMA_LOG_INFO("%s:   knee selected: %llu (%.3f%%)\n", __func__,
                            (unsigned long long) stats.n_knee_selected, knee_pct);
                }
                LLAMA_LOG_INFO("%s:   selected vs M=6 weighted reconstruction MSE ratio: %.9g\n", __func__, weighted_ratio);
                LLAMA_LOG_INFO("%s:   selected vs M=6 ordinary reconstruction MSE ratio: %.9g\n", __func__, ratio);
            } else {
                LLAMA_LOG_INFO("%s:   selected vs M=6 reconstruction MSE ratio: %.9g\n", __func__, ratio);
            }
        }
    }

    if (!params->imatrix && params->dry_run && will_require_imatrix) {
        LLAMA_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n",
                       __func__
        );
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, ml.n_tensors);
    }
}

//
// interface implementation
//

llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q8_0,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.dry_run                     =*/ false,
        /*.quant_wht                   =*/ false,
        /*.quant_wht_full              =*/ false,
        /*.quant_wht_skip_types        =*/ nullptr,
        /*.quant_wht_dim               =*/ 256,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_type                 =*/ nullptr,
        /*.prune_layers                =*/ nullptr,
        /*.nvfp4_scale_mode            =*/ LLAMA_NVFP4_SCALE_MODE_M6,
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}

//
// Helper functions for external tools exposed in llama-ext.h
//

quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params) {
    return new quantize_state_impl(*model, params);
}

void llama_quant_free(quantize_state_impl * qs) {
    delete qs;
}

llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc) {
    struct llama_model_params mparams = llama_model_default_params();
    auto arch = llm_arch_from_string(desc->architecture);
    auto * model = llama_model_create(arch, mparams);
    model->arch = arch;

    // infer llm_type: only LLM_TYPE_70B matters for quantization logic
    if (model->arch == LLM_ARCH_LLAMA && desc->n_layer == 80 && desc->n_head != desc->n_head_kv) {
        model->type = LLM_TYPE_70B;
    }

    model->hparams.n_embd             = desc->n_embd;
    model->hparams.n_embd_head_k_full = desc->n_embd_head_k;
    model->hparams.n_embd_head_v_full = desc->n_embd_head_v;
    model->hparams.n_layer_all        = desc->n_layer;
    model->hparams.n_expert           = desc->n_expert;

    for (uint32_t i = 0; i < desc->n_layer; i++) {
        model->hparams.n_head_arr[i]    = desc->n_head;
        model->hparams.n_head_kv_arr[i] = desc->n_head_kv;
        model->hparams.n_ff_arr[i]      = desc->n_ff;
    }

    return model;
}

bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor) {
    return tensor_allows_quantization(qs->params, qs->model.arch, tensor);
}

void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors) {
    // reset per-computation state
    qs->n_attention_wv      = 0;
    qs->n_ffn_down          = 0;
    qs->n_ffn_gate          = 0;
    qs->n_ffn_up            = 0;
    qs->i_attention_wv      = 0;
    qs->i_ffn_down          = 0;
    qs->i_ffn_gate          = 0;
    qs->i_ffn_up            = 0;
    qs->n_fallback          = 0;
    qs->has_imatrix         = false;
    qs->has_tied_embeddings = true;

    // build metadata from tensor names
    std::vector<tensor_metadata> metadata(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) {
        metadata[i].name = ggml_get_name(tensors[i]);
    }

    // initialize counters and categories
    init_quantize_state_counters(*qs, metadata);

    // use a local copy of params with the requested ftype
    llama_model_quantize_params local_params = *qs->params;
    local_params.ftype = ftype;

    ggml_type default_type = llama_ftype_get_default_type(ftype);

    // compute types
    for (size_t i = 0; i < n_tensors; i++) {
        result_types[i] = llama_tensor_get_type(*qs, &local_params, tensors[i], default_type, metadata[i]);
    }
}
