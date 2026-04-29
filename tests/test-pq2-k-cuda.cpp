#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int64_t PQ2_K_TEST_M = 128;
constexpr int64_t PQ2_K_TEST_K = 1024;
constexpr int64_t PQ2_K_MMVQ_N = 1;
constexpr int64_t PQ2_K_MMQ_N  = 128;

constexpr double PQ2_K_MAX_ABS_TOL = 5e-2;
constexpr double PQ2_K_REL_RMS_TOL = 5e-3;

struct diff_stats {
    double max_abs = 0.0;
    double rms = 0.0;
    double rel_rms = 0.0;
};

struct input_payload {
    std::vector<uint8_t> raw;
    std::vector<float> ref;
};

struct test_case {
    const char * label;
    ggml_type type_b;
    int64_t n;
    uint32_t seed;
};

static std::vector<float> make_uniform_f32(size_t count, uint32_t seed, float min_value = -1.0f, float max_value = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min_value, max_value);

    std::vector<float> data(count);
    for (float & value : data) {
        value = dist(rng);
    }

    return data;
}

static std::vector<uint8_t> quantize_pq2_k(const std::vector<float> & weights_f32, int64_t k, int64_t m) {
    std::vector<uint8_t> weights_q(ggml_row_size(GGML_TYPE_PQ2_K, k) * m);
    std::vector<float> imatrix(k, 1.0f);
    const float * imatrix_ptr = ggml_quantize_requires_imatrix(GGML_TYPE_PQ2_K) ? imatrix.data() : nullptr;

    const size_t written = ggml_quantize_chunk(GGML_TYPE_PQ2_K, weights_f32.data(), weights_q.data(), 0, m, k, imatrix_ptr);
    GGML_ASSERT(written == weights_q.size());

    return weights_q;
}

static std::vector<float> dequantize_pq2_k(const std::vector<uint8_t> & weights_q, int64_t nels) {
    const ggml_to_float_t to_float = ggml_get_type_traits(GGML_TYPE_PQ2_K)->to_float;
    GGML_ASSERT(to_float != nullptr);

    std::vector<float> weights_f32(nels);
    to_float(weights_q.data(), weights_f32.data(), nels);
    return weights_f32;
}

static input_payload make_input_payload(ggml_type type, const std::vector<float> & input_f32) {
    input_payload payload;
    payload.ref = input_f32;

    if (type == GGML_TYPE_F32) {
        payload.raw.resize(input_f32.size() * sizeof(float));
        std::memcpy(payload.raw.data(), input_f32.data(), payload.raw.size());
        return payload;
    }

    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> input_f16(input_f32.size());
        ggml_fp32_to_fp16_row(input_f32.data(), input_f16.data(), input_f32.size());

        payload.ref.resize(input_f16.size());
        ggml_fp16_to_fp32_row(input_f16.data(), payload.ref.data(), input_f16.size());

        payload.raw.resize(input_f16.size() * sizeof(ggml_fp16_t));
        std::memcpy(payload.raw.data(), input_f16.data(), payload.raw.size());
        return payload;
    }

    GGML_ABORT("unsupported activation type");
}

static std::vector<float> reference_mul_mat(
        const std::vector<float> & weights_f32,
        const std::vector<float> & input_f32,
        int64_t m,
        int64_t n,
        int64_t k) {
    std::vector<float> out(m * n, 0.0f);

    for (int64_t col = 0; col < n; ++col) {
        const float * input_col = input_f32.data() + col * k;
        float * out_col = out.data() + col * m;

        for (int64_t row = 0; row < m; ++row) {
            const float * weight_row = weights_f32.data() + row * k;
            double sum = 0.0;

            for (int64_t idx = 0; idx < k; ++idx) {
                sum += (double) weight_row[idx] * input_col[idx];
            }

            out_col[row] = (float) sum;
        }
    }

    return out;
}

static diff_stats compute_diff_stats(const std::vector<float> & ref, const std::vector<float> & got) {
    GGML_ASSERT(ref.size() == got.size());

    diff_stats stats;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const double err = (double) got[i] - ref[i];
        const double abs_err = std::fabs(err);

        stats.max_abs = std::max(stats.max_abs, abs_err);
        sum_sq_err += err * err;
        sum_sq_ref += (double) ref[i] * ref[i];
    }

    stats.rms = std::sqrt(sum_sq_err / std::max<size_t>(1, ref.size()));
    stats.rel_rms = std::sqrt(sum_sq_err / std::max(sum_sq_ref, 1e-30));
    return stats;
}

static ggml_backend_t init_cuda_backend() {
    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = ggml_backend_reg_name(reg);

        if (reg_name == nullptr || std::strcmp(reg_name, "CUDA") != 0) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend != nullptr);

        size_t free_mem = 0;
        size_t total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        std::printf("Using CUDA backend %s (%s), memory %zu MiB free / %zu MiB total\n",
                ggml_backend_dev_name(dev),
                ggml_backend_dev_description(dev),
                free_mem / (1024 * 1024),
                total_mem / (1024 * 1024));

        return backend;
    }

    return nullptr;
}

static bool run_case(
        ggml_backend_t backend,
        const test_case & tc,
        const std::vector<uint8_t> & weights_q,
        const std::vector<float> & weights_ref) {
    const std::vector<float> input_f32 = make_uniform_f32((size_t) (PQ2_K_TEST_K * tc.n), tc.seed);
    const input_payload input = make_input_payload(tc.type_b, input_f32);
    const std::vector<float> ref = reference_mul_mat(weights_ref, input.ref, PQ2_K_TEST_M, tc.n, PQ2_K_TEST_K);

    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false);
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_K, PQ2_K_TEST_K, PQ2_K_TEST_M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, tc.type_b, PQ2_K_TEST_K, tc.n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    GGML_ASSERT(out != nullptr);
    GGML_ASSERT(out->type == GGML_TYPE_F32);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    GGML_ASSERT((size_t) ggml_nbytes(a) == weights_q.size());
    ggml_backend_tensor_set(a, weights_q.data(), 0, weights_q.size());
    ggml_backend_tensor_set(b, input.raw.data(), 0, input.raw.size());

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: ggml_backend_graph_compute failed for %s: %s\n",
                __func__, tc.label, ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> got(ref.size());
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

    const diff_stats stats = compute_diff_stats(ref, got);
    std::printf("%s type_b=%s m=%lld n=%lld k=%lld max_abs=%0.8f rms=%0.8f rel_rms=%0.8f\n",
            tc.label,
            ggml_type_name(tc.type_b),
            (long long) PQ2_K_TEST_M,
            (long long) tc.n,
            (long long) PQ2_K_TEST_K,
            stats.max_abs,
            stats.rms,
            stats.rel_rms);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return stats.max_abs <= PQ2_K_MAX_ABS_TOL && stats.rel_rms <= PQ2_K_REL_RMS_TOL;
}

} // namespace

int main() {
    ggml_log_set(nullptr, nullptr);

    ggml_backend_t backend = init_cuda_backend();
    if (backend == nullptr) {
        std::printf("No CUDA backend found, skipping PQ2_K CUDA compare.\n");
        return 0;
    }

    const std::vector<float> weights_src = make_uniform_f32((size_t) (PQ2_K_TEST_M * PQ2_K_TEST_K), 0xC0FFEEu);
    const std::vector<uint8_t> weights_q = quantize_pq2_k(weights_src, PQ2_K_TEST_K, PQ2_K_TEST_M);
    const std::vector<float> weights_ref = dequantize_pq2_k(weights_q, PQ2_K_TEST_M * PQ2_K_TEST_K);

    const test_case cases[] = {
        { "MMVQ", GGML_TYPE_F32, PQ2_K_MMVQ_N, 0x1001u },
        { "MMVQ", GGML_TYPE_F16, PQ2_K_MMVQ_N, 0x1002u },
        { "MMQ",  GGML_TYPE_F32, PQ2_K_MMQ_N,  0x2001u },
        { "MMQ",  GGML_TYPE_F16, PQ2_K_MMQ_N,  0x2002u },
    };

    bool ok = true;
    for (const test_case & tc : cases) {
        ok = run_case(backend, tc, weights_q, weights_ref) && ok;
    }

    ggml_backend_free(backend);

    if (!ok) {
        std::fprintf(stderr, "PQ2_K CUDA compare exceeded tolerances: max_abs<=%g rel_rms<=%g\n",
                PQ2_K_MAX_ABS_TOL, PQ2_K_REL_RMS_TOL);
        return 1;
    }

    return 0;
}