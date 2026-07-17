#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

enum ggml_nvfp4_scale_mode {
    GGML_NVFP4_SCALE_MODE_M6       = 0,
    GGML_NVFP4_SCALE_MODE_M4       = 1,
    GGML_NVFP4_SCALE_MODE_ADAPTIVE = 2,
};

enum ggml_nvfp4_imatrix_select_mode {
    GGML_NVFP4_IMATRIX_SELECT_TWO_OBJECTIVE = 0,
    GGML_NVFP4_IMATRIX_SELECT_ORDINARY      = 1,
    GGML_NVFP4_IMATRIX_SELECT_WEIGHTED_GUARD = 2,
    GGML_NVFP4_IMATRIX_SELECT_WEIGHTED_RAW  = 3,
};

enum ggml_nvfp4_search_algo {
    GGML_NVFP4_SEARCH_RICH      = 0,
    GGML_NVFP4_SEARCH_CJSO      = 1,
    GGML_NVFP4_SEARCH_RICH_CJSO = 2,
};

typedef struct ggml_nvfp4_scale_stats {
    uint64_t n_subblocks;
    uint64_t n_weighted_subblocks;
    uint64_t n_ordinary_best_selected;
    uint64_t n_weighted_best_selected;
    uint64_t n_knee_selected;
    uint64_t n_weighted_guard_accepted;
    uint64_t n_weighted_guard_fallback;
    uint64_t n_m4;
    uint64_t n_m5;
    uint64_t n_m6;
    double   mse_m6;
    double   mse_selected;
    double   weighted_mse_m6;
    double   weighted_mse_selected;
    uint64_t candidate_count_total;
    uint64_t n_anchor_top1;
    uint64_t n_slot1;
    uint64_t n_slot15;
    uint64_t n_slot2;
    uint64_t n_slot3;
    uint64_t n_slot_other;
    uint64_t n_source_rich;
    uint64_t n_source_cjso_ordinary;
    uint64_t n_source_cjso_weighted;
    uint64_t n_source_both;
    uint64_t n_source_fallback;
    uint64_t n_cjso_ordinary_candidates;
    uint64_t n_cjso_weighted_candidates;
    uint64_t n_cjso_delta_m4;
    uint64_t n_cjso_delta_m3;
    uint64_t n_cjso_delta_m2;
    uint64_t n_cjso_delta_m1;
    uint64_t n_cjso_delta_0;
    uint64_t n_cjso_delta_p1;
    uint64_t n_cjso_delta_p2;
    uint64_t n_cjso_delta_p3;
    uint64_t n_cjso_delta_p4;
    uint64_t n_cjso_delta_other;
    uint64_t n_rsf_tensors;
    uint64_t n_rsf_0875;
    uint64_t n_rsf_09375;
    uint64_t n_rsf_1000;
    uint64_t n_rsf_10625;
    uint64_t n_rsf_1125;
    double   rsf_sample_error_baseline;
    double   rsf_sample_error_selected;
} ggml_nvfp4_scale_stats;

GGML_API void ggml_nvfp4_set_scale_mode(enum ggml_nvfp4_scale_mode mode);
GGML_API void ggml_nvfp4_reset_scale_stats(void);
GGML_API int  ggml_nvfp4_get_scale_stats(ggml_nvfp4_scale_stats * stats);
GGML_API const char * ggml_nvfp4_adaptive_candidates(void);
GGML_API int  ggml_nvfp4_rich_scale_search_enabled(void);
GGML_API int  ggml_nvfp4_rich_scale_code_radius(void);
GGML_API enum ggml_nvfp4_search_algo ggml_nvfp4_get_search_algo(void);
GGML_API const char * ggml_nvfp4_search_algo_name(enum ggml_nvfp4_search_algo algo);
GGML_API int  ggml_nvfp4_cjso_radius(void);
GGML_API int  ggml_nvfp4_rsf_lite_enabled(void);
GGML_API float ggml_nvfp4_choose_rsf_multiplier(const float * src, int64_t nrow, int64_t n_per_row, const float * quant_weights);
GGML_API void ggml_nvfp4_set_rsf_multiplier_override(float multiplier);
GGML_API void ggml_nvfp4_clear_rsf_multiplier_override(void);
GGML_API enum ggml_nvfp4_imatrix_select_mode ggml_nvfp4_get_imatrix_select_mode(void);
GGML_API const char * ggml_nvfp4_imatrix_select_mode_name(enum ggml_nvfp4_imatrix_select_mode mode);
GGML_API const char * ggml_nvfp4_imatrix_select_criterion(void);
GGML_API float ggml_nvfp4_imatrix_ordinary_mse_guard(void);
GGML_API void ggml_nvfp4_set_debug_context(const char * tensor_name, const float * data, int64_t n_per_row);

// Quantization
GGML_API void quantize_row_q1_0_ref(const float * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q2_0_ref(const float * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_nl_ref (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_xs_ref (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_s_ref  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_s_ref  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API void dequantize_row_q1_0(const block_q1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q2_0(const block_q2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_pq2_0_ref(const float * GGML_RESTRICT x, block_pq2 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq3_0_ref(const float * GGML_RESTRICT x, block_pq3 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq4_0_ref(const float * GGML_RESTRICT x, block_pq4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq4_0_64_ref(const float * GGML_RESTRICT x, block_pq4_d64 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq2_K_ref(const float * GGML_RESTRICT x, block_pq2_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq3_K_ref(const float * GGML_RESTRICT x, block_pq3_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq4_K_ref(const float * GGML_RESTRICT x, block_pq4_K * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq2_0(const block_pq2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq3_0(const block_pq3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq4_0(const block_pq4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq4_0_64(const block_pq4_d64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq2_K(const block_pq2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq3_K(const block_pq3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq4_K(const block_pq4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_pq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq4_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_tq2_1_ref(const float * GGML_RESTRICT x, block_tq2 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq3_1_ref(const float * GGML_RESTRICT x, block_tq3 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq4_1_ref(const float * GGML_RESTRICT x, block_tq4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq4_1_64_ref(const float * GGML_RESTRICT x, block_tq4_d64 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_1(const block_tq2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq3_1(const block_tq3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq4_1(const block_tq4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq4_1_64(const block_tq4_d64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_tq2_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq3_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq4_1_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_nvfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void iq2xs_init_impl(enum ggml_type type);
GGML_API void iq2xs_free_impl(enum ggml_type type);
GGML_API void iq3xs_init_impl(int grid_size);
GGML_API void iq3xs_free_impl(int grid_size);

#ifdef __cplusplus
}
#endif
