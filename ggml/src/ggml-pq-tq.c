/*
 * PQ/TQ KV-cache compression via PolarQuant + optional 1-bit QJL compensation.
 * Based on: arXiv 2504.19874.
 *
  * the public cache-type names are pq2/pq3/pq4 and tq2/tq3/tq4.
 */

#define _USE_MATH_DEFINES
#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-pqk-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <float.h>

/* ---------- constants ---------- */

#define PQ_TQ_SEED_ROTATION 42
#define PQ_TQ_SEED_QJL      1042
#define PQ_TQ_D128             128  /* pq/tq4 block dimension (= QK_PQ_TQ_4); head dim D is separate */
#define PQ_TQ_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 1-bit: ±sqrt(2/(pi*d)) */
static const float CENTROIDS_1BIT[2] = { -0.070711f, 0.070711f };  /* for d=128 */

/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static const float CENTROIDS_4BIT_TABLE[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

static int nearest_centroid_2bit(float val);
static int nearest_centroid_3bit(float val);
static int nearest_centroid_4bit(float val);

/* Fixed FWHT + sign matrices for 128-wide weight groups. */
static const float PQ_TQ_WHT_S1_128[128] = {
     1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
};

static const float PQ_TQ_WHT_S2_128[128] = {
     1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
};

static void pq_tq_fwht_128(float * x) {
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt128 = 0.08838834764831845f;
    for (int i = 0; i < 128; ++i) {
        x[i] *= inv_sqrt128;
    }
}

static void pq_tq_rotate_forward_128(float * x) {
    for (int i = 0; i < 128; ++i) {
        x[i] *= PQ_TQ_WHT_S1_128[i];
    }
    pq_tq_fwht_128(x);
    for (int i = 0; i < 128; ++i) {
        x[i] *= PQ_TQ_WHT_S2_128[i];
    }
}

static void pq_tq_rotate_inverse_128(float * x) {
    for (int i = 0; i < 128; ++i) {
        x[i] *= PQ_TQ_WHT_S2_128[i];
    }
    pq_tq_fwht_128(x);
    for (int i = 0; i < 128; ++i) {
        x[i] *= PQ_TQ_WHT_S1_128[i];
    }
}

static void pq_tq_quantize_group_pq2_128(const float * GGML_RESTRICT src, block_pq2 * GGML_RESTRICT dst) {
    float norm_sq = 0.0f;
    for (int i = 0; i < 128; ++i) {
        norm_sq += src[i] * src[i];
    }

    const float norm = sqrtf(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    float rotated[128];
    for (int i = 0; i < 128; ++i) {
        rotated[i] = src[i] * inv_norm;
    }
    pq_tq_rotate_forward_128(rotated);

    uint8_t qidx[128];
    float recon_norm_sq = 0.0f;
    for (int i = 0; i < 128; ++i) {
        qidx[i] = (uint8_t) nearest_centroid_2bit(rotated[i]);
        const float c = CENTROIDS_2BIT[qidx[i]];
        recon_norm_sq += c * c;
    }

    const float recon_norm = sqrtf(recon_norm_sq);
    const ggml_half norm_h = GGML_FP32_TO_FP16(recon_norm > 1e-10f ? norm / recon_norm : norm);

    for (int blk = 0; blk < 4; ++blk) {
        dst[blk].norm = norm_h;
        memset(dst[blk].qs, 0, sizeof(dst[blk].qs));
        for (int i = 0; i < 32; ++i) {
            const int local = blk * 32 + i;
            dst[blk].qs[i/4] |= (uint8_t) ((qidx[local] & 0x3u) << (2 * (i & 3)));
        }
    }
}

static void pq_tq_dequantize_group_pq2_128(const block_pq2 * GGML_RESTRICT src, float * GGML_RESTRICT dst) {
    float rotated[128];
    for (int blk = 0; blk < 4; ++blk) {
        const float norm = GGML_FP16_TO_FP32(src[blk].norm);
        for (int i = 0; i < 32; ++i) {
            const uint8_t q = (src[blk].qs[i/4] >> (2 * (i & 3))) & 0x3u;
            rotated[blk * 32 + i] = CENTROIDS_2BIT[q] * norm;
        }
    }
    pq_tq_rotate_inverse_128(rotated);
    memcpy(dst, rotated, sizeof(rotated));
}

static void pq_tq_quantize_group_pq3_128(const float * GGML_RESTRICT src, block_pq3 * GGML_RESTRICT dst) {
    float norm_sq = 0.0f;
    for (int i = 0; i < 128; ++i) {
        norm_sq += src[i] * src[i];
    }

    const float norm = sqrtf(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    float rotated[128];
    for (int i = 0; i < 128; ++i) {
        rotated[i] = src[i] * inv_norm;
    }
    pq_tq_rotate_forward_128(rotated);

    uint8_t qidx[128];
    float recon_norm_sq = 0.0f;
    for (int i = 0; i < 128; ++i) {
        qidx[i] = (uint8_t) nearest_centroid_3bit(rotated[i]);
        const float c = CENTROIDS_3BIT[qidx[i]];
        recon_norm_sq += c * c;
    }

    const float recon_norm = sqrtf(recon_norm_sq);
    const ggml_half norm_h = GGML_FP32_TO_FP16(recon_norm > 1e-10f ? norm / recon_norm : norm);

    for (int blk = 0; blk < 4; ++blk) {
        dst[blk].norm = norm_h;
        memset(dst[blk].qs, 0, sizeof(dst[blk].qs));
        memset(dst[blk].signs, 0, sizeof(dst[blk].signs));

        for (int i = 0; i < 32; ++i) {
            const int local = blk * 32 + i;
            const uint8_t q = qidx[local];
            dst[blk].qs[i/4]    |= (uint8_t) ((q & 0x3u) << (2 * (i & 3)));
            dst[blk].signs[i/8] |= (uint8_t) (((q >> 2) & 0x1u) << (i & 7));
        }
    }
}

static void pq_tq_dequantize_group_pq3_128(const block_pq3 * GGML_RESTRICT src, float * GGML_RESTRICT dst) {
    float rotated[128];
    for (int blk = 0; blk < 4; ++blk) {
        const float norm = GGML_FP16_TO_FP32(src[blk].norm);
        for (int i = 0; i < 32; ++i) {
            const uint8_t q2 = (src[blk].qs[i/4] >> (2 * (i & 3))) & 0x3u;
            const uint8_t s  = (src[blk].signs[i/8] >> (i & 7)) & 0x1u;
            rotated[blk * 32 + i] = CENTROIDS_3BIT[q2 | (s << 2)] * norm;
        }
    }
    pq_tq_rotate_inverse_128(rotated);
    memcpy(dst, rotated, sizeof(rotated));
}

static void pq_tq_quantize_group_pq4_128(const float * GGML_RESTRICT src, block_pq4 * GGML_RESTRICT dst) {
    float norm_sq = 0.0f;
    for (int i = 0; i < 128; ++i) {
        norm_sq += src[i] * src[i];
    }

    const float norm = sqrtf(norm_sq);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    float rotated[128];
    for (int i = 0; i < 128; ++i) {
        rotated[i] = src[i] * inv_norm;
    }
    pq_tq_rotate_forward_128(rotated);

    float recon_norm_sq = 0.0f;
    memset(dst->qs, 0, sizeof(dst->qs));
    for (int i = 0; i < 128; ++i) {
        const uint8_t q = (uint8_t) nearest_centroid_4bit(rotated[i]);
        const float c = CENTROIDS_4BIT_TABLE[q];
        recon_norm_sq += c * c;
        dst->qs[i/2] |= (uint8_t) ((q & 0xFu) << (4 * (i & 1)));
    }

    const float recon_norm = sqrtf(recon_norm_sq);
    dst->norm  = GGML_FP32_TO_FP16(recon_norm > 1e-10f ? norm / recon_norm : norm);
    dst->rnorm = GGML_FP32_TO_FP16(0.0f);
}

static void pq_tq_dequantize_group_pq4_128(const block_pq4 * GGML_RESTRICT src, float * GGML_RESTRICT dst) {
    float rotated[128];
    const float norm = GGML_FP16_TO_FP32(src->norm);
    for (int i = 0; i < 128; ++i) {
        const uint8_t q = (src->qs[i/2] >> (4 * (i & 1))) & 0xFu;
        rotated[i] = CENTROIDS_4BIT_TABLE[q] * norm;
    }
    pq_tq_rotate_inverse_128(rotated);
    memcpy(dst, rotated, sizeof(rotated));
}

/* ---------- rotation matrix (lazy init) ---------- */

static float pq_tq_rotation[PQ_TQ_D128 * PQ_TQ_D128];
static float pq_tq_rotation_t[PQ_TQ_D128 * PQ_TQ_D128]; /* transpose */
static int   pq_tq_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t pq_tq_prng_state;

static void pq_tq_prng_seed(uint64_t seed) {
    pq_tq_prng_state = seed;
}

static double pq_tq_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    pq_tq_prng_state = pq_tq_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(pq_tq_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    pq_tq_prng_state = pq_tq_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(pq_tq_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void pq_tq_init_rotation(void) {
    if (pq_tq_rotation_initialized) return;

    const int d = PQ_TQ_D128;

    /* Generate random Gaussian matrix */
    pq_tq_prng_seed(PQ_TQ_SEED_ROTATION);
    float G[PQ_TQ_D128 * PQ_TQ_D128];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)pq_tq_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in pq_tq_rotation */
    memcpy(pq_tq_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += pq_tq_rotation[i * d + j] * pq_tq_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                pq_tq_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += pq_tq_rotation[i * d + j] * pq_tq_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                pq_tq_rotation[i * d + k] -= dot * pq_tq_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            pq_tq_rotation_t[i * d + j] = pq_tq_rotation[j * d + i];
        }
    }

    pq_tq_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float tq_qjl_matrix[PQ_TQ_D128 * PQ_TQ_D128];
static float tq_qjl_matrix_t[PQ_TQ_D128 * PQ_TQ_D128];
static int   tq_qjl_initialized = 0;

static void tq_init_qjl(void) {
    if (tq_qjl_initialized) return;

    const int d = PQ_TQ_D128;
    pq_tq_prng_seed(PQ_TQ_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        tq_qjl_matrix[i] = (float)pq_tq_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            tq_qjl_matrix_t[i * d + j] = tq_qjl_matrix[j * d + i];
        }
    }

    tq_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.145560f) return 0;
    if (val < -0.103361f) return 1;
    if (val < -0.079142f) return 2;
    if (val < -0.060009f) return 3;
    if (val < -0.043430f) return 4;
    if (val < -0.028293f) return 5;
    if (val < -0.013963f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.013963f) return 8;
    if (val <  0.028293f) return 9;
    if (val <  0.043430f) return 10;
    if (val <  0.060009f) return 11;
    if (val <  0.079142f) return 12;
    if (val <  0.103361f) return 13;
    if (val <  0.145560f) return 14;
    return 15;
}

/* ---------- PQ2_0: 2-bit PolarQuant (4 centroids) ---------- */

void quantize_row_pq2_0_ref(const float * GGML_RESTRICT x, block_pq2 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_2_GROUP == 0);
    const int ng = k / QK_PQ_TQ_2_GROUP;
    for (int g = 0; g < ng; ++g) {
        pq_tq_quantize_group_pq2_128(x + g * QK_PQ_TQ_2_GROUP, y + g * (QK_PQ_TQ_2_GROUP / QK_PQ_TQ_2));
    }
}

void dequantize_row_pq2_0(const block_pq2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_2_GROUP == 0);
    const int ng = k / QK_PQ_TQ_2_GROUP;
    for (int g = 0; g < ng; ++g) {
        pq_tq_dequantize_group_pq2_128(x + g * (QK_PQ_TQ_2_GROUP / QK_PQ_TQ_2), y + g * QK_PQ_TQ_2_GROUP);
    }
}

size_t quantize_pq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_2_GROUP == 0);

    size_t row_size = (n_per_row / QK_PQ_TQ_2) * sizeof(block_pq2);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_pq2_0_ref(
            src + row * n_per_row,
            (block_pq2 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- PQ3_0: 3-bit PolarQuant ---------- */

void quantize_row_pq3_0_ref(const float * GGML_RESTRICT x, block_pq3 * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles quantize on GPU. CPU path is simplified.
    assert(k % QK_PQ_TQ_3_GROUP == 0);
    const int ng = k / QK_PQ_TQ_3_GROUP;
    for (int g = 0; g < ng; ++g) {
        pq_tq_quantize_group_pq3_128(x + g * QK_PQ_TQ_3_GROUP, y + g * (QK_PQ_TQ_3_GROUP / QK_PQ_TQ_3));
    }
}

void dequantize_row_pq3_0(const block_pq3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_PQ_TQ_3_GROUP == 0);
    const int ng = k / QK_PQ_TQ_3_GROUP;
    for (int g = 0; g < ng; ++g) {
        pq_tq_dequantize_group_pq3_128(x + g * (QK_PQ_TQ_3_GROUP / QK_PQ_TQ_3), y + g * QK_PQ_TQ_3_GROUP);
    }
}

size_t quantize_pq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_3_GROUP == 0);

    size_t row_size = (n_per_row / QK_PQ_TQ_3) * sizeof(block_pq3);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_pq3_0_ref(
            src + row * n_per_row,
            (block_pq3 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- PQ4_0: 4-bit PolarQuant (16 centroids) ---------- */

void quantize_row_pq4_0_ref(const float * GGML_RESTRICT x, block_pq4 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_4 == 0);
    const int nb = k / QK_PQ_TQ_4;
    for (int block = 0; block < nb; ++block) {
        pq_tq_quantize_group_pq4_128(x + block * QK_PQ_TQ_4, y + block);
    }
}

void dequantize_row_pq4_0(const block_pq4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_4 == 0);
    const int nb = k / QK_PQ_TQ_4;

    /* 4-bit PolarQuant: nibble unpack → centroid → inverse rotate → scale */    
    for (int block = 0; block < nb; ++block) {
        pq_tq_dequantize_group_pq4_128(x + block, y + block * QK_PQ_TQ_4);
    }
}

size_t quantize_pq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_4 == 0);

    size_t row_size = (n_per_row / QK_PQ_TQ_4) * sizeof(block_pq4);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_pq4_0_ref(
            src + row * n_per_row,
            (block_pq4 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* Internal D=64 pq4 layouts ------------------------------------------------ */

void quantize_row_pq4_0_64_ref(const float * GGML_RESTRICT x, block_pq4_d64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_4_D64 == 0);
    const int nb = k / QK_PQ_TQ_4_D64;
    for (int block = 0; block < nb; ++block) {
        const float * src = x + block * QK_PQ_TQ_4_D64;
        float norm_sq = 0.0f;
        for (int i = 0; i < QK_PQ_TQ_4_D64; ++i) {
            norm_sq += src[i] * src[i];
        }
        const float norm = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

        uint8_t indices[QK_PQ_TQ_4_D64];
        float recon_norm_sq = 0.0f;
        memset(y[block].qs, 0, sizeof(y[block].qs));
        for (int i = 0; i < QK_PQ_TQ_4_D64; ++i) {
            indices[i] = (uint8_t)nearest_centroid_4bit(src[i] * inv_norm);
            recon_norm_sq += CENTROIDS_4BIT_TABLE[indices[i]] * CENTROIDS_4BIT_TABLE[indices[i]];
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }

        const float recon_norm = sqrtf(recon_norm_sq);
        const float corrected_norm = recon_norm > 1e-10f ? norm / recon_norm : norm;
        y[block].norm  = GGML_FP32_TO_FP16(corrected_norm);
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_pq4_0_64(const block_pq4_d64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_4_D64 == 0);
    const int nb = k / QK_PQ_TQ_4_D64;
    for (int block = 0; block < nb; ++block) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int i = 0; i < QK_PQ_TQ_4_D64; ++i) {
            const uint8_t qb = x[block].qs[i / 2];
            const uint8_t idx = (qb >> ((i % 2) * 4)) & 0xF;
            y[block * QK_PQ_TQ_4_D64 + i] = CENTROIDS_4BIT_TABLE[idx] * norm;
        }
    }
}

size_t quantize_pq4_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_4_D64 == 0);

    const size_t row_size = (n_per_row / QK_PQ_TQ_4_D64) * sizeof(block_pq4_d64);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_pq4_0_64_ref(
            src + row * n_per_row,
            (block_pq4_d64 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- PQn_K: rotated 256-wide PolarQuant K-family ---------- */

static const uint32_t PQK_WHT_SIGNS1_256_BITS[8] = {
    0xc3284666u, 0xce93b542u, 0x79141579u, 0x9aa89715u,
    0x9b0404dau, 0x0af8ae67u, 0xef41f700u, 0xd712a44au
};

static const uint32_t PQK_WHT_SIGNS2_256_BITS[8] = {
    0x6e2e718eu, 0x82fc60a0u, 0xb7719342u, 0x67487f5au,
    0xbfd09d07u, 0xaeadc1c4u, 0xd5c0b687u, 0x6c1b19a0u
};

static float pqk_wht_apply_sign_256(float x, const uint32_t * signs, int idx) {
    const uint32_t sign = ((signs[idx >> 5] >> (idx & 31)) & 1u) << 31;
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits ^= sign;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static void pqk_fwht_256(float * data) {
    for (int len = 1; len < QK_K; len <<= 1) {
        for (int base = 0; base < QK_K; base += 2 * len) {
            for (int i = 0; i < len; ++i) {
                const float a = data[base + i + 0];
                const float b = data[base + i + len];
                data[base + i + 0] = a + b;
                data[base + i + len] = a - b;
            }
        }
    }
}

static void pqk_rotate_forward_256(float * data) {
    for (int i = 0; i < QK_K; ++i) {
        data[i] = pqk_wht_apply_sign_256(data[i], PQK_WHT_SIGNS1_256_BITS, i);
    }
    pqk_fwht_256(data);
    for (int i = 0; i < QK_K; ++i) {
        data[i] = pqk_wht_apply_sign_256(data[i] * 0.0625f, PQK_WHT_SIGNS2_256_BITS, i);
    }
}

static void pqk_rotate_inverse_256(float * data) {
    for (int i = 0; i < QK_K; ++i) {
        data[i] = pqk_wht_apply_sign_256(data[i], PQK_WHT_SIGNS2_256_BITS, i);
    }
    pqk_fwht_256(data);
    for (int i = 0; i < QK_K; ++i) {
        data[i] = pqk_wht_apply_sign_256(data[i] * 0.0625f, PQK_WHT_SIGNS1_256_BITS, i);
    }
}

typedef float (*pqk_centroid_fn_t)(uint8_t q);

typedef struct {
    int levels;
    float max_centroid;
    pqk_centroid_fn_t centroid_fn;
} pqk_codebook_spec;

typedef struct {
    float band_master[GGML_PQK_BAND_COUNT];
    uint8_t scale_q[GGML_PQK_SUBBLOCK_COUNT];
    uint8_t qidx[QK_K];
} pqk_quantized_block_tmp;

static float pqk_centroid_2bit_host(uint8_t q) {
    return ggml_pqk_centroid_2bit(q);
}

static float pqk_centroid_3bit_host(uint8_t q) {
    return ggml_pqk_centroid_3bit(q);
}

static float pqk_centroid_4bit_host(uint8_t q) {
    return ggml_pqk_centroid_4bit(q);
}

static const pqk_codebook_spec PQK_CODEBOOK_2BIT = {
    4,
    GGML_PQ2K_MAX_CENTROID,
    pqk_centroid_2bit_host,
};

static const pqk_codebook_spec PQK_CODEBOOK_3BIT = {
    8,
    GGML_PQ3K_MAX_CENTROID,
    pqk_centroid_3bit_host,
};

static const pqk_codebook_spec PQK_CODEBOOK_4BIT = {
    16,
    GGML_PQ4K_MAX_CENTROID,
    pqk_centroid_4bit_host,
};

static int pqk_compare_float(const void * a, const void * b) {
    const float fa = *(const float *) a;
    const float fb = *(const float *) b;
    return (fa > fb) - (fa < fb);
}

static uint8_t pqk_nearest_centroid_generic(float value, int levels, pqk_centroid_fn_t centroid_fn) {
    uint8_t best_q = 0;
    float best_err = fabsf(value - centroid_fn(0));
    for (int q = 1; q < levels; ++q) {
        const float err = fabsf(value - centroid_fn((uint8_t) q));
        if (err < best_err) {
            best_err = err;
            best_q = (uint8_t) q;
        }
    }
    return best_q;
}

static float pqk_subblock_error_generic(const float * src, const uint8_t * qidx, float scale, pqk_centroid_fn_t centroid_fn) {
    float err = 0.0f;
    for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
        const float diff = src[i] - scale * centroid_fn(qidx[i]);
        err += diff * diff;
    }
    return err / GGML_PQK_SUBBLOCK_SIZE;
}

static float pqk_fit_subblock_fast(const float * src, uint8_t * qidx, int levels, float max_centroid, pqk_centroid_fn_t centroid_fn, float * err_out) {
    float max_abs = 0.0f;
    for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    if (max_abs < 1e-12f) {
        memset(qidx, 0, GGML_PQK_SUBBLOCK_SIZE);
        if (err_out != NULL) {
            *err_out = 0.0f;
        }
        return 0.0f;
    }

    float scale = max_abs / max_centroid;
    if (scale < 1e-12f) {
        memset(qidx, 0, GGML_PQK_SUBBLOCK_SIZE);
        if (err_out != NULL) {
            *err_out = 0.0f;
        }
        return 0.0f;
    }

    float numer = 0.0f;
    float denom = 0.0f;
    for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
        const uint8_t q = pqk_nearest_centroid_generic(src[i] / scale, levels, centroid_fn);
        const float c = centroid_fn(q);
        qidx[i] = q;
        numer += src[i] * c;
        denom += c * c;
    }

    if (denom < 1e-20f) {
        memset(qidx, 0, GGML_PQK_SUBBLOCK_SIZE);
        if (err_out != NULL) {
            *err_out = 0.0f;
        }
        return 0.0f;
    }

    scale = fmaxf(numer / denom, 0.0f);
    if (scale < 1e-12f) {
        memset(qidx, 0, GGML_PQK_SUBBLOCK_SIZE);
        if (err_out != NULL) {
            *err_out = 0.0f;
        }
        return 0.0f;
    }

    numer = 0.0f;
    denom = 0.0f;
    for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
        const uint8_t q = pqk_nearest_centroid_generic(src[i] / scale, levels, centroid_fn);
        const float c = centroid_fn(q);
        qidx[i] = q;
        numer += src[i] * c;
        denom += c * c;
    }

    if (denom >= 1e-20f) {
        scale = fmaxf(numer / denom, 0.0f);
    }

    if (scale < 1e-12f) {
        memset(qidx, 0, GGML_PQK_SUBBLOCK_SIZE);
        if (err_out != NULL) {
            *err_out = 0.0f;
        }
        return 0.0f;
    }

    if (err_out != NULL) {
        *err_out = pqk_subblock_error_generic(src, qidx, scale, centroid_fn);
    }

    return scale;
}

static float pqk_quantize_subblock_with_scale(const float * src, uint8_t * qidx, int levels, float scale, pqk_centroid_fn_t centroid_fn) {
    if (scale < 1e-12f) {
        memset(qidx, 0, GGML_PQK_SUBBLOCK_SIZE);
        float err = 0.0f;
        for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
            err += src[i] * src[i];
        }
        return err / GGML_PQK_SUBBLOCK_SIZE;
    }

    for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
        qidx[i] = pqk_nearest_centroid_generic(src[i] / scale, levels, centroid_fn);
    }

    return pqk_subblock_error_generic(src, qidx, scale, centroid_fn);
}

static float pqk_geometric_mean(const float * values, int count) {
    float log_sum = 0.0f;
    int n = 0;
    for (int i = 0; i < count; ++i) {
        if (values[i] > 1e-12f) {
            log_sum += logf(values[i]);
            ++n;
        }
    }
    return n > 0 ? expf(log_sum / n) : 0.0f;
}

static float pqk_max_value(const float * values, int count) {
    float max_value = 0.0f;
    for (int i = 0; i < count; ++i) {
        if (values[i] > max_value) {
            max_value = values[i];
        }
    }
    return max_value;
}

static uint8_t pqk_encode_local_scale(float master, float local);

static float pqk_quantized_master_loss(float log2_master, const float * values, int count) {
    const float master = exp2f(log2_master);
    const float delta = 2.0f * GGML_PQK_LOG_SCALE_STEP;
    float loss = 0.0f;

    for (int i = 0; i < count; ++i) {
        const float local = values[i];
        if (local <= 1e-12f) {
            continue;
        }

        const float decoded = ggml_pqk_decode_local_scale(master, pqk_encode_local_scale(master, local));
        if (decoded <= 1e-12f) {
            continue;
        }

        const float err = fabsf(log2f(local / decoded));
        if (err <= delta) {
            loss += 0.5f * err * err;
        } else {
            loss += delta * (err - 0.5f * delta);
        }
    }

    return loss;
}

static float pqk_choose_band_master_fast(const float * values, int count) {
    float log_values[GGML_PQK_SUBBLOCKS_PER_BAND];
    int n = 0;

    for (int i = 0; i < count; ++i) {
        if (values[i] > 1e-12f) {
            log_values[n++] = log2f(values[i]);
        }
    }

    if (n == 0) {
        return 0.0f;
    }

    qsort(log_values, n, sizeof(float), pqk_compare_float);

    float candidates[3];
    int nc = 0;

    candidates[nc++] = log2f(pqk_geometric_mean(values, count));

    if ((n & 1) != 0) {
        candidates[nc++] = log_values[n / 2];
    } else {
        candidates[nc++] = 0.5f * (log_values[n / 2 - 1] + log_values[n / 2]);
    }

    candidates[nc++] = log2f(pqk_max_value(values, count));

    float best_log2_master = candidates[0];
    float best_loss = pqk_quantized_master_loss(best_log2_master, values, count);

    for (int i = 1; i < nc; ++i) {
        if (fabsf(candidates[i] - candidates[i - 1]) < 1e-6f) {
            continue;
        }

        const float loss = pqk_quantized_master_loss(candidates[i], values, count);
        if (loss < best_loss) {
            best_loss = loss;
            best_log2_master = candidates[i];
        }
    }

    return exp2f(best_log2_master);
}

static int pqk_collect_band_master_candidates(const float * values, int count, float * candidates) {
    float log_values[GGML_PQK_SUBBLOCKS_PER_BAND];
    int n = 0;

    for (int i = 0; i < count; ++i) {
        if (values[i] > 1e-12f) {
            log_values[n++] = log2f(values[i]);
        }
    }

    if (n == 0) {
        return 0;
    }

    qsort(log_values, n, sizeof(float), pqk_compare_float);

    int nc = 0;
    candidates[nc++] = log2f(pqk_geometric_mean(values, count));

    if ((n & 1) != 0) {
        candidates[nc++] = log_values[n / 2];
    } else {
        candidates[nc++] = 0.5f * (log_values[n / 2 - 1] + log_values[n / 2]);
    }

    candidates[nc++] = log2f(pqk_max_value(values, count));
    return nc;
}

static uint8_t pqk_encode_local_scale(float master, float local) {
    if (master <= 0.0f || local <= 1e-12f) {
        return 0;
    }

    const float log_ratio = log2f(local / master);
    if (log_ratio <= GGML_PQK_LOG_SCALE_MIN) {
        return 1;
    }
    if (log_ratio >= GGML_PQK_LOG_SCALE_MAX) {
        return 63;
    }

    int q = 1 + (int) roundf((log_ratio - GGML_PQK_LOG_SCALE_MIN) / GGML_PQK_LOG_SCALE_STEP);
    if (q < 1) {
        q = 1;
    } else if (q > 63) {
        q = 63;
    }
    return (uint8_t) q;
}

static void pqk_quantize_block_fast(const float * src, pqk_quantized_block_tmp * dst, const pqk_codebook_spec * spec) {
    float rotated[QK_K];
    float local_exact[GGML_PQK_SUBBLOCK_COUNT];

    memcpy(rotated, src, sizeof(rotated));
    pqk_rotate_forward_256(rotated);

    for (int sb = 0; sb < GGML_PQK_SUBBLOCK_COUNT; ++sb) {
        uint8_t qidx[GGML_PQK_SUBBLOCK_SIZE];
        float err = 0.0f;
        local_exact[sb] = pqk_fit_subblock_fast(
                rotated + sb * GGML_PQK_SUBBLOCK_SIZE,
                qidx,
                spec->levels,
                spec->max_centroid,
                spec->centroid_fn,
                &err);
    }

    for (int band = 0; band < GGML_PQK_BAND_COUNT; ++band) {
        float band_locals[GGML_PQK_SUBBLOCKS_PER_BAND];
        for (int i = 0; i < GGML_PQK_SUBBLOCKS_PER_BAND; ++i) {
            const int sb = band * GGML_PQK_SUBBLOCKS_PER_BAND + i;
            band_locals[i] = local_exact[sb];
        }
        dst->band_master[band] = pqk_choose_band_master_fast(band_locals, GGML_PQK_SUBBLOCKS_PER_BAND);
    }

    for (int sb = 0; sb < GGML_PQK_SUBBLOCK_COUNT; ++sb) {
        const int band = sb / GGML_PQK_SUBBLOCKS_PER_BAND;
        dst->scale_q[sb] = pqk_encode_local_scale(dst->band_master[band], local_exact[sb]);
        const float scale = ggml_pqk_decode_local_scale(dst->band_master[band], dst->scale_q[sb]);
        pqk_quantize_subblock_with_scale(
                rotated + sb * GGML_PQK_SUBBLOCK_SIZE,
                dst->qidx + sb * GGML_PQK_SUBBLOCK_SIZE,
                spec->levels,
                scale,
                spec->centroid_fn);
    }
}

static uint8_t pqk_clamp_local_scale_q(int q) {
    if (q < 1) {
        return 1;
    }
    if (q > 63) {
        return 63;
    }
    return (uint8_t) q;
}

static void pqk_quantize_block_with_masters(
        const float * rotated, const float * local_exact, const float * band_master,
        pqk_quantized_block_tmp * dst, const pqk_codebook_spec * spec, int local_q_delta) {
    dst->band_master[0] = band_master[0];
    dst->band_master[1] = band_master[1];

    for (int sb = 0; sb < GGML_PQK_SUBBLOCK_COUNT; ++sb) {
        const int band = sb / GGML_PQK_SUBBLOCKS_PER_BAND;
        uint8_t qscale = pqk_encode_local_scale(dst->band_master[band], local_exact[sb]);
        if (qscale != 0 && local_q_delta != 0) {
            qscale = pqk_clamp_local_scale_q((int) qscale + local_q_delta);
        }

        dst->scale_q[sb] = qscale;
        const float scale = ggml_pqk_decode_local_scale(dst->band_master[band], qscale);
        pqk_quantize_subblock_with_scale(
                rotated + sb * GGML_PQK_SUBBLOCK_SIZE,
                dst->qidx + sb * GGML_PQK_SUBBLOCK_SIZE,
                spec->levels,
                scale,
                spec->centroid_fn);
    }
}

static float pqk_weighted_loss_original_domain(
        const float * rotated, const pqk_quantized_block_tmp * q, const pqk_codebook_spec * spec,
        const float * weights) {
    float residual[QK_K];

    for (int sb = 0; sb < GGML_PQK_SUBBLOCK_COUNT; ++sb) {
        const int band = sb / GGML_PQK_SUBBLOCKS_PER_BAND;
        const float scale = ggml_pqk_decode_local_scale(q->band_master[band], q->scale_q[sb]);
        for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
            const int idx = sb * GGML_PQK_SUBBLOCK_SIZE + i;
            residual[idx] = rotated[idx] - scale * spec->centroid_fn(q->qidx[idx]);
        }
    }

    // PQ_K quantizes in the rotated domain, but imatrix weights live in the
    // original coordinates. Score candidates after rotating the residual back.
    pqk_rotate_inverse_256(residual);

    float loss = 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        loss += weights[i] * residual[i] * residual[i];
    }
    return loss;
}

static void pqk_quantize_block_weighted(
        const float * src, const float * weights, pqk_quantized_block_tmp * dst,
        const pqk_codebook_spec * spec) {
    float rotated[QK_K];
    float local_exact[GGML_PQK_SUBBLOCK_COUNT];
    float band_master[GGML_PQK_BAND_COUNT];

    memcpy(rotated, src, sizeof(rotated));
    pqk_rotate_forward_256(rotated);

    for (int sb = 0; sb < GGML_PQK_SUBBLOCK_COUNT; ++sb) {
        uint8_t qidx[GGML_PQK_SUBBLOCK_SIZE];
        float err = 0.0f;
        local_exact[sb] = pqk_fit_subblock_fast(
                rotated + sb * GGML_PQK_SUBBLOCK_SIZE,
                qidx,
                spec->levels,
                spec->max_centroid,
                spec->centroid_fn,
                &err);
    }

    for (int band = 0; band < GGML_PQK_BAND_COUNT; ++band) {
        float band_locals[GGML_PQK_SUBBLOCKS_PER_BAND];
        for (int i = 0; i < GGML_PQK_SUBBLOCKS_PER_BAND; ++i) {
            const int sb = band * GGML_PQK_SUBBLOCKS_PER_BAND + i;
            band_locals[i] = local_exact[sb];
        }
        band_master[band] = pqk_choose_band_master_fast(band_locals, GGML_PQK_SUBBLOCKS_PER_BAND);
    }

    pqk_quantize_block_with_masters(rotated, local_exact, band_master, dst, spec, 0);
    float best_loss = pqk_weighted_loss_original_domain(rotated, dst, spec, weights);

    for (int band = 0; band < GGML_PQK_BAND_COUNT; ++band) {
        float band_locals[GGML_PQK_SUBBLOCKS_PER_BAND];
        for (int i = 0; i < GGML_PQK_SUBBLOCKS_PER_BAND; ++i) {
            const int sb = band * GGML_PQK_SUBBLOCKS_PER_BAND + i;
            band_locals[i] = local_exact[sb];
        }

        float candidates[3];
        const int nc = pqk_collect_band_master_candidates(band_locals, GGML_PQK_SUBBLOCKS_PER_BAND, candidates);
        for (int i = 0; i < nc; ++i) {
            const float candidate_master = exp2f(candidates[i]);
            if (fabsf(candidate_master - band_master[band]) <= 1e-12f * fmaxf(1.0f, band_master[band])) {
                continue;
            }

            float trial_master[GGML_PQK_BAND_COUNT] = { band_master[0], band_master[1] };
            pqk_quantized_block_tmp trial;
            trial_master[band] = candidate_master;
            pqk_quantize_block_with_masters(rotated, local_exact, trial_master, &trial, spec, 0);

            const float loss = pqk_weighted_loss_original_domain(rotated, &trial, spec, weights);
            if (loss < best_loss) {
                best_loss = loss;
                *dst = trial;
                band_master[0] = dst->band_master[0];
                band_master[1] = dst->band_master[1];
            }
        }
    }

    // Keep the refinement deliberately small: the initial sub-block search stays
    // unchanged, and imatrix only adjudicates nearby encoded local-scale choices.
    for (int delta = -1; delta <= 1; delta += 2) {
        pqk_quantized_block_tmp trial;
        pqk_quantize_block_with_masters(rotated, local_exact, band_master, &trial, spec, delta);
        const float loss = pqk_weighted_loss_original_domain(rotated, &trial, spec, weights);
        if (loss < best_loss) {
            best_loss = loss;
            *dst = trial;
        }
    }
}

static void pqk_quantize_block_generic(const float * src, pqk_quantized_block_tmp * dst, const pqk_codebook_spec * spec) {
    pqk_quantize_block_fast(src, dst, spec);
}

static void pqk_quantize_block_generic_weighted(
        const float * src, const float * weights, pqk_quantized_block_tmp * dst,
        const pqk_codebook_spec * spec) {
    if (weights != NULL) {
        pqk_quantize_block_weighted(src, weights, dst, spec);
    } else {
        pqk_quantize_block_fast(src, dst, spec);
    }
}

static void pqk_store_scales(ggml_half * d, uint8_t * scales, const pqk_quantized_block_tmp * src) {
    d[0] = GGML_FP32_TO_FP16(src->band_master[0]);
    d[1] = GGML_FP32_TO_FP16(src->band_master[1]);

    memset(scales, 0, K_SCALE_SIZE);
    for (int sb = 0; sb < GGML_PQK_SUBBLOCK_COUNT; ++sb) {
        ggml_pqk_scale_set(scales, sb, src->scale_q[sb]);
    }
}

static void pqk_store_block_pq2_K(block_pq2_K * dst, const pqk_quantized_block_tmp * src) {
    pqk_store_scales(dst->d, dst->scales, src);
    memset(dst->qs, 0, sizeof(dst->qs));
    for (int i = 0; i < QK_K; ++i) {
        dst->qs[i / 4] |= (uint8_t)((src->qidx[i] & 0x3u) << (2 * (i & 3)));
    }
}

static void pqk_store_block_pq3_K(block_pq3_K * dst, const pqk_quantized_block_tmp * src) {
    pqk_store_scales(dst->d, dst->scales, src);
    memset(dst->qs, 0, sizeof(dst->qs));
    memset(dst->hmask, 0, sizeof(dst->hmask));
    for (int i = 0; i < QK_K; ++i) {
        const uint8_t q = src->qidx[i];
        dst->qs[i / 4] |= (uint8_t)((q & 0x3u) << (2 * (i & 3)));
        dst->hmask[i / 8] |= (uint8_t)(((q >> 2) & 0x1u) << (i & 7));
    }
}

static void pqk_store_block_pq4_K(block_pq4_K * dst, const pqk_quantized_block_tmp * src) {
    pqk_store_scales(dst->d, dst->scales, src);
    memset(dst->qs, 0, sizeof(dst->qs));
    for (int i = 0; i < QK_K; ++i) {
        dst->qs[i / 2] |= (uint8_t)((src->qidx[i] & 0xFu) << (4 * (i & 1)));
    }
}

void quantize_row_pq2_K_ref(const float * GGML_RESTRICT x, block_pq2_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int block = 0; block < nb; ++block) {
        pqk_quantized_block_tmp tmp;
        pqk_quantize_block_generic(x + block * QK_K, &tmp, &PQK_CODEBOOK_2BIT);
        pqk_store_block_pq2_K(y + block, &tmp);
    }
}

void quantize_row_pq3_K_ref(const float * GGML_RESTRICT x, block_pq3_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int block = 0; block < nb; ++block) {
        pqk_quantized_block_tmp tmp;
        pqk_quantize_block_generic(x + block * QK_K, &tmp, &PQK_CODEBOOK_3BIT);
        pqk_store_block_pq3_K(y + block, &tmp);
    }
}

void quantize_row_pq4_K_ref(const float * GGML_RESTRICT x, block_pq4_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int block = 0; block < nb; ++block) {
        pqk_quantized_block_tmp tmp;
        pqk_quantize_block_generic(x + block * QK_K, &tmp, &PQK_CODEBOOK_4BIT);
        pqk_store_block_pq4_K(y + block, &tmp);
    }
}

static void quantize_row_pq2_K_impl(
        const float * GGML_RESTRICT x, block_pq2_K * GGML_RESTRICT y, int64_t k,
        const float * GGML_RESTRICT weights) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int block = 0; block < nb; ++block) {
        pqk_quantized_block_tmp tmp;
        pqk_quantize_block_generic_weighted(x + block * QK_K, weights ? weights + block * QK_K : NULL, &tmp, &PQK_CODEBOOK_2BIT);
        pqk_store_block_pq2_K(y + block, &tmp);
    }
}

static void quantize_row_pq3_K_impl(
        const float * GGML_RESTRICT x, block_pq3_K * GGML_RESTRICT y, int64_t k,
        const float * GGML_RESTRICT weights) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int block = 0; block < nb; ++block) {
        pqk_quantized_block_tmp tmp;
        pqk_quantize_block_generic_weighted(x + block * QK_K, weights ? weights + block * QK_K : NULL, &tmp, &PQK_CODEBOOK_3BIT);
        pqk_store_block_pq3_K(y + block, &tmp);
    }
}

static void quantize_row_pq4_K_impl(
        const float * GGML_RESTRICT x, block_pq4_K * GGML_RESTRICT y, int64_t k,
        const float * GGML_RESTRICT weights) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int block = 0; block < nb; ++block) {
        pqk_quantized_block_tmp tmp;
        pqk_quantize_block_generic_weighted(x + block * QK_K, weights ? weights + block * QK_K : NULL, &tmp, &PQK_CODEBOOK_4BIT);
        pqk_store_block_pq4_K(y + block, &tmp);
    }
}

void dequantize_row_pq2_K(const block_pq2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int block = 0; block < nb; ++block) {
        float rotated[QK_K];
        for (int sb = 0; sb < GGML_PQK_SUBBLOCK_COUNT; ++sb) {
            const int band = sb / GGML_PQK_SUBBLOCKS_PER_BAND;
            const float master = GGML_FP16_TO_FP32(x[block].d[band]);
            const float scale = ggml_pqk_decode_local_scale(master, ggml_pqk_scale_get(x[block].scales, sb));
            for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
                const int idx = sb * GGML_PQK_SUBBLOCK_SIZE + i;
                const uint8_t q = (x[block].qs[idx / 4] >> (2 * (idx & 3))) & 0x3u;
                rotated[idx] = scale * ggml_pqk_centroid_2bit(q);
            }
        }
        pqk_rotate_inverse_256(rotated);
        memcpy(y + block * QK_K, rotated, sizeof(rotated));
    }
}

void dequantize_row_pq3_K(const block_pq3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int block = 0; block < nb; ++block) {
        float rotated[QK_K];
        for (int sb = 0; sb < GGML_PQK_SUBBLOCK_COUNT; ++sb) {
            const int band = sb / GGML_PQK_SUBBLOCKS_PER_BAND;
            const float master = GGML_FP16_TO_FP32(x[block].d[band]);
            const float scale = ggml_pqk_decode_local_scale(master, ggml_pqk_scale_get(x[block].scales, sb));
            for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
                const int idx = sb * GGML_PQK_SUBBLOCK_SIZE + i;
                const uint8_t ql = (x[block].qs[idx / 4] >> (2 * (idx & 3))) & 0x3u;
                const uint8_t qh = (x[block].hmask[idx / 8] >> (idx & 7)) & 0x1u;
                rotated[idx] = scale * ggml_pqk_centroid_3bit((uint8_t)(ql | (qh << 2)));
            }
        }
        pqk_rotate_inverse_256(rotated);
        memcpy(y + block * QK_K, rotated, sizeof(rotated));
    }
}

void dequantize_row_pq4_K(const block_pq4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int block = 0; block < nb; ++block) {
        float rotated[QK_K];
        for (int sb = 0; sb < GGML_PQK_SUBBLOCK_COUNT; ++sb) {
            const int band = sb / GGML_PQK_SUBBLOCKS_PER_BAND;
            const float master = GGML_FP16_TO_FP32(x[block].d[band]);
            const float scale = ggml_pqk_decode_local_scale(master, ggml_pqk_scale_get(x[block].scales, sb));
            for (int i = 0; i < GGML_PQK_SUBBLOCK_SIZE; ++i) {
                const int idx = sb * GGML_PQK_SUBBLOCK_SIZE + i;
                const uint8_t q = (x[block].qs[idx / 2] >> (4 * (idx & 1))) & 0xFu;
                rotated[idx] = scale * ggml_pqk_centroid_4bit(q);
            }
        }
        pqk_rotate_inverse_256(rotated);
        memcpy(y + block * QK_K, rotated, sizeof(rotated));
    }
}

size_t quantize_pq2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);

    const size_t row_size = (n_per_row / QK_K) * sizeof(block_pq2_K);
    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix != NULL) {
            quantize_row_pq2_K_impl(src + row * n_per_row, (block_pq2_K *)((char *)dst + row * row_size), n_per_row, imatrix);
        } else {
            quantize_row_pq2_K_ref(src + row * n_per_row, (block_pq2_K *)((char *)dst + row * row_size), n_per_row);
        }
    }
    return nrows * row_size;
}

size_t quantize_pq3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);

    const size_t row_size = (n_per_row / QK_K) * sizeof(block_pq3_K);
    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix != NULL) {
            quantize_row_pq3_K_impl(src + row * n_per_row, (block_pq3_K *)((char *)dst + row * row_size), n_per_row, imatrix);
        } else {
            quantize_row_pq3_K_ref(src + row * n_per_row, (block_pq3_K *)((char *)dst + row * row_size), n_per_row);
        }
    }
    return nrows * row_size;
}

size_t quantize_pq4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);

    const size_t row_size = (n_per_row / QK_K) * sizeof(block_pq4_K);
    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix != NULL) {
            quantize_row_pq4_K_impl(src + row * n_per_row, (block_pq4_K *)((char *)dst + row * row_size), n_per_row, imatrix);
        } else {
            quantize_row_pq4_K_ref(src + row * n_per_row, (block_pq4_K *)((char *)dst + row * row_size), n_per_row);
        }
    }
    return nrows * row_size;
}

/* ===================================================================== */
/* QJL VARIANTS (_1): base MSE + 1-bit residual sign correction          */
/* ===================================================================== */

#define TQ_QJL_CORR_CPU 0.0705348f  /* sqrt(2/(pi*128)) */

/* TQ2_1 ------------------------------------------------------------ */
void quantize_row_tq2_1_ref(const float * GGML_RESTRICT x, block_tq2 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_2 == 0);
    const int nb = k / QK_PQ_TQ_2;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_PQ_TQ_2; j++) norm += x[i*QK_PQ_TQ_2 + j] * x[i*QK_PQ_TQ_2 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_PQ_TQ_2 / 4);
        memset(y[i].qjl, 0, QK_PQ_TQ_2 / 8);
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_tq2_1(const block_tq2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_2 == 0);
    const int nb = k / QK_PQ_TQ_2;
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);
        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        for (int j = 0; j < QK_PQ_TQ_2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t s = (x[block].qjl[j/8] >> (j%8)) & 0x1;
            y[block * QK_PQ_TQ_2 + j] = CENTROIDS_2BIT[idx] * norm + (2.0f*s - 1.0f) * rnorm * TQ_QJL_CORR_CPU;
        }
    }
}

size_t quantize_tq2_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_2 == 0);
    size_t row_size = (n_per_row / QK_PQ_TQ_2) * sizeof(block_tq2);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq2_1_ref(src + row * n_per_row, (block_tq2 *)((char *)dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}

/* TQ3_1 ------------------------------------------------------------ */
void quantize_row_tq3_1_ref(const float * GGML_RESTRICT x, block_tq3 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_3 == 0);
    const int nb = k / QK_PQ_TQ_3;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_PQ_TQ_3; j++) norm += x[i*QK_PQ_TQ_3 + j] * x[i*QK_PQ_TQ_3 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_PQ_TQ_3 / 4);
        memset(y[i].signs, 0, QK_PQ_TQ_3 / 8);
        memset(y[i].qjl, 0, QK_PQ_TQ_3 / 8);
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_tq3_1(const block_tq3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_3 == 0);
    const int nb = k / QK_PQ_TQ_3;
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);
        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        for (int j = 0; j < QK_PQ_TQ_3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1  = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            uint8_t s = (x[block].qjl[j/8] >> (j%8)) & 0x1;
            y[block * QK_PQ_TQ_3 + j] = CENTROIDS_3BIT[idx] * norm + (2.0f*s - 1.0f) * rnorm * TQ_QJL_CORR_CPU;
        }
    }
}

size_t quantize_tq3_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_3 == 0);
    size_t row_size = (n_per_row / QK_PQ_TQ_3) * sizeof(block_tq3);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq3_1_ref(src + row * n_per_row, (block_tq3 *)((char *)dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}

/* TQ4_1 ------------------------------------------------------------ */
void quantize_row_tq4_1_ref(const float * GGML_RESTRICT x, block_tq4 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_4 == 0);
    const int nb = k / QK_PQ_TQ_4;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_PQ_TQ_4; j++) norm += x[i*QK_PQ_TQ_4 + j] * x[i*QK_PQ_TQ_4 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].qs, 0, QK_PQ_TQ_4 / 2);
        memset(y[i].qjl, 0, QK_PQ_TQ_4 / 8);
    }
}

void dequantize_row_tq4_1(const block_tq4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    pq_tq_init_rotation();
    assert(k % QK_PQ_TQ_4 == 0);
    const int nb = k / QK_PQ_TQ_4;
    const int d = QK_PQ_TQ_4;
    static const float C4[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);
        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        float rotated[PQ_TQ_D128];
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i/2] >> ((i%2)*4)) & 0xF;
            uint8_t s = (x[block].qjl[i/8] >> (i%8)) & 0x1;
            rotated[i] = C4[idx] * norm + (2.0f*s - 1.0f) * rnorm * TQ_QJL_CORR_CPU;
        }
        float * dst2 = y + block * d;
        matvec(pq_tq_rotation_t, rotated, dst2, d);
    }
}

size_t quantize_tq4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_4 == 0);
    size_t row_size = (n_per_row / QK_PQ_TQ_4) * sizeof(block_tq4);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq4_1_ref(src + row * n_per_row, (block_tq4 *)((char *)dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}

void quantize_row_tq4_1_64_ref(const float * GGML_RESTRICT x, block_tq4_d64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_4_D64 == 0);
    const int nb = k / QK_PQ_TQ_4_D64;
    for (int block = 0; block < nb; ++block) {
        const float * src = x + block * QK_PQ_TQ_4_D64;
        float norm_sq = 0.0f;
        for (int i = 0; i < QK_PQ_TQ_4_D64; ++i) {
            norm_sq += src[i] * src[i];
        }
        const float norm = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

        uint8_t indices[QK_PQ_TQ_4_D64];
        float recon_norm_sq = 0.0f;
        memset(y[block].qs, 0, sizeof(y[block].qs));
        memset(y[block].qjl, 0, sizeof(y[block].qjl));
        for (int i = 0; i < QK_PQ_TQ_4_D64; ++i) {
            indices[i] = (uint8_t)nearest_centroid_4bit(src[i] * inv_norm);
            recon_norm_sq += CENTROIDS_4BIT_TABLE[indices[i]] * CENTROIDS_4BIT_TABLE[indices[i]];
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }

        const float recon_norm = sqrtf(recon_norm_sq);
        const float corrected_norm = recon_norm > 1e-10f ? norm / recon_norm : norm;
        float residual_sq = 0.0f;
        for (int i = 0; i < QK_PQ_TQ_4_D64; ++i) {
            const float recon = CENTROIDS_4BIT_TABLE[indices[i]] / (recon_norm + 1e-10f);
            const float residual = src[i] * inv_norm - recon;
            if (residual >= 0.0f) {
                y[block].qjl[i / 8] |= (uint8_t)(1u << (i % 8));
            }
            residual_sq += residual * residual;
        }

        y[block].norm  = GGML_FP32_TO_FP16(corrected_norm);
        y[block].rnorm = GGML_FP32_TO_FP16(norm * sqrtf(residual_sq * 128.0f / (float)QK_PQ_TQ_4_D64));
    }
}

void dequantize_row_tq4_1_64(const block_tq4_d64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_4_D64 == 0);
    const int nb = k / QK_PQ_TQ_4_D64;
    for (int block = 0; block < nb; ++block) {
        const float norm  = GGML_FP16_TO_FP32(x[block].norm);
        const float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        for (int i = 0; i < QK_PQ_TQ_4_D64; ++i) {
            const uint8_t qb = x[block].qs[i / 2];
            const uint8_t idx = (qb >> ((i % 2) * 4)) & 0xF;
            const uint8_t s = (x[block].qjl[i / 8] >> (i % 8)) & 0x1;
            y[block * QK_PQ_TQ_4_D64 + i] = CENTROIDS_4BIT_TABLE[idx] * norm + (2.0f * s - 1.0f) * rnorm * TQ_QJL_CORR_CPU;
        }
    }
}

size_t quantize_tq4_1_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_4_D64 == 0);

    const size_t row_size = (n_per_row / QK_PQ_TQ_4_D64) * sizeof(block_tq4_d64);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_tq4_1_64_ref(
            src + row * n_per_row,
            (block_tq4_d64 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
