/*
 * PQ/TQ KV-cache compression via PolarQuant + optional 1-bit QJL compensation.
 * Based on: arXiv 2504.19874.
 *
  * the public cache-type names are pq2/pq3/pq4 and tq2/tq3/tq4.
 */

#define _USE_MATH_DEFINES
#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

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
    assert(k % QK_PQ_TQ_2 == 0);
    const int nb = k / QK_PQ_TQ_2;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_PQ_TQ_2; j++) norm += x[i*QK_PQ_TQ_2 + j] * x[i*QK_PQ_TQ_2 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_PQ_TQ_2 / 4);
    }
}

void dequantize_row_pq2_0(const block_pq2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ_TQ_2 == 0);
    const int nb = k / QK_PQ_TQ_2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_PQ_TQ_2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_PQ_TQ_2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_pq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_2 == 0);

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
    assert(k % QK_PQ_TQ_3 == 0);
    const int nb = k / QK_PQ_TQ_3;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_PQ_TQ_3; j++) norm += x[i*QK_PQ_TQ_3 + j] * x[i*QK_PQ_TQ_3 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_PQ_TQ_3 / 4);
        memset(y[i].signs, 0, QK_PQ_TQ_3 / 8);
    }
}

void dequantize_row_pq3_0(const block_pq3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_PQ_TQ_3 == 0);
    const int nb = k / QK_PQ_TQ_3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_PQ_TQ_3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_PQ_TQ_3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_pq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_PQ_TQ_3 == 0);

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
    pq_tq_init_rotation();
    tq_init_qjl();

    assert(k % QK_PQ_TQ_4 == 0);
    const int nb = k / QK_PQ_TQ_4;
    const int d  = QK_PQ_TQ_4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[PQ_TQ_D128];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Rotate */
        float rotated[PQ_TQ_D128];
        matvec(pq_tq_rotation, normalized, rotated, d);

        /* Step 3: 4-bit quantization (16 centroids) */
        static const float CENTROIDS_4BIT[16] = {
            -0.173926f, -0.117195f, -0.089527f, -0.068756f,
            -0.051262f, -0.035597f, -0.020989f, -0.006938f,
             0.006938f,  0.020989f,  0.035597f,  0.051262f,
             0.068756f,  0.089527f,  0.117195f,  0.173926f
        };
        uint8_t indices[PQ_TQ_D128];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Norm correction */
        float recon_norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            recon_norm_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);

        /* Pack */
        y[block].norm  = GGML_FP32_TO_FP16(norm);

        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_pq4_0(const block_pq4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    pq_tq_init_rotation();

    assert(k % QK_PQ_TQ_4 == 0);
    const int nb = k / QK_PQ_TQ_4;
    const int d  = QK_PQ_TQ_4;

    /* 4-bit PolarQuant: nibble unpack → centroid → inverse rotate → scale */    
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float rotated[QK_PQ_TQ_4];
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            rotated[i] = CENTROIDS_4BIT[idx];
        }
        float * dst = y + block * d;
        matvec(pq_tq_rotation_t, rotated, dst, d);
        for (int i = 0; i < d; i++) dst[i] *= norm;
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
