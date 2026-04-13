#pragma once

/*
 * Fast Walsh-Hadamard Transform for PQ/TQ
 *
 * FWHT-128 via butterfly network: O(n log n) = 896 ops
 * vs dense matrix multiply: O(n²) = 16384 ops
 * Speedup: ~18.3×
 */

// ===== Fast Walsh-Hadamard Transform: 128-element butterfly network =====

__device__ __forceinline__ void pq_tq_fwht_128_forward(float* x) {
    // In-place FWHT via butterfly passes
    // Each pass: stride h = 1, 2, 4, 8, 16, 32, 64
    // Total: 7 stages × 128/2 butterflies = 448 butterflies
    // Plus normalization factor: 1/sqrt(128) ≈ 0.0883883

#pragma unroll(1)
    for (int h = 1; h < 128; h *= 2) {
#pragma unroll(1)
        for (int i = 0; i < 128; i += 2 * h) {
#pragma unroll(4)
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;      // Addition
                x[j + h] = a - b;      // Subtraction
            }
        }
    }

    // Normalization: scale by 1/sqrt(128)
    const float norm = 0.08838834764831845f;  // 1/sqrt(128)
#pragma unroll(8)
    for (int i = 0; i < 128; i++) {
        x[i] *= norm;
    }
}

// ===== Rotation functions: apply FWHT with sign preprocessing =====

__device__ __forceinline__ void pq_tq_rotate_forward(
    float* x,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2) {
    // Forward rotation: signs1 → FWHT → signs2

    // Step 1: Pre-multiply by signs1
#pragma unroll(8)
    for (int i = 0; i < 128; i++) {
        x[i] *= signs1[i];
    }

    // Step 2: Execute FWHT
    pq_tq_fwht_128_forward(x);

    // Step 3: Post-multiply by signs2
#pragma unroll(8)
    for (int i = 0; i < 128; i++) {
        x[i] *= signs2[i];
    }
}

__device__ __forceinline__ void pq_tq_rotate_inverse(
    float* x,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2) {
    // Inverse rotation: same as forward (FWHT is self-inverse)
    // But sign application order is reversed

    // Step 1: Pre-multiply by signs2
#pragma unroll(8)
    for (int i = 0; i < 128; i++) {
        x[i] *= signs2[i];
    }

    // Step 2: Execute FWHT (self-inverse)
    pq_tq_fwht_128_forward(x);

    // Step 3: Post-multiply by signs1
#pragma unroll(8)
    for (int i = 0; i < 128; i++) {
        x[i] *= signs1[i];
    }
}

// ============================================================================
// Cooperative (multi-thread) WHT for use in CUDA kernels with shared memory.
// 128 threads per block. For D=64: 64 active. For D=128: all 128. For D=256: each handles 2 elements.
// sh[] must be __shared__ float[D], pre-loaded with normalized data.
// After return, sh[] contains WHT-rotated data (SIGNS applied, scaled).
// NOTE: These must NOT use __restrict__ on the sh pointer (NVCC aliasing bug on sm_120a).
// ============================================================================

// Forward rotation: signs1 → FWHT butterfly → scale+signs2
template<int D>
static __device__ __forceinline__ void pq_tq_coop_wht_forward(float * sh, const int tid) {
    static_assert(D == 64 || D == 128 || D == 256, "Unsupported WHT dimension");

    if constexpr (D == 64) {
        if (tid < 64) {
            sh[tid] *= PQ_TQ_WHT_SIGNS1_64[tid];
        }
        if (tid >= 64) sh[tid] = 0.0f;
        __syncthreads();

#pragma unroll
        for (int h = 1; h < 64; h <<= 1) {
            const float a = sh[tid];
            const float b = sh[tid ^ h];
            __syncthreads();
            sh[tid] = ((tid & h) == 0) ? (a + b) : (b - a);
            __syncthreads();
        }

        constexpr float inv_sqrt64 = 0.125f; // 1/sqrt(64)
        if (tid < 64) {
            sh[tid] *= inv_sqrt64 * PQ_TQ_WHT_SIGNS2_64[tid];
        }
        __syncthreads();

    } else if constexpr (D == 128) {
        sh[tid] *= PQ_TQ_WHT_SIGNS1[tid];
        __syncthreads();

#pragma unroll
        for (int h = 1; h < 128; h <<= 1) {
            const float a = sh[tid];
            const float b = sh[tid ^ h];
            __syncthreads();
            sh[tid] = ((tid & h) == 0) ? (a + b) : (b - a);
            __syncthreads();
        }

        constexpr float inv_sqrt128 = 0.08838834764831845f; // 1/sqrt(128)
        sh[tid] *= inv_sqrt128 * PQ_TQ_WHT_SIGNS2[tid];
        __syncthreads();

    } else if constexpr (D == 256) {
        sh[tid]       *= PQ_TQ_WHT_SIGNS1_256[tid];
        sh[tid + 128] *= PQ_TQ_WHT_SIGNS1_256[tid + 128];
        __syncthreads();

#pragma unroll
        for (int h = 1; h < 128; h <<= 1) {
            const float a0 = sh[tid],       b0 = sh[tid ^ h];
            const float a1 = sh[tid + 128], b1 = sh[(tid ^ h) + 128];
            __syncthreads();
            sh[tid]       = ((tid & h) == 0) ? (a0 + b0) : (b0 - a0);
            sh[tid + 128] = ((tid & h) == 0) ? (a1 + b1) : (b1 - a1);
            __syncthreads();
        }

        {
            const float lo = sh[tid], hi = sh[tid + 128];
            __syncthreads();
            sh[tid]       = lo + hi;
            sh[tid + 128] = lo - hi;
            __syncthreads();
        }

        constexpr float inv_sqrt256 = 0.0625f; // 1/sqrt(256) = 1/16
        sh[tid]       *= inv_sqrt256 * PQ_TQ_WHT_SIGNS2_256[tid];
        sh[tid + 128] *= inv_sqrt256 * PQ_TQ_WHT_SIGNS2_256[tid + 128];
        __syncthreads();
    }
}

// Inverse rotation: signs2 → FWHT butterfly → scale+signs1
template<int D>
static __device__ __forceinline__ void pq_tq_coop_wht_inverse(float * sh, const int tid) {
    static_assert(D == 64 || D == 128 || D == 256, "Unsupported WHT dimension");

    if constexpr (D == 64) {
        if (tid < 64) {
            sh[tid] *= PQ_TQ_WHT_SIGNS2_64[tid];
        }
        if (tid >= 64) sh[tid] = 0.0f;
        __syncthreads();

#pragma unroll
        for (int h = 1; h < 64; h <<= 1) {
            const float a = sh[tid];
            const float b = sh[tid ^ h];
            __syncthreads();
            sh[tid] = ((tid & h) == 0) ? (a + b) : (b - a);
            __syncthreads();
        }

        constexpr float inv_sqrt64 = 0.125f; // 1/sqrt(64)
        if (tid < 64) {
            sh[tid] *= inv_sqrt64 * PQ_TQ_WHT_SIGNS1_64[tid];
        }
        __syncthreads();

    } else if constexpr (D == 128) {
        sh[tid] *= PQ_TQ_WHT_SIGNS2[tid];
        __syncthreads();

#pragma unroll
        for (int h = 1; h < 128; h <<= 1) {
            const float a = sh[tid];
            const float b = sh[tid ^ h];
            __syncthreads();
            sh[tid] = ((tid & h) == 0) ? (a + b) : (b - a);
            __syncthreads();
        }

        constexpr float inv_sqrt128 = 0.08838834764831845f;
        sh[tid] *= inv_sqrt128 * PQ_TQ_WHT_SIGNS1[tid];
        __syncthreads();

    } else if constexpr (D == 256) {
        sh[tid]       *= PQ_TQ_WHT_SIGNS2_256[tid];
        sh[tid + 128] *= PQ_TQ_WHT_SIGNS2_256[tid + 128];
        __syncthreads();

#pragma unroll
        for (int h = 1; h < 128; h <<= 1) {
            const float a0 = sh[tid],       b0 = sh[tid ^ h];
            const float a1 = sh[tid + 128], b1 = sh[(tid ^ h) + 128];
            __syncthreads();
            sh[tid]       = ((tid & h) == 0) ? (a0 + b0) : (b0 - a0);
            sh[tid + 128] = ((tid & h) == 0) ? (a1 + b1) : (b1 - a1);
            __syncthreads();
        }

        {
            const float lo = sh[tid], hi = sh[tid + 128];
            __syncthreads();
            sh[tid]       = lo + hi;
            sh[tid + 128] = lo - hi;
            __syncthreads();
        }

        constexpr float inv_sqrt256 = 0.0625f;
        sh[tid]       *= inv_sqrt256 * PQ_TQ_WHT_SIGNS1_256[tid];
        sh[tid + 128] *= inv_sqrt256 * PQ_TQ_WHT_SIGNS1_256[tid + 128];
        __syncthreads();
    }
}

// Legacy aliases for backward compatibility (D=128 only)
static __device__ __forceinline__ void pq_tq_coop_wht_forward_128(float * sh, const int tid) {
    pq_tq_coop_wht_forward<128>(sh, tid);
}

static __device__ __forceinline__ void pq_tq_coop_wht_inverse_128(float * sh, const int tid) {
    pq_tq_coop_wht_inverse<128>(sh, tid);
}

// ===== Test/Debug: Matrix-vector multiplication (fallback for testing) =====

__device__ __forceinline__ void pq_tq_matvec_multiply(
    const float* __restrict__ M,  // 128×128 matrix (row-major)
    const float* __restrict__ x,  // 128-element input
    float* __restrict__ y) {      // 128-element output

    // y = M @ x (matrix is 128×128 row-major)
    // This is O(n²) - used only for validation, not production

#pragma unroll(1)
    for (int i = 0; i < 128; i++) {
        float sum = 0.0f;
#pragma unroll(4)
        for (int j = 0; j < 128; j++) {
            sum += M[i * 128 + j] * x[j];
        }
        y[i] = sum;
    }
}
