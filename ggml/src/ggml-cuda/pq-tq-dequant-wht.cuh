#pragma once

// WHT-fused dequantization for PQ/TQ types.
// Dequantizes pq/tq blocks AND applies cooperative inverse WHT in a single kernel.
// Output is in NORMAL domain (not rotated), eliminating the need for separate
// Q forward WHT and output inverse WHT graph ops.
//
// Each CUDA block processes one 128-element WHT group:
//   1. 128 threads each dequant 1 element from pq/tq block(s)
//   2. Cooperative in-place FWHT butterfly (7 stages, 14 __syncthreads)
//   3. Write output as fp16

#include "common.cuh"
#include "convert.cuh"
#include "pq-tq-common.cuh"
#include "pq-tq-fwht.cuh"

// ============================================================================
// Single-element dequantize device functions (no WHT, just centroid decode)
// ============================================================================

static __device__ __forceinline__ float pq_dequant_elem_2_0(const void * vx, int64_t global_elem) {
    const block_pq2 * x = (const block_pq2 *)vx;
    const int ib = global_elem / QK_PQ_TQ_2;
    const int il = global_elem % QK_PQ_TQ_2;
    const float norm = __half2float(x[ib].norm);
    const int qs_byte = il / 4;
    const int qs_shift = (il % 4) * 2;
    const uint8_t q = (x[ib].qs[qs_byte] >> qs_shift) & 0x3;
    return PQ_TQ_CENTROIDS_2BIT[q] * norm;
}

static __device__ __forceinline__ float pq_dequant_elem_3_0(const void * vx, int64_t global_elem) {
    const block_pq3 * x = (const block_pq3 *)vx;
    const int ib = global_elem / QK_PQ_TQ_3;
    const int il = global_elem % QK_PQ_TQ_3;
    const float norm = __half2float(x[ib].norm);
    const int qs_byte = il / 4;
    const int qs_shift = (il % 4) * 2;
    const int s_byte = il / 8;
    const int s_shift = il % 8;
    const uint8_t q = (x[ib].qs[qs_byte] >> qs_shift) & 0x3;
    const uint8_t s = (x[ib].signs[s_byte] >> s_shift) & 0x1;
    return PQ_TQ_CENTROIDS_3BIT[q | (s << 2)] * norm;
}

template <typename block_pq4, int block_qk>
static __device__ __forceinline__ float pq_dequant_elem_4_0_impl(const void * vx, int64_t global_elem) {
    const block_pq4 * x = (const block_pq4 *)vx;
    const int ib = global_elem / block_qk;
    const int il = global_elem % block_qk;
    const float norm = __half2float(x[ib].norm);
    const uint8_t qb = x[ib].qs[il / 2];
    return (il & 1) ? (PQ_TQ_CENTROIDS_4BIT[qb >> 4] * norm)
                     : (PQ_TQ_CENTROIDS_4BIT[qb & 0xF] * norm);
}

static __device__ __forceinline__ float pq_dequant_elem_4_0(const void * vx, int64_t global_elem) {
    return pq_dequant_elem_4_0_impl<block_pq4, QK_PQ_TQ_4>(vx, global_elem);
}

static __device__ __forceinline__ float pq_dequant_elem_4_0_64(const void * vx, int64_t global_elem) {
    return pq_dequant_elem_4_0_impl<block_pq4_d64, QK_PQ_TQ_4_D64>(vx, global_elem);
}

static constexpr float TQ_DEQUANT_QJL_CORR = 0.0705348f; // sqrt(2/(pi*128))

static __device__ __forceinline__ float tq_dequant_elem_2_1(const void * vx, int64_t global_elem) {
    const block_tq2 * x = (const block_tq2 *)vx;
    const int ib = global_elem / QK_PQ_TQ_2;
    const int il = global_elem % QK_PQ_TQ_2;
    const float norm  = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const int qs_byte = il / 4;
    const int qs_shift = (il % 4) * 2;
    const uint8_t q = (x[ib].qs[qs_byte] >> qs_shift) & 0x3;
    const int qjl_byte = il / 8;
    const uint8_t j = (x[ib].qjl[qjl_byte] >> (il % 8)) & 1u;
    return PQ_TQ_CENTROIDS_2BIT[q] * norm + (2.0f * j - 1.0f) * rnorm * TQ_DEQUANT_QJL_CORR;
}

static __device__ __forceinline__ float tq_dequant_elem_3_1(const void * vx, int64_t global_elem) {
    const block_tq3 * x = (const block_tq3 *)vx;
    const int ib = global_elem / QK_PQ_TQ_3;
    const int il = global_elem % QK_PQ_TQ_3;
    const float norm  = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const int qs_byte = il / 4;
    const int qs_shift = (il % 4) * 2;
    const int s_byte = il / 8;
    const int s_shift = il % 8;
    const uint8_t q = (x[ib].qs[qs_byte] >> qs_shift) & 0x3;
    const uint8_t s = (x[ib].signs[s_byte] >> s_shift) & 0x1;
    const uint8_t j = (x[ib].qjl[il / 8] >> (il % 8)) & 1u;
    return PQ_TQ_CENTROIDS_3BIT[q | (s << 2)] * norm + (2.0f * j - 1.0f) * rnorm * TQ_DEQUANT_QJL_CORR;
}

template <typename block_tq4, int block_qk>
static __device__ __forceinline__ float tq_dequant_elem_4_1_impl(const void * vx, int64_t global_elem) {
    const block_tq4 * x = (const block_tq4 *)vx;
    const int ib = global_elem / block_qk;
    const int il = global_elem % block_qk;
    const float norm  = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const uint8_t qb = x[ib].qs[il / 2];
    const float centroid = (il & 1) ? PQ_TQ_CENTROIDS_4BIT[qb >> 4]
                                    : PQ_TQ_CENTROIDS_4BIT[qb & 0xF];
    const int qjl_byte = il / 8;
    const uint8_t j = (x[ib].qjl[qjl_byte] >> (il % 8)) & 1u;
    return centroid * norm + (2.0f * j - 1.0f) * rnorm * TQ_DEQUANT_QJL_CORR;
}

static __device__ __forceinline__ float tq_dequant_elem_4_1(const void * vx, int64_t global_elem) {
    return tq_dequant_elem_4_1_impl<block_tq4, QK_PQ_TQ_4>(vx, global_elem);
}

static __device__ __forceinline__ float tq_dequant_elem_4_1_64(const void * vx, int64_t global_elem) {
    return tq_dequant_elem_4_1_impl<block_tq4_d64, QK_PQ_TQ_4_D64>(vx, global_elem);
}

// ============================================================================
// Type dispatch tag
// ============================================================================
enum class PqTqTypeTag { T2_0, T3_0, T4_0, T2_1, T3_1, T4_1, T4_0_64, T4_1_64 };

template <PqTqTypeTag TT>
static __device__ __forceinline__ float pq_tq_dequant_elem(const void * vx, int64_t global_elem);

template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::T2_0>(const void * vx, int64_t e) { return pq_dequant_elem_2_0(vx, e); }
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::T3_0>(const void * vx, int64_t e) { return pq_dequant_elem_3_0(vx, e); }
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::T4_0>(const void * vx, int64_t e) { return pq_dequant_elem_4_0(vx, e); }
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::T2_1>(const void * vx, int64_t e) { return tq_dequant_elem_2_1(vx, e); }
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::T3_1>(const void * vx, int64_t e) { return tq_dequant_elem_3_1(vx, e); }
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::T4_1>(const void * vx, int64_t e) { return tq_dequant_elem_4_1(vx, e); }
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::T4_0_64>(const void * vx, int64_t e) { return pq_dequant_elem_4_0_64(vx, e); }
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::T4_1_64>(const void * vx, int64_t e) { return tq_dequant_elem_4_1_64(vx, e); }

// ============================================================================
// Kernel: dequantize + cooperative inverse WHT → fp16 output
// ============================================================================
// Each block = 128 threads = one WHT group (wht_dim = 64, 128, or 256).
// D-aware: for D=64 only 64 threads active, for D=256 each handles 2 elements.
// Elements within a WHT group are contiguous in memory.

template <PqTqTypeTag TT>
static __global__ void k_dequant_pq_tq_unrotated_fp16(
        const void * __restrict__ vx, half * __restrict__ y,
        const int64_t k, const int wht_dim) {

    __shared__ float sh[256];  // max D=256

    const int tid = threadIdx.x;       // 0..127
    const int64_t gid = blockIdx.x;    // WHT group index
    const int64_t base = gid * wht_dim;

    // Phase 1: Dequantize into shared memory (D-aware)
    if (wht_dim <= 64) {
        if (tid < wht_dim) sh[tid] = pq_tq_dequant_elem<TT>(vx, base + tid);
    } else if (wht_dim <= 128) {
        sh[tid] = pq_tq_dequant_elem<TT>(vx, base + tid);
    } else {
        sh[tid]       = pq_tq_dequant_elem<TT>(vx, base + tid);
        sh[tid + 128] = pq_tq_dequant_elem<TT>(vx, base + tid + 128);
    }
    __syncthreads();

    // Phase 2: Cooperative inverse WHT (SIGNS + butterfly + scale + SIGNS)
    if      (wht_dim == 64)  pq_tq_coop_wht_inverse<64>(sh, tid);
    else if (wht_dim == 256) pq_tq_coop_wht_inverse<256>(sh, tid);
    else                     pq_tq_coop_wht_inverse<128>(sh, tid);

    // Phase 3: Write output as fp16 (D-aware)
    if (wht_dim <= 64) {
        if (tid < wht_dim) y[base + tid] = __float2half(sh[tid]);
    } else if (wht_dim <= 128) {
        y[base + tid] = __float2half(sh[tid]);
    } else {
        y[base + tid]       = __float2half(sh[tid]);
        y[base + tid + 128] = __float2half(sh[tid + 128]);
    }
}

// ============================================================================
// Host wrapper: D-aware dispatch (called directly with wht_dim, not via function pointer)
// ============================================================================

template <PqTqTypeTag TT>
static void dequant_pq_tq_unrotated_fp16_wht(
        const void * vx, half * y, int64_t k, int wht_dim, cudaStream_t stream) {
    GGML_ASSERT(wht_dim == 64 || wht_dim == 128 || wht_dim == 256);
    GGML_ASSERT(k % wht_dim == 0);
    const int64_t n_groups = k / wht_dim;
    k_dequant_pq_tq_unrotated_fp16<TT><<<n_groups, 128, 0, stream>>>(vx, y, k, wht_dim);
}

// Legacy wrapper matching to_fp16_cuda_t signature (assumes D=128)
template <PqTqTypeTag TT>
static void dequant_pq_tq_unrotated_fp16(
        const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequant_pq_tq_unrotated_fp16_wht<TT>(vx, y, k, 128, stream);
}

// ============================================================================
// D-aware dispatcher: calls kernel with correct wht_dim for pq/tq types
// ============================================================================

static void dequant_pq_tq_unrotated_fp16_dispatch(
        ggml_type type, const void * vx, half * y, int64_t k, int wht_dim, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_PQ2_0: dequant_pq_tq_unrotated_fp16_wht<PqTqTypeTag::T2_0>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ3_0: dequant_pq_tq_unrotated_fp16_wht<PqTqTypeTag::T3_0>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ4_0: dequant_pq_tq_unrotated_fp16_wht<PqTqTypeTag::T4_0>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ2_1: dequant_pq_tq_unrotated_fp16_wht<PqTqTypeTag::T2_1>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ3_1: dequant_pq_tq_unrotated_fp16_wht<PqTqTypeTag::T3_1>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ4_1: dequant_pq_tq_unrotated_fp16_wht<PqTqTypeTag::T4_1>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ4_0_64: dequant_pq_tq_unrotated_fp16_wht<PqTqTypeTag::T4_0_64>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ4_1_64: dequant_pq_tq_unrotated_fp16_wht<PqTqTypeTag::T4_1_64>(vx, y, k, wht_dim, stream); break;
        default: GGML_ABORT("unsupported pq/tq type for WHT-fused dequant");
    }
}

// ============================================================================
// Dispatcher: returns WHT-fused dequant function pointer for pq/tq types
// Returns nullptr for non-pq/tq types.
// ============================================================================

static to_fp16_cuda_t ggml_get_to_fp16_pq_tq_unrotated_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_PQ2_0: return dequant_pq_tq_unrotated_fp16<PqTqTypeTag::T2_0>;
        case GGML_TYPE_PQ3_0: return dequant_pq_tq_unrotated_fp16<PqTqTypeTag::T3_0>;
        case GGML_TYPE_PQ4_0: return dequant_pq_tq_unrotated_fp16<PqTqTypeTag::T4_0>;
        case GGML_TYPE_TQ2_1: return dequant_pq_tq_unrotated_fp16<PqTqTypeTag::T2_1>;
        case GGML_TYPE_TQ3_1: return dequant_pq_tq_unrotated_fp16<PqTqTypeTag::T3_1>;
        case GGML_TYPE_TQ4_1: return dequant_pq_tq_unrotated_fp16<PqTqTypeTag::T4_1>;
        case GGML_TYPE_PQ4_0_64: return dequant_pq_tq_unrotated_fp16<PqTqTypeTag::T4_0_64>;
        case GGML_TYPE_TQ4_1_64: return dequant_pq_tq_unrotated_fp16<PqTqTypeTag::T4_1_64>;
        default: return nullptr;
    }
}

// Helper: check if a ggml_type is a pq/tq type
static inline bool ggml_is_pq_tq_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_PQ3_0:
        case GGML_TYPE_PQ4_0:
        case GGML_TYPE_TQ2_1:
        case GGML_TYPE_TQ3_1:
        case GGML_TYPE_TQ4_1:
        case GGML_TYPE_PQ4_0_64:
        case GGML_TYPE_TQ4_1_64:
            return true;
        default:
            return false;
    }
}
