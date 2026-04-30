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
#include "ggml-pqk-common.h"
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

template <typename block_pqk>
static __device__ __forceinline__ float pqk_dequant_scale(const block_pqk * x, const int ib, const int subblock) {
    const int band = subblock / GGML_PQK_SUBBLOCKS_PER_BAND;
    const float master = __half2float(x[ib].d[band]);
    return ggml_pqk_decode_local_scale(master, ggml_pqk_scale_get(x[ib].scales, subblock));
}

static __device__ __forceinline__ float pq2_k_dequant_scale(const block_pq2_K * x, const int ib, const int subblock) {
    const int band = subblock / GGML_PQ2_K_SUBBLOCKS_PER_BAND;
    const float master = __half2float(x[ib].d[band]);
    return ggml_pq2_k_decode_local_scale(master, ggml_pq2_k_scale_get(x[ib].scales, subblock));
}

static __device__ __forceinline__ float pq3_k_dequant_scale(const block_pq3_K * x, const int ib, const int subblock) {
    const int band = subblock / GGML_PQ3_K_SUBBLOCKS_PER_BAND;
    const float master = __half2float(x[ib].d[band]);
    return ggml_pq3_k_decode_local_scale(master, ggml_pq3_k_scale_get(x[ib].scales, subblock));
}

static __device__ __forceinline__ float pq4_k_dequant_scale(const block_pq4_K * x, const int ib, const int subblock) {
    const int band = subblock / GGML_PQ4_K_SUBBLOCKS_PER_BAND;
    const float master = __half2float(x[ib].d[band]);
    return ggml_pq4_k_decode_local_scale(master, ggml_pq4_k_scale_get(x[ib].scales, subblock));
}

static __device__ __forceinline__ float pq_dequant_elem_2_k(const void * vx, int64_t global_elem) {
    const block_pq2_K * x = (const block_pq2_K *) vx;
    const int ib = global_elem / QK_K;
    const int il = global_elem % QK_K;
    const int subblock = il / GGML_PQ2_K_SUBBLOCK_SIZE;
    const float scale = pq2_k_dequant_scale(x, ib, subblock);
    const uint8_t q = (x[ib].qs[il / 4] >> (2 * (il & 3))) & 0x3u;
    return ggml_pqk_centroid_2bit(q) * scale;
}

static __device__ __forceinline__ float pq_dequant_elem_3_k(const void * vx, int64_t global_elem) {
    const block_pq3_K * x = (const block_pq3_K *) vx;
    const int ib = global_elem / QK_K;
    const int il = global_elem % QK_K;
    const int subblock = il / GGML_PQ3_K_SUBBLOCK_SIZE;
    const float scale = pq3_k_dequant_scale(x, ib, subblock);
    const uint8_t ql = (x[ib].qs[il / 4] >> (2 * (il & 3))) & 0x3u;
    const uint8_t qh = (x[ib].hmask[il / 8] >> (il & 7)) & 0x1u;
    return ggml_pqk_centroid_3bit((uint8_t)(ql | (qh << 2))) * scale;
}

static __device__ __forceinline__ float pq_dequant_elem_4_k(const void * vx, int64_t global_elem) {
    const block_pq4_K * x = (const block_pq4_K *) vx;
    const int ib = global_elem / QK_K;
    const int il = global_elem % QK_K;
    const float scale = pq4_k_dequant_scale(x, ib, il / GGML_PQ4_K_SUBBLOCK_SIZE);
    const uint8_t q = (x[ib].qs[il / 2] >> (4 * (il & 1))) & 0xFu;
    return ggml_pqk_centroid_4bit(q) * scale;
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
enum class PqTqTypeTag { T2_0, T3_0, T4_0, T2_1, T3_1, T4_1, T4_0_64, T4_1_64, P2_K, P3_K, P4_K };

// ============================================================================
// Pair dequantize device functions (2 consecutive elements at once).
// Used by the WHT-fused fp16/fp32 conversion path to cut scalar centroid decode
// traffic roughly in half before entering the cooperative inverse WHT.
// ============================================================================

template <PqTqTypeTag TT>
static __device__ __forceinline__ float2 pq_tq_dequant_pair(const void * vx, int64_t global_pair);

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::T2_0>(const void * vx, int64_t global_pair) {
    const block_pq2 * x = (const block_pq2 *) vx;
    constexpr int pairs_per_block = QK_PQ_TQ_2 / 2;
    const int ib = global_pair / pairs_per_block;
    const int ip = global_pair % pairs_per_block;
    const float norm = __half2float(x[ib].norm);
    const uint8_t qb = x[ib].qs[ip / 2];
    const uint8_t q4 = (ip & 1) ? (qb >> 4) : (qb & 0x0F);
    const float2 pair = pq_tq_centroid_pair_2bit(q4);
    return make_float2(pair.x * norm, pair.y * norm);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::T3_0>(const void * vx, int64_t global_pair) {
    const block_pq3 * x = (const block_pq3 *) vx;
    constexpr int pairs_per_block = QK_PQ_TQ_3 / 2;
    const int ib = global_pair / pairs_per_block;
    const int ip = global_pair % pairs_per_block;
    const float norm = __half2float(x[ib].norm);
    const uint8_t qb = x[ib].qs[ip / 2];
    const uint8_t q4 = (ip & 1) ? (qb >> 4) : (qb & 0x0F);
    const uint8_t sb = x[ib].signs[ip / 4];
    const uint8_t sign2 = (sb >> ((ip & 0x3) * 2)) & 0x03u;
    const float2 pair = PQ_TQ_PAIR_LUT_3BIT[(sign2 << 4) | q4];
    return make_float2(pair.x * norm, pair.y * norm);
}

template <typename block_pq4, int block_qk>
static __device__ __forceinline__ float2 pq_dequant_pair_4_0_impl(const void * vx, int64_t global_pair) {
    const block_pq4 * x = (const block_pq4 *) vx;
    constexpr int pairs_per_block = block_qk / 2;
    const int ib = global_pair / pairs_per_block;
    const int ip = global_pair % pairs_per_block;
    const float norm = __half2float(x[ib].norm);
    const float2 pair = pq_tq_centroid_pair_4bit(x[ib].qs[ip]);
    return make_float2(pair.x * norm, pair.y * norm);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::T4_0>(const void * vx, int64_t global_pair) {
    return pq_dequant_pair_4_0_impl<block_pq4, QK_PQ_TQ_4>(vx, global_pair);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::T4_0_64>(const void * vx, int64_t global_pair) {
    return pq_dequant_pair_4_0_impl<block_pq4_d64, QK_PQ_TQ_4_D64>(vx, global_pair);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::T2_1>(const void * vx, int64_t global_pair) {
    const block_tq2 * x = (const block_tq2 *) vx;
    constexpr int pairs_per_block = QK_PQ_TQ_2 / 2;
    const int ib = global_pair / pairs_per_block;
    const int ip = global_pair % pairs_per_block;
    const float norm = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const uint8_t qb = x[ib].qs[ip / 2];
    const uint8_t q4 = (ip & 1) ? (qb >> 4) : (qb & 0x0F);
    const uint8_t jb = x[ib].qjl[ip / 4];
    const float2 pair = pq_tq_centroid_pair_2bit(q4);
    const float2 corr = tq_qjl_correction_pair(jb, (ip & 0x3) * 2, rnorm);
    return make_float2(pair.x * norm + corr.x, pair.y * norm + corr.y);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::T3_1>(const void * vx, int64_t global_pair) {
    const block_tq3 * x = (const block_tq3 *) vx;
    constexpr int pairs_per_block = QK_PQ_TQ_3 / 2;
    const int ib = global_pair / pairs_per_block;
    const int ip = global_pair % pairs_per_block;
    const float norm = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const uint8_t qb = x[ib].qs[ip / 2];
    const uint8_t q4 = (ip & 1) ? (qb >> 4) : (qb & 0x0F);
    const uint8_t sb = x[ib].signs[ip / 4];
    const uint8_t sign2 = (sb >> ((ip & 0x3) * 2)) & 0x03u;
    const uint8_t jb = x[ib].qjl[ip / 4];
    const float2 pair = PQ_TQ_PAIR_LUT_3BIT[(sign2 << 4) | q4];
    const float2 corr = tq_qjl_correction_pair(jb, (ip & 0x3) * 2, rnorm);
    return make_float2(pair.x * norm + corr.x, pair.y * norm + corr.y);
}

template <typename block_tq4, int block_qk>
static __device__ __forceinline__ float2 tq_dequant_pair_4_1_impl(const void * vx, int64_t global_pair) {
    const block_tq4 * x = (const block_tq4 *) vx;
    constexpr int pairs_per_block = block_qk / 2;
    const int ib = global_pair / pairs_per_block;
    const int ip = global_pair % pairs_per_block;
    const float norm = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const float2 pair = pq_tq_centroid_pair_4bit(x[ib].qs[ip]);
    const uint8_t jb = x[ib].qjl[ip / 4];
    const float2 corr = tq_qjl_correction_pair(jb, (ip & 0x3) * 2, rnorm);
    return make_float2(pair.x * norm + corr.x, pair.y * norm + corr.y);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::T4_1>(const void * vx, int64_t global_pair) {
    return tq_dequant_pair_4_1_impl<block_tq4, QK_PQ_TQ_4>(vx, global_pair);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::T4_1_64>(const void * vx, int64_t global_pair) {
    return tq_dequant_pair_4_1_impl<block_tq4_d64, QK_PQ_TQ_4_D64>(vx, global_pair);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::P2_K>(const void * vx, int64_t global_pair) {
    const block_pq2_K * x = (const block_pq2_K *) vx;
    constexpr int pairs_per_block = QK_K / 2;
    const int ib = global_pair / pairs_per_block;
    const int ip = global_pair % pairs_per_block;
    const int il = 2 * ip;
    const int subblock = il / GGML_PQ2_K_SUBBLOCK_SIZE;
    const float scale = pq2_k_dequant_scale(x, ib, subblock);
    const uint8_t qb = x[ib].qs[il / 4];
    const int shift = 2 * (il & 3);
    const uint8_t q0 = (qb >> shift) & 0x3u;
    const uint8_t q1 = (qb >> (shift + 2)) & 0x3u;
    return make_float2(ggml_pqk_centroid_2bit(q0) * scale, ggml_pqk_centroid_2bit(q1) * scale);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::P3_K>(const void * vx, int64_t global_pair) {
    const block_pq3_K * x = (const block_pq3_K *) vx;
    constexpr int pairs_per_block = QK_K / 2;
    const int ib = global_pair / pairs_per_block;
    const int ip = global_pair % pairs_per_block;
    const int il = 2 * ip;
    const int subblock = il / GGML_PQ3_K_SUBBLOCK_SIZE;
    const float scale = pq3_k_dequant_scale(x, ib, subblock);
    const uint8_t qb = x[ib].qs[il / 4];
    const int shift = 2 * (il & 3);
    const uint8_t q0 = ((qb >> shift) & 0x3u) | (((x[ib].hmask[il / 8] >> (il & 7)) & 0x1u) << 2);
    const int il1 = il + 1;
    const uint8_t q1 = ((qb >> (shift + 2)) & 0x3u) | (((x[ib].hmask[il1 / 8] >> (il1 & 7)) & 0x1u) << 2);
    return make_float2(ggml_pqk_centroid_3bit(q0) * scale, ggml_pqk_centroid_3bit(q1) * scale);
}

template <>
__device__ __forceinline__ float2 pq_tq_dequant_pair<PqTqTypeTag::P4_K>(const void * vx, int64_t global_pair) {
    const block_pq4_K * x = (const block_pq4_K *) vx;
    constexpr int pairs_per_block = QK_K / 2;
    const int ib = global_pair / pairs_per_block;
    const int ip = global_pair % pairs_per_block;
    const int il = 2 * ip;
    const float scale = pq4_k_dequant_scale(x, ib, il / GGML_PQ4_K_SUBBLOCK_SIZE);
    const uint8_t qb = x[ib].qs[ip];
    return make_float2(ggml_pqk_centroid_4bit(qb & 0xFu) * scale, ggml_pqk_centroid_4bit(qb >> 4) * scale);
}

static __device__ __forceinline__ float pq_tq_warp_fwht_step(const float x, const int lane, const int h) {
    const float y = __shfl_xor_sync(0xffffffff, x, h);
    return ((lane & h) == 0) ? (x + y) : (y - x);
}

static __device__ __forceinline__ float pq_tq_warp_fwht_32(float x, const int lane) {
    x = pq_tq_warp_fwht_step(x, lane, 1);
    x = pq_tq_warp_fwht_step(x, lane, 2);
    x = pq_tq_warp_fwht_step(x, lane, 4);
    x = pq_tq_warp_fwht_step(x, lane, 8);
    x = pq_tq_warp_fwht_step(x, lane, 16);
    return x;
}

template <typename dst_t>
static __device__ __forceinline__ void pq_tq_inverse_wht_store_64(
        float * sh, const int tid, dst_t * __restrict__ y, const int64_t base) {
    if (tid < 64) {
        float x = sh[tid] * PQ_TQ_WHT_SIGNS2_64[tid];
        x = pq_tq_warp_fwht_32(x, tid & 31);
        sh[tid] = x;
    }
    __syncthreads();

    float a = 0.0f;
    float b = 0.0f;
    if (tid < 64) {
        a = sh[tid];
        b = sh[tid ^ 32];
    }
    __syncthreads();

    if (tid < 64) {
        constexpr float inv_sqrt64 = 0.125f;
        const float x = ((tid & 32) == 0) ? (a + b) : (b - a);
        y[base + tid] = ggml_cuda_cast<dst_t>(x * inv_sqrt64 * PQ_TQ_WHT_SIGNS1_64[tid]);
    }
}

template <typename dst_t>
static __device__ __forceinline__ void pq_tq_inverse_wht_store_128(
        float * sh, const int tid, dst_t * __restrict__ y, const int64_t base) {
    float x = sh[tid] * PQ_TQ_WHT_SIGNS2[tid];
    x = pq_tq_warp_fwht_32(x, tid & 31);
    sh[tid] = x;
    __syncthreads();

    float a = sh[tid];
    float b = sh[tid ^ 32];
    __syncthreads();
    sh[tid] = ((tid & 32) == 0) ? (a + b) : (b - a);
    __syncthreads();

    a = sh[tid];
    b = sh[tid ^ 64];
    constexpr float inv_sqrt128 = 0.08838834764831845f;
    x = ((tid & 64) == 0) ? (a + b) : (b - a);
    y[base + tid] = ggml_cuda_cast<dst_t>(x * inv_sqrt128 * PQ_TQ_WHT_SIGNS1[tid]);
}

template <typename dst_t>
static __device__ __forceinline__ void pq_tq_inverse_wht_store_256(
        float * sh, const int tid, dst_t * __restrict__ y, const int64_t base) {
    float x0 = sh[tid] * PQ_TQ_WHT_SIGNS2_256[tid];
    float x1 = sh[tid + 128] * PQ_TQ_WHT_SIGNS2_256[tid + 128];
    const int lane = tid & 31;
    x0 = pq_tq_warp_fwht_32(x0, lane);
    x1 = pq_tq_warp_fwht_32(x1, lane);
    sh[tid] = x0;
    sh[tid + 128] = x1;
    __syncthreads();

    float a0 = sh[tid];
    float b0 = sh[tid ^ 32];
    float a1 = sh[tid + 128];
    float b1 = sh[(tid ^ 32) + 128];
    __syncthreads();
    sh[tid]       = ((tid & 32) == 0) ? (a0 + b0) : (b0 - a0);
    sh[tid + 128] = ((tid & 32) == 0) ? (a1 + b1) : (b1 - a1);
    __syncthreads();

    a0 = sh[tid];
    b0 = sh[tid ^ 64];
    a1 = sh[tid + 128];
    b1 = sh[(tid ^ 64) + 128];
    x0 = ((tid & 64) == 0) ? (a0 + b0) : (b0 - a0);
    x1 = ((tid & 64) == 0) ? (a1 + b1) : (b1 - a1);

    constexpr float inv_sqrt256 = 0.0625f;
    const float y0 = x0 + x1;
    const float y1 = x0 - x1;
    y[base + tid]       = ggml_cuda_cast<dst_t>(y0 * inv_sqrt256 * PQ_TQ_WHT_SIGNS1_256[tid]);
    y[base + tid + 128] = ggml_cuda_cast<dst_t>(y1 * inv_sqrt256 * PQ_TQ_WHT_SIGNS1_256[tid + 128]);
}

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
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::P2_K>(const void * vx, int64_t e) { return pq_dequant_elem_2_k(vx, e); }
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::P3_K>(const void * vx, int64_t e) { return pq_dequant_elem_3_k(vx, e); }
template<> __device__ __forceinline__ float pq_tq_dequant_elem<PqTqTypeTag::P4_K>(const void * vx, int64_t e) { return pq_dequant_elem_4_k(vx, e); }

// ============================================================================
// Kernel: dequantize + cooperative inverse WHT → fp16 output
// ============================================================================
// Each block = 128 threads = one WHT group (wht_dim = 64, 128, or 256).
// D-aware: for D=64 only 64 threads active, for D=256 each handles 2 elements.
// Elements within a WHT group are contiguous in memory.

template <typename dst_t, PqTqTypeTag TT>
static __global__ void k_dequant_pq_tq_unrotated(
        const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t k, const int wht_dim) {

    __shared__ float sh[256];  // max D=256

    const int tid = threadIdx.x;       // 0..127
    const int64_t gid = blockIdx.x;    // WHT group index
    const int64_t base = gid * wht_dim;

    // Phase 1: Dequantize into shared memory, 2 elements per active thread.
    // This halves the centroid decode work before the inverse WHT for pq/tq 2/3/4.
    const int npairs = wht_dim / 2;
    if (tid < npairs) {
        const float2 pair = pq_tq_dequant_pair<TT>(vx, gid * npairs + tid);
        sh[2 * tid + 0] = pair.x;
        sh[2 * tid + 1] = pair.y;
    }
    __syncthreads();

    // Phase 2: Inverse WHT and write output in the normal domain.
    if      (wht_dim == 64)  pq_tq_inverse_wht_store_64(sh, tid, y, base);
    else if (wht_dim == 256) pq_tq_inverse_wht_store_256(sh, tid, y, base);
    else                     pq_tq_inverse_wht_store_128(sh, tid, y, base);
}

// ============================================================================
// Host wrapper: D-aware dispatch (called directly with wht_dim, not via function pointer)
// ============================================================================

template <typename dst_t, PqTqTypeTag TT>
static void dequant_pq_tq_unrotated_wht(
        const void * vx, dst_t * y, int64_t k, int wht_dim, cudaStream_t stream) {
    GGML_ASSERT(wht_dim == 64 || wht_dim == 128 || wht_dim == 256);
    GGML_ASSERT(k % wht_dim == 0);
    const int64_t n_groups = k / wht_dim;
    k_dequant_pq_tq_unrotated<dst_t, TT><<<n_groups, 128, 0, stream>>>(vx, y, k, wht_dim);
}

// Legacy wrapper matching to_fp16_cuda_t signature (assumes D=128)
template <PqTqTypeTag TT>
static void dequant_pq_tq_unrotated_fp16(
        const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequant_pq_tq_unrotated_wht<half, TT>(vx, y, k, 128, stream);
}

template <PqTqTypeTag TT>
static void dequant_pq_tq_unrotated_fp32(
        const void * vx, float * y, int64_t k, cudaStream_t stream) {
    dequant_pq_tq_unrotated_wht<float, TT>(vx, y, k, 128, stream);
}

template <PqTqTypeTag TT>
static void dequant_pq_tq_unrotated_fp16_256(
        const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequant_pq_tq_unrotated_wht<half, TT>(vx, y, k, 256, stream);
}

template <PqTqTypeTag TT>
static void dequant_pq_tq_unrotated_fp32_256(
        const void * vx, float * y, int64_t k, cudaStream_t stream) {
    dequant_pq_tq_unrotated_wht<float, TT>(vx, y, k, 256, stream);
}

// ============================================================================
// D-aware dispatcher: calls kernel with correct wht_dim for pq/tq types
// ============================================================================

static void dequant_pq_tq_unrotated_fp16_dispatch(
        ggml_type type, const void * vx, half * y, int64_t k, int wht_dim, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_PQ2_0: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::T2_0>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ3_0: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::T3_0>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ4_0: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::T4_0>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ2_1: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::T2_1>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ3_1: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::T3_1>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ4_1: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::T4_1>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ4_0_64: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::T4_0_64>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ4_1_64: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::T4_1_64>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ2_K: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::P2_K>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ3_K: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::P3_K>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ4_K: dequant_pq_tq_unrotated_wht<half, PqTqTypeTag::P4_K>(vx, y, k, wht_dim, stream); break;
        default: GGML_ABORT("unsupported pq/tq type for WHT-fused fp16 dequant");
    }
}

static void dequant_pq_tq_unrotated_fp32_dispatch(
        ggml_type type, const void * vx, float * y, int64_t k, int wht_dim, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_PQ2_0: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::T2_0>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ3_0: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::T3_0>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ4_0: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::T4_0>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ2_1: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::T2_1>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ3_1: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::T3_1>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ4_1: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::T4_1>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ4_0_64: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::T4_0_64>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_TQ4_1_64: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::T4_1_64>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ2_K: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::P2_K>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ3_K: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::P3_K>(vx, y, k, wht_dim, stream); break;
        case GGML_TYPE_PQ4_K: dequant_pq_tq_unrotated_wht<float, PqTqTypeTag::P4_K>(vx, y, k, wht_dim, stream); break;
        default: GGML_ABORT("unsupported pq/tq type for WHT-fused fp32 dequant");
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
        case GGML_TYPE_PQ2_K: return dequant_pq_tq_unrotated_fp16_256<PqTqTypeTag::P2_K>;
        case GGML_TYPE_PQ3_K: return dequant_pq_tq_unrotated_fp16_256<PqTqTypeTag::P3_K>;
        case GGML_TYPE_PQ4_K: return dequant_pq_tq_unrotated_fp16_256<PqTqTypeTag::P4_K>;
        default: return nullptr;
    }
}

static to_fp32_cuda_t ggml_get_to_fp32_pq_tq_unrotated_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_PQ2_0: return dequant_pq_tq_unrotated_fp32<PqTqTypeTag::T2_0>;
        case GGML_TYPE_PQ3_0: return dequant_pq_tq_unrotated_fp32<PqTqTypeTag::T3_0>;
        case GGML_TYPE_PQ4_0: return dequant_pq_tq_unrotated_fp32<PqTqTypeTag::T4_0>;
        case GGML_TYPE_TQ2_1: return dequant_pq_tq_unrotated_fp32<PqTqTypeTag::T2_1>;
        case GGML_TYPE_TQ3_1: return dequant_pq_tq_unrotated_fp32<PqTqTypeTag::T3_1>;
        case GGML_TYPE_TQ4_1: return dequant_pq_tq_unrotated_fp32<PqTqTypeTag::T4_1>;
        case GGML_TYPE_PQ4_0_64: return dequant_pq_tq_unrotated_fp32<PqTqTypeTag::T4_0_64>;
        case GGML_TYPE_TQ4_1_64: return dequant_pq_tq_unrotated_fp32<PqTqTypeTag::T4_1_64>;
        case GGML_TYPE_PQ2_K: return dequant_pq_tq_unrotated_fp32_256<PqTqTypeTag::P2_K>;
        case GGML_TYPE_PQ3_K: return dequant_pq_tq_unrotated_fp32_256<PqTqTypeTag::P3_K>;
        case GGML_TYPE_PQ4_K: return dequant_pq_tq_unrotated_fp32_256<PqTqTypeTag::P4_K>;
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
        case GGML_TYPE_PQ2_K:
        case GGML_TYPE_PQ3_K:
        case GGML_TYPE_PQ4_K:
            return true;
        default:
            return false;
    }
}
