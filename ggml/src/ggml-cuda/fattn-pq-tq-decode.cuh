#pragma once

// Three-stage PQ/TQ-native decode attention:
//   Stage 1: Q preprocess kernel  — forward WHT + q8_1 quantize (once per decode step)
//   Stage 2: Split-K main kernel  — dp4a KQ + softmax + V accum (reads pre-processed Q)
//   Stage 3: Final inverse WHT    — one inverse WHT on combined output (once per decode step)
//
// Eliminates per-block redundant Q preprocessing and inverse WHT when parallel_blocks > 1.
// Operates directly on pq/tq blocks without materializing K_f16/V_f16.
// Supports all 6 pq/tq types (pq/tq2/3/4 × _0/_1).
// Supported head dimensions: D=64 (native WHT64), D=128 (native WHT128), D=256 (native WHT256).

#include "common.cuh"
#include "fattn-common.cuh"
#include "pq-tq-fwht.cuh"

// ============================================================================
// Stage 1: Q Preprocess Kernel
// One block per head. 128 threads per block.
// Loads Q float → forward WHT per 128-group → q8_1 quantize → write to global buffer.
// ============================================================================

template<int D>
__launch_bounds__(128, 1)
static __global__ void pq_tq_q_preprocess(
        const float * __restrict__ Q,          // [D, 1, n_heads, n_seq] row-major
        int          * __restrict__ Q_q8_i32,  // [n_heads * D/4] output
        float2       * __restrict__ Q_q8_ds,   // [n_heads * D/32] output
        const float scale,
        const int nb01,                        // Q byte stride per token (unused for decode, Q has 1 token)
        const int nb02,                        // Q byte stride per head
        const int nb03) {                      // Q byte stride per sequence

    constexpr int nthreads = 128;
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;

    // Grid: blockIdx.x = head, blockIdx.y = sequence
    const int head     = blockIdx.x;
    const int sequence = blockIdx.y;

    // Input: Q pointer for this head+sequence
    const float * Q_f = (const float *)((const char *)Q + nb03*sequence + nb02*head);

    // Output: q8 buffer for this head
    const int head_idx = sequence * gridDim.x + head;
    int    * out_i32 = Q_q8_i32 + head_idx * (D/4);
    float2 * out_ds  = Q_q8_ds  + head_idx * (D/32);

    __shared__ float  sh_wht[D > nthreads ? D : nthreads];
    __shared__ int    sh_q_i32[D/4];
    __shared__ float2 sh_q_ds[D/32];

    // Load Q into shared memory
    for (int i = tid; i < D; i += nthreads) {
        sh_wht[i] = Q_f[i];
    }
    __syncthreads();

    // Native forward WHT (D=64, 128, or 256 — single call, no grouped loop)
    pq_tq_coop_wht_forward<D>(sh_wht, tid);

    // Keep WHT input and q8 output in separate shared buffers.
    // Reusing the same storage violates the quantizer's __restrict__ contract and can corrupt Q.

    if (threadIdx.y == 0) { // Warp 0 only
        constexpr int nthreads_quantize = D/4 < WARP_SIZE ? D/4 : WARP_SIZE;
#pragma unroll
        for (int i0 = 0; i0 < D/4; i0 += nthreads_quantize) {
            quantize_q8_1_to_shared<float2, nthreads_quantize>
                (sh_wht + i0*4, scale, sh_q_i32 + i0, sh_q_ds + i0/QI8_1);
        }
    }
    __syncthreads();

    // Write q8 data from shared to global output
    for (int i = tid; i < D/4; i += nthreads) {
        out_i32[i] = sh_q_i32[i];
    }
    for (int i = tid; i < D/32; i += nthreads) {
        out_ds[i] = sh_q_ds[i];
    }
}

// ============================================================================
// Stage 3: Final Inverse WHT Kernel
// Applied once after combine (or after single-block main kernel).
// One block per head. 128 threads per block.
// Reads rotated-domain output from dst, applies inverse WHT, writes back.
// ============================================================================

template<int D>
__launch_bounds__(128, 1)
static __global__ void pq_tq_final_inverse_wht(
        float * __restrict__ dst,    // [D, 1, n_heads, n_seq] — read and write in-place
        const int ne02,              // n_heads
        const int ne01) {            // Q->ne[1] (tokens per decode, always 1)

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;

    // Grid: blockIdx.x = col (always 0 for decode), blockIdx.y = head, blockIdx.z = sequence
    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    // Index matches flash_attn_combine_results layout:
    // j_dst_unrolled = (sequence * ne01 + col) * ne02 + head
    const int j_dst = (sequence * ne01 + col) * ne02 + head;

    float * out = dst + j_dst * D;

    constexpr int sh_size = D > 128 ? D : 128;
    __shared__ float sh[sh_size];

    // Load D elements into shared memory (128 threads; for D>128 each thread loads multiple)
    for (int i = tid; i < D; i += 128) {
        sh[i] = out[i];
    }
    __syncthreads();

    // Native inverse WHT for this D
    pq_tq_coop_wht_inverse<D>(sh, tid);

    // Write back D elements
    for (int i = tid; i < D; i += 128) {
        out[i] = sh[i];
    }
}

// ============================================================================
// Split-K combine + final inverse WHT fused into one kernel.
// Used first for D=64, where the extra post kernel launch shows up clearly.
// ============================================================================

template<int D>
static __global__ void pq_tq_combine_inverse_wht(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        const int parallel_blocks) {
    static_assert(D == 64 || D == 128 || D == 256, "Unsupported WHT dimension");
    constexpr int nthreads = D == 256 ? 128 : D;

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;
    const int j_dst    = (sequence*ne01 + col)*ne02 + head;
    const int tid      = threadIdx.x;

    VKQ_parts += j_dst * parallel_blocks * D;
    VKQ_meta  += j_dst * parallel_blocks;
    dst       += j_dst * D;

    extern __shared__ unsigned char smem_raw[];
    float2 * meta = (float2 *) smem_raw;
    float  * sh   = (float  *) (meta + parallel_blocks);

    for (int i = tid; i < parallel_blocks; i += nthreads) {
        meta[i] = VKQ_meta[i];
    }
    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = fmaxf(kqmax, meta[l].x);
    }

    if constexpr (D == 256) {
        float num0 = 0.0f;
        float num1 = 0.0f;
        float den  = 0.0f;
        for (int l = 0; l < parallel_blocks; ++l) {
            const float scale = expf(meta[l].x - kqmax);
            num0 += scale * VKQ_parts[l*D + tid];
            num1 += scale * VKQ_parts[l*D + tid + 128];
            den  += scale * meta[l].y;
        }

        sh[tid]       = num0 / den;
        sh[tid + 128] = num1 / den;
        __syncthreads();

        pq_tq_coop_wht_inverse<256>(sh, tid);

        dst[tid]       = sh[tid];
        dst[tid + 128] = sh[tid + 128];
    } else {
        float num = 0.0f;
        float den = 0.0f;
        for (int l = 0; l < parallel_blocks; ++l) {
            const float scale = expf(meta[l].x - kqmax);
            num += scale * VKQ_parts[l*D + tid];
            den += scale * meta[l].y;
        }

        sh[tid] = num / den;
        __syncthreads();

        pq_tq_coop_wht_inverse<D>(sh, tid);

        dst[tid] = sh[tid];
    }
}

// ============================================================================
// Stage 2: Split-K Main Kernel (reads pre-processed Q, outputs rotated-domain)
// Template: D = head dimension (64, 128, or 256), type_KV = pq/tq ggml_type.
// ============================================================================

template<int D, int tile_rows, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__launch_bounds__(tile_rows, 1)
static __global__ void flash_attn_pq_tq_decode(
        const char * __restrict__ Q_unused,    // unused — Q is read from Q_q8 buffers
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unsupported variants at compile time:
    if (use_logit_softcap && !(D == 64 || D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q_unused, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03, nb01, nb02, nb03,
            ne10, ne11, ne12, ne13, nb11, nb12, nb13,
            nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    // ---- Compile-time configuration ----
    constexpr int nthreads    = tile_rows;
    constexpr int nwarps      = nthreads / WARP_SIZE;
    constexpr int nthreads_KQ = D == 64 ? 16 : 32;
    constexpr int nthreads_V  = 32;
    constexpr int V_rows_per_thread = (D >= 128) ? 4 : 2; // D=64: 32 threads × 2 = 64 elements
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V; // = 1
    constexpr int KQ_rows_per_iter  = WARP_SIZE / nthreads_KQ;

    constexpr vec_dot_KQ_t   vec_dot_KQ   = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();

    constexpr int ne_KQ      = nthreads;                        // 128
    constexpr int ne_combine  = nwarps * V_cols_per_iter * D;   // 4*D
    constexpr int ne_shared   = ne_KQ > ne_combine ? ne_KQ : ne_combine;

    static_assert(D == 64 || D == 128 || D == 256, "pq/tq decode kernel supports D=64, D=128 and D=256");
    static_assert(tile_rows == 32 || tile_rows == 64 || tile_rows == 128 || (D == 64 && tile_rows == 256),
        "pq/tq decode kernel supports 32-row, 64-row, 128-row tiles, plus 256-row tiles for D=64");
    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    // ---- Per-thread state ----
    float2 VKQ[(D/2)/nthreads_V] = {{0.0f, 0.0f}};
    float  KQ_max_val = -FLT_MAX/2.0f;
    float  KQ_sum_val = 0.0f;

    __shared__ float KQ_sh[ne_shared];

    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    // ---- Grid mapping: ic0=0 (ncols=1), head, sequence ----
    const int sequence = blockIdx.z / ne02;
    const int head     = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12;

    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh = mask ? (const half *)(mask + nb33*(sequence % ne33)) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // ================================================================
    // Phase 1: Load pre-processed Q from contiguous Q_q8 buffer
    // ================================================================
    // Q was already forward-WHT'd and quantized to q8_1 by pq_tq_q_preprocess.
    // The host wrapper passes Q_q8_buf.ptr as the Q arg (Q_unused).
    // Buffer layout: [int32 data for all heads | float2 data for all heads]
    // Offset to float2 section = ne02 * ne03 * (D/4) ints.

    const int head_idx = sequence * ne02 + head;
    const int    * Q_q8_i32 = (const int    *)Q_unused + head_idx * (D/4);
    const float2 * Q_q8_ds  = (const float2 *)((const int *)Q_unused + ne02 * ne03 * (D/4))
                             + head_idx * (D/32);

    int    Q_i32[D/(int(sizeof(int))*nthreads_KQ) > 0 ? D/(int(sizeof(int))*nthreads_KQ) : 1];
    float2 Q_ds [D/(int(sizeof(int))*nthreads_KQ) > 0 ? D/(int(sizeof(int))*nthreads_KQ) : 1];

    // Zero-init for D < nthreads_KQ*sizeof(int) (D=64: only 16 int32s, threads 16-31 contribute 0)
    Q_i32[0] = 0;
    Q_ds [0] = {0.0f, 0.0f};

    // Load q8_1 data from global to registers.
#pragma unroll
    for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
        const int i = i0 + (threadIdx.x % nthreads_KQ);
        if (i < int(D/sizeof(int))) {
            Q_i32[i0/nthreads_KQ] = Q_q8_i32[i];
            Q_ds [i0/nthreads_KQ] = Q_q8_ds[i/QI8_1];
        }
    }

    // ================================================================
    // DEBUG: Dump pq/tq K data and Q data for first head/seq/block to verify encoding
    // ================================================================
    // (In-kernel diagnostic removed — using host-side readback instead)

    // ================================================================
    // Phase 2: Streaming KQ + Softmax + V Accumulation (split-K)
    // ================================================================

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;

    // Split-K: each y-block starts at a different KV offset, strides by gridDim.y.
    K     += blockIdx.y * nthreads * nb11;
    V     += blockIdx.y * nthreads * nb21;
    maskh += blockIdx.y * nthreads;

    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21,
             maskh += (mask ? gridDim.y*nthreads : 0)) {
        const int valid_rows_this_tile = k_VKQ_max - k_VKQ_0 < nthreads ? k_VKQ_max - k_VKQ_0 : nthreads;
        const bool full_tile = valid_rows_this_tile == nthreads;

        // ---- KQ dot product for nthreads (128) K rows ----
        float KQ_reg;
        float KQ_max_new = KQ_max_val;

        if (full_tile) {
#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < int(nthreads_KQ); ++i_KQ_0) {
                const int i_KQ = threadIdx.y*WARP_SIZE + i_KQ_0*KQ_rows_per_iter + threadIdx.x/nthreads_KQ;

                float sum = vec_dot_KQ(K + i_KQ*nb11, nullptr, Q_i32, Q_ds);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask) {
                    sum += slope*__half2float(maskh[i_KQ]);
                }

                KQ_max_new = fmaxf(KQ_max_new, sum + FATTN_KQ_MAX_OFFSET);

                if (threadIdx.x % nthreads_KQ == uint32_t(i_KQ_0)) {
                    KQ_reg = sum;
                }
            }
        } else {
#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < int(nthreads_KQ); ++i_KQ_0) {
                const int i_KQ = threadIdx.y*WARP_SIZE + i_KQ_0*KQ_rows_per_iter + threadIdx.x/nthreads_KQ;

                float sum = -FLT_MAX/2.0f;
                if (i_KQ < valid_rows_this_tile) {
                    sum = vec_dot_KQ(K + i_KQ*nb11, nullptr, Q_i32, Q_ds);
                    sum = warp_reduce_sum<nthreads_KQ>(sum);

                    if (use_logit_softcap) {
                        sum = logit_softcap*tanhf(sum);
                    }

                    if (mask) {
                        sum += slope*__half2float(maskh[i_KQ]);
                    }
                }

                KQ_max_new = fmaxf(KQ_max_new, sum + FATTN_KQ_MAX_OFFSET);

                if (threadIdx.x % nthreads_KQ == uint32_t(i_KQ_0)) {
                    KQ_reg = sum;
                }
            }
        }

        // ---- Softmax: rescale accumulators ----
#pragma unroll
        for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
            KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, offset, WARP_SIZE));
        }
        const float KQ_max_scale = expf(KQ_max_val - KQ_max_new);
        KQ_max_val = KQ_max_new;

        KQ_reg = expf(KQ_reg - KQ_max_val);
        KQ_sum_val = KQ_sum_val*KQ_max_scale + KQ_reg;
        const int kq_slot = threadIdx.y*WARP_SIZE + (threadIdx.x % nthreads_KQ)*KQ_rows_per_iter + threadIdx.x/nthreads_KQ;
        KQ_sh[kq_slot] = KQ_reg;

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[i_VKQ_0/nthreads_V].x *= KQ_max_scale;
            VKQ[i_VKQ_0/nthreads_V].y *= KQ_max_scale;
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif

        // ---- V accumulation ----
        if (full_tile) {
#pragma unroll
            for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
                const int k = threadIdx.y*WARP_SIZE + k0;
                const float KQ_k = KQ_sh[k];

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    float2 tmp[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread);

#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        VKQ[i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x * KQ_k;
                        VKQ[i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y * KQ_k;
                    }
                }
            }
        } else {
#pragma unroll
            for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
                const int k = threadIdx.y*WARP_SIZE + k0;
                if (k >= valid_rows_this_tile) {
                    continue;
                }

                const float KQ_k = KQ_sh[k];

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    float2 tmp[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread);

#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        VKQ[i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x * KQ_k;
                        VKQ[i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y * KQ_k;
                    }
                }
            }
        }
    }

    // ---- Attention sinks ----
    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

        const float kqmax_new_j = fmaxf(sink, KQ_max_val);
        const float KQ_max_scale = expf(KQ_max_val - kqmax_new_j);
        KQ_max_val = kqmax_new_j;

        KQ_sum_val = KQ_sum_val*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max_val) : 0.0f);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[i_VKQ_0/nthreads_V].x *= KQ_max_scale;
            VKQ[i_VKQ_0/nthreads_V].y *= KQ_max_scale;
        }
    }

    // ================================================================
    // Phase 3: Cross-warp combine + Write ROTATED-DOMAIN output
    // (No inverse WHT here — that's done by pq_tq_final_inverse_wht)
    // ================================================================

    __shared__ float KQ_max_shared[WARP_SIZE];
    __shared__ float KQ_sum_shared[WARP_SIZE];

    if (threadIdx.y == 0) {
        KQ_max_shared[threadIdx.x] = -FLT_MAX/2.0f;
        KQ_sum_shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        KQ_max_shared[threadIdx.y] = KQ_max_val;
    }
    __syncthreads();

    float kqmax_new = KQ_max_shared[threadIdx.x];
    kqmax_new = warp_reduce_max(kqmax_new);
    const float kqmax_scale = expf(KQ_max_val - kqmax_new);
    KQ_max_val = kqmax_new;

    // Write rescaled VKQ to shared for cross-warp summation:
    float2 * VKQ_tmp = (float2 *) KQ_sh + threadIdx.y*(D/2);

    if constexpr (D == 64) {
        // V_rows_per_thread=2: each thread has exactly 1 float2 in VKQ, write directly
        VKQ[0].x *= kqmax_scale;
        VKQ[0].y *= kqmax_scale;
        VKQ_tmp[threadIdx.x] = VKQ[0];
    } else {
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
    }

    KQ_sum_val *= kqmax_scale;
    KQ_sum_val = warp_reduce_sum(KQ_sum_val);
    if (threadIdx.x == 0) {
        KQ_sum_shared[threadIdx.y] = KQ_sum_val;
    }

    __syncthreads();

    // Cross-warp combine → write rotated-domain partial/final output.
    // D >= 128 and nthreads == 128, so all threads participate.
    {
        float KQ_sum_total = KQ_sum_shared[threadIdx.x];
        KQ_sum_total = warp_reduce_sum(KQ_sum_total);

        const int j_dst = sequence*int(ne01.z)*ne02 + head;

        if (gridDim.y == 1) {
            // Single-block path: fuse inverse WHT to avoid a separate kernel launch.
            // Combine warp results → normalize → native inverse WHT → write final.

            if constexpr (D == 64) {
                // D < nthreads: only first 64 threads have valid data
                float val = 0.0f;
                if (tid < D) {
#pragma unroll
                    for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                        for (int v = 0; v < V_cols_per_iter; ++v) {
                            val += KQ_sh[w*V_cols_per_iter*D + v*D + tid];
                        }
                    }
                    val /= KQ_sum_total;
                }
                __syncthreads();

                if (tid < D) {
                    KQ_sh[tid] = val;
                }
                __syncthreads();

                pq_tq_coop_wht_inverse<64>(KQ_sh, tid);

                if (tid < D) {
                    dst[j_dst*D + tid] = KQ_sh[tid];
                }

            } else if constexpr (D == 128) {
                constexpr int vals_per_thread = D / nthreads;
                float vals[vals_per_thread] = {0.0f};
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
#pragma unroll
                        for (int off = 0; off < D; off += nthreads) {
                            vals[off / nthreads] += KQ_sh[w*V_cols_per_iter*D + v*D + tid + off];
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < vals_per_thread; ++i) {
                    vals[i] /= KQ_sum_total;
                }
                __syncthreads();

#pragma unroll
                for (int off = 0; off < D; off += nthreads) {
                    KQ_sh[tid + off] = vals[off / nthreads];
                }
                __syncthreads();

                pq_tq_coop_wht_inverse<128>(KQ_sh, tid);

#pragma unroll
                for (int off = 0; off < D; off += nthreads) {
                    dst[j_dst*D + tid + off] = KQ_sh[tid + off];
                }

            } else if constexpr (D == 256) {
                // D > nthreads: each thread handles 2 elements
                float val0 = 0.0f, val1 = 0.0f;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        val0 += KQ_sh[w*V_cols_per_iter*D + v*D + tid];
                        val1 += KQ_sh[w*V_cols_per_iter*D + v*D + tid + 128];
                    }
                }
                val0 /= KQ_sum_total;
                val1 /= KQ_sum_total;
                __syncthreads(); // ensure all warp reads complete before overwriting

                KQ_sh[tid]       = val0;
                KQ_sh[tid + 128] = val1;
                __syncthreads();

                pq_tq_coop_wht_inverse<256>(KQ_sh, tid);

                dst[j_dst*D + tid]       = KQ_sh[tid];
                dst[j_dst*D + tid + 128] = KQ_sh[tid + 128];
            }
        } else {
            // Multi-block path: write rotated-domain partials for combine step.
#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                const int idx = i0 + tid;
                if (idx < D) {
                    float val = 0.0f;
#pragma unroll
                    for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                        for (int v = 0; v < V_cols_per_iter; ++v) {
                            val += KQ_sh[w*V_cols_per_iter*D + v*D + idx];
                        }
                    }
                    dst[(j_dst*gridDim.y + blockIdx.y)*D + idx] = val;
                }
            }

            if (tid == 0) {
                dst_meta[(sequence*int(ne01.z)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max_val, KQ_sum_total);
            }
        }
    }

#else
    GGML_UNUSED_VARS(Q_unused, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// ============================================================================
// Host launch wrapper: three-stage orchestration
//   1. pq_tq_q_preprocess — forward WHT + q8_1 quantize (once)
//   2. flash_attn_pq_tq_decode — split-K main kernel (reads pre-processed Q)
//   3. flash_attn_combine_results (if parallel_blocks > 1)
//   4. pq_tq_final_inverse_wht — one inverse WHT on combined output (once)
// ============================================================================

template <int D, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_pq_tq_decode_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    ggml_tensor       * KQV   = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // Extract op params.
    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const int n_heads = Q->ne[2];
    const int n_seq   = Q->ne[3];
    const int n_heads_total = n_heads * n_seq;

    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_heads))));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int kernel_rows_per_block = D == 64 ? 256 : 32;

    // ================================================================
    // Stage 2: Select kernel variant + compute parallel_blocks
    // ================================================================
    fattn_kernel_t fattn_kernel;
    if (kernel_rows_per_block == 32) {
        if (logit_softcap == 0.0f) {
            fattn_kernel = flash_attn_pq_tq_decode<D, 32, type_K, type_V, false>;
        } else {
            fattn_kernel = flash_attn_pq_tq_decode<D, 32, type_K, type_V, true>;
        }
    } else if (kernel_rows_per_block == 64) {
        if (logit_softcap == 0.0f) {
            fattn_kernel = flash_attn_pq_tq_decode<D, 64, type_K, type_V, false>;
        } else {
            fattn_kernel = flash_attn_pq_tq_decode<D, 64, type_K, type_V, true>;
        }
    } else if (kernel_rows_per_block == 128) {
        if (logit_softcap == 0.0f) {
            fattn_kernel = flash_attn_pq_tq_decode<D, 128, type_K, type_V, false>;
        } else {
            fattn_kernel = flash_attn_pq_tq_decode<D, 128, type_K, type_V, true>;
        }
    } else {
        if constexpr (D == 64) {
            GGML_ASSERT(kernel_rows_per_block == 256);
            if (logit_softcap == 0.0f) {
                fattn_kernel = flash_attn_pq_tq_decode<D, 256, type_K, type_V, false>;
            } else {
                fattn_kernel = flash_attn_pq_tq_decode<D, 256, type_K, type_V, true>;
            }
        } else {
            GGML_ABORT("fatal error: unsupported pq/tq tile size");
        }
    }

    const int ntiles_KV = (K->ne[1] + kernel_rows_per_block - 1) / kernel_rows_per_block;

    const int gqa_ratio  = Q->ne[2] / K->ne[2];
    const int ntiles_dst = 1 * gqa_ratio * K->ne[2] * Q->ne[3]; // = n_heads_total

    const int nwarps = kernel_rows_per_block / WARP_SIZE;
    const dim3 block_main(WARP_SIZE, nwarps, 1);

    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, fattn_kernel, block_main.x * block_main.y, 0));
    GGML_ASSERT(max_blocks_per_sm > 0);

    int parallel_blocks = std::min(max_blocks_per_sm, ntiles_KV);

    {
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int pb = parallel_blocks; pb <= ntiles_KV; ++pb) {
            const int nblocks = ntiles_dst * pb;
            const int nwaves  = (nblocks + blocks_per_wave - 1) / blocks_per_wave;
            const int eff     = 100 * nblocks / (nwaves * blocks_per_wave);

            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }
            if (eff > efficiency_percent_best) {
                nwaves_best             = nwaves;
                efficiency_percent_best = eff;
                parallel_blocks         = pb;
            }
        }
    }

    // ================================================================
    // Stage 1: Q Preprocess — forward WHT + q8_1 quantize (once)
    // ================================================================
    // Combined Q_q8 buffer layout (contiguous):
    //   [int32 data: n_heads_total * (D/4) ints] [float2 data: n_heads_total * (D/32) float2s]
    // The main kernel recovers the float2 offset using ne02 * ne03 * (D/4).
    const size_t q8_i32_bytes = size_t(n_heads_total) * (D/4) * sizeof(int);
    const size_t q8_ds_bytes  = size_t(n_heads_total) * (D/32) * sizeof(float2);

    ggml_cuda_pool_alloc<char> Q_q8_buf(pool, q8_i32_bytes + q8_ds_bytes);

    int    * Q_q8_i32 = (int    *) Q_q8_buf.ptr;
    float2 * Q_q8_ds  = (float2 *)(Q_q8_buf.ptr + q8_i32_bytes);

    {
        const dim3 block_pre(WARP_SIZE, 4, 1);
        const dim3 grid_pre(n_heads, n_seq);

        pq_tq_q_preprocess<D><<<grid_pre, block_pre, 0, main_stream>>>(
            (const float *) Q->data,
            Q_q8_i32, Q_q8_ds,
            scale,
            Q->nb[1], Q->nb[2], Q->nb[3]);
        CUDA_CHECK(cudaGetLastError());
    }
    // ================================================================
    // Stage 2: Split-K Main Kernel
    // ================================================================
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks * ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks * ggml_nrows(KQV));
    }

    const dim3 grid_main(1, parallel_blocks, n_heads_total); // ntiles_x = 1 for decode
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    fattn_kernel<<<grid_main, block_main, 0, main_stream>>>(
        (const char *) Q_q8_buf.ptr,               // Q_unused: combined Q_q8 buffer
        (const char *) K->data,
        (const char *) V->data,
        mask  ? (const char *) mask->data  : nullptr,
        sinks ? (const char *) sinks->data : nullptr,
        nullptr,
        parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data,
        dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0);
    CUDA_CHECK(cudaGetLastError());

    // ================================================================
    // Combine split-K partials + inverse WHT (separate kernels)
    // ================================================================
    if (parallel_blocks > 1) {
        const dim3 grid_iwht(Q->ne[1], n_heads, n_seq);

        if constexpr (D == 64) {
            const dim3 block_combine_iwht(D, 1, 1);
            const size_t nbytes_shared_combine_iwht = parallel_blocks * sizeof(float2) + D * sizeof(float);

            pq_tq_combine_inverse_wht<D>
                <<<grid_iwht, block_combine_iwht, nbytes_shared_combine_iwht, main_stream>>>
                ((const float *) dst_tmp.ptr, (const float2 *) dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
            CUDA_CHECK(cudaGetLastError());
        } else {
            const dim3 block_combine(D, 1, 1);
            const size_t nbytes_shared_combine = parallel_blocks * sizeof(float2);

            flash_attn_combine_results<D>
                <<<grid_iwht, block_combine, nbytes_shared_combine, main_stream>>>
                (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
            CUDA_CHECK(cudaGetLastError());

            const dim3 block_iwht(WARP_SIZE, 4, 1);

            pq_tq_final_inverse_wht<D><<<grid_iwht, block_iwht, 0, main_stream>>>(
                (float *) KQV->data, (int)Q->ne[2], (int)Q->ne[1]);
            CUDA_CHECK(cudaGetLastError());
        }
    }

}
