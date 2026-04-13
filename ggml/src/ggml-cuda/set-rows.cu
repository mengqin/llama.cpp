#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "pq-tq-common.cuh"
#include "pq-tq-fwht.cuh"

// =============================================================================
// PQ/TQ set_rows: 128-thread parallel WHT kernels
//
// One CUDA block (128 threads) processes one rotation group (128 elements).
// Norm reduction uses warp shuffles; WHT uses shared-memory butterfly.
// This replaces the old 1-thread-per-group serial approach (~25x speedup).
// =============================================================================


// =============================================================================
// D-aware helper: load data, compute L2 norm, normalize, apply forward WHT.
// After return: sh[0..wht_dim-1] = rotated values, *norm_sh = input L2 norm.
// Requires 128 threads. For D=64: threads 64-127 idle. For D=256: each handles 2.
// =============================================================================
static __device__ __forceinline__ void pq_tq_set_rows_load_wht(
    float * sh, float * warp_buf, float * norm_sh,
    const float * __restrict__ src, const int t, const int wht_dim,
    const bool scale_to_centroid_variance = true) {

    float raw = 0.0f, raw1 = 0.0f;
    if (wht_dim <= 64) {
        if (t < wht_dim) raw = src[t];
    } else if (wht_dim <= 128) {
        raw = src[t];
    } else {
        raw = src[t];
        raw1 = src[t + 128];
    }

    float sq = raw * raw + raw1 * raw1;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        sq += __shfl_xor_sync(0xffffffffu, sq, mask);
    if ((t & 31) == 0) warp_buf[t >> 5] = sq;
    __syncthreads();
    if (t == 0) *norm_sh = sqrtf(warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3] + 1e-10f);
    __syncthreads();

    if (wht_dim <= 64) {
        if (t < wht_dim) sh[t] = raw / *norm_sh;
    } else if (wht_dim <= 128) {
        sh[t] = raw / *norm_sh;
    } else {
        sh[t] = raw / *norm_sh;
        sh[t + 128] = raw1 / *norm_sh;
    }
    __syncthreads();

    if      (wht_dim == 64)  pq_tq_coop_wht_forward<64>(sh, t);
    else if (wht_dim == 256) pq_tq_coop_wht_forward<256>(sh, t);
    else                     pq_tq_coop_wht_forward<128>(sh, t);

    // Scale WHT coefficients to match centroid distribution (optimized for D=128).
    // For D<128 coefficients are too spread (variance 1/D > 1/128) → scale down.
    // For D>128 coefficients are too narrow (variance 1/D < 1/128) → scale up.
    // The recon_norm correction in the stored norm automatically compensates.
    if (scale_to_centroid_variance && wht_dim != 128) {
        const float wht_coeff_scale = sqrtf((float)wht_dim / 128.0f);
        if (wht_dim <= 64) {
            if (t < wht_dim) sh[t] *= wht_coeff_scale;
        } else {
            sh[t] *= wht_coeff_scale;
            if (wht_dim > 128) sh[t + 128] *= wht_coeff_scale;
        }
        __syncthreads();
    }
}

// D-aware quantize+recon-norm helper for 2-bit centroids.
// Writes to qidx_sh[0..wht_dim-1], returns per-thread recon sq via rsq.
static __device__ __forceinline__ void pq_tq_set_rows_quant2(
    float * sh, uint8_t * qidx_sh, const int t, const int wht_dim,
    float & rsq_out) {

    uint8_t qidx = 0;
    float c = 0.0f, rsq = 0.0f;
    if (wht_dim <= 64 && t >= wht_dim) {
        // idle thread
    } else {
        qidx = pq_tq_find_nearest_centroid_2bit(sh[t]);
        c = PQ_TQ_CENTROIDS_2BIT[qidx];
        qidx_sh[t] = qidx;
        rsq = c * c;
    }
    if (wht_dim > 128) {
        uint8_t q1 = pq_tq_find_nearest_centroid_2bit(sh[t + 128]);
        float c1 = PQ_TQ_CENTROIDS_2BIT[q1];
        qidx_sh[t + 128] = q1;
        rsq += c1 * c1;
    }
    rsq_out = rsq;
}

// D-aware quantize+recon-norm helper for 3-bit centroids (2-bit magnitude + sign).
static __device__ __forceinline__ void pq_tq_set_rows_quant3(
    float * sh, uint8_t * qidx_sh, const int t, const int wht_dim,
    float & rsq_out) {

    uint8_t qidx = 0;
    float c = 0.0f, rsq = 0.0f;
    if (wht_dim <= 64 && t >= wht_dim) {
        // idle thread
    } else {
        qidx = pq_tq_find_nearest_centroid_3bit(sh[t]);
        c = PQ_TQ_CENTROIDS_3BIT[qidx];
        qidx_sh[t] = qidx;
        rsq = c * c;
    }
    if (wht_dim > 128) {
        uint8_t q1 = pq_tq_find_nearest_centroid_3bit(sh[t + 128]);
        float c1 = PQ_TQ_CENTROIDS_3BIT[q1];
        qidx_sh[t + 128] = q1;
        rsq += c1 * c1;
    }
    rsq_out = rsq;
}

// D-aware quantize+recon-norm helper for 4-bit centroids.
static __device__ __forceinline__ void pq_tq_set_rows_quant4(
    float * sh, uint8_t * qidx_sh, const int t, const int wht_dim,
    float & rsq_out) {

    uint8_t qidx = pq_tq_find_nearest_centroid_4bit(sh[t]);
    float c = PQ_TQ_CENTROIDS_4BIT[qidx];
    qidx_sh[t] = qidx;
    float rsq = c * c;
    if (wht_dim > 128) {
        uint8_t q1 = pq_tq_find_nearest_centroid_4bit(sh[t + 128]);
        float c1 = PQ_TQ_CENTROIDS_4BIT[q1];
        qidx_sh[t + 128] = q1;
        rsq += c1 * c1;
    }
    rsq_out = rsq;
}

// D-aware warp-reduction of per-thread value to warp_buf[4].
static __device__ __forceinline__ void pq_tq_set_rows_reduce(
    float val, float * warp_buf, const int t) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, mask);
    if ((t & 31) == 0) warp_buf[t >> 5] = val;
    __syncthreads();
}

// When D!=128 uses centroid-variance scaling, QJL residuals must be computed in
// the same unscaled normalized domain as centroid/recon_norm, otherwise the
// residual becomes s*x - x_hat instead of x - x_hat.
static __device__ __forceinline__ float pq_tq_set_rows_qjl_residual_fix(
    const float rotated_coeff, const float centroid, const float inv_recon, const int wht_dim) {
    float rotated = rotated_coeff;
    if (wht_dim != 128) {
        rotated *= sqrtf(128.0f / (float)wht_dim);
    }
    return rotated - centroid * inv_recon;
}

// --- pq4: one CUDA block (128 threads) per pq4 block ---
template <typename idx_t, typename block_pq4, int block_qk>
static __global__ void __launch_bounds__(128, 2) k_set_rows_pq4(
        const float        * __restrict__ src0,
        const idx_t        * __restrict__ src1,
        block_pq4       * __restrict__ dst,
        const int          wht_dim,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const uint3   ne00_fd,
        const uint3   ne01_fd,
        const uint3   ne02_fd,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {

    const int t = threadIdx.x;               // element index within block: 0..127
    const int64_t g = (int64_t)blockIdx.x;  // which pq4 block

    __shared__ float   sh[256];    // working buffer (max D=256)
    __shared__ float   warp_buf[4]; // cross-warp reduction (4 warps)
    __shared__ float   norm_sh;    // broadcast: L2 norm of input block
    __shared__ uint8_t qidx_sh[256]; // quantized centroid indices

    // 1. Decode block index → source/dest pointers
    const int64_t i_base = g * (int64_t)wht_dim;
    uint32_t tmp = (uint32_t)i_base;
    uint2 dm;
    dm = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y;
    const int64_t i03 = (int64_t)tmp;

    const int64_t i12     = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11     = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10     = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float       * src_block = (src0 + i01*s01 + i02*s02 + i03*s03) + i00;
    block_pq4 * dst_blocks     = dst + (dst_row*s1 + i02*s2 + i03*s3) / (int64_t)sizeof(block_pq4)
                                       + i00 / block_qk;

    // 2. D-aware load + L2 norm + forward WHT
    pq_tq_set_rows_load_wht(sh, warp_buf, &norm_sh, src_block, t, wht_dim);

    // 3. Quantize to 4-bit centroids
    float rsq;
    pq_tq_set_rows_quant4(sh, qidx_sh, t, wht_dim, rsq);

    // 4. Recon norm reduction
    pq_tq_set_rows_reduce(rsq, warp_buf, t);

    // 5. Pack nibbles (D-aware)
    const int n_pq4 = wht_dim / block_qk;  // 1, 2, or 4
    const int pack_thr4 = wht_dim / 2;
    if (t < pack_thr4) {
        const int bytes_per_block = block_qk / 2;
        const int blk = t / bytes_per_block;
        const int idx = t % bytes_per_block;
        dst_blocks[blk].qs[idx] = qidx_sh[blk*block_qk + 2*idx] | (uint8_t)(qidx_sh[blk*block_qk + 2*idx + 1] << 4);
    }

    // 6. Write block headers
    if (t == 0) {
        const float recon_norm = sqrtf(warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3] + 1e-10f);
        const ggml_half nh = __float2half((recon_norm > 1e-10f) ? (norm_sh / recon_norm) : norm_sh);
        for (int i = 0; i < n_pq4; i++) {
            dst_blocks[i].norm  = nh;
            dst_blocks[i].rnorm = __float2half(0.0f);
        }
    }

    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
}

// pq2 set_rows kernel: 2-bit PolarQuant (4 centroids, no signs)
// One 128-thread CUDA block per QK_PQ_TQ_2_GROUP=128 rotation group.
template<typename idx_t>
static __global__ void __launch_bounds__(128, 2) k_set_rows_pq2(
        const float    * __restrict__ src0,
        const idx_t    * __restrict__ src1,
        block_pq2 * __restrict__ dst,
        const int          wht_dim,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const uint3   ne00_fd,
        const uint3   ne01_fd,
        const uint3   ne02_fd,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {

    const int t = threadIdx.x;
    const int64_t g = (int64_t)blockIdx.x;

    __shared__ float   sh[256];
    __shared__ float   warp_buf[4];
    __shared__ float   norm_sh;
    __shared__ uint8_t qidx_sh[256];

    // 1. Decode group index → source/dest pointers
    const int64_t i_base = g * (int64_t)wht_dim;
    uint32_t tmp = (uint32_t)i_base;
    uint2 dm;
    dm = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y;
    const int64_t i03 = (int64_t)tmp;

    const int64_t i12     = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11     = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10     = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float       * src_group = (src0 + i01*s01 + i02*s02 + i03*s03) + i00;
    block_pq2 * dst_blocks   = dst + (dst_row*s1 + i02*s2 + i03*s3) / (int64_t)sizeof(block_pq2)
                                       + i00 / QK_PQ_TQ_2;

    // 2. D-aware load + L2 norm + forward WHT
    // D=64 pq2/tq2 behaves better with the paper-faithful WHT normalization.
    pq_tq_set_rows_load_wht(sh, warp_buf, &norm_sh, src_group, t, wht_dim, wht_dim != 64);

    // 3. Quantize to 2-bit centroids
    float rsq;
    pq_tq_set_rows_quant2(sh, qidx_sh, t, wht_dim, rsq);

    // 4. Recon norm reduction
    pq_tq_set_rows_reduce(rsq, warp_buf, t);

    // 5. Pack 2-bit qs (D-aware: wht_dim/4 threads)
    const int n_blocks_2 = wht_dim / QK_PQ_TQ_2;
    const int pack_thr_2 = wht_dim / 4;
    if (t < pack_thr_2) {
        const int base = t * 4;
        const uint8_t packed = ((qidx_sh[base + 0] & 0x3u) << 0)
                             | ((qidx_sh[base + 1] & 0x3u) << 2)
                             | ((qidx_sh[base + 2] & 0x3u) << 4)
                             | ((qidx_sh[base + 3] & 0x3u) << 6);
        dst_blocks[t >> 3].qs[t & 7] = packed;
    }

    // 6. Write corrected norm to all sub-blocks
    if (t == 0) {
        const float recon_norm = sqrtf(warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3] + 1e-10f);
        const ggml_half norm_h = __float2half((recon_norm > 1e-10f) ? (norm_sh / recon_norm) : norm_sh);
        for (int i = 0; i < n_blocks_2; i++) dst_blocks[i].norm = norm_h;
    }

    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
}

// --- pq3: one CUDA block (128 threads) per rotation group (128 elements = 4×block_pq3) ---
template <typename idx_t>
static __global__ void __launch_bounds__(128, 2) k_set_rows_pq3(
        const float    * __restrict__ src0,
        const idx_t    * __restrict__ src1,
        block_pq3 * __restrict__ dst,
        const int          wht_dim,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const uint3   ne00_fd,
        const uint3   ne01_fd,
        const uint3   ne02_fd,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {

    const int t = threadIdx.x;               // element index: 0..127
    const int64_t g = (int64_t)blockIdx.x;  // which rotation group

    __shared__ float   sh[256];
    __shared__ float   warp_buf[4];
    __shared__ float   norm_sh;
    __shared__ uint8_t qidx_sh[256]; // 3-bit indices (bits [1:0]=q2, bit[2]=sign)

    // 1. Decode group index → source/dest pointers
    const int64_t i_base = g * (int64_t)wht_dim;
    uint32_t tmp = (uint32_t)i_base;
    uint2 dm;
    dm = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y;
    const int64_t i03 = (int64_t)tmp;

    const int64_t i12     = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11     = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10     = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float       * src_group = (src0 + i01*s01 + i02*s02 + i03*s03) + i00;
    block_pq3 * dst_blocks   = dst + (dst_row*s1 + i02*s2 + i03*s3) / (int64_t)sizeof(block_pq3)
                                       + i00 / QK_PQ_TQ_3;

    // 2. D-aware load + L2 norm + forward WHT
    pq_tq_set_rows_load_wht(sh, warp_buf, &norm_sh, src_group, t, wht_dim);

    // 3. Quantize to 3-bit centroids
    float rsq;
    pq_tq_set_rows_quant3(sh, qidx_sh, t, wht_dim, rsq);

    // 4. Recon norm reduction
    pq_tq_set_rows_reduce(rsq, warp_buf, t);

    // 5. Pack 2-bit qs (D-aware)
    const int n_blocks_3 = wht_dim / QK_PQ_TQ_3;
    const int pack_thr_3 = wht_dim / 4;
    if (t < pack_thr_3) {
        const int base = t * 4;
        const uint8_t packed = ((qidx_sh[base + 0] & 0x3u) << 0)
                             | ((qidx_sh[base + 1] & 0x3u) << 2)
                             | ((qidx_sh[base + 2] & 0x3u) << 4)
                             | ((qidx_sh[base + 3] & 0x3u) << 6);
        dst_blocks[t >> 3].qs[t & 7] = packed;
    }

    // 6. Pack 1-bit signs (D-aware)
    const int sign_thr_3 = wht_dim / 8;
    if (t < sign_thr_3) {
        const int base = (t >> 2) * 32 + (t & 3) * 8;
        const uint8_t packed = ((qidx_sh[base + 0] >> 2) & 1u) << 0
                             | ((qidx_sh[base + 1] >> 2) & 1u) << 1
                             | ((qidx_sh[base + 2] >> 2) & 1u) << 2
                             | ((qidx_sh[base + 3] >> 2) & 1u) << 3
                             | ((qidx_sh[base + 4] >> 2) & 1u) << 4
                             | ((qidx_sh[base + 5] >> 2) & 1u) << 5
                             | ((qidx_sh[base + 6] >> 2) & 1u) << 6
                             | ((qidx_sh[base + 7] >> 2) & 1u) << 7;
        dst_blocks[t >> 2].signs[t & 3] = packed;
    }

    // 7. Write corrected norm to all sub-blocks
    if (t == 0) {
        const float recon_norm = sqrtf(warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3] + 1e-10f);
        const ggml_half norm_h = __float2half((recon_norm > 1e-10f) ? (norm_sh / recon_norm) : norm_sh);
        for (int i = 0; i < n_blocks_3; i++) dst_blocks[i].norm = norm_h;
    }

    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
}

// Launcher: pq4 — one 128-thread CUDA block per pq4 block
template<typename idx_t, typename block_pq4, int block_qk>
static void set_rows_cuda_pq4(
        const float * src0_d, const idx_t * src1_d, block_pq4 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const int wht_dim, cudaStream_t stream) {
    GGML_ASSERT(ne00 % wht_dim == 0);
    GGML_ASSERT(wht_dim >= block_qk);

    const int64_t n_blocks = (ne00 * ne01 * ne02 * ne03) / wht_dim;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (n_blocks > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_pq4<idx_t, block_pq4, block_qk><<<(int)n_blocks, 128, 0, stream>>>(
            src0_d, src1_d, dst_d, wht_dim, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

// Launcher: pq2 — one 128-thread CUDA block per 128-element rotation group
template<typename idx_t>
static void set_rows_cuda_pq2(
        const float * src0_d, const idx_t * src1_d, block_pq2 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const int wht_dim, cudaStream_t stream) {
    GGML_ASSERT(ne00 % wht_dim == 0);

    const int64_t n_groups = (ne00 * ne01 * ne02 * ne03) / wht_dim;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (n_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_pq2<<<(int)n_groups, 128, 0, stream>>>(
            src0_d, src1_d, dst_d, wht_dim, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

// Launcher: pq3 — one 128-thread CUDA block per 128-element rotation group
template<typename idx_t>
static void set_rows_cuda_pq3(
        const float * src0_d, const idx_t * src1_d, block_pq3 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const int wht_dim, cudaStream_t stream) {
    GGML_ASSERT(ne00 % wht_dim == 0);

    const int64_t n_groups = (ne00 * ne01 * ne02 * ne03) / wht_dim;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (n_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_pq3<<<(int)n_groups, 128, 0, stream>>>(
            src0_d, src1_d, dst_d, wht_dim, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

// =============================================================================
// QJL set_rows kernels (_1 variants): same WHT + quantize + residual signs
// =============================================================================

// --- tq4 set_rows ---
template <typename idx_t, typename block_tq4, int block_qk>
static __global__ void __launch_bounds__(128, 2) k_set_rows_tq4_qjl(
        const float        * __restrict__ src0,
        const idx_t        * __restrict__ src1,
        block_tq4          * __restrict__ dst,
        const int          wht_dim,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int t = threadIdx.x;
    const int64_t g = (int64_t)blockIdx.x;

    __shared__ float   sh[256];
    __shared__ float   warp_buf[4];
    __shared__ float   norm_sh;
    __shared__ float   recon_norm_sh;
    __shared__ uint8_t qidx_sh[256];

    const int64_t i_base = g * (int64_t)wht_dim;
    uint32_t tmp = (uint32_t)i_base;
    uint2 dm;
    dm = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y;
    const int64_t i03 = (int64_t)tmp;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10 = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src_block = (src0 + i01*s01 + i02*s02 + i03*s03) + i00;
    block_tq4 * dst_blocks = dst + (dst_row*s1 + i02*s2 + i03*s3) / (int64_t)sizeof(block_tq4) + i00 / block_qk;

    // D-aware load + norm + WHT
    pq_tq_set_rows_load_wht(sh, warp_buf, &norm_sh, src_block, t, wht_dim);

    // Quantize 4-bit + recon norm
    float rsq;
    pq_tq_set_rows_quant4(sh, qidx_sh, t, wht_dim, rsq);
    pq_tq_set_rows_reduce(rsq, warp_buf, t);
    if (t == 0) recon_norm_sh = sqrtf(warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3] + 1e-10f);
    __syncthreads();

    // QJL: residual + ballot signs
    const int n_tq4 = wht_dim / block_qk;
    const int warps_per_block = block_qk / WARP_SIZE;
    const float inv_recon = 1.0f / (recon_norm_sh + 1e-10f);
    const bool active_lo = t < (wht_dim < 128 ? wht_dim : 128);
    const float rotated_lo = active_lo ? sh[t] : 0.0f;
    const float c_lo = active_lo ? PQ_TQ_CENTROIDS_4BIT[qidx_sh[t]] : 0.0f;
    const float residual_lo = pq_tq_set_rows_qjl_residual_fix(rotated_lo, c_lo, inv_recon, wht_dim);
    const uint32_t signs_lo = __ballot_sync(0xffffffffu, active_lo && residual_lo >= 0.0f);
    float rsq2 = active_lo ? residual_lo * residual_lo : 0.0f;
    float residual_hi = 0.0f;
    uint32_t signs_hi = 0;
    if (wht_dim > 128) {
        const float c_hi = PQ_TQ_CENTROIDS_4BIT[qidx_sh[t + 128]];
        residual_hi = pq_tq_set_rows_qjl_residual_fix(sh[t + 128], c_hi, inv_recon, wht_dim);
        signs_hi = __ballot_sync(0xffffffffu, residual_hi >= 0.0f);
        rsq2 += residual_hi * residual_hi;
    }
    pq_tq_set_rows_reduce(rsq2, warp_buf, t);

    // Pack nibbles
    const int pack_thr4 = wht_dim / 2;
    if (t < pack_thr4) {
        const int bytes_per_block = block_qk / 2;
        const int blk = t / bytes_per_block;
        const int idx = t % bytes_per_block;
        dst_blocks[blk].qs[idx] = qidx_sh[blk*block_qk + 2*idx] | (uint8_t)(qidx_sh[blk*block_qk + 2*idx + 1] << 4);
    }
    // QJL signs (low half)
    const int warp_id = t >> 5;
    if ((t & 31) == 0 && warp_id < warps_per_block) {
        dst_blocks[0].qjl[warp_id*4+0] = (uint8_t)((signs_lo>> 0)&0xFF);
        dst_blocks[0].qjl[warp_id*4+1] = (uint8_t)((signs_lo>> 8)&0xFF);
        dst_blocks[0].qjl[warp_id*4+2] = (uint8_t)((signs_lo>>16)&0xFF);
        dst_blocks[0].qjl[warp_id*4+3] = (uint8_t)((signs_lo>>24)&0xFF);
    }
    if (wht_dim > 128 && (t & 31) == 0 && warp_id < warps_per_block) {
        dst_blocks[1].qjl[warp_id*4+0] = (uint8_t)((signs_hi>> 0)&0xFF);
        dst_blocks[1].qjl[warp_id*4+1] = (uint8_t)((signs_hi>> 8)&0xFF);
        dst_blocks[1].qjl[warp_id*4+2] = (uint8_t)((signs_hi>>16)&0xFF);
        dst_blocks[1].qjl[warp_id*4+3] = (uint8_t)((signs_hi>>24)&0xFF);
    }
    if (t == 0) {
        const float sn = (recon_norm_sh > 1e-10f) ? (norm_sh / recon_norm_sh) : norm_sh;
        const ggml_half nh = __float2half(sn);
        const float sum_rsq = warp_buf[0]+warp_buf[1]+warp_buf[2]+warp_buf[3];
        // Scale rnorm by sqrt(128/wht_dim) so dequant's fixed PQ_TQ_DEQUANT_QJL_CORR=sqrt(2/(pi*128))
        // gives the correct sqrt(2/(pi*wht_dim)) correction for any D.
        const ggml_half rh = __float2half(norm_sh * sqrtf(sum_rsq * 128.0f / (float)wht_dim));
        for (int i = 0; i < n_tq4; i++) { dst_blocks[i].norm = nh; dst_blocks[i].rnorm = rh; }
    }
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
}

// --- tq2 set_rows ---
template <typename idx_t>
static __global__ void __launch_bounds__(128, 2) k_set_rows_tq2_qjl(
        const float        * __restrict__ src0,
        const idx_t        * __restrict__ src1,
        block_tq2     * __restrict__ dst,
        const int          wht_dim,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int t = threadIdx.x;
    const int64_t g = (int64_t)blockIdx.x;

    __shared__ float   sh[256];
    __shared__ float   warp_buf[4];
    __shared__ float   norm_sh;
    __shared__ float   recon_norm_sh;
    __shared__ uint8_t qidx_sh[256];

    const int64_t i_base = g * (int64_t)wht_dim;
    uint32_t tmp = (uint32_t)i_base;
    uint2 dm;
    dm = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y;
    const int64_t i03 = (int64_t)tmp;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10 = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src_block = (src0 + i01*s01 + i02*s02 + i03*s03) + i00;
    block_tq2 * dst_blocks = dst + (dst_row*s1 + i02*s2 + i03*s3) / (int64_t)sizeof(block_tq2) + i00 / QK_PQ_TQ_2;

    // D-aware load + norm + WHT
    pq_tq_set_rows_load_wht(sh, warp_buf, &norm_sh, src_block, t, wht_dim);

    // Quantize 2-bit + recon norm
    float rsq;
    pq_tq_set_rows_quant2(sh, qidx_sh, t, wht_dim, rsq);
    pq_tq_set_rows_reduce(rsq, warp_buf, t);
    if (t == 0) recon_norm_sh = sqrtf(warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3] + 1e-10f);
    __syncthreads();

    // QJL: residual + ballot signs
    const int n_blocks_2q = wht_dim / QK_PQ_TQ_2;
    const float inv_recon = 1.0f / (recon_norm_sh + 1e-10f);
    const float c_lo = (wht_dim <= 64 && t >= wht_dim) ? 0.0f : PQ_TQ_CENTROIDS_2BIT[qidx_sh[t]];
    const float residual_lo = (wht_dim <= 64 && t >= wht_dim) ? 0.0f
        : pq_tq_set_rows_qjl_residual_fix(sh[t], c_lo, inv_recon, wht_dim);
    const uint32_t signs_lo = __ballot_sync(0xffffffffu, residual_lo >= 0.0f);
    float rsq2 = residual_lo * residual_lo;
    float residual_hi = 0.0f;
    uint32_t signs_hi = 0;
    if (wht_dim > 128) {
        const float c_hi = PQ_TQ_CENTROIDS_2BIT[qidx_sh[t + 128]];
        residual_hi = pq_tq_set_rows_qjl_residual_fix(sh[t + 128], c_hi, inv_recon, wht_dim);
        signs_hi = __ballot_sync(0xffffffffu, residual_hi >= 0.0f);
        rsq2 += residual_hi * residual_hi;
    }
    pq_tq_set_rows_reduce(rsq2, warp_buf, t);

    // Pack qs
    const int pack_thr_2q = wht_dim / 4;
    if (t < pack_thr_2q) {
        const int base = t * 4;
        dst_blocks[t >> 3].qs[t & 7] = ((qidx_sh[base+0]&0x3u)<<0)|((qidx_sh[base+1]&0x3u)<<2)|((qidx_sh[base+2]&0x3u)<<4)|((qidx_sh[base+3]&0x3u)<<6);
    }
    // QJL signs (low half)
    const int warp_id = t >> 5;
    if (warp_id < n_blocks_2q && (t & 31) == 0) {
        dst_blocks[warp_id].qjl[0] = (uint8_t)((signs_lo>> 0)&0xFF);
        dst_blocks[warp_id].qjl[1] = (uint8_t)((signs_lo>> 8)&0xFF);
        dst_blocks[warp_id].qjl[2] = (uint8_t)((signs_lo>>16)&0xFF);
        dst_blocks[warp_id].qjl[3] = (uint8_t)((signs_lo>>24)&0xFF);
    }
    if (wht_dim > 128 && (t & 31) == 0) {
        dst_blocks[warp_id + n_blocks_2q/2].qjl[0] = (uint8_t)((signs_hi>> 0)&0xFF);
        dst_blocks[warp_id + n_blocks_2q/2].qjl[1] = (uint8_t)((signs_hi>> 8)&0xFF);
        dst_blocks[warp_id + n_blocks_2q/2].qjl[2] = (uint8_t)((signs_hi>>16)&0xFF);
        dst_blocks[warp_id + n_blocks_2q/2].qjl[3] = (uint8_t)((signs_hi>>24)&0xFF);
    }
    if (t == 0) {
        const float sn = (recon_norm_sh > 1e-10f) ? (norm_sh / recon_norm_sh) : norm_sh;
        const ggml_half nh = __float2half(sn);
        const float sum_rsq = warp_buf[0]+warp_buf[1]+warp_buf[2]+warp_buf[3];
        const ggml_half rh = __float2half(norm_sh * sqrtf(sum_rsq * 128.0f / (float)wht_dim));
        for (int i = 0; i < n_blocks_2q; i++) { dst_blocks[i].norm = nh; dst_blocks[i].rnorm = rh; }
    }
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
}

// --- tq3 set_rows ---
template <typename idx_t>
static __global__ void __launch_bounds__(128, 2) k_set_rows_tq3_qjl(
        const float        * __restrict__ src0,
        const idx_t        * __restrict__ src1,
        block_tq3     * __restrict__ dst,
        const int          wht_dim,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int t = threadIdx.x;
    const int64_t g = (int64_t)blockIdx.x;

    __shared__ float   sh[256];
    __shared__ float   warp_buf[4];
    __shared__ float   norm_sh;
    __shared__ float   recon_norm_sh;
    __shared__ uint8_t qidx_sh[256];

    const int64_t i_base = g * (int64_t)wht_dim;
    uint32_t tmp = (uint32_t)i_base;
    uint2 dm;
    dm = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y;
    const int64_t i03 = (int64_t)tmp;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10 = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src_block = (src0 + i01*s01 + i02*s02 + i03*s03) + i00;
    block_tq3 * dst_blocks = dst + (dst_row*s1 + i02*s2 + i03*s3) / (int64_t)sizeof(block_tq3) + i00 / QK_PQ_TQ_3;

    // D-aware load + norm + WHT
    pq_tq_set_rows_load_wht(sh, warp_buf, &norm_sh, src_block, t, wht_dim);

    // Quantize 3-bit + recon norm
    float rsq;
    pq_tq_set_rows_quant3(sh, qidx_sh, t, wht_dim, rsq);
    pq_tq_set_rows_reduce(rsq, warp_buf, t);
    if (t == 0) recon_norm_sh = sqrtf(warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3] + 1e-10f);
    __syncthreads();

    // QJL: residual + ballot signs
    const int n_blocks_3q = wht_dim / QK_PQ_TQ_3;
    const float inv_recon = 1.0f / (recon_norm_sh + 1e-10f);
    const float c_lo = (wht_dim <= 64 && t >= wht_dim) ? 0.0f : PQ_TQ_CENTROIDS_3BIT[qidx_sh[t]];
    const float residual_lo = (wht_dim <= 64 && t >= wht_dim) ? 0.0f
        : pq_tq_set_rows_qjl_residual_fix(sh[t], c_lo, inv_recon, wht_dim);
    const uint32_t signs_lo = __ballot_sync(0xffffffffu, residual_lo >= 0.0f);
    float rsq2 = residual_lo * residual_lo;
    float residual_hi = 0.0f;
    uint32_t signs_hi = 0;
    if (wht_dim > 128) {
        const float c_hi = PQ_TQ_CENTROIDS_3BIT[qidx_sh[t + 128]];
        residual_hi = pq_tq_set_rows_qjl_residual_fix(sh[t + 128], c_hi, inv_recon, wht_dim);
        signs_hi = __ballot_sync(0xffffffffu, residual_hi >= 0.0f);
        rsq2 += residual_hi * residual_hi;
    }
    pq_tq_set_rows_reduce(rsq2, warp_buf, t);

    // Pack qs
    const int pack_thr_3q = wht_dim / 4;
    if (t < pack_thr_3q) {
        const int base = t * 4;
        dst_blocks[t >> 3].qs[t & 7] = ((qidx_sh[base+0]&0x3u)<<0)|((qidx_sh[base+1]&0x3u)<<2)|((qidx_sh[base+2]&0x3u)<<4)|((qidx_sh[base+3]&0x3u)<<6);
    }
    // Pack signs
    const int sign_thr_3q = wht_dim / 8;
    if (t < sign_thr_3q) {
        const int base = (t >> 2) * 32 + (t & 3) * 8;
        dst_blocks[t>>2].signs[t&3] = ((qidx_sh[base+0]>>2)&1u)<<0|((qidx_sh[base+1]>>2)&1u)<<1|((qidx_sh[base+2]>>2)&1u)<<2|((qidx_sh[base+3]>>2)&1u)<<3
                                      |((qidx_sh[base+4]>>2)&1u)<<4|((qidx_sh[base+5]>>2)&1u)<<5|((qidx_sh[base+6]>>2)&1u)<<6|((qidx_sh[base+7]>>2)&1u)<<7;
    }
    // QJL signs (low half)
    const int warp_id = t >> 5;
    if (warp_id < n_blocks_3q && (t & 31) == 0) {
        dst_blocks[warp_id].qjl[0] = (uint8_t)((signs_lo>> 0)&0xFF);
        dst_blocks[warp_id].qjl[1] = (uint8_t)((signs_lo>> 8)&0xFF);
        dst_blocks[warp_id].qjl[2] = (uint8_t)((signs_lo>>16)&0xFF);
        dst_blocks[warp_id].qjl[3] = (uint8_t)((signs_lo>>24)&0xFF);
    }
    if (wht_dim > 128 && (t & 31) == 0) {
        dst_blocks[warp_id + n_blocks_3q/2].qjl[0] = (uint8_t)((signs_hi>> 0)&0xFF);
        dst_blocks[warp_id + n_blocks_3q/2].qjl[1] = (uint8_t)((signs_hi>> 8)&0xFF);
        dst_blocks[warp_id + n_blocks_3q/2].qjl[2] = (uint8_t)((signs_hi>>16)&0xFF);
        dst_blocks[warp_id + n_blocks_3q/2].qjl[3] = (uint8_t)((signs_hi>>24)&0xFF);
    }
    if (t == 0) {
        const float sn = (recon_norm_sh > 1e-10f) ? (norm_sh / recon_norm_sh) : norm_sh;
        const ggml_half nh = __float2half(sn);
        const float sum_rsq = warp_buf[0]+warp_buf[1]+warp_buf[2]+warp_buf[3];
        const ggml_half rh = __float2half(norm_sh * sqrtf(sum_rsq * 128.0f / (float)wht_dim));
        for (int i = 0; i < n_blocks_3q; i++) { dst_blocks[i].norm = nh; dst_blocks[i].rnorm = rh; }
    }
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
}

// Launchers for QJL variants
template<typename idx_t, typename block_tq4, int block_qk>
static void set_rows_cuda_tq4_qjl(
        const float * src0_d, const idx_t * src1_d, block_tq4 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, const int wht_dim, cudaStream_t stream) {
    GGML_ASSERT(ne00 % wht_dim == 0);
    GGML_ASSERT(wht_dim >= block_qk);
    const int64_t n_blocks = (ne00 * ne01 * ne02 * ne03) / wht_dim;
    const int64_t s01 = nb01/sizeof(float); const int64_t s02 = nb02/sizeof(float); const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t); const int64_t s11 = nb11/sizeof(idx_t); const int64_t s12 = nb12/sizeof(idx_t);
    if (n_blocks > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        k_set_rows_tq4_qjl<idx_t, block_tq4, block_qk><<<(int)n_blocks, 128, 0, stream>>>(
            src0_d, src1_d, dst_d, wht_dim, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3,
            init_fastdiv_values((uint32_t)ne00), init_fastdiv_values((uint32_t)ne01), init_fastdiv_values((uint32_t)ne02),
            init_fastdiv_values((uint32_t)ne11), init_fastdiv_values((uint32_t)ne12));
    }
}

template<typename idx_t>
static void set_rows_cuda_tq2_qjl(
        const float * src0_d, const idx_t * src1_d, block_tq2 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, const int wht_dim, cudaStream_t stream) {
    GGML_ASSERT(ne00 % wht_dim == 0);
    const int64_t n_groups = (ne00 * ne01 * ne02 * ne03) / wht_dim;
    const int64_t s01 = nb01/sizeof(float); const int64_t s02 = nb02/sizeof(float); const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t); const int64_t s11 = nb11/sizeof(idx_t); const int64_t s12 = nb12/sizeof(idx_t);
    if (n_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        k_set_rows_tq2_qjl<<<(int)n_groups, 128, 0, stream>>>(
            src0_d, src1_d, dst_d, wht_dim, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3,
            init_fastdiv_values((uint32_t)ne00), init_fastdiv_values((uint32_t)ne01), init_fastdiv_values((uint32_t)ne02),
            init_fastdiv_values((uint32_t)ne11), init_fastdiv_values((uint32_t)ne12));
    }
}

template<typename idx_t>
static void set_rows_cuda_tq3_qjl(
        const float * src0_d, const idx_t * src1_d, block_tq3 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3, const int wht_dim, cudaStream_t stream) {
    GGML_ASSERT(ne00 % wht_dim == 0);
    const int64_t n_groups = (ne00 * ne01 * ne02 * ne03) / wht_dim;
    const int64_t s01 = nb01/sizeof(float); const int64_t s02 = nb02/sizeof(float); const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t); const int64_t s11 = nb11/sizeof(idx_t); const int64_t s12 = nb12/sizeof(idx_t);
    if (n_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        k_set_rows_tq3_qjl<<<(int)n_groups, 128, 0, stream>>>(
            src0_d, src1_d, dst_d, wht_dim, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3,
            init_fastdiv_values((uint32_t)ne00), init_fastdiv_values((uint32_t)ne01), init_fastdiv_values((uint32_t)ne02),
            init_fastdiv_values((uint32_t)ne11), init_fastdiv_values((uint32_t)ne12));
    }
}

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_PQ2_0) {
        const int wht_dim = dst->op_params[0] > 0 ? dst->op_params[0] : 128;
        set_rows_cuda_pq2(
            src0_d, src1_d, (block_pq2*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            wht_dim, stream
        );
    } else if (dst->type == GGML_TYPE_PQ3_0) {
        const int wht_dim = dst->op_params[0] > 0 ? dst->op_params[0] : 128;
        set_rows_cuda_pq3(
            src0_d, src1_d, (block_pq3*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            wht_dim, stream
        );
    } else if (dst->type == GGML_TYPE_PQ4_0) {
        const int wht_dim = dst->op_params[0] > 0 ? dst->op_params[0] : 128;
        set_rows_cuda_pq4<idx_t, block_pq4, QK_PQ_TQ_4>(
            src0_d, src1_d, (block_pq4*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            wht_dim, stream
        );
    } else if (dst->type == GGML_TYPE_PQ4_0_64) {
        const int wht_dim = dst->op_params[0] > 0 ? dst->op_params[0] : 64;
        set_rows_cuda_pq4<idx_t, block_pq4_d64, QK_PQ_TQ_4_D64>(
            src0_d, src1_d, (block_pq4_d64*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            wht_dim, stream
        );
    } else if (dst->type == GGML_TYPE_TQ2_1) {
        const int wht_dim = dst->op_params[0] > 0 ? dst->op_params[0] : 128;
        set_rows_cuda_tq2_qjl(
            src0_d, src1_d, (block_tq2*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            wht_dim, stream
        );
    } else if (dst->type == GGML_TYPE_TQ3_1) {
        const int wht_dim = dst->op_params[0] > 0 ? dst->op_params[0] : 128;
        set_rows_cuda_tq3_qjl(
            src0_d, src1_d, (block_tq3*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            wht_dim, stream
        );
    } else if (dst->type == GGML_TYPE_TQ4_1) {
        const int wht_dim = dst->op_params[0] > 0 ? dst->op_params[0] : 128;
        set_rows_cuda_tq4_qjl<idx_t, block_tq4, QK_PQ_TQ_4>(
            src0_d, src1_d, (block_tq4*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            wht_dim, stream
        );
    } else if (dst->type == GGML_TYPE_TQ4_1_64) {
        const int wht_dim = dst->op_params[0] > 0 ? dst->op_params[0] : 64;
        set_rows_cuda_tq4_qjl<idx_t, block_tq4_d64, QK_PQ_TQ_4_D64>(
            src0_d, src1_d, (block_tq4_d64*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            wht_dim, stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
