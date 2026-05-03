#include "quantize.cuh"
#include "pq-tq-fwht.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstring>

static __device__ __forceinline__ float pqk_wht_apply_sign_256(const float x, const uint32_t * signs, const int idx) {
    const uint32_t sign = ((signs[idx >> 5] >> (idx & 31)) & 1u) << 31;
    return __uint_as_float(__float_as_uint(x) ^ sign);
}

static __device__ __forceinline__ void pqk_coop_wht_forward_256_fast(float * sh, const int tid, float & xr0, float & xr1) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    constexpr unsigned mask = 0xFFFFFFFFu;
    const int lane = tid & (WARP_SIZE - 1);

    float x0 = pqk_wht_apply_sign_256(sh[tid],       PQ_TQ_WHT_SIGNS1_256_BITS, tid);
    float x1 = pqk_wht_apply_sign_256(sh[tid + 128], PQ_TQ_WHT_SIGNS1_256_BITS, tid + 128);

#pragma unroll
    for (int h = 1; h < WARP_SIZE; h <<= 1) {
        const float y0 = __shfl_xor_sync(mask, x0, h, WARP_SIZE);
        const float y1 = __shfl_xor_sync(mask, x1, h, WARP_SIZE);
        x0 = (lane & h) == 0 ? x0 + y0 : y0 - x0;
        x1 = (lane & h) == 0 ? x1 + y1 : y1 - x1;
    }

    sh[tid]       = x0;
    sh[tid + 128] = x1;
    __syncthreads();

#pragma unroll
    for (int h = WARP_SIZE; h < 128; h <<= 1) {
        const float a0 = sh[tid],       b0 = sh[tid ^ h];
        const float a1 = sh[tid + 128], b1 = sh[(tid ^ h) + 128];
        __syncthreads();
        sh[tid]       = ((tid & h) == 0) ? (a0 + b0) : (b0 - a0);
        sh[tid + 128] = ((tid & h) == 0) ? (a1 + b1) : (b1 - a1);
        __syncthreads();
    }

    const float lo = sh[tid], hi = sh[tid + 128];
    constexpr float inv_sqrt256 = 0.0625f;
    xr0 = pqk_wht_apply_sign_256((lo + hi) * inv_sqrt256, PQ_TQ_WHT_SIGNS2_256_BITS, tid);
    xr1 = pqk_wht_apply_sign_256((lo - hi) * inv_sqrt256, PQ_TQ_WHT_SIGNS2_256_BITS, tid + 128);
#else
    pq_tq_coop_wht_forward<256>(sh, tid);
    xr0 = sh[tid];
    xr1 = sh[tid + 128];
#endif
}

static bool ggml_cuda_quant_wht_type_supported(const ggml_type type) {
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

static void ggml_cuda_quant_wht_log_once(const ggml_type type, const char * path) {
    if (getenv("GGML_CUDA_LOG_QUANT_WHT") == nullptr) {
        return;
    }
    static bool logged_mmvq[GGML_TYPE_COUNT] = {};
    static bool logged_mmq[GGML_TYPE_COUNT]  = {};
    const int type_idx = (int) type;
    if (type_idx < 0 || type_idx >= GGML_TYPE_COUNT) {
        return;
    }
    bool * logged = strcmp(path, "MMQ") == 0 ? logged_mmq : logged_mmvq;
    if (logged[type_idx]) {
        return;
    }
    logged[type_idx] = true;
    GGML_LOG_INFO("%s: quant_wht enabled, dim=256, type=%s, activation WHT preprocess=yes, path=%s\n",
            __func__, ggml_type_name(type), path);
}

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

template<int D>
__launch_bounds__(128, 1)
static __global__ void quantize_q8_1_pq_wht(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    static_assert(D == 128, "unsupported PQ WHT group size");

    const int tid = threadIdx.x;
    const int64_t base_i0 = (int64_t) blockIdx.x * D;

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    __shared__ float sh[D];

    const int64_t i0 = base_i0 + tid;
    sh[tid] = i0 < ne00 ? x[i3*s03 + i2*s02 + i1*s01 + i0] : 0.0f;
    __syncthreads();

    pq_tq_coop_wht_forward<D>(sh, tid);

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const float xr = sh[tid];

    float amax = fabsf(xr);
    float sum  = xr;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xr / d);

    block_q8_1 * y = (block_q8_1 *) vy;
    const int64_t ib_base = (((i3*ne2.z + i2) * ne1 + i1) * ne0 + base_i0) / QK8_1;
    y[ib_base + warp_id].qs[lane_id] = q;

    if (lane_id == 0) {
        y[ib_base + warp_id].ds = make_half2(d, sum);
    }
}

__launch_bounds__(128, 1)
static __global__ void quantize_q8_1_pqk_wht(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    constexpr int D = QK_K;

    const int tid = threadIdx.x;
    const int64_t base_i0 = (int64_t) blockIdx.x * D;

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    __shared__ float sh[D];

    const int64_t i0 = base_i0 + tid;
    sh[tid]       = i0       < ne00 ? x[i3*s03 + i2*s02 + i1*s01 + i0]       : 0.0f;
    sh[tid + 128] = i0 + 128 < ne00 ? x[i3*s03 + i2*s02 + i1*s01 + i0 + 128] : 0.0f;
    __syncthreads();

    float xr0;
    float xr1;
    pqk_coop_wht_forward_256_fast(sh, tid, xr0, xr1);

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    float amax0 = fabsf(xr0);
    float amax1 = fabsf(xr1);
    float sum0  = xr0;
    float sum1  = xr1;
    amax0 = warp_reduce_max<QK8_1>(amax0);
    amax1 = warp_reduce_max<QK8_1>(amax1);
    sum0  = warp_reduce_sum<QK8_1>(sum0);
    sum1  = warp_reduce_sum<QK8_1>(sum1);

    const float d0 = amax0 / 127.0f;
    const float d1 = amax1 / 127.0f;
    const int8_t q0 = amax0 == 0.0f ? 0 : roundf(xr0 / d0);
    const int8_t q1 = amax1 == 0.0f ? 0 : roundf(xr1 / d1);

    block_q8_1 * y = (block_q8_1 *) vy;
    const int64_t ib_base = (((i3*ne2.z + i2) * ne1 + i1) * ne0 + base_i0) / QK8_1;
    y[ib_base + warp_id].qs[lane_id] = q0;
    y[ib_base + 4 + warp_id].qs[lane_id] = q1;

    if (lane_id == 0) {
        y[ib_base + warp_id].ds = make_half2(d0, sum0);
        y[ib_base + 4 + warp_id].ds = make_half2(d1, sum1);
    }
}

__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    // FP4 E2M1: max exponent (unbiased) is 2.
    constexpr int FP4_E2M1_EMAX = 2;

    const float e = log2f(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = __float2int_rn(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return static_cast<uint8_t>(biased);
}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_fp4_mmq * y = (block_fp4_mmq *) vy;

    const int64_t block_fp4_mmq_size = 8 * QK_MXFP4;  // 256 values
    const int64_t ib0                = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
    const int64_t ib = ib0 + (warp_start_offset / block_fp4_mmq_size) * ne1 + blockIdx.x;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;
    char2 * yqs2 = (char2 *) y[ib].qs;

    const int64_t base_pos = i03 * s03 + i02 * s02 + i01 * s01;

    uint8_t scales[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i0 = warp_start_offset + b * vals_per_scale + lane_id_32;
        const float xi = (i0 < ne00) ? x[base_pos + i0] : 0.0f;

        float amax = fabsf(xi);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e = compute_e8m0_scale(amax);
        scales[b] = e;
        const float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float scaled_val = xi * inv_s;

        const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
        const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
        const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
        const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));

            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = *(char2 *) &fp4_packed;
        }
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            char2 q;
            q.x = (q_hi_0 << 4) | q_lo_0;
            q.y = (q_hi_1 << 4) | q_lo_1;
            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = q;
        }
#endif // CUDART_VERSION >= 12080
    }

    if (lane_id_32 == 0) {
        // Store 2 scales packed into 1 uint32
        y[ib].d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
    }
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

template<int D>
__launch_bounds__(128, 1)
static __global__ void quantize_mmq_q8_1_pq_wht(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    static_assert(D == 128, "unsupported PQ WHT group size");

    const int tid = threadIdx.x;
    const int64_t base_i0 = (int64_t) blockIdx.y * D;

    if (base_i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = ids ? ids[i1] : i1;

    __shared__ float sh[D];

    const int64_t i0 = base_i0 + tid;
    sh[tid] = i0 < ne00 ? x[i3*s03 + i2*s02 + i01*s01 + i0] : 0.0f;
    __syncthreads();

    pq_tq_coop_wht_forward<D>(sh, tid);

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
    const int64_t ib = blockIdx.z * ((int64_t) ne1 * (ne0 / D)) + (base_i0 / D) * ne1 + blockIdx.x;

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const float xr = sh[tid];

    float amax = fabsf(xr);
    amax = warp_reduce_max<QK8_1>(amax);

    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xr / d);

    y[ib].qs[warp_id * QK8_1 + lane_id] = q;

    if (lane_id == 0) {
        y[ib].d4[warp_id] = d;
    }
}

template <mmq_q8_1_ds_layout ds_layout>
__launch_bounds__(128, 1)
static __global__ void quantize_mmq_q8_1_pqk_wht(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    constexpr int D = QK_K;

    const int tid = threadIdx.x;
    const int64_t base_i0 = (int64_t) blockIdx.y * D;

    if (base_i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = ids ? ids[i1] : i1;

    __shared__ float sh[D];
    __shared__ float amax32[8];

    const int64_t i0 = base_i0 + tid;
    sh[tid]       = i0       < ne00 ? x[i3*s03 + i2*s02 + i01*s01 + i0]       : 0.0f;
    sh[tid + 128] = i0 + 128 < ne00 ? x[i3*s03 + i2*s02 + i01*s01 + i0 + 128] : 0.0f;
    __syncthreads();

    float xr0;
    float xr1;
    pqk_coop_wht_forward_256_fast(sh, tid, xr0, xr1);

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
    const int64_t ib0 = blockIdx.z * ((int64_t) ne1 * (ne0 / (4*QK8_1)))
                      + (base_i0 / (4*QK8_1)) * ne1 + blockIdx.x;
    const int64_t ib1 = ib0 + ne1;

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    float amax0 = fabsf(xr0);
    float amax1 = fabsf(xr1);
    float sum0  = xr0;
    float sum1  = xr1;
    amax0 = warp_reduce_max<QK8_1>(amax0);
    amax1 = warp_reduce_max<QK8_1>(amax1);
    sum0  = warp_reduce_sum<QK8_1>(sum0);
    sum1  = warp_reduce_sum<QK8_1>(sum1);

    float d0 = amax0 / 127.0f;
    float d1 = amax1 / 127.0f;

    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (lane_id == 0) {
            amax32[warp_id]     = amax0;
            amax32[warp_id + 4] = amax1;
        }
        __syncthreads();

        const int scale_group = warp_id / 2;
        d0 = fmaxf(amax32[2*scale_group],     amax32[2*scale_group + 1]) / 127.0f;
        d1 = fmaxf(amax32[2*scale_group + 4], amax32[2*scale_group + 5]) / 127.0f;
    }

    const int8_t q0 = amax0 == 0.0f ? 0 : roundf(xr0 / d0);
    const int8_t q1 = amax1 == 0.0f ? 0 : roundf(xr1 / d1);

    y[ib0].qs[warp_id * QK8_1 + lane_id] = q0;
    y[ib1].qs[warp_id * QK8_1 + lane_id] = q1;

    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        float sum16_0 = xr0;
        float sum16_1 = xr1;
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            sum16_0 += __shfl_xor_sync(0xFFFFFFFF, sum16_0, offset, 16);
            sum16_1 += __shfl_xor_sync(0xFFFFFFFF, sum16_1, offset, 16);
        }

        if (lane_id == 0 && (warp_id & 1) == 0) {
            y[ib0].d2s6[warp_id / 2] = d0;
            y[ib1].d2s6[warp_id / 2] = d1;
        }
        const int sum_group = warp_id * 2 + lane_id / 16;
        if ((lane_id & 15) == 0 && sum_group < 6) {
            y[ib0].d2s6[2 + sum_group] = sum16_0;
            y[ib1].d2s6[2 + sum_group] = sum16_1;
        }
    } else if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        if (lane_id == 0) {
            y[ib0].ds4[warp_id] = make_half2(d0, sum0);
            y[ib1].ds4[warp_id] = make_half2(d1, sum1);
        }
    } else {
        if (lane_id == 0) {
            y[ib0].d4[warp_id] = d0;
            y[ib1].d4[warp_id] = d1;
        }
    }
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, const bool quant_wht, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    if (type_src0 == GGML_TYPE_PQ2_0 || type_src0 == GGML_TYPE_PQ3_0 || type_src0 == GGML_TYPE_PQ4_0) {
        GGML_ASSERT(ne00 % QK_PQ_TQ_2_GROUP == 0);
        GGML_ASSERT(ne0  % QK_PQ_TQ_2_GROUP == 0);

        const dim3 num_blocks(ne0 / QK_PQ_TQ_2_GROUP, ne1, ne2*ne3);
        const dim3 block_size(QK_PQ_TQ_2_GROUP, 1, 1);
        quantize_q8_1_pq_wht<QK_PQ_TQ_2_GROUP><<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
        return;
    }

    if (type_src0 == GGML_TYPE_PQ2_K || type_src0 == GGML_TYPE_PQ3_K || type_src0 == GGML_TYPE_PQ4_K ||
            (quant_wht && ggml_cuda_quant_wht_type_supported(type_src0))) {
        GGML_ASSERT(ne00 % QK_K == 0);
        GGML_ASSERT(ne0  % QK_K == 0);

        if (quant_wht && ggml_cuda_quant_wht_type_supported(type_src0)) {
            ggml_cuda_quant_wht_log_once(type_src0, "MMVQ");
        }
        const dim3 num_blocks(ne0 / QK_K, ne1, ne2*ne3);
        const dim3 block_size(128, 1, 1);
        quantize_q8_1_pqk_wht<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
        return;
    }

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, const bool quant_wht, cudaStream_t stream) {
    if (type_src0 == GGML_TYPE_PQ2_0 || type_src0 == GGML_TYPE_PQ3_0 || type_src0 == GGML_TYPE_PQ4_0) {
        GGML_ASSERT(ne00 % QK_PQ_TQ_2_GROUP == 0);
        GGML_ASSERT(ne0  % QK_PQ_TQ_2_GROUP == 0);

        const dim3 num_blocks(ne1, ne0 / QK_PQ_TQ_2_GROUP, ne2*ne3);
        const dim3 block_size(QK_PQ_TQ_2_GROUP, 1, 1);
        quantize_mmq_q8_1_pq_wht<QK_PQ_TQ_2_GROUP>
            <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
        return;
    }

    if (type_src0 == GGML_TYPE_PQ2_K || type_src0 == GGML_TYPE_PQ3_K || type_src0 == GGML_TYPE_PQ4_K ||
            (quant_wht && ggml_cuda_quant_wht_type_supported(type_src0))) {
        GGML_ASSERT(ne00 % QK_K == 0);
        GGML_ASSERT(ne0  % QK_K == 0);

        if (quant_wht && ggml_cuda_quant_wht_type_supported(type_src0)) {
            ggml_cuda_quant_wht_log_once(type_src0, "MMQ");
        }
        const dim3 num_blocks(ne1, ne0 / QK_K, ne2*ne3);
        const dim3 block_size(128, 1, 1);
        switch (mmq_get_q8_1_ds_layout(type_src0)) {
            case MMQ_Q8_1_DS_LAYOUT_D4:
                quantize_mmq_q8_1_pqk_wht<MMQ_Q8_1_DS_LAYOUT_D4>
                    <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
                break;
            case MMQ_Q8_1_DS_LAYOUT_DS4:
                quantize_mmq_q8_1_pqk_wht<MMQ_Q8_1_DS_LAYOUT_DS4>
                    <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
                break;
            case MMQ_Q8_1_DS_LAYOUT_D2S6:
                quantize_mmq_q8_1_pqk_wht<MMQ_Q8_1_DS_LAYOUT_D2S6>
                    <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }

    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_mmq_mxfp4_cuda(const float *                    x,
                             const int32_t *                  ids,
                             void *                           vy,
                             [[maybe_unused]] const ggml_type type_src0,
                             const int64_t                    ne00,
                             const int64_t                    s01,
                             const int64_t                    s02,
                             const int64_t                    s03,
                             const int64_t                    ne0,
                             const int64_t                    ne1,
                             const int64_t                    ne2,
                             const int64_t                    ne3,
                             [[maybe_unused]] const bool       quant_wht,
                             cudaStream_t                     stream) {
    GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

    constexpr int nwarps = 8;
    constexpr int vals_per_warp  = 2 * QK_MXFP4;
    constexpr int vals_per_block = nwarps * vals_per_warp;

    const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
    const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3    block_size(WARP_SIZE, nwarps, 1);

    quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
