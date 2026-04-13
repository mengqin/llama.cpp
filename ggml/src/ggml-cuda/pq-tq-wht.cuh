#pragma once

#include "common.cuh"
#include "pq-tq-common.cuh"
#include "pq-tq-fwht.cuh"

static __global__ void k_pq_tq_wht(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n_groups,
        const int direction,
        const int wht_dim) {
    __shared__ float sh[256];

    const int tid = threadIdx.x;
    const int group = blockIdx.x;

    if (group >= n_groups) {
        return;
    }

    const int base = group * wht_dim;

    if (wht_dim <= 64) {
        if (tid < wht_dim) {
            sh[tid] = src[base + tid];
        }
    } else if (wht_dim <= 128) {
        sh[tid] = src[base + tid];
    } else {
        sh[tid]       = src[base + tid];
        sh[tid + 128] = src[base + tid + 128];
    }
    __syncthreads();

    if (direction == 0) {
        if (wht_dim == 64) {
            pq_tq_coop_wht_forward<64>(sh, tid);
        } else if (wht_dim == 256) {
            pq_tq_coop_wht_forward<256>(sh, tid);
        } else {
            pq_tq_coop_wht_forward<128>(sh, tid);
        }
    } else {
        if (wht_dim == 64) {
            pq_tq_coop_wht_inverse<64>(sh, tid);
        } else if (wht_dim == 256) {
            pq_tq_coop_wht_inverse<256>(sh, tid);
        } else {
            pq_tq_coop_wht_inverse<128>(sh, tid);
        }
    }

    if (wht_dim <= 64) {
        if (tid < wht_dim) {
            dst[base + tid] = sh[tid];
        }
    } else if (wht_dim <= 128) {
        dst[base + tid] = sh[tid];
    } else {
        dst[base + tid]       = sh[tid];
        dst[base + tid + 128] = sh[tid + 128];
    }
}

static void ggml_cuda_pq_tq_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int wht_dim = (int) src->ne[0];
    GGML_ASSERT(wht_dim == 64 || wht_dim == 128 || wht_dim == 256);

    const int direction = ggml_get_op_params_i32(dst, 0);

    const int64_t n_elements = ggml_nelements(src);
    GGML_ASSERT(n_elements % wht_dim == 0);
    const int64_t n_groups = n_elements / wht_dim;

    const float * src_d = (const float *) src->data;
    float * dst_d = (float *) dst->data;

    k_pq_tq_wht<<<n_groups, 128, 0, ctx.stream()>>>(src_d, dst_d, n_groups, direction, wht_dim);
}
