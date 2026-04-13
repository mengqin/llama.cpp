#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// ===== PQ/TQ dequantize =====

__constant__ static const float PQ_TQ_CENTROIDS_2BIT_DEQNT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

__constant__ static const float PQ_TQ_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

__constant__ static const float PQ_TQ_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

#define QR_PQ2_0 1
#define QR_PQ3_0 1
#define QR_PQ4_0 1

static __device__ __forceinline__ void dequantize_pq2_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_pq2 * x = (const block_pq2 *) vx;
    const float norm = __half2float(x[ib].norm);

    const int qs_byte0  = iqs / 4;
    const int qs_shift0 = (iqs % 4) * 2;
    const int qs_byte1  = (iqs + 1) / 4;
    const int qs_shift1 = ((iqs + 1) % 4) * 2;

    const uint8_t q0 = (x[ib].qs[qs_byte0] >> qs_shift0) & 0x3;
    const uint8_t q1 = (x[ib].qs[qs_byte1] >> qs_shift1) & 0x3;

    v.x = PQ_TQ_CENTROIDS_2BIT_DEQNT[q0] * norm;
    v.y = PQ_TQ_CENTROIDS_2BIT_DEQNT[q1] * norm;
}

static __device__ __forceinline__ void dequantize_pq3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_pq3 * x = (const block_pq3 *) vx;
    const float norm = __half2float(x[ib].norm);

    const int qs_byte0  = iqs / 4;
    const int qs_shift0 = (iqs % 4) * 2;
    const int s_byte0   = iqs / 8;
    const int s_shift0  = iqs % 8;

    const int qs_byte1  = (iqs + 1) / 4;
    const int qs_shift1 = ((iqs + 1) % 4) * 2;
    const int s_byte1   = (iqs + 1) / 8;
    const int s_shift1  = (iqs + 1) % 8;

    const uint8_t q0 = (x[ib].qs[qs_byte0] >> qs_shift0) & 0x3;
    const uint8_t s0 = (x[ib].signs[s_byte0] >> s_shift0) & 0x1;
    const uint8_t q1 = (x[ib].qs[qs_byte1] >> qs_shift1) & 0x3;
    const uint8_t s1 = (x[ib].signs[s_byte1] >> s_shift1) & 0x1;

    v.x = PQ_TQ_CENTROIDS_3BIT[q0 | (s0 << 2)] * norm;
    v.y = PQ_TQ_CENTROIDS_3BIT[q1 | (s1 << 2)] * norm;
}

template <typename block_pq4, int block_qk>
static __device__ __forceinline__ void dequantize_pq4_0_impl(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_pq4 * x = (const block_pq4 *) vx;
    const float norm = __half2float(x[ib].norm);
    const uint8_t qb = x[ib].qs[iqs / 2];

    v.x = PQ_TQ_CENTROIDS_4BIT[qb & 0xF] * norm;
    v.y = PQ_TQ_CENTROIDS_4BIT[qb >> 4]  * norm;
    GGML_UNUSED(block_qk);
}

static __device__ __forceinline__ void dequantize_pq4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_pq4_0_impl<block_pq4, QK_PQ_TQ_4>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_pq4_0_64(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_pq4_0_impl<block_pq4_d64, QK_PQ_TQ_4_D64>(vx, ib, iqs, v);
}

static constexpr float TQ_QJL_CORR = 0.0705348f;

#define QR_TQ2_1 1
#define QR_TQ3_1 1
#define QR_TQ4_1 1

static __device__ __forceinline__ void dequantize_tq2_1(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq2 * x = (const block_tq2 *) vx;
    const float norm  = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);

    const int qs_byte0 = iqs / 4;
    const int qs_shift0 = (iqs % 4) * 2;
    const int qs_byte1 = (iqs + 1) / 4;
    const int qs_shift1 = ((iqs + 1) % 4) * 2;
    const uint8_t q0 = (x[ib].qs[qs_byte0] >> qs_shift0) & 0x3;
    const uint8_t q1 = (x[ib].qs[qs_byte1] >> qs_shift1) & 0x3;

    const int qjl_byte = iqs / 8;
    const uint8_t s0 = (x[ib].qjl[qjl_byte] >> (iqs % 8)) & 1u;
    const uint8_t s1 = (x[ib].qjl[qjl_byte] >> ((iqs + 1) % 8)) & 1u;

    v.x = PQ_TQ_CENTROIDS_2BIT_DEQNT[q0] * norm + (2.0f*s0 - 1.0f) * rnorm * TQ_QJL_CORR;
    v.y = PQ_TQ_CENTROIDS_2BIT_DEQNT[q1] * norm + (2.0f*s1 - 1.0f) * rnorm * TQ_QJL_CORR;
}

static __device__ __forceinline__ void dequantize_tq3_1(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq3 * x = (const block_tq3 *) vx;
    const float norm  = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);

    const int qs_byte0 = iqs / 4;
    const int qs_shift0 = (iqs % 4) * 2;
    const int s_byte0 = iqs / 8;
    const int s_shift0 = iqs % 8;
    const int qs_byte1 = (iqs + 1) / 4;
    const int qs_shift1 = ((iqs + 1) % 4) * 2;
    const int s_byte1 = (iqs + 1) / 8;
    const int s_shift1 = (iqs + 1) % 8;

    const uint8_t q0 = (x[ib].qs[qs_byte0] >> qs_shift0) & 0x3;
    const uint8_t h0 = (x[ib].signs[s_byte0] >> s_shift0) & 0x1;
    const uint8_t q1 = (x[ib].qs[qs_byte1] >> qs_shift1) & 0x3;
    const uint8_t h1 = (x[ib].signs[s_byte1] >> s_shift1) & 0x1;
    const uint8_t j0 = (x[ib].qjl[iqs / 8] >> (iqs % 8)) & 1u;
    const uint8_t j1 = (x[ib].qjl[(iqs + 1) / 8] >> ((iqs + 1) % 8)) & 1u;

    v.x = PQ_TQ_CENTROIDS_3BIT[q0 | (h0 << 2)] * norm + (2.0f*j0 - 1.0f) * rnorm * TQ_QJL_CORR;
    v.y = PQ_TQ_CENTROIDS_3BIT[q1 | (h1 << 2)] * norm + (2.0f*j1 - 1.0f) * rnorm * TQ_QJL_CORR;
}

template <typename block_tq4, int block_qk>
static __device__ __forceinline__ void dequantize_tq4_1_impl(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq4 * x = (const block_tq4 *) vx;
    const float norm  = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const uint8_t qb = x[ib].qs[iqs / 2];
    const int qjl_byte = iqs / 8;
    const uint8_t j0 = (x[ib].qjl[qjl_byte] >> (iqs % 8)) & 1u;
    const uint8_t j1 = (x[ib].qjl[qjl_byte] >> ((iqs + 1) % 8)) & 1u;

    v.x = PQ_TQ_CENTROIDS_4BIT[qb & 0xF] * norm + (2.0f*j0 - 1.0f) * rnorm * TQ_QJL_CORR;
    v.y = PQ_TQ_CENTROIDS_4BIT[qb >> 4]  * norm + (2.0f*j1 - 1.0f) * rnorm * TQ_QJL_CORR;
    GGML_UNUSED(block_qk);
}

static __device__ __forceinline__ void dequantize_tq4_1(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tq4_1_impl<block_tq4, QK_PQ_TQ_4>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tq4_1_64(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tq4_1_impl<block_tq4_d64, QK_PQ_TQ_4_D64>(vx, ib, iqs, v);
}
