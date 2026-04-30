#pragma once

#include <stdint.h>
#include <math.h>

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
#define GGML_PQK_HOST_DEVICE __host__ __device__
#else
#define GGML_PQK_HOST_DEVICE
#endif

#define GGML_PQK_SUBBLOCK_SIZE 16
#define GGML_PQK_SUBBLOCK_COUNT (QK_K / GGML_PQK_SUBBLOCK_SIZE)
#define GGML_PQK_BAND_COUNT 2
#define GGML_PQK_SUBBLOCKS_PER_BAND (GGML_PQK_SUBBLOCK_COUNT / GGML_PQK_BAND_COUNT)
#define GGML_PQK_SCALE_LEVELS 64
#define GGML_PQK_LOG_SCALE_MIN (-4.0f)
#define GGML_PQK_LOG_SCALE_MAX (0.0f)
#define GGML_PQK_LOG_SCALE_STEP ((GGML_PQK_LOG_SCALE_MAX - GGML_PQK_LOG_SCALE_MIN) / 62.0f)

#define GGML_PQ2_K_SUBBLOCK_SIZE 8
#define GGML_PQ2_K_SUBBLOCK_COUNT (QK_K / GGML_PQ2_K_SUBBLOCK_SIZE)
#define GGML_PQ2_K_SUBBLOCKS_PER_BAND (GGML_PQ2_K_SUBBLOCK_COUNT / GGML_PQK_BAND_COUNT)
#define GGML_PQ2_K_SCALE_LEVELS 16
#define GGML_PQ2_K_LOG_SCALE_MIN (-4.0f)
#define GGML_PQ2_K_LOG_SCALE_MAX (0.0f)
#define GGML_PQ2_K_LOG_SCALE_STEP ((GGML_PQ2_K_LOG_SCALE_MAX - GGML_PQ2_K_LOG_SCALE_MIN) / 14.0f)

#define GGML_PQ3_K_SUBBLOCK_SIZE 8
#define GGML_PQ3_K_SUBBLOCK_COUNT (QK_K / GGML_PQ3_K_SUBBLOCK_SIZE)
#define GGML_PQ3_K_SUBBLOCKS_PER_BAND (GGML_PQ3_K_SUBBLOCK_COUNT / GGML_PQK_BAND_COUNT)
#define GGML_PQ3_K_SCALE_LEVELS 16
#define GGML_PQ3_K_LOG_SCALE_MIN (-4.0f)
#define GGML_PQ3_K_LOG_SCALE_MAX (0.0f)
#define GGML_PQ3_K_LOG_SCALE_STEP ((GGML_PQ3_K_LOG_SCALE_MAX - GGML_PQ3_K_LOG_SCALE_MIN) / 14.0f)

#define GGML_PQ4_K_SUBBLOCK_SIZE 8
#define GGML_PQ4_K_SUBBLOCK_COUNT (QK_K / GGML_PQ4_K_SUBBLOCK_SIZE)
#define GGML_PQ4_K_SUBBLOCKS_PER_BAND (GGML_PQ4_K_SUBBLOCK_COUNT / GGML_PQK_BAND_COUNT)
#define GGML_PQ4_K_SCALE_LEVELS 16
#define GGML_PQ4_K_LOG_SCALE_MIN (-4.0f)
#define GGML_PQ4_K_LOG_SCALE_MAX (0.0f)
#define GGML_PQ4_K_LOG_SCALE_STEP ((GGML_PQ4_K_LOG_SCALE_MAX - GGML_PQ4_K_LOG_SCALE_MIN) / 14.0f)

#define GGML_PQ2K_MAX_CENTROID 1.468733483f
#define GGML_PQ3K_MAX_CENTROID 1.996037246f
#define GGML_PQ4K_MAX_CENTROID 2.413635748f

static GGML_PQK_HOST_DEVICE inline uint8_t ggml_pqk_bits_get(const uint8_t * data, int idx, int bits) {
    const int bit0 = idx * bits;
    uint8_t value = 0;
    for (int bit = 0; bit < bits; ++bit) {
        const int pos = bit0 + bit;
        value |= (uint8_t)(((data[pos >> 3] >> (pos & 7)) & 0x1u) << bit);
    }
    return value;
}

static inline void ggml_pqk_bits_set(uint8_t * data, int idx, int bits, uint8_t value) {
    const int bit0 = idx * bits;
    for (int bit = 0; bit < bits; ++bit) {
        const int pos = bit0 + bit;
        const uint8_t mask = (uint8_t)(1u << (pos & 7));
        const int byte = pos >> 3;
        if ((value >> bit) & 0x1u) {
            data[byte] |= mask;
        } else {
            data[byte] &= (uint8_t) ~mask;
        }
    }
}

static GGML_PQK_HOST_DEVICE inline uint8_t ggml_pqk_scale_get(const uint8_t * scales, int idx) {
    return ggml_pqk_bits_get(scales, idx, 6);
}

static inline void ggml_pqk_scale_set(uint8_t * scales, int idx, uint8_t value) {
    ggml_pqk_bits_set(scales, idx, 6, value);
}

static GGML_PQK_HOST_DEVICE inline uint8_t ggml_pq2_k_scale_get(const uint8_t * scales, int idx) {
    return ggml_pqk_bits_get(scales, idx, 4);
}

static inline void ggml_pq2_k_scale_set(uint8_t * scales, int idx, uint8_t value) {
    ggml_pqk_bits_set(scales, idx, 4, value);
}

static GGML_PQK_HOST_DEVICE inline uint8_t ggml_pq3_k_scale_get(const uint8_t * scales, int idx) {
    return ggml_pqk_bits_get(scales, idx, 4);
}

static inline void ggml_pq3_k_scale_set(uint8_t * scales, int idx, uint8_t value) {
    ggml_pqk_bits_set(scales, idx, 4, value);
}

static GGML_PQK_HOST_DEVICE inline uint8_t ggml_pq4_k_scale_get(const uint8_t * scales, int idx) {
    return ggml_pqk_bits_get(scales, idx, 4);
}

static inline void ggml_pq4_k_scale_set(uint8_t * scales, int idx, uint8_t value) {
    ggml_pqk_bits_set(scales, idx, 4, value);
}

static GGML_PQK_HOST_DEVICE inline float ggml_pqk_decode_local_scale(float master, uint8_t q) {
    if (master <= 0.0f || q == 0) {
        return 0.0f;
    }
    return master * exp2f(GGML_PQK_LOG_SCALE_MIN + (float)(q - 1) * GGML_PQK_LOG_SCALE_STEP);
}

static GGML_PQK_HOST_DEVICE inline float ggml_pq2_k_decode_local_scale(float master, uint8_t q) {
    if (master <= 0.0f || q == 0) {
        return 0.0f;
    }
    return master * exp2f(GGML_PQ2_K_LOG_SCALE_MIN + (float)(q - 1) * GGML_PQ2_K_LOG_SCALE_STEP);
}

static GGML_PQK_HOST_DEVICE inline float ggml_pq3_k_decode_local_scale(float master, uint8_t q) {
    if (master <= 0.0f || q == 0) {
        return 0.0f;
    }
    return master * exp2f(GGML_PQ3_K_LOG_SCALE_MIN + (float)(q - 1) * GGML_PQ3_K_LOG_SCALE_STEP);
}

static GGML_PQK_HOST_DEVICE inline float ggml_pq4_k_decode_local_scale(float master, uint8_t q) {
    if (master <= 0.0f || q == 0) {
        return 0.0f;
    }
    return master * exp2f(GGML_PQ4_K_LOG_SCALE_MIN + (float)(q - 1) * GGML_PQ4_K_LOG_SCALE_STEP);
}

// Universal PQK codebooks generated from the 16D spherical model used by
// scripts/train_pqk_universal_codebooks.py.
static GGML_PQK_HOST_DEVICE inline float ggml_pqk_centroid_2bit(uint8_t q) {
    switch (q & 0x3u) {
        case 0: return -1.468733483f;
        case 1: return -0.450043231f;
        case 2: return  0.450043231f;
        default: return 1.468733483f;
    }
}

static GGML_PQK_HOST_DEVICE inline float ggml_pqk_centroid_3bit(uint8_t q) {
    switch (q & 0x7u) {
        case 0: return -1.996037246f;
        case 1: return -1.286192434f;
        case 2: return -0.734445201f;
        case 3: return -0.239624284f;
        case 4: return  0.239624284f;
        case 5: return  0.734445201f;
        case 6: return  1.286192434f;
        default: return 1.996037246f;
    }
}

static GGML_PQK_HOST_DEVICE inline float ggml_pqk_centroid_4bit(uint8_t q) {
    switch (q & 0xFu) {
        case 0:  return -2.413635748f;
        case 1:  return -1.895488127f;
        case 2:  return -1.512641483f;
        case 3:  return -1.189715161f;
        case 4:  return -0.900380605f;
        case 5:  return -0.631294575f;
        case 6:  return -0.374426627f;
        case 7:  return -0.124176311f;
        case 8:  return  0.124176311f;
        case 9:  return  0.374426627f;
        case 10: return  0.631294575f;
        case 11: return  0.900380605f;
        case 12: return  1.189715161f;
        case 13: return  1.512641483f;
        case 14: return  1.895488127f;
        default: return 2.413635748f;
    }
}
