#pragma once

#include "common.cuh"

// ===== CONSTANT MEMORY: Precomputed centroids and midpoints =====

// 2-bit centroids: Lloyd-Max for N(0, 1/128)
__constant__ static const float PQ_TQ_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

__constant__ static const float PQ_TQ_MAG_2BIT[2] = {
    0.039994f, 0.133462f
};

// 3-bit centroids: Lloyd-Max for N(0, 1/128)
__constant__ static const float PQ_TQ_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

__constant__ static const float PQ_TQ_MAG_3BIT[4] = {
    0.021460f, 0.065717f, 0.117832f, 0.190685f
};

// 3-bit midpoints (7 values define 8 regions)
__constant__ static const float PQ_TQ_MID_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// 4-bit centroids: 16 optimal centroids for PolarQuant
__constant__ static const float PQ_TQ_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

__constant__ static const float PQ_TQ_MAG_4BIT[8] = {
    0.006938f, 0.020989f, 0.035597f, 0.051262f,
    0.068756f, 0.089527f, 0.117195f, 0.173926f
};

static constexpr float PQ_TQ_DP4A_SCALE_2BIT = 914.0f;
static constexpr float PQ_TQ_DP4A_INV_SCALE_2BIT = 1.0f / PQ_TQ_DP4A_SCALE_2BIT;
static constexpr float PQ_TQ_DP4A_SCALE_3BIT = 640.0f;
static constexpr float PQ_TQ_DP4A_INV_SCALE_3BIT = 1.0f / PQ_TQ_DP4A_SCALE_3BIT;
static constexpr float PQ_TQ_DP4A_SCALE_4BIT = 704.0f;
static constexpr float PQ_TQ_DP4A_INV_SCALE_4BIT = 1.0f / PQ_TQ_DP4A_SCALE_4BIT;

// QJL correction: expected |residual_i| = sqrt(2/(pi*d)) * rnorm
// D-dependent correction scales:
static constexpr float TQ_QJL_CORRECTION_SCALE_64  = 0.0997356f;  // sqrt(2/(pi*64))
static constexpr float TQ_QJL_CORRECTION_SCALE_128 = 0.0705348f;  // sqrt(2/(pi*128))
static constexpr float TQ_QJL_CORRECTION_SCALE_256 = 0.0498678f;  // sqrt(2/(pi*256))
// Legacy alias for D=128 (backward compat):
static constexpr float TQ_QJL_CORRECTION_SCALE = TQ_QJL_CORRECTION_SCALE_128;
// For dp4a path: signs are mapped to ±127, so correction_per_dp4a = QJL_CORRECTION_SCALE / 127
static constexpr float TQ_QJL_DP4A_CORRECTION_SCALE = TQ_QJL_CORRECTION_SCALE / 127.0f;

// D-dependent QJL correction scale (compile-time dispatch)
template<int D> static constexpr float tq_qjl_correction_scale();
template<> constexpr float tq_qjl_correction_scale<64>()  { return TQ_QJL_CORRECTION_SCALE_64;  }
template<> constexpr float tq_qjl_correction_scale<128>() { return TQ_QJL_CORRECTION_SCALE_128; }
template<> constexpr float tq_qjl_correction_scale<256>() { return TQ_QJL_CORRECTION_SCALE_256; }

// Compile-time check for supported PQ/TQ head dimensions
template<int D> struct pq_tq_supported_dim : std::false_type {};
template<> struct pq_tq_supported_dim<64>  : std::true_type {};
template<> struct pq_tq_supported_dim<128> : std::true_type {};
template<> struct pq_tq_supported_dim<256> : std::true_type {};

// Runtime check for supported PQ/TQ head dimensions
static inline bool pq_tq_is_supported_dim(int d) {
    return d == 64 || d == 128 || d == 256;
}

static constexpr int pq_tq_round_i8_host(const float value) {
    return value >= 0.0f ? (int) (value + 0.5f) : (int) (value - 0.5f);
}

static constexpr uint16_t pq_tq_pack_i8_pair_host(const float x0, const float x1) {
    return (uint16_t) (uint8_t) (int8_t) pq_tq_round_i8_host(x0)
        | ((uint16_t) (uint8_t) (int8_t) pq_tq_round_i8_host(x1) << 8);
}

static constexpr uint32_t tq_qjl_pack_signs_i8_host(const uint8_t bits) {
    return  (uint32_t) (uint8_t) (int8_t) ((bits & 0x1u) ? 127 : -127)
        | (((uint32_t) (uint8_t) (int8_t) ((bits & 0x2u) ? 127 : -127)) <<  8)
        | (((uint32_t) (uint8_t) (int8_t) ((bits & 0x4u) ? 127 : -127)) << 16)
        | (((uint32_t) (uint8_t) (int8_t) ((bits & 0x8u) ? 127 : -127)) << 24);
}

static constexpr float pq_tq_centroid_2bit_host(const uint8_t q2) {
    constexpr float centroids[4] = {-0.133462f, -0.039994f, 0.039994f, 0.133462f};
    return centroids[q2];
}

static constexpr float pq_tq_centroid_3bit_host(const uint8_t q2, const uint8_t sign) {
    constexpr float mags[4] = { 0.021460f, 0.065717f, 0.117832f, 0.190685f };
    return sign ? mags[q2] : -mags[3 - q2];
}

static constexpr float pq_tq_centroid_4bit_host(const uint8_t q4) {
    constexpr float mags[8] = {
        0.006938f, 0.020989f, 0.035597f, 0.051262f,
        0.068756f, 0.089527f, 0.117195f, 0.173926f,
    };
    const uint8_t sign = q4 >> 3;
    return sign ? mags[q4 & 0x7u] : -mags[0x7u - (q4 & 0x7u)];
}

#define PQ_TQ_PAIR_LUT_ROW_16(ENTRY, base) \
    ENTRY((base) + 0), ENTRY((base) + 1), ENTRY((base) + 2), ENTRY((base) + 3), \
    ENTRY((base) + 4), ENTRY((base) + 5), ENTRY((base) + 6), ENTRY((base) + 7), \
    ENTRY((base) + 8), ENTRY((base) + 9), ENTRY((base) + 10), ENTRY((base) + 11), \
    ENTRY((base) + 12), ENTRY((base) + 13), ENTRY((base) + 14), ENTRY((base) + 15)

#define PQ_TQ_PAIR2_ENTRY(idx) { \
    pq_tq_centroid_2bit_host((idx) & 0x3u), \
    pq_tq_centroid_2bit_host(((idx) >> 2) & 0x3u) \
}

__constant__ static const float2 PQ_TQ_PAIR_LUT_2BIT[16] = {
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR2_ENTRY, 0),
};

#define PQ_TQ_DP4A_PAIR2_ENTRY(idx) \
    pq_tq_pack_i8_pair_host( \
        pq_tq_centroid_2bit_host((idx) & 0x3u) * PQ_TQ_DP4A_SCALE_2BIT, \
        pq_tq_centroid_2bit_host(((idx) >> 2) & 0x3u) * PQ_TQ_DP4A_SCALE_2BIT)

__constant__ static const uint16_t PQ_TQ_DP4A_PAIR_LUT_2BIT[16] = {
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR2_ENTRY, 0),
};

#define PQ_TQ_DP4A_VAL2_ENTRY(idx) \
    (int8_t) pq_tq_round_i8_host(pq_tq_centroid_2bit_host((idx) & 0x3u) * PQ_TQ_DP4A_SCALE_2BIT)

__constant__ static const int8_t PQ_TQ_DP4A_VAL_2BIT[16] = {
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_VAL2_ENTRY, 0),
};

#define PQ_TQ_PAIR3_ENTRY(idx) { \
    pq_tq_centroid_3bit_host((idx) & 0x3u, ((idx) >> 4) & 0x1u), \
    pq_tq_centroid_3bit_host(((idx) >> 2) & 0x3u, ((idx) >> 5) & 0x1u) \
}

__constant__ static const float2 PQ_TQ_PAIR_LUT_3BIT[64] = {
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR3_ENTRY,  0),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR3_ENTRY, 16),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR3_ENTRY, 32),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR3_ENTRY, 48),
};

#define PQ_TQ_DP4A_PAIR3_ENTRY(idx) \
    pq_tq_pack_i8_pair_host( \
        pq_tq_centroid_3bit_host((idx) & 0x3u, ((idx) >> 4) & 0x1u) * PQ_TQ_DP4A_SCALE_3BIT, \
        pq_tq_centroid_3bit_host(((idx) >> 2) & 0x3u, ((idx) >> 5) & 0x1u) * PQ_TQ_DP4A_SCALE_3BIT)

__constant__ static const uint16_t PQ_TQ_DP4A_PAIR_LUT_3BIT[64] = {
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR3_ENTRY,  0),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR3_ENTRY, 16),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR3_ENTRY, 32),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR3_ENTRY, 48),
};

#define PQ_TQ_DP4A_VAL3_ENTRY(idx) \
    (int8_t) pq_tq_round_i8_host(pq_tq_centroid_3bit_host((idx) & 0x3u, ((idx) >> 2) & 0x1u) * PQ_TQ_DP4A_SCALE_3BIT)

__constant__ static const int8_t PQ_TQ_DP4A_VAL_3BIT[8] = {
    PQ_TQ_DP4A_VAL3_ENTRY(0), PQ_TQ_DP4A_VAL3_ENTRY(1), PQ_TQ_DP4A_VAL3_ENTRY(2), PQ_TQ_DP4A_VAL3_ENTRY(3),
    PQ_TQ_DP4A_VAL3_ENTRY(4), PQ_TQ_DP4A_VAL3_ENTRY(5), PQ_TQ_DP4A_VAL3_ENTRY(6), PQ_TQ_DP4A_VAL3_ENTRY(7),
};

__constant__ static const int8_t PQ_TQ_DP4A_VAL_3BIT_16[16] = {
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_VAL3_ENTRY, 0),
};

#define TQ_QJL_DP4A_SIGNS_ENTRY(idx) \
    (int) tq_qjl_pack_signs_i8_host((uint8_t) (idx))

__constant__ static const int TQ_QJL_DP4A_SIGNS_16[16] = {
    PQ_TQ_PAIR_LUT_ROW_16(TQ_QJL_DP4A_SIGNS_ENTRY, 0),
};

#define PQ_TQ_PAIR4_ENTRY(idx) { \
    pq_tq_centroid_4bit_host((idx) & 0xFu), \
    pq_tq_centroid_4bit_host(((idx) >> 4) & 0xFu) \
}

__constant__ static const float2 PQ_TQ_PAIR_LUT_4BIT[256] = {
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY,   0),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY,  16),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY,  32),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY,  48),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY,  64),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY,  80),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY,  96),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY, 112),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY, 128),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY, 144),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY, 160),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY, 176),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY, 192),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY, 208),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY, 224),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_PAIR4_ENTRY, 240),
};

#define PQ_TQ_DP4A_PAIR4_ENTRY(idx) \
    pq_tq_pack_i8_pair_host( \
        pq_tq_centroid_4bit_host((idx) & 0xFu) * PQ_TQ_DP4A_SCALE_4BIT, \
        pq_tq_centroid_4bit_host(((idx) >> 4) & 0xFu) * PQ_TQ_DP4A_SCALE_4BIT)

__constant__ static const uint16_t PQ_TQ_DP4A_PAIR_LUT_4BIT[256] = {
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY,   0),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY,  16),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY,  32),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY,  48),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY,  64),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY,  80),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY,  96),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY, 112),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY, 128),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY, 144),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY, 160),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY, 176),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY, 192),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY, 208),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY, 224),
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_PAIR4_ENTRY, 240),
};

#define PQ_TQ_DP4A_VAL4_ENTRY(idx) \
    (int8_t) pq_tq_round_i8_host(pq_tq_centroid_4bit_host(idx) * PQ_TQ_DP4A_SCALE_4BIT)

__constant__ static const int8_t PQ_TQ_DP4A_VAL_4BIT[16] = {
    PQ_TQ_PAIR_LUT_ROW_16(PQ_TQ_DP4A_VAL4_ENTRY, 0),
};

#undef PQ_TQ_DP4A_PAIR4_ENTRY
#undef PQ_TQ_DP4A_VAL4_ENTRY
#undef PQ_TQ_DP4A_VAL2_ENTRY
#undef PQ_TQ_DP4A_VAL3_ENTRY
#undef PQ_TQ_DP4A_PAIR3_ENTRY
#undef PQ_TQ_DP4A_PAIR2_ENTRY
#undef PQ_TQ_PAIR4_ENTRY
#undef PQ_TQ_PAIR3_ENTRY
#undef PQ_TQ_PAIR2_ENTRY
#undef PQ_TQ_PAIR_LUT_ROW_16

// Half2 pair LUTs for V dequantization (avoids float<->half conversions at runtime).
// Each uint32_t entry encodes a half2: low 16 bits = first centroid, high 16 bits = second centroid.
// PQ_TQ_PAIR_LUT_2BIT_H2[idx]:  idx = 4-bit nibble encoding 2×2-bit indices, .x = centroid(idx&3), .y = centroid(idx>>2)
__constant__ static const uint32_t PQ_TQ_PAIR_LUT_2BIT_H2[16] = {
    0xB045B045, 0xB045A91F, 0xB045291F, 0xB0453045,
    0xA91FB045, 0xA91FA91F, 0xA91F291F, 0xA91F3045,
    0x291FB045, 0x291FA91F, 0x291F291F, 0x291F3045,
    0x3045B045, 0x3045A91F, 0x3045291F, 0x30453045
};

// PQ_TQ_PAIR_LUT_4BIT_H2[byte]: .x = centroid(byte & 0xF), .y = centroid(byte >> 4)
// PQ_TQ_PAIR_LUT_3BIT_H2[idx]:  idx = (sign2<<4)|q4, matches PQ_TQ_PAIR_LUT_3BIT indexing
__constant__ static const uint32_t PQ_TQ_PAIR_LUT_4BIT_H2[256] = {
    0xB191B191, 0xB191AF80, 0xB191ADBB, 0xB191AC66, 0xB191AA90, 0xB191A88E, 0xB191A560, 0xB1919F1B,
    0xB1911F1B, 0xB1912560, 0xB191288E, 0xB1912A90, 0xB1912C66, 0xB1912DBB, 0xB1912F80, 0xB1913191,
    0xAF80B191, 0xAF80AF80, 0xAF80ADBB, 0xAF80AC66, 0xAF80AA90, 0xAF80A88E, 0xAF80A560, 0xAF809F1B,
    0xAF801F1B, 0xAF802560, 0xAF80288E, 0xAF802A90, 0xAF802C66, 0xAF802DBB, 0xAF802F80, 0xAF803191,
    0xADBBB191, 0xADBBAF80, 0xADBBADBB, 0xADBBAC66, 0xADBBAA90, 0xADBBA88E, 0xADBBA560, 0xADBB9F1B,
    0xADBB1F1B, 0xADBB2560, 0xADBB288E, 0xADBB2A90, 0xADBB2C66, 0xADBB2DBB, 0xADBB2F80, 0xADBB3191,
    0xAC66B191, 0xAC66AF80, 0xAC66ADBB, 0xAC66AC66, 0xAC66AA90, 0xAC66A88E, 0xAC66A560, 0xAC669F1B,
    0xAC661F1B, 0xAC662560, 0xAC66288E, 0xAC662A90, 0xAC662C66, 0xAC662DBB, 0xAC662F80, 0xAC663191,
    0xAA90B191, 0xAA90AF80, 0xAA90ADBB, 0xAA90AC66, 0xAA90AA90, 0xAA90A88E, 0xAA90A560, 0xAA909F1B,
    0xAA901F1B, 0xAA902560, 0xAA90288E, 0xAA902A90, 0xAA902C66, 0xAA902DBB, 0xAA902F80, 0xAA903191,
    0xA88EB191, 0xA88EAF80, 0xA88EADBB, 0xA88EAC66, 0xA88EAA90, 0xA88EA88E, 0xA88EA560, 0xA88E9F1B,
    0xA88E1F1B, 0xA88E2560, 0xA88E288E, 0xA88E2A90, 0xA88E2C66, 0xA88E2DBB, 0xA88E2F80, 0xA88E3191,
    0xA560B191, 0xA560AF80, 0xA560ADBB, 0xA560AC66, 0xA560AA90, 0xA560A88E, 0xA560A560, 0xA5609F1B,
    0xA5601F1B, 0xA5602560, 0xA560288E, 0xA5602A90, 0xA5602C66, 0xA5602DBB, 0xA5602F80, 0xA5603191,
    0x9F1BB191, 0x9F1BAF80, 0x9F1BADBB, 0x9F1BAC66, 0x9F1BAA90, 0x9F1BA88E, 0x9F1BA560, 0x9F1B9F1B,
    0x9F1B1F1B, 0x9F1B2560, 0x9F1B288E, 0x9F1B2A90, 0x9F1B2C66, 0x9F1B2DBB, 0x9F1B2F80, 0x9F1B3191,
    0x1F1BB191, 0x1F1BAF80, 0x1F1BADBB, 0x1F1BAC66, 0x1F1BAA90, 0x1F1BA88E, 0x1F1BA560, 0x1F1B9F1B,
    0x1F1B1F1B, 0x1F1B2560, 0x1F1B288E, 0x1F1B2A90, 0x1F1B2C66, 0x1F1B2DBB, 0x1F1B2F80, 0x1F1B3191,
    0x2560B191, 0x2560AF80, 0x2560ADBB, 0x2560AC66, 0x2560AA90, 0x2560A88E, 0x2560A560, 0x25609F1B,
    0x25601F1B, 0x25602560, 0x2560288E, 0x25602A90, 0x25602C66, 0x25602DBB, 0x25602F80, 0x25603191,
    0x288EB191, 0x288EAF80, 0x288EADBB, 0x288EAC66, 0x288EAA90, 0x288EA88E, 0x288EA560, 0x288E9F1B,
    0x288E1F1B, 0x288E2560, 0x288E288E, 0x288E2A90, 0x288E2C66, 0x288E2DBB, 0x288E2F80, 0x288E3191,
    0x2A90B191, 0x2A90AF80, 0x2A90ADBB, 0x2A90AC66, 0x2A90AA90, 0x2A90A88E, 0x2A90A560, 0x2A909F1B,
    0x2A901F1B, 0x2A902560, 0x2A90288E, 0x2A902A90, 0x2A902C66, 0x2A902DBB, 0x2A902F80, 0x2A903191,
    0x2C66B191, 0x2C66AF80, 0x2C66ADBB, 0x2C66AC66, 0x2C66AA90, 0x2C66A88E, 0x2C66A560, 0x2C669F1B,
    0x2C661F1B, 0x2C662560, 0x2C66288E, 0x2C662A90, 0x2C662C66, 0x2C662DBB, 0x2C662F80, 0x2C663191,
    0x2DBBB191, 0x2DBBAF80, 0x2DBBADBB, 0x2DBBAC66, 0x2DBBAA90, 0x2DBBA88E, 0x2DBBA560, 0x2DBB9F1B,
    0x2DBB1F1B, 0x2DBB2560, 0x2DBB288E, 0x2DBB2A90, 0x2DBB2C66, 0x2DBB2DBB, 0x2DBB2F80, 0x2DBB3191,
    0x2F80B191, 0x2F80AF80, 0x2F80ADBB, 0x2F80AC66, 0x2F80AA90, 0x2F80A88E, 0x2F80A560, 0x2F809F1B,
    0x2F801F1B, 0x2F802560, 0x2F80288E, 0x2F802A90, 0x2F802C66, 0x2F802DBB, 0x2F802F80, 0x2F803191,
    0x3191B191, 0x3191AF80, 0x3191ADBB, 0x3191AC66, 0x3191AA90, 0x3191A88E, 0x3191A560, 0x31919F1B,
    0x31911F1B, 0x31912560, 0x3191288E, 0x31912A90, 0x31912C66, 0x31912DBB, 0x31912F80, 0x31913191
};

// PQ_TQ_PAIR_LUT_3BIT_H2[idx]: idx = (sign2<<4)|q4, .x = centroid(q4&3,sign2&1), .y = centroid((q4>>2)&3,(sign2>>1)&1)
__constant__ static const uint32_t PQ_TQ_PAIR_LUT_3BIT_H2[64] = {
    0xB21AB21A, 0xB21AAF8B, 0xB21AAC35, 0xB21AA57E, 0xAF8BB21A, 0xAF8BAF8B, 0xAF8BAC35, 0xAF8BA57E,
    0xAC35B21A, 0xAC35AF8B, 0xAC35AC35, 0xAC35A57E, 0xA57EB21A, 0xA57EAF8B, 0xA57EAC35, 0xA57EA57E,
    0xB21A257E, 0xB21A2C35, 0xB21A2F8B, 0xB21A321A, 0xAF8B257E, 0xAF8B2C35, 0xAF8B2F8B, 0xAF8B321A,
    0xAC35257E, 0xAC352C35, 0xAC352F8B, 0xAC35321A, 0xA57E257E, 0xA57E2C35, 0xA57E2F8B, 0xA57E321A,
    0x257EB21A, 0x257EAF8B, 0x257EAC35, 0x257EA57E, 0x2C35B21A, 0x2C35AF8B, 0x2C35AC35, 0x2C35A57E,
    0x2F8BB21A, 0x2F8BAF8B, 0x2F8BAC35, 0x2F8BA57E, 0x321AB21A, 0x321AAF8B, 0x321AAC35, 0x321AA57E,
    0x257E257E, 0x257E2C35, 0x257E2F8B, 0x257E321A, 0x2C35257E, 0x2C352C35, 0x2C352F8B, 0x2C35321A,
    0x2F8B257E, 0x2F8B2C35, 0x2F8B2F8B, 0x2F8B321A, 0x321A257E, 0x321A2C35, 0x321A2F8B, 0x321A321A
};

// 2-bit midpoints (3 values define 4 regions)
__constant__ static const float PQ_TQ_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// 4-bit midpoints (15 values define 16 regions)
__constant__ static const float PQ_TQ_MID_4BIT[15] = {
    -0.145560f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013963f,  0.000000f,
     0.013963f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145560f
};

// FWHT sign arrays (precomputed from seed=42 and seed=1042)
__constant__ static const float PQ_TQ_WHT_SIGNS1[128] = {
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f
};

__constant__ static const float PQ_TQ_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f
};

// D=64 dedicated FWHT sign arrays (xorshift32: seed=2525 for S1, seed=2547 for S2)
// The first 64 elements of PQ_TQ_WHT_SIGNS1/2 have pathologically structured patterns
// (Walsh-matrix pattern in SIGNS2[0..63]: 8 same-sign runs), which degrades WHT quality
// for D=64.  These dedicated arrays have balanced ±1 counts and max same-sign run ≤ 3.
__constant__ static const float PQ_TQ_WHT_SIGNS1_64[64] = {
     1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
};
__constant__ static const float PQ_TQ_WHT_SIGNS2_64[64] = {
     1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
};

// D=256 FWHT sign arrays (xorshift32: seed=4242 for S1, seed=104242 for S2)
// For D=128, the existing PQ_TQ_WHT_SIGNS1/2[128] are used.
// For D=256, these dedicated 256-element arrays are used (native WHT256).
__constant__ static const float PQ_TQ_WHT_SIGNS1_256[256] = {
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f
};

__constant__ static const float PQ_TQ_WHT_SIGNS2_256[256] = {
     1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f
};

static __device__ __forceinline__ float pq_tq_centroid_2bit(const uint8_t q2) {
    return PQ_TQ_CENTROIDS_2BIT[q2];
}

static __device__ __forceinline__ float2 pq_tq_centroid_pair_2bit(const uint8_t q4) {
    return PQ_TQ_PAIR_LUT_2BIT[q4 & 0x0Fu];
}

#ifdef FP16_AVAILABLE
static __device__ __forceinline__ half2 pq_tq_centroid_pair_2bit_h2(const uint8_t q4) {
    uint32_t bits = PQ_TQ_PAIR_LUT_2BIT_H2[q4 & 0x0Fu];
    half2 result;
    memcpy(&result, &bits, sizeof(half2));
    return result;
}
#endif // FP16_AVAILABLE

static __device__ __forceinline__ int pq_tq_centroid_pack_2bit_i8(const uint8_t qb) {
    const uint32_t lo = PQ_TQ_DP4A_PAIR_LUT_2BIT[qb & 0x0Fu];
    const uint32_t hi = PQ_TQ_DP4A_PAIR_LUT_2BIT[qb >> 4];
    return (int) (lo | (hi << 16));
}

static __device__ __forceinline__ float pq_tq_centroid_3bit(const uint8_t q2, const uint8_t sign) {
    const float mag = PQ_TQ_MAG_3BIT[q2 ^ (sign ? 0x0u : 0x3u)];
    return sign ? mag : -mag;
}

static __device__ __forceinline__ float2 pq_tq_centroid_pair_3bit(const uint8_t q4, const uint8_t sign2) {
    return make_float2(
        PQ_TQ_CENTROIDS_3BIT[((sign2 & 0x1u) << 2) | (q4 & 0x3u)],
        PQ_TQ_CENTROIDS_3BIT[(((sign2 >> 1) & 0x1u) << 2) | ((q4 >> 2) & 0x3u)]);
}

// Half2 pair lookup: eliminates float<->half conversions in V dequant hot path.
// Returns half2{centroid(byte & 0xF), centroid(byte >> 4)} directly as half2.
#ifdef FP16_AVAILABLE
static __device__ __forceinline__ half2 pq_tq_centroid_pair_4bit_h2(const uint8_t byte) {
    uint32_t bits = PQ_TQ_PAIR_LUT_4BIT_H2[byte];
    half2 result;
    memcpy(&result, &bits, sizeof(half2));
    return result;
}

// Returns half2{centroid_3bit(q4&3,sign2&1), centroid_3bit((q4>>2)&3,(sign2>>1)&1)}.
static __device__ __forceinline__ half2 pq_tq_centroid_pair_3bit_h2(const uint8_t q4, const uint8_t sign2) {
    uint32_t bits = PQ_TQ_PAIR_LUT_3BIT_H2[(sign2 << 4) | q4];
    half2 result;
    memcpy(&result, &bits, sizeof(half2));
    return result;
}
#endif // FP16_AVAILABLE

static __device__ __forceinline__ int pq_tq_centroid_pack_3bit_i8(const uint8_t qb, const uint8_t sb) {
    const uint32_t v0 = (uint8_t) PQ_TQ_DP4A_VAL_3BIT[((sb & 0x1u) << 2) | (qb & 0x3u)];
    const uint32_t v1 = (uint8_t) PQ_TQ_DP4A_VAL_3BIT[(((sb >> 1) & 0x1u) << 2) | ((qb >> 2) & 0x3u)];
    const uint32_t v2 = (uint8_t) PQ_TQ_DP4A_VAL_3BIT[(((sb >> 2) & 0x1u) << 2) | ((qb >> 4) & 0x3u)];
    const uint32_t v3 = (uint8_t) PQ_TQ_DP4A_VAL_3BIT[(((sb >> 3) & 0x1u) << 2) | ((qb >> 6) & 0x3u)];
    return (int) (v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
}

static __device__ __forceinline__ float pq_tq_centroid_4bit(const uint8_t q4) {
    const uint8_t sign = q4 >> 3;
    const float mag = PQ_TQ_MAG_4BIT[(q4 & 0x7u) ^ (sign ? 0x0u : 0x7u)];
    return sign ? mag : -mag;
}

static __device__ __forceinline__ float2 pq_tq_centroid_pair_4bit(const uint8_t q8) {
    return make_float2(PQ_TQ_CENTROIDS_4BIT[q8 & 0x0Fu], PQ_TQ_CENTROIDS_4BIT[q8 >> 4]);
}

static __device__ __forceinline__ int pq_tq_centroid_pack_4bit_i8(const uint16_t qpair) {
    const uint32_t v0 = (uint8_t) PQ_TQ_DP4A_VAL_4BIT[ qpair        & 0x0Fu];
    const uint32_t v1 = (uint8_t) PQ_TQ_DP4A_VAL_4BIT[(qpair >>  4) & 0x0Fu];
    const uint32_t v2 = (uint8_t) PQ_TQ_DP4A_VAL_4BIT[(qpair >>  8) & 0x0Fu];
    const uint32_t v3 = (uint8_t) PQ_TQ_DP4A_VAL_4BIT[(qpair >> 12) & 0x0Fu];
    return (int) (v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
}

// ===== DEVICE MEMORY: Not used - FWHT uses constant memory sign arrays =====

// GPU rotation matrices are not allocated (FWHT approach uses constant memory instead)

// ===== Device helper: Find nearest centroid via midpoints =====

__device__ __forceinline__ uint8_t pq_tq_find_nearest_centroid_4bit(float val) {
    // Binary search using 15 midpoints
    if (val < PQ_TQ_MID_4BIT[0]) return 0;
    if (val < PQ_TQ_MID_4BIT[1]) return 1;
    if (val < PQ_TQ_MID_4BIT[2]) return 2;
    if (val < PQ_TQ_MID_4BIT[3]) return 3;
    if (val < PQ_TQ_MID_4BIT[4]) return 4;
    if (val < PQ_TQ_MID_4BIT[5]) return 5;
    if (val < PQ_TQ_MID_4BIT[6]) return 6;
    if (val < PQ_TQ_MID_4BIT[7]) return 7;
    if (val < PQ_TQ_MID_4BIT[8]) return 8;
    if (val < PQ_TQ_MID_4BIT[9]) return 9;
    if (val < PQ_TQ_MID_4BIT[10]) return 10;
    if (val < PQ_TQ_MID_4BIT[11]) return 11;
    if (val < PQ_TQ_MID_4BIT[12]) return 12;
    if (val < PQ_TQ_MID_4BIT[13]) return 13;
    if (val < PQ_TQ_MID_4BIT[14]) return 14;
    return 15;
}

__device__ __forceinline__ uint8_t pq_tq_find_nearest_centroid_3bit(float val) {
    // Binary search using 7 midpoints
    if (val < PQ_TQ_MID_3BIT[0]) return 0;
    if (val < PQ_TQ_MID_3BIT[1]) return 1;
    if (val < PQ_TQ_MID_3BIT[2]) return 2;
    if (val < PQ_TQ_MID_3BIT[3]) return 3;
    if (val < PQ_TQ_MID_3BIT[4]) return 4;
    if (val < PQ_TQ_MID_3BIT[5]) return 5;
    if (val < PQ_TQ_MID_3BIT[6]) return 6;
    return 7;
}

__device__ __forceinline__ uint8_t pq_tq_find_nearest_centroid_2bit(float val) {
    if (val < PQ_TQ_MID_2BIT[0]) return 0;
    if (val < PQ_TQ_MID_2BIT[1]) return 1;
    if (val < PQ_TQ_MID_2BIT[2]) return 2;
    return 3;
}

// ===== QJL sign packing for dp4a: 4 sign bits → int32 of ±127 =====
// Given a byte where each bit is a residual sign (bit=1 → positive, bit=0 → negative),
// extract 4 consecutive bits starting at bit position `shift` and pack as int8x4 (±127).
__device__ __forceinline__ int tq_qjl_pack_signs_i8(uint8_t qjl_byte, int shift) {
    // Extract 4 bits
    const uint8_t bits = (qjl_byte >> shift) & 0x0Fu;
    // Map each bit: 0→-127, 1→+127
    const int8_t s0 = (bits & 1u) ? 127 : -127;
    const int8_t s1 = (bits & 2u) ? 127 : -127;
    const int8_t s2 = (bits & 4u) ? 127 : -127;
    const int8_t s3 = (bits & 8u) ? 127 : -127;
    return (int)(uint32_t)((uint8_t)s0 | ((uint8_t)s1 << 8) | ((uint8_t)s2 << 16) | ((uint8_t)s3 << 24));
}

// QJL correction for half2 V dequant: returns correction for 2 elements
__device__ __forceinline__ float2 tq_qjl_correction_pair(uint8_t qjl_byte, int shift, float rnorm) {
    const uint8_t bits = (qjl_byte >> shift) & 0x3u;
    const float s0 = (bits & 1u) ? TQ_QJL_CORRECTION_SCALE : -TQ_QJL_CORRECTION_SCALE;
    const float s1 = (bits & 2u) ? TQ_QJL_CORRECTION_SCALE : -TQ_QJL_CORRECTION_SCALE;
    return make_float2(s0 * rnorm, s1 * rnorm);
}
