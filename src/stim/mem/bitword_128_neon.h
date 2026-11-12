/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_MEM_SIMD_WORD_128_NEON_H
#define _STIM_MEM_SIMD_WORD_128_NEON_H
#ifdef __ARM_NEON

#include <algorithm>
#include <array>
#include <bit>
#include <arm_neon.h>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "stim/mem/bitword.h"

namespace stim {

template <>
struct alignas(16) bitword<128> {
    constexpr static size_t BIT_SIZE = 128;
    constexpr static size_t BIT_POW = 7;

    union {
        uint8x16_t val;
        uint64_t u64[2];
        uint8_t u8[16];
    };

    static void *aligned_malloc(size_t bytes) {
        void *ptr = nullptr;
        if (posix_memalign(&ptr, 16, bytes) != 0)
            return nullptr;
        return ptr;
    }

    static void aligned_free(void *ptr) {
        free(ptr);
    }

    inline bitword() : val(vdupq_n_u8(0)) {}

    inline bitword(uint8x16_t val) : val(val) {}

    inline bitword(std::array<uint64_t, 2> arr) {
        val = vreinterpretq_u8_u64(vcombine_u64(
            vcreate_u64(arr[0]),  // Low 64 bits
            vcreate_u64(arr[1])   // High 64 bits
        ));
    }

    inline bitword(uint64_t low) {
        val = vreinterpretq_u8_u64(vcombine_u64(
            vcreate_u64(low),
            vcreate_u64(0)
        ));
    }

    inline bitword(int64_t sval) {
        uint64_t high = (sval < 0) ? ~0ULL : 0ULL;
        val = vreinterpretq_u8_u64(vcombine_u64(
            vcreate_u64(static_cast<uint64_t>(sval)),
            vcreate_u64(high)
        ));
    }

    inline bitword(int sval) {
        uint64_t high = (sval < 0) ? ~0ULL : 0ULL;
        val = vreinterpretq_u8_u64(vcombine_u64(
            vcreate_u64(static_cast<uint64_t>(sval)),
            vcreate_u64(high)
        ));
    }

    // Optimized tile functions using direct NEON intrinsics
    inline static bitword<128> tile8(uint8_t pattern) {
        return {vdupq_n_u8(pattern)};
    }

    inline static bitword<128> tile16(uint16_t pattern) {
        return {vreinterpretq_u8_u16(vdupq_n_u16(pattern))};
    }

    inline static bitword<128> tile32(uint32_t pattern) {
        return {vreinterpretq_u8_u32(vdupq_n_u32(pattern))};
    }

    inline static bitword<128> tile64(uint64_t pattern) {
        return {vreinterpretq_u8_u64(vdupq_n_u64(pattern))};
    }

    inline std::array<uint64_t, 2> to_u64_array() const {
        return {u64[0], u64[1]};  // Direct access, no swapping needed
    }

    inline operator bool() const {
        return vmaxvq_u8(val) != 0;
    }

    inline operator int() const {
        return static_cast<int>(static_cast<int64_t>(*this));
    }

    inline operator uint64_t() const {
        if (u64[1] != 0) {
            throw std::invalid_argument("Too large for uint64_t");
        }
        return u64[0];
    }

    inline operator int64_t() const {
        int64_t result = static_cast<int64_t>(u64[0]);
        uint64_t expected_high = (result < 0) ? ~0ULL : 0ULL;
        if (u64[1] != expected_high) {
            throw std::invalid_argument("Out of bounds of int64_t");
        }
        return result;
    }

    // Bitwise operations using NEON intrinsics
    inline bitword<128> &operator^=(const bitword<128> &other) {
        val = veorq_u8(val, other.val);
        return *this;
    }

    inline bitword<128> &operator&=(const bitword<128> &other) {
        val = vandq_u8(val, other.val);
        return *this;
    }

    inline bitword<128> &operator|=(const bitword<128> &other) {
        val = vorrq_u8(val, other.val);
        return *this;
    }

    inline bitword<128> operator^(const bitword<128> &other) const {
        return {veorq_u8(val, other.val)};
    }

    inline bitword<128> operator&(const bitword<128> &other) const {
        return {vandq_u8(val, other.val)};
    }

    inline bitword<128> operator|(const bitword<128> &other) const {
        return {vorrq_u8(val, other.val)};
    }

    inline bitword<128> andnot(const bitword<128> &other) const {
        return {vbicq_u8(other.val, val)};  // ARM BIC: other & ~val
    }

    inline bitword<128> operator~() const {
        return {vmvnq_u8(val)};
    }

    inline uint16_t popcount() const {
        // Use ARM's efficient popcount instruction when available
//#ifdef __ARM_FEATURE_CRYPTO
        uint8x16_t cnt = vcntq_u8(val);
        return vaddvq_u8(cnt);  // Sum all bytes efficiently
//        return std::popcount(u64[0]) + std::popcount(u64[1]);
    }

inline bitword<128> shifted(int offset) const {
    if (offset == 0) return *this;

    uint64x2_t v = vreinterpretq_u64_u8(val);

    if (offset > 0) {
        if (offset >= 128) return {vdupq_n_u8(0)};

        if (offset >= 64) {
            // Shift by 64+ bits: move low to high, zero low
            uint64_t low_val = vgetq_lane_u64(v, 0);
            uint64x2_t result = vsetq_lane_u64(0, v, 0);

            if (offset == 64) {
                result = vsetq_lane_u64(low_val, result, 1);
            } else {
                // Use variable shift for the remainder
                int remaining = offset - 64;
                uint64_t shifted_val = low_val << remaining;
                result = vsetq_lane_u64(shifted_val, result, 1);
            }
            return {vreinterpretq_u8_u64(result)};
        } else {
            // Shift by <64 bits: use variable shifts
            uint64_t low = vgetq_lane_u64(v, 0);
            uint64_t high = vgetq_lane_u64(v, 1);

            uint64_t new_low = low << offset;
            uint64_t carry = low >> (64 - offset);
            uint64_t new_high = (high << offset) | carry;

            uint64x2_t result = vsetq_lane_u64(new_low, v, 0);
            result = vsetq_lane_u64(new_high, result, 1);
            return {vreinterpretq_u8_u64(result)};
        }
    } else {
        offset = -offset;
        if (offset >= 128) return {vdupq_n_u8(0)};

        if (offset >= 64) {
            // Shift by 64+ bits: move high to low, zero high
            uint64_t high_val = vgetq_lane_u64(v, 1);
            uint64x2_t result = vsetq_lane_u64(0, v, 1);

            if (offset == 64) {
                result = vsetq_lane_u64(high_val, result, 0);
            } else {
                // Use variable shift for the remainder
                int remaining = offset - 64;
                uint64_t shifted_val = high_val >> remaining;
                result = vsetq_lane_u64(shifted_val, result, 0);
            }
            return {vreinterpretq_u8_u64(result)};
        } else {
            // Shift by <64 bits: use variable shifts
            uint64_t low = vgetq_lane_u64(v, 0);
            uint64_t high = vgetq_lane_u64(v, 1);

            uint64_t new_high = high >> offset;
            uint64_t carry = high << (64 - offset);
            uint64_t new_low = (low >> offset) | carry;

            uint64x2_t result = vsetq_lane_u64(new_low, v, 0);
            result = vsetq_lane_u64(new_high, result, 1);
            return {vreinterpretq_u8_u64(result)};
        }
    }
}

    inline std::string str() const {
        std::stringstream out;
        out << *this;
        return out.str();
    }

    /// One step of the recursive 128x128 bit-block transpose, applied to a pair of rows (k, k + shift):
    /// swaps the high `shift` bits of each 2*shift-bit group of row k with the low ones of row k + shift.
    /// Each step is one or two NEON instructions per row:
    ///   shift 1, 2:   shift + BSL (bit select under an alternating mask)
    ///   shift 4:      SLI / SRI (shift and insert within each byte)
    ///   shift 8..64:  TRN1 / TRN2 (interleave even / odd lanes of 8, 16, 32 or 64 bits)
    template <uint64_t shift>
    static inline void transpose_step(uint8x16_t &x, uint8x16_t &y) {
        uint8x16_t new_x;
        uint8x16_t new_y;
        if constexpr (shift == 1 || shift == 2) {
            const uint8x16_t mask = vdupq_n_u8(shift == 1 ? 0x55 : 0x33);
            new_x = vbslq_u8(mask, x, vshlq_n_u8(y, shift));
            new_y = vbslq_u8(mask, vshrq_n_u8(x, shift), y);
        } else if constexpr (shift == 4) {
            new_x = vsliq_n_u8(x, y, 4);
            new_y = vsriq_n_u8(y, x, 4);
        } else if constexpr (shift == 8) {
            new_x = vtrn1q_u8(x, y);
            new_y = vtrn2q_u8(x, y);
        } else if constexpr (shift == 16) {
            new_x = vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(x), vreinterpretq_u16_u8(y)));
            new_y = vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(x), vreinterpretq_u16_u8(y)));
        } else if constexpr (shift == 32) {
            new_x = vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(x), vreinterpretq_u32_u8(y)));
            new_y = vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(x), vreinterpretq_u32_u8(y)));
        } else {
            static_assert(shift == 64);
            new_x = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(x), vreinterpretq_u64_u8(y)));
            new_y = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(x), vreinterpretq_u64_u8(y)));
        }
        x = new_x;
        y = new_y;
    }

    /// Applies transpose_step<shift * row_spacing> to every pair (i, i + shift) of N rows held in registers,
    /// where the registers hold rows spaced `row_spacing` apart in the block.
    template <uint64_t shift, uint64_t row_spacing, size_t N>
    static inline void transpose_steps_in_registers(uint8x16_t (&r)[N]) {
        for (size_t i = 0; i < N; i++) {
            if (!(i & shift)) {
                transpose_step<shift * row_spacing>(r[i], r[i + shift]);
            }
        }
    }

    /// Transposes a 128x128 bit block in place. Row k of the block is data[k * stride].
    ///
    /// The seven steps (shift 1, 2, ..., 64) commute, so they are grouped to keep rows in registers:
    /// steps 1-8 run on 16 consecutive rows at a time, and steps 16-64 on 8 rows spaced 16 apart.
    /// That is two trips through memory instead of seven.
    static void inplace_transpose_square(bitword<128> *data, size_t stride) {
        for (size_t base = 0; base < 128; base += 16) {
            uint8x16_t r[16];
            for (size_t i = 0; i < 16; i++) {
                r[i] = data[stride * (base + i)].val;
            }
            transpose_steps_in_registers<1, 1>(r);
            transpose_steps_in_registers<2, 1>(r);
            transpose_steps_in_registers<4, 1>(r);
            transpose_steps_in_registers<8, 1>(r);
            for (size_t i = 0; i < 16; i++) {
                data[stride * (base + i)].val = r[i];
            }
        }
        for (size_t base = 0; base < 16; base++) {
            uint8x16_t r[8];
            for (size_t i = 0; i < 8; i++) {
                r[i] = data[stride * (base + 16 * i)].val;
            }
            transpose_steps_in_registers<1, 16>(r);
            transpose_steps_in_registers<2, 16>(r);
            transpose_steps_in_registers<4, 16>(r);
            for (size_t i = 0; i < 8; i++) {
                data[stride * (base + 16 * i)].val = r[i];
            }
        }
    }
};

}  // namespace stim

#endif
#endif
