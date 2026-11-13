// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "stim/py/numpy.pybind.h"

#include <array>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

using namespace stim;
using namespace stim_pybind;

namespace {

// Maps a byte to 8 bytes of 0/1, in little-endian bit order (bit k of the byte becomes byte k).
struct ByteToBoolsTable {
    std::array<uint64_t, 256> entries;
    constexpr ByteToBoolsTable() : entries() {
        for (size_t b = 0; b < 256; b++) {
            uint64_t v = 0;
            for (size_t k = 0; k < 8; k++) {
                if ((b >> k) & 1) {
                    v |= uint64_t{1} << (8 * k);
                }
            }
            entries[b] = v;
        }
    }
};
constexpr ByteToBoolsTable BYTE_TO_BOOLS{};

/// Writes `num_bits` bools (each exactly 0 or 1) to the contiguous buffer `out`, from the little-endian bit
/// packed buffer `in`. Writes exactly `num_bits` bytes; reads ceil(num_bits / 8) bytes.
inline void expand_bits_to_bools(const uint8_t *in, size_t num_bits, uint8_t *out) {
    size_t num_full_bytes = num_bits >> 3;
    size_t k = 0;
#ifdef __ARM_NEON
    // 16 input bytes -> 128 output bytes: replicate each byte across 8 lanes, test one bit per lane.
    static const uint8_t lane_bits[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t bit_mask = vld1q_u8(lane_bits);
    const uint8x16_t one = vdupq_n_u8(1);
    uint8x16_t idx = vcombine_u8(vdup_n_u8(0), vdup_n_u8(1));
    const uint8x16_t two = vdupq_n_u8(2);
    for (; k + 16 <= num_full_bytes; k += 16) {
        uint8x16_t v = vld1q_u8(in + k);
        uint8x16_t i = idx;
        for (size_t j = 0; j < 8; j++) {
            vst1q_u8(out + 8 * k + 16 * j, vandq_u8(vtstq_u8(vqtbl1q_u8(v, i), bit_mask), one));
            i = vaddq_u8(i, two);
        }
    }
#endif
    for (; k < num_full_bytes; k++) {
        memcpy(out + 8 * k, &BYTE_TO_BOOLS.entries[in[k]], 8);
    }
    size_t tail = num_bits & 7;
    if (tail) {
        uint8_t b = in[num_full_bytes];
        uint8_t *o = out + 8 * num_full_bytes;
        for (size_t t = 0; t < tail; t++) {
            o[t] = (b >> t) & 1;
        }
    }
}

/// Same as expand_bits_to_bools, but output element t goes to out + t * stride.
inline void expand_bits_to_bools_strided(const uint8_t *in, size_t num_bits, uint8_t *out, ptrdiff_t stride) {
    for (size_t t = 0; t < num_bits; t++) {
        *out = (in[t >> 3] >> (t & 7)) & 1;
        out += stride;
    }
}

}  // namespace

static pybind11::object transposed_simd_bit_table_to_numpy_uint8(
    const simd_bit_table<MAX_BITWORD_WIDTH> &table,
    size_t table_shape0,
    size_t table_shape1,
    pybind11::object out_buffer) {
    size_t num_major_bytes_in = (table_shape0 + 7) / 8;

    if (out_buffer.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out_buffer = numpy.attr("empty")(pybind11::make_tuple(table_shape1, num_major_bytes_in), numpy.attr("uint8"));
    }

    if (!pybind11::isinstance<pybind11::array_t<uint8_t>>(out_buffer)) {
        throw std::invalid_argument("Output buffer wasn't a numpy.ndarray[np.uint8].");
    }
    auto buf = pybind11::cast<pybind11::array_t<uint8_t>>(out_buffer);
    if (buf.ndim() != 2) {
        throw std::invalid_argument("Output buffer wasn't two dimensional.");
    }
    if ((size_t)buf.shape(0) != table_shape1 || (size_t)buf.shape(1) != num_major_bytes_in) {
        std::stringstream ss;
        ss << "Expected output buffer to have shape=(" << table_shape1 << ", " << num_major_bytes_in << ")";
        ss << " but its shape is (" << buf.shape(0) << ", " << buf.shape(1) << ").";
        throw std::invalid_argument(ss.str());
    }

    if (table_shape0 && table_shape1) {
        // Work in W x W bit blocks using the word type's own (SIMD) square transpose, so a single block
        // transpose covers as many bits as the table's words hold.
        constexpr size_t W = MAX_BITWORD_WIDTH;
        std::vector<bitword<W>> transpose_buffer(W);

        uint8_t *base = buf.mutable_data(0, 0);
        auto stride0 = buf.strides(0);
        auto stride1 = buf.strides(1);
        for (size_t b0 = 0; b0 < table_shape0; b0 += W) {
            size_t n0 = std::min(W, table_shape0 - b0);
            size_t n0_bytes = (n0 + 7) / 8;
            for (size_t b1 = 0; b1 < table_shape1; b1 += W) {
                size_t n1 = std::min(W, table_shape1 - b1);

                // Read a W x W block of bits out of the table.
                for (size_t d0 = 0; d0 < n0; d0++) {
                    transpose_buffer[d0] = table[b0 + d0].ptr_simd[b1 / W];
                }
                for (size_t d0 = n0; d0 < W; d0++) {
                    transpose_buffer[d0] = bitword<W>::tile8(0);
                }

                // Transpose the block of bits.
                bitword<W>::inplace_transpose_square(transpose_buffer.data(), 1);

                // Write the transposed block of bits into the numpy array.
                uint8_t *row_out = base + stride0 * b1 + stride1 * (b0 / 8);
                for (size_t d1 = 0; d1 < n1; d1++) {
                    const uint8_t *row_in = transpose_buffer[d1].u8;
                    if (stride1 == 1 && n0 == W) {
                        memcpy(row_out, row_in, W / 8);  // Constant size: compiles to plain vector stores.
                    } else {
                        // Partial block (few bits per shot) or strided output: a short byte loop is cheaper
                        // than a memcpy call per row.
                        uint8_t *cell_out = row_out;
                        for (size_t k = 0; k < n0_bytes; k++) {
                            *cell_out = row_in[k];
                            cell_out += stride1;
                        }
                    }
                    row_out += stride0;
                }
            }
        }
    }

    return out_buffer;
}

static pybind11::object transposed_simd_bit_table_to_numpy_bool8(
    const simd_bit_table<MAX_BITWORD_WIDTH> &table,
    size_t num_major_in,
    size_t num_minor_in,
    pybind11::object out_buffer) {
    if (out_buffer.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out_buffer = numpy.attr("empty")(pybind11::make_tuple(num_minor_in, num_major_in), numpy.attr("bool_"));
    }

    if (!pybind11::isinstance<pybind11::array_t<bool>>(out_buffer)) {
        throw std::invalid_argument("Output buffer wasn't a numpy.ndarray[np.bool_].");
    }
    auto buf = pybind11::cast<pybind11::array_t<bool>>(out_buffer);
    if (buf.ndim() != 2) {
        throw std::invalid_argument("Output buffer wasn't two dimensional.");
    }
    if ((size_t)buf.shape(0) != num_minor_in || (size_t)buf.shape(1) != num_major_in) {
        std::stringstream ss;
        ss << "Expected output buffer to have shape=(" << num_minor_in << ", " << num_major_in << ")";
        ss << " but its shape is (" << buf.shape(0) << ", " << buf.shape(1) << ").";
        throw std::invalid_argument(ss.str());
    }

    if (num_major_in && num_minor_in) {
        // Same W x W block transpose as transposed_simd_bit_table_to_numpy_uint8, but each transposed row is
        // expanded into bools. Writing one shot's bits at a time keeps the writes row-contiguous instead of
        // walking down one output column per table row.
        //
        // The loop order is the reverse of the uint8 version: for each block of W shots, sweep across all
        // majors. That fills each block of W output rows completely before moving on, so the (8x larger)
        // bool output is written in one sequential pass instead of being revisited once per major block.
        // The reads this scatters (W/8 bytes from each table row) stay in cache across neighboring blocks.
        constexpr size_t W = MAX_BITWORD_WIDTH;
        std::vector<bitword<W>> transpose_buffer(W);

        uint8_t *base = reinterpret_cast<uint8_t *>(buf.mutable_data(0, 0));
        ptrdiff_t stride0 = buf.strides(0);
        ptrdiff_t stride1 = buf.strides(1);
        for (size_t b1 = 0; b1 < num_minor_in; b1 += W) {
            size_t n1 = std::min(W, num_minor_in - b1);
            for (size_t b0 = 0; b0 < num_major_in; b0 += W) {
                size_t n0 = std::min(W, num_major_in - b0);

                for (size_t d0 = 0; d0 < n0; d0++) {
                    transpose_buffer[d0] = table[b0 + d0].ptr_simd[b1 / W];
                }
                for (size_t d0 = n0; d0 < W; d0++) {
                    transpose_buffer[d0] = bitword<W>::tile8(0);
                }
                bitword<W>::inplace_transpose_square(transpose_buffer.data(), 1);

                uint8_t *row_out = base + stride0 * (ptrdiff_t)b1 + stride1 * (ptrdiff_t)b0;
                for (size_t d1 = 0; d1 < n1; d1++) {
                    const uint8_t *row_in = transpose_buffer[d1].u8;
                    if (stride1 == 1) {
                        expand_bits_to_bools(row_in, n0, row_out);
                    } else {
                        expand_bits_to_bools_strided(row_in, n0, row_out, stride1);
                    }
                    row_out += stride0;
                }
            }
        }
    }

    return out_buffer;
}

static pybind11::object simd_bit_table_to_numpy_uint8(
    const simd_bit_table<MAX_BITWORD_WIDTH> &table, size_t num_major, size_t num_minor, pybind11::object out_buffer) {
    size_t num_minor_bytes = (num_minor + 7) / 8;
    if (out_buffer.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out_buffer = numpy.attr("empty")(pybind11::make_tuple(num_major, num_minor_bytes), numpy.attr("uint8"));
    }

    if (!pybind11::isinstance<pybind11::array_t<uint8_t>>(out_buffer)) {
        throw std::invalid_argument("Output buffer wasn't a numpy.ndarray[np.uint8].");
    }
    auto buf = pybind11::cast<pybind11::array_t<uint8_t>>(out_buffer);
    if (buf.ndim() != 2) {
        throw std::invalid_argument("Output buffer wasn't two dimensional.");
    }
    if ((size_t)buf.shape(0) != num_major || (size_t)buf.shape(1) != num_minor_bytes) {
        std::stringstream ss;
        ss << "Expected output buffer to have shape=(" << num_major << ", " << num_minor_bytes << ")";
        ss << " but its shape is (" << buf.shape(0) << ", " << buf.shape(1) << ").";
        throw std::invalid_argument(ss.str());
    }

    uint8_t mask = 0b11111111;
    if (num_minor & 7) {
        mask = (1 << (num_minor & 7)) - 1;
    }

    if (num_major && num_minor) {
        auto stride = buf.strides(1);
        if (stride == 1) {
            for (size_t major = 0; major < num_major; major++) {
                auto row = table[major];
                memcpy(buf.mutable_data(major, 0), row.u8, num_minor_bytes);
                *buf.mutable_data(major, num_minor_bytes - 1) &= mask;
            }
        } else {
            for (size_t major = 0; major < num_major; major++) {
                auto row = table[major];
                auto ptr = buf.mutable_data(major, 0);
                for (size_t minor = 0; minor < num_minor_bytes; minor += 1) {
                    *ptr = row.u8[minor];
                    ptr += stride;
                }
                *(ptr - stride) &= mask;
            }
        }
    }

    return out_buffer;
}

static pybind11::object simd_bit_table_to_numpy_bool8(
    const simd_bit_table<MAX_BITWORD_WIDTH> &table, size_t num_major, size_t num_minor, pybind11::object out_buffer) {
    if (out_buffer.is_none()) {
        auto numpy = pybind11::module::import("numpy");
        out_buffer = numpy.attr("empty")(pybind11::make_tuple(num_major, num_minor), numpy.attr("bool_"));
    }
    if (!pybind11::isinstance<pybind11::array_t<bool>>(out_buffer)) {
        throw std::invalid_argument("Output buffer wasn't a numpy.ndarray[np.bool_].");
    }
    auto buf = pybind11::cast<pybind11::array_t<bool>>(out_buffer);
    if (buf.ndim() != 2) {
        throw std::invalid_argument("Output buffer wasn't two dimensional.");
    }
    if ((size_t)buf.shape(0) != num_major || (size_t)buf.shape(1) != num_minor) {
        std::stringstream ss;
        ss << "Expected output buffer to have shape=(" << num_major << ", " << num_minor << ")";
        ss << " but its shape is (" << buf.shape(0) << ", " << buf.shape(1) << ").";
        throw std::invalid_argument(ss.str());
    }

    if (num_major && num_minor) {
        ptrdiff_t stride = buf.strides(1);
        for (size_t major = 0; major < num_major; major++) {
            const uint8_t *row_in = table[major].u8;
            uint8_t *out_ptr = reinterpret_cast<uint8_t *>(buf.mutable_data(major, 0));
            if (stride == 1) {
                expand_bits_to_bools(row_in, num_minor, out_ptr);
            } else {
                expand_bits_to_bools_strided(row_in, num_minor, out_ptr, stride);
            }
        }
    }

    return out_buffer;
}

pybind11::object stim_pybind::simd_bit_table_to_numpy(
    const simd_bit_table<MAX_BITWORD_WIDTH> &table,
    size_t num_major,
    size_t num_minor,
    bool bit_pack_result,
    bool transposed,
    pybind11::object out_buffer) {
    if (transposed) {
        if (bit_pack_result) {
            return transposed_simd_bit_table_to_numpy_uint8(table, num_major, num_minor, out_buffer);
        } else {
            return transposed_simd_bit_table_to_numpy_bool8(table, num_major, num_minor, out_buffer);
        }
    } else {
        if (bit_pack_result) {
            return simd_bit_table_to_numpy_uint8(table, num_major, num_minor, out_buffer);
        } else {
            return simd_bit_table_to_numpy_bool8(table, num_major, num_minor, out_buffer);
        }
    }
}

void stim_pybind::memcpy_bits_from_numpy_to_simd_bit_table(
    size_t num_major,
    size_t num_minor,
    const pybind11::object &src,
    stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> &dst) {
    if (pybind11::isinstance<pybind11::array_t<uint8_t>>(src)) {
        auto arr = pybind11::cast<pybind11::array_t<uint8_t>>(src);
        size_t num_minor_bytes = (num_minor + 7) / 8;
        auto u = arr.unchecked();
        for (size_t major = 0; major < num_major; major++) {
            auto row = dst[major];
            for (size_t minor_byte = 0; minor_byte < num_minor_bytes; minor_byte++) {
                uint8_t v = u(major, minor_byte);
                row.u8[minor_byte] = v;
            }
            // Clear overwrite.
            for (size_t k = num_minor; k < num_minor_bytes * 8; k++) {
                row[k] = false;
            }
        }
    } else if (pybind11::isinstance<pybind11::array_t<bool>>(src)) {
        auto arr = pybind11::cast<pybind11::array_t<bool>>(src);
        auto u = arr.unchecked();
        for (size_t major = 0; major < num_major; major++) {
            auto row = dst[major];
            for (size_t minor = 0; minor < num_minor; minor++) {
                row[minor] = u(major, minor);
            }
        }
    } else {
        throw std::invalid_argument("Expected a 2-dimensional numpy array with dtype=np.uint8 or dtype=np.bool_");
    }
}

simd_bit_table<MAX_BITWORD_WIDTH> bit_packed_numpy_uint8_array_to_transposed_simd_table(
    const pybind11::array_t<uint8_t> &data_u8, size_t expected_bits_per_shot, size_t *num_shots_out) {
    if (data_u8.ndim() != 2) {
        throw std::invalid_argument("data must be a 2-dimensional numpy array with dtype=np.uint8 or dtype=np.bool_");
    }

    size_t num_shots = data_u8.shape(0);
    *num_shots_out = num_shots;

    size_t expected_bytes_per_shot = (expected_bits_per_shot + 7) / 8;
    size_t actual_bytes_per_shot = data_u8.shape(1);
    if (actual_bytes_per_shot != expected_bytes_per_shot) {
        std::stringstream ss;
        ss << "Expected " << expected_bits_per_shot << " bits per shot. ";
        ss << "Got bit packed data (dtype=np.uint8) but data.shape[1]=";
        ss << actual_bytes_per_shot << " != math.ceil(" << expected_bits_per_shot
           << " / 8)=" << expected_bytes_per_shot;
        throw std::invalid_argument(ss.str());
    }

    simd_bit_table<MAX_BITWORD_WIDTH> result(actual_bytes_per_shot * 8, num_shots);

    auto u = data_u8.unchecked();
    for (size_t a = 0; a < num_shots; a++) {
        for (size_t b = 0; b < actual_bytes_per_shot; b++) {
            uint8_t v = u(a, b);
            for (size_t k = 0; k < 8; k++) {
                result[b * 8 + k][a] |= ((v >> k) & 1) != 0;
            }
        }
    }

    return result;
}

simd_bit_table<MAX_BITWORD_WIDTH> bit_packed_numpy_bool8_array_to_transposed_simd_table(
    const pybind11::array_t<bool> &data_bool8, size_t expected_bits_per_shot, size_t *num_shots_out) {
    size_t num_shots = data_bool8.shape(0);
    *num_shots_out = num_shots;

    if (data_bool8.ndim() != 2) {
        throw std::invalid_argument("data must be a 2-dimensional numpy array with dtype=np.uint8 or dtype=np.bool_");
    }

    size_t actual_bits_per_shot = data_bool8.shape(1);
    if (actual_bits_per_shot != expected_bits_per_shot) {
        std::stringstream ss;
        ss << "Expected " << expected_bits_per_shot << " bits per shot. ";
        ss << "Got unpacked boolean data (dtype=np.bool_) but data.shape[1]=" << actual_bits_per_shot;
        throw std::invalid_argument(ss.str());
    }
    simd_bit_table<MAX_BITWORD_WIDTH> result(actual_bits_per_shot, num_shots);

    auto u = data_bool8.unchecked();
    for (size_t a = 0; a < num_shots; a++) {
        for (size_t b = 0; b < actual_bits_per_shot; b++) {
            result[b][a] |= u(a, b);
        }
    }

    return result;
}

simd_bit_table<MAX_BITWORD_WIDTH> stim_pybind::numpy_array_to_transposed_simd_table(
    const pybind11::object &data, size_t bits_per_shot, size_t *num_shots_out) {
    if (pybind11::isinstance<pybind11::array_t<uint8_t>>(data)) {
        return bit_packed_numpy_uint8_array_to_transposed_simd_table(
            pybind11::cast<pybind11::array_t<uint8_t>>(data), bits_per_shot, num_shots_out);
    } else if (pybind11::isinstance<pybind11::array_t<bool>>(data)) {
        return bit_packed_numpy_bool8_array_to_transposed_simd_table(
            pybind11::cast<pybind11::array_t<bool>>(data), bits_per_shot, num_shots_out);
    } else {
        throw std::invalid_argument("data must be a 2-dimensional numpy array with dtype=np.uint8 or dtype=np.bool_");
    }
}

pybind11::object bits_to_numpy_bool8(simd_bits_range_ref<MAX_BITWORD_WIDTH> bits, size_t num_bits) {
    bool *buffer = new bool[num_bits];
    expand_bits_to_bools(bits.u8, num_bits, reinterpret_cast<uint8_t *>(buffer));

    pybind11::capsule free_when_done(buffer, [](void *f) {
        delete[] reinterpret_cast<bool *>(f);
    });

    return pybind11::array_t<bool>({(pybind11::ssize_t)num_bits}, {(pybind11::ssize_t)1}, buffer, free_when_done);
}

pybind11::object bits_to_numpy_uint8_packed(simd_bits_range_ref<MAX_BITWORD_WIDTH> bits, size_t num_bits) {
    size_t num_bytes = (num_bits + 7) / 8;
    uint8_t *buffer = new uint8_t[num_bytes];
    memcpy(buffer, bits.u8, num_bytes);

    pybind11::capsule free_when_done(buffer, [](void *f) {
        delete[] reinterpret_cast<uint8_t *>(f);
    });

    return pybind11::array_t<uint8_t>({(pybind11::ssize_t)num_bytes}, {(pybind11::ssize_t)1}, buffer, free_when_done);
}

pybind11::object stim_pybind::simd_bits_to_numpy(
    simd_bits_range_ref<MAX_BITWORD_WIDTH> bits, size_t num_bits, bool bit_packed) {
    if (bit_packed) {
        return bits_to_numpy_uint8_packed(bits, num_bits);
    }
    return bits_to_numpy_bool8(bits, num_bits);
}

void stim_pybind::memcpy_bits_from_numpy_to_simd(
    size_t num_bits, const pybind11::object &src, simd_bits_range_ref<MAX_BITWORD_WIDTH> dst) {
    if (pybind11::isinstance<pybind11::array_t<uint8_t>>(src)) {
        auto arr = pybind11::cast<pybind11::array_t<uint8_t>>(src);
        if (arr.ndim() == 1) {
            size_t num_bytes = (num_bits + 7) / 8;
            auto u = arr.unchecked();
            for (size_t k = 0; k < num_bytes; k++) {
                uint8_t v = u(k);
                dst.u8[k] = v;
            }

            // Clear overwrite.
            for (size_t k = num_bits; k < num_bytes * 8; k++) {
                dst[k] = false;
            }
            return;
        }
    } else if (pybind11::isinstance<pybind11::array_t<bool>>(src)) {
        auto arr = pybind11::cast<pybind11::array_t<bool>>(src);
        if (arr.ndim() == 1) {
            auto u = arr.unchecked();
            for (size_t k = 0; k < num_bits; k++) {
                dst[k] = u(k);
            }
            return;
        }
    }

    throw std::invalid_argument("Expected a 1-dimensional numpy array with dtype=np.uint8 or dtype=np.bool_");
}
