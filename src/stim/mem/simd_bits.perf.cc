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

#include "stim/mem/simd_bits.h"

#include <cstring>

#include "stim/perf.perf.h"

using namespace stim;

// Test RNG performance in both scalar and SIMD contexts
BENCHMARK(rng_scalar_context) {
    size_t n = 10 * 1000;
    simd_bits<64> d(n);  // W=64 context
    std::mt19937_64 rng;
    benchmark_go([&]() {
        auto num_calls = (n + 63) >> 6;
        for (size_t k = 0; k < num_calls; k++) {
            (void)(volatile uint64_t)rng();
        }
    }).goal_nanos(450);
}

BENCHMARK(rng_simd_context) {
    size_t n = 10 * 1000;
    simd_bits<MAX_BITWORD_WIDTH> d(n);  // widest SIMD context available
    std::mt19937_64 rng;
    benchmark_go([&]() {
        auto num_calls = (n + 63) >> 6;
        for (size_t k = 0; k < num_calls; k++) {
            (void)(volatile uint64_t)rng();
        }
    }).goal_nanos(450);
}

BENCHMARK(simd_bits_randomize_10K) {
    size_t n = 10 * 1000;
    simd_bits<MAX_BITWORD_WIDTH> data(n);
    std::mt19937_64 rng(0);
    benchmark_go([&]() {
        data.randomize(n, rng);
    })
        .goal_nanos(450)
        .show_rate("Bits", n);
}

BENCHMARK(simd_bits_xor_10K) {
    size_t n = 10 * 1000;
    simd_bits<MAX_BITWORD_WIDTH> d1(n);
    simd_bits<MAX_BITWORD_WIDTH> d2(n);
    benchmark_go([&]() {
        d2 ^= d1;
    })
        .goal_nanos(20)
        .show_rate("Bits", n);
}

BENCHMARK(simd_bits_not_zero_100K) {
    size_t n = 10 * 1000;
    simd_bits<MAX_BITWORD_WIDTH> d(n);
    d[600] = true;
    size_t total = 0;
    benchmark_go([&]() {
        total += d.not_zero();
    })
        .goal_nanos(32)
        .show_rate("Bits", n);
    if (total == 0) {
        std::cerr << "data dependency";
    }
}
