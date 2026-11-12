// Microbenchmarks for the simd_bits kernels whose stim_perf versions can be
// optimized away. An empty asm with a "memory" clobber after each repetition
// forces the compiler to actually redo the work every time.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "stim/mem/simd_bits.h"

using namespace stim;
constexpr size_t W = MAX_BITWORD_WIDTH;

template <typename F>
double best_ns(F &&f, size_t reps) {
    std::vector<double> samples;
    for (int trial = 0; trial < 15; trial++) {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t k = 0; k < reps; k++) {
            f();
            asm volatile("" ::: "memory");
        }
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / reps);
    }
    std::sort(samples.begin(), samples.end());
    return samples[0];  // fastest of 15 trials
}

int main() {
    size_t sink = 0;
    for (size_t n : {10000, 100000}) {
        simd_bits<W> a(n), b(n);
        std::mt19937_64 rng(5);
        a.randomize(n, rng);
        b.randomize(n, rng);
        size_t reps = 20000000 / n;
        double xor_ns = best_ns([&] { b ^= a; }, reps);
        double and_ns = best_ns([&] { b &= a; b |= a; }, reps) / 2;
        simd_bits<W> z(n);
        z[n - 1] = true;  // forces a full scan
        double nz_ns = best_ns([&] { sink += z.not_zero(); }, reps);
        double pop_ns = best_ns([&] { sink += a.popcnt(); }, reps);
        double inter_ns = best_ns([&] { sink += a.intersects(z); }, reps);
        std::printf("W=%zu n=%zu xor=%.1f and_or=%.1f not_zero=%.1f popcnt=%.1f intersects=%.1f (ns)\n", W, n, xor_ns,
                    and_ns, nz_ns, pop_ns, inter_ns);
    }
    if (sink == 42) std::printf("!");
}
