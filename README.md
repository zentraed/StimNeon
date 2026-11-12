# StimNeon: Stim with ARM NEON SIMD

StimNeon is a fork of [Stim](https://github.com/quantumlib/Stim) that adds a 128-bit
[ARM NEON](https://developer.arm.com/Architectures/Neon) SIMD backend, so Stim can use
128-bit words on Apple Silicon and other 64-bit ARM CPUs.

Upstream Stim has SIMD backends only for x86 (SSE2 at 128 bits, AVX2 at 256 bits). On ARM it
falls back to plain 64-bit words. StimNeon adds a `bitword<128>` implemented with NEON
intrinsics and selects it automatically when the compiler targets NEON.

StimNeon is based on upstream commit
[`ffe0d04`](https://github.com/quantumlib/Stim/commit/ffe0d043598ed5df0851a3e6b4920359a23fafbd)
(October 28, 2026), the latest commit on Stim's `main` branch at the time of writing. The
upstream Stim README follows [below](#upstream-stim-readme).

## Results at a glance

Measured on an Apple M1 Pro against unmodified Stim at the same commit, built with the same
compiler and flags:

- **Large speedups for bit-matrix transposes and tableau updates:** 10K × 10K bit-table
  transposes are **3.3–5.5×** faster, CNOT on a 10K-qubit tableau is **2.2×** faster,
  multiplying 10K-qubit Clifford strings is **1.5×** faster, and reference samples of a large
  surface-code circuit are **1.7×** faster.
- **Large speedups for noisy sampling:** depolarizing noise in the frame simulator is
  **3.2–3.9×** faster, a d=11 surface-code frame simulation is **2.7×** faster, detector error
  model sampling is **2.3×** faster, and 256-shot sampling from the command line is
  **2.2–2.3×** faster. Most of this comes from a faster random-number sampler for macOS (see
  [Optimization notes](#optimization-notes)), which doesn't depend on NEON.
- **Moderate speedups for single-shot sampling:** **1.31–1.40×** faster.
- **No change for most of Stim:** 48 of 72 benchmarks are within measurement noise. The
  geometric-mean speedup across all 72 is **1.28×**.
- **No clear regressions.** One benchmark measured slower, and its fastest round matched the
  original (see [below](#benchmarks-that-changed)).

All 1,442 C++ tests pass, including under AddressSanitizer and UndefinedBehaviorSanitizer.

## What changed

| File | Change |
|---|---|
| [`src/stim/mem/bitword_128_neon.h`](src/stim/mem/bitword_128_neon.h) | **New.** `bitword<128>` built on `uint8x16_t`: bitwise ops (`veorq`, `vandq`, `vorrq`, `vbicq`, `vmvnq`), `popcount` via `vcntq_u8` + `vaddvq_u8`, `operator bool` via `vmaxvq_u8`, shifts, and the 128×128 in-place bit-block transpose used by `simd_bit_table` (see [Optimization notes](#optimization-notes)). Only compiled when `__ARM_NEON` is defined. |
| [`src/stim/mem/simd_word.h`](src/stim/mem/simd_word.h) | Sets `MAX_BITWORD_WIDTH = 128` when `__ARM_NEON` is defined and includes the NEON header. |
| [`src/stim/mem/simd_word.test.h`](src/stim/mem/simd_word.test.h) | Runs the word-size-parameterized tests at 64 and 128 bits on NEON. |
| [`src/stim.h`](src/stim.h) | Includes the NEON header. |
| [`src/stim/mem/simd_bits_range_ref.h`](src/stim/mem/simd_bits_range_ref.h) | `for_each_word` loops are unrolled 4× for words wider than 64 bits. |
| [`src/stim/mem/simd_bits_range_ref.inl`](src/stim/mem/simd_bits_range_ref.inl) | Hand-unrolled loops for `^=`, `\|=` and `&=`; `not_zero()` uses 8 independent accumulators for words wider than 64 bits; `countr_zero()` scans whole SIMD words. These apply to every backend, not just NEON. |
| [`src/stim/simulators/frame_simulator_util.inl`](src/stim/simulators/frame_simulator_util.inl) | Batch sampling of 64 or fewer shots uses 64-bit words. |
| [`src/stim/util_top/reference_sample_tree.cc`](src/stim/util_top/reference_sample_tree.cc), [`.h`](src/stim/util_top/reference_sample_tree.h) | Reference samples of circuits with 64 or fewer qubits use 64-bit words. |
| [`src/stim/util_bot/probability_util.cc`](src/stim/util_bot/probability_util.cc), [`.h`](src/stim/util_bot/probability_util.h) | `RareErrorIterator` samples the gap to the next error as `floor(log(u) / log(1 - p))` instead of calling `std::geometric_distribution`. Applies to every platform; the speedup is on macOS. |
| [`src/stim/simulators/frame_simulator.inl`](src/stim/simulators/frame_simulator.inl) | `reset_all()` fills only the real qubits' rows of the Z frame with random bits, and zeroes the padding rows. |
| [`src/stim/py/numpy.pybind.cc`](src/stim/py/numpy.pybind.cc) | Bit-packed results are converted to numpy in W×W blocks with the word type's own SIMD transpose (128×128 on NEON). |
| [`src/stim/mem/bitword_128_sse.h`](src/stim/mem/bitword_128_sse.h) | Small cleanups to the SSE backend (`alignas(16)`, direct `u64` access in `operator bool` and `to_u64_array`). x86 only. |
| [`CMakeLists.txt`](CMakeLists.txt) | Builds out of the box on macOS with the default Apple clang (see [Building](#building-on-macos)). |
| [`src/stim/mem/simd_bits.perf.cc`](src/stim/mem/simd_bits.perf.cc), [`simd_word.perf.cc`](src/stim/mem/simd_word.perf.cc) | Extra RNG benchmarks; `simd_compat_popcnt` now times `simd_bits::popcnt()`. |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | macOS wheels build on `macos-15` and `macos-15-intel` (GitHub retired `macos-14`). Publishing to PyPI runs only in quantumlib/Stim. |
| [`neon_benchmarks/`](neon_benchmarks) | **New.** Scripts, microbenchmark and raw data behind the performance numbers below. |
| [`src/search.py`](src/search.py) | **New.** Development helper that flags code that may build a `SpanRef` or `CircuitInstruction` pointing at a temporary `std::vector`. |

## Building on macOS

No extra setup is needed beyond Xcode's command-line tools and CMake:

```bash
git clone https://github.com/zentraed/StimNeon.git
cd StimNeon
cmake -S . -B build
cmake --build build -j
build/out/stim_test_o3   # optimized test suite
build/out/stim_test      # tests under AddressSanitizer + UBSan
build/out/stim_perf      # benchmarks
```

Binaries are written to `build/out/`. The NEON backend is selected automatically on arm64.

CMake changes that make this work:

- **GoogleTest is downloaded and built with the project's compiler** (v1.15.2, through CMake's
  `FetchContent`) instead of being taken from whatever is installed. A preinstalled GoogleTest
  built by another toolchain caused link errors and AddressSanitizer crashes. Apple clang also
  searches `/usr/local/include` before `-isystem` directories, so installed GoogleTest headers
  would override the downloaded ones; the downloaded headers are therefore passed with `-I`.
  The first configure needs internet access. To use an installed GoogleTest instead, pass
  `-DSTIM_FETCH_GTEST=OFF` (this is the default on Linux).
- **The build targets the machine's own architecture** (`uname -m`) unless you set
  `-DCMAKE_OSX_ARCHITECTURES` yourself.
- **ARM processors (`arm64`/`ARM64`) are recognized explicitly.** The test build uses `-Og`
  there to keep floating-point results consistent with the reference diagrams.

## Performance comparison

### Setup

- **Machine:** Apple M1 Pro (10 cores, 16 GB), macOS 26.3.1, on AC power.
- **Compiler:** Apple clang 21.0.0, `-O3 -DNDEBUG -fno-strict-aliasing`, identical for both
  variants.
- **Variants:**
  - **Original:** upstream Stim at `ffe0d04`, downloaded from GitHub. On ARM it uses 64-bit
    words.
  - **StimNeon:** this repository.
- **Benchmark code:** identical in both builds. The harness was modified only to also print
  unrounded timings.
- **Method:** 5 rounds of all 74 `stim_perf` benchmarks, interleaved (Original, StimNeon,
  repeat) so that machine-state drift affects both variants equally. Each benchmark runs for
  0.5 s per round. The tables report the median of the 5 rounds. Speedup is original time ÷
  new time, so above 1 means faster.

### Benchmarks that changed

Benchmarks whose change exceeds both 5% and their own run-to-run spread:

| Benchmark | Original | StimNeon | Speedup |
|---|---:|---:|---:|
| `simd_bit_table_inplace_square_transpose_diam10K` | 8.31 ms | 1.52 ms | **5.47×** |
| `biased_random_1024_1percent` | 874 ns | 185 ns | **4.72×** |
| `biased_random_1024_99percent` | 886 ns | 191 ns | **4.63×** |
| `FrameSimulator_depolarize1_100Kqubits_1Ksamples_per1000` | 8.09 ms | 2.08 ms | **3.88×** |
| `biased_random_1024_0point01percent` | 91.8 ns | 25.5 ns | **3.60×** |
| `biased_random_1024_0point1percent` | 197 ns | 59.0 ns | **3.34×** |
| `simd_bit_table_out_of_place_transpose_diam10K` | 8.40 ms | 2.57 ms | **3.27×** |
| `FrameSimulator_depolarize2_100Kqubits_1Ksamples_per1000` | 4.27 ms | 1.33 ms | **3.21×** |
| `FrameSimulator_surface_code_rotated_memory_z_d11_r100_batch1024` | 9.05 ms | 3.42 ms | **2.65×** |
| `DemSampler_surface_code_rotated_memory_z_distance11_100rounds_1024stripes` | 37.9 ms | 16.2 ms | **2.34×** |
| `main_sample256_pauliframe_b8_rep_d1000_r100` | 13.8 ms | 6.04 ms | **2.28×** |
| `tableau_cnot_10Kqubits` | 345 ns | 160 ns | **2.16×** |
| `main_sample256_detectors_b8_rep_d1000_r100` | 15.1 ms | 7.05 ms | **2.15×** |
| `reference_sample_tree_surface_code_d31_r1000000000` | 17.7 ms | 10.2 ms | **1.74×** |
| `TableauSimulator_CX_10Kqubits` | 2.11 ms | 1.33 ms | **1.58×** |
| `biased_random_1024_40percent` | 657 ns | 432 ns | **1.52×** |
| `CliffordString_multiplication_10K` | 497 ns | 328 ns | **1.51×** |
| `biased_random_1024_90percent` | 648 ns | 456 ns | **1.42×** |
| `main_sample1_pauliframe_b8_rep_d1000_r100` | 3.83 ms | 2.73 ms | **1.40×** |
| `main_sample1_detectors_b8_rep_d1000_r100` | 4.58 ms | 3.45 ms | **1.33×** |
| `main_sample1_tableau_rep_d1000_r100` | 23.7 ms | 18.1 ms | **1.31×** |
| `tableau_random_100` | 530 µs | 486 µs | **1.09×** |
| `tableau_random_256` | 3.74 ms | 3.48 ms | **1.08×** |
| `read_r8_dense_per10` | 18.5 µs | 22.8 µs | 0.81×\* |

The `biased_random_*`, `FrameSimulator_depolarize*`, `DemSampler_*` and most of the
`FrameSimulator_surface_code_*` and `main_sample256_*` gains come from the faster
random-number sampler, which doesn't depend on NEON. The transposes, the tableau benchmarks and
`CliffordString_multiplication_10K` come from the NEON backend.

\* `read_r8_dense_per10` reads a measurement file and doesn't use the sampling code changed
since the previous run of this comparison, where it measured 1.01×. StimNeon's five rounds
ranged from 18.4 to 24.9 µs, and its fastest round was within 1% of the original's (18.3 µs). The slow rounds
are most likely machine noise or code-layout effects: the same code lands at different
addresses in two differently built binaries, which changes alignment and branch prediction.
Layout alone can shift a benchmark by about 25%, so small differences elsewhere should be read
with that in mind.

The other 48 benchmarks (circuit parsing, most measurement-record I/O, error analysis,
streaming sampling, Pauli-string arithmetic, `SparseXorTable`, tableau iteration) changed by
less than their noise. Full results for every benchmark are in
[`neon_benchmarks/results/stim_perf_summary.csv`](neon_benchmarks/results/stim_perf_summary.csv).

### Regressions fixed during development

The first NEON version, measured against the earlier Stim base `901b630`, was slower than the
original on several benchmarks. The optimizations in [Optimization notes](#optimization-notes)
brought them back to within noise of the original. The two single-shot sampling benchmarks are
now faster than the original, mostly because of the later random-number sampler fix:

| Benchmark | First version (vs. `901b630`) | StimNeon (vs. `ffe0d04`) | Fix |
|---|---:|---:|---|
| `main_sample1_pauliframe_b8_rep_d1000_r100` | 0.66× | **1.40×** | 64-bit words for ≤64 shots |
| `main_sample1_detectors_b8_rep_d1000_r100` | 0.69× | **1.33×** | 64-bit words for ≤64 shots |
| `tableau_random_10` | 0.79× | 1.08× | faster NEON transpose |
| `reference_sample_tree_nested_circuit` | 0.91× | 0.97× | 64-bit words for ≤64 qubits |
| `tableau_random_1000` | 0.90× | 0.97× | mostly unrolled `for_each_word` |
| `FrameSimulator_hadamard_100Kqubits_1Ksamples` | 0.86× | 0.91×\*\* | unrolled `for_each_word` |
| `simd_compat_popcnt` | 0.96× | 1.00× | upstream's `popcnt()` loop restored |

\*\* `FrameSimulator_hadamard_100Kqubits_1Ksamples` had a run-to-run spread of 50% in this run
(1.03× in the previous one), so 0.91× is within its noise.

### Bit-array kernels (microbenchmark)

Two `stim_perf` benchmarks, `simd_bits_xor_10K` and `simd_bits_not_zero_100K`, repeat
identical work on every repetition. The compiler can hoist or merge that work, so their
timings do not measure the kernel. They are excluded from the summary above.

[`neon_benchmarks/micro.cc`](neon_benchmarks/micro.cc) measures these kernels properly by
inserting a compiler barrier after each repetition. Fastest of 5 interleaved rounds × 15
trials:

| Operation | Original, 100K bits | StimNeon | Speedup | Speedup at 10K bits |
|---|---:|---:|---:|---:|
| `a ^= b` | 188 ns | 195 ns | 0.97× | 1.00× |
| `a &= b`, `a \|= b` (per op) | 188 ns | 195 ns | 0.97× | 0.97× |
| `popcnt()` | 189 ns | 189 ns | 1.00× | 0.98× |
| `intersects()` | 173 ns | 173 ns | 1.00× | 0.98× |
| `not_zero()` | 111 ns | 93 ns | **1.19×** | 0.78× |

Raw output is in [`neon_benchmarks/results/micro_runs.log`](neon_benchmarks/results/micro_runs.log).
Clang already compiles upstream's 64-bit loops to NEON instructions automatically, so for
simple loops over bit arrays the original is effectively a NEON implementation, and there is
little left to gain.

## Optimization notes

Each of these was found by measuring, and checked against the original after the change.

- **Geometric sampling by inversion: 3.3–4.7× faster rare-error sampling on macOS.** Profiling
  noisy surface-code sampling showed 55–60% of the time in `std::poisson_distribution`,
  `std::normal_distribution`, `std::gamma_distribution` and `log`, all called from
  `std::geometric_distribution`. libc++, the C++ library on macOS, implements the geometric
  distribution as a negative binomial with k = 1, which draws a gamma sample and then a
  Poisson sample. `RareErrorIterator`, which finds the next error in a noise channel, now
  computes the gap as `floor(log(u) / log(1 - p))`, with `u` uniform on (0, 1] at 53-bit
  precision. That is exact, since P(gap ≥ k) = (1 − p)^k, and costs one 64-bit random draw
  and one `log`. Measured error rates over 2 × 10⁹ trials at p = 10⁻⁴, 10⁻³ and 5 × 10⁻³ were
  within 2 standard deviations of p. This is what speeds up the `biased_random_*`,
  depolarizing-noise and DEM-sampling benchmarks. It doesn't depend on NEON. On Linux,
  libstdc++ already computes the geometric distribution this way, so the change should make no
  speed difference there.
- **No random bits for padding rows.** `reset_all()` filled the whole Z-frame table with random
  bits. The table rounds its row count up to a multiple of the word size, so at W = 128 a
  188-qubit circuit randomized 256 rows, against 192 at W = 64, and a 17-qubit circuit
  randomized 128 rows against 64. Only real qubits' rows are randomized now, and the padding is
  zeroed.
- **numpy conversion uses the SIMD transpose.** Python's bit-packed sampling results were
  converted to numpy by transposing 64×64 blocks and writing the output one byte at a time.
  They are now transposed in W×W blocks with the word type's own `inplace_transpose_square`
  (the 128×128 NEON transpose described below), and each full row is written with a fixed
  16-byte copy.

- **128×128 bit-block transpose: 5.3× faster than the first version.** The first version ran
  seven passes over all 128 rows in memory, at about 8 instructions per row pair per pass. The
  rewrite keeps rows in NEON registers. Steps 1–8 run on 16 consecutive rows at a time and steps
  16–64 on 8 rows spaced 16 apart, so the block makes two trips through memory instead of seven
  (the steps commute). Each step maps onto one or two NEON instructions per row: `TRN1`/`TRN2`
  for the 8- to 64-bit steps, `SLI`/`SRI` for the 4-bit step, and shift + `BSL` for the 1- and
  2-bit steps. One 128×128 block now takes 97 ns, against 519 ns before and 265 ns for
  upstream's 64×64 scalar block. This speeds up large transposes, and stops small tableaus from
  paying for a slow 128-bit block. It was checked against a naive bit-by-bit transpose on 1,000
  random blocks with row strides from 1 to 16.
- **Small batches use 64-bit words.** Stim pads each batch of shots to a whole number of words,
  so one shot on 128-bit words simulates 127 padding lanes. Batch sampling of 64 or fewer shots,
  and reference samples of circuits with 64 or fewer qubits, now use 64-bit words. Larger
  inputs still use 128-bit words.
- **`for_each_word` is unrolled for wide words.** Clang auto-vectorizes and unrolls loops over
  64-bit words, but leaves loops over SIMD word types alone. The first version's Hadamard swap
  ran one 16-byte word per loop iteration, while the 64-bit version moved 128 bytes per
  iteration. Requesting 4× unrolling for wide words (`#pragma clang loop unroll_count(4)`, or
  `#pragma GCC unroll 4`) fixes every operation built on `for_each_word` at once.
- **`not_zero()` uses 8 accumulators.** With one accumulator, each `vorrq_u8` waits for the
  previous one. 8 independent accumulators let the ORs overlap. The 64-bit path keeps its
  simple loop, because manual unrolling there stops Clang from vectorizing it (it became 2×
  slower in testing).
- **`intersects()` and `popcnt()` use upstream's simple loops again.** The first version's
  rewrites measured 0.70× and 0.97× of the original.

## Known issues and next steps

- **`tableau_iter_unsigned_3q` runs at 0.92–0.95×** (0.95× in the latest run, within its
  noise). The benchmark itself asks for
  `TableauIterator<MAX_BITWORD_WIDTH>`, so the word size can't change underneath it. The
  remaining cost is copying 3-qubit tableaus padded to 128 bits. Code that iterates over tiny
  tableaus can use `TableauIterator<64>` instead.
- **`not_zero()` on short arrays** (10,000 bits) still runs at 0.78× the original, about 3 ns,
  because the leftover words after the unrolled part are handled one at a time.
- **Seeded results differ from upstream.** With a fixed `--seed`, StimNeon gives different
  (equally valid) random results than upstream Stim on the same machine. The geometric sampler
  consumes random numbers differently, a different word size draws random bits in a different
  order, and padding rows no longer use random bits. Stim's documentation already states that
  seeded results can differ between versions and machines.
- **Untested on x86.** The `for_each_word` unrolling and the 8-accumulator `not_zero()` also
  apply to the SSE2 and AVX2 backends. From the generated x86 code, `not_zero()` should be up
  to 2–3× faster there, but none of this has been measured on an x86 machine.
- **Benchmark noise and layout effects.** A few benchmarks, such as
  `FrameSimulator_hadamard_100Kqubits_1Ksamples` and several `read_*` benchmarks, have
  run-to-run spreads above 20%, and code layout can shift results by up to about 25% (see
  `read_r8_dense_per10` above). Small differences are not meaningful.

## Reproducing the benchmarks

```bash
neon_benchmarks/run_comparison.sh               # upstream ffe0d04 vs. HEAD: 5 rounds, ~7 minutes
neon_benchmarks/run_comparison.sh 1 0.05        # quick smoke test
PREV=<commit> neon_benchmarks/run_comparison.sh # also compare another local commit
python3 neon_benchmarks/analyze.py              # summarize the committed results
```

The script downloads the upstream commit from GitHub, builds both variants in a temporary
directory (your checkout is not modified), runs `stim_perf` and the microbenchmark in
interleaved rounds, and prints the summary. Set `BASE=<commit>` to compare against a different
upstream commit. Close other applications and keep the machine on AC power while it runs.

## License

StimNeon is distributed under the same [Apache 2.0 license](LICENSE) as Stim. Stim is
developed by Google's Quantum AI team; StimNeon is an independent fork and is not affiliated
with or endorsed by Google.

---

<a name="upstream-stim-readme"></a>
*The upstream Stim README follows.*

# Stim

<img align="right" width="125em" alt="Stim logo" src="https://raw.githubusercontent.com/quantumlib/Stim/refs/heads/main/doc/logo_128x128.svg">

High-performance simulation of quantum stabilizer circuits for quantum error correction.

◼︎︎&nbsp;&nbsp;[What is Stim?](#what-is-stim)<br>
◼︎︎&nbsp;&nbsp;[How do I use Stim?](#how-use-stim)<br>
◼︎&nbsp;&nbsp;[How does Stim work?](#how-stim-work)<br>
◼︎&nbsp;&nbsp;[How do I cite Stim?](#how-cite-stim)<br>
◼︎&nbsp;&nbsp;[*Subproject*: Sinter decoding sampler](glue/sample)<br>
◼︎&nbsp;&nbsp;[*Subproject*: Crumble interactive editor](glue/crumble)<br>

## <a name="what-is-stim"></a>What is Stim?

Stim is a tool for high performance simulation and analysis of quantum stabilizer circuits,
especially quantum error correction (QEC) circuits.
Typically Stim is used as a Python package (`pip install stim`), though Stim can also be used as
a command-line tool or a C++ library.

*   [Watch the 15 minute lightning talk presenting Stim at QPL2021](https://youtu.be/7m_JrJIskPM?t=895)
*   [Watch Stim being used to estimate the threshold of the honeycomb code over a
    weekend](https://www.youtube.com/watch?v=E9yj0o1LGII)

Stim's key features:

1.  **<em>Really</em> fast simulation of stabilizer circuits**.
    Have a circuit with thousands of qubits and millions of operations?
    [`stim.Circuit.compile_sampler()`](doc/python_api_reference_vDev.md#stim.Circuit.compile_sampler) will perform a few
    seconds of analysis and then produce an object that can sample shots at kilohertz rates.

2.  **Semi-automatic decoder configuration**.
    [`stim.Circuit.detector_error_model()`](doc/python_api_reference_vDev.md#stim.Circuit.detector_error_model) converts
    a noisy circuit into a detector error model (a [Tanner graph](https://en.wikipedia.org/wiki/Tanner_graph)) which can
    be used to configure decoders.
    Adding the option `decompose_operations=True` will additionally suggest how hyper errors can be decomposed into
    graphlike errors, making it easier to configure matching-based decoders.

3.  **Useful building blocks for working with stabilizers**, such as
    [`stim.PauliString`](doc/python_api_reference_vDev.md#stim.PauliString),
    [`stim.Tableau`](doc/python_api_reference_vDev.md#stim.Tableau),
    and [`stim.TableauSimulator`](doc/python_api_reference_vDev.md#stim.TableauSimulator).

Stim's main limitations are:

1.  There is no support for non-Clifford operations, such as T gates and Toffoli gates. Only stabilizer operations are
    supported.

2.  `stim.Circuit` only supports Pauli noise channels (eg. no amplitude decay). For more complex noise you must manually
    drive a `stim.TableauSimulator`.

3.  `stim.Circuit` only supports single-control Pauli feedback. For multi-control feedback, or non-Pauli feedback, you
    must manually drive a `stim.TableauSimulator`.

Stim's design philosophy:

*   **Performance is king.**
    The goal is not to be fast *enough*, it is to be fast in an absolute sense.
    Think of it this way.
    The difference between doing one thing per second (human speeds) and doing ten billion things
    per second (computer speeds) is 100 decibels (100 factors of 1.26).
    Because software slowdowns tend to compound exponentially, the choices we make can be thought of multiplicatively;
    they can be thought of as spending or saving decibels.
    For example, under default usage, Python is 100 times slower than C++.
    That's 20dB of the 100dB budget!
    A *fifth* of the multiplicative performance budget allocated to *language choice*!
    Too expensive!
    Although Stim will never achieve the glory of [30 GiB per second of
    FizzBuzz](https://codegolf.stackexchange.com/a/236630/74349), it at least *wishes* it could.

*   **Bottom up.**
    Stim is intended to be like an assembly language: a mostly straightforward layer upon which more complex layers
    can be built.
    The user may define QEC constructions at some high level, perhaps as a set of stabilizers or as a parity check
    matrix, but these concepts are explained to Stim at a low level (e.g., as circuits).
    Stim is not necessarily the abstraction that the user wants, but Stim wants to implement low-level
    pieces simple enough and fast enough that the high-level pieces that the user wants can be built on top.

*   **Backwards compatibility.**
    Stim's Python package uses semantic versioning.
    Within a major version (1.X), Stim guarantees backwards compatibility of its Python API and of its command-line API.
    Note Stim DOESN'T guarantee backwards compatibility of the underlying C++ API.

## <a name="how-use-stim"></a>How do I use Stim?

See the [Getting Started Notebook](doc/getting_started.ipynb).

Stuck?
[Get help on the quantum computing stack exchange](https://quantumcomputing.stackexchange.com)
and use the [`stim`](https://quantumcomputing.stackexchange.com/questions/tagged/stim) tag.

See the reference documentation:

*   [Stim Python API Reference](doc/python_api_reference_vDev.md)
*   [Stim Supported Gates Reference](doc/gates.md)
*   [Stim Command Line Reference](doc/usage_command_line.md)
*   [Stim Circuit File Format (.stim)](doc/file_format_stim_circuit.md)
*   [Stim Detector Error model Format (.dem)](doc/file_format_dem_detector_error_model.md)
*   [Stim Results Format Reference](doc/result_formats.md)
*   [Stim Internal Developer Reference](doc/developer_documentation.md)

## <a name="how-stim-work"></a>How does Stim work?

See [the paper describing Stim](https://quantum-journal.org/papers/q-2021-07-06-497/).
Stim makes three core improvements over previous stabilizer simulators:

1.  **Vectorized code.**
    Stim's hot loops are heavily vectorized, using 256 bit wide AVX instructions.
    This makes them very fast.
    For example, Stim can multiply Pauli strings with 100 billion terms in one second.

2.  **Reference Frame Sampling.**
    When bulk sampling, Stim only uses a general stabilizer simulator for an initial reference sample.
    After that, it cheaply derives as many samples as needed by propagating simulated errors diffed against the
    reference.
    This simple trick is *ridiculously* cheaper than the alternative: constant cost per gate, instead of linear cost
    or even quadratic cost.

3.  **Inverted Stabilizer Tableau.**
    When doing general stabilizer simulation, Stim tracks the inverse of the stabilizer tableau that was historically
    used.
    This has the unexpected benefit of making measurements that commute with the current stabilizers take
    linear time instead of quadratic time. This is beneficial in error correcting codes, because the measurements
    they perform are usually redundant and so commute with the current stabilizers.

## <a name="how-cite-stim"></a>How do I cite Stim?

When using Stim for research, [please cite](https://quantum-journal.org/papers/q-2021-07-06-497/):

```latex
@article{gidney2021stim,
  doi = {10.22331/q-2021-07-06-497},
  url = {https://doi.org/10.22331/q-2021-07-06-497},
  title = {Stim: a fast stabilizer circuit simulator},
  author = {Gidney, Craig},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens
                in den Quantenwissenschaften}},
  volume = 5,
  pages = 497,
  month = jul,
  year = 2021
}
```

## Contact

For any questions or concerns not addressed here, please email quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software
Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Copyright 2025 Google LLC.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="./doc/quantum-ai-vertical.svg">
  </a>
</div>
