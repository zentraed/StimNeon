# Proposal: fast bool (unpacked) numpy output for StimNeon samplers

## Summary

StimNeon's 128-bit NEON words make bit-packed sampling 4–5x faster than the
PyPI Stim 1.16.0 wheel on Apple silicon. When the caller asks for the default
unpacked `np.bool_` output, that gain disappears: StimNeon is no faster than
PyPI Stim, and sometimes slower. The cause is the numpy conversion in
`src/stim/py/numpy.pybind.cc`, which writes bool output one bit at a time
through a strided pointer. This proposal replaces that loop with the block
transpose the packed path already uses, followed by a byte-to-8-bools
expansion.

## Evidence

Measured on an M5 Pro, single thread, best of 5 runs, rotated surface-code
memory circuit with `rounds = distance` and noise 1e-3. Numbers are shots per
second from `compile_sampler().sample(n)` and
`compile_detector_sampler().sample(n)`, with and without `bit_packed=True`:

| circuit, shots per call | output | PyPI bool | PyPI packed | StimNeon bool | StimNeon packed |
|---|---|---|---|---|---|
| d=5, 100k | measurements | 2.29M | 4.38M | 1.70M | 17.7M |
| d=5, 100k | detectors | 3.50M | 5.11M | 4.05M | 20.9M |
| d=11, 100k | measurements | 209K | 418K | 201K | 1.44M |
| d=11, 100k | detectors | 228K | 437K | 238K | 1.78M |
| d=11, 16k | measurements | 325K | 393K | 414K | 1.89M |

The d=5, 16k row was not recorded in this first run; it is in the baseline
table under "Results" below.

What the table shows:

- With packed output, StimNeon is about 4x faster than PyPI Stim. With bool
  output, the two are roughly equal. StimNeon's d=5 measurement bool number
  (1.70M against PyPI's 2.29M) is 26% lower, just outside the ±20% run-to-run
  noise. It is not explained yet. The likeliest causes are noise, or the
  128-bit build padding each table row to 128 shots, which makes the
  bit-at-a-time loop do slightly more work. Since the rewrite removes that loop
  entirely, this gap is not investigated further.
- StimNeon's bool output is 5–10x slower than its packed output. Bool output
  writes 8x more bytes, so some slowdown is expected, but not this much.
- Bool output speeds up much more than packed output when the batch shrinks
  from 100k to 16k shots. At d=11, StimNeon bool goes up 2.06x (201K → 414K)
  and StimNeon packed goes up 1.31x (1.44M → 1.89M); PyPI packed stays flat.
  That points to the memory access pattern, not the amount of computation.
- A quick lower bound: `np.unpackbits` applied to StimNeon's packed output is
  already 5–8x faster than StimNeon's native bool output (see the `packed +
  unpackbits` column under "Results"). Any C++ fix has to beat that.

Timings vary by about ±20% between runs. The benchmark script is in the
appendix.

## Root cause

`stim::simd_bit_table_to_numpy` (`src/stim/py/numpy.pybind.cc`, about line
221) chooses one of four converters. Every table-to-numpy path goes through it:

- samplers: `compiled_measurement_sampler.pybind.cc:34`,
  `compiled_detector_sampler.pybind.cc:62,77,80`,
  `dem_sampler.pybind.cc:73,76,78`;
- `measurements_to_detection_events.pybind.cc:132,137`;
- `frame_simulator.pybind.cc:279–298` (`FlipSimulator.to_numpy`) and `:353`
  (`get_*_flips` with no index);
- `tableau.pybind.cc:323–326` (`Tableau.to_numpy`, non-transposed path).

`simd_bits_to_numpy` (one-dimensional, via `bits_to_numpy_bool8`) is used by
`PauliString.to_numpy`, `Tableau.to_numpy` (signs), `CliffordString`, and
`FlipSimulator.get_*_flips` with an index.

- **`transposed_simd_bit_table_to_numpy_uint8`** (packed, transposed; about
  line 20) is already fast. It reads W×W bit blocks, calls
  `bitword<W>::inplace_transpose_square` (NEON on the 128-bit build), and
  stores whole rows.
- **`transposed_simd_bit_table_to_numpy_bool8`** (about line 96) is the hot
  path for `sample()` without `bit_packed`, and it is the slow one. For each
  table row (one measurement or detector) it walks every shot with
  `*ptr = row[minor]; ptr += stride;`. That is one bit extraction per
  element, and each write lands `num_major` bytes after the previous one,
  i.e. one output column at a time. Each 64-byte output cache line is written
  in 64 separate passes over the majors. At d=11 with 100k shots the output is
  about 144 MB, larger than any cache, so each line is fetched again on every
  pass. That explains the batch-size effect.
- **`simd_bit_table_to_numpy_bool8`** (not transposed; about line 186) and
  **`bits_to_numpy_bool8`** (about line 353) extract bits one at a time too.
  Their writes are contiguous, so they matter less, but they should use the
  same expansion helper.

## Proposed change

1. **A byte-to-8-bools expansion helper.** It turns one byte into 8 bytes of
   0 or 1, in little-endian bit order (bit k becomes byte k), which matches
   Stim's packed format. The portable version is a 256-entry `uint64_t`
   lookup table. On NEON, a fast path expands 16 bytes into 128 bools at once:
   `vqtbl1q_u8` to replicate bytes, `vtstq_u8` against `{1,2,4,...,128}`,
   then `vandq_u8` with 1. Only keep the NEON path if it measurably beats the
   lookup table. Every value written must be exactly 0 or 1, because numpy
   assumes bools are 0/1.

2. **Rewrite `transposed_simd_bit_table_to_numpy_bool8`** to follow the same
   steps as the uint8 version: read a W×W block, transpose it in place, then
   for each of the `n1` shots in the block expand the `n0` transposed bits
   straight into that shot's output row (`n0` bools, contiguous when
   `stride1 == 1`). This turns column-strided bit writes into row-contiguous
   block writes and reuses the NEON transpose that already exists.
   - Keep a strided fallback for output buffers where `strides(1) != 1`.
     Callers can pass their own buffer (`dets_out`/`obs_out` on the detector
     sampler, `output_*` on `FlipSimulator.to_numpy`), and pybind's
     `array_t<bool>` check accepts non-contiguous views, including
     Fortran-order arrays, step slices and negative strides.
   - Handle partial blocks (`n0 < W`, `n1 < W`) and widths that are not a
     multiple of 8 without writing past `num_major` columns.

3. **Switch `simd_bit_table_to_numpy_bool8` and `bits_to_numpy_bool8`** to the
   expansion helper, a byte or word at a time, with a scalar tail.

4. **Keep everything generic over `MAX_BITWORD_WIDTH`.** The same file must
   still compile and pass tests on the 64-bit and 256-bit (AVX2) word builds,
   not only on NEON.

## Out of scope

- The reverse direction, numpy bool to `simd_bit_table`
  (`bit_packed_numpy_bool8_array_to_transposed_simd_table` and
  `memcpy_bits_from_numpy_*`). Write it down as a follow-up if profiling
  shows it is also slow.
- Releasing the GIL during the conversion loop. It is cheap to add, since the
  loop only touches raw memory, but it changes threading behavior and should
  be its own change.
- The full table copy in `CompiledDetectorSampler::sample_to_numpy` when
  `append_observables` or `prepend_observables` is set
  (`compiled_detector_sampler.pybind.cc:70`). It is a known extra cost on
  those paths.
- Any change to the Python API, default arguments or output dtypes.
- The sampler kernels themselves.

## Acceptance criteria

1. **Correctness.** For every converter changed, bool output must equal
   `np.unpackbits(packed, axis=-1, bitorder="little")[..., :n]` of the
   corresponding packed output. Because each `sample()` call advances the RNG,
   the bool and packed outputs must come from two samplers built with the same
   seed. Seeded output also differs between word widths, so comparisons are
   only valid within one build. Add pytest cases next to the existing ones
   (`src/stim/py/compiled_measurement_sampler_pybind_test.py`,
   `compiled_detector_sampler_pybind_test.py`,
   `src/stim/simulators/dem_sampler_pybind_test.py`,
   `src/stim/simulators/frame_simulator_pybind_test.py`,
   `src/stim/stabilizers/tableau_pybind_test.py`,
   `src/stim/stabilizers/pauli_string_pybind_test.py`) covering:
   - bit counts and shot counts of 0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129,
     255, 256, 257 and 300, so that block edges are tested for W = 64, 128 and
     256;
   - caller-supplied `dets_out`/`obs_out` buffers: Fortran-order, step-sliced
     rows, step-sliced columns and negative strides;
   - `separate_observables=True`, `append_observables=True`,
     `prepend_observables=True` and `transposed=True` on the detector sampler;
   - with a fixed seed, bool output identical to the output before the
     change, so the RNG stream is untouched.
2. **The full existing test suite passes:** `pytest src/` run with the
   `stimneon` conda environment's Python after a reinstall.
3. **Performance.** With the appendix benchmark, StimNeon bool output at
   d=5 and d=11 with 100k shots should be at least 3x faster than today.
   Aim to come within about 2x of StimNeon's packed output, and beat
   `packed + np.unpackbits`. Bool throughput at 100k shots should be within
   25% of the 16k number. Some drop is expected, because 100k-shot output no
   longer fits in cache. Report before and after numbers in the same format as
   the results table.
4. **No packed-path regression.** `bit_packed=True` median throughput, over 15
   repetitions, regresses by no more than 5% against today's build.

## Build and run

- Reinstall into the benchmark environment after each change:
  `/opt/anaconda3/envs/stimneon/bin/pip install ~/PythonConda/Stimneon/StimNeon`
  (about 10 minutes per build).
- PyPI baseline interpreter: `/opt/anaconda3/envs/stimpypi/bin/python`
  (stim 1.16.0).
- Both environments are native arm64. Check with
  `python -c "import platform; print(platform.machine())"`. The Anaconda base
  install is x86_64, so do not use it.
- Optional: add a bool-output case to `neon_benchmarks/micro.cc` or
  `run_comparison.sh` so later changes are measured against upstream.

## Results

Implemented in `src/stim/py/numpy.pybind.cc` as proposed, with one change
found during benchmarking. The bool converter loops over shot blocks on the
outside and major blocks on the inside, the reverse of the uint8 converter.
With the uint8 order, each shot's output row (1441 bytes at d=11) was filled
in 12 separate 128-byte passes. The partial cache lines at block edges were
evicted and fetched again between passes, and the native bool path lost to
`packed + np.unpackbits` at d=11. With the swapped order each block of output
rows is written in one sequential sweep. The scattered table reads that this
causes, W/8 bytes per table row, stay in cache across neighboring blocks.

The NEON expansion path was kept. In a standalone microbenchmark (in cache) it
was about 2x faster than the lookup table: 1.3–1.7 ms against 2.5–3.0 ms to
expand 3.2 MB to 26 MB.

### Throughput

Measured on an Apple M1 Pro (10 cores, 16 GB), macOS 26.3.1, numpy 2.5.4. StimNeon build,
thousands of shots per second, median of 15 timed runs per
cell. The "after" columns are the per-cell median of three full benchmark runs.

| circuit, shots | meas bool before | meas bool after | speedup | det bool before | det bool after | speedup | meas packed after | meas packed + unpackbits after |
|---|---|---|---|---|---|---|---|---|
| d=5, 100k | 1,761 | 14,604 | 8.3x | 3,398 | 19,651 | 5.8x | 15,307 | 11,066 |
| d=5, 16k | 4,206 | 13,692 | 3.3x | 4,709 | 18,646 | 4.0x | 18,571 | 13,052 |
| d=11, 100k | 227 | 1,237 | 5.4x | 225 | 1,368 | 6.1x | 1,586 | 1,120 |
| d=11, 16k | 408 | 1,217 | 3.0x | 361 | 1,452 | 4.0x | 1,586 | 1,208 |

PyPI stim 1.16.0, same script, for reference (meas bool / meas packed /
det bool / det packed): d=5 100k 1,049 / 2,934 / 2,618 / 2,461; d=11 100k
153 / 418 / 229 / 451. StimNeon bool output is now 6–14x faster than PyPI
bool output, and faster than PyPI packed output.

### Interleaved ratios

Run-to-run noise on this machine was large during the session: packed
throughput, whose code did not change, moved between 0.99M and 2.0M shots/s
at d=11 across runs. To remove drift, `bool`, `packed` and `packed +
unpackbits` were timed interleaved, 40 rounds each, in one process
(measurement sampler, median times):

| circuit, shots | bool / packed time | bool / (packed + unpackbits) time |
|---|---|---|
| d=5, 100k | 1.02x | 0.75x |
| d=5, 16k | 1.27x | 0.97x |
| d=11, 100k | 1.35x | 0.93x |
| d=11, 16k | 1.41x | 0.98x |

### Acceptance criteria

1. **Correctness: met.** 94 new parametrized tests in the six files listed
   under criterion 1. They cover sizes 0 through 300 (including the 63/64/65,
   127/128/129 and 255/256/257 edges), Fortran-order, step-sliced and
   negative-stride `dets_out`/`obs_out`, the observable options and
   `transposed=True`, `DemSampler`, `FlipSimulator.to_numpy` (both
   orientations) and `get_measurement_flips`, `Tableau.to_numpy` and
   `PauliString.to_numpy`. All also passed against the old build. Seeded bool
   outputs from the measurement, detector (append and separate observables)
   and DEM samplers are byte-identical before and after the change. The file
   also compiles cleanly (`-fsyntax-only`) for the 64-bit (no NEON) and AVX2
   word widths. Tests have only been run on the NEON build.
2. **Full suite: met.** `pytest src/`: 574 passed, 2 skipped.
3. **Performance: met.** At 100k shots: 5.4–8.3x faster (target 3x), within
   1.02–1.35x of packed (target 2x), and faster than `packed + np.unpackbits`
   in every case. Bool throughput at 100k is within 7% of the 16k number,
   and is the higher of the two at both distances.
4. **No packed regression: not measurable to 5% on this machine, no evidence
   of one.** `transposed_simd_bit_table_to_numpy_uint8` and the dispatcher are
   unchanged. Across runs, after-change packed medians were both above and
   below the single before-change run (d=11 100k measurements 1,834 before
   against 986–1,998 after; detectors 1,426 before against 1,716–2,316 after).
   Confirming ≤5% needs both builds installed side by side and timed
   interleaved, on a quiet machine.

## Appendix: benchmark script

Run it with each interpreter:
`/opt/anaconda3/envs/stimneon/bin/python packtest.py [reps]`.
It reports the median of `reps` timed runs (default 15), in thousands of
shots per second. Columns: `mb`/`mp` measurement bool/packed, `mu` measurement
packed followed by `np.unpackbits`, `db`/`dp` detector bool/packed.

```python
import sys, time, statistics, stim, numpy as np
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
def timeit(fn, n):
    fn(n)
    ts = []
    for _ in range(REPS):
        s = time.perf_counter(); fn(n); ts.append(time.perf_counter() - s)
    return n / statistics.median(ts)
for d in (5, 11):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3,
        after_reset_flip_probability=1e-3)
    m, det = c.compile_sampler(), c.compile_detector_sampler()
    nm = c.num_measurements
    for n in (100_000, 16_384):
        r = dict(
            mb=timeit(lambda k: m.sample(k), n),
            mp=timeit(lambda k: m.sample(k, bit_packed=True), n),
            mu=timeit(lambda k: np.unpackbits(m.sample(k, bit_packed=True), axis=1, bitorder="little")[:, :nm], n),
            db=timeit(lambda k: det.sample(k), n),
            dp=timeit(lambda k: det.sample(k, bit_packed=True), n),
        )
        print(f"d={d:>2} n={n:>6}  " + "  ".join(f"{k} {v/1e3:>9,.0f}K" for k, v in r.items()), flush=True)
```
