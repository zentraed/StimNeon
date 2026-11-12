"""Summarize interleaved stim_perf runs: median per variant, speedups, noise.

Usage: python3 analyze.py [RUNS_DIR] [--csv]

RUNS_DIR holds <variant>.<round>.txt files written by run_comparison.sh (default:
results/stim_perf_runs next to this script). The variant named A-original is the
baseline; every other variant is compared against it. Speedup = original time / new
time, so above 1 means faster.
"""
import glob
import math
import os
import re
import statistics
import sys

BASELINE = "A-original"
# These two repeat identical work every repetition, so the compiler can hoist or
# merge it and the timings don't measure the kernel. micro.cc measures them properly.
UNRELIABLE = {"simd_bits_xor_10K", "simd_bits_not_zero_100K"}
LINE = re.compile(r"(\S+)\tEXACT_NS=([0-9.eE+-]+)\s*$")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
runs = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "stim_perf_runs")

data = {}  # variant -> name -> [ns...]
order = []
for path in sorted(glob.glob(os.path.join(runs, "*.*.txt"))):
    variant = os.path.basename(path).split(".")[0]
    for line in open(path):
        m = LINE.search(line)
        if not m:
            continue
        name, ns = m.group(1), float(m.group(2))
        data.setdefault(variant, {}).setdefault(name, []).append(ns)
        if name not in order:
            order.append(name)
if BASELINE not in data:
    sys.exit(f"no {BASELINE}.*.txt files in {runs}")
others = sorted(v for v in data if v != BASELINE)


def spread(xs):
    """Half the (max - min) range relative to the median, in percent."""
    return 100 * (max(xs) - min(xs)) / 2 / statistics.median(xs)


def fmt_ns(ns):
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3g} {unit}"
    return f"{ns:.3g} ns"


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


rows = []  # (name, baseline_ns, {variant: ns}, {variant: speedup}, noise_pct, runs)
for name in order:
    if not all(name in data[v] for v in [BASELINE] + others):
        continue
    base = statistics.median(data[BASELINE][name])
    meds = {v: statistics.median(data[v][name]) for v in others}
    noise = max(spread(data[v][name]) for v in [BASELINE] + others)
    rows.append((name, base, meds, {v: base / meds[v] for v in others}, noise, len(data[BASELINE][name])))

if "--csv" in sys.argv:
    print(",".join(["benchmark", "original_ns"] + [f"{v}_ns" for v in others] + [f"{v}_speedup" for v in others]
                   + ["max_spread_pct", "runs"]))
    for name, base, meds, sp, noise, n in rows:
        print(",".join([name, f"{base:.4g}"] + [f"{meds[v]:.4g}" for v in others] + [f"{sp[v]:.3f}" for v in others]
                       + [f"{noise:.1f}", str(n)]))
    sys.exit()

print(f"{'benchmark':52} {'original':>10}" + "".join(f" {v[:14]:>14} {'speedup':>7}" for v in others) + f" {'noise%':>6}")
for name, base, meds, sp, noise, n in rows:
    print(f"{name:52} {fmt_ns(base):>10}" + "".join(f" {fmt_ns(meds[v]):>14} {sp[v]:7.2f}" for v in others)
          + f" {noise:6.1f}")

print()
print(f"excluded from summary (optimized away by the compiler): {', '.join(sorted(UNRELIABLE))}")
rows = [r for r in rows if r[0] not in UNRELIABLE]
print(f"benchmarks: {len(rows)}, runs per variant: {rows[0][5] if rows else 0}")
for v in others:
    sp = [r[3][v] for r in rows]
    faster = sum(1 for r in rows if r[3][v] > 1 + max(0.05, r[4] / 100))
    slower = sum(1 for r in rows if r[3][v] < 1 - max(0.05, r[4] / 100))
    print(f"{v}: geomean speedup {geo(sp):.3f}x; clearly faster {faster}, clearly slower {slower}, "
          f"within noise {len(rows) - faster - slower}")
