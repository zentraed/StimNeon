#!/usr/bin/env bash
# Benchmark StimNeon against the upstream Stim commit it is based on.
#
# Builds stim_perf and the bit-array microbenchmark for each variant in a scratch
# directory (your checkout is not modified):
#   A-original   upstream Stim at BASE, downloaded from BASE_URL
#   B-stimneon   this repo's HEAD
#   C-previous   optional: another local commit to compare, e.g. PREV=<commit>
# All variants get identical benchmark sources, then run in interleaved rounds.
#
# Usage: [PREV=<commit>] neon_benchmarks/run_comparison.sh [ROUNDS] [TARGET_SECONDS]
set -euo pipefail

ROUNDS="${1:-5}"
TARGET_SECONDS="${2:-0.5}"
BASE_URL="${BASE_URL:-https://github.com/quantumlib/Stim.git}"
BASE="${BASE:-ffe0d043598ed5df0851a3e6b4920359a23fafbd}"
PREV="${PREV:-}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
WORK="${WORK:-$(mktemp -d -t stimneon-bench)}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"  # parallel compile jobs; set JOBS=4 to keep the machine usable
echo "Working directory: $WORK"

VARIANTS=(A-original B-stimneon)
git init -q "$WORK/A-original"
git -C "$WORK/A-original" fetch -q --depth 1 "$BASE_URL" "$BASE"
git -C "$WORK/A-original" checkout -q FETCH_HEAD
git -C "$REPO" worktree add -q --detach "$WORK/B-stimneon" HEAD
if [ -n "$PREV" ]; then
    VARIANTS+=(C-previous)
    git -C "$REPO" worktree add -q --detach "$WORK/C-previous" "$PREV"
fi

for v in "${VARIANTS[@]}"; do
    # Identical benchmark code everywhere. simd_bits.perf.cc comes from upstream because
    # StimNeon's version adds a W=128 benchmark the original cannot compile on ARM.
    cp "$WORK/A-original/src/stim/mem/simd_bits.perf.cc" "$WORK/$v/src/stim/mem/simd_bits.perf.cc.upstream"
    mv "$WORK/$v/src/stim/mem/simd_bits.perf.cc.upstream" "$WORK/$v/src/stim/mem/simd_bits.perf.cc"
    git -C "$REPO" show "HEAD:src/stim/mem/simd_word.perf.cc" > "$WORK/$v/src/stim/mem/simd_word.perf.cc"
    # The harness prints results rounded to 2 significant figures; also print exact ns.
    sed -i.bak 's|            std::cout << benchmark.name << "\\n";|            std::cout << benchmark.name << "\\tEXACT_NS=" << (actual_seconds_per_rep * 1e9) << "\\n";|' \
        "$WORK/$v/src/stim/main.perf.cc"
    grep -q EXACT_NS "$WORK/$v/src/stim/main.perf.cc" || { echo "failed to patch harness in $v" >&2; exit 1; }

    cmake -S "$WORK/$v" -B "$WORK/build-$v" -DCMAKE_BUILD_TYPE=Release -DSTIM_FETCH_GTEST=OFF > "$WORK/cmake-$v.log"
    make -C "$WORK/build-$v" -j"$JOBS" stim_perf > "$WORK/make-$v.log"
    c++ -std=gnu++20 -O3 -DNDEBUG -fno-strict-aliasing -I "$WORK/$v/src" "$REPO/neon_benchmarks/micro.cc" \
        "$WORK/$v/src/stim/mem/simd_util.cc" "$WORK/$v/src/stim/mem/bit_ref.cc" -o "$WORK/micro-$v"
    echo "built $v"
done

mkdir -p "$WORK/runs"
for r in $(seq 1 "$ROUNDS"); do
    for v in "${VARIANTS[@]}"; do
        "$WORK/build-$v/out/stim_perf" --target_seconds="$TARGET_SECONDS" > "$WORK/runs/$v.$r.txt"
        "$WORK/micro-$v" | sed "s/^/$v /" >> "$WORK/runs/micro_runs.log"
    done
    echo "round $r/$ROUNDS done"
done

python3 -I "$REPO/neon_benchmarks/analyze.py" "$WORK/runs"
echo
echo "Raw results: $WORK/runs"
echo "Clean up with: rm -rf \"$WORK\" && git -C \"$REPO\" worktree prune"
