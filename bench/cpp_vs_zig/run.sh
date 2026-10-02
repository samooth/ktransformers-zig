#!/usr/bin/env bash
# C++ reference kernel vs Zig port, same host, same shapes, same inputs.
#
#   ./bench/cpp_vs_zig/run.sh
#
# Env:
#   KT_REF   path to the ktransformers repo (default: /ai/repos/2026/ktransformers)
#   OPT      zig optimize mode (default: ReleaseFast)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
REF="${KT_REF:-/ai/repos/2026/ktransformers}"
AVX2_DIR="$REF/ktransformers/kt-kernel/operators/avx2"
OUT="$HERE/out"

[ -f "$AVX2_DIR/avx2_bf16_gemm.hpp" ] || {
  echo "error: reference AVX2 headers not found at $AVX2_DIR (set KT_REF)" >&2
  exit 1
}

mkdir -p "$OUT"
cd "$ROOT"

echo "== building C++ reference harness (-mavx2 -mfma, headers UNMODIFIED) =="
g++ -O3 -DNDEBUG -mavx2 -mfma -fno-math-errno \
    -I"$AVX2_DIR" -I"$HERE/shim" -I"$HERE" \
    "$HERE/cpp_bench.cpp" -o "$HERE/cpp_bench"

echo "== building Zig harness (-Doptimize=${OPT:-ReleaseFast}) =="
zig build-exe -OReleaseFast -mcpu=native \
    -lc --dep kt -Mroot="$HERE/zig_bench.zig" \
    -Mkt=src/root.zig \
    -femit-bin="$HERE/zig_bench"

# KT_PIN=<cpu> pins both harnesses to one core. Worthwhile on a busy box:
# shared L3/memory bandwidth from a neighbour perturbs timings by 2-4x.
PIN="${KT_PIN:-}"
run() { if [ -n "$PIN" ]; then taskset -c "$PIN" "$@"; else "$@"; fi; }

echo "== running (${PIN:+pinned to cpu $PIN}) =="
KT_BENCH_OUT="$OUT" run "$HERE/cpp_bench" > "$OUT/cpp.txt"
KT_BENCH_OUT="$OUT" run "$HERE/zig_bench" "$OUT" > "$OUT/zig.txt" 2>&1  # std.debug.print goes to stderr

python3 "$HERE/compare.py" "$OUT"