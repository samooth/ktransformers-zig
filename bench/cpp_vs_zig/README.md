# C++ reference vs Zig port — kernel-level A/B

Measures the Zig port's kernels against the **unmodified** reference kernels
from the ktransformers repo, on the same host, same shapes, same inputs, and
cross-checks the two numerically.

This is the harness that answered "why is the Zig port slower than the C++?"
for the AVX2 path, and it is the regression gate for SIMD work: if a change
makes the port fall back toward parity with a naive C++ kernel, it shows up
here immediately.

## What it compares

| side | kernel |
|---|---|
| C++ | `kt-kernel/operators/avx2/avx2_bf16_gemm.hpp` — `avx2::gemm_bf16`, compiled straight from the reference tree |
| C++ | `avx2::act_fn` from `avx2_bf16_utils.hpp` — the SwiGLU activation |
| Zig | `src/kernels/amx/gemm_224_bf16.zig` — `gemmExpert`, `GemmKernel224BF.applySwiGLU` |

Shapes are DeepSeek-V3 MoE projections (`hidden=7168`, `inter=2048`):
decode (`m=1`), small decode, and prefill at `m=32/128/512`.

## Running it

```bash
./bench/cpp_vs_zig/run.sh              # auto-detects the reference checkout
KT_REF=/path/to/ktransformers ./bench/cpp_vs_zig/run.sh
KT_PIN=9 ./bench/cpp_vs_zig/run.sh     # pin both harnesses to one core
```

| env var | meaning |
|---|---|
| `KT_REF` | ktransformers checkout. Default `/ai/repos/2026/ktransformers`. The script looks for `$KT_REF/ktransformers/kt-kernel/operators/avx2/avx2_bf16_gemm.hpp` and fails fast with a clear message if it is missing. |
| `KT_PIN` | `taskset -c $KT_PIN` both harnesses. |

**Use `KT_PIN` on a shared box, and only trust an idle machine.** A neighbour
thread competing for L3 and DRAM perturbs these numbers by 2-4x — larger than
most of the effects being measured. `run.sh` reports the load-sensitive rows
(m=1 and m=512) as noise; the m=8/32/128 rows are stable to about ±5% even
when contended, and those are the ones to compare on.

Builds are `-O3 -mavx2 -mfma` for C++ and `-OReleaseFast -mcpu=native` for
Zig. Always compare against a ReleaseFast Zig build; `zig build` alone defaults
to Debug.

## Why there is a `shim/`

`avx2_bf16_utils.hpp` includes `"llama.cpp/ggml.h"`, and llama.cpp is not
vendored in the ktransformers tree. `shim/llama.cpp/ggml.h` supplies the four
symbols those headers actually need — `ggml_bf16_t`, `ggml_compute_fp32_t`,
`GGML_BF16_TO_FP32`, `GGML_FP32_TO_BF16` — with bit-identical semantics to
`ggml-impl.h`. **The reference kernels themselves are used unmodified**; only
the missing third-party typedefs are stubbed. Verified with `objdump` that the
compiled binary really contains `vpmovzxwd` + `vpslld` + `vfmadd231ps` with four
independent accumulators, so the comparison is not accidentally measuring a
deoptimised C++ build.

## Reading the output

```
GEMM|<name> [m=..,n=..,k=..]|<ms>|<GFLOP/s>
```

`Zig vs C++` is the speed ratio (>1 means the port wins). The tail of the run
cross-checks every output matrix: `maxdiff` should stay around 1e-7 relative
(summation order only), and the activation row should report **0 differing
elements** — the two `exp` implementations differ far below bf16 resolution, so
after rounding to bf16 they agree bit for bit.

## `zig_probe.zig` — the attribution tool

`run.sh` tells you *that* the port is slower. `zig_probe.zig` tells you
*why*, by running four variants of the same dot product that differ in exactly
one structural property each:

| variant | accumulators | mul/add | loop order |
|---|---|---|---|
| V0 | 1 | mul+add | m-outer (the original shape) |
| V1 | 4 | mul+add | m-outer |
| V2 | 4 | FMA | m-outer |
| V3 | 4 | FMA | n-outer (= the C++ reference's shape) |

Build and run it directly:

```bash
zig build-exe -OReleaseFast -mcpu=native -lc \
    --dep kt -Mroot=bench/cpp_vs_zig/zig_probe.zig -Mkt=src/root.zig \
    -femit-bin=/tmp/zig_probe && /tmp/zig_probe
```

The measured conclusion (see `RESULTS.md`): **loop order is essentially the
whole story** — V2 -> V3 is a 2.7-3.4x jump at m>=32, while V0 -> V1 -> V2
together buy only ~10%. Accumulator count and FMA matter at m=1 and are
invisible once the kernel is bandwidth-bound.

This is the tool to reach for when a kernel looks slow: change one structural
property at a time and measure, instead of guessing.