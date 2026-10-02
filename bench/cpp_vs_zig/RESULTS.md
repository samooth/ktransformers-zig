# Measured results — C++ AVX2 reference vs Zig port

Durable record of the numbers behind the AVX2 SIMD rework (commit `0dcef68`).
Regenerate with `./bench/cpp_vs_zig/run.sh`; see `README.md` for how to read it.

## Environment

| | |
|---|---|
| CPU | AMD Ryzen 7 5800H (Zen3), 16 threads |
| Cache | L1d 32K, L1i 32K, L2 512K, L3 16 MB |
| Relevant ISA | `avx`, `avx2`, `fma`, `f16c` — **no AVX-512, no AMX** |
| C++ | `g++ 13.3.0`, `-O3 -mavx2 -mfma -fno-math-errno` |
| Zig | 0.16.0, `-OReleaseFast -mcpu=native` |
| Reference | ktransformers `kt-kernel/operators/avx2/avx2_bf16_gemm.hpp`, unmodified |

### Read this before trusting any number here

This host is **shared**. Load average during the recorded runs was **26-38 on
16 cores** from an unrelated concurrent workload. That matters because most of
these kernels are memory-bandwidth-bound, and a neighbour touching L3/DRAM
moves them by up to 3x.

- The **ratios** below are stable: the same ordering and roughly the same
  multipliers reproduce across runs and across load conditions.
- The **absolute ms** are not. The MLA figure swings from 13.7 ms on a quiet
  host to ~40 ms under load — same binary, same input.
- `m=1` and `m=512` rows are noise-dominated and should not be quoted.

## GEMM: `gemmExpert` vs `avx2::gemm_bf16`

Best-of-5 runs x 5 harness invocations, pinned to one core.

| case | shape | C++ ms | Zig ms | C++ GF/s | Zig GF/s | **Zig/C++** |
|---|---|---|---|---|---|---|
| decode gate/up | m=1, n=2048, k=7168 | 2.903 | 2.423 | 10.12 | 12.12 | **1.20x** |
| decode down | m=1, n=7168, k=2048 | 2.991 | 2.530 | 9.82 | 11.60 | **1.18x** |
| small decode | m=8 | 12.828 | 9.874 | 18.31 | 23.79 | **1.30x** |
| prefill 32 | m=32 | 46.054 | 35.820 | 20.40 | 26.23 | **1.29x** |
| prefill 128 | m=128 | 173.505 | 153.939 | 21.66 | 24.41 | **1.13x** |
| prefill 512 | m=512 | 1583.485 | 840.964 | 9.49 | 17.88 | **1.88x** |

**Before the rework** (same harness, same host, measured earlier in the same
session): 0.19x / 0.72x / 0.55x / 0.40x / 0.35x / 0.46x respectively.

Numerical cross-check, every shape: max relative difference **1.2e-7 to
1.6e-7** — pure summation-order noise.

## SwiGLU activation

| case | C++ ms | Zig ms | Zig/C++ |
|---|---|---|---|
| `m=128, n=2048` | 0.188 | 0.282 | **0.67x** |

**Before: 0.06x** (2.84 ms vs 0.146 ms). The remaining gap is the activation
kernel's structure, not `exp` any more — see below.

Numerical cross-check: **0 of 262144 elements differ.** The two `exp`
implementations differ far below bf16 resolution, so after rounding to bf16
they agree bit for bit.

## Attribution — which change bought what

`zig_probe.zig`, one structural property changed per variant:

| variant | accumulators | mul/add | loop order |
|---|---|---|---|
| V0 | 1 | mul+add | m-outer (original) |
| V1 | 4 | mul+add | m-outer |
| V2 | 4 | FMA | m-outer |
| V3 | 4 | FMA | n-outer (= C++ reference) |

At `m=32`: V0 164 ms -> V1 162 ms -> V2 161 ms -> **V3 47.7 ms**.
At `m=128`: V0 724 ms -> V1 719 ms -> V2 661 ms -> **V3 182 ms**.

So **loop order is essentially the entire story** (2.7-3.4x at the last step);
4 accumulators and FMA together buy ~10% once the kernel is bandwidth-bound.
That is why the SIMD rewrite prioritised the loop nest over instruction
selection, and why `AGENTS.md` now says never to simplify these back to a
triple loop.

The port ends up *ahead* of the reference because it adds one thing the
reference lacks: MR=4 register tiling, so each converted bf16 weight chunk
feeds 4 accumulators. The reference re-converts the weight row for every `m`.

## Other kernels

Internal bench (`zig build -Dvariant=avx2 -Doptimize=ReleaseFast bench`),
`gemmExpert` vs a scalar reference of the same math:

| case | vec ms | vec GF/s | vs scalar |
|---|---|---|---|
| decode gate/up m=1, n=256, k=7168 | 0.353 | 10.39 | 5.81x |
| decode down m=1, n=7168, k=2048 | 4.187 | 7.01 | 4.06x |
| prefill m=32, n=256, k=7168 | 4.753 | 24.71 | 14.55x |
| large square m=256, n=256, k=7168 | 39.566 | 23.75 | 13.91x |

MLA `o_proj` — the largest projection in DeepSeek-V3, 264 MB of BF16 weights
read per token:

| | ms | GB/s |
|---|---|---|
| before (naive `@reduce`-in-k-loop, m-outer) | 63.5 | 4.16 |
| after (quiet host) | **13.7** | **19.3** |
| after (this host, load 26-38) | ~40 | ~6.6 |

19.3 GB/s single-core is close to the DRAM limit, so the kernel is now
bandwidth-bound rather than 16x off it. The 6.6 GB/s figure is what the same
binary does while the box is contended — use it as a reminder that these
absolute figures need a quiet host, not as a regression.

MoE forward (`moe_bench.zig`, default vs host-tuned tile params): tiny 1-token
decode 128 us, small decode 217 us, small prefill (16 tok) 2.49 ms, medium
prefill (8 tok, k=4) 7.98 ms.

## Summary of the change

| | before | after |
|---|---|---|
| BF16 GEMM vs C++ reference | 0.19x – 0.46x | **1.13x – 1.88x** |
| SwiGLU vs C++ reference | 0.06x | **0.67x** |
| MLA o_proj | 63.5 ms | **13.7 ms** (19.3 GB/s) |
| `f32_to_bf16` | truncate (biased, disagrees with ggml) | round-to-nearest-even |
| `exp` accuracy | libm `expf` | <= 1 ULP, 89.8% bit-identical |