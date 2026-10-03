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

## Cross-check re-measurement (2026-10-03, after the K-quant AVX2 work)

Same harness, same host, run again to confirm the table above still holds.
The box was heavily oversubscribed during this run (**load 74 on 16 cores**,
from a concurrent build), so both harnesses were run 3x alternating, pinned to
one core, and each side's best time was kept:

| case | shape | C++ ms | Zig ms | Zig/C++ today | recorded | verdict |
|---|---|---|---|---|---|---|
| decode gate/up | m=1, n=2048, k=7168 | 2.270 | 2.724 | 0.83x | 1.20x | noise-dominated (see below) |
| decode down | m=1, n=7168, k=2048 | 2.351 | 2.312 | 1.02x | 1.18x | noise-dominated |
| small decode | m=8 | 15.148 | 10.325 | **1.47x** | 1.30x | reproduced |
| prefill 32 | m=32 | 48.366 | 35.664 | **1.36x** | 1.29x | reproduced |
| prefill 128 | m=128 | 185.474 | 159.706 | **1.16x** | 1.13x | reproduced |
| prefill 512 | m=512 | 2139.05 | 1187.72 | **1.80x** | 1.88x | reproduced |
| SwiGLU | m=128, n=2048 | 0.2046 | 0.3009 | **0.68x** | 0.67x | reproduced |

Numerical cross-check unchanged: max relative difference 1.2e-7 to 1.6e-7 on
every GEMM shape, and 0 of 262144 elements differ in the activation.

**Why the `m=1` rows move.** The same Zig kernel on the same shape measures
**1.47 ms** in a quiet session (`bench/kquant_bench.zig`, which takes the best
of 6 runs) and **2.72 ms** inside this harness under load — the C++ side
moves by a similar factor (2.27 ms here vs 2.90 ms recorded). `m=1` moves
2.9x less data than `m=8`, so it is the row most exposed to whatever else the
box is doing. Treat it as "parity, ±20%", not as a regression; the
`m>=8` rows are the ones to gate on.

## Operator families the C++ reference does NOT cover

The A/B above is the whole of what can be compared against `kt-kernel`. For the
GGML K-quants (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K) and for Q8_0 and the nine IQ
formats there is **no AVX2 kernel in the reference at all** — `operators/avx2/`
contains BF16, FP8, MXFP4, MXFP8, GPTQ-INT4 and raw-INT4 only. Those formats
are therefore benchmarked against their own scalar versions instead (see
`docs/perf-roadmap.md` P0.1 and `bench/kquant_bench.zig`), and their
cross-implementation reference would have to be ported in from llamafile's
`ik_llama.cpp` sgemm, which is not in this tree.

| family | reference AVX2 kernel | Zig status on a non-AMX host |
|---|---|---|
| BF16 GEMM | yes (10 intrinsics) | vectorized, 1.16x-1.80x of reference |
| SwiGLU | (same header) | vectorized, 0.68x of reference |
| FP8 E4M3/E5M2 | yes (12 + 4) | scalar + AMX tile |
| MXFP4 / MXFP8 | yes (42 / 18) | scalar + AMX tile |
| GPTQ-INT4 / raw-INT4 | yes (4,8,7 / 18,7) | scalar + AMX tile; no raw-INT4 kernel at all |
| Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K | **none** | vectorized (6.3x-15.7x over own scalar) |
| Q8_0, IQ4_NL, IQ4_XS, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | **none** | scalar |
