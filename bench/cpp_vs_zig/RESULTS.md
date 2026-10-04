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

> Superseded by the control-armed re-measurement below, which has a noise floor
> attached to every ratio. Kept for the record of what was claimed and when;
> where the two disagree, the control-armed table is the one to believe.

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
| BF16 GEMM vs C++ reference | 0.19x – 0.46x | **1.21x – 1.34x** (control-armed, quiet box; see below) |
| SwiGLU vs C++ reference | 0.06x | **0.68x** |
| MLA o_proj | 63.5 ms | **13.7 ms** (19.3 GB/s) |
| `f32_to_bf16` | truncate (biased, disagrees with ggml) | round-to-nearest-even |
| `exp` accuracy | libm `expf` | <= 1 ULP, 89.8% bit-identical |

## Control-armed re-measurement (2026-10-04)

Produced by `zig run bench/cpp_vs_zig/measure.zig -- --iters 3 --pin 9`, the first
pass over this table that carries its own **noise floor** instead of asking the
reader to trust a ratio. Each side runs twice per iteration as a same-code
control (`cpp`/`cppctl`, `zig`/`ctl`), four arms interleaved and pinned. A row's
floor is the widest spread over all samples of identical code per side,
compounded across the two sides; a ratio inside the floor is **not printed**.

Three passes at load 2.2–3.6 / 16 cores. Ratios agree to a few percent wherever
they resolve, and the same rows resolve each time:

| case | shape | pass A | pass B | pass C | verdict |
|---|---|---|---|---|---|
| decode gate/up | m=1 | unmeas. (1.073x) | unmeas. | unmeas. (1.092x) | **not measurable** |
| decode down | m=1 | 1.04x (1.037x) | unmeas. | unmeas. (1.085x) | **not measurable** |
| small decode | m=8 | 1.33x (1.018x) | — | — | faster, stable |
| prefill 32 | m=32 | 1.34x (1.122x) | 1.29x (1.135x) | — | faster, stable |
| prefill 128 | m=128 | 1.26x (1.061x) | 1.21x (1.046x) | — | faster, stable |
| prefill 512 | m=512 | 1.23x (1.053x) | 1.26x (1.054x) | — | faster, stable |
| SwiGLU | m=128, n=2048 | 0.68x (1.084x) | 0.69x (1.077x) | — | **slower, stable** |

Bracketed values are each pass's measured floor. Pass A, full absolute numbers:

| case | C++ ms | Zig ms |
|---|---|---|
| decode gate/up m=1 | 1.2852 | 1.2973 |
| decode down m=1 | 1.3795 | 1.3261 |
| small decode m=8 | 8.0595 | 6.0741 |
| prefill 32 | 32.0278 | 23.9246 |
| prefill 128 | 125.7672 | 100.1174 |
| prefill 512 | 509.1928 | 414.7809 |
| SwiGLU m=128 | 0.1372 | 0.2017 |

### What this does to the recorded table

- **The `m>=8` GEMM range is 1.21x – 1.34x, not "1.13x – 1.88x".** The old range
  accumulated across mixed load conditions and most of its spread was the host.
  The BF16 GEMM work really is ahead of the reference; the honest figure is
  tighter and lower than advertised.
- **`prefill 512` is 1.23–1.26x, not 1.88x.** The 1.88x was a real measurement of
  a run at load 74 where this harness took 2139 ms against today's 509 ms for the
  same binary. Bandwidth-bound row, oversubscribed box.
- **`prefill 128` is 1.21–1.26x, not the 0.78x `run.sh` printed on the day.**
  Same cause reading low; a published 0.78x here reads as a regression and was
  noise.
- **The two `m=1` rows are not measurable.** They have three recorded answers on
  each side (1.20x/0.83x/0.79x and 1.18x/1.02x/0.45x) and a floor wider than most
  of them. The one time `decode down` cleared the floor it read **1.04x**, i.e.
  parity. The old prose verdict — "treat it as parity, ±20%" — was right and is
  now measured rather than asserted.
- **SwiGLU 0.68x reproduces exactly** (0.67 / 0.68 / 0.68 / 0.69 across today,
  0.67x recorded). It is the only row that was ever a reliable signal, which fits
  P1.3: it is the only row small enough to pin.

### The floor is load-dependent, and that is the actionable part

| load average | measured floor |
|---|---|
| 2 – 4 | 1.02x – 1.14x |
| 17 – 27 | ~1.17x |
| ~80 | ~1.25x |

The 14–23% noise figures quoted elsewhere (TODO.md, roadmap P1.3) were taken at
load 17–80. On a **quiet** box the floor is 1.02–1.14x, five or six rows resolve,
and the GEMM spread is 1.21–1.34x. Re-measuring is worth doing; so is choosing
when.

### Reproducing, and checking the instrument

    zig run bench/cpp_vs_zig/measure.zig -- --self-test --iters 3 --pin 9

Both arms become the same binary, so the true ratio is exactly 1.0 and **every row
must read UNMEASURABLE**. This is how the floor was established and it should be
re-run before trusting any table from this tool. It has earned that warning: the
first two implementations of the floor both looked fine and failed the self-test
— one printed a spurious "0.80x, slower" at load 80, another printed "0.96x,
slower" twice in three on a *quiet* box — because the minimum is an extreme order
statistic and two small samples of it agree by luck. A third bug found the same
way left the `ctl` arm uninitialised, silently dropping the Zig-side control from
every floor.

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
| GPTQ-INT4 / raw-INT4 | yes (4,8,7 / 18,7) | GPTQ-INT4 vectorized (AVX2 + AMX tile); **no raw-INT4 kernel at all** |
| Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K | **none** | vectorized (6.3x-15.7x over own scalar) |
| Q8_0, IQ4_NL, IQ4_XS, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | **none** | scalar |
## Absolute mode — formats the C++ reference has no kernel for

`measure.zig` normally answers "is the port faster than the reference", which is
a quotient of two measurements and needs a control on each side. Some formats have
no C++ row at all — `cpp_bench.cpp` covers BF16 GEMM and the activation, and
nothing else — and for those there is no quotient to resolve. Absolute mode
reports the Zig ms, the harness-reported figure, and `dev(zig, ctl)` as a band,
with **no ratio and no verdict**. It deliberately does not print UNMEASURABLE:
that word means "a comparison this pass could not resolve", and reusing it would
make a successful measurement read as a failed one.

The band is `zig_bench` against **itself** in the same pass. That is the entire
error budget of a single measurement — one code path, sampled N times — and it is
why one control suffices here when the ratio needs two: a ratio divides two
independent measurements, so each side contributes error and the two compound.
The distinction is asserted in the code so the ratio path cannot be "simplified"
onto one control later.

INT4 GPTQ is the first row in this mode, and it is the first row in the Zig
harness with **i8 activations** — which is the axis `kquant_simd.gemmBF16xQuant`
hardcodes (`widen16`), so the tree's "shared driver" cannot express INT4 or the
INT8 path at all today. The row is the evidence for why that driver needs a
decode slot and a widening slot rather than being widened until it fits.

Recorded pass: commit `2d91e7a`, `--iters 7 --pin 9`, filter `gptq`, load average
27.5/24.5/13.6, min-of-7 across process invocations with each harness doing its
own min-of-9 in-process.

| row | shape | Zig ms | **band** | GB/s over weight bytes |
|---|---|---|---|---|
| int4 gptq | m=1, n=2048, k=7168 | 1.860 | **±27.0%** | 3.70 |
| int4 gptq m4 | m=4, n=2048, k=7168 | 2.675 | **±51.5%** | 2.26 |

**The band is the finding, not the figure.** An earlier pass at load 5 gave 4.07
and 2.89 GB/s for these two rows; this one at load 27 gives 3.70 and 2.26. That
movement is what ±27% and ±51% mean, and it is why neither figure should be
quoted without its band or off an idle host. The m=4 row is the wider of the two
because it does four times the arithmetic over the same weight stream.

`--self-test --cpp-absent` runs no C++ arm at all, which forces every row through
the absolute path; inside `--self-test` that path prints its band and **refuses
the figure**, because both arms are then the same binary and nothing has been
validated against an independent implementation. `--cpp-absent` without
`--self-test` is refused outright.

The 1.49x that INT4 shows against the **Q8_0** control (`bench/int4_bench.zig`,
TODO.md Stream L) is *not* in this table: Q8_0 is not a harness row, so that
comparison carries no harness band and should be read as the weaker of the two
claims until it is either added to the harness or re-run on an idle host.
