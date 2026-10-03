# Performance roadmap — closing the gap with the C++ AVX2 reference

**Goal: the Zig port beats the reference kernel-for-kernel on every operator
family, not just BF16 GEMM.**

This is the working document for that effort. Every number here was measured on
the development host (see [Measurement protocol](#measurement-protocol)) and
every item is scoped to specific files so multiple devs can work in parallel.

---

## Where we stand

### vs the C++ AVX2 reference (`bench/cpp_vs_zig/run.sh`)

| case | shape | C++ ms | Zig ms | Zig/C++ |
|---|---|---|---|---|
| decode gate/up | m=1, n=2048, k=7168 | 6.066 | 3.219 | **1.88x** |
| decode down | m=1, n=7168, k=2048 | 6.897 | 3.815 | **1.81x** |
| small decode | m=8 | 35.262 | 10.926 | **3.23x** |
| prefill | m=32 | 51.988 | 41.945 | **1.24x** |
| prefill | m=128 | 226.407 | 167.866 | **1.35x** |
| prefill | m=512 | 3013.111 | 2119.151 | **1.42x** |
| **SwiGLU activation** | m=128, n=2048 | 0.180 | 0.304 | **0.59x** |

BF16 GEMM is won, and won *because* of one structural property the reference
lacks: MR=4 register tiling, so each converted bf16 weight chunk feeds four
accumulators. The reference re-converts the weight row for every `m`.

### Coverage: this is the real problem

The reference ships **10 non-BF16 AVX2 kernels**. On this host the Zig port
runs **none** of them — every one falls back to scalar, because the AMX tile
path is dead on Zen3 (`AmxFeatures.available == false`).

| operator family | C++ AVX2 header | `_mm256*` uses | Zig on AVX2 |
|---|---|---|---|
| BF16 GEMM | `avx2_bf16_gemm.hpp` | 10 | **vectorized — winning** |
| FP8 E4M3/E5M2 MoE | `fp8-moe.hpp`, `fp8_dequant.hpp` | 12, 4 | scalar + AMX tile |
| MXFP4 MoE | `mxfp4-moe.hpp` | **42** | scalar + AMX tile |
| MXFP8 MoE | `mxfp8-moe.hpp` | 18 | scalar + AMX tile |
| GPTQ-INT4 | `gptq_int4-moe.hpp` + `_dequant` + `_avxvnni` | 4, 8, 7 | scalar + AMX tile |
| raw-INT4 | `rawint4-moe.hpp` + `_avxvnni` | **18**, 7 | scalar + AMX tile |
| Q4_K | (llamafile upstream) | — | **vectorized (2.4x)** |
| Q5_K, Q6_K, Q8_K, Q2_K, Q3_K | (llamafile upstream) | — | **vectorized (P0.1 done)** |
| Q8_0, IQ1_S, IQ1_M, IQ2_\*, IQ3_\*, IQ4_XS, IQ4_NL | (llamafile upstream) | — | scalar |

### Measured headroom, per path

One LlamaMoe-shaped projection: `m=1, n=2048, k=7168`. The ceiling is the
vectorized BF16 kernel on the same host, same shape.

| path | time | weights | bandwidth | % of ceiling |
|---|---|---|---|---|
| BF16 (vectorized) — **ceiling** | 3.70 ms | 29.4 MB | 7.9 GB/s | 100% |
| Q4_K×Q8_0 **scalar** (was) | 20.37 ms | 8.3 MB | 0.41 GB/s | 5% |
| Q4_K×Q8_0 **vectorized** (now) | 8.38 ms | 8.3 MB | 0.99 GB/s | **14%** |
| Q8_K scalar (was) | 22.24 ms | 16.7 MB | 0.75 GB/s | **10%** |
| Q6_K scalar (was) | 38.69 ms | 12.0 MB | 0.31 GB/s | **4%** |
| Q8_K **vectorized** (P0.1) | 1.27 ms | 16.7 MB | **13.2 GB/s** | **66%** |
| Q6_K **vectorized** (P0.1) | 2.27 ms | 12.0 MB | **5.3 GB/s** | **27%** |
| SwiGLU activation | 0.30 ms | 1.5 MB | — | see P1.3 |

Every scalar quantized path is **6-25x off** the ceiling. This is not
prefetching or block sizes — they are dequant/arithmetic-bound. The only fix
is SIMD.

---

## The four rules that make kernels fast here

Established by measurement (`bench/cpp_vs_zig/zig_probe.zig`), not folklore.
Apply these to every kernel and most of the work is done.

1. **Loop order beats instruction selection.** n-outer / m-inner so the *weight*
   matrix streams from DRAM once instead of once per token row. Measured
   3.4-4.0x — an order of magnitude more than accumulators or FMA. Never
   simplify a GEMM back to a triple loop.
2. **Register-tile M.** The dequant conversion dominates the uop budget
   (bf16→f32 is 2 uops per 8 elements per operand, ~0.5 uops/MAC, vs 0.125
   for an FMA). With MR=4 rows in flight each converted chunk feeds 4
   accumulators. This is why we beat the reference on BF16.
3. **Explicit `@mulAdd`.** Zig defaults `fp-contract=off`; plain `a*b+c` emits
   `vmulps`+`vaddps`. Before the rewrite there were zero `vfmadd` in the `.so`.
4. **Never `@exp` in a vector loop.** No vector lowering exists; you get one
   libcall per lane. Use `amx.expScalar` / `amx.expVec` (≤1 ULP, verified).

Also: **`-Dvariant` does not pin CPU features** — it only sets C macros, so the
feature set comes from the native target. Always build
`-Doptimize=ReleaseFast`; `zig build` alone defaults to Debug.

---

## Worklist

Ordered by expected payoff per unit of effort. P0 items are independent of each
other and can run in parallel.

### P0.1 — K-quant vectorization · `gemm_224_{q2_k,q3_k,q4_k,q5_k,q6_k,q8_k}.zig` — **DONE**

Six formats now have an AVX2 path (Q4_K included: `gemmQ4_KScalar` was the one
Stream A left behind). One shared driver, `kquant_simd.zig::gemmBF16xQuant`,
parameterized per format by a comptime `Cfg` supplying the per-super-block
scale table and the 16-lane unpack — so the six kernels cannot drift apart, and
`gemmQ*_KScalar` stays as the readable definition of each format's math.

**Measured** (m=1, n=2048, k=7168, pinned, min of 6 runs, scalar and AVX2 in
the same binary/run — `bench/kquant_bench.zig`; the ceiling is the vectorized
BF16 kernel on the same shape, 1.47 ms / 20.0 GB/s. Bandwidth is over WEIGHT
bytes only, recomputed from the best time. The host is shared, so absolute ms
move by 2-3x between runs — the ratio within a row is what is stable, which is
why both paths are in the same binary and the same run):

| format | scalar | AVX2 | speedup | % of ceiling BW |
|---|---|---|---|---|
| Q8_K | 18.24 ms / 0.92 GB/s | **1.27 ms / 13.19 GB/s** | **14.4x** | 66% |
| Q6_K | 27.42 ms / 0.44 | **2.27 ms / 5.31** | **12.1x** | 27% |
| Q5_K | 26.60 ms / 0.38 | **2.20 ms / 4.60** | **12.1x** | 23% |
| Q4_K | 20.95 ms / 0.39 | **2.02 ms / 4.08** | **10.4x** | 20% |
| Q3_K | 15.26 ms / 0.41 | **2.15 ms / 2.94** | **7.1x** | 15% |
| Q2_K | 14.61 ms / 0.33 | **2.33 ms / 2.07** | **6.3x** | 10% |

Numerical agreement at that K: **rel-L2 1.7e-7 – 2.3e-7** between the two paths
(normalized by the output's L2 norm; against an f64 ground truth both are
< 1e-6, so this is f32 accumulation order, not an approximation in the kernel).

m=8: 10.7x–15.7x across the same six.

**What actually produced the win** (in order of measured effect; all three are
in the driver, so all six formats get them):

1. **Fold the scale/min into the weights** (`acc += a*(s*q - m)`) instead of
   accumulating raw quants per group and scaling the partial sums. That buys
   ONE `@reduce` per output element instead of one per scale group; at Q6_K /
   Q3_K / Q2_K granularity (one group = 16 weights) a horizontal reduce costs
   more than the broadcast-multiply it replaces. It also makes the vector path
   numerically comparable with `dequantizeRow*` + dot, which is what the tests
   assert.
2. **Comptime scale-group index.** The first cut used a runtime `g`, and the
   disassembly showed ~20 scalar instructions per 16-weight group (decomposing
   `g` into chunk/plane/half) *plus* a 4-way indirect jump for the plane
   switch. Making `g` a comptime parameter folds every byte offset into the
   load displacement: measured **1.19x (Q6_K), 1.29x (Q5_K), 1.52x (Q3_K)**,
   neutral for Q8_K/Q4_K (whose groups are 256/32 weights). Cost: ~16
   specializations of the unpack per format, ~2 KB of code.
3. **Hoist the scale table** (`prepare`, once per super-block) so the f16→f32
   conversion of `d` and the per-group scale multiply are not redone for every
   16 weights — ~30% of the inner-loop uop budget before the hoist.

Also: MR=4 register tiling over the M axis (one unpacked weight vector feeds
four accumulators) and n-outer/m-inner loop order, both inherited from the BF16
kernel (rule 1 and rule 2 above).

**On the "≥60% of the BF16 ceiling bandwidth" target: met for Q8_K (66%), not
reachable for the sub-byte formats on AVX2 with bf16 activations.** The
ceiling is DRAM-bound (28 MB of weights at 20 GB/s). A 4-bit format at the
same shape moves 8 MB — which *fits* in this host's 16 MB L3 — so bandwidth is
not its limit: arithmetic is. Counting the emitted instructions (disassembly,
m=1 path), the inner loop is ~16 uops per 16 weights ≈ **1.0 uop/weight**, of
which 6 are the bf16 activation widening and 4 are the quant→f32 conversion.
At 4 uops/cycle and ~3.5 GHz that caps any implementation of this shape at
~14-16 G weights/s, i.e. **~8 GB/s for Q4_K** (0.5625 B/weight) and ~12 GB/s
for Q6_K. Measured 4.08 and 5.31 GB/s = **~50% and ~45% of the arithmetic
ceiling** — the remaining gap is LLVM's 16-lane ops splitting into 128-bit
halves plus loop-carried pressure, not a missing optimization of the same
shape. Getting past ~8 GB/s needs a different operand type, not a wider
vector: with **integer** activations (`vpmaddwd`) the conversion cost per
weight halves, which is exactly why `gemmQ4KxQ8_0` (Q8_0 activations) is the
faster path and why LlamaMoe uses it for decode.

**Remaining lever (unclaimed):** at m=1 the activation widening is repeated for
every one of the n output columns. Blocking n by 4 (one widened activation
feeding 4 column accumulators — the mirror of the MR=4 tiling that already
gives m=8 its 2x) should buy ~1.5x. Register budget works out: 4 columns ×
16 lanes of f32 = 8 ymm, +2 activation +2 weight temporaries = 12 of 16.

### P0.2 — FP8 E4M3/E5M2 · `gemm_224_fp8.zig`

**Why:** direct port target — the reference has `fp8-moe.hpp` (12 AVX2
intrinsics) and `fp8_dequant.hpp` (4) to diff against, so there is no
design risk. FP8 is what `kt_dsv3_*` / the FP8 layerwise-transport path uses.

**What to do.** `gemmFullTileScalar` is the scalar entry. E4M3 decode is a
shift + mask + exponent-bias fixup; do it 8 lanes at a time and apply the
per-row scale with one `vmulps`. E5M2 adds the sign bit. If the scales are
per-16 or per-32 groups (check the reference), unroll to that group size and
accumulate separately per group.

**Acceptance.** Existing FP8 tests pass; a new vectorized sibling matches
`gemmFullTileScalar` to f32 rounding on random blocks; the dequant
round-trip stays byte-exact.

---

### P0.3 — MXFP4 / MXFP8 · `gemm_224_{mxfp4,mxfp8}.zig`

**Why:** the *heaviest* vectorization in the reference — `mxfp4-moe.hpp` has 42
`_mm256` intrinsic uses. Whatever they are doing to get that, it is
presumably the win.

**What to do.** MXFP4 = 4-bit weights with a **shared exponent per 32
elements**; MXFP8 = 8-bit with a shared scale per 32. So the inner loop is
32 weights → 1 scale. The 4-bit unpack is best done with a `vpshufb` lookup
table over the packed bytes (one LUT handles all 16 nibble values → f32),
rather than shifts and masks. Accumulate 32-element groups separately and
apply the group scale before adding to the total, which is exactly what the
per-32 scale layout wants.

**Acceptance.** Existing MXFP tests (2 exact-value tests each) keep passing,
plus a GEMM-vs-dequantized-reference test with varied per-32 scales.

---

### P0.4 — INT4 families · `gemm_224_int4.zig` (+ raw-INT4 / GPTQ-INT4)

**Why:** the reference has three AVX2 INT4 kernels (`rawint4-moe` with 18
intrinsics, `gptq_int4-moe`, `gptq_int4_avxvnni-moe`) — the most INT4 coverage
of any family. Zig has one `gemm_224_int4.zig` (GPTQ-style groups of 32 with
scales+zero-points) and no raw-INT4 kernel.

**What to do.** GPTQ-INT4 is the friendliest: 4-bit values, a scale and a
zero-point per 32-element group, so it is a `vpshufb` unpack plus an
`vfmadd` chain with a per-group scale and subtract. Raw-INT4 is the simplest
(nibble unpack, one global scale). The `_avxvnni` variants need VNNI — out of
scope on this host, keep the scalar path as the fallback.

**Also:** there is no raw-INT4 kernel in Zig at all. Either add one or record
deliberately that INT4 covers GPTQ only.

**Acceptance.** Existing INT4 tests (`16x16x32 identity`, partial-group case)
pass; new GEMM-vs-dequantized-reference test.

---

### P1.1 — Finish Q4_K×Q8_0 (another ~2.4x available)

Currently 14% of ceiling. Three known costs, in order of size:

1. **`dot_1` is recomputed for every output column.** It is the sum of the
   activations and does not depend on `j` at all — at n=2048 that is 2048x
   redundant work. Fix by restructuring to outer loop over
   (i, super-block, group) with `j` innermost, which turns weight access into a
   constant-stride walk that the prefetcher handles. Alternative: a small
   per-row scratch of `m * (k/64) * 2` floats.
2. **The four `@reduce(.Add)` horizontal reductions per 64 weights dominate.**
   Accumulate at `@Vector(32, f32)` across a whole super-block and reduce once,
   or restructure so the nibble halves share an accumulator before reduction.
3. **`getScaleMinK4` and `f16_to_f32` are scalar with a branch** — 8 + 4 calls
   per super-block. Hoist the scale unpack: all 8 (sc, m) pairs can be
   extracted once per super-block into scalars.

---

### P1.2 — Vectorize the dequant-to-BF16 paths · all `gemm_224_*.zig`

P0.1 closed the *GEMM* side for all six K-quants (both activation types for
Q4_K), but the `dequantizeRow*` functions themselves — which `gemmQ8_0Scalar`,
all the IQ* kernels, `kt_to_float`, and the AMX tile paths still call — are
still scalar per element. So this item is now about the standalone dequantize
rows, not about the K-quant GEMMs.

The `dequantizeRow*` functions themselves are scalar too and run per weight
row — vectorizing dequant alone buys a lot before the GEMM is touched. Cheap
first win: `dequantizeRowQ4_K` writes 256 floats per block with a scalar
`d1*(q&0xF) - m1`; that is 4 `vpmovzxwd` + shifts + 4 FMAs per block.

**Extend the fused Q8_0-activation path beyond Q4_K.** `runMatmulScratch`
special-cases one format; making it table-driven over `(weight_type,
has_fused_path)` would let Q6_K/Q8_K skip the BF16 round-trip at m=1 too.
Note the payoff is smaller now: with integer activations `vpmaddwd` halves the
conversion cost per weight, which is why P0.1's numbers for the bf16-activation
path (28% of the BF16 ceiling's bandwidth for Q6_K) are where a Q8_0 path
would roughly double. Worth doing for Q6_K, which is the second most common
GGUF format.

---

### P1.3 — SwiGLU activation (0.59x) — diagnosed, needs register pressure work

Instruction counts for the 8-element body: **Zig 48, C++ 40.** Not
algorithmic — both do `vpmovzxwd` + `vpslld` + an FMA polynomial. It is
register allocation: `expVec` has ~13 simultaneously-live constants
(7 polynomial coefficients + `INV_LN2`, `LN2_HI`, `LN2_LO`, `0.5`, `1.0`, two
clamp bounds) against 16 ymm, so LLVM emits **6 `vbroadcastss` inside the
loop** rematerializing constants, where C++ hoists all of them before it.

Zig also emits 8 FMAs to C++'s 6 — a deliberate accuracy choice (degree-6 vs
degree-5: 1.0e-8 vs ~1e-6 relative truncation). Do not "fix" that by
regressing accuracy.

**Fix by cutting peak live constants, in this order:**
1. Split `expVec` into two phases — split/floor/pow2, *then* the polynomial —
   so the first group's constants are dead before the second's are born.
2. Hoist the clamp bounds out of the per-vector path (clamp once per call).
3. Fold `0.5` and `1.0` into fewer live values by algebraic rearrangement.
4. *(Last resort)* drop to degree-5 and give up the accuracy margin.

**Tried and reverted — do not repeat.** Replacing the `[VEC_LEN]f32` array
staging in `applySwiGLU` with direct `*align(1)` vector loads plus a
vectorized RNE narrowing store. Measured back-to-back against the same binary
six times: **median 1.02x**, and the `.so` grew 29 instructions. The array
staging is not the cause.

---

### P2.1 — Promote shared primitives into the arch layer

`widen8` lives in `gemm_224_bf16.zig` but every quant kernel needs the same
bf16 widening. Move it (and a matching RNE narrowing store) to
`src/kernels/arch/amx.zig` so P0.1-P0.4 share one copy. **Do this first** — it
is the one change that makes the parallel streams easier, and it is the only
item here that requires editing `arch/amx.zig` (coordinate via a CLAIM).

---

## Where the K-quant GEMM tests live

`tests/kernels/simd_kernels_tests.zig`, one per format, asserting
`fused == dequantizeRow* + f32 dot`. **Not** in `gemm_224_*_k.zig`: a `test`
block inside a file that belongs to the `kt` module is never executed by
`zig build test` (Zig collects tests from the root file's module only), which
is how two broken copies of those tests sat in `gemm_224_q{3,6}_k.zig`
unnoticed. `tests/kernels/test_kernels.zig` holds the authoritative copies of
the per-format block/quantize/GEMM tests.

## Parallel-work rules

Non-negotiable, learned the hard way:

1. **CLAIM in `TODO.md` before touching anything** (protocol at the top of that
   file). It works: `REMAINING.md` had gone stale precisely because four of
   five tasks were already done and nobody re-checked.
2. **Do not edit `build.zig`, `test_kernels.zig`, `src/root.zig`, or
   `src/kernels/arch/amx.zig`** without coordinating via a claim. These are the
   four shared files; everything else is one-per-stream.
3. **One operator family per stream.** Each item above is a disjoint file set.
4. **Ground every "done" in a test that could have failed.** The strongest
   available invariant for a quantized kernel is
   `fused == dequantizeRow* + f32 dot` — no approximation in it. Two Q4_K
   tests passed for a week with a 128-byte out-of-bounds read because their
   data masked it.
5. **Record the before/after.** `TODO.md` open items are verified by re-running
   the check that marked them done; an unmeasured claim is indistinguishable
   from a stale one.

---

## Measurement protocol

The host is **shared**. Most of these kernels are memory-bound, so absolute
milliseconds move by up to 3x with load; ratios are stable.

```bash
./bench/cpp_vs_zig/run.sh              # A/B vs the reference
KT_PIN=9 ./bench/cpp_vs_zig/run.sh     # pin both harnesses to one core
zig build -Doptimize=ReleaseFast bench # internal gemm + MoE forward
zig build test                         # 245 tests
python3 tools/verify_abi.py            # 112 symbols + arity
```

For per-kernel sizing, compare against the **vectorized BF16 kernel on the same
shape** as the ceiling — it is the honest "what this hardware can do" number,
and it is what makes "% of ceiling" comparable across formats with different
block sizes.

`bench/cpp_vs_zig/README.md` has the harness details; `RESULTS.md` has the
committed measurement record. Note that `zig build` alone defaults to **Debug**,
and that `m=1`/`m=512` rows are noise-dominated on a busy box.

---

## Already done, for context

- BF16 GEMM: n-outer/m-inner + MR=4 tiling + `@mulAdd` → **1.24x-3.23x** of the
  reference.
- MLA/MHA matmuls: 4.6x (63.5 ms → 13.7 ms for DSV3 `o_proj`), now
  bandwidth-bound.
- `exp`: replaced per-lane libm calls with a Cephes construction, ≤1 ULP and
  89.8% bit-identical to `@exp`.
- Q4_K×Q8_0 fused dot: 2.4x (scalar → AVX2).
- K-quant GEMMs (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K): **6.3x–15.7x** over scalar via
  one shared AVX2 driver (`kquant_simd.zig`); Q8_K now at 66% of the BF16
  ceiling's bandwidth, the rest at ~45-50% of theirs.
- MLA o_proj went from 0.16x of the DRAM limit to bandwidth-bound.
---

## Ready-to-use task prompts

The worklist items above are pre-packaged as copy-paste prompts in
**`docs/perf-task-prompts.md`** — one per P0/P1 item, each self-contained
(repo paths, the four shared files not to touch, the measured numbers, the
technique, the acceptance criterion, the gates, and what to report back).
Paste one into a fresh agent session.
