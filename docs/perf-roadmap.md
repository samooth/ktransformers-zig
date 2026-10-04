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

### P0.2 — FP8 E4M3/E5M2 · `gemm_224_fp8.zig` — **DONE** (E5M2 was the missing half)

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

**What actually happened.** E4M3 was done; E5M2 had no encoder, no decoder, no
LUT, no enum value, no C API and no GEMM, so this entry was half a deliverable
under a name that read as finished. Both formats now share ONE GEMM body
(`gemmAvx2With`), instantiated per codec, because a second hand-written copy of
that loop is the same duplication that let the E4M3 and MXFP8 codecs drift.
E5M2's differences from E4M3 are all traps: bias 15 not 7, NO infinities (so the
largest finite value is 57344 and saturation must clip to `0x7B` rather than
emit the NaN byte that decodes to zero), subnormals down to 2^-16, and the
mantissa carry still landing in the exponent. Both C API pairs now have tests —
the E4M3 pair had none, because it was never declared `pub export fn` and no
suite could name it. `kt_matmul_fp8` remains E4M3-only; an E5M2 GEMM entry point
is future work for whoever has a caller, and the kernel is emitted and tested
today via the root.zig comptime reference.

---

### P0.3 — MXFP4 / MXFP8 · `gemm_224_{mxfp4,mxfp8}.zig` — **DONE** (with a correction to the plan)

**The plan above was wrong about the order, and finding that out was most of
the work.** It said "vectorize the 4-bit unpack with a `vpshufb` lookup table".
Vectorizing first would have optimized a format that could not read a real
MXFP checkpoint at all, so the block layout and the codec had to be fixed
first. Both are now correct and both GEMMs are vectorized.

**Block layout was wrong.** `scale` was `amx.bf16` (2 bytes), making MXFP8 34
bytes and MXFP4 18 against the format's **33 / 17**, with a scale encoding MXFP
does not have. The scale is **ue8m0**: one byte, an unbiased power of two,
`2^(b-127)`. That appears nowhere else in the tree — the encoding was never
implemented, only mis-laid-out. `ue8m0_to_f32` is now a pure shift,
`@as(u32, b) << 23`, taken verbatim from `mxfp8-moe.hpp::convert_ue8m0_to_fp32`
including its two documented edges (b=0 → 0.0 rather than 2^-127, b=255 →
overflow into NaN). `@sizeOf` is pinned by a `comptime` guard and a test.

Note the reference does *not* use `vpshufb` for this either — it uses a
256-entry FP32 LUT plus `_mm256_i32gather_ps` for E4M3, and a 16-entry table for
E2M1. The `vpshufb` idea does not work for FP32 output anyway: `vpshufb`
produces bytes, so it can only feed a bf16 result directly, not f32.

**E4M3 codec was wrong in six places.** Worst was the encoder masking the
3-bit mantissa round-overflow (`& 0x7`) instead of carrying it into the
exponent, so `3.875` encoded as **2.0 instead of 4.0** — a 2x error on 1/16 of
the entire normal range. Saturation also returned `0x7F`/`0xFF`, which are the
NaN encodings, so any weight above 448 became NaN; the decoder read subnormals
as normals (up to 1024x too large) and `0x7F` as 480, above `E4M3_MAX`; and
all seven subnormal encodings were flushed to zero. Now **254/256 exact round
trip**, the 2 exceptions being `0x7F`/`0xFF`, which correctly decode to 0 by
the reference's "treat as 0 to avoid propagation".

**Both GEMMs vectorized** (`gemmAvx2`), built like `kquant_simd`: n-outer so
the weight block streams once, MR=4, explicit `@mulAdd`, one `@reduce` per
output row. The reference's deferred-hsum rationale applies and is adopted —
MXFP keeps the scale separate from the payload rather than baking it into the
dequant, so the horizontal reduce hoists out of the k loop. Folding the ue8m0
scale into the weight vector is exact, since a power of two only changes the
exponent.

| format | m=1 | m=4 | m=8 |
|---|---|---|---|
| MXFP8 (33 B/blk) | 2x | 7x | 7x |
| MXFP4 (17 B/blk) | 3x | 11x | 10x |

at n=2048, k=7168, pinned, ReleaseFast, best of 5, vs the scalar triple loop in
the same binary. MXFP4 gains most, as expected: at 4 bits per weight the stream
is half the size, so loop order and FMA tiling matter more than the unpack.

`fp4x8` emits the **low nibble first**, matching `unpackMXFP4`. That ordering
gets its own test rather than relying on the oracle comparison, because a
swapped pair still produces plausible numbers instead of garbage.

**Acceptance met.** The two exact-value MXFP tests still pass, plus
GEMM-vs-scalar-oracle across 7 shapes each (m % MR, n=1, ragged k), the
nibble-order test, the block-size audit, and ue8m0 identity over all 256 bytes.

**Open:** m=1 is only 2-3x because the kernel does 8 scalar table loads per 8
weights. The vectorized **Q8_0** GEMM (P0.5) needs no table and reaches 7.8
GB/s at m=1 against MXFP8's 3.0 and MXFP4's 2.6 from the same shape, which
isolates the table as the cost. An asm `vgatherdps` is the obvious next
experiment.

---

### P0.4 — INT4 families · `gemm_224_int4.zig` — **DONE** (GPTQ), with the plan's primitive corrected

**The plan below prescribed the wrong instruction, and the reference proves it.**
It said to unpack with `vpshufb`. GPTQ-INT4 is **affine** in the nibble -
`w = (q - 8) * d` - so the conversion is one `vcvtdq2ps` plus one FMA, and a
lookup table only pays for a NON-linear codebook. The reference's own
`gptq_int4_dequant.hpp` agrees: `vpsrlvd` + `vpand` + `cvtepi32_ps` + `sub` +
`mul`, with no shuffle and no LUT anywhere in the file. Do not reach for
`pshufb` on an affine format; that is the FP8/IQ shape, not this one.

What the reference's variable shift *does* buy is the nibble interleave: the
block is byte-interleaved, so bits 4t..4t+3 of the little-endian u32 over bytes
0..3 are exactly weights 0..7 - one broadcast, one `vpsrlvd`, one mask.

**Delivered** (`2b6dbb7` kernel, `464252a` tests, `4e6e245` bench, `c9b52f6`
results plus the raw-INT4 decision). Three things that matter, in order:

1. **Bit-exactness is a GUARANTEE here, not a passing test.** i8 activations and
   4-bit weights make every product and partial sum an exact integer times an
   8-bit significand, and the tests hold activations to `|a| <= 8` so the integer
   sum stays under 2^16. Nothing in that chain can round. Anyone changing this
   kernel may rely on exactness **only while that precondition holds** - relax the
   activation bound and the oracle stops being bit-comparable, which is the trap
   to know about before trying to optimise it.
2. **The decode is 47% of the inner loop, and that is the headroom.** 43
   instructions per 32 weights, of which 20 are decode; a decode-free variant is
   about 2x. There is no cheap fix - five ops per 8 weights is the arithmetic
   floor for an f32 result. The two routes past it are an integer `vpmaddubsw`
   dot needing the per-group activation-sum correction, or an AVX-512 `vpermb`
   path. **The first route is exactly the raw-INT4 trick**, so the ABI question
   below and this 2x are the same piece of code.
3. **Do not quote the 15-24x as kernel quality.** The scalar baseline is
   pathological: ~13 cycles/weight, two multiplies per weight into a serial f32
   add chain, against 4-5 for the K-quant scalars on the same shape. What the
   speedup establishes is the loop order. The GB/s column is the number to carry,
   and it has no floor yet - `cpp_bench.cpp` has no INT4 coverage, so the harness
   can produce absolute throughput plus a control arm here, but not a ratio.

**Raw-INT4: deliberately not done, and it is an ABI question, not a missing
trick.** `KT_TYPE_I4` is the GPTQ block (18 B per 32 weights), a different
layout; there is no in-tree consumer (`kt_matmul_int4` takes the GPTQ block); and
the scale is a separate `scales[n * k/group]` array with a runtime group size
where GPTQ's is fixed at 32 - a different signature, not a flag. It is ~40 lines
once the header lands, and `nibble8` is already
`rawint4_dequant_8x4bit_unscaled` with `d = 1`. The decision deferred is whether
to fold the scale per group (the reference's way; needs an hsum per group per
output) or into the weights (needs none, but makes the accumulator non-integer).
The bit-exact oracle does **not** carry over, because raw-INT4 activations are
bf16.

**The original acceptance, for the record.** Existing INT4 tests (`16x16x32`
identity, partial-group case) pass; new GEMM-vs-dequantized-reference test.


### P0.5 — Q8_0 GEMM · `gemm_224_q8_0.zig` — **DONE** (4x at m=1, 13x at m=4/m=8)

Not on the original worklist, and it should have been: **Q8_0 is the GGUF
default weight format AND the LlamaMoe activation format**, so this was the
hottest quantized GEMM in the tree. It was also the last one still on an
`m/n/k` triple loop, and in the worst of the three orders — m-outer meant the
whole n x k weight matrix was re-streamed from DRAM once per token row.

Three call sites moved to the vectorized path: `kt_matmul_q8_0`, the
`gemmQuant` `KT_TYPE_Q8_0` arm (so every Q8_0-weight LlamaMoe forward), and the
root.zig comptime ref. `gemmQ8_0Scalar` stays as the oracle.

| m | scalar | avx2 | speedup |
|---|---|---|---|
| 1 | 11 ms | 2 ms | 4x |
| 4 | 45 ms | 3 ms | 13x |
| 8 | 90 ms | 6 ms | 13x |

at n=2048, k=7168, pinned, ReleaseFast, best of 5.

**READ THE FORMAT IN EVERY Q8 FIGURE IN THIS FILE.** Q8_0 and Q8_K are different
formats at different weights-per-byte, and both appear here:

| format | bytes/weight | recorded | where |
|---|---|---|---|
| **Q8_0** | 1.0625 | **7.8 GB/s** at m=1 | P0.5, this entry |
| **Q8_K** | 2.25 | **13.19 GB/s** | P0.1 tables, two places |

They are **not comparable**, and treating them as one "Q8" number has already
misled: the INT4 session took a Q8_0 control (correctly computed at 1.063
B/weight) and compared it against the 13.19 GB/s recorded for **Q8_K**, and read
the host as 5x slower than it was — the same-format gap is about 3x, and a
control measured in the SAME run cancels the host out entirely. Use the 7.8 GB/s
above, or better a same-run control. See `docs/measuring.md`.


**This entry is also the evidence for P0.3's open question.** Q8_0 needs no
lookup table — its i8→f32 step is pure `vpmovsxbd` + `vcvtdq2ps` — and reaches
**7.8 GB/s** at m=1, against 3.0 GB/s for MXFP8 and 2.6 GB/s for MXFP4 from the
same shape with a table lookup per vector. So the ~2.6x gap in the MX kernels is
the table, not the memory system, which makes an asm `vgatherdps` the right next
experiment rather than an assumption that the scalar loads are near the ceiling.

Disassembly (`gemm_224_q8_0.gemmAvx2`, out-of-line here, 424 insns): 28
`vfmadd`, 141 `ymm` ops, only 5 horizontal reduces, and exactly 16 `vpmovsxbd` +
16 `vcvtdq2ps` — the sign extension is mandatory, Q8_0 quants being signed and
including -128. The 35 scalar FP instructions are the per-block f16 scale
conversion (one per 32 weights) and the `@reduce` epilogue.

**Acceptance met.** Oracle comparison across 7 shapes (m % MR, n=1, k to 7168)
plus a separate sweep proving every element of [0,m) x [0,n) is written — the
scalar version zeroes `c` then accumulates while the vector version assigns
directly, so they are only equivalent with no uninitialised holes.

### P1.1 — Finish Q4_K×Q8_0 — **MOOT, superseded (see below)**

**This item is obsolete.** It proposed ~2.4x more on the fused `Q4_K x Q8_0`
kernel, all three costs being properties of *that* kernel's structure. The
fused kernel has since been removed from the inference path entirely, because
the generic AVX2 K-quant driver (P0.1) turned out to be **3.3x faster than the
fused kernel** on the decode shape — not 1.24x, which is the number this item
assumed.

Measured at m=1, n=2048, k=7168, pinned, three cores, same weights:

| path | time | bandwidth |
|---|---|---|
| fused Q8_0 (skips the activation dequant) | 5727 / 5775 / 5858 us | 1 GB/s |
| dequant (4 us) + generic bf16 driver | 1778 / 1787 / 1758 us | **4 GB/s** |

The dequant that the fused path avoids costs 4 us — 0.2% of the fused path's own
runtime — so the generic driver wins by 3.3x while doing strictly more work.
Optimizing the fused kernel further would have optimized the wrong kernel.

Two consequences worth keeping:

1. **The numerics changed, deliberately.** The fused arm skipped the BF16
   activation intermediate and was ~0.2% per activation more accurate. It was
   also the only format in `LlamaMoe` with that advantage — Q8_0, the other five
   K-quants and the nine IQ formats all dequantize activations to BF16 — so Q4_K
   is now consistent with its siblings instead of being the outlier.
2. **A test asserted a falsehood and hid it.** `llamafile_moe_capi_tests`
   claimed the BF16 round-trip was "exact for Q8_0" and compared the BF16 path
   against an F32 reference under a 1e-2 bound. It passed only because the
   fused arm was what ran. `dequantQ8_0ToBF16` is `f32_to_bf16(a_q * a_d)`, and
   while `a_q` (8-bit) is exact in bf16 the f16 scale `d` has an 11-bit
   significand against bf16's 8, so ~3 low bits are lost. The reference now
   models both BF16 activation round-trips, restoring the ~1e-5 agreement the
   test always meant to assert:

   | comparison | rel error |
   |---|---|
   | fused vs F32 reference | 6.2e-6 (what it asserted) |
   | bf16 driver vs F32 reference | 7.8e-2 (fails the bound) |
   | bf16 driver vs F32 ref, bf16 rounding modelled | 9.5e-6 |

`gemmQ4KxQ8_0` / `gemmQ4KxQ8_0Scalar` still exist, correct and tested, as the
readable statement of the fused formulation and a reference for their tests.
Deleting them plus their 4 tests is an open call.

**What would still be worth doing** is the activation dequant itself, which now
sits in front of all 16 formats (see P1.2).

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

**Do NOT extend the fused Q8_0-activation path to Q6_K/Q8_K.** This paragraph
previously recommended exactly that. It is now known to be wrong, and the
measurement is in P1.1: the one fused path that ever existed (Q4_K) turned out
to be **3.3x slower** than the generic bf16-activation driver on the decode
shape, 5727 us against 1778 us at m=1 n=2048 k=7168. The activation dequant it
skips costs 4 us out of 5727 — 0.2% — so "skipping the BF16 round-trip" is not
where the time is. Generalizing it to Q6_K would have generalized the slower
path.

The real remaining cost in front of all 16 formats is the **dequantize-to-BF16
step itself**, not the choice of activation type. That is the first paragraph of
this item and it is where the effort belongs.

---

### P1.3 — SwiGLU activation (0.68x) — **CLOSED as a measured negative result**

**The register-pressure diagnosis that used to be recorded here was wrong, and
fixing it does nothing.** Do not re-run steps 1-4 below. Measured 2026-10-04;
full write-up in TODO.md under "Stream D".

The corrected instruction counts for the 8-element body: **Zig 47, C++ 35** (not
48/40), with **5 `vbroadcastss`** inside the Zig body (not 6). 9 of the 12
instructions we spend over the reference are the price of reducing in exp
space rather than log2 space — the explicit floor (4 ops against their 1) and
the LN2_HI/LN2_LO split (3 ops they do not need). That reduction is what makes
us ~1e-8 accurate against their ~1e-6, so it is not a defect. Only the 5
broadcasts are register pressure.

Fix 1 below was implemented for real and measured: broadcasts **5 → 5** (no
change), body 47 → 49 instructions (worse), Zig timing 0.2038 ms → 0.2038 ms
(unchanged to four significant figures). A controlled probe confirmed the rest:
removing the divide, removing the clamp, and manual 2x/4x/8x unrolling are all
indistinguishable from baseline. Only deleting arithmetic outright moves the
number. The activation is bound by the bf16 widen, the narrowing store, and FP
vector throughput — not by register allocation.

**Noise floor, stated so nobody mistakes a future null result for a bug:** on
this shared host an activation change must beat ~25% to be measurable at all. A
byte-identical copy of the baseline differed from it by 14-23% across runs.

Reopen this only if you are willing to adopt the reference's own
`exp_avx2` (log2 space, ~1e-6) and give up the accuracy margin. That is a
different trade from the one this item asked for.

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


**CONSTRAINT, learned the hard way: the shared thing is the SHAPE; the primitive
is per-format. Do NOT put `vpshufb` in a shared driver.**

- `pshufb` buys a **non-linear** codebook (FP8 E4M3/E5M2, the IQ families). It is
  the wrong primitive for an **affine** one: GPTQ-INT4 is `w = (q-8)*d`, and the
  reference's `gptq_int4_dequant.hpp` contains no shuffle and no LUT at all.
  Importing `pshufb` for every format would give the affine formats a dependency
  they do not need, and a slower path besides.
- Formats that legitimately share a driver should **instantiate** it. FP8's
  `gemmAvx2With(decode8, tailDot)` takes the two format-specific steps as comptime
  parameters, so `gemmAvx2E5M2` is a wrapper and not a second copy of the loop.
  That is the shape to converge on: a driver with a **decode slot**, each format
  supplying its own decode.
- **"Left local" must mean local file with an identical shape, not a local file
  with a private shape.** There are now three hand-written GEMM bodies (K-quant via
  `kquant_simd`, FP8, INT4). Two duplications have already cost us: the
  E4M3/MXFP8 codec pair, and `gemmAvx2E5M2` before it was parameterised. INT4 is
  the newest, and writing it in the instantiable shape is what makes a future
  shared driver a one-argument move rather than a rewrite. `kquant_simd` hardcodes
  bf16 activations and per-format `QK` asserts, which is why INT4 and FP8 are not
  in it — a reason to keep them out, not a reason to let the shapes drift.

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
zig build test                         # 268 tests
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
