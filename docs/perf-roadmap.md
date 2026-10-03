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
| Q5_K, Q6_K, Q8_K, Q2_K, Q3_K | (llamafile upstream) | — | scalar |
| Q8_0, IQ1_S, IQ1_M, IQ2_\*, IQ3_\*, IQ4_XS, IQ4_NL | (llamafile upstream) | — | scalar |

### Measured headroom, per path

One LlamaMoe-shaped projection: `m=1, n=2048, k=7168`. The ceiling is the
vectorized BF16 kernel on the same host, same shape.

| path | time | weights | bandwidth | % of ceiling |
|---|---|---|---|---|
| BF16 (vectorized) — **ceiling** | 3.70 ms | 29.4 MB | 7.9 GB/s | 100% |
| Q4_K×Q8_0 **scalar** (was) | 20.37 ms | 8.3 MB | 0.41 GB/s | 5% |
| Q4_K×Q8_0 **vectorized** (now) | 8.38 ms | 8.3 MB | 0.99 GB/s | **14%** |
| Q8_K scalar | 22.24 ms | 16.7 MB | 0.75 GB/s | **10%** |
| Q6_K scalar | 38.69 ms | 12.0 MB | 0.31 GB/s | **4%** |
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

### P0.1 — K-quant vectorization · `gemm_224_{q2_k,q3_k,q5_k,q6_k,q8_k}.zig`

**Why:** measured 4% (Q6_K) and 10% (Q8_K) of ceiling. Q6_K is the second most
common GGUF format after Q4_K, so this is the largest single user-visible win
available. Five files, one structural pattern — one dev can take all five or
split them.

**What to do.** Each is `gemmQ*KScalar`: a per-element scalar loop doing
`acc += dequant(q[i]) * a[i]`. Replace the inner 32/256-element chunk with
vector ops. The ggml formulation per sub-block is
`Σ q·a` where `a` arrives as bf16 (widen with `vpmovzxwd` + `vpslld`) and `q` is
the unpacked quant:

- **Q6_K**: 6-bit values split as 4 low bits in `ql[128]` + 2 high bits in
  `qh[64]`. Reconstruct with shift/mask, or better: `vpshufb` nibble LUT to
  turn the packed bytes into f32 directly. Scales are i8 × f16 `d`.
- **Q8_K**: simplest — `qs[256]` are i8 and `d` is f32, so this is just a
  sign-extend + `vpmaddwd`-style i16 dot. **Start here**; it is the smallest
  and validates the pattern for the others.
- **Q5_K**: 5-bit (4 low in `q[256]`, 1 high bit in `qh[32]`).
- **Q2_K / Q3_K**: 2-bit and 3-bit, both with 16-way scale/min packing in
  `scales[16]` — the scale unpacking is the fiddly part, and it is *identical*
  to `getScaleMinK4` which already exists in `gemm_224_q4_k.zig`.

**Reuse, do not reinvent:**
- `gemm_224_q4_k.zig::gemmQ4KxQ8_0` — the finished AVX2 version of this exact
  shape, including the `*align(1)` unaligned-load idiom these blocks all need
  (they are 2-aligned structs packed at non-multiple-of-16 strides).
- `getScaleMinK4` / `makeQkx2Quants` in `gemm_224_q4_k.zig`.
- `dequantizeRowQ4_K` is the ground truth these must match.

**Acceptance.** Existing per-format tests (`block layout byte-exact`,
`quantize/dequant round trip`, `all-zero`, `scalar GEMM`) keep passing
unmodified, plus a **new** `GEMM == dequantize-then-dot` test per format at
~2e-5 relative tolerance over blocks with deliberately varied scale
structure. That test shape is not optional: the Q4_K nibble-interleave bug
survived two tests that filled `qs` with `0x11`, because equal nibbles hide a
swapped interleave.

**Target:** each format reaches ≥60% of the BF16 ceiling bandwidth.

---

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

`gemmQ4_KScalar` and every other format's dequant path are still scalar, and
they are what runs for **m > 1** and for every format other than Q4_K at m=1
(`runMatmulScratch` only has a fused fast path for Q4_K + m=1). So P0.1-P0.4
close the m=1 decode case; this closes prefill and the general case.

The `dequantizeRow*` functions themselves are scalar too and run per weight
row — vectorizing dequant alone buys a lot before the GEMM is touched. Cheap
first win: `dequantizeRowQ4_K` writes 256 floats per block with a scalar
`d1*(q&0xF) - m1`; that is 4 `vpmovzxwd` + shifts + 4 FMAs per block.

**Extend the fused Q8_0 fast path beyond Q4_K.** `runMatmulScratch` special-
cases one format; making it table-driven over `(weight_type, has_fused_path)`
would let Q6_K/Q8_K skip the BF16 round-trip at m=1 too.

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
- MLA o_proj went from 0.16x of the DRAM limit to bandwidth-bound.