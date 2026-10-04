# Performance task prompts

Copy-paste one of these into a fresh agent session. Each is self-contained: an
agent with no prior context on this repo should be able to execute it.

Background, measurements, per-item technique and acceptance rationale:
`docs/perf-roadmap.md`. These are the same items, pre-packaged for handoff.

**Status: all UNCLAIMED.** Each dev must CLAIM in `TODO.md` before editing
(protocol at the top of that file).

---

## P0.1 — Vectorize the K-quant GEMMs (start with Q8_K)

> You are working in `/ai/repos/2026/ktransformers-zig`, a Zig 0.16 port of the
> ktransformers CPU inference kernels. A C++ reference implementation lives at
> `/ai/repos/2026/ktransformers` (kernels under
> `ktransformers/kt-kernel/operators/`).
>
> **BEFORE YOU EDIT ANYTHING**
> - Read `AGENTS.md`, `docs/perf-roadmap.md` and `LESSONS_ZIG.md`.
>   `LESSONS_ZIG.md` has the Zig 0.16 API gotchas that will otherwise cost you
>   an hour.
> - CLAIM this task in `TODO.md`.
> - Do **NOT** edit `build.zig`, `tests/kernels/test_kernels.zig`,
>   `src/root.zig`, or `src/kernels/arch/amx.zig`. Those four files are shared
>   across parallel work streams. Everything else you need is yours alone.
>
> **TASK**
> Give the K-quant GEMMs an AVX2 path. Right now `gemmQ8_KScalar`,
> `gemmQ6_KScalar`, `gemmQ5_KScalar`, `gemmQ3_KScalar` and `gemmQ2_KScalar` in
> `src/kernels/amx/gemm_224_*.zig` are pure scalar, and on a non-AMX host they
> are the only code that runs (the AMX tile path is gated on
> `AmxFeatures.available`, which is false here).
>
> Measured at m=1, n=2048, k=7168 — one LlamaMoe-shaped expert projection:
>
> | path | time | bandwidth | % of ceiling |
> |---|---|---|---|
> | Q8_K scalar | 22.24 ms | 0.75 GB/s | 10% |
> | Q6_K scalar | 38.69 ms | 0.31 GB/s | 4% |
> | ceiling (vectorized BF16, same host + shape) | 3.70 ms | 7.9 GB/s | 100% |
>
> These paths are dequant/arithmetic-bound, not bandwidth-bound, so block sizes
> and prefetching will not help you. Only SIMD will.
>
> **START WITH `gemm_224_q8_k.zig` ONLY.** `BlockQ8_K` is the easiest of the
> five: `d` is f32 and `qs` is 256 signed bytes, so the dot product is a
> sign-extend plus an integer dot (`vpmovsxbd` + `vpmaddwd`-style, or widen to
> f32 and use `vfmadd`). Get that landed, then pick up Q6_K in the same session
> if there is time. The other three share the pattern once you have one.
>
> **MODEL YOUR IMPLEMENTATION ON `gemmQ4KxQ8_0` in
> `src/kernels/amx/gemm_224_q4_k.zig`** — it is the finished AVX2 version of
> exactly this shape and it already documents the two things that will bite you:
>
> - **The unaligned-load idiom.** These blocks are 2-aligned `extern struct`s
>   packed at non-multiple-of-16 strides, so `BlockQ8_K.qs` is 16-aligned on
>   even blocks and 2-aligned on odd ones. A plain vector load is a segfault
>   waiting to happen. Use:
>   `@as(*align(1) const @Vector(16, i8), @ptrCast(ptr)).*`
> - **n-outer / m-inner loop order.**
>
> **RULES THAT ACTUALLY MATTER (each measured, not folklore)**
> - n-outer / m-inner so the weight matrix streams from DRAM once, not once per
>   token row. Worth 3.4-4.0x — more than everything else combined.
> - Register-tile M (MR=4) so each converted chunk feeds several accumulators.
>   This is why we currently beat the C++ reference on BF16.
> - Explicit `@mulAdd`. Zig defaults `fp-contract` to off, so plain `a*b+c` emits
>   `vmulps` + `vaddps` instead of one `vfmadd`.
> - Never `@exp` in a vector loop. There is no vector lowering; you get one libm
>   call per lane. Use `amx.expScalar` / `amx.expVec`.
> - Build with `-Doptimize=ReleaseFast` for anything you measure. Plain
>   `zig build` defaults to Debug and all your numbers will be meaningless.
>
> **ACCEPTANCE — this is the part people get wrong, so read it twice**
> Add a test asserting `vectorizedGEMM == dequantizeRowQ8_K + F32 dot` at 2e-5
> relative tolerance, over several blocks with deliberately varied scale
> structure — **and** make the fixture such that a swapped nibble interleave or
> a wrong scale index **cannot** pass. Two Q4_K tests passed for a week while the
> kernel had a 128-byte out-of-bounds read, because the fixture filled `qs` with
> `0x11`: equal nibbles hide the bug. Also assert the vectorized path agrees
> with the existing scalar path.
>
> The existing per-format tests (`block layout byte-exact`,
> `quantize/dequant round trip`, `all-zero`, `scalar GEMM`) must keep passing
> unmodified.
>
> **GATES**
> - `zig build -Doptimize=ReleaseFast` — must build clean
> - `zig build test` — must stay green
> - `python3 tools/verify_abi.py` — 112 symbols + arity
>
> **REPORT BACK**
> - before/after time and bandwidth for the same shape
> - the new test names
> - confirmation the gates are green
> - if you cannot beat the scalar path, say so and stop. Do not land a neutral
>   change; "it was 1.02x" is a useful result and the repo has a precedent of
>   recording one.

---

## P0.2 — Vectorize the FP8 GEMM (lowest risk)

> **Status 2026-10-03 — read before starting.** Stream B is claimed and in
> flight (`4f222ba`); the decoder half already landed in `1e8369a` (the 256-entry
> LUT, which fixed subnormal decoding and the 0x7F NaN encoding). The prompt
> below predates all of that and is kept for the GEMM half.
>
> **The encoder is still wrong and has no test.** `f32_to_fp8e4m3` in
> `gemm_224_fp8.zig` is reachable from three call sites in `main.zig` (2020,
> 2058, 2540), so it is live in the C ABI. Measured by encoding and decoding
> again:
>
> | input | `gemm_224_fp8` | `gemm_224_mxfp8` (correct) |
> |---|---|---|
> | 500.0 | 0x7F → **0.0** | 0x7E → 448 |
> | 3.875 | 0x40 → **2.0** | 0x48 → 4.0 |
> | 0.0078125 | 0x00 → **0.0** | 0x04 → 0.0078125 |
>
> Three defects: saturation emits 0x7F/0xFF, which is the **NaN** encoding rather
> than ±448; subnormals are flushed to zero, destroying the whole 2^-9…2^-6 band;
> and the mantissa carry is masked away (`m3 + 1` then `& 0x7` without
> incrementing the exponent), which is the 2x error. **The correct
> implementation is already in this tree**, in `gemm_224_mxfp8.zig`, with the
> carry case commented — copy it rather than writing a fourth one. It survived
> because the only codec round-trip test (`simd_kernels_tests.zig`) exercises the
> MXFP8 copy, so `gemm_fp8.f32_to_fp8e4m3` has no coverage at all.
>
> Two more corrections to the plan below: the reference's `fp8x8_to_fp32x8`
> uses `_mm256_i32gather_ps` over a 256-entry table — a single `vpshufb` cannot
> serve 256 entries, so do not wait for the arch-layer shuffle; and this port
> keeps **one f32 scale per row of B**, not the reference's 128x128 block-wise
> scales, so the scale folds in once per row rather than per block.

> You are working in `/ai/repos/2026/ktransformers-zig`, a Zig 0.16 port of the
> ktransformers CPU inference kernels. A C++ reference implementation lives at
> `/ai/repos/2026/ktransformers` (kernels under
> `ktransformers/kt-kernel/operators/`).
>
> **BEFORE YOU EDIT ANYTHING**
> - Read `AGENTS.md`, `docs/perf-roadmap.md` and `LESSONS_ZIG.md`.
> - CLAIM this task in `TODO.md`.
> - Do **NOT** edit `build.zig`, `tests/kernels/test_kernels.zig`,
>   `src/root.zig`, or `src/kernels/arch/amx.zig`.
>
> **TASK**
> `gemmFullTileScalar` in `src/kernels/amx/gemm_224_fp8.zig` is scalar; give it an
> AVX2 path. On a non-AMX host it is the only FP8 code that runs, and FP8 backs
> `kt_dsv3_*` and the FP8 layerwise-transport path.
>
> **This is a direct port — use the reference.**
> `ktransformers/kt-kernel/operators/avx2/fp8-moe.hpp` (12 `_mm256` uses) and
> `fp8_dequant.hpp` (4) implement exactly this for AVX2. Diff against them; you
> have no design risk here, only translation.
>
> E4M3 decode is shift + mask + exponent-bias fixup, 8 lanes at a time, with
> the scale applied by one `vmulps`. E5M2 adds the sign bit. If the reference
> applies scales per-16 or per-32 group, unroll to that group size and
> accumulate per-group before scaling — that is what the group layout is for.
>
> Model the loop structure on `gemmQ4KxQ8_0` in
> `src/kernels/amx/gemm_224_q4_k.zig` for the `*align(1)` unaligned-load idiom
> and the n-outer/m-inner ordering. Use explicit `@mulAdd` (Zig's `fp-contract`
> is off).
>
> **ACCEPTANCE**
> - Existing FP8 tests keep passing.
> - A new test pins the vectorized path against `gemmFullTileScalar` to f32
>   rounding on random blocks.
> - The dequant round-trip stays byte-exact.
>
> **GATES:** `zig build -Doptimize=ReleaseFast`, `zig build test`,
> `python3 tools/verify_abi.py`.
>
> **REPORT BACK:** before/after time and bandwidth against the vectorized BF16
> kernel (the ceiling) at the same shape, new test names, gate status.

---

## P0.3 — Vectorize MXFP4 / MXFP8

> You are working in `/ai/repos/2026/ktransformers-zig`, a Zig 0.16 port of the
> ktransformers CPU inference kernels. A C++ reference implementation lives at
> `/ai/repos/2026/ktransformers`.
>
> **BEFORE YOU EDIT ANYTHING**
> - Read `AGENTS.md`, `docs/perf-roadmap.md` and `LESSONS_ZIG.md`.
> - CLAIM this task in `TODO.md`.
> - Do **NOT** edit `build.zig`, `tests/kernels/test_kernels.zig`,
>   `src/root.zig`, or `src/kernels/arch/amx.zig`.
>
> **TASK**
> `src/kernels/amx/gemm_224_mxfp4.zig` and `gemm_224_mxfp8.zig` are scalar on
> non-AMX hosts. The C++ reference vectorizes these more heavily than any other
> family — `kt-kernel/operators/avx2/mxfp4-moe.hpp` has **42** `_mm256` intrinsic
> uses (`mxfp8-moe.hpp` has 18). Start by reading them: whatever they are doing
> to justify 42 intrinsics is the win.
>
> MXFP4 is 4-bit weights with a **shared exponent per 32 elements**; MXFP8 is
> 8-bit with a shared scale per 32. So the inner loop is 32 weights -> 1 scale.
> Accumulate 32-element groups separately and apply the group scale before
> adding into the total. For the 4-bit unpack prefer a `vpshufb` lookup table
> over the packed bytes (one LUT maps all 16 nibble values to f32) rather than
> shifts and masks.
>
> Use `gemmQ4KxQ8_0` in `src/kernels/amx/gemm_224_q4_k.zig` for the `*align(1)`
> unaligned-load idiom and the n-outer/m-inner loop order. Explicit `@mulAdd`.
>
> **ACCEPTANCE**
> - The two existing exact-value MXFP tests keep passing.
> - Add a GEMM-vs-`dequantizeRow` test with varied per-32 group scales at ~2e-5
>   relative tolerance, using fixture data that cannot pass with a swapped
>   unpack order.
>
> **GATES:** `zig build -Doptimize=ReleaseFast`, `zig build test`,
> `python3 tools/verify_abi.py`.
>
> **REPORT BACK:** before/after time and bandwidth against the vectorized BF16
> ceiling, new test names, gate status.

---

## P0.4 — INT4 families (GPTQ-INT4 + raw-INT4)

> You are working in `/ai/repos/2026/ktransformers-zig`, a Zig 0.16 port of the
> ktransformers CPU inference kernels. A C++ reference implementation lives at
> `/ai/repos/2026/ktransformers`.
>
> **BEFORE YOU EDIT ANYTHING**
> - Read `AGENTS.md`, `docs/perf-roadmap.md` and `LESSONS_ZIG.md`.
> - CLAIM this task in `TODO.md`.
> - Do **NOT** edit `build.zig`, `tests/kernels/test_kernels.zig`,
>   `src/root.zig`, or `src/kernels/arch/amx.zig`.
>
> **TASK**
> The reference has three AVX2 INT4 kernels — the most INT4 coverage of any
> family: `kt-kernel/operators/avx2/rawint4-moe.hpp` (18 `_mm256` uses),
> `gptq_int4-moe.hpp`, `gptq_int4_dequant.hpp` (8). The Zig port has only
> `src/kernels/amx/gemm_224_int4.zig` (GPTQ-style: 4-bit values with a scale and
> a zero-point per 32-element group) and it is scalar.
>
> GPTQ-INT4 is the friendliest starting point: `vpshufb` unpack, then an `vfmadd`
> chain with a per-group scale and subtract. Raw-INT4 is simpler still (nibble
> unpack, one global scale) but **there is no raw-INT4 kernel in Zig at all** —
> decide whether to add one, or record deliberately that INT4 covers GPTQ only.
> The `_avxvnni` reference variants need VNNI, which this host lacks; keep the
> scalar path as their fallback and say so in a comment.
>
> Use `gemmQ4KxQ8_0` in `src/kernels/amx/gemm_224_q4_k.zig` for the `*align(1)`
> unaligned-load idiom and the n-outer/m-inner loop order. Explicit `@mulAdd`.
>
> **ACCEPTANCE**
> - Existing INT4 tests (`16x16x32 identity`, the partial-group case) keep
>   passing.
> - Add a GEMM-vs-`dequantizeRow` test at ~2e-5 relative tolerance, with fixture
>   data that cannot pass with a swapped nibble interleave.
>
> **GATES:** `zig build -Doptimize=ReleaseFast`, `zig build test`,
> `python3 tools/verify_abi.py`.
>
> **REPORT BACK:** before/after bandwidth against the vectorized BF16 ceiling,
> new test names, gate status.

---

## P1.3 — SwiGLU activation (diagnosed; needs register-pressure work)

> Re-measured 2026-10-03: **0.68x**, not the 0.59x recorded below, and the line
> numbers are current as of commit `2416b68`. Supersedes the earlier copy.

You are working in `/ai/repos/2026/ktransformers-zig`, a Zig 0.16 port of the
ktransformers CPU inference kernels. A C++ reference implementation lives at
`/ai/repos/2026/ktransformers`; compare against
`ktransformers/kt-kernel/operators/avx2/avx2_bf16_utils.hpp` (`act_fn` at :118).

**BEFORE YOU EDIT ANYTHING**
- Read `AGENTS.md`, `docs/perf-roadmap.md` (item P1.3) and `LESSONS_ZIG.md`. The
  Zig 0.16 API gotchas in LESSONS_ZIG.md are real and will cost you an hour.
- CLAIM this task in `TODO.md` **and** put a marker at the top of each file you
  intend to edit — `src/kernels/arch/amx.zig` and possibly
  `src/kernels/amx/gemm_224_bf16.zig`:
  `// CLAIM: <handle> — SwiGLU activation — <date>`. Claims go in both places:
  the moment one matters is when someone opens the file they are about to change.
- `src/kernels/arch/amx.zig` is normally off limits and you must say in your
  claim that you are touching it. It was held exclusively by the `vpshufb` work
  (Stream J), which has now landed and released it — confirm it is still clean
  (`git status src/kernels/arch/amx.zig`) before you start, and be the only
  session in that file.
- Do not edit `build.zig`, `tests/kernels/test_kernels.zig`, `src/root.zig`.
  Append-only in `tests/kernels/simd_kernels_tests.zig` and your own TODO.md
  section. Stage by filename when you commit — never `git add -A`; there is
  uncommitted work from another session in this tree (Stream B, FP8).
- Claim format is `CLAIMED (<handle>) since <date>`. No tool or model names in
  claims, commits or commit trailers. Never push. Commit after each verified step.

**WHAT IS IN THAT FILE NOW.** `2416b68` added `pshufb16` / `pshufbTable` and a
shared `ByteVec = @Vector(32, u8)` at the top of `amx.zig`. It is additive and
does not touch `expVec`, but it shifted every line below it, so use the anchors
below rather than any older note. It also moved the file's constants **into the
same arch layer** you are about to edit — if you add a helper there, expect it
to be shared with everything.

**THE TASK**
`amx.expVec` (`src/kernels/arch/amx.zig:675`) is why the activation sits at
**0.68x** of the C++ reference. The gap is register allocation, not algorithm:
`expVec` needs about thirteen simultaneously-live constants against sixteen ymm
registers — seven polynomial coefficients (`EXP_C0`..`EXP_C6`, lines 639-645),
`EXP_INV_LN2`, `EXP_LN2_HI`, `EXP_LN2_LO`, 0.5, 1.0 and the two clamp bounds
(`EXP_LO`/`EXP_HI`, lines 634-635) — so LLVM rematerializes constants with
`vbroadcastss` inside the loop where the C++ hoists all of them before it. A
session that measured this recorded **48 instructions in the Zig 8-element body
against 40 in C++**, and six `vbroadcastss` where C++ has none. Treat those as a
starting point and re-derive them (see MEASUREMENT — LLVM unrolls this loop
heavily, so an unnormalized count is meaningless).

Fix by cutting peak live constants, in this order, stopping as soon as you are
at parity:
1. Split `expVec` into two phases — clamp, argument split, floor, exponent
   synthesis first, the polynomial second — so the first group's constants are
   dead before the second's are born. Most likely to work; does not change the
   arithmetic.
2. Hoist the clamp bounds out of the per-vector path. They are loop-invariant
   per call; clamp once per call.
3. Fold 0.5 and 1.0 out of the live set algebraically.

Last resort only, and only with a measured justification in your report: drop
the polynomial from degree 6 to degree 5.

**A WARNING WORTH READING TWICE, because the same shape of bug was just fixed in
another kernel.** The IQ4_XS stream expected its win from `vpshufb` and got
none; the actual bottleneck was `f16_to_f32(blk.d)` sitting *inside* a
per-sub-block helper, so the conversion ran eight times per block at the head of
a serial dependency chain with only four FMAs per sub-block to hide it at m=1.
Hoisting it was the whole 3x → 17-19x. **So before you start optimizing
constants, establish what the dependency chain actually is** at m=1: dump the
loop and look at what feeds the first FMA. The lesson generalizes — a
conversion that looks like it belongs to the inner loop may belong to the block.

**TWO WARNINGS, BOTH FROM WORK ALREADY DONE HERE**
- Do **not** drop to degree 5 to save a register. Zig emits 8 FMAs where C++ emits
  6 on purpose: degree-6 truncation is 1.0e-8 relative against ~1e-6, and the
  port is currently *more* accurate than the kernel it replaced. Reverting that
  is an accuracy regression dressed as a speedup.
- Do **not** retry the array-staging fix. Replacing the `[VEC_LEN]f32` staging in
  `applySwiGLU` (`gemm_224_bf16.zig:240`) with direct `*align(1)` vector loads
  plus a vectorized RNE narrowing store measured **1.02x median over six
  back-to-back runs** (range 0.80-1.28x) and made the binary 29 instructions
  larger. Tried, reverted, do not repeat.
- Do **not** change `pub const VecF32 = @Vector(8, f32)` (now line 750) or
  `VEC_LEN`. They are shared with the BF16 GEMM, MLA and every quantized kernel;
  widening the vector is a different task with a much larger blast radius.

**TESTS — `tests/kernels/simd_kernels_tests.zig`, never next to the code.** A
`test` block inside a file that belongs to the `kt` module is never executed by
`zig build test`; Zig collects tests from the root file's module only. Assert,
for all three variants (`swigluVec` :764, `swigluClampVec` :778, `swigluOaiVec`
:785):
- `expVec` still within 1 ULP of `@exp`, and the 90% bit-identical rate is not
  regressed;
- `expVec == expScalar` lane for lane, including the clamp at both ends;
- the activation output stays **bit-identical** to the reference. It currently
  is: 0 of 262144 elements differ. A change that shows up only as a 1e-7 relative
  wobble is a bug here.

**MEASUREMENT.** The host is shared and absolute milliseconds move by up to 3x
with the load — today another agent's inference job put it at load 31 on 16
cores, and the same case measured 10x, 14x and 19x on three consecutive runs.
Never compare against a remembered number.

    KT_PIN=9 ./bench/cpp_vs_zig/run.sh

The row you care about is `SwiGLU activation swiglu act [m=128,n=2048]`. Take
the best of at least three runs and state the load average with the number.
Baseline as of today, pinned, best of 3: C++ 0.2046 ms, Zig 0.3009 ms, 0.68x —
re-measure it yourself, do not quote that.

For instruction counts:

    zig build-exe -OReleaseFast -mcpu=native -lc --dep kt \
        -Mroot=bench/cpp_vs_zig/zig_bench.zig -Mkt=src/root.zig \
        -femit-bin=/tmp/zb
    nm /tmp/zb | grep applySwiGLU
    objdump -d --start-address=0x<SYMBOL> /tmp/zb

Count inside the loop body only, and normalize: the loop is heavily unrolled, so
report both the raw count and the count per 8 elements processed. The number
that must move is `vbroadcastss` inside the body.

**ACCEPTANCE**
- The existing exp and SwiGLU tests pass, and the activation remains
  bit-identical to the reference.
- The activation moves from 0.68x toward parity. Parity is the goal; anything
  above 0.85x is a strong result.
- `vbroadcastss` per 8 elements in the loop body strictly decreases, reported
  next to the times.
- If the register work does not pay off, say so with numbers and leave the code
  correct. A measured negative result with a clear write-up is a useful outcome
  here; a lucky micro-optimization with a regression hidden behind noise is not.

**GATES, all of them:**
    zig build -Doptimize=ReleaseFast
    zig build test
    python3 tools/verify_abi.py
    python3 tools/audit_layout.py
    zig build all-variants
    zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu

**REPORT BACK**
- Before/after table: C++ ms, Zig ms, ratio, load average stated.
- Instruction counts: raw and per-8-element, for the loop body.
- What the dependency chain at m=1 actually was, and which of the three fixes
  you applied — and why you stopped there.
- Confirmation that the output is still bit-identical, with the element count.
- Anything this prompt got wrong.

**FOOTGUNS, all verified in this repo**
- `@exp` on a vector does not vectorize: compiler-rt ships only scalar `expf` and
  Zig does not target libmvec, so it becomes eight scalar libcalls with spills,
  17x slower. Never reintroduce it in a vector loop.
- Zig defaults `fp-contract` off, so `a*b+c` emits `vmulps`+`vaddps` instead of one
  `vfmadd`. Use `@mulAdd` where you touch arithmetic.
- `@splat` is one-argument in Zig 0.16 and the value must be typed:
  `@as(VecF32, @splat(@as(f32, 0.5)))`.
- A vector shift needs a vector of the shift-logic type, typed even for a
  comptime count: `v >> @as(@Vector(8, u5), @splat(@as(u5, 23)))`.
- Vector element access needs a comptime index; stage through an array.
- Run `sentrux_rescan` then `sentrux_check_rules` after touching `amx.zig`.
