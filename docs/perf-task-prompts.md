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

> You are working in `/ai/repos/2026/ktransformers-zig`, a Zig 0.16 port of the
> ktransformers CPU inference kernels. A C++ reference implementation lives at
> `/ai/repos/2026/ktransformers`; compare against
> `ktransformers/kt-kernel/operators/avx2/avx2_bf16_utils.hpp` (`act_fn`).
>
> **BEFORE YOU EDIT ANYTHING**
> - Read `AGENTS.md`, `docs/perf-roadmap.md` and `LESSONS_ZIG.md`.
> - CLAIM this task in `TODO.md`.
> - This task requires editing `src/kernels/arch/amx.zig`, which is normally off
>   limits. Your CLAIM must say so explicitly, and it must be the only task in
>   flight that touches that file.
>
> **TASK**
> `amx.expVec` in `src/kernels/arch/amx.zig` is the reason the SwiGLU activation
> sits at 0.59x of the C++ reference. Instruction counts for the 8-element body
> are **Zig 48, C++ 40**, and the gap is register allocation, not algorithm:
> `expVec` needs ~13 simultaneously-live constants (7 polynomial coefficients +
> `INV_LN2`, `LN2_HI`, `LN2_LO`, `0.5`, `1.0`, two clamp bounds) against 16
> ymm registers, so LLVM emits **6 `vbroadcastss` inside the loop** to
> rematerialize constants where the C++ hoists them all before it.
>
> **Fix by cutting peak live constants, in this order:**
> 1. Split `expVec` into two phases — argument split / floor / truncate /
>    exponent synthesis, *then* the polynomial — so the first group's constants
>    are dead before the second's are born.
> 2. Hoist the clamp bounds out of the per-vector path (clamp once per call).
> 3. Algebraically remove `0.5` / `1.0` from the live set.
>
> **Two warnings.**
> - Do **not** reduce the polynomial to degree 5 to save a register. Zig emits 8
>   FMAs to C++'s 6 **on purpose** (degree-6 truncation is 1.0e-8 relative vs
>   ~1e-6), and reverting that is an accuracy regression, not a fix.
> - Do **not** retry the array-staging fix. Replacing the `[VEC_LEN]f32` staging
>   in `applySwiGLU` with direct `*align(1)` vector loads plus a vectorized RNE
>   narrowing store measured **1.02x median over six back-to-back runs** and made
>   the binary 29 instructions larger. It was tried and reverted.
>
> **ACCEPTANCE**
> - The existing exp tests still pass (`tests/kernels/simd_kernels_tests.zig`
>   pins <= 1 ULP and 8-lane `expVec == expScalar`).
> - SwiGLU moves from 0.59x toward parity on `bench/cpp_vs_zig/run.sh`.
> - The activation output stays **bit-identical** to the reference — it
>   currently is, 0 of 262144 elements differ. Do not regress that.
> - Report instruction counts before/after alongside the time.
>
> **GATES:** `zig build -Doptimize=ReleaseFast`, `zig build test`,
> `python3 tools/verify_abi.py`.