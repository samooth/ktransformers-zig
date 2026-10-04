# ktransformers-zig TODO

## Status: Beta (15/15 GGML formats + 268 tests; SIMD at/above C++ parity)

Last updated: 2026-10-03

### Coordination protocol (multi-agent)
**CLAIM before you code.** Before starting a new task, add a claim line
in the relevant section: `- [ ] <task> — CLAIMED (<handle>) since <date>`.
This prevents parallel collisions (we hit one with LlamaMoe: two agents
independently ported the same LLAMA_MOE_TP to Zig in the same session).
Claims are removed when the task is done; abandoned claims can be reclaimed.
`<handle>` identifies the working session to the other agents (a branch, a
session name, an initials tag). Never put a tool, product or model name in a
claim, a commit message or a commit trailer — the repository records the work
and the reasoning, not who or what typed it. Commit messages carry no
attribution trailers of any kind.

**Build: WORKING** - `zig build` produces `zig-out/lib/libkt_kernel_ext.so`; 6 x86 variants + aarch64 neon cross-build.
**Tests: WORKING** - `zig build test` runs all suites (kernels incl. 15 GGML formats + dispatch/ABI audits, MLA + MLA C-API, Qwen3 MoE + layer C-API + GGUF e2e, FP8 transport, GGML C API, to/from_float C-API, aarch64, NEON kernel, allocator C-API, LlamaMoe C-API, K_BLOCK runtime, math-helpers C-API (swiglu 3 variants/rms_norm/rope/softmax), SIMD primitives + GEMM micro-kernels + the six K-quant AVX2 GEMMs) = **268 total pass across 18 suites, 0 leaks, exit 0**.
**Multi-variant: WORKING** - `zig build all-variants` produces **6** `.so` (`libkt_kernel_ext_{variant}.so` for avx2, avx512_base, avx512_vnni, avx512_vbmi, avx512_bf16, amx), each exporting **112 C API symbols** (triple-gated: exports + arity via `tools/verify_abi.py`; layout via `tools/audit_layout.py`). Runs in CI (`wheels.yml` `abi-gate` job). **There is no neon leg**: `all-variants` is x86-only, and the aarch64 build is a separate one-shot `zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu`. Verified by moving the stale `libkt_kernel_ext_neon.so` out of `zig-out/lib`, running `all-variants`, and getting zero neon outputs. The old "+ neon (8 installed names)" here was wrong, and it stayed wrong through a cycle in which aarch64 was actually broken - the same blind spot as the missing aarch64 CI leg.
**Bench: WORKING** - `zig build -Doptimize=ReleaseFast bench` (B2 + moe_bench extensions): gemmExpert vectorized vs scalar ref at DeepSeek-V3 shapes — **2.8x (down, memory-bound) to 5.3x (prefill) measured**, maxdiff ≤ 1.7e-6; MoE forward end-to-end with default-vs-tuned tile params.
**Runtime: WORKING** - work-stealing worker pool + **sched_setaffinity pinning (A3)** + **waitIdle/kt_cpuinfer_sync drain (B3)**; NUMA topology via /sys and /proc/cpuinfo + **mbind in NumaAllocator (A3)**; **L1/L2/L3 sysfs detection + selectTileParams (A4)**.
**SFT/LoRA: FORWARD+BACKWARD COMPLETE** - training path done; C API exports (forward_sft/backward/update_lora_weights); smoke test passes.
**Allocator: INJECTABLE (B1)** - `kt_set_default_allocator` C-ABI vtable; all context types capture-and-free symmetrically.
**Model Orchestration: COMPLETE** - DeepseekV3DecoderLayer + DeepseekV3Model + DeepseekV3ForCausalLM (C-API `kt_dsv3_*`), Qwen3MoeDecoderLayer + Qwen3MoeModel + Qwen3MoeForCausalLM (C-API `kt_qwen3moe_*`), AND LlamaMoe for GGUF checkpoints (C-API `kt_llama_moe_*`, all 16 GGML weight formats per projection via the comptime-audited `gemmQuant` dispatch).
**GGML: 15/15 FORMATS COMPLETE** - Q8_0, Q4_K, Q5_K, Q6_K, Q8_K, Q2_K, Q3_K, IQ4_XS, IQ4_NL, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M — all with quantize + dequantize + GEMM + tests. Kmap init histogram-based (22x) + process-cached.
**MHA engine + GGUF parser (Qwen3 dev)**: vanilla MHA `MhaEngine` + GGUF v3 parser in `src/io/gguf.zig` — committed in `a61271e`.

### Open items (verified against the tree on 2026-10-03)

Consolidated from `REMAINING.md` and the lane-c external audit
(`TO_IMPROVE.md`). Everything those two files listed as a task has been
re-checked against the current source; the resolved ones are listed so the
next reader does not re-investigate them. `REMAINING.md` has been deleted in
favour of this section — it had gone stale (four of its five tasks were
already completed and it did not mention the SIMD work at all).

### Parallel work streams (2026-10-03) — CLAIM before starting

**Full roadmap with per-item technique, acceptance criteria and payoff targets:
`docs/perf-roadmap.md`.** That is the document to read before starting any
kernel work; the summary below is the index.

The AVX2 SIMD audit found that after the GEMM/activation rework, **BF16 is the
only kernel with a real AVX2 vector path**. Everything else has an AMX tile
path (dead on any non-AMX host, `AmxFeatures.available == false`) plus a
scalar fallback. Measured cost at a LlamaMoe-shaped projection
(`m=1, n=2048, k=7168`):

| | time | weight bytes | bandwidth |
|---|---|---|---|
| Q4_K x Q8_0 (scalar) | 20.37 ms | 8.26 MB | **0.41 GB/s** |
| BF16 (vectorized), same n/k | 3.70 ms | 29.36 MB | **7.94 GB/s** |

The scalar quantized path is **5.5x slower while moving 3.6x fewer bytes** — so
it is dequant/arithmetic-bound, not bandwidth-bound, and uses ~2% of the
19.3 GB/s a vectorized kernel sustains on this host. This is missing SIMD, not
instruction selection.

These streams are deliberately scoped to **disjoint files** so they can run in
parallel. Two rules so they do not collide:

1. **Do not edit `build.zig`, `test_kernels.zig`, `src/root.zig`, or
   `src/kernels/arch/amx.zig`.** Test-file registration is pre-wired below;
   kernel files are one-per-stream. If you genuinely need an arch-layer
   change, say so in the claim rather than editing it.
2. **CLAIM here first** (the protocol at the top of this file), then work.
   Put the claim in **two** places: the section below, and a one-line comment at
   the top of each file you intend to edit:

   ```zig
   // CLAIM: <handle> — <scope> — <date> — holds <shared file> until committed
   ```

   `grep -rn "^// CLAIM:" src/` answers "is this file free?" without anyone having
   to remember to open this file — and that is the moment it matters, because a
   claim only protects the sessions that know about it. A claim recorded here
   but not in the file is a claim nobody sees; a claim in the file but not here
   is a claim nobody counts.
3. **Before you commit, check what you staged.** Another session's uncommitted
   work can be sitting in the same tree, and `git add -A` will absorb it. Stage
   the files you touched by name. If you did not write a hunk, it does not go in
   your commit, however reasonable it looks next to your change.

**Stream A follow-up — Q4_K m=1 routing. DONE (3.3x, commit `ab09aa5`).**

The K-quant AVX2 driver (commit `f6b1f0b`) superseded the Stream A fused
`gemmQ4KxQ8_0` kernel, but `runMatmulScratch` still routes Q4_K at `m == 1` to
the slower one. Measured on three cores, same weights, m=1 n=2048 k=7168:

| path | time | bandwidth |
|---|---|---|
| fused Q8_0 (skips the dequant) | 5727 / 5775 / 5858 us | 1 GB/s |
| dequant (4 us) + generic bf16 driver | 1778 / 1787 / 1758 us | **4 GB/s** |

The dequant that the fused path avoids costs 4 us — 0.2% of the total — so the
generic driver is **3.3x faster while doing strictly more work**.

**Numerics consequence (deliberate, not a free win).** The fused arm skipped
the BF16 activation intermediate, so it was ~0.2% per activation MORE accurate
than the generic path. It was also the only format in `LlamaMoe` with that
advantage — Q8_0, the other five K-quants and the nine IQ formats all
dequantize activations to BF16, so Q4_K was the outlier and is now consistent
with its siblings. `dequantQ8_0ToBF16` computes `f32_to_bf16(a_q * a_d)`:
`a_q` (8-bit) is exact in bf16 but the f16 scale `d` has an 11-bit significand
against bf16's 8, so ~3 low bits are lost. There is no C++ reference for this
path (ktransformers has no K-quant MoE kernel at all — it is Zig-native), so
consistency with the rest of the library is the criterion, not reference parity.

**A test asserted the wrong thing and hid this.** `llamafile_moe_capi_tests`
claimed the BF16 round-trip was "exact for Q8_0" and compared the BF16 path
against an F32 reference with a 1e-2 bound; it only ever passed because the
fused arm was what ran. Fixed to model both BF16 activation round-trips, which
restores the ~1e-5 agreement the test always meant to assert:

| comparison | rel error |
|---|---|
| fused vs F32 reference | 6.2e-6 (what it asserted) |
| bf16 driver vs F32 reference | 7.8e-2 (fails the bound) |
| bf16 driver vs F32 ref with bf16 rounding modelled | 9.5e-6 |

The driver was never wrong. The fused kernels stay (correct, tested, and still
the readable statement of the Q8_0 formulation) but are off the inference path;
deleting them plus their 4 tests is a separate call.

**Stream A — Q4_K x Q8_0 fused dot. DONE (2.4x, superseded).**
`gemmQ4KxQ8_0` added alongside the scalar version (which stays as the
readable statement of the math). 20.37 ms -> 8.38 ms at m=1,n=2048,k=7168
(**2.4x**); 0.41 -> 0.99 GB/s. `gemmQ4_KScalar` was still scalar — **CLOSED
by Stream C** (`gemmQ4_K`, 10.5x over scalar; see the Stream C entry).

**Remaining headroom on the Q8_0-activation kernel (unclaimed):** at
0.99 GB/s it is still only ~5% of what a vectorized kernel sustains, i.e.
another 4-8x is available. The dominant remaining cost is that `dot_1` (the
activation sums) does not depend on `j` yet is recomputed for every one of the
2048 output columns. Hoisting it out of the n loop needs either a small
per-row scratch or an outer loop over (i, super-block, group) with `j`
innermost — which turns weight access into a constant-stride walk. That is
also the same fix as the NR-tiling item under "actually open" below. Secondary:
`getScaleMinK4` and `f16_to_f32` are still scalar with a branch, 8+4 calls per
super-block, and the four `@reduce(.Add)` horizontal reductions per 64 weights
dominate the rest.

**Stream B — FP8 E4M3/E5M2. CLAIMED (fp8-gemm-avx2) since 2026-10-04.**
`src/kernels/amx/gemm_224_fp8.zig`, `gemmFullTileScalar` (line ~355). The C++
reference has `operators/avx2/fp8-moe.hpp` + `fp8_dequant.hpp`, so this is a
direct port target with a reference to diff against. Acceptance: the existing
FP8 tests pass, plus `gemmFullTileScalar`'s new vectorized sibling matches it
to f32 rounding on random blocks.

**The decode fix is committed (1e8369a, 2026-10-04).** The hunk that was sitting
uncommitted is now its own commit — the 256-entry `FP8_E4M3_LUT`, bit-identical
to the reference's `FP8LUT::init`, adopted as written. It fixed two real decode
bugs: E4M3 subnormals (exponent field 0, mantissa 1-7) were decoded as normal,
yielding about 1.0-2.0 instead of 2^-9 .. 7*2^-9, and `0x7F`, the single NaN
encoding, was decoded to 480.0, which is above `E4M3_MAX` = 448 and therefore
not representable. Committed separately so it can no longer be lost to a
`git checkout`.

**E5M2 is NOT in this stream.** There is no E5M2 anywhere in the tree — no
codec, no type id in `src/main.zig` or `include/kt_kernel.h` — and none in the
reference (`fp8_dequant.hpp` is E4M3-only), so roadmap P0.2's "E5M2 adds the
sign bit" has nothing to add to yet. Adding a type id is a change to the C ABI
contract that Python and the pybind11 shim both depend on, so it needs its own
claim on `include/kt_kernel.h`, not a kernel detail.

**E4M3 GEMM: DONE (3.2x at m=1, 9.2x at m=4). `gemmAvx2` in
`gemm_224_fp8.zig`, wired through `gemmFullTile` so it is reachable from the
exported `kt_matmul_fp8` with no `root.zig` change. `b88e90b`.**

`gemmFullTileScalar` was an m/n/k triple loop, so the n x k weight matrix was
re-streamed from DRAM once per token row. The replacement is n-outer / m-inner
with MR=4, one decoded weight vector feeding four accumulators, explicit
`@mulAdd`, and one `@reduce` per OUTPUT element — the reference's four-accumulator
fold reaches the same place with the four chains on the M axis instead of the k
axis, for 4 registers rather than 16.

At n=2048 k=7168, pinned to cpu 9, ReleaseFast, best of 4 runs of 9 reps, both
paths in the same binary (`bench/fp8_bench.zig`). Load average 22-25 on 16
threads throughout, so absolute ms are ~2x worse than a quiet host and only the
ratios mean anything — note the Q8_0 control below reads 4.5 GB/s where
Stream F recorded 7.8:

| path | m=1 | m=4 |
|---|---|---|
| `gemmFullTileScalar` (before) | 13.36 ms / 1.08 GB/s | 54.80 ms / 0.27 GB/s |
| **`gemmAvx2` (after)** | **4.13 ms / 3.55 GB/s — 3.2x** | **5.97 ms / 2.46 GB/s — 9.2x** |
| same loop, asm `vgatherdps` | 8.05 ms / 1.82 GB/s — **0.51x** | 9.47 ms / 1.50 GB/s — 0.63x |
| same loop, table-free arithmetic decode | 4.14 ms / 3.54 GB/s — 1.00x | 6.36 ms / 2.31 GB/s — 0.94x |
| decode-free floor (loop cost alone) | 2.94 ms / 5.00 GB/s | 4.24 ms / 3.46 GB/s |
| Q8_0 `gemmAvx2`, table-free control | 3.45 ms / 4.52 GB/s | 4.81 ms / 3.24 GB/s |

Agreement between the two FP8 paths, L2-normalised max-abs: 2.0e-7 at m=1 and
1.3e-7 at m=4 — f32 summation order, not either kernel.

**The asm `vgatherdps` is a LOSS on this host (Zen 3), 1.6x-1.9x slower than
eight scalar table loads.** That closes roadmap P0.2's "an asm `vgatherdps` is
the obvious next experiment" and Stream E/F's copy of it: on this CPU the
gather costs ~5.1 ms per 14.7M weights against ~1.2 ms for the plain loads,
i.e. ~14 cycles per 8 lanes versus ~4. Do not spend the arch-layer slot on a
gather. Two traps worth writing down: the AT&T operand order for `vgatherdps`
is REVERSED relative to Intel (the destination is the LAST operand and the mask
is the first) — written the Intel way it assembles, runs, and returns all zeros,
because the mask register is then the uninitialised destination; and the asm
needs an x86 guard or `zig build -Dtarget=aarch64-linux-gnu` fails with
"couldn't allocate output register for constraint 'x'".

**But the LUT is not the bottleneck either — it is second.** At m=1 the decode
is 1.2 ms of the 4.13 ms total (29%), and the floor with NO decode at all is
2.94 ms, so even a free decode would only buy 1.4x. The other 71% is weight
streaming plus the loop, and the table-free Q8_0 control (3.45 ms) is SLOWER
than the FP8 decode-free floor (2.94 ms), which says the remaining cost is the
memory system on a loaded host, not the format. The arithmetic decode costs the
same as the table at m=1 (15 ALU ops against 16 load/insert ops — both land at
the ~4-uops/cycle issue limit) and is slightly WORSE at m=4, so there is nothing
to win there either. What is left is the loop order and the memory system.

That is the input for whoever owns a shared decode primitive: `vpshufb` cannot
serve 256 entries, but if any single-nibble (16-entry) decode helps, the MX
formats are where it pays — their tables ARE 16 entries, and `pshufb16` already
exists for them. For E4M3 the two-level decomposition is worth at most the 1.4x
above.

**Inner loop, disassembly (ReleaseFast, x86-64).** Scalar triple loop: **7
instructions per weight** (3 to widen the bf16 activation, 1 byte load, 1
`vmulss` against the table that also applies the scale, 1 `vmulss`, 1 `vaddss`)
with a serial `vaddss` dependency chain. `gemmAvx2`, MR=4 path: **32
instructions per 8 weights = 1.0 per MAC** (8 byte loads, 8 table loads, 4
activation `vpmovzxwd`, 4 `vpslld`, 4 `vfmadd`, 4 loop overhead). The m=1
row-tail path is 32 instructions per 8 MACs = 4 per MAC, against the scalar 7.
Both entry points are `noinline` so each keeps a symbol in the `.so`.

**Tests** (`tests/kernels/simd_kernels_tests.zig`, `6153821`): decode exact for
all 256 encodings by BIT PATTERN against an independent transcription of the
reference rule, including the negative zeros the reference also produces; the
GEMM against an f64 once-scaled oracle over 9 shapes (m = 1/4/5, n = 1/2/7,
k = 8/16/32 exact, past one 32-step, and with 1/5/7-element tails) with both
outputs NaN-poisoned; and the "scale applied once, after the accumulation"
choice pinned BIT-EXACTLY (0x3B866667 once-scaled, which is the f64 truth
rounded once, against 0x3B866666 per-element — one ULP apart, so no tolerance
would have separated them).

**The E4M3 ENCODER was still the four-bug version until 6a64d45.** `1e8369a`
fixed the decoder and `b88e90b` the GEMM; nothing had touched
`f32_to_fp8e4m3`, which is live in the C ABI (`kt_fp8_quantize`,
`kt_fp8_quantize_block`, `kt_quantize_fp8_e4m3`, `quantizeRowBF16ToFP8`).
Re-measured over all 256 encodings, decode -> encode: **14 of 256 round trips
failed** (16 counting 0x7F/0xFF, which are the NaN hole and cannot round trip
by definition), worst case **100% relative error**, twice over:

| input | old byte | old value | new byte | new value |
|---|---|---|---|---|
| 500.0 / -500.0 / 1e9 / inf | 0x7F / 0xFF | **0** | 0x7E / 0xFE | +-448 |
| 3.875 | 0x40 | **2.0** | 0x48 | 4.0 |
| 1.0625 (exact tie) | 0x39 | **1.125** | 0x38 | 1.0 |
| 2^-9 ... 7*2^-9 (7 subnormals) | 0x00,0x00,0x00,0x00,0x02,0x04,0x06 | **all wrong** | 0x01..0x07 | exact |

The saturation row is the one that matters: 0x7F is the NaN encoding and the
decoder maps it to 0, so any weight above 448 was silently DELETED, not
clipped. Now 0 of 256 fail. `gemm_224_fp8.zig` OWNS the codec (LUT, decode,
encode) and `gemm_224_mxfp8.zig` imports it, re-exporting the same names — the
de-duplication that TODO.md had deferred on the grounds that Stream B was in
flight. Two editable copies of a codec is how the MX one got fixed and this one
did not. Test `793efb8`, verified to have teeth: against the old encoder it
fails the 256-sweep with exactly 14 mismatches.

**NOTE for the arch layer: the pshufb guard has landed.** The "NOTE for the
arch layer" paragraph below diagnosed `amx.pshufb16` (2416b68) as breaking the
aarch64 cross-build with no target guard. Stream D (which holds
`src/kernels/arch/amx.zig`) has since added a `comptime builtin.cpu.arch ==
.x86_64` guard with a scalar fallback, and `zig build -Dvariant=neon
-Dtarget=aarch64-linux-gnu` now PASSES again (2026-10-04, re-verified here). No
handoff is needed. Worth recording for whoever wrote it: the fallback indexes
`table[(k & 16) + (idx[k] & 15)]`, i.e. it keeps the per-128-bit-lane offset,
which is the actual `vpshufb` semantics on a ymm — a fallback that indexed
`table[idx[k] & 15]` would be right only while callers happen to build their
table with `pshufbTable` (which duplicates it into both lanes). An independent
guard written here used the lane-less form and would have been the weaker of the
two, which is the argument for letting the claim holder land their own.

**Not touched, still open.** `FP8BufferB.fromMatBF16` and `batchedGemm` have no
callers in the tree (`kt_matmul_fp8` is the only consumer, and `gemmQuant`'s
`isWeightTypeSupported` list must NOT grow FP8: it is a per-row-scale format,
not a GGML block format, so LlamaMoe has no route to it). The AMX path in
`gemmFullTile` still has its own known gaps — `loadB` fills 16 of the 32 rows
the tile config asks for and `lda` is passed as elements where `tile_loadd`
wants bytes — but fixing it needs the C++ `BufferBFP8Impl` packing ported
first, and it cannot be tested on this host.

**NOTE for the arch layer (found while gating, NOT caused by this stream):**
`amx.pshufb16` (2416b68) has no target guard, so
`zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu` now fails with
"couldn't allocate output register for constraint 'x'". Verified by building
`4f222ba` (aarch64: PASS) against `2416b68` (aarch64: same error), and by
building this tree with only that one function guarded (aarch64: PASS, with
`gemmAvx2` present in the aarch64 object — so this stream is portable and
`@Vector`-only).

**Stream C — K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K. DONE (6.3x–15.7x).**
`src/kernels/amx/gemm_224_{q2_k,q3_k,q4_k,q5_k,q6_k,q8_k}.zig` + the new shared
driver `src/kernels/amx/kquant_simd.zig`. Each format gained a `gemmQ*_K`
AVX2 sibling of `gemmQ*_KScalar` (which stays as the readable statement of the
math the tests pin against). The driver is ONE generic implementation
(`gemmBF16xQuant`) parameterized per format by a comptime `Cfg` that supplies
`prepare` (per-super-block scale table) and `unpackVec` (16 lanes of one scale
group); all six formats share it, so they cannot drift apart again.
Measured at a LlamaMoe shape (m=1, n=2048, k=7168, pinned, min of 6 runs —
`bench/kquant_bench.zig`; scalar vs AVX2 in the SAME binary/run, which is the
only reliable comparison on this shared host):

| format | scalar | AVX2 | speedup | % of BF16 ceiling BW |
|---|---|---|---|---|
| Q8_K | 18.24 ms (0.92 GB/s) | **1.27 ms (13.19 GB/s)** | **14.4x** | 66% |
| Q6_K | 27.42 ms (0.44) | **2.27 ms (5.31)** | **12.1x** | 27% |
| Q5_K | 26.60 ms (0.38) | **2.20 ms (4.60)** | **12.1x** | 23% |
| Q4_K | 20.95 ms (0.39) | **2.02 ms (4.08)** | **10.4x** | 20% |
| Q3_K | 15.26 ms (0.41) | **2.15 ms (2.94)** | **7.1x** | 15% |
| Q2_K | 14.61 ms (0.33) | **2.33 ms (2.07)** | **6.3x** | 10% |

(ceiling = the vectorized BF16 kernel on the same shape: 1.47 ms / 20.0 GB/s.)
Numerical agreement of the two paths at that K: **rel-L2 1.7e-7 – 2.3e-7**
(vector vs scalar, normalized by the output's L2 norm — against an f64
ground truth both are < 1e-6).

At m=8 the same formats are 10.7x–15.7x. Q8_K clears the roadmap's "≥60% of
the BF16 ceiling bandwidth" target; the sub-byte formats do not and, on this
host, cannot — see the analysis in `docs/perf-roadmap.md` P0.1 (they are
uop-bound at ~1.0 uop/weight because the activations arrive as bf16, while the
BF16 "ceiling" is DRAM-bound at 20 GB/s; the arithmetic ceiling for a 4-bit
format with bf16 activations is ~8 GB/s, so 4.08 is ~50% of what the
instruction set can do).
Design notes that carry the performance, all measured: (a) fold the
scale/min into the weights so there is ONE `@reduce` per output element
instead of one per scale group (at 16-weight granularity a horizontal reduce
costs more than the broadcast-multiply it replaces); (b) make the scale-group
index COMPTIME — a runtime `g` costs ~20 scalar ops of index arithmetic per
16-weight group plus a 4-way indirect jump, worth 1.19x (Q6_K) to 1.52x
(Q3_K); (c) `prepare` the per-super-block scale table once instead of
recomputing the f16 conversion per group. Dispatch is wired into
`gemmQuant` (LlamaMoe) and `kt_matmul_q{4,5,6,8}_k` (C API) — without that the
new functions would never be emitted into the .so.
Tests: 6 new, in `tests/kernels/simd_kernels_tests.zig`, one per format,
`fused == dequantizeRow + f32 dot` (no approximation) with deliberately
non-uniform per-group scales, at m ∈ {1, 4, 5} and 3 super-blocks, plus the
scalar path on the same assertion. Suite total 268 across 18 suites, `zig build test` exit 0.
Also here: `gemmQ4_KScalar` (the BF16-activation Q4_K entry, i.e. Stream A's
remaining item) got the same treatment and is now dispatched too.

**BUG FOUND while doing Stream C (pre-existing, still open):** `test` blocks
inside `src/kernels/amx/gemm_224_*.zig` are **never executed** — the kernel
modules sit behind the `kt` module boundary, so `zig build test` collects tests
only from the root file's own module (`tests/kernels/test_kernels.zig` holds
the authoritative copies of all 15 formats' tests). Two of those dead copies
fail when run: `gemm_224_q6_k.zig` "Q6_K scalar GEMM constant weights" builds
weights for only the first of the two 128-weight chunks (ql[0..64] and
qh[0..32] set, ql[64..128] and qh[32..64] left zero) so it expects 256.0 and
gets -4032 — the live copy in test_kernels.zig:2201 was already rewritten to
the dequantize-and-compare form and passes; and
`gemm_224_q3_k.zig` "Q3_K quantize/dequantize round trip accuracy" asserts
rel < 0.20 and measures **0.329** (Q3_K is ~3.5 bits/weight, so either the
threshold or the quantizer deserves a second look). Both need test_kernels.zig
or a live-suite home, which this stream did not touch.


**Stream D — SwiGLU activation, 0.57x. CLAIMED (swiglu-regalloc) since
2026-10-04.**
**RELEASED 2026-10-04 — `src/kernels/arch/amx.zig` is free again.** The claim
covered `expVec`/`expScalar` plus append-only test lines. The `expVec` work
concluded with no source change (see the negative result below); the only
`amx.zig` edit that landed was the `pshufb16` aarch64 guard, committed alone as
`fb258fa`. The three new tests were appended to
`tests/kernels/simd_kernels_tests.zig` and are now in `HEAD`.

**This claim edits `src/kernels/arch/amx.zig`** (`expVec`/`expScalar` only) plus
append-only test lines in `tests/kernels/simd_kernels_tests.zig` and this
section. The arch-layer hold is free: Stream J (`iq4xs-avx2`) committed its
`pshufb16` work as `2416b68` and released the exclusive claim, which had
explicitly blocked this item until then. `amx.zig` is clean at HEAD, and the only
other live claim in the tree (Stream B / `fp8-gemm-avx2`,
`src/kernels/amx/gemm_224_fp8.zig`) does not overlap. Any *other* task that needs
`amx.zig` must wait for this one the way Stream J waited for P1.3.
The activation is the one place the port is still clearly behind. Measured
instruction counts for the 8-element body: **Zig 48, C++ 40.** The difference
is not algorithmic — both do `vpmovzxwd` + `vpslld` and an FMA polynomial — it
is register allocation:

- Zig emits **6 `vbroadcastss` per iteration** that C++ does not have: the
  polynomial constants are rematerialized from memory inside the loop because
  `expVec` has ~13 simultaneously-live constants (7 polynomial coefficients
  + INV_LN2, LN2_HI, LN2_LO, 0.5, 1.0, and the two clamp bounds) against 16
  ymm. C++ hoists its constants into registers before the loop.
- Zig emits 8 FMAs to C++'s 6 — a deliberate accuracy choice (degree-6 vs
  degree-5, 1.0e-8 vs ~1e-6 relative truncation error). Not a bug.
- 2 `vmovaps` + 1 `vextracti128` of register shuffling.

So the fix is to cut peak live constants, not to change the math. Ideas, in
order of expected value: restructure `expVec` into two phases (split, floor,
pow2 — then the polynomial) so the first group's constants are dead before the
second's are born; hoist the clamp bounds out by clamping once per call instead
of per vector; or drop to a degree-5 polynomial and give up the accuracy
margin. The first two keep <=1 ULP.

**Tried and reverted** (do not repeat): replacing the `[VEC_LEN]f32` array
staging in `applySwiGLU` with direct `*align(1)` vector loads and a vectorized
RNE narrowing store. Measured back-to-back against the same binary six times:
median **1.02x** (range 0.80-1.28x), and the .so grew by 29 instructions
overall. The array staging was not the cause of the rematerialization.

**The register-pressure diagnosis above is WRONG. Measured 2026-10-04
(`swiglu-regalloc`): fix #1 does nothing, and this item is CLOSED as a measured
negative result. `expVec` is unchanged.**

First the counts, which are wrong in both directions. Disassembling the real
specialized `applySwiGLU` (the bench calls it with limit=0 and alpha=0, so LLVM
folds the variant branches away and the whole 3-variant source becomes one
branchless 2x-unrolled loop) against the reference's `swiglu_ref`:

| per 8 elements | Zig | C++ |
|---|---|---|
| instructions in the loop body | **47** | **35** |
| `vbroadcastss` inside the body | 5 | 0 |
| FMAs | 7 | 6 |

So 47 vs 35, not "48 vs 40": the gap is 12 instructions, wider than recorded,
and the broadcast count is 5, not 6. The C++ figure is the surprise — at 35 the
reference is 8 instructions *tighter* than believed, which is most of the gap.

Where the 12 go, from the two disassemblies side by side:

- **6 `vbroadcastss`.** Real, and unique to us; GCC hoists all 8 of its
  constants into the prologue.
- **2 on the floor.** C++ gets `j` from one `vcvtps2dq`, because
  round-to-nearest-even comes free there. We need `mul` + `add 0.5` +
  `vroundps` + `vcvttps2dq`, because our split is in exp space and needs an
  explicit floor rather than a nearest-integer conversion.
- **3 on the LN2_HI/LN2_LO split** (`jf*HI`, `jf*LO`, the second subtract).
  The same fact as the previous bullet from the other side: C++ reduces in log2
  space and needs no split at all.
- **1 on the clamp**: C++ clamps one end in f32 and the other in integer
  (`vminps` + `vpminsd`/`vpmaxsd`); we clamp both ends in f32.

**9 of the 12 are the price of reducing in exp space, which is exactly what
makes us more accurate than the reference (1.0e-8 vs ~1e-6).** Only the 6
broadcasts are register pressure, and they are the only part fix #1 addresses.

Fix #1 was then implemented for real in `expVec` — phase 1 (clamp, split,
floor, pow2) with its constants dead before phase 2 (the polynomial) — and
measured against HEAD:

| per 8 elements | HEAD | two-phase |
|---|---|---|
| instructions | 47 | **49** |
| `vbroadcastss` | 5 | **5** |
| FMAs | 7 | 7 |

It moved the broadcasts **not at all** and made the body two instructions
longer. Timing by interleaved A/B of the two built harnesses on one pinned
core, minimum per side (the box is shared; interference only ever adds time, so
the min is the robust statistic, and C++ alone swings 0.137-0.190 ms run to
run):

| | C++ | Zig | ratio |
|---|---|---|---|
| HEAD, load 3.96, min of 9 | 0.1386 ms | 0.2038 ms | 0.68x |
| two-phase, load 4.47, min of 7 | 0.1367 ms | 0.2038 ms | 0.67x |

The Zig side is unchanged to four significant figures. Reverted.

Why is visible in a controlled probe (one structural property changed per
variant, best-of-15, pinned, ns/element): removing the divide, removing the
clamp, the two-phase restructure, and manual 2x/4x/8x unrolling are **all
indistinguishable from baseline across five runs**. Only two things move the
number, and both remove arithmetic rather than rearrange it:

| variant | ns/elem |
|---|---|
| shipped | 0.89 - 1.17 |
| all of the above | inside that same band |
| deg-2 polynomial | 0.77 - 1.02 |
| loads + bf16 store only, no math | 0.20 - 0.25 |

The noise floor is the binding constraint here: one probe variant is a
byte-identical copy of the baseline and still differs from it by 14-23%, which
is larger than any effect worth chasing. **An activation micro-optimization
measured on this host has to beat ~25% to be distinguishable from noise at
all**, and the prescribed fix was worth 0%.

**So this is not a register-allocation problem and never was.** The activation
is bound by the bf16 widen plus the narrowing store plus FP vector throughput.
Closing the rest of the gap means giving up the exp-space reduction — adopting
the reference's ~1e-6 log2-space `exp_avx2` and the accuracy margin with it,
which is the trade this stream was told not to make. **Closed, code left
correct.** To reopen it, the honest framing is "replace our exp with the
reference's", not "cut peak live constants": the second is measured not to
work.

**What did land** are the three tests the decision rests on, because leaving
`expVec` alone is only safe if its properties are pinned and they were not:
`expVec` had never been compared against `@exp` (only `expScalar` had a ULP
bound, so the vector path every MoE/MLA activation runs had no accuracy test at
all), the ~89% bit-identical rate existed only in prose, and agreement with the
C++ reference was checked only by `bench/cpp_vs_zig/compare.py` — i.e. only by
running the benchmark, never by `zig build test`. All three are in
`tests/kernels/simd_kernels_tests.zig` now. Measured values they pin: **max 1
ULP, 89.37% bit-identical** over 1.5M samples strictly inside the clamp. The
bit-identity budget they use against the reference is 1-in-100000, not zero: on
a different input distribution the plain variant differs on **2 of 262144**
elements, both exact bf16 rounding-midpoint straddles (gate=-1.609375 gives
1.5507811 vs 1.5507815, midpoint 1.55078125). "0 of 262144" is a property of the
benchmark's inputs, not of the port — worth knowing before anyone writes it
down as an invariant.

**Kept as documentation:** the probe harness pattern is the transferable part
— one structural property per variant, interleaved, pinned, minimum-of-N. Two
items in this file were closed on it (this one, and the IQ4_XS "expected win
worthless" result), and on both occasions the thing that moved the number was
not the thing the diagnosis predicted.

**Actually open:**

- [ ] **NR-tiling is the next lever for the K-quant GEMMs** — at m=1 the
  activation widening (6 uops per 16 weights, ~40% of the inner loop) is
  repeated for every one of the n output columns. Blocking n by 4 (one
  widened activation feeding 4 column accumulators, the mirror image of the
  MR=4 tiling that already gives m=8 its 2x) should buy ~1.5x. Register
  budget: 4 columns × 16 lanes of f32 accumulator = 8 ymm, +2 for the
  activations +2 for weight temps = 12 of 16. Dispatch on `m` the way a real
  kernel does (NR for decode, MR for prefill). Driver:
  `src/kernels/amx/kquant_simd.zig::gemmBF16xQuant`.
- [x] **CLOSED by Stream F (`c8b2217`):** `gemmQ8_0Scalar` was the last
  un-vectorized BF16 GEMM — 4x at m=1, 13x at m=4/m=8, and `gemmQuant` /
  `kt_matmul_q8_0` now dispatch to `gemmAvx2`. Kept here as the record of why
  it mattered: Q8_0 is both the GGUF default weight format and the LlamaMoe
  activation format, so it was the hottest quantized GEMM in the tree.
- [ ] **Two modules now both export an entry point called `gemmAvx2`**
  (`gemm_224_q8_0.zig`, `gemm_224_iq4_nl.zig`; the MX and K-quant kernels use
  format-qualified names like `gemmQ8_K`). They are distinct symbols so nothing
  is broken, and both call sites name the module explicitly, but the name no
  longer says which format it belongs to — in a codebase whose recurring lesson
  is that an implicit contract costs a debug session, prefer
  `gemmQ8_0Avx2` / `gemmIQ4_NL_AVx2` when these files are next touched.
- [ ] **Two dead tests in `gemm_224_q{3,6}_k.zig` fail if actually run** —
  see the Stream C note above for both diagnoses. They are invisible because
  in-kernel-file `test` blocks are not executed (Stream C note).
- [ ] **`posix_madvise` is stubbed** (`src/runtime/memory.zig:173,179`) —
  `madviseSequential` / `madviseRandom` take their arguments and do nothing,
  so weight-loading streams get no readahead hint and no DONTNEED. Low risk,
  real win on large GGUF loads. (TO_IMPROVE P0.3)
- [ ] **`LlamaMoe.tp_part_idx` is stored but never used** —
  `llamafile_moe.zig:256,283,302` accept and store it, but nothing selects a
  subpool from it, so every TP shard runs on the default pool. Either wire it
  to `worker_pool.subpoolForCpu` or delete the field so it stops implying
  routing that does not happen.
- [ ] **No end-to-end validation against a real model** — every suite runs on
  synthetic shapes. The GGUF parser is exercised against a real file
  (`qwen3_gguf_e2e_tests.zig`) but no suite checks logit agreement against a
  reference implementation. (TO_IMPROVE P2.1)

**Verified resolved (do not re-open):**

- ABI gates in CI — `verify_abi.py` runs in the `wheels.yml` `abi-gate` job.
- Q4_K x Q8_0 fused matmul — `gemmQ4KxQ8_0Scalar`, wired into
  `runMatmulScratch`; covered by 3 tests including a dequant-and-dot
  ground-truth check. The nibble-interleave bug in it is fixed.
- `kt_llama_moe_forward_many` — commit `1130d0b`.
- `K_BLOCK` runtime tuning for INT8/INT4/FP8/MXFP4/MXFP8 — commit `fe437aa`.
- down-projection memory-bound gap — closed by the GEMM loop-order fix (see
  the SIMD section below): 1.13x-1.88x of the C++ reference at m>=8.
- `src/main.zig.backup*` snapshots — gone (TO_IMPROVE P0.1).
- `src/numa/numa_worker.zig` API rot — no remaining `std.fs.cwd` /
  `std.mem.page_size` uses (TO_IMPROVE P0.2).

---

**Stream E — MXFP4/MXFP8. DONE (layout + codec fixed; both GEMMs vectorized).
CLAIMED (p0.3-mx) since 2026-10-03, completed 2026-10-03.**

`src/kernels/amx/gemm_224_{mxfp4,mxfp8}.zig`. The plan in the roadmap was
"vectorize the MXFP GEMMs". Vectorizing them would have optimized a format that
could not read a real MXFP checkpoint at all, so the order had to change.

**Both block layouts were wrong.** `scale` was `amx.bf16`, making MXFP8 34 bytes
and MXFP4 18 against the format's 33 and 17, and using a scale encoding MXFP
does not have. The reference is explicit
(`operators/avx2/mxfp8-moe.hpp`: "Scale: ue8m0 per-group (group_size=32)",
"ue8m0 byte b -> 2^(b-127)"), as is GGUF. `ue8m0` appeared NOWHERE in this tree
— the scale encoding was never implemented, only mis-laid-out. Now one byte,
`ue8m0_to_f32` is a pure shift (`@as(u32, b) << 23`, verbatim from the reference
including its documented b=0 and b=255 edges), and `@sizeOf` is pinned by a
`comptime` guard plus a test so the layout cannot drift again.

The pre-existing tests could not have caught this: they populated `block.scale`
through our own field type, so they were self-consistent by construction no
matter how wrong the layout was. Same class as the KT_TYPE/block-size
regression in eec0c22, which is why the size is now asserted rather than
assumed.

**The E4M3 codec was wrong in six places** (in `gemm_224_mxfp8.zig`, which keeps
its own private copy of the decoder and encoder):
- decoder: subnormals (exp field 0) decoded as normals, up to 1024x too large
- decoder: 0x7F, the only NaN encoding in E4M3, decoded to 480 — above
  E4M3_MAX, so not even representable
- encoder: the 3-bit mantissa round-overflow was MASKED (`& 0x7`) instead of
  carried into the exponent, so `3.875` encoded as **2.0 instead of 4.0** — a 2x
  error on 1/16 of the entire normal range
- encoder: saturation returned 0x7F/0xFF, which are the NaN encodings, so any
  weight above 448 became NaN
- encoder: all seven subnormal encodings were flushed to zero
- encoder: round-half-up instead of round-to-nearest-even

Verified: **254/256 exact round trip**, the 2 exceptions being 0x7F/0xFF, which
correctly decode to 0 by the reference's "treat as 0 to avoid propagation".

**MXFP8 GEMM vectorized** (`gemmAvx2`, replacing an m/n/k triple loop that
re-streamed the whole weight matrix from DRAM once per token row). Built like
`kquant_simd.gemmBF16xQuant`: n-outer/m-inner, MR=4, explicit `@mulAdd`, one
`@reduce` per output row. Folding the ue8m0 scale into the weight vector is
EXACT, because scaling by a power of two only changes the exponent. The
reference's deferred-hsum rationale is adopted — MXFP keeps the scale separate
from the payload instead of baking it into the dequant, so the horizontal
reduce hoists out of the k loop ("~10x fewer hsums than per-group hsum").

Measured at n=2048, k=7168, pinned, ReleaseFast, best of 5, against the scalar
triple loop in the same binary:

| m | scalar | avx2 | speedup |
|---|---|---|---|
| 1 | 10 ms | 5 ms | 2x |
| 4 | 46 ms | 6 ms | 7x |
| 8 | 88 ms | 12 ms | 7x |

All four SIMD rules verified in the disassembly, not the comments: 16 `vfmadd`
(exactly MR=4 x 4 vectors-per-group), 0 scalar FP mul/add, 4 standalone
`vmulps` (the scale folds, one per vector), and only 4 horizontal reduces in the
whole kernel — deferred-hsum confirmed rather than per-group.

**MXFP4 GEMM vectorized too** (`gemmAvx2` in `gemm_224_mxfp4.zig`), same shape:
n-outer/m-inner, MR=4, explicit `@mulAdd`, one `@reduce` per output row, ue8m0
scale folded into the weight vector. `fp4x8` turns 4 bytes into 8 E2M1 values,
**low nibble first** — the order `unpackMXFP4` uses. That ordering is pinned by a
dedicated test, not just by the oracle comparison, because a swapped pair still
produces plausible-looking numbers instead of garbage.

| format | m | scalar | avx2 | speedup |
|---|---|---|---|---|
| MXFP8 (33 B/blk) | 1 | 10 ms | 5 ms | 2x |
| MXFP8 | 4 | 46 ms | 6 ms | 7x |
| MXFP8 | 8 | 88 ms | 12 ms | 7x |
| MXFP4 (17 B/blk) | 1 | 11 ms | 3 ms | 3x |
| MXFP4 | 4 | 45 ms | 4 ms | **11x** |
| MXFP4 | 8 | 90 ms | 8 ms | **10x** |

at n=2048, k=7168, pinned, ReleaseFast, best of 5, against the scalar triple
loop in the same binary. MXFP4 gains the most, which is expected: at 4 bits per
weight the weight stream is half the size, so the loop order and the FMA tiling
matter more than the unpack.

Two bugs caught while porting, both by the tests rather than by reading:
- the unroll count was `GROUP_SIZE / 16` instead of `/ 8`, so each 32-weight
  group was only half-consumed (16 of 32 weights). The equivalence test failed
  across every shape, and so did the pre-existing `MXFP4 GEMM 16x16x32` test.
- the nibble-order test itself overflowed `u8`: `(i + 9) << 4` exceeds a nibble
  for i >= 7. Worth remembering that `<< 4` on a `u8` panics rather than
  wrapping when the high bits would be lost.

Both MX kernels verify identically in the disassembly: 16 `vfmadd` (MR=4 x 4
vectors-per-group), 4 standalone `vmulps` (the scale folds, one per vector), and
only 4 horizontal reduces in the whole kernel — deferred-hsum confirmed rather
than per-group. The 28 scalar FP instructions present in each sit entirely after
the hot loop, in the cold ragged-k tail (`tailDot`), which only runs when
`k % 32 != 0`.

**REMAINING (not perf, correctness/interop):**
1. **De-duplicate the E4M3 codec.** Two copies now exist:
   `gemm_224_fp8.zig` (Stream B's file) and `gemm_224_mxfp8.zig` (fixed).
   `mxfp8` should import from `fp8` and keep only the MX-specific parts.
   Deliberately deferred while Stream B is in flight. **Note for whoever owns
   Stream B: that file's copy of `f32_to_fp8e4m3` still has all four encoder
   bugs listed above** — the mantissa carry (2x error), NaN saturation, flushed
   subnormals, and round-half-up.
2. **m=1 is the weak case (2-3x) and the LUT loads are why** — 8 scalar table
   loads per 8 weights, ~2.6-3 G weights/s measured. The reference uses
   `_mm256_i32gather_ps` (one instruction); worth trying an asm gather before
   calling this the ceiling. This is the next perf item.
3. `kt_matmul_mxfp8` / `kt_matmul_mxfp4` C-API coverage is untested — the same
   gap class Stream B hit. The GEMMs are now fast, but nothing above them is
   pinned.

**Stream F — Q8_0 GEMM. DONE (4x at m=1, 13x at m=4/m=8). CLAIMED (q8-0-avx2)
since 2026-10-03.**

`src/kernels/amx/gemm_224_q8_0.zig`, plus the three call sites: `kt_matmul_q8_0`
(main.zig), the `gemmQuant` `KT_TYPE_Q8_0` arm in `llamafile_moe.zig` (i.e.
every Q8_0-weight forward), and the root.zig comptime ref.

Q8_0 is the GGUF default weight format AND the LlamaMoe activation format, so
this was the hottest quantized GEMM in the tree — and it was the last one still
on an m/n/k triple loop, in the worst of the three orders: m-outer meant the
whole n x k weight matrix was re-streamed from DRAM once per token row. Same
four rules as the K-quant and MX kernels (n-outer/m-inner, MR=4, explicit
`@mulAdd`, one `@reduce` per output element). Folding `d` into the weight vector
is exact: `d` is f16 and the quants are small integers, so `qs * d` is
representable in f32.

| m | scalar | avx2 | speedup |
|---|---|---|---|
| 1 | 11 ms | 2 ms | 4x |
| 4 | 45 ms | 3 ms | 13x |
| 8 | 90 ms | 6 ms | 13x |

at n=2048, k=7168, pinned, ReleaseFast, best of 5, vs the scalar triple loop in
the same binary.

**Q8_0 is now the evidence for the MX m=1 hypothesis below.** At m=1 it reaches
7.8 GB/s with NO lookup table (the i8->f32 step is pure `vpmovsxbd` +
`vcvtdq2ps`), while MXFP8 reaches 3.0 GB/s and MXFP4 2.6 GB/s from the same
shape with a table lookup per vector. So the ~2.6x gap is the table, not the
memory system — which is what makes an asm `vgatherdps` worth trying.

Disassembly of `gemm_224_q8_0.gemmAvx2` (a real out-of-line function here, 424
insns): 28 `vfmadd`, 141 `ymm` ops, only 5 horizontal reduces for the whole
kernel, and exactly 16 `vpmovsxbd` + 16 `vcvtdq2ps` — the documented
sign-extending i8->f32 path (sign extension is mandatory; Q8_0 quants are
signed and include -128). The 35 scalar FP instructions are the per-block f16
scale conversion (one per 32 weights) and the `@reduce` epilogue, not the
per-element path.

Two tests: the oracle comparison across 7 shapes (m % MR, n = 1, k up to 7168),
and a separate check that every element of [0,m) x [0,n) is written — the scalar
version zeroes `c` and then accumulates, the vector version assigns directly, so
that is only equivalent if there are no uninitialised holes. That test sweeps m =
1..9 with `c` pre-poisoned with NaN.

Worth recording: the equivalence tolerance is 1e-3, not 1e-4, and that is a
measurement rather than laziness. The test's `rnd` returns activations up to ~512
and the dots reach ~10^5, so a partially-cancelling sum shows the f32
summation-order difference amplified to 1.6e-4 (an absolute 0.06 on a value of
381, over 256 terms). Every real defect shows up as O(1) instead — the Q4_K
routing bug was rel=4.2, the MXFP4 encoder bug a clean factor of 2.

**Stream G — IQ* family block-offset bug. DONE (correctness, found while
scoping the IQ vectorization). CLAIMED (iq-block-offset) since 2026-10-03.**

Chasing the next perf item (the nine scalar IQ formats) turned up a
family-wide correctness bug first, which is the more important of the two.

**All nine IQ scalar GEMMs ignored the block offset.** Inside the per-block
loop they indexed the activation as `a[i * lda + k0]` with `k0` in
`0..GROUP`, never adding `blk * GROUP`. So every block of a row was multiplied
by the SAME first GROUP activations:

```
sum += amx.bf16_to_f32(a[i * lda + k0]) * scratch[k0];   // missing blk * QK_K
```

That is correct only when `nb == 1`, i.e. `k == GROUP` — which is the only
shape the pre-existing tests used. Measured against a correct reference on
IQ4_NL: **rel error 2.2 at k=64, 8.9 at k=128, 1.5 at k=256**, and exact at
k=32. Every real checkpoint has k >> GROUP, and `gemmQuant` dispatches all nine
formats, so **every IQ\* model in GGUF was computing garbage**.

Two API shapes, both affected: `(a, b, c, m, n, k, lda, ldb, ldc)` with `QK_K`
(IQ1_M, IQ1_S, IQ2_*, IQ3_*) and `(input, weight, output, m, n, k, input_ld,
weight_ld, output_ld)` with `QK4_NL` (IQ4_XS, IQ4_NL). Note `QK_K` is 256 for
IQ1_M but 32 for most of the others, which is why the bug survived a per-format
eyeball.

Fixed in all nine, each with a comment naming the bug, plus two regression tests
(IQ4_NL and IQ1_M — one per API shape) that build the reference by hand from the
formats' own `dequantizeRow*` and sweep k past one group.

**Lesson worth keeping: a test that only ever uses one shape cannot see a
missing index.** Every one of these GEMMs was "covered" — the suites are 263
tests and include per-format quantize/dequantize and scalar-GEMM tests — but all
of them used `k == GROUP`, which is precisely the one value of k where the
block offset is zero. A loop whose trip count is 1 cannot be wrong about its
stride. When auditing an index, check the trip count first: if it is 1 in every
test, the index is untested.

Two smaller things found while writing the tests:
- IQ1_M's hidden f16 scale is stored as 4 bits inside the `scales` array, so
  **random block bytes legitimately produce NaN/Inf weights**. The assertion has
  to be NaN-aware (kernel and reference must agree on WHICH outputs are
  non-finite), not `isFinite`.
- My first draft of the IQ1_M test used k = 32 and 64 against `QK_K = 256`, so
  `nb = 0` and both cases were **vacuous** — a test that looks like coverage and
  asserts nothing. Fixed to multiples of 256. Worth flagging because it is the
  same failure mode as the bug it was written for.

**Not done here: the IQ\* vectorization itself.** The layouts are correct now
(verified round-trip by the existing per-format tests), so the perf work is
unblocked. Structure looks favourable: 32-weight groups with a per-group scale,
structurally the same as Q8_0/MXFP4, so a `kquant_simd`-style shared driver with
a per-format comptime `Cfg` is the likely shape.

**Stream H — IQ4_NL GEMM. DONE (3x at m=1, 10x at m=4). CLAIMED (iq4nl-avx2)
since 2026-10-03.** First of the nine IQ formats to get the four-rule treatment.

| m | scalar | avx2 | speedup |
|---|---|---|---|
| 1 | 18 ms | 4 ms | 3x |
| 4 | 73 ms | 7 ms | 10x |
| 8 | 144 ms | 14 ms | 9x |

at n=2048, k=7168, pinned, ReleaseFast, best of 5. The win exceeds the SIMD
alone because `gemmIQ4_NLScalar` dequantizes every block into a 32-float
`scratch` and only then does 32 scalar MACs; the vector path never materializes
dequantized weights at all. The `gemmQuant` `KT_TYPE_IQ4_NL` arm now dispatches
to `gemmAvx2`.

**IQ4_NL is SPLIT-HALF, the opposite of MXFP4.** `dequantizeRowIQ4_NL` puts the
low nibbles of all 16 bytes at weights 0..15 and the high nibbles at 16..31:

```
y[j]      = d * kvalues[qs[j] & 0xf]      j = 0..15
y[j + 16] = d * kvalues[qs[j] >> 4]       j = 0..15
```

MXFP4 instead has byte i supplying weights 2i and 2i+1. Reusing the MXFP4 unpack
here would permute every weight while leaving the magnitudes plausible, so the
order is asserted **directly** in the test (checking `dequantizeRowIQ4_NL`
against the `KVALUES` table at known indices) as well as through the oracle
comparison. This is the second time in this session that a copy-adapted unpack
was the hazard — the first was MXFP4's low-nibble-first order.

Disassembly of `gemm_224_iq4_nl.gemmAvx2` (594 insns): 20 `vfmadd`, 91 `ymm`,
only 5 horizontal reduces. **Zero `vpshufb`** — the 16-entry `KVALUES` lookup is
8 scalar table reads per vector, the same pattern as the MX kernels, so the m=1
case is lookup-bound in the same way (4 ms for 8.3 MB = 2.1 GB/s, against
Q8_0's table-free 7.8 GB/s). `vpshufb` would be the natural fix since the table
is 16 bytes and `vpshufb` is exactly a 16-entry byte shuffle, but it needs
inline asm.

**Stream I — IQ4_XS GEMM. DONE (see Stream J for the final number: 5-9x at m=1,
17-19x at m=4/m=8 after the pshufb + scale-hoist follow-up). CLAIMED (iq4xs-avx2)
since 2026-10-03.** Second of the nine IQ formats.

| m | scalar | avx2 | speedup |
|---|---|---|---|
| 1 | 18 ms | 4 ms | 3x |
| 4 | 74 ms | 6 ms | 12x |
| 8 | 145 ms | 11 ms | 12x |

at n=2048, k=7168, pinned, ReleaseFast, best of 5.

The unpack is the same SPLIT-HALF shape as IQ4_NL (per 32-weight sub-block: low
nibbles of its 16 bytes are weights 0..15, high nibbles are 16..31), but the
scale is **per sub-block rather than per block**. The 6-bit field is split across
two arrays and has to be reassembled before use:

```
low  4 bits: (scales_l[ib/2] >> 4*(ib%2)) & 0xf
high 2 bits: (scales_h >> 2*ib) & 3,  shifted up by 4
signed by subtracting 32, then multiplied by the f16 super-block scale
```

`sub-block loop is runtime, the 4 Vec8 inside it comptime` — inlining the eight
sub-blocks too would unroll 128 FMAs per 256-weight block for no benefit, since
the per-sub-block scale is one scalar op feeding 128 FMAs.

**The 6-bit scale index is asserted directly, and the test had to be written
twice for it.** A first attempt hardcoded `scales_h = 0b11_10_01_00_01_10_00_11`
and a hand-derived `highs` array — which read the literal MSB-first by eye while
the implementation reads `(x >> 2*ib) & 3`, i.e. LSB-first. It failed on the
second sub-block with a plausible-looking number. Both fields are now built by
an explicit loop and every sub-block gets a distinct `(low, high)` pair, so
reading the wrong sub-block cannot pass. A bit-order mistake that produces a
plausible magnitude is exactly what a hand-written literal invites.

Also worth recording: the expectation had to be computed from the **stored f16**
(`f16(0.01)` is 0.0100021362, not 0.01), or a 1e-9 assertion is off by 2e-5
before the code under test has done anything wrong.

**IQ\* progress: 2 of 9 done (IQ4_NL, IQ4_XS), both 4-bit and both on the same
split-half unpack. 7 remaining**, and they are materially harder:

| format | group | what makes it harder |
|---|---|---|
| IQ2_XXS | 256 | 32 u16 grid-index+sign entries; needs the `kmap`/`kneighbors` grid tables |
| IQ2_XS | 256 | as above **plus** 16 packed 4-bit scales |
| IQ2_S | 256 | codebook + per-16-element scales |
| IQ3_XXS | 256 | grid tables + packed 6-bit scales + a sign plane |
| IQ3_S | 256 | codebook + per-32 scales + reorder |
| IQ1_S | 256 | 1-bit grid indices + a delta code + kmap |
| IQ1_M | 256 | 1-bit grid + `qs`/`qh`/`scales` split across three arrays, QK_K = 256 |

The four 2-/3-bit families need the grid-index lookup machinery (`iq2xs_init` /
`iq3xs_init` already exist per AGENTS.md) lifted into the inner loop, which is a
different exercise from a nibble unpack plus a scale. The two 1-bit formats are
worse: a grid lookup **and** a delta/second-difference code across elements,
which is inherently sequential rather than a per-element table.

**Suggested order for whoever continues:** IQ2_XXS and IQ2_XS together (shared
grid machinery, and IQ2_XS is a real DeepSeek format), then IQ3_XXS/IQ3_S
together, then the two 1-bit formats last. The open `vpshufb` item below would
also lift IQ4_NL, IQ4_XS and MXFP4 together, so doing it before the remaining
seven may be worth more than a seventh format.

**Stream J — `vpshufb` table lookup + the real IQ4_XS bottleneck. DONE. CLAIMED
(pshufb) since 2026-10-03.**

**The prediction was wrong, and finding out why was the value.** I expected
`vpshufb` to fix the m=1 case, on the theory that 8 scalar table loads per 8
weights were the bottleneck (IQ4_XS at m=1 was 2 GB/s against Q8_0's table-free
7.8 GB/s). Measured: **no reliable difference at all.** The win came from
something else entirely.

**What actually mattered: `f16_to_f32(blk.d)` was inside the per-sub-block scale
helper.** IQ4_XS has one f16 super-block scale and EIGHT 32-weight sub-blocks, so
that conversion was being redone 8 times per 256-weight block — and it sat at the
head of a serial dependency chain immediately before the FMAs. At m=1 there are
only four FMAs per sub-block to hide the latency, so the chain, not the table,
was the bottleneck. Passing `d` in instead of re-reading it:

| m | before | after |
|---|---|---|
| 1 | 4-6 ms | **1-2 ms** |
| 4 | 5-7 ms | **3 ms** |
| 8 | 11-15 ms | **7 ms** |

i.e. 3x -> 5-9x at m=1 and 12x -> 17-19x at m=4/m=8.

`vpshufb` is kept because it replaces 32 scalar table loads with 2 instructions
and cannot be slower, but its effect was not measurable in isolation here and the
commit does not claim otherwise.

**The primitive** (`amx.pshufb16`) is in the arch layer, which is where a generic
byte shuffle belongs. Getting the AT&T operand order wrong is silent: `vpshufb`
takes the DATA operand second, and writing it first still compiles and still
emits bytes — but the table then acts as the index, and `pshufb` writes zero for
any index with bit 7 set, so every output becomes 0. It produced exactly 0/32 on
the first attempt. The round-trip test uses table values 200..215, all of which
have bit 7 set, specifically so that mistake cannot hide.

`KVALUES_IQ4NL` is 16 bytes, which is byte-for-byte the shape `vpshufb` indexes,
so the whole 16-entry non-linear table is now one instruction. `kvalues16` lives
in `gemm_224_iq4_xs.zig` (which owns `KVALUES_IQ4NL`) and `gemm_224_iq4_nl.zig`
imports it rather than duplicating. Both formats now unpack 16 weights per
shuffle.

**CAVEAT ON TODAY'S TIMINGS.** Partway through this work another agent started a
multi-threaded CPU inference job (`zig-ai-engine --ppl` on Qwen3.5-0.8B) and the
host went to load average 17 on 16 cores. Every number in this session's later
entries was taken under that contention. Scalar-vs-AVX2 ratios are measured
back-to-back in one binary so they are more robust than absolute times, but even
they drift: the same m=8 case measured 10x, 14x and 19x on three consecutive
runs. **Treat today's ratios as lower bounds and re-measure on an idle host
before quoting them anywhere.** The 3x -> 17-19x move above is large enough to
survive that noise; the pshufb null result is not resolvable either way.

**A suite test can hang instead of failing.** `zig build test` stalled for 1300
seconds at `test_kernels.test.Subpool pins worker threads to the configured CPU
set (A3)` — the process sat at 0% CPU in `Sl` state with 2 threads, so it was
blocked, not starved. No test has a timeout, so a hang blocks CI indefinitely
rather than reporting a failure. Worth a per-test timeout; recorded here rather
than fixed, since the worker pool is not this stream's file.

**Verification.** Because the parallel runner stalls, every suite was verified by
running its compiled binary directly: **268 tests, all passing** (266 plus the two
new `vpshufb` tests). `verify_abi.py` 112/112; avx2 and amx variants build.

**Housekeeping left by this stream.**
- The arch-layer hold is **released**: `src/kernels/arch/amx.zig` is committed and
  clean, so roadmap **P1.3 (the SwiGLU activation) is unblocked** and can be
  claimed. Note that `expVec` moved down the file when `pshufb` landed —
  it is now at line 675, with the EXP_* constants at 634-645 — so any anchor
  written before that commit is stale.
- `pshufb16` now shares the file with everything else, and `ByteVec`
  (`@Vector(32, u8)`) is a new shared type there. Any task that needs the arch
  layer must claim it again; it is no longer held.
- **One forward-looking warning, still open:** `KVALUES_IQ4NL` has no bytes with
  bit 7 set, so the round-trip test proves the operand order for that table
  only. Before `pshufb16` is reused for the 2-bit and 1-bit tables, check
  whether those tables carry signed entries — a table byte with bit 7 set makes
  the swapped-operand-order bug return zeros, and it would still compile.
- The uncommitted FP8 lookup table that briefly sat next to this work was
  committed separately (`1e8369a`, `4f222ba`) and is Stream B's; nothing from
  that stream was absorbed here.
- The CI test step is now bounded (`timeout 2400 zig build test`), so the hang
  recorded above fails the job instead of stalling it.

**Stream L — INT4 (GPTQ) GEMM. DONE (5 commits: a99f20c claim, 2b6dbb7 kernel,
464252a tests, 4e6e245 bench, c9b52f6 results + the raw-INT4 decision). CLAIMED
(int4-avx2) since 2026-10-04; the work is committed and the in-file CLAIM marker
in `gemm_224_int4.zig` is STALE and should be released by whoever still holds it.**

**Bit-exactness here is a guarantee, not a pass count — and it has a precondition.**
i8 activations x 4-bit weights make every product and partial sum an exact integer
times an 8-bit significand, and the tests hold activations to |a| <= 8 so the
integer sum stays under 2^16. Nothing in that chain can round. **Anyone optimising
this kernel may rely on exactness only while that bound holds**; relax it and the
oracle stops being bit-comparable, so the tests would need loosening too. This is
worth knowing before the 47%-decode work below, not after.

**The plan's primitive was wrong and the reference proves it.** P0.4 said to unpack
with `vpshufb`; GPTQ is affine in the nibble (`w = (q-8)*d`), so it is one
`vcvtdq2ps` plus one FMA, and the reference's `gptq_int4_dequant.hpp` has no
shuffle and no LUT at all. `pshufb` buys a NON-linear codebook. The roadmap now
says so, so nobody re-derives the wrong instruction.

Numbers, with the caveat attached: 3.68 GB/s at m=1 (scalar 0.23), measured at
load 57-62 with no control arm, so those are floors being quoted as point
estimates — `cpp_bench.cpp` has no INT4 coverage, so the harness can only give
absolute throughput plus a control here, never a ratio. **Do not quote the 15-24x
as kernel quality**: the scalar baseline is ~13 cycles/weight (two multiplies into
a serial f32 chain) against 4-5 for the K-quant scalars on the same shape, so what
the ratio establishes is the loop order. The useful host-relative number is against
a same-run control: his own Q8_0 control read 2.47 GB/s in the same run, so INT4 is
1.49x that — but compare it to Q8_0's recorded 7.8 GB/s, NOT the Q8_K figure of
13.19 GB/s two hundred lines away in the roadmap.

Headroom: the decode is 47% of the inner loop (43 instructions per 32 weights, 20
of them decode); a decode-free variant is ~2x, and five ops per 8 weights is the
arithmetic floor for an f32 result. The two routes past it are an integer
`vpmaddubsw` dot needing the per-group activation-sum correction, or an AVX-512
`vpermb` path. **The first is exactly the raw-INT4 trick**, so the deferred ABI
decision and this 2x are the same code.

Raw-INT4: deliberately not done — `KT_TYPE_I4` is the GPTQ block (18 B/32), a
different layout; no in-tree consumer; and the scale is a separate array with a
RUNTIME group size where GPTQ's is fixed at 32, so it is a different signature
rather than a flag. ~40 lines once the header lands (`nibble8` is already
`rawint4_dequant_8x4bit_unscaled` with d = 1). Open decision: fold the scale per
group (reference's way, needs an hsum per group per output) or into the weights
(needs none, non-integer accumulator). The bit-exact oracle does NOT carry over —
raw-INT4 activations are bf16.
`src/kernels/amx/gemm_224_int4.zig`, roadmap P0.4. The last quant family in the
tree with no vectorized path: `gemmFullTile` dispatches to `gemmFullTileScalar`
on every non-AMX host, and that is an m/n/k triple loop, so the weight matrix is
re-streamed from DRAM once per token row. The reference has three AVX2 INT4
kernels (`gptq_int4-moe`, `rawint4-moe`, `gptq_int4_avxvnni-moe`).

Scope as agreed: the scalar reference stays and becomes `pub` (it is the oracle),
the AVX2 GPTQ path lands beside it, tests go in
`tests/kernels/simd_kernels_tests.zig` (that file is open in another session, so
they land as a separate append-only commit once it is free), and raw-INT4 is
either added or recorded as deliberately out of scope with the reason. The
`_avxvnni` variant is out of scope: this host has no VNNI and the scalar path
stays the fallback for it.

**Stream M — FP8 E5M2. DONE (3 commits). CLAIMED (fp8-e5m2) since 2026-10-04.**
`src/kernels/amx/gemm_224_fp8.zig`, `src/main.zig`, `include/kt_kernel.h`,
`tests/kernels/simd_kernels_tests.zig` (+ new `fp8_capi_tests.zig`), `build.zig`,
`docs/api.md`, `docs/perf-roadmap.md`. Claim released.

E5M2 was the last unclaimed piece of roadmap P0.2 and the reason "FP8 E4M3/E5M2"
was half a deliverable: E4M3 had everything, E5M2 had nothing — no encoder, no
decoder, no LUT, no enum value, no C API, no GEMM. Three commits: the codec, the
C API plus the coverage the E4M3 pair never had, and the GEMM sibling.

Delivered: codec (LUT built by bit placement, exact, comptime, builds on aarch64
too), `kt_quantize_fp8_e5m2` / `kt_dequantize_fp8_e5m2`, and `gemmAvx2E5M2`.
ABI 112 -> 114.

The structural decision worth keeping: **both formats share ONE GEMM body**
(`gemmAvx2With`), instantiated per codec, rather than a second copy of the loop.
That is the E4M3/MXFP8 lesson applied preemptively — the codec duplication was
fixed by hand last time, at the cost of four real encoder bugs surviving a
decoder fix, a GEMM rewrite and a claim cycle.

Two real bugs were caught in the new code while writing it, both by writing the
test before trusting the code:
- the LUT folded the sign into the value instead of applying it last, so 0x80
  decoded to +0.0 and re-encoded to 0x00 — negative zero silently lost;
- the encoder's sticky mask included the round bit, which silently degrades
  round-to-nearest-even into round-half-up.
Both mutation-tested: reinstating either bug fails a named test.

Also closed: the E4M3 C API pair had **no tests at all**, because it was declared
`export fn` rather than `pub export fn` and no suite could name it. It is `pub`
now and covered alongside E5M2.

`kt_matmul_fp8` stays E4M3-only — an E5M2 GEMM entry point is future work for
whoever has a caller. The kernel is emitted into the `.so` (via a root.zig
comptime reference) and tested today.

**Stream K — IQ2_XXS GEMM. DONE. The earlier "weak result" entry was WRONG and
is corrected below. CLAIMED (iq2xs-avx2) since 2026-10-04.**

**RETRACTED: "this format is structurally unpack-bound and only worth 4x" was
wrong.** The 1x/4x measurement was real, but the cause was my own `grid8`, not
the format.

`grid8` originally extracted the eight magnitude bytes with `inline for` +
array staging, which meant eight **scalar 64-bit shifts** feeding eight scalar
int-to-float conversions — and AVX2 has no 64-bit integer shift, so LLVM
generated something terrible. Rewritten with explicit vector ops it reinterprets
the u64 as 8 bytes and lets the compiler emit `vpmovzxbd` + `vcvtdq2ps`, with the
sign as a `1 - bit*2` shift/subtract (KMASK makes sign bit j, and the convention
is bit SET => negative).

| m | scalar | grid8 with array staging | grid8 with vector ops |
|---|---|---|---|
| 1 | 272 ms | 283 ms (1x) | **3 ms (69x)** |
| 4 | 1108 ms | 295 ms (4x) | **7 ms (151x)** |
| 8 | 2217 ms | 604 ms (4x) | **13 ms (165x)** |

Verified correct, not fast-but-wrong: both paths' checksums agree
(275183.31 vs 275184.00 at m=1, 2201466.50 vs 2201472.00 at m=8 — f32 summation
order).

**Read the ratio with care.** 69-165x is inflated by a pathological scalar
baseline (272 ms for 14.7M weights is ~55 cycles/weight, because the scalar path
calls `dequantizeRow` per block per output element). The honest measure is the
> **SUPERSEDED 2026-10-04 by the control-armed re-measure below.** The 1.26 and
> 1.06 GB/s figures were taken on a host at load 13-51 from "~3 ms" and "4 ms"
> readings. Re-measured with a control arm per side, the two anchors are
> **0.93 GB/s each — 1.01x apart, a tie**, against a floor of 1.17-1.52x. The
> 1.19x spread between them is not a property of the formats. Absolute, and
> unchanged: **3.78 MB in ~3 ms = 1.26 GB/s at m=1**, which puts IQ2_XXS in the same
ballpark as IQ4_XS (~2.6 GB/s), MXFP4 (~2.6) and well below Q8_0 (7.8). So the
format is not weak; the earlier number was my bug.

**How the wrong conclusion was reached, because the method is the lesson.** The
micro-benchmark that "showed" vector ops were slower (845 ps vs 760 ps per
`grid8` call) was broken twice over: the loop accumulated into a single `f32`, so
it measured latency rather than throughput, and the host was at load 13-51.
`perf` is unavailable here (`perf_event_paranoid=4`), so the bisection never
measured the right thing. The GEMM-level ratio — both paths in one binary, one
after the other — was the only trustworthy signal, and it said 100x. **On a
contended host, believe the end-to-end ratio and distrust micro-benchmarks.**

**Coordinator point 3 — super-block indexing, verified by mutation rather than
by inspection.** Dropping `blk * QK_K` from the activation index in the row-tail
path makes the new equivalence test FAIL, and restoring it makes the suite pass
again (48/48). So the test does cover multi-block shapes; it uses k = 256, 512
and 768, and a block-0-only test would not have caught it.

**Coordinator point 4 — table ownership.** `KSIGNS_IQ2XS` and `KMASK_IQ2XS` are
used from `gemm_224_iq2_xxs.zig`, never re-declared. `gemm_224_iq2_xs.zig`
already aliases them (`KMASK_IQ2XS = iq2xxs.KMASK_IQ2XS`), which is the pattern
to follow.

**This is a much weaker result than every other format in this session, and the
reason is structural rather than a mistake.** Recorded as a negative result
because "it vectorizes, therefore it's fast" is exactly the assumption this
format breaks.

| format | m=1 | m=4/m=8 |
|---|---|---|
| IQ4_XS | 5-9x | 17-19x |
| Q8_0 | 4x | 13x |
| MXFP4 | 3x | 11x |
| **IQ2_XXS** | **1x** | **4x** |

| m | scalar | avx2 | speedup |
|---|---|---|---|
| 1 | 313 ms | 283 ms | 1x |
| 4 | 1332 ms | 295 ms | 4x |
| 8 | 2463 ms | 604 ms | 4x |

at n=2048, k=7168, pinned, ReleaseFast, best of 3, reproducible across three
consecutive runs.

**Why it is unpack-bound.** IQ2_XXS needs, per 8 weights, TWO table lookups
(`IQ2XXS_GRID[fp]` — 256 entries of u64, 2 KB — and `KSIGNS_IQ2XS[idx]`, 128
entries) plus eight byte extractions out of a u64. AVX2 has **no 64-bit integer
shift**, so those eight extractions are scalar work that LLVM can only partly
bundle (disassembly shows 8 `vcvtdq2ps` + 4 `vpmovzxbd`, so it does vectorize the
int-to-float part, but the 64-bit shifts stay scalar). The unpack cost is paid
per output element and is INDEPENDENT of m, so MR=4 can only amortize it 4x —
which is precisely the 4x observed, and why m=1 is flat. Time is nearly constant
in m (283 / 295 / 604 ms), which is the signature of an unpack-bound kernel.

**UNEXPLAINED ANOMALY — do not trust these absolute numbers.** 3.8 MB in ~290 ms
is 13 MB/s, roughly 40x worse than IQ4_XS on the same host at the same moment
(7.8 MB in 15-18 ms). An op-count estimate puts this kernel at single-digit
milliseconds. A 90x gap between the estimate and the measurement means something
is badly wrong that I have not found, and it is the first thing to check before
building on this. Candidates I did not rule out: the 2 KB grid table thrashing
L1 under realistic (non-constant) fingerprints; the scalar 64-bit shifts not
pipelining; or the benchmark's constant `0x35` payloads producing a pathological
access pattern that a real model would not. The 4x ratio is real; the absolute
throughput is not understood.

**Kept the dispatch change** (never slower: 283 vs 313 ms at m=1) but the vector
path should not be considered a win for this format until the anomaly is
explained.

**A real bug caught en route, and it is the reason the claim names the oracle.**
`grid8` first extracted **nibbles** (`>> 4*j`) where `dequantizeRowIQ2_XXS` reads
`grid_bytes[j]`, i.e. a **byte** index into the u64. Wrong for all 256 weights.
Confirmed by hand: grid `0x192B19190819082B`, byte 0 = 0x2B = 43, and
6.179741 / 0.14375 = 43.0. This is precisely the "packed exponent nibble in the
same u32 as the sign indices" hazard the claim warned about — a clean-room
reading of the spec gets it wrong, and only diffing against the existing
dequantizer catches it.

Also: the f16 trap again. `blk.d = 0.05` is stored as f16 = 0.050048828125, so a
hand-rolled reference using the f32 literal disagrees by ~5e-4 and looks like a
kernel bug. Two tests in this session have now been derailed by it.

**IQ2_XS DONE too, reusing `grid8` unchanged** — the layouts are identical
(8 magnitude bytes per grid word, same sign convention), only the scale differs:
16 packed 4-bit scales instead of one exponent nibble, low scale for sub-blocks
0-1 of each group and high for 2-3.

| m | scalar | avx2 | speedup | absolute |
|---|---|---|---|---|
| 1 | 180 ms | 4 ms | 39x | 4.24 MB / 4 ms = 1.06 GB/s |
| 4 | 730 ms | 7 ms | 101x | |
| 8 | 1494 ms | 19 ms | 78x | |

Checksums agree on all three (e.g. 10216204.0 vs 10215942.0 at m=1).

**Both formats' `gemmQuant` arms now dispatch to `gemmAvx2`, and both scalar
references are kept** as the readable statement of the math and the oracle the
vector paths are diffed against.

**IQ2_S, IQ3_XXS and IQ3_S DONE — 7 of 9 IQ formats now on `gemmAvx2`.**

All three reuse `iq2xxs.grid8` **unchanged**, which is the payoff for putting the
unpack in one owner:

- **IQ2_S** is structurally IQ2_XS: a u64 grid word of 8 magnitude bytes and a
  RAW signs byte. Only the index construction differs (`lo | hi`, two bits lifted
  out of `qh`).
- **IQ3_XXS / IQ3_S** each need **two** u32 grid words per sub-block (4 magnitude
  bytes each). Packing them as `g1 | (g2 << 32)` puts g1's bytes at lanes 0..3 and
  g2's at lanes 4..7, and both dequantizers apply signs bit *j* to lane *j* — which
  is exactly what `grid8` already does. So the 8-weight unpack needed **no new
  helper at all**. That is the single best argument for the shared-owner rule.

Three per-format traps, each of which would have been a silent wrong answer:
- IQ3_XXS's group scale uses **0.5**; the IQ2 formats use 0.25.
- IQ3_S's group scale is `1 + 2*scale`, a completely different formula again.
- IQ3_S's index-high bit sits at **bit 8** (masked 0x100, not 0x300), and `qh`
  advances **2 bytes per group pair** while the signs area advances 4 per group.

Each got a comment naming the trap next to the code.

All three passed their equivalence test **on the first attempt** (SIMD suite
49 -> 52), which is what the shared-ownership discipline buys: the unpack was
written once, correctly, and the two IQ3 variants needed no unpack work at all.
Tests sweep k = 256/512/768 with m = 5 so both the MR path and the row tail run,
and `nb > 0` is asserted so no case is vacuous.

**Not benchmarked** — the host is at load 4-51 and a full benchmark is scheduled
for an idle box, so no ratios are claimed here for these three. Absolute
throughput will be ~1 GB/s at m=1 based on the IQ2_XXS/IQ2_XS results, but that is
an estimate, not a measurement.

**IQ1_S and IQ1_M DONE. ALL NINE IQ FORMATS ARE NOW ON `gemmAvx2` — zero scalar
`gemmQuant` arms remain.**

Both turned out to be *simpler* than the "sequential delta code" warning above
suggested. There is no per-element sign at all: the unpack is a grid lookup plus
a group-wide delta, i.e. `dl * (q + delta)`, with `delta` one value per group
whose sign comes from a bit of `qh`.

**The catch, and it would have been a silent wrong answer: the 1-bit formats use
SIGNED magnitude bytes.** Their `gridBytes` returns `[8]i8`, so 0xFF is -1, not
255. Reusing the unsigned `grid8` would be off by 256 on every negative magnitude
while the values stayed plausible. So `grid8s` was added beside `grid8` in the
shared owner — both variants now live side by side, because the IQ2/IQ3 formats
genuinely are unsigned. `@intCast` from `@Vector(8, i8)` to `@Vector(8, i32)`
sign-extends, which is exactly the `vpmovsxbd` wanted.

Bookkeeping traps, each named in a comment beside the code:

- **IQ1_S** packs three scale fields into one u16 per group: 12 bits of
  index-highs (3 each for l = 0..3), a 3-bit scale at bits 12..14, and the delta
  sign at bit 15. Delta is added to the magnitude BEFORE the scale multiply.
- **IQ1_M**'s hidden f16 global scale is *shuffled across the first four*
  `scales` bytes (`extractScale`), so it is per BLOCK and cannot be hoisted per
  group the way every other format's scale could. `qh` advances TWO bytes per
  32-group with the first two sub-blocks reading `qh0` and the last two `qh1`,
  and the per-group scale is two 3-bit fields at bit offsets `6*(ib32%2)` and
  `6*(ib32%2)+3`, applying dl1 to the first two sub-blocks and dl2 to the last.

Both passed their equivalence test first time (SIMD suite 52 -> 54). Their tests
fill blocks through the **byte view** deliberately: IQ1_M's hidden f16 scale lives
in the `scales` nibbles, so random bytes legitimately produce NaN weights, and the
comparison is NaN-aware rather than `isFinite` — the same trap that cost time
twice earlier today.

**Not benchmarked**, per instruction: the host is contended and a full benchmark
is scheduled for an idle box. No ratios are claimed for IQ2_S / IQ3_XXS / IQ3_S /
IQ1_S / IQ1_M. The formats that have been measured share the same unpack path, so
these should land in the same range — an estimate, not a measurement.

> **The range does not exist (2026-10-04).** Re-measured control-armed, the two
> anchors are a **tie at ~0.93 GB/s** (1.01x apart, floor 1.17-1.52x), so
> "1.06-1.26 GB/s" was two contended-host readings of one value. The shared unpack
> path predicts equal throughput, which is what the re-measure shows — the
> intuition was right and the arithmetic turned a tie into a band. **Reframed
> question: do the five land at ~0.93 GB/s?** One point, one floor, well posed.

**Format-anchored tests for `grid8s` — the signedness trap is now pinned, not
just self-consistent.** The IQ1 equivalence tests only prove `gemmAvx2` and
`gemmIQ1_SScalar` agree with EACH OTHER; if both were built on the same wrong
assumption about signedness they would agree perfectly and be worthless. Two
tests close that:

1. Sweeps `IQ1S_GRID` for entries containing a byte >= 0x80 and asserts, per
   lane, that `grid8s` returns the **signed** reading while `grid8` returns the
   **unsigned** one — including `0xFF -> -1.0` vs `255.0` — and that the two
   helpers DISAGREE wherever the byte is negative. Pinning both behaviours is
   deliberate: a future "cleanup" that merged them would break this test instead
   of silently corrupting one format family. Asserts `found > 0` and
   `any_negative` so it cannot pass vacuously if the table ever changes.
2. Builds an IQ1_S block whose 32 sub-blocks all point at negative-magnitude
   grid entries and asserts the full chain: `grid8s` equals the signed magnitude,
   and `dequantizeRowIQ1_S` equals `dl * (q + 0.125)` from the format formula
   (which the suite already pins by hand-tracing against the C reference).

**Verified by mutation, not by inspection.** Making `grid8s` unsigned — exactly
the mistake flagged in review — is caught by FOUR tests, and the new one names the
failure outright: `expected -1, found 255`. Reverted and re-verified green.

**A process note worth keeping.** The first attempt to verify this ran a STALE
test binary and appeared to pass 64/64 while the test file had not compiled at
all (`@floatCast must have a known result type`). The suite name and pass count
looked entirely plausible. Now every verification here checks the build log for
`error:` BEFORE running a binary, and cross-checks the binary mtime against the
build. A green test count from a stale binary is worse than no result, because it
is trusted.

**`numa_pool_tests.Subpool` failed once mid-session and is a load flake**, not a
regression: `expected 6400, found 6391` on a timing-sensitive worker-completion
counter, with this stream touching neither `src/runtime/` nor the pool. Passes 3/3
in isolation. Worth a retry bound if it ever reaches CI.

**Stream L (mine) — P1.2 dequant-to-BF16 paths: Q4_K / Q5_K / Q8_K DONE, 3x.**

The roadmap's "cheap first win", and the reason it is worth doing is leverage:
this path sits in front of **all six K-quants** via `kt_to_float` /
`kt_from_float`, so a vectorized dequant lifts every format that routes through
it, not just one.

Three of six done. Each `*Vec` is a **rewrite of the same math**, not a second
definition of it, and each is pinned **bit-exactly** against the scalar body with
fuzzed scales/mins and payloads — a uniform scale would let a wrong
`getScaleMinK4` index pass, which is the same reason the K-quant GEMM tests fuzz
their scale fields.

| format | structure that makes it vectorizable |
|---|---|
| Q4_K | per 64 weights: 32 low nibbles sharing d1/m1, then 32 high sharing d2/m2 — 4 x Vec8 each with one broadcast scale |
| Q5_K | same, plus Q5_K's 5th bit from `qh` (masked in, scaled by 16, no per-lane branch) |
| Q8_K | `dst = d * f32(qs[j])` — widen and multiply |

Measured on Q4_K at k=7168 (4 KB of weights -> 28 KB of f32): **4 us -> 1 us,
3x, 18 GB/s of f32 out**, checksums **bit-identical** to the scalar path.

Two format facts worth writing down because they are copy-from-sibling traps:
- **Q8_K's `d` is f32**, not f16 like the other five K-quants. Writing
  `f16_to_f32(blk.d)` does not compile, so this one is safe by accident.
- **Q5_K's `qh` never advances.** It is indexed 0..32 for *every* chunk; chunks
  are separated by shifting the hi-bit mask left by 2 each time. Advancing `qh`
  per chunk would be the natural-looking bug.

Still scalar in this path: **Q6_K, Q2_K, Q3_K** (6-bit, 2-bit and 3-bit payloads
respectively — more unpacking per element than Q4/Q5's nibble, and Q2/Q3 need
their shift-plane / k-map machinery). Also untouched: the `*ToBF16` wrappers,
which still run a scalar f32->bf16 loop after the (now vectorized) dequant.

**Stream K complete.** Next unclaimed work: roadmap P0.4's remaining INT4 families
(raw-INT4 and GPTQ-INT4 kernels do not exist yet — another stream has claimed
`gemm_224_int4.zig`), P1.2's dequant-to-BF16 paths, and P2.1's primitive
de-duplication (the E4M3 codec still exists in two files).


`src/kernels/amx/gemm_224_{iq2_xxs,iq2_xs}.zig`. The next real DeepSeek formats;
both are 256-weight super-blocks with a 32 x 16-bit packed field that has to be
reinterpreted as 2 x u32 before anything can be unpacked.

Much heavier than the 4-bit formats: **two table lookups per 8 weights**, not a
nibble unpack. Per 8-weight sub-block it needs `IQ2XXS_GRID[fingerprint]` (a
256-entry u64 table, 2 KB) and `KSIGNS_IQ2XS[sign_idx]` (128 entries), then 8
nibble extractions from the grid and 8 sign bits from the signs byte. The saving
grace is that both tables are tiny and L1-resident.

Two things that make it more tractable than it first looks:
- `KMASK_IQ2XS = {1,2,4,8,16,32,64,128}`, so `signs & KMASK[j]` is exactly
  **bit j of the signs byte**. No mask table is needed at all.
- `KSIGNS_IQ2XS` is 128 entries, which does NOT fit a `vpshufb` (16 per lane),
  so it stays a scalar load — but it is only 128 bytes and always hot in L1.

**The oracle for both is the existing `dequantizeRow*`, not my reading of the
GGUF spec.** These formats pack an exponent nibble into the same u32 as the sign
indices, which is exactly the kind of detail that is easy to misread and that a
"clean-room" reimplementation gets wrong. The vector path must reproduce the
existing dequantizer's output exactly; that is the acceptance criterion.

## ✅ Completed (2026-10-03 — AVX2 SIMD rework; C++ parity, several real bugs)

Benchmarked against the reference's own AVX2 kernels
(`kt-kernel/operators/avx2/avx2_bf16_gemm.hpp`) with the new harness
`bench/cpp_vs_zig/run.sh` (compiles the reference headers unmodified behind a
25-line `ggml.h` shim; same host, same shapes, same inputs, numerical
cross-check). **See `bench/cpp_vs_zig/README.md` for how to run it and
`bench/cpp_vs_zig/RESULTS.md` for the durable measurement record** — note that
host is shared, so absolute ms figures need a quiet box; the ratios are stable. Before this work the port was **0.06x–0.46x** of the reference;
it is now **1.19x–1.79x** on GEMM and **0.71x** on SwiGLU (bit-identical
output). Attribution was measured with a controlled variant probe
(`bench/cpp_vs_zig/zig_probe.zig`) that changes one structural property at a
time, not guessed.

- [x] **`@exp` on `@Vector(8, f32)` scalarizes — 17x on SwiGLU** — LLVM has
  no vector lowering for `expf`; compiler-rt ships only a scalar one and Zig
  does not target glibc's libmvec, so every vector became 8 `call
  compiler_rt.exp.expf` with register spills (verified in the disassembly).
  Replaced with `amx.expScalar` / `amx.expVec`: a Cephes-style construction
  (`x = j·ln2 + f`, `exp = 2^j·(1 + f·R(f))`) that was **derived and fitted,
  not transcribed** — three separate bugs in the first drafts are documented
  inline (exponent-field shift applied before the sign OR; log2-space split
  paired with an exp-space polynomial, silently 40% off; Horner walked the
  coefficients upward and evaluated the reversed polynomial).
  **Verified ≤ 1 ULP, 89.8% bit-identical to `@exp` over 1.5M points** —
  tighter than the reference's own ~1e-6 polynomial, so the port ends up more
  accurate than what it replaced rather than bit-mismatched with it.
- [x] **GEMM loop order — 3.4x–4.0x, the dominant factor** — `gemmExpert` was
  m-outer/n-inner, which re-streams the *entire* weight matrix (29 MB for a
  DSV3 gate/up projection, past this host's L3) from DRAM once per token row.
  Now n-outer/m-inner: one weight row stays L1-resident across all m, weights
  are read from DRAM exactly once, only the small activation matrix is
  re-read (out of L2/L3). Measured with a 4-variant probe: loop order alone
  accounted for essentially the whole gap (164→47.7 ms at m=32); accumulator
  count and FMA were ~1.1x each by comparison.
- [x] **BF16→F32 conversion, not the FMA, is the uop budget** — 2 uops per 8
  elements per operand vs 2 operands per 8 MACs. Added MR=4 register tiling so
  each converted 32-element weight chunk feeds 4 accumulators (32→20
  conversions per k-block). This is why the port now *beats* the reference,
  which re-converts the weight row for every m.
- [x] **MLA/MHA matmuls — 4.6x** — both engines carried their own copy of a
  naive triple loop with `@reduce(.Add, ...)` *inside* the k loop (a full
  horizontal reduce every 16 MACs). MLA's o_proj `[m=1,n=7168,k=18432]` went
  **63.5 ms → 13.7 ms**, i.e. 4.16 → 19.3 GB/s, which is memory-bound — at the
  hardware limit rather than 16x off it. Consolidated into one shared
  `amx.gemmF32Bf16` in the arch layer so the two copies cannot drift again.
- [x] **Explicit GEMM accumulation contract + tp>1 bug fix** — `gemmExpert`
  OVERWRITES its output (callers legitimately pass uninitialised memory);
  `gemmExpertAccum` ADDS to it. `forwardDown` was calling the overwrite-only
  entry once per tensor-parallel rank into one shared FP32 buffer, so **every
  rank but the last was silently discarded** whenever `tp_count > 1`. Making
  the contract explicit fixes it; regression test added.
- [x] **`f32_to_bf16` now rounds to nearest even** — it truncated, biasing every
  value toward zero and disagreeing with ggml's `ggml_fp32_to_bf16` and the
  reference's `avx2::store_fp32_to_bf16`. `f32_to_bf16_trunc` keeps the old
  behaviour. bf16→f32→bf16 still round-trips exactly for all 65536 patterns.
- [x] **Vectorized softmax** — `amx.softmaxF32` shared by `kt_softmax`, MHA,
  MLA and DeepSeek-V3 routing; the old bodies made one scalar libm call per
  element (129k calls/token for a 129k vocab).
- [x] **`swigluClampVec` clamped `gate` from below** — the scalar
  `swiglu_clamp` and the C++ `act_fn` clamp `gate` from above only (`up` is
  symmetric), so the vectorized and scalar paths disagreed whenever a gate
  value fell below `-limit`. The pre-existing test could not see it because its
  inputs never reached the clamp.
- [x] **Test infra unblocked (pre-existing)** — `tools/test_runner.zig` did not
  compile on Zig 0.16.0 (`@errorReturnTrace()` returns `*builtin.StackTrace`,
  `dumpStackTrace` wants `*const debug.StackTrace`), so `zig build test`
  could not run at all; plus 5 `var`-never-mutated errors in the MLA suites.
  `src/mla/mla_tests.zig` also had to import through the `kt` hub instead of
  by relative path, because its root lives inside `src/` and a direct
  `@import("mla_core.zig")` would then reject any `../kernels/...` import.
- [x] **Tests: 241/243 pass** — new `tests/kernels/simd_kernels_tests.zig`
  (17 tests: exp ULP, expVec==expScalar incl. clamp, RNE property test +
  all-65536 round-trip, softmax tails/in-place, gemmExpert vs naive reference
  across MR/32/8/1 tails and non-default `ld`s, overwrite-vs-accumulate,
  `gemmExpertAccum` tp>1 guard, `gemmF32Bf16`, `matmulF32Offset`). The 2
  failures are the `Q4_KxQ8_0` tests added by a concurrent agent's
  uncommitted WIP, which fail on their own in-progress work.

### Known gap left in place (deliberately)

`-Dvariant` still does **not** pin Zig CPU features — it only sets C macros for
the pybind path, so the feature set comes from the native target. Verified:
`zig build -Dvariant=avx2 -Dtarget=x86_64-linux-gnu` emits SSE2-only code
while printing "Building variant: avx2". Not fixed here because wiring AMX
features into an AVX2 host path would emit instructions this hardware cannot
execute, and there is no AMX hardware to validate against. Documented in
AGENTS.md instead. **F16C remains irrelevant**: the weights are BF16, and AVX2
has no BF16 conversion instruction — the widening is `vpmovzxwd` + `vpslld
$16`, which both implementations already emit.

---

## ✅ Completed (2026-08-31 — dev-feedback audit, IMPROVE.md P0-P3 ALL RESOLVED)

Verified audit of external dev feedback (see IMPROVE.md: which claims were
true vs false, with file:line evidence). All real items fixed:

- [x] **A2 — AMX guard was comptime-broken (P0)** — `AmxFeatures.available`
  used `@hasField(std.Target.Cpu.Feature.Set, "amx_int8")` which is ALWAYS
  false in Zig 0.16 (Feature.Set is a packed bitmask, not a named-field
  struct) — every AMX intrinsic silently no-op'd even on AMX hardware and
  on `-Dvariant=amx` builds. Fixed with comptime
  `std.Target.x86.featureSetHas(.amx_tile)` + runtime `detectAmxSupport()`
  (CPUID leaf 7 EBX bits 22/25 + arch_prctl XTILEDATA permission, cached in
  an atomic). All intrinsic guards updated. (commit ff57125)
- [x] **D1/D2/D3 — MoE latent buffer/index bugs (P0)** — D1: sequential
  forward allocated `expert_down_out` as `hidden` (1 token) but
  forwardDown writes `count*hidden` (overflow when an expert got >1
  token; hidden because tests used qlen=1). D2: `forwardParallel` used
  `gi*h+h` — loop index as stride (corruption for qlen>1; hidden for the
  same reason). D3: `forwardGateUp/Down` passed `ldc=inter` while the
  scratch is `m*n` with `n=inter/tp_count` (overflow for tp>1). All
  fixed + 2 regression tests exercising qlen=3 (multi-token-per-expert and
  cross-expert). (commit ff57125)
- [x] **A1 — gemmExpert was pure scalar (P1)** — hot path of
  MoE/MLP/Linear/Gate was a triple for-loop. Vectorized the K axis with
  `@Vector(8,f32)` (AVX2 `vmulps` verified in disassembly) + scalar tail
  for k%8. New `amx.reduceAddFp32` helper. 2 regression tests (aligned k,
  k=37 tail boundary) vs hand-computed reference. **Measured (B2): 5.2x
  decode gate/up, 5.3x prefill, 2.8x decode down (memory-bound), 5.0x
  large square; maxdiff ≤ 1.7e-6.** (commit 7515468)
- [x] **A3 — worker pool had no NUMA pinning (P1)** — `Subpool.init` now
  takes `?[]const usize` CPU list; every worker calls
  `numa.setThreadAffinity` (sched_setaffinity) as its FIRST action.
  `WorkerPoolConfig.subpool_thread_cpus` plumbs lists through.
  `NumaAllocator.alloc` calls real `mbind` (closes the old "TODO:
  numactl/libnuma"). Fixed Zig 0.16 comptime_int|runtime rot in
  `bindMemory` en route. root.zig comptime fn-refs force the NUMA
  helpers into the .so (`nm` verified). Regression test: spawned worker
  observes itself on the pinned CPU set via getcpu. (commit c35a526)
- [x] **D4 — kt_gate_forward ignored DeepSeek-V3 routing fields (P1)** —
  n_group/topk_group/norm_topk_prob/routed_scaling_factor/e_score_correction_bias
  were accepted and discarded; topk_weights were raw logits (not
  probabilities). Implemented the full reference algorithm
  (kt-kernel python/sft/layer.py:696-728): sigmoid scores + bias
  correction + group-top2 sum + topk_group selection + masked final
  top-k + sigmoid weights + optional normalize + scaling. Malformed
  configs fall back to the legacy naive path (no panic on model load).
  GateContext captures the routing config. 2 regression tests mirroring
  the Python reference test (hand-computed ids+weights, normalized sum
  == routed_scaling_factor). (commit f5443f3)
- [x] **B2 — benchmark suite (P2)** — `bench/gemm_bench.zig` + `zig build
  bench` step: gemmExpert vs pure-scalar reference at 4 DeepSeek-V3
  shapes, best-of-5 CLOCK_MONOTONIC, GFLOPS + speedup + maxdiff
  cross-check. Zig 0.16 notes: std.Timer doesn't exist (clock_gettime
  syscall pattern); `{e:.2}` is the scientific-notation format.
  (commit 7d2db9c)
- [x] **A4 — no L1/L2/L3 detection (P2)** — `CpuInfo` gains
  l1d/l1i/l2/l3_bytes populated from
  `/sys/devices/system/cpu/cpu0/cache/index*/` (Zig 0.16 Io API:
  `std.Io.Dir.cwd` + `readPositionalAll`; `std.fs.cwd` is gone). New
  `selectTileParams(cpu)` derives GEMM n_block/k_block from real L2
  (50%-of-L2 working-set budget, tile-aligned). Host measured:
  L1d=32K L2=512K L3=16M → k_block=448 (evidence the fixed
  K_BLOCK=1792 constant exceeds this host's L2). Kernels still use the
  fixed constants — wiring selectTileParams in is the follow-up. 2 tests.
  (commit ca0485a)
- [x] **B3 — kt_cpuinfer_sync was a no-op (P2)** — `Subpool.pending_jobs`
  atomic (incr at doWorkStealingJob entry, decr under mutex + done_cv
  broadcast at exit) + `waitIdle(allow_n)`. `kt_cpuinfer_sync` drains
  every subpool to the caller's allowance. Contract now holds even if
  submit becomes fire-and-forget. Test: two job bursts drain to 0
  pending (accounting bugs fail by deadlock). (commit ca0485a)
- [x] **B1 — page_allocator hardcoded everywhere (P2)** — new
  `kt_set_default_allocator(vtable|null)` C-ABI export (symbol 87):
  userdata + alloc/free/resize fn pointers, malloc semantics. Adapter
  bridges to std.mem.Allocator (layouts are NOT compatible — real
  conversion, not a cast). ALL 9 context types (WorkerPool, TpMoe,
  TpMoeSft, MlaContext, Gate, Linear, MLP, Fp8Transport) capture the
  allocator at `*_new` and free through the CAPTURED one — swapping the
  default between new and free is safe. Per-call scratch (MLA/Linear/
  MLP) uses the context allocator. Bonus bug fixed: TpMoeSft.deinit
  freed grad_inter with hardcoded page_allocator (invalid-free). 2-test
  suite `tests/kernels/allocator_capi_tests.zig` with a tracking vtable:
  1 alloc/1 free balanced, 0 live bytes, even after mid-lifecycle
  default swap. (commit 15a8ea5)
- [x] **Rejected feedback (documented in IMPROVE.md C1-C5)** —
  kt_mla_forward is NOT a placeholder; amx.zig HAS real instructions;
  loadWeights is NOT a no-op; there ARE 15+ numerical correctness
  tests; the dev's kt-*.tar.gz drop-in files do NOT exist on this
  system.

---

## 🟡 High Priority (Next Steps)

### Runtime (follow-ups from the audit)

- [x] **Wire `selectTileParams` into the kernels — CLOSED (ce849ec, validated by 196a6c4)** — the tune hook (`tuneTileParamsForHost` in main.zig) installs measured cache-derived K_BLOCK/N_BLOCK into the BF16 kernel. Two fixes landed in ce849ec: (1) **idempotent guard** — detectCpu allocates, so a process-wide static makes the first call win (was re-running the sysfs probe on every kt_cpuinfer_new); (2) **defensive wiring at every operator _new** — kt_moe_new/kt_moe_new_sft/kt_dsv3_{layer,model,causallm}_new all construct TpMoe (whose init reads GemmKernel224BF.K_BLOCK for BufferA/B/C sizing) and now call the tune first, so callers that build their own pool still get host-sized expert buffers. **Measured validation** (moe_bench, 196a6c4): host-derived K_BLOCK=448 on a 512K-L2 Ryzen gives **-13.2% / -20.3%** on 16-tok / 8-tok prefill shapes vs the 1792 default (working set per block-step drops 9MB -> 2.3MB), noise-level on decode shapes — the A4 design intent, delivered. Note: INT8/MXFP kernels still use fixed K_BLOCK=3584 comptime consts (only the BF16 kernel has the runtime override) — extending the `pub var` pattern to them is optional follow-up if a host ever shows pressure on their shapes.
- [x] **NUMA helpers + kt_worker_pool auto-population — CLOSED (bd7e712 + 476a96e)** — bd7e712 (other dev, 2026-09-01) revived the rotted helpers: numa_topology full IO-layer rewrite (std.Io.Dir.cwd, readSmallFile fixed-buffer, node-probe via ENOENT loop), allocNuma via rawAlloc (comptime-enum workaround), getThreadAffinity return-convention BUG FIX (kernel returns bytes-copied, not errno — every success was misread as failure). 476a96e (this session) wired the remaining piece: kt_worker_pool_new_config auto-populates `subpool_thread_cpus` from NumaTopology.detect when the caller declares a multi-node subpool_numa_map (empty node -> unpinned subpool; single-node/detect-fail -> no pinning). Ownership: WorkerPoolConfig.owns_cpu_lists frees outer+inner slices at deinit; the C path now sets owns_config=true (fixes a pre-existing leak: inline-built numa_map/thread_counts were never freed). 3 C-API tests in tests/kernels/numa_pool_tests.zig (suite 8). Remaining rot: numa_worker.zig (unreferenced, unowned — fix on first use).
- [x] **Route internal scratch through the MoE allocator — CLOSED (58b0cb1)** — `routeExperts`/`routeExpertsDeepSeek`/`routeExpertsWithOpts` scratch (logits + score rows + group scores/mask) now allocates through the caller-threaded allocator: `GateContext.allocator` at kt_gate_forward (B1-captured), `self.allocator` in DeepseekV3DecoderLayer.forward, testing-allocator at test sites. Buffers no longer escape a kt_set_default_allocator-installed tracker. No ABI change (internal API). Routing tests green: Gate+MoE e2e, group-top2 vs Python reference, both fallbacks.
- [x] **`kt_mla_forward` qlen_count > 1 — CLOSED (139bf63 + 2e8ce43)** — paged/batched path: qlen_count==1 keeps the legacy sequential route; >1 loops per sequence through `MlaEngine.forwardPaged` with per-sequence page tables (mla-tp.hpp:84 contract), page_table_lens validated (>= ceil(kv_len/tokens_per_page)). `MlaKvCache` gained paged accessors (`pagedGetPageInfo/pagedWriteToken/pagedGetNopePtr/pagedGetRopePtr/ensurePageCount`) + save/load/dump/fromDump (32-byte "KVCA" header; page_table NOT serialized — scheduler's job, same as kvcache_load_dump.cpp). **Equivalence tests (2e8ce43 — the coverage 139bf63 shipped without)**: (a) scrambled==identity — same logical content on different physical pages, bit-for-bit (non-zero weights + non-zero-output guard = non-vacuous); (b) paged==legacy — single sequence through identity table == sequential path bit-for-bit. **Test (a) caught a real OOB on first run**: forwardPagedSequence sized the page pool from the LAST table entry only; a scrambled table with its max page index earlier (e.g. {7,6}) read past the pool (`index 7, len 7` in pagedGetRopePtr). Fixed: scan all n_pt_entries for max_page. 143 tests, 0 leaks, ABI 96/96 triple gate PASS.

### Python Packaging
- [x] **Minimal pybind11 wrapper — DONE (4040b5b)** — see the Tier-1 entry
  under Completed GGML/pybind11 below. (`python/kt_kernel/` ctypes wrapper
  remains the zero-dependency surface.)

### CI/CD
- [x] **Run the ABI gates in CI** — `wheels.yml` has `abi-gate` job
  (builds all variants, runs `verify_abi.py` + `audit_layout.py` +
  `zig build test`; other dev added in `7db2936`).
- [x] **Python binding for `kt_set_default_allocator`** — symbol 87 has
  ctypes wrapper (lines 542-566 of `python/kt_kernel/__init__.py`):
  `kt_allocator_vtable` Structure (userdata + alloc/free/resize
  CFUNCTYPEs) + `kt_set_default_allocator(POINTER(vtable))` call
  (None restores default). End-to-end test at
  `tests/test_allocator.py`: installs a tracker allocator, runs
  kt_worker_pool_new(0) + kt_worker_pool_free() round-trip, asserts
  5 allocs/5 frees/408 bytes (matching the WorkerPool's internal
  buffer setup), then asserts kt_set_default_allocator(None) actually
  stops the tracker (counts stay frozen across a second round-trip).

---

## ✅ Completed (2026-08-29)

### Critical Build Fixes
- [x] **Module System**: `build.zig` now uses `src/root.zig` as root source file
- [x] **`src/root.zig`**: Created with proper submodule re-exports
- [x] **`zig build`**: Produces `zig-out/lib/libkt_kernel_ext.so`
- [x] **`zig build test`**: All tests compile successfully

### Zig 0.16 Migration Fixes
- [x] **`@intFromFloat` → `@intFromFloat` with typed cast**: Fixed in `buffers.zig`
- [x] **`std.atomic.Bool` → `std.atomic.Value(bool)`**: Fixed in `worker_pool.zig`
- [x] **`std.Thread.Pool` removed**: Replaced with placeholder in `WorkerPool`
- [x] **`std.DoublyLinkedList(T)` → `std.DoublyLinkedList`**: Fixed API call
- [x] **`std.time.sleep()` removed**: Replaced with spin-wait
- [x] **`alignedAlloc` signature change**: Updated to new 3-arg API
- [x] **Pointer types need optional**: Updated `[*]T = null` to `?[*]T = null`
- [x] **`packed struct` with arrays**: Changed to `extern struct`
- [x] **Packed struct fix in `arch/amx.zig`**: TileConfig now uses extern struct
- [x] **AMX features check**: Use `@hasField` for comptime safety
- [x] **moe.zig indentation**: Fixed ternary indentation
- [x] **main.zig forward references**: Removed all undeclared function references
- [x] **cpu_detect.zig `||` → `or`**: Fixed all string comparisons
- [x] **Test infrastructure**: Tests now compile and run

---

## 🟡 High Priority (Next Steps)

### Task assignments (2026-08-30, two agents working concurrently)

**Dev A — MLA C API wiring + warmUp** (files: `src/main.zig`, `src/kernels/moe/moe.zig` warmUp only):
- [x] **Wire `kt_mla_*` C API to the real MLA engine** — replaced 5 placeholders (kt_mla_new/free/forward/prefill/decode) + kt_mla_update_kv_cache. MlaContext wrapper struct holds MlaEngine + MlaKvCache. BF16↔F32 conversion at C/Zig boundary. `*anyopaque` cast to `[*]const u16` for BF16 weights. External kv_cache param documented as ignored. Engine end-to-end test added.
- [x] **Upgrade `TpMoe.warmUp` to real pre-touch** — replaced no-op with first-byte touch of gate/up/down BF16 storage. Guarded by weights_loaded and bf16_weights_alloced. Touches first page of each storage to prime the TLB.

**Dev B — MXFP4/MXFP8 kernels** (files: `gemm_224_mxfp{4,8}.zig`, `src/root.zig` re-exports, `tests` append-only, docs):
- [x] **Complete MXFP4/MXFP8 kernels** — both files now in root.zig's import graph (re-exports + comptime fn-refs force analysis per the LAZY-ANALYSIS lesson). mxfp4 bogus `unpackMXFP4(block, &sum)` line removed; mxfp8 u8 cast in `f32_to_fp8e4m3` fixed. Per-32-block scale folded into the dequant (FP8 pattern adapted to block scale). AMX tile path mirrors `gemm_224_fp8.zig` (config/cleanC/storeC/loadA/runTile/loadB + gemmFullTile dispatch, scalar fallback preserved). Two exact-value tests in `test_kernels.zig`; 27 kernels + 11 MLA = 38 total pass, 0 leaks. References: `operators/amx/mxfp8-moe.hpp`, `fp4-moe.hpp`, `avx2/mxfp4-moe.hpp`.

**Exclusivity notes (2026-08-30, live two-agent work)**: Dev A owns main.zig + moe.zig; Dev B owns mxfp files + root.zig. Shared files only by the stated constraints (tests append-only, docs append/mark). `zig build test` currently exits 0 with 0 leaks (38/38) — keep it that way.

### Kernels
- [x] **Complete INT4 GPTQ kernel (`gemm_224_int4.zig`)** — replaced scalar fallback with AMX tile path (`tileint8dpd` + on-the-fly INT4→INT8 dequantization). Lo/hi nibble pattern from C++ `GemmKernel224Int4` (amx_kernels.hpp:1559-1848). Per-group scales applied at K-block boundary via `applyScales` (INT32 → FP32). Correctness test added to `tests/kernels/test_kernels.zig`. Non-AMX hosts fall back to the preserved scalar implementation.
- [x] **Complete FP8 E4M3 kernel (`gemm_224_fp8.zig`)** — replaced scalar fallback with AMX tile path that reuses the BF16 GEMM tiles (`tilebf16dpd` accumulates into FP32, no INT32 scratch). On-the-fly FP8→BF16 dequantization in `loadB` (byte-level, 1:1: each FP8 byte becomes one BF16 short). Per-row scale applied at end of each `(m, n)` block (not per K-step). Tile config matches BF16 (TILE_M=16, TILE_K=32, TILE_N=16, VNNI_BLK=2). Correctness test added (skips on non-AMX). Non-AMX hosts fall back to scalar.
- [x] **Complete MXFP4/MXFP8 kernels** — both wired into root.zig's import graph; AMX tile path + scalar fallback. 2 exact-value tests; 27 kernels + 11 MLA = 38 total pass, 0 leaks.
- [x] **Vectorize `applySwiGLU` with `std.simd`** (Dev B) — VecF32=8 vectorized helpers (`swigluVec`/`swigluClampVec`/`swigluOaiVec`) in `arch/amx.zig`; `applySwiGLU` refactored to branch-once on variant + vector inner loop (8-wide) + scalar tail for `n % 8`. Verified bit-exact vs scalar across all three variants and `n`=5..17 (covers tail, exact-width, multi-vector); 1 new correctness test in test_kernels.zig.
- [x] **Add actual AMX inline assembly (previously placeholder)** — `ldtilecfg`, `tilerelease`, `tileloadd`, `tilestored`, `tilezero`, `tilebf16dpd`, `tileint8dpd` all wired in `src/kernels/arch/amx.zig` with proper comptime tile-register immediates. Includes XFEATURE_XTILEDATA permission request via `arch_prctl`. Correctness tests added to `tests/kernels/test_kernels.zig` (skip on non-AMX hardware).

### MoE Layer
- [x] **Implement `TpMoe.loadWeights()` with online quantization (BF16 → INT8/INT4)** — BF16 path now packs all 3 projections (gate, up, down) with per-TP-rank slicing. INT8 path unchanged. Module-level BF16 weight storage used to avoid struct field changes.
- [x] **Complete the MoE compute path** — `loadWeights` BF16 packing (all 3 projections), `forwardGateUp`/`forwardDown` real bodies (was no-ops with TODOs), `forward` expert GEMM enabled (gates, SwiGLU, down, routing weight accumulation). Forward equivalence test passes.
- [x] **Implement `merge_results()` with AVX512 FP32 add + BF16 convert**
- [x] **Expert routing with top-k selection (GEMM-based, naive top-k)** — `routeExperts` upgraded from scalar dot product per (token, expert) to one batched BF16 GEMM (`gemm_bf16.gemmExpert`) + naive O(n*k) top-k. Pool parameter made optional (`?*WorkerPool`) so the C API (no pool) and MoE forward (real pool) both work. Naive top-k is fine for k=2-8; SIMD/heap optimization is a follow-up.
- [x] **Replace 5 placeholder TpMoe methods** — `deinit` (frees the two init-allocated slices, page_allocator convention; buffer structs are non-owning POD views), `warmUp` (no-op + TODO until loadWeights populates ptrs), `loadWeightsWithMap` (double-load + logical-slot remap), `forwardGateUp`/`forwardDown` (guarded no-ops with real bodies in comments, blocked on the loadWeights BF16 bug). End-to-end test with zero inputs passes; 5 TpMoe placeholder methods now safe.
- [x] **Fix `gemmExpert` `weight_ld` OOB in MoE expert compute** — `moe.computeExpert` (3 sites) and `TpMoe.forwardGateUp`/`forwardDown` (3 sites) were passing `weight_ld = n` instead of `k`, causing OOB reads of up to `n^2 - n + k - 1` past the end of `n*k`-element weight buffers. The OOB happened to land in page allocator zero-fill on Linux so existing tests passed by accident. All 6 sites now use `weight_ld = k` matching the `[n, k]` row-major layout. See LESSONS_ZIG.md §"gemmExpert weight layout".

### C API Completeness
- [x] **MLA attention — code complete in `src/mla/`, re-exported from `root.zig`** (config, cache, core modules wired into the library; 11/11 standalone tests passing). C API integration (`kt_mla_*` in `main.zig`) assigned Dev A — see Task assignments above.
- [x] **Gate (`kt_gate_*` functions)** — `kt_gate_new` allocates a `GateContext` (BF16 weight pointer + dims); `kt_gate_free` destroys it; `kt_gate_forward` calls `moe.routeExperts` with the stored weights and `null` pool. BF16-only (`@panic` on other dtypes, matching the rest of the C API).
- [x] **Linear/MLP (`kt_linear_*`, `kt_mlp_*`)** — both wrap a real GEMM (BF16 weight + BF16 input → F32 output → BF16 output). Linear = 1 GEMM; MLP = gate GEMM + up GEMM + F32 SwiGLU + BF16 round-trip + down GEMM. Also fixed `kt_linear_config_t` / `kt_mlp_config_t` / `kt_gate_config_t` to match the C header field names (was: `weight_ptr`/`dtype`; now: `weight`/`weight_type` for Gate, `proj`/`proj_type`/`hidden_type` for Linear, `gate_proj`/`up_proj`/`down_proj`/`*_type`/`hidden_type` for MLP). Linear+MLP end-to-end test in test_kernels.zig exercises the same pipeline via `gemm_bf16` directly. 25/25 tests pass.
- [x] **SFT/LoRA backward pass** (Dev B) — `TpMoeSft.backward` computes grad_input, 6 LoRA grads (gate/up/down × A/B), base-weight grads, routing-weight grads. `kt_moe_backward` C API export. Smoke test passes (43/43 total).
- [x] **FP8 layerwise transport (`kt_fp8_*`) — #4 DONE (2026-08-31, verified by coordinator)** — all 10 symbols implemented in `src/main.zig` (+351 lines): `kt_fp8_layerwise_init`, `kt_fp8_transport_{new,free,close,closed,rank,tp_size,num_experts,run_producer,wait}`. Callback-only plumbing per spec (real dequant+GEMM stays in the caller via `kt_matmul_fp8`); `cuda_device`/`local_gpu_ptrs`/`timeout_ms` documented no-ops (CPU port); pointer math matches reference layout `(slot * tp_size + rank) * 4 + kind`, slot = expert % 2 (fp8_layerwise_transport.cpp:409-417); `Fp8ControlHeader` extern struct mirrors fp8_layerwise_transport.cpp:49-56. Verification (evidence standard): 6 variants build exit 0; `zig build test` 47/47 (34 kernels + 11 MLA + 2 new FP8 transport tests), 0 leaks, exit 0; `tools/verify_abi.py` confirms all 10 transport symbols exported by every variant — only `kt_mla_load_weights` remains missing (next item). Tests: `tests/kernels/fp8_transport_tests.zig` (lifecycle + exact reference pointer-math assertions + null-handle sentinels), wired as third suite in `build.zig`.
- [x] **`kt_mla_load_weights` C API — DONE (2026-08-31) — ABI is now 100% complete against the header.** Mirrors the C++ reference semantics (TP_MLA_Common::load_weights sets `weights_loaded`; forward() throws "Not Loaded" without it — mla-tp.hpp:39,86): `MlaContext.weights_loaded` flag set by `kt_mla_load_weights`; `mlaForwardImpl` (covers forward/prefill/decode) panics with a clear message if called before load_weights. `kt_mla_new` additionally validates all 7 weight pointers non-null up front (C header types them non-nullable `*anyopaque`, but a caller bug passing 0 would otherwise surface as a far-away segfault). All 7 `kt_mla_*` exports marked `pub` for Zig-level test access (no ABI change). Verification: `tools/verify_abi.py` **full PASS — all 53 declared symbols exported by all 6 variants** (first time); `zig build test` 48/48 (34 kernels + 11 MLA + 2 fp8 + 1 new MLA C-API lifecycle test `tests/kernels/mla_capi_tests.zig`, fourth suite), 0 leaks, exit 0; 6 variants build exit 0.

---

## 🟢 Medium Priority

### Build System
- [x] **Test runner actually runs tests** — replaced `test_step.dependOn(&test_obj.step)` with `test_step.dependOn(&b.addRunArtifact(test_obj).step)` in build.zig for both test modules. Fixed use-after-free in `src/runtime/cpu_detect.zig::detectCpuLinux` (model_name pointed into a soon-to-be-freed buffer). Fixed 3 `.flags = std.ArrayList(u8).init(allocator)` type mismatches.
- [x] **Multi-variant build** — `zig build all-variants` builds all 6 variants with distinct names (`libkt_kernel_ext_{variant}.so`). Per-variant C macros preserved; single-variant `-Dvariant=` path kept for dev.
- [x] **Variant-specific library names** — each variant installs as `libkt_kernel_ext_{suffix}.so`.

### Runtime
- [x] **Worker pool work-stealing** — replaced busy-wait spin + broken queue with atomic-counter work-stealing via `std.c.pthread_mutex_t`/`pthread_cond_t`. Workers block on a condvar when idle (no CPU burn). `doWorkStealingJob(count, fn)` distributes `count` tasks across all subpool threads. Verified: 1000 tasks across 4 threads complete correctly.
- [x] **NUMA topology** — `numaNodeOfCpu()` reads `/sys/devices/system/cpu/cpu{N}/topology/physical_package_id`; `getCpuCountPerNuma()` parses `/proc/cpuinfo` (correctly returns 16 CPUs on 1 NUMA for Ryzen 5800H).
- [x] **Verify worker_pool.zig** — work-stealing pool with NUMA subpools compiles and runs.
- [x] **Wire work-stealing into MoE kernels** (Dev A+b, commit b762a2e) — `TpMoe.forward` now gates on `config.pool`: parallel path uses `doWorkStealingJob` with per-expert private BF16 scratch (zeroed, sequential reduction); sequential fallback unchanged. 44/44 tests pass including pool-vs-sequential equivalence. Review gate (data-race-free reduction, allocator symmetry, sequential fallback, BF16-scratch semantics) confirmed.
- [x] **Wire work-stealing into SFT/LoRA path** (Dev A+b, commit d1e0adb, reviewed 2026-08-30) — `TpMoeSft.forward_sft` mirrors the MoE pattern: branches on `self.moe.config.pool`, dispatches per-expert tasks via `doWorkStealingJob` on `subpool[0]`, each task owns its BF16 scratch (`[][]amx.bf16`), writes a disjoint `save_for_backward` cache slice via precomputed `expert_token_offset[e]`, sequential reduction after join (BF16 → F32 via `bf16_to_f32`, weighted by routing w). SFT-specific `g_parallel` (moe_sft.zig:67) is module-scoped — decoupled from moe.zig's context. Review gate (a–e + backward-cache) confirmed: 6 variants build, 45/45 tests pass (34 kernels incl. new "SFT forward_sft: work-stealing pool matches sequential (equivalence)" + 11 MLA), 0 leaks. ABI verifier PASS on all non-FP8 symbols (10 `kt_fp8_transport_*` + `kt_mla_load_weights` missing are pre-existing #4 carve-outs, not #5 defects).

---

## 🔵 Low Priority / Nice to Have

### Python Packaging
- [x] **Minimal ctypes wrapper** (`python/kt_kernel/__init__.py`) — pure-Python `ctypes` wrapper that `dlopen`s the variant `.so` (auto-detected from `/proc/cpuinfo`), exposes `kt_version`, `kt_get_cpu_variant`, worker pool, CPUInfer, Linear, Gate, MLP, and bf16 conversion functions. Smoke-tested: `import kt_kernel; print(kt_kernel.kt_version())` works.
- [x] **`pyproject.toml` + `setup.py` for `kt-kernel` wheel** — setuptools with package-data for the 6 variant `.so` files; custom `build_ext` runs `zig build all-variants` and bundles them into `python/kt_kernel/`. Verified: `python3 setup.py build_ext --inplace` → `import kt_kernel` → `kt_version()` returns `b'0.6.1-zig'`.
- [x] **CI/CD for multi-variant wheel building** — `.github/workflows/wheels.yml` builds all 6 variants and packages the wheel.
- [x] **Minimal pybind11 wrapper in C++ that loads Zig `.so`** — DONE (4040b5b): `bindings/kt_kernel_pybind.cpp` + `build.sh`, module named `kt_kernel_ext` (C++ reference drop-in), gated by `tools/audit_layout.py`. Full entry under GGML/pybind11 Tier-1 below.

### Advanced Features
- [x] **SFT/LoRA backward pass** — forward_sft (Dev A) + backward (Dev B) complete with LoRA kernels; kt_moe_forward_sft/kt_moe_backward/kt_moe_update_lora_weights C API exports. 43/43 tests pass.
- [x] **Speculative decoding (MTP head) — RESOLVED: out of scope for this .so** (verified 2026-08-31) — MTP/NextN speculative decoding in ktransformers is a **serving-engine feature that runs on the GPU via SGLang/vLLM EAGLE flags** (`--speculative-algorithm EAGLE`, doc/en/DeepSeek-V4-Flash.md:188-198), NOT a CPU kernel in kt-kernel. Evidence: (1) zero MTP/nextn implementation anywhere in kt-kernel C++ or the kernel-side Python — only the config field `num_nextn_predict_layers=1` + weight-checkpoint skipping of the extra layer (check.py:67, sft/artifacts.py:319-350 reject MTP tensors); (2) LESSONS_KTRANSFORMERS.md:123 lists it as a feature of the *framework*, not the kernel library; (3) the archive's modeling_deepseek_v3.py has no NextN class. The draft head itself is a small decoder (embed×2 → linear → decoder layer → norm → lm_head) that the serving engine orchestrates on-GPU against the CPU-offloaded main model via the FP8 layerwise transport (#4, landed). If CPU-side drafting is ever wanted, every primitive already exists here (DecoderLayer + lm_head + our model-orchestration tower) — but that would be new design work, not a port. Closing the TODO as reference-absent, same verdict as the operators/kml/ discovery.
- [x] ARM/KML backend (NEON/SVE) — RESOLVED: 2026-09-02
  - [x] **aarch64 variant detection** — `selectBestVariant` picks `sve2` > `sve` > `neon` for `cpu.vendor == .arm` or `cpu.arch == "aarch64"`. 7 tests in `tests/kernels/aarch64_detect_test.zig`.
  - [x] **NEON Phase-2: BF16 GEMM kernel** — `src/kernels/arch/gemm_224_bf16_neon.zig` (205 lines) parallels the A1 work in `src/kernels/amx/gemm_224_bf16.zig` with `@Vector(4,f32)` (4 f32 lanes = 128-bit NEON register). Comptime-dispatched: x86_64 builds hit the SSE fallback; aarch64 cross-build emits NEON fmla/fmul. Dispatcher (`gemmExpert`) routes to scalar or vectorized based on the NEON comptime gate. 4 tests in `tests/kernels/neon_kernel_test.zig`: vecF32 layout, dispatcher routing, vectorized-vs-scalar reference, correctness on x86_64 fallback.
  - [x] **NEON feature detection** — `src/kernels/arch/neon.zig` exposes `NeonFeatures.available` (comptime flag), `selectBestVariant` honors it. 1 test in `tests/kernels/neon_arch_test.zig`.
  - [x] **Cross-compiled aarch64 `.so`** — `zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu` produces `zig-out/lib/libkt_kernel_ext_neon.so` (5.4 MB ARM ELF, 112 symbols). `tools/verify_abi.py` gates it PASS: 112/112 header prototypes exported, arity audit clean. File header verified: ELF 64-bit LSB aarch64, dynamically linked, with debug_info. Build-time footgun guard: passing `-Dvariant=neon` without `-Dtarget=aarch64-linux-gnu` exits with a clear error (no silent x86 "neon" build).
  - **Remaining follow-ups (out of scope for this workstream)**: SVE/SVE2 GEMM kernels (the dispatcher already routes to `sve2 > sve > neon` but the actual vectorized kernels beyond the BF16 one are not yet ported; further work only justified when a real aarch64 host is in the dev rotation).
- [x] **GGML Q8_0 kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q8_0.zig`: byte-exact `BlockQ8_0` (34 bytes: f16 `d` + 32×i8 `qs`, `@sizeOf` compile-checked) vs ggml-common.h; quantize/dequantize matching ggml-quants.c:276/553 (`d = amax/127`, `qs = round(x·1/d)`, `y = qs·d`); f16↔f32 via Zig's **native f16** `@floatCast`/`@bitCast` (first attempt hand-rolled the bit math — buggy subnormal path, replaced by native casts, lesson for LESSONS); scalar GEMM (BF16 activations × Q8_0 weights → F32, on-the-fly dequant). 5 exact-value tests in test_kernels.zig (f16 round-trip, 34-byte layout, quantize/dequant tolerance, 2 GEMM constants). Wired into root.zig with comptime fn-refs (emission verified via nm). Remaining formats: Q4_K → Q6_K → Q5_K (defer Q8_K — super-block scale/min layout is the fiddliest and least used); C-API quantize/dequantize exports are Phase 2.
- [x] **GGML Q4_K kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q4_k.zig`: byte-exact `BlockQ4_K` (144 bytes: f16 `d`+`dmin`, 12-byte 6-bit packed `scales`, 128-byte 4-bit `qs`, `@sizeOf` compile-checked) vs ggml-common.h; `nearestInt` (magic-number trick, ggml-quants.c:621), `getScaleMinK4` (6-bit scale/min unpack, :880), full `makeQkx2Quants` error-minimizing search (:799, 20-step), `quantizeRowQ4_K` (:1457), `dequantizeRowQ4_K` (:1529); scalar GEMM (BF16 × Q4_K → F32). 5 tests in test_kernels.zig (nearest_int known values, 144-byte layout, scale/min unpack scheme, round-trip accuracy <15% rel on smooth data, GEMM constant 256.0). Wired into root.zig (emission verified via nm).
- [x] **GGML Q6_K kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q6_k.zig`: byte-exact `BlockQ6_K` (210 bytes: `ql[128]` low nibbles + `qh[64]` 2-bit highs + 16×i8 `scales` + f16 `d`, compile-checked); `nearestInt`, `makeQxQuants` (rmse_type=1 weighted search + ±9 refine loop, ggml-quants.c:628), `quantizeRowQ6_K` (:1869 incl. all-zero fast path), `dequantizeRowQ6_K` (:1939 — 6-bit quants = ql nibble | qh 2-bit<<4, minus 32); scalar GEMM. Zig gotcha found: C packs L as `uint8_t` — Zig `i8 >> 4` is arithmetic and `(3<<6)` overflows i8, must cast to u8 before the qh packing. 4 tests in test_kernels.zig (210-byte layout offsets, round-trip rel err < 10% on smooth data, all-zero block → d=0 → all-zero output, GEMM vs dequantized reference). Wired into root.zig; emission verified via nm.
- [x] **GGML Q5_K kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q5_k.zig`: byte-exact `BlockQ5_K` (176 bytes: f16 `d`+`dmin`, 12-byte 6-bit packed `scales` [same scheme as Q4_K], `qh[32]` high-bit plane, `qs[128]` low nibbles, compile-checked); quantize reuses `makeQkx2Quants` with q5_K's parameters (nmax=31, rmin=-0.5, nstep=15); dequant: `y = d·sc·((qs nibble) + (qh bit?16:0)) − dmin·m`. Two subtle reference behaviors caught by tests: (1) the `qh` pointer NEVER advances in the C code — all 4 chunks of 64 weights reuse the same 32 qh bytes, separated by shifting bit masks (m1=1,m2=2, <<2 per chunk); (2) the output pointer `y` advances every chunk (×64), unlike the block-strided intuition. 5 tests in test_kernels.zig (176-byte layout offsets, round-trip rel err < 12%, all-zero input → exact zero, qh high-bit plane exercised by wide-range data, GEMM vs dequantized reference <1% quant error). Wired into root.zig; emission verified via nm. Remaining: Q8_K (DONE — see the Q8_K entry below).
- [x] **GGML C-API wrappers — Phase 2 DONE (2026-08-31)** — 12 new exports in `src/main.zig` + 12 declarations in `include/kt_kernel.h`: `kt_quantize/dequantize_q{8_0,4_k,5_k,6_k}` (one-row F32↔block conversions) and `kt_matmul_q{8_0,4_k,5_k,6_k}` (BF16 activations × quantized weights → F32, ldb in BLOCKS). Since the block layouts are byte-exact vs GGUF, Python can pass weights loaded from .gguf files straight into the matmuls. `verify_abi.py` now gates **65 symbols × 7 variants, full PASS**. 7 C-API lifecycle tests in `tests/kernels/ggml_capi_tests.zig` (fifth suite, rooted at main.zig): per-format round trips through the real exports + constant-weight matmuls. main.zig GGML/amx aliases marked `pub` for test access (no ABI effect). **GGML workstream COMPLETE for the 4 workhorse formats** (Q8_0, Q4_K, Q5_K, Q6_K); Q8_K landed after (see next entries).
- [x] **GGML Q8_K kernel — DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q8_k.zig`: byte-exact `BlockQ8_K` (292 bytes: **f32** `d` [not f16!], 256×i8 `qs`, 16×i16 `bsums` — the dot-product sidecar, compile-checked). The simplest K-quant math: signed `iscale = -127/max` (max carries sign — the IQ2_XXS-compat choice), `qs = MIN(127, nearestInt(iscale·x))` with **no lower clamp needed** (|iscale·x| ≤ 127 by construction), `bsums[j] = Σ qs` per 16-group, all-zero fast path. `kt_quantize/dequantize/matmul_q8_k` C-API + header declarations — ABI now **68 symbols**. 5 kernel tests + 2 C-API tests (292-byte layout, round-trip ≤ d/2 exact bound + rel <2%, bsums exact-group-sum invariant, zero fast path, GEMM vs dequantized reference <0.5%). **GGML workstream COMPLETE — all 5 formats.**
- [x] **GGML Q2_K kernel — DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q2_k.zig`: byte-exact `BlockQ2_K` (84 bytes: 16×u8 `scales` [4-bit scale+min, ONE byte per sub-block — unlike Q4_K/Q5_K's 6-bit scheme], 64×u8 `qs` [2-bit quants, 4/byte], f16 `d`+`dmin`, compile-checked). Quantize (:891): per-16 `makeQkx2Quants(nmax=3, rmin=-0.5, nstep=15, use_mad=TRUE)` with |x| weights (the q2_K-specific params), 4-bit scale/min packing with q4scale=15, clamp [0,3], 2-bit plane packing. Dequant (:961): per-128 chunk, 4 shift planes × 2 sub-blocks (q[l], q[l+16]) — `y = d·sc_lo·((qs>>shift)&3) − dmin·sc_hi`. Two Zig gotchas caught by tests: (1) `u3` shift loop counter wraps at 8 (6+2→0) and re-enters → scales[] OOB; use a usize plane counter and `@intCast` at the use site. (2) The honest round-trip floor for 2.625 bits/weight is rel ~0.39 on smooth data (verified against a Python port of the ggml reference) — threshold set to 45% with that evidence, not a guess. 4 tests (84-byte layout, round trip < 45% rel, all-zero, GEMM vs dequantized ref). Wired into root.zig; emission verified via nm. 111 tests total.
- [x] **GGML Q3_K kernel — DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q3_k.zig`: byte-exact `BlockQ3_K` (110 bytes: `hmask[32]` high-bit plane, `qs[64]` 2-bit quants, `scales[12]` packed 6-bit, f16 `d`, compile-checked). The most layout-intricate format: 16 int8 scales packed into 12 bytes via a 4-way nibble+2-bit crisscross; dequant unpacks via the aux[kmask1/kmask2] bit-shuffle. `make_q3_quants` (nmax=4, 5-try coordinate descent). The high bit SUBTRACTS 4 (inverting the +16 pattern of Q5_K). Three bugs caught by tests+debug: (1) my unpackScales masked aux[2] top byte (scales[11] lives in bits 24..31) — no effect on this data but latent; (2) `u3` shift counter wraps (same as Q2_K); (3) the `m` hmask-bit counter must advance across ALL 8 planes (2 chunks x 4 planes) — resetting it per chunk made chunk-1 weights read wrong hmask bits (rel 2.37 → 0.33 after fix, debug-verified). Honest threshold: 35% (measured 0.33 with the Python-port cross-check). 4 tests (110-byte layout, round trip < 35% rel, all-zero, GEMM vs dequantized ref). 115 tests total. **All 7 standard K-quant formats + Q8_0 now complete.**
- [x] **GGML IQ4_XS kernel — DONE (2026-08-31)** — `src/kernels/amx/gemm_224_iq4_xs.zig`: byte-exact `BlockIQ4_XS` (136 bytes: f16 `d`, u16 `scales_h` [8×2-bit highs crisscrossed], `scales_l[4]` [8×4-bit lows, 2/byte], `qs[128]` [4-bit indices], compile-checked). **The first non-linear format**: quant values are INDICES into `kvalues_iq4nl` {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113} looked up via `bestIndexInt8` (binary search). Quantizer: shared `quantizeRowIQ4NLImpl` (per-32 weighted least-squares with ntry=7 refinement + the d=-max_scale/32 negative-scale convention). Honest threshold: 25% (measured 0.21 — the non-linear table + 6-bit per-32 scales have their own noise floor; sub-block scales come out tiny on smooth sinusoids, debug-verified). 5 tests (136-byte layout, kvalues table, round trip < 25% rel, all-zero, GEMM vs dequantized ref). 120 tests total. **First IQ-family format landed.**
- [x] **GGML IQ2_XXS kernel — DONE (2026-09-01)** — `src/kernels/amx/gemm_224_iq2_xxs.zig`: byte-exact `BlockIQ2_XXS` (66 bytes: f16 `d` + `qs[32]u16`, compile-checked). **Grid-based 2.0625 bpw**: each u16 encodes a grid-index byte + 7-bit sign-selector; per-32 group a 4-bit db factor in the top nibble of `aux32[1]`. Three byte-exact lookup tables transcribed: `IQ2XXS_GRID` (256×u64, 8 magnitude bytes each from the {0x08,0x19,0x2b} alphabet), `KSIGNS_IQ2XS` (128×u8), `KMASK_IQ2XS` (8×u8). Quantize: **stubbed with @panic** (requires runtime kmap/kneighbors init via `iq2xs_init_impl` — separate workstream, documented). Dequant + scalar GEMM: real, byte-exact vs ggml-quants.c:2488. LE byte-order gotcha caught by the hand-crafted-value test: grid64 byte 0 = LSB (= 0x2b for grid[1]), not the printed hex MSB (= 0x08). 5 tests (66-byte layout, 3 lookup tables vs ggml, zero block, hand-crafted known values, GEMM with 2 weight columns). 125 tests total.
- [x] **GGML IQ3_XXS kernel — DONE (2026-09-01)** — `src/kernels/amx/gemm_224_iq3_xxs.zig`: byte-exact `BlockIQ3_XXS` (98 bytes: f16 `d` + `qs[96]` [64 grid indices + 32 scales+signs], compile-checked). **Grid-based 3.0625 bpw**: per 32-weight group, 4 sub-blocks of 8; each sub-block reads TWO u32 grid entries (4 magnitude bytes each, alphabet {0x04..0x3e}); db = d·(0.5+(aux32>>28))·0.5 from the top nibble of the group's scale area; sign palette shared with IQ2_XXS (KSIGNS_IQ2XS). 256×u32 `IQ3XXS_GRID` table byte-exact from ggml-common.h. Quantize: **DONE (2026-09-01, follow-up commit) — the @panic stub was replaced** by the wrapper in `gemm_224_iq3_xxs.zig:quantizeRowIQ3_XXS` (lazy-init global kmap + mutex) delegating to `iq3_quantize.quantizeRowIQ3_XXS_WithInit` (see the IQ3 quantize entry below). Dequant + scalar GEMM: real, byte-exact vs ggml-quants.c:2575. 5 tests (98-byte layout, grid tables vs ggml, zero block, hand-crafted known values, GEMM 2-column) + 3 new (kmap exact+correspondence, off-grid validity, quantize round-trip).
- [x] **IQ2_XXS kmap/kneighbors init — DONE (2026-09-01)** — `src/kernels/amx/iq2xs_init.zig`: port of `iq2xs_init_impl` (ggml-quants.c:2853-3255). `KGRID_2BIT_256` (256×u16 packed 2-bit fingerprints) byte-exact from the reference. `initIq2XsData`: builds the 8-byte position grid (pos[k]=2·nibble+1), the exact-match kmap (all 256 grid entries map to themselves — verified), and the kneighbors table for the 43436 off-grid fingerprints (2-pass: count + fill, nwant=2 nearest via Hamming distance, sorted). 2 tests verify: (a) all 256 exact matches land correctly; (b) all 43436 off-grid entries have valid negative-encoded offsets pointing into kneighbors with count>0. This unblocks the full IQ2_XXS quantize (the @panic stub). 132 tests total. **IQ2_XXS quantize now unblocked — the remaining work is the nearest-grid search loop (quantize_row_iq2_xxs_impl) using these tables.** **IQ2_XXS full quantize — DONE (2026-09-01)**: `src/kernels/amx/iq2_quantize.zig` — port of `quantize_row_iq2_xxs_impl` (ggml-quants.c:3294-3465): `makeQPQuants` (weighted LS + 5-try coordinate descent), `findBestNeighbour` (on/off-grid refinement), sign extraction with odd-parity flip, ±6-scale grid-search refinement via the kmap, d=max_scale/31 global scale, 4-bit per-32 scale in the top nibble. Constant-0.5 debug verification confirms correct algorithm (values map to the {8,25,43} magnitude alphabet at the right scale; dst[1]=0.4686 ≈ 25·db). Sinusoid test threshold is 150% rel — algorithm-correctness gate, not precision (the reference always uses importance weights; without them, low-|x| values get poorly quantized BY DESIGN). 133 tests. **IQ2_XXS fully complete: quantize + dequantize + GEMM + kmap init.** **All standard GGML quant formats now complete** (7 linear K-quants + Q8_0, IQ4_XS, IQ2_XXS, IQ3_XXS = 10 formats). **IQ3_XXS full quantize — DONE (2026-09-01)**: `src/kernels/amx/iq3xs_init.zig` — port of `iq3xs_init_impl` (ggml-quants.c:3703-3902, grid_size=256): `KGRID_Q3XS_256` byte-exact, kmap **4096** entries (4× 3-bit nibbles, not the 43692 of IQ2's 4×2-bit), positions 4 bytes (odd 1..15), kneighbors nwant=2, 2-pass count+fill. `src/kernels/amx/iq3_quantize.zig` — port of `quantize_row_iq3_xxs_impl` (:3938-4150): **NO makeQPQuants initial fit** (unlike IQ2) — the ±15-scale grid search itself produces the levels (kMaxQ=8, 3-bit), sigma2 = **2·**sumx2/QK_K (differs from IQ2), GROUP_MAX_EPS = 1e-8, off-grid redo pass at the final scale, d = max_scale/31 × **1.0125 fudge factor** (:4137), 7-bit sign packing at strides {0,7,14,21}, 4-bit per-32 scale in the top nibble. **Grid-index correspondence verified with evidence** (Python cross-check vs the C tables): IQ3XXS_GRID[gi] bytes = ALPHABET[nibble(grid[gi])] bijection {0..7}→{4,12,20,28,36,44,52,62}, and both Zig tables byte-exact vs the C — so a kmap grid_index is a valid IQ3XXS_GRID dequant index (round-trip correct by construction). Verification: constant-0.5 → mean 0.4917, max err 0.139 (3-bit grid expected); sinusoid → RMS rel err 20.8% (null importance weights, same rationale as IQ2). 140 tests total. **IQ3_XXS fully complete: quantize + dequantize + GEMM + kmap init — the quantize-with-kmap recipe now proven twice (IQ2_XXS, IQ3_XXS).** **IQ2_XS — DONE (2026-09-02, c6f41e6)**: `src/kernels/amx/gemm_224_iq2_xs.zig` + `iq2xs_init.zig` generalized (initIq2GridData core + KGRID_2BIT_512 byte-exact + initIq2XsData512, nwant=2). 2.3125 bpw, 74-byte block: **9-bit** 512-grid index | signs<<9 in u16 qs + **separate** u8 scales[8] (4-bit dual-nibble per 32 weights — l=0,1→lo nibble, l=2,3→hi). db = d·(0.5+nibble)·**0.25** (XXS uses 0.5). Quantize (:3472-3650): 16-weight groups, ±9-scale search, off-grid redo, d=max_scale/31 (no fudge). Round-trip evidence: const-0.5 → max err 0.037 (XXS: 0.149 — the denser grid measurably finer); sinusoid RMS rel 0.367 (XXS 0.42), 150% max-rel bound (null importance weights, same rationale). 5 tests. Tables byte-exact vs C (Python cross-check).  **IQ2_S — DONE (2026-09-02, ede64ea)**: `src/kernels/amx/gemm_224_iq2_s.zig` + `iq2xs_init.zig` KGRID_2BIT_1024 + initIq2SData (**nwant=1** — the only 2-bit format refining off-grid to the single nearest neighbor). 82-byte block: **10-bit** grid index split qs-lo(8)+qh-hi(2, 4/byte @2-bit stride); **RAW 8-bit sign bytes** (no KSIGNS palette, no parity trick — the S format spends its extra bits buying back the sign bit); quantize: sigma2=2·sumx2/QK_K, null-weight fallback **0.25·sigma2+x²**, no initial fit, d=max_scale/31·**0.9875**. 6 tests green on FIRST run (incl. qh-hi-bits selecting grid [256] — the 10-bit split proof).  **IQ3_S — DONE (2026-09-02, a543876)**: `src/kernels/amx/gemm_224_iq3_s.zig` + `iq3xs_init.zig` 3-bit core generalized + KGRID_Q3XS_512 + initIq3SData (**nwant=3**, ggml:3762). 110-byte block: **9-bit** index qs-lo+qh-hi (1 bit/sub-block-of-4, 8/byte @slot%8), **RAW signs** [32]u8, scales[4] own array with the NEW scheme **db = d·(1+2·nibble)** (2-bit family: 0.25·(0.5+n)). Quantize (:4169-4350): 8×4-weight sub-blocks/group, is_on_grid starts **FALSE** + redo pass runs for ALL sub-blocks (ggml:4283 commented-out `continue`), d=max_scale/31·**1.033**. 5 tests green on FIRST run; const-0.5 probe → mean 0.5038 (finest grid format).  **IQ1_S — DONE (2026-09-02, 127bbef)**: `src/kernels/amx/gemm_224_iq1_s.zig` + `iq2xs_init.zig` KGRID_1BIT_2048 + initIq1SData (**nwant=3**, same 43692 kmap — nibble {0,1,2}→byte {ff(-1),00(0),01(+1)}). 50-byte block: qs[32] lo + qh[8]u16 packing 4×3-bit index-his(12b) + **3-bit scale**(bits 12-14) + **delta-sign bit**(15) — no separate scales array. Dequant (:2651): dl=d·(2n+1), delta=±0.125, y=dl·(grid[j]+delta) — the delta shifts the 1-bit code to effective {−1.125, 0.125, 1.125}. Quantize (:4508-4670) **EXACT — no grid search**: qsort (value,idx) pairs + prefix sums + exhaustive (i1,i2) split search against BOTH delta shifts; **d=max_scale/15** (3-bit scale, not 31) × 1.125 fudge ('Don\'t ask' — ggml:4661). findBestNeighbour2 with the **exhaustive 2048-entry fallback** (empty-neighbor fingerprints). **Round-trip test caught a real bug on first run**: the refine pass must read the runtime POSITION grid ({1,3,5}) not the dequant grid ({−1,0,1}) — (pg−1)/2 over −1 overflows the xg index. Gotchas: 'i8' AND 'i1' shadow primitives. 6 tests; const-0.5 → mean 0.5624 (format floor — 0.5 is unreachable at 3 levels).  **IQ1_M — DONE (2026-09-02, d0b0211) — ALL 15 GGML FORMATS COMPLETE**: `src/kernels/amx/gemm_224_iq1_m.zig`. 56-byte block: qs[32]+qh[16]+scales[8], **NO d field** — the f16 scale hides in the top nibbles of sc[0..3] (extractScale reconstruction). Delta flips INDEPENDENTLY per group-half (4 mask variants 0x00/0x80/0x08/0x88); quantize = SSD split search against all 4 delta combos simultaneously + negative-scale k-involution (0↔3, 1↔2) + **GLOBAL d refit** over quantized 3-bit scales + 1.1125 fudge. Alignment lesson: extern struct of u8 arrays = natural alignment 1 — the C's u16* scales access is silent UB; Zig panics; all access byte-composed (sc16 helper). **Plus the kmap-init 22x speedup** (d0b0211): qsort-then-scan replaced by bounded HISTOGRAM (d2 ≤ 8·36=288) single-pass — same neighbor set + same order; iq1s 130.6s→6.0s, iq2s 51.7s→3.0s. Process-wide kmap CACHE (mutex + page_allocator, *Owned variants for old semantics). Fixed a latent landmine: gemm_224_iq3_xxs.zig:241 used nonexistent std.Thread.Mutex (never compiled — lazy analysis hid it; std.atomic.Mutex is the 0.16 name). 134 kernels tests, 186 OK/0 leaks total. **THE GGML FORMAT SET IS COMPLETE: 15/15** (Q8_0 + 7 K-quants + IQ4_XS + IQ4_NL + IQ2_XXS/XS/S + IQ3_XXS/S + IQ1_S/M) — every kt_type_t enum value now has a real kernel with quantize+dequantize+GEMM+tests.
- [x] **DeepseekV3DecoderLayer — model-orchestration layer DONE (2026-08-31)** — `src/kernels/moe/deepseekv3_layer.zig`: the Zig-native port of the Python reference layer forward (kt-kernel/examples/modeling_deepseek_v3.py:1196-1226 + MoEGate:394-461). Single-sequence orchestration wiring the existing kernels: residual → RMSNorm(attn_norm_w) → MlaEngine.forward (continuous KV, kv_start_pos param) → residual add → RMSNorm(ffn_norm_w) → routeExpertsDeepSeek (sigmoid + e_score_correction_bias + group-top2 noaux_tc + norm_topk_prob + routed_scaling_factor) → TpMoe.forward (work-stealing when pool configured) → residual add → BF16 out. Owned contexts: MlaEngine + MlaKvCache + TpMoe created in init, freed in deinit; weight pointers borrowed (caller-owned, standard C-API convention); per-call scratch allocated symmetric. NOTE: the C++ reference's operators/kml/deepseekv3.hpp does NOT exist in the checkout (ARM/KML-only include, dir absent) — the Python modeling file was the real spec. Fixed a stray root.zig import left by the parallel session (operators/deepseekv3_layer.zig — nonexistent path). Test: 2-step decode (kv_start_pos 0 then 1) with zero weights asserting exact residual semantics (output == input through BF16), 0 leaks. **Model-orchestration: DecoderLayer + C-API + pybind11 done.** C-API: kt_dsv3_layer_new/forward/free + kt_dsv3_layer_config_t extern struct (34 fields, config by POINTER — pointer-passing, not by-value; ABI 87 -> 90 symbols, double-gated). New suite tests/kernels/dsv3_layer_capi_tests.zig (2 decode steps through the extern-struct boundary, exact residual semantics). pybind shim: m.dsv3.DeepseekV3DecoderLayer + LayerConfig (size_t pointer fields with per-field conversion; safe defaults — rope_theta=10000 because 0.0 produces NaN in RoPE, found live via pybind e2e). **DeepseekV3Model + ForCausalLM DONE (follow-up commit)** — src/kernels/moe/deepseekv3_model.zig: Model = N × DeepseekV3DecoderLayer (ping-pong hidden buffers) + final RMSNorm (inline f32 helper); ForCausalLM = Model + lm_head GEMM (gemmExpert, [qlen,hidden]×[vocab,hidden]^T → f32 logits). C-API: kt_dsv3_model_new/forward/free + kt_dsv3_causallm_new/forward/free + kt_dsv3_model_config_t (embeds kt_dsv3_layer_config_t as per-layer template + num_layers/final_norm/lm_head/vocab_size). ABI 90 → 96 symbols, double-gated. 2 kernel tests (2-layer residual chain + final-norm=1 → exact 1.0; lm_head weights=1 → logits=64 exact), 106 total.
- [x] **Python ctypes GGML bindings — DONE (2026-08-31)** — `python/kt_kernel/__init__.py`: block-size constants (`Q8_0_BLOCK_BYTES`=34 … `Q8_K_BLOCK_BYTES`=292), raw bindings for all 15 `kt_quantize/dequantize/matmul_q{8_0,4_k,5_k,6_k,8_k}` exports, plus convenience API: `quantize_row(fmt, src)` (list→packed block bytes), `dequantize_row(fmt, blocks, k)` (bytes→c_float array, with expected-size validation), `matmul_quantized(fmt, a_f32, b_blocks, m, n, k)` (auto f32→BF16 activation conversion via kt_f32_to_bf16). Footgun guards: ValueError on non-multiple-of-QK lengths, unknown formats, and buffer-size mismatches. Verified end-to-end: all 5 formats round-trip within kernel tolerances, q8_0 GEMM ≈32.0 exact, error paths raise. Bundled variant .so files refreshed to 68-symbol builds (the stale bundle lacked the GGML symbols — found via the import failing with undefined symbol).
- [x] **ABI signature parity audit — Tier 0 DONE (2026-08-31)** — full 86-symbol sweep of header prototypes vs Zig export signatures. Found and fixed 3 real divergences (stack-corruption risk for header-following callers): `kt_mla_new` (dropped the extra cpuinfer param; header contract is 1 arg), `kt_mla_forward` (rewritten to the 8-arg paged/batched C contract: qlens/page_tables/kv_lens arrays; qlen_count==1 fully supported, >1 panics with a clear message — paged-attention indirection is future work; page_tables accepted-but-unused documented), `kt_gate_forward` (header updated to the 7-arg Zig form with topk_ids/topk_weights outputs — the ctypes wrapper + tests already used it; 4-arg C++ minimum noted as not exported). Also gated 18 previously-unheadered .so exports (4 math helpers kt_apply_swiglu/rms_norm/rope + kt_softmax, MLA prefill/decode/update_kv_cache conveniences, MoE forward_gate_up/down, FP8 quantize helpers, scalar matmuls bf16/int8/int4/fp8, worker_pool_get_thread_num) in a clearly-marked 'Zig extensions' header section — ABI: 68 → 86 symbols. **New gates**: `tools/audit_arity.py` (header/Zig arity parser with fn-ptr + multi-line + trailing-comma handling) + wired into `verify_abi.py` as a second gate (FAIL on any name or arity divergence). Side effect: the audit surfaced two latent compile errors in `kt_matmul_int4` (b_scratch pointer-type mismatch, applyScales b/c types) — fixed; INT4 AMX-path tests now actually run (previously silently un-analyzed under lazy compilation). Python wrapper: 18 new raw bindings + argtypes, 61 names in __all__.
- [x] **pybind11 drop-in wrapper — Tier 1 DONE (2026-08-31)** — `bindings/kt_kernel_pybind.cpp` + `bindings/build.sh`: a pybind11 module named `kt_kernel_ext` (matching the C++ reference's PYBIND11_MODULE name, so `from kt_kernel_ext.moe import MOE, MOEConfig` works unchanged) that links against the Zig `.so` and routes every method through the C API. Surface: `moe.MOE` (construct-from-config, load_weights, warm_up, forward, forward_sft), `moe.MOEConfig` (full field set incl. embedded quant_config), `mla.MLA` + `mla.MLAConfig` (paged forward, qlen_count==1), `kvcache.ggml_type` enum, `WorkerPool`/`CPUInfer` (submit is synchronous; the Zig work-stealing pool is reached via the MoE pool config), `kt_version`/`kt_get_cpu_variant`. **The hard part was the ABI**: config structs are passed BY VALUE (~340/272 bytes) — a C++ mirror struct with wrong field order silently corrupts the callee (found live: the shim originally missed `num_gpu_experts`/`gpu_experts_mask`/`physical_to_logical_map` + embedded `quant_config`, and typed `kt_type_t` enums as u8 instead of c_int). Three-layer fix: (1) shim structs transcribed field-for-field from the Zig `extern struct`s; (2) new Zig ABI probes `kt_abi_size_*`/`kt_abi_field_offset` (via `@offsetOf`, test-only exports); (3) new gate `tools/audit_layout.py` — loads the probes from the .so, computes the C++ shim's Itanium-ABI offsets, FAILs on any sizeof/offset divergence. Layout gate caught the MLA u8-vs-int enum bug live. End-to-end verified: MOE construct→load_weights→forward (zero weights→zero output exact) and MLA construct→load_weights→forward through the real pybind11 path with numpy buffers. Third Python surface alongside the ctypes wrapper; build via bindings/build.sh (pybind11 from torch include).

- [x] **LlamaMoe (Zig extension) — DONE + RECONCILED (2026-09-03)** — ports
  the C++ `LLAMA_MOE_TP` from `ktransformers/kt-kernel/operators/llamafile/moe.hpp`
  (the framework's default backend for GGUF-loaded checkpoints). The
  Zig class (`src/kernels/moe/llamafile_moe.zig`) implements the
  full per-expert algorithm: BF16 input → Q8_0 quant → SwiGLU
  → per-format GEMM × Q8_0 activations → F32 weighted sum.
  **Reconciliation (5177bdf)**: two agents ported this in parallel
  (llama_moe.zig vs llamafile_moe.zig); unified into llamafile_moe.zig
  — the 16-format `gemmQuant` dispatch + gpu_experts_mask merged into
  the C-API-first base, duplicate deleted. **Supported weight types:
  ALL 16 GGML formats** (Q8_0, Q*_K, IQ2/IQ3/IQ4/IQ1 families) —
  each projection may differ. C ABI:
  `kt_llama_moe_{new,free,load_weights,forward}` (4 symbols) +
  `kt_llama_moe_config_t` extern struct. ABI: 105 → 112 symbols,
  verify_abi.py PASS on all 8 variants. The matmul path dequantizes
  Q8_0 → BF16 and calls the per-format GEMM kernels (byte-exact vs
  llama.cpp, but does an extra dequant/quant round trip on the
  activation). Tests: `tests/kernels/llamafile_moe_capi_tests.zig`
  (lifecycle, pool-null rejection, KT_TYPE ABI audit) +
  `test_kernels.zig` (Q8_0/Q4_K constant-value dispatch, 16-format
  zero-weight dispatch sweep). **P0 bug fixed (eec0c22)**: the
  16-format extension originally shipped 11 invented KT_TYPE values
  (e.g. IQ3_S=29 vs header 25 — collisions dispatched the WRONG
  kernel silently) + 9 wrong block byte-sizes; guarded now by a
  comptime `@sizeOf` audit, a KT_TYPE-vs-enum ABI test, and the
  16-arm dispatch test. **Limitations (follow-ups)**: (1) true fused
  q×q matmuls to skip the BF16 round-trip; (2) `forward_many`
  (batched per-m-block work-stealing) — the C ABI calls per-token
  `forward_one` for qlen > 1 (correct, not optimal); (3) TP path is
  `tp_part_idx=0` only. **Closes the §1 gap** from the "qué falta" analysis.

### Documentation
- [x] **API documentation** — `docs/api.md` (hand-written; toolchain is Zig 0.16.0-dev.2535 which has **no `zig doc` subcommand** and no `-femit-docs` flag, so the doc is not auto-generated). Integrator's quick-reference: family map + per-family signature blocks covering **ALL 112 kt_* symbols** (verified: every header prototype is mentioned; `comm` audit 2026-09-03), pointers + minimal Python ctypes example, ABI contract (header is the source of truth; `tools/verify_abi.py` is the gate; multi-variant layout), known gaps (the 10 non-linear GGML formats have no per-type C exports BY DESIGN — reachable via kt_llama_moe_* or the Zig API; documented as such). The 4 math-helper exports (`kt_apply_swiglu`/`kt_apply_rms_norm`/`kt_apply_rope`/`kt_softmax`) were headered + ABI-gated in `ff57125` and are now documented in api.md's Math helpers section with the family-map row.
- [x] **Porting guide from C++** — `docs/porting-guide.md` (181 lines, committed in `e2c900d`): C++ reference file map → Zig counterpart map, lazy-analysis wiring pattern, comptime fn-ref pattern for `.so` symbol emission, AMX arch-gate, the bf16 f16 cast (Zig 0.16 native f16 vs hand-rolled bit math), and the `@Vector` indexing gotchas. Linked from AGENTS.md.

---

## 📊 Progress Tracking
| Component | Status | % |
|-----------|--------|---|
| Runtime (pool/queue/memory/cpu) | Work-stealing pool + sched_setaffinity pinning (A3) + waitIdle sync (B3) + mbind NumaAllocator + L1/L2/L3 detection (A4); MoE + SFT paths parallel | 100% |
| AMX Intrinsics | Real inline asm + runtime detectAmxSupport guard (A2 fix: comptime featureSetHas + CPUID + arch_prctl) | 95% |
| Buffer Packing | Working | 80% |
| BF16 GEMM | gemmExpert vectorized (@Vector(8,f32), A1) — 5.2x decode / 5.3x prefill measured (B2); AMX tile path verified | 85% |
| INT8 GEMM | Working | 70% |
| INT4/FP8/MXFP4/8 GEMM | INT4, FP8, MXFP4, MXFP8 all done (AMX + scalar fallback) | 100% |
| GGML Quant | All 5 formats (Q8_0/Q4_K/Q5_K/Q6_K/Q8_K) byte-exact + C-API matmuls + Python bindings | 100% |
| MoE Orchestration | Forward + work-stealing parallel + DeepSeek-V3 group-top2 routing (D4); D1/D2/D3 buffer bugs fixed with qlen>1/tp>1 regression tests | 100% |
| SFT/LoRA Training | Forward + backward complete + work-stealing parallel forward; 4 C API exports | 100% |
| C API | 87/87 header symbols exported by all variants; TRIPLE GATE: exports + arity (verify_abi.py) + layout (audit_layout.py); injectable allocator (B1) | 100% |
| Build System | Multi-variant (6 x86 variants + aarch64 neon, distinct .so names) + bench step (B2) | 90% |
| Tests | 68 kernels + 11 MLA + 2 FP8 + 1 MLA C API + 9 GGML C API + 7 aarch64 + 1 neon + 2 allocator C API = 101 total pass, 0 leaks, exit 0 | 100% |
| Python Integration | ctypes wrapper (61+ names incl. GGML) + pybind11 drop-in (4040b5b, layout-gated) + wheel + CI; kt_set_default_allocator binding pending | 85% |

---

## Notes

- **Reference**: Original C++ in `/ai/repos/2026/ktransformers/ktransformers/kt-kernel/`
- **Zig Version**: 0.16.0-dev.2535+b5bd49460
- **Target**: Drop-in replacement for `kt_kernel_ext.so` (6 CPU variants)
- **Python Compat**: Match `kt_kernel_ext.so` C API exactly for pybind11
- **Key Files Modified**:
  - `build.zig`: Uses `src/root.zig` as root, test step enabled
  - `src/root.zig`: Re-exports all submodules
  - `src/main.zig`: Cleaned up forward references
  - `src/kernels/amx/buffers.zig`: Fixed @intFromFloat
  - `src/kernels/arch/amx.zig`: Fixed packed struct, AMX feature check
  - `src/kernels/moe/moe.zig`: Fixed indentation, pointer optionals
  - `src/runtime/cpu_detect.zig`: Fixed || → or
  - `src/runtime/memory.zig`: Fixed alignedAlloc API
  - `src/runtime/worker_pool.zig`: Fixed atomic, DoublyLinkedList, Thread.Pool
  - `tests/kernels/test_kernels.zig`: Fixed import path, type casts

---

## ✅ Additional Fixes (2026-08-29 - Session 2)

### Build Error Resolution
- [x] **Fixed all 31 build errors** in Zig 0.16
- [x] **Fixed 6 main.zig errors** (parentheses, pointer types, opaque types)
- [x] **Fixed 18 moe.zig errors** (type casting, pointer arithmetic, @memset)
- [x] **Fixed 4 gemm_224_int8 errors** (pointer casting, setTile, quantization)
- [x] **Fixed 2 worker_pool.zig errors** (std.Io.Mutex changes)
- [x] **Added 5 missing TpMoe methods** (deinit, warmUp, loadWeightsWithMap, forwardGateUp, forwardDown)
- [x] **Fixed @memset API** (now 2-arg version in Zig 0.16)
- [x] **Fixed all setTile calls** with proper u8/u16 casting
- [x] **Fixed pointer arithmetic** throughout kernels with @ptrCast patterns
- [x] **Fixed f32↔int conversions** (@intFromFloat, @floatFromInt)

### Documentation
- [x] **Updated LESSONS_ZIG.md** with 30+ Zig 0.16 lessons learned
- [x] **Added build success summary** to LESSONS_ZIG.md

## Reference Repos

  The reference repo are is at /ai/repos/2026/ktransformers


### Current Status (2026-08-31)
- **Dev A (2026-08-30)**: Linear/MLP C API wired; kt_*_config_t structs
  synced with C header; MoE gemmExpert weight_ld OOB fixed (6 sites);
  multi-variant build (`zig build all-variants` → 6 .so); minimal Python
  ctypes wrapper; kt_get_cpu_variant lazy-init fixed; SFT/LoRA forward half
  (LoRA kernels, TpMoeSft, forward_sft, C API); runtime hardening (work-stealing
  worker pool, NUMA topology); allocator safety fix (TpMoe stores allocator).
- **Dev B (2026-08-30)**: MXFP4/MXFP8 kernels complete; vectorized applySwiGLU
  complete (bit-exact vs scalar); SFT/LoRA backward half complete (backward
  method, kt_moe_backward export).
- **Audit session (2026-08-31)**: external-dev feedback verified against the
  code (IMPROVE.md) and every real item fixed: P0 guard AMX + MoE buffer bugs
  (ff57125), gemmExpert vectorization A1 (7515468, 5.2x measured), NUMA
  pinning A3 (c35a526), DeepSeek-V3 routing D4 (f5443f3), benchmark suite B2
  (7d2db9c), cache detection A4 + cpuinfer_sync B3 (ca0485a), injectable
  allocator B1 (15a8ea5). ABI 86→87 symbols; tests 90→101; all P0-P3 closed.
- **Last fully-green**: 101 tests pass, 0 leaks, `zig build test` exit 0;
  `verify_abi.py` 87/87 + arity PASS on all variants; `zig build bench` green.
