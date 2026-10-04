# AGENTS.md - ktransformers-zig

Guidance for AI agents working on this repo. **Read `LESSONS_ZIG.md` before
writing any Zig code** — this toolchain (0.16.0-dev) differs significantly
from what you likely remember, and the lessons file contains dozens of
verified gotchas.

## Project Context

Port of ktransformers CPU kernels (C++/AMX/AVX2/AVX512) to Zig, as a drop-in
replacement for `kt_kernel_ext.so` (C API in `include/kt_kernel.h`, loaded
from Python via pybind11).

- **Zig version**: 0.16.0-dev.2535+ (APIs verified against this build; see LESSONS_ZIG.md)
- **C++ reference**: `/ai/repos/2026/ktransformers/ktransformers/kt-kernel/`
  - `operators/amx/la/amx_kernels.hpp` — GemmKernel224 BF16/INT8/INT4
  - `operators/amx/moe_base.hpp`, `operators/amx/moe.hpp` — AMX MoE
  - `operators/moe-tp.hpp` — TP_MOE_Common (monolithic forward)
  - `operators/llamafile/mla.hpp`, `operators/mla-tp.hpp` — MLA attention
  - `operators/rope.hpp` — RoPE/YaRN reference
  - `cpu_backend/worker_pool.h`, `ext_bindings.cpp`
- **External kernels** (optional, from `/home/t0m4s/repos/2026/zig-ai`):
  `src/moe/cpu_gemv.zig` (quantized GEMV), `src/moe/cpu_executor.zig` — add to
  `build.zig.zon` when needed.

## Build & Test

```bash
zig build                 # default AVX2 -> zig-out/lib/libkt_kernel_ext.so
zig build -Dvariant=amx   # avx2|avx512_base|avx512_vnni|avx512_vbmi|avx512_bf16|amx|neon
zig build all-variants    # all 6 x86 variants as libkt_kernel_ext_{variant}.so
zig build test            # runs ALL suites — tests EXECUTE via addRunArtifact
zig build test-aarch64    # cross-builds the ARM suites and RUNS them under
                          # qemu-aarch64-static (needs qemu-user-static)
zig test src/mla/mla_tests.zig             # standalone MLA suite
zig build -Doptimize=ReleaseFast bench     # GEMM micro-benchmark (B2)
python3 tools/verify_abi.py                 # ABI double gate: exports + arity
python3 tools/audit_layout.py               # ABI layout gate (pybind11 shim)
sentrux check .                            # architecture rules gate (see below)
```

Caveats (verified the hard way — see LESSONS_ZIG.md for details):
- `zig build` installs the DEFAULT variant as `libkt_kernel_ext.so`; `all-variants` installs per-variant names.
- Tests only run because build.zig uses `addRunArtifact`; `test_step.dependOn(&test_obj.step)` alone only compiles.
- The test runner is `tools/test_runner.zig` in simple mode — the default `--listen=-` IPC handshake fails in this Zig 0.16.0-dev environment.
- If a build error survives a no-op rebuild, suspect stale `.zig-cache`: `rm -rf .zig-cache zig-out`.
- Benchmark timings in Debug are meaningless — always `-Doptimize=ReleaseFast`.
- `verify_abi.py` is the ABI gate: it FAILs if any header prototype lacks a matching Zig export (or arity diverges). Run it after touching `include/kt_kernel.h` or any `export fn`.

## Code Structure

Imported (reachable from `src/root.zig`, compiled into the .so):
```
src/
├── root.zig                    # module re-exports + comptime fn-refs forcing analysis
├── main.zig                    # C API exports (kt_*) — mirrors include/kt_kernel.h
├── runtime/{memory,worker_pool,task_queue,cpu_detect}.zig
│     # worker_pool: work-stealing + sched_setaffinity pinning (A3) + waitIdle (B3)
│     # cpu_detect: L1/L2/L3 sysfs detection + selectTileParams (A4)
│     # memory: NumaAllocator with real mbind (A3)
├── numa/                       # NUMA syscalls; select helpers forced into the .so
│                               # via root.zig comptime fn-refs (setThreadAffinity,
│                               # bindMemory, NumaAllocator, ...). NOTE: some unused
│                               # helpers still have Zig 0.16 API rot (std.fs.cwd,
│                               # std.mem.page_size, alignedAlloc signature) — they
│                               # only compile once referenced; fix on first use.
├── kernels/
│   ├── arch/amx.zig            # AMX inline asm + detectAmxSupport runtime guard (A2)
│   │                           # + VecF32 helpers (swigluVec, reduceAddFp32)
│   ├── arch/neon.zig           # ARM NEON comptime gate (aarch64 cross-build)
│   ├── amx/{buffers,kquant_simd,gemm_224_{bf16,int8,int4,fp8,q8_0,q4_k,q5_k,q6_k,q8_k,
│          mxfp4,mxfp8,q2_k,q3_k,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_xs,iq4_nl,
│          iq1_s,iq1_m}}.zig
│   │                          # + iq2xs_init/iq3xs_init (kmap/kneighbors, histogram+cached)
│   └── moe/{moe,llamafile_moe}.zig
│                               # TpMoe: routing (legacy + DeepSeek-V3 group-top2, D4),
│                               # loadWeights (all 3 projections), vectorized gemmExpert (A1)
│                               # LlamaMoe: GGML-quantized MoE for GGUF checkpoints —
│                               # gemmQuant 16-format dispatch (comptime @sizeOf-audited;
│                               # KT_TYPE values ABI-test-pinned vs main.zig's enum)
└── mla/                        # MLA attention (DeepSeek-V2/V3): config/cache/core

bench/gemm_bench.zig            # B2: gemmExpert vs scalar ref at DeepSeek-V3 shapes
bindings/kt_kernel_pybind.cpp   # Tier-1 pybind11 drop-in (module name kt_kernel_ext)
tools/{verify_abi,audit_arity,audit_layout}.py  # ABI gates (export + arity + layout)
```

NOT imported (dead code — excluded by Zig's lazy analysis; don't assume they compile):
`kernels/amx/gemm_224_{bf16,int8,int4,fp8}_avx512.zig` (also imports a
nonexistent `arch/amx_intrinsics.zig`). Also: `main.zig.backup*` are
stale snapshots. (MXFP4/MXFP8 ARE wired into root.zig now.)

## Status (see TODO.md for details)

- Build: working; 6 x86 variants **and** the aarch64 neon cross-build
  (`zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu`), which is now
  green. It was broken from `2416b68` until `amx.pshufb16` got the comptime
  target guard it needed: `asm` is not lazily analysed, so its `"x"`
  constraint had no register class on aarch64 and the build died with
  "couldn't allocate output register for constraint 'x'" while all six x86
  variants and the whole test suite passed. The guard keeps the instruction on
  x86 and falls back to a lane-exact scalar lookup elsewhere, so the asm path
  is unchanged and the round-trip test covers both. **No CI job built aarch64
  at all** while this was broken — the matrix is six x86 variants and
  `zig build all-variants` is x86-only — which is how it reached `main` green.
  `09660de` added an aarch64 leg to the abi-gate job so the next one fails.
 - C API: **112 symbols** gated by `verify_abi.py` triple gate (exports +
  arity + layout via audit_layout.py). All operator families implemented:
  MoE (incl. SFT/LoRA forward+backward), MLA, Gate (DeepSeek-V3
  group-top2 routing), Linear, MLP, FP8 transport, math helpers,
  `kt_set_default_allocator` (injectable allocator, B1),
  `kt_dsv3_*` (DeepseekV3 model-orchestration),
  `kt_qwen3moe_*` (Qwen3 model-orchestration), and
  `kt_llama_moe_*` (LlamaMoe for GGUF checkpoints, all 16 GGML
  weight formats per projection), `kt_to_float`/`kt_from_float`/
  `kt_type_row_bytes` (generic block↔F32 dispatch). User docs:
  `docs/api.md` covers every symbol (112/112, comm-audited).
- GGML: **15/15 — every kt_type_t format complete** — Q8_0 + the 7
  K-quants (Q2_K..Q8_K), IQ4_XS + IQ4_NL (non-linear 4-bit),
  IQ2_XXS/IQ2_XS/IQ2_S (2-bit grid family, full kmap/kneighbors
  quantize), IQ3_XXS/IQ3_S (3-bit grid family), IQ1_S/IQ1_M (1-bit +
  delta code). Each has quantize + dequantize + scalar GEMM + tests.
  The kmap init is histogram-based (22x faster than the qsort port,
  d0b0211) and process-cached. Commits: 7d10033..d0b0211.
- MoE: `loadWeights` packs all 3 projections; `forwardGateUp`/
  `forwardDown` are real; `gemmExpert` is vectorized (`@Vector(8,f32)`
  K-loop, A1, ~5.2x on decode shapes); D1/D2/D3 buffer-overflow/index
  bugs fixed with regression tests (qlen>1, tp>1). Routing
  (routeExperts / routeExpertsDeepSeek) threads a captured
  `std.mem.Allocator` for its scratch buffers (B1 closure).
- MLA: complete in `src/mla/`; `kt_mla_*` C API fully implemented
  (new/load_weights/forward/prefill/decode/update_kv_cache/free).
  `kt_mla_forward` supports qlen_count==1 (legacy sequential) AND
  qlen_count>1 (paged/batched via per-sequence page_tables; matches
  the C++ mla-tp.hpp:84 forward(qlens, page_tables, kv_lens,
  input, output) contract exactly). MlaKvCache has save/load
  (binary file) for context resume (kvcache_load_dump.cpp mirror).
- DeepseekV3 Orchestration: `DeepseekV3DecoderLayer` +
  `DeepseekV3Model` + `DeepseekV3ForCausalLM` (Zig-native, in
  `src/kernels/moe/`) with C-API (`kt_dsv3_*`) and pybind11 shim.
- Qwen3 Orchestration: `Qwen3MoeDecoderLayer` + `Qwen3MoeModel` +
  `Qwen3MoeForCausalLM` (Zig-native, in `src/kernels/qwen3/`) with
  C-API (`kt_qwen3moe_*`, 9 symbols). RMSNorm → MHA → residual,
  RMSNorm → gate+MoE → residual (standard MHA, no MLA).
- LlamaMoe (reconciled 2026-09-03): `LlamaMoe` in
  `src/kernels/moe/llamafile_moe.zig` — GGML-quantized MoE for GGUF
  checkpoints, ALL 16 weight formats per projection via `gemmQuant`
  (three kernel signature conventions mapped per family), gpu_experts_mask,
  C-API `kt_llama_moe_*`. LESSON: the 16-format extension once shipped
  11 invented KT_TYPE values + 9 wrong block sizes (eec0c22) — now
  guarded by comptime `@sizeOf` audit + KT_TYPE ABI test + 16-arm
  dispatch test. Never hand-copy enum values; import or pin them.
- Vanilla MHA: `MhaEngine` in `src/kernels/attn/mha.zig` (matmulQKV,
  softmax, matmulO). Companion to MLA; same `*Engine.init/deinit/
  forward/decode` API shape. Standalone `matmulF32` / `rmsNormInline` /
  `softmaxInPlace` for callers that don't need a full engine.
- GGUF parser: `src/io/gguf.zig` — GGUF v3 header + tensor table. Used
  by the Qwen3 test suite to fabricate minimal GGUF blobs in memory;
  reusable for real `.gguf` file loading (future work).
- AMX: `detectAmxSupport()` runtime CPUID + arch_prctl guard (A2 fix —
  the old `@hasField` comptime check was ALWAYS false, silently
  no-op'ing every AMX intrinsic even on AMX hardware).
- NUMA: `sched_setaffinity` pinning wired into Subpool worker spawn;
  `mbind` wired into runtime NumaAllocator (A3); `src/numa/` module
  fully revived (NumaTopology.detect + allocNuma + getThreadAffinity
  work byte-exact; getThreadAffinity bug fixed in route — kernel
  returns bytes-copied, not errno).
- Tests: **268 across 18 suites, 0 leaks** (148 in the kernels suite
  incl. all 15 GGML formats + the 16-arm gemmQuant dispatch sweep; 40
  SIMD/suite-6f incl. the 6 K-quant AVX2 GEMM equivalence tests, the MXFP8,
  MXFP4, Q8_0, IQ4_NL and IQ4_XS AVX2 GEMMs, the MX block-layout / ue8m0 / E4M3 codec
  audits, and the IQ scalar-GEMM block-offset regression tests; 11
  MLA + 4+3 MLA C-API + 2 FP8 + 9 GGML C API + 7 aarch64 + NEON +
  allocator + Qwen3 + Qwen3 C-API + GGUF e2e + to/from_float + 3
  LlamaMoe C-API incl. the KT_TYPE ABI audit + K_BLOCK runtime + 6 math-helpers (swiglu 3 variants/rms_norm/rope/softmax, numerics-pinned)). Bench:
  `zig build bench` (2.8-9.3x measured speedup A1) + `moe_bench.zig`
  (end-to-end MoE forward with tile-param tuning: -20.3% on prefill
  8×4 at K=448 vs the 1792 default on this Ryzen 512K-L2 — validates
  the A4 wiring empirically) + `kquant_bench.zig` (per-format K-quant
  scalar-vs-AVX2).
- IMPROVE.md documents a verified external-dev-feedback audit: all
  P0-P3 items resolved (see its Parte E table for the commit map).

## Key Patterns

### Zig 0.16 quick rules (full list in LESSONS_ZIG.md)
- `std.ArrayList(T)` is UNMANAGED: `.empty` init, `list.append(allocator, x)`, `list.deinit(allocator)`.
- `allocator.alignedAlloc(T, .@"64", n)` — alignment is a comptime enum, `n` is ELEMENT count.
- Unused params are errors (`_ = param;` or rename `_param`); `&&`→`and`; no `@intToFloat`; `std.mem.eql` for strings.
- Vector element writes need comptime index — stage through an array (`const v: Vec16 = arr;` coerces both ways).
- Test/entrypoint signatures: `test "x" {}` unchanged; `pub fn main(init: std.process.Init) !void`.

### AMX inline asm (verified working, see arch/amx.zig)
```zig
asm volatile (
    "ldtilecfg (%[cfg])"
    :
    : [cfg] "r" (cfg),
    : "memory"
);
// Tile reg must be an immediate: pass as u8 with "n" constraint,
// splice with %%tmm%[tmm]; memory operands look like (%[base],%[stride],1)
```
Requires one-time `arch_prctl` XFEATURE_XTILEDATA enable (done in `tile_loadconfig`).

### Library wiring (the lazy-analysis trap)
A `pub const mod = @import(...)` re-export in root.zig does NOT emit the
module's code. Reference the functions in a `comptime { _ = &mod.fn; }` block,
and verify with `nm zig-out/lib/libkt_kernel_ext.so | grep <symbol>`
(plain `nm` shows internal `t` symbols; `nm -D` only shows `export fn` C API).

### C API export
```zig
export fn kt_moe_forward(moe_ptr: *KT_MOE, ...) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    m.forward(...);
}
```
Opaque handles (`KT_MOE` etc.) are `opaque {}`; wrappers cast to the real type.
**Allocator (B1)**: contexts capture the allocator at `*_new` (from
`kt_set_default_allocator` if installed, else page_allocator) and free
through the CAPTURED one — a `*_free` never reads the current default,
so swapping the default between new and free is safe. New C-API paths
must follow this pattern (see `kt_moe_free`, `kt_mla_free`). Tests that
call exports directly need them declared `pub export fn` in main.zig
(same pattern as `kt_mla_*`/`kt_gate_*`).

### TpMoe per-expert decomposition (Zig-port design; C++ forward is monolithic)
`forwardGateUp`/`forwardDown` split the expert pass so Python can interleave
custom ops (e.g. alternate SwiGLU variants). Pattern: iterate `tp_count`
ranks, slice per-rank output offsets, accumulate down-proj in FP32 -> BF16.
Both are real implementations (the guarded no-op era ended with the
loadWeights fix); D3 fixed the `ldc` for tp_count > 1.

## SIMD on AVX2 (the perf contract — read before touching a kernel)

Benchmarked against the C++ reference's own AVX2 kernels
(`kt-kernel/operators/avx2/avx2_bf16_gemm.hpp`) with
`bench/cpp_vs_zig/run.sh` — see that directory's `README.md` (how to run,
`KT_REF`/`KT_PIN`) and `RESULTS.md` (the measurement record); as of the SIMD
rework the Zig port is **1.21x-1.34x the reference** on BF16 GEMM and
**0.68x** on SwiGLU (with bit-identical output). Four rules carry almost all of
it.

**That range is NOT currently reproducible, and the harness now says so itself.**
A consolidated control-armed pass on 2026-10-04 (`zig run
bench/cpp_vs_zig/measure.zig -- --iters 5 --pin 9`, no filter, commit `add41d5`
lineage) resolved **0 of 7 comparison rows**, every one of them reported as
`UNMEASURABLE — setup-dominated`: `zig_bench` spends 16x-68x longer building its
inputs than timing the kernel, so a process-per-arm sampler is mostly measuring
the setup and the resulting floors are 2.0x-18.4x wide. A 1.2x effect is
untestable through that sampler, so the range above stands as a record of what
was measured on a filtered pass, not as a claim the harness can currently
confirm. The remedy is in the harness, not on this host: hoist the setup out of
the process or sample in-process (`SETUP_DOMINATED` in
`bench/cpp_vs_zig/measure.zig`, which prints `setup:meas` and names which kind
of unresolvable each row is). Do not re-quote the range until a pass resolves
the BF16 rows.

That range was previously written up here as 1.13x-1.88x, and it was wrong: it
accumulated across mixed load conditions and most of the spread was the host, not
the kernels. The 1.88x was a real measurement of a run at load 74 where the same
binary took 2139 ms against 509 ms on a quiet box. Re-measured with a
same-code control arm on each side, the range is **tighter and lower than
advertised** — the work really is ahead of the reference, but by less than this
file used to claim. Read `docs/measuring.md` before quoting ANY ratio from this
file: the noise floor is load-dependent (1.02x-1.14x at load 2-4, ~1.25x at load
80), so a ratio without the floor beside it is not a result.

1. **`@exp` does not vectorize.** compiler-rt has only scalar `expf` and Zig
   does not target libmvec, so `@exp` on `@Vector(8, f32)` becomes 8 scalar
   libcalls with stack spills (17x slower than a polynomial). Use
   `amx.expScalar` / `amx.expVec` — the Cephes construction in
   `arch/amx.zig`, derived (not transcribed) and verified to <= 1 ULP, which
   is *tighter* than the reference's ~1e-6 polynomial. Never reintroduce
   `@exp` in a vector loop; `grep -rn "@exp" src/` should only hit comments.
2. **F16C is irrelevant here.** The weights are BF16, and AVX2 has no BF16
   conversion instruction — the widening is `vpmovzxwd` + `vpslld $16`, which
   LLVM already emits for both sides. F16C is IEEE-f16. Don't spend effort
   "enabling F16C"; `KT_TYPE_F16` exists in the ABI but has no kernel.
3. **Loop order is worth more than instruction selection.** `gemmExpert` and
   `gemmF32Bf16` are n-outer / m-inner so the *weight* matrix (the huge
   operand) streams from DRAM once instead of once per token row. Measured
   3.4-4.0x — far more than the ~1.1x from FMA or extra accumulators. Never
   "simplify" these back to a m/n/k triple loop.
4. **GEMM accumulation is an explicit choice.** `gemmExpert` OVERWRITES its
   output (callers may pass uninitialised memory); `gemmExpertAccum` ADDS to
   it (tensor-parallel rank loops). This split fixed a real bug:
   `forwardDown` used to call the overwrite-only entry point once per tp rank
   into one shared buffer, silently discarding every rank but the last.

Also: `f32_to_bf16` rounds to nearest even (matching ggml / the reference);
`f32_to_bf16_trunc` keeps the old truncating behaviour. And `-Dvariant` does
**not** set Zig CPU features — it only sets C macros for the pybind path, so
the feature set comes from the native target. `zig build -Dtarget=x86_64-linux-gnu
-Dvariant=avx2` still prints "Building variant: avx2" while producing **nothing**:
that target is baseline x86_64, which has no 32-byte vector type, so
`amx.ByteVec = @Vector(32, u8)` fails to compile with "TODO moveStrategy for
@Vector(32, u8)". This was true before the pshufb work as well — the `-Dvariant`
macro not pinning features has always been the underlying issue — but it is worth
stating plainly: **the only supported builds are the native target and the six
x86 variants.** Narrowing `ByteVec` to 16 lanes with two shuffles would restore
the baseline target, and nobody needs it yet.

Tests for all of the above live in `tests/kernels/simd_kernels_tests.zig`
(kept separate from `test_kernels.zig`) — which is also the ONLY place a new
kernel test can go: a `test` block inside a file belonging to the `kt` module
is never executed by `zig build test` (Zig collects tests from the root file's
module), which is how two broken copies hid in `gemm_224_q{3,6}_k.zig`.

**The six GGML K-quants have the same treatment** (`kquant_simd.zig`: one
comptime-parameterized AVX2 driver, `gemmQ*_K` beside each `gemmQ*_KScalar`,
6.3x-15.7x measured, wired into `gemmQuant` + `kt_matmul_q{4,5,6,8}_k`). The
scalar versions are kept deliberately: they are the readable statement of each
format's math and the reference the new tests compare against. Note their
`test` blocks are dead (module boundary) — the live K-quant GEMM tests are in
`tests/kernels/simd_kernels_tests.zig`.

**Next performance work is in `docs/perf-roadmap.md`.** Short version: BF16
GEMM is won (1.21x-1.34x of the C++ reference, re-measured with a control arm;
the old 1.13x-1.88x was mostly the host) and the six K-quants are now
vectorized too (6.3x-15.7x over their own scalar versions), but the remaining
*quantized* families — Q8_0 and the nine IQ* formats — are still scalar on
non-AMX hosts, and the C++ reference has 10 AVX2 kernels we do not run at all
(FP8, MXFP4, MXFP8, GPTQ-INT4, raw-INT4; it has no K-quant kernel, so the
K-quant work has no cross-implementation reference to diff against). Those,
plus the 0.68x SwiGLU activation, is where the remaining gap lives.

## sentrux (architecture sensor — MCP + CLI)

`.sentrux/rules.toml` is the **layering contract** for this repo, enforced by
the sentrux rules engine. In this environment sentrux is exposed via the
**MCP tools** (`sentrux_scan`, `sentrux_check_rules`, ...); the standalone
CLI (`sentrux check .`) exists upstream via `brew install sentrux/tap/sentrux`
/ the install.sh script if you want the gate in CI.

### The contract (know this before adding imports)
Layer order (lower may be imported by higher, **never** the reverse):
`numa/io leaves(0) -> arch intrinsics(1) -> mla engine(2) -> kernels(3) ->
runtime(4) -> root hub(5) -> C exports(6) -> tests/bench(7) -> bindings(8)`.

Hard boundaries: `src/**` never imports `tests/**` or `bench/**`;
`src/main.zig` reaches kernels only through `root.zig` (direct kernel imports
bypass the comptime fn-ref wiring); kernels never import `main.zig`;
`src/numa/**` is a std-only leaf; `src/kernels/arch/**` is imported-by-all and
must not import any specific kernel.

**Two INTENTIONAL exceptions — do NOT "fix" them** (documented inline in
rules.toml):
1. `root.zig <-> main.zig` cycle: `root.zig:241` forces `main.zig` analysis
   so the `export fn kt_*` are EMITTED into the `.so` (Zig drops exports
   nothing references — the LAZY-ANALYSIS trap). Hence `max_cycles = 1`.
2. `root.zig` fan-out 47: it IS the designated module hub + comptime fn-ref
   block. Hence `no_god_files = false`.

Also intentional: grid/lookup tables have a single owner file that sibling
kernels import (`gemm_224_iq1_s.zig` owns `IQ1S_GRID`; `gemm_224_iq2_xxs.zig`
owns `KSIGNS`/`KMASK`). Same-layer data sharing is fine — don't split it.

### How to use the MCP tools (agent workflow)

1. **Session start**: `sentrux_session_start` BEFORE your first edit (saves
   the baseline). At the end, `sentrux_session_end` reports whether your
   session degraded anything. If you don't have the MCP, the CLI equivalents
   are `sentrux gate --save .` before / `sentrux gate .` after.
2. **After adding/removing imports or new files**: `sentrux_rescan` then
   `sentrux_check_rules`. The check FAILs on any new cycle, any layer
   inversion, or any boundary break — fix it or, if the change is a
   legitimate new pattern, update rules.toml WITH a comment explaining why
   (the file's comments are the source of truth for intent).
3. **`sentrux_health`**: `quality_signal` 0-10000. Useful for *deltas* over
   your session (it went 4976 -> 5388 when a real import cycle was removed);
   do NOT chase the absolute number — root.zig's fan-out keeps it structurally
   capped, and that's by design.
4. **Ignore `sentrux_test_gaps` entirely** for this repo: it reports 0%
   coverage because it can't follow the `@import("kt")` module alias that
   build.zig wires for every test suite. The real count is `zig build test`
   (160+ across the suites). Decisions about what to test come from TODO.md
   and the coverage of the file you're touching, NOT from this tool.

### Proven value (why this gate exists)
First run with a correct layer model (b3f2916): found a REAL circular import
`gemm_224_iq3_xxs <-> iq3_quantize` (introduced with the IQ3_XXS quantize
work, a8f7455) sitting inside a dead wrapper that also declared the
nonexistent `std.Thread.Mutex`. Cycle removed, wrapper deleted, signal +412.
That's the class of bug the rules gate now catches at commit time instead
of two sessions later.

### Rules format gotcha
The schema is `[constraints]` + `[[layers]]` + `[[boundaries]]` (from the
sentrux upstream docs) — a `[[rules]]` table parses as ZERO rules and
silently passes. If `check_rules` ever reports `rules_checked: 0` after a
rules.toml edit, that's what happened.


## Documentation

- `README.md` — project overview
- `TODO.md` — task tracking (source of truth for current status)
- `IMPROVE.md` — verified audit of external dev feedback (P0-P3 all resolved; commit map in Parte E)
- `LESSONS_ZIG.md` — Zig 0.16 gotchas, verified AMX asm patterns, lazy-analysis/integration lessons (**read first**)
- `include/kt_kernel.h` — C API contract (do not change without coordinating: Python/pybind11 depends on it; additive changes to the clearly-marked 'Zig extensions' section are the established pattern)
- `docs/api.md` — C API reference for integrators
- `docs/perf-roadmap.md` — **read before any kernel work.** Per-operator list of
  what is still scalar, measured headroom, the four rules that make kernels
  fast here, acceptance criteria, and the parallel-work rules
- `docs/perf-task-prompts.md` — the same items pre-packaged as copy-paste
  prompts, for handing a stream to another dev
- `docs/porting-guide.md` — C++ → Zig porting patterns


  The reference repo is at /ai/repos/2026/ktransformers


## Coordination notes

- `PLAN.md`, `LESSONS_KTRANSFORMERS.md` referenced by older docs do not exist.
- Multiple agents have worked in this tree concurrently: before running
  `git stash`/`git checkout`, check `git status` for live WIP that isn't yours.

### Claims — including on SHARED files

`grep -rn "^// CLAIM:" src/` answers "is this file free?" without anyone having
to ask. Claim a file before editing it, and release the marker in the same
commit that lands the work, so the two never disagree.

**Every shared file is CLAIMABLE, and a claim is enforced by a script, not by
good intentions.** `TODO.md`, `tests/kernels/simd_kernels_tests.zig`,
`docs/*.md` and `bench/cpp_vs_zig/` have all been swept by another session's
commit. `git add <file>` stages the file's **entire working state**, so two
sessions with uncommitted edits in one file always cross-contaminate whichever
lands first — and `bench/cpp_vs_zig/measure.zig` has now been swept three times,
**twice leaving HEAD not compiling**, because a staged commit took another
session's half-finished edits with it.

**Do not "stage by filename". It is the wrong instinct and it is what caused all
of the above.** Three options, in order of preference:

1. **Stage only your own hunks** (`git add -p`, or `git diff | git apply
   --cached` on a filtered diff).
2. **Wait** for the session holding the file.
3. **If their work genuinely belongs in your commit**, say so — and that is
   checkable:

       tools/check_staged_claims.sh

   Run it before committing. It collects the claim handles in every staged file
   and requires each to be declared in your message as `Handles: <handle>`, so a
   commit that sweeps someone else's work cannot land silently. A marker documents
   intent; this enforces it.

   Installed here as an **advisory** `prepare-commit-msg` hook, which appends the
   warning to your message and always exits 0 — so every session sees it and none
   is blocked. A blocking hook would break other sessions mid-flight, because
   `.git/hooks/` is per-*repository*, not per-user: there is no per-session hook,
   which is worth knowing before you ask for one. Re-install with:

       cp tools/prepare-commit-msg .git/hooks/ && chmod +x .git/hooks/prepare-commit-msg

   (`.git/hooks/` is not versioned, so the hook is kept at
   `tools/prepare-commit-msg` and copied in.)

If a shared file is held and you cannot wait: put your write-up in your own notes
file and hand the text to the coordinator to fold in. If you are the coordinator,
integrating someone else's text is your job, not theirs.

Two more coordination rules learned the hard way:

- **`docs/measuring.md` governs any performance claim.** A ratio without its
  noise floor is not a result, and a floor quoted from an earlier session is
  fabricated rather than conservative.
- **A negative result is a deliverable.** Implement the fix, measure it, revert
  it, and write up why it did not work. Three of the most useful findings in this
  repo are a register-pressure diagnosis that was wrong, an instruction count
  that was off by 8, and a kernel that turned out to be 165x faster than a
  microbenchmark claimed.
