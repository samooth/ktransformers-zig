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
zig test src/mla/mla_tests.zig             # standalone MLA suite
zig build -Doptimize=ReleaseFast bench     # GEMM micro-benchmark (B2)
python3 tools/verify_abi.py                 # ABI double gate: exports + arity
python3 tools/audit_layout.py               # ABI layout gate (pybind11 shim)
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
│   ├── amx/{buffers,gemm_224_{bf16,int8,int4,fp8,q8_0,q4_k,q5_k,q6_k,q8_k,mxfp4,mxfp8}}.zig
│   └── moe/moe.zig             # TpMoe: routing (legacy + DeepSeek-V3 group-top2, D4),
│                               # loadWeights (all 3 projections), vectorized gemmExpert (A1)
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

- Build: working; 6 x86 variants + aarch64 neon cross-build.
 - C API: **105 symbols** gated by `verify_abi.py` triple gate (exports +
  arity + layout via audit_layout.py). All operator families implemented:
  MoE (incl. SFT/LoRA forward+backward), MLA, Gate (DeepSeek-V3
  group-top2 routing), Linear, MLP, FP8 transport, math helpers,
  `kt_set_default_allocator` (injectable allocator, B1),
  `kt_dsv3_*` (DeepseekV3 model-orchestration), and
  `kt_qwen3moe_*` (Qwen3 model-orchestration).
- GGML: **10/10 standard formats complete** — Q8_0, Q4_K, Q5_K, Q6_K,
  Q8_K (linear/4-8 bit), Q2_K, Q3_K (linear/2-3 bit), IQ4_XS, IQ2_XXS
  (grid-based with full kmap/kneighbors init for quantize), IQ3_XXS
  (grid-based), IQ4_NL (non-linear 4-bit, 32-weight super-blocks).
  IQ2_XXS/IQ3_XXS quantize is now real (was stubbed, see commits
  7d10033 + b003990); only IQ2/IQ3 "S" and "M" variants remain for
  if a real model needs them.
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
- Tests: 73 kernels + 11 MLA + 9 MLA C-API + 2 FP8 + 9 GGML C API +
  7 aarch64 + 4 NEON kernel + 2 allocator C API + 9 Qwen3 MoE =
  **156 tests, all passing, 0 leaks**. Bench: `zig build bench`
  (2.8-9.3x measured speedup A1) + `moe_bench.zig` (end-to-end MoE
  forward with tile-param tuning: -20.3% on prefill 8×4 at K=448
  vs the 1792 default on this Ryzen 512K-L2 — validates the A4
  wiring empirically).
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

## Documentation

- `README.md` — project overview
- `TODO.md` — task tracking (source of truth for current status)
- `IMPROVE.md` — verified audit of external dev feedback (P0-P3 all resolved; commit map in Parte E)
- `LESSONS_ZIG.md` — Zig 0.16 gotchas, verified AMX asm patterns, lazy-analysis/integration lessons (**read first**)
- `include/kt_kernel.h` — C API contract (do not change without coordinating: Python/pybind11 depends on it; additive changes to the clearly-marked 'Zig extensions' section are the established pattern)
- `docs/api.md` — C API reference for integrators
- `docs/porting-guide.md` — C++ → Zig porting patterns


  The reference repo is at /ai/repos/2026/ktransformers


## Coordination notes

- `PLAN.md`, `LESSONS_KTRANSFORMERS.md` referenced by older docs do not exist.
- Multiple agents have worked in this tree concurrently: before running
  `git stash`/`git checkout`, check `git status` for live WIP that isn't yours.
