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
zig build -Dvariant=amx   # avx2|avx512_base|avx512_vnni|avx512_vbmi|avx512_bf16|amx
zig build test            # runs both suites (kernels + MLA) — tests EXECUTE now
zig test src/mla/mla_tests.zig   # standalone MLA suite
```

Caveats (verified the hard way — see LESSONS_ZIG.md for details):
- All variants currently install the SAME name `libkt_kernel_ext.so`.
- Tests only run because build.zig uses `addRunArtifact`; `test_step.dependOn(&test_obj.step)` alone only compiles.
- If a build error survives a no-op rebuild, suspect stale `.zig-cache`: `rm -rf .zig-cache zig-out`.

## Code Structure

Imported (reachable from `src/root.zig`, compiled into the .so):
```
src/
├── root.zig                    # module re-exports + comptime fn-refs forcing analysis
├── main.zig                    # C API exports (kt_*) — mirrors include/kt_kernel.h
├── runtime/{memory,worker_pool,task_queue,cpu_detect}.zig
├── numa/                       # NUMA topology/memory/worker (re-exported, but no
│                               # comptime fn-refs yet -> not emitted into the .so)
├── kernels/
│   ├── arch/amx.zig            # AMX inline asm (ldtilecfg, tilebf16dpd, ...) + XFEATURE enable
│   ├── amx/{buffers,gemm_224_{bf16,int8,int4,fp8}}.zig
│   └── moe/moe.zig             # TpMoe: routing, loadWeights, 5 per-expert methods
└── mla/                        # MLA attention (DeepSeek-V2/V3): config/cache/core
```

NOT imported (dead code — excluded by Zig's lazy analysis; don't assume they compile):
`kernels/amx/gemm_224_{bf16,int8,int4,fp8}_avx512.zig` (also imports a
nonexistent `arch/amx_intrinsics.zig`), `gemm_224_mxfp{4,8}.zig`. Note
`numa/` IS re-exported from root.zig but has no comptime fn-refs, so its
functions are not emitted into the .so yet. Also: `main.zig.backup*` are
stale snapshots.

## Status (see TODO.md for details)

- Build: working; all 6 variants compile (default AVX2 installed).
- C API: `kt_moe_*`, `kt_mla_*` (placeholder bodies), plus conversions exported.
- MoE: `loadWeights` BF16 branch only packs `gate_proj` (known bug);
  `forwardGateUp`/`forwardDown` are documented no-ops pending that fix;
  `forward`'s expert GEMM is disabled (placeholder).
- MLA: complete in `src/mla/` (absorbed attention, latent KV cache);
  `kt_mla_*` C API still placeholder.
- Tests: kernels 21 + MLA 11, run for real via addRunArtifact; known
  failures tracked in TODO.md.

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
`kt_moe_new` allocates with `std.heap.page_allocator` — `deinit` methods must
free with the same (they take no allocator param).

### TpMoe per-expert decomposition (Zig-port design; C++ forward is monolithic)
`forwardGateUp`/`forwardDown` split the expert pass so Python can interleave
custom ops (e.g. alternate SwiGLU variants). Pattern: iterate `tp_count`
ranks, slice per-rank output offsets, accumulate down-proj in FP32 -> BF16.
Currently shipped as guarded no-ops with the real version in comments —
enabled by the loadWeights fix.

## Documentation

- `README.md` — project overview
- `TODO.md` — task tracking (source of truth for current status)
- `LESSONS_ZIG.md` — Zig 0.16 gotchas, verified AMX asm patterns, lazy-analysis/integration lessons (**read first**)
- `include/kt_kernel.h` — C API contract (do not change without coordinating: Python/pybind11 depends on it)


  The reference repo is at /ai/repos/2026/ktransformers


## Coordination notes

- `PLAN.md`, `LESSONS_KTRANSFORMERS.md` referenced by older docs do not exist.
- Multiple agents have worked in this tree concurrently: before running
  `git stash`/`git checkout`, check `git status` for live WIP that isn't yours.
