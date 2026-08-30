# ktransformers-zig TODO

## Status: Alpha (Build + Tests Working)

Last updated: 2026-08-30

**Build: WORKING** - `zig build` produces `zig-out/lib/libkt_kernel_ext.so`
**Tests: WORKING** - `zig build test` RUNS all suites: 22 kernels + 11 MLA pass, 0 leaks, exit 0

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

### Runtime
- [ ] Verify `worker_pool.zig` compiles and works with NUMA subpools (basic version works)
- [ ] Verify `task_queue.zig` lock-free SPSC/MPMC queues
- [ ] Verify `cpu_detect.zig` feature detection matches C++ version
- [ ] Implement proper `workerLoop` (currently placeholder spin-wait)

### Kernels
- [x] **Complete INT4 GPTQ kernel (`gemm_224_int4.zig`)** — replaced scalar fallback with AMX tile path (`tileint8dpd` + on-the-fly INT4→INT8 dequantization). Lo/hi nibble pattern from C++ `GemmKernel224Int4` (amx_kernels.hpp:1559-1848). Per-group scales applied at K-block boundary via `applyScales` (INT32 → FP32). Correctness test added to `tests/kernels/test_kernels.zig`. Non-AMX hosts fall back to the preserved scalar implementation.
- [x] **Complete FP8 E4M3 kernel (`gemm_224_fp8.zig`)** — replaced scalar fallback with AMX tile path that reuses the BF16 GEMM tiles (`tilebf16dpd` accumulates into FP32, no INT32 scratch). On-the-fly FP8→BF16 dequantization in `loadB` (byte-level, 1:1: each FP8 byte becomes one BF16 short). Per-row scale applied at end of each `(m, n)` block (not per K-step). Tile config matches BF16 (TILE_M=16, TILE_K=32, TILE_N=16, VNNI_BLK=2). Correctness test added (skips on non-AMX). Non-AMX hosts fall back to scalar.
- [ ] Complete MXFP4/MXFP8 kernels (assigned Dev B — see Task assignments above)
- [ ] Implement `applySwiGLU` vectorized version using `std.simd`
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
- [ ] FP8 layerwise transport (`kt_fp8_*` functions)
- [ ] Backward pass functions (currently removed from main.zig)

---

## 🟢 Medium Priority

### Build System
- [x] **Test runner actually runs tests** — replaced `test_step.dependOn(&test_obj.step)` with `test_step.dependOn(&b.addRunArtifact(test_obj).step)` in build.zig for both test modules. Fixed use-after-free in `src/runtime/cpu_detect.zig::detectCpuLinux` (model_name pointed into a soon-to-be-freed buffer). Fixed 3 `.flags = std.ArrayList(u8).init(allocator)` type mismatches.
- [ ] Re-enable multi-variant build (AVX2, AVX512_base, AVX512_VNNI, AVX512_VBMI, AVX512_BF16, AMX)
- [ ] Add variant-specific library names (currently all build as `kt_kernel_ext`)

### Testing
- [ ] Run actual tests and verify all pass (compilation works, execution untested)
- [ ] Integration tests vs C++ reference outputs
- [ ] Benchmark harness comparing Zig vs C++ performance

---

## 🔵 Low Priority / Nice to Have

### Python Packaging
- [ ] `pyproject.toml` for `kt-kernel` wheel
- [ ] Minimal pybind11 wrapper in C++ that loads Zig `.so`
- [ ] CI/CD for multi-variant wheel building

### Advanced Features
- [ ] SFT/LoRA backward pass kernels
- [ ] Speculative decoding (MTP head)
- [ ] ARM/KML backend (NEON/SVE)
- [ ] GGML quantization types compatibility

### Documentation
- [ ] API documentation (zig doc)
- [ ] Porting guide from C++

---

## 📊 Progress Tracking
| Component | Status | % |
|-----------|--------|---|
| Runtime (pool/queue/memory/cpu) | Working, basic impl | 75% |
| AMX Intrinsics | Real inline asm (tile_loadconfig/loadd/stored/zero/dpbf16ps/dpbssd) | 90% |
| Buffer Packing | Working | 80% |
| BF16 GEMM | Working | 70% |
| INT8 GEMM | Working | 70% |
| INT4/FP8/MXFP4/8 GEMM | INT4 and FP8 done (AMX), MXFP4/8 in progress (Dev B) | 50% |
| MoE Orchestration | Forward path complete (gate+up+SwiGLU+down+routing); Gate C API wired; weight_ld OOB fixed | 95% |
| C API | MLA + MoE + Gate + Linear + MLP complete; FP8/Backward still placeholder | 85% |
| Build System | Single variant | 60% |
| Tests | All 25 kernels pass (incl. Gate+MoE and Linear+MLP end-to-end), 0 leaks, `zig build test` exits 0 | 100% |
| Python Integration | Not started | 0% |

> **2026-08-30 status note**: Build is currently broken by Dev B's
> uncommitted MXFP4/MXFP8 WIP in `src/root.zig` and
> `src/kernels/amx/gemm_224_mxfp8.zig` (two issues: `fromMatBF16`
> referenced as a free fn but it's a struct method, and a u8/u32 cast
> at gemm_224_mxfp8.zig:51). Per AGENTS.md coordination guidance, Dev A
> does not touch Dev B's WIP. The moe.zig weight_ld OOB fix and the
> Linear+MLP C API work are in place and will be testable once Dev B's
> work compiles. Last known good: 25/25 kernels + 11/11 MLA, exit 0.

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


### Current Status (2026-08-30)
- **Dev A (this session)**: Linear/MLP C API wired; kt_*_config_t structs
  synced with C header; MoE gemmExpert weight_ld OOB fixed (6 sites).
  25/25 kernels + 11/11 MLA passed before Dev B's WIP broke the build.
- **Dev B (concurrent)**: MXFP4/MXFP8 kernel work in progress; build
  currently broken in their files. See status note above.
- **Last fully-green**: 25/25 kernels + 11/11 MLA, exit 0.
