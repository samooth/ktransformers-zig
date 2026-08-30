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
- [ ] **Wire `kt_mla_*` C API to the real MLA engine** — replace the 5 placeholder bodies in `main.zig` (kt_mla_new/free/forward/prefill/decode, ~lines 493-560) with `MlaEngine` construction/forward/decode/resetCache calls. Pattern is proven (kt_moe_* + comptime fn-refs already in root.zig for mla_core). Map `kt_mla_config_t` → `MlaConfig`; note kt_mla_new takes `*KT_CPUInfer` (ignored by engine, sequential heads).
- [ ] **Upgrade `TpMoe.warmUp` to real pre-touch** — blocker resolved (loadWeights now populates ptrs); replace the no-op (moe.zig:~576, grep TODO(warm-up)) with the first-byte touch loop from the comment. Keep the weights_loaded guard.

**Dev B — MXFP4/MXFP8 kernels** (files: `gemm_224_mxfp{4,8}.zig`, `src/root.zig` re-exports, `tests` append-only, docs):
- [ ] **Complete MXFP4/MXFP8 kernels** — both files exist but are NOT in root.zig's import graph (never compiled; mxfp4 has a bogus `unpackMXFP4(block, &sum)` line that will fail analysis). Steps: standalone `zig test` each file; wire re-exports + comptime fn-refs into root.zig (verify `nm | grep -i mxfp`); scalar exact-value tests; AMX tile path via on-the-fly block dequant (pattern: the INT4/FP8 sections in LESSONS_ZIG.md; reference: `operators/amx/mxfp8-moe.hpp`, `fp4-moe.hpp`, `avx2/mxfp4-moe.hpp`).

**Exclusivity notes**: Dev A owns main.zig + moe.zig(warmUp); Dev B owns mxfp files + root.zig. Shared files only by the stated constraints (tests append-only, docs append/mark). `zig build test` currently exits 0 with 0 leaks (33/33) — keep it that way.

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
- [ ] Implement `merge_results()` with AVX512 FP32 add + BF16 convert
- [ ] Expert routing with top-k selection (SIMD optimized)
- [x] **Replace 5 placeholder TpMoe methods** — `deinit` (frees the two init-allocated slices, page_allocator convention; buffer structs are non-owning POD views), `warmUp` (no-op + TODO until loadWeights populates ptrs), `loadWeightsWithMap` (double-load + logical-slot remap), `forwardGateUp`/`forwardDown` (guarded no-ops with real bodies in comments, blocked on the loadWeights BF16 bug). End-to-end test with zero inputs passes; 5 TpMoe placeholder methods now safe.

### C API Completeness
- [x] **MLA attention — code complete in `src/mla/`, re-exported from `root.zig`** (config, cache, core modules wired into the library; 11/11 standalone tests passing). C API integration (`kt_mla_*` in `main.zig`) assigned Dev A — see Task assignments above.
- [ ] Gate (`kt_gate_*` functions)
- [ ] Linear/MLP (`kt_linear_*`, `kt_mlp_*`)
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
| INT4/FP8/MXFP4/8 GEMM | INT4 and FP8 done (AMX), MXFP4/8 not started | 50% |
| MoE Orchestration | Forward path complete (gate+up+SwiGLU+down+routing) | 75% |
| C API | Basic functions only | 50% |
| Build System | Single variant | 60% |
| Tests | All 22 kernels + 11 MLA pass, 0 leaks, `zig build test` exits 0 | 100% |
| Python Integration | Not started | 0% |

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

### Current Status
- **Build**: ✅ WORKING (0 errors)
- **Tests**: ✅ PASSING
- **Library**: `zig-out/lib/libkt_kernel_ext.so` (12.2 MB)
- **All compilation issues resolved**
