# ktransformers-zig TODO

## Status: Alpha (Build + Tests Working)

Last updated: 2026-08-29

**Build: WORKING** - `zig build` produces `zig-out/lib/libkt_kernel_ext.so`
**Tests: WORKING** - `zig build test` compiles successfully

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

### Runtime
- [ ] Verify `worker_pool.zig` compiles and works with NUMA subpools (basic version works)
- [ ] Verify `task_queue.zig` lock-free SPSC/MPMC queues
- [ ] Verify `cpu_detect.zig` feature detection matches C++ version
- [ ] Implement proper `workerLoop` (currently placeholder spin-wait)

### Kernels
- [ ] Complete INT4 GPTQ kernel (`gemm_224_int4.zig`)
- [ ] Complete FP8 E4M3 kernel (`gemm_224_fp8.zig`)
- [ ] Complete MXFP4/MXFP8 kernels
- [ ] Implement `applySwiGLU` vectorized version using `std.simd`
- [ ] Add actual AMX inline assembly (currently placeholder)

### MoE Layer
- [ ] Implement `TpMoe.loadWeights()` with online quantization (BF16 → INT8/INT4)
- [ ] Implement `merge_results()` with AVX512 FP32 add + BF16 convert
- [ ] Expert routing with top-k selection (SIMD optimized)

### C API Completeness
- [ ] MLA attention (`kt_mla_*` functions)
- [ ] Gate (`kt_gate_*` functions)
- [ ] Linear/MLP (`kt_linear_*`, `kt_mlp_*`)
- [ ] FP8 layerwise transport (`kt_fp8_*` functions)
- [ ] Backward pass functions (currently removed from main.zig)

---

## 🟢 Medium Priority

### Build System
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
| AMX Intrinsics | Placeholder (no inline asm) | 50% |
| Buffer Packing | Working | 80% |
| BF16 GEMM | Working | 70% |
| INT8 GEMM | Working | 70% |
| INT4/FP8/MXFP4/8 GEMM | Not started | 0% |
| MoE Orchestration | Partial | 40% |
| C API | Basic functions only | 50% |
| Build System | Single variant | 60% |
| Tests | Compiling, need to run | 50% |
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
