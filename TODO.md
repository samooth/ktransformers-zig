# ktransformers-zig TODO

## Status: Alpha (Build Working)

Last updated: 2026-08-29

**Build: WORKING** - `zig build` produces `zig-out/lib/libkt_kernel_ext.so`

---

## 🔴 Critical: Fix Build Errors (Blocking)

### Module System
- [ ] **Fix Zig module imports** - `@import("arch/amx.zig")` fails. Zig's `addIncludePath()` doesn't work for Zig modules. Need either:
  - `src/root.zig` that re-exports all modules with `pub const amx = @import("kernels/arch/amx.zig");` etc.
  - Or use `b.addModule()` for each submodule and wire imports in `build.zig`

### Code Fixes (25+ errors)

#### `src/kernels/amx/buffers.zig`
- [ ] `requiredSize` unused params → prefix with `_`
- [ ] `quantizeRowBF16ToInt8` line 308: `@truncate(val / scale.* + 0.5)` → `@intCast(val / scale.* + 0.5)`
- [ ] `zero` calc line 336: `@as(u8, @intCast(-min_val / scale + 0.5))`
- [ ] `f32_MAX` → `std.math.max(f32)` (lines 328, 337)
- [ ] `fromMatTransposed`: unused `_src_k` param
- [ ] `getSubmat`: unused `n_blocks`, `n_within`, `k_within` → `_` prefix
- [ ] `toMat`: unused `_n_block`, `_n_step` params

#### `src/kernels/amx/gemm_224_bf16.zig`
- [ ] `split_range_n`: `_n` param
- [ ] `gemmExpert`: `_m`, `_n`, `_k` params

#### `src/kernels/amx/gemm_224_int8.zig`
- [ ] Missing closing `}` at end of file (line 319)

#### `src/kernels/arch/amx.zig`
- [ ] Line 69: inline asm constraint syntax - check Zig 0.16 asm format

#### `src/kernels/moe/moe.zig`
- [ ] Line 426: ternary in function arg needs parens around each branch

#### `src/main.zig`
- [ ] Line 295: `@ptrCast(*worker_pool.WorkerPool, cpuinfer)` → single arg `@ptrCast(cpuinfer)`
- [ ] Remove unused imports: `memory`, `task_queue`, `gemm_bf16`, `gemm_int8`

---

## 🟡 High Priority

### Build System
- [ ] Create `src/root.zig` that re-exports all modules for proper Zig module resolution
- [ ] Re-enable multi-variant build (AVX2, AVX512_base, AVX512_VNNI, AVX512_VBMI, AVX512_BF16, AMX)
- [ ] Add `zig build test` step working

### Runtime
- [ ] Verify `worker_pool.zig` compiles and works with NUMA subpools
- [ ] Verify `task_queue.zig` lock-free SPSC/MPMC queues
- [ ] Verify `cpu_detect.zig` feature detection matches C++ version

### Kernels
- [ ] Complete INT4 GPTQ kernel (`gemm_224_int4.zig`)
- [ ] Complete FP8 E4M3 kernel (`gemm_224_fp8.zig`)
- [ ] Complete MXFP4/MXFP8 kernels
- [ ] Implement `applySwiGLU` vectorized version using `std.simd`

---

## 🟢 Medium Priority

### MoE Layer
- [ ] Implement `TpMoe.loadWeights()` with online quantization (BF16 → INT8/INT4)
- [ ] Implement `merge_results()` with AVX512 FP32 add + BF16 convert
- [ ] Expert routing with top-k selection (SIMD optimized)

### C API Completeness
- [ ] MLA attention (`kt_mla_*` functions)
- [ ] Gate (`kt_gate_*` functions)
- [ ] Linear/MLP (`kt_linear_*`, `kt_mlp_*`)
- [ ] FP8 layerwise transport (`kt_fp8_*` functions)

### Testing
- [ ] Unit tests passing (`zig build test`)
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
| Runtime (pool/queue/memory/cpu) | Code done, untested | 70% |
| AMX Intrinsics | Code done | 80% |
| Buffer Packing | Code done, needs fixes | 60% |
| BF16 GEMM | Code done, needs fixes | 60% |
| INT8 GEMM | Code done, needs fixes | 60% |
| INT4/FP8/MXFP4/8 GEMM | Not started | 0% |
| MoE Orchestration | Partial | 40% |
| C API | Partial | 50% |
| Build System | Single variant | 60% |
| Tests | Written, unrunnable | 30% |
| Python Integration | Not started | 0% |

---

## Notes

- **Reference**: Original C++ in `/ai/repos/2026/ktransformers/ktransformers/kt-kernel/`
- **Zig Version**: 0.16.0-dev.2535+b5bd49460
- **Target**: Drop-in replacement for `kt_kernel_ext.so` (6 CPU variants)
- **Python Compat**: Match `kt_kernel_ext.so` C API exactly for pybind11