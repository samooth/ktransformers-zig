# Porting Guide: C++ ktransformers → Zig ktransformers-zig

This guide helps developers understand how the original C++ `kt_kernel_ext.so`
was ported to Zig, and how to extend or maintain the Zig implementation.

## Architecture Comparison

### C++ (original)

```
ktransformers/ktransformers/kt-kernel/
├── operators/
│   ├── amx/
│   │   ├── moe_base.hpp        # AMX MoE base class
│   │   ├── moe.hpp             # AMX MoE + TP
│   │   └── la/
│   │       └── amx_kernels.hpp # AMX GEMM kernels
│   ├── avx/                    # AVX512/AVX2 kernels
│   └── moe-tp.hpp              # Tensor-parallel MoE
├── cpu_backend/
│   └── worker_pool.h           # Thread pool
└── ext_bindings.cpp            # pybind11 → Python
```

### Zig (this repo)

```
ktransformers-zig/
├── src/
│   ├── main.zig                # C API exports (replaces ext_bindings.cpp)
│   ├── runtime/
│   │   ├── worker_pool.zig     # Work-stealing thread pool
│   │   ├── task_queue.zig      # Lock-free SPSC/MPMC queues
│   │   ├── memory.zig          # SIMD-aware allocators
│   │   └── cpu_detect.zig      # CPU feature detection
│   └── kernels/
│       ├── arch/
│       │   └── amx.zig         # AMX inline asm + activation functions
│       ├── amx/
│       │   ├── buffers.zig     # Buffer types
│       │   ├── gemm_224_bf16.zig
│       │   ├── gemm_224_int8.zig
│       │   ├── gemm_224_int4.zig
│       │   ├── gemm_224_fp8.zig
│       │   ├── gemm_224_mxfp4.zig
│       │   └── gemm_224_mxfp8.zig
│       ├── moe/
│       │   ├── moe.zig         # TpMoe (inference)
│       │   └── moe_sft.zig     # TpMoeSft (training/SFT)
│       └── amx/
│           └── lora_kernels.zig # LoRA GEMM kernels
├── include/
│   └── kt_kernel.h             # C API (contract with Python)
├── python/kt_kernel/
│   └── __init__.py             # ctypes wrapper + variant auto-detect
├── build.zig                   # Multi-variant build system
└── tests/kernels/
    └── test_kernels.zig        # Kernel test suite
```

## Key Design Decisions

### 1. C API Compatibility

The Zig implementation maintains **binary compatibility** with the original
C++ API defined in `include/kt_kernel.h`. All `kt_*` export functions use
`callconv(.C)` and opaque handles (`KT_MOE`, `KT_MLA`, etc.) so the existing
pybind11 Python wrapper can load the Zig-built `.so` without modification.

### 2. Variant Selection

| C++ (compile-time) | Zig (build-time) |
|--------------------|------------------|
| Template specialization per arch | `-Dvariant=` build option |
| Single binary per arch | `zig build all-variants` → 6 `.so` files |
| Runtime dispatch via feature detect | Runtime dispatch via `/proc/cpuinfo` in Python wrapper |

The Python wrapper (`python/kt_kernel/__init__.py`) auto-detects the best
variant at import time and loads the matching `.so`.

### 3. AMX Inline Assembly

C++ uses Intel intrinsics (`_mm512_tile_dpbf16_ps`). Zig uses inline assembly
with comptime tile-register immediates:

```zig
// Zig: AMX BF16 dot product (arch/amx.zig)
asm volatile (
    "tilebf16ps %%tmm%[dst], %%tmm%[src], %%tmm%[acc]"
    : [dst] "+t" (dst_reg)
    : [src] "t" (src_reg), [acc] "t" (acc_reg)
    : "memory"
);
```

Requires one-time `arch_prctl(ARCH_ENABLE_XFEATURE_XTILEDATA)` enablement.

### 4. Memory Management

| C++ | Zig |
|-----|-----|
| `std::vector`, RAII | `std.mem.Allocator` pattern |
| `new`/`delete` | Page allocator for weights, arena for scratch |
| Manual NUMA-aware alloc | `numaNodeOfCpu()` + per-NUMA subpools |

### 5. Thread Pool

| C++ (`worker_pool.h`) | Zig (`worker_pool.zig`) |
|-----------------------|------------------------|
| `std::mutex` + `std::condition_variable` | `std.c.pthread_mutex_t`/`pthread_cond_t` |
| `std::atomic<int>` work counter | `std.atomic.Value(usize)` |
| `std::barrier` for phases | Condition variable + completion counter |

Zig 0.16 lacks `std.Thread.Mutex` (added in 0.17), so the pool uses POSIX
primitives directly. Workers block on a condvar when idle (no CPU burn).

## Build & Install

```bash
# Build single variant (default AVX2)
zig build

# Build all 6 variants
zig build all-variants

# Build wheel (bundles all .sos)
python setup.py build_ext --inplace
# or
pip install -e .

# Run tests
zig build test
```

## Python Usage

```python
import kt_kernel as kt

# Auto-detects CPU variant and loads the right .so
print(kt.kt_version())       # b'0.6.1-zig'
print(kt.kt_get_cpu_variant())  # b'avx2'

# BF16 conversion
import numpy as np
f32_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
bf16_arr = np.empty(3, dtype=np.uint16)
kt.kt_f32_to_bf16(f32_arr.ctypes.data, bf16_arr.ctypes.data, 3)
```

## Testing

```bash
# Full suite (kernels + MLA)
zig build test

# Standalone MLA suite
zig test src/mla/mla_tests.zig
```

Tests run for real via `addRunArtifact` (build.zig pattern). Kernel tests
cover: AMX activation functions, BF16/INT8/INT4/FP8/MXFP4/MXFP8 GEMM, MoE
forward equivalence, SFT forward+backward, work-stealing pool, NUMA topology.

## Adding a New Kernel

1. Create `src/kernels/amx/gemm_224_<new>.zig`
2. Define `GemmKernel224<New>` struct with `gemmFullTile` + scalar fallback
3. Add `pub const` re-export in `src/root.zig`
4. Force comptime analysis: `_ = &gemm_<new>.GemmKernel224<New>.gemmFullTile;`
5. Add test in `tests/kernels/test_kernels.zig` (append-only)

## Lessons Learned

See `LESSONS_ZIG.md` for verified Zig 0.16 gotchas:
- `@splat` is 1-arg only (length inferred from context)
- `@Vector` has no `.len` member (use a separate `pub const`)
- Vector element access needs comptime index (stage via array coercion)
- `std.Thread.Mutex` doesn't exist in 0.16 (use `std.c` pthread)
- `try` can't be used in `void`-returning functions
- `AutoHashMap.getOrPut` doesn't zero-initialize new entries
