# AGENTS.md - ktransformers-zig

Guidance for AI agents working on this project.

## Project Context

Porting ktransformers CPU kernels (C++/AMX/AVX2/AVX512) to Zig for drop-in replacement of `kt_kernel_ext.so`.

**Target**: 6 CPU variants (AVX2, AVX512_base, AVX512_VNNI, AVX512_VBMI, AVX512_BF16, AMX) with Python pybind11 compatibility.

**Reference**: Original C++ at `/ai/repos/2026/ktransformers/ktransformers/kt-kernel/`

## Build System

```bash
# Build (default AVX2)
zig build

# Specific variant
zig build -Dvariant=amx

# Test
zig build test

# Install
zig build install
```

Output: `zig-out/lib/kt_kernel_ext_<variant>.so`

**Zig version**: 0.16.0-dev.2535+

## Code Structure

```
src/
├── main.zig              # C API exports (kt_version, kt_moe_*, etc.)
├── runtime/
│   ├── memory.zig        # SimdArena, NumaAllocator, SharedBuffer, BufferPool
│   ├── worker_pool.zig   # NUMA subpools, work stealing
│   ├── task_queue.zig    # SPSC + MPMC lock-free queues
│   └── cpu_detect.zig    # CPU vendor/features, variant selection
└── kernels/
    ├── arch/amx.zig      # AMX tile intrinsics (ldtilecfg, tdpbf16ps, etc.)
    ├── amx/
    │   ├── buffers.zig   # BufferA/B/C packing, quantization
    │   ├── gemm_224_bf16.zig
    │   └── gemm_224_int8.zig
    └── moe/moe.zig       # TpMoe: expert routing, computeExpert, loadWeights
```

## Key Patterns

### Zig 0.16 Specifics
- `@import("relative/path.zig")` for modules (not `addIncludePath`)
- `std.meta.stringToEnum(T, str)` not `EnumVariant`
- Unused params/locals are errors → prefix with `_`
- `@round()` takes 1 arg; use `@intCast(x + 0.5)` for rounding
- `f32_MAX` → `std.math.max(f32)`

### AMX Intrinsics
```zig
asm volatile (
    "ldtilecfg [%0]"
    :
    : "r" (cfg)
    : "memory"
);
```
Tile regs: `TileReg.tmm0` through `TileReg.tmm7`

### CRTP Pattern (from C++)
```zig
// Base holds common logic, derived provides kernel config
const Kernel = struct {
    pub const TILE_M = 16;
    pub fn config() void { ... }
};

const MoE = struct {
    fn computeExpert(...) void { Kernel.config(); ... }
};
```

### C API Export
```zig
export fn kt_moe_forward(moe_ptr: *KT_MOE, ...) void {
    const m = @ptrCast(moe.TpMoe, moe_ptr);
    m.forward(...);
}
```

## Current Status

**Build: WORKING** - `zig build` produces `zig-out/lib/libkt_kernel_ext.so`

See `TODO.md` for remaining work and `LESSONS_ZIG.md` for Zig 0.16 syntax gotchas.

### Immediate Fixes Needed
1. Module imports: `@import("arch/amx.zig")` fails - need `src/root.zig` re-exports
2. `buffers.zig`: unused params, `@truncate` → `@intCast`, `f32_MAX` → `std.math.max(f32)`
3. `gemm_224_int8.zig`: missing `}`
4. `arch/amx.zig:69`: inline asm syntax
5. `moe.zig:426`: ternary in arg needs parens
6. `main.zig:295`: `@ptrCast` single arg

## Testing

```bash
zig build test
```
Tests in `tests/kernels/test_kernels.zig` cover: BF16 conversion, buffer sizes, CPU detection, worker pool, memory arena, SwiGLU, quantization.

## Python Integration

C API in `include/kt_kernel.h` matches ktransformers. Python loads via pybind11 wrapper:
```python
# In ktransformers Python package
from kt_kernel import MOE  # loads kt_kernel_ext.so
```

## Reference Implementation

Original C++ at `/ai/repos/2026/ktransformers/ktransformers/kt-kernel/`:
- `operators/amx/la/amx_kernels.hpp` - GemmKernel224BF/INT8/INT4
- `operators/amx/moe_base.hpp` - AMX_MOE_BASE
- `operators/amx/moe.hpp` - AMX_MOE_TP
- `operators/moe-tp.hpp` - TP_MOE_Common
- `cpu_backend/worker_pool.h` - WorkerPool
- `ext_bindings.cpp` - pybind11 exports

## Zig-AI Dependency

Some CPU kernels from `/home/t0m4s/repos/2026/zig-ai`:
- `src/moe/cpu_gemv.zig` - Quantized GEMV
- `src/moe/cpu_executor.zig` - Worker pool
- `kernels/dequant_*.cu` - GGUF dequant CUDA

Add to `build.zig.zon` when needed.

## Documentation

- `README.md` - Project overview
- `TODO.md` - Task tracking with priorities
- `PLAN.md` - Full porting plan (20 weeks)
- `LESSONS_ZIG.md` - Zig 0.16 lessons learned (READ THIS FIRST for Zig syntax/idioms)
- `LESSONS_KTRANSFORMERS.md` - ktransformers architecture notes

## Critical: Zig 0.16 Gotchas (from LESSONS_ZIG.md)

**READ `LESSONS_ZIG.md` BEFORE WRITING ANY ZIG CODE** - the syntax differs significantly from older versions:

1. **Module imports**: Use `src/root.zig` as root, re-export submodules with `pub const x = @import("...");`. Do NOT use `addIncludePath()` for Zig modules.

2. **Unused parameters**: Prefix with `_` AND add explicit `_ = param;` if still flagged.

3. **`&&` is ambiguous** - use `and` instead.

4. **String comparison**: Use `std.mem.eql(u8, a, b)` not `==`.

5. **Struct declarations at top level**: Use `pub const X = struct { ... };` not `pub struct X { ... };`.

6. **Single-line if/else**: Wrap in braces: `if (cond) { ... } else { ... }`.

7. **Alignment** is a reserved keyword - use `alignment`.

8. **AMX inline asm syntax**: Use `asm (...)` not `asm volatile (...)` in Zig 0.16.

9. **`std.StringArray`** doesn't exist - use `std.ArrayList([]const u8)`.

10. **`@intToFloat`/`@intToInt`** don't exist - use `@as(T, @intCast(val))`.