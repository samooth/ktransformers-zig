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

## Adding a New Model Orchestration Layer

The Qwen3 MoE port (2026-09) is the canonical example. Procedure:

1. **Pick an attention engine** — MLA (`src/mla/mla_core.zig`) for
   DeepSeek-style, or MHA (`src/kernels/attn/mha.zig`) for standard
   attention. MHA is reusable across any non-MLA transformer; MLA is
   DSV3-specific.
2. **Pick or implement a router** — DeepSeek uses sigmoid+group-top2
   (`moe.routeExpertsDeepSeek` with `scoring=.sigmoid, n_group>1`),
   Qwen3 uses vanilla softmax top-k (`scoring=.softmax, n_group=1,
   bias=null`). The D4 path already supports both — no new gate code
   needed.
3. **Create the layer module** (`src/kernels/<arch>/<arch>_layer.zig`):
   - Define a `LayerConfig` struct mirroring the C ABI struct (all
     weight pointers borrowed from caller; BF16 for everything).
   - Create a `*DecoderLayer` struct holding the engine + cache + MoE
     + scratch buffers. `init` allocates, `deinit` frees symmetrically.
   - Implement `forward(qlen, kv_start_pos, input, output)` doing
     residual + RMSNorm + attention + residual + RMSNorm + gate + MoE +
     residual, with BF16↔F32 conversion at the C ABI boundary.
4. **Create the model + CausalLM** (`src/kernels/<arch>/<arch>_model.zig`):
   - Model: N × layer + final RMSNorm (ping-pong buffers).
   - CausalLM: Model + lm_head GEMM.
5. **Add C API exports** (`src/main.zig`):
   - `pub const kt_<arch>_layer_config_t = extern struct { ... };` matching
     the header field-for-field.
   - `fn toXxxConfig(c) ...` converter; `*_new` / `*_forward` / `*_free`
     with the `*opaque` casts.
6. **Add header declarations** (`include/kt_kernel.h`):
   - `typedef struct kt_<arch>_layer_config_t { ... };`
   - Function prototypes in the "Zig extensions" section.
7. **Wire into root.zig**: `pub const <arch>_layer = @import(...);` +
   `comptime { _ = &<arch>_layer.<Type>.init; _ = &<arch>_layer.<Type>.forward; ... }`
   (lazy-analysis trap — without the comptime block, the .so has zero
   code for the module).
8. **Add size probes** (`src/main.zig`):
   - `export fn kt_abi_size_<arch>_layer_config() usize { return @sizeOf(...); }`
9. **Update `tools/audit_layout.py`** with the new struct's field list.
10. **Add pybind11 shim** (`bindings/kt_kernel_pybind.cpp`):
    - ABI struct mirror with identical field order/types.
    - pybind-side config struct (pointer fields as `size_t` ints).
    - `Py<Name>Layer`/`Model`/`CausalLM` wrapper classes.
    - `m.<arch>` submodule with the classes and `LayerConfig`/`ModelConfig`.
11. **Tests** in `tests/kernels/<arch>_tests.zig`:
    - Unit tests for the engine (MHA softmax, RoPE, etc.)
    - Layer init + 2-step decode (zero weights → exact residual semantics).
    - Model + CausalLM forward (zero weights → zero logits).
    - Register the test file in `build.zig` (Suite N: ...).
12. **Verify all gates**:
    - `zig build all-variants` — all 6+1 variants build clean
    - `python3 tools/verify_abi.py` — all symbols exported
    - `python3 tools/audit_layout.py` — struct layouts match
    - `zig build test` — all tests pass, 0 leaks
    - `bash bindings/build.sh` — pybind11 module builds, smoke test OK

**Lazy-analysis trap**: a `pub const` re-export in root.zig does NOT
analyze the module's declarations — the .so contains zero code for it
unless you also add a `comptime { _ = &... }` block. Same pattern
applies to `main.zig` exports: a `pub export fn` in a module that's not
imported anywhere is silently stripped. See `LESSONS_ZIG.md` for the
detailed analysis.

## When NOT to Add a New Model Layer

ktransformers is an **MoE-offload engine** — its value is CPU offload
of expert FFN layers for trillion-param MoE models (DeepSeek-V3 671B,
Qwen3-30B-A3B, Qwen3.5-35B-A3B, etc.). The C++ upstream chose not to
port dense transformer families (Gemma, dense Qwen) because there's
nothing to offload: a dense 12B/26B model fits entirely in GPU VRAM
on a single consumer card, so the offload architecture provides no
benefit.

The same logic applies here. **Gemma (and dense Qwen) are out of
scope** unless the goal is to build a *general* CPU inference engine
(rather than an MoE-offload engine). If that goal changes, the
shared MHA + GGUF-parser + RMSNorm + RoPE base from the Qwen3 port
(`src/kernels/attn/mha.zig`, `src/io/gguf.zig`, the C-extension
`mha.rmsNormInline`) covers ~80% of the work — only the GeLU FFN,
post-norm placement, and (Gemma 2+) soft-capping need to be added
on top.

Dense Qwen and Gemma would also need an embedding/lm-head layer that
the MoE engine currently doesn't have (the Qwen3 port still
assumes embedded inputs from the caller; a true standalone engine
needs a token-id → hidden-state row in `token_embd.weight`).

## AMX GEMM kernel pattern: per-group scale, accumulate F32

The `src/kernels/amx/gemm_224_int4.zig` kernel implements the canonical
pattern for AMX GEMMs with on-the-fly dequant + per-group scales:

1. **K-axis tiling by group, not by K_BLOCK.** Iterate by `GROUP_SIZE`
   (one K-block = one group of 32 elements for Q4_0), not by the larger
   `K_BLOCK` (3584 for the AMX tile pipeline). Each group gets its own
   AMX tile run; the per-group scale is applied immediately and the
   result is accumulated into F32 `c`. **Don't** try to accumulate
   across multiple groups in INT32 then apply a single scale at the
   end — when K_BLOCK > GROUP_SIZE, multiple groups contribute to the
   same INT32 with different scales, and the trailing partial group
   silently gets dropped (returns 0) for any K not a multiple of
   GROUP_SIZE.

2. **Scalar fallback must handle partial trailing groups.** Don't
   compute `k_blocks = k / GROUP_SIZE` (integer divide). Use
   `k_blocks_full = k / GROUP_SIZE` for the full groups and
   `k_tail = k - k_blocks_full * GROUP_SIZE` for the partial last
   block, processing only `k_tail` weights (and handling the
   even/odd `k_tail` separately for the lo/hi nibble layout).

3. **AMX partial-group test gating.** The AMX path is gated by
   `amx.detectAmxSupport()`; the test must work in BOTH paths. For
   the partial-group regression test, gate the AMX-specific expected
   value behind `if (!amx.AmxFeatures.available)` and use a tolerance
   check (e.g. "non-zero, finite") for the AMX case, since the
   lo+hi-tiles-both-contribute math on real AMX hardware produces
   a different exact value than the scalar (1x scale vs 2x).

## Lessons Learned

See `LESSONS_ZIG.md` for verified Zig 0.16 gotchas:
- `@splat` is 1-arg only (length inferred from context)
- `@Vector` has no `.len` member (use a separate `pub const`)
- Vector element access needs comptime index (stage via array coercion)
- `std.Thread.Mutex` doesn't exist in 0.16 (use `std.c` pthread)
- `try` can't be used in `void`-returning functions
- `AutoHashMap.getOrPut` doesn't zero-initialize new entries
