# ktransformers-zig

High-performance CPU inference kernels for Mixture-of-Experts (MoE) models, ported from [ktransformers](https://github.com/kvcache-ai/ktransformers) to Zig.

## Overview

This project ports the core CPU kernels from ktransformers (C++/AMX/AVX2/AVX512) to Zig, maintaining API compatibility with the Python `kt_kernel` package while leveraging Zig's comptime metaprogramming, SIMD intrinsics, and memory safety.

**Target**: Drop-in replacement for `kt_kernel_ext.so` with 6 CPU variants (AVX2, AVX512_base, AVX512_VNNI, AVX512_VBMI, AVX512_BF16, AMX).

## Architecture

```
ktransformers-zig/
├── build.zig              # Build system with multi-variant support
├── build.zig.zon          # Package configuration
├── include/
│   └── kt_kernel.h        # C API header (matches ktransformers)
├── src/
│   ├── main.zig           # C API exports
│   ├── runtime/           # Thread pool, task queue, memory, CPU detection
│   │   ├── memory.zig
│   │   ├── worker_pool.zig
│   │   ├── task_queue.zig
│   │   └── cpu_detect.zig
│   └── kernels/
│       ├── arch/          # Architecture-specific intrinsics
│       │   └── amx.zig    # AMX tile instructions
│       ├── amx/           # AMX GEMM kernels
│       │   ├── buffers.zig
│       │   ├── gemm_224_bf16.zig
│       │   └── gemm_224_int8.zig
│       └── moe/           # MoE orchestration
│           └── moe.zig
└── tests/
    └── kernels/
        └── test_kernels.zig
```

## Features

- **CPU Variants**: AVX2, AVX512_base, AVX512_VNNI, AVX512_VBMI, AVX512_BF16, AMX
- **Quantization**: BF16, INT8 (per-row), INT4 (GPTQ), FP8 E4M3, MXFP4, MXFP8
- **Tensor Parallelism**: NUMA-aware weight splitting across sockets
- **Python Compatible**: C API matches `kt_kernel_ext.so` for pybind11

## Building

```bash
# Build default variant (AVX2)
zig build

# Build specific variant
zig build -Dvariant=amx
zig build -Dvariant=avx512_bf16

# Run tests
zig build test

# Install artifacts
zig build install
```

Artifacts installed to `zig-out/lib/kt_kernel_ext_<variant>.so`

## Python Integration

The C API in `include/kt_kernel.h` matches the original ktransformers C++ API. A minimal pybind11 wrapper can load the Zig-built `.so` as a drop-in replacement:

```python
# Original
from kt_kernel import MOE

# With Zig build (after installing wheel)
from kt_kernel import MOE  # same API
```

## Requirements

- Zig 0.16+
- x86_64 CPU with AVX2+ (AMX requires Sapphire Rapids+)
- Linux (primary target)

## Project Status

See [TODO.md](TODO.md) for current status and roadmap.

## License

Apache-2.0 (same as ktransformers)