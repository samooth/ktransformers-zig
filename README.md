# ktransformers-zig

High-performance CPU inference kernels for Mixture-of-Experts (MoE) models, ported from [ktransformers](https://github.com/kvcache-ai/ktransformers) to Zig.

## Status

**Build: WORKING** - `zig build` produces `zig-out/lib/libkt_kernel_ext.so`

## Overview

This project ports the core CPU kernels from ktransformers (C++/AMX/AVX2/AVX512) to Zig, maintaining API compatibility with the Python `kt_kernel` package while leveraging Zig's comptime metaprogramming, SIMD intrinsics, and memory safety.

**Target**: Drop-in replacement for `kt_kernel_ext.so` with 6 CPU variants (AVX2, AVX512_base, AVX512_VNNI, AVX512_VBMI, AVX512_BF16, AMX).

## Architecture

```
ktransformers-zig/
├── build.zig              # Build system with multi-variant support
├── build.zig.zon          # Package configuration
├── include/
│   └── kt_kernel.h        # C API contract (112 symbols, gated by tools/verify_abi.py)
├── bindings/              # pybind11 shim (drop-in for kt_kernel_ext)
├── python/                # ctypes bindings + wheel packaging
├── src/
│   ├── main.zig           # C API exports (kt_*)
│   ├── root.zig           # module hub + comptime fn-refs
│   ├── io/gguf.zig        # GGUF v3 parser
│   ├── runtime/           # Thread pool (NUMA-pinned), task queue, memory, CPU/L1-L3 detect
│   ├── numa/              # NUMA syscalls (topology, mbind, affinity)
│   ├── mla/               # MLA attention (DeepSeek-V2/V3) + KV cache
│   └── kernels/
│       ├── arch/          # AMX/NEON intrinsics + runtime CPU guards
│       ├── amx/           # GEMM kernels: BF16, INT8/INT4, FP8, MXFP4/8,
│       │                  #   + all 15 GGML formats (Q8_0, Q*_K, IQ*_*)
│       │                  #   with quantize+dequantize+GEMM per format
│       ├── attn/          # vanilla MHA engine
│       ├── qwen3/         # Qwen3-MoE model orchestration
│       └── moe/           # MoE (DeepSeek routing), LlamaMoe (GGML-quantized
│                          #   GGUF MoE, all 16 weight formats), SFT/LoRA
├── tests/                 # 210+ tests across the suites (zig build test)
├── bench/                 # GEMM + MoE micro-benchmarks (ReleaseFast)
└── tools/                 # ABI gates: verify_abi.py, audit_layout.py
```

## Features

- **CPU Variants**: AVX2, AVX512_base, AVX512_VNNI, AVX512_VBMI, AVX512_BF16, AMX (+ aarch64 NEON cross-build)
- **Quantization**: BF16, INT8 (per-row), INT4 (GPTQ), FP8 E4M3, MXFP4, MXFP8, and **all 15 GGML block formats** (Q8_0, Q2_K–Q8_K, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, IQ1_S/M) — quantize + dequantize + GEMM, byte-exact vs llama.cpp
- **MoE backends**: DeepSeek-V3 group-top2 routing, LlamaMoe for GGUF checkpoints (16 weight formats per projection), SFT/LoRA forward+backward
- **Attention**: MLA (DeepSeek-V2/V3, paged/batched) + vanilla MHA
- **Model orchestration**: DeepseekV3 and Qwen3-MoE decoder stacks (C API `kt_dsv3_*` / `kt_qwen3moe_*`)
- **Runtime**: NUMA-aware worker pool (sched_setaffinity pinning, mbind), injectable allocator (`kt_set_default_allocator`)
- **Python Compatible**: C API matches `kt_kernel_ext.so` (pybind11 shim + ctypes path)

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

## Attribution / NOTICE

This project is a Zig port of KTransformers (https://github.com/kvcache-ai/KTransformers),
originally developed by MADSys Lab @ Tsinghua University, Approaching.AI, and 9#AISoft.
The original work is licensed under the Apache License, Version 2.0.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.