// Root module re-exports all submodules for proper Zig module resolution
// Also exports C API functions from main.zig so they are linked into the library

pub const memory = @import("runtime/memory.zig");
pub const worker_pool = @import("runtime/worker_pool.zig");
pub const task_queue = @import("runtime/task_queue.zig");
pub const cpu_detect = @import("runtime/cpu_detect.zig");
pub const numa = @import("numa/numa.zig");
pub const amx = @import("kernels/arch/amx.zig");
pub const buffers = @import("kernels/amx/buffers.zig");
pub const gemm_bf16 = @import("kernels/amx/gemm_224_bf16.zig");
pub const gemm_int4 = @import("kernels/amx/gemm_224_int4.zig");
pub const gemm_fp8 = @import("kernels/amx/gemm_224_fp8.zig");
pub const gemm_int8 = @import("kernels/amx/gemm_224_int8.zig");
pub const moe = @import("kernels/moe/moe.zig");
pub const moe_sft = @import("kernels/moe/moe_sft.zig");
pub const mla_config = @import("mla/mla_config.zig");
pub const mla_cache = @import("mla/mla_cache.zig");
pub const mla_core = @import("mla/mla_core.zig");
pub const lora_kernels = @import("kernels/amx/lora_kernels.zig");

// MXFP4/MXFP8 kernels: re-exports alone are not enough — without a
// `comptime { _ = &fn; }` block the .so contains zero MXFP code (lazy
// analysis strips the module). See "Integrating a Standalone Zig Module
// (MLA)" in LESSONS_ZIG.md for the mechanism this is built on.
pub const gemm_mxfp4 = @import("kernels/amx/gemm_224_mxfp4.zig");
pub const gemm_mxfp8 = @import("kernels/amx/gemm_224_mxfp8.zig");

// GGML quantization kernels (Phase 1: Q8_0). Same lazy-analysis rule applies.
pub const gemm_q8_0 = @import("kernels/amx/gemm_224_q8_0.zig");
pub const gemm_q4_k = @import("kernels/amx/gemm_224_q4_k.zig");
comptime {
    _ = &gemm_q8_0.BlockQ8_0;
    _ = &gemm_q8_0.f32_to_f16;
    _ = &gemm_q8_0.f16_to_f32;
    _ = &gemm_q8_0.quantizeRowQ8_0;
    _ = &gemm_q8_0.dequantizeRowQ8_0;
    _ = &gemm_q8_0.dequantizeRowQ8_0ToBF16;
    _ = &gemm_q8_0.gemmQ8_0Scalar;
    _ = &gemm_q4_k.BlockQ4_K;
    _ = &gemm_q4_k.f32_to_f16;
    _ = &gemm_q4_k.f16_to_f32;
    _ = &gemm_q4_k.nearestInt;
    _ = &gemm_q4_k.getScaleMinK4;
    _ = &gemm_q4_k.quantizeRowQ4_K;
    _ = &gemm_q4_k.dequantizeRowQ4_K;
    _ = &gemm_q4_k.dequantizeRowQ4_KToBF16;
    _ = &gemm_q4_k.gemmQ4_KScalar;
}
comptime {
    _ = &gemm_mxfp4.GemmKernel224MXFP4.gemmFullTile;
    _ = &gemm_mxfp4.MXFP4Block;
    _ = &gemm_mxfp4.fp4e2m1_to_f32;
    _ = &gemm_mxfp4.MXFP4BufferB.fromMatBF16;
    _ = &gemm_mxfp8.GemmKernel224MXFP8.gemmFullTile;
    _ = &gemm_mxfp8.MXFP8Block;
    _ = &gemm_mxfp8.fp8e4m3_to_f32;
    _ = &gemm_mxfp8.MXFP8BufferB.fromMatBF16;
}

// Force full semantic analysis of the MLA modules. A bare `pub const`
// namespace re-export does NOT analyze the module's declarations (lazy
// analysis); referencing the functions themselves in comptime does.
// Without this, the .so contains zero MLA code despite the re-exports.
comptime {
    _ = &mla_core.rmsNorm;
    _ = &mla_core.matmulF32;
    _ = &mla_core.matmulF32Offset;
    _ = &mla_core.MlaEngine.init;
    _ = &mla_core.MlaEngine.deinit;
    _ = &mla_core.MlaEngine.forward;
    _ = &mla_core.MlaEngine.decode;
    _ = &mla_core.MlaEngine.resetCache;
}

// Force main.zig (C API: export fn kt_*) to be semantically analyzed and
// exported into the shared library. A plain `const _ = @import(...)` is
// lazily stripped by the compiler and produces zero exported symbols;
// wrapping in comptime forces the analysis. Verified experimentally.
comptime {
    _ = @import("main.zig");
}
