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
pub const deepseekv3_layer = @import("kernels/moe/deepseekv3_layer.zig");
pub const deepseekv3_model = @import("kernels/moe/deepseekv3_model.zig");
pub const mla_config = @import("mla/mla_config.zig");
pub const mla_cache = @import("mla/mla_cache.zig");
pub const mla_core = @import("mla/mla_core.zig");
pub const lora_kernels = @import("kernels/amx/lora_kernels.zig");
pub const gemm_bf16_neon = @import("kernels/arch/gemm_224_bf16_neon.zig");

// MXFP4/MXFP8 kernels: re-exports alone are not enough — without a
// `comptime { _ = &fn; }` block the .so contains zero MXFP code (lazy
// analysis strips the module). See "Integrating a Standalone Zig Module
// (MLA)" in LESSONS_ZIG.md for the mechanism this is built on.
pub const gemm_mxfp4 = @import("kernels/amx/gemm_224_mxfp4.zig");
pub const gemm_mxfp8 = @import("kernels/amx/gemm_224_mxfp8.zig");

// GGML quantization kernels (Phase 1: Q8_0). Same lazy-analysis rule applies.
pub const gemm_q8_0 = @import("kernels/amx/gemm_224_q8_0.zig");
pub const gemm_q4_k = @import("kernels/amx/gemm_224_q4_k.zig");
pub const gemm_q6_k = @import("kernels/amx/gemm_224_q6_k.zig");
pub const gemm_q5_k = @import("kernels/amx/gemm_224_q5_k.zig");
pub const gemm_q8_k = @import("kernels/amx/gemm_224_q8_k.zig");
pub const gemm_q2_k = @import("kernels/amx/gemm_224_q2_k.zig");
pub const gemm_q3_k = @import("kernels/amx/gemm_224_q3_k.zig");
pub const gemm_iq4_xs = @import("kernels/amx/gemm_224_iq4_xs.zig");
pub const gemm_iq4_nl = @import("kernels/amx/gemm_224_iq4_nl.zig");
pub const gemm_iq2_xxs = @import("kernels/amx/gemm_224_iq2_xxs.zig");
pub const gemm_iq3_xxs = @import("kernels/amx/gemm_224_iq3_xxs.zig");
pub const iq2xs_init = @import("kernels/amx/iq2xs_init.zig");
pub const iq2_quantize = @import("kernels/amx/iq2_quantize.zig");
pub const gemm_iq2_xs = @import("kernels/amx/gemm_224_iq2_xs.zig");
pub const gemm_iq2_s = @import("kernels/amx/gemm_224_iq2_s.zig");
pub const gemm_iq3_s = @import("kernels/amx/gemm_224_iq3_s.zig");
pub const gemm_iq1_s = @import("kernels/amx/gemm_224_iq1_s.zig");
pub const gemm_iq1_m = @import("kernels/amx/gemm_224_iq1_m.zig");
pub const llama_moe = @import("kernels/moe/llama_moe.zig");
pub const iq3xs_init = @import("kernels/amx/iq3xs_init.zig");
pub const iq3_quantize = @import("kernels/amx/iq3_quantize.zig");
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
    _ = &gemm_q6_k.BlockQ6_K;
    _ = &gemm_q6_k.f32_to_f16;
    _ = &gemm_q6_k.f16_to_f32;
    _ = &gemm_q6_k.nearestInt;
    _ = &gemm_q6_k.quantizeRowQ6_K;
    _ = &gemm_q6_k.dequantizeRowQ6_K;
    _ = &gemm_q6_k.dequantizeRowQ6_KToBF16;
    _ = &gemm_q6_k.gemmQ6_KScalar;
    _ = &gemm_q5_k.BlockQ5_K;
    _ = &gemm_q5_k.f32_to_f16;
    _ = &gemm_q5_k.f16_to_f32;
    _ = &gemm_q5_k.nearestInt;
    _ = &gemm_q5_k.getScaleMinK4;
    _ = &gemm_q5_k.quantizeRowQ5_K;
    _ = &gemm_q5_k.dequantizeRowQ5_K;
    _ = &gemm_q5_k.dequantizeRowQ5_KToBF16;
    _ = &gemm_q5_k.gemmQ5_KScalar;
    _ = &gemm_q8_k.BlockQ8_K;
    _ = &gemm_q8_k.nearestInt;
    _ = &gemm_q8_k.quantizeRowQ8_K;
    _ = &gemm_q8_k.dequantizeRowQ8_K;
    _ = &gemm_q8_k.dequantizeRowQ8_KToBF16;
    _ = &gemm_q8_k.gemmQ8_KScalar;
    _ = &gemm_q2_k.BlockQ2_K;
    _ = &gemm_q2_k.f32_to_f16;
    _ = &gemm_q2_k.f16_to_f32;
    _ = &gemm_q2_k.nearestInt;
    _ = &gemm_q2_k.quantizeRowQ2_K;
    _ = &gemm_q2_k.dequantizeRowQ2_K;
    _ = &gemm_q2_k.dequantizeRowQ2_KToBF16;
    _ = &gemm_q2_k.gemmQ2_KScalar;
    _ = &gemm_q3_k.BlockQ3_K;
    _ = &gemm_q3_k.f32_to_f16;
    _ = &gemm_q3_k.f16_to_f32;
    _ = &gemm_q3_k.nearestInt;
    _ = &gemm_q3_k.quantizeRowQ3_K;
    _ = &gemm_q3_k.dequantizeRowQ3_K;
    _ = &gemm_q3_k.dequantizeRowQ3_KToBF16;
    _ = &gemm_q3_k.gemmQ3_KScalar;
    _ = &gemm_iq4_xs.BlockIQ4_XS;
    _ = &gemm_iq4_xs.KVALUES_IQ4NL;
    _ = &gemm_iq4_xs.f32_to_f16;
    _ = &gemm_iq4_xs.f16_to_f32;
    _ = &gemm_iq4_xs.quantizeRowIQ4_XS;
    _ = &gemm_iq4_xs.dequantizeRowIQ4_XS;
    _ = &gemm_iq4_xs.dequantizeRowIQ4_XSToBF16;
    _ = &gemm_iq4_xs.gemmIQ4_XSScalar;
    _ = &gemm_iq4_nl.BlockIQ4_NL;
    _ = &gemm_iq4_nl.f32_to_f16;
    _ = &gemm_iq4_nl.f16_to_f32;
    _ = &gemm_iq4_nl.dequantizeRowIQ4_NL;
    _ = &gemm_iq4_nl.dequantizeRowIQ4_NLToBF16;
    _ = &gemm_iq4_nl.gemmIQ4_NLScalar;
    _ = &gemm_iq2_xxs.BlockIQ2_XXS;
    _ = &gemm_iq2_xxs.IQ2XXS_GRID;
    _ = &gemm_iq2_xxs.KSIGNS_IQ2XS;
    _ = &gemm_iq2_xxs.KMASK_IQ2XS;
    _ = &gemm_iq2_xxs.dequantizeRowIQ2_XXS;
    _ = &gemm_iq2_xxs.dequantizeRowIQ2_XXSToBF16;
    _ = &gemm_iq2_xxs.gemmIQ2_XXSScalar;
    _ = &gemm_iq3_xxs.BlockIQ3_XXS;
    _ = &gemm_iq3_xxs.IQ3XXS_GRID;
    _ = &gemm_iq3_xxs.dequantizeRowIQ3_XXS;
    _ = &gemm_iq3_xxs.dequantizeRowIQ3_XXSToBF16;
    _ = &gemm_iq3_xxs.gemmIQ3_XXSScalar;
    _ = &iq2xs_init.initIq2XsData;
    _ = &iq2xs_init.freeIq2XsData;
    _ = &iq2xs_init.KGRID_2BIT_256;
    _ = &iq2xs_init.KMAP_SIZE;
    _ = &iq2_quantize.quantizeRowIQ2_XXS_WithInit;
    _ = &gemm_iq2_xs.BlockIQ2_XS;
    _ = &gemm_iq2_xs.IQ2XS_GRID;
    _ = &gemm_iq2_xs.dequantizeRowIQ2_XS;
    _ = &gemm_iq2_xs.dequantizeRowIQ2_XSToBF16;
    _ = &gemm_iq2_xs.gemmIQ2_XSScalar;
    _ = &gemm_iq2_xs.quantizeRowIQ2_XS_WithInit;
    _ = &iq2xs_init.KGRID_2BIT_512;
    _ = &iq2xs_init.initIq2XsData512;
    _ = &gemm_iq2_s.BlockIQ2_S;
    _ = &gemm_iq2_s.IQ2S_GRID;
    _ = &gemm_iq2_s.dequantizeRowIQ2_S;
    _ = &gemm_iq2_s.dequantizeRowIQ2_SToBF16;
    _ = &gemm_iq2_s.gemmIQ2_SScalar;
    _ = &gemm_iq2_s.quantizeRowIQ2_S_WithInit;
    _ = &iq2xs_init.KGRID_2BIT_1024;
    _ = &iq2xs_init.initIq2SData;
    _ = &gemm_iq3_s.BlockIQ3_S;
    _ = &gemm_iq3_s.IQ3S_GRID;
    _ = &gemm_iq3_s.dequantizeRowIQ3_S;
    _ = &gemm_iq3_s.dequantizeRowIQ3_SToBF16;
    _ = &gemm_iq3_s.gemmIQ3_SScalar;
    _ = &gemm_iq3_s.quantizeRowIQ3_S_WithInit;
    _ = &iq3xs_init.KGRID_Q3XS_512;
    _ = &iq3xs_init.initIq3SData;
    _ = &gemm_iq1_s.BlockIQ1_S;
    _ = &gemm_iq1_s.IQ1S_GRID;
    _ = &gemm_iq1_s.dequantizeRowIQ1_S;
    _ = &gemm_iq1_s.dequantizeRowIQ1_SToBF16;
    _ = &gemm_iq1_s.gemmIQ1_SScalar;
    _ = &gemm_iq1_s.quantizeRowIQ1_S_WithInit;
    _ = &iq2xs_init.KGRID_1BIT_2048;
    _ = &iq2xs_init.initIq1SData;
    _ = &llama_moe.LlamaMoe.init;
    _ = &llama_moe.LlamaMoe.forwardOne;
    _ = &llama_moe.rowBytes;
    _ = &gemm_iq1_m.BlockIQ1_M;
    _ = &gemm_iq1_m.extractScale;
    _ = &gemm_iq1_m.dequantizeRowIQ1_M;
    _ = &gemm_iq1_m.dequantizeRowIQ1_MToBF16;
    _ = &gemm_iq1_m.gemmIQ1_MScalar;
    _ = &gemm_iq1_m.quantizeRowIQ1_M_WithInit;
    _ = &iq3xs_init.initIq3XsData;
    _ = &iq3xs_init.freeIq3XsData;
    _ = &iq3xs_init.KGRID_Q3XS_256;
    _ = &iq3xs_init.KMAP_SIZE;
    _ = &iq3_quantize.quantizeRowIQ3_XXS_WithInit;
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

// Force full semantic analysis of the NUMA modules. Same lazy-analysis
// rule as MLA. A3: the runtime/worker_pool.zig imports the NUMA
// syscalls (sched_setaffinity, mbind) at comptime, so the analysis
// propagates naturally to setThreadAffinity and bindMemory. We force
// those (and the NumaAllocator vtable) here so external users linking
// against the .so can reach them without depending on internal module
// paths. The topology detector and NUMA allocator helpers were revived
// from Zig 0.16 API rot (std.fs.cwd → std.Io, std.mem.page_size →
// std.heap.page_size_min, runtime-aligned rawAlloc) — they now compile
// and are forced here as well.
comptime {
    _ = &numa.memory.setThreadAffinity;
    _ = &numa.memory.pinThreadToCpu;
    _ = &numa.memory.getThreadAffinity;
    _ = &numa.memory.bindMemory;
    _ = &numa.memory.setThreadNumaPolicy;
    _ = &numa.memory.getThreadNumaPolicy;
    _ = &numa.memory.allocNuma;
    _ = &numa.memory.NumaAllocator.init;
    _ = &numa.topology.NumaTopology.detect;
    _ = &numa.topology.NumaTopology.deinit;
    _ = &numa.topology.NumaTopology.nodeForCpu;
    _ = &numa.topology.NumaTopology.cpusForNode;
    _ = &numa.worker.NumaWorker.spawn;
}

// Force main.zig (C API: export fn kt_*) to be semantically analyzed and
// exported into the shared library. A plain `const _ = @import(...)` is
// lazily stripped by the compiler and produces zero exported symbols;
// wrapping in comptime forces the analysis. Verified experimentally.
comptime {
    _ = @import("main.zig");
}

// Force analysis of the DeepseekV3DecoderLayer orchestration module.
comptime {
    _ = &deepseekv3_layer.DeepseekV3DecoderLayer.init;
    _ = &deepseekv3_layer.DeepseekV3DecoderLayer.forward;
    _ = &deepseekv3_layer.DeepseekV3DecoderLayer.deinit;
    _ = &deepseekv3_model.DeepseekV3Model.init;
    _ = &deepseekv3_model.DeepseekV3Model.forward;
    _ = &deepseekv3_model.DeepseekV3Model.deinit;
    _ = &deepseekv3_model.DeepseekV3ForCausalLM.init;
    _ = &deepseekv3_model.DeepseekV3ForCausalLM.forward;
    _ = &deepseekv3_model.DeepseekV3ForCausalLM.deinit;
}

// Force analysis of the Qwen3 MoE orchestration (MHA engine + decoder
// layer + model + CausalLM). Same lazy-analysis rule as MLA / DSV3.
pub const mha = @import("kernels/attn/mha.zig");
pub const gguf = @import("io/gguf.zig");
pub const qwen3_layer = @import("kernels/qwen3/qwen3_layer.zig");
pub const qwen3_model = @import("kernels/qwen3/qwen3_model.zig");
pub const llamafile_moe = @import("kernels/moe/llamafile_moe.zig");
comptime {
    _ = &mha.MhaEngine.init;
    _ = &mha.MhaEngine.deinit;
    _ = &mha.MhaEngine.forward;
    _ = &mha.MhaEngine.decode;
    _ = &mha.MhaKvCache.init;
    _ = &mha.MhaKvCache.deinit;
    _ = &mha.matmulF32;
    _ = &mha.rmsNormInline;
    _ = &mha.softmaxInPlace;
    _ = &gguf.parse;
    _ = &gguf.Header.findTensor;
    _ = &qwen3_layer.Qwen3MoeDecoderLayer.init;
    _ = &qwen3_layer.Qwen3MoeDecoderLayer.forward;
    _ = &qwen3_layer.Qwen3MoeDecoderLayer.deinit;
    _ = &qwen3_model.Qwen3MoeModel.init;
    _ = &qwen3_model.Qwen3MoeModel.forward;
    _ = &qwen3_model.Qwen3MoeModel.deinit;
    _ = &qwen3_model.Qwen3MoeForCausalLM.init;
    _ = &qwen3_model.Qwen3MoeForCausalLM.forward;
    _ = &qwen3_model.Qwen3MoeForCausalLM.deinit;
    _ = &llamafile_moe.LlamaMoe.init;
    _ = &llamafile_moe.LlamaMoe.deinit;
    _ = &llamafile_moe.LlamaMoe.forward;
    _ = &llamafile_moe.LlamaMoe.loadWeights;
}

// NEON Phase-2: force analysis of the NEON-vectorized BF16 GEMM. The
// cross-build (`zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu`)
// lowers the comptime @Vector(4,f32) to native NEON fmul/fmla. On
// x86_64 the same source lowers to SSE 128-bit (half the lane count
// of the x86 A1 kernel, but algorithmically correct — the
// comptime gate `if (neon.NeonFeatures.available)` ensures only
// the relevant path emits). Without these fn-refs the .so contains
// zero NEON code (lazy analysis strips the module on aarch64 just
// like it does for MLA / lora_kernels).
comptime {
    _ = &gemm_bf16_neon.GemmExpertBF16NEON.gemmExpert;
    _ = &gemm_bf16_neon.GemmExpertBF16NEON.gemmExpertNeon;
    _ = &gemm_bf16_neon.GemmExpertBF16NEON.gemmExpertScalar;
    _ = &gemm_bf16_neon.GemmExpertBF16NEON.VecF32;
    _ = &gemm_bf16_neon.GemmExpertBF16NEON.reduceAdd;
}
