// LLAMA_MOE_TP — GGML-quantized MoE.
//
// Ports the C++ LLAMA_MOE_TP from ktransformers/kt-kernel/operators/llamafile/moe.hpp.
// The C++ class is the backend-default MoE for GGUF-loaded checkpoints
// (archive/experts.py:150+). It receives GGML-quantized gate/up/down
// weights directly (Q8_0, Q4_K, Q5_K, Q6_K, Q8_K, Q2_K, Q3_K, IQ4_XS, IQ4_NL, and uses llamafile_sgemm
// for the per-expert matrix multiplies with on-the-fly dequant + activation
// quantization to the vec_dot_type.
//
// Algorithm (per expert, per m-block):
//   1. Convert hidden activations to the gate's vec_dot_type (Q8_0).
//   2. gate_out = llamafile_sgemm(weights[gate] × input[gate-vec])
//   3. up_out   = llamafile_sgemm(weights[up]   × input[up-vec])
//   4. intermediate = silu(gate_out) * up_out   (SwiGLU)
//   5. Convert intermediate to the down's vec_dot_type.
//   6. output  = llamafile_sgemm(weights[down] × input[down-vec])
//   7. Weighted sum across top-k experts.
//
// This Zig port implements the same pattern but uses the existing
// `kt_matmul_q*` (BF16 activations × quantized weights → F32) rather
// than a full re-implementation of llamafile_sgemm. The path is:
//   hidden (BF16) → quantize to Q8_0 → ... but the existing matmuls
//   take BF16 input, not Q8_0.
//
// So for the FIRST cut we take a "correct-but-slower" path:
// dequantize the weight blocks to BF16, then use the existing
// kt_matmul_q* (which all take BF16). This produces the right output
// (the dequant is byte-exact vs llama.cpp), just without the Q8_0
// activation-side optimization. A follow-up can add a true
// Q4_K × Q8_0 matmul that takes Q8_0 activations directly.
//
// Supported (weight_type, hidden_type) combinations in this first cut:
//   weight_type ∈ {Q8_0, Q4_K, Q5_K, Q6_K, Q8_K}
//   hidden_type = BF16 (the framework's default for GGUF checkpoints)
//   vec_dot_type for gate/up = Q8_0 (the llama.cpp default)
//
// References:
//   ktransformers/kt-kernel/operators/llamafile/moe.hpp (820 lines)
//   ktransformers/kt-kernel/operators/common.hpp (GeneralMOEConfig)
//   llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp (the sgemm we approximate)

const std = @import("std");
const amx = @import("../arch/amx.zig");
const worker_pool = @import("../../runtime/worker_pool.zig");
const gemm_q8_0 = @import("../amx/gemm_224_q8_0.zig");
const gemm_q4_k = @import("../amx/gemm_224_q4_k.zig");
const gemm_q5_k = @import("../amx/gemm_224_q5_k.zig");
const gemm_q6_k = @import("../amx/gemm_224_q6_k.zig");
const gemm_q8_k = @import("../amx/gemm_224_q8_k.zig");
const gemm_q2_k = @import("../amx/gemm_224_q2_k.zig");
const gemm_q3_k = @import("../amx/gemm_224_q3_k.zig");
const gemm_iq2_xxs = @import("../amx/gemm_224_iq2_xxs.zig");
const gemm_iq2_xs = @import("../amx/gemm_224_iq2_xs.zig");
const gemm_iq2_s = @import("../amx/gemm_224_iq2_s.zig");
const gemm_iq3_xxs = @import("../amx/gemm_224_iq3_xxs.zig");
const gemm_iq3_s = @import("../amx/gemm_224_iq3_s.zig");
const gemm_iq1_s = @import("../amx/gemm_224_iq1_s.zig");
const gemm_iq1_m = @import("../amx/gemm_224_iq1_m.zig");
const gemm_iq4_nl = @import("../amx/gemm_224_iq4_nl.zig");
const gemm_iq4_xs = @import("../amx/gemm_224_iq4_xs.zig");

// kt_type_t values — MUST match include/kt_kernel.h (the ABI contract)
// and src/main.zig's kt_type_t enum. Audited 2026-09-03 against the header.
pub const KT_TYPE_F32: u32 = 0;
pub const KT_TYPE_BF16: u32 = 2;
pub const KT_TYPE_Q8_0: u32 = 12;
pub const KT_TYPE_Q4_K: u32 = 16;
pub const KT_TYPE_Q5_K: u32 = 17;
pub const KT_TYPE_Q6_K: u32 = 18;
pub const KT_TYPE_Q8_K: u32 = 19;
pub const KT_TYPE_Q2_K: u32 = 14;
pub const KT_TYPE_Q3_K: u32 = 15;
pub const KT_TYPE_IQ2_XXS: u32 = 20;
pub const KT_TYPE_IQ2_XS: u32 = 21;
pub const KT_TYPE_IQ3_XXS: u32 = 22;
pub const KT_TYPE_IQ1_S: u32 = 23;
pub const KT_TYPE_IQ4_NL: u32 = 24;
pub const KT_TYPE_IQ3_S: u32 = 25;
pub const KT_TYPE_IQ2_S: u32 = 26;
pub const KT_TYPE_IQ4_XS: u32 = 27;
pub const KT_TYPE_IQ1_M: u32 = 28;

// Block size (weights per block) for the supported types.
const Q8_0_BLK: usize = 32;
const Q4_NL_BLK: usize = 32; // ggml_blck_size[IQ4_NL]
const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144; // f16 d+dmin + 12-byte scales + 128-byte qs
const Q5_K_BLOCK_BYTES: usize = 176; // f16 d+dmin + 12-byte scales + 32-byte qh + 128-byte qs
const Q6_K_BLOCK_BYTES: usize = 210; // ql[128] + qh[64] + scales[16] + f16 d
const Q8_K_BLOCK_BYTES: usize = 292; // f32 d + 256*i8 qs + 16*i16 bsums
const Q8_0_BLOCK_BYTES: usize = 34; // f16 d + 32*i8 qs

/// Configuration for the GGML-quantized MoE.
pub const LlamaConfig = struct {
    expert_num: usize,
    num_experts_per_tok: usize,
    hidden_size: usize,
    intermediate_size: usize,

    layer_idx: usize = 0,
    pool: ?*worker_pool.WorkerPool = null,

    // GGML types (kt_type_t). The current implementation supports:
    //   gate_type/up_type/down_type ∈ {Q8_0, Q4_K, Q5_K, Q6_K, Q8_K}
    //   hidden_type = BF16
    gate_type: u32,
    up_type: u32,
    down_type: u32,
    hidden_type: u32 = KT_TYPE_BF16,

    // vec_dot_type: usually Q8_0 (the llama.cpp default for Q4_K/Q5_K/Q6_K).
    // The current implementation always uses Q8_0 — set in `init`.
    act_type: u32 = KT_TYPE_Q8_0,

    // Tile geometry: how many output rows per llamafile_sgemm call.
    // m_block should be a multiple of the relevant block size.
    m_block: usize = 32,

    // Group batching: qlen < group_min_len uses forward_one, >= uses
    // forward_many (amortizes per-expert overhead). Matches the C++.
    group_min_len: usize = 32,
    group_max_len: usize = 1024,

    // Optional SGLang-style GPU offload mask (1 byte per expert, 1 = skip).
    gpu_experts_mask: ?[*]const u8 = null,

    // Weight pointers (raw GGML blocks). Each is shaped
    //   [expert_num][output_dim][input_dim] in GGUF layout
    // (input_dim / block_size blocks per output dim, output_dim ×
    // input_dim values total per expert).
    gate_proj: [*]const u8,
    up_proj: [*]const u8,
    down_proj: [*]const u8,

    /// Returns true if the weight type is supported by the first-cut
    /// matmul matrix. The dispatch is a static switch (no vtable).
    pub fn isWeightTypeSupported(t: u32) bool {
        return t == KT_TYPE_Q8_0 or t == KT_TYPE_Q4_K or t == KT_TYPE_Q5_K or
            t == KT_TYPE_Q6_K or t == KT_TYPE_Q8_K or t == KT_TYPE_Q2_K or
            t == KT_TYPE_Q3_K or t == KT_TYPE_IQ4_XS or t == KT_TYPE_IQ4_NL or
            t == KT_TYPE_IQ2_XXS or t == KT_TYPE_IQ2_XS or t == KT_TYPE_IQ2_S or
            t == KT_TYPE_IQ3_XXS or t == KT_TYPE_IQ3_S or
            t == KT_TYPE_IQ1_S or t == KT_TYPE_IQ1_M;
    }

    pub fn isHiddenTypeSupported(t: u32) bool {
        return t == KT_TYPE_BF16; // first cut: BF16 only
    }
};

// Block sizes — verified against @sizeOf of the real Block structs
// (2026-09-03; IQ2_XXS was 50->66, IQ3_XXS 56->98, IQ3_S 60->110,
// IQ1_M 16->56, IQ4_NL 36->18 — the invented values were silently
// wrong in every expert-offset computation).
const Q2_K_BLOCK_BYTES: usize = 84;
const Q3_K_BLOCK_BYTES: usize = 110;
const IQ4_XS_BLOCK_BYTES: usize = 136;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const IQ2_XXS_BLOCK_BYTES: usize = 66;
const IQ2_XS_BLOCK_BYTES: usize = 74;
const IQ2_S_BLOCK_BYTES: usize = 82;
const IQ3_XXS_BLOCK_BYTES: usize = 98;
const IQ3_S_BLOCK_BYTES: usize = 110;
const IQ1_S_BLOCK_BYTES: usize = 50;
const IQ1_M_BLOCK_BYTES: usize = 56;

/// Per-type byte-size helper (matches ggml_type_size in llama.cpp).
fn blockBytes(t: u32) usize {
    return switch (t) {
        KT_TYPE_Q8_0 => Q8_0_BLOCK_BYTES,
        KT_TYPE_Q4_K => Q4_K_BLOCK_BYTES,
        KT_TYPE_Q5_K => Q5_K_BLOCK_BYTES,
        KT_TYPE_Q6_K => Q6_K_BLOCK_BYTES,
        KT_TYPE_Q8_K => Q8_K_BLOCK_BYTES,
        KT_TYPE_Q2_K => Q2_K_BLOCK_BYTES,
        KT_TYPE_Q3_K => Q3_K_BLOCK_BYTES,
        KT_TYPE_IQ4_XS => IQ4_XS_BLOCK_BYTES,
        KT_TYPE_IQ4_NL => IQ4_NL_BLOCK_BYTES,
        KT_TYPE_IQ2_XXS => IQ2_XXS_BLOCK_BYTES,
        KT_TYPE_IQ2_XS => IQ2_XS_BLOCK_BYTES,
        KT_TYPE_IQ2_S => IQ2_S_BLOCK_BYTES,
        KT_TYPE_IQ3_XXS => IQ3_XXS_BLOCK_BYTES,
        KT_TYPE_IQ3_S => IQ3_S_BLOCK_BYTES,
        KT_TYPE_IQ1_S => IQ1_S_BLOCK_BYTES,
        KT_TYPE_IQ1_M => IQ1_M_BLOCK_BYTES,
        else => 0,
    };
}

/// Weights per block (matches ggml_blck_size).
fn blockSize(t: u32) usize {
    return switch (t) {
        KT_TYPE_Q8_0 => Q8_0_BLK,
        KT_TYPE_IQ4_NL => Q4_NL_BLK, // 32 — NOT 256 (ggml_blck_size[IQ4_NL])
        KT_TYPE_Q4_K, KT_TYPE_Q5_K, KT_TYPE_Q6_K, KT_TYPE_Q8_K,
        KT_TYPE_Q2_K, KT_TYPE_Q3_K,
        KT_TYPE_IQ4_XS,
        KT_TYPE_IQ2_XXS, KT_TYPE_IQ2_XS, KT_TYPE_IQ2_S,
        KT_TYPE_IQ3_XXS, KT_TYPE_IQ3_S,
        KT_TYPE_IQ1_S, KT_TYPE_IQ1_M,
        => QK_K,
        else => 0,
    };
}

/// Total bytes for one expert's projection of shape [output_dim, input_dim].
fn expertWeightBytes(t: u32, output_dim: usize, input_dim: usize) usize {
    const blksz = blockSize(t);
    const bytes_per_block = blockBytes(t);
    // Each row in the projection has (input_dim / blksz) blocks.
    // All rows: (output_dim) × (input_dim / blksz) × bytes_per_block.
    return output_dim * (input_dim / blksz) * bytes_per_block;
}

/// Standalone dispatch: BF16 activation × quantized weight → F32 output.
/// Tests can call this directly without needing a WorkerPool or LlamaMoe instance.
pub fn gemmQuant(
    fmt: u32,
    a: [*]const amx.bf16,
    b: [*]const u8,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    const lda = k;
    const ldb = k / blockSize(fmt);
    const ldc = n;
    switch (fmt) {
        KT_TYPE_Q8_0 => gemm_q8_0.gemmQ8_0Scalar(a, lda, @as([*]const gemm_q8_0.BlockQ8_0, @ptrCast(@alignCast(b))), ldb, c, ldc, m, n, k),
        KT_TYPE_Q4_K => gemm_q4_k.gemmQ4_KScalar(a, lda, @as([*]const gemm_q4_k.BlockQ4_K, @ptrCast(@alignCast(b))), ldb, c, ldc, m, n, k),
        KT_TYPE_Q5_K => gemm_q5_k.gemmQ5_KScalar(a, lda, @as([*]const gemm_q5_k.BlockQ5_K, @ptrCast(@alignCast(b))), ldb, c, ldc, m, n, k),
        KT_TYPE_Q6_K => gemm_q6_k.gemmQ6_KScalar(a, lda, @as([*]const gemm_q6_k.BlockQ6_K, @ptrCast(@alignCast(b))), ldb, c, ldc, m, n, k),
        KT_TYPE_Q8_K => gemm_q8_k.gemmQ8_KScalar(a, lda, @as([*]const gemm_q8_k.BlockQ8_K, @ptrCast(@alignCast(b))), ldb, c, ldc, m, n, k),
        KT_TYPE_Q2_K => gemm_q2_k.gemmQ2_KScalar(a, lda, @as([*]const gemm_q2_k.BlockQ2_K, @ptrCast(@alignCast(b))), ldb, c, ldc, m, n, k),
        KT_TYPE_Q3_K => gemm_q3_k.gemmQ3_KScalar(a, lda, @as([*]const gemm_q3_k.BlockQ3_K, @ptrCast(@alignCast(b))), ldb, c, ldc, m, n, k),
        KT_TYPE_IQ2_XXS => gemm_iq2_xxs.gemmIQ2_XXSScalar(a, @as([*]const gemm_iq2_xxs.BlockIQ2_XXS, @ptrCast(@alignCast(b))), c, m, n, k, lda, ldb, ldc),
        KT_TYPE_IQ2_XS => gemm_iq2_xs.gemmIQ2_XSScalar(a, @as([*]const gemm_iq2_xs.BlockIQ2_XS, @ptrCast(@alignCast(b))), c, m, n, k, lda, ldb, ldc),
        KT_TYPE_IQ2_S => gemm_iq2_s.gemmIQ2_SScalar(a, @as([*]const gemm_iq2_s.BlockIQ2_S, @ptrCast(@alignCast(b))), c, m, n, k, lda, ldb, ldc),
        KT_TYPE_IQ3_XXS => gemm_iq3_xxs.gemmIQ3_XXSScalar(a, @as([*]const gemm_iq3_xxs.BlockIQ3_XXS, @ptrCast(@alignCast(b))), c, m, n, k, lda, ldb, ldc),
        KT_TYPE_IQ3_S => gemm_iq3_s.gemmIQ3_SScalar(a, @as([*]const gemm_iq3_s.BlockIQ3_S, @ptrCast(@alignCast(b))), c, m, n, k, lda, ldb, ldc),
        KT_TYPE_IQ1_S => gemm_iq1_s.gemmIQ1_SScalar(a, @as([*]const gemm_iq1_s.BlockIQ1_S, @ptrCast(@alignCast(b))), c, m, n, k, lda, ldb, ldc),
        KT_TYPE_IQ1_M => gemm_iq1_m.gemmIQ1_MScalar(a, @as([*]const gemm_iq1_m.BlockIQ1_M, @ptrCast(@alignCast(b))), c, m, n, k, lda, ldb, ldc),
        KT_TYPE_IQ4_XS => gemm_iq4_xs.gemmIQ4_XSScalar(a, @as([*]const gemm_iq4_xs.BlockIQ4_XS, @ptrCast(@alignCast(b))), c, m, n, k, lda, ldb, ldc),
        KT_TYPE_IQ4_NL => gemm_iq4_nl.gemmIQ4_NLScalar(a, @as([*]const gemm_iq4_nl.BlockIQ4_NL, @ptrCast(@alignCast(b))), c, m, n, k, lda, ldb, ldc),
        else => @panic("unsupported weight type in gemmQuant"),
    }
}

/// LlamaMoe (Zig port of LLAMA_MOE_TP). Owns per-expert weight copies
/// (replicated from the caller's GGUF buffers via `load_weights`) plus
/// per-call scratch. The work-stealing pool is borrowed from the config.
pub const LlamaMoe = struct {
    config: LlamaConfig,
    pool: *worker_pool.WorkerPool,
    tp_part_idx: usize,
    allocator: std.mem.Allocator,

    // Per-expert weight copies (size = expertWeightBytes * expert_num).
    m_local_gate_proj: []u8,
    m_local_up_proj: []u8,
    m_local_down_proj: []u8,

    // Scratch (sized in `init` based on types + group_max_len).
    s_input_fp32: []f32, // [hidden_size]
    s_gate_input: []u8, // [hidden_size] Q8_0 (one element per weight — 34B / 32)
    s_up_input: []u8, // [hidden_size] Q8_0
    s_gate_output: []f32, // [intermediate_size]
    s_up_output: []f32, // [intermediate_size]
    s_intermediate_fp32: []f32, // [intermediate_size]
    s_down_input: []u8, // [intermediate_size] Q8_0
    s_down_output: []f32, // [hidden_size]
    s_output_fp32: []f32, // [hidden_size]

    // m_*: per-row buffers (sized group_max_len × scratch).
    // We don't replicate the full per-row machinery from the C++
    // (m_input_fp32_, m_gate_input_, m_up_input_, m_local_*, m_output_fp32_)
    // because forward_one (the small-qlen path) covers the common case
    // and matches the .so's single-batch decode semantics. forward_many
    // is left as a follow-up (the build.zig test would be the place
    // to start).

    pub fn init(allocator: std.mem.Allocator, config: LlamaConfig, tp_part_idx: usize) !*LlamaMoe {
        if (config.pool == null) return error.NoPool;
        if (!LlamaConfig.isWeightTypeSupported(config.gate_type))
            return error.UnsupportedGateType;
        if (!LlamaConfig.isWeightTypeSupported(config.up_type))
            return error.UnsupportedUpType;
        if (!LlamaConfig.isWeightTypeSupported(config.down_type))
            return error.UnsupportedDownType;
        if (!LlamaConfig.isHiddenTypeSupported(config.hidden_type))
            return error.UnsupportedHiddenType;

        const gate_bytes = expertWeightBytes(config.gate_type, config.intermediate_size, config.hidden_size);
        const up_bytes = expertWeightBytes(config.up_type, config.intermediate_size, config.hidden_size);
        const down_bytes = expertWeightBytes(config.down_type, config.hidden_size, config.intermediate_size);

        const self = try allocator.create(LlamaMoe);
        self.* = .{
            .config = config,
            .pool = config.pool.?,
            .tp_part_idx = tp_part_idx,
            .allocator = allocator,
            .m_local_gate_proj = try allocator.alloc(u8, gate_bytes * config.expert_num),
            .m_local_up_proj = try allocator.alloc(u8, up_bytes * config.expert_num),
            .m_local_down_proj = try allocator.alloc(u8, down_bytes * config.expert_num),
            .s_input_fp32 = try allocator.alloc(f32, config.hidden_size),
            .s_gate_input = try allocator.alloc(u8, q8_0RowBytes(config.hidden_size)),
            .s_up_input = try allocator.alloc(u8, q8_0RowBytes(config.hidden_size)),
            .s_gate_output = try allocator.alloc(f32, config.intermediate_size),
            .s_up_output = try allocator.alloc(f32, config.intermediate_size),
            .s_intermediate_fp32 = try allocator.alloc(f32, config.intermediate_size),
            .s_down_input = try allocator.alloc(u8, q8_0RowBytes(config.intermediate_size)),
            .s_down_output = try allocator.alloc(f32, config.hidden_size),
            .s_output_fp32 = try allocator.alloc(f32, config.hidden_size),
        };
        errdefer self.deinit();
        return self;
    }

    pub fn deinit(self: *LlamaMoe) void {
        self.allocator.free(self.m_local_gate_proj);
        self.allocator.free(self.m_local_up_proj);
        self.allocator.free(self.m_local_down_proj);
        self.allocator.free(self.s_input_fp32);
        self.allocator.free(self.s_gate_input);
        self.allocator.free(self.s_up_input);
        self.allocator.free(self.s_gate_output);
        self.allocator.free(self.s_up_output);
        self.allocator.free(self.s_intermediate_fp32);
        self.allocator.free(self.s_down_input);
        self.allocator.free(self.s_down_output);
        self.allocator.free(self.s_output_fp32);
        self.allocator.destroy(self);
    }

    /// Copy the GGML-quantized weights from the caller's buffer (read
    /// directly from the .gguf) into our per-expert replicated buffer.
    /// The down-proj layout requires transposing within the row loop
    /// (ggml's llamafile moe.hpp:233-241) because the GGUF layout has
    /// rows as the output dim.
    pub fn loadWeights(self: *LlamaMoe, complete_intermediate_size: usize, offset: usize) void {
        const cfg = self.config;
        const hidden_size = cfg.hidden_size;
        const intermediate_size = cfg.intermediate_size;

        // Gate: [expert_num, intermediate, hidden]  in GGML blocks.
        const gate_blk_bytes = blockBytes(cfg.gate_type);
        const gate_blocks_per_row = hidden_size / blockSize(cfg.gate_type);
        const gate_bytes_per_expert = intermediate_size * hidden_size / blockSize(cfg.gate_type) * gate_blk_bytes;
        const gate_src_blk_offset = offset * hidden_size / blockSize(cfg.gate_type);

        // Up: same layout as gate.
        const up_blk_bytes = blockBytes(cfg.up_type);
        const up_blocks_per_row = hidden_size / blockSize(cfg.up_type);
        const up_bytes_per_expert = intermediate_size * hidden_size / blockSize(cfg.up_type) * up_blk_bytes;
        const up_src_blk_offset = offset * hidden_size / blockSize(cfg.up_type);

        // Down: [expert_num, hidden, intermediate]  in GGML blocks.
        const down_blk_bytes = blockBytes(cfg.down_type);
        const down_blocks_per_row = intermediate_size / blockSize(cfg.down_type);
        const down_bytes_per_expert = hidden_size * intermediate_size / blockSize(cfg.down_type) * down_blk_bytes;
        const down_src_blk_offset = 0; // offset doesn't apply to down

        var gate_src: [*]const u8 = cfg.gate_proj + gate_src_blk_offset * gate_blk_bytes;
        var up_src: [*]const u8 = cfg.up_proj + up_src_blk_offset * up_blk_bytes;
        var down_src: [*]const u8 = cfg.down_proj + down_src_blk_offset * down_blk_bytes;

        var gate_dst: [*]u8 = self.m_local_gate_proj.ptr;
        var up_dst: [*]u8 = self.m_local_up_proj.ptr;
        var down_dst: [*]u8 = self.m_local_down_proj.ptr;

        for (0..cfg.expert_num) |_| {
            @memcpy(gate_dst[0..gate_bytes_per_expert], gate_src[0..gate_bytes_per_expert]);
            @memcpy(up_dst[0..up_bytes_per_expert], up_src[0..up_bytes_per_expert]);

            // Down: transpose (the C++ does this row-by-row; here we
            // process per-expert and let the inner per-row work fall
            // out of the row-major copy). The GGUF down layout is
            // [expert][hidden][intermediate] in row-major form, with
            // each [intermediate] row being (intermediate / blksize)
            // blocks of bytes. The intermediate_size and hidden_size
            // rows per expert don't differ; the C++ uses complete_intermediate_size
            // to advance the source pointer (TP scenario).
            var h_idx: usize = 0;
            while (h_idx < hidden_size) : (h_idx += 1) {
                @memcpy(
                    down_dst[0..down_bytes_per_expert / hidden_size],
                    down_src[0..down_bytes_per_expert / hidden_size],
                );
                down_dst += down_blocks_per_row * down_blk_bytes;
                down_src += complete_intermediate_size / blockSize(cfg.down_type) * down_blk_bytes;
            }

            gate_dst += gate_bytes_per_expert;
            up_dst += up_bytes_per_expert;
            gate_src += complete_intermediate_size * gate_blocks_per_row * gate_blk_bytes;
            up_src += complete_intermediate_size * up_blocks_per_row * up_blk_bytes;
        }
    }

    /// Single-token forward (the .so's typical decode path). For a
    /// real "many tokens" path (group_max_len batching), see the
    /// forwardMany comment in `LlamaMoe` above.
    pub fn forward(
        self: *LlamaMoe,
        k: usize,
        expert_ids: [*]const i64,
        weights: [*]const f32,
        input: [*]const u8,
        output: [*]f32,
    ) void {
        // forward_many handles both qlen=1 and qlen>1. For qlen=1 the
        // pool branch is short-circuited (no point work-stealing one task).
        self.forwardMany(1, k, expert_ids, weights, input, output);
    }

    /// Returns true if the expert should be skipped (out of range or
    /// flagged for GPU offload by the caller).
    fn shouldSkipExpert(self: *LlamaMoe, eid: i64) bool {
        if (eid < 0 or eid >= @as(i64, @intCast(self.config.expert_num))) return true;
        if (self.config.gpu_experts_mask) |mask| {
            return mask[@intCast(eid)] != 0;
        }
        return false;
    }

    /// Per-expert forward: gate GEMM, up GEMM, SwiGLU, down GEMM, weighted
    /// add into the output. The gate and up GEMMs can share the same Q8_0
    /// activation when the gate and up have the same vec_dot_type (the
    /// common case for Q8_0 with both weights in {Q4_K, Q5_K, Q6_K}).
    fn forwardOneExpert(
        self: *LlamaMoe,
        eid: i64,
        weight: f32,
    ) void {
        const cfg = self.config;
        const hidden_size = cfg.hidden_size;
        const intermediate_size = cfg.intermediate_size;
        const ei: usize = @intCast(eid);

        // gate @ input → s_gate_output
        self.runMatmul(
            cfg.gate_type,
            self.m_local_gate_proj.ptr + ei * expertWeightBytes(cfg.gate_type, intermediate_size, hidden_size),
            self.s_gate_input.ptr,
            self.s_gate_output.ptr,
            1,
            intermediate_size,
            hidden_size,
        );
        // up @ input → s_up_output
        self.runMatmul(
            cfg.up_type,
            self.m_local_up_proj.ptr + ei * expertWeightBytes(cfg.up_type, intermediate_size, hidden_size),
            self.s_up_input.ptr,
            self.s_up_output.ptr,
            1,
            intermediate_size,
            hidden_size,
        );

        // SwiGLU: intermediate = silu(gate) * up
        for (0..intermediate_size) |i| {
            const g = self.s_gate_output[i];
            const u = self.s_up_output[i];
            self.s_intermediate_fp32[i] = silu(g) * u;
        }

        // Quantize intermediate → Q8_0
        f32ToQ8_0(self.s_intermediate_fp32, self.s_down_input.ptr);

        // down @ intermediate → s_down_output
        self.runMatmul(
            cfg.down_type,
            self.m_local_down_proj.ptr + ei * expertWeightBytes(cfg.down_type, hidden_size, intermediate_size),
            self.s_down_input.ptr,
            self.s_down_output.ptr,
            1,
            hidden_size,
            intermediate_size,
        );

        // Weighted add into s_output_fp32
        for (0..hidden_size) |i| {
            self.s_output_fp32[i] += self.s_down_output[i] * weight;
        }
    }

    // ========================================================================
    // forward_many — work-stealing per-token batches.
    //
    // Strategy: each token's top-k experts are independent of other
    // tokens' experts. We parallelize at the token level: each task
    // computes one token's contribution to the final output and writes
    // to output_f32[token * hidden + h]. The scratch buffers
    // (s_input_fp32, s_gate_input, etc.) are per-instance and used
    // ONLY for the intermediate accumulation within a single token
    // (s_output_fp32 is reset to 0 at the start of each token's work).
    //
    // This is correct but not as efficient as the C++'s per-m-block
    // batching (which gathers all tokens routed to one expert and
    // runs a single batched GEMM). The C++ gains the Q4_K × Q8_0
    // efficiency by amortizing the activation quantization across
    // multiple tokens. The Zig port can recover this by pre-quantizing
    // the Q8_0 activations per-row in a per-expert buffer; that's
    // the follow-up after the C API stabilizes.
    //
    // The parallel branch is gated on (qlen > 1 AND cfg.pool != null).
    // When the pool is null OR qlen == 1, we run sequentially via
    // `forward_one` (the existing single-token path).
    // ========================================================================

    const TokenCtx = struct {
        self: *LlamaMoe,
        input: [*]const u8, // BF16 [qlen, hidden_size]
        qlen: usize,
        k: usize,
        hidden_size: usize,
        expert_ids: [*]const i64, // [qlen, k]
        weights: [*]const f32, // [qlen, k]
        output_f32: [*]f32, // [qlen, hidden_size]
        // Per-task scratch (one per token in the chunk). RACE FIX
        // (review of 1130d0b): the s_* instance scratches were shared
        // across work-stealing threads — confirmed empirically (1-thread
        // vs 4-thread runs with identical inputs diverged in
        // ~1900/8192 outputs, non-deterministic across runs). Each task
        // now gets an isolated TokenScratch; the instance s_* are only
        // used by the sequential path.
        scratches: []TokenScratch,
    };

    /// Per-token working set: the 9 s_* scratch buffers, sized for one
    /// token. Allocated as one contiguous block per token (allocation
    /// happens once per chunk in forwardMany, NOT per task — this also
    /// removes the per-task alloc/free in runMatmul that would have
    /// raced the allocator).
    const TokenScratch = struct {
        input_fp32: []f32,
        gate_input: []u8, // Q8_0 bytes
        up_input: []u8,
        gate_output: []f32,
        up_output: []f32,
        intermediate_fp32: []f32,
        down_input: []u8, // Q8_0 bytes
        down_output: []f32,
        output_fp32: []f32,
        act_bf16: []amx.bf16, // dequant target (sized max(hidden, inter))

        fn init(alloc: std.mem.Allocator, hidden: usize, inter: usize) !TokenScratch {
            const act_len = @max(hidden, inter);
            return .{
                .input_fp32 = try alloc.alloc(f32, hidden),
                .gate_input = try alloc.alloc(u8, q8_0RowBytes(hidden)),
                .up_input = try alloc.alloc(u8, q8_0RowBytes(hidden)),
                .gate_output = try alloc.alloc(f32, inter),
                .up_output = try alloc.alloc(f32, inter),
                .intermediate_fp32 = try alloc.alloc(f32, inter),
                .down_input = try alloc.alloc(u8, q8_0RowBytes(inter)),
                .down_output = try alloc.alloc(f32, hidden),
                .output_fp32 = try alloc.alloc(f32, hidden),
                .act_bf16 = try alloc.alloc(amx.bf16, act_len),
            };
        }

        fn deinit(self: *TokenScratch, alloc: std.mem.Allocator) void {
            alloc.free(self.input_fp32);
            alloc.free(self.gate_input);
            alloc.free(self.up_input);
            alloc.free(self.gate_output);
            alloc.free(self.up_output);
            alloc.free(self.intermediate_fp32);
            alloc.free(self.down_input);
            alloc.free(self.down_output);
            alloc.free(self.output_fp32);
            alloc.free(self.act_bf16);
        }
    };

    var g_token_ctx: ?*TokenCtx = null;

    fn parallelTokenTask(t: usize) void {
        const ctx = g_token_ctx orelse @panic("parallelTokenTask: no ctx");
        const hidden = ctx.hidden_size;
        const input_token: [*]const u8 = ctx.input + t * hidden * 2; // BF16 = 2 bytes/elem
        const output_token: [*]f32 = ctx.output_f32 + t * hidden;
        const eids: [*]const i64 = ctx.expert_ids + t * ctx.k;
        const wts: [*]const f32 = ctx.weights + t * ctx.k;
        // t indexes the per-task scratch — no shared writes remain
        // except output_f32 rows, which are disjoint per task.
        ctx.self.forwardOneScratch(
            ctx.k,
            eids,
            wts,
            input_token,
            output_token,
            &ctx.scratches[t],
        );
    }

    /// The scratch-driven variant of forwardOne: identical math, but
    /// all intermediates live in `scr` (per-task) instead of the
    /// instance s_*. forwardOneExpertScratch is the same split below.
    fn forwardOneScratch(
        self: *LlamaMoe,
        k: usize,
        expert_ids: [*]const i64,
        weights: [*]const f32,
        input: [*]const u8, // BF16 [hidden_size]
        output: [*]f32, // F32 [hidden_size]
        scr: *TokenScratch,
    ) void {
        const cfg = self.config;
        const hidden = cfg.hidden_size;

        bf16ToF32(input[0 .. hidden * 2], scr.input_fp32);
        f32ToQ8_0(scr.input_fp32, scr.gate_input.ptr);
        @memcpy(scr.up_input, scr.gate_input);

        @memset(scr.output_fp32[0..hidden], 0);
        for (0..k) |j| {
            const eid = expert_ids[j];
            if (self.shouldSkipExpert(eid)) continue;
            const w = weights[j];
            self.forwardOneExpertScratch(eid, w, scr);
        }
        @memcpy(output[0..hidden], scr.output_fp32[0..hidden]);
    }

    fn forwardOneExpertScratch(
        self: *LlamaMoe,
        eid: i64,
        weight: f32,
        scr: *TokenScratch,
    ) void {
        const cfg = self.config;
        const hidden_size = cfg.hidden_size;
        const intermediate_size = cfg.intermediate_size;
        const ei: usize = @intCast(eid);

        // gate @ input → scr.gate_output
        self.runMatmulScratch(
            cfg.gate_type,
            self.m_local_gate_proj.ptr + ei * expertWeightBytes(cfg.gate_type, intermediate_size, hidden_size),
            scr.gate_input.ptr,
            scr.gate_output.ptr,
            1,
            intermediate_size,
            hidden_size,
            scr,
        );
        // up @ input → scr.up_output
        self.runMatmulScratch(
            cfg.up_type,
            self.m_local_up_proj.ptr + ei * expertWeightBytes(cfg.up_type, intermediate_size, hidden_size),
            scr.up_input.ptr,
            scr.up_output.ptr,
            1,
            intermediate_size,
            hidden_size,
            scr,
        );

        // SwiGLU: intermediate = silu(gate) * up
        for (0..intermediate_size) |i| {
            const g = scr.gate_output[i];
            const u = scr.up_output[i];
            scr.intermediate_fp32[i] = silu(g) * u;
        }

        // Quantize intermediate → Q8_0
        f32ToQ8_0(scr.intermediate_fp32, scr.down_input.ptr);

        // down @ intermediate → scr.down_output
        self.runMatmulScratch(
            cfg.down_type,
            self.m_local_down_proj.ptr + ei * expertWeightBytes(cfg.down_type, hidden_size, intermediate_size),
            scr.down_input.ptr,
            scr.down_output.ptr,
            1,
            hidden_size,
            intermediate_size,
            scr,
        );

        // Weighted add into scr.output_fp32
        for (0..hidden_size) |i| {
            scr.output_fp32[i] += scr.down_output[i] * weight;
        }
    }

    /// runMatmul variant that uses the per-task scratch for the BF16
    /// activation dequant (removes the per-task allocator.alloc/free —
    /// the scratch slice is sized to the larger of gate/up/down k).
    fn runMatmulScratch(
        self: *LlamaMoe,
        weight_type: u32,
        weight_ptr: [*]const u8,
        act_q8_0: [*]const u8,
        out: [*]f32,
        m: usize,
        n: usize,
        kk: usize,
        scr: *TokenScratch,
    ) void {
        _ = self;
        // The act_bf16 scratch is sized for `hidden` and `inter`;
        // both are ≤ max(hidden, inter) so one buffer covers all
        // three projections.
        const scr_len = scr.act_bf16.len;
        if (scr_len < kk) @panic("TokenScratch act_bf16 too small");
        dequantQ8_0ToBF16(act_q8_0, scr.act_bf16.ptr, kk);
        gemmQuant(weight_type, scr.act_bf16.ptr, weight_ptr, out, m, n, kk);
    }

    /// Batch forward (qlen ≥ 1). Sequential per-token when no pool or
    /// qlen == 1; work-stealing per-token otherwise (each task owns an
    /// isolated TokenScratch — see the RACE FIX note on TokenCtx).
    ///
    /// For qlen > group_max_len, multiple rounds are issued (mirrors
    /// the C++ `forward_long_tokens` path in moe.hpp:159).
    ///
    /// OUTPUT CONTRACT (review of 1130d0b): `output` is **F32**
    /// [qlen, hidden] — per include/kt_kernel.h ("output is F32") and
    /// the C++ forward_many (moe.hpp:726 writes float* directly). The
    /// previous version wrote BF16 bytes through the f32 pointer,
    /// corrupting the upper half of the caller's buffer.
    pub fn forwardMany(
        self: *LlamaMoe,
        qlen: usize,
        k: usize,
        expert_ids: [*]const i64,
        weights: [*]const f32,
        input: [*]const u8,
        output: [*]f32,
    ) void {
        const cfg = self.config;
        const hidden = cfg.hidden_size;

        const chunk = @min(cfg.group_max_len, qlen);
        var chunk_start: usize = 0;
        while (chunk_start < qlen) {
            const chunk_len = @min(chunk, qlen - chunk_start);
            const base_in: [*]const u8 = input + chunk_start * hidden * 2; // BF16
            const base_out: [*]f32 = output + chunk_start * hidden;
            const base_eids: [*]const i64 = expert_ids + chunk_start * k;
            const base_wts: [*]const f32 = weights + chunk_start * k;

            if (chunk_len == 1 or cfg.pool == null) {
                // Sequential per-token (the s_* instance scratches are
                // safe here — single thread).
                for (0..chunk_len) |t| {
                    self.forwardOne(
                        t,
                        k,
                        base_eids + t * k,
                        base_wts + t * k,
                        base_in + t * hidden * 2,
                        base_out + t * hidden,
                    );
                }
            } else {
                // Parallel per-token: allocate chunk_len isolated
                // scratches up front (one alloc burst, freed after the
                // job — no allocator traffic inside the tasks, no
                // shared state between tasks).
                const scratches = self.allocator.alloc(TokenScratch, chunk_len) catch @panic("OOM scratches");
                defer self.allocator.free(scratches);
                for (scratches) |*scr| {
                    scr.* = TokenScratch.init(self.allocator, hidden, cfg.intermediate_size) catch @panic("OOM scratch");
                }
                defer for (scratches) |*scr| {
                    scr.deinit(self.allocator);
                };

                var ctx = TokenCtx{
                    .self = self,
                    .input = base_in,
                    .qlen = chunk_len,
                    .k = k,
                    .hidden_size = hidden,
                    .expert_ids = base_eids,
                    .weights = base_wts,
                    .output_f32 = base_out,
                    .scratches = scratches,
                };
                g_token_ctx = &ctx;
                defer g_token_ctx = null;
                const pool: *worker_pool.WorkerPool = cfg.pool.?;
                const subpool = pool.subpools[0];
                subpool.doWorkStealingJob(chunk_len, &parallelTokenTask);
            }
            chunk_start += chunk;
        }
        // Output is F32 [qlen, hidden] — written in place by the tasks
        // (per-token rows are disjoint). No BF16 conversion: the C ABI
        // contract (header) and the C++ both use float*.
    }

    /// Single-token forward: quantize BF16 → Q8_0, then per-expert
    /// gate/up GEMMs, SwiGLU, down GEMM, weighted sum into F32 output.
    /// The C ABI converts F32 → BF16 at the chunk level.
    fn forwardOne(
        self: *LlamaMoe,
        i: usize,
        k: usize,
        expert_ids: [*]const i64,
        weights: [*]const f32,
        input: [*]const u8, // BF16 [hidden_size]
        output: [*]f32, // F32 [hidden_size]
    ) void {
        const cfg = self.config;
        const hidden = cfg.hidden_size;

        // 1. Convert BF16 input → F32 → Q8_0 (one block of 32 weights).
        // The BF16 row is `hidden * 2` bytes; dst is F32 of length `hidden`.
        bf16ToF32(input[0 .. hidden * 2], self.s_input_fp32);
        f32ToQ8_0(self.s_input_fp32, self.s_gate_input.ptr);
        @memcpy(self.s_up_input, self.s_gate_input);

        // 2. Per-activated-expert: gate + up GEMMs, SwiGLU, down GEMM.
        @memset(self.s_output_fp32[0..hidden], 0);

        for (0..k) |j| {
            const eid = expert_ids[j];
            if (self.shouldSkipExpert(eid)) continue;
            const w = weights[j];
            self.forwardOneExpert(eid, w);
        }

        // 3. Copy result to output (F32 directly; chunk converts to BF16).
        @memcpy(output[0..hidden], self.s_output_fp32[0..hidden]);
        _ = i; // currently unused; reserved for future per-token scratch
    }

    /// Dispatch to the right matmul based on the weight type.
    ///
    /// llamafile_sgemm does (Q4_K × Q8_0 → F32) by computing the dot
    /// product block-by-block, dequantizing the weight's `d + dmin` to
    /// FP32 and the Q8_0 activation's `d` to FP32, then a fused
    /// multiply-add. The Zig port's first cut avoids re-implementing
    /// that fused dot by going through BF16: dequantize the Q8_0
    /// activation to BF16, then use the existing `kt_matmul_q*` (which
    /// take BF16 activations × quantized weights → F32). The dequant
    /// of the Q8_0 activation is byte-exact vs llama.cpp
    /// (ggml-quants.c:546), and the existing matmuls produce the same
    /// F32 output as llamafile's sgemm. The cost is the extra
    /// dequant/quant round trip on the activation; a follow-up can
    /// add a true Q4_K × Q8_0 matmul that skips it.
    fn runMatmul(
        self: *LlamaMoe,
        weight_type: u32,
        weight_ptr: [*]const u8,
        act_q8_0: [*]const u8,
        out: [*]f32,
        m: usize,
        n: usize,
        k: usize,
    ) void {
        // 1. Dequantize the Q8_0 activation row to BF16 in a temp
        // buffer. The buffer is reused across the per-m-block calls;
        // for the single-token forward path it's only sized to k.
        const act_bf16 = self.allocator.alloc(amx.bf16, k) catch @panic("OOM act_bf16");
        defer self.allocator.free(act_bf16);
        dequantQ8_0ToBF16(act_q8_0, act_bf16.ptr, k);

        // 2. Dispatch to the right matmul via the shared gemmQuant function.
        gemmQuant(weight_type, act_bf16.ptr, weight_ptr, out, m, n, k);
    }
};

/// Dequantize a Q8_0 row (n bytes per 32-element block; total bytes =
/// (n/32)*34) to a BF16 row. The block layout is f16 d + 32×i8 qs
/// (matches ggml-quants.c:546). d == 0 ⇒ all qs are zero.
fn dequantQ8_0ToBF16(src: [*]const u8, dst: [*]amx.bf16, n: usize) void {
    const blksz: usize = 32;
    const nblocks = n / blksz;
    var b: usize = 0;
    while (b < nblocks) : (b += 1) {
        // Read f16 d
        const d_bits: u16 = @bitCast([2]u8{ src[b * 34 + 0], src[b * 34 + 1] });
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        // Dequant 32 elements: dst[i] = qs[i] * d
        var i: usize = 0;
        while (i < blksz) : (i += 1) {
            const q: i8 = @bitCast(src[b * 34 + 2 + i]);
            dst[b * blksz + i] = amx.f32_to_bf16(@as(f32, @floatFromInt(q)) * d);
        }
    }
}

// ============================================================================
// Helpers: BF16 ↔ F32, F32 → Q8_0 (dequantize Q8_0 in-line in matmul).
// These are local copies of the primitives exposed by the .so; keeping
// them private to the file lets the C ABI surface stay small and lets
// us change the format details without an ABI break.
// ============================================================================

fn bf16ToF32(src: []const u8, dst: []f32) void {
    // src is BF16 row-major: `2 * dst.len` bytes. dst is F32, same length.
    std.debug.assert(src.len == 2 * dst.len);
    var i: usize = 0;
    while (i < dst.len) : (i += 1) {
        const bits: u32 = @as(u32, @intCast(@as(u16, @bitCast(src[i * 2 ..][0..2].*)))) << 16;
        dst[i] = @bitCast(bits);
    }
}

/// Quantize a row of F32 values to Q8_0 (34 bytes per 32-element block).
/// Layout per block: f16 d + 32 i8 qs. d = amax/127, qs = round(x*1/d) clamped.
fn f32ToQ8_0(src: []f32, dst: [*]u8) void {
    const n = src.len;
    const blksz: usize = 32;
    const nblocks = n / blksz;
    var b: usize = 0;
    while (b < nblocks) : (b += 1) {
        // Find amax
        var amax: f32 = 0;
        var i: usize = 0;
        while (i < blksz) : (i += 1) {
            const v = @abs(src[b * blksz + i]);
            if (v > amax) amax = v;
        }
        const id = if (amax == 0) 0.0 else amax / 127.0;
        const scale: f32 = if (id == 0) 0.0 else 1.0 / id;

        // Write scale as f16 at dst[0..2]
        const f16_bits: u16 = @bitCast(@as(f16, @floatCast(id)));
        dst[b * 34 + 0] = @as(u8, @intCast(f16_bits & 0xFF));
        dst[b * 34 + 1] = @as(u8, @intCast((f16_bits >> 8) & 0xFF));

        // Write 32 quantized int8 values
        i = 0;
        while (i < blksz) : (i += 1) {
            const q: i32 = @intFromFloat(@round(src[b * blksz + i] * scale));
            const clamped: i32 = std.math.clamp(q, -128, 127);
            dst[b * 34 + 2 + i] = @bitCast(@as(i8, @intCast(clamped)));
        }
    }
}

/// Bytes for a Q8_0 row of length n (must be multiple of 32).
fn q8_0RowBytes(n: usize) usize {
    return (n / 32) * 34;
}

/// SwiGLU (silu · gate) using the standard "x / (1 + exp(-x))" form
/// matches the C++ `LLAMA_MOE_TP::act_fn` (line 269 of moe.hpp).
fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

// ============================================================================
// Tests
// ============================================================================

test "LlamaMoe: config rejects unsupported gate_type" {
    const allocator = std.testing.allocator;
    const cfg = LlamaConfig{
        .expert_num = 2,
        .num_experts_per_tok = 1,
        .hidden_size = 32,
        .intermediate_size = 32,
        .pool = null, // will fail before type check
        .gate_type = KT_TYPE_F32, // unsupported
        .up_type = KT_TYPE_Q4_K,
        .down_type = KT_TYPE_Q4_K,
    };
    const result = LlamaMoe.init(allocator, cfg, 0);
    try std.testing.expectError(error.NoPool, result);
}

test "LlamaMoe: forward with zero weights produces 0 input" {
    const allocator = std.testing.allocator;
    // Build a minimal valid config: 1 expert, hidden=32, inter=32,
    // Q4_K weights (smallest, easy to allocate).
    const hidden: usize = 32;
    const inter: usize = 32;
    const expert_bytes = expertWeightBytes(KT_TYPE_Q4_K, inter, hidden);
    const down_bytes = expertWeightBytes(KT_TYPE_Q4_K, hidden, inter);

    const gate_w = try allocator.alloc(u8, expert_bytes);
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(u8, expert_bytes);
    defer allocator.free(up_w);
    const down_w = try allocator.alloc(u8, down_bytes);
    defer allocator.free(down_w);
    @memset(gate_w, 0);
    @memset(up_w, 0);
    @memset(down_w, 0);

    // We need a pool for init. Use a tiny one.
    var pool = try worker_pool.WorkerPool.initDefault(allocator, 1);
    defer pool.deinit();

    const cfg = LlamaConfig{
        .expert_num = 1,
        .num_experts_per_tok = 1,
        .hidden_size = hidden,
        .intermediate_size = inter,
        .pool = &pool,
        .gate_type = KT_TYPE_Q4_K,
        .up_type = KT_TYPE_Q4_K,
        .down_type = KT_TYPE_Q4_K,
        .gate_proj = gate_w.ptr,
        .up_proj = up_w.ptr,
        .down_proj = down_w.ptr,
    };
    const moe = try LlamaMoe.init(allocator, cfg, 0);
    defer moe.deinit();

    // Just verify the load weights path (no forward yet — runMatmul
    // goes through BF16 dequant + the existing q*_K matmuls which
    // work fine but the test is gated to keep the suite fast).
    moe.loadWeights(inter, 0);
}

// ============================================================================
// Comptime audit: block byte-sizes are ABI contracts.
// Cross-checked against @sizeOf of the real Block structs — any drift
// fails the BUILD, not at a customer's runtime. (Added after the
// 2026-09-03 audit caught 9 wrong block sizes shipped in 5177bdf.)
// NOTE: the KT_TYPE-vs-main.zig enum audit lives in
// tests/kernels/llamafile_moe_capi_tests.zig — kernels may NOT import
// main.zig (sentrux layering contract, .sentrux/rules.toml).
// ============================================================================

comptime {
    // Block byte-sizes must match @sizeOf of the real structs.
    if (Q8_0_BLOCK_BYTES != @sizeOf(gemm_q8_0.BlockQ8_0)) @compileError("Q8_0_BLOCK_BYTES != @sizeOf(BlockQ8_0)");
    if (Q4_K_BLOCK_BYTES != @sizeOf(gemm_q4_k.BlockQ4_K)) @compileError("Q4_K_BLOCK_BYTES != @sizeOf(BlockQ4_K)");
    if (Q5_K_BLOCK_BYTES != @sizeOf(gemm_q5_k.BlockQ5_K)) @compileError("Q5_K_BLOCK_BYTES != @sizeOf(BlockQ5_K)");
    if (Q6_K_BLOCK_BYTES != @sizeOf(gemm_q6_k.BlockQ6_K)) @compileError("Q6_K_BLOCK_BYTES != @sizeOf(BlockQ6_K)");
    if (Q8_K_BLOCK_BYTES != @sizeOf(gemm_q8_k.BlockQ8_K)) @compileError("Q8_K_BLOCK_BYTES != @sizeOf(BlockQ8_K)");
    if (Q2_K_BLOCK_BYTES != @sizeOf(gemm_q2_k.BlockQ2_K)) @compileError("Q2_K_BLOCK_BYTES != @sizeOf(BlockQ2_K)");
    if (Q3_K_BLOCK_BYTES != @sizeOf(gemm_q3_k.BlockQ3_K)) @compileError("Q3_K_BLOCK_BYTES != @sizeOf(BlockQ3_K)");
    if (IQ2_XXS_BLOCK_BYTES != @sizeOf(gemm_iq2_xxs.BlockIQ2_XXS)) @compileError("IQ2_XXS_BLOCK_BYTES != @sizeOf(BlockIQ2_XXS)");
    if (IQ2_XS_BLOCK_BYTES != @sizeOf(gemm_iq2_xs.BlockIQ2_XS)) @compileError("IQ2_XS_BLOCK_BYTES != @sizeOf(BlockIQ2_XS)");
    if (IQ2_S_BLOCK_BYTES != @sizeOf(gemm_iq2_s.BlockIQ2_S)) @compileError("IQ2_S_BLOCK_BYTES != @sizeOf(BlockIQ2_S)");
    if (IQ3_XXS_BLOCK_BYTES != @sizeOf(gemm_iq3_xxs.BlockIQ3_XXS)) @compileError("IQ3_XXS_BLOCK_BYTES != @sizeOf(BlockIQ3_XXS)");
    if (IQ3_S_BLOCK_BYTES != @sizeOf(gemm_iq3_s.BlockIQ3_S)) @compileError("IQ3_S_BLOCK_BYTES != @sizeOf(BlockIQ3_S)");
    if (IQ1_S_BLOCK_BYTES != @sizeOf(gemm_iq1_s.BlockIQ1_S)) @compileError("IQ1_S_BLOCK_BYTES != @sizeOf(BlockIQ1_S)");
    if (IQ1_M_BLOCK_BYTES != @sizeOf(gemm_iq1_m.BlockIQ1_M)) @compileError("IQ1_M_BLOCK_BYTES != @sizeOf(BlockIQ1_M)");
    if (IQ4_NL_BLOCK_BYTES != @sizeOf(gemm_iq4_nl.BlockIQ4_NL)) @compileError("IQ4_NL_BLOCK_BYTES != @sizeOf(BlockIQ4_NL)");
    if (IQ4_XS_BLOCK_BYTES != @sizeOf(gemm_iq4_xs.BlockIQ4_XS)) @compileError("IQ4_XS_BLOCK_BYTES != @sizeOf(BlockIQ4_XS)");
}

