// LlamaMoe — the GGML-quantized MoE (the "llamafile" backend class of the
// C++ reference: operators/llamafile/moe.hpp LLAMA_MOE_TP).
//
// This is the MoE the ktransformers framework uses as its DEFAULT backend
// for GGUF models: it receives PRE-QUANTIZED expert weights (gate/up/down,
// each possibly a different GGML format) straight from the GGUF loader —
// no BF16 staging, no online quantization. The C++ computes it with
// llamafile_sgemm + a vec_dot_type activation quantization; this port
// dispatches to the 15 GGML GEMM kernels (activation BF16 x quantized
// weights -> F32), which is the same math with the vec_dot semantics of
// the BF16 hidden_type case (the dominant GGUF path).
//
// Weight layout (matches the C++ exactly — same byte offsets the
// framework's GGUFLoader produces):
//   gate_proj: [expert_num * intermediate_size, hidden_size] rows, blocks
//              of gate_type, row-major: expert e's gate slice starts at
//              (e * intermediate_size) * hidden_size elements.
//   up_proj:   same shape/type.
//   down_proj: [expert_num * hidden_size, intermediate_size] rows of
//              down_type: expert e's down slice at
//              (e * hidden_size) * intermediate_size elements.
//
// Forward (per token, k activated experts — mirrors forward_one):
//   1. skip invalid/GPU experts (gpu_experts_mask + expert_id range)
//   2. per activated expert: gate GEMM + up GEMM (BF16 act x quant W)
//   3. intermediate = swiglu(gate) * up   (F32; act_fn = silu)
//   4. down GEMM per expert: [1, intermediate] x quant W -> [1, hidden]
//   5. output += down_out * routing weight (first match in expert_ids,
//      exactly like the C++ weight lookup at moe.hpp:437-443)
//
// The C++ parallelizes with m_block chunks + work-stealing; this port is
// sequential per expert (correctness-first; the pool can be wired the
// same way as TpMoe.forward when perf matters — the kernels themselves
// are the hot path and are already vectorized).

const std = @import("std");
const amx = @import("../arch/amx.zig");

const gemm_q8_0 = @import("../amx/gemm_224_q8_0.zig");
const gemm_q4_k = @import("../amx/gemm_224_q4_k.zig");
const gemm_q5_k = @import("../amx/gemm_224_q5_k.zig");
const gemm_q6_k = @import("../amx/gemm_224_q6_k.zig");
const gemm_q8_k = @import("../amx/gemm_224_q8_k.zig");
const gemm_q2_k = @import("../amx/gemm_224_q2_k.zig");
const gemm_q3_k = @import("../amx/gemm_224_q3_k.zig");
const gemm_iq4_xs = @import("../amx/gemm_224_iq4_xs.zig");
const gemm_iq4_nl = @import("../amx/gemm_224_iq4_nl.zig");
const gemm_iq2_xxs = @import("../amx/gemm_224_iq2_xxs.zig");
const gemm_iq2_xs = @import("../amx/gemm_224_iq2_xs.zig");
const gemm_iq2_s = @import("../amx/gemm_224_iq2_s.zig");
const gemm_iq3_xxs = @import("../amx/gemm_224_iq3_xxs.zig");
const gemm_iq3_s = @import("../amx/gemm_224_iq3_s.zig");
const gemm_iq1_s = @import("../amx/gemm_224_iq1_s.zig");
const gemm_iq1_m = @import("../amx/gemm_224_iq1_m.zig");

/// The format id space (mirrors the kt_type_t / ggml_type values the
/// framework passes — see include/kt_kernel.h).
pub const Type = enum(c_int) {
    bf16 = 2,
    q8_0 = 12,
    q2_k = 14,
    q3_k = 15,
    q4_k = 16,
    q5_k = 17,
    q6_k = 18,
    q8_k = 19,
    iq2_xxs = 20,
    iq2_xs = 21,
    iq3_xxs = 22,
    iq1_s = 23,
    iq4_nl = 24,
    iq3_s = 25,
    iq2_s = 26,
    iq4_xs = 27,
    iq1_m = 28,
    _,

    /// Elements per block for this format.
    pub fn blockElems(self: Type) usize {
        return switch (self) {
            // 256-elem super-blocks: the K-quants + IQ 2/3/1 families.
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .iq4_xs => 256,
            // 32-elem blocks.
            .q8_0, .iq4_nl => 32,
            // Elementwise.
            .bf16 => 1,
            _ => 0,
        };
    }
};

/// Bytes of one quantized block for the format (sizeof(Block)).
pub fn rowBytes(fmt: Type) usize {
    return switch (fmt) {
        .q8_0 => @sizeOf(gemm_q8_0.BlockQ8_0),
        .q4_k => @sizeOf(gemm_q4_k.BlockQ4_K),
        .q5_k => @sizeOf(gemm_q5_k.BlockQ5_K),
        .q6_k => @sizeOf(gemm_q6_k.BlockQ6_K),
        .q8_k => @sizeOf(gemm_q8_k.BlockQ8_K),
        .q2_k => @sizeOf(gemm_q2_k.BlockQ2_K),
        .q3_k => @sizeOf(gemm_q3_k.BlockQ3_K),
        .iq4_xs => @sizeOf(gemm_iq4_xs.BlockIQ4_XS),
        .iq4_nl => @sizeOf(gemm_iq4_nl.BlockIQ4_NL),
        .iq2_xxs => @sizeOf(gemm_iq2_xxs.BlockIQ2_XXS),
        .iq2_xs => @sizeOf(gemm_iq2_xs.BlockIQ2_XS),
        .iq2_s => @sizeOf(gemm_iq2_s.BlockIQ2_S),
        .iq3_xxs => @sizeOf(gemm_iq3_xxs.BlockIQ3_XXS),
        .iq3_s => @sizeOf(gemm_iq3_s.BlockIQ3_S),
        .iq1_s => @sizeOf(gemm_iq1_s.BlockIQ1_S),
        .iq1_m => @sizeOf(gemm_iq1_m.BlockIQ1_M),
        .bf16 => 2,
        _ => 0,
    };
}

fn comptimeType(comptime fmt: Type) type {
    return switch (fmt) {
        .q8_0 => gemm_q8_0.BlockQ8_0,
        .q4_k => gemm_q4_k.BlockQ4_K,
        .q5_k => gemm_q5_k.BlockQ5_K,
        .q6_k => gemm_q6_k.BlockQ6_K,
        .q8_k => gemm_q8_k.BlockQ8_K,
        .q2_k => gemm_q2_k.BlockQ2_K,
        .q3_k => gemm_q3_k.BlockQ3_K,
        .iq4_xs => gemm_iq4_xs.BlockIQ4_XS,
        .iq4_nl => gemm_iq4_nl.BlockIQ4_NL,
        .iq2_xxs => gemm_iq2_xxs.BlockIQ2_XXS,
        .iq2_xs => gemm_iq2_xs.BlockIQ2_XS,
        .iq2_s => gemm_iq2_s.BlockIQ2_S,
        .iq3_xxs => gemm_iq3_xxs.BlockIQ3_XXS,
        .iq3_s => gemm_iq3_s.BlockIQ3_S,
        .iq1_s => gemm_iq1_s.BlockIQ1_S,
        .iq1_m => gemm_iq1_m.BlockIQ1_M,
        else => void,
    };
}

/// One GEMM (M x N x K) of BF16 activations x quantized weights -> F32.
/// `weight` points at the (n, k) row-major quantized matrix (blocks).
/// All kernels share the (a, lda, b, ldb, c, ldc, m, n, k) convention
/// with lda/ldc in ELEMENTS and ldb in BLOCKS (k / block_elems).
fn gemmQuant(
    fmt: Type,
    a: [*]const amx.bf16, // [m, k] BF16 activations
    w: [*]const u8, // [n, k] quantized weight rows (byte blocks)
    c: [*]f32, // [m, n] F32 out
    m: usize,
    n: usize,
    kk: usize,
) void {
    const b: [*]const u8 = w;
    switch (fmt) {
        // K-quant family: (a, lda, b, ldb, c, ldc, m, n, k), ldb in blocks.
        .q8_0 => gemm_q8_0.gemmQ8_0Scalar(a, kk, @ptrCast(@alignCast(b)), kk / gemm_q8_0.QK8_0, c, n, m, n, kk),
        .q4_k => gemm_q4_k.gemmQ4_KScalar(a, kk, @ptrCast(@alignCast(b)), kk / 256, c, n, m, n, kk),
        .q5_k => gemm_q5_k.gemmQ5_KScalar(a, kk, @ptrCast(@alignCast(b)), kk / 256, c, n, m, n, kk),
        .q6_k => gemm_q6_k.gemmQ6_KScalar(a, kk, @ptrCast(@alignCast(b)), kk / 256, c, n, m, n, kk),
        .q8_k => gemm_q8_k.gemmQ8_KScalar(a, kk, @ptrCast(@alignCast(b)), kk / 256, c, n, m, n, kk),
        .q2_k => gemm_q2_k.gemmQ2_KScalar(a, kk, @ptrCast(@alignCast(b)), kk / 256, c, n, m, n, kk),
        .q3_k => gemm_q3_k.gemmQ3_KScalar(a, kk, @ptrCast(@alignCast(b)), kk / 256, c, n, m, n, kk),
        // IQ family (a/b/c naming): (a, b, c, m, n, k, lda, ldb, ldc), ldb blocks.
        .iq2_xxs => gemm_iq2_xxs.gemmIQ2_XXSScalar(a, @ptrCast(@alignCast(b)), c, m, n, kk, kk, kk / 256, n),
        .iq2_xs => gemm_iq2_xs.gemmIQ2_XSScalar(a, @ptrCast(@alignCast(b)), c, m, n, kk, kk, kk / 256, n),
        .iq2_s => gemm_iq2_s.gemmIQ2_SScalar(a, @ptrCast(@alignCast(b)), c, m, n, kk, kk, kk / 256, n),
        .iq3_xxs => gemm_iq3_xxs.gemmIQ3_XXSScalar(a, @ptrCast(@alignCast(b)), c, m, n, kk, kk, kk / 256, n),
        .iq3_s => gemm_iq3_s.gemmIQ3_SScalar(a, @ptrCast(@alignCast(b)), c, m, n, kk, kk, kk / 256, n),
        .iq1_s => gemm_iq1_s.gemmIQ1_SScalar(a, @ptrCast(@alignCast(b)), c, m, n, kk, kk, kk / 256, n),
        .iq1_m => gemm_iq1_m.gemmIQ1_MScalar(a, @ptrCast(@alignCast(b)), c, m, n, kk, kk, kk / 256, n),
        // IQ4 family (named-params): (input, weight, output, m, n, k, ild, wld, old).
        .iq4_xs => gemm_iq4_xs.gemmIQ4_XSScalar(a, @ptrCast(@alignCast(b)), c, m, n, kk, kk, kk / 256, n),
        .iq4_nl => gemm_iq4_nl.gemmIQ4_NLScalar(a, @ptrCast(@alignCast(b)), c, m, n, kk, kk, kk / gemm_iq4_nl.QK4_NL, n),
        else => @panic("LlamaMoe: unsupported weight type"),
    }
}

pub const LlamaMoeConfig = struct {
    expert_num: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts_per_tok: usize,

    gate_proj: [*]const u8,
    up_proj: [*]const u8,
    down_proj: [*]const u8,
    gate_type: Type = .bf16,
    up_type: Type = .bf16,
    down_type: Type = .bf16,

    /// Optional bool mask: true = expert on GPU (skip on CPU), like the
    /// C++ gpu_experts_mask. Null = all experts local.
    gpu_experts_mask: ?[*]const u8 = null,
};

pub const LlamaMoe = struct {
    config: LlamaMoeConfig,
    allocator: std.mem.Allocator,

    pub fn init(config: LlamaMoeConfig, allocator: std.mem.Allocator) LlamaMoe {
        return .{ .config = config, .allocator = allocator };
    }

    fn shouldSkipExpert(self: *const LlamaMoe, expert_id: i64) bool {
        if (expert_id < 0) return true;
        if (expert_id >= @as(i64, @intCast(self.config.expert_num))) return true;
        if (self.config.gpu_experts_mask) |mask| {
            if (mask[@intCast(expert_id)] != 0) return true;
        }
        return false;
    }

    /// The routing weight of `expert_id`: the FIRST match in expert_ids
    /// (the C++ break-loop lookup at moe.hpp:437-443).
    fn weightOf(expert_ids: []const i64, weights: []const f32, expert_id: i64) f32 {
        for (expert_ids, 0..) |id, i| {
            if (id == expert_id) return weights[i];
        }
        return 0;
    }

    fn silu(x: f32) f32 {
        return x / (1.0 + @exp(-x));
    }

    /// Forward for ONE token with k activated experts (forward_one).
    /// input:  [hidden_size] BF16
    /// output: [hidden_size] F32 (zeroed then accumulated per expert)
    pub fn forwardOne(
        self: *LlamaMoe,
        input: [*]const amx.bf16,
        output: [*]f32,
        k: usize,
        expert_ids: []const i64,
        weights: []const f32,
    ) void {
        const cfg = self.config;
        std.debug.assert(k == expert_ids.len and expert_ids.len == weights.len);
        @memset(output[0..cfg.hidden_size], 0);

        // Per activated expert: gate+up+swiglu+down, accumulate.
        for (expert_ids) |expert_id| {
            if (self.shouldSkipExpert(expert_id)) continue;
            const e: usize = @intCast(expert_id);

            // gate GEMM: [1, hidden] x [inter, hidden]^T -> [1, inter]
            // Weight rows are quantized blocks: element offset ->
            // (elems / block_elems) * block_bytes.
            const gate_blck = cfg.gate_type.blockElems();
            const gate_row_blocks = cfg.hidden_size / gate_blck;
            const gate_w = cfg.gate_proj +
                (e * cfg.intermediate_size * gate_row_blocks) * rowBytes(cfg.gate_type);
            const gate_out = self.allocator.alloc(f32, cfg.intermediate_size) catch @panic("OOM");
            defer self.allocator.free(gate_out);
            gemmQuant(cfg.gate_type, input, gate_w, gate_out.ptr, 1, cfg.intermediate_size, cfg.hidden_size);

            // up GEMM
            const up_blck = cfg.up_type.blockElems();
            const up_row_blocks = cfg.hidden_size / up_blck;
            const up_w = cfg.up_proj +
                (e * cfg.intermediate_size * up_row_blocks) * rowBytes(cfg.up_type);
            const up_out = self.allocator.alloc(f32, cfg.intermediate_size) catch @panic("OOM");
            defer self.allocator.free(up_out);
            gemmQuant(cfg.up_type, input, up_w, up_out.ptr, 1, cfg.intermediate_size, cfg.hidden_size);

            // down GEMM: [1, inter] x [hidden, inter]^T -> [1, hidden]
            // (activations converted to BF16 first — matches the
            // from_float(vec_dot) step; BF16 precision, documented)
            const inter_bf16 = self.allocator.alloc(amx.bf16, cfg.intermediate_size) catch @panic("OOM");
            defer self.allocator.free(inter_bf16);
            for (0..cfg.intermediate_size) |i| {
                inter_bf16[i] = amx.f32_to_bf16(silu(gate_out[i]) * up_out[i]);
            }
            const down_blck = cfg.down_type.blockElems();
            const down_row_blocks = cfg.intermediate_size / down_blck;
            const down_w = cfg.down_proj +
                (e * cfg.hidden_size * down_row_blocks) * rowBytes(cfg.down_type);
            const down_out = self.allocator.alloc(f32, cfg.hidden_size) catch @panic("OOM");
            defer self.allocator.free(down_out);
            gemmQuant(cfg.down_type, inter_bf16.ptr, down_w, down_out.ptr, 1, cfg.hidden_size, cfg.intermediate_size);

            const w = weightOf(expert_ids, weights, expert_id);
            for (0..cfg.hidden_size) |i| {
                output[i] += down_out[i] * w;
            }
        }
    }
};

comptime {
    // Force analysis of every dispatch arm so lazy compilation cannot
    // hide a broken kernel signature (the lesson from the ABI audit).
    for (.{ .q8_0, .q4_k, .q5_k, .q6_k, .q8_k, .q2_k, .q3_k, .iq4_xs, .iq4_nl, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m }) |fmt| {
        _ = comptimeType(fmt);
    }
}
