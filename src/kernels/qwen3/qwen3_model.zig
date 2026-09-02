// Qwen3MoeModel + Qwen3MoeForCausalLM — model-level orchestration.
//
// Mirrors deepseekv3_model.zig exactly:
//   Qwen3MoeModel:    embed -> N x Qwen3MoeDecoderLayer -> final RMSNorm
//   Qwen3MoeForCausalLM: Model + lm_head (hidden -> vocab logits)
//
// Embeddings are the caller's job in the C-API port (token ids ->
// [hidden_size] rows are computed Python-side; this layer accepts the
// embedded hidden states directly — matches the Qwen3 reference which
// also receives inputs_embeds). lm_head is a plain BF16 GEMM.
//
// Weight pointers are caller-owned (borrowed), per the C-API convention.

const std = @import("std");
const amx = @import("../arch/amx.zig");
const layer_mod = @import("qwen3_layer.zig");
const gemm_bf16 = @import("../amx/gemm_224_bf16.zig");

pub const ModelConfig = struct {
    num_layers: usize,
    layer: layer_mod.LayerConfig, // shared dims/weights template per layer
    final_norm_weight: [*]const amx.bf16, // [hidden_size] RMSNorm
};

pub const Qwen3MoeModel = struct {
    config: ModelConfig,
    allocator: std.mem.Allocator,
    layers: []*layer_mod.Qwen3MoeDecoderLayer,
    scratch: []amx.bf16, // ping-pong buffer: [2][max_qlen * hidden]

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !*Qwen3MoeModel {
        const layers = try allocator.alloc(*layer_mod.Qwen3MoeDecoderLayer, config.num_layers);
        for (layers, 0..) |*slot, i| {
            slot.* = try layer_mod.Qwen3MoeDecoderLayer.init(allocator, config.layer);
            _ = i;
        }
        const n = config.layer.max_qlen * config.layer.hidden_size;
        const self = try allocator.create(Qwen3MoeModel);
        self.* = .{
            .config = config,
            .allocator = allocator,
            .layers = layers,
            .scratch = try allocator.alloc(amx.bf16, 2 * n),
        };
        return self;
    }

    pub fn deinit(self: *Qwen3MoeModel) void {
        for (self.layers) |l| l.deinit();
        self.allocator.free(self.layers);
        self.allocator.free(self.scratch);
        self.allocator.destroy(self);
    }

    /// hidden_states: embedded inputs [qlen, hidden] BF16.
    /// output: post-norm final hidden [qlen, hidden] BF16.
    pub fn forward(
        self: *Qwen3MoeModel,
        qlen: usize,
        kv_start_pos: usize,
        hidden_states: [*]const amx.bf16,
        output: [*]amx.bf16,
    ) void {
        const n = qlen * self.config.layer.hidden_size;

        // Ping-pong through layers: cur holds the running hidden state.
        var cur: [*]const amx.bf16 = hidden_states;
        var buf_idx: usize = 0;
        const bufs = [2][*]amx.bf16{
            self.scratch.ptr,
            self.scratch.ptr + n,
        };

        for (self.layers) |layer| {
            const dst = bufs[buf_idx];
            layer.forward(qlen, kv_start_pos, cur, dst);
            cur = dst;
            buf_idx ^= 1;
        }

        // Final RMSNorm — f32 in/out with a per-row scratch, then BF16.
        const hidden = self.config.layer.hidden_size;
        const eps: f32 = 1e-6;
        var row_scratch: [4096]f32 = undefined;
        std.debug.assert(hidden <= row_scratch.len);
        const x_f32 = self.allocator.alloc(f32, n) catch @panic("OOM");
        defer self.allocator.free(x_f32);
        for (0..n) |i| x_f32[i] = amx.bf16_to_f32(cur[i]);
        for (0..qlen) |t| {
            rmsNormInline(
                x_f32[t * hidden ..][0..hidden],
                @ptrCast(self.config.final_norm_weight),
                row_scratch[0..hidden],
                eps,
            );
            @memcpy(x_f32[t * hidden ..][0..hidden], row_scratch[0..hidden]);
        }
        for (0..n) |i| output[i] = amx.f32_to_bf16(x_f32[i]);
    }
};

/// Final-norm helper (mirrors the inline impl in deepseekv3_model.zig;
/// kept here too so this file is self-contained).
fn rmsNormInline(x: []const f32, weight: [*]const u16, out: []f32, eps: f32) void {
    var sum_sq: f32 = 0;
    for (x) |v| sum_sq += v * v;
    const rms = std.math.sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    const inv = 1.0 / rms;
    for (x, 0..) |v, i| {
        out[i] = v * inv * amx.bf16_to_f32(weight[i]);
    }
}

pub const CausalLMConfig = struct {
    model: ModelConfig,
    lm_head: [*]const amx.bf16, // [vocab_size, hidden_size] BF16
    vocab_size: usize,
};

pub const Qwen3MoeForCausalLM = struct {
    config: CausalLMConfig,
    allocator: std.mem.Allocator,
    model: *Qwen3MoeModel,
    logits_scratch: []f32, // [max_qlen * vocab_size]

    pub fn init(allocator: std.mem.Allocator, config: CausalLMConfig) !*Qwen3MoeForCausalLM {
        const model = try Qwen3MoeModel.init(allocator, config.model);
        const self = try allocator.create(Qwen3MoeForCausalLM);
        self.* = .{
            .config = config,
            .allocator = allocator,
            .model = model,
            .logits_scratch = try allocator.alloc(f32, config.model.layer.max_qlen * config.vocab_size),
        };
        return self;
    }

    pub fn deinit(self: *Qwen3MoeForCausalLM) void {
        self.model.deinit();
        self.allocator.free(self.logits_scratch);
        self.allocator.destroy(self);
    }

    /// Full forward: embedded hidden states -> vocab logits (f32).
    pub fn forward(
        self: *Qwen3MoeForCausalLM,
        qlen: usize,
        kv_start_pos: usize,
        hidden_states: [*]const amx.bf16,
        logits: [*]f32,
    ) void {
        const n = qlen * self.config.model.layer.hidden_size;
        const final_hidden = self.allocator.alloc(amx.bf16, n) catch @panic("OOM");
        defer self.allocator.free(final_hidden);
        self.model.forward(qlen, kv_start_pos, hidden_states, final_hidden.ptr);

        // lm_head: [qlen, hidden] x [vocab, hidden]^T -> [qlen, vocab]
        gemm_bf16.gemmExpert(
            final_hidden.ptr,
            self.config.lm_head,
            logits,
            qlen,
            self.config.vocab_size,
            self.config.model.layer.hidden_size,
            self.config.model.layer.hidden_size, // input_ld = k (row-major)
            self.config.model.layer.hidden_size, // weight_ld = k (row-major [n,k])
            self.config.vocab_size, // output_ld = n
        );
    }
};
