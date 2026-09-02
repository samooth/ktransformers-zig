// Qwen3MoeDecoderLayer — model-orchestration layer for Qwen3 MoE.
//
// Ports the Qwen3 MoE forward pass (sft/layer.py:461-560, modeling
// Qwen3 MoE-style layers with standard MHA + MoE FFN).
//
//   residual = x
//   x = RMSNorm(x, attn_norm)            // pre-norm
//   x = MHA(x)                            // standard Q/K/V + RoPE
//   x = residual + x
//   residual = x
//   x = RMSNorm(x, ffn_norm)              // pre-norm
//   topk_ids, topk_weights = Qwen3Gate(x)  // softmax top-k, no group routing
//   x = MoE(x, topk_ids, topk_weights)     // BF16 GEMM + SwiGLU
//   x = residual + x
//
// This mirrors deepseekv3_layer.zig exactly in structure, but the
// attention is MHA (standard Q/K/V + GQA + RoPE) instead of MLA, and
// the gate is softmax-based (no group-top2) instead of
// sigmoid+group-top2.

const std = @import("std");
const amx = @import("../arch/amx.zig");
const moe_mod = @import("../moe/moe.zig");
const worker_pool = @import("../../runtime/worker_pool.zig");
const mha = @import("../attn/mha.zig");

pub const LayerConfig = struct {
    hidden_size: usize,
    // MHA dims
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_qlen: usize,
    max_kvlen: usize,
    rope_theta: f64 = 1000000.0, // Qwen3 default; much larger than DSV3's 10000
    // MoE dims
    expert_num: usize,
    num_experts_per_tok: usize,
    intermediate_size: usize,
    pool: ?*worker_pool.WorkerPool = null,

    // --- weight pointers (BF16 unless noted); all caller-owned ---
    // MHA
    q_proj: [*]const amx.bf16, // [num_heads*head_dim, hidden_size]
    k_proj: [*]const amx.bf16, // [num_kv_heads*head_dim, hidden_size]
    v_proj: [*]const amx.bf16, // [num_kv_heads*head_dim, hidden_size]
    o_proj: [*]const amx.bf16, // [hidden_size, num_heads*head_dim]
    // Layer norms (RMSNorm weights, [hidden_size] each)
    attn_norm_weight: [*]const amx.bf16,
    ffn_norm_weight: [*]const amx.bf16,
    // Router + experts
    gate_weight: [*]const amx.bf16, // [expert_num, hidden_size]
    gate_proj: [*]const amx.bf16, // [expert_num, intermediate, hidden]
    up_proj: [*]const amx.bf16,
    down_proj: [*]const amx.bf16,
};

pub const Qwen3MoeDecoderLayer = struct {
    config: LayerConfig,
    allocator: std.mem.Allocator,
    engine: *mha.MhaEngine,
    cache: *mha.MhaKvCache,
    moe: *moe_mod.TpMoe,

    // Scratch allocated once in init (sized to max_qlen)
    attn_norm_out: []f32, // [max_qlen * hidden]
    ffn_norm_out: []f32,
    attn_out: []f32, // [max_qlen * hidden]
    ffn_out: []amx.bf16, // [max_qlen * hidden]
    topk_ids: []i64, // [max_qlen * num_experts_per_tok]
    topk_weights: []f32,
    residual: []f32, // [max_qlen * hidden]

    pub fn init(allocator: std.mem.Allocator, config: LayerConfig) !*Qwen3MoeDecoderLayer {
        const mha_cfg = mha.MhaConfig{
            .hidden_size = config.hidden_size,
            .num_heads = config.num_heads,
            .num_kv_heads = config.num_kv_heads,
            .head_dim = config.head_dim,
            .max_qlen = config.max_qlen,
            .max_kvlen = config.max_kvlen,
            .rope_theta = config.rope_theta,
            .q_proj = @ptrCast(config.q_proj),
            .k_proj = @ptrCast(config.k_proj),
            .v_proj = @ptrCast(config.v_proj),
            .o_proj = @ptrCast(config.o_proj),
        };
        const cache = try allocator.create(mha.MhaKvCache);
        cache.* = try mha.MhaKvCache.init(
            allocator,
            config.num_kv_heads,
            config.head_dim,
            config.max_kvlen,
        );
        const engine = try allocator.create(mha.MhaEngine);
        engine.* = try mha.MhaEngine.init(allocator, mha_cfg, cache);

        const moe_inst = try allocator.create(moe_mod.TpMoe);
        moe_inst.* = try moe_mod.TpMoe.init(.{
            .expert_num = @intCast(config.expert_num),
            .num_experts_per_tok = @intCast(config.num_experts_per_tok),
            .hidden_size = @intCast(config.hidden_size),
            .intermediate_size = @intCast(config.intermediate_size),
            .gate_proj = @constCast(config.gate_proj),
            .up_proj = @constCast(config.up_proj),
            .down_proj = @constCast(config.down_proj),
            .pool = config.pool,
        }, allocator);
        moe_inst.loadWeights();

        const hidden = config.hidden_size;
        const n_tok = config.max_qlen * config.num_experts_per_tok;
        const self = try allocator.create(Qwen3MoeDecoderLayer);
        self.* = .{
            .config = config,
            .allocator = allocator,
            .engine = engine,
            .cache = cache,
            .moe = moe_inst,
            .attn_norm_out = try allocator.alloc(f32, config.max_qlen * hidden),
            .ffn_norm_out = try allocator.alloc(f32, config.max_qlen * hidden),
            .attn_out = try allocator.alloc(f32, config.max_qlen * hidden),
            .ffn_out = try allocator.alloc(amx.bf16, config.max_qlen * hidden),
            .topk_ids = try allocator.alloc(i64, n_tok),
            .topk_weights = try allocator.alloc(f32, n_tok),
            .residual = try allocator.alloc(f32, config.max_qlen * hidden),
        };
        return self;
    }

    pub fn deinit(self: *Qwen3MoeDecoderLayer) void {
        self.engine.deinit();
        self.cache.deinit();
        self.moe.deinit();
        self.allocator.destroy(self.engine);
        self.allocator.destroy(self.cache);
        self.allocator.destroy(self.moe);
        self.allocator.free(self.attn_norm_out);
        self.allocator.free(self.ffn_norm_out);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.ffn_out);
        self.allocator.free(self.topk_ids);
        self.allocator.free(self.topk_weights);
        self.allocator.free(self.residual);
        self.allocator.destroy(self);
    }

    pub fn forward(
        self: *Qwen3MoeDecoderLayer,
        qlen: usize,
        kv_start_pos: usize,
        input: [*]const amx.bf16,
        output: [*]amx.bf16,
    ) void {
        const cfg = self.config;
        const hidden = cfg.hidden_size;
        const eps: f32 = 1e-6;
        const n = qlen * hidden;

        // BF16 -> f32 work buffer
        var x = self.attn_out[0..n];
        for (0..n) |i| x[i] = amx.bf16_to_f32(input[i]);

        // ============ block 1: attention (pre-norm + MHA + residual) ============
        @memcpy(self.residual[0..n], x);
        for (0..qlen) |t| {
            mha.rmsNormInline(
                x[t * hidden ..][0..hidden],
                @ptrCast(cfg.attn_norm_weight),
                self.attn_norm_out[0..hidden],
                eps,
            );
            @memcpy(x[t * hidden ..][0..hidden], self.attn_norm_out[0..hidden]);
        }
        self.engine.forward(x.ptr, self.attn_norm_out.ptr, qlen, kv_start_pos);
        for (0..n) |i| x[i] = self.residual[i] + self.attn_norm_out[i];

        // ============ block 2: FFN/MoE (pre-norm + gate + MoE + residual) ============
        @memcpy(self.residual[0..n], x);
        for (0..qlen) |t| {
            mha.rmsNormInline(
                x[t * hidden ..][0..hidden],
                @ptrCast(cfg.ffn_norm_weight),
                self.ffn_norm_out[0..hidden],
                eps,
            );
            @memcpy(x[t * hidden ..][0..hidden], self.ffn_norm_out[0..hidden]);
        }
        const bridge_bf16 = self.allocator.alloc(amx.bf16, n) catch @panic("OOM");
        defer self.allocator.free(bridge_bf16);
        for (0..n) |i| bridge_bf16[i] = amx.f32_to_bf16(x[i]);

        // Qwen3 gate: vanilla top-k softmax (no group routing, no bias).
        // Use the existing routeExpertsDeepSeek with the right opts.
        moe_mod.routeExpertsDeepSeek(
            self.allocator,
            bridge_bf16.ptr,
            cfg.gate_weight,
            qlen,
            hidden,
            cfg.expert_num,
            cfg.num_experts_per_tok,
            .{
                .scoring = .softmax, // Qwen3: softmax over expert dim
                .bias = null, // no e_score_correction_bias in Qwen3
                .n_group = 1, // no group-top2
                .topk_group = 1,
                .norm_topk_prob = false,
                .routed_scaling_factor = 1.0,
            },
            self.topk_ids.ptr,
            self.topk_weights.ptr,
            null, // sequential routing
        );

        // MoE experts forward (work-stealing if a pool is configured)
        self.moe.forward(
            qlen,
            @intCast(cfg.num_experts_per_tok),
            self.topk_ids.ptr,
            self.topk_weights.ptr,
            bridge_bf16.ptr,
            self.ffn_out.ptr,
            false,
        );

        // residual add -> BF16 output
        for (0..n) |i| {
            const v = self.residual[i] + amx.bf16_to_f32(self.ffn_out[i]);
            output[i] = amx.f32_to_bf16(v);
        }
    }
};
