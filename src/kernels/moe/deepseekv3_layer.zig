// DeepseekV3DecoderLayer — model-orchestration layer for the Zig port.
//
// Ports the Python reference (kt-kernel/examples/modeling_deepseek_v3.py):
//   forward (layer): residual = x; x = RMSNorm(x, attn_norm_w)
//                    x = MLA(x); x = residual + x
//                    residual = x; x = RMSNorm(x, ffn_norm_w)
//                    x = MoE(gate, x);  x = residual + x
//   MoEGate (noaux_tc group-top2): sigmoid scores + e_score_correction_bias,
//     group top-2 sums -> topk_group groups -> top_k experts, weights
//     normalized (norm_topk_prob) and scaled (routed_scaling_factor).
//
// The orchestration is single-sequence (qlen tokens, continuous KV) — the
// serving-model paged-attention variant is future work (see moe.zig for the
// same constraint on the MoE side). All kernels already exist:
//   rmsNorm (mla_core), MlaEngine.forward, routeExpertsDeepSeek, TpMoe.forward.
//
// Owned contexts are created here and freed in deinit; weight pointers are
// borrowed (caller-owned, as everywhere in this C API).

const std = @import("std");
const amx = @import("../arch/amx.zig");
const moe_mod = @import("moe.zig");
const mla_config_mod = @import("../../mla/mla_config.zig");
const mla_cache_mod = @import("../../mla/mla_cache.zig");
const mla_core = @import("../../mla/mla_core.zig");
const worker_pool = @import("../../runtime/worker_pool.zig");

pub const LayerConfig = struct {
    hidden_size: usize,
    // MLA dims
    q_lora_rank: usize,
    num_heads: usize,
    nope_size: usize,
    rope_size: usize,
    kv_lora_rank: usize,
    max_qlen: usize,
    max_kvlen: usize,
    token_count_in_page: usize,
    rope_theta: f64 = 10000.0,
    // MoE dims
    expert_num: usize,
    num_experts_per_tok: usize,
    intermediate_size: usize,
    n_group: usize = 1,
    topk_group: usize = 1,
    norm_topk_prob: bool = true,
    routed_scaling_factor: f32 = 1.0,
    pool: ?*worker_pool.WorkerPool = null,

    // --- weight pointers (BF16 unless noted); all caller-owned ---
    // MLA
    q_a_proj: [*]const amx.bf16,
    q_a_norm: [*]const amx.bf16,
    q_b_proj: [*]const amx.bf16,
    kv_a_proj_with_mqa: [*]const amx.bf16,
    kv_a_norm: [*]const amx.bf16,
    kv_b_proj: [*]const amx.bf16,
    o_proj: [*]const amx.bf16,
    // Layer norms (RMSNorm weights, [hidden_size] each)
    attn_norm_weight: [*]const amx.bf16,
    ffn_norm_weight: [*]const amx.bf16,
    // Router + experts
    gate_weight: [*]const amx.bf16, // [expert_num, hidden_size]
    e_score_correction_bias: ?[*]const f32 = null, // noaux_tc; null = no bias
    gate_proj: [*]const amx.bf16, // [expert_num, inter, hidden]
    up_proj: [*]const amx.bf16,
    down_proj: [*]const amx.bf16,
};

pub const DeepseekV3DecoderLayer = struct {
    config: LayerConfig,
    allocator: std.mem.Allocator,
    engine: *mla_core.MlaEngine,
    cache: *mla_cache_mod.MlaKvCache,
    moe: *moe_mod.TpMoe,
    // Scratch allocated once in init (sized to max_qlen):
    attn_out: []f32, // [max_qlen * hidden]
    ffn_out: []amx.bf16, // [max_qlen * hidden]
    topk_ids: []i64, // [max_qlen * num_experts_per_tok]
    topk_weights: []f32,
    residual: []f32, // [max_qlen * hidden]

    pub fn init(allocator: std.mem.Allocator, config: LayerConfig) !*DeepseekV3DecoderLayer {
        // MLA engine (kv cache derived from max dims)
        const mla_cfg = mla_config_mod.MlaConfig{
            .hidden_size = config.hidden_size,
            .q_lora_rank = config.q_lora_rank,
            .num_heads = config.num_heads,
            .nope_size = config.nope_size,
            .rope_size = config.rope_size,
            .kv_lora_rank = config.kv_lora_rank,
            .max_qlen = config.max_qlen,
            .max_kvlen = config.max_kvlen,
            .token_count_in_page = config.token_count_in_page,
            .rope_theta = config.rope_theta,
            .q_a_proj = @ptrCast(config.q_a_proj),
            .q_a_norm = @ptrCast(config.q_a_norm),
            .q_b_proj = @ptrCast(config.q_b_proj),
            .kv_a_proj_with_mqa = @ptrCast(config.kv_a_proj_with_mqa),
            .kv_a_norm = @ptrCast(config.kv_a_norm),
            .kv_b_proj = @ptrCast(config.kv_b_proj),
            .o_proj = @ptrCast(config.o_proj),
        };
        const max_pages = (config.max_kvlen / @max(config.token_count_in_page, 1)) + 1;
        const cache = try allocator.create(mla_cache_mod.MlaKvCache);
        cache.* = try mla_cache_mod.MlaKvCache.init(allocator, mla_cfg, max_pages);
        const engine = try allocator.create(mla_core.MlaEngine);
        engine.* = try mla_core.MlaEngine.init(allocator, mla_cfg, cache);

        // MoE
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
        // loadWeights packs the BF16 weight pointers into the module-level
        // storage the per-expert compute reads.
        moe_inst.loadWeights();

        const hidden = config.hidden_size;
        const n_tok = config.max_qlen * config.num_experts_per_tok;
        const self = try allocator.create(DeepseekV3DecoderLayer);
        self.* = .{
            .config = config,
            .allocator = allocator,
            .engine = engine,
            .cache = cache,
            .moe = moe_inst,
            .attn_out = try allocator.alloc(f32, config.max_qlen * hidden),
            .ffn_out = try allocator.alloc(amx.bf16, config.max_qlen * hidden),
            .topk_ids = try allocator.alloc(i64, n_tok),
            .topk_weights = try allocator.alloc(f32, n_tok),
            .residual = try allocator.alloc(f32, config.max_qlen * hidden),
        };
        return self;
    }

    pub fn deinit(self: *DeepseekV3DecoderLayer) void {
        self.engine.deinit();
        self.cache.deinit();
        self.moe.deinit();
        self.allocator.destroy(self.engine);
        self.allocator.destroy(self.cache);
        self.allocator.destroy(self.moe);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.ffn_out);
        self.allocator.free(self.topk_ids);
        self.allocator.free(self.topk_weights);
        self.allocator.free(self.residual);
        self.allocator.destroy(self);
    }

    /// One forward pass over `qlen` tokens. `input`/`output` are BF16
    /// [qlen, hidden]; `kv_start_pos` positions of context already exist.
    /// Single sequence (the continuous-KV serving shape).
    pub fn forward(
        self: *DeepseekV3DecoderLayer,
        qlen: usize,
        kv_start_pos: usize,
        input: [*]const amx.bf16,
        output: [*]amx.bf16,
    ) void {
        const cfg = self.config;
        const hidden = cfg.hidden_size;
        const eps: f32 = 1e-6;
        const n = qlen * hidden;

        // BF16 -> f32 hidden-state work buffer
        const x = self.attn_out[0..n];
        for (0..n) |i| x[i] = amx.bf16_to_f32(input[i]);

        // Per-call scratch (f32 norm output + BF16 bridges)
        const norm_scratch = self.allocator.alloc(f32, n) catch @panic("OOM");
        defer self.allocator.free(norm_scratch);
        const bridge_bf16 = self.allocator.alloc(amx.bf16, n) catch @panic("OOM");
        defer self.allocator.free(bridge_bf16);
        const mla_out = self.allocator.alloc(f32, n) catch @panic("OOM");
        defer self.allocator.free(mla_out);

        // ============ block 1: attention (modeling_deepseek_v3.py:1196) ============
        @memcpy(self.residual[0..n], x);
        for (0..qlen) |t| {
            mla_core.rmsNorm(
                x[t * hidden ..].ptr,
                @ptrCast(cfg.attn_norm_weight),
                norm_scratch[t * hidden ..].ptr,
                hidden,
                eps,
            );
        }
        @memcpy(x, norm_scratch);

        for (0..n) |i| bridge_bf16[i] = amx.f32_to_bf16(x[i]);
        self.engine.forward(norm_scratch.ptr, mla_out.ptr, qlen, kv_start_pos) catch @panic("MLA forward failed");
        for (0..n) |i| x[i] = self.residual[i] + mla_out[i];

        // ============ block 2: FFN/MoE (modeling_deepseek_v3.py:1215) ============
        @memcpy(self.residual[0..n], x);
        for (0..qlen) |t| {
            mla_core.rmsNorm(
                x[t * hidden ..].ptr,
                @ptrCast(cfg.ffn_norm_weight),
                norm_scratch[t * hidden ..].ptr,
                hidden,
                eps,
            );
        }
        @memcpy(x, norm_scratch);
        for (0..n) |i| bridge_bf16[i] = amx.f32_to_bf16(x[i]);

        // Router: DeepSeek-V3 noaux_tc group-top2 (:394-461)
        moe_mod.routeExpertsDeepSeek(
            bridge_bf16.ptr,
            cfg.gate_weight,
            qlen,
            hidden,
            cfg.expert_num,
            cfg.num_experts_per_tok,
            .{
                .scoring = .sigmoid,
                .bias = cfg.e_score_correction_bias,
                .n_group = cfg.n_group,
                .topk_group = cfg.topk_group,
                .norm_topk_prob = cfg.norm_topk_prob,
                .routed_scaling_factor = cfg.routed_scaling_factor,
            },
            self.topk_ids.ptr,
            self.topk_weights.ptr,
            null, // routing is O(qlen*k); sequential
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
