// Standalone test for the kt_qwen3moe_* C API (model orchestration).
//
// Mirrors dsv3_layer_capi_tests.zig but for the Qwen3 surface:
//   kt_qwen3moe_layer_new(config) -> kt_qwen3moe_layer_forward
//   (2 decode steps) -> kt_qwen3moe_layer_free
//   + kt_qwen3moe_model_* lifecycle
//   + kt_qwen3moe_causallm_* lifecycle
//
// Exercises the full export path through the real C ABI: the
// by-pointer config marshaling, the opaque-handle lifecycle, and
// the softmax top-k gate path (vs DSV3's sigmoid+group-top2).
//
// Sanity check (with zero weights): RMSNorm(0-w)=0 → attn 0 →
// MoE 0 → residual passthrough → output == input.

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig (wired by build.zig)

test "kt_qwen3moe_layer C API: new -> 2 decode steps -> free" {
    const allocator = testing.allocator;

    const hidden: usize = 64;
    const num_heads: usize = 4;
    const num_kv_heads: usize = 2; // GQA: 2 query heads per KV head
    const head_dim: usize = 8;
    const max_qlen: usize = 2;
    const max_kvlen: usize = 8;
    const expert_num: usize = 4;
    const top_k: usize = 2;
    const inter: usize = 32;

    // Zeroed BF16 weight buffers (deterministic). Use a single big
    // buffer for all projections — they're all sized to the same hidden
    // alignment in this minimal test.
    const w = try allocator.alloc(u16, 64 * 1024);
    defer allocator.free(w);
    @memset(w, 0);
    // gate weight must be non-zero for the gate to produce a valid
    // softmax distribution (all-zero gate gives 0 logits → softmax
    // → undefined top-k). Use a small constant (BF16 0.5 = 0x3F00) so
    // the gate outputs 0.5 × 4 = 2.0 per expert → uniform softmax →
    // experts 0, 1, 2, 3 in top-k=2 (deterministic by argmax order).
    const gate_w = try allocator.alloc(u16, expert_num * hidden);
    defer allocator.free(gate_w);
    for (0..gate_w.len) |i| gate_w[i] = 0x3F00;

    const config = kt.kt_qwen3moe_layer_config_t{
        .hidden_size = hidden,
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .max_qlen = max_qlen,
        .max_kvlen = max_kvlen,
        .rope_theta = 1000000.0, // Qwen3 default
        .expert_num = expert_num,
        .num_experts_per_tok = top_k,
        .intermediate_size = inter,
        .pool = null,
        .q_proj = w.ptr,
        .k_proj = w.ptr,
        .v_proj = w.ptr,
        .o_proj = w.ptr,
        .attn_norm_weight = w.ptr,
        .ffn_norm_weight = w.ptr,
        .gate_weight = gate_w.ptr,
        .gate_proj = w.ptr,
        .up_proj = w.ptr,
        .down_proj = w.ptr,
    };

    const layer = kt.kt_qwen3moe_layer_new(&config);

    const inp = try allocator.alloc(u16, hidden);
    defer allocator.free(inp);
    // BF16(0.5) = 0x3F00
    @memset(inp, 0x3F00);
    const out = try allocator.alloc(u16, hidden);
    defer allocator.free(out);

    // Step 1: kv_start_pos = 0. Zero projections ⇒ attention output
    // is 0. RMSNorm(0-w)=0 ⇒ attn 0. MoE: gate logits = 0.5×hidden×1
    // (BF16 0.5 × BF16 1) per expert, top-k by argmax order (0, 1).
    // Zero expert weights ⇒ MoE output 0 ⇒ residual passthrough ⇒
    // out ≈ 0.5 (the input).
    @memset(out, 0);
    kt.kt_qwen3moe_layer_forward(layer, 1, 0, inp.ptr, out.ptr);
    for (0..hidden) |i| {
        const bits: u32 = @as(u32, out[i]) << 16;
        const v: f32 = @bitCast(bits);
        try testing.expectApproxEqAbs(@as(f32, 0.5), v, 0.02);
    }

    // Step 2: kv_start_pos = 1 (1 cached token from step 1).
    @memset(out, 0);
    kt.kt_qwen3moe_layer_forward(layer, 1, 1, inp.ptr, out.ptr);
    for (0..hidden) |i| {
        const bits: u32 = @as(u32, out[i]) << 16;
        const v: f32 = @bitCast(bits);
        try testing.expectApproxEqAbs(@as(f32, 0.5), v, 0.02);
    }

    kt.kt_qwen3moe_layer_free(layer);
}

test "kt_qwen3moe_layer C API: 2-token prefill" {
    const allocator = testing.allocator;

    const hidden: usize = 16;
    const num_heads: usize = 2;
    const num_kv_heads: usize = 1;
    const head_dim: usize = 8;
    const max_qlen: usize = 4;
    const max_kvlen: usize = 8;
    const expert_num: usize = 2;
    const top_k: usize = 1;
    const inter: usize = 8;

    const w = try allocator.alloc(u16, 16 * 1024);
    defer allocator.free(w);
    @memset(w, 0);
    const gate_w = try allocator.alloc(u16, expert_num * hidden);
    defer allocator.free(gate_w);
    @memset(gate_w, 0x3F00);

    const config = kt.kt_qwen3moe_layer_config_t{
        .hidden_size = hidden,
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .max_qlen = max_qlen,
        .max_kvlen = max_kvlen,
        .rope_theta = 1000000.0,
        .expert_num = expert_num,
        .num_experts_per_tok = top_k,
        .intermediate_size = inter,
        .pool = null,
        .q_proj = w.ptr,
        .k_proj = w.ptr,
        .v_proj = w.ptr,
        .o_proj = w.ptr,
        .attn_norm_weight = w.ptr,
        .ffn_norm_weight = w.ptr,
        .gate_weight = gate_w.ptr,
        .gate_proj = w.ptr,
        .up_proj = w.ptr,
        .down_proj = w.ptr,
    };

    const layer = kt.kt_qwen3moe_layer_new(&config);

    // 2 tokens, all BF16(1.0) = 0x3F80.
    var inp = [_]u16{0x3F80} ** (2 * hidden);
    var out: [2 * hidden]u16 = .{0} ** (2 * hidden);

    kt.kt_qwen3moe_layer_forward(layer, 2, 0, &inp, &out);

    // Zero projections ⇒ attention output 0 ⇒ RMSNorm(0-w)=0
    // ⇒ residual passthrough ⇒ out ≈ 1.0 (the input).
    for (0..2 * hidden) |i| {
        const bits: u32 = @as(u32, out[i]) << 16;
        const v: f32 = @bitCast(bits);
        try testing.expectApproxEqAbs(@as(f32, 1.0), v, 0.02);
    }

    kt.kt_qwen3moe_layer_free(layer);
}

test "kt_qwen3moe_model C API: new -> forward -> free" {
    const allocator = testing.allocator;

    const hidden: usize = 16;
    const num_heads: usize = 2;
    const num_kv_heads: usize = 1;
    const head_dim: usize = 8;
    const max_qlen: usize = 2;
    const max_kvlen: usize = 4;
    const expert_num: usize = 2;
    const top_k: usize = 1;
    const inter: usize = 8;
    const num_layers: usize = 1;
    const vocab_size: usize = 4;

    const w = try allocator.alloc(u16, 16 * 1024);
    defer allocator.free(w);
    @memset(w, 0);
    const gate_w = try allocator.alloc(u16, expert_num * hidden);
    defer allocator.free(gate_w);
    @memset(gate_w, 0x3F00);
    const norm_w = try allocator.alloc(u16, hidden);
    defer allocator.free(norm_w);
    // BF16(1.0) for the final RMSNorm weight so the residual path
    // preserves the input through to the model output.
    @memset(norm_w, 0x3F80);
    const lm_head = try allocator.alloc(u16, vocab_size * hidden);
    defer allocator.free(lm_head);
    @memset(lm_head, 0);

    const config = kt.kt_qwen3moe_model_config_t{
        .num_layers = num_layers,
        .layer = .{
            .hidden_size = hidden,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .max_qlen = max_qlen,
            .max_kvlen = max_kvlen,
            .rope_theta = 1000000.0,
            .expert_num = expert_num,
            .num_experts_per_tok = top_k,
            .intermediate_size = inter,
            .pool = null,
            .q_proj = w.ptr,
            .k_proj = w.ptr,
            .v_proj = w.ptr,
            .o_proj = w.ptr,
            .attn_norm_weight = w.ptr,
            .ffn_norm_weight = w.ptr,
            .gate_weight = gate_w.ptr,
            .gate_proj = w.ptr,
            .up_proj = w.ptr,
            .down_proj = w.ptr,
        },
        .final_norm_weight = norm_w.ptr,
        .lm_head = lm_head.ptr,
        .vocab_size = vocab_size,
    };

    const model = kt.kt_qwen3moe_model_new(&config);

    var inp = [_]u16{0x3F80} ** hidden; // BF16(1.0)
    var out: [hidden]u16 = .{0} ** hidden;
    kt.kt_qwen3moe_model_forward(model, 1, 0, &inp, &out);

    // Zero attn + zero experts + RMSNorm(1.0 * x) = x, plus zero
    // residual contribution from anywhere ⇒ out ≈ 1.0.
    for (0..hidden) |i| {
        const bits: u32 = @as(u32, out[i]) << 16;
        const v: f32 = @bitCast(bits);
        try testing.expectApproxEqAbs(@as(f32, 1.0), v, 0.02);
    }

    kt.kt_qwen3moe_model_free(model);
}

test "kt_qwen3moe_causallm C API: forward -> logits" {
    const allocator = testing.allocator;

    const hidden: usize = 8;
    const num_heads: usize = 1;
    const num_kv_heads: usize = 1;
    const head_dim: usize = 8;
    const max_qlen: usize = 1;
    const max_kvlen: usize = 2;
    const expert_num: usize = 1;
    const top_k: usize = 1;
    const inter: usize = 4;
    const num_layers: usize = 1;
    const vocab_size: usize = 4;

    const w = try allocator.alloc(u16, 4 * 1024);
    defer allocator.free(w);
    @memset(w, 0);
    const gate_w = try allocator.alloc(u16, expert_num * hidden);
    defer allocator.free(gate_w);
    @memset(gate_w, 0x3F00);
    const norm_w = try allocator.alloc(u16, hidden);
    defer allocator.free(norm_w);
    @memset(norm_w, 0x3F80);
    const lm_head = try allocator.alloc(u16, vocab_size * hidden);
    defer allocator.free(lm_head);
    @memset(lm_head, 0);

    const config = kt.kt_qwen3moe_model_config_t{
        .num_layers = num_layers,
        .layer = .{
            .hidden_size = hidden,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .max_qlen = max_qlen,
            .max_kvlen = max_kvlen,
            .rope_theta = 1000000.0,
            .expert_num = expert_num,
            .num_experts_per_tok = top_k,
            .intermediate_size = inter,
            .pool = null,
            .q_proj = w.ptr,
            .k_proj = w.ptr,
            .v_proj = w.ptr,
            .o_proj = w.ptr,
            .attn_norm_weight = w.ptr,
            .ffn_norm_weight = w.ptr,
            .gate_weight = gate_w.ptr,
            .gate_proj = w.ptr,
            .up_proj = w.ptr,
            .down_proj = w.ptr,
        },
        .final_norm_weight = norm_w.ptr,
        .lm_head = lm_head.ptr,
        .vocab_size = vocab_size,
    };

    const clm = kt.kt_qwen3moe_causallm_new(&config);

    var inp = [_]u16{0x3F80} ** hidden; // BF16(1.0)
    var logits = [_]f32{0.0} ** vocab_size;
    kt.kt_qwen3moe_causallm_forward(clm, 1, 0, &inp, &logits);

    // Zero lm_head ⇒ logits all 0.
    for (logits) |l| try testing.expectEqual(@as(f32, 0.0), l);

    kt.kt_qwen3moe_causallm_free(clm);
}
