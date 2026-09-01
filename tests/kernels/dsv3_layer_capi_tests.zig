// Standalone test for the kt_dsv3_layer_* C API (model orchestration).
//
// Exercises the full export path through the real C ABI:
//   kt_dsv3_layer_new(config) -> kt_dsv3_layer_forward (2 decode steps)
//   -> kt_dsv3_layer_free — with the exact residual semantics verified in
//   the kernel-layer test (test_kernels.zig) re-checked here through the
//   extern-struct config marshaling (the by-value/pointer boundary is what
//   this suite adds coverage for, plus the opaque-handle lifecycle).

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig (wired by build.zig)

test "kt_dsv3_layer C API: new -> 2 decode steps -> free" {
    const allocator = testing.allocator;

    const hidden: usize = 64;
    const q_lora_rank: usize = 32;
    const num_heads: usize = 4;
    const nope_size: usize = 8;
    const rope_size: usize = 4;
    const kv_lora_rank: usize = 16;
    const max_qlen: usize = 2;
    const max_kvlen: usize = 16;
    const tpp: usize = 4;
    const expert_num: usize = 4;
    const top_k: usize = 2;
    const inter: usize = 32;

    // Zeroed BF16 weight buffers (deterministic: RMSNorm(0-w) -> 0 ->
    // attn 0 -> output == input through the residual path).
    const w = try allocator.alloc(u16, 64 * 1024);
    defer allocator.free(w);
    @memset(w, 0);
    const gate_w = try allocator.alloc(u16, expert_num * hidden);
    defer allocator.free(gate_w);
    @memset(gate_w, 0);

    const config = kt.kt_dsv3_layer_config_t{
        .hidden_size = hidden,
        .q_lora_rank = q_lora_rank,
        .num_heads = num_heads,
        .nope_size = nope_size,
        .rope_size = rope_size,
        .kv_lora_rank = kv_lora_rank,
        .max_qlen = max_qlen,
        .max_kvlen = max_kvlen,
        .token_count_in_page = tpp,
        .rope_theta = 10000.0,
        .expert_num = expert_num,
        .num_experts_per_tok = top_k,
        .intermediate_size = inter,
        .n_group = 2,
        .topk_group = 1,
        .norm_topk_prob = 1,
        .routed_scaling_factor = 1.0,
        .pool = null,
        .q_a_proj = w.ptr,
        .q_a_norm = w.ptr,
        .q_b_proj = w.ptr,
        .kv_a_proj_with_mqa = w.ptr,
        .kv_a_norm = w.ptr,
        .kv_b_proj = w.ptr,
        .o_proj = w.ptr,
        .attn_norm_weight = w.ptr,
        .ffn_norm_weight = w.ptr,
        .gate_weight = gate_w.ptr,
        .e_score_correction_bias = null,
        .gate_proj = w.ptr,
        .up_proj = w.ptr,
        .down_proj = w.ptr,
    };

    const layer = kt.kt_dsv3_layer_new(&config);

    const inp = try allocator.alloc(u16, hidden);
    defer allocator.free(inp);
    // BF16(0.5) = 0x3F00 (f32 0.5 = 0x3F000000 -> upper 16 bits)
    @memset(inp, 0x3F00);
    const out = try allocator.alloc(u16, hidden);
    defer allocator.free(out);

    // Step 1: kv_start_pos = 0. Zero weights => output == input through the
    // residual path (RMSNorm(0-w)=0 -> attn 0 -> MoE 0 -> residual passthrough).
    // BF16 decode: u16 -> upper half of f32 bits.
    @memset(out, 0);
    kt.kt_dsv3_layer_forward(layer, 1, 0, inp.ptr, out.ptr);
    for (0..hidden) |i| {
        const bits: u32 = @as(u32, out[i]) << 16;
        const v: f32 = @bitCast(bits);
        try testing.expectApproxEqAbs(@as(f32, 0.5), v, 0.01);
    }

    // Step 2: kv_start_pos = 1 (1 cached token)
    @memset(out, 0);
    kt.kt_dsv3_layer_forward(layer, 1, 1, inp.ptr, out.ptr);
    for (0..hidden) |i| {
        const bits: u32 = @as(u32, out[i]) << 16;
        const v: f32 = @bitCast(bits);
        try testing.expectApproxEqAbs(@as(f32, 0.5), v, 0.01);
    }

    kt.kt_dsv3_layer_free(layer);
}
