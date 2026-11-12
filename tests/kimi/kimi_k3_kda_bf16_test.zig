const std = @import("std");
const testing = std.testing;

const root = @import("kt");
const amx = root.amx;
const cfgmod = root.kimi_k3_config;
const kda = root.kimi_k3_kda_bf16;

test "kda bf16 layer smoke" {
    const allocator = std.heap.page_allocator;
    // tiny config
    var cfg = cfgmod.KimiK3Config{
        .hidden = 4,
        .n_layers = 1,
        .vocab = 8,
        .rms_eps = 1e-5,
        .kda_heads = 2,
        .kda_head_dim = 2,
        .conv_k = 2,
        .gate_lb = -5.0,
        .n_heads = 2,
        .q_lora = 4,
        .kv_lora = 2,
        .qk_nope = 2,
        .qk_rope = 2,
        .v_head = 2,
        .mla_out_gate = true,
        .n_experts = 2,
        .topk = 1,
        .n_shared = 1,
        .latent = 2,
        .moe_inter = 2,
        .routed_scale = 1.0,
        .moe_renorm = true,
        .latent_norm = true,
        .first_dense = 1,
        .dense_inter = 4,
        .attn_res_block = 1,
        .situ_b1 = 4.0,
        .situ_b2 = 25.0,
        .n_full_attn = 0,
    };

    const E: usize = cfg.hidden;
    const H: usize = cfg.kda_heads;
    const D: usize = cfg.kda_head_dim;
    const P: usize = H * D;
    const K: usize = cfg.conv_k;

    // allocate bf16 weights (small, ones)
    const mkBf16 = struct {
        fn f(n: usize) []amx.bf16 {
            const s = allocator.alloc(amx.bf16, n) catch unreachable;
            for (s) |*v| v.* = amx.f32_to_bf16(0.1);
            return s;
        }
    }.f;

    var w = kda.KdaBf16Weights{
        .q = mkBf16(P * E).ptr,
        .k = mkBf16(P * E).ptr,
        .v = mkBf16(P * E).ptr,
        .g = mkBf16(P * E).ptr,
        .b = mkBf16(H * E).ptr,
        .f_a = mkBf16(D * E).ptr,
        .f_b = mkBf16(P * D).ptr,
        .o = mkBf16(E * P).ptr,
        .q_conv = (blk: {
            const s = allocator.alloc(f32, P * K) catch unreachable;
            for (s) |*v| v.* = 0.1;
            break :blk s.ptr;
        }),
        .k_conv = (blk: {
            const s = allocator.alloc(f32, P * K) catch unreachable;
            for (s) |*v| v.* = 0.1;
            break :blk s.ptr;
        }),
        .v_conv = (blk: {
            const s = allocator.alloc(f32, P * K) catch unreachable;
            for (s) |*v| v.* = 0.1;
            break :blk s.ptr;
        }),
        .A_log = (blk: {
            const s = allocator.alloc(f32, H) catch unreachable;
            for (s) |*v| v.* = 0.0;
            break :blk s.ptr;
        }),
        .dt_bias = (blk: {
            const s = allocator.alloc(f32, P) catch unreachable;
            for (s) |*v| v.* = 0.0;
            break :blk s.ptr;
        }),
        .o_norm = (blk: {
            const s = allocator.alloc(f32, D) catch unreachable;
            for (s) |*v| v.* = 1.0;
            break :blk s.ptr;
        }),
    };

    const T: usize = 2;
    const x = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(x);
    for (x) |*v| v.* = 0.5;
    const out = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(out);

    kda.kdaLayerFullBf16(out, x, &w, &cfg, T);

    for (out) |v| try testing.expect(std.math.isFinite(v));
}