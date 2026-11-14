const std = @import("std");
const testing = std.testing;

const root = @import("kt");
const amx = root.amx;
const cfgmod = root.kimi_k3_config;
const kda = root.kimi_k3_kda_bf16;

test "kda bf16 incremental matches full" {
    const allocator = std.heap.page_allocator;
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

    const mkBf16 = struct {
        fn f(n: usize) []amx.bf16 {
            const s = allocator.alloc(amx.bf16, n) catch unreachable;
            for (s) |*v| v.* = amx.f32_to_bf16(0.1);
            return s;
        }
    }.f;
    const mkF32 = struct {
        fn f(n: usize, val: f32) []f32 {
            const s = allocator.alloc(f32, n) catch unreachable;
            for (s) |*v| v.* = val;
            return s;
        }
    }.f;

    const w = kda.KdaBf16Weights{
        .q = mkBf16(P * E).ptr,
        .k = mkBf16(P * E).ptr,
        .v = mkBf16(P * E).ptr,
        .g = mkBf16(P * E).ptr,
        .b = mkBf16(H * E).ptr,
        .f_a = mkBf16(D * E).ptr,
        .f_b = mkBf16(P * D).ptr,
        .o = mkBf16(E * P).ptr,
        .q_conv = mkF32(P * K, 0.1).ptr,
        .k_conv = mkF32(P * K, 0.1).ptr,
        .v_conv = mkF32(P * K, 0.1).ptr,
        .A_log = mkF32(H, 0.0).ptr,
        .dt_bias = mkF32(P, 0.0).ptr,
        .o_norm = mkF32(D, 1.0).ptr,
    };

    const T: usize = 2;
    const x = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(x);
    for (x) |*v| v.* = 0.5;
    const out_full = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(out_full);

    kda.kdaLayerFullBf16(out_full, x, &w, &cfg, T);

    // incremental
    const hist = K - 1;
    var st = kda.KdaLayerState{
        .S = mkF32(H * D * D, 0.0),
        .conv = mkF32(3 * P * hist, 0.0),
    };
    const out0 = allocator.alloc(f32, E) catch unreachable;
    defer allocator.free(out0);
    const out1 = allocator.alloc(f32, E) catch unreachable;
    defer allocator.free(out1);

    kda.kdaLayerIncBf16(out0, x[0..E], &w, &cfg, &st);
    kda.kdaLayerIncBf16(out1, x[E..2*E], &w, &cfg, &st);

    for (0..E) |i| try testing.expectApproxEqAbs(out_full[i], out0[i], 1e-3);
    for (0..E) |i| try testing.expectApproxEqAbs(out_full[E + i], out1[i], 1e-3);
}
