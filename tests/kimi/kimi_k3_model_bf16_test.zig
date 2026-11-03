const std = @import("std");
const testing = std.testing;

const root = @import("kt");
const amx = root.amx;
const cfgmod = root.kimi_k3_config;
const layermod = root.kimi_k3_layer_bf16;
const modelmod = root.kimi_k3_model_bf16;
const kda = root.kimi_k3_kda_bf16;
const ops = root.kimi_k3_ops;

test "kimi model bf16 smoke" {
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
    const dense_inter: usize = cfg.dense_inter;

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

    const kda_w = kda.KdaBf16Weights{
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

    const layer_w = layermod.KimiLayerBf16Weights{
        .in_norm = mkF32(E, 1.0),
        .post_norm = mkF32(E, 1.0),
        .attn_res_norm = mkF32(E, 1.0),
        .attn_res_proj = mkF32(E, 1.0),
        .mlp_res_norm = mkF32(E, 1.0),
        .mlp_res_proj = mkF32(E, 1.0),
        .is_mla = false,
        .kda = kda_w,
        .mla = null,
        .is_dense = true,
        .dense_inter = dense_inter,
        .dense_gate = mkF32(dense_inter * E, 0.1),
        .dense_up = mkF32(dense_inter * E, 0.1),
        .dense_down = mkF32(E * dense_inter, 0.1),
        .moe = null,
        .moe_cfg = ops.LatentMoeCfg{
            .hidden = E,
            .latent = 2,
            .moe_inter = 2,
            .n_shared = 1,
            .n_experts = 2,
            .topk = 1,
            .situ_b1 = cfg.situ_b1,
            .situ_b2 = cfg.situ_b2,
            .routed_scale = cfg.routed_scale,
            .moe_renorm = cfg.moe_renorm,
            .latent_norm = cfg.latent_norm,
            .rms_eps = cfg.rms_eps,
        },
    };

    const layers = allocator.alloc(layermod.KimiLayerBf16Weights, 1) catch unreachable;
    layers[0] = layer_w;

    const final_norm = mkF32(E, 1.0);
    const model = modelmod.KimiModelBf16.init(allocator, &cfg, layers, final_norm, 2) catch unreachable;
    defer model.deinit();

    const T: usize = 2;
    const x = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(x);
    for (x) |*v| v.* = 0.5;
    const out = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(out);

    model.forward(&cfg, T, x, out);

    for (out) |v| try testing.expect(std.math.isFinite(v));
}