const std = @import("std");
const testing = std.testing;

const root = @import("kt");
const amx = root.amx;
const cfgmod = root.kimi_k3_config;
const mla = root.kimi_k3_mla_bf16;

test "mla bf16 layer smoke" {
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
    const H: usize = cfg.n_heads;
    const qn: usize = cfg.qk_nope;
    const qr: usize = cfg.qk_rope;
    const vh: usize = cfg.v_head;
    const qh: usize = qn + qr;
    const kv_lora: usize = cfg.kv_lora;
    const kvw: usize = kv_lora + qr;

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

    var w = mla.MlaBf16Weights{
        .q_a = mkBf16(cfg.q_lora * E).ptr,
        .q_a_norm = mkF32(cfg.q_lora, 1.0).ptr,
        .q_b = mkBf16(H * qh * cfg.q_lora).ptr,
        .kv_a = mkBf16(kvw * E).ptr,
        .kv_a_norm = mkF32(kv_lora, 1.0).ptr,
        .kv_b = mkBf16(H * (qn + vh) * kv_lora).ptr,
        .o = mkBf16(E * H * vh).ptr,
        .g = mkBf16(H * vh * E).ptr,
    };

    const T: usize = 2;
    const x = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(x);
    for (x) |*v| v.* = 0.5;
    const out = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(out);

    mla.mlaLayerFullBf16(out, x, &w, &cfg, T);

    for (out) |v| try testing.expect(std.math.isFinite(v));
}