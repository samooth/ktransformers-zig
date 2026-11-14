const std = @import("std");
const testing = std.testing;

const root = @import("kt");
const amx = root.amx;
const cfgmod = root.kimi_k3_config;
const mla = root.kimi_k3_mla_bf16;

test "mla bf16 incremental matches full" {
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
        .n_full_attn = 1,
        .full_attn = [_]i32{1} ++ [_]i32{0} ** 255,
    };

    const E: usize = cfg.hidden;
    const H: usize = cfg.n_heads;
    const qn: usize = cfg.qk_nope;
    const qr: usize = cfg.qk_rope;
    const vh: usize = cfg.v_head;
    const qh: usize = qn + qr;
    const kv_lora: usize = cfg.kv_lora;
    const kvw: usize = kv_lora + qr;
    const kvd: usize = qn + vh;

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
        .kv_b = mkBf16(H * kvd * kv_lora).ptr,
        .o = mkBf16(E * H * vh).ptr,
        .g = mkBf16(H * vh * E).ptr,
    };

    const T: usize = 2;
    const x = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(x);
    for (x) |*v| v.* = 0.5;
    const out_full = allocator.alloc(f32, T * E) catch unreachable;
    defer allocator.free(out_full);

    mla.mlaLayerFullBf16(out_full, x, &w, &cfg, T);

    // incremental
    var st = mla.MlaLayerState{
        .kv = mkF32(4 * H * kvd, 0.0),
        .rope = mkF32(4 * qr, 0.0),
        .cached = 0,
        .cap = 4,
    };
    const out0 = allocator.alloc(f32, E) catch unreachable;
    defer allocator.free(out0);
    const out1 = allocator.alloc(f32, E) catch unreachable;
    defer allocator.free(out1);

    mla.mlaLayerIncBf16(out0, x[0..E], &w, &cfg, &st);
    mla.mlaLayerIncBf16(out1, x[E..2*E], &w, &cfg, &st);

    for (0..E) |i| try testing.expectApproxEqAbs(out_full[i], out0[i], 1e-3);
    for (0..E) |i| try testing.expectApproxEqAbs(out_full[E + i], out1[i], 1e-3);
}
