const std = @import("std");
const testing = std.testing;
const kt = @import("kt");

test "kt_kimi_kda_layer C API: new -> forward -> free" {
    const allocator = std.heap.page_allocator;

    const hidden: usize = 4;
    const heads: usize = 2;
    const head_dim: usize = 2;
    const P: usize = heads * head_dim;
    const conv_k: usize = 2;
    const E: usize = hidden;

    const mkBf16 = struct {
        fn f(n: usize) []u16 {
            const s = allocator.alloc(u16, n) catch unreachable;
            for (s) |*v| v.* = 0x3DCC;
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

    const q_w = mkBf16(P * E);
    const k_w = mkBf16(P * E);
    const v_w = mkBf16(P * E);
    const g_w = mkBf16(P * E);
    const b_w = mkBf16(heads * E);
    const fa_w = mkBf16(head_dim * E);
    const fb_w = mkBf16(P * head_dim);
    const o_w = mkBf16(E * P);
    const q_conv_w = mkF32(P * conv_k, 0.1);
    const k_conv_w = mkF32(P * conv_k, 0.1);
    const v_conv_w = mkF32(P * conv_k, 0.1);
    const A_log_w = mkF32(heads, 0.0);
    const dt_bias_w = mkF32(P, 0.0);
    const o_norm_w = mkF32(head_dim, 1.0);

    var cfg = kt.kt_kimi_kda_layer_config_t{
        .hidden = hidden,
        .kda_heads = heads,
        .kda_head_dim = head_dim,
        .conv_k = conv_k,
        .gate_lb = -5.0,
        .rms_eps = 1e-5,
        .q = q_w.ptr,
        .k = k_w.ptr,
        .v = v_w.ptr,
        .g = g_w.ptr,
        .b = b_w.ptr,
        .f_a = fa_w.ptr,
        .f_b = fb_w.ptr,
        .o = o_w.ptr,
        .q_conv = &q_conv_w[0],
        .k_conv = &k_conv_w[0],
        .v_conv = &v_conv_w[0],
        .A_log = &A_log_w[0],
        .dt_bias = &dt_bias_w[0],
        .o_norm = &o_norm_w[0],
    };

    const layer = kt.kt_kimi_kda_layer_new(&cfg);
    defer kt.kt_kimi_kda_layer_free(layer);

    const T: usize = 2;
    const x = try allocator.alloc(f32, T * E);
    defer allocator.free(x);
    for (x) |*v| v.* = 0.5;
    const out = try allocator.alloc(f32, T * E);
    defer allocator.free(out);

    kt.kt_kimi_kda_layer_forward(layer, T, x.ptr, out.ptr);

    for (out) |v| try testing.expect(std.math.isFinite(v));
}