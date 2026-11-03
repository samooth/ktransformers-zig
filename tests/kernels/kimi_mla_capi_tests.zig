const std = @import("std");
const testing = std.testing;
const kt = @import("kt");

test "kt_kimi_mla_layer C API: new -> forward -> free" {
    const allocator = std.heap.page_allocator;

    const hidden: usize = 4;
    const H: usize = 2;
    const q_lora: usize = 4;
    const kv_lora: usize = 2;
    const qn: usize = 2;
    const qr: usize = 2;
    const vh: usize = 2;
    const qh: usize = qn + qr;
    const kvw: usize = kv_lora + qr;

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

    const q_a_w = mkBf16(q_lora * hidden);
    const q_b_w = mkBf16(H * qh * q_lora);
    const kv_a_w = mkBf16(kvw * hidden);
    const kv_b_w = mkBf16(H * (qn + vh) * kv_lora);
    const o_w = mkBf16(hidden * H * vh);
    const g_w = mkBf16(H * vh * hidden);
    const q_a_norm_w = mkF32(q_lora, 1.0);
    const kv_a_norm_w = mkF32(kv_lora, 1.0);

    var cfg = kt.kt_kimi_mla_layer_config_t{
        .hidden = hidden,
        .n_heads = H,
        .q_lora = q_lora,
        .kv_lora = kv_lora,
        .qk_nope = qn,
        .qk_rope = qr,
        .v_head = vh,
        .rms_eps = 1e-5,
        .q_a = q_a_w.ptr,
        .q_b = q_b_w.ptr,
        .kv_a = kv_a_w.ptr,
        .kv_b = kv_b_w.ptr,
        .o = o_w.ptr,
        .g = g_w.ptr,
        .q_a_norm = &q_a_norm_w[0],
        .kv_a_norm = &kv_a_norm_w[0],
    };

    const layer = kt.kt_kimi_mla_layer_new(&cfg);
    defer kt.kt_kimi_mla_layer_free(layer);

    const T: usize = 2;
    const x = try allocator.alloc(f32, T * hidden);
    defer allocator.free(x);
    for (x) |*v| v.* = 0.5;
    const out = try allocator.alloc(f32, T * hidden);
    defer allocator.free(out);

    kt.kt_kimi_mla_layer_forward(layer, T, x.ptr, out.ptr);

    for (out) |v| try testing.expect(std.math.isFinite(v));
}