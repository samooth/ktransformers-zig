// BF16 Gated MLA layer (NoPE) using ktransformers' BF16 GEMM kernels.

const std = @import("std");
const amx = @import("../arch/amx.zig");
const gemm_bf16 = @import("../amx/gemm_224_bf16.zig");
const cfgmod = @import("../../kimi/kimi_k3_config.zig");

pub const MlaBf16Weights = struct {
    q_a: [*]const amx.bf16, // [q_lora][E]
    q_a_norm: [*]const f32, // [q_lora]
    q_b: [*]const amx.bf16, // [H*qh][q_lora]
    kv_a: [*]const amx.bf16, // [kv_lora+qr][E]
    kv_a_norm: [*]const f32, // [kv_lora]
    kv_b: [*]const amx.bf16, // [H*kvd][kv_lora]
    o: [*]const amx.bf16, // [E][H*vh]
    g: ?[*]const amx.bf16, // [H*vh][E] or null
};

pub const MlaLayerState = struct {
    kv: []f32, // [cap*H*kvd]
    rope: []f32, // [cap*qr]
    cached: usize,
    cap: usize,
};

fn matmulBf16(y: []f32, x_bf16: [*]const amx.bf16, W: [*]const amx.bf16, in: usize, out: usize) void {
    gemm_bf16.gemmExpert(x_bf16, W, y.ptr, 1, out, in, in, in, out);
}

fn sigmoidF32(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn rmsnormF32(y: []f32, x: []const f32, w: []const f32, n: usize, eps: f32) void {
    var ss: f64 = 0.0;
    for (0..n) |i| ss += @as(f64, x[i]) * @as(f64, x[i]);
    const inv: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(n)))) + eps);
    for (0..n) |i| y[i] = w[i] * x[i] * inv;
}

pub fn mlaLayerFullBf16(
    out: []f32,
    x: []const f32,
    w: *const MlaBf16Weights,
    c: *const cfgmod.KimiK3Config,
    T: usize,
) void {
    const E = c.hidden;
    const H = c.n_heads;
    const qn = c.qk_nope;
    const qr = c.qk_rope;
    const vh = c.v_head;
    const qh = qn + qr;
    const kv_lora = c.kv_lora;
    const kvw = kv_lora + qr;
    const kvd = qn + vh;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(qh)));

    // bf16 bridge for input (per token handled inside loop via slices)
    const xbf = std.heap.page_allocator.alloc(amx.bf16, T * E) catch return;
    defer std.heap.page_allocator.free(xbf);
    for (0..T * E) |i| xbf[i] = amx.f32_to_bf16(x[i]);

    const q = std.heap.page_allocator.alloc(f32, T * H * qh) catch return;
    defer std.heap.page_allocator.free(q);
    const kv = std.heap.page_allocator.alloc(f32, T * H * kvd) catch return;
    defer std.heap.page_allocator.free(kv);
    const rp = std.heap.page_allocator.alloc(f32, T * qr) catch return;
    defer std.heap.page_allocator.free(rp);
    const ql = std.heap.page_allocator.alloc(f32, c.q_lora) catch return;
    defer std.heap.page_allocator.free(ql);
    const ct = std.heap.page_allocator.alloc(f32, kvw) catch return;
    defer std.heap.page_allocator.free(ct);
    const acc = std.heap.page_allocator.alloc(f32, H * vh) catch return;
    defer std.heap.page_allocator.free(acc);
    const gbuf = std.heap.page_allocator.alloc(f32, H * vh) catch return;
    defer std.heap.page_allocator.free(gbuf);
    const sc = std.heap.page_allocator.alloc(f32, T) catch return;
    defer std.heap.page_allocator.free(sc);

    for (0..T) |t| {
        const xt_bf = xbf[t * E ..].ptr;
        // q
        matmulBf16(ql, xt_bf, w.q_a, E, c.q_lora);
        rmsnormF32(ql, ql, w.q_a_norm[0..c.q_lora], c.q_lora, c.rms_eps);
        const ql_bf = std.heap.page_allocator.alloc(amx.bf16, c.q_lora) catch return;
        defer std.heap.page_allocator.free(ql_bf);
        for (0..c.q_lora) |i| ql_bf[i] = amx.f32_to_bf16(ql[i]);
        matmulBf16(q[t * H * qh ..][0 .. H * qh], ql_bf.ptr, w.q_b, c.q_lora, H * qh);

        // kv_a
        matmulBf16(ct, xt_bf, w.kv_a, E, kvw);
        rmsnormF32(ct[0..kv_lora], ct[0..kv_lora], w.kv_a_norm[0..kv_lora], kv_lora, c.rms_eps);
        for (0..qr) |i| rp[t * qr + i] = ct[kv_lora + i];
        const ct_bf = std.heap.page_allocator.alloc(amx.bf16, kv_lora) catch return;
        defer std.heap.page_allocator.free(ct_bf);
        for (0..kv_lora) |i| ct_bf[i] = amx.f32_to_bf16(ct[i]);
        matmulBf16(kv[t * H * kvd ..][0 .. H * kvd], ct_bf.ptr, w.kv_b, kv_lora, H * kvd);
    }

    for (0..T) |t| {
        for (0..H) |h| {
            const qt = q[t * H * qh + h * qh ..][0..qh];
            var m: f32 = -std.math.inf(f32);
            for (0..(t + 1)) |s| {
                const ks = kv[s * H * kvd + h * kvd ..][0..kvd];
                const kr = rp[s * qr ..][0..qr];
                var dd: f64 = 0.0;
                for (0..qn) |i| dd += @as(f64, qt[i]) * @as(f64, ks[i]);
                for (0..qr) |i| dd += @as(f64, qt[qn + i]) * @as(f64, kr[i]);
                sc[s] = @as(f32, @floatCast(dd)) * scale;
                if (sc[s] > m) m = sc[s];
            }
            var zv: f64 = 0.0;
            for (0..(t + 1)) |s| {
                sc[s] = @exp(sc[s] - m);
                zv += @as(f64, sc[s]);
            }
            const accp = acc[h * vh ..][0..vh];
            for (0..vh) |j| accp[j] = 0.0;
            for (0..(t + 1)) |s| {
                const pr = sc[s] / @as(f32, @floatCast(zv));
                const vs = kv[s * H * kvd + h * kvd + qn ..][0..vh];
                for (0..vh) |j| accp[j] += pr * vs[j];
            }
        }
        if (w.g) |gw| {
            const xt_bf = xbf[t * E ..].ptr;
            matmulBf16(gbuf, xt_bf, gw, E, H * vh);
            for (0..H * vh) |i| acc[i] *= sigmoidF32(gbuf[i]);
        }
        const acc_bf = std.heap.page_allocator.alloc(amx.bf16, H * vh) catch return;
        defer std.heap.page_allocator.free(acc_bf);
        for (0..H * vh) |i| acc_bf[i] = amx.f32_to_bf16(acc[i]);
        matmulBf16(out[t * E ..][0..E], acc_bf.ptr, w.o, H * vh, E);
    }
}

pub fn mlaLayerIncBf16(
    out: []f32,
    x: []const f32,
    w: *const MlaBf16Weights,
    c: *const cfgmod.KimiK3Config,
    st: *MlaLayerState,
) void {
    const E = c.hidden;
    const H = c.n_heads;
    const qn = c.qk_nope;
    const qr = c.qk_rope;
    const vh = c.v_head;
    const qh = qn + qr;
    const kv_lora = c.kv_lora;
    const kvw = kv_lora + qr;
    const kvd = qn + vh;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(qh)));

    var xb = std.heap.page_allocator.alloc(amx.bf16, E) catch return;
    defer std.heap.page_allocator.free(xb);
    for (0..E) |i| xb[i] = amx.f32_to_bf16(x[i]);

    const ql = std.heap.page_allocator.alloc(f32, c.q_lora) catch return;
    defer std.heap.page_allocator.free(ql);
    var ct = std.heap.page_allocator.alloc(f32, kvw) catch return;
    defer std.heap.page_allocator.free(ct);

    var q = std.heap.page_allocator.alloc(f32, H * qh) catch return;
    defer std.heap.page_allocator.free(q);

    matmulBf16(ql, xb.ptr, w.q_a, E, c.q_lora);
    rmsnormF32(ql, ql, w.q_a_norm[0..c.q_lora], c.q_lora, c.rms_eps);
    const ql_bf = std.heap.page_allocator.alloc(amx.bf16, c.q_lora) catch return;
    defer std.heap.page_allocator.free(ql_bf);
    for (0..c.q_lora) |i| ql_bf[i] = amx.f32_to_bf16(ql[i]);
    matmulBf16(q, ql_bf.ptr, w.q_b, c.q_lora, H * qh);

    matmulBf16(ct, xb.ptr, w.kv_a, E, kvw);
    rmsnormF32(ct[0..kv_lora], ct[0..kv_lora], w.kv_a_norm[0..kv_lora], kv_lora, c.rms_eps);

    const p = st.cached;
    for (0..qr) |i| st.rope[p * qr + i] = ct[kv_lora + i];

    const ct_bf = std.heap.page_allocator.alloc(amx.bf16, kv_lora) catch return;
    defer std.heap.page_allocator.free(ct_bf);
    for (0..kv_lora) |i| ct_bf[i] = amx.f32_to_bf16(ct[i]);
    matmulBf16(st.kv[p * H * kvd ..][0 .. H * kvd], ct_bf.ptr, w.kv_b, kv_lora, H * kvd);

    var acc = std.heap.page_allocator.alloc(f32, H * vh) catch return;
    defer std.heap.page_allocator.free(acc);
    var sc = std.heap.page_allocator.alloc(f32, p + 1) catch return;
    defer std.heap.page_allocator.free(sc);

    for (0..H) |h| {
        const qt = q[h * qh ..][0..qh];
        var m: f32 = -std.math.inf(f32);
        for (0..(p + 1)) |s| {
            const ks = st.kv[s * H * kvd + h * kvd ..][0..kvd];
            const kr = st.rope[s * qr ..][0..qr];
            var dd: f64 = 0.0;
            for (0..qn) |i| dd += @as(f64, qt[i]) * @as(f64, ks[i]);
            for (0..qr) |i| dd += @as(f64, qt[qn + i]) * @as(f64, kr[i]);
            sc[s] = @as(f32, @floatCast(dd)) * scale;
            if (sc[s] > m) m = sc[s];
        }
        var zv: f64 = 0.0;
        for (0..(p + 1)) |s| {
            sc[s] = @exp(sc[s] - m);
            zv += @as(f64, sc[s]);
        }
        const accp = acc[h * vh ..][0..vh];
        for (0..vh) |j| accp[j] = 0.0;
        for (0..(p + 1)) |s| {
            const pr = sc[s] / @as(f32, @floatCast(zv));
            const vs = st.kv[s * H * kvd + h * kvd + qn ..][0..vh];
            for (0..vh) |j| accp[j] += pr * vs[j];
        }
    }

    const gbuf = std.heap.page_allocator.alloc(f32, H * vh) catch return;
    defer std.heap.page_allocator.free(gbuf);
    if (w.g) |gw| {
        matmulBf16(gbuf, xb.ptr, gw, E, H * vh);
        for (0..H * vh) |i| acc[i] *= sigmoidF32(gbuf[i]);
    }
    const acc_bf = std.heap.page_allocator.alloc(amx.bf16, H * vh) catch return;
    defer std.heap.page_allocator.free(acc_bf);
    for (0..H * vh) |i| acc_bf[i] = amx.f32_to_bf16(acc[i]);
    matmulBf16(out, acc_bf.ptr, w.o, H * vh, E);

    st.cached += 1;
}
