// BF16 KDA layer: Kimi Delta Attention using ktransformers' BF16 GEMM kernels.
//
// Adaptation of kdaLayerFull (kimi_k3_model.zig) to BF16 weights, following the
// deepseekv3 pattern: linear projections go through amx.gemm_bf16.gemmExpert
// (BF16 in, F32 out), while conv kernels, A_log, dt_bias and o_norm stay F32
// (elementwise / tiny, per k3-in-c).

const std = @import("std");
const amx = @import("../arch/amx.zig");
const gemm_bf16 = @import("../amx/gemm_224_bf16.zig");
const cfgmod = @import("../../kimi/kimi_k3_config.zig");
const ops = @import("../../kimi/kimi_k3_ops.zig");

pub const KdaBf16Weights = struct {
    q: [*]const amx.bf16, // [P][E]
    k: [*]const amx.bf16,
    v: [*]const amx.bf16,
    g: [*]const amx.bf16, // [P][E]
    b: [*]const amx.bf16, // [H][E]
    f_a: [*]const amx.bf16, // [D][E]
    f_b: [*]const amx.bf16, // [P][D]
    o: [*]const amx.bf16, // [E][P]
    // elementwise (fp32)
    q_conv: [*]const f32, // [P][conv_k]
    k_conv: [*]const f32,
    v_conv: [*]const f32,
    A_log: [*]const f32, // [H]
    dt_bias: [*]const f32, // [P]
    o_norm: [*]const f32, // [D]
};

pub const KdaLayerState = struct {
    S: []f32, // [H*D*D]
    conv: []f32, // [3*P*hist]
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

pub fn kdaLayerFullBf16(
    out: []f32,
    x: []const f32,
    w: *const KdaBf16Weights,
    c: *const cfgmod.KimiK3Config,
    T: usize,
) void {
    const E = c.hidden;
    const H = c.kda_heads;
    const D = c.kda_head_dim;
    const P = H * D;
    const K = c.conv_k;
    const hist = K - 1;

    // bf16 bridge for input
    const xbf = std.heap.page_allocator.alloc(amx.bf16, T * E) catch return;
    defer std.heap.page_allocator.free(xbf);
    for (0..T * E) |i| xbf[i] = amx.f32_to_bf16(x[i]);

    var q = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(q);
    var kbuf = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(kbuf);
    var vbuf = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(vbuf);
    var z = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(z);
    var bt = std.heap.page_allocator.alloc(f32, T * H) catch return;
    defer std.heap.page_allocator.free(bt);
    const o = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(o);
    const gb = std.heap.page_allocator.alloc(f32, P) catch return;
    defer std.heap.page_allocator.free(gb);
    const alpha = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(alpha);

    for (0..T) |t| {
        const xt_bf = xbf[t * E ..].ptr;
        matmulBf16(q[t * P ..][0..P], xt_bf, w.q, E, P);
        matmulBf16(kbuf[t * P ..][0..P], xt_bf, w.k, E, P);
        matmulBf16(vbuf[t * P ..][0..P], xt_bf, w.v, E, P);
        matmulBf16(bt[t * H ..][0..H], xt_bf, w.b, E, H);
        const fa = std.heap.page_allocator.alloc(f32, D) catch return;
        defer std.heap.page_allocator.free(fa);
        matmulBf16(fa, xt_bf, w.f_a, E, D);
        const fa_bf = std.heap.page_allocator.alloc(amx.bf16, D) catch return;
        defer std.heap.page_allocator.free(fa_bf);
        for (0..D) |i| fa_bf[i] = amx.f32_to_bf16(fa[i]);
        matmulBf16(z[t * P ..][0..P], fa_bf.ptr, w.f_b, D, P);
    }

    // ShortConv (manual, history zero) — reuse f32 conv
    inline for (.{ 0, 1, 2 }) |which| {
        const src = if (which == 0) q else if (which == 1) kbuf else vbuf;
        const convw = if (which == 0) w.q_conv else if (which == 1) w.k_conv else w.v_conv;
        for (0..P) |ch| {
            var buf: [16]f32 = [_]f32{0} ** 16;
            for (0..T) |t| {
                const cur = src[t * P + ch];
                var acc: f32 = convw[ch * K + hist] * cur;
                for (0..hist) |j| acc += convw[ch * K + j] * buf[j];
                var j: usize = 0;
                while (j + 1 < hist) : (j += 1) buf[j] = buf[j + 1];
                if (hist > 0) buf[hist - 1] = cur;
                src[t * P + ch] = acc * sigmoidF32(acc);
            }
        }
    }

    // L2Norm q,k
    for (0..T) |t| {
        for (0..H) |h| {
            var ss: f64 = 0.0;
            for (0..D) |d| {
                const val = q[t * P + h * D + d];
                ss += @as(f64, val) * @as(f64, val);
            }
            const inv: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(D)))) + 1e-6);
            for (0..D) |d| q[t * P + h * D + d] *= inv;
            ss = 0.0;
            for (0..D) |d| {
                const val = kbuf[t * P + h * D + d];
                ss += @as(f64, val) * @as(f64, val);
            }
            const invk: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(D)))) + 1e-6);
            for (0..D) |d| kbuf[t * P + h * D + d] *= invk;
        }
    }
    for (0..T) |t| {
        for (0..H) |h| bt[t * H + h] = sigmoidF32(bt[t * H + h]);
    }
    // decay (A_log per head)
    for (0..T) |t| {
        const a_log = w.A_log;
        const dt_bias = w.dt_bias;
        const lb = c.gate_lb;
        for (0..H) |h| {
            const a = @exp(a_log[h]);
            for (0..D) |d| {
                const i = h * D + d;
                const zz = z[t * P + i];
                const u = a * (zz + dt_bias[i]);
                const gi = lb * (1.0 / (1.0 + @exp(-u)));
                alpha[t * P + i] = @exp(gi);
            }
        }
    }

    const qscale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var S = std.heap.page_allocator.alloc(f32, H * D * D) catch return;
    defer std.heap.page_allocator.free(S);
    for (S) |*s| s.* = 0.0;
    for (0..H) |h| {
        for (0..T) |t| {
            const off = t * P + h * D;
            var qh = std.heap.page_allocator.alloc(f32, D) catch return;
            defer std.heap.page_allocator.free(qh);
            for (0..D) |d| qh[d] = q[off + d] * qscale;
            // kdaStep (inline minimal)
            const Sk = S[h * D * D ..][0 .. D * D];
            // decay
            for (0..D) |i| {
                const a = alpha[off + i];
                for (0..D) |j| Sk[i * D + j] *= a;
            }
            // read u
            var u: [256]f32 = [_]f32{0} ** 256;
            for (0..D) |j| u[j] = 0.0;
            for (0..D) |i| {
                const ki = kbuf[off + i];
                if (ki == 0.0) continue;
                for (0..D) |j| u[j] += ki * Sk[i * D + j];
            }
            // write
            for (0..D) |i| {
                const ki = kbuf[off + i];
                if (ki == 0.0) continue;
                for (0..D) |j| {
                    Sk[i * D + j] += ki * bt[t * H + h] * (vbuf[off + j] - u[j]);
                }
            }
            // output
            for (0..D) |j| o[off + j] = 0.0;
            for (0..D) |i| {
                const qi = qh[i];
                if (qi == 0.0) continue;
                for (0..D) |j| o[off + j] += qi * Sk[i * D + j];
            }
        }
    }

    // norm + gate + o_proj
    for (0..T) |t| {
        const ot = o[t * P ..][0..P];
        for (0..H) |h| rmsnormF32(ot[h * D ..][0..D], ot[h * D ..][0..D], w.o_norm[0..D], D, c.rms_eps);
        const xt_bf = xbf[t * E ..].ptr;
        matmulBf16(gb, xt_bf, w.g, E, P);
        for (0..P) |i| ot[i] *= sigmoidF32(gb[i]);
        const ot_bf = std.heap.page_allocator.alloc(amx.bf16, P) catch return;
        defer std.heap.page_allocator.free(ot_bf);
        for (0..P) |i| ot_bf[i] = amx.f32_to_bf16(ot[i]);
        matmulBf16(out[t * E ..][0..E], ot_bf.ptr, w.o, P, E);
    }
}

pub fn kdaLayerIncBf16(
    out: []f32,
    x: []const f32,
    w: *const KdaBf16Weights,
    c: *const cfgmod.KimiK3Config,
    st: *KdaLayerState,
) void {
    const E = c.hidden;
    const H = c.kda_heads;
    const D = c.kda_head_dim;
    const P = H * D;
    const K = c.conv_k;
    const hist = K - 1;

    var xb = std.heap.page_allocator.alloc(amx.bf16, E) catch return;
    defer std.heap.page_allocator.free(xb);
    for (0..E) |i| xb[i] = amx.f32_to_bf16(x[i]);

    var q = std.heap.page_allocator.alloc(f32, P) catch return;
    defer std.heap.page_allocator.free(q);
    var k = std.heap.page_allocator.alloc(f32, P) catch return;
    defer std.heap.page_allocator.free(k);
    var v = std.heap.page_allocator.alloc(f32, P) catch return;
    defer std.heap.page_allocator.free(v);
    const z = std.heap.page_allocator.alloc(f32, P) catch return;
    defer std.heap.page_allocator.free(z);
    var bt = std.heap.page_allocator.alloc(f32, H) catch return;
    defer std.heap.page_allocator.free(bt);
    var o = std.heap.page_allocator.alloc(f32, P) catch return;
    defer std.heap.page_allocator.free(o);
    const gb = std.heap.page_allocator.alloc(f32, P) catch return;
    defer std.heap.page_allocator.free(gb);
    var alpha = std.heap.page_allocator.alloc(f32, P) catch return;
    defer std.heap.page_allocator.free(alpha);

    matmulBf16(q, xb.ptr, w.q, E, P);
    matmulBf16(k, xb.ptr, w.k, E, P);
    matmulBf16(v, xb.ptr, w.v, E, P);
    matmulBf16(bt, xb.ptr, w.b, E, H);
    const fa = std.heap.page_allocator.alloc(f32, D) catch return;
    defer std.heap.page_allocator.free(fa);
    matmulBf16(fa, xb.ptr, w.f_a, E, D);
    const fa_bf = std.heap.page_allocator.alloc(amx.bf16, D) catch return;
    defer std.heap.page_allocator.free(fa_bf);
    for (0..D) |i| fa_bf[i] = amx.f32_to_bf16(fa[i]);
    matmulBf16(z, fa_bf.ptr, w.f_b, D, P);

    // ShortConv carried state
    const srcs = [_][]f32{ q, k, v };
    const convws = [_][]const f32{ w.q_conv[0..P*K], w.k_conv[0..P*K], w.v_conv[0..P*K] };
    for (0..3) |wi| {
        const src = srcs[wi];
        const convw = convws[wi];
        for (0..P) |ch| {
            var buf: [16]f32 = [_]f32{0} ** 16;
            const off = wi * P * hist + ch * hist;
            if (hist > 0) {
                for (0..hist) |hh| buf[hh] = st.conv[off + hh];
            }
            const cur = src[ch];
            var acc: f32 = convw[ch * K + hist] * cur;
            for (0..hist) |j| acc += convw[ch * K + j] * buf[j];
            if (hist > 0) {
                var j: usize = 0;
                while (j + 1 < hist) : (j += 1) buf[j] = buf[j + 1];
                buf[hist - 1] = cur;
                for (0..hist) |hh| st.conv[off + hh] = buf[hh];
            }
            src[ch] = acc * sigmoidF32(acc);
        }
    }

    // L2Norm q,k
    for (0..H) |h| {
        var ss: f64 = 0.0;
        for (0..D) |d| {
            const val = q[h * D + d];
            ss += @as(f64, val) * @as(f64, val);
        }
        const inv: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(D)))) + 1e-6);
        for (0..D) |d| q[h * D + d] *= inv;
        ss = 0.0;
        for (0..D) |d| {
            const val = k[h * D + d];
            ss += @as(f64, val) * @as(f64, val);
        }
        const invk: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(D)))) + 1e-6);
        for (0..D) |d| k[h * D + d] *= invk;
    }

    for (0..H) |h| bt[h] = sigmoidF32(bt[h]);

    {
        const gtmp = std.heap.page_allocator.alloc(f32, P) catch return;
        defer std.heap.page_allocator.free(gtmp);
        ops.kdaDecay(gtmp, alpha, z, w.A_log[0..H], w.dt_bias[0..P], H, D, c.gate_lb);
    }

    const qscale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    for (0..H) |h| {
        var qh = std.heap.page_allocator.alloc(f32, D) catch return;
        defer std.heap.page_allocator.free(qh);
        for (0..D) |d| qh[d] = q[h * D + d] * qscale;
        ops.kdaStep(
            st.S[h * D * D ..][0 .. D * D],
            o[h * D ..][0..D],
            qh,
            k[h * D ..][0..D],
            v[h * D ..][0..D],
            alpha[h * D ..][0..D],
            bt[h],
            D,
            D,
        );
    }

    for (0..H) |h| rmsnormF32(o[h * D ..][0..D], o[h * D ..][0..D], w.o_norm[0..D], D, c.rms_eps);
    matmulBf16(gb, xb.ptr, w.g, E, P);
    for (0..P) |i| o[i] *= sigmoidF32(gb[i]);
    const obf = std.heap.page_allocator.alloc(amx.bf16, P) catch return;
    defer std.heap.page_allocator.free(obf);
    for (0..P) |i| obf[i] = amx.f32_to_bf16(o[i]);
    matmulBf16(out, obf.ptr, w.o, P, E);
}
