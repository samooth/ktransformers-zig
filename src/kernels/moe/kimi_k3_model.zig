//! Kimi K3 CPU model (port of k3 layer + k3_run forward).
//!
//! Phase 2 of the port: pure CPU, full-recompute forward over T tokens, using
//! the Phase 1 ops. Incremental decode + state serialisation land in Phase 3.

const std = @import("std");
const ops = @import("../../kimi/kimi_k3_ops.zig");
const cfgmod = @import("../../kimi/kimi_k3_config.zig");
pub const KimiK3Config = cfgmod.KimiK3Config;

pub const KdaWeights = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    g: []f32,
    o: []f32,
    q_conv: []f32,
    k_conv: []f32,
    v_conv: []f32,
    f_a: []f32,
    f_b: []f32,
    b: []f32,
    A_log: []f32,
    dt_bias: []f32,
    o_norm: []f32,
};

pub const MlaWeights = struct {
    q_a: []f32,
    q_a_norm: []f32,
    q_b: []f32,
    kv_a: []f32,
    kv_a_norm: []f32,
    kv_b: []f32,
    o: []f32,
    g: []f32,
    has_gate: bool,
};

pub const LayerWeights = struct {
    in_norm: []f32,
    post_norm: []f32,
    attn_res_norm: []f32,
    attn_res_proj: []f32,
    mlp_res_norm: []f32,
    mlp_res_proj: []f32,

    is_mla: bool,
    kda: ?KdaWeights,
    mla: ?MlaWeights,

    is_dense: bool,
    dense_gate: []f32,
    dense_up: []f32,
    dense_down: []f32,
    moe: ?ops.LatentMoeWeights,

    moe_cfg: ops.LatentMoeCfg,
};

pub const ModelWeights = struct {
    embed: []f32,
    final_norm: []f32,
    lm_head: []f32,
    layers: []const LayerWeights,
};

fn matmulF32(y: []f32, x: []const f32, W: []const f32, in: usize, out: usize) void {
    for (0..out) |o| {
        var acc: f64 = 0.0;
        for (0..in) |i| acc += @as(f64, W[o * in + i]) * @as(f64, x[i]);
        y[o] = @floatCast(acc);
    }
}

fn rmsnormF32(y: []f32, x: []const f32, w: []const f32, n: usize, eps: f32) void {
    var ss: f64 = 0.0;
    for (0..n) |i| ss += @as(f64, x[i]) * @as(f64, x[i]);
    const inv: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(n)))) + eps);
    for (0..n) |i| y[i] = w[i] * x[i] * inv;
}

fn sigmoidF32(x: f32) f32 {
    return @floatCast(1.0 / (1.0 + @exp(-@as(f64, x))));
}

pub fn kdaLayerFull(out: []f32, x: []f32, w: *const KdaWeights, c: *const KimiK3Config, T: usize) void {
    const E = c.hidden;
    const H = c.kda_heads;
    const D = c.kda_head_dim;
    const P = H * D;
    const K = c.conv_k;
    const hist = K - 1;

    var q = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(q);
    var k = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(k);
    var v = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(v);
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
        const xt = x[t * E ..][0..E];
        matmulF32(q[t * P ..][0..P], xt, w.q, E, P);
        matmulF32(k[t * P ..][0..P], xt, w.k, E, P);
        matmulF32(v[t * P ..][0..P], xt, w.v, E, P);
        matmulF32(bt[t * H ..][0..H], xt, w.b, E, H);
        const fa = std.heap.page_allocator.alloc(f32, D) catch return;
        defer std.heap.page_allocator.free(fa);
        matmulF32(fa, xt, w.f_a, E, D);
        matmulF32(z[t * P ..][0..P], fa, w.f_b, D, P);
    }

    // ShortConv (manual, history zero for full recompute)
    inline for (.{ 0, 1, 2 }) |which| {
        const src = if (which == 0) q else if (which == 1) k else v;
        const convw = if (which == 0) w.q_conv else if (which == 1) w.k_conv else w.v_conv;
        for (0..P) |ch| {
            var buf: [16]f32 = [_]f32{0} ** 16;
            for (0..T) |t| {
                const cur = src[t * P + ch];
                var acc: f32 = convw[ch * K + hist] * cur;
                for (0..hist) |j| acc += convw[ch * K + j] * buf[j];
                // shift
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
                const val = k[t * P + h * D + d];
                ss += @as(f64, val) * @as(f64, val);
            }
            const invk: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(D)))) + 1e-6);
            for (0..D) |d| k[t * P + h * D + d] *= invk;
        }
    }
    // beta sigmoid
    for (0..T) |t| {
        for (0..H) |h| bt[t * H + h] = sigmoidF32(bt[t * H + h]);
    }
    // decay
    {
        const gtmp = std.heap.page_allocator.alloc(f32, P) catch return;
        defer std.heap.page_allocator.free(gtmp);
        for (0..T) |t| {
            ops.kdaDecay(gtmp, alpha[t * P ..][0..P], z[t * P ..][0..P], w.A_log, w.dt_bias, H, D, c.gate_lb);
        }
    }

    // recurrence
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
            ops.kdaStep(
                S[h * D * D ..][0 .. D * D],
                o[off..][0..D],
                qh,
                k[off..][0..D],
                v[off..][0..D],
                alpha[off..][0..D],
                bt[t * H + h],
                D,
                D,
            );
        }
    }

    // norm + gate + o_proj
    for (0..T) |t| {
        const ot = o[t * P ..][0..P];
        for (0..H) |h| rmsnormF32(ot[h * D ..][0..D], ot[h * D ..][0..D], w.o_norm, D, c.rms_eps);
        matmulF32(gb, x[t * E ..][0..E], w.g, E, P);
        for (0..P) |i| ot[i] *= sigmoidF32(gb[i]);
        matmulF32(out[t * E ..][0..E], ot, w.o, P, E);
    }
}

pub fn mlaLayerFull(out: []f32, x: []f32, w: *const MlaWeights, c: *const KimiK3Config, T: usize) void {
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

    var q = std.heap.page_allocator.alloc(f32, T * H * qh) catch return;
    defer std.heap.page_allocator.free(q);
    var ct = std.heap.page_allocator.alloc(f32, kvw) catch return;
    defer std.heap.page_allocator.free(ct);
    const ql = std.heap.page_allocator.alloc(f32, c.q_lora) catch return;
    defer std.heap.page_allocator.free(ql);
    var kv = std.heap.page_allocator.alloc(f32, T * H * kvd) catch return;
    defer std.heap.page_allocator.free(kv);
    var rp = std.heap.page_allocator.alloc(f32, T * qr) catch return;
    defer std.heap.page_allocator.free(rp);
    var acc = std.heap.page_allocator.alloc(f32, H * vh) catch return;
    defer std.heap.page_allocator.free(acc);
    const gbuf = std.heap.page_allocator.alloc(f32, H * vh) catch return;
    defer std.heap.page_allocator.free(gbuf);
    var sc = std.heap.page_allocator.alloc(f32, T) catch return;
    defer std.heap.page_allocator.free(sc);

    for (0..T) |t| {
        const xt = x[t * E ..][0..E];
        matmulF32(ql, xt, w.q_a, E, c.q_lora);
        rmsnormF32(ql, ql, w.q_a_norm, c.q_lora, c.rms_eps);
        matmulF32(q[t * H * qh ..][0 .. H * qh], ql, w.q_b, c.q_lora, H * qh);

        matmulF32(ct, xt, w.kv_a, E, kvw);
        rmsnormF32(ct[0..kv_lora], ct[0..kv_lora], w.kv_a_norm, kv_lora, c.rms_eps);
        for (0..qr) |i| rp[t * qr + i] = ct[kv_lora + i];
        matmulF32(kv[t * H * kvd ..][0 .. H * kvd], ct[0..kv_lora], w.kv_b, kv_lora, H * kvd);
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
        if (w.has_gate) {
            matmulF32(gbuf, x[t * E ..][0..E], w.g, E, H * vh);
            for (0..H * vh) |i| acc[i] *= sigmoidF32(gbuf[i]);
        }
        matmulF32(out[t * E ..][0..E], acc, w.o, H * vh, E);
    }
}

fn denseMlpSlice(out: []f32, x: []f32, w_gate: []f32, w_up: []f32, w_down: []f32, dense_inter: usize, E: usize, b1: f32, b2: f32) void {
    var dgu = std.heap.page_allocator.alloc(f32, 2 * dense_inter) catch return;
    defer std.heap.page_allocator.free(dgu);
    const act = std.heap.page_allocator.alloc(f32, dense_inter) catch return;
    defer std.heap.page_allocator.free(act);
    matmulF32(dgu[0..dense_inter], x, w_gate, E, dense_inter);
    matmulF32(dgu[dense_inter .. 2 * dense_inter], x, w_up, E, dense_inter);
    ops.situGlu(act, dgu, dense_inter, b1, b2);
    matmulF32(out, act, w_down, dense_inter, E);
}

pub fn layerForward(h: []f32, block_residual: []f32, n_blocks: *usize, w: *const LayerWeights, c: *const KimiK3Config, layer_idx: usize, T: usize) void {
    const E = c.hidden;
    const maxb = c.n_layers / c.attn_res_block + 2;

    var pref = std.heap.page_allocator.alloc(f32, T * E) catch return;
    defer std.heap.page_allocator.free(pref);
    @memcpy(pref, h);

    var have_prefix: bool = true;

    if (n_blocks.* > 0) {
        var src = std.heap.page_allocator.alloc(f32, (maxb + 1) * E) catch return;
        defer std.heap.page_allocator.free(src);
        var foldA = std.heap.page_allocator.alloc(f32, E) catch return;
        defer std.heap.page_allocator.free(foldA);
        for (0..E) |i| foldA[i] = w.attn_res_norm[i] * w.attn_res_proj[i];
        for (0..T) |t| {
            for (0..n_blocks.*) |b| {
                @memcpy(src[b * E ..][0..E], block_residual[t * maxb * E + b * E ..][0..E]);
            }
            @memcpy(src[n_blocks.* * E ..][0..E], pref[t * E ..][0..E]);
            ops.attnRes(h[t * E ..][0..E], src[0 .. (n_blocks.* + 1) * E], foldA, n_blocks.* + 1, E, c.rms_eps);
        }
    }

    if (layer_idx % c.attn_res_block == 0) {
        for (0..T) |t| {
            @memcpy(block_residual[t * maxb * E + n_blocks.* * E ..][0..E], pref[t * E ..][0..E]);
        }
        n_blocks.* += 1;
        have_prefix = false;
    }

    var hin = std.heap.page_allocator.alloc(f32, T * E) catch return;
    defer std.heap.page_allocator.free(hin);
    var tmp = std.heap.page_allocator.alloc(f32, T * E) catch return;
    defer std.heap.page_allocator.free(tmp);
    for (0..T) |t| rmsnormF32(hin[t * E ..][0..E], h[t * E ..][0..E], w.in_norm, E, c.rms_eps);

    if (w.is_mla) {
        if (w.mla) |mla| {
            var mla_copy = mla;
            mlaLayerFull(tmp, hin, &mla_copy, c, T);
        }
    } else {
        if (w.kda) |kda| {
            var kda_copy = kda;
            kdaLayerFull(tmp, hin, &kda_copy, c, T);
        }
    }

    if (have_prefix) {
        for (0..T * E) |i| pref[i] += tmp[i];
    } else {
        @memcpy(pref, tmp);
        have_prefix = true;
    }

    {
        var src = std.heap.page_allocator.alloc(f32, (maxb + 1) * E) catch return;
        defer std.heap.page_allocator.free(src);
        var foldM = std.heap.page_allocator.alloc(f32, E) catch return;
        defer std.heap.page_allocator.free(foldM);
        for (0..E) |i| foldM[i] = w.mlp_res_norm[i] * w.mlp_res_proj[i];
        for (0..T) |t| {
            for (0..n_blocks.*) |b| {
                @memcpy(src[b * E ..][0..E], block_residual[t * maxb * E + b * E ..][0..E]);
            }
            @memcpy(src[n_blocks.* * E ..][0..E], pref[t * E ..][0..E]);
            ops.attnRes(h[t * E ..][0..E], src[0 .. (n_blocks.* + 1) * E], foldM, n_blocks.* + 1, E, c.rms_eps);
        }
    }

    for (0..T) |t| rmsnormF32(hin[t * E ..][0..E], h[t * E ..][0..E], w.post_norm, E, c.rms_eps);

    if (w.is_dense) {
        for (0..T) |t| {
            denseMlpSlice(tmp[t * E ..][0..E], hin[t * E ..][0..E], w.dense_gate, w.dense_up, w.dense_down, c.dense_inter, E, c.situ_b1, c.situ_b2);
        }
    } else {
        if (w.moe) |moe| {
            var moe_copy = moe;
            for (0..T) |t| {
                ops.latentMoe(tmp[t * E ..][0..E], hin[t * E ..][0..E], &moe_copy, &w.moe_cfg);
            }
        }
    }

    for (0..T * E) |i| pref[i] += tmp[i];
    @memcpy(h, pref);
}

pub fn forward(out: []f32, x_tokens: []const u32, w: *const ModelWeights, c: *const KimiK3Config, T: usize) void {
    const E = c.hidden;
    const maxb = c.n_layers / c.attn_res_block + 2;

    for (0..T) |t| {
        const id = x_tokens[t];
        for (0..E) |i| out[t * E + i] = w.embed[id * E + i];
    }

    const block_residual = std.heap.page_allocator.alloc(f32, T * maxb * E) catch return;
    defer std.heap.page_allocator.free(block_residual);
    for (block_residual) |*b| b.* = 0.0;

    var nb: usize = 0;
    for (0..c.n_layers) |li| {
        const lw = &w.layers[li];
        layerForward(out, block_residual, &nb, lw, c, li, T);
    }

    for (0..T) |t| {
        rmsnormF32(out[t * E ..][0..E], out[t * E ..][0..E], w.final_norm, E, c.rms_eps);
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Incremental decode (single token, carried state)
// ---------------------------------------------------------------------------

pub const KdaLayerState = struct {
    S: []f32, // [H*D*D]
    conv: []f32, // [3*P*hist]  (q,k,v history)
};

pub const MlaLayerState = struct {
    kv: []f32, // [cap*H*kvd]
    rope: []f32, // [cap*qr]
    cached: usize,
    cap: usize,
};

pub const LayerState = struct {
    is_mla: bool,
    kda: ?KdaLayerState,
    mla: ?MlaLayerState,
};

pub const ModelState = struct {
    layers: []LayerState,
    n_blocks: usize,
    block_residual: []f32, // [maxb*E] for single token
    maxb: usize,
    E: usize,
};

pub fn initModelState(allocator: std.mem.Allocator, c: *const KimiK3Config, max_seq: usize) ?ModelState {
    const E = c.hidden;
    const maxb = c.n_layers / c.attn_res_block + 2;
    var ms = ModelState{
        .layers = allocator.alloc(LayerState, c.n_layers) catch return null,
        .n_blocks = 0,
        .block_residual = allocator.alloc(f32, maxb * E) catch return null,
        .maxb = maxb,
        .E = E,
    };
    for (ms.block_residual) |*b| b.* = 0.0;
    for (0..c.n_layers) |li| {
        const lw_is_mla = (li + 1 <= c.n_full_attn and blk: {
            // check if li is in full_attn list
            var found = false;
            for (c.full_attn[0..c.n_full_attn]) |v| {
                if (v == @as(i32, @intCast(li + 1))) {
                    found = true;
                    break;
                }
            }
            break :blk found;
        });
        _ = lw_is_mla;
        const is_mla_layer = blk2: {
            var found = false;
            for (c.full_attn[0..c.n_full_attn]) |v| {
                if (v == @as(i32, @intCast(li + 1))) {
                    found = true;
                    break;
                }
            }
            break :blk2 found;
        };
        if (is_mla_layer) {
            const H = c.n_heads;
            const qn = c.qk_nope;
            const vh = c.v_head;
            const kvd = qn + vh;
            const qr = c.qk_rope;
            ms.layers[li] = LayerState{
                .is_mla = true,
                .kda = null,
                .mla = MlaLayerState{
                    .kv = allocator.alloc(f32, max_seq * H * kvd) catch return null,
                    .rope = allocator.alloc(f32, max_seq * qr) catch return null,
                    .cached = 0,
                    .cap = max_seq,
                },
            };
            for (ms.layers[li].mla.?.kv) |*v| v.* = 0.0;
            for (ms.layers[li].mla.?.rope) |*v| v.* = 0.0;
        } else {
            const H = c.kda_heads;
            const D = c.kda_head_dim;
            const P = H * D;
            const K = c.conv_k;
            const hist = K - 1;
            const conv_size = if (hist > 0) 3 * P * hist else 3 * P;
            ms.layers[li] = LayerState{
                .is_mla = false,
                .kda = KdaLayerState{
                    .S = allocator.alloc(f32, H * D * D) catch return null,
                    .conv = allocator.alloc(f32, conv_size) catch return null,
                },
                .mla = null,
            };
            for (ms.layers[li].kda.?.S) |*v| v.* = 0.0;
            for (ms.layers[li].kda.?.conv) |*v| v.* = 0.0;
        }
    }
    return ms;
}

fn kdaLayerInc(
    out: []f32, // [E]
    x: []const f32, // [E]
    w: *const KdaWeights,
    c: *const KimiK3Config,
    st: *KdaLayerState,
) void {
    const E = c.hidden;
    const H = c.kda_heads;
    const D = c.kda_head_dim;
    const P = H * D;
    const K = c.conv_k;
    const hist = K - 1;

    // projections
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

    matmulF32(q, x, w.q, E, P);
    matmulF32(k, x, w.k, E, P);
    matmulF32(v, x, w.v, E, P);
    matmulF32(bt, x, w.b, E, H);
    const fa = std.heap.page_allocator.alloc(f32, D) catch return;
    defer std.heap.page_allocator.free(fa);
    matmulF32(fa, x, w.f_a, E, D);
    matmulF32(z, fa, w.f_b, D, P);

    // ShortConv per channel with carried state
    const srcs = [_][]f32{ q, k, v };
    const convws = [_][]const f32{ w.q_conv, w.k_conv, w.v_conv };
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
            // shift history
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

    // beta
    for (0..H) |h| bt[h] = sigmoidF32(bt[h]);

    // decay
    {
        const gtmp = std.heap.page_allocator.alloc(f32, P) catch return;
        defer std.heap.page_allocator.free(gtmp);
        ops.kdaDecay(gtmp, alpha, z, w.A_log, w.dt_bias, H, D, c.gate_lb);
    }

    // recurrence (single step) using carried S
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

    // norm + gate + o_proj
    for (0..H) |h| rmsnormF32(o[h * D ..][0..D], o[h * D ..][0..D], w.o_norm, D, c.rms_eps);
    matmulF32(gb, x, w.g, E, P);
    for (0..P) |i| o[i] *= sigmoidF32(gb[i]);
    matmulF32(out, o, w.o, P, E);
}

pub fn mlaLayerInc(
    out: []f32,
    x: []const f32,
    w: *const MlaWeights,
    c: *const KimiK3Config,
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

    const ql = std.heap.page_allocator.alloc(f32, c.q_lora) catch return;
    defer std.heap.page_allocator.free(ql);
    var ct = std.heap.page_allocator.alloc(f32, kvw) catch return;
    defer std.heap.page_allocator.free(ct);

    // q
    var q = std.heap.page_allocator.alloc(f32, H * qh) catch return;
    defer std.heap.page_allocator.free(q);
    matmulF32(ql, x, w.q_a, E, c.q_lora);
    rmsnormF32(ql, ql, w.q_a_norm, c.q_lora, c.rms_eps);
    matmulF32(q, ql, w.q_b, c.q_lora, H * qh);

    // kv_a
    matmulF32(ct, x, w.kv_a, E, kvw);
    rmsnormF32(ct[0..kv_lora], ct[0..kv_lora], w.kv_a_norm, kv_lora, c.rms_eps);
    const p = st.cached;
    // rope slot
    for (0..qr) |i| st.rope[p * qr + i] = ct[kv_lora + i];
    // kv_b -> expanded kv at position p
    matmulF32(st.kv[p * H * kvd ..][0 .. H * kvd], ct[0..kv_lora], w.kv_b, kv_lora, H * kvd);

    // attention
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
    if (w.has_gate) {
        matmulF32(gbuf, x, w.g, E, H * vh);
        for (0..H * vh) |i| acc[i] *= sigmoidF32(gbuf[i]);
    }
    matmulF32(out, acc, w.o, H * vh, E);

    st.cached += 1;
}

pub fn layerForwardInc(
    h: []f32, // [E] in/out
    block_residual: []f32, // [maxb*E]
    n_blocks: *usize,
    w: *const LayerWeights,
    c: *const KimiK3Config,
    layer_idx: usize,
    st: *LayerState,
) void {
    const E = c.hidden;
    const maxb = c.n_layers / c.attn_res_block + 2;

    // E may be > 4096 in real model, but for tiny ok; use heap for safety
    const pref_heap = std.heap.page_allocator.alloc(f32, E) catch return;
    defer std.heap.page_allocator.free(pref_heap);
    @memcpy(pref_heap, h);

    var have_prefix: bool = true;

    // aggregation before attention
    if (n_blocks.* > 0) {
        var src = std.heap.page_allocator.alloc(f32, (maxb + 1) * E) catch return;
        defer std.heap.page_allocator.free(src);
        var foldA = std.heap.page_allocator.alloc(f32, E) catch return;
        defer std.heap.page_allocator.free(foldA);
        for (0..E) |i| foldA[i] = w.attn_res_norm[i] * w.attn_res_proj[i];
        for (0..n_blocks.*) |b| {
            @memcpy(src[b * E ..][0..E], block_residual[b * E ..][0..E]);
        }
        @memcpy(src[n_blocks.* * E ..][0..E], pref_heap);
        ops.attnRes(h, src[0 .. (n_blocks.* + 1) * E], foldA, n_blocks.* + 1, E, c.rms_eps);
    }

    // boundary
    if (layer_idx % c.attn_res_block == 0) {
        @memcpy(block_residual[n_blocks.* * E ..][0..E], pref_heap);
        n_blocks.* += 1;
        have_prefix = false;
    }

    const hin = std.heap.page_allocator.alloc(f32, E) catch return;
    defer std.heap.page_allocator.free(hin);
    const tmp = std.heap.page_allocator.alloc(f32, E) catch return;
    defer std.heap.page_allocator.free(tmp);
    rmsnormF32(hin, h, w.in_norm, E, c.rms_eps);

    if (w.is_mla) {
        if (w.mla) |mla| {
            var mla_copy = mla;
            if (st.mla) |*ms| mlaLayerInc(tmp, hin, &mla_copy, c, ms);
        }
    } else {
        if (w.kda) |kda| {
            var kda_copy = kda;
            if (st.kda) |*ks| kdaLayerInc(tmp, hin, &kda_copy, c, ks);
        }
    }

    if (have_prefix) {
        for (0..E) |i| pref_heap[i] += tmp[i];
    } else {
        @memcpy(pref_heap, tmp);
        have_prefix = true;
    }

    // aggregation before MLP (unconditional)
    {
        var src = std.heap.page_allocator.alloc(f32, (maxb + 1) * E) catch return;
        defer std.heap.page_allocator.free(src);
        var foldM = std.heap.page_allocator.alloc(f32, E) catch return;
        defer std.heap.page_allocator.free(foldM);
        for (0..E) |i| foldM[i] = w.mlp_res_norm[i] * w.mlp_res_proj[i];
        for (0..n_blocks.*) |b| {
            @memcpy(src[b * E ..][0..E], block_residual[b * E ..][0..E]);
        }
        @memcpy(src[n_blocks.* * E ..][0..E], pref_heap);
        ops.attnRes(h, src[0 .. (n_blocks.* + 1) * E], foldM, n_blocks.* + 1, E, c.rms_eps);
    }

    rmsnormF32(hin, h, w.post_norm, E, c.rms_eps);

    if (w.is_dense) {
        denseMlpSlice(tmp, hin, w.dense_gate, w.dense_up, w.dense_down, c.dense_inter, E, c.situ_b1, c.situ_b2);
    } else {
        if (w.moe) |moe| {
            var moe_copy = moe;
            ops.latentMoe(tmp, hin, &moe_copy, &w.moe_cfg);
        }
    }

    for (0..E) |i| pref_heap[i] += tmp[i];
    @memcpy(h, pref_heap);
}

pub fn forwardInc(
    out: []f32, // [E]
    token: u32,
    ms: *ModelState,
    w: *const ModelWeights,
    c: *const KimiK3Config,
) void {
    const E = c.hidden;
    // reset per-token state
    ms.n_blocks = 0;
    for (ms.block_residual) |*b| b.* = 0.0;
    // embedding
    for (0..E) |i| out[i] = w.embed[token * E + i];

    for (0..c.n_layers) |li| {
        const lw = &w.layers[li];
        layerForwardInc(out, ms.block_residual, &ms.n_blocks, lw, c, li, &ms.layers[li]);
    }
    // final norm
    rmsnormF32(out, out, w.final_norm, E, c.rms_eps);
}

// ---------------------------------------------------------------------------
// Phase 5.8b.4: Chunked prefill for MLA (T tokens, KV cache carry-over)
// ---------------------------------------------------------------------------

/// Process a chunk of T MLA tokens with KV-cache carry-over.
/// Each token t attends to cached (prior chunks) + tokens 0..t (within chunk).
pub fn mlaLayerChunked(
    out: []f32, // [T][E]
    x: []const f32, // [T][E]
    w: *const MlaWeights,
    c: *const KimiK3Config,
    st: *MlaLayerState,
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

    const cached = st.cached;

    // projections for all T tokens
    var q = std.heap.page_allocator.alloc(f32, T * H * qh) catch return;
    defer std.heap.page_allocator.free(q);
    var ct = std.heap.page_allocator.alloc(f32, T * kvw) catch return;
    defer std.heap.page_allocator.free(ct);
    var kv = std.heap.page_allocator.alloc(f32, T * H * kvd) catch return;
    defer std.heap.page_allocator.free(kv);
    var rp = std.heap.page_allocator.alloc(f32, T * qr) catch return;
    defer std.heap.page_allocator.free(rp);

    const ql = std.heap.page_allocator.alloc(f32, c.q_lora) catch return;
    defer std.heap.page_allocator.free(ql);

    for (0..T) |t| {
        const xt = x[t * E ..][0..E];
        matmulF32(ql, xt, w.q_a, E, c.q_lora);
        rmsnormF32(ql, ql, w.q_a_norm, c.q_lora, c.rms_eps);
        matmulF32(q[t * H * qh ..][0 .. H * qh], ql, w.q_b, c.q_lora, H * qh);

        matmulF32(ct[t * kvw ..][0..kvw], xt, w.kv_a, E, kvw);
        rmsnormF32(ct[t * kvw ..][0..kv_lora], ct[t * kvw ..][0..kv_lora], w.kv_a_norm, kv_lora, c.rms_eps);
        for (0..qr) |i| rp[t * qr + i] = ct[t * kvw + kv_lora + i];
        matmulF32(kv[t * H * kvd ..][0 .. H * kvd], ct[t * kvw ..][0..kv_lora], w.kv_b, kv_lora, H * kvd);
    }

    // append all T to KV cache at positions cached..cached+T-1
    for (0..T) |t| {
        const pos = cached + t;
        @memcpy(st.kv[pos * H * kvd ..][0 .. H * kvd], kv[t * H * kvd ..][0 .. H * kvd]);
        @memcpy(st.rope[pos * qr ..][0..qr], rp[t * qr ..][0..qr]);
    }

    // attention per token (causal over 0..cached+t)
    var acc = std.heap.page_allocator.alloc(f32, H * vh) catch return;
    defer std.heap.page_allocator.free(acc);
    const gbuf = std.heap.page_allocator.alloc(f32, H * vh) catch return;
    defer std.heap.page_allocator.free(gbuf);
    var sc = std.heap.page_allocator.alloc(f32, cached + T) catch return;
    defer std.heap.page_allocator.free(sc);

    for (0..T) |t| {
        const pos = cached + t;
        for (0..H) |h| {
            const qt = q[t * H * qh + h * qh ..][0..qh];
            var m: f32 = -std.math.inf(f32);
            for (0..(pos + 1)) |s| {
                const ks = st.kv[s * H * kvd + h * kvd ..][0..kvd];
                const kr = st.rope[s * qr ..][0..qr];
                var dd: f64 = 0.0;
                for (0..qn) |i| dd += @as(f64, qt[i]) * @as(f64, ks[i]);
                for (0..qr) |i| dd += @as(f64, qt[qn + i]) * @as(f64, kr[i]);
                sc[s] = @as(f32, @floatCast(dd)) * scale;
                if (sc[s] > m) m = sc[s];
            }
            var zv: f64 = 0.0;
            for (0..(pos + 1)) |s| {
                sc[s] = @exp(sc[s] - m);
                zv += @as(f64, sc[s]);
            }
            const accp = acc[h * vh ..][0..vh];
            for (0..vh) |j| accp[j] = 0.0;
            for (0..(pos + 1)) |s| {
                const pr = sc[s] / @as(f32, @floatCast(zv));
                const vs = st.kv[s * H * kvd + h * kvd + qn ..][0..vh];
                for (0..vh) |j| accp[j] += pr * vs[j];
            }
        }
        if (w.has_gate) {
            matmulF32(gbuf, x[t * E ..][0..E], w.g, E, H * vh);
            for (0..H * vh) |i| acc[i] *= sigmoidF32(gbuf[i]);
        }
        matmulF32(out[t * E ..][0..E], acc, w.o, H * vh, E);
    }

    st.cached += T;
}

/// Chunked prefill for a full KDA layer (T tokens, state carry-over).
pub fn kdaLayerChunked(
    out: []f32, // [T][E]
    x: []const f32, // [T][E]
    w: *const KdaWeights,
    c: *const KimiK3Config,
    conv_state: []f32, // [3*P*hist]
    S: []f32, // [H*D*D]
    T: usize,
) void {
    const E = c.hidden;
    const H = c.kda_heads;
    const D = c.kda_head_dim;
    const P = H * D;
    const K = c.conv_k;
    const hist = K - 1;

    // projections (same as kdaLayerFull)
    var q = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(q);
    var k = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(k);
    var v = std.heap.page_allocator.alloc(f32, T * P) catch return;
    defer std.heap.page_allocator.free(v);
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
        const xt = x[t * E ..][0..E];
        matmulF32(q[t * P ..][0..P], xt, w.q, E, P);
        matmulF32(k[t * P ..][0..P], xt, w.k, E, P);
        matmulF32(v[t * P ..][0..P], xt, w.v, E, P);
        matmulF32(bt[t * H ..][0..H], xt, w.b, E, H);
        const fa = std.heap.page_allocator.alloc(f32, D) catch return;
        defer std.heap.page_allocator.free(fa);
        matmulF32(fa, xt, w.f_a, E, D);
        matmulF32(z[t * P ..][0..P], fa, w.f_b, D, P);
    }

    // ShortConv with CARRIED state (per channel, 3 groups: q,k,v)
    const convws = [_][]const f32{ w.q_conv, w.k_conv, w.v_conv };
    const srcs = [_][]f32{ q, k, v };
    for (0..3) |wi| {
        const src = srcs[wi];
        const convw = convws[wi];
        for (0..P) |ch| {
            var buf: [16]f32 = [_]f32{0} ** 16;
            const off = wi * P * (if (hist > 0) hist else 1) + ch * (if (hist > 0) hist else 1);
            if (hist > 0) {
                for (0..hist) |hh| buf[hh] = conv_state[off + hh];
            }
            for (0..T) |t| {
                const cur = src[t * P + ch];
                var acc: f32 = convw[ch * K + hist] * cur;
                for (0..hist) |j| acc += convw[ch * K + j] * buf[j];
                // shift
                if (hist > 0) {
                    var j: usize = 0;
                    while (j + 1 < hist) : (j += 1) buf[j] = buf[j + 1];
                    buf[hist - 1] = cur;
                }
                src[t * P + ch] = acc * sigmoidF32(acc);
            }
            if (hist > 0) {
                for (0..hist) |hh| conv_state[off + hh] = buf[hh];
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
                const val = k[t * P + h * D + d];
                ss += @as(f64, val) * @as(f64, val);
            }
            const invk: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(D)))) + 1e-6);
            for (0..D) |d| k[t * P + h * D + d] *= invk;
        }
    }

    // beta sigmoid
    for (0..T) |t| {
        for (0..H) |h| bt[t * H + h] = sigmoidF32(bt[t * H + h]);
    }
    // decay
    {
        const gtmp = std.heap.page_allocator.alloc(f32, P) catch return;
        defer std.heap.page_allocator.free(gtmp);
        for (0..T) |t| {
            ops.kdaDecay(gtmp, alpha[t * P ..][0..P], z[t * P ..][0..P], w.A_log, w.dt_bias, H, D, c.gate_lb);
        }
    }

    // recurrence with CARRIED S
    const qscale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    for (0..H) |h| {
        for (0..T) |t| {
            const off = t * P + h * D;
            var qh = std.heap.page_allocator.alloc(f32, D) catch return;
            defer std.heap.page_allocator.free(qh);
            for (0..D) |d| qh[d] = q[off + d] * qscale;
            ops.kdaStep(
                S[h * D * D ..][0 .. D * D],
                o[off ..][0..D],
                qh,
                k[off ..][0..D],
                v[off ..][0..D],
                alpha[off ..][0..D],
                bt[t * H + h],
                D,
                D,
            );
        }
    }

    // norm + gate + o_proj
    for (0..T) |t| {
        const ot = o[t * P ..][0..P];
        for (0..H) |h| rmsnormF32(ot[h * D ..][0..D], ot[h * D ..][0..D], w.o_norm, D, c.rms_eps);
        matmulF32(gb, x[t * E ..][0..E], w.g, E, P);
        for (0..P) |i| ot[i] *= sigmoidF32(gb[i]);
        matmulF32(out[t * E ..][0..E], ot, w.o, P, E);
    }
}

/// Process a chunk of T tokens through a full layer (KDA or MLA) with state carry-over.
pub fn layerForwardChunked(
    h: []f32, // [T][E] in/out
    block_residual: []f32, // [T][maxb][E]
    n_blocks: *usize,
    w: *const LayerWeights,
    c: *const KimiK3Config,
    layer_idx: usize,
    T: usize,
    // state pointers (carried)
    conv_state: ?[]f32,
    S_state: ?[]f32,
    mla_state: ?*MlaLayerState,
) void {
    const E = c.hidden;
    const maxb = c.n_layers / c.attn_res_block + 2;

    // AttnRes pre-attention (per token)
    var pref = std.heap.page_allocator.alloc(f32, T * E) catch return;
    defer std.heap.page_allocator.free(pref);
    @memcpy(pref, h);

    var have_prefix: bool = true;

    if (n_blocks.* > 0) {
        var src = std.heap.page_allocator.alloc(f32, (maxb + 1) * E) catch return;
        defer std.heap.page_allocator.free(src);
        var foldA = std.heap.page_allocator.alloc(f32, E) catch return;
        defer std.heap.page_allocator.free(foldA);
        for (0..E) |i| foldA[i] = w.attn_res_norm[i] * w.attn_res_proj[i];
        for (0..T) |t| {
            for (0..n_blocks.*) |b| {
                @memcpy(src[b * E ..][0..E], block_residual[t * maxb * E + b * E ..][0..E]);
            }
            @memcpy(src[n_blocks.* * E ..][0..E], pref[t * E ..][0..E]);
            ops.attnRes(h[t * E ..][0..E], src[0 .. (n_blocks.* + 1) * E], foldA, n_blocks.* + 1, E, c.rms_eps);
        }
    }

    if (layer_idx % c.attn_res_block == 0) {
        for (0..T) |t| {
            @memcpy(block_residual[t * maxb * E + n_blocks.* * E ..][0..E], pref[t * E ..][0..E]);
        }
        n_blocks.* += 1;
        have_prefix = false;
    }

    var hin = std.heap.page_allocator.alloc(f32, T * E) catch return;
    defer std.heap.page_allocator.free(hin);
    var tmp = std.heap.page_allocator.alloc(f32, T * E) catch return;
    defer std.heap.page_allocator.free(tmp);
    for (0..T) |t| rmsnormF32(hin[t * E ..][0..E], h[t * E ..][0..E], w.in_norm, E, c.rms_eps);

    // Attention: KDA or MLA (chunked with state carry-over)
    if (w.is_mla) {
        if (w.mla) |mla| {
            if (mla_state) |ms| {
                mlaLayerChunked(tmp, hin, &mla, c, ms, T);
            } else {
                mlaLayerFull(tmp, hin, &mla, c, T);
            }
        }
    } else {
        if (w.kda) |kda| {
            if (conv_state != null and S_state != null) {
                kdaLayerChunked(tmp, hin, &kda, c, conv_state.?, S_state.?, T);
            } else {
                kdaLayerFull(tmp, hin, &kda, c, T);
            }
        }
    }

    if (have_prefix) {
        for (0..T * E) |i| pref[i] += tmp[i];
    } else {
        @memcpy(pref, tmp);
        have_prefix = true;
    }

    // AttnRes before MLP
    {
        var src = std.heap.page_allocator.alloc(f32, (maxb + 1) * E) catch return;
        defer std.heap.page_allocator.free(src);
        var foldM = std.heap.page_allocator.alloc(f32, E) catch return;
        defer std.heap.page_allocator.free(foldM);
        for (0..E) |i| foldM[i] = w.mlp_res_norm[i] * w.mlp_res_proj[i];
        for (0..T) |t| {
            for (0..n_blocks.*) |b| {
                @memcpy(src[b * E ..][0..E], block_residual[t * maxb * E + b * E ..][0..E]);
            }
            @memcpy(src[n_blocks.* * E ..][0..E], pref[t * E ..][0..E]);
            ops.attnRes(h[t * E ..][0..E], src[0 .. (n_blocks.* + 1) * E], foldM, n_blocks.* + 1, E, c.rms_eps);
        }
    }

    for (0..T) |t| rmsnormF32(hin[t * E ..][0..E], h[t * E ..][0..E], w.post_norm, E, c.rms_eps);

    if (w.is_dense) {
        for (0..T) |t| {
            denseMlpSlice(tmp[t * E ..][0..E], hin[t * E ..][0..E], w.dense_gate, w.dense_up, w.dense_down, c.dense_inter, E, c.situ_b1, c.situ_b2);
        }
    } else {
        if (w.moe) |moe| {
            var moe_copy = moe;
            for (0..T) |t| {
                ops.latentMoe(tmp[t * E ..][0..E], hin[t * E ..][0..E], &moe_copy, &w.moe_cfg);
            }
        }
    }

    for (0..T * E) |i| pref[i] += tmp[i];
    @memcpy(h, pref);
}

/// Chunked forward: process tokens in a chunk of `chunk_size` with state carry-over.
/// Falls back to sequential forwardInc for correctness; state is carried via ms.
pub fn forwardChunked(
    out: []f32, // [N][E]
    x_tokens: []const u32, // [N]
    w: *const ModelWeights,
    c: *const KimiK3Config,
    ms: *ModelState,
    chunk_size: usize,
) void {
    const E = c.hidden;
    const N = x_tokens.len;
    // Process in chunks
    var i: usize = 0;
    while (i < N) {
        const chunk_end = @min(i + chunk_size, N);
        // Process each token in chunk with forwardInc (state carry-over)
        var ci: usize = i;
        while (ci < chunk_end) : (ci += 1) {
            const tok = x_tokens[ci];
            const h_tok = std.heap.page_allocator.alloc(f32, E) catch return;
            defer std.heap.page_allocator.free(h_tok);
            forwardInc(h_tok, tok, ms, w, c);
            // Store output
            @memcpy(out[ci * E ..][0..E], h_tok);
        }
        i = chunk_end;
    }
}
