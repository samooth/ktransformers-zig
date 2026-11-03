//! Kimi K3 CPU ops (port of zig-ai's kimi_k3_ops.zig), scalar reference in f32.
//!
//! Phase 1 of the port: pure CPU, f32, no SIMD. These are the kernels whose
//! contract matters (order, indexing, nibble order) and which have no direct
//! equivalent in ktransformers-zig yet. Kernels that already exist (RMSNorm,
//! MLA, MoE, MXFP4) are reused from their modules; only the K3-specific
//! variants and the novel ones (SiTU-GLU, KDA decay/step, AttnRes, router,
//! LatentMoE) live here.
//!
//! All functions operate on flat slices and take dimensions explicitly so they
//! can be tested against small hand-computed fixtures.

const std = @import("std");

// ---------------------------------------------------------------------------
// ShortConv (causal depthwise conv + fused SiLU)
// ---------------------------------------------------------------------------

/// Causal depthwise convolution with fused SiLU.
/// w is [channels][k]; state holds (k-1) previous inputs per channel, oldest first,
/// and is UPDATED IN PLACE. state may be null (treat history as zero).
/// x/y layout: [T][channels] (row-major by token). w layout: [channels][k].
pub fn shortConv(
    y: []f32,
    x: []const f32,
    w: []const f32,
    state: ?[]f32,
    channels: usize,
    k: usize,
    T: usize,
) void {
    const hist = k - 1;
    var buf = [_]f32{0} ** 64; // channels handled per-channel; hist <= 64 for K3 (4-1)
    // hist is small (3 for K3); guard generically with a heap buffer if needed.
    for (0..channels) |c| {
        if (hist > 0) {
            if (state) |st| {
                for (0..hist) |j| buf[j] = st[c * hist + j];
            } else {
                for (0..hist) |j| buf[j] = 0.0;
            }
        }
        for (0..T) |t| {
            const cur = x[t * channels + c];
            var acc: f32 = w[c * k + hist] * cur;
            for (0..hist) |j| acc += w[c * k + j] * buf[j];
            // shift history
            if (hist > 0) {
                var j: usize = 0;
                while (j + 1 < hist) : (j += 1) buf[j] = buf[j + 1];
                buf[hist - 1] = cur;
            }
            // SiLU: x * sigmoid(x)
            const sig: f32 = 1.0 / (1.0 + @exp(-acc));
            y[t * channels + c] = acc * sig;
        }
        if (state) |st| {
            if (hist > 0) {
                for (0..hist) |j| st[c * hist + j] = buf[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SiTU-GLU
// ---------------------------------------------------------------------------

/// SiTU-GLU over a 2*n input laid out as [gate | up].
///   a = b1 * tanh(gate / b1) * sigmoid(gate)   (sigmoid sees the UNCAPPED gate)
///   u = b2 * tanh(up / b2)
///   y = a * u
pub fn situGlu(out: []f32, x: []const f32, n: usize, b1: f32, b2: f32) void {
    const gate = x;
    const up = x[n..];
    for (0..n) |i| {
        const g = gate[i];
        const a = b1 * std.math.tanh(g / b1) * (1.0 / (1.0 + @exp(-g)));
        const u = b2 * std.math.tanh(up[i] / b2);
        out[i] = a * u;
    }
}

// ---------------------------------------------------------------------------
// KDA decay
// ---------------------------------------------------------------------------

/// Decay chain, per head h and channel d:
///   z     = (input, overwritten with alpha output)
///   a     = exp(A_log[h])          (A_log PER HEAD)
///   u     = a * (z[i] + dt_bias[i])
///   g     = lb * sigmoid(u)        in (lb, 0]
///   alpha = exp(g)                 in (e^lb, 1]
/// Writes g_out[i]=g, alpha_out[i]=alpha. z is the f_b(f_a(x)) input.
pub fn kdaDecay(
    g_out: []f32,
    alpha_out: []f32,
    z: []const f32,
    A_log: []const f32,
    dt_bias: []const f32,
    H: usize,
    D: usize,
    lb: f32,
) void {
    for (0..H) |h| {
        const a = @exp(A_log[h]);
        for (0..D) |d| {
            const i = h * D + d;
            const u = a * (z[i] + dt_bias[i]);
            const gi = lb * (1.0 / (1.0 + @exp(-u)));
            g_out[i] = gi;
            alpha_out[i] = @exp(gi);
        }
    }
}

// ---------------------------------------------------------------------------
// KDA recurrence step (one head)
// ---------------------------------------------------------------------------

/// One KDA recurrence step for one head. S is [dk][dv] row-major (dk*dv floats).
/// Order (load bearing):
///   1. decay   S[i][:] *= alpha[i]
///   2. read    u = S^T k
///   3. write   S += k * (beta*(v-u))^T
///   4. output  o = S^T q   from the ALREADY UPDATED state
/// q must arrive pre-scaled by dk^-0.5 (caller's job).
pub fn kdaStep(
    S: []f32,
    o: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    alpha: []const f32,
    beta: f32,
    dk: usize,
    dv: usize,
) void {
    // 1. decay
    for (0..dk) |i| {
        const a = alpha[i];
        for (0..dv) |j| {
            S[i * dv + j] *= a;
        }
    }
    // 2. read u = S^T k
    var u: [256]f32 = [_]f32{0} ** 256;
    for (0..dv) |j| u[j] = 0.0;
    for (0..dk) |i| {
        const ki = k[i];
        if (ki == 0.0) continue;
        for (0..dv) |j| {
            u[j] += ki * S[i * dv + j];
        }
    }
    // 3. write
    for (0..dk) |i| {
        const ki = k[i];
        if (ki == 0.0) continue;
        for (0..dv) |j| {
            S[i * dv + j] += ki * beta * (v[j] - u[j]);
        }
    }
    // 4. output o = S^T q
    for (0..dv) |j| o[j] = 0.0;
    for (0..dk) |i| {
        const qi = q[i];
        if (qi == 0.0) continue;
        for (0..dv) |j| {
            o[j] += qi * S[i * dv + j];
        }
    }
}

// ---------------------------------------------------------------------------
// AttnRes
// ---------------------------------------------------------------------------

/// AttnRes aggregation over nsrc sources of width n.
///   keys = RMSNorm(sources) (over n, eps)
///   score = dot(key, fold)   (fold already carries norm.weight*proj.weight)
///   out = softmax(score) @ sources (RAW sources)
/// src layout: [nsrc][n] (flat). fold length n.
pub fn attnRes(
    out: []f32,
    src: []const f32,
    fold: []const f32,
    nsrc: usize,
    n: usize,
    eps: f32,
) void {
    // scores
    var scores: [64]f32 = [_]f32{0} ** 64;
    for (0..nsrc) |s| {
        const v = src[s * n ..][0..n];
        // rmsnorm of v
        var ss: f64 = 0.0;
        for (v) |x| ss += @as(f64, x) * @as(f64, x);
        const inv: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(n)))) + eps);
        var acc: f64 = 0.0;
        for (0..n) |i| {
            acc += @as(f64, v[i] * inv) * @as(f64, fold[i]);
        }
        scores[s] = @floatCast(acc);
    }
    // softmax
    var m: f32 = scores[0];
    for (1..nsrc) |s| {
        if (scores[s] > m) m = scores[s];
    }
    var z: f64 = 0.0;
    for (0..nsrc) |s| {
        scores[s] = @exp(scores[s] - m);
        z += @as(f64, scores[s]);
    }
    for (0..nsrc) |s| scores[s] = scores[s] / @as(f32, @floatCast(z));

    // out = sum_s p_s * src[s]
    for (0..n) |i| out[i] = 0.0;
    for (0..nsrc) |s| {
        const p = scores[s];
        const v = src[s * n ..][0..n];
        for (0..n) |i| out[i] += p * v[i];
    }
}

// ---------------------------------------------------------------------------
// Router (LatentMoE)
// ---------------------------------------------------------------------------

/// MoE routing, one token.
///   logits = W[e] . x  (float32, no bias)
///   scores = sigmoid(logits)  (independent)
///   sel = topk(scores + bias)  (bias steers selection only)
///   w = scores[sel]  (unbiased)
///   w /= sum(w)+1e-20  (if renorm)
///   w *= routed_scale
/// idx,w written descending score. Fills at most topk entries.
pub fn router(
    idx: []i32,
    w: []f32,
    x: []const f32,
    W: []const f32,
    bias: ?[]const f32,
    hidden: usize,
    n_experts: usize,
    topk: usize,
    renorm: bool,
    routed_scale: f32,
) void {
    // scores
    var score_buf: [4096]f32 = undefined;
    var choice_buf: [4096]f32 = undefined;
    // n_experts <= 4096 for K3 (896); guard
    for (0..n_experts) |e| {
        const row = W[e * hidden ..][0..hidden];
        var acc: f64 = 0.0;
        for (0..hidden) |i| acc += @as(f64, row[i]) * @as(f64, x[i]);
        const sc: f32 = @floatCast(1.0 / (1.0 + @exp(-@as(f64, acc))));
        score_buf[e] = sc;
        const b = if (bias) |bb| bb[e] else 0.0;
        choice_buf[e] = sc + b;
    }
    // topk by repeated max
    for (0..topk) |j| {
        var best: isize = -1;
        var bv: f32 = -std.math.inf(f32);
        for (0..n_experts) |e| {
            if (choice_buf[e] > bv) {
                bv = choice_buf[e];
                best = @intCast(e);
            }
        }
        if (best < 0) {
            idx[j] = 0;
            w[j] = 0.0;
            continue;
        }
        const be: usize = @intCast(best);
        idx[j] = @intCast(be);
        w[j] = score_buf[be]; // unbiased
        choice_buf[be] = -std.math.inf(f32);
    }
    if (renorm and topk > 1) {
        var s: f64 = 0.0;
        for (0..topk) |j| s += @as(f64, w[j]);
        const inv: f32 = @floatCast(1.0 / (s + 1e-20));
        for (0..topk) |j| w[j] *= inv;
    }
    for (0..topk) |j| w[j] *= routed_scale;
}

// ---------------------------------------------------------------------------
// LatentMoE (one token, resident fp32 expert bank)
// ---------------------------------------------------------------------------

pub const LatentMoeWeights = struct {
    gate: []const f32, // [n_experts][hidden]
    bias: []const f32, // [n_experts]
    down: []const f32, // [latent][hidden]  (hidden->latent)
    up: []const f32, // [hidden][latent]   (latent->hidden)
    latent_norm: []const f32, // [latent]
    sh1: []const f32, // [SI][hidden]  SI = moe_inter*n_shared
    sh3: []const f32, // [SI][hidden]
    sh2: []const f32, // [hidden][SI]
    // resident expert bank (fixture-only path)
    w1: []const f32, // [n_experts][moe_inter][latent]
    w3: []const f32,
    w2: []const f32, // [n_experts][latent][moe_inter]
};

pub const LatentMoeCfg = struct {
    hidden: usize,
    latent: usize,
    moe_inter: usize,
    n_shared: usize,
    n_experts: usize,
    topk: usize,
    situ_b1: f32,
    situ_b2: f32,
    routed_scale: f32,
    moe_renorm: bool,
    latent_norm: bool,
    rms_eps: f32,
};

fn matmulF32(y: []f32, x: []const f32, W: []const f32, in: usize, out: usize) void {
    for (0..out) |o| {
        var acc: f64 = 0.0;
        for (0..in) |i| acc += @as(f64, W[o * in + i]) * @as(f64, x[i]);
        y[o] = @floatCast(acc);
    }
}

fn rmsnormInPlace(y: []f32, w: []const f32, n: usize, eps: f32) void {
    var ss: f64 = 0.0;
    for (0..n) |i| ss += @as(f64, y[i]) * @as(f64, y[i]);
    const inv: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(n)))) + eps);
    for (0..n) |i| y[i] = w[i] * y[i] * inv;
}

pub fn latentMoe(out: []f32, x: []const f32, w: *const LatentMoeWeights, c: *const LatentMoeCfg) void {
    const E = c.hidden;
    const L = c.latent;
    const I = c.moe_inter;
    const SI = I * c.n_shared;

    // 1. route
    var idx_buf: [64]i32 = [_]i32{0} ** 64;
    var wt_buf: [64]f32 = [_]f32{0} ** 64;
    router(idx_buf[0..c.topk], wt_buf[0..c.topk], x, w.gate, w.bias, E, c.n_experts, c.topk, c.moe_renorm, c.routed_scale);

    // 2. down-project to latent
    var z: [4096]f32 = undefined;
    matmulF32(z[0..L], x, w.down, E, L);

    // 3. experts in latent space
    var accL: [4096]f32 = [_]f32{0} ** 4096;
    for (0..L) |i| accL[i] = 0.0;
    for (0..c.topk) |j| {
        const ei: usize = @intCast(idx_buf[j]);
        const wj = wt_buf[j];
        // gate/up
        var gu: [8192]f32 = undefined;
        matmulF32(gu[0..I], z[0..L], w.w1[ei * I * L ..][0..(I * L)], L, I);
        matmulF32(gu[I .. I + I], z[0..L], w.w3[ei * I * L ..][0..(I * L)], L, I);
        // situ glu
        var act: [4096]f32 = undefined;
        situGlu(act[0..I], gu[0 .. 2 * I], I, c.situ_b1, c.situ_b2);
        // down
        var edn: [4096]f32 = undefined;
        matmulF32(edn[0..L], act[0..I], w.w2[ei * L * I ..][0..(L * I)], I, L);
        for (0..L) |i| accL[i] += wj * edn[i];
    }

    // 4. RMSNorm on aggregate
    if (c.latent_norm) rmsnormInPlace(accL[0..L], w.latent_norm, L, c.rms_eps);

    // 5. up-project
    matmulF32(out, accL[0..L], w.up, L, E);

    // 6. shared expert on original input, added unweighted
    var sgu: [8192]f32 = undefined;
    matmulF32(sgu[0..SI], x, w.sh1, E, SI);
    matmulF32(sgu[SI .. SI + SI], x, w.sh3, E, SI);
    var sact: [4096]f32 = undefined;
    situGlu(sact[0..SI], sgu[0 .. 2 * SI], SI, c.situ_b1, c.situ_b2);
    var sdn: [4096]f32 = undefined;
    matmulF32(sdn[0..E], sact[0..SI], w.sh2, SI, E);
    for (0..E) |i| out[i] += sdn[i];
}