// Tests for Kimi K3 CPU ops (Phase 1).

const std = @import("std");
const testing = std.testing;

const root = @import("kt");
const ops = root.kimi_k3_ops;

test "situGlu basic" {
    const n: usize = 2;
    var x = [_]f32{ 1.0, 2.0, 0.5, -0.5 };
    var out = [_]f32{ 0.0, 0.0 };
    ops.situGlu(&out, &x, n, 1.0, 1.0);
    // a = 1*tanh(1)*sigmoid(1) ; u = 1*tanh(0.5) ; y = a*u
    const a = 1.0 * std.math.tanh(@as(f32, 1.0)) * (1.0 / (1.0 + @exp(-@as(f32, 1.0))));
    const u = 1.0 * std.math.tanh(@as(f32, 0.5));
    try testing.expectApproxEqAbs(a * u, out[0], 1e-5);
}

test "kdaDecay per head" {
    const H: usize = 1;
    const D: usize = 2;
    const z = [_]f32{ 0.0, 0.0 };
    const A_log = [_]f32{0.0}; // exp(0)=1
    const dt_bias = [_]f32{ 0.0, 0.0 };
    var g = [_]f32{ 0.0, 0.0 };
    var alpha = [_]f32{ 0.0, 0.0 };
    ops.kdaDecay(&g, &alpha, &z, &A_log, &dt_bias, H, D, -1.0);
    // u = 1*(0+0)=0; g = -1*sigmoid(0) = -0.5; alpha = exp(-0.5)
    try testing.expectApproxEqAbs(-0.5, g[0], 1e-5);
    try testing.expectApproxEqAbs(@exp(-0.5), alpha[0], 1e-5);
}

test "kdaStep recurrence order" {
    const dk: usize = 1;
    const dv: usize = 1;
    var S = [_]f32{2.0};
    const q = [_]f32{1.0};
    const k = [_]f32{1.0};
    const v = [_]f32{5.0};
    const alpha = [_]f32{0.5};
    const beta: f32 = 1.0;
    var o = [_]f32{0.0};
    ops.kdaStep(&S, &o, &q, &k, &v, &alpha, beta, dk, dv);
    // decay: S=1.0; u=S*k=1.0; write S += k*1*(5-1)=1+4=5; o=S*q=5
    try testing.expectApproxEqAbs(5.0, o[0], 1e-5);
    try testing.expectApproxEqAbs(5.0, S[0], 1e-5);
}

test "attnRes aggregation" {
    const nsrc: usize = 2;
    const n: usize = 2;
    // sources: [1,0], [0,1]
    const src = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const fold = [_]f32{ 1.0, 1.0 };
    var out = [_]f32{ 0.0, 0.0 };
    ops.attnRes(&out, &src, &fold, nsrc, n, 1e-6);
    // keys = rmsnorm(src) (unit norm) ; scores = dot(key, fold) = sum of key
    // both keys have equal norm; scores equal -> softmax 0.5 each -> out = 0.5*[1,0]+0.5*[0,1]=[0.5,0.5]
    try testing.expectApproxEqAbs(0.5, out[0], 1e-4);
    try testing.expectApproxEqAbs(0.5, out[1], 1e-4);
}

test "router topk" {
    const hidden: usize = 2;
    const n_experts: usize = 3;
    const topk: usize = 2;
    // W rows: [1,0],[0,1],[1,1]
    const W = [_]f32{ 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    const x = [_]f32{ 1.0, 1.0 };
    var idx = [_]i32{ 0, 0 };
    var w = [_]f32{ 0.0, 0.0 };
    ops.router(&idx, &w, &x, &W, null, hidden, n_experts, topk, true, 1.0);
    // scores: sigmoid(1)=0.731, sigmoid(1)=0.731, sigmoid(2)=0.881 -> top2: expert2 (0.881), then tie 0/1
    // selected idx should include 2
    var has2 = false;
    for (idx) |v| if (v == 2) {
        has2 = true;
    };
    try testing.expect(has2);
}

test "latentMoe basic" {
    const E: usize = 2;
    const L: usize = 2;
    const I: usize = 2;
    const n_experts: usize = 2;
    const topk: usize = 1;
    var gate = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var bias = [_]f32{ 0.0, 0.0 };
    var down = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var up = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var latent_norm = [_]f32{ 1.0, 1.0 };
    var sh1 = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var sh3 = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var sh2 = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var w1 = [_]f32{ 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0 };
    var w3 = [_]f32{ 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0 };
    var w2 = [_]f32{ 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0 };
    const lw = ops.LatentMoeWeights{
        .gate = &gate,
        .bias = &bias,
        .down = &down,
        .up = &up,
        .latent_norm = &latent_norm,
        .sh1 = &sh1,
        .sh3 = &sh3,
        .sh2 = &sh2,
        .w1 = &w1,
        .w3 = &w3,
        .w2 = &w2,
    };
    const cfg = ops.LatentMoeCfg{
        .hidden = E,
        .latent = L,
        .moe_inter = I,
        .n_shared = 1,
        .n_experts = n_experts,
        .topk = topk,
        .situ_b1 = 1.0,
        .situ_b2 = 1.0,
        .routed_scale = 1.0,
        .moe_renorm = false,
        .latent_norm = true,
        .rms_eps = 1e-6,
    };
    const x = [_]f32{ 1.0, 1.0 };
    var out = [_]f32{ 0.0, 0.0 };
    ops.latentMoe(&out, &x, &lw, &cfg);
    // just ensure finite
    try testing.expect(std.math.isFinite(out[0]));
    try testing.expect(std.math.isFinite(out[1]));
}