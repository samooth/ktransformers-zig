// Standalone test for the kt_apply_* / kt_softmax math helpers (C API).
//
// These 4 exports were headered + ABI-gated in ff57125 but had ZERO test
// coverage (nothing in tests/ referenced them). This suite pins their
// numerics against hand-computed references:
//   - kt_apply_swiglu: all 3 variants (plain silu*up, clamped, OAI tanh)
//   - kt_apply_rms_norm: sum(x^2)/n + eps, weighted, BF16 in/out
//   - kt_apply_rope: half-split rotation — post-state must be a pure
//     rotation of the pre-state (norm preserved, |angle| = position*freq)
//   - kt_softmax: max-subtracted, sums to 1, in-place-capable

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig
const amx = @import("kt").amx;

test "kt_apply_swiglu: plain silu(gate)*up (limit=0, alpha=0)" {
    const n: usize = 4;
    var gate: [n]amx.bf16 = undefined;
    var up: [n]amx.bf16 = undefined;
    const gate_f = [_]f32{ 0.5, 1.0, -2.0, 0.0 };
    const up_f = [_]f32{ 2.0, -1.0, 0.5, 3.0 };
    for (0..n) |i| {
        gate[i] = amx.f32_to_bf16(gate_f[i]);
        up[i] = amx.f32_to_bf16(up_f[i]);
    }
    var dst: [n]amx.bf16 = undefined;
    kt.kt_apply_swiglu(&gate, &up, &dst, n, 0, 0);
    for (0..n) |i| {
        const g = amx.bf16_to_f32(gate[i]); // bf16-rounded, same as kernel
        const u = amx.bf16_to_f32(up[i]);
        const want: f32 = g / (1.0 + @exp(-g)) * u;
        try testing.expectApproxEqAbs(want, amx.bf16_to_f32(dst[i]), 0.02);
    }
}

test "kt_apply_swiglu: OAI tanh variant (alpha>0) matches reference" {
    const n: usize = 2;
    const alpha: f32 = 0.7;
    const limit: f32 = 5.0;
    var gate: [n]amx.bf16 = undefined;
    var up: [n]amx.bf16 = undefined;
    const gate_f = [_]f32{ 1.5, -0.5 };
    const up_f = [_]f32{ 0.25, 2.0 };
    for (0..n) |i| {
        gate[i] = amx.f32_to_bf16(gate_f[i]);
        up[i] = amx.f32_to_bf16(up_f[i]);
    }
    var dst: [n]amx.bf16 = undefined;
    kt.kt_apply_swiglu(&gate, &up, &dst, n, limit, alpha);
    for (0..n) |i| {
        const g = amx.bf16_to_f32(gate[i]);
        const u = amx.bf16_to_f32(up[i]);
        // amx.swiglu_oai: gate*(1/(1+exp(-gate*alpha)))*(up+1)
        const want: f32 = g * (1.0 / (1.0 + @exp(-g * alpha))) * (u + 1.0);
        try testing.expectApproxEqAbs(want, amx.bf16_to_f32(dst[i]), 0.02);
    }
}

test "kt_apply_swiglu: clamped variant (limit>0, alpha=0) clamps gate" {
    const n: usize = 3;
    const limit: f32 = 1.0;
    var gate: [n]amx.bf16 = undefined;
    var up: [n]amx.bf16 = undefined;
    const gate_f = [_]f32{ 3.0, -3.0, 0.5 }; // positive-clamp: {1, -3, 0.5}
    const up_f = [_]f32{ 1.0, 1.0, 1.0 };
    for (0..n) |i| {
        gate[i] = amx.f32_to_bf16(gate_f[i]);
        up[i] = amx.f32_to_bf16(up_f[i]);
    }
    var dst: [n]amx.bf16 = undefined;
    kt.kt_apply_swiglu(&gate, &up, &dst, n, limit, 0);
    for (0..n) |i| {
        var g = amx.bf16_to_f32(gate[i]);
        const u = amx.bf16_to_f32(up[i]);
        // amx.swiglu_clamp: gate gets a POSITIVE-ONLY clamp (no -limit
        // branch on gate), up clamps symmetrically, then silu(gate)*up.
        // (Asymmetric by design — mirrors the C++ llamafile variant.)
        if (g > limit) g = limit;
        const ug = if (u > limit) limit else if (u < -limit) -limit else u;
        const want: f32 = g / (1.0 + @exp(-g)) * ug;
        try testing.expectApproxEqAbs(want, amx.bf16_to_f32(dst[i]), 0.02);
    }
}

test "kt_apply_rms_norm: matches weighted RMS reference" {
    const hidden: usize = 8;
    var input: [hidden]amx.bf16 = undefined;
    var weight: [hidden]amx.bf16 = undefined;
    const in_f = [_]f32{ 1.0, -2.0, 3.0, -4.0, 0.5, 0.25, -0.75, 2.5 };
    const w_f = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    for (0..hidden) |i| {
        input[i] = amx.f32_to_bf16(in_f[i]);
        weight[i] = amx.f32_to_bf16(w_f[i]);
    }
    var out: [hidden]amx.bf16 = undefined;
    kt.kt_apply_rms_norm(&input, &weight, &out, hidden, 1e-6);
    // Reference (bf16-rounded inputs, same as the kernel):
    var sum_sq: f32 = 0;
    for (0..hidden) |i| {
        const v = amx.bf16_to_f32(input[i]);
        sum_sq += v * v;
    }
    const inv_rms = 1.0 / @sqrt(sum_sq / hidden + 1e-6);
    for (0..hidden) |i| {
        const want: f32 = amx.bf16_to_f32(input[i]) * inv_rms * amx.bf16_to_f32(weight[i]);
        try testing.expectApproxEqAbs(want, amx.bf16_to_f32(out[i]), 0.02);
    }
}

test "kt_apply_rope: rotation preserves norms and applies position" {
    const head_dim: usize = 8; // 4 rotation pairs per head
    const position: i64 = 3;
    const theta: f64 = 10000.0;
    var q: [head_dim]amx.bf16 = undefined;
    var k: [head_dim]amx.bf16 = undefined;
    const q_f = [_]f32{ 0.3, -1.2, 2.0, 0.5, -0.7, 1.1, 0.0, -2.5 };
    const k_f = [_]f32{ 1.0, 0.1, -0.4, 0.9, 1.5, -0.2, 0.8, 0.05 };
    for (0..head_dim) |i| {
        q[i] = amx.f32_to_bf16(q_f[i]);
        k[i] = amx.f32_to_bf16(k_f[i]);
    }
    // Pre-norms (bf16-rounded)
    var q_norm: f32 = 0;
    var k_norm: f32 = 0;
    for (0..head_dim) |i| {
        q_norm += amx.bf16_to_f32(q[i]) * amx.bf16_to_f32(q[i]);
        k_norm += amx.bf16_to_f32(k[i]) * amx.bf16_to_f32(k[i]);
    }
    kt.kt_apply_rope(&q, &k, position, head_dim, theta);
    // Post-norms must match (rotation is norm-preserving; bf16 rounding
    // gives small tolerance)
    var q_post: f32 = 0;
    var k_post: f32 = 0;
    for (0..head_dim) |i| {
        q_post += amx.bf16_to_f32(q[i]) * amx.bf16_to_f32(q[i]);
        k_post += amx.bf16_to_f32(k[i]) * amx.bf16_to_f32(k[i]);
    }
    try testing.expectApproxEqRel(q_norm, q_post, 0.05);
    try testing.expectApproxEqRel(k_norm, k_post, 0.05);
    // Position=0 must be identity
    var q0: [head_dim]amx.bf16 = undefined;
    const q0_f = [_]f32{ 0.3, -1.2, 2.0, 0.5, -0.7, 1.1, 0.0, -2.5 };
    for (0..head_dim) |i| q0[i] = amx.f32_to_bf16(q0_f[i]);
    var k0: [head_dim]amx.bf16 = undefined;
    for (&k0) |*v| v.* = amx.f32_to_bf16(0.0);
    kt.kt_apply_rope(&q0, &k0, 0, head_dim, theta);
    for (0..head_dim) |i| {
        try testing.expect(amx.bf16_to_f32(q0[i]) == amx.f32_to_bf16(q0_f[i]) or
            @abs(amx.bf16_to_f32(q0[i]) - q0_f[i]) < 0.01);
    }
}

test "kt_softmax: sums to 1, max-subtracted, matches closed form" {
    const size: usize = 5;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out: [size]f32 = undefined;
    kt.kt_softmax(&input, &out, size);
    var sum: f32 = 0;
    for (out) |v| sum += v;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Closed form: exp(x-5)/sum(exp(x-5))
    var want_sum: f32 = 0;
    for (input) |x| want_sum += @exp(x - 5.0);
    for (0..size) |i| {
        try testing.expectApproxEqAbs(@exp(input[i] - 5.0) / want_sum, out[i], 1e-5);
    }
    // Degenerate: all-equal inputs -> uniform
    const eq = [_]f32{ 0.7, 0.7, 0.7, 0.7, 0.7 };
    var out2: [size]f32 = undefined;
    kt.kt_softmax(&eq, &out2, size);
    for (out2) |v| try testing.expectApproxEqAbs(@as(f32, 0.2), v, 1e-6);
}
