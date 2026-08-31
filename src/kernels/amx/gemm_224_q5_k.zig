// GEMM Kernel 224 Q5_K (GGML K-quant) for AMX/AVX512
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256):
//   block_q5_K {
//     ggml_half d, dmin;      // super-block scales (scales / mins)
//     uint8_t scales[12];    // 8 sub-block scales+mins, 6-bit packed
//     uint8_t qh[QK_K/8];    // 32 bytes: quants, high bit (bit 4)
//     uint8_t qs[QK_K/2];    // 128 bytes: quants, low 4 bits
//   } = 176 bytes per 256 weights (8 sub-blocks of 32)
//
// Weight reconstruction (ggml-quants.c:1731 dequantize_row_q5_K):
//   y = d * sc * ((qs nibble) + (qh bit ? 16 : 0)) - dmin * m
// with (sc, m) from get_scale_min_k4 and the qh bit masks u1=1,u2=2
// shifting left by 2 per 64-weight chunk.
//
// Quantization (:1644) mirrors q4_K but with nmax=31, rmin=-0.5, nstep=15;
// the 0..31 levels split into a 4-bit plane (qs) and a 1-bit plane (qh).
//
// Phase 1 of the GGML-quant workstream. Q8_0, Q4_K, Q6_K have landed.
// C-API wrappers are Phase 2.

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// Q5_K Block Structure (byte-exact vs ggml's block_q5_K)
// ============================================================================

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
const SUB_BLOCK: usize = 32; // weights per sub-block
const N_SUB: usize = QK_K / SUB_BLOCK; // = 8

pub const BlockQ5_K = extern struct {
    d: u16, // f16: super-block scale for quantized scales
    dmin: u16, // f16: super-block scale for quantized mins
    scales: [K_SCALE_SIZE]u8, // 6-bit packed per-sub-block (sc, m)
    qh: [QK_K / 8]u8, // high bit of each 5-bit quant
    qs: [QK_K / 2]u8, // low 4 bits of each 5-bit quant
};

comptime {
    if (@sizeOf(BlockQ5_K) != 176) @compileError("BlockQ5_K must be 176 bytes");
}

// ============================================================================
// f16 <-> f32 (native f16, same as the other GGML kernels)
// ============================================================================

pub fn f32_to_f16(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

pub fn f16_to_f32(x: u16) f32 {
    const h: f16 = @bitCast(x);
    return @floatCast(h);
}

// ============================================================================
// ggml helpers (ported exactly)
// ============================================================================

pub fn nearestInt(fval: f32) i32 {
    std.debug.assert(@abs(fval) <= 4194303.0);
    const val: f32 = fval + 12582912.0;
    const i: i32 = @bitCast(val);
    return (i & 0x007f_ffff) - 0x0040_0000;
}

/// ggml-quants.c:880 get_scale_min_k4 — unpack 6-bit (sc, m) pair j.
pub fn getScaleMinK4(j: usize, q: [*]const u8, d: *u8, m: *u8) void {
    if (j < 4) {
        d.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        d.* = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m.* = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// ggml-quants.c:799 make_qkx2_quants — error-minimizing scale/min search.
/// q5_K calls it with nmax=31, rmin=-0.5, rdelta=0.1, nstep=15.
fn makeQkx2Quants(
    n: usize,
    nmax: i32,
    x: []const f32,
    weights: []const f32,
    L: []u8,
    the_min: *f32,
    Laux: []u8,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) f32 {
    var min = x[0];
    var max = x[0];
    var sum_w = weights[0];
    var sum_x = sum_w * x[0];
    for (1..n) |i| {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        const w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (0..n) |i| L[i] = 0;
        the_min.* = -min;
        return 0.0;
    }
    var iscale = @as(f32, @floatFromInt(nmax)) / (max - min);
    var scale = 1.0 / iscale;
    var best_error: f32 = 0;
    for (0..n) |i| {
        const l = nearestInt(iscale * (x[i] - min));
        L[i] = @intCast(std.math.clamp(l, 0, nmax));
        var diff = scale * @as(f32, @floatFromInt(L[i])) + min - x[i];
        diff = if (use_mad) @abs(diff) else diff * diff;
        const w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        the_min.* = -min;
        return scale;
    }
    var is: i32 = 0;
    while (is <= nstep) : (is += 1) {
        iscale = (rmin + rdelta * @as(f32, @floatFromInt(is)) + @as(f32, @floatFromInt(nmax))) / (max - min);
        var sum_l: f32 = 0;
        var sum_l2: f32 = 0;
        var sum_xl: f32 = 0;
        for (0..n) |i| {
            const l = nearestInt(iscale * (x[i] - min));
            const li = std.math.clamp(l, 0, nmax);
            Laux[i] = @intCast(li);
            const w = weights[i];
            const lf: f32 = @floatFromInt(li);
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        const D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            var this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            var this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            var cur_error: f32 = 0;
            for (0..n) |i| {
                var diff = this_scale * @as(f32, @floatFromInt(Laux[i])) + this_min - x[i];
                diff = if (use_mad) @abs(diff) else diff * diff;
                const w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (0..n) |i| L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    the_min.* = -min;
    return scale;
}

// ============================================================================
// Quantize / Dequantize (byte-exact vs ggml-quants.c)
// ============================================================================

/// ggml-quants.c:1644 quantize_row_q5_K_ref. k must be a multiple of 256.
pub fn quantizeRowQ5_K(src: []const f32, dst: [*]BlockQ5_K, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;

    var L: [QK_K]u8 = undefined;
    var Laux: [SUB_BLOCK]u8 = undefined;
    var w: [SUB_BLOCK]f32 = undefined;
    var mins: [N_SUB]f32 = undefined;
    var scales: [N_SUB]f32 = undefined;

    for (0..nb) |i| {
        const x = src[i * QK_K ..][0..QK_K];
        const blk = &dst[i];

        var max_scale: f32 = 0;
        var max_min: f32 = 0;
        for (0..N_SUB) |j| {
            const xs = x[j * SUB_BLOCK ..][0..SUB_BLOCK];
            var sum_x2: f32 = 0;
            for (xs) |v| sum_x2 += v * v;
            const av_x = @sqrt(sum_x2 / SUB_BLOCK);
            for (0..SUB_BLOCK) |l| w[l] = av_x + @abs(xs[l]);
            // q5_K: nmax=31, rmin=-0.5, nstep=15 (vs q4_K's 15, -1, 20)
            scales[j] = makeQkx2Quants(SUB_BLOCK, 31, xs, &w, L[j * SUB_BLOCK ..][0..SUB_BLOCK], &mins[j], &Laux, -0.5, 0.1, 15, false);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }

        const inv_scale: f32 = if (max_scale > 0) 63.0 / max_scale else 0.0;
        const inv_min: f32 = if (max_min > 0) 63.0 / max_min else 0.0;
        for (0..N_SUB) |j| {
            const ls: u8 = @min(63, @as(u8, @intCast(std.math.clamp(nearestInt(inv_scale * scales[j]), 0, 63))));
            const lm: u8 = @min(63, @as(u8, @intCast(std.math.clamp(nearestInt(inv_min * mins[j]), 0, 63))));
            if (j < 4) {
                blk.scales[j] = ls;
                blk.scales[j + 4] = lm;
            } else {
                blk.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                blk.scales[j - 4] |= ((ls >> 4) << 6);
                blk.scales[j] |= ((lm >> 4) << 6);
            }
        }
        blk.d = f32_to_f16(max_scale / 63.0);
        blk.dmin = f32_to_f16(max_min / 63.0);

        // Re-quantize with the final super-block scales (levels 0..31)
        var sc: u8 = undefined;
        var m: u8 = undefined;
        for (0..N_SUB) |j| {
            getScaleMinK4(j, &blk.scales, &sc, &m);
            const d = f16_to_f32(blk.d) * @as(f32, @floatFromInt(sc));
            if (d == 0) continue;
            const dm = f16_to_f32(blk.dmin) * @as(f32, @floatFromInt(m));
            for (0..SUB_BLOCK) |ii| {
                const l = nearestInt((x[j * SUB_BLOCK + ii] + dm) / d);
                L[j * SUB_BLOCK + ii] = @intCast(std.math.clamp(l, 0, 31));
            }
        }

        // Pack: low 4 bits into qs nibbles, bit 4 into qh. NOTE: qh is indexed
        // 0..32 directly (the C reference never advances the qh pointer) —
        // the 4 chunks of 64 weights reuse the same 32 qh bytes, separated by
        // the shifting m1/m2 bit masks.
        @memset(&blk.qh, 0);
        var m1: u8 = 1;
        var m2: u8 = 2;
        var ql_off: usize = 0;
        var n: usize = 0;
        while (n < QK_K) : (n += 64) {
            for (0..32) |j| {
                var l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16;
                    blk.qh[j] |= m1;
                }
                var l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16;
                    blk.qh[j] |= m2;
                }
                blk.qs[ql_off + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
            ql_off += 32;
        }
    }
}

/// ggml-quants.c:1731 dequantize_row_q5_K. k must be a multiple of 256.
pub fn dequantizeRowQ5_K(src: [*]const BlockQ5_K, dst: []f32, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var out: usize = 0;
    for (0..nb) |i| {
        const blk = &src[i];
        const d = f16_to_f32(blk.d);
        const min = f16_to_f32(blk.dmin);
        var is: usize = 0;
        var hi_mask1: u8 = 1;
        var hi_mask2: u8 = 2;
        var q_off: usize = 0;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            var sc: u8 = undefined;
            var m: u8 = undefined;
            getScaleMinK4(is, &blk.scales, &sc, &m);
            const d1 = d * @as(f32, @floatFromInt(sc));
            const m1 = min * @as(f32, @floatFromInt(m));
            getScaleMinK4(is + 1, &blk.scales, &sc, &m);
            const d2 = d * @as(f32, @floatFromInt(sc));
            const m2 = min * @as(f32, @floatFromInt(m));
            // qh is indexed 0..32 for every chunk (matches the C reference:
            // the qh pointer never advances; chunks are separated by the
            // shifting hi bit masks).
            for (0..32) |l| {
                const hi1: u8 = if ((blk.qh[l] & hi_mask1) != 0) 16 else 0;
                dst[out + l] = d1 * (@as(f32, @floatFromInt(blk.qs[q_off + l] & 0xF)) + @as(f32, @floatFromInt(hi1))) - m1;
            }
            for (0..32) |l| {
                const hi2: u8 = if ((blk.qh[l] & hi_mask2) != 0) 16 else 0;
                dst[out + l + 32] = d2 * (@as(f32, @floatFromInt(blk.qs[q_off + l] >> 4)) + @as(f32, @floatFromInt(hi2))) - m2;
            }
            // C advances y every chunk (*y++ in both inner loops)
            out += 64;
            q_off += 32;
            is += 2;
            hi_mask1 <<= 2;
            hi_mask2 <<= 2;
        }
    }
}

/// Dequantize Q5_K to BF16 (for BF16 GEMM tile paths).
pub fn dequantizeRowQ5_KToBF16(src: [*]const BlockQ5_K, dst: [*]amx.bf16, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowQ5_K(src + i, f[0..], QK_K);
        for (0..QK_K) |j| dst[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x Q5_K weights -> F32)
// ============================================================================

pub fn gemmQ5_KScalar(
    a: [*]const amx.bf16,
    lda: usize,
    b: [*]const BlockQ5_K,
    ldb: usize,
    c: [*]f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| c[i * ldc + j] = 0;
    }
    const kb = k / QK_K;
    var bq: [QK_K]f32 = undefined;
    for (0..n) |j| {
        const src = b + j * ldb;
        for (0..kb) |blk| {
            dequantizeRowQ5_K(src + blk, bq[0..], QK_K);
            for (0..m) |i| {
                var acc = c[i * ldc + j];
                const a_row = a + i * lda;
                for (0..QK_K) |t| {
                    acc += amx.bf16_to_f32(a_row[blk * QK_K + t]) * bq[t];
                }
                c[i * ldc + j] = acc;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "Q5_K block layout byte-exact (176 bytes)" {
    try testing.expectEqual(@as(usize, 176), @sizeOf(BlockQ5_K));
    const blk = std.mem.zeroes(BlockQ5_K);
    const bytes: [*]const u8 = @ptrCast(&blk);
    // d [0,2), dmin [2,4), scales [4,16), qh [16,48), qs [48,176)
    try testing.expectEqual(@as(u8, 0), bytes[0]);
    try testing.expectEqual(@as(u8, 0), bytes[3]);
    try testing.expectEqual(@as(u8, 0), bytes[4]);
    try testing.expectEqual(@as(u8, 0), bytes[15]);
    try testing.expectEqual(@as(u8, 0), bytes[16]);
    try testing.expectEqual(@as(u8, 0), bytes[47]);
    try testing.expectEqual(@as(u8, 0), bytes[48]);
    try testing.expectEqual(@as(u8, 0), bytes[175]);
}

test "Q5_K quantize/dequantize round trip accuracy" {
    const k = QK_K; // one super-block
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5 + 0.1 * @as(f32, @floatFromInt(i % 7));
    }

    var blk: [1]BlockQ5_K = undefined;
    quantizeRowQ5_K(&src, &blk, k);

    var dst: [k]f32 = undefined;
    dequantizeRowQ5_K(&blk, &dst, k);

    var max_abs_err: f32 = 0;
    var sum_abs_x: f32 = 0;
    for (0..k) |i| {
        max_abs_err = @max(max_abs_err, @abs(src[i] - dst[i]));
        sum_abs_x += @abs(src[i]);
    }
    const rel = max_abs_err / (sum_abs_x / k);
    // ~5.5 bits/weight: between Q4_K (15%) and Q6_K (10%); allow 12%
    try testing.expect(rel < 0.12);
}

test "Q5_K all-zero input dequantizes to zero" {
    const k = QK_K;
    var src: [k]f32 = [_]f32{0.0} ** k;
    var blk: [1]BlockQ5_K = undefined;
    quantizeRowQ5_K(&src, &blk, k);

    var dst: [k]f32 = undefined;
    dequantizeRowQ5_K(&blk, &dst, k);
    for (dst) |v| try testing.expectEqual(@as(f32, 0), v);
}

test "Q5_K high-bit plane (qh) exercised" {
    // Values spanning the full 0..31 level range must use the qh bit plane;
    // verify a quantize->dequantize round trip flips qh bits for large levels.
    const k = QK_K;
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @as(f32, @floatFromInt(i % 32)) * 0.03 - 0.2; // spans wide range
    }

    var blk: [1]BlockQ5_K = undefined;
    quantizeRowQ5_K(&src, &blk, k);

    var qh_nonzero: usize = 0;
    for (blk[0].qh) |v| {
        if (v != 0) qh_nonzero += 1;
    }
    // Wide-range data must set some high bits; all-zero qh would mean the
    // 5th bit was never used (i.e., only 4-bit levels were ever produced).
    try testing.expect(qh_nonzero > 0);
}

test "Q5_K scalar GEMM vs dequantized reference" {
    const M = 4;
    const N = 4;
    const K = QK_K;

    var a: [M * K]amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    var src: [K]f32 = undefined;
    for (&src) |*v| v.* = 1.0;
    var b: [N]BlockQ5_K = undefined;
    for (0..N) |j| quantizeRowQ5_K(&src, @ptrCast(&b[j]), K);

    var c: [M * N]f32 = undefined;
    gemmQ5_KScalar(&a, K, &b, 1, &c, N, M, N, K);

    var w: [K]f32 = undefined;
    dequantizeRowQ5_K(@ptrCast(&b[0]), &w, K);
    var expected: f32 = 0;
    for (w) |v| expected += v;
    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(expected, c[i * N + j], 1e-3);
        }
    }
    // Constant 1.0 row quantizes tightly
    try testing.expect(@abs(expected - 256.0) / 256.0 < 0.01);
}
