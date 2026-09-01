// GEMM Kernel 224 Q2_K (GGML K-quant) for NEON/AVX2 fallback
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256):
//   block_q2_K {
//     uint8_t scales[QK_K/16];  // 16 bytes: 4-bit scale + 4-bit min pairs
//     uint8_t qs[QK_K/4];       // 64 bytes: 2-bit quants
//     ggml_half d, dmin;        // super-block scales
//   } = 84 bytes per 256 weights (16 sub-blocks of 16)
//
// Weight reconstruction (ggml-quants.c:961 dequantize_row_q2_K):
//   per sub-block j: sc = scales[j] (4-bit scale lo nibble, 4-bit min hi)
//   dl = d * (sc & 0xF); ml = dmin * (sc >> 4)
//   y = dl * ((qs 2-bit) & 3) - ml, with the qs 2-bit planes advancing
//   per chunk of 128 (shift 0,2,4,6 within; 32-byte groups)
//
// Quantize (:891): per-16 make_qkx2_quants(nmax=3, rmin=-0.5, nstep=15,
// use_mad=TRUE — the q2_K-specific params), 4-bit scale/min packing with
// q4scale=15, final re-quantize clamp [0,3], 2-bit plane packing.
//
// Note the asymmetric scale/min packing: scales[j] is a SINGLE byte holding
// scale (lo nibble) AND min (hi nibble) — unlike Q4_K/Q5_K's 6-bit scheme.
//
// Follows the 5 landed formats (Q8_0, Q4_K, Q5_K, Q6_K, Q8_K).

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// Q2_K Block Structure (byte-exact vs ggml's block_q2_K)
// ============================================================================

pub const QK_K: usize = 256;
const SUB_BLOCK: usize = 16; // weights per sub-block
const N_SUB: usize = QK_K / SUB_BLOCK; // = 16

pub const BlockQ2_K = extern struct {
    scales: [N_SUB]u8, // 4-bit (scale, min) pairs — ONE byte per sub-block
    qs: [QK_K / 4]u8, // 2-bit quants, 4 per byte
    d: u16, // f16: super-block scale for quantized scales
    dmin: u16, // f16: super-block scale for quantized mins
};

comptime {
    if (@sizeOf(BlockQ2_K) != 84) @compileError("BlockQ2_K must be 84 bytes");
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

/// ggml-quants.c:799 make_qkx2_quants — q2_K calls it with
/// (n=16, nmax=3, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=TRUE).
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

/// ggml-quants.c:891 quantize_row_q2_K_ref. k must be a multiple of 256.
pub fn quantizeRowQ2_K(src: []const f32, dst: [*]BlockQ2_K, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    const q4scale: f32 = 15.0;

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
            // q2_K weights are |x| (fabs), and use_mad=true
            for (0..SUB_BLOCK) |l| w[l] = @abs(xs[l]);
            scales[j] = makeQkx2Quants(SUB_BLOCK, 3, xs, &w, L[j * SUB_BLOCK ..][0..SUB_BLOCK], &mins[j], &Laux, -0.5, 0.1, 15, true);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }

        if (max_scale > 0) {
            const iscale = q4scale / max_scale;
            for (0..N_SUB) |j| {
                const l = nearestInt(iscale * scales[j]);
                blk.scales[j] = @intCast(std.math.clamp(l, 0, 255));
            }
            blk.d = f32_to_f16(max_scale / q4scale);
        } else {
            for (0..N_SUB) |j| blk.scales[j] = 0;
            blk.d = f32_to_f16(0.0);
        }
        if (max_min > 0) {
            const iscale_min = q4scale / max_min;
            for (0..N_SUB) |j| {
                const l = nearestInt(iscale_min * mins[j]);
                blk.scales[j] |= @intCast((std.math.clamp(l, 0, 15)) << 4);
            }
            blk.dmin = f32_to_f16(max_min / q4scale);
        } else {
            blk.dmin = f32_to_f16(0.0);
        }

        // Re-quantize with the final super-block scales (levels 0..3)
        for (0..N_SUB) |j| {
            const d = f16_to_f32(blk.d) * @as(f32, @floatFromInt(blk.scales[j] & 0xF));
            if (d == 0) continue;
            const dm = f16_to_f32(blk.dmin) * @as(f32, @floatFromInt(blk.scales[j] >> 4));
            for (0..SUB_BLOCK) |ii| {
                const l = nearestInt((x[j * SUB_BLOCK + ii] + dm) / d);
                L[j * SUB_BLOCK + ii] = @intCast(std.math.clamp(l, 0, 3));
            }
        }

        // Pack 2-bit quants: 4 per byte, 32-byte groups per 128 weights
        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                blk.qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }
    }
}

/// ggml-quants.c:961 dequantize_row_q2_K. k must be a multiple of 256.
pub fn dequantizeRowQ2_K(src: [*]const BlockQ2_K, dst: []f32, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var out: usize = 0;
    for (0..nb) |i| {
        const blk = &src[i];
        const d = f16_to_f32(blk.d);
        const min = f16_to_f32(blk.dmin);

        var is: usize = 0;
        var q_off: usize = 0;
        var n: usize = 0;
        while (n < QK_K) : (n += 128) {
            // 4 shift planes (0,2,4,6) x 2 sub-blocks each = 8 scale reads
            // per 128-chunk; use a usize counter and cast to u3 at the use
            // site (a u3 loop counter wraps at 8 and re-enters).
            var plane: usize = 0;
            while (plane < 4) : (plane += 1) {
                const shift: u3 = @intCast(plane * 2);
                // sub-block A: qs[0..16], sub-block B: qs[16..32]
                var sc = blk.scales[is];
                is += 1;
                const dl_a = d * @as(f32, @floatFromInt(sc & 0xF));
                const ml_a = min * @as(f32, @floatFromInt(sc >> 4));
                for (0..16) |l| {
                    dst[out + l] = dl_a * @as(f32, @floatFromInt((blk.qs[q_off + l] >> shift) & 3)) - ml_a;
                }
                out += 16;

                sc = blk.scales[is];
                is += 1;
                const dl_b = d * @as(f32, @floatFromInt(sc & 0xF));
                const ml_b = min * @as(f32, @floatFromInt(sc >> 4));
                for (0..16) |l| {
                    dst[out + l] = dl_b * @as(f32, @floatFromInt((blk.qs[q_off + l + 16] >> shift) & 3)) - ml_b;
                }
                out += 16;
            }
            q_off += 32;
        }
    }
}

/// Dequantize Q2_K to BF16 (for BF16 GEMM tile paths).
pub fn dequantizeRowQ2_KToBF16(src: [*]const BlockQ2_K, dst: [*]amx.bf16, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowQ2_K(src + i, f[0..], QK_K);
        for (0..QK_K) |j| dst[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x Q2_K weights -> F32)
// ============================================================================

pub fn gemmQ2_KScalar(
    a: [*]const amx.bf16,
    lda: usize,
    b: [*]const BlockQ2_K,
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
            dequantizeRowQ2_K(src + blk, bq[0..], QK_K);
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

test "Q2_K block layout byte-exact (84 bytes)" {
    try testing.expectEqual(@as(usize, 84), @sizeOf(BlockQ2_K));
    const blk = std.mem.zeroes(BlockQ2_K);
    const bytes: [*]const u8 = @ptrCast(&blk);
    // scales [0,16), qs [16,80), d [80,82), dmin [82,84)
    try testing.expectEqual(@as(u8, 0), bytes[0]);
    try testing.expectEqual(@as(u8, 0), bytes[15]);
    try testing.expectEqual(@as(u8, 0), bytes[16]);
    try testing.expectEqual(@as(u8, 0), bytes[79]);
    try testing.expectEqual(@as(u8, 0), bytes[80]);
    try testing.expectEqual(@as(u8, 0), bytes[81]);
    try testing.expectEqual(@as(u8, 0), bytes[82]);
    try testing.expectEqual(@as(u8, 0), bytes[83]);
}

test "Q2_K quantize/dequantize round trip accuracy" {
    const k = QK_K;
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5 + 0.1 * @as(f32, @floatFromInt(i % 7));
    }

    var blk: [1]BlockQ2_K = undefined;
    quantizeRowQ2_K(&src, &blk, k);

    var dst: [k]f32 = undefined;
    dequantizeRowQ2_K(&blk, &dst, k);

    var max_abs_err: f32 = 0;
    var sum_abs_x: f32 = 0;
    for (0..k) |i| {
        max_abs_err = @max(max_abs_err, @abs(src[i] - dst[i]));
        sum_abs_x += @abs(src[i]);
    }
    const rel = max_abs_err / (sum_abs_x / k);
    // ~2.625 bits/weight: the loosest format; allow 25%
    try testing.expect(rel < 0.25);
}

test "Q2_K all-zero input dequantizes to zero" {
    const k = QK_K;
    var src: [k]f32 = [_]f32{0.0} ** k;
    var blk: [1]BlockQ2_K = undefined;
    quantizeRowQ2_K(&src, &blk, k);
    try testing.expectEqual(@as(u16, 0), blk[0].d);

    var dst: [k]f32 = undefined;
    dequantizeRowQ2_K(&blk, &dst, k);
    for (dst) |v| try testing.expectEqual(@as(f32, 0), v);
}

test "Q2_K scalar GEMM vs dequantized reference" {
    const M = 4;
    const N = 4;
    const K = QK_K;

    var a: [M * K]amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    var src: [K]f32 = undefined;
    for (&src) |*v| v.* = 1.0;
    var b: [N]BlockQ2_K = undefined;
    for (0..N) |j| quantizeRowQ2_K(&src, @ptrCast(&b[j]), K);

    var c: [M * N]f32 = undefined;
    gemmQ2_KScalar(&a, K, &b, 1, &c, N, M, N, K);

    var w: [K]f32 = undefined;
    dequantizeRowQ2_K(@ptrCast(&b[0]), &w, K);
    var expected: f32 = 0;
    for (w) |v| expected += v;
    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(expected, c[i * N + j], 1.5);
        }
    }
}
