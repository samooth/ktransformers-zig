// GEMM Kernel 224 Q6_K (GGML K-quant) for AMX/AVX512
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256):
//   block_q6_K {
//     uint8_t ql[QK_K/2];      // 128 bytes: quants, lower 4 bits
//     uint8_t qh[QK_K/4];      //  64 bytes: quants, upper 2 bits
//     int8_t  scales[QK_K/16]; //  16 bytes: per-sub-block scales (8-bit)
//     ggml_half d;             // super-block scale
//   } = 210 bytes per 256 weights (16 sub-blocks of 16)
//
// Weight reconstruction (ggml-quants.c:1939 dequantize_row_q6_K):
//   q  = (ql nibble | qh 2-bit << 4) - 32
//   y  = d * scales[ib] * q            (ib = sub-block index)
// Quantization (:1869) uses make_qx_quants (rmse_type=1, 16 weights,
// nmax=32) + nearest_int. Both ported exactly below.
//
// Phase 1 of the GGML-quant workstream. Q8_0 and Q4_K have landed
// (gemm_224_q8_0.zig, gemm_224_q4_k.zig); Q5_K follows. C-API is Phase 2.

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// Q6_K Block Structure (byte-exact vs ggml's block_q6_K)
// ============================================================================

pub const QK_K: usize = 256;
const SUB: usize = 16; // weights per sub-block
const N_SUB: usize = QK_K / SUB; // = 16

pub const BlockQ6_K = extern struct {
    ql: [QK_K / 2]u8, // lower 4 bits of each 6-bit quant
    qh: [QK_K / 4]u8, // upper 2 bits of each 6-bit quant
    scales: [N_SUB]i8, // per-sub-block scales, 8-bit signed
    d: u16, // f16 super-block scale
};

comptime {
    if (@sizeOf(BlockQ6_K) != 210) @compileError("BlockQ6_K must be 210 bytes");
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

const GROUP_MAX_EPS: f32 = 1e-15;

/// ggml-quants.c:628 make_qx_quants — rmse_type=1 path (used by q6_K).
fn makeQxQuants(n: usize, nmax: i32, x: []const f32, L: []i8, rmse_type: i32, qw: ?[]const f32) f32 {
    var max: f32 = 0;
    var amax: f32 = 0;
    for (0..n) |i| {
        const ax = @abs(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < GROUP_MAX_EPS) {
        for (0..n) |i| L[i] = 0;
        return 0.0;
    }
    var iscale = -@as(f32, @floatFromInt(nmax)) / max;
    if (rmse_type == 0) {
        for (0..n) |i| {
            const l = nearestInt(iscale * x[i]);
            L[i] = @intCast(nmax + std.math.clamp(l, -nmax, nmax - 1));
        }
        return 1.0 / iscale;
    }
    var sumlx: f32 = 0;
    var suml2: f32 = 0;
    for (0..n) |i| {
        const l = nearestInt(iscale * x[i]);
        const li = std.math.clamp(l, -nmax, nmax - 1);
        L[i] = @intCast(li + nmax);
        const w: f32 = if (qw) |q| q[i] else if (rmse_type == 1) x[i] * x[i] else if (rmse_type == 2) 1.0 else if (rmse_type == 3) @abs(x[i]) else @sqrt(@abs(x[i]));
        sumlx += w * x[i] * @as(f32, @floatFromInt(li));
        suml2 += w * @as(f32, @floatFromInt(li)) * @as(f32, @floatFromInt(li));
    }
    var scale: f32 = if (suml2 != 0) sumlx / suml2 else 0.0;
    var best: f32 = scale * sumlx;
    var is: i32 = -9;
    while (is <= 9) : (is += 1) {
        if (is == 0) continue;
        iscale = -(@as(f32, @floatFromInt(nmax)) + 0.1 * @as(f32, @floatFromInt(is))) / max;
        sumlx = 0;
        suml2 = 0;
        for (0..n) |i| {
            const l = nearestInt(iscale * x[i]);
            const li = std.math.clamp(l, -nmax, nmax - 1);
            const w: f32 = if (qw) |q| q[i] else if (rmse_type == 1) x[i] * x[i] else if (rmse_type == 2) 1.0 else if (rmse_type == 3) @abs(x[i]) else @sqrt(@abs(x[i]));
            const lf: f32 = @floatFromInt(li);
            sumlx += w * x[i] * lf;
            suml2 += w * lf * lf;
        }
        if (suml2 > 0 and sumlx * sumlx > best * suml2) {
            for (0..n) |i| {
                const l = nearestInt(iscale * x[i]);
                L[i] = @intCast(nmax + std.math.clamp(l, -nmax, nmax - 1));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

// ============================================================================
// Quantize / Dequantize (byte-exact vs ggml-quants.c)
// ============================================================================

/// ggml-quants.c:1869 quantize_row_q6_K_ref. k must be a multiple of 256.
pub fn quantizeRowQ6_K(src: []const f32, dst: [*]BlockQ6_K, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;

    var L: [QK_K]i8 = undefined;
    var scales: [N_SUB]f32 = undefined;

    for (0..nb) |i| {
        const x = src[i * QK_K ..][0..QK_K];
        const blk = &dst[i];

        var max_scale: f32 = 0;
        var max_abs_scale: f32 = 0;
        for (0..N_SUB) |ib| {
            const scale = makeQxQuants(SUB, 32, x[ib * SUB ..][0..SUB], L[ib * SUB ..][0..SUB], 1, null);
            scales[ib] = scale;
            const abs_scale = @abs(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            // All-zero super-block: zero everything, d = 0
            blk.* = std.mem.zeroes(BlockQ6_K);
            blk.d = f32_to_f16(0.0);
            continue;
        }

        const iscale = -128.0 / max_scale;
        blk.d = f32_to_f16(1.0 / iscale);
        for (0..N_SUB) |ib| {
            // ggml: MIN(127, nearest_int(iscale*scales[ib])) stored in int8.
            // |iscale*scales[ib]| <= 128 by construction (max_scale is the
            // max |scale|), so -128..127; clamp both sides for float safety.
            const qi = std.math.clamp(nearestInt(iscale * scales[ib]), -128, 127);
            blk.scales[ib] = @intCast(qi);
        }

        // Re-quantize with the final super-block scale
        for (0..N_SUB) |j| {
            const d = f16_to_f32(blk.d) * @as(f32, @floatFromInt(blk.scales[j]));
            if (d == 0) continue;
            for (0..SUB) |ii| {
                var l = nearestInt(x[j * SUB + ii] / d);
                l = std.math.clamp(l, -32, 31);
                L[j * SUB + ii] = @intCast(l + 32);
            }
        }

        // Pack 6-bit quants: ql = low nibbles, qh = 2-bit high parts
        var ql_off: usize = 0;
        var qh_off: usize = 0;
        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                // L entries are 0..63 (l+32 offset already applied); the C
                // code packs them as uint8_t — cast before shifting to match
                // (i8 >> in Zig is arithmetic and 3<<6 overflows i8).
                const q1: u8 = @as(u8, @intCast(L[j + l])) & 0xF;
                const q2: u8 = @as(u8, @intCast(L[j + l + 32])) & 0xF;
                const q3: u8 = @as(u8, @intCast(L[j + l + 64])) & 0xF;
                const q4: u8 = @as(u8, @intCast(L[j + l + 96])) & 0xF;
                blk.ql[ql_off + l] = q1 | (q3 << 4);
                blk.ql[ql_off + l + 32] = q2 | (q4 << 4);
                blk.qh[qh_off + l] = (@as(u8, @intCast(L[j + l])) >> 4) | ((@as(u8, @intCast(L[j + l + 32])) >> 4) << 2) | ((@as(u8, @intCast(L[j + l + 64])) >> 4) << 4) | ((@as(u8, @intCast(L[j + l + 96])) >> 4) << 6);
            }
            ql_off += 64;
            qh_off += 32;
        }
    }
}

/// ggml-quants.c:1939 dequantize_row_q6_K. k must be a multiple of 256.
pub fn dequantizeRowQ6_K(src: [*]const BlockQ6_K, dst: []f32, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var out: usize = 0;
    for (0..nb) |i| {
        const blk = &src[i];
        const d = f16_to_f32(blk.d);
        var ql_off: usize = 0;
        var qh_off: usize = 0;
        var sc_off: usize = 0;
        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                const is = l / 16;
                const q1: i32 = @as(i32, blk.ql[ql_off + l] & 0xF) | ((@as(i32, blk.qh[qh_off + l] >> 0) & 3) << 4);
                const q2: i32 = @as(i32, blk.ql[ql_off + l + 32] & 0xF) | ((@as(i32, blk.qh[qh_off + l] >> 2) & 3) << 4);
                const q3: i32 = @as(i32, blk.ql[ql_off + l] >> 4) | ((@as(i32, blk.qh[qh_off + l] >> 4) & 3) << 4);
                const q4: i32 = @as(i32, blk.ql[ql_off + l + 32] >> 4) | ((@as(i32, blk.qh[qh_off + l] >> 6) & 3) << 4);
                dst[out + l] = d * @as(f32, @floatFromInt(blk.scales[sc_off + is + 0])) * @as(f32, @floatFromInt(q1 - 32));
                dst[out + l + 32] = d * @as(f32, @floatFromInt(blk.scales[sc_off + is + 2])) * @as(f32, @floatFromInt(q2 - 32));
                dst[out + l + 64] = d * @as(f32, @floatFromInt(blk.scales[sc_off + is + 4])) * @as(f32, @floatFromInt(q3 - 32));
                dst[out + l + 96] = d * @as(f32, @floatFromInt(blk.scales[sc_off + is + 6])) * @as(f32, @floatFromInt(q4 - 32));
            }
            out += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

/// Dequantize Q6_K to BF16 (for BF16 GEMM tile paths).
pub fn dequantizeRowQ6_KToBF16(src: [*]const BlockQ6_K, dst: [*]amx.bf16, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowQ6_K(src + i, f[0..], QK_K);
        for (0..QK_K) |j| dst[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x Q6_K weights -> F32)
// ============================================================================

pub fn gemmQ6_KScalar(
    a: [*]const amx.bf16,
    lda: usize,
    b: [*]const BlockQ6_K,
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
            dequantizeRowQ6_K(src + blk, bq[0..], QK_K);
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

test "Q6_K block layout byte-exact (210 bytes)" {
    try testing.expectEqual(@as(usize, 210), @sizeOf(BlockQ6_K));
    const blk = std.mem.zeroes(BlockQ6_K);
    const bytes: [*]const u8 = @ptrCast(&blk);
    // ql at 0 (128 bytes), qh at 128 (64 bytes), scales at 192 (16), d at 208 (2)
    try testing.expectEqual(@as(u8, 0), bytes[0]);
    try testing.expectEqual(@as(u8, 0), bytes[127]);
    try testing.expectEqual(@as(u8, 0), bytes[128]);
    try testing.expectEqual(@as(u8, 0), bytes[191]);
    try testing.expectEqual(@as(u8, 0), bytes[192]);
    try testing.expectEqual(@as(u8, 0), bytes[207]);
    try testing.expectEqual(@as(u8, 0), bytes[208]);
    try testing.expectEqual(@as(u8, 0), bytes[209]);
}

test "Q6_K quantize/dequantize round trip accuracy" {
    const k = QK_K; // one super-block
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5 + 0.1 * @as(f32, @floatFromInt(i % 7));
    }

    var blk: [1]BlockQ6_K = undefined;
    quantizeRowQ6_K(&src, &blk, k);

    var dst: [k]f32 = undefined;
    dequantizeRowQ6_K(&blk, &dst, k);

    var max_abs_err: f32 = 0;
    var sum_abs_x: f32 = 0;
    for (0..k) |i| {
        max_abs_err = @max(max_abs_err, @abs(src[i] - dst[i]));
        sum_abs_x += @abs(src[i]);
    }
    const rel = max_abs_err / (sum_abs_x / k);
    // Q6_K is ~6.5 bits/weight: error should be well under Q4_K's; allow 10%
    try testing.expect(rel < 0.10);
}

test "Q6_K all-zero super-block zeros d" {
    const k = QK_K;
    var src: [k]f32 = [_]f32{0.0} ** k;
    var blk: [1]BlockQ6_K = undefined;
    quantizeRowQ6_K(&src, &blk, k);
    try testing.expectEqual(@as(u16, 0), blk[0].d);

    var dst: [k]f32 = undefined;
    dequantizeRowQ6_K(&blk, &dst, k);
    for (dst) |v| try testing.expectEqual(@as(f32, 0), v);
}

test "Q6_K scalar GEMM constant weights" {
    const M = 4;
    const N = 4;
    const K = QK_K;

    var a: [M * K]amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    // Build weights that dequantize to exactly 1.0:
    // y = d * sc * q  with q = 1 => q-32 = ... choose d=1, sc=1, q=1
    // q (6-bit stored) = 1 => (q - 32) = -31 ... instead pick the value that
    // gives y = 1: d=1, sc=1, quantized q with (q-32)=1 => stored = 33.
    var b: [N]BlockQ6_K = undefined;
    for (&b) |*blk| {
        blk.* = std.mem.zeroes(BlockQ6_K);
        blk.d = f32_to_f16(1.0);
        for (&blk.scales) |*s| s.* = 1; // sc = 1 for all 16 sub-blocks
        // stored quant 33 = 0b100001: low nibble 1, high 2 bits 2
        // ql low nibble = 1; qh 2-bit field = 2 for all four slots
        for (0..32) |l| {
            blk.ql[l] = 1; // q1/q3 low nibble
            blk.ql[l + 32] = 1; // q2/q4 low nibble
            // qh byte: bits 0-1,2-3,4-5,6-7 each = 2
            blk.qh[l] = (2 << 0) | (2 << 2) | (2 << 4) | (2 << 6);
        }
    }

    var c: [M * N]f32 = undefined;
    gemmQ6_KScalar(&a, K, &b, 1, &c, N, M, N, K);

    for (0..M) |i| {
        for (0..N) |j| {
            // All 256 weights = 1.0 => C = 256
            try testing.expectApproxEqAbs(@as(f32, 256.0), c[i * N + j], 0.5);
        }
    }
}
