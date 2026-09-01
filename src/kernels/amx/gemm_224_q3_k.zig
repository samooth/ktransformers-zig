// GEMM Kernel 224 Q3_K (GGML K-quant)
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256):
//   block_q3_K {
//     uint8_t hmask[QK_K/8];  // 32 bytes: quants high bit
//     uint8_t qs[QK_K/4];     // 64 bytes: quants low 2 bits
//     uint8_t scales[12];     // 12 bytes: 6-bit-ish packed int8 scales
//     ggml_half d;            // super-block scale
//   } = 110 bytes per 256 weights (16 sub-blocks of 16)
//
// Weight reconstruction (ggml-quants.c:1305):
//   The 12 scale bytes unpack into 16 int8 scales via a bit-shuffle
//   (kmask1=0x03030303, kmask2=0x0f0f0f0f — see dequantizeRowQ3_K).
//   Per sub-block j: dl = d * (scales[j] - 32)
//   y = dl * ((qs 2-bit) - ((hmask bit set) ? 0 : 4))
//   — i.e. the "high bit" SUBTRACTS 4, inverting the usual +16 pattern.
//
// Quantize (:1229): make_q3_quants (nmax=4, do_rmse, 5-try coordinate
// descent) -> iscale = -32/max -> 6-bit scale packing across the 12 bytes
// -> re-quantize clamp [-4,3] +4 -> high-bit extraction into hmask ->
// 2-bit plane packing.
//
// This is the most layout-intricate of the landed formats (Q8_0, Q4_K,
// Q5_K, Q6_K, Q8_K, Q2_K) — the scale unpack alone is 6 bit-manipulation
// lines in the reference.

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// Q3_K Block Structure (byte-exact vs ggml's block_q3_K)
// ============================================================================

pub const QK_K: usize = 256;
const SUB_BLOCK: usize = 16;
const N_SUB: usize = QK_K / SUB_BLOCK; // = 16

pub const BlockQ3_K = extern struct {
    hmask: [QK_K / 8]u8, // 32 bytes: high bit per weight (bit m, byte j)
    qs: [QK_K / 4]u8, // 64 bytes: low 2 bits, 4 per byte
    scales: [12]u8, // packed 6-bit-ish int8 scales (16 values)
    d: u16, // f16 super-block scale
};

comptime {
    if (@sizeOf(BlockQ3_K) != 110) @compileError("BlockQ3_K must be 110 bytes");
}

// ============================================================================
// f16 <-> f32 (native f16)
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
// ggml helpers
// ============================================================================

pub fn nearestInt(fval: f32) i32 {
    std.debug.assert(@abs(fval) <= 4194303.0);
    const val: f32 = fval + 12582912.0;
    const i: i32 = @bitCast(val);
    return (i & 0x007f_ffff) - 0x0040_0000;
}

const GROUP_MAX_EPS: f32 = 1e-15;

/// ggml-quants.c:697 make_q3_quants — nmax=4, do_rmse=true, 5-try
/// coordinate-descent refinement. Returns scale; L gets l+nmax (0..7).
fn makeQ3Quants(n: usize, nmax: i32, x: []const f32, L: []i8, do_rmse: bool) f32 {
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
    const iscale = -@as(f32, @floatFromInt(nmax)) / max;
    if (do_rmse) {
        var sumlx: f32 = 0;
        var suml2: f32 = 0;
        for (0..n) |i| {
            const l = nearestInt(iscale * x[i]);
            const li = std.math.clamp(l, -nmax, nmax - 1);
            L[i] = @intCast(li);
            const w = x[i] * x[i];
            const lf: f32 = @floatFromInt(li);
            sumlx += w * x[i] * lf;
            suml2 += w * lf * lf;
        }
        var itry: usize = 0;
        while (itry < 5) : (itry += 1) {
            var n_changed: usize = 0;
            for (0..n) |i| {
                const w = x[i] * x[i];
                const lf: f32 = @floatFromInt(L[i]);
                var slx = sumlx - w * x[i] * lf;
                if (slx > 0) {
                    var sl2 = suml2 - w * lf * lf;
                    var new_l = nearestInt(x[i] * sl2 / slx);
                    new_l = std.math.clamp(new_l, -nmax, nmax - 1);
                    if (new_l != L[i]) {
                        const nf: f32 = @floatFromInt(new_l);
                        slx += w * x[i] * nf;
                        sl2 += w * nf * nf;
                        if (sl2 > 0 and slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = @intCast(new_l);
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if (n_changed == 0) break;
        }
        for (0..n) |i| L[i] += @intCast(nmax);
        return if (suml2 > 0.0) sumlx / suml2 else 0.0;
    }
    for (0..n) |i| {
        const l = nearestInt(iscale * x[i]);
        const li = std.math.clamp(l, -nmax, nmax - 1);
        L[i] = @intCast(li + nmax);
    }
    return 1.0 / iscale;
}

// ============================================================================
// Scale packing / unpacking (the intricate 12-byte scheme)
// ============================================================================

/// Pack 16 int8 scales (values already in the 0..63 representation) into
/// the 12-byte scales array, exactly as ggml-quants.c:1248-1261.
fn packScales(scales: *[12]u8, sc: []const i16) void {
    @memset(scales, 0);
    for (0..N_SUB) |j| {
        // l is the 0..63 representation: (nearest(-32/max*scale) +32) & 63
        var l: i16 = sc[j];
        l = std.math.clamp(l, 0, 63);
        if (j < 8) {
            scales[j] = @intCast(l & 0xF);
        } else {
            scales[j - 8] |= @intCast((l & 0xF) << 4);
        }
        l >>= 4;
        // scales[8 + j%4] |= (l << (2*(j/4))) — the 2-bit high part
        const shift: u3 = @intCast(2 * (j / 4));
        const jp4 = 8 + j % 4;
        scales[jp4] |= @intCast((l & 3) << shift);
    }
}

/// Unpack the 12-byte scales into 16 int8 values (the dequant side,
/// ggml-quants.c:1323-1328). Returns scales with the -32 offset NOT yet
/// applied (the caller does `scales[i] - 32`).
fn unpackScales(scales: *const [12]u8, out: *[N_SUB]i8) void {
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;
    var aux: [4]u32 = undefined;
    // memcpy 12 bytes into aux[0..2] (scales[8..11] fully occupies aux[2]'s
    // 32 bits — do NOT mask the top byte; scales[11] lives in bits 24..31)
    aux[0] = std.mem.readInt(u32, scales[0..4], .little);
    aux[1] = std.mem.readInt(u32, scales[4..8], .little);
    aux[2] = std.mem.readInt(u32, scales[8..12], .little);
    aux[3] = 0;
    const tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    // Reinterpret aux as 16 int8 scales
    const bytes: [*]const u8 = @ptrCast(&aux);
    for (0..N_SUB) |i| out[i] = @bitCast(bytes[i]);
}

// ============================================================================
// Quantize / Dequantize
// ============================================================================

/// ggml-quants.c:1229 quantize_row_q3_K_ref. k must be a multiple of 256.
pub fn quantizeRowQ3_K(src: []const f32, dst: [*]BlockQ3_K, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;

    var L: [QK_K]i8 = undefined;
    var scales_f: [N_SUB]f32 = undefined;

    for (0..nb) |i| {
        const x = src[i * QK_K ..][0..QK_K];
        const blk = &dst[i];

        var max_scale: f32 = 0;
        var amax: f32 = 0;
        for (0..N_SUB) |j| {
            scales_f[j] = makeQ3Quants(SUB_BLOCK, 4, x[j * SUB_BLOCK ..][0..SUB_BLOCK], L[j * SUB_BLOCK ..][0..SUB_BLOCK], true);
            const scale = @abs(scales_f[j]);
            if (scale > amax) {
                amax = scale;
                max_scale = scales_f[j];
            }
        }

        if (max_scale != 0) {
            const iscale = -32.0 / max_scale;
            // Pack: (nearest(iscale*scale) + 32) in 0..63
            var sc_packed: [N_SUB]i16 = undefined;
            for (0..N_SUB) |j| {
                const l = nearestInt(iscale * scales_f[j]);
                sc_packed[j] = @intCast(std.math.clamp(l, -32, 31) + 32);
            }
            packScales(&blk.scales, &sc_packed);
            blk.d = f32_to_f16(1.0 / iscale);
        } else {
            @memset(&blk.scales, 0);
            blk.d = f32_to_f16(0.0);
        }

        // Re-quantize with the final scales (levels [-4,3] + 4 = [0,7])
        var sc_unp: [N_SUB]i8 = undefined;
        unpackScales(&blk.scales, &sc_unp);
        for (0..N_SUB) |j| {
            const d = f16_to_f32(blk.d) * @as(f32, @floatFromInt(sc_unp[j] - 32));
            if (d == 0) continue;
            for (0..SUB_BLOCK) |ii| {
                const l = nearestInt(x[j * SUB_BLOCK + ii] / d);
                L[j * SUB_BLOCK + ii] = @intCast(std.math.clamp(l, -4, 3) + 4);
            }
        }

        // Extract high bits (L > 3) into hmask; 8 weights per bit position m
        @memset(&blk.hmask, 0);
        var m: usize = 0;
        var hm: u8 = 1;
        for (0..QK_K) |j| {
            if (L[j] > 3) {
                blk.hmask[m] |= hm;
                L[j] -= 4;
            }
            m += 1;
            if (m == QK_K / 8) {
                m = 0;
                hm <<= 1;
            }
        }

        // Pack 2-bit planes: 4 per byte
        var j: usize = 0;
        while (j < QK_K) : (j += 128) {
            for (0..32) |l| {
                const q: u8 = @as(u8, @intCast(L[j + l])) | (@as(u8, @intCast(L[j + l + 32])) << 2) | (@as(u8, @intCast(L[j + l + 64])) << 4) | (@as(u8, @intCast(L[j + l + 96])) << 6);
                blk.qs[j / 4 + l] = q;
            }
        }
    }
}

/// ggml-quants.c:1305 dequantize_row_q3_K. k must be a multiple of 256.
pub fn dequantizeRowQ3_K(src: [*]const BlockQ3_K, dst: []f32, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var out: usize = 0;
    for (0..nb) |i| {
        const blk = &src[i];
        const d_all = f16_to_f32(blk.d);

        var sc: [N_SUB]i8 = undefined;
        unpackScales(&blk.scales, &sc);

        var is: usize = 0;
        var q_off: usize = 0;
        var m: u8 = 1; // advances across ALL 8 planes (2 chunks x 4 planes) — C:1321
        var n: usize = 0;
        while (n < QK_K) : (n += 128) {
            var plane: usize = 0;
            while (plane < 4) : (plane += 1) {
                const shift: u3 = @intCast(plane * 2);
                // Sub-block A: q[0..16], Sub-block B: q[16..32]
                var dl = d_all * @as(f32, @floatFromInt(sc[is] - 32));
                is += 1;
                for (0..16) |l| {
                    const q_val: i32 = @as(i32, blk.qs[q_off + l] >> shift) & 3;
                    const hb: i32 = if ((blk.hmask[l] & m) != 0) 0 else 4;
                    dst[out + l] = dl * @as(f32, @floatFromInt(q_val - hb));
                }
                out += 16;

                dl = d_all * @as(f32, @floatFromInt(sc[is] - 32));
                is += 1;
                for (0..16) |l| {
                    const q_val: i32 = @as(i32, blk.qs[q_off + l + 16] >> shift) & 3;
                    const hb: i32 = if ((blk.hmask[l + 16] & m) != 0) 0 else 4;
                    dst[out + l] = dl * @as(f32, @floatFromInt(q_val - hb));
                }
                out += 16;
                m <<= 1;
            }
            q_off += 32;
        }
    }
}

/// Dequantize Q3_K to BF16.
pub fn dequantizeRowQ3_KToBF16(src: [*]const BlockQ3_K, dst: [*]amx.bf16, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowQ3_K(src + i, f[0..], QK_K);
        for (0..QK_K) |j| dst[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x Q3_K weights -> F32)
// ============================================================================

pub fn gemmQ3_KScalar(
    a: [*]const amx.bf16,
    lda: usize,
    b: [*]const BlockQ3_K,
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
            dequantizeRowQ3_K(src + blk, bq[0..], QK_K);
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

test "Q3_K block layout byte-exact (110 bytes)" {
    try testing.expectEqual(@as(usize, 110), @sizeOf(BlockQ3_K));
    const blk = std.mem.zeroes(BlockQ3_K);
    const bytes: [*]const u8 = @ptrCast(&blk);
    // hmask [0,32), qs [32,96), scales [96,108), d [108,110)
    try testing.expectEqual(@as(u8, 0), bytes[0]);
    try testing.expectEqual(@as(u8, 0), bytes[31]);
    try testing.expectEqual(@as(u8, 0), bytes[32]);
    try testing.expectEqual(@as(u8, 0), bytes[95]);
    try testing.expectEqual(@as(u8, 0), bytes[96]);
    try testing.expectEqual(@as(u8, 0), bytes[107]);
    try testing.expectEqual(@as(u8, 0), bytes[108]);
    try testing.expectEqual(@as(u8, 0), bytes[109]);
}

test "Q3_K scale pack/unpack round trip" {
    // Round-trip all 64 scale values through pack/unpack
    var packed_bytes: [12]u8 = undefined;
    var sc: [N_SUB]i16 = undefined;
    for (0..N_SUB) |j| sc[j] = @intCast(j * 4); // 0,4,8...60
    packScales(&packed_bytes, &sc);
    var unp: [N_SUB]i8 = undefined;
    unpackScales(&packed_bytes, &unp);
    for (0..N_SUB) |j| {
        try testing.expectEqual(@as(i8, @intCast(sc[j])), unp[j]);
    }
}

test "Q3_K quantize/dequantize round trip accuracy" {
    const k = QK_K;
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5 + 0.1 * @as(f32, @floatFromInt(i % 7));
    }

    var blk: [1]BlockQ3_K = undefined;
    quantizeRowQ3_K(&src, &blk, k);

    var dst: [k]f32 = undefined;
    dequantizeRowQ3_K(&blk, &dst, k);

    var max_abs_err: f32 = 0;
    var sum_abs_x: f32 = 0;
    for (0..k) |i| {
        max_abs_err = @max(max_abs_err, @abs(src[i] - dst[i]));
        sum_abs_x += @abs(src[i]);
    }
    const rel = max_abs_err / (sum_abs_x / k);
    // ~3.4375 bits/weight (incl. the high bit): between Q2_K (45%) and
    // Q4_K (15%); allow 20%.
    try testing.expect(rel < 0.20);
}

test "Q3_K all-zero input dequantizes to zero" {
    const k = QK_K;
    var src: [k]f32 = [_]f32{0.0} ** k;
    var blk: [1]BlockQ3_K = undefined;
    quantizeRowQ3_K(&src, &blk, k);
    try testing.expectEqual(@as(u16, 0), blk[0].d);

    var dst: [k]f32 = undefined;
    dequantizeRowQ3_K(&blk, &dst, k);
    for (dst) |v| try testing.expectEqual(@as(f32, 0), v);
}

test "Q3_K scalar GEMM vs dequantized reference" {
    const M = 4;
    const N = 4;
    const K = QK_K;

    var a: [M * K]amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    var src: [K]f32 = undefined;
    for (&src) |*v| v.* = 1.0;
    var b: [N]BlockQ3_K = undefined;
    for (0..N) |j| quantizeRowQ3_K(&src, @ptrCast(&b[j]), K);

    var c: [M * N]f32 = undefined;
    gemmQ3_KScalar(&a, K, &b, 1, &c, N, M, N, K);

    var w: [K]f32 = undefined;
    dequantizeRowQ3_K(@ptrCast(&b[0]), &w, K);
    var expected: f32 = 0;
    for (w) |v| expected += v;
    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(expected, c[i * N + j], 1.0);
        }
    }
}
