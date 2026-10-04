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
const Dq6Vec: type = @Vector(8, f32);

/// 16 weights' worth of 6-bit Q6_K values -> two `Dq6Vec`.
///
/// A Q6_K value is a 4-bit nibble from `ql` plus 2 bits from `qh` shifted up by
/// 4, minus 32. `nib` selects the low (0) or high (4) nibble plane; `hshift`
/// selects which 2-bit field of `qh` (0, 2, 4 or 6).
///
/// The scalar body's OR is really an ADD: the nibble is 4 bits and `h2 << 4`
/// starts at bit 4, so the fields never overlap. Written as an add so the vector
/// form is a plain widen + shift + add.
inline fn q6_16(ql: *const u8, qh: *const u8, comptime nib: u4, comptime hshift: u3) [2]Dq6Vec {
    const qb: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(ql)).*;
    const hb: @Vector(16, u8) = @as(*align(1) const @Vector(16, u8), @ptrCast(qh)).*;
    const lo: @Vector(16, u8) = if (nib == 0)
        qb & @as(@Vector(16, u8), @splat(0x0F))
    else
        qb >> @as(@Vector(16, u4), @splat(nib));
    const h2: @Vector(16, u8) = (hb >> @as(@Vector(16, u4), @splat(hshift))) & @as(@Vector(16, u8), @splat(3));
    const q6: @Vector(16, i32) = @as(@Vector(16, i32), @intCast(lo)) +
        (@as(@Vector(16, i32), @intCast(h2)) << @as(@Vector(16, u5), @splat(4)));
    const f: @Vector(16, f32) = @floatFromInt(q6 - @as(@Vector(16, i32), @splat(32)));
    // Stage through an array: `f[0..8]` on a @Vector is not addressable.
    const fa: [16]f32 = f;
    return .{ fa[0..8].*, fa[8..16].* };
}

/// AVX2 Q6_K dequantize-to-F32. Bit-exact with `dequantizeRowQ6_K`.
///
/// The natural vector unit is 16 weights, not 8: each 16-value group has ONE
/// scale, and in the scalar body's `l` loop that scale steps at `l == 16`
/// (`is = l / 16`). Four ql planes x two halves = 8 units per 128-weight chunk.
///
/// THE TRAP, which cost a wrong answer first time: the **ql offset depends on
/// the plane, not just the half**. Planes 1 and 3 read `ql[l + 32]`; planes 0 and
/// 2 read `ql[l]`. qh is indexed by `l` for EVERY plane (only the 2-bit field
/// moves), so its offset does NOT get the +32. Getting that wrong yields
/// plausible magnitudes — it reads a valid neighbouring nibble — and the
/// bit-exact test is what catches it.
pub fn dequantizeRowQ6_KVec(src: [*]const BlockQ6_K, dst: []f32, k: usize) void {
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
            inline for (0..4) |plane| { // ql nibble plane x qh 2-bit field
                inline for (0..2) |half| { // which 16 values of the 32-long l loop
                    const qp = ql_off + half * 16 + (if (plane % 2 == 1) @as(usize, 32) else 0);
                    const hp = qh_off + half * 16;
                    const sv: Dq6Vec = @splat(d * @as(f32, @floatFromInt(blk.scales[sc_off + half + plane * 2])));
                    const pair = q6_16(&blk.ql[qp], &blk.qh[hp], if (plane < 2) 0 else 4, plane * 2);
                    dst[out + plane * 32 + half * 16 + 0 ..][0..8].* = sv * pair[0];
                    dst[out + plane * 32 + half * 16 + 8 ..][0..8].* = sv * pair[1];
                }
            }
            out += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

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
// AVX2 GEMM (BF16 activations x Q6_K weights -> F32)
// ============================================================================
//
// Same math as `gemmQ6_KScalar` above (kept as the readable statement of the
// math the tests pin against `dequantizeRowQ6_K`), with the 256-element inner
// loop vectorized. Q6_K is the second most common GGUF format after Q4_K, so
// this is the largest single user-visible win in the K-quant family: see
// bench/kquant_bench.zig for the measured numbers (docs/perf-roadmap.md P0.1).
//
// The driver and the shared vector primitives live in kquant_simd.zig; this
// file supplies the bit layout. The mapping below was derived from
// `dequantizeRowQ6_K` and is the part that has to be right:
//
//   The dequantizer walks 2 chunks of 128 weights. Inside a chunk, index `l`
//   in 0..32 produces FOUR 32-value runs, in weight order: `l`, `l+32`,
//   `l+64`, `l+96`, using (ql nibble, qh 2-bit field, scale) triples
//   (low, shift 0), (low, shift 2), (high, shift 4), (high, shift 6) and
//   scales[sc_off + l/16 + 0, +2, +4, +6]. Writing the weight index inside the
//   chunk as `w = l + 32*q`, the 16-weight group index is `gc = w/16 = l/16 +
//   2q`, i.e. `is = gc & 1` and `q = gc >> 1` — which is exactly the pair
//   (`plane`, scale offset) the dequantizer uses, and the scale index collapses
//   to `sc_off + gc`. So group `g` of the super-block (0..15) reads 16 bytes of
//   `ql` and 16 bytes of `qh` — 32 lanes' worth — with NO lane shuffling:
//       l        = (gc & 1)*16 + lane       // which 16-byte run
//       plane    = gc >> 1                 // 0..3: qh 2-bit field, and the
//                                           // nibble is high iff plane >= 2
//       ql byte  = chunk*64 + l + 32*(plane & 1)
//       qh byte  = chunk*32 + l
//   NOTE the 32-byte ql stride is selected by `plane`, not by the run index —
//   getting that wrong was the one bug the equivalence test caught (see
//   LESSONS_ZIG.md §9).

const ksimd = @import("kquant_simd.zig");

const SimdCfg = struct {
    pub const Block = BlockQ6_K;
    pub const QK: usize = QK_K;
    /// One i8 scale per 16 weights.
    pub const GROUP: usize = SUB;
    /// `d * scales[i]` for all 16 sub-blocks, hoisted out of the inner loop
    /// (see kquant_simd.zig note 4).
    pub const Scales = [N_SUB]f32;

    pub inline fn prepare(blk: *const BlockQ6_K) Scales {
        const d = f16_to_f32(blk.d);
        var t: Scales = undefined;
        for (0..N_SUB) |i| t[i] = d * @as(f32, @floatFromInt(blk.scales[i]));
        return t;
    }

    /// `y = (d * scales[g]) * (q - 32)` for 16 consecutive weights — the same
    /// association order as the scalar dequantizer (`(d*sc) * (q-32)`), so the
    /// unpacked values are bit-identical to `dequantizeRowQ6_K`'s.
    pub inline fn unpackVec(blk: *const BlockQ6_K, sc: *const Scales, comptime g: usize, comptime v: usize) ksimd.Vec {
        std.debug.assert(v == 0); // GROUP == VW: exactly one vector per group
        const chunk = g >> 3; // 0..1 (the dequantizer's 128-weight chunk)
        const gc = g & 7;
        const l_hi = gc & 1; // 0/1 -> which 16-byte run inside the 32-wide pair
        const plane = gc >> 1; // 0..3 -> which 2-bit field of `qh`
        const ql_off = chunk * 64 + l_hi * 16 + (plane & 1) * 32 + v * ksimd.VW;
        const qh_off = chunk * 32 + l_hi * 16 + v * ksimd.VW;

        const qh = ksimd.loadU8(&blk.qh[qh_off]);
        // Immediate shifts (a switch, not a runtime shift) so this lowers to
        // vpsrlw/vpsllw rather than the per-lane vpsrlvd.
        const hi2: ksimd.VecU8 = switch (plane) {
            0 => qh & ksimd.mask3(),
            1 => (qh >> ksimd.shr8(2)) & ksimd.mask3(),
            2 => (qh >> ksimd.shr8(4)) & ksimd.mask3(),
            else => (qh >> ksimd.shr8(6)) & ksimd.mask3(),
        };
        const ql = ksimd.loadU8(&blk.ql[ql_off]);
        const lo4: ksimd.VecU8 = if (plane < 2) ql & ksimd.mask4() else ql >> ksimd.shr8(4);

        const q = lo4 | (hi2 << ksimd.shl8(4));
        return (ksimd.u8ToF32(q) - ksimd.splatF(32.0)) * ksimd.splatF(sc[g]);
    }
};

/// AVX2 Q6_K GEMM. Drop-in replacement for `gemmQ6_KScalar` (same signature,
/// same result to f32 rounding).
pub fn gemmQ6_K(
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
    ksimd.gemmBF16xQuant(SimdCfg, a, lda, b, ldb, c, ldc, m, n, k);
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
