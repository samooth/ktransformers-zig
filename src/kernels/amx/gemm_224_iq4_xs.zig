// GEMM Kernel 224 IQ4_XS (GGML non-linear 4-bit i-quant)
//
// Canonical GGUF layout (llama.cpp ggml-common.h):
//   block_iq4_xs {
//     ggml_half d;                          // super-block scale (f16)
//     uint16_t scales_h;                    // 8 × 2-bit high halves of scales
//     uint8_t  scales_l[QK_K/64];           // 8 × 4-bit low halves (packed 2/byte)
//     uint8_t  qs[QK_K/2];                  // 4-bit quants (128 bytes)
//   } = 2 + 2 + 4 + 128 = 136 bytes per 256 weights
//
// Weight reconstruction (ggml-quants.c:2743 dequantize_row_iq4_xs):
//   For each sub-block ib of 32 weights (8 sub-blocks total):
//     ls = (scales_l[ib/2] >> 4*(ib%2)) & 0xf              // low 4 bits
//        | ((scales_h >> 2*ib) & 3) << 4                  // high 2 bits
//     ls is a 6-bit signed offset in [0..63] (stored as 0..63, dl = d * (ls-32))
//     dl = d * (ls - 32)
//     Each 4-bit nibble from qs indexes kvalues_iq4nl[] (the non-linear
//     table). For each 16-byte half of qs, the 4-bit pairs (lo, hi)
//     produce 32 weights: y[0..15] = dl*table[qs_lo], y[16..31] = dl*table[qs_hi].
//
// kvalues_iq4nl (16 i8, byte-exact vs llama.cpp):
//   -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
// Designed for non-uniform data: most mass near zero, large values for outliers.
//
// Quantization (ggml-quants.c:4966 quantize_row_iq4_nl_impl, ntry=7 for IQ4_XS):
//   1. Compute sigma2 = 2*sum(x^2)/QK_K
//   2. For each sub-block of 32 weights:
//      - Compute per-element weight[j] (x^2, or quant_weights[j]*sqrt(sigma2+x^2))
//      - Find amax, derive d_guess = -max/values[0]  (i.e. d*table[0] = -max, anchoring)
//      - Initial pass: quantize x/d_guess via best_index_int8(16, table, ...)
//      - Solve for d via weighted least-squares (sumqx, sumq2)
//      - Local search ±ntry around d_guess to reduce quantization error
//   3. After all sub-blocks, pack scales to 6 bits via max_scale/32 reference:
//      scales_l[ib/2] packs two 4-bit lows, scales_h[ib/8] packs 8 2-bit highs
//   4. Encode the chosen L values (table index) into qs (4-bit nibble pairs).
//
// Ported 1:1 from llama.cpp ggml-quants.c (commit-era stable; matches the
// byte layout of all 5+ IQ4_XS models available in /ai/models).

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// Block Structure (byte-exact vs ggml's block_iq4_xs)
// ============================================================================

pub const QK_K: usize = 256;
const SUB_BLOCK: usize = 32;
const N_SUB: usize = QK_K / SUB_BLOCK; // 8

pub const BlockIQ4_XS = extern struct {
    d: u16, // f16: super-block scale
    scales_h: u16, // 8 × 2-bit high halves of the 6-bit per-sub-block scales
    scales_l: [QK_K / 64]u8, // 4-bit low halves (2 sub-block scales per byte)
    qs: [QK_K / 2]u8, // 4-bit quants (128 bytes, 256 nibbles)
};

comptime {
    if (@sizeOf(BlockIQ4_XS) != 136) @compileError("BlockIQ4_XS must be 136 bytes");
}

// ============================================================================
// f16 <-> f32 (native f16, same approach as Q4_K / Q8_0 / Q6_K)
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
// kvalues_iq4nl — the non-linear 4-bit quant lookup (16 i8 values)
// ============================================================================

pub const KVALUES_IQ4NL: [16]i8 = .{
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

comptime {
    // Verify the table matches llama.cpp (defensive: a typo here would
    // silently produce wrong dequantization on real GGUF weights).
    const expected: [16]i8 = .{
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };
    if (!std.meta.eql(KVALUES_IQ4NL, expected)) @compileError("KVALUES_IQ4NL drift");
}

// ============================================================================
// best_index_int8 — binary search for the nearest table entry
// ============================================================================

fn bestIndexInt8(n: usize, values: *const [16]i8, x: f32) u8 {
    // Matches ggml-quants.c:28 exactly. The values array is sorted (the
    // iq4nl table is strictly increasing), so binary search is correct.
    if (x <= @as(f32, @floatFromInt(values[0]))) return 0;
    if (x >= @as(f32, @floatFromInt(values[n - 1]))) return @intCast(n - 1);
    var ml: usize = 0;
    var mu: usize = n - 1;
    while (mu - ml > 1) {
        const mav = (ml + mu) / 2;
        if (x < @as(f32, @floatFromInt(values[mav]))) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    const v_mu_1 = @as(f32, @floatFromInt(values[mu - 1]));
    const v_mu = @as(f32, @floatFromInt(values[mu]));
    return @intCast(if (x - v_mu_1 < v_mu - x) mu - 1 else mu);
}

// ============================================================================
// nearest_int — the standard float-to-int quantization step (matches C)
// ============================================================================

fn nearestInt(x: f32) i32 {
    // C: (x < 0 || ((x - @as(int,x)) == 0.5f)) ? @intFromFloat(@floor(x)) : @intFromFloat(@round(x))
    if (x < 0.0) return @intFromFloat(@floor(x));
    const xi: f32 = @floatFromInt(@as(i32, @intFromFloat(x)));
    if (x - xi == 0.5) return @intFromFloat(@floor(x));
    return @intFromFloat(@round(x));
}

// ============================================================================
// quantize_row_iq4_xs_ref — port of quantize_row_iq4_nl_impl(QK_K, 32, ..., 7)
// ============================================================================

fn quantizeRowIq4Xs(
    x: [*]const f32, // input: [QK_K] F32 (one super-block)
    out: *BlockIQ4_XS, // output block (caller pre-allocates)
) void {
    // scratch
    var l_table: [QK_K]u8 = undefined; // chosen kvalues_iq4nl index per element
    var weight: [SUB_BLOCK]f32 = undefined;
    var scales: [N_SUB]f32 = undefined;

    @memset(&out.qs, 0);

    // --- step 1: per-block scale via d = -max/kvalues[0] = -max/-127 = max/127 ---
    // (kvalues[0] is -127, so d is positive when max > 0)
    var sigma2: f32 = 0;
    for (0..QK_K) |j| sigma2 += x[j] * x[j];
    sigma2 *= 2.0 / @as(f32, @floatFromInt(QK_K));

    var amax_scale: f32 = 0;
    var max_scale: f32 = 0;
    const ntry: i32 = 7;
    const values = &KVALUES_IQ4NL;
    const v0: f32 = @floatFromInt(values[0]); // -127

    for (0..N_SUB) |ib| {
        const xb: [*]const f32 = x + ib * SUB_BLOCK;
        // weight[j] = xb[j]*xb[j]  (no quant_weights; matches `NULL` branch)
        for (0..SUB_BLOCK) |j| weight[j] = xb[j] * xb[j];
        // amax / max
        var amax: f32 = 0;
        var max: f32 = 0;
        for (0..SUB_BLOCK) |j| {
            const ax = @abs(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax < 1e-9) { // GROUP_MAX_EPS, matches C
            scales[ib] = 0;
            continue;
        }
        var d = -max / v0; // since v0 = -127
        const id: f32 = 1.0 / d;
        var sumqx: f32 = 0;
        var sumq2: f32 = 0;
        const lb_slice: [*]u8 = l_table[ib * SUB_BLOCK ..][0..SUB_BLOCK];
        for (0..SUB_BLOCK) |j| {
            const al = id * xb[j];
            const l = bestIndexInt8(16, values, al);
            lb_slice[j] = l;
            const q: f32 = @floatFromInt(values[l]);
            const w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = if (sumq2 > 0) sumqx / sumq2 else 0.0;
        var best = d * sumqx;
        // Local search ±ntry around the initial d
        var itry: i32 = -ntry;
        while (itry <= ntry) : (itry += 1) {
            const id2: f32 = @as(f32, @floatFromInt(itry + @as(i32, @intFromFloat(v0)))) / max;
            var sqx: f32 = 0;
            var sq2: f32 = 0;
            for (0..SUB_BLOCK) |j| {
                const al = id2 * xb[j];
                const l = bestIndexInt8(16, values, al);
                lb_slice[j] = l;
                const q: f32 = @floatFromInt(values[l]);
                const w = weight[j];
                sqx += w * q * xb[j];
                sq2 += w * q * q;
            }
            if (sq2 > 0 and sqx * sqx > best * sq2) {
                d = sqx / sq2;
                best = d * sqx;
            }
        }
        scales[ib] = d;
        const abs_d = @abs(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    // --- step 2: pack scales to 6 bits (max_scale / 32 reference) ---
    // d is written as f16: d = -max_scale/32 (the C code uses -max_scale/32)
    const d: f32 = -max_scale / 32.0;
    out.d = f32_to_f16(d);
    out.scales_h = 0;
    const id: f32 = if (d != 0) 1.0 / d else 0.0;
    for (0..N_SUB) |ib| {
        var l: i32 = nearestInt(id * scales[ib]);
        l = std.math.clamp(l, -32, 31);
        const dl: f32 = d * @as(f32, @floatFromInt(l));
        const idl: f32 = if (dl != 0) 1.0 / dl else 0.0;
        const lb_slice: [*]u8 = l_table[ib * SUB_BLOCK ..][0..SUB_BLOCK];
        const xb: [*]const f32 = x + ib * SUB_BLOCK;
        for (0..SUB_BLOCK) |j| {
            lb_slice[j] = bestIndexInt8(16, values, idl * xb[j]);
        }
        l += 32; // re-anchor to 0..63
        const l_l: u8 = @intCast(l & 0xf);
        const l_h: u8 = @intCast((l >> 4) & 3);
        if (ib % 2 == 0) {
            out.scales_l[ib / 2] = l_l;
        } else {
            out.scales_l[ib / 2] |= l_l << 4;
        }
        out.scales_h |= @as(u16, l_h) << @intCast(2 * (ib % 8));
    }

    // --- step 3: encode L indices into qs nibbles (lo 4, hi 4) ---
    for (0..N_SUB) |i| {
        for (0..SUB_BLOCK / 2) |j| {
            const lo = l_table[i * SUB_BLOCK + j];
            const hi = l_table[i * SUB_BLOCK + j + 16];
            out.qs[i * (SUB_BLOCK / 2) + j] = @as(u8, lo) | (@as(u8, hi) << 4);
        }
    }
}

// ============================================================================
// dequantize_row_iq4_xs — port of dequantize_row_iq4_xs (byte-exact vs C)
// ============================================================================

pub fn dequantizeRowIQ4_XS(x: [*]const BlockIQ4_XS, y: [*]f32, k: usize) void {
    // k must be a multiple of QK_K; the caller (dequantize_row wrapper) ensures
    // this. The block is QK_K=256 F32 outputs per BlockIQ4_XS input.
    // Zig 0.16: `y` is a many-pointer ([*]f32) — to advance the cursor
    // by N elements across sub-blocks, we carry an index counter
    // (`yoff`) instead of doing `y += 32` directly. This is the same
    // pattern Q4_K/Q5_K/Q6_K use (their authors ran into the same
    // compiler rule on many-pointer arithmetic).
    const nb = k / QK_K;
    var yoff: usize = 0;

    for (0..nb) |i| {
        const blk = &x[i];
        const qs = blk.qs[0..];
        const d = f16_to_f32(blk.d);

        for (0..N_SUB) |ib| {
            // Unpack the 6-bit scale offset for this 32-weight sub-block.
            // (scales_l[ib/2] >> 4*(ib%2)) & 0xf  -> low 4 bits
            // ((scales_h >> 2*ib) & 3) << 4       -> high 2 bits, OR'd in
            const l_low = (blk.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0xf;
            const l_high: u8 = @intCast((blk.scales_h >> @intCast(2 * ib)) & 3);
            const ls: u32 = @as(u32, l_low) | (@as(u32, l_high) << 4);
            // The stored value 0..63 is signed by subtracting 32 in the dequant.
            const dl = d * (@as(f32, @floatFromInt(ls)) - 32.0);

            // 32 weights: 16 from low nibbles of qs[0..16], 16 from high nibbles of qs[0..16].
            // (qs[0..16] for this 32-weight sub-block starts at qs[ib*16]).
            const qso = qs[ib * 16 ..][0..16];
            for (0..16) |j| {
                const v_lo = KVALUES_IQ4NL[qso[j] & 0xf];
                const v_hi = KVALUES_IQ4NL[qso[j] >> 4];
                y[yoff + j + 0] = dl * @as(f32, @floatFromInt(v_lo));
                y[yoff + j + 16] = dl * @as(f32, @floatFromInt(v_hi));
            }
            yoff += 32;
        }
    }
}

// ============================================================================
// quantize_row_iq4_xs — row-level entry (k is a multiple of QK_K)
// ============================================================================

pub fn quantizeRowIQ4_XS(x: [*]const f32, y: [*]BlockIQ4_XS, k: usize) void {
    // Zig 0.16: arithmetic on a many-pointer (`y + i`) is rejected
    // for the same reason as in dequantizeRowIQ4_XS. The internal
    // quantizeRowIq4Xs takes a single-block *BlockIQ4_XS, so we
    // @ptrCast the many-pointer to a single pointer.
    const nb = k / QK_K;
    for (0..nb) |i| {
        const blk: *BlockIQ4_XS = @ptrCast(@alignCast(y + i));
        quantizeRowIq4Xs(x + i * QK_K, blk);
    }
}

// ============================================================================
// scalar GEMM: BF16 activations × IQ4_XS weights -> F32 output
// ============================================================================

// ============================================================================
// AVX2 GEMM
// ============================================================================
//
// Same four rules as everywhere else in this tree: n-outer so the weight block
// streams from DRAM once, MR=4 so four activation rows share one unpacked
// weight vector, explicit `@mulAdd` because FP contraction is off in this
// toolchain, and one `@reduce` per output element.
//
// The unpack is the same SPLIT-HALF shape as IQ4_NL — for each 32-weight
// sub-block, the low nibbles of its 16 bytes are weights 0..15 and the high
// nibbles are weights 16..31 — but the scale is per-sub-block rather than per
// block. The 6-bit scale is split across two arrays and must be reassembled
// before use:
//
//   low  4 bits: (scales_l[ib/2] >> 4*(ib%2)) & 0xf
//   high 2 bits: (scales_h >> 2*ib) & 3,  shifted up by 4
//   signed by subtracting 32, then multiplied by the f16 super-block scale
//
// Getting the two halves from the wrong sub-block still yields a plausible
// number, so the scale index is asserted directly in the test.

const MR: usize = 4;
const Vec: type = @Vector(8, f32);

/// 8 bf16 -> 8 f32, unaligned.
inline fn widen8(p: [*]const amx.bf16) Vec {
    return amx.widen8(p);
}

/// The dequant scale for sub-block `ib`: the 6-bit field reassembled from
/// `scales_l`/`scales_h`, signed by subtracting 32, times the super-block f16
/// scale.
///
/// `d` is passed in rather than read from `blk.d` because that conversion is
/// per-BLOCK, not per-sub-block: computing it here meant re-converting the same
/// f16 eight times per 256-weight block. Worse, it sat at the head of a serial
/// dependency chain immediately before the FMAs, and at m=1 there are only four
/// FMAs per sub-block to hide the latency. Hoisting it is what actually moved
/// the m=1 number — NOT the vpshufb, which made no measurable difference here.
inline fn subBlockScale(blk: *const BlockIQ4_XS, d: f32, ib: usize) f32 {
    const l_low = (blk.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0xf;
    const l_high: u8 = @intCast((blk.scales_h >> @intCast(2 * ib)) & 3);
    const ls: u32 = @as(u32, l_low) | (@as(u32, l_high) << 4);
    return d * (@as(f32, @floatFromInt(ls)) - 32.0);
}


/// `KVALUES_IQ4NL` as raw bytes, then broadcast into both 128-bit lanes so one
/// `vpshufb` can serve indices from either half of the result.
const KV_BYTES: [16]u8 = blk: {
    var t: [16]u8 = undefined;
    for (KVALUES_IQ4NL, 0..) |v, i| t[i] = @bitCast(v); // i8 -> u8, same byte
    break :blk t;
};
pub const KV_VEC: amx.ByteVec = amx.pshufbTable(&KV_BYTES);

/// 16 packed nibbles -> 16 f32, as two `Vec8`, in ONE `vpshufb`.
///
/// This replaces 16 scalar table loads. `KVALUES_IQ4NL` is 16 bytes, which is
/// exactly what `vpshufb` indexes, so the whole 16-entry non-linear table is one
/// instruction. `shift` 0 takes the low nibbles (weights 0..15 of a sub-block),
/// 4 takes the high ones (weights 16..31).
///
/// The two `Vec8` come from one 16-byte result widened twice: sign-extend with
/// `@intCast` (a `vpmovsxbd`) then convert (a `vcvtdq2ps`).
pub inline fn kvalues16(p: *const u8, comptime shift: u4) [2]Vec {
    // Arrays, not vectors, for the indexed part: `vec[i]` needs a comptime `i`.
    const bytes: [16]u8 = @as(*align(1) const [16]u8, @ptrCast(p)).*;
    var idx: [16]u8 = undefined;
    for (0..16) |i| {
        idx[i] = if (shift == 0) bytes[i] & 0x0F else bytes[i] >> shift;
    }
    // `pshufb` on a ymm works lane-wise, so the 16 indices go into both
    // 128-bit halves (KV_VEC has the table in both, so either half works).
    var idx32: [32]u8 = undefined;
    for (0..16) |i| {
        idx32[i] = idx[i];
        idx32[i + 16] = idx[i];
    }
    const looked: [32]u8 = amx.pshufb16(KV_VEC, idx32);

    const lof: Vec = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), @bitCast(looked[0..8].*)))));
    const hif: Vec = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), @bitCast(looked[8..16].*)))));
    return .{ lof, hif };
}

/// AVX2 IQ4_XS GEMM. `c` is OVERWRITTEN on every element of [0,m) x [0,n).
/// `k` must be a multiple of QK_K, as for the scalar version.
pub fn gemmAvx2(
    a: [*]const amx.bf16,
    b: [*]const BlockIQ4_XS,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    const nb = k / QK_K;
    var j: usize = 0;
    while (j < n) : (j += 1) {
        const brow = b + j * ldb;

        var i: usize = 0;
        while (i + MR <= m) : (i += MR) {
            var acc = [_]Vec{@splat(0.0)} ** MR;
            for (0..nb) |blk| {
                const blk_ptr = &brow[blk];
                const abase = blk * QK_K;
                const dsuper = f16_to_f32(blk_ptr.d); // hoisted: per block, not per sub-block
                // 8 sub-blocks of 32. Runtime `ib` so this does not unroll 128
                // vfmadd per block; the 4 Vec8 inside each sub-block are
                // comptime so their byte offsets fold into the load displacement.
                for (0..N_SUB) |ib| {
                    const dv: Vec = @splat(subBlockScale(blk_ptr, dsuper, ib));
                    const qbase = ib * 16;
                    // Two vpshufb per sub-block: low nibbles -> weights 0..15,
                    // high nibbles -> weights 16..31. Each yields two Vec8.
                    inline for (0..2) |nh| {
                        const halves = kvalues16(&blk_ptr.qs[qbase], if (nh == 0) 0 else 4);
                        inline for (0..2) |half| {
                            const koff = abase + ib * SUB_BLOCK + nh * 16 + half * 8;
                            const w = halves[half] * dv;
                            inline for (0..MR) |r| {
                                acc[r] = @mulAdd(Vec, widen8(a + (i + r) * lda + koff), w, acc[r]);
                            }
                        }
                    }
                }
            }
            inline for (0..MR) |r| c[(i + r) * ldc + j] = @reduce(.Add, acc[r]);
        }

        // Row tail (m % MR), one row at a time.
        while (i < m) : (i += 1) {
            var acc: Vec = @splat(0.0);
            for (0..nb) |blk| {
                const blk_ptr = &brow[blk];
                const abase = blk * QK_K;
                const dsuper = f16_to_f32(blk_ptr.d); // hoisted: per block, not per sub-block
                for (0..N_SUB) |ib| {
                    const dv: Vec = @splat(subBlockScale(blk_ptr, dsuper, ib));
                    const qbase = ib * 16;
                    inline for (0..2) |nh| {
                        const halves = kvalues16(&blk_ptr.qs[qbase], if (nh == 0) 0 else 4);
                        inline for (0..2) |half| {
                            const koff = abase + ib * SUB_BLOCK + nh * 16 + half * 8;
                            acc = @mulAdd(Vec, widen8(a + i * lda + koff), halves[half] * dv, acc);
                        }
                    }
                }
            }
            c[i * ldc + j] = @reduce(.Add, acc);
        }
    }
}

pub fn gemmIQ4_XSScalar(
    input: [*]const amx.bf16, // [m, k] BF16, row-major
    weight: [*]const BlockIQ4_XS, // [n, k/QK_K] blocks, row-major
    output: [*]f32, // [m, n] F32, row-major
    m: usize,
    n: usize,
    k: usize,
    input_ld: usize, // row stride of input in elements
    weight_ld: usize, // row stride of weight in BLOCKS
    output_ld: usize, // row stride of output in elements
) void {
    // Dequantize one block of B into an F32 scratch on the fly per (i,j)
    // pair (cheapest for the scalar path; an AMX path will use tiles).
    const nb = k / QK_K;
    var scratch: [QK_K]f32 = undefined;
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..nb) |blk| {
                dequantizeRowIQ4_XS(weight + j * weight_ld + blk, &scratch, QK_K);
                // BUG (pre-existing): the activation index omitted `blk * QK_K`, so every
                // block was multiplied by the SAME first QK_K activations. Correct
                // only when nb == 1 (k == QK_K); wrong for every real checkpoint.
                // Pinned by "IQ scalar GEMMs honour the block offset" in
                // tests/kernels/simd_kernels_tests.zig.
                for (0..QK_K) |k0| {
                    sum += amx.bf16_to_f32(input[i * input_ld + blk * QK_K + k0]) * scratch[k0];
                }
            }
            output[i * output_ld + j] = sum;
        }
    }
}

// ============================================================================
// dequantizeRowIQ4_XSToBF16 — for the AMX path / direct test convenience
// ============================================================================

pub fn dequantizeRowIQ4_XSToBF16(x: [*]const BlockIQ4_XS, y: [*]amx.bf16, k: usize) void {
    const nb = k / QK_K;
    var f32_buf: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        @memset(&f32_buf, 0);
        dequantizeRowIQ4_XS(x + i, &f32_buf, QK_K);
        for (0..QK_K) |j| y[i * QK_K + j] = amx.f32_to_bf16(f32_buf[j]);
    }
}
