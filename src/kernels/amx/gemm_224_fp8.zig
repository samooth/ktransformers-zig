// GEMM Kernel 224 FP8 E4M3 for AMX/AVX512
// Ported from ktransformers kt-kernel/operators/amx/la/amx_kernels.hpp
// FP8 E4M3 weights with BF16 activations

const std = @import("std");
const amx = @import("../arch/amx.zig");
const buffers = @import("buffers.zig");

// ============================================================================
// FP8 E4M3 Type — THE SINGLE OWNER OF THE CODEC
// ============================================================================
//
// `gemm_224_mxfp8.zig` imports this file's codec from here and keeps only its
// MX-specific parts. It used to carry its own copy of all three functions, and
// that is exactly how the two copies drifted: the MX encoder was rewritten
// (correctly) while this one kept all four of its bugs, and the MX decoder's
// replacement was then written out again here. Two editable copies of a codec
// is not a redundancy, it is a future bug report. If you need E4M3
// encode/decode, import it; do not re-derive it.

/// FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
/// Range: ~0.00195 to 448.0
pub const fp8_e4m3 = u8;

/// E4M3 constants
pub const E4M3_MAX: f32 = 448.0;
pub const E4M3_MIN: f32 = 0.001953125; // 2^-9

/// f32 -> FP8 E4M3, round-to-nearest-even, saturating.
///
/// The version this replaces was wrong in four ways, all reachable from
/// ordinary weight data and all measured at 2026-10-04 (see TODO.md Stream B):
///
///  1. The 3-bit mantissa round-overflow was MASKED (`rounded_mant & 0x7`)
///     instead of carried into the exponent, so every value whose mantissa
///     rounded up from 7 encoded one power of two too small — 3.875 became
///     0x40 (2.0) instead of 0x48 (4.0). A 2x error on 1/16 of the whole
///     normal range.
///  2. Saturation returned 0x7F / 0xFF, which are the NaN encodings, so any
///     weight above E4M3_MAX became NaN: 500.0 encoded to 0x7F, which the
///     decoder then maps to 0.0, so an over-range weight was silently DELETED.
///     The largest FINITE encoding is 0x7E (exp 15, man 6) = 448.
///  3. Every subnormal was flushed to zero — the `|x| < 2^-9` guard discarded
///     all seven valid subnormal encodings, and the `e4m3_exp < 0` guard
///     mis-encoded the [2^-7, 2^-6) band into the subnormal field.
///  4. Rounding was round-half-up rather than round-to-nearest-even.
///
/// This is live in the C ABI: `kt_fp8_quantize`, `kt_fp8_quantize_block` and
/// `kt_quantize_fp8_e4m3` all call it (src/main.zig:2020, 2058, 2540), as does
/// `quantizeRowBF16ToFP8` below. Identical to the encoder `gemm_224_mxfp8.zig`
/// carried before the two were merged here.
pub fn f32_to_fp8e4m3(x: f32) fp8_e4m3 {
    if (std.math.isNan(x)) return if (std.math.signbit(x)) 0xFF else 0x7F;

    const s: u8 = if (std.math.signbit(x)) 0x80 else 0x00;
    const a = @abs(x);

    // 0x7E = 448 is the largest finite encoding; 0x7F is NaN.
    if (a >= E4M3_MAX) return s | 0x7E;

    // Subnormal band: value = man * 2^-9 with man in 0..7. 2^-9 is exactly
    // representable, so scaling by 512 is exact and the rounding below is
    // exact too. A result of 8 means it rounded up past the subnormals and
    // belongs in the normal band as 0x08 (2^-6).
    if (a < 0.015625) {
        const scaled = a * 512.0;
        const fl = @floor(scaled);
        var man: u8 = @intFromFloat(fl);
        const frac = scaled - fl;
        if (frac > 0.5 or (frac == 0.5 and (man & 1) == 1)) man += 1;
        return s | (if (man >= 8) 0x08 else man);
    }

    const u: u32 = @bitCast(a);
    const exp = @as(i32, @intCast((u >> 23) & 0xFF)) - 127;
    const mant = u & 0x007FFFFF;

    var e_out: u8 = @intCast(exp + 7); // bias 7
    const m3 = (mant >> 20) & 0x7;
    const round_bit = (mant >> 19) & 1;
    const sticky = mant & 0x7FFFF != 0;
    var mm: u8 = @intCast(m3);
    // Round-to-nearest-even: round up on >half, or on an exact half when the
    // retained value is odd (so 2.5 -> 2, not 3).
    if (round_bit == 1 and (sticky or (m3 & 1) == 1)) mm += 1;
    // Carry the mantissa overflow into the exponent. THIS IS THE FIX for the
    // 2x error; masking it here is what broke 3.875.
    if (mm == 8) {
        mm = 0;
        e_out += 1;
    }
    if (e_out > 15) return s | 0x7E; // 2^9 and up saturate
    return s | (e_out << 3) | mm;
}

/// FP8 E4M3 -> F32 lookup table, built at comptime.
///
/// This is the exact table the C++ reference builds at startup in
/// `operators/avx2/fp8_dequant.hpp` (`FP8LUT::init`), which gathers from it
/// with `_mm256_i32gather_ps`. Matching it bit-for-bit is the point: it is
/// what makes a cross-implementation accuracy diff meaningful.
///
/// It replaces a pure bit-reconstruction decoder that was wrong twice over:
///   - subnormals (exponent field 0, mantissa 1-7) were decoded as NORMAL,
///     yielding ~1.0-2.0 instead of `man/8 * 2^-6` (2^-9 .. 7*2^-9);
///   - 0x7F, the single NaN encoding in E4M3, decoded to 480.0, which is
///     above E4M3_MAX=448 and therefore not even representable.
///
/// 0x7F maps to 0.0 here, matching the reference: "Only 0x7F is NaN in E4M3.
/// Treat as 0 to avoid propagation." Exponent 15 with mantissa 0-6 are real
/// finite values (256..448) and decode normally.
pub const FP8_E4M3_LUT: [256]f32 = blk: {
    var t: [256]f32 = undefined;
    for (0..256) |i| {
        const bits: u8 = @intCast(i);
        const sign: u1 = @intCast((i >> 7) & 1);
        const exp: i32 = @intCast((i >> 3) & 0xF);
        const man: u32 = @intCast(bits & 0x7);

        var v: f32 = undefined;
        if (exp == 0 and man == 0) {
            v = 0.0;
        } else if (exp == 0) {
            v = @as(f32, @floatFromInt(man)) * (1.0 / 8.0) * 0.015625; // man/8 * 2^-6
        } else if (exp == 15 and man == 7) {
            v = 0.0; // the one NaN encoding; see above
        } else {
            // 2^(exp-7) * (1 + man/8), built straight into the f32 exponent
            // field. NOT std.math.scalbn/ldexp: those take a
            // `comptime_float`/f32 union and cannot be evaluated at comptime
            // in Zig 0.16 ("access of union field 'float' while field
            // 'comptime_float' is active", plus a branch-quota overflow), so
            // this table failed to build. exponent = exp + 120 is exactly
            // (exp - 7) + 127, and man << 20 is exactly the 3-bit mantissa.
            v = @bitCast((@as(u32, @as(u32, exp) + 120) << 23) | (@as(u32, man) << 20));
        }
        t[i] = if (sign == 1) -v else v;
    }
    break :blk t;
};

/// Convert FP8 E4M3 to f32. One L1-resident table load; see
/// `FP8_E4M3_LUT` for why this is a lookup and not bit arithmetic.
pub fn fp8e4m3_to_f32(x: fp8_e4m3) f32 {
    return FP8_E4M3_LUT[x];
}

/// Convert FP8 E4M3 to BF16
pub fn fp8e4m3_to_bf16(x: fp8_e4m3) amx.bf16 {
    return amx.f32_to_bf16(fp8e4m3_to_f32(x));
}

// ============================================================================
// FP8 E5M2 Type
// ============================================================================
//
// 1 sign bit, 5 exponent bits, 2 mantissa bits, bias 15. This is the second
// OCP FP8 format and the one DeepSeek-style FP8 transports use for activations,
// where the extra exponent range matters more than mantissa precision.
//
// IT IS NOT E4M3 WITH WIDER FIELDS, and every difference below is a place a
// by-analogy port goes wrong. The E4M3 encoder in this file carried four real
// bugs and the list is the checklist for this one:
//
//   1. **There are no infinities.** The entire exponent-31 field is NaN, so
//      `E5M2_MAX` is `0x7B` (exp 30, man 3) = 1.75 * 2^15 = 57344, NOT 0x7E.
//      Encoding saturation to 0x7E would produce a NaN, and the decoder maps
//      NaN to 0 — which is how the E4M3 encoder deleted every weight above
//      range instead of clipping it.
//   2. **Bias is 15, not 7.** Getting this wrong shifts every value by 2^8.
//   3. **Subnormals reach 2^-16** across four encodings (1..3 * 2^-16 plus
//      zero), not 2^-9 across seven. A guard that flushes them deletes real
//      small values.
//   4. **The mantissa carry still lands in the exponent.** 2-bit mantissa or
//      not: rounding 1.75 up gives 2.0, which is an exponent bump, not a
//      mantissa mask. Masking it is the 2x bug E4M3 had at 3.875.
///
/// Rounding is round-to-nearest-even throughout, matching E4M3 and matching the
/// C++ reference.

/// FP8 E5M2 format: 1 sign bit, 5 exponent bits, 2 mantissa bits
/// Range: 2^-16 to 57344.0
pub const fp8_e5m2 = u8;

/// Largest finite E5M2 value: 0x7B = 1.75 * 2^15. There is no infinity.
pub const E5M2_MAX: f32 = 57344.0;

/// Smallest positive normal: 2^-14.
pub const E5M2_MIN_NORMAL: f32 = 0.00006103515625; // 2^-14

/// Smallest positive subnormal: 2^-16.
pub const E5M2_MIN_SUBNORMAL: f32 = 0.0000152587890625; // 2^-16

/// The four subnormal magnitudes, exactly representable in f32 (each is a
/// small multiple of 2^-16, needing at most 2 mantissa bits).
pub const E5M2_SUBNORMALS: [4]f32 = .{
    0.0,
    1.0 / 65536.0, // 2^-16
    2.0 / 65536.0, // 2^-15
    3.0 / 65536.0,
};

/// The NaN encoding. E5M2's NaN field is the whole top exponent (0x7C..0x7F
/// and their signed twins); 0x7F is quiet NaN with both mantissa bits set.
pub const E5M2_NAN: fp8_e5m2 = 0x7F;

/// Encode f32 to FP8 E5M2, saturating rather than producing NaN.
pub fn f32_to_fp8e5m2(x: f32) fp8_e5m2 {
    if (std.math.isNan(x)) return if (std.math.signbit(x)) 0xFF else E5M2_NAN;

    const s: u8 = if (std.math.signbit(x)) 0x80 else 0x00;
    const a = @abs(x);

    // 0x7B = 57344 is the largest finite encoding. Everything at or above it
    // clips there; the NaN encodings are NOT reachable by saturation.
    if (a >= E5M2_MAX) return s | 0x7B;

    // Subnormal band: value = man * 2^-16 with man in 0..3. 2^-16 is exactly
    // representable, so scaling by 65536 is exact and the rounding below is
    // exact too. A result of 4 means it rounded up past the subnormals and
    // belongs in the normal band as 0x04 (2^-14).
    if (a < E5M2_MIN_NORMAL) {
        const scaled = a * 65536.0;
        const fl = @floor(scaled);
        var man: u8 = @intFromFloat(fl);
        const frac = scaled - fl;
        if (frac > 0.5 or (frac == 0.5 and (man & 1) == 1)) man += 1;
        return s | (if (man >= 4) 0x04 else man);
    }

    const u: u32 = @bitCast(a);
    const exp = @as(i32, @intCast((u >> 23) & 0xFF)) - 127;
    const mant = u & 0x007FFFFF;

    var e_out: u8 = @intCast(exp + 15); // bias 15, NOT 7
    const m2 = (mant >> 21) & 0x3;
    const round_bit = (mant >> 20) & 1;
    // Sticky is the bits STRICTLY BELOW the round bit, so it must mask off
    // bit 20 as well — `mant & 0x1FFFFF` includes the round bit itself, which
    // makes `sticky` true whenever `round_bit` is, and silently degrades
    // round-to-nearest-even into round-half-up. The E4M3 encoder above keeps 3
    // bits with the round bit at 19 and masks 0x7FFFF (bits 0-18) for the same
    // reason: with 2 retained bits the mask is bits 0-19.
    const sticky = mant & 0x000FFFFF != 0;
    var mm: u8 = @intCast(m2);
    // Round-to-nearest-even, same rule as E4M3.
    if (round_bit == 1 and (sticky or (m2 & 1) == 1)) mm += 1;
    // The carry into the exponent. 1.75 rounds up to 2.0, which lives in the
    // exponent field; masking it here is the 2x error E4M3 had at 3.875.
    if (mm == 4) {
        mm = 0;
        e_out += 1;
    }
    if (e_out > 30) return s | 0x7B; // 2^15 and up saturate to max finite
    return s | (e_out << 2) | mm;
}

/// FP8 E5M2 -> F32 lookup table, built at comptime.
///
/// Same shape as the E4M3 table, and for the same reason: a LUT is what the
/// GEMM gathers from, and building it once at comptime keeps the decode path
/// branch-free.
///
/// The normal band is assembled by bit placement rather than arithmetic, so it
/// is exact by construction: E5M2's 5 exponent bits and 2 mantissa bits fit
/// f32's 8 and 23 with a +112 bias shift and a 21-bit mantissa shift, with no
/// rounding anywhere. The subnormal band cannot be bit-assembled (it is a
/// different encoding, not a small exponent), so it comes from the exact
/// constant table.
pub const FP8_E5M2_LUT: [256]f32 = blk: {
    var t: [256]f32 = undefined;
    for (0..256) |i| {
        const sign: u1 = @intCast((i >> 7) & 1);
        const e: u8 = (i >> 2) & 0x1F;
        const m: u8 = i & 0x03;

        var v: f32 = undefined;
        if (e == 31) {
            // No infinity in E5M2: the whole top exponent is NaN.
            v = std.math.nan(f32);
        } else if (e == 0) {
            // Index 0 is +0.0; the sign is applied below, which is what keeps
            // 0x80 decoding to NEGATIVE zero rather than +0.0. Folding the sign
            // into the value instead — `if (m == 0) 0.0` — silently drops it,
            // and then 0x80 fails to round-trip through the encoder. The
            // 256-encoding sweep is what caught that.
            v = E5M2_SUBNORMALS[m];
        } else {
            // Built straight into the f32 fields rather than with ldexp, for
            // the reason spelled out in the E4M3 table above: ldexp cannot be
            // evaluated at comptime in Zig 0.16. Here it does not need to be,
            // because exponent +112 is exactly (e - 15) + 127 and m << 21 is
            // exactly the 2-bit mantissa, so the normal band is bit-exact with
            // no rounding at all.
            const bits: u32 = (@as(u32, e + 112) << 23) | (@as(u32, m) << 21);
            v = @bitCast(bits);
        }
        // Applied last, so -0.0 survives. NaN keeps its sign too, harmlessly.
        t[i] = if (sign == 1) -v else v;
    }
    break :blk t;
};

/// Decode FP8 E5M2 to F32.
pub fn fp8e5m2_to_f32(x: fp8_e5m2) f32 {
    return FP8_E5M2_LUT[x];
}

/// Convert FP8 E5M2 to BF16.
pub fn fp8e5m2_to_bf16(x: fp8_e5m2) amx.bf16 {
    return amx.f32_to_bf16(fp8e5m2_to_f32(x));
}

// ============================================================================
// FP8 Quantization Helpers
// ============================================================================

/// Quantize a row of BF16 values to FP8 E4M3 with per-row scale
/// Uses per-channel (per-row) scaling for better accuracy
pub fn quantizeRowBF16ToFP8(src: [*]const amx.bf16, dst: [*]fp8_e4m3, scale: *f32, k: usize) void {
    var max_val: f32 = 0;
    for (0..k) |i| {
        const val = amx.bf16_to_f32(src[i]);
        const abs_val = if (val < 0) -val else val;
        if (abs_val > max_val) max_val = abs_val;
    }

    // Scale so max_val maps to ~448 (E4M3 max)
    scale.* = if (max_val > 0) max_val / E4M3_MAX else 1.0;
    const inv_scale = 1.0 / scale.*;

    for (0..k) |i| {
        const val = amx.bf16_to_f32(src[i]) * inv_scale;
        dst[i] = f32_to_fp8e4m3(val);
    }
}

/// Dequantize FP8 row to BF16
pub fn dequantizeRowFP8ToBF16(src: [*]const fp8_e4m3, scale: f32, dst: [*]amx.bf16, k: usize) void {
    for (0..k) |i| {
        dst[i] = amx.f32_to_bf16(fp8e4m3_to_f32(src[i]) * scale);
    }
}

// ============================================================================
// Kernel Configuration for FP8 E4M3
// ============================================================================

pub const GemmKernel224FP8 = struct {
    pub const dt = fp8_e4m3;
    pub const output_t = f32;
    pub const ELEMENT_SIZE = 1;

    pub const TILE_M = 16;
    pub const TILE_K = 32;  // FP8 -> BF16, then BF16 tile path. TILE_K=32 matches BF16 GEMM.
    pub const TILE_N = 16;
    pub const VNNI_BLK = 2;

    pub const M_STEP = 32;
    pub const N_STEP = 32;
    pub const K_STEP = 32;

    // A4: runtime-overridable block sizes (mirrors gemm_224_bf16.zig).
    pub var N_BLOCK: usize = 64;
    pub var K_BLOCK: usize = 3584;

    pub const DEFAULT_N_BLOCK: usize = 64;
    pub const DEFAULT_K_BLOCK: usize = 3584;

    /// Override the block sizes from measured cache hierarchy. Invalid
    /// or out-of-range values keep the defaults.
    pub fn setTileParams(n_block_in: usize, k_block_in: usize) void {
        if (n_block_in >= N_STEP and n_block_in % N_STEP == 0) {
            N_BLOCK = n_block_in;
        }
        if (k_block_in >= K_STEP and k_block_in % K_STEP == 0) {
            K_BLOCK = k_block_in;
        }
    }

    /// Restore the compiled-in defaults.
    pub fn resetTileParams() void {
        N_BLOCK = DEFAULT_N_BLOCK;
        K_BLOCK = DEFAULT_K_BLOCK;
    }

    pub fn name() []const u8 { return "FP8_E4M3"; }

    pub fn recommended_nth(n: usize) usize {
        return (n + N_BLOCK - 1) / N_BLOCK;
    }

    // =======================================================================
    // Tile Configuration (mirrors BF16 GEMM: 2x A, 2x B, 4x C, all BF16/FP32)
    // =======================================================================

    /// Set the 8-tile configuration: 2x A (16x32 BF16), 2x B (16x32 BF16),
    /// 4x C (16x16 FP32). Identical to GemmKernel224BF16.
    pub fn config() void {
        if (!amx.detectAmxSupport()) return;
        var tile_config = amx.TileConfig.init();

        // Tile 0,1: A matrices (16 x 32 BF16) — 32 bytes per row
        for (0..2) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), TILE_M, @as(u16, @intCast(TILE_K * @sizeOf(amx.bf16))));
        }

        // Tile 2,3: B matrices (16 x 32 BF16 VNNI, 2 tiles of 16x32)
        for (2..4) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), @as(u16, @intCast(TILE_K / VNNI_BLK)), @as(u16, @intCast(TILE_N * VNNI_BLK * @sizeOf(amx.bf16))));
        }

        // Tile 4,5,6,7: C matrices (16 x 16 FP32)
        for (4..8) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), TILE_M, @as(u16, @intCast(TILE_N * @sizeOf(f32))));
        }

        amx.tile_loadconfig(&tile_config);
    }

    pub fn cleanC() void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_zero(amx.TileReg.tmm4);
        amx.tile_zero(amx.TileReg.tmm5);
        amx.tile_zero(amx.TileReg.tmm6);
        amx.tile_zero(amx.TileReg.tmm7);
    }

    pub fn loadC(c: [*]f32, ldc: usize) void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_loadd(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_loadd(amx.TileReg.tmm5, @ptrCast(@as([*]f32, @ptrCast(c)) + TILE_N), ldc);
        amx.tile_loadd(amx.TileReg.tmm6, @ptrCast(@as([*]f32, @ptrCast(c)) + ldc * TILE_M), ldc);
        amx.tile_loadd(amx.TileReg.tmm7, @ptrCast(@as([*]f32, @ptrCast(c)) + ldc * TILE_M + TILE_N), ldc);
    }

    pub fn storeC(c: [*]f32, ldc: usize) void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_stored(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_stored(amx.TileReg.tmm5, @ptrCast(@as([*]f32, @ptrCast(c)) + TILE_N), ldc);
        amx.tile_stored(amx.TileReg.tmm6, @ptrCast(@as([*]f32, @ptrCast(c)) + ldc * TILE_M), ldc);
        amx.tile_stored(amx.TileReg.tmm7, @ptrCast(@as([*]f32, @ptrCast(c)) + ldc * TILE_M + TILE_N), ldc);
    }

    /// Load 32 rows of BF16 activations (M=16) × 32 K into tmm0/tmm1.
    /// `lda` is the row stride in bytes of the A matrix.
    pub fn loadA(a: [*]const amx.bf16, lda: usize) void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_loadd(amx.TileReg.tmm0, @ptrCast(a), lda);
        amx.tile_loadd(amx.TileReg.tmm1, @ptrCast(@as([*]const amx.bf16, @ptrCast(a)) + TILE_M * lda), lda);
    }

    /// Run the 2x2x4 tile dot product: 4 tilebf16dpd instructions.
    /// FP32 accumulator (no INT32 scratch needed).
    pub fn runTile() void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_dpbf16ps(amx.TileReg.tmm4, amx.TileReg.tmm0, amx.TileReg.tmm2);
        amx.tile_dpbf16ps(amx.TileReg.tmm5, amx.TileReg.tmm0, amx.TileReg.tmm3);
        amx.tile_dpbf16ps(amx.TileReg.tmm6, amx.TileReg.tmm1, amx.TileReg.tmm2);
        amx.tile_dpbf16ps(amx.TileReg.tmm7, amx.TileReg.tmm1, amx.TileReg.tmm3);
    }

    // =======================================================================
    // FP8 On-the-fly Dequantization (FP8 -> BF16 for tilebf16dpd)
    // =======================================================================

    /// Dequantize 16 rows of FP8 (B) into a 16×32 BF16 scratch buffer (32
    /// bytes of FP8 → 64 bytes of BF16, since each FP8 is 1 byte and each
    /// BF16 is 2 bytes).
    /// Then load as tiles tmm2 (rows 0..15) and tmm3 (rows 16..31).
    /// `ldb` is the row stride of `b` in elements of fp8_e4m3 (1 byte each).
    pub fn loadB(b: [*]const fp8_e4m3, ldb: usize, scratch: *[2 * TILE_N * TILE_K]amx.bf16) void {
        if (!amx.detectAmxSupport()) return;
        // Dequantize 16 rows for tmm2.
        for (0..TILE_N) |i| {
            for (0..TILE_K) |k| {
                scratch[i * TILE_K + k] = fp8e4m3_to_bf16((@as([*]const fp8_e4m3, @ptrCast(b)))[i * ldb + k]);
            }
        }
        // Zero the second 16 rows (for tmm3) — in the simple case we only have
        // 16 rows of B per (m_begin, n_begin) block. The C++ packs more rows
        // here when N_STEP=32 spans two N_STEPs of the same column-major
        // layout, but we keep it simple: tmm3 will be loaded with zeros
        // and contribute nothing to the dot product. (This works because
        // tilebf16dpd accumulates — adding 0 to a 0 tile is harmless.)
        for (TILE_N..2 * TILE_N) |i| {
            for (0..TILE_K) |k| {
                scratch[i * TILE_K + k] = amx.f32_to_bf16(0.0);
            }
        }
        // Load into tmm2 (rows 0..15) and tmm3 (rows 16..31).
        // Stride is in bytes; each BF16 row is TILE_K * 2 = 64 bytes.
        const stride_bytes: usize = TILE_K * @sizeOf(amx.bf16);
        amx.tile_loadd(amx.TileReg.tmm2, @ptrCast(scratch), stride_bytes);
        amx.tile_loadd(amx.TileReg.tmm3, @ptrCast(scratch[TILE_N * TILE_K ..].ptr), stride_bytes);
    }

    // =======================================================================
    // GEMM (AMX-tile path)
    // =======================================================================

    /// Compute GEMM with FP8 weights and BF16 activations using AMX.
    /// a: [m, k] BF16 activations
    /// b: [n, k] FP8 E4M3 weights (with per-row scale at b_scales[j])
    /// c: [m, n] FP32 output
    ///
    /// On non-AMX hardware, this falls back to a scalar implementation.
    pub fn gemmFullTile(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const fp8_e4m3, b_scales: [*]const f32, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        if (!amx.detectAmxSupport()) {
            gemmAvx2(a, lda, b, b_scales, ldb, c, ldc, m, n, k);
            return;
        }

        // Tile config is set once per call (idempotent: ldtilecfg is cheap).
        config();

        // Scratch buffer for dequantized B rows. 32 rows × 32 BF16 = 2048 bytes
        // (32 rows because tmm2 + tmm3 each cover 16 rows; the second 16 are
        // zeroed since we don't span two N-blocks per tile-load in this simple
        // implementation).
        var b_scratch: [2 * TILE_N * TILE_K]amx.bf16 align(64) = undefined;

        for (0..m) |m_begin| {
            const m_end = @min(m_begin + M_STEP, m);
            for (0..n) |n_begin| {
                const n_end = @min(n_begin + N_STEP, n);
                const c_tile = c + m_begin * ldc + n_begin;

                // Clean C tiles at the start of each (m, n) block.
                cleanC();

                // Inner K-loop: K_STEP=32 steps.
                var k_begin: usize = 0;
                while (k_begin < k) : (k_begin += K_STEP) {
                    const a_ptr = a + m_begin * lda + k_begin;
                    const b_ptr = b + n_begin * ldb + k_begin;

                    GemmKernel224FP8.loadA(a_ptr, lda);
                    GemmKernel224FP8.loadB(b_ptr, ldb, &b_scratch);
                    GemmKernel224FP8.runTile();
                }

                // Store FP32 tiles to c.
                GemmKernel224FP8.storeC(c_tile, N_STEP * @sizeOf(f32));

                // Apply per-row scale: c[i, j] *= b_scales[j] for j in [n_begin, n_end).
                for (m_begin..m_end) |i| {
                    for (n_begin..n_end) |j| {
                        (@as([*]f32, @ptrCast(c)))[i * ldc + j] *= b_scales[j];
                    }
                }
            }
        }
    }

    /// Scalar fallback for non-AMX hosts, and the oracle the AVX2 path is
    /// tested against. Kept as the readable statement of the math; see
    /// `gemmAvx2` below for why it is no longer what runs.
    ///
    /// `noinline` so that where this IS called — the test suite and
    /// `bench/fp8_bench.zig`, both of which reference it — it keeps a symbol
    /// and its inner loop stays disassemblable next to the vectorized one.
    /// Nothing in the library calls it any more, so unlike `gemmAvx2` it does
    /// NOT appear in the shipped `.so`; `root.zig` has no comptime reference
    /// for it and `root.zig` is a shared file.
    pub noinline fn gemmFullTileScalar(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const fp8_e4m3, b_scales: [*]const f32, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                const scale = b_scales[j];
                for (0..k) |kk| {
                    const a_val = amx.bf16_to_f32(a[i * lda + kk]);
                    const b_val = fp8e4m3_to_f32(b[j * ldb + kk]) * scale;
                    sum += a_val * b_val;
                }
                (@as([*]f32, @ptrCast(c)))[i * ldc + j] = sum;
            }
        }
    }

    // =======================================================================
    // AVX2 path (non-AMX hosts) — the GEMM that actually runs
    // =======================================================================
    //
    // `gemmFullTileScalar` is an m/n/k triple loop, which is the one loop order
    // the SIMD contract forbids: the weight matrix — the huge operand — is
    // re-streamed from DRAM once per token row. On any non-AMX host (including
    // the development box) `gemmFullTile` lands on it, so it WAS the FP8 GEMM.
    // This is the replacement, built like `kquant_simd.gemmBF16xQuant` and its
    // two siblings (`gemm_224_q8_0.zig`, `gemm_224_mxfp8.zig`):
    //
    //   n-outer / m-inner  the n x k weight block is walked once, in order.
    //                       Same as the reference's `gemm_fp8`, which also puts
    //                       `ni` outside `mi`, and worth 3.4-4.0x on BF16.
    //   MR = 4             four activation rows share one decoded weight
    //                       vector, so the 8 table lookups amortize over 4 FMAs
    //   explicit @mulAdd   FP contraction is off in this toolchain, so a plain
    //                       `a*w + acc` would be a separate multiply and add
    //   deferred hsum      ONE @reduce per OUTPUT element, not per 32-weight
    //                       step (see below)
    //
    // On the accumulators: the reference runs FOUR `__m256` accumulators over
    // each 32-element k step and folds them with one `hsum_avx2`
    // (fp8-moe.hpp:178-201). That is the same intent — independent chains for
    // ILP, a single fold, the scale applied last — reached differently here:
    // MR=4 gets the four independent chains from the M axis instead of the k
    // axis, which costs 4 registers rather than 16 and leaves ONE reduction
    // for the whole row rather than one per 32 weights. Nothing is lost; the
    // fold happens later.
    //
    // On the scale: the reference has 128x128 block-wise scales indexed
    // `b.d[n_block_idx * n_blocks_k + k_block_idx]` and multiplies each k
    // block's folded sum by it. This port has ONE f32 scale per row of B
    // (`FP8BufferB.scales`, filled by `fromMatBF16`), so the scale is constant
    // across the whole k of a row: accumulate raw and multiply ONCE at the end,
    // exactly as `kquant_simd.gemmBF16xQuant` does for the K-quants. That is
    // also strictly more accurate — it is the formulation `gemmFullTileScalar`
    // approximates by folding `scale` into each product.

    const MR: usize = 4;
    const Vec: type = @Vector(8, f32);

    /// 8 bf16 -> 8 f32. Single pointer parameter and `align(1)`: an
    /// activation row can start at any byte offset.
    inline fn widen8(p: [*]const amx.bf16) Vec {
        return amx.widen8(p);
    }

    /// 8 E4M3 bytes -> 8 f32. The local stand-in for the reference's
    /// `fp8x8_to_fp32x8`, which widens with `_mm256_cvtepu8_epi32` and then
    /// does `_mm256_i32gather_ps` against the same 256-entry table.
    ///
    /// A single `vpshufb` cannot replace the gather: the table is 16x16, and
    /// `vpshufb` indexes 16 BYTES. So this keeps the table and pays eight
    /// scalar L1 loads per 8 weights — which is the known cost of this format
    /// (see the report in TODO.md Stream B). `gemmAvx2Gather` below is the
    /// asm-gather experiment against that cost.
    inline fn fp8x8(p: [*]const fp8_e4m3) Vec {
        return lut8(&FP8_E4M3_LUT, p);
    }

    /// Scalar dot over the ragged tail of a k that is not a multiple of 8.
    /// Deliberately does NOT apply the row scale: the caller folds that in
    /// once, after the reduction, exactly as the vector path does.
    inline fn tailDot(a_row: [*]const amx.bf16, brow: [*]const fp8_e4m3, kbase: usize, rem: usize) f32 {
        return tailDotLut(a_row, brow, kbase, rem, &FP8_E4M3_LUT);
    }

    /// 8 E5M2 bytes -> 8 f32, from the E5M2 table. Identical shape to `fp8x8`:
    /// both formats are one byte per weight, so only the table differs.
    inline fn fp8x8E5M2(p: [*]const fp8_e5m2) Vec {
        return lut8(&FP8_E5M2_LUT, p);
    }

    /// Scalar dot over a ragged tail, for E5M2.
    inline fn tailDotE5M2(a_row: [*]const amx.bf16, brow: [*]const fp8_e5m2, kbase: usize, rem: usize) f32 {
        return tailDotLut(a_row, brow, kbase, rem, &FP8_E5M2_LUT);
    }

    /// The two codec-specific steps, factored out so the GEMM body below is
    /// written ONCE and instantiated per format. A second hand-written copy of
    /// that loop is exactly the duplication that let the E4M3 and MXFP8 codecs
    /// drift apart, and a GEMM loop drifting from its scalar oracle is the same
    /// failure with a worse blast radius.
    inline fn lut8(comptime lut: *const [256]f32, p: [*]const u8) Vec {
        const raw: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(p)).*;
        var out: [8]f32 = undefined;
        inline for (0..8) |i| out[i] = lut[raw[i]];
        return out;
    }

    inline fn tailDotLut(
        a_row: [*]const amx.bf16,
        brow: [*]const u8,
        kbase: usize,
        rem: usize,
        comptime lut: *const [256]f32,
    ) f32 {
        var acc: f32 = 0;
        for (0..rem) |t| {
            acc += amx.bf16_to_f32(a_row[kbase + t]) * lut[brow[kbase + t]];
        }
        return acc;
    }

    /// AVX2 FP8 E4M3 GEMM. `c` is OVERWRITTEN on every element of [0,m) x
    /// [0,n) — it is assigned, never accumulated, matching
    /// `gemmFullTileScalar`. The per-row scale is applied once, after the
    /// reduction.
    /// `noinline` so the kernel keeps its own symbol in the `.so`. LLVM
    /// otherwise folds it into `gemmFullTile` (and that into `kt_matmul_fp8`),
    /// which is correct but leaves nothing to disassemble and nothing for `nm`
    /// to show — the failure mode this repo has hit twice.
    pub noinline fn gemmAvx2(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const fp8_e4m3, b_scales: [*]const f32, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        gemmAvx2With(fp8x8, tailDot, a, lda, b, b_scales, ldb, c, ldc, m, n, k);
    }

    /// The same kernel for E5M2 weights. Separate symbol, identical arithmetic
    /// with the E5M2 table substituted — see `gemmAvx2With` for why the body is
    /// shared rather than copied.
    pub noinline fn gemmAvx2E5M2(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const fp8_e5m2, b_scales: [*]const f32, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        gemmAvx2With(fp8x8E5M2, tailDotE5M2, a, lda, b, b_scales, ldb, c, ldc, m, n, k);
    }

    /// The AVX2 GEMM body, parameterised on the two codec-specific steps.
    ///
    /// `decode8` turns 8 weight bytes into 8 f32 lanes; `tail` handles the
    /// ragged k tail. Everything else — the MR-row blocking that lets MR rows of
    /// A share one decoded weight vector, the single horizontal reduction per
    /// output element, the scale applied once after the reduction — is
    /// identical for both formats and lives here exactly once.
    ///
    /// Both formats are one byte per weight, so `b` is `[*]const u8` here and
    /// the typed wrappers above keep the public signatures honest.
    inline fn gemmAvx2With(
        comptime decode8: anytype,
        comptime tail: anytype,
        a: [*]const amx.bf16, lda: usize,
        b: [*]const u8, b_scales: [*]const f32, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize,
    ) void {
        if (m == 0 or n == 0) return;
        const kv = k / 8; // full 8-weight vectors
        const rem = k - kv * 8; // ragged tail, < 8

        var j: usize = 0;
        while (j < n) : (j += 1) {
            const brow = b + j * ldb;

            var i: usize = 0;
            // Main path: MR rows of A share every decoded weight vector.
            while (i + MR <= m) : (i += MR) {
                var acc = [_]Vec{@splat(0.0)} ** MR;
                for (0..kv) |v| {
                    const w = decode8(brow + v * 8);
                    const kb = v * 8;
                    inline for (0..MR) |r| {
                        acc[r] = @mulAdd(Vec, widen8(a + (i + r) * lda + kb), w, acc[r]);
                    }
                }
                inline for (0..MR) |r| {
                    var sum = @reduce(.Add, acc[r]); // ONE hsum per output element
                    if (rem != 0) sum += tail(a + (i + r) * lda, brow, kv * 8, rem);
                    c[(i + r) * ldc + j] = sum * b_scales[j];
                }
            }
            // Row tail (m % MR), one row at a time.
            while (i < m) : (i += 1) {
                var acc: Vec = @splat(0.0);
                for (0..kv) |v| {
                    acc = @mulAdd(Vec, widen8(a + i * lda + v * 8), decode8(brow + v * 8), acc);
                }
                var sum = @reduce(.Add, acc);
                if (rem != 0) sum += tail(a + i * lda, brow, kv * 8, rem);
                c[i * ldc + j] = sum * b_scales[j];
            }
        }
    }

    /// Batched GEMM for MoE with FP8 weights
    pub fn batchedGemm(
        experts: []struct { 
            a: [*]const amx.bf16, 
            b: [*]const fp8_e4m3,
            b_scales: [*]const f32,
            c: [*]f32,
            m: usize, n: usize, k: usize,
            lda: usize, ldb: usize, ldc: usize
        }
    ) void {
        for (experts) |exp| {
            gemmFullTile(exp.a, exp.lda, exp.b, exp.b_scales, exp.ldb, 
                        exp.c, exp.ldc, exp.m, exp.n, exp.k);
        }
    }
};

// ============================================================================
// FP8 BufferB with Per-Row Scales
// ============================================================================

pub const FP8BufferB = struct {
    ptr: [*]fp8_e4m3,
    scales: [*]f32,
    n: usize,
    k: usize,
    n_step: usize,
    k_step: usize,
    k_block: usize,
    n_block: usize,

    pub fn requiredSize(n: usize, k: usize) usize {
        return n * k * @sizeOf(fp8_e4m3) + n * @sizeOf(f32);
    }

    pub fn init(n: usize, k: usize, ptr: *fp8_e4m3, n_step: usize, k_step: usize, 
                k_block: usize, n_block: usize) FP8BufferB {
        const scale_offset = n * k;
        return FP8BufferB{
            .ptr = ptr,
            .scales = @ptrCast(@alignCast(ptr + scale_offset)),
            .n = n,
            .k = k,
            .n_step = n_step,
            .k_step = k_step,
            .k_block = k_block,
            .n_block = n_block,
        };
    }

    /// Pack BF16 weights to FP8 with per-row quantization
    pub fn fromMatBF16(self: *FP8BufferB, src: [*]const amx.bf16, src_ld: usize) void {
        for (0..self.n) |n_idx| {
            const row_src = src[n_idx * src_ld..][0..self.k];
            const row_dst = self.ptr[n_idx * self.k..][0..self.k];
            const scale_ptr = &self.scales[n_idx];

            quantizeRowBF16ToFP8(row_src.ptr, row_dst.ptr, scale_ptr, self.k);
        }
    }
};
