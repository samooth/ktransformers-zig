// AVX512 FP8 E4M3 GEMM Kernel for ktransformers-zig
// Dequantizes FP8 on-the-fly and computes with AVX512 f32
// Target: AVX512-F (Intel Skylake+, AMD Zen 4+)
// Future: AVX512-FP8 (Granite Rapids, Diamond Rapids)

const std = @import("std");

/// FP8 E4M3 type
pub const fp8_e4m3 = u8;

/// E4M3 constants
pub const E4M3_MAX: f32 = 448.0;

/// FP8 E4M3 lookup table (index 0-255)
/// Precomputed for fast dequantization
const FP8_E4M3_TABLE: [256]f32 = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| {
        const sign = (i >> 7) & 1;
        const exp = @as(i32, @intCast((i >> 3) & 0x0F)) - 7;
        const mant = @as(u32, i & 0x07);
        const f32_exp = @as(u32, @intCast(exp + 127)) << 23;
        const f32_mant = mant << 20;
        const f32_sign = @as(u32, sign) << 31;
        table[i] = @bitCast(f32_sign | f32_exp | f32_mant);
    }
    break :blk table;
};

/// Convert FP8 E4M3 to f32 using lookup table (fastest)
pub inline fn fp8ToF32Table(x: fp8_e4m3) f32 {
    return FP8_E4M3_TABLE[x];
}

/// Convert FP8 E4M3 to f32 (compute)
pub inline fn fp8ToF32(x: fp8_e4m3) f32 {
    const sign = (x >> 7) & 1;
    const exp = @as(i32, @intCast((x >> 3) & 0x0F)) - 7;
    const mant = @as(u32, x & 0x07);
    const f32_exp = @as(u32, @intCast(exp + 127)) << 23;
    const f32_mant = mant << 20;
    const f32_sign = @as(u32, sign) << 31;
    return @bitCast(f32_sign | f32_exp | f32_mant);
}

/// Convert f32 to FP8 E4M3
pub inline fn f32ToFp8(x: f32) fp8_e4m3 {
    if (x > E4M3_MAX) return 0x7F;
    if (x < -E4M3_MAX) return 0xFF;
    if (@abs(x) < 0.001953125 and x != 0) {
        return if (x < 0) 0x80 else 0x00;
    }
    const u = @as(u32, @bitCast(x));
    const sign = (u >> 31) & 1;
    const exp = @as(i32, @intCast((u >> 23) & 0xFF)) - 127;
    const mant = u & 0x007FFFFF;
    const e4m3_exp = exp + 7;
    if (e4m3_exp < 0) return if (sign == 1) 0x80 else 0x00;
    if (e4m3_exp > 15) return if (sign == 1) 0xFF else 0x7F;
    const m3 = (mant >> 20) & 0x7;
    const round_bit = (mant >> 19) & 1;
    const rounded_mant = if (round_bit == 1) m3 + 1 else m3;
    const result = (@as(u8, @intCast(e4m3_exp)) << 3) | (rounded_mant & 0x7);
    return if (sign == 1) result | 0x80 else result;
}

/// Dequantize a block of FP8 to f32 using @Vector
/// Uses lookup table for speed
pub inline fn dequantBlockFp8ToF32(
    src: [*]const fp8_e4m3,
    scale: f32,
    dst: [*]f32,
    count: usize,
) void {
    const vec_len: usize = 16;
    var i: usize = 0;
    while (i + vec_len <= count) : (i += vec_len) {
        // Load 16 FP8 values
        var indices: @Vector(16, u8) = undefined;
        for (0..16) |j| {
            indices[j] = src[i + j];
        }

        // Dequantize using lookup table
        var result: @Vector(16, f32) = undefined;
        for (0..16) |j| {
            result[j] = FP8_E4M3_TABLE[indices[j]] * scale;
        }

        // Store
        dst[i..][0..vec_len].* = result;
    }

    // Scalar tail
    while (i < count) : (i += 1) {
        dst[i] = FP8_E4M3_TABLE[src[i]] * scale;
    }
}

/// AVX512 FP8 GEMM with on-the-fly dequantization
/// A: [m, k] f32 activations
/// B: [n, k] FP8 weights with per-row scales
/// C: [m, n] f32 output
pub fn gemmFp8Avx512(
    a: [*]const f32, lda: usize,
    b: [*]const fp8_e4m3, b_scales: [*]const f32, ldb: usize,
    c: [*]f32, ldc: usize,
    m: usize, n: usize, k: usize,
) void {
    const vec_len: usize = 16;

    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j + vec_len <= n) : (j += vec_len) {
            var acc: @Vector(16, f32) = @splat(0.0);

            for (0..k) |kk| {
                const a_val: f32 = a[i * lda + kk];
                var b_vec: @Vector(16, f32) = undefined;

                for (0..16) |jj| {
                    const b_idx = (j + jj) * ldb + kk;
                    const scale = b_scales[j + jj];
                    b_vec[jj] = FP8_E4M3_TABLE[b[b_idx]] * scale;
                }

                acc += @as(@Vector(16, f32), @splat(a_val)) * b_vec;
            }

            for (0..16) |jj| {
                c[i * ldc + j + jj] = acc[jj];
            }
        }

        // Scalar tail
        while (j < n) : (j += 1) {
            var sum: f32 = 0;
            const scale = b_scales[j];
            for (0..k) |kk| {
                sum += a[i * lda + kk] * FP8_E4M3_TABLE[b[j * ldb + kk]] * scale;
            }
            c[i * ldc + j] = sum;
        }
    }
}

/// Optimized FP8 GEMM: dequantize B to f32 first, then use AVX512 GEMM
pub fn gemmFp8DequantThenGemm(
    a: [*]const f32, lda: usize,
    b: [*]const fp8_e4m3, b_scales: [*]const f32, ldb: usize,
    c: [*]f32, ldc: usize,
    m: usize, n: usize, k: usize,
) void {
    // Step 1: Dequantize B to f32
    const b_f32 = std.heap.page_allocator.alloc(f32, n * k) catch @panic("OOM");
    defer std.heap.page_allocator.free(b_f32);

    for (0..n) |j| {
        const scale = b_scales[j];
        var kk: usize = 0;
        while (kk + 16 <= k) : (kk += 16) {
            var indices: @Vector(16, u8) = undefined;
            for (0..16) |idx| {
                indices[idx] = b[j * ldb + kk + idx];
            }
            var result: @Vector(16, f32) = undefined;
            for (0..16) |idx| {
                result[idx] = FP8_E4M3_TABLE[indices[idx]] * scale;
            }
            b_f32[j * k + kk ..][0..16].* = result;
        }
        while (kk < k) : (kk += 1) {
            b_f32[j * k + kk] = FP8_E4M3_TABLE[b[j * ldb + kk]] * scale;
        }
    }

    // Step 2: AVX512 f32 GEMM
    const Vec16f32 = @Vector(16, f32);
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j + 16 <= n) : (j += 16) {
            var acc: Vec16f32 = @splat(0.0);
            var kk: usize = 0;
            while (kk < k) : (kk += 1) {
                const a_val: f32 = a[i * lda + kk];
                var b_vec: Vec16f32 = undefined;
                for (0..16) |jj| {
                    b_vec[jj] = b_f32[(j + jj) * k + kk];
                }
                acc += @as(Vec16f32, @splat(a_val)) * b_vec;
            }
            for (0..16) |jj| {
                c[i * ldc + j + jj] = acc[jj];
            }
        }
        while (j < n) : (j += 1) {
            var sum: f32 = 0;
            for (0..k) |kk| {
                sum += a[i * lda + kk] * b_f32[j * k + kk];
            }
            c[i * ldc + j] = sum;
        }
    }
}

/// Quantize f32 block to FP8 with per-row scaling (AVX512-optimized)
pub fn quantizeBlockF32ToFp8Avx512(
    src: [*]const f32,
    dst: [*]fp8_e4m3,
    scale: *f32,
    count: usize,
) void {
    // Find max abs (vectorized)
    var max_val: f32 = 0;
    var i: usize = 0;
    while (i + 16 <= count) : (i += 16) {
        const v: @Vector(16, f32) = src[i..][0..16].*;
        const abs_v = @abs(v);
        const max_vec = @reduce(.Max, abs_v);
        if (max_vec > max_val) max_val = max_vec;
    }
    while (i < count) : (i += 1) {
        const v = src[i];
        const abs_v = if (v < 0) -v else v;
        if (abs_v > max_val) max_val = abs_v;
    }

    scale.* = if (max_val > 0) max_val / E4M3_MAX else 1.0;
    const inv_scale = 1.0 / scale.*;

    // Quantize (vectorized)
    i = 0;
    while (i + 16 <= count) : (i += 16) {
        const v: @Vector(16, f32) = src[i..][0..16].*;
        const scaled = v * @as(@Vector(16, f32), @splat(inv_scale));
        for (0..16) |j| {
            dst[i + j] = f32ToFp8(scaled[j]);
        }
    }
    while (i < count) : (i += 1) {
        dst[i] = f32ToFp8(src[i] * inv_scale);
    }
}
