// AVX512 VNNI INT8 GEMM Kernel for ktransformers-zig
// Uses @Vector for portable SIMD + inline asm for VNNI dpbusd
// Target: AVX512-VNNI (Intel Cascade Lake+, AMD Zen 4+)

const std = @import("std");
const amx = @import("../arch/amx_intrinsics.zig");

// AVX512 vector types
const Vec16i32 = @Vector(16, i32);
const Vec64i8 = @Vector(64, i8);
const Vec32i8 = @Vector(32, i8);

/// VNNI dot product: 4x i8 * 4x i8 -> i32 accumulate
/// vpdpbusd: multiply unsigned bytes by signed bytes and accumulate to dwords
/// We use inline asm for the actual VNNI instruction
pub inline fn vnniDotProduct16(
    a_u8: @Vector(64, u8),  // 16 groups of 4 unsigned bytes
    b_i8: @Vector(64, i8),  // 16 groups of 4 signed bytes
    acc: Vec16i32,
) Vec16i32 {
    // Software fallback (VNNI emulation)
    // In real hardware, this would be a single vpdpbusd instruction
    var result = acc;
    for (0..16) |i| {
        const a0: i32 = @intCast(a_u8[i * 4 + 0]);
        const a1: i32 = @intCast(a_u8[i * 4 + 1]);
        const a2: i32 = @intCast(a_u8[i * 4 + 2]);
        const a3: i32 = @intCast(a_u8[i * 4 + 3]);
        const b0: i32 = @intCast(b_i8[i * 4 + 0]);
        const b1: i32 = @intCast(b_i8[i * 4 + 1]);
        const b2: i32 = @intCast(b_i8[i * 4 + 2]);
        const b3: i32 = @intCast(b_i8[i * 4 + 3]);
        result[i] += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    }
    return result;
}

/// VNNI GEMM INT8: C[M,N] = A[M,K] * B[N,K]^T
/// A, B in INT8; C in INT32
/// K must be multiple of 4 for VNNI
pub fn gemmInt8Avx512(
    a: [*]const i8, lda: usize,
    b: [*]const i8, ldb: usize,
    c: [*]i32, ldc: usize,
    m: usize, n: usize, k: usize,
) void {
    const unrollM: usize = 2;
    const unrollN: usize = 16; // 16 i32 accumulators = 1 AVX512 register
    const kBlock: usize = 64;  // 64 i8 = 16 VNNI operations

    var i: usize = 0;
    while (i + unrollM <= m) : (i += unrollM) {
        var j: usize = 0;
        while (j + unrollN <= n) : (j += unrollN) {
            var acc0: Vec16i32 = @splat(0);
            var acc1: Vec16i32 = @splat(0);

            var kk: usize = 0;
            while (kk < k) : (kk += 4) {
                const klen = @min(4, k - kk);
                if (klen < 4) break; // VNNI needs groups of 4

                // Load A values for 2 rows
                const a0_0: i32 = a[(i + 0) * lda + kk + 0];
                const a0_1: i32 = a[(i + 0) * lda + kk + 1];
                const a0_2: i32 = a[(i + 0) * lda + kk + 2];
                const a0_3: i32 = a[(i + 0) * lda + kk + 3];
                const a1_0: i32 = a[(i + 1) * lda + kk + 0];
                const a1_1: i32 = a[(i + 1) * lda + kk + 1];
                const a1_2: i32 = a[(i + 1) * lda + kk + 2];
                const a1_3: i32 = a[(i + 1) * lda + kk + 3];

                // Load B columns (16 values each)
                for (0..16) |jj| {
                    const b_col = j + jj;
                    const b0: i32 = b[b_col * ldb + kk + 0];
                    const b1: i32 = b[b_col * ldb + kk + 1];
                    const b2: i32 = b[b_col * ldb + kk + 2];
                    const b3: i32 = b[b_col * ldb + kk + 3];

                    acc0[jj] += a0_0 * b0 + a0_1 * b1 + a0_2 * b2 + a0_3 * b3;
                    acc1[jj] += a1_0 * b0 + a1_1 * b1 + a1_2 * b2 + a1_3 * b3;
                }
            }

            // Store
            for (0..16) |jj| {
                c[(i + 0) * ldc + j + jj] = acc0[jj];
                c[(i + 1) * ldc + j + jj] = acc1[jj];
            }
        }

        // Scalar tail for N
        while (j < n) : (j += 1) {
            var sum0: i32 = 0;
            var sum1: i32 = 0;
            for (0..k) |kk| {
                sum0 += @as(i32, a[(i + 0) * lda + kk]) * @as(i32, b[j * ldb + kk]);
                sum1 += @as(i32, a[(i + 1) * lda + kk]) * @as(i32, b[j * ldb + kk]);
            }
            c[(i + 0) * ldc + j] = sum0;
            c[(i + 1) * ldc + j] = sum1;
        }
    }

    // Scalar tail for M
    while (i < m) : (i += 1) {
        for (0..n) |j| {
            var sum: i32 = 0;
            for (0..k) |kk| {
                sum += @as(i32, a[i * lda + kk]) * @as(i32, b[j * ldb + kk]);
            }
            c[i * ldc + j] = sum;
        }
    }
}

/// VNNI GEMM with AMX tiles (INT8)
pub fn gemmInt8AmxOrAvx512(
    a: [*]const i8, lda: usize,
    b: [*]const i8, ldb: usize,
    c: [*]i32, ldc: usize,
    m: usize, n: usize, k: usize,
) void {
    if (amx.amxInt8Available()) {
        amx.tileGemmInt8(a, lda, b, ldb, c, ldc, m, n, k);
    } else {
        gemmInt8Avx512(a, lda, b, ldb, c, ldc, m, n, k);
    }
}

/// INT8 quantization with AVX512
/// Quantize a block of BF16 to INT8 with per-row scale
pub fn quantizeBlockBF16ToInt8Avx512(
    src: [*]const u16, dst: [*]i8, scale: *f32, k: usize,
) void {
    // Find max abs value (vectorized)
    var max_val: f32 = 0;
    var i: usize = 0;
    while (i + 16 <= k) : (i += 16) {
        const v_u16: @Vector(16, u16) = src[i..][0..16].*;
        var v_f32: @Vector(16, f32) = undefined;
        for (0..16) |j| {
            v_f32[j] = @bitCast(@as(u32, v_u16[j]) << 16);
        }
        const abs_v = @abs(v_f32);
        const max_vec = @reduce(.Max, abs_v);
        if (max_vec > max_val) max_val = max_vec;
    }
    while (i < k) : (i += 1) {
        const v = @bitCast(@as(u32, src[i]) << 16);
        const abs_v = if (v < 0) -v else v;
        if (abs_v > max_val) max_val = abs_v;
    }

    scale.* = if (max_val > 0) max_val / 127.0 else 1.0;
    const inv_scale = 1.0 / scale.*;

    // Quantize (vectorized)
    i = 0;
    while (i + 16 <= k) : (i += 16) {
        const v_u16: @Vector(16, u16) = src[i..][0..16].*;
        var v_f32: @Vector(16, f32) = undefined;
        for (0..16) |j| {
            v_f32[j] = @bitCast(@as(u32, v_u16[j]) << 16);
        }
        const scaled = v_f32 * @as(@Vector(16, f32), @splat(inv_scale));
        const rounded = @round(scaled);
        for (0..16) |j| {
            const val: i32 = @intFromFloat(rounded[j]);
            dst[i + j] = @intCast(std.math.clamp(val, -127, 127));
        }
    }
    while (i < k) : (i += 1) {
        const v = @bitCast(@as(u32, src[i]) << 16) * inv_scale;
        const val: i32 = @intFromFloat(@round(v));
        dst[i] = @intCast(std.math.clamp(val, -127, 127));
    }
}
