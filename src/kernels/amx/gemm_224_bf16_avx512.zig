// AVX512 BF16 GEMM Kernel for ktransformers-zig
// Uses @Vector for portable SIMD + inline asm for AMX tile ops
// Target: AVX512-BF16 (Intel Cooper Lake, Sapphire Rapids, Granite Rapids)

const std = @import("std");
const amx = @import("../arch/amx_intrinsics.zig");

// AVX512 vector types
const Vec16f32 = @Vector(16, f32);
const Vec16u16 = @Vector(16, u16);
const Vec32u16 = @Vector(32, u16);
const Vec16i32 = @Vector(16, i32);

/// Convert 16 f32 to 16 BF16 (u16) using AVX512-BF16
/// Fallback: software conversion if intrinsics unavailable
pub inline fn cvt16f32ToBf16(vec: Vec16f32) Vec16u16 {
    // Software fallback for BF16 conversion
    // Round to nearest even, truncate mantissa
    var result: Vec16u16 = undefined;
    const u = @as([16]u32, @bitCast(vec));
    for (0..16) |i| {
        const sign = (u[i] >> 31) & 1;
        const exp = (u[i] >> 23) & 0xFF;
        const mant = u[i] & 0x007FFFFF;
        // Round: add 0x00008000 to mantissa before truncating
        const rounded = u[i] + 0x00008000;
        const bf16 = (rounded >> 16) & 0xFFFF;
        result[i] = @intCast(bf16);
    }
    return result;
}

/// Convert 16 BF16 (u16) to 16 f32
pub inline fn cvt16Bf16ToF32(vec: Vec16u16) Vec16f32 {
    var result: Vec16f32 = undefined;
    for (0..16) |i| {
        const u = @as(u32, vec[i]) << 16;
        result[i] = @bitCast(u);
    }
    return result;
}

/// BF16 dot product of 2 vectors of 32 BF16 elements each
/// Returns f32 sum (2 accumulations of 16-element dot products)
pub inline fn bf16DotProduct32(a: [*]const u16, b: [*]const u16) f32 {
    // Load first 16 elements
    const a0: Vec16u16 = a[0..16].*;
    const b0: Vec16u16 = b[0..16].*;
    const a0_f32 = cvt16Bf16ToF32(a0);
    const b0_f32 = cvt16Bf16ToF32(b0);
    const prod0 = a0_f32 * b0_f32;

    // Load second 16 elements
    const a1: Vec16u16 = a[16..32].*;
    const b1: Vec16u16 = b[16..32].*;
    const a1_f32 = cvt16Bf16ToF32(a1);
    const b1_f32 = cvt16Bf16ToF32(b1);
    const prod1 = a1_f32 * b1_f32;

    return @reduce(.Add, prod0) + @reduce(.Add, prod1);
}

/// AVX512 BF16 GEMM: C[M,N] = A[M,K] * B[N,K]^T
/// A, B in BF16 (u16), C in FP32
pub fn gemmBF16Avx512(
    a: [*]const u16, lda: usize,
    b: [*]const u16, ldb: usize,
    c: [*]f32, ldc: usize,
    m: usize, n: usize, k: usize,
) void {
    const unrollM: usize = 2;  // Process 2 rows of A at a time
    const unrollN: usize = 16; // Process 16 cols of B at a time (1 AVX512 register)
    const kBlock: usize = 32;  // Process 32 elements of K at a time

    var i: usize = 0;
    while (i + unrollM <= m) : (i += unrollM) {
        var j: usize = 0;
        while (j + unrollN <= n) : (j += unrollN) {
            // Initialize 2x16 accumulators to zero
            var acc0: Vec16f32 = @splat(0.0);
            var acc1: Vec16f32 = @splat(0.0);

            var kk: usize = 0;
            while (kk < k) : (kk += kBlock) {
                const klen = @min(kBlock, k - kk);

                // Prefetch next A/B blocks
                // @prefetch(a + (i + 2) * lda + kk, .{ .rw = .read, .locality = 3 });

                // Load A rows
                for (0..klen) |kidx| {
                    const a0_val: f32 = @bitCast(@as(u32, a[(i + 0) * lda + kk + kidx]) << 16);
                    const a1_val: f32 = @bitCast(@as(u32, a[(i + 1) * lda + kk + kidx]) << 16);

                    // Load B column (16 values)
                    var b_vec: Vec16u16 = undefined;
                    for (0..16) |jj| {
                        b_vec[jj] = b[(j + jj) * ldb + kk + kidx];
                    }
                    const b_f32 = cvt16Bf16ToF32(b_vec);

                    // FMA: acc += a * b
                    const a0_broadcast: Vec16f32 = @splat(a0_val);
                    const a1_broadcast: Vec16f32 = @splat(a1_val);
                    acc0 += a0_broadcast * b_f32;
                    acc1 += a1_broadcast * b_f32;
                }
            }

            // Store results
            const c0 = c + (i + 0) * ldc + j;
            const c1 = c + (i + 1) * ldc + j;
            for (0..16) |jj| {
                c0[jj] = acc0[jj];
                c1[jj] = acc1[jj];
            }
        }

        // Handle remaining N columns (scalar)
        while (j < n) : (j += 1) {
            var sum0: f32 = 0;
            var sum1: f32 = 0;
            for (0..k) |kk| {
                const a0 = @bitCast(@as(u32, a[(i + 0) * lda + kk]) << 16);
                const a1 = @bitCast(@as(u32, a[(i + 1) * lda + kk]) << 16);
                const bval = @bitCast(@as(u32, b[j * ldb + kk]) << 16);
                sum0 += a0 * bval;
                sum1 += a1 * bval;
            }
            c[(i + 0) * ldc + j] = sum0;
            c[(i + 1) * ldc + j] = sum1;
        }
    }

    // Handle remaining rows (scalar)
    while (i < m) : (i += 1) {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                const a_val = @bitCast(@as(u32, a[i * lda + kk]) << 16);
                const b_val = @bitCast(@as(u32, b[j * ldb + kk]) << 16);
                sum += a_val * b_val;
            }
            c[i * ldc + j] = sum;
        }
    }
}

/// BF16 GEMM with AMX tiles (highest performance)
/// Falls back to AVX512 if AMX not available
pub fn gemmBF16AmxOrAvx512(
    a: [*]const u16, lda: usize,
    b: [*]const u16, ldb: usize,
    c: [*]f32, ldc: usize,
    m: usize, n: usize, k: usize,
) void {
    if (amx.amxBf16Available()) {
        amx.tileGemmBF16(a, lda, b, ldb, c, ldc, m, n, k);
    } else {
        gemmBF16Avx512(a, lda, b, ldb, c, ldc, m, n, k);
    }
}

/// SwiGLU with AVX512: gate = silu(gate) * up
/// Processes 16 elements at a time
pub fn swigluAvx512(
    gate: [*]const u16, up: [*]const u16, dst: [*]u16, count: usize,
) void {
    const vec_len: usize = 16;
    var i: usize = 0;
    while (i + vec_len <= count) : (i += vec_len) {
        const g_u16: Vec16u16 = gate[i..][0..vec_len].*;
        const u_u16: Vec16u16 = up[i..][0..vec_len].*;

        const g_f32 = cvt16Bf16ToF32(g_u16);
        const u_f32 = cvt16Bf16ToF32(u_u16);

        // SiLU: x * sigmoid(x)
        const neg_g = -g_f32;
        const exp_neg = @exp(neg_g);
        const sigmoid = 1.0 / (1.0 + exp_neg);
        const silu = g_f32 * sigmoid;

        const result = silu * u_f32;
        const result_u16 = cvt16f32ToBf16(result);

        dst[i..][0..vec_len].* = result_u16;
    }

    // Scalar tail
    while (i < count) : (i += 1) {
        const g = @bitCast(@as(u32, gate[i]) << 16);
        const u = @bitCast(@as(u32, up[i]) << 16);
        const sigmoid = 1.0 / (1.0 + @exp(-g));
        const silu = g * sigmoid;
        const result = silu * u;
        dst[i] = @intCast((@as(u32, @bitCast(result)) + 0x00008000) >> 16);
    }
}

/// RMSNorm with AVX512
pub fn rmsNormAvx512(
    input: [*]const u16, weight: [*]const u16, output: [*]u16,
    rows: usize, cols: usize, eps: f32,
) void {
    for (0..rows) |r| {
        // Compute sum of squares
        var sum_sq: f32 = 0;
        var i: usize = 0;
        while (i + 16 <= cols) : (i += 16) {
            const v: Vec16u16 = input[r * cols + i ..][0..16].*;
            const vf = cvt16Bf16ToF32(v);
            sum_sq += @reduce(.Add, vf * vf);
        }
        while (i < cols) : (i += 1) {
            const v = @bitCast(@as(u32, input[r * cols + i]) << 16);
            sum_sq += v * v;
        }

        const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(cols)) + eps);
        const inv_rms = 1.0 / rms;

        // Normalize and scale
        i = 0;
        while (i + 16 <= cols) : (i += 16) {
            const in_v: Vec16u16 = input[r * cols + i ..][0..16].*;
            const w_v: Vec16u16 = weight[i..][0..16].*;
            const in_f = cvt16Bf16ToF32(in_v);
            const w_f = cvt16Bf16ToF32(w_v);
            const out_f = in_f * @as(Vec16f32, @splat(inv_rms)) * w_f;
            output[r * cols + i ..][0..16].* = cvt16f32ToBf16(out_f);
        }
        while (i < cols) : (i += 1) {
            const in_v = @bitCast(@as(u32, input[r * cols + i]) << 16);
            const w_v = @bitCast(@as(u32, weight[i]) << 16);
            const out_v = in_v * inv_rms * w_v;
            output[r * cols + i] = @intCast((@as(u32, @bitCast(out_v)) + 0x00008000) >> 16);
        }
    }
}
