// GEMM Kernel 224 FP8 E4M3 for AMX/AVX512
// Ported from ktransformers kt-kernel/operators/amx/la/amx_kernels.hpp
// FP8 E4M3 weights with BF16 activations

const std = @import("std");
const amx = @import("../arch/amx.zig");
const buffers = @import("buffers.zig");

// ============================================================================
// FP8 E4M3 Type
// ============================================================================

/// FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
/// Range: ~0.00195 to 448.0
pub const fp8_e4m3 = u8;

/// E4M3 constants
pub const E4M3_MAX: f32 = 448.0;
pub const E4M3_MIN: f32 = 0.001953125; // 2^-9

/// Convert f32 to FP8 E4M3 (saturate to range)
pub fn f32_to_fp8e4m3(x: f32) fp8_e4m3 {
    if (x > E4M3_MAX) return 0x7F; // max positive
    if (x < -E4M3_MAX) return 0xFF; // max negative
    if (@abs(x) < E4M3_MIN and x != 0) {
        // Subnormal or zero
        return if (x < 0) 0x80 else 0x00;
    }

    // Simple conversion: extract sign, exponent, mantissa
    const u = @as(u32, @bitCast(x));
    const sign = (u >> 31) & 1;
    const exp = @as(i32, @intCast((u >> 23) & 0xFF)) - 127;
    const mant = u & 0x007FFFFF;

    // E4M3: bias = 7, so exp + 7
    const e4m3_exp = exp + 7;
    if (e4m3_exp < 0) return if (sign == 1) 0x80 else 0x00; // underflow to zero
    if (e4m3_exp > 15) return if (sign == 1) 0xFF else 0x7F; // overflow to max

    // 3 mantissa bits (round)
    const m3 = (mant >> 20) & 0x7;
    const round_bit = (mant >> 19) & 1;
    const rounded_mant = if (round_bit == 1) m3 + 1 else m3;

    const result: u8 = (@as(u8, @intCast(e4m3_exp)) << 3) | @as(u8, @intCast(rounded_mant & 0x7));
    return if (sign == 1) result | @as(u8, 0x80) else result;
}

/// Convert FP8 E4M3 to f32
pub fn fp8e4m3_to_f32(x: fp8_e4m3) f32 {
    const sign = (x >> 7) & 1;
    const exp = @as(i32, @intCast((x >> 3) & 0x0F)) - 7;
    const mant = @as(u32, x & 0x07);

    // Reconstruct f32
    const f32_exp = @as(u32, @intCast(exp + 127)) << 23;
    const f32_mant = mant << 20;
    const f32_sign = @as(u32, sign) << 31;

    return @bitCast(f32_sign | f32_exp | f32_mant);
}

/// Convert FP8 E4M3 to BF16
pub fn fp8e4m3_to_bf16(x: fp8_e4m3) amx.bf16 {
    return amx.f32_to_bf16(fp8e4m3_to_f32(x));
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
    pub const TILE_K = 64;  // FP8 uses 64 for K (like INT8)
    pub const TILE_N = 16;
    pub const VNNI_BLK = 4;

    pub const M_STEP = 32;
    pub const N_STEP = 32;
    pub const K_STEP = 64;

    pub const N_BLOCK = 64;
    pub const K_BLOCK = 3584;

    pub fn name() []const u8 { return "FP8_E4M3"; }

    pub fn recommended_nth(n: usize) usize {
        return (n + N_BLOCK - 1) / N_BLOCK;
    }

    // =======================================================================
    // FP8 GEMM with dequantization
    // =======================================================================

    /// Compute GEMM with FP8 weights and BF16 activations
    /// a: [m, k] BF16 activations
    /// b: [n, k] FP8 E4M3 weights (with per-row scale)
    /// c: [m, n] FP32 output
    pub fn gemmFullTile(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const fp8_e4m3, b_scales: [*]const f32, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        // Scalar fallback with on-the-fly dequantization
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                const scale = b_scales[j];
                for (0..k) |kk| {
                    const a_val = amx.bf16_to_f32(a[i * lda + kk]);
                    const b_val = fp8e4m3_to_f32(b[j * ldb + kk]) * scale;
                    sum += a_val * b_val;
                }
                c[i * ldc + j] = sum;
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
