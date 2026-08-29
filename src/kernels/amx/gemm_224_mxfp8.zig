// GEMM Kernel 224 MXFP8 for AMX/AVX512
// MXFP8: Microscaling FP8 format (E4M3 or E5M2) with per-block scales
// Used in MiniMax M3 and other models

const std = @import("std");
const amx = @import("../arch/amx.zig");
const buffers = @import("buffers.zig");

// ============================================================================
// MXFP8 Type (E4M3 with microscaling)
// ============================================================================

/// Reuse FP8 E4M3 from fp8 kernel
pub const fp8_e4m3 = u8;

/// MXFP8 block: 32 FP8 weights + 1 scale (BF16) = 34 bytes
pub const MXFP8Block = extern struct {
    scale: amx.bf16,
    qs: [32]fp8_e4m3,
};

/// FP8 E4M3 conversion (from gemm_224_fp8.zig)
const E4M3_MAX: f32 = 448.0;

pub fn fp8e4m3_to_f32(x: fp8_e4m3) f32 {
    const sign = (x >> 7) & 1;
    const exp = @as(i32, @intCast((x >> 3) & 0x0F)) - 7;
    const mant = @as(u32, x & 0x07);
    const f32_exp = @as(u32, @intCast(exp + 127)) << 23;
    const f32_mant = mant << 20;
    const f32_sign = @as(u32, sign) << 31;
    return @bitCast(f32_sign | f32_exp | f32_mant);
}

pub fn f32_to_fp8e4m3(x: f32) fp8_e4m3 {
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

// ============================================================================
// Kernel Configuration for MXFP8
// ============================================================================

pub const GemmKernel224MXFP8 = struct {
    pub const dt = MXFP8Block;
    pub const output_t = f32;
    pub const ELEMENT_SIZE = 1;

    pub const TILE_M = 16;
    pub const TILE_K = 64;
    pub const TILE_N = 16;
    pub const VNNI_BLK = 4;

    pub const M_STEP = 32;
    pub const N_STEP = 32;
    pub const K_STEP = 64;

    pub const N_BLOCK = 64;
    pub const K_BLOCK = 3584;
    pub const GROUP_SIZE = 32;

    pub fn name() []const u8 { return "MXFP8_E4M3"; }

    pub fn recommended_nth(n: usize) usize {
        return (n + N_BLOCK - 1) / N_BLOCK;
    }

    /// GEMM with MXFP8 weights and BF16 activations
    pub fn gemmFullTile(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const MXFP8Block, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        const blocks_per_row = (k + GROUP_SIZE - 1) / GROUP_SIZE;

        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                for (0..blocks_per_row) |blk| {
                    const block = &b[j * ldb + blk];
                    const scale = amx.bf16_to_f32(block.scale);
                    const k_start = blk * GROUP_SIZE;
                    const k_end = @min(k, k_start + GROUP_SIZE);

                    for (k_start..k_end) |kk| {
                        const a_val = amx.bf16_to_f32(a[i * lda + kk]);
                        const b_val = fp8e4m3_to_f32(block.qs[kk - k_start]) * scale;
                        sum += a_val * b_val;
                    }
                }
                c[i * ldc + j] = sum;
            }
        }
    }
};

// ============================================================================
// MXFP8 BufferB
// ============================================================================

pub const MXFP8BufferB = struct {
    ptr: [*]MXFP8Block,
    n: usize,
    k: usize,
    n_step: usize,
    k_step: usize,
    k_block: usize,
    n_block: usize,
    group_size: usize,

    pub fn requiredSize(n: usize, k: usize, group_size: usize) usize {
        const blocks_per_row = (k + group_size - 1) / group_size;
        return n * blocks_per_row * @sizeOf(MXFP8Block);
    }

    pub fn init(n: usize, k: usize, ptr: *MXFP8Block, n_step: usize, k_step: usize,
                k_block: usize, n_block: usize, group_size: usize) MXFP8BufferB {
        return MXFP8BufferB{
            .ptr = ptr,
            .n = n,
            .k = k,
            .n_step = n_step,
            .k_step = k_step,
            .k_block = k_block,
            .n_block = n_block,
            .group_size = group_size,
        };
    }

    /// Pack BF16 weights to MXFP8 with per-block quantization
    pub fn fromMatBF16(self: *MXFP8BufferB, src: [*]const amx.bf16, src_ld: usize) void {
        const blocks_per_row = (self.k + self.group_size - 1) / self.group_size;

        for (0..self.n) |n_idx| {
            for (0..blocks_per_row) |blk| {
                const k_start = blk * self.group_size;
                const k_end = @min(self.k, k_start + self.group_size);
                const k_actual = k_end - k_start;

                var max_val: f32 = 0;
                for (0..k_actual) |k_off| {
                    const val = amx.bf16_to_f32(src[n_idx * src_ld + k_start + k_off]);
                    const abs_val = if (val < 0) -val else val;
                    if (abs_val > max_val) max_val = abs_val;
                }

                const scale = if (max_val > 0) max_val / E4M3_MAX else 1.0;
                const inv_scale = 1.0 / scale;

                const block = &self.ptr[n_idx * blocks_per_row + blk];
                block.scale = amx.f32_to_bf16(scale);

                for (0..k_actual) |k_off| {
                    const val = amx.bf16_to_f32(src[n_idx * src_ld + k_start + k_off]) * inv_scale;
                    block.qs[k_off] = f32_to_fp8e4m3(val);
                }
                // Zero padding
                for (k_actual..32) |k_off| {
                    block.qs[k_off] = 0;
                }
            }
        }
    }
};
