// GEMM Kernel 224 INT4 GPTQ for AMX/AVX512
// Ported from ktransformers kt-kernel/operators/amx/la/amx_kernels.hpp
// INT4 weights with per-group scales and zeros (GPTQ-style quantization)

const std = @import("std");
const amx = @import("../arch/amx.zig");
const buffers = @import("buffers.zig");

// ============================================================================
// INT4 Block Structure (GPTQ-style: block_q4_0 from GGML)
// Each block: 32 weights (4 bytes) + 1 scale (f16/bf16) = 18 bytes for 32 weights
// ============================================================================

pub const BlockQ4_0 = extern struct {
    d: amx.bf16,        // scale
    qs: [16]u8,         // 32 nibbles packed (4 bits each)
};

/// Unpack 32 INT4 values from a BlockQ4_0 into 32 i8 values
/// offset: 0 or 16 (which half of the block)
pub fn unpackQ4_0(block: *const BlockQ4_0, dst: [*]i8, offset: usize) void {
    const scale = amx.bf16_to_f32(block.d);

    for (0..16) |i| {
        const byte = block.qs[i];
        const low = @as(i8, @intCast(byte & 0x0F));
        const high = @as(i8, @intCast((byte >> 4) & 0x0F));

        dst[offset + i * 2] = @intCast((@as(i32, low) - 8) * scale);
        dst[offset + i * 2 + 1] = @intCast((@as(i32, high) - 8) * scale);
    }
}

/// Dequantize a row of BlockQ4_0 to BF16
pub fn dequantizeQ4_0Row(src: [*]const BlockQ4_0, dst: [*]amx.bf16, k: usize, src_ld: usize) void {
    const blocks_per_row = k / 32;
    for (0..blocks_per_row) |blk| {
        const block = &src[blk * src_ld];
        const scale = amx.bf16_to_f32(block.d);
        for (0..16) |i| {
            const byte = block.qs[i];
            const low = @as(f32, @floatFromInt(byte & 0x0F)) - 8.0;
            const high = @as(f32, @floatFromInt((byte >> 4) & 0x0F)) - 8.0;
            dst[blk * 32 + i * 2] = amx.f32_to_bf16(low * scale);
            dst[blk * 32 + i * 2 + 1] = amx.f32_to_bf16(high * scale);
        }
    }
}

// ============================================================================
// Kernel Configuration for INT4 GPTQ
// ============================================================================

pub const GemmKernel224Int4 = struct {
    pub const dt = i8;           // Dequantized to INT8 for computation
    pub const weight_t = BlockQ4_0;
    pub const output_t = f32;
    pub const ELEMENT_SIZE = 1;  // After dequantization

    pub const TILE_M = 16;
    pub const TILE_K = 64;       // INT8 uses 64 for K
    pub const TILE_N = 16;
    pub const VNNI_BLK = 4;

    pub const M_STEP = 32;
    pub const N_STEP = 32;
    pub const K_STEP = 64;

    pub const N_BLOCK = 64;
    pub const K_BLOCK = 3584;
    pub const GROUP_SIZE = 32;   // GPTQ group size

    pub fn name() []const u8 { return "INT4_GPTQ"; }

    pub fn recommended_nth(n: usize) usize {
        return (n + N_BLOCK - 1) / N_BLOCK;
    }

    // =======================================================================
    // Dequantization + GEMM
    // =======================================================================

    /// Dequantize INT4 weights to INT8 on-the-fly and compute GEMM
    /// a: [m, k] INT8 activations
    /// b: [n, k/32] BlockQ4_0 weights (column-major, each row is one output dim)
    /// c: [m, n] FP32 output
    pub fn gemmFullTile(
        a: *const i8, lda: usize,
        b: *const BlockQ4_0, ldb: usize,
        c: *f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        // For now, scalar fallback with on-the-fly dequantization
        // Real implementation would use AVX512 VNNI with dequantization

        const k_blocks = k / GROUP_SIZE;

        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                for (0..k_blocks) |blk| {
                    const block = &(@as([*]const BlockQ4_0, @ptrCast(b))[j * ldb + blk]);
                    const scale = amx.bf16_to_f32(block.d);

                    for (0..16) |q| {
                        const byte = block.qs[q];
                        const low = @as(f32, @floatFromInt(byte & 0x0F)) - 8.0;
                        const high = @as(f32, @floatFromInt((byte >> 4) & 0x0F)) - 8.0;

                        const a_idx_low = blk * GROUP_SIZE + q * 2;
                        const a_idx_high = a_idx_low + 1;

                        if (a_idx_low < k) {
                            sum += @as(f32, @floatFromInt((@as([*]const i8, @ptrCast(a)))[i * lda + a_idx_low])) * low * scale;
                        }
                        if (a_idx_high < k) {
                            sum += @as(f32, @floatFromInt((@as([*]const i8, @ptrCast(a)))[i * lda + a_idx_high])) * high * scale;
                        }
                    }
                }
                (@as([*]f32, @ptrCast(c)))[i * ldc + j] = sum;
            }
        }
    }

    /// Batched GEMM for MoE with INT4 weights
    pub fn batchedGemm(
        experts: []struct { 
            a: *const i8, 
            b: *const BlockQ4_0, 
            c: *f32,
            m: usize, n: usize, k: usize,
            lda: usize, ldb: usize, ldc: usize
        }
    ) void {
        for (experts) |exp| {
            gemmFullTile(exp.a, exp.lda, exp.b, exp.ldb, exp.c, exp.ldc, 
                        exp.m, exp.n, exp.k);
        }
    }
};

// ============================================================================
// INT4 BufferB with GPTQ-style packing
// ============================================================================

pub const Int4BufferB = struct {
    ptr: [*]BlockQ4_0,
    n: usize,
    k: usize,
    n_step: usize,
    k_step: usize,
    k_block: usize,
    n_block: usize,
    group_size: usize,

    pub fn requiredSize(n: usize, k: usize, group_size: usize) usize {
        const blocks_per_row = (k + group_size - 1) / group_size;
        return n * blocks_per_row * @sizeOf(BlockQ4_0);
    }

    pub fn init(n: usize, k: usize, ptr: *BlockQ4_0, n_step: usize, k_step: usize, 
                k_block: usize, n_block: usize, group_size: usize) Int4BufferB {
        return Int4BufferB{
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

    /// Pack FP32/BF16 weights to INT4 GPTQ format with per-group quantization
    pub fn fromMatBF16(self: *Int4BufferB, src: [*]const amx.bf16, src_ld: usize) void {
        const blocks_per_row = (self.k + self.group_size - 1) / self.group_size;

        for (0..self.n) |n_idx| {
            for (0..blocks_per_row) |blk| {
                const k_start = blk * self.group_size;
                const k_end = @min(self.k, k_start + self.group_size);
                const k_actual = k_end - k_start;

                // Find max abs value for scale
                var max_val: f32 = 0;
                for (0..k_actual) |k_off| {
                    const val = amx.bf16_to_f32(src[n_idx * src_ld + k_start + k_off]);
                    const abs_val = if (val < 0) -val else val;
                    if (abs_val > max_val) max_val = abs_val;
                }

                const scale = if (max_val > 0) max_val / 7.0 else 1.0;
                const inv_scale = 1.0 / scale;

                const block = &self.ptr[n_idx * blocks_per_row + blk];
                block.d = amx.f32_to_bf16(scale);

                // Quantize 32 values to 4-bit
                for (0..16) |q| {
                    const k0 = k_start + q * 2;
                    const k1 = k0 + 1;

                    const v0 = if (k0 < self.k) 
                        amx.bf16_to_f32(src[n_idx * src_ld + k0]) * inv_scale + 8.0
                    else 
                        8.0;
                    const v1 = if (k1 < self.k) 
                        amx.bf16_to_f32(src[n_idx * src_ld + k1]) * inv_scale + 8.0
                    else 
                        8.0;

                    const q0: u8 = @intCast(@max(0, @min(15, @as(i32, @intFromFloat(v0)))));
                    const q1: u8 = @intCast(@max(0, @min(15, @as(i32, @intFromFloat(v1)))));

                    block.qs[q] = (q1 << 4) | q0;
                }
            }
        }
    }
};
