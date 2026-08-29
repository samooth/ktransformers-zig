// GEMM Kernel 224 MXFP4 for AMX/AVX512
// MXFP4: Microscaling FP4 format (E2M1) with per-block scales
// Used in some MoE implementations for extreme weight compression

const std = @import("std");
const amx = @import("../arch/amx.zig");
const buffers = @import("buffers.zig");

// ============================================================================
// MXFP4 Type (E2M1)
// ============================================================================

/// FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
/// Values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
pub const fp4_e2m1 = u4;  // 4-bit value

/// MXFP4 block: 32 FP4 weights + 1 scale (BF16) = 18 bytes
pub const MXFP4Block = extern struct {
    scale: amx.bf16,
    qs: [16]u8,  // 32 nibbles packed
};

/// FP4 E2M1 lookup table (index 0-15)
const FP4_TABLE = [_]f32{
    0.0,   0.5,   1.0,   1.5,
    2.0,   3.0,   4.0,   6.0,
    -0.0,  -0.5,  -1.0,  -1.5,
    -2.0,  -3.0,  -4.0,  -6.0,
};

/// Convert FP4 E2M1 to f32 using lookup table
pub fn fp4e2m1_to_f32(x: u4) f32 {
    return FP4_TABLE[x];
}

/// Convert f32 to nearest FP4 E2M1 value
pub fn f32_to_fp4e2m1(x: f32) u4 {
    var best_idx: u4 = 0;
    var best_diff: f32 = @abs(x - FP4_TABLE[0]);

    for (1..16) |i| {
        const diff = @abs(x - FP4_TABLE[i]);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = @intCast(i);
        }
    }
    return best_idx;
}

/// Unpack 32 FP4 values from MXFP4Block
pub fn unpackMXFP4(block: *const MXFP4Block, dst: [*]f32) void {
    const scale = amx.bf16_to_f32(block.scale);
    for (0..16) |i| {
        const byte = block.qs[i];
        const low = @as(u4, @intCast(byte & 0x0F));
        const high = @as(u4, @intCast((byte >> 4) & 0x0F));
        dst[i * 2] = fp4e2m1_to_f32(low) * scale;
        dst[i * 2 + 1] = fp4e2m1_to_f32(high) * scale;
    }
}

// ============================================================================
// Kernel Configuration for MXFP4
// ============================================================================

pub const GemmKernel224MXFP4 = struct {
    pub const dt = MXFP4Block;
    pub const output_t = f32;
    pub const ELEMENT_SIZE = 0.5;  // 4 bits = 0.5 bytes

    pub const TILE_M = 16;
    pub const TILE_K = 64;
    pub const TILE_N = 16;
    pub const VNNI_BLK = 4;

    pub const M_STEP = 32;
    pub const N_STEP = 32;
    pub const K_STEP = 64;

    pub const N_BLOCK = 64;
    pub const K_BLOCK = 3584;
    pub const GROUP_SIZE = 32;  // 32 weights per block

    pub fn name() []const u8 { return "MXFP4_E2M1"; }

    pub fn recommended_nth(n: usize) usize {
        return (n + N_BLOCK - 1) / N_BLOCK;
    }

    /// GEMM with MXFP4 weights and BF16 activations
    pub fn gemmFullTile(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const MXFP4Block, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        const blocks_per_row = (k + GROUP_SIZE - 1) / GROUP_SIZE;

        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                for (0..blocks_per_row) |blk| {
                    const block = &b[j * ldb + blk];
                    unpackMXFP4(block, &sum);  // Simplified - real impl would accumulate
                    // Proper implementation:
                    var unpacked: [32]f32 = undefined;
                    unpackMXFP4(block, &unpacked);
                    const k_start = blk * GROUP_SIZE;
                    const k_end = @min(k, k_start + GROUP_SIZE);
                    for (k_start..k_end) |kk| {
                        sum += amx.bf16_to_f32(a[i * lda + kk]) * unpacked[kk - k_start];
                    }
                }
                c[i * ldc + j] = sum;
            }
        }
    }
};

// ============================================================================
// MXFP4 BufferB
// ============================================================================

pub const MXFP4BufferB = struct {
    ptr: [*]MXFP4Block,
    n: usize,
    k: usize,
    n_step: usize,
    k_step: usize,
    k_block: usize,
    n_block: usize,
    group_size: usize,

    pub fn requiredSize(n: usize, k: usize, group_size: usize) usize {
        const blocks_per_row = (k + group_size - 1) / group_size;
        return n * blocks_per_row * @sizeOf(MXFP4Block);
    }

    pub fn init(n: usize, k: usize, ptr: *MXFP4Block, n_step: usize, k_step: usize,
                k_block: usize, n_block: usize, group_size: usize) MXFP4BufferB {
        return MXFP4BufferB{
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

    /// Pack BF16 weights to MXFP4
    pub fn fromMatBF16(self: *MXFP4BufferB, src: [*]const amx.bf16, src_ld: usize) void {
        const blocks_per_row = (self.k + self.group_size - 1) / self.group_size;

        for (0..self.n) |n_idx| {
            for (0..blocks_per_row) |blk| {
                const k_start = blk * self.group_size;
                const k_end = @min(self.k, k_start + self.group_size);
                const k_actual = k_end - k_start;

                // Find max abs for scale
                var max_val: f32 = 0;
                for (0..k_actual) |k_off| {
                    const val = amx.bf16_to_f32(src[n_idx * src_ld + k_start + k_off]);
                    const abs_val = if (val < 0) -val else val;
                    if (abs_val > max_val) max_val = abs_val;
                }

                const scale = if (max_val > 0) max_val / 6.0 else 1.0;
                const inv_scale = 1.0 / scale;

                const block = &self.ptr[n_idx * blocks_per_row + blk];
                block.scale = amx.f32_to_bf16(scale);

                for (0..16) |q| {
                    const k0 = k_start + q * 2;
                    const k1 = k0 + 1;

                    const v0 = if (k0 < self.k)
                        f32_to_fp4e2m1(amx.bf16_to_f32(src[n_idx * src_ld + k0]) * inv_scale)
                    else
                        0;
                    const v1 = if (k1 < self.k)
                        f32_to_fp4e2m1(amx.bf16_to_f32(src[n_idx * src_ld + k1]) * inv_scale)
                    else
                        0;

                    block.qs[q] = (@as(u8, v1) << 4) | @as(u8, v0);
                }
            }
        }
    }
};
