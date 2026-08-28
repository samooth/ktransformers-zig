// AMX Buffer types for ktransformers-zig
// Ported from ktransformers kt-kernel/operators/amx/la/amx_buffers.hpp

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// Buffer A: Row-major input matrix (M x K)
// ============================================================================

pub fn BufferA(comptime K: type) type {
    return struct {
        ptr: [*]K,
        max_m: usize,
        k: usize,
        m_step: usize,
        k_step: usize,
        k_block: usize,
        n_block: usize,

        pub fn requiredSize(max_m: usize) usize {
            return max_m * @sizeOf(K);
        }

        pub fn init(max_m: usize, k: usize, ptr: *K, m_step: usize, k_step: usize, k_block: usize, n_block: usize) BufferA(K) {
            return BufferA(K){
                .ptr = ptr,
                .max_m = max_m,
                .k = k,
                .m_step = m_step,
                .k_step = k_step,
                .k_block = k_block,
                .n_block = n_block,
            };
        }

        pub fn setData(self: *BufferA(K), new_ptr: *K) void {
            self.ptr = new_ptr;
        }

        /// Pack row-major BF16 matrix into BufferA format
        /// Input: [m, k] row-major, Output: AMX-packed [k_blocks][m_blocks][k_step][m_step]
        pub fn fromMat(self: *BufferA(K), m: usize, src: [*]const K, src_ld: usize,
                       m_step: usize, k_step: usize, k_block: usize) void {
            const m_blocks = (m + m_step - 1) / m_step;
            const k_blocks = (self.k + k_block - 1) / k_block;

            for (0..m_blocks)  | m_blk |  {
                const m_start = m_blk * m_step;
                const m_end = @min(m, m_start + m_step);
                const m_actual = m_end - m_start;

                for (0..k_blocks)  | k_blk |  {
                    const k_start = k_blk * k_block;
                    const k_end = @min(self.k, k_start + k_block);
                    const k_actual = k_end - k_start;

                    for (0..k_actual)  | k_off |  {
                        const k_idx = k_start + k_off;
                        for (0..m_actual)  | m_off |  {
                            const m_idx = m_start + m_off;
                            const dst_idx = k_blk * (m_blocks * m_step * k_step) +
                                           m_blk * (k_step * m_step) +
                                           k_off * m_step +
                                           m_off;
                            self.ptr[dst_idx] = src[m_idx * src_ld + k_idx];
                        }
                    }
                }
            }
        }

        /// Get submatrix pointer for kernel
        pub fn getSubmat(self: *BufferA(K), m: usize, k: usize, m_begin: usize, k_begin: usize) *K {
            const m_blocks = (m + self.m_step - 1) / self.m_step;
            const k_blk_idx = k_begin / self.k_block;
            const k_within = k_begin % self.k_block;
            _ = @min(self.k_block, k - k_begin);
            return self.ptr + k_blk_idx * (m_blocks * self.m_step * self.k_step) +
                           m_begin * (self.k_step * self.m_step) +
                           k_within * self.m_step;
        }
    };
}

// ============================================================================
// Buffer B: Column-major VNNI weight matrix (N x K)
// ============================================================================

pub fn BufferB(comptime K: type) type {
    return struct {
        ptr: [*]K,
        scales: [*]f32,
        n: usize,
        k: usize,
        n_step: usize,
        k_step: usize,
        k_block: usize,
        n_block: usize,
        has_scales: bool,

        pub fn requiredSize(n: usize, k: usize, elem_size: usize, has_scales: bool) usize {
            const scale_size = if (has_scales) n * @sizeOf(f32) else 0;
            return n * k * elem_size + scale_size;
        }

        pub fn init(n: usize, k: usize, ptr: *K, n_step: usize, k_step: usize, k_block: usize, n_block: usize, has_scales: bool) BufferB(K) {
            const scale_offset = n * k;
            return BufferB(K){
                .ptr = ptr,
                .scales = if (has_scales) @ptrCast(@alignCast(ptr + scale_offset)) else undefined,
                .n = n,
                .k = k,
                .n_step = n_step,
                .k_step = k_step,
                .k_block = k_block,
                .n_block = n_block,
                .has_scales = has_scales,
            };
        }

        pub fn setData(self: *BufferB(K), new_ptr: *K) void {
            self.ptr = new_ptr;
            const scale_offset = self.n * self.k;
            self.scales = if (self.has_scales) @ptrCast(@alignCast(new_ptr + scale_offset)) else undefined;
        }

        /// Pack column-major weight matrix into BufferB format with VNNI layout
        pub fn fromMat(self: *BufferB(K), src: [*]const K, src_ld: usize,
                       n_step: usize, k_step: usize, k_block: usize, n_block: usize) void {
            const n_blocks = (self.n + n_block - 1) / n_block;
            const k_blocks = (self.k + k_block - 1) / k_block;

            for (0..n_blocks)  | n_blk |  {
                const n_start = n_blk * n_block;
                const n_end = @min(self.n, n_start + n_block);
                const n_actual = n_end - n_start;

                for (0..k_blocks)  | k_blk |  {
                    const k_start = k_blk * k_block;
                    const k_end = @min(self.k, k_start + k_block);
                    const k_actual = k_end - k_start;

                    for (0..n_actual)  | n_off |  {
                        const n_idx = n_start + n_off;
                        for (0..k_actual)  | k_off |  {
                            const k_idx = k_start + k_off;
                            const dst_idx = n_blk * (self.k * n_block) +
                                           k_blk * (n_block * k_step * n_step) +
                                           n_off * (k_step * n_step) +
                                           k_off * n_step;
                            self.ptr[dst_idx] = src[n_idx * src_ld + k_idx];
                        }
                    }
                }
            }
        }

        /// Pack transposed matrix (for down_proj)
pub fn fromMatTransposed(self: *BufferB(K), src: [*]const K,
                                  n_step: usize, k_step: usize, k_block: usize, n_block: usize) void {
            // Transpose: src is [src_n, src_k], we want [src_k, src_n]
            const n_blocks = (self.n + n_block - 1) / n_block;
            const k_blocks = (self.k + k_block - 1) / k_block;

            for (0..n_blocks)  | n_blk |  {
                const n_start = n_blk * n_block;
                const n_end = @min(self.n, n_start + n_block);
                const n_actual = n_end - n_start;

                for (0..k_blocks)  | k_blk |  {
                    const k_start = k_blk * k_block;
                    const k_end = @min(self.k, k_start + k_block);
                    const k_actual = k_end - k_start;

                    for (0..n_actual)  | n_off |  {
                        const n_idx = n_start + n_off;
                        for (0..k_actual)  | k_off |  {
                            const k_idx = k_start + k_off;
                            // Transposed access: src[k_idx][n_idx]
                            const dst_idx = n_blk * (self.k * n_block) +
                                           k_blk * (n_block * k_step * n_step) +
                                           n_off * (k_step * n_step) +
                                           k_off * n_step;
                            self.ptr[dst_idx] = src[k_idx * self.k + n_idx];
                        }
                    }
                }
            }
        }

        /// Get submatrix pointer
        pub fn getSubmat(self: *BufferB(K), n: usize, k: usize, n_begin: usize, k_begin: usize) *K {
            const n_blk_idx = n_begin / self.n_block;
            const n_within = n_begin % self.n_block;
            const n_blk_size = @min(self.n_block, n - n_blk_idx * self.n_block);
            const k_blk_idx = k_begin / self.k_block;
            const k_within = k_begin % self.k_block;
            _ = @min(self.k_block, k - k_blk_idx * self.k_block);
            return self.ptr + n_blk_idx * (self.k * n_blk_size) +
                           k_blk_idx * (n_blk_size * self.k_step * self.n_step) +
                           n_within * (self.k_step * self.n_step) +
                           k_within * self.n_step;
        }
    };
}

// ============================================================================
// Buffer C: FP32 output matrix (M x N)
// ============================================================================

pub fn BufferC(comptime K: type) type {
    return struct {
        ptr: [*]K,
        max_m: usize,
        n: usize,
        m_step: usize,
        n_step: usize,
        n_block: usize,

        pub fn requiredSize(max_m: usize, n: usize) usize {
            return max_m * n * @sizeOf(K);
        }

        pub fn init(max_m: usize, n: usize, ptr: *K, m_step: usize, n_step: usize, n_block: usize) BufferC(K) {
            return BufferC(K){
                .ptr = ptr,
                .max_m = max_m,
                .n = n,
                .m_step = m_step,
                .n_step = n_step,
                .n_block = n_block,
            };
        }

        pub fn setData(self: *BufferC(K), new_ptr: *K) void {
            self.ptr = new_ptr;
        }

        /// Convert FP32 BufferC to BF16 output matrix
        pub fn toMat(self: *BufferC(K), m: usize, dst: [*]amx.bf16, dst_ld: usize) void {
            const m_blocks = (m + self.m_step - 1) / self.m_step;
            const n_blocks = (self.n + self.n_block - 1) / self.n_block;

            for (0..m_blocks)  | m_blk |  {
                const m_start = m_blk * self.m_step;
                const m_end = @min(m, m_start + self.m_step);
                const m_actual = m_end - m_start;

                for (0..n_blocks)  | n_blk |  {
                    const n_start = n_blk * self.n_block;
                    const n_end = @min(self.n, n_start + self.n_block);
                    const n_actual = n_end - n_start;

                    for (0..m_actual)  | m_off |  {
                        const m_idx = m_start + m_off;
                        for (0..n_actual)  | n_off_step |  {
                            const n_step_idx = n_off_step * self.n_step;
                            const n_end_step = @min(n_actual, n_step_idx + self.n_step);
                            const n_step_actual = n_end_step - n_step_idx;
                            const n_idx = n_start + n_step_idx;

                            const src_idx = m_blk * (m_blocks * self.m_step * self.n_step) +
                                           n_blk * (self.m_step * self.n_step) +
                                           m_off * self.n_step +
                                           n_step_idx;

                            // Vectorized convert FP32 -> BF16
                            for (0..n_step_actual)  | i |  {
                                dst[m_idx * dst_ld + n_idx + i] = amx.f32_to_bf16(self.ptr[src_idx + i]);
                            }
                        }
                    }
                }
            }
        }

        /// Get submatrix pointer
        pub fn getSubmat(self: *BufferC(K), m: usize, n: usize, m_begin: usize, n_begin: usize) *K {
            const m_blocks = (m + self.m_step - 1) / self.m_step;
            const n_blk_idx = n_begin / self.n_block;
            const n_within = n_begin % self.n_block;
            const n_blk_size = @min(self.n_block, n - n_blk_idx * self.n_block);
            return self.ptr + n_blk_idx * (m_blocks * self.m_step * n_blk_size) +
                           m_begin * (self.n_step * n_blk_size) +
                           n_within * self.m_step;
        }
    };
}

// ============================================================================
// Quantization Buffer Helpers
// ============================================================================

/// Quantize BF16 row to INT8 with per-row scale
pub fn quantizeRowBF16ToInt8(src: [*]const amx.bf16, dst: [*]i8, scale: *f32, k: usize) void {
    var max_val: f32 = 0;
    for (0..k)  | i |  {
        const val = amx.bf16_to_f32(src[i]);
        const abs_val = if (val < 0) -val else val;
        if (abs_val > max_val) max_val = abs_val;
    }
    scale.* = if (max_val > 0) max_val / 127.0 else 0;

    for (0..k)  | i |  {
        const val = amx.bf16_to_f32(src[i]);
        const quantized = @as(i8, @intCast(val / scale.* + 0.5));
        dst[i] = if (quantized < -128) -128 else if (quantized > 127) 127 else quantized;
    }
}

/// Dequantize INT8 row to BF16
pub fn dequantizeRowInt8ToBF16(src: [*]const i8, scale: f32, dst: [*]amx.bf16, k: usize) void {
    for (0..k)  | i |  {
        dst[i] = amx.f32_to_bf16(@as(f32, @intCast(src[i])) * scale);
    }
}

/// Quantize BF16 to INT4 (GPTQ style, 2 values per byte)
pub fn quantizeRowBF16ToInt4(src: [*]const amx.bf16, dst: [*]u8, scales: [*]f32, zeros: [*]u8, k: usize, group_size: usize) void {
    const groups = (k + group_size - 1) / group_size;
    for (0..groups)  | g |  {
        const start = g * group_size;
        const end = @min(k, start + group_size);

        var min_val: f32 = std.math.max(f32);
        var max_val: f32 = -std.math.max(f32);
        for (start..end)  | i |  {
            const val = amx.bf16_to_f32(src[i]);
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        const scale = (max_val - min_val) / 15.0;
        const zero = @as(u8, @intCast(-min_val / scale + 0.5));

        scales[g] = scale;
        zeros[g] = zero;

        for (start..end)  | i |  {
            const val = amx.bf16_to_f32(src[i]);
            const quantized = @as(u8, @intCast(val / scale + @as(f32, zero) + 0.5));
            const clamped = if (quantized < 0) 0 else if (quantized > 15) 15 else quantized;
            if (i % 2 == 0) {
                dst[i / 2] = clamped;
            } else {
                dst[i / 2] |= clamped << 4;
            }
        }
    }
}

/// Dequantize INT4 to BF16
pub fn dequantizeRowInt4ToBF16(src: [*]const u8, scales: [*]const f32, zeros: [*]const u8, dst: [*]amx.bf16, k: usize, group_size: usize) void {
    const groups = (k + group_size - 1) / group_size;
    for (0..groups)  | g |  {
        const start = g * group_size;
        const end = @min(k, start + group_size);
        const scale = scales[g];
        const zero = zeros[g];

        for (start..end)  | i |  {
            const byte = src[i / 2];
            const nibble = if (i % 2 == 0) byte & 0xF else byte >> 4;
            const val = (@as(f32, @intCast(nibble)) - @as(f32, @intCast(zero))) * scale;
            dst[i] = amx.f32_to_bf16(val);
        }
    }
}

// ============================================================================
// SIMD Helpers for AVX512
// ============================================================================

/// Copy 32 BF16 elements (512 bits)
pub fn avx512_copy_32xbf16(src: *const amx.bf16, dst: *amx.bf16) void {
    // This would use VMOVDQU32 in real implementation
    @memcpy(dst[0..32], src[0..32]);
}

/// Convert 32 BF16 to 32 FP32 (2 ZMM registers)
pub fn avx512_32xbf16_to_32xfp32(src: *const amx.bf16, dst0: *f32, dst1: *f32) void {
    for (0..16)  | i |  {
        dst0[i] = amx.bf16_to_f32(src[i]);
        dst1[i] = amx.bf16_to_f32(src[i + 16]);
    }
}

/// Convert 32 FP32 to 32 BF16 (2 ZMM registers)
pub fn avx512_32xfp32_to_32xbf16(src0: *const f32, src1: *const f32, dst: *amx.bf16) void {
    for (0..16)  | i |  {
        dst[i] = amx.f32_to_bf16(src0[i]);
        dst[i + 16] = amx.f32_to_bf16(src1[i]);
    }
}

/// Transpose 16x16 32-bit elements (for VNNI layout)
pub fn transpose_16x16_32bit(data: *u32) void {
    // Placeholder for VPERM-based transpose
    _ = data;
}

/// Transpose 16x8 32-bit elements
pub fn transpose_16x8_32bit(data: *u32) void {
    _ = data;
}