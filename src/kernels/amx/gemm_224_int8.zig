// GEMM Kernel 224 INT8 for AMX
// Ported from ktransformers kt-kernel/operators/amx/la/amx_kernels.hpp GemmKernel224Int8

const std = @import("std");
const amx = @import("../arch/amx.zig");
const buffers = @import("buffers.zig");

// ============================================================================
// Kernel Configuration
// ============================================================================

pub const GemmKernel224Int8 = struct {
    pub const dt = i8;
    pub const output_t = i32;
    pub const ELEMENT_SIZE = 1;

    pub const TILE_M = 16;
    pub const TILE_K = 64;
    pub const TILE_N = 16;
    pub const VNNI_BLK = 4;

    pub const M_STEP = 32;  // TILE_M * 2
    pub const N_STEP = 32;  // TILE_N * 2
    pub const K_STEP = 64;  // TILE_K

    pub const N_BLOCK = 64;
    pub const K_BLOCK = 3584;

    pub fn name() []const u8 { return "INT8"; }

    pub fn recommended_nth(n: usize) usize {
        return (n + N_BLOCK - 1) / N_BLOCK;
    }

    pub fn split_range_n(_n: usize, ith: usize) struct { n_start: usize, n_end: usize } {
        const n_start = N_BLOCK * ith;
        const n_end = @min(_n, N_BLOCK * (ith + 1));
        return .{ .n_start = n_start, .n_end = n_end };
    }

    // ========================================================================
    // Tile Configuration
    // ========================================================================

    pub fn config() void {
        var tile_config = amx.TileConfig.init();

        // Tile 0,1: A matrices (16 x 64 INT8)
        for (0..2) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), TILE_M, @as(u16, @intCast(TILE_K * @sizeOf(dt))));
        }

        // Tile 2,3: B matrices (16 x 64 INT8 VNNI)
        for (2..4) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), @as(u16, @intCast(TILE_K / VNNI_BLK)), @as(u16, @intCast(TILE_N * VNNI_BLK * @sizeOf(dt))));
        }

        // Tile 4,5,6,7: C matrices (16 x 16 INT32)
        for (4..8) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), TILE_M, @as(u16, @intCast(TILE_N * @sizeOf(output_t))));
        }

        amx.tile_loadconfig(&tile_config);
    }

    // ========================================================================
    // INT8 Preparation (VNNI requires XOR with 0x80 for some backends)
    // ========================================================================

    pub fn prepareA(value: i8) i8 {
        // For oneDNN VNNI backend, XOR with 0x80
        // This is a compile-time option in the original
        return value;
    }

    // ========================================================================
    // Core GEMM Operations
    // ========================================================================

    pub fn loadA(a: *const dt, lda: usize) void {
        amx.tile_loadd(amx.TileReg.tmm0, @ptrCast(a), lda);
        amx.tile_loadd(amx.TileReg.tmm1, @ptrCast(@as([*]const dt, @ptrCast(a)) + TILE_M * lda), lda);
    }

    pub fn loadB(b: *const dt, ldb: usize) void {
        amx.tile_loadd(amx.TileReg.tmm2, @ptrCast(b), ldb);
        amx.tile_loadd(amx.TileReg.tmm3, @ptrCast(@as([*]const dt, @ptrCast(b)) + TILE_N * ldb), ldb);
    }

    pub fn cleanC() void {
        amx.tile_zero(amx.TileReg.tmm4);
        amx.tile_zero(amx.TileReg.tmm5);
        amx.tile_zero(amx.TileReg.tmm6);
        amx.tile_zero(amx.TileReg.tmm7);
    }

    pub fn loadC(c: *output_t, ldc: usize) void {
        amx.tile_loadd(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_loadd(amx.TileReg.tmm5, @ptrCast(@as([*]output_t, @ptrCast(c)) + TILE_N), ldc);
        amx.tile_loadd(amx.TileReg.tmm6, @ptrCast(@as([*]output_t, @ptrCast(c)) + ldc * TILE_M), ldc);
        amx.tile_loadd(amx.TileReg.tmm7, @ptrCast(@as([*]output_t, @ptrCast(c)) + ldc * TILE_M + TILE_N), ldc);
    }

    pub fn storeC(c: *output_t, ldc: usize) void {
        amx.tile_stored(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_stored(amx.TileReg.tmm5, @ptrCast(@as([*]output_t, @ptrCast(c)) + TILE_N), ldc);
        amx.tile_stored(amx.TileReg.tmm6, @ptrCast(@as([*]output_t, @ptrCast(c)) + ldc * TILE_M), ldc);
        amx.tile_stored(amx.TileReg.tmm7, @ptrCast(@as([*]output_t, @ptrCast(c)) + ldc * TILE_M + TILE_N), ldc);
    }

    pub fn runTile() void {
        // C[0] += A[0] * B[0]
        amx.tile_dpbssd(amx.TileReg.tmm4, amx.TileReg.tmm0, amx.TileReg.tmm2);
        // C[1] += A[0] * B[1]
        amx.tile_dpbssd(amx.TileReg.tmm5, amx.TileReg.tmm0, amx.TileReg.tmm3);
        // C[2] += A[1] * B[0]
        amx.tile_dpbssd(amx.TileReg.tmm6, amx.TileReg.tmm1, amx.TileReg.tmm2);
        // C[3] += A[1] * B[1]
        amx.tile_dpbssd(amx.TileReg.tmm7, amx.TileReg.tmm1, amx.TileReg.tmm3);
    }

    // ========================================================================
    // Buffer Types
    // ========================================================================

    pub const BufferA = buffers.BufferA(dt);
    pub const BufferB = buffers.BufferB(dt);
    pub const BufferC = buffers.BufferC(output_t);

    // ========================================================================
    // Quantization Support
    // ========================================================================

    /// Quantize BF16 row to INT8 with per-row scale
    pub fn quantizeRowBF16ToInt8(src: [*]const amx.bf16, dst: [*]i8, scale: *f32, k: usize) void {
        var max_val: f32 = 0;
        for (0..k) |i| {
            const val = amx.bf16_to_f32(src[i]);
            const abs_val = if (val < 0) -val else val;
            if (abs_val > max_val) max_val = abs_val;
        }
        scale.* = if (max_val > 0) max_val / 127.0 else 0;

        for (0..k) |i| {
            const val = amx.bf16_to_f32(src[i]);
            const quantized = @as(i8, @intFromFloat(val / scale.* + 0.5));
            dst[i] = if (quantized < -128) -128 else if (quantized > 127) 127 else quantized;
        }
    }

    /// Dequantize INT8 row to BF16
    pub fn dequantizeRowInt8ToBF16(src: [*]const i8, scale: f32, dst: [*]amx.bf16, k: usize) void {
        for (0..k) |i| {
            dst[i] = amx.f32_to_bf16(@as(f32, @floatFromInt(src[i])) * scale);
        }
    }

    // ========================================================================
    // Full Tile GEMM
    // ========================================================================

    pub fn gemmFullTile(
        a: *const dt, lda: usize,
        b: *const dt, ldb: usize,
        c: *output_t, ldc: usize,
        k: usize
    ) void {
        config();

        var k_processed: usize = 0;
        while (k_processed < k) : (k_processed += K_STEP) {
            const k_remain = k - k_processed;
            const k_step = @min(K_STEP, k_remain);

            if (k_step == K_STEP) {
                loadA(@ptrCast((@as([*]const dt, @ptrCast(a)) + k_processed)), lda);
                loadB(@ptrCast((@as([*]const dt, @ptrCast(b)) + k_processed)), ldb);

                if (k_processed == 0) {
                    cleanC();
                } else {
                    loadC(c, ldc);
                }

                runTile();
                storeC(c, ldc);
            } else {
                gemmPartialTile(@ptrCast(@as([*]const dt, @ptrCast(a)) + k_processed), lda, @ptrCast(@as([*]const dt, @ptrCast(b)) + k_processed), ldb, c, ldc, k_step);
            }
        }
    }

    fn gemmPartialTile(
        a: *const dt, lda: usize,
        b: *const dt, ldb: usize,
        c: *output_t, ldc: usize,
        k_step: usize
    ) void {
        for (0..M_STEP) |m| {
            for (0..N_STEP) |n| {
                var sum: i32 = 0;
                for (0..k_step) |k| {
                    sum += @as(i32, @intCast((@as([*]const i8, @ptrCast(a)))[m * lda + k])) * @as(i32, @intCast((@as([*]const i8, @ptrCast(b)))[n * ldb + k]));
                }
                (@as([*]i32, @ptrCast(c)))[m * ldc + n] += sum;
            }
        }
    }

    // ========================================================================
    // Batched GEMM for MoE
    // ========================================================================

    pub fn batchedGemm(
        experts: []struct { a: *const dt, b: *const dt, c: *output_t },
        _lda: usize, _ldb: usize, _ldc: usize
    ) void {
        for (experts) |exp| {
            gemmFullTile(exp.a, _lda, exp.b, _ldb, exp.c, _ldc, 0);
        }
    }
};

// ============================================================================
// INT8 BufferB with Scales (for quantized weights)
// ========================================================================

/// Extended BufferB with per-row scales for INT8 weights
pub const Int8BufferB = struct {
    ptr: [*]i8,
    scales: [*]f32,
    n: usize,
    k: usize,
    n_step: usize,
    k_step: usize,
    k_block: usize,
    n_block: usize,

    pub fn requiredSize(n: usize, k: usize) usize {
        return n * k * @sizeOf(i8) + n * @sizeOf(f32);
    }

    pub fn init(n: usize, k: usize, ptr: *i8, n_step: usize, k_step: usize, k_block: usize, n_block: usize) Int8BufferB {
        const scale_offset = n * k;
        return Int8BufferB{
            .ptr = @as([*]i8, @ptrCast(ptr)),
            .scales = @as([*]f32, @ptrCast(@alignCast(@as([*]i8, @ptrCast(ptr)) + scale_offset))),
            .n = n,
            .k = k,
            .n_step = n_step,
            .k_step = k_step,
            .k_block = k_block,
            .n_block = n_block,
        };
    }

    /// Pack BF16 weights to INT8 BufferB with quantization
    pub fn fromMatBF16(self: *Int8BufferB, src: [*]const amx.bf16, src_ld: usize) void {
        const n_blocks = (self.n + self.n_block - 1) / self.n_block;
        const k_blocks = (self.k + self.k_block - 1) / self.k_block;

        for (0..n_blocks) |n_blk| {
            const n_start = n_blk * self.n_block;
            const n_end = @min(self.n, n_start + self.n_block);
            const n_actual = n_end - n_start;

            for (0..k_blocks) |k_blk| {
                const k_start = k_blk * self.k_block;
                const k_end = @min(self.k, k_start + self.k_block);
                const k_actual = k_end - k_start;

                for (0..n_actual) |n_off| {
                    const n_idx = n_start + n_off;

                    // Quantize this row
                    const row_src = src[n_idx * src_ld + k_start..][0..k_actual];
                    var row_dst = self.ptr[n_blk * (self.k * self.n_block) +
                                           k_blk * (self.n_block * self.k_step * self.n_step) +
                                           n_off * (self.k_step * self.n_step) ..][0..k_actual];

                    // Use buffers.quantizeRowBF16ToInt8
                    var scale: f32 = 0;
                    buffers.quantizeRowBF16ToInt8(row_src.ptr, row_dst.ptr, &scale, k_actual);
                    self.scales[n_idx] = scale;
                }
            }
        }
    }

    /// Dequantize INT8 BufferB to BF16
    pub fn toMatBF16(self: *Int8BufferB, dst: [*]amx.bf16, dst_ld: usize) void {
        const n_blocks = (self.n + self.n_block - 1) / self.n_block;
        const k_blocks = (self.k + self.k_block - 1) / self.k_block;

        for (0..n_blocks) |n_blk| {
            const n_start = n_blk * self.n_block;
            const n_end = @min(self.n, n_start + self.n_block);
            const n_actual = n_end - n_start;

            for (0..k_blocks) |k_blk| {
                const k_start = k_blk * self.k_block;
                const k_end = @min(self.k, k_start + self.k_block);
                const k_actual = k_end - k_start;

                for (0..n_actual) |n_off| {
                    const n_idx = n_start + n_off;
                    const scale = self.scales[n_idx];

                    const row_src = self.ptr[n_blk * (self.k * self.n_block) +
                                           k_blk * (self.n_block * self.k_step * self.n_step) +
                                           n_off * (self.k_step * self.n_step) ..][0..k_actual];
                    var row_dst = dst[n_idx * dst_ld + k_start..][0..k_actual];

                    buffers.dequantizeRowInt8ToBF16(row_src.ptr, scale, row_dst.ptr, k_actual);
                }
            }
        }
    }
};
