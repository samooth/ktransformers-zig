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
    const rounded_mant: u8 = if (round_bit == 1) @as(u8, @intCast(m3 + 1)) else @as(u8, @intCast(m3));
    const result: u8 = (@as(u8, @intCast(e4m3_exp)) << 3) | (rounded_mant & 0x7);
    return if (sign == 1) result | @as(u8, 0x80) else result;
}

// ============================================================================
// Kernel Configuration for MXFP8
// ============================================================================

pub const GemmKernel224MXFP8 = struct {
    pub const dt = MXFP8Block;
    pub const output_t = f32;
    pub const ELEMENT_SIZE = 1;

    pub const TILE_M = 16;
    pub const TILE_K = 32;  // MXFP8 -> BF16, then BF16 tile path. TILE_K=32 matches BF16 GEMM.
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

    pub const GROUP_SIZE = 32;

    pub fn name() []const u8 { return "MXFP8_E4M3"; }

    pub fn recommended_nth(n: usize) usize {
        return (n + N_BLOCK - 1) / N_BLOCK;
    }

    /// GEMM with MXFP8 weights and BF16 activations.
    /// a: [m, k] BF16 activations
    /// b: [n, blocks_per_row] MXFP8Block weights (ldb in BLOCKS)
    /// c: [m, n] FP32 output
    ///
    /// AMX path: dequant each 32-element block (value * block_scale) to BF16
    /// on the fly and reuse the BF16 tile path (tilebf16dpd -> FP32 C tiles).
    /// Scalar path on non-AMX hosts (identical math).
    pub fn gemmFullTile(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const MXFP8Block, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        if (!amx.detectAmxSupport()) {
            gemmFullTileScalar(a, lda, b, ldb, c, ldc, m, n, k);
            return;
        }

        config();

        // Scratch for one K_STEP=32 column strip of dequantized B (BF16),
        // laid out as 32 rows (2x TILE_N) x 32 K for tmm2/tmm3.
        var b_scratch: [2 * TILE_N * TILE_K]amx.bf16 align(64) = undefined;

        for (0..m) |m_begin| {
            for (0..n) |n_begin| {
                const c_tile = c + m_begin * ldc + n_begin;
                cleanC();

                var k_begin: usize = 0;
                while (k_begin < k) : (k_begin += K_STEP) {
                    const a_ptr = a + m_begin * lda + k_begin;
                    GemmKernel224MXFP8.loadA(a_ptr, lda);
                    GemmKernel224MXFP8.loadB(b, ldb, n_begin, k_begin, k, &b_scratch);
                    GemmKernel224MXFP8.runTile();
                }

                GemmKernel224MXFP8.storeC(c_tile, N_STEP * @sizeOf(f32));
            }
        }
    }

    /// Scalar fallback for non-AMX hosts.
    fn gemmFullTileScalar(
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

    // =======================================================================
    // Tile helpers (mirror the FP8/BF16 kernels; strides in BYTES)
    // =======================================================================

    /// Tile config: identical to GemmKernel224FP8 (2x A 16x32 BF16,
    /// 2x B 16x32 BF16 VNNI, 4x C 16x16 FP32).
    pub fn config() void {
        if (!amx.detectAmxSupport()) return;
        var tile_config = amx.TileConfig.init();

        for (0..2) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), TILE_M, @as(u16, @intCast(TILE_K * @sizeOf(amx.bf16))));
        }
        for (2..4) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), @as(u16, @intCast(TILE_K / VNNI_BLK)), @as(u16, @intCast(TILE_N * VNNI_BLK * @sizeOf(amx.bf16))));
        }
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

    pub fn storeC(c: [*]f32, ldc: usize) void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_stored(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_stored(amx.TileReg.tmm5, @ptrCast(@as([*]f32, @ptrCast(c)) + TILE_N), ldc);
        amx.tile_stored(amx.TileReg.tmm6, @ptrCast(@as([*]f32, @ptrCast(c)) + ldc * TILE_M), ldc);
        amx.tile_stored(amx.TileReg.tmm7, @ptrCast(@as([*]f32, @ptrCast(c)) + ldc * TILE_M + TILE_N), ldc);
    }

    /// Load 32 rows of BF16 activations into tmm0/tmm1. `lda` is the A row
    /// stride in BYTES.
    pub fn loadA(a: [*]const amx.bf16, lda: usize) void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_loadd(amx.TileReg.tmm0, @ptrCast(a), lda);
        amx.tile_loadd(amx.TileReg.tmm1, @ptrCast(@as([*]const amx.bf16, @ptrCast(a)) + TILE_M * lda), lda);
    }

    /// Run the 2x2x4 BF16 dot product (FP32 accumulate).
    pub fn runTile() void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_dpbf16ps(amx.TileReg.tmm4, amx.TileReg.tmm0, amx.TileReg.tmm2);
        amx.tile_dpbf16ps(amx.TileReg.tmm5, amx.TileReg.tmm0, amx.TileReg.tmm3);
        amx.tile_dpbf16ps(amx.TileReg.tmm6, amx.TileReg.tmm1, amx.TileReg.tmm2);
        amx.tile_dpbf16ps(amx.TileReg.tmm7, amx.TileReg.tmm1, amx.TileReg.tmm3);
    }

    /// Dequantize B rows [n_begin, n_begin+TILE_N) over K range
    /// [k_begin, min(k_begin+K_STEP, k)) into the BF16 scratch, then tile-load
    /// tmm2/tmm3. Block scales are folded into the values themselves, so no
    /// post-scale pass is needed (unlike the FP8 kernel's per-row scales).
    /// ldb is in BLOCKS per row.
    pub fn loadB(
        b: [*]const MXFP8Block,
        ldb: usize,
        n_begin: usize,
        k_begin: usize,
        k_total: usize,
        scratch: *[2 * TILE_N * TILE_K]amx.bf16,
    ) void {
        if (!amx.detectAmxSupport()) return;
        const k_end = @min(k_begin + K_STEP, k_total);
        const k_count = k_end - k_begin;

        for (0..TILE_N) |i| {
            const n_idx = n_begin + i;
            for (0..TILE_K) |kk| {
                const val: f32 = if (kk < k_count) blk: {
                    const k_abs = k_begin + kk;
                    const blk_idx = k_abs / GROUP_SIZE;
                    const in_blk = k_abs % GROUP_SIZE;
                    const block = &b[n_idx * ldb + blk_idx];
                    break :blk fp8e4m3_to_f32(block.qs[in_blk]) * amx.bf16_to_f32(block.scale);
                } else 0.0;
                scratch[i * TILE_K + kk] = amx.f32_to_bf16(val);
            }
        }
        for (TILE_N..2 * TILE_N) |i| {
            for (0..TILE_K) |kk| {
                scratch[i * TILE_K + kk] = amx.f32_to_bf16(0.0);
            }
        }
        const stride_bytes: usize = TILE_K * @sizeOf(amx.bf16);
        amx.tile_loadd(amx.TileReg.tmm2, @ptrCast(scratch), stride_bytes);
        amx.tile_loadd(amx.TileReg.tmm3, @ptrCast(scratch[TILE_N * TILE_K ..].ptr), stride_bytes);
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

// ============================================================================
// Tests (standalone: zig test src/kernels/amx/gemm_224_mxfp8.zig)
// ============================================================================

test "fp8 e4m3 conversion round trip" {
    try std.testing.expect(fp8e4m3_to_f32(0x38) == 1.0);
    const vals = [_]f32{ 1.0, 2.0, 0.25, -1.0, 448.0 };
    for (vals) |v| {
        const enc = f32_to_fp8e4m3(v);
        const dec = fp8e4m3_to_f32(enc);
        try std.testing.expectApproxEqRel(v, dec, 0.2);
    }
}

test "MXFP8 scalar GEMM exact values" {
    // M=16, N=16, K=32 (one block per row). A row 0 all 1.0; B all FP8 1.0
    // with scale 1.0 -> c[0, j] = 32. Other rows 0.
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 32;

    var a: [M * K]amx.bf16 align(64) = undefined;
    var b: [N]MXFP8Block align(64) = undefined;
    var c: [M * N]f32 align(64) = undefined;

    for (0..M) |i| {
        for (0..K) |j| {
            a[i * K + j] = if (i == 0) amx.f32_to_bf16(1.0) else amx.f32_to_bf16(0.0);
        }
    }
    for (0..N) |j| {
        b[j].scale = amx.f32_to_bf16(1.0);
        for (0..32) |q| b[j].qs[q] = 0x38; // 1.0
    }
    for (0..c.len) |i| c[i] = 0.0;

    GemmKernel224MXFP8.gemmFullTileScalar(&a, K, &b, 1, &c, N, M, N, K);

    for (0..M) |i| {
        for (0..N) |j| {
            const expected: f32 = if (i == 0) 32.0 else 0.0;
            try std.testing.expect(@abs(c[i * N + j] - expected) < 1e-4);
        }
    }
}

test "MXFP8 block scale applied" {
    // Same as above but scale = 2.0 -> c[0, j] = 64.
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 32;

    var a: [M * K]amx.bf16 align(64) = undefined;
    var b: [N]MXFP8Block align(64) = undefined;
    var c: [M * N]f32 align(64) = undefined;

    for (0..M) |i| {
        for (0..K) |j| {
            a[i * K + j] = if (i == 0) amx.f32_to_bf16(1.0) else amx.f32_to_bf16(0.0);
        }
    }
    for (0..N) |j| {
        b[j].scale = amx.f32_to_bf16(2.0);
        for (0..32) |q| b[j].qs[q] = 0x38;
    }
    for (0..c.len) |i| c[i] = 0.0;

    GemmKernel224MXFP8.gemmFullTileScalar(&a, K, &b, 1, &c, N, M, N, K);

    try std.testing.expect(@abs(c[0] - 64.0) < 1e-3);
    try std.testing.expect(@abs(c[1] - 64.0) < 1e-3);
}
