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

    /// GEMM with MXFP4 weights and BF16 activations.
    /// a: [m, k] BF16 activations
    /// b: [n, blocks_per_row] MXFP4Block weights (ldb in BLOCKS)
    /// c: [m, n] FP32 output
    ///
    /// AMX path: dequant each 32-nibble block (nibble -> E2M1 value *
    /// block_scale) to BF16 on the fly and reuse the BF16 tile path
    /// (tilebf16dpd -> FP32 C tiles). Block scales are folded into the
    /// values themselves, so no post-scale pass is needed.
    /// Scalar path on non-AMX hosts (identical math).
    /// Nibble order: even k -> low nibble, odd k -> high nibble
    /// (C++ avx2/mxfp4-moe.hpp:314).
    pub fn gemmFullTile(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const MXFP4Block, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        if (!amx.AmxFeatures.available) {
            gemmFullTileScalar(a, lda, b, ldb, c, ldc, m, n, k);
            return;
        }

        config();

        var b_scratch: [2 * TILE_N * TILE_K]amx.bf16 align(64) = undefined;

        for (0..m) |m_begin| {
            for (0..n) |n_begin| {
                const c_tile = c + m_begin * ldc + n_begin;
                cleanC();

                var k_begin: usize = 0;
                while (k_begin < k) : (k_begin += K_STEP) {
                    const a_ptr = a + m_begin * lda + k_begin;
                    GemmKernel224MXFP4.loadA(a_ptr, lda);
                    GemmKernel224MXFP4.loadB(b, ldb, n_begin, k_begin, k, &b_scratch);
                    GemmKernel224MXFP4.runTile();
                }

                GemmKernel224MXFP4.storeC(c_tile, N_STEP * @sizeOf(f32));
            }
        }
    }

    /// Scalar fallback for non-AMX hosts.
    fn gemmFullTileScalar(
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

    // =======================================================================
    // Tile helpers (mirror the MXFP8/FP8/BF16 kernels; strides in BYTES)
    // =======================================================================

    /// Tile config: identical to GemmKernel224BF16 (2x A 16x32 BF16,
    /// 2x B 16x32 BF16 VNNI, 4x C 16x16 FP32).
    pub fn config() void {
        if (!amx.AmxFeatures.available) return;
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
        if (!amx.AmxFeatures.available) return;
        amx.tile_zero(amx.TileReg.tmm4);
        amx.tile_zero(amx.TileReg.tmm5);
        amx.tile_zero(amx.TileReg.tmm6);
        amx.tile_zero(amx.TileReg.tmm7);
    }

    pub fn storeC(c: [*]f32, ldc: usize) void {
        if (!amx.AmxFeatures.available) return;
        amx.tile_stored(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_stored(amx.TileReg.tmm5, @ptrCast(@as([*]f32, @ptrCast(c)) + TILE_N), ldc);
        amx.tile_stored(amx.TileReg.tmm6, @ptrCast(@as([*]f32, @ptrCast(c)) + ldc * TILE_M), ldc);
        amx.tile_stored(amx.TileReg.tmm7, @ptrCast(@as([*]f32, @ptrCast(c)) + ldc * TILE_M + TILE_N), ldc);
    }

    /// Load 32 rows of BF16 activations into tmm0/tmm1. `lda` is the A row
    /// stride in BYTES.
    pub fn loadA(a: [*]const amx.bf16, lda: usize) void {
        if (!amx.AmxFeatures.available) return;
        amx.tile_loadd(amx.TileReg.tmm0, @ptrCast(a), lda);
        amx.tile_loadd(amx.TileReg.tmm1, @ptrCast(@as([*]const amx.bf16, @ptrCast(a)) + TILE_M * lda), lda);
    }

    /// Run the 2x2x4 BF16 dot product (FP32 accumulate).
    pub fn runTile() void {
        if (!amx.AmxFeatures.available) return;
        amx.tile_dpbf16ps(amx.TileReg.tmm4, amx.TileReg.tmm0, amx.TileReg.tmm2);
        amx.tile_dpbf16ps(amx.TileReg.tmm5, amx.TileReg.tmm0, amx.TileReg.tmm3);
        amx.tile_dpbf16ps(amx.TileReg.tmm6, amx.TileReg.tmm1, amx.TileReg.tmm2);
        amx.tile_dpbf16ps(amx.TileReg.tmm7, amx.TileReg.tmm1, amx.TileReg.tmm3);
    }

    /// Dequantize B rows [n_begin, n_begin+TILE_N) over K range
    /// [k_begin, min(k_begin+K_STEP, k)) into the BF16 scratch, then tile-load
    /// tmm2/tmm3. Block scales are folded into the values themselves.
    /// ldb is in BLOCKS per row.
    pub fn loadB(
        b: [*]const MXFP4Block,
        ldb: usize,
        n_begin: usize,
        k_begin: usize,
        k_total: usize,
        scratch: *[2 * TILE_N * TILE_K]amx.bf16,
    ) void {
        if (!amx.AmxFeatures.available) return;
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
                    // Even k -> low nibble, odd k -> high nibble (C++ ref
                    // avx2/mxfp4-moe.hpp:314).
                    const packed_byte = block.qs[in_blk / 2];
                    const nib: u4 = if (in_blk % 2 == 0)
                        @intCast(packed_byte & 0x0F)
                    else
                        @intCast((packed_byte >> 4) & 0x0F);
                    break :blk fp4e2m1_to_f32(nib) * amx.bf16_to_f32(block.scale);
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

// ============================================================================
// Tests (standalone: zig test src/kernels/amx/gemm_224_mxfp4.zig)
// ============================================================================

test "fp4 e2m1 conversion round trip" {
    // FP4 table: index 2 = 1.0, index 10 = -1.0
    try std.testing.expect(fp4e2m1_to_f32(2) == 1.0);
    try std.testing.expect(fp4e2m1_to_f32(10) == -1.0);
    const vals = [_]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -1.0, -6.0 };
    for (vals) |v| {
        const enc = f32_to_fp4e2m1(v);
        try std.testing.expect(fp4e2m1_to_f32(enc) == v);
    }
}

test "MXFP4 scalar GEMM exact values" {
    // M=16, N=16, K=32 (one block per row). A row 0 all 1.0; B all FP4 1.0
    // (nibble 2) with scale 1.0 -> c[0, j] = 32. Other rows 0.
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 32;

    var a: [M * K]amx.bf16 align(64) = undefined;
    var b: [N]MXFP4Block align(64) = undefined;
    var c: [M * N]f32 align(64) = undefined;

    for (0..M) |i| {
        for (0..K) |j| {
            a[i * K + j] = if (i == 0) amx.f32_to_bf16(1.0) else amx.f32_to_bf16(0.0);
        }
    }
    // Every nibble = 2 (FP4 1.0): low nibble 2, high nibble 2 -> byte 0x22.
    for (0..N) |j| {
        b[j].scale = amx.f32_to_bf16(1.0);
        for (0..16) |q| b[j].qs[q] = 0x22;
    }
    for (0..c.len) |i| c[i] = 0.0;

    GemmKernel224MXFP4.gemmFullTileScalar(&a, K, &b, 1, &c, N, M, N, K);

    for (0..M) |i| {
        for (0..N) |j| {
            const expected: f32 = if (i == 0) 32.0 else 0.0;
            try std.testing.expect(@abs(c[i * N + j] - expected) < 1e-4);
        }
    }
}

test "MXFP4 nibble order matches C++ reference" {
    // avx2/mxfp4-moe.hpp:314: even k -> LOW nibble, odd k -> HIGH nibble.
    // Set byte 0 of qs to 0x12 (low=2 -> 1.0, high=1 -> 0.5) with scale 1.0,
    // all other bytes to 0x00 (both nibbles 0). Then A[0,0]=1, A[0,1]=0
    // picks out element 0; A[0,0]=0, A[0,1]=1 picks out element 1.
    var b: MXFP4Block align(64) = undefined;
    b.scale = amx.f32_to_bf16(1.0);
    @memset(&b.qs, 0);
    b.qs[0] = 0x12; // element 0 = low nibble (2 -> 1.0), element 1 = high (1 -> 0.5)

    var unpacked: [32]f32 = undefined;
    unpackMXFP4(&b, &unpacked);
    try std.testing.expect(unpacked[0] == 1.0); // low nibble first
    try std.testing.expect(unpacked[1] == 0.5); // high nibble second
}
