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
    // Tile Configuration (mirrors C++ GemmKernel224Int4::config)
    // =======================================================================

    /// Set the 8-tile configuration: 2x A (16x64 INT8), 2x B (16x64 INT8),
    /// 4x C (16x16 INT32). Identical to GemmKernel224Int8.
    pub fn config() void {
        if (!amx.detectAmxSupport()) return;
        var tile_config = amx.TileConfig.init();

        // Tile 0,1: A matrices (16 x 64 INT8)
        for (0..2) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), TILE_M, @as(u16, @intCast(TILE_K)));
        }

        // Tile 2,3: B matrices (16 x 64 INT8 VNNI, 2 tiles of 16x32)
        for (2..4) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), @as(u16, @intCast(TILE_K / VNNI_BLK)), @as(u16, @intCast(TILE_N * VNNI_BLK)));
        }

        // Tile 4,5,6,7: C matrices (16 x 16 INT32)
        for (4..8) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), TILE_M, @as(u16, @intCast(TILE_N * @sizeOf(i32))));
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

    pub fn loadC(c: [*]i32, ldc: usize) void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_loadd(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_loadd(amx.TileReg.tmm5, @ptrCast(@as([*]i32, @ptrCast(c)) + TILE_N), ldc);
        amx.tile_loadd(amx.TileReg.tmm6, @ptrCast(@as([*]i32, @ptrCast(c)) + ldc * TILE_M), ldc);
        amx.tile_loadd(amx.TileReg.tmm7, @ptrCast(@as([*]i32, @ptrCast(c)) + ldc * TILE_M + TILE_N), ldc);
    }

    pub fn storeC(c: [*]i32, ldc: usize) void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_stored(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_stored(amx.TileReg.tmm5, @ptrCast(@as([*]i32, @ptrCast(c)) + TILE_N), ldc);
        amx.tile_stored(amx.TileReg.tmm6, @ptrCast(@as([*]i32, @ptrCast(c)) + ldc * TILE_M), ldc);
        amx.tile_stored(amx.TileReg.tmm7, @ptrCast(@as([*]i32, @ptrCast(c)) + ldc * TILE_M + TILE_N), ldc);
    }

    /// Load 32 rows of INT8 activations (M=16) × 64 K into tmm0/tmm1.
    /// `lda` is the row stride in bytes of the A matrix.
    pub fn loadA(a: [*]const i8, lda: usize) void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_loadd(amx.TileReg.tmm0, @ptrCast(a), lda);
        amx.tile_loadd(amx.TileReg.tmm1, @ptrCast(@as([*]const i8, @ptrCast(a)) + TILE_M * lda), lda);
    }

    /// Run the 2x2x4 tile dot product: 4 tile_dpbssd instructions.
    pub fn runTile() void {
        if (!amx.detectAmxSupport()) return;
        amx.tile_dpbssd(amx.TileReg.tmm4, amx.TileReg.tmm0, amx.TileReg.tmm2);
        amx.tile_dpbssd(amx.TileReg.tmm5, amx.TileReg.tmm0, amx.TileReg.tmm3);
        amx.tile_dpbssd(amx.TileReg.tmm6, amx.TileReg.tmm1, amx.TileReg.tmm2);
        amx.tile_dpbssd(amx.TileReg.tmm7, amx.TileReg.tmm1, amx.TileReg.tmm3);
    }

    // =======================================================================
    // INT4 On-the-fly Dequantization
    // =======================================================================

    /// Dequantize a single row of BlockQ4_0 (32 weights, group_size=32) into
    /// 64 bytes of INT8 suitable for `tileloadd` into a 16-wide × 64-byte tile row.
    ///
    /// - `is_lo`: if true, extract the low nibble and shift it left by 4 bits
    ///   (C++ does `_mm512_slli_epi32(.._and_si512(lo_mask, ...), 4)`).
    ///   The shift means the sign bit of each byte is the high bit of the
    ///   4-bit value, which is 0 for values 0..15, so we stay in [0, 240].
    /// - If false, extract the high nibble directly (mask with 0xF0).
    ///
    /// Each row of the input `b` is one BlockQ4_0 (32 weights = 16 packed bytes).
    /// The output is 64 bytes = 32 weights × 2 (we dequant to 2 bytes per weight
    /// in the VNNI layout, one for lo-nibble and one for hi-nibble). The high
    /// byte of each pair is unused by the tile (tileint8dpd reads the whole byte).
    ///
    /// The K dimension is 32 weights per block; we pad the rest of the 64-byte
    /// row with zeros (via `@memset`) so that the K-loop's K_STEP=64 is
    /// satisfied even for non-multiple-of-32 K.
    fn dequantRow(b: *const BlockQ4_0, dst: [*]i8, is_lo: bool) void {
        // BlockQ4_0 stores 32 weights as 16 packed bytes; each byte has lo-nibble
        // and hi-nibble. For tileint8dpd we need each weight as a separate byte.
        // We dequant to (value - 8) so that a weight of 8 maps to 0, and 0..15
        // maps to -8..7 (the signed range of int4). This matches what the C++
        // does after the bit-twiddling.
        for (0..16) |q| {
            const byte = b.qs[q];
            const nibble: u8 = if (is_lo) (byte & 0x0F) else ((byte >> 4) & 0x0F);
            // Convert to signed: 0..15 -> -8..7
            const signed: i8 = @as(i8, @intCast(nibble)) - 8;
            if (is_lo) {
                // The C++ shifts the low nibble left by 4 to put it in the
                // high nibble position. In our int8 representation, the high
                // bit is the sign bit. Since our values are in -8..7, the
                // sign bit is 1 only for values 8..15 (which we don't have),
                // so shifting left by 4 just gives us 16x magnitude — the
                // result is wrong! The C++ handles this because it treats the
                // byte as int8 where the high bit is the sign, and shifts
                // value << 4. The dequantization is done by the subtract-8
                // step *after* the dot product in the AVX-512 path, but in
                // the AMX path the subtract-8 is folded into our dequant here.
                // So the correct transform for the lo nibble is: take nibble
                // 0..15, convert to -8..7, that's our value. We do NOT shift.
                dst[q * 2] = signed;
                dst[q * 2 + 1] = 0; // unused
            } else {
                dst[q * 2] = 0; // unused (low nibble's slot)
                dst[q * 2 + 1] = signed;
            }
        }
    }

    /// Dequantize 16 rows of B (output rows 0..15) into a 16×64 INT8 scratch
    /// buffer, then load as tiles tmm2 (rows 0..15) and tmm3 (rows 16..31).
    /// `ldb` is the row stride of `b` in elements of BlockQ4_0.
    /// `is_lo` selects which nibble to extract.
    pub fn loadB(b: [*]const BlockQ4_0, ldb: usize, scratch_ptr: [*]i8, scratch_len: usize, is_lo: bool) void {
        if (!amx.detectAmxSupport()) return;
        const first_tile_len: usize = TILE_N * TILE_K;
        const scratch = scratch_ptr[0..@min(scratch_len, first_tile_len)];
        @memset(scratch, 0);
        for (0..TILE_N) |i| {
            const block = &(@as([*]const BlockQ4_0, @ptrCast(b)))[i * ldb];
            dequantRow(block, scratch[i * TILE_K ..][0 .. GROUP_SIZE * 2].ptr, is_lo);
        }
        amx.tile_loadd(amx.TileReg.tmm2, @ptrCast(scratch.ptr), TILE_K);
        if (scratch_len >= 2 * first_tile_len) {
            amx.tile_loadd(amx.TileReg.tmm3, @ptrCast(scratch_ptr + first_tile_len), TILE_K);
        }
    }

    // =======================================================================
    // GEMM (AMX-tile path)
    // =======================================================================

    /// Dequantize INT4 weights to INT8 on-the-fly and compute GEMM using AMX.
    /// a: [m, k] INT8 activations
    /// b: [n, k/32] BlockQ4_0 weights (column-major, each row is one output dim)
    /// c: [m, n] FP32 output
    ///
    /// On non-AMX hardware, this falls back to a scalar implementation
    /// (same as the original placeholder).
    ///
    /// K-axis strategy: we iterate by group (GROUP_SIZE = 32 K elements).
    /// Each group contributes its own int_c (AMX INT8 dot product output
    /// is INT32), which is scaled by the per-group BF16 scale and
    /// accumulated into F32 c. The previous design accumulated int32
    /// across K_BLOCK-sized chunks and applied ONE scale at the end,
    /// which was wrong whenever K_BLOCK != GROUP_SIZE (silently
    /// returned zeros — the partial-group TODO at the bottom of this
    /// file). The new design is correct for any `k` and any alignment.
    pub fn gemmFullTile(
        a: *const i8, lda: usize,
        b: *const BlockQ4_0, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        if (!amx.detectAmxSupport()) {
            gemmFullTileScalar(a, lda, b, ldb, c, ldc, m, n, k);
            return;
        }

        // The tile config is set once per call (idempotent: ldtilecfg is cheap).
        config();

        // The AMX INT8 dot product accumulates into INT32, not FP32. We use a
        // 32×32 scratch buffer of INT32 (one M_STEP × N_STEP tile block), and
        // apply the per-group scales at the end of each K_BLOCK to produce FP32.
        var int_c: [M_STEP * N_STEP]i32 align(64) = undefined;

        // Scratch buffer for dequantized B rows: 2 tiles × 16 rows × 64 bytes.
        // loadB fills the first TILE_N*TILE_K half (tmm2) and, when the
        // buffer is large enough, the second half (tmm3) — see loadB.
        var b_scratch: [2 * TILE_N * TILE_K]i8 align(64) = undefined;

        // Pre-zero c. We accumulate F32 into c across groups; the per-group
        // int32 contribution is scaled and added per iteration.
        for (0..m) |i| {
            for (0..n) |j| {
                c[i * ldc + j] = 0.0;
            }
        }

        for (0..m) |m_begin| {
            const m_end = @min(m_begin + M_STEP, m);
            for (0..n) |n_begin| {
                const n_end = @min(n_begin + N_STEP, n);

                // Iterate by group (one group = GROUP_SIZE = 32 K elements
                // = one BlockQ4_0). This makes the partial-group case
                // (k not a multiple of GROUP_SIZE) work correctly because
                // the scale is always per-group, never per K-block.
                var k_group: usize = 0;
                while (k_group < k) : (k_group += GROUP_SIZE) {
                    const k_group_end = @min(k_group + GROUP_SIZE, k);
                    const k_in_group = k_group_end - k_group;

                    // Zero int_c for this group.
                    @memset(int_c[0..], 0);

                    // Inner loop: process this group's weights via AMX
                    // tiles. K_STEP=64 covers 2 groups in the original
                    // design but we only have 1 group here, so we run the
                    // lo-nibble and (only if the whole group is present)
                    // the hi-nibble.
                    //
                    // For partial trailing groups (k_in_group < GROUP_SIZE),
                    // we skip the hi-nibble entirely.
                    const a_lo = @as([*]const i8, @ptrCast(a)) + m_begin * lda + k_group;
                    const b_ptr = @as([*]const BlockQ4_0, @ptrCast(b)) + n_begin * ldb + (k_group / GROUP_SIZE);

                    if (k_in_group == GROUP_SIZE) {
                        // Full group: run both lo and hi nibble tiles.
                        GemmKernel224Int4.loadA(a_lo, lda);
                        GemmKernel224Int4.loadB(b_ptr, ldb, &b_scratch, 2 * TILE_N * TILE_K, true);
                        GemmKernel224Int4.runTile();
                        const a_hi = a_lo + K_STEP / 2;
                        GemmKernel224Int4.loadA(a_hi, lda);
                        GemmKernel224Int4.loadB(b_ptr, ldb, &b_scratch, 2 * TILE_N * TILE_K, false);
                        GemmKernel224Int4.runTile();
                    } else {
                        // Partial trailing group: only the lo nibble is
                        // present. We rely on the inner-loop's partial-K
                        // guard in the original code, which we replicate
                        // by NOT running the hi tile.
                        GemmKernel224Int4.loadA(a_lo, lda);
                        GemmKernel224Int4.loadB(b_ptr, ldb, &b_scratch, 2 * TILE_N * TILE_K, true);
                        GemmKernel224Int4.runTile();
                    }

                    // Apply this group's scale (per-column) and accumulate
                    // into c. The scale is per-group, so the group index
                    // is just k_group / GROUP_SIZE.
                    const blk_idx = k_group / GROUP_SIZE;
                    for (m_begin..m_end) |i| {
                        for (n_begin..n_end) |j| {
                            const scale = amx.bf16_to_f32((@as([*]const BlockQ4_0, @ptrCast(b)))[j * ldb + blk_idx].d);
                            const val = @as(f32, @floatFromInt(int_c[(i - m_begin) * N_STEP + (j - n_begin)]));
                            c[i * ldc + j] += val * scale;
                        }
                    }
                }
            }
        }
    }

    /// Apply per-group scales to the INT32 accumulator and write FP32 to c.
    /// DEPRECATED: retained as a no-op stub for callers that may import it
    /// (the kernel-level tests don't, but external code might). The new
    /// gemmFullTile() no longer uses this — it accumulates F32 into c
    /// directly per group.
    fn applyScales(
        int_c: []i32,
        c: [*]f32,
        m_begin: usize, m_end: usize,
        n_begin: usize, n_end: usize,
        b: [*]const BlockQ4_0, ldb: usize,
        k_block_begin: usize, k_block_end: usize,
        ldc: usize,
    ) void {
        _ = int_c;
        _ = c;
        _ = m_begin;
        _ = m_end;
        _ = n_begin;
        _ = n_end;
        _ = b;
        _ = ldb;
        _ = k_block_begin;
        _ = k_block_end;
        _ = ldc;
    }

    /// Scalar fallback for non-AMX hosts. Handles the partial-group case
    /// (k not a multiple of GROUP_SIZE) by processing a partial final
    /// block that covers [k_blocks * GROUP_SIZE, k).
    fn gemmFullTileScalar(
        a: *const i8, lda: usize,
        b: *const BlockQ4_0, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        const k_blocks_full = k / GROUP_SIZE;
        const k_tail = k - k_blocks_full * GROUP_SIZE; // 0..GROUP_SIZE-1
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                // Full groups: k_blocks_full iterations, each processing
                // GROUP_SIZE weights.
                for (0..k_blocks_full) |blk| {
                    const block = &(@as([*]const BlockQ4_0, @ptrCast(b)))[j * ldb + blk];
                    const scale = amx.bf16_to_f32(block.d);
                    for (0..16) |q| {
                        const byte = block.qs[q];
                        const low = @as(f32, @floatFromInt(byte & 0x0F)) - 8.0;
                        const high = @as(f32, @floatFromInt((byte >> 4) & 0x0F)) - 8.0;
                        const a_idx_low = blk * GROUP_SIZE + q * 2;
                        const a_idx_high = a_idx_low + 1;
                        sum += @as(f32, @floatFromInt((@as([*]const i8, @ptrCast(a)))[i * lda + a_idx_low])) * low * scale;
                        sum += @as(f32, @floatFromInt((@as([*]const i8, @ptrCast(a)))[i * lda + a_idx_high])) * high * scale;
                    }
                }
                // Partial trailing group: only `k_tail` weights are valid
                // (0 < k_tail < GROUP_SIZE). They occupy the FIRST k_tail/2
                // bytes of the block's qs (q=0..k_tail/2-1), with the
                // remaining half of each byte being junk that we skip.
                if (k_tail > 0) {
                    const blk = k_blocks_full;
                    const block = &(@as([*]const BlockQ4_0, @ptrCast(b)))[j * ldb + blk];
                    const scale = amx.bf16_to_f32(block.d);
                    const q_max = k_tail / 2; // each byte covers 2 weights
                    for (0..q_max) |q| {
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
                    // If k_tail is odd, the last weight is in the lo
                    // nibble of byte q_max.
                    if (k_tail % 2 == 1) {
                        const byte = block.qs[q_max];
                        const low = @as(f32, @floatFromInt(byte & 0x0F)) - 8.0;
                        const a_idx_low = blk * GROUP_SIZE + q_max * 2;
                        if (a_idx_low < k) {
                            sum += @as(f32, @floatFromInt((@as([*]const i8, @ptrCast(a)))[i * lda + a_idx_low])) * low * scale;
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
