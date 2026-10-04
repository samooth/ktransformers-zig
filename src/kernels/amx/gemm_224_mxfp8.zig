// GEMM Kernel 224 MXFP8 for AMX/AVX512
// MXFP8: Microscaling FP8 format (E4M3 or E5M2) with per-block scales
// Used in MiniMax M3 and other models

const std = @import("std");
const amx = @import("../arch/amx.zig");
const buffers = @import("buffers.zig");
const fp8 = @import("gemm_224_fp8.zig");

// ============================================================================
// MXFP8 Type (E4M3 with microscaling)
// ============================================================================

/// Reuse FP8 E4M3 from fp8 kernel: the alias now lives with the rest of the
/// imported codec further down, next to the other re-exports.

/// MXFP8 block: 32 FP8 weights + 1 scale (BF16) = 34 bytes
/// MXFP8 group: 32 FP8 E4M3 values sharing one scale.
///
/// The scale is **ue8m0** (a.k.a. e8m0) — ONE byte, an unbiased power of two
/// — not a bf16. This used to be `scale: amx.bf16`, which made the block 34
/// bytes against the format's 33 and used a scale encoding that does not
/// exist in MXFP. Consequence: these blocks could not describe a real
/// MXFP8 checkpoint at all. Same failure mode as the KT_TYPE/block-size
/// regression in eec0c22 (11 invented enum values, 9 wrong block sizes),
/// which is why `@sizeOf` is now pinned by a test below.
///
/// Layout matches the C++ reference (kt-kernel/operators/avx2/mxfp8-moe.hpp:
/// "Scale: ue8m0 per-group (group_size=32)") and GGUF's MXFP8 tensor type.
pub const MXFP8Block = extern struct {
    scale: u8, // ue8m0
    qs: [32]fp8_e4m3,
};

comptime {
    // 1 scale byte + 32 payload bytes. If this fires, the layout drifted and
    // every already-serialized checkpoint becomes unreadable.
    if (@sizeOf(MXFP8Block) != 33) @compileError("MXFP8Block must be 33 bytes (ue8m0 + 32 x E4M3)");
}

/// ue8m0 -> f32: byte `b` encodes 2^(b-127).
///
/// A pure shift, no table: setting the f32 exponent field to `b` with a zero
/// mantissa gives exactly 2^(b-127). Verbatim from the reference's
/// `convert_ue8m0_to_fp32`, including its two documented edges — b=0 yields
/// 0.0 rather than the 2^-127 a strict reading wants, and b=255 overflows the
/// f32 exponent into NaN. Matching the reference bit-for-bit is what makes a
/// cross-implementation accuracy diff meaningful, so neither is "fixed" here.
pub inline fn ue8m0_to_f32(b: u8) f32 {
    return @bitCast(@as(u32, b) << 23);
}

/// f32 -> ue8m0: round the magnitude to the nearest power of two, then encode
/// its exponent as an unbiased byte. Inverse of `ue8m0_to_f32` over the
/// representable range; out-of-range inputs saturate to the byte endpoints.
pub fn f32_to_ue8m0(x: f32) u8 {
    if (!(x > 0.0)) return 0; // also catches NaN
    const e = std.math.ilogb(x); // floor(log2(|x|))
    const m = std.math.scalbn(x, -e); // in [1, 2)
    const rounded = if (m >= 1.5) e + 1 else e;
    return @intCast(std.math.clamp(rounded + 127, 0, 255));
}

// ============================================================================
// E4M3 codec — IMPORTED, NOT COPIED
// ============================================================================
//
// This file used to own a second copy of `FP8_E4M3_LUT`, `fp8e4m3_to_f32` and
// `f32_to_fp8e4m3`, and the two copies drifted: the MX encoder was rewritten
// (correctly, four real bugs) while `gemm_224_fp8.zig` kept all four of its own,
// and the MX decoder's replacement was written out again there. Two editable
// copies of a codec is a future bug report, so `gemm_224_fp8.zig` is now the
// single owner and this file re-exports its names, which keeps every existing
// caller (`root.zig`, `src/main.zig`, the test suites) working unchanged.
//
// Same-layer data sharing is the pattern the sentrux rules describe for the
// grid/lookup tables (`gemm_224_iq1_s.zig` owns `IQ1S_GRID` for its siblings);
// this is the same thing one level up.

/// E4M3 -> f32, via the 256-entry table. Owned by `gemm_224_fp8.zig`, which
/// carries the rationale: the bit-reconstruction decoder this replaced was
/// wrong for the subnormals (exp field 0 read as normal, up to 1024x too large)
/// and for 0x7F, the only NaN encoding, which it returned as 480.0 — above
/// E4M3_MAX and therefore not even representable.
pub const fp8_e4m3 = fp8.fp8_e4m3;
pub const E4M3_MAX = fp8.E4M3_MAX;
pub const FP8_E4M3_LUT = fp8.FP8_E4M3_LUT;
pub const fp8e4m3_to_f32 = fp8.fp8e4m3_to_f32;
/// f32 -> E4M3, round-to-nearest-even, saturating at 0x7E. The four bugs it
/// used to carry in THIS file are fixed in the copy that is now the owner; see
/// `gemm_224_fp8.zig::f32_to_fp8e4m3` for them, and TODO.md for the case that
/// exposed the remaining copy (500.0 -> 0x7F -> 0.0).
pub const f32_to_fp8e4m3 = fp8.f32_to_fp8e4m3;

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
            gemmAvx2(a, lda, b, ldb, c, ldc, m, n, k);
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

    // ========================================================================
    // AVX2 path (non-AMX hosts)
    // ========================================================================
    //
    // `gemmFullTileScalar` is an m/n/k triple loop, which is the one loop order
    // the SIMD contract forbids: the weight matrix — the huge operand — is
    // re-streamed from DRAM once per token row. This is the replacement, and
    // it is built the same way as `kquant_simd.gemmBF16xQuant`:
    //
    //   n-outer / m-inner  the n×k weight block is walked once, in order
    //   MR = 4             four activation rows share one unpacked weight
    //                       vector, so the LUT lookups amortize over 4 FMAs
    //   explicit @mulAdd   FP contraction is off in this toolchain, so an
    //                       `a*w + acc` would be a separate multiply and add
    //   deferred hsum      ONE @reduce per output row, not per 32-weight group.
    //                       The reference makes the same point about its own
    //                       kernel: "~10x fewer hsums than per-group hsum"
    //                       (mxfp8-moe.hpp), because MXFP keeps the scale
    //                       separate from the payload rather than baking it
    //                       into the dequant.
    //
    // Folding the scale into the weight vector is EXACT: ue8m0 is a power of
    // two, so scaling by it only changes the exponent and cannot round.

    const MR: usize = 4;
    const Vec: type = @Vector(8, f32);

    /// 8 bf16 -> 8 f32. Single pointer parameter and `align(1)`: an
    /// activation row can start at any offset.
    inline fn widen8(p: [*]const amx.bf16) Vec {
        return amx.widen8(p);
    }

    /// 8 E4M3 -> 8 f32 via the table. Single pointer parameter because
    /// `&block.qs[i]` sits at offset 1 in the block (after the scale byte) and
    /// so is only 1-byte aligned.
    inline fn fp8x8(p: *const fp8_e4m3) Vec {
        const raw: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(p)).*;
        const idx: @Vector(8, u32) = @intCast(raw);
        var out: [8]f32 = undefined;
        inline for (0..8) |i| out[i] = FP8_E4M3_LUT[idx[i]];
        return out;
    }

    /// Scalar dot over the ragged tail of a k that is not a multiple of
    /// GROUP_SIZE. Only the last group can be partial.
    inline fn tailDot(a_row: [*]const amx.bf16, blk: *const MXFP8Block, kbase: usize, rem: usize) f32 {
        const s = ue8m0_to_f32(blk.scale);
        var acc: f32 = 0;
        for (0..rem) |t| {
            acc += amx.bf16_to_f32(a_row[kbase + t]) * fp8e4m3_to_f32(blk.qs[t]) * s;
        }
        return acc;
    }

    /// AVX2 MXFP8 GEMM. `c` is OVERWRITTEN (matches `gemmFullTileScalar`, which
    /// assigns rather than accumulates).
    pub fn gemmAvx2(
        a: [*]const amx.bf16, lda: usize,
        b: [*]const MXFP8Block, ldb: usize,
        c: [*]f32, ldc: usize,
        m: usize, n: usize, k: usize
    ) void {
        const full = k / GROUP_SIZE; // complete 32-weight groups
        const rem = k - full * GROUP_SIZE; // ragged tail, < 32

        var j: usize = 0;
        while (j < n) : (j += 1) {
            const brow = b + j * ldb;

            var i: usize = 0;
            while (i + MR <= m) : (i += MR) {
                var acc = [_]Vec{@splat(0.0)} ** MR;
                for (0..full) |blk| {
                    const blk_ptr = &brow[blk];
                    const sv: Vec = @splat(ue8m0_to_f32(blk_ptr.scale));
                    const kb = blk * GROUP_SIZE;
                    inline for (0..GROUP_SIZE / 8) |v| {
                        const w = fp8x8(&blk_ptr.qs[v * 8]) * sv;
                        inline for (0..MR) |r| {
                            acc[r] = @mulAdd(Vec, widen8(a + (i + r) * lda + kb + v * 8), w, acc[r]);
                        }
                    }
                }
                inline for (0..MR) |r| {
                    var sum = @reduce(.Add, acc[r]); // ONE hsum per output element
                    if (rem != 0) sum += tailDot(a + (i + r) * lda, &brow[full], full * GROUP_SIZE, rem);
                    c[(i + r) * ldc + j] = sum;
                }
            }

            // Row tail (m % MR), one row at a time.
            while (i < m) : (i += 1) {
                var acc: Vec = @splat(0.0);
                for (0..full) |blk| {
                    const blk_ptr = &brow[blk];
                    const sv: Vec = @splat(ue8m0_to_f32(blk_ptr.scale));
                    const kb = blk * GROUP_SIZE;
                    inline for (0..GROUP_SIZE / 8) |v| {
                        acc = @mulAdd(Vec, widen8(a + i * lda + kb + v * 8), fp8x8(&blk_ptr.qs[v * 8]) * sv, acc);
                    }
                }
                var sum = @reduce(.Add, acc);
                if (rem != 0) sum += tailDot(a + i * lda, &brow[full], full * GROUP_SIZE, rem);
                c[i * ldc + j] = sum;
            }
        }
    }

    /// Scalar reference for non-AMX hosts: the readable statement of the
    /// math, and the oracle the AVX2 path is tested against. Public for the
    /// same reason `gemmQ*_KScalar` is: the live equivalence test lives in
    /// tests/kernels/simd_kernels_tests.zig and has to reach both sides.
    pub fn gemmFullTileScalar(
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
                    const scale = ue8m0_to_f32(block.scale);
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
                    break :blk fp8e4m3_to_f32(block.qs[in_blk]) * ue8m0_to_f32(block.scale);
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
                block.scale = f32_to_ue8m0(scale);

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
        b[j].scale = 127; // ue8m0 for 2^(127-127) = 1.0
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
        b[j].scale = 128; // ue8m0 for 2^(128-127) = 2.0
        for (0..32) |q| b[j].qs[q] = 0x38;
    }
    for (0..c.len) |i| c[i] = 0.0;

    GemmKernel224MXFP8.gemmFullTileScalar(&a, K, &b, 1, &c, N, M, N, K);

    try std.testing.expect(@abs(c[0] - 64.0) < 1e-3);
    try std.testing.expect(@abs(c[1] - 64.0) < 1e-3);
}
