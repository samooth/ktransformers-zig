// GEMM Kernel 224 IQ3_XXS (GGML non-linear 3.0625-bpw grid quant)
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256):
//   block_iq3_xxs {
//     ggml_half d;              // 2 bytes: global super-block scale
//     uint8_t qs[3*QK_K/8];     // 96 bytes: 64 grid indices (8 per group)
//                                //   + 32 bytes scales+signs (4 per 32-group)
//   } = 98 bytes per 256 weights
//
// Weight reconstruction (ggml-quants.c:2575):
//   Per 32-weight group (8 sub-blocks of 8):
//     aux32 = 4 bytes from scales_and_signs area (qs[64+4*ib32 .. +4])
//     db = d * (0.5 + (aux32 >> 28)) * 0.5  // 4-bit scale in top nibble
//     Per sub-block l (0..3):
//       signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]  // shared sign palette
//       grid1 = iq3xxs_grid[qs[2*l]]    // u32: 4 magnitude bytes (first 4 weights)
//       grid2 = iq3xxs_grid[qs[2*l+1]]  // u32: 4 magnitude bytes (last 4 weights)
//       y[j+0] = db * grid1[j] * sign; y[j+4] = db * grid2[j] * sign
//
// The 256-entry iq3xxs_grid is a u32 table (4 packed magnitude bytes per
// entry from the 8-value alphabet {0x04..0x3e}); ksigns_iq2xs/kmask_iq2xs
// are the SAME tables as IQ2_XXS (shared sign palette + mask).

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// IQ3_XXS Block Structure (byte-exact vs ggml's block_iq3_xxs)
// ============================================================================

pub const QK_K: usize = 256;
const N_GROUPS: usize = QK_K / 32; // = 8 groups of 32

pub const BlockIQ3_XXS = extern struct {
    d: u16, // f16 global scale
    qs: [3 * QK_K / 8]u8, // 64 grid indices + 32 scales+signs
};

comptime {
    if (@sizeOf(BlockIQ3_XXS) != 98) @compileError("BlockIQ3_XXS must be 98 bytes");
}

// ============================================================================
// Lookup tables (byte-exact from ggml-common.h)
// ============================================================================

/// kmask_iq2xs: shared with IQ2_XXS (bit j applies to weight j).
pub const KMASK_IQ2XS = [8]u8{ 1, 2, 4, 8, 16, 32, 64, 128 };

/// ksigns_iq2xs: shared with IQ2_XXS (128-entry 7-bit sign palette).
pub const KSIGNS_IQ2XS = [128]u8{
    0,   129, 130, 3,   132, 5,   6,   135, 136, 9,   10,  139, 12,  141, 142, 15,
    144, 17,  18,  147, 20,  149, 150, 23,  24,  153, 154, 27,  156, 29,  30,  159,
    160, 33,  34,  163, 36,  165, 166, 39,  40,  169, 170, 43,  172, 45,  46,  175,
    48,  177, 178, 51,  180, 53,  54,  183, 184, 57,  58,  187, 60,  189, 190, 63,
    192, 65,  66,  195, 68,  197, 198, 71,  72,  201, 202, 75,  204, 77,  78,  207,
    80,  209, 210, 83,  212, 85,  86,  215, 216, 89,  90,  219, 92,  221, 222, 95,
    96,  225, 226, 99,  228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

/// iq3xxs_grid: 256-entry u32 table. Each entry is 4 packed magnitude
/// bytes (one per weight half-sub-block) from the 8-value alphabet
/// {0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e}.
pub const IQ3XXS_GRID = [256]u32{
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c,
    0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c,
    0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14,
    0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404,
    0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c,
    0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424,
    0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c,
    0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c,
    0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e,
    0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c,
    0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404,
    0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424,
    0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c,
    0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c,
    0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04,
    0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404,
    0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c,
    0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c,
    0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14,
    0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c,
    0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e,
    0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c,
    0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e,
    0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c,
    0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04,
    0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434,
    0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c,
    0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424,
    0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04,
    0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c,
    0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14,
    0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c,
    0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// ============================================================================
// f16 <-> f32 (native f16)
// ============================================================================

pub fn f32_to_f16(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

pub fn f16_to_f32(x: u16) f32 {
    const h: f16 = @bitCast(x);
    return @floatCast(h);
}

// ============================================================================
// Dequantize (byte-exact vs ggml-quants.c:2575)
// ============================================================================

/// k must be a multiple of 256.
pub fn dequantizeRowIQ3_XXS(x: [*]const BlockIQ3_XXS, y: [*]f32, k: usize) void {
    const nb = k / QK_K;
    var yoff: usize = 0;

    for (0..nb) |i| {
        const blk = &x[i];
        const d = f16_to_f32(blk.d);
        // First 64 bytes of qs are grid indices; next 32 are scales+signs
        const grid_area: [*]const u8 = &blk.qs;
        const scale_area: [*]const u8 = blk.qs[64..].ptr;

        for (0..N_GROUPS) |ib32| {
            // Read 4 bytes of scale+sign data as a u32
            var aux32: u32 = undefined;
            const sa: [*]u8 = @ptrCast(&aux32);
            @memcpy(sa[0..4], scale_area[ib32 * 4 .. ib32 * 4 + 4]);

            // db = d * (0.5 + (aux32 >> 28)) * 0.5
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32 >> 28))) * 0.5;

            for (0..4) |l| {
                const signs: u8 = KSIGNS_IQ2XS[(aux32 >> @intCast(7 * l)) & 127];
                const g1_idx: usize = grid_area[ib32 * 8 + 2 * l];
                const g2_idx: usize = grid_area[ib32 * 8 + 2 * l + 1];
                const grid1: u32 = IQ3XXS_GRID[g1_idx];
                const grid2: u32 = IQ3XXS_GRID[g2_idx];
                const g1_bytes: [*]const u8 = @ptrCast(&grid1);
                const g2_bytes: [*]const u8 = @ptrCast(&grid2);

                for (0..4) |j| {
                    const m1: f32 = @floatFromInt(g1_bytes[j]);
                    const s1: f32 = if ((signs & KMASK_IQ2XS[j]) != 0) -1.0 else 1.0;
                    y[yoff + j] = db * m1 * s1;
                    const m2: f32 = @floatFromInt(g2_bytes[j]);
                    const s2: f32 = if ((signs & KMASK_IQ2XS[j + 4]) != 0) -1.0 else 1.0;
                    y[yoff + j + 4] = db * m2 * s2;
                }
                yoff += 8;
            }
        }
    }
}

/// Dequantize IQ3_XXS to BF16.
pub fn dequantizeRowIQ3_XXSToBF16(x: [*]const BlockIQ3_XXS, y: [*]amx.bf16, k: usize) void {
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowIQ3_XXS(x + i, &f, QK_K);
        for (0..QK_K) |j| y[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x IQ3_XXS weights -> F32)
// ============================================================================

pub fn gemmIQ3_XXSScalar(
    a: [*]const amx.bf16,
    b: [*]const BlockIQ3_XXS,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    const nb = k / QK_K;
    var scratch: [QK_K]f32 = undefined;
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..nb) |blk| {
                dequantizeRowIQ3_XXS(b + j * ldb + blk, &scratch, QK_K);
                for (0..QK_K) |k0| {
                    sum += amx.bf16_to_f32(a[i * lda + k0]) * scratch[k0];
                }
            }
            c[i * ldc + j] = sum;
        }
    }
}

// ============================================================================
// Quantize — lives in iq3_quantize.zig (quantizeRowIQ3_XXS_WithInit),
// which owns the kmap plumbing (iq3xs_init) and this file's block type.
// The former convenience wrapper here was REMOVED: it created a
// gemm_224_iq3_xxs <-> iq3_quantize import cycle (sentrux rule
// max_cycles=0 caught it), declared a std.Thread.Mutex that does not
// exist in this std (std.atomic.Mutex — latent, hidden by lazy
// analysis), and had zero callers. Callers use the WithInit form with
// initIq3XsData (process-cached since d0b0211).
