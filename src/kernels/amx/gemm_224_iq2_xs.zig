// GEMM Kernel 224 IQ2_XS (GGML non-linear 2.3125-bpw grid quant)
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256):
//   block_iq2_xs {
//     ggml_half d;              // 2 bytes: global super-block scale
//     uint16_t qs[QK_K/8];      // 64 bytes: per-8 grid_index(9b) | signs(7b)<<9
//     uint8_t  scales[QK_K/32]; // 8 bytes: per-16 4-bit scale, packed 2/byte
//   } = 74 bytes per 256 weights
//
// Weight reconstruction (ggml-quants.c:2516):
//   Per 16-weight group (ib32), 2 sub-blocks of 8:
//     db[l/2] = d * (0.5 + (scales[ib32] >> 4*(l/2) & 0xf)) * 0.25
//     Per sub-block l (0..1):
//       grid  = iq2xs_grid[qs[4*ib32+l] & 511]   // u64: 8 magnitude bytes
//       signs = ksigns_iq2xs[qs[4*ib32+l] >> 9]
//       y[j] = db[l/2] * grid[j] * sign        // j = 0..7
//
// vs IQ2_XXS (gemm_224_iq2_xxs.zig):
//   - grid is 512 entries (9-bit index) not 256
//   - groups are 16 weights (2 sub-blocks), not 32 (4 sub-blocks)
//   - the per-16 scale is a SEPARATE u8 array (4-bit dual nibble),
//     not packed into the qs u32 words
//   - db factor is 0.25, not 0.5
//
// The kmap for quantize is the same 43692-entry 2-bit kmap, but built
// from KGRID_2BIT_512 (iq2xs_init.initIq2XsData512).

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// IQ2_XS Block Structure (byte-exact vs ggml's block_iq2_xs)
// ============================================================================

pub const QK_K: usize = 256;
const N_GROUPS: usize = QK_K / 16; // = 16 groups of 16

pub const BlockIQ2_XS = extern struct {
    d: u16, // f16 global scale
    qs: [QK_K / 8]u16, // 32 x (grid_index | signs<<9)
    scales: [QK_K / 32]u8, // 8 bytes: 16 x 4-bit, packed 2/byte
};

comptime {
    if (@sizeOf(BlockIQ2_XS) != 74) @compileError("BlockIQ2_XS must be 74 bytes");
}

// ============================================================================
// Lookup tables (byte-exact from ggml-common.h / shared with IQ2_XXS)
// ============================================================================

/// kmask_iq2xs / ksigns_iq2xs shared with IQ2_XXS.
const iq2xxs = @import("gemm_224_iq2_xxs.zig");
pub const KMASK_IQ2XS = iq2xxs.KMASK_IQ2XS;
pub const KSIGNS_IQ2XS = iq2xxs.KSIGNS_IQ2XS;

/// iq2xs_grid: 512-entry u64 table (8 packed magnitude bytes per entry,
/// alphabet {0x08, 0x19, 0x2b} — same as IQ2_XXS but denser).
pub const IQ2XS_GRID = [512]u64{
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
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
// Dequantize (byte-exact vs ggml-quants.c:2516)
// ============================================================================

/// k must be a multiple of 256.
pub fn dequantizeRowIQ2_XS(x: [*]const BlockIQ2_XS, y: [*]f32, k: usize) void {
    const nb = k / QK_K;
    var yoff: usize = 0;

    for (0..nb) |i| {
        const blk = &x[i];
        const d = f16_to_f32(blk.d);

        for (0..QK_K / 32) |ib32| {
            const db_lo = d * (0.5 + @as(f32, @floatFromInt(blk.scales[ib32] & 0xf))) * 0.25;
            const db_hi = d * (0.5 + @as(f32, @floatFromInt(blk.scales[ib32] >> 4))) * 0.25;

            for (0..4) |l| {
                const db = if (l / 2 == 0) db_lo else db_hi;
                const word: u16 = blk.qs[4 * ib32 + l];
                const grid: u64 = IQ2XS_GRID[word & 511];
                const signs: u8 = KSIGNS_IQ2XS[(word >> 9) & 127];
                const grid_bytes: [*]const u8 = @ptrCast(&grid);

                for (0..8) |j| {
                    const m: f32 = @floatFromInt(grid_bytes[j]);
                    const s: f32 = if ((signs & KMASK_IQ2XS[j]) != 0) -1.0 else 1.0;
                    y[yoff + j] = db * m * s;
                }
                yoff += 8;
            }
        }
    }
}

/// Dequantize IQ2_XS to BF16.
pub fn dequantizeRowIQ2_XSToBF16(x: [*]const BlockIQ2_XS, y: [*]amx.bf16, k: usize) void {
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowIQ2_XS(x + i, &f, QK_K);
        for (0..QK_K) |j| y[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x IQ2_XS weights -> F32)
// ============================================================================

pub fn gemmIQ2_XSScalar(
    a: [*]const amx.bf16,
    b: [*]const BlockIQ2_XS,
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
                dequantizeRowIQ2_XS(b + j * ldb + blk, &scratch, QK_K);
                for (0..QK_K) |k0| {
                    sum += amx.bf16_to_f32(a[i * lda + k0]) * scratch[k0];
                }
            }
            c[i * ldc + j] = sum;
        }
    }
}

// ============================================================================
// Quantize — full implementation (kmap from iq2xs_init.initIq2XsData512)
// ============================================================================
//
// Port of ggml-quants.c:3472-3650 (quantize_row_iq2_xs_impl).
// vs IQ2_XXS quantize:
//   - 16-weight groups (2 sub-blocks of 8), not 32 (4 sub-blocks)
//   - ±9-scale search (not ±6), same kMaxQ=3
//   - q2[2*(QK_K/16)] u16: grid_index | signs<<9 (per sub-block)
//   - the per-16 scale is NOT packed into q2 — it lives in y.scales[]
//     as 4-bit pairs (low nibble = even group, high = odd)

const init_mod = @import("iq2xs_init.zig");

fn nearestInt(fval: f32) i32 {
    std.debug.assert(@abs(fval) <= 4194303.0);
    const val: f32 = fval + 12582912.0;
    const i: i32 = @bitCast(val);
    return (i & 0x007f_ffff) - 0x0040_0000;
}

fn findBestNeighbour(
    data: *const init_mod.Iq2GridData,
    neighbors_start: usize,
    xval: []const f32,
    weight: []const f32,
    scale: f32,
    L: [*]u8,
) usize {
    const num_neighbors = data.kneighbors[neighbors_start];
    std.debug.assert(num_neighbors > 0);
    var best_d2: f32 = std.math.floatMax(f32);
    var grid_index: usize = std.math.maxInt(usize);
    for (1..num_neighbors + 1) |j| {
        const nb = data.kneighbors[neighbors_start + j];
        const gpos = data.grid[nb];
        var d2: f32 = 0;
        for (0..8) |i| {
            const q: f32 = @floatFromInt(gpos[i]);
            const diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = nb;
        }
    }
    std.debug.assert(grid_index != std.math.maxInt(usize));
    const gpos = data.grid[grid_index];
    for (0..8) |i| L[i] = @intCast(@divTrunc(gpos[i] - 1, 2));
    return grid_index;
}

const KMAX: i32 = 3;
const GROUP_MAX_EPS: f32 = 1e-15;

/// Weighted LS fit of 0..nmax-1 levels (the makeQPQuants shared recipe —
/// same as IQ2_XXS; duplicated here with the 16-wide shapes inlined).
fn makeQPQuants16(n: usize, nmax: i32, x: []const f32, L: []u8, qw: []const f32) f32 {
    var max: f32 = 0;
    for (0..n) |i| max = @max(max, x[i]);
    if (max < GROUP_MAX_EPS) {
        for (0..n) |i| L[i] = 0;
        return 0.0;
    }
    var iscale = @as(f32, @floatFromInt(nmax)) / max;
    for (0..n) |i| L[i] = @intCast(std.math.clamp(nearestInt(iscale * x[i]), 0, nmax));
    const scale = 1.0 / iscale;
    var best_mse: f32 = 0;
    for (0..n) |i| {
        const diff = x[i] - scale * @as(f32, @floatFromInt(L[i]));
        const w = qw[i];
        best_mse += w * diff * diff;
    }
    var is: i32 = -4;
    while (is <= 4) : (is += 1) {
        if (is == 0) continue;
        const iscale_is = (0.1 * @as(f32, @floatFromInt(is)) + @as(f32, @floatFromInt(nmax))) / max;
        const scale_is = 1.0 / iscale_is;
        var mse: f32 = 0;
        for (0..n) |i| {
            const l = std.math.clamp(nearestInt(iscale_is * x[i]), 0, nmax);
            const diff = x[i] - scale_is * @as(f32, @floatFromInt(l));
            const w = qw[i];
            mse += w * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    var sumlx: f32 = 0;
    var suml2: f32 = 0;
    for (0..n) |i| {
        const l = std.math.clamp(nearestInt(iscale * x[i]), 0, nmax);
        L[i] = @intCast(l);
        const w = qw[i];
        const lf: f32 = @floatFromInt(l);
        sumlx += w * x[i] * lf;
        suml2 += w * lf * lf;
    }
    var itry: usize = 0;
    while (itry < 5) : (itry += 1) {
        var n_changed: usize = 0;
        for (0..n) |i| {
            const w = qw[i];
            const lf: f32 = @floatFromInt(L[i]);
            const slx = sumlx - w * x[i] * lf;
            const sl2 = suml2 - w * lf * lf;
            if (slx > 0 and sl2 > 0) {
                var new_l = nearestInt(x[i] * sl2 / slx);
                new_l = @min(new_l, nmax);
                if (new_l != L[i]) {
                    const nf: f32 = @floatFromInt(new_l);
                    const nsx = slx + w * x[i] * nf;
                    const nsl2 = sl2 + w * nf * nf;
                    if (nsx * nsx * suml2 > sumlx * sumlx * nsl2) {
                        L[i] = @intCast(new_l);
                        sumlx = nsx;
                        suml2 = nsl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if (n_changed == 0) break;
    }
    return if (suml2 > 0) sumlx / suml2 else 0.0;
}

/// Quantize a full row using the 512-entry kmap. Caller must call
/// init_mod.initIq2XsData512 first (NOT initIq2XsData — different grid).
pub fn quantizeRowIQ2_XS_WithInit(
    data: *const init_mod.Iq2GridData,
    src: []const f32,
    dst: [*]BlockIQ2_XS,
    k: usize,
    quant_weights: ?[]const f32,
) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;

    var scales: [QK_K / 16]f32 = undefined;
    var weight: [16]f32 = undefined;
    var xval: [16]f32 = undefined;
    var L: [16]u8 = undefined;
    var Laux: [16]u8 = undefined;
    var waux: [16]f32 = undefined;
    var is_on_grid: [2]bool = undefined;
    var is_on_grid_aux: [2]bool = undefined;
    var block_signs: [2]u8 = undefined;
    var q2: [2 * (QK_K / 16)]u16 = undefined;

    for (0..nb) |ibl| {
        const blk = &dst[ibl];
        blk.d = f32_to_f16(0.0);
        @memset(q2[0..], 0);
        @memset(blk.scales[0..], 0);

        var max_scale: f32 = 0;
        const xbl = src[ibl * QK_K ..][0..QK_K];

        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        const sigma2 = sumx2 / QK_K;

        for (0..QK_K / 16) |ib| {
            const xb = xbl[16 * ib ..][0..16];

            for (0..16) |i| {
                // Null quant_weights: |x| as the weight (the reference
                // ALWAYS passes importance weights; null is our test path).
                weight[i] = if (quant_weights) |qwv| qwv[ibl * QK_K + 16 * ib + i] * @sqrt(sigma2 + xb[i] * xb[i]) else xb[i] * xb[i];
                waux[i] = @sqrt(weight[i]);
            }

            // Sign extraction with odd-parity flip (2 groups of 8)
            for (0..2) |kk| {
                var nflip: usize = 0;
                var s: u8 = 0;
                for (0..8) |i| {
                    if (xb[8 * kk + i] >= 0) {
                        xval[8 * kk + i] = xb[8 * kk + i];
                    } else {
                        xval[8 * kk + i] = -xb[8 * kk + i];
                        nflip += 1;
                        s |= @as(u8, 1) << @intCast(i);
                    }
                }
                if (nflip % 2 == 1) {
                    var imin: usize = 0;
                    var min = weight[8 * kk + imin] * xb[8 * kk + imin] * xb[8 * kk + imin];
                    for (1..8) |i| {
                        const ax = weight[8 * kk + i] * xb[8 * kk + i] * xb[8 * kk + i];
                        if (ax < min) {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * kk + imin] = -xval[8 * kk + imin];
                    s ^= @as(u8, 1) << @intCast(imin);
                }
                block_signs[kk] = s & 127;
            }

            var max = xval[0];
            for (1..16) |i| max = @max(max, xval[i]);
            @memset(L[0..16], 0);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                continue;
            }

            // ±9-scale grid search
            var best: f32 = 0;
            var scale = max / @as(f32, @floatFromInt(2 * KMAX - 1));
            is_on_grid[0] = true;
            is_on_grid[1] = true;
            var is: i32 = -9;
            while (is <= 9) : (is += 1) {
                const id = (@as(f32, @floatFromInt(2 * KMAX - 1)) + 0.1 * @as(f32, @floatFromInt(is))) / max;
                const this_scale = 1.0 / id;
                for (0..2) |kk| {
                    for (0..8) |i| {
                        const l = nearestInt(0.5 * (id * xval[8 * kk + i] - 1));
                        Laux[8 * kk + i] = @intCast(std.math.clamp(l, 0, KMAX - 1));
                    }
                    var u: u16 = 0;
                    for (0..8) |i| u |= @as(u16, Laux[8 * kk + i]) << @intCast(2 * i);
                    is_on_grid_aux[kk] = true;
                    if (data.kmap[u] < 0) {
                        is_on_grid_aux[kk] = false;
                        const enc = data.kmap[u];
                        const nb_start: usize = @intCast(-(enc + 1));
                        _ = findBestNeighbour(data, nb_start, xval[8 * kk ..][0..8], waux[8 * kk ..][0..8], this_scale, Laux[8 * kk ..].ptr);
                    }
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..16) |i| {
                    const w = weight[i];
                    const q: f32 = 2.0 * @as(f32, @floatFromInt(Laux[i])) + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 and sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    @memcpy(L[0..16], Laux[0..16]);
                    @memcpy(is_on_grid[0..2], is_on_grid_aux[0..2]);
                }
            }

            // Off-grid redo at final scale
            var n_not_ongrid: usize = 0;
            for (0..2) |kk| {
                if (!is_on_grid[kk]) n_not_ongrid += 1;
            }
            if (n_not_ongrid > 0 and scale > 0) {
                const id = 1.0 / scale;
                for (0..2) |kk| {
                    if (is_on_grid[kk]) continue;
                    var u: u16 = 0;
                    for (0..8) |i| {
                        const l = std.math.clamp(nearestInt(0.5 * (id * xval[8 * kk + i] - 1)), 0, KMAX - 1);
                        u |= @as(u16, @intCast(l)) << @intCast(2 * i);
                        L[8 * kk + i] = @intCast(l);
                    }
                    if (data.kmap[u] < 0) {
                        const enc = data.kmap[u];
                        const nb_start: usize = @intCast(-(enc + 1));
                        _ = findBestNeighbour(data, nb_start, xval[8 * kk ..][0..8], waux[8 * kk ..][0..8], scale, L[8 * kk ..].ptr);
                    }
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..16) |i| {
                    const w = weight[i];
                    const q: f32 = 2.0 * @as(f32, @floatFromInt(L[i])) + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }

            if (scale < 0) {
                scale = -scale;
                for (0..2) |kk| block_signs[kk] = (~block_signs[kk]) & 127;
            }

            // Pack: q2 = grid_index | signs<<9 (verify on-grid)
            for (0..2) |kk| {
                var u: u16 = 0;
                for (0..8) |i| u |= @as(u16, L[8 * kk + i]) << @intCast(2 * i);
                if (data.kmap[u] < 0) {
                    @panic("iq2_xs quantize: internal error — L not on grid after refinement");
                }
                q2[2 * ib + kk] = @as(u16, @intCast(data.kmap[u])) | (@as(u16, block_signs[kk]) << 9);
            }
            scales[ib] = scale;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) {
            @memset(blk.qs[0..], 0);
            continue;
        }

        const d = max_scale / 31.0;
        blk.d = f32_to_f16(d);
        const id = 1.0 / d;
        for (0..QK_K / 16) |ib| {
            const l = std.math.clamp(nearestInt(0.5 * (id * scales[ib] - 1)), 0, 15);
            if (ib % 2 == 0) {
                blk.scales[ib / 2] = @intCast(l);
            } else {
                blk.scales[ib / 2] |= @as(u8, @intCast(l)) << 4;
            }
        }
        @memcpy(blk.qs[0..], q2[0..]);
    }
}
