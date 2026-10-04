// CLAIM: iq2xs-avx2 — IQ2_XXS/IQ2_XS GEMM — 2026-10-04
// GEMM Kernel 224 IQ2_XXS (GGML non-linear 2.0625-bpw grid quant)
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256):
//   block_iq2_xxs {
//     ggml_half d;             // 2 bytes: global super-block scale
//     uint16_t qs[QK_K/8];     // 64 bytes: 32 sub-blocks of 8 weights,
//                               //   each a 16-bit grid-index+signs pair
//   } = 66 bytes per 256 weights
//
// Weight reconstruction (ggml-quants.c:2488 dequantize_row_iq2_xxs):
//   Per 32-weight group (8 sub-blocks of 8):
//     aux32[0..1] = 4 consecutive u16 qs entries reinterpreted as 2×u32
//     db = d * (0.5 + (aux32[1] >> 28)) * 0.25   // 4-bit scale in the top nibble
//     Per sub-block l (0..3), 8 weights each:
//       grid = iq2xxs_grid[aux8[l]]   // u64: 8 packed magnitude bytes
//       signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127]  // 7-bit sign selector
//       y[j] = db * grid_byte[j] * (signs & kmask[j] ? -1 : 1)
//
// The 256-entry iq2xxs_grid is a u64 table (8 magnitude bytes per entry);
// ksigns_iq2xs is the 128-entry 7-bit sign palette. All tables are
// byte-exact transcriptions from ggml-common.h.
//
// Quantize: NOT IMPLEMENTED (out of scope for this port). The reference
// requires runtime-initialized kmap (fingerprint->grid) and kneighbors
// tables via iq2xs_init_impl — a separate workstream. The dequant+matmul
// path suffices for loading GGUF models whose weights come pre-quantized.

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// IQ2_XXS Block Structure (byte-exact vs ggml's block_iq2_xxs)
// ============================================================================

pub const QK_K: usize = 256;
const SUB_BLOCKS_PER_32: usize = 4; // 4 x 8 weights per 32-weight group
const N_GROUPS: usize = QK_K / 32; // = 8 groups of 32

pub const BlockIQ2_XXS = extern struct {
    d: u16, // f16 global scale
    qs: [QK_K / 8]u16, // 32 x 16-bit grid-index + sign entries
};

comptime {
    if (@sizeOf(BlockIQ2_XXS) != 66) @compileError("BlockIQ2_XXS must be 66 bytes");
}

// ============================================================================
// Lookup tables (byte-exact from ggml-common.h)
// ============================================================================

/// kmask_iq2xs: bit j of the sign byte applies to weight j.
pub const KMASK_IQ2XS = [8]u8{ 1, 2, 4, 8, 16, 32, 64, 128 };

/// ksigns_iq2xs: 128-entry 7-bit sign palette.
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

/// iq2xxs_grid: 256-entry u64 table. Each entry is 8 packed magnitude
/// bytes (one per weight of the sub-block); the byte values are drawn
/// from {0x08, 0x19, 0x2b} — the 3-bit magnitude alphabet.
pub const IQ2XXS_GRID = [256]u64{
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b080808, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x190819192b080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b0808,
    0x19190808082b082b, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
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
// Dequantize (byte-exact vs ggml-quants.c:2488)
// ============================================================================

/// k must be a multiple of 256.
pub fn dequantizeRowIQ2_XXS(x: [*]const BlockIQ2_XXS, y: [*]f32, k: usize) void {
    const nb = k / QK_K;
    var yoff: usize = 0;

    for (0..nb) |i| {
        const blk = &x[i];
        const d = f16_to_f32(blk.d);

        for (0..N_GROUPS) |ib32| {
            // Reinterpret 4 consecutive u16 entries as 2 u32
            const qs_base = ib32 * 4;
            var aux32: [2]u32 = undefined;
            aux32[0] = @as(u32, blk.qs[qs_base]) | (@as(u32, blk.qs[qs_base + 1]) << 16);
            aux32[1] = @as(u32, blk.qs[qs_base + 2]) | (@as(u32, blk.qs[qs_base + 3]) << 16);
            const aux8: [*]const u8 = @ptrCast(&aux32);

            // db = d * (0.5 + (aux32[1] >> 28)) * 0.25
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32[1] >> 28))) * 0.25;

            for (0..SUB_BLOCKS_PER_32) |l| {
                const grid64: u64 = IQ2XXS_GRID[aux8[l]];
                const grid_bytes: [*]const u8 = @ptrCast(&grid64);
                const sign_idx: usize = (aux32[1] >> @intCast(7 * l)) & 127;
                const signs: u8 = KSIGNS_IQ2XS[sign_idx];

                for (0..8) |j| {
                    const m: f32 = @floatFromInt(grid_bytes[j]);
                    const sign: f32 = if ((signs & KMASK_IQ2XS[j]) != 0) -1.0 else 1.0;
                    y[yoff + j] = db * m * sign;
                }
                yoff += 8;
            }
        }
    }
}

/// Dequantize IQ2_XXS to BF16.
pub fn dequantizeRowIQ2_XXSToBF16(x: [*]const BlockIQ2_XXS, y: [*]amx.bf16, k: usize) void {
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowIQ2_XXS(x + i, &f, QK_K);
        for (0..QK_K) |j| y[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x IQ2_XXS weights -> F32)
// ============================================================================

// ============================================================================
// AVX2 GEMM
// ============================================================================
//
// Same four rules as everywhere else: n-outer so the weight block streams from
// DRAM once, MR=4 so four activation rows share one unpacked weight vector,
// explicit `@mulAdd` because FP contraction is off in this toolchain, and one
// `@reduce` per output element.
//
// This format is the heaviest of the ones vectorized so far: TWO table lookups
// per 8 weights (a 256-entry u64 grid table and a 128-entry sign table) rather
// than a nibble unpack. What keeps it cheap is that both tables are tiny and
// L1-resident — 2 KB and 128 B — so both are scalar loads on purpose.
//
// `KMASK_IQ2XS = {1,2,4,8,16,32,64,128}` means `signs & KMASK[j]` is simply bit
// j of the signs byte, so the sign vector needs no mask table: broadcast the
// byte, shift by 0..7, and select.
//
// ORACLE DISCIPLINE: `grid8` reproduces `dequantizeRowIQ2_XXS` exactly, it is
// not an independent reading of the GGUF layout. These formats pack an exponent
// nibble into the same u32 as the sign indices, which is precisely the sort of
// detail a clean-room reimplementation gets wrong, so the vector path is
// required to match the existing dequantizer bit for bit.

const MR: usize = 4;
const Vec: type = @Vector(8, f32);

/// 8 bf16 -> 8 f32, unaligned.
inline fn widen8(p: [*]const amx.bf16) Vec {
    const raw: @Vector(8, u16) = @as(*align(1) const @Vector(8, u16), @ptrCast(p)).*;
    return @bitCast(@as(@Vector(8, u32), raw) << @as(@Vector(8, u5), @splat(16)));
}

/// 8 weights from one grid word and one signs byte: **byte** j of the grid is
/// the magnitude of weight j, and bit j of `signs` is its sign.
///
/// BYTE, not nibble. The first version of this extracted nibbles (`>> 4*j`) and
/// was wrong for all 256 weights; `dequantizeRowIQ2_XXS` reads
/// `grid_bytes[j]`, i.e. a byte index into the u64. Checked by hand against a
/// random block: grid `0x192B19190819082B`, byte 0 = 0x2B = 43, and the
/// reference value divided by the group scale is 6.179741 / 0.14375 = 43.0.
///
/// The sign needs no mask table: `KMASK_IQ2XS = {1,2,4,...,128}` means
/// `signs & KMASK[j]` is exactly bit j of the byte.
///
/// WRITTEN WITH EXPLICIT VECTOR OPS, not `inline for` + array staging. The
/// staged version measured **749 ps per call — 1.8x the cost of BOTH table
/// lookups combined** (grid 249 ps, signs 184 ps) — because eight scalar 64-bit
/// shifts are what the compiler had to work with, and AVX2 has no 64-bit
/// integer shift. Reinterpreting the u64 as 8 bytes lets it emit
/// `vpmovzxbd` + `vcvtdq2ps` instead.
///
/// The sign multiplier is `(bit << 1) - 1`, i.e. -1 for a clear bit and +1 for a
/// set bit, so the sign is a shift/subtract/convert rather than a select.
inline fn grid8(grid64: u64, signs: u8) Vec {
    const bytes: @Vector(8, u8) = @bitCast(grid64);
    const mag: Vec = @floatFromInt(@as(@Vector(8, i32), @intCast(bytes)));

    const sb: @Vector(8, u8) = @splat(signs);
    const shifts: @Vector(8, u3) = @Vector(8, u3){ 0, 1, 2, 3, 4, 5, 6, 7 };
    // i32 throughout, and the multiplier is `1 - bit*2`: the convention is
    // bit SET => NEGATIVE (matching `signs & KMASK[j] != 0 -> -1`), so `bit*2-1`
    // would be exactly inverted. The sink check in bench caught that: the
    // checksum came out negated.
    const bits: @Vector(8, i32) = @intCast((sb >> shifts) & @as(@Vector(8, u8), @splat(1)));
    const mult: Vec = @floatFromInt(@as(@Vector(8, i32), @splat(1)) - bits * @as(@Vector(8, i32), @splat(2)));
    return mag * mult;
}

/// AVX2 IQ2_XXS GEMM. `c` is OVERWRITTEN on every element of [0,m) x [0,n).
/// `k` must be a multiple of QK_K, as for the scalar version.
pub fn gemmAvx2(
    a: [*]const amx.bf16,
    b: [*]const BlockIQ2_XXS,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    const nb = k / QK_K;
    var j: usize = 0;
    while (j < n) : (j += 1) {
        const brow = b + j * ldb;

        var i: usize = 0;
        while (i + MR <= m) : (i += MR) {
            var acc = [_]Vec{@splat(0.0)} ** MR;
            for (0..nb) |blk| {
                const blk_ptr = &brow[blk];
                // Per-BLOCK, hoisted: the f16 super-block scale is the same for
                // all 256 weights. Doing this per group was the IQ4_XS mistake.
                const dsuper = f16_to_f32(blk_ptr.d);
                const abase = blk * QK_K;
                for (0..N_GROUPS) |ib32| {
                    // 4 u16 -> 2 u32, exactly as dequantizeRow does.
                    const qs_base = ib32 * 4;
                    const aux0: u32 = @as(u32, blk_ptr.qs[qs_base]) | (@as(u32, blk_ptr.qs[qs_base + 1]) << 16);
                    const aux1: u32 = @as(u32, blk_ptr.qs[qs_base + 2]) | (@as(u32, blk_ptr.qs[qs_base + 3]) << 16);
                    // The 8 grid fingerprints are the 8 bytes of those two u32.
                    const fp: [8]u8 = .{
                        @truncate(aux0), @truncate(aux0 >> 8), @truncate(aux0 >> 16), @truncate(aux0 >> 24),
                        @truncate(aux1), @truncate(aux1 >> 8), @truncate(aux1 >> 16), @truncate(aux1 >> 24),
                    };
                    // db = d * (0.5 + exponent) * 0.25
                    const db: f32 = dsuper * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
                    inline for (0..SUB_BLOCKS_PER_32) |l| {
                        const grid64 = IQ2XXS_GRID[fp[l]];
                        const signs = KSIGNS_IQ2XS[(aux1 >> @intCast(7 * l)) & 127];
                        const w = grid8(grid64, signs) * @as(Vec, @splat(db));
                        const koff = abase + ib32 * 32 + l * 8;
                        inline for (0..MR) |r| {
                            acc[r] = @mulAdd(Vec, widen8(a + (i + r) * lda + koff), w, acc[r]);
                        }
                    }
                }
            }
            inline for (0..MR) |r| c[(i + r) * ldc + j] = @reduce(.Add, acc[r]);
        }

        // Row tail (m % MR), one row at a time.
        while (i < m) : (i += 1) {
            var acc: Vec = @splat(0.0);
            for (0..nb) |blk| {
                const blk_ptr = &brow[blk];
                const dsuper = f16_to_f32(blk_ptr.d);
                const abase = blk * QK_K;
                for (0..N_GROUPS) |ib32| {
                    const qs_base = ib32 * 4;
                    const aux0: u32 = @as(u32, blk_ptr.qs[qs_base]) | (@as(u32, blk_ptr.qs[qs_base + 1]) << 16);
                    const aux1: u32 = @as(u32, blk_ptr.qs[qs_base + 2]) | (@as(u32, blk_ptr.qs[qs_base + 3]) << 16);
                    const fp: [8]u8 = .{
                        @truncate(aux0), @truncate(aux0 >> 8), @truncate(aux0 >> 16), @truncate(aux0 >> 24),
                        @truncate(aux1), @truncate(aux1 >> 8), @truncate(aux1 >> 16), @truncate(aux1 >> 24),
                    };
                    const db: f32 = dsuper * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
                    inline for (0..SUB_BLOCKS_PER_32) |l| {
                        const grid64 = IQ2XXS_GRID[fp[l]];
                        const signs = KSIGNS_IQ2XS[(aux1 >> @intCast(7 * l)) & 127];
                        const koff = abase + ib32 * 32 + l * 8;
                        acc = @mulAdd(Vec, widen8(a + i * lda + koff), grid8(grid64, signs) * @as(Vec, @splat(db)), acc);
                    }
                }
            }
            c[i * ldc + j] = @reduce(.Add, acc);
        }
    }
}

pub fn gemmIQ2_XXSScalar(
    a: [*]const amx.bf16, // [m, k] BF16 row-major
    b: [*]const BlockIQ2_XXS, // [n, k/QK_K] blocks row-major
    c: [*]f32, // [m, n] F32 row-major
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize, // in BLOCKS
    ldc: usize,
) void {
    const nb = k / QK_K;
    var scratch: [QK_K]f32 = undefined;
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..nb) |blk| {
                dequantizeRowIQ2_XXS(b + j * ldb + blk, &scratch, QK_K);
                // BUG (pre-existing): the activation index omitted `blk * QK_K`, so every
                // block was multiplied by the SAME first QK_K activations. Correct
                // only when nb == 1 (k == QK_K); wrong for every real checkpoint.
                // Pinned by "IQ scalar GEMMs honour the block offset" in
                // tests/kernels/simd_kernels_tests.zig.
                for (0..QK_K) |k0| {
                    sum += amx.bf16_to_f32(a[i * lda + blk * QK_K + k0]) * scratch[k0];
                }
            }
            c[i * ldc + j] = sum;
        }
    }
}

// ============================================================================
// Quantize — NOT IMPLEMENTED (out of scope)
// ============================================================================

pub fn quantizeRowIQ2_XXS(_: [*]const f32, _: [*]BlockIQ2_XXS, _: usize) void {
    @panic("quantizeRowIQ2_XXS: not implemented — requires the runtime kmap/kneighbors init (iq2xs_init_impl), a separate workstream. The dequant+matmul path handles pre-quantized GGUF weights.");
}
