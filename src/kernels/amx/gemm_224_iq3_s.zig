// GEMM Kernel 224 IQ3_S (GGML non-linear 3.4375-bpw grid quant)
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256,
// IQ3S_N_SCALE = QK_K/64 = 4):
//   block_iq3_s {
//     ggml_half d;             // 2 bytes: global super-block scale
//     uint8_t qs[QK_K/4];      // 64 bytes: 8 grid-index lo bytes per 32-group
//     uint8_t qh[QK_K/32];     // 8 bytes: grid-index hi bit per sub-block-of-4
//     uint8_t signs[QK_K/8];   // 32 bytes: RAW sign bytes (1 per 8 weights)
//     uint8_t scales[4];       // 4 bytes: per-32 4-bit scale, packed 2/byte
//   } = 110 bytes per 256 weights
//
// Weight reconstruction (ggml-quants.c:2607):
//   Per 32-weight group PAIR (ib32 steps 2, sharing one scales byte):
//     db1 = d * (1 + 2*(scales[ib32/2] & 0xf))   // NOTE: (1+2n), not 0.25*(0.5+n)
//     db2 = d * (1 + 2*(scales[ib32/2] >>  4))
//     Per sub-block l (0..3): TWO u32 grid entries (4 magnitudes each),
//     9-bit index = qs[2l] | (qh[byte] << (8-2l)) & 256  /  qs[2l+1] | ... << (7-2l)
//     y[j+0..4) = db * grid1[j] * sign ; y[j+4..8) = db * grid2[j] * sign
//
// vs IQ3_XXS: groups are 32 weights with EIGHT 4-weight sub-blocks
// (XXS: four 8-weight sub-blocks), signs RAW (no KSIGNS palette), the
// scale scheme is (1+2*nibble) with a SEPARATE 4-byte array, and d
// gets the 1.033 fudge (XXS: 1.0125).
//
// Quantize note (ggml:4246): is_on_grid[] starts FALSE (every other
// format starts true) and the redo pass at the final scale runs for
// ALL sub-blocks (the `continue` is commented out at :4283) — every
// group gets the grid-byte re-derivation.

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// IQ3_S Block Structure (byte-exact vs ggml's block_iq3_s)
// ============================================================================

pub const QK_K: usize = 256;
const IQ3S_N_SCALE: usize = QK_K / 64; // 4

pub const BlockIQ3_S = extern struct {
    d: u16, // f16 global scale
    qs: [QK_K / 4]u8, // 64 grid-index lo bytes (8 per 32-group)
    qh: [QK_K / 32]u8, // 8 bytes: index hi bits, 1 bit per sub-block-of-4
    signs: [QK_K / 8]u8, // 32 RAW sign bytes (1 per 8 weights)
    scales: [IQ3S_N_SCALE]u8, // 4 x dual-nibble per-32 scales
};

comptime {
    if (@sizeOf(BlockIQ3_S) != 110) @compileError("BlockIQ3_S must be 110 bytes");
}

// ============================================================================
// Lookup tables
// ============================================================================

const iq2xxs = @import("gemm_224_iq2_xxs.zig");
pub const KMASK_IQ2XS = iq2xxs.KMASK_IQ2XS;

pub const IQ3S_GRID = [512]u32{
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301,
    0x01010303, 0x01010305, 0x01010309, 0x0101030d, 0x01010501, 0x01010503,
    0x0101050b, 0x01010707, 0x01010901, 0x01010905, 0x0101090b, 0x0101090f,
    0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301,
    0x01030303, 0x0103030b, 0x01030501, 0x01030507, 0x0103050f, 0x01030703,
    0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b, 0x01030f05, 0x01050101,
    0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b,
    0x0105090f, 0x01050b03, 0x01050b07, 0x01050f01, 0x01050f07, 0x01070107,
    0x01070303, 0x0107030b, 0x01070501, 0x01070505, 0x01070703, 0x01070707,
    0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509,
    0x01090705, 0x01090901, 0x01090907, 0x01090b03, 0x01090f01, 0x010b0105,
    0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d, 0x010b0707, 0x010b0903,
    0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105,
    0x010f0109, 0x010f0501, 0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01,
    0x010f0b09, 0x03010101, 0x03010103, 0x03010105, 0x03010109, 0x03010301,
    0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03,
    0x03010f05, 0x03030101, 0x03030103, 0x03030107, 0x0303010d, 0x03030301,
    0x03030309, 0x03030503, 0x03030701, 0x03030707, 0x03030903, 0x03030b01,
    0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907,
    0x03050b0b, 0x03050d01, 0x03050f05, 0x03070103, 0x03070109, 0x0307010f,
    0x03070301, 0x03070307, 0x03070503, 0x0307050f, 0x03070701, 0x03070709,
    0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01,
    0x03090b09, 0x030b0103, 0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701,
    0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509, 0x030d050f, 0x030d0909,
    0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103,
    0x05010107, 0x0501010b, 0x0501010f, 0x05010301, 0x05010305, 0x05010309,
    0x0501030d, 0x05010503, 0x05010507, 0x0501050f, 0x05010701, 0x05010705,
    0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301,
    0x05030307, 0x0503030f, 0x05030505, 0x0503050b, 0x05030703, 0x05030709,
    0x05030905, 0x05030b03, 0x05050103, 0x05050109, 0x0505010f, 0x05050503,
    0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303,
    0x05070505, 0x05070509, 0x05070703, 0x05070707, 0x05070905, 0x05070b01,
    0x05070d0d, 0x05090103, 0x0509010f, 0x05090501, 0x05090507, 0x05090705,
    0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101,
    0x050d0105, 0x050d010f, 0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b,
    0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907, 0x050f0b01, 0x07010105,
    0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03,
    0x07010d07, 0x07010f03, 0x07030103, 0x07030107, 0x0703010b, 0x07030309,
    0x07030503, 0x07030507, 0x07030901, 0x07030d01, 0x07030f05, 0x07030f0d,
    0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f,
    0x07070701, 0x07070903, 0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07,
    0x07090107, 0x07090303, 0x0709030d, 0x07090505, 0x07090703, 0x07090b05,
    0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903,
    0x070f0103, 0x070f0107, 0x070f0501, 0x070f0505, 0x070f070b, 0x09010101,
    0x09010109, 0x09010305, 0x09010501, 0x09010509, 0x0901050f, 0x09010705,
    0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03,
    0x09030b0b, 0x09050103, 0x09050107, 0x09050301, 0x0905030b, 0x09050503,
    0x09050707, 0x09050901, 0x09050b0f, 0x09050d05, 0x09050f01, 0x09070109,
    0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03,
    0x090b010b, 0x090b010f, 0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709,
    0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701, 0x090f0907, 0x090f0b03,
    0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107,
    0x0b03010b, 0x0b030305, 0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101,
    0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d, 0x0b050b07, 0x0b070105,
    0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d,
    0x0b0b0305, 0x0b0b050d, 0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105,
    0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307, 0x0d01030b, 0x0d010703,
    0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01,
    0x0d070101, 0x0d070309, 0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907,
    0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709, 0x0d0b0d01, 0x0d0d010b,
    0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05,
    0x0f030105, 0x0f030303, 0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103,
    0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503, 0x0f050701, 0x0f050b03,
    0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105,
    0x0f0d0703, 0x0f0f0101,
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
// Dequantize (byte-exact vs ggml-quants.c:2607)
// ============================================================================

/// k must be a multiple of 256.
pub fn dequantizeRowIQ3_S(x: [*]const BlockIQ3_S, y: [*]f32, k: usize) void {
    const nb = k / QK_K;
    var yoff: usize = 0;

    for (0..nb) |i| {
        const blk = &x[i];
        const d = f16_to_f32(blk.d);

        var qh_base: usize = 0; // qh advances 2 bytes per group-pair
        var signs_base: usize = 0; // signs advances 4 per group
        for (0..QK_K / 32) |ib32| {
            const db1 = d * (1.0 + 2.0 * @as(f32, @floatFromInt(blk.scales[ib32 / 2] & 0xf)));
            const db2 = d * (1.0 + 2.0 * @as(f32, @floatFromInt(blk.scales[ib32 / 2] >> 4)));
            // ib32 even -> qh[byte 0 of the pair], odd -> qh[byte 1]
            const qh_byte = blk.qh[qh_base + (ib32 % 2)];

            for (0..4) |l| {
                const db = if (ib32 % 2 == 0) db1 else db2;
                const lo1: usize = blk.qs[ib32 * 8 + 2 * l];
                const hi1: usize = (@as(usize, qh_byte) << @intCast(8 - 2 * l)) & 256;
                const lo2: usize = blk.qs[ib32 * 8 + 2 * l + 1];
                const hi2: usize = (@as(usize, qh_byte) << @intCast(7 - 2 * l)) & 256;
                const g1: u32 = IQ3S_GRID[lo1 | hi1];
                const g2: u32 = IQ3S_GRID[lo2 | hi2];
                const g1_bytes: [*]const u8 = @ptrCast(&g1);
                const g2_bytes: [*]const u8 = @ptrCast(&g2);
                const signs: u8 = blk.signs[signs_base + l];

                for (0..4) |j| {
                    const s1: f32 = if ((signs & KMASK_IQ2XS[j]) != 0) -1.0 else 1.0;
                    const m1: f32 = @floatFromInt(g1_bytes[j]);
                    y[yoff + j] = db * m1 * s1;
                    const s2: f32 = if ((signs & KMASK_IQ2XS[j + 4]) != 0) -1.0 else 1.0;
                    const m2: f32 = @floatFromInt(g2_bytes[j]);
                    y[yoff + j + 4] = db * m2 * s2;
                }
                yoff += 8;
            }
            if (ib32 % 2 == 1) qh_base += 2;
            signs_base += 4;
        }
    }
}

/// Dequantize IQ3_S to BF16.
pub fn dequantizeRowIQ3_SToBF16(x: [*]const BlockIQ3_S, y: [*]amx.bf16, k: usize) void {
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowIQ3_S(x + i, &f, QK_K);
        for (0..QK_K) |j| y[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x IQ3_S weights -> F32)
// ============================================================================

pub fn gemmIQ3_SScalar(
    a: [*]const amx.bf16,
    b: [*]const BlockIQ3_S,
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
                dequantizeRowIQ3_S(b + j * ldb + blk, &scratch, QK_K);
                for (0..QK_K) |k0| {
                    sum += amx.bf16_to_f32(a[i * lda + k0]) * scratch[k0];
                }
            }
            c[i * ldc + j] = sum;
        }
    }
}

// ============================================================================
// Quantize (ggml-quants.c:4169-4350, block_size=32)
// ============================================================================

const init_mod = @import("iq3xs_init.zig");

fn nearestInt(fval: f32) i32 {
    std.debug.assert(@abs(fval) <= 4194303.0);
    const val: f32 = fval + 12582912.0;
    const i: i32 = @bitCast(val);
    return (i & 0x007f_ffff) - 0x0040_0000;
}

fn findBestNeighbour(
    data: *const init_mod.Iq3GridData,
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
        for (0..4) |i| {
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
    for (0..4) |i| L[i] = @intCast(@divTrunc(gpos[i] - 1, 2));
    return grid_index;
}

const KMAX_Q: i32 = 8;

/// Quantize a full row using the 512-entry 3-bit kmap (nwant=3). Caller
/// must call init_mod.initIq3SData first.
pub fn quantizeRowIQ3_S_WithInit(
    data: *const init_mod.Iq3GridData,
    src: []const f32,
    dst: [*]BlockIQ3_S,
    k: usize,
    quant_weights: ?[]const f32,
) void {
    const block_size: usize = 32;
    const bs4 = block_size / 4; // 8
    const bs8 = block_size / 8; // 4
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;

    var scales: [QK_K / block_size]f32 = undefined;
    var weight: [block_size]f32 = undefined;
    var xval: [block_size]f32 = undefined;
    var L: [block_size]u8 = undefined;
    var Laux: [block_size]u8 = undefined;
    var waux: [block_size]f32 = undefined;
    var is_on_grid: [bs4]bool = undefined;
    var is_on_grid_aux: [bs4]bool = undefined;
    var block_signs: [bs8]u8 = undefined;

    for (0..nb) |ibl| {
        const blk = &dst[ibl];
        blk.* = std.mem.zeroes(BlockIQ3_S);
        blk.d = f32_to_f16(0.0);

        var max_scale: f32 = 0;
        const xbl = src[ibl * QK_K ..][0..QK_K];

        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        const sigma2 = 2 * sumx2 / QK_K;

        var qs_off: usize = 0;
        var signs_off: usize = 0;

        for (0..QK_K / block_size) |ib| {
            const xb = xbl[block_size * ib ..][0..block_size];

            for (0..block_size) |i| {
                weight[i] = if (quant_weights) |qwv| qwv[ibl * QK_K + block_size * ib + i] * @sqrt(sigma2 + xb[i] * xb[i]) else xb[i] * xb[i];
                waux[i] = @sqrt(weight[i]);
            }

            // RAW sign extraction (no parity trick)
            for (0..bs8) |kk| {
                var s: u8 = 0;
                for (0..8) |i| {
                    if (xb[8 * kk + i] >= 0) {
                        xval[8 * kk + i] = xb[8 * kk + i];
                    } else {
                        xval[8 * kk + i] = -xb[8 * kk + i];
                        s |= @as(u8, 1) << @intCast(i);
                    }
                }
                block_signs[kk] = s;
            }

            var max = xval[0];
            for (1..block_size) |i| max = @max(max, xval[i]);
            @memset(L[0..block_size], 0);
            if (max == 0) {
                scales[ib] = 0;
                continue;
            }

            // ±9-scale grid search with 0.2 steps (3-bit family)
            var best: f32 = 0;
            var scale = max / @as(f32, @floatFromInt(2 * KMAX_Q - 1));
            // NOTE: starts FALSE (unlike XXS) — every sub-block must win
            // the grid check in the search below.
            for (0..bs4) |kk| is_on_grid[kk] = false;
            var is: i32 = -9;
            while (is <= 9) : (is += 1) {
                const id = (@as(f32, @floatFromInt(2 * KMAX_Q - 1)) + 0.2 * @as(f32, @floatFromInt(is))) / max;
                const this_scale = 1.0 / id;
                for (0..bs4) |kk| {
                    for (0..4) |i| {
                        const l = nearestInt(0.5 * (id * xval[4 * kk + i] - 1));
                        Laux[4 * kk + i] = @intCast(std.math.clamp(l, 0, KMAX_Q - 1));
                    }
                    var u: u16 = 0;
                    for (0..4) |i| u |= @as(u16, Laux[4 * kk + i]) << @intCast(3 * i);
                    is_on_grid_aux[kk] = true;
                    if (data.kmap[u] < 0) {
                        is_on_grid_aux[kk] = false;
                        const enc = data.kmap[u];
                        const nb_start: usize = @intCast(-(enc + 1));
                        _ = findBestNeighbour(data, nb_start, xval[4 * kk ..][0..4], waux[4 * kk ..][0..4], this_scale, Laux[4 * kk ..].ptr);
                    }
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..block_size) |i| {
                    const w = weight[i];
                    const q: f32 = 2.0 * @as(f32, @floatFromInt(Laux[i])) + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 and sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    @memcpy(L[0..block_size], Laux[0..block_size]);
                    @memcpy(is_on_grid[0..bs4], is_on_grid_aux[0..bs4]);
                }
            }

            // Off-grid redo — runs for ALL sub-blocks (the `continue` is
            // commented out at ggml:4283): re-derive L from grid bytes.
            var n_not_ongrid: usize = 0;
            for (0..bs4) |kk| {
                if (!is_on_grid[kk]) n_not_ongrid += 1;
            }
            if (n_not_ongrid > 0 and scale > 0) {
                const id = 1.0 / scale;
                for (0..bs4) |kk| {
                    var u: u16 = 0;
                    for (0..4) |i| {
                        const l = std.math.clamp(nearestInt(0.5 * (id * xval[4 * kk + i] - 1)), 0, KMAX_Q - 1);
                        u |= @as(u16, @intCast(l)) << @intCast(3 * i);
                    }
                    var grid_index: usize = undefined;
                    if (data.kmap[u] < 0) {
                        const enc = data.kmap[u];
                        const nb_start: usize = @intCast(-(enc + 1));
                        grid_index = findBestNeighbour(data, nb_start, xval[4 * kk ..][0..4], waux[4 * kk ..][0..4], scale, L[4 * kk ..].ptr);
                    } else {
                        grid_index = @intCast(data.kmap[u]);
                    }
                    const gpos = data.grid[grid_index];
                    for (0..4) |i| L[4 * kk + i] = @intCast(@divTrunc(gpos[i] - 1, 2));
                }
                var sumqx: f32 = 0;
                var sumq2: f32 = 0;
                for (0..block_size) |i| {
                    const w = weight[i];
                    const q: f32 = 2.0 * @as(f32, @floatFromInt(L[i])) + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }

            if (scale < 0) {
                scale = -scale;
                for (0..bs8) |kk| block_signs[kk] = ~block_signs[kk];
            }

            // Pack: qs = index-lo, qh |= index-hi (1 bit per sub-block-of-4,
            // 8 per byte at (ib*bs4+k)%8 stride), signs area = RAW bytes.
            for (0..bs4) |kk| {
                var u: u16 = 0;
                for (0..4) |i| u |= @as(u16, L[4 * kk + i]) << @intCast(3 * i);
                if (data.kmap[u] < 0) {
                    @panic("iq3_s quantize: internal error — L not on grid after refinement");
                }
                const grid_index: usize = @intCast(data.kmap[u]);
                const slot = ib * bs4 + kk;
                blk.qs[qs_off + kk] = @truncate(grid_index);
                blk.qh[slot / 8] |= @as(u8, @truncate((grid_index >> 8) << @intCast(slot % 8)));
            }
            qs_off += bs4;
            for (0..bs8) |kk| blk.signs[signs_off + kk] = block_signs[kk];
            signs_off += bs8;

            scales[ib] = scale;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) {
            continue;
        }

        const d = max_scale / 31.0;
        blk.d = f32_to_f16(d * 1.033); // S fudge (ggml:4339)
        const id = 1.0 / d;
        var ib: usize = 0;
        while (ib < QK_K / block_size) : (ib += 2) {
            const l1 = std.math.clamp(nearestInt(0.5 * (id * scales[ib] - 1)), 0, 15);
            const l2 = std.math.clamp(nearestInt(0.5 * (id * scales[ib + 1] - 1)), 0, 15);
            blk.scales[ib / 2] = @as(u8, @intCast(l1)) | (@as(u8, @intCast(l2)) << 4);
        }
    }
}
