// GEMM Kernel 224 IQ1_M (GGML non-linear 1.75-bpw grid quant)
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256,
// IQ1M_BLOCK_SIZE = 16):
//   block_iq1_m {
//     uint8_t  qs[QK_K/8];      // 32 bytes: grid-index lo (2 per 16-group)
//     uint8_t  qh[QK_K/16];     // 16 bytes: grid-index hi (4b/sub-block x2)
//                               //         + delta mask bits (0x08 / 0x80)
//     uint8_t  scales[QK_K/32]; // 8 bytes: 3-bit per-16 scales @ 3*(ib%4)
//                               //         + the f16 global scale HIDDEN in
//                               //         the high nibbles of sc[0..3]
//   } = 56 bytes per 256 weights
//
// Weight reconstruction (ggml-quants.c:2675):
//   scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000)
//   d = f16(scale)
//   Per 32-weight PAIR of groups (ib32), 4 sub-blocks of 8:
//     dl1 = d * (2*((sc[ib32/2] >> (6*(ib%2)+0)) & 7) + 1)   // group 2*ib32
//     dl2 = d * (2*((sc[ib32/2] >> (6*(ib%2)+3)) & 7) + 1)   // group 2*ib32+1
//     idx[0] = qs[0] | ((qh[0] << 8) & 0x700)   delta[0] = qh[0] & 0x08 ? -δ : +δ
//     idx[1] = qs[1] | ((qh[0] << 4) & 0x700)   delta[1] = qh[0] & 0x80 ? -δ : +δ
//     idx[2] = qs[2] | ((qh[1] << 8) & 0x700)   delta[2] = qh[1] & 0x08 ? -δ : +δ
//     idx[3] = qs[3] | ((qh[1] << 4) & 0x700)   delta[3] = qh[1] & 0x80 ? -δ : +δ
//     y = dl * (grid[j] + delta)   with delta = ±IQ1S_DELTA
//
// vs IQ1_S: 16-weight groups (2 sub-blocks), the delta can flip
// INDEPENDENTLY per half-group (4 mask variants at quantize time),
// the f16 global scale lives inside the scales nibble shuffle, and
// the quantizer ends with a GLOBAL d refit over the quantized scales.

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// IQ1_M Block Structure (byte-exact vs ggml's block_iq1_m)
// ============================================================================

pub const QK_K: usize = 256;
const IQ1M_BLOCK_SIZE: usize = 16;
const IQ1S_DELTA: f32 = 0.125;

pub const BlockIQ1_M = extern struct {
    qs: [QK_K / 8]u8, // 32 grid-index lo bytes (2 per 16-group)
    qh: [QK_K / 16]u8, // 16: index-his (4b x2) + delta mask bits
    scales: [QK_K / 32]u8, // 8: 3-bit scales + the hidden f16 scale nibbles
};

comptime {
    if (@sizeOf(BlockIQ1_M) != 56) @compileError("BlockIQ1_M must be 56 bytes");
}

// ============================================================================
// Grid (shared with IQ1_S)
// ============================================================================

const iq1s_mod = @import("gemm_224_iq1_s.zig");
pub const IQ1S_GRID = iq1s_mod.IQ1S_GRID;

inline fn gridBytes(entry: u64) [8]i8 {
    var out: [8]i8 = undefined;
    inline for (0..8) |j| {
        out[j] = @bitCast(@as(u8, @truncate(entry >> @intCast(8 * j))));
    }
    return out;
}

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

/// Compose a little-endian u16 from two scale bytes (the scales array
/// has natural alignment 1 — an extern struct of u8 — so u16 pointer
/// casts would be misaligned; compose by bytes instead).
inline fn sc16(sc: []const u8, i: usize) u16 {
    return @as(u16, sc[2 * i]) | (@as(u16, sc[2 * i + 1]) << 8);
}

/// Extract the hidden global f16 scale from the scales nibble shuffle.
pub fn extractScale(sc: []const u8) u16 {
    return (sc16(sc, 0) >> 12) | ((sc16(sc, 1) >> 8) & 0x00f0) | ((sc16(sc, 2) >> 4) & 0x0f00) | (sc16(sc, 3) & 0xf000);
}

// ============================================================================
// Dequantize (byte-exact vs ggml-quants.c:2675)
// ============================================================================

/// k must be a multiple of 256.
pub fn dequantizeRowIQ1_M(x: [*]const BlockIQ1_M, y: [*]f32, k: usize) void {
    const nb = k / QK_K;
    var yoff: usize = 0;

    for (0..nb) |i| {
        const blk = &x[i];
        const d = f16_to_f32(extractScale(&blk.scales));

        for (0..QK_K / 32) |ib32| {
            const scw = sc16(&blk.scales, ib32 / 2);
            const dl1 = d * (2.0 * @as(f32, @floatFromInt((scw >> @intCast(6 * (ib32 % 2) + 0)) & 0x7)) + 1.0);
            const dl2 = d * (2.0 * @as(f32, @floatFromInt((scw >> @intCast(6 * (ib32 % 2) + 3)) & 0x7)) + 1.0);

            // qh bytes advance 2 per 32-group-pair; 2 sub-blocks per half.
            const qh0 = blk.qh[2 * ib32 + 0];
            const qh1 = blk.qh[2 * ib32 + 1];

            const idx = [4]usize{
                blk.qs[ib32 * 4 + 0] | ((@as(usize, qh0) << 8) & 0x700),
                blk.qs[ib32 * 4 + 1] | ((@as(usize, qh0) << 4) & 0x700),
                blk.qs[ib32 * 4 + 2] | ((@as(usize, qh1) << 8) & 0x700),
                blk.qs[ib32 * 4 + 3] | ((@as(usize, qh1) << 4) & 0x700),
            };
            const delta = [4]f32{
                if ((qh0 & 0x08) != 0) -IQ1S_DELTA else IQ1S_DELTA,
                if ((qh0 & 0x80) != 0) -IQ1S_DELTA else IQ1S_DELTA,
                if ((qh1 & 0x08) != 0) -IQ1S_DELTA else IQ1S_DELTA,
                if ((qh1 & 0x80) != 0) -IQ1S_DELTA else IQ1S_DELTA,
            };

            for (0..2) |l| {
                const dl = if (l == 0) dl1 else dl2;
                for (0..8) |j| {
                    const gb = gridBytes(IQ1S_GRID[idx[2 * l]]);
                    const q: f32 = @floatFromInt(gb[j]);
                    y[yoff + j] = dl * (q + delta[2 * l]);
                }
                yoff += 8;
                for (0..8) |j| {
                    const gb = gridBytes(IQ1S_GRID[idx[2 * l + 1]]);
                    const q: f32 = @floatFromInt(gb[j]);
                    y[yoff + j] = dl * (q + delta[2 * l + 1]);
                }
                yoff += 8;
            }
        }
    }
}

/// Dequantize IQ1_M to BF16.
pub fn dequantizeRowIQ1_MToBF16(x: [*]const BlockIQ1_M, y: [*]amx.bf16, k: usize) void {
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowIQ1_M(x + i, &f, QK_K);
        for (0..QK_K) |j| y[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x IQ1_M weights -> F32)
// ============================================================================

pub fn gemmIQ1_MScalar(
    a: [*]const amx.bf16,
    b: [*]const BlockIQ1_M,
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
                dequantizeRowIQ1_M(b + j * ldb + blk, &scratch, QK_K);
                for (0..QK_K) |k0| {
                    sum += amx.bf16_to_f32(a[i * lda + k0]) * scratch[k0];
                }
            }
            c[i * ldc + j] = sum;
        }
    }
}

// ============================================================================
// Quantize (ggml-quants.c:4692-4944)
// ============================================================================

const init_mod = @import("iq2xs_init.zig");

fn nearestInt(fval: f32) i32 {
    std.debug.assert(@abs(fval) <= 4194303.0);
    const val: f32 = fval + 12582912.0;
    const i: i32 = @bitCast(val);
    return (i & 0x007f_ffff) - 0x0040_0000;
}

const GROUP_MAX_EPS: f32 = 1e-15;

fn findBestNeighbour2(
    data: *const init_mod.Iq2GridData,
    neighbors_start: usize,
    xval: []const f32,
    weight: []const f32,
    scale: f32,
    xg: [3]f32,
    L: [*]u8,
) usize {
    const num_neighbors = data.kneighbors[neighbors_start];
    var best_score: f32 = std.math.floatMax(f32);
    var grid_index: usize = std.math.maxInt(usize);
    if (num_neighbors > 0) {
        for (1..num_neighbors + 1) |j| {
            const nb = data.kneighbors[neighbors_start + j];
            const gpos = data.grid[nb];
            var d2: f32 = 0;
            for (0..8) |i| {
                const q: f32 = xg[@intCast(@divTrunc(gpos[i] - 1, 2))];
                const diff = scale * q - xval[i];
                const w = weight[i];
                d2 += w * diff * diff;
            }
            if (d2 < best_score) {
                best_score = d2;
                grid_index = nb;
            }
        }
    }
    if (grid_index == std.math.maxInt(usize)) {
        for (0..2048) |i| {
            const gpos = data.grid[i];
            var d2: f32 = 0;
            for (0..8) |j| {
                const q: f32 = xg[@intCast(@divTrunc(gpos[j] - 1, 2))];
                const w = weight[j];
                const diff = scale * q - xval[j];
                d2 += w * diff * diff;
            }
            if (d2 < best_score) {
                best_score = d2;
                grid_index = i;
            }
        }
    }
    std.debug.assert(grid_index != std.math.maxInt(usize));
    const gpos = data.grid[grid_index];
    for (0..8) |i| L[i] = @intCast(@divTrunc(gpos[i] - 1, 2));
    return grid_index;
}

/// Quantize a full row using the 2048-entry 1-bit kmap. Caller must call
/// init_mod.initIq1SData first (shared with IQ1_S).
pub fn quantizeRowIQ1_M_WithInit(
    data: *const init_mod.Iq2GridData,
    src: []const f32,
    dst: [*]BlockIQ1_M,
    k: usize,
    quant_weights: ?[]const f32,
) void {
    const block_size = IQ1M_BLOCK_SIZE;
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;

    const x_p = [3]f32{ -1.0 + IQ1S_DELTA, IQ1S_DELTA, 1.0 + IQ1S_DELTA };
    const x_m = [3]f32{ -1.0 - IQ1S_DELTA, -IQ1S_DELTA, 1.0 - IQ1S_DELTA };
    const masks = [4]u8{ 0x00, 0x80, 0x08, 0x88 };

    var scales: [QK_K / block_size]f32 = undefined;
    var weight: [block_size]f32 = undefined;
    var pairs: [2 * block_size]f32 = undefined;
    var L: [block_size]u8 = undefined;
    var index: [block_size / 8]u16 = undefined;
    var shifts: [QK_K / block_size]i8 = undefined;
    var sumqx4: [4]f32 = undefined;
    var sumq24: [4]f32 = undefined;

    for (0..nb) |ibl| {
        const blk = &dst[ibl];
        @memset(blk.qs[0..], 0);
        @memset(blk.qh[0..], 0);
        @memset(blk.scales[0..], 0);

        var max_scale: f32 = 0;
        const xbl = src[ibl * QK_K ..][0..QK_K];

        var sumx2: f32 = 0;
        for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
        const sigma2 = 2 * sumx2 / QK_K;

        for (0..QK_K / block_size) |ib| {
            const xb = xbl[block_size * ib ..][0..block_size];

            for (0..block_size) |i| {
                weight[i] = if (quant_weights) |qwv| qwv[ibl * QK_K + block_size * ib + i] * @sqrt(sigma2 + xb[i] * xb[i]) else xb[i] * xb[i];
            }
            var max: f32 = @abs(xb[0]);
            for (1..block_size) |i| max = @max(max, @abs(xb[i]));
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                shifts[ib] = 0;
                @memset(L[0..block_size], 1);
                continue;
            }

            // Sort (value, index) pairs.
            for (0..block_size) |j| {
                pairs[2 * j] = xb[j];
                const idx_bits: u32 = @intCast(j);
                pairs[2 * j + 1] = @bitCast(idx_bits);
            }
            std.mem.sort([2]f32, @as([][2]f32, @ptrCast(pairs[0 .. 2 * block_size]))[0..block_size], {}, struct {
                fn lt(_: void, a: [2]f32, b: [2]f32) bool {
                    return a[0] < b[0];
                }
            }.lt);

            // Exhaustive split search across the FOUR delta combinations:
            //   k=0: (+,+) k=1: (+,-) k=2: (-,+) k=3: (-,-)
            // The first half of the group (i < block_size/2) uses the k-th
            // sign for both candidates; the second half uses the OTHER
            // assignment (see the C's sumqx[0..3] accumulation).
            var best_score: f32 = -std.math.floatMax(f32);
            var scale = max;
            var besti1: i32 = -1;
            var besti2: i32 = -1;
            var best_k: i32 = -1;
            var b1: usize = 0;
            while (b1 <= block_size) : (b1 += 1) {
                var b2: usize = b1;
                while (b2 <= block_size) : (b2 += 1) {
                    @memset(sumqx4[0..], 0);
                    @memset(sumq24[0..], 0);
                    // level 0: j in [0, b1)
                    for (0..b1) |j| {
                        const i: usize = @as(u32, @bitCast(pairs[2 * j + 1]));
                        const w = weight[i];
                        const v = xb[i];
                        const sp = w * x_p[0] * v;
                        const sm = w * x_m[0] * v;
                        const qp = w * x_p[0] * x_p[0];
                        const qm = w * x_m[0] * x_m[0];
                        if (i < block_size / 2) {
                            sumqx4[0] += sp; sumq24[0] += qp;
                            sumqx4[1] += sp; sumq24[1] += qp;
                            sumqx4[2] += sm; sumq24[2] += qm;
                            sumqx4[3] += sm; sumq24[3] += qm;
                        } else {
                            sumqx4[0] += sp; sumq24[0] += qp;
                            sumqx4[2] += sp; sumq24[2] += qp;
                            sumqx4[1] += sm; sumq24[1] += qm;
                            sumqx4[3] += sm; sumq24[3] += qm;
                        }
                    }
                    // level 1: j in [b1, b2)
                    for (b1..b2) |j| {
                        const i: usize = @as(u32, @bitCast(pairs[2 * j + 1]));
                        const w = weight[i];
                        const v = xb[i];
                        const sp = w * x_p[1] * v;
                        const sm = w * x_m[1] * v;
                        const qp = w * x_p[1] * x_p[1];
                        const qm = w * x_m[1] * x_m[1];
                        if (i < block_size / 2) {
                            sumqx4[0] += sp; sumq24[0] += qp;
                            sumqx4[1] += sp; sumq24[1] += qp;
                            sumqx4[2] += sm; sumq24[2] += qm;
                            sumqx4[3] += sm; sumq24[3] += qm;
                        } else {
                            sumqx4[0] += sp; sumq24[0] += qp;
                            sumqx4[2] += sp; sumq24[2] += qp;
                            sumqx4[1] += sm; sumq24[1] += qm;
                            sumqx4[3] += sm; sumq24[3] += qm;
                        }
                    }
                    // level 2: j in [b2, block_size)
                    for (b2..block_size) |j| {
                        const i: usize = @as(u32, @bitCast(pairs[2 * j + 1]));
                        const w = weight[i];
                        const v = xb[i];
                        const sp = w * x_p[2] * v;
                        const sm = w * x_m[2] * v;
                        const qp = w * x_p[2] * x_p[2];
                        const qm = w * x_m[2] * x_m[2];
                        if (i < block_size / 2) {
                            sumqx4[0] += sp; sumq24[0] += qp;
                            sumqx4[1] += sp; sumq24[1] += qp;
                            sumqx4[2] += sm; sumq24[2] += qm;
                            sumqx4[3] += sm; sumq24[3] += qm;
                        } else {
                            sumqx4[0] += sp; sumq24[0] += qp;
                            sumqx4[2] += sp; sumq24[2] += qp;
                            sumqx4[1] += sm; sumq24[1] += qm;
                            sumqx4[3] += sm; sumq24[3] += qm;
                        }
                    }
                    for (0..4) |kk| {
                        if (sumq24[kk] > 0 and sumqx4[kk] * sumqx4[kk] > best_score * sumq24[kk]) {
                            scale = sumqx4[kk] / sumq24[kk];
                            best_score = scale * sumqx4[kk];
                            besti1 = @intCast(b1);
                            besti2 = @intCast(b2);
                            best_k = @intCast(kk);
                        }
                    }
                }
            }
            if (besti1 < 0 or besti2 < 0 or best_k < 0) {
                scales[ib] = 0;
                shifts[ib] = 0;
                @memset(L[0..block_size], 1);
                continue;
            }

            for (0..@as(usize, @intCast(besti1))) |j| {
                const i: usize = @as(u32, @bitCast(pairs[2 * j + 1]));
                L[i] = 0;
            }
            for (@as(usize, @intCast(besti1))..@as(usize, @intCast(besti2))) |j| {
                const i: usize = @as(u32, @bitCast(pairs[2 * j + 1]));
                L[i] = 1;
            }
            for (@as(usize, @intCast(besti2))..block_size) |j| {
                const i: usize = @as(u32, @bitCast(pairs[2 * j + 1]));
                L[i] = 2;
            }
            if (scale < 0) {
                for (0..block_size) |j| L[j] = 2 - L[j];
                scale = -scale;
                best_k = if (best_k == 0) 3 else if (best_k == 1) 2 else if (best_k == 2) 1 else 0;
            }

            // Grid lookup (2 sub-blocks of 8), delta choice per half.
            var all_on_grid = true;
            for (0..block_size / 8) |kk| {
                const xx: [3]f32 = if (kk == 0)
                    (if (best_k < 2) x_p else x_m)
                else
                    (if (@mod(best_k, 2) == 0) x_p else x_m);
                var u: u16 = 0;
                for (0..8) |j| u |= @as(u16, L[8 * kk + j]) << @intCast(2 * j);
                if (data.kmap[u] >= 0) {
                    index[kk] = @intCast(data.kmap[u]);
                } else {
                    all_on_grid = false;
                    const enc = data.kmap[u];
                    const nb_start: usize = @intCast(-(enc + 1));
                    index[kk] = @intCast(findBestNeighbour2(data, nb_start, xb[8 * kk ..][0..8], weight[8 * kk ..][0..8], scale, xx, L[8 * kk ..].ptr));
                }
            }
            if (!all_on_grid) {
                var sumqx_f: f32 = 0;
                var sumq2_f: f32 = 0;
                for (0..block_size / 8) |kk| {
                    const xx: [3]f32 = if (kk == 0)
                        (if (best_k < 2) x_p else x_m)
                    else
                        (if (@mod(best_k, 2) == 0) x_p else x_m);
                    const gpos = data.grid[index[kk]];
                    for (0..8) |j| {
                        const w = weight[8 * kk + j];
                        const q = xx[@intCast(@divTrunc(gpos[j] - 1, 2))];
                        sumqx_f += w * q * xb[8 * kk + j];
                        sumq2_f += w * q * q;
                    }
                }
                if (sumqx_f > 0 and sumq2_f > 0) scale = sumqx_f / sumq2_f;
            }

            blk.qs[2 * ib + 0] = @truncate(index[0]);
            blk.qs[2 * ib + 1] = @truncate(index[1]);
            blk.qh[ib] = @as(u8, @truncate(index[0] >> 8)) | @as(u8, @truncate(index[1] >> 8)) << 4;
            scales[ib] = scale;
            shifts[ib] = @intCast(best_k);
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) {
            continue;
        }

        // Final encode: 3-bit scales into the u16 view, delta masks into qh,
        // then the GLOBAL d refit (over all groups with their quantized
        // scale factors) before packing d into the nibble shuffle.
        var d = max_scale / 15.0;
        const id = 1.0 / d;
        var sumqx_f: f32 = 0;
        var sumq2_f: f32 = 0;
        for (0..QK_K / block_size) |ib| {
            var l = nearestInt(0.5 * (id * scales[ib] - 1));
            l = std.math.clamp(l, 0, 7);
            // sc[ib/4] |= l << 3*(ib%4) via byte RMW (LE u16 view):
            const scw = ib / 4;
            const sh = 3 * (ib % 4);
            const add: u16 = @as(u16, @intCast(l)) << @intCast(sh);
            const cur = sc16(&blk.scales, scw);
            const upd = cur | add;
            blk.scales[2 * scw] = @truncate(upd);
            blk.scales[2 * scw + 1] = @truncate(upd >> 8);
            blk.qh[ib] |= masks[@intCast(shifts[ib])];

            const xb = xbl[block_size * ib ..][0..block_size];
            for (0..block_size) |i| {
                weight[i] = if (quant_weights) |qwv| qwv[ibl * QK_K + block_size * ib + i] * @sqrt(sigma2 + xb[i] * xb[i]) else xb[i] * xb[i];
            }
            for (0..block_size / 8) |kk| {
                const xx: [3]f32 = if (kk == 0)
                    (if (shifts[ib] < 2) x_p else x_m)
                else
                    (if (@mod(shifts[ib], 2) == 0) x_p else x_m);
                // NOTE: read back through the PACKED form like the C does
                // (qs + qh<<... & 0x700) — the dequant-consistent view.
                const gidx: usize = @as(usize, blk.qs[2 * ib + kk]) | ((@as(usize, blk.qh[ib]) << @intCast(8 - 4 * kk)) & 0x700);
                const gpos = data.grid[gidx];
                for (0..8) |j| {
                    const w = weight[8 * kk + j];
                    const q = xx[@intCast(@divTrunc(gpos[j] - 1, 2))] * (2.0 * @as(f32, @floatFromInt(l)) + 1.0);
                    sumqx_f += w * q * xb[8 * kk + j];
                    sumq2_f += w * q * q;
                }
            }
        }
        if (sumq2_f > 0) d = sumqx_f / sumq2_f;
        // The fudge + nibble shuffle into sc[0..3] bits 12+ (byte RMW).
        const s16 = f32_to_f16(d * 1.1125); // 1.1125 fudge (ggml:4938)
        inline for (0..4) |wi| {
            const mask: u16 = switch (wi) {
                0 => (s16 & 0x000f) << 12,
                1 => (s16 & 0x00f0) << 8,
                2 => (s16 & 0x0f00) << 4,
                else => (s16 & 0xf000) << 0,
            };
            const upd = sc16(&blk.scales, wi) | mask;
            blk.scales[2 * wi] = @truncate(upd);
            blk.scales[2 * wi + 1] = @truncate(upd >> 8);
        }
    }
}
