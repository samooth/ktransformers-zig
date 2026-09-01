// IQ3_XXS quantizer — uses the runtime kmap/kneighbors from iq3xs_init.zig.
//
// Port of ggml-quants.c:3938-4150 (quantize_row_iq3_xxs_impl, grid_size=256).
//
// Algorithm per 256-weight super-block:
//   Per 32-weight group:
//     1. Split into 8 sub-blocks of 4; extract signs (odd-parity trick),
//        get |x| into xval
//     2. ±15-scale grid search: for each candidate scale, quantize the 4
//        values per sub-block to 3-bit levels l = round(0.5*(id*x - 1))
//        clamped to 0..kMaxQ-1, pack into a u16 fingerprint
//        (4 nibbles x 3 bits), lookup in kmap; if off-grid, use the
//        kneighbors refinement (iq3_find_best_neighbour). Track the best
//        scale via sumqx² > best*sumq².
//     3. If any sub-block ended off-grid, redo just those at the final
//        scale and re-derive L from the grid bytes.
//     4. Encode: q3[8*ib + k] = grid_index (one byte per sub-block),
//        scales_and_signs[ib] = 4 sign bytes packed at 7-bit stride.
//   Then: d = max_scale/31 * 1.0125 (fudge factor, ggml:4137), per-group
//   4-bit scale l = round(0.5*(id*scale - 1)) clamped 0..15 in the top
//   nibble of scales_and_signs[ib].
//
// Unlike IQ2_XXS there is NO make_qp_quants initial fit — the grid
// search itself produces the levels (kMaxQ = 8 levels, 3 bits).

const std = @import("std");
const iq3xs = @import("gemm_224_iq3_xxs.zig");
const init = @import("iq3xs_init.zig");

const QK_K: usize = 256;
const KMAX_Q: i32 = 8;
const GROUP_MAX_EPS: f32 = 1e-8; // GROUP_MAX_EPS_IQ3_XXS (ggml-quants.c:21)

fn nearestInt(fval: f32) i32 {
    std.debug.assert(@abs(fval) <= 4194303.0);
    const val: f32 = fval + 12582912.0;
    const i: i32 = @bitCast(val);
    return (i & 0x007f_ffff) - 0x0040_0000;
}

fn f32ToF16(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

/// ggml-quants.c:3914 iq3_find_best_neighbour — try each neighbor grid
/// entry, pick the one with min weighted squared error against
/// (scale*grid - xval). Writes L[0..4] and returns the grid index.
fn findBestNeighbour(
    data: *const init.Iq3GridData,
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

/// Quantize a full row using the runtime kmap. Caller must call
/// init.initIq3XsData first. `quant_weights` are the importance weights
/// (the reference always passes them in practice; null -> w = x²).
pub fn quantizeRowIQ3_XXS_WithInit(
    data: *const init.Iq3GridData,
    src: []const f32,
    dst: [*]iq3xs.BlockIQ3_XXS,
    k: usize,
    quant_weights: ?[]const f32,
) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;

    var scales: [QK_K / 32]f32 = undefined;
    var weight: [32]f32 = undefined;
    var xval: [32]f32 = undefined;
    var L: [32]u8 = undefined;
    var Laux: [32]u8 = undefined;
    var waux: [32]f32 = undefined;
    var is_on_grid: [8]bool = undefined;
    var is_on_grid_aux: [8]bool = undefined;
    var block_signs: [8]u8 = undefined;
    // q3 layout: 3*QK_K/8 bytes of grid indices + QK_K/32 bytes of
    // scales_and_signs (accessed as u32 via the alias below).
    var q3: [3 * (QK_K / 8) + QK_K / 32]u8 = undefined;
    const scales_and_signs: []u32 = @as([*]u32, @ptrCast(@alignCast(q3[QK_K / 4 ..].ptr)))[0 .. QK_K / 32];
    const quant_size = 3 * (QK_K / 8); // grid indices only — blk.qs is 96 bytes
    // (the 32-byte scales_and_signs area is written via the q3 alias inside
    // the loop; blk.qs is exactly 3*QK_K/8 == 96 bytes for IQ3_XXS)

    for (0..nb) |ibl| {
        const blk = &dst[ibl];
        blk.d = f32ToF16(0.0);
        @memset(q3[0..], 0);

        var max_scale: f32 = 0;
        const xbl = src[ibl * QK_K ..][0..QK_K];

        for (0..QK_K / 32) |ib| {
            const xb = xbl[32 * ib ..][0..32];
            const qw: ?[]const f32 = if (quant_weights) |qwv| qwv[ibl * QK_K + 32 * ib ..][0..32] else null;

            var sumx2: f32 = 0;
            for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
            const sigma2 = 2 * sumx2 / QK_K; // NOTE: 2* (ggml:3996) — differs from IQ2's sigma2

            for (0..32) |i| {
                weight[i] = if (qw) |q| q[i] * @sqrt(sigma2 + xb[i] * xb[i]) else xb[i] * xb[i];
                waux[i] = @sqrt(weight[i]);
            }

            // Extract signs; make each 8-group have even parity (odd-parity flip trick)
            for (0..4) |kk| {
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
            for (1..32) |i| max = @max(max, xval[i]);
            @memset(L[0..32], 0);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                continue;
            }

            // ±15-scale grid search
            var best: f32 = 0;
            var scale = max / @as(f32, @floatFromInt(2 * KMAX_Q - 1));
            for (0..8) |kk| is_on_grid[kk] = true;
            var is: i32 = -15;
            while (is <= 15) : (is += 1) {
                const id = (@as(f32, @floatFromInt(2 * KMAX_Q - 1)) + 0.2 * @as(f32, @floatFromInt(is))) / max;
                const this_scale = 1.0 / id;
                for (0..8) |kk| {
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
                for (0..32) |i| {
                    const w = weight[i];
                    const q: f32 = 2.0 * @as(f32, @floatFromInt(Laux[i])) + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 and sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    @memcpy(L[0..32], Laux[0..32]);
                    @memcpy(is_on_grid[0..8], is_on_grid_aux[0..8]);
                }
            }

            // If any sub-block is off-grid at the chosen scale, redo those
            // and re-derive L from the grid bytes (ggml:4070-4098)
            var n_not_ongrid: usize = 0;
            for (0..8) |kk| {
                if (!is_on_grid[kk]) n_not_ongrid += 1;
            }
            if (n_not_ongrid > 0 and scale > 0) {
                const id = 1.0 / scale;
                for (0..8) |kk| {
                    if (is_on_grid[kk]) continue;
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
                for (0..32) |i| {
                    const w = weight[i];
                    const q: f32 = 2.0 * @as(f32, @floatFromInt(L[i])) + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }

            if (scale < 0) {
                // Should never happen, but flip scale positive + invert signs
                scale = -scale;
                for (0..4) |kk| block_signs[kk] = (~block_signs[kk]) & 127;
            }

            // Pack grid indices into q3; verify L is on-grid
            for (0..8) |kk| {
                var u: u16 = 0;
                for (0..4) |i| u |= @as(u16, L[4 * kk + i]) << @intCast(3 * i);
                if (data.kmap[u] < 0) {
                    @panic("iq3 quantize: internal error — L not on grid after refinement");
                }
                q3[8 * ib + kk] = @intCast(data.kmap[u]);
            }
            scales_and_signs[ib] = @as(u32, block_signs[0]) |
                (@as(u32, block_signs[1]) << 7) |
                (@as(u32, block_signs[2]) << 14) |
                (@as(u32, block_signs[3]) << 21);
            scales[ib] = scale;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) {
            @memset(blk.qs[0..quant_size], 0);
            continue;
        }

        const d = max_scale / 31.0;
        blk.d = f32ToF16(d * 1.0125); // fudge factor (ggml:4137)
        const id = 1.0 / d;
        for (0..QK_K / 32) |ib| {
            const l = std.math.clamp(nearestInt(0.5 * (id * scales[ib] - 1)), 0, 15);
            scales_and_signs[ib] |= @as(u32, @intCast(l)) << 28;
        }
        @memcpy(blk.qs[0..quant_size], q3[0..quant_size]);
    }
}
