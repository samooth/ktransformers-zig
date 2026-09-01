// IQ2_XXS quantizer — uses the runtime kmap/kneighbors from iq2xs_init.zig.
//
// Port of ggml-quants.c:3294-3465 (quantize_row_iq2_xxs_impl).
//
// Algorithm per 256-weight super-block:
//   Per 32-weight sub-block:
//     1. Split into 4 groups of 8; extract signs (odd-parity trick), get |x|
//     2. make_qp_quants: weighted least-squares fit of 0..3 levels
//     3. ±6-scale refinement: for each scale, quantize to 2-bit nibbles,
//        pack into a u16 fingerprint, lookup in kmap; if off-grid, use
//        the kneighbors refinement (iq2_find_best_neighbour). Track the
//        best scale via sumqx² > best·sumq².
//     4. Final L values from the best grid entry
//   Then encode: d = max_scale/31 (global f16 scale), per-32 scale 4-bit
//   in the top nibble of aux32[1], grid indices + sign palette entries
//   into the 32 u16 qs fields.

const std = @import("std");
const amx = @import("../arch/amx.zig");
const iq2xs = @import("gemm_224_iq2_xxs.zig");
const init = @import("iq2xs_init.zig");

const QK_K: usize = 256;
const KMAX: i32 = 3;
const GROUP_MAX_EPS: f32 = 1e-15;

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

/// ggml-quants.c:1076 make_qp_quants — weighted LS fit of 0..nmax-1 levels.
fn makeQPQuants(n: usize, nmax: i32, x: []const f32, L: []u8, qw: []const f32) f32 {
    var max: f32 = 0;
    for (0..n) |i| max = @max(max, x[i]);
    if (max < GROUP_MAX_EPS) {
        for (0..n) |i| L[i] = 0;
        return 0.0;
    }
    const iscale_init = @as(f32, @floatFromInt(nmax)) / max;
    var iscale = iscale_init;
    for (0..n) |i| L[i] = @intCast(std.math.clamp(nearestInt(iscale * x[i]), 0, nmax));
    const scale_init = 1.0 / iscale;
    const scale = scale_init;
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

/// ggml-quants.c:3270 iq2_find_best_neighbour — try each neighbor grid entry,
/// pick the one with min weighted squared error against (scale*grid - xval).
fn findBestNeighbour(
    data: *const init.Iq2GridData,
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

/// Quantize a full row using the runtime kmap. Caller must call
/// init.initIq2XsData first. The `quant_weights` are the importance-
/// weighting factors (same as the reference — typically x² or similar).
pub fn quantizeRowIQ2_XXS_WithInit(
    data: *const init.Iq2GridData,
    src: []const f32,
    dst: [*]iq2xs.BlockIQ2_XXS,
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
    var block_signs: [4]u8 = undefined;
    var q2: [2 * (QK_K / 32)]u32 = undefined;

    for (0..nb) |ibl| {
        const blk = &dst[ibl];
        blk.d = f32ToF16(0.0);
        @memset(q2[0..], 0);

        var max_scale: f32 = 0;
        const xbl = src[ibl * QK_K ..][0..QK_K];

        for (0..QK_K / 32) |ib| {
            const xb = xbl[32 * ib ..][0..32];
            // Use |x| as the default weight when quant_weights is null
            const qw: ?[]const f32 = if (quant_weights) |qwv| qwv[ibl * QK_K + 32 * ib ..][0..32] else null;

            var sumx2: f32 = 0;
            for (0..QK_K) |i| sumx2 += xbl[i] * xbl[i];
            const sigma2 = sumx2 / QK_K;

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
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                @memset(L[0..32], 0);
                continue;
            }

            var scale = makeQPQuants(32, KMAX + 1, xval[0..32], L[0..32], weight[0..32]);
            const eff_max = scale * @as(f32, @floatFromInt(KMAX));
            if (eff_max <= 0) {
                scales[ib] = 0;
                @memset(L[0..32], 0);
                continue;
            }

            // ±6-scale refinement with grid lookup
            var best: f32 = 0;
            var is: i32 = -6;
            while (is <= 6) : (is += 1) {
                const id = (2 * @as(f32, @floatFromInt(KMAX)) - 1 + 0.1 * @as(f32, @floatFromInt(is))) / eff_max;
                const this_scale = 1.0 / id;
                for (0..4) |kk| {
                    for (0..8) |i| {
                        const l = nearestInt(0.5 * (id * xval[8 * kk + i] - 1));
                        Laux[8 * kk + i] = @intCast(std.math.clamp(l, 0, KMAX - 1));
                    }
                    var u: u16 = 0;
                    for (0..8) |i| u |= @as(u16, Laux[8 * kk + i]) << @intCast(2 * i);
                    if (data.kmap[u] >= 0) {
                        // on-grid: L from the grid entry
                        const gpos = data.grid[@intCast(data.kmap[u])];
                        for (0..8) |i| Laux[8 * kk + i] = @intCast(@divTrunc(gpos[i] - 1, 2));
                    } else {
                        // off-grid: neighbor refinement
                        const enc = data.kmap[u];
                        const nb_start: usize = @intCast(-(enc + 1));
                        _ = findBestNeighbour(data, nb_start, xval[8 * kk ..][0..8], waux[8 * kk ..][0..8], this_scale, Laux[8 * kk ..].ptr);
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
                }
            }

            // Final L from the best scale's grid entries
            if (scale > 0) {
                const id = 1.0 / scale;
                for (0..4) |kk| {
                    var u: u16 = 0;
                    for (0..8) |i| {
                        const l = std.math.clamp(nearestInt(0.5 * (id * xval[8 * kk + i] - 1)), 0, KMAX - 1);
                        u |= @as(u16, @intCast(l)) << @intCast(2 * i);
                    }
                    if (data.kmap[u] >= 0) {
                        const gpos = data.grid[@intCast(data.kmap[u])];
                        for (0..8) |i| L[8 * kk + i] = @intCast(@divTrunc(gpos[i] - 1, 2));
                    } else {
                        const enc = data.kmap[u];
                        const nb_start: usize = @intCast(-(enc + 1));
                        _ = findBestNeighbour(data, nb_start, xval[8 * kk ..][0..8], waux[8 * kk ..][0..8], scale, L[8 * kk ..].ptr);
                    }
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
                // Flip scale positive + invert signs
                scale = -scale;
                for (0..4) |kk| block_signs[kk] = (~block_signs[kk]) & 127;
            }

            // Pack grid indices + signs into q2
            for (0..4) |kk| {
                var u: u16 = 0;
                for (0..8) |i| u |= @as(u16, L[8 * kk + i]) << @intCast(2 * i);
                if (data.kmap[u] < 0) {
                    @panic("iq2 quantize: internal error — L not on grid after refinement");
                }
                q2[2 * ib + 0] |= @as(u32, @intCast(data.kmap[u])) << @intCast(8 * kk);
                q2[2 * ib + 1] |= @as(u32, block_signs[kk]) << @intCast(7 * kk);
            }
            scales[ib] = scale;
            max_scale = @max(max_scale, scale);
        }

        if (max_scale == 0) {
            @memset(blk.qs[0 .. QK_K / 8], 0);
            continue;
        }

        const d = max_scale / 31.0;
        blk.d = f32ToF16(d);
        const id = 1.0 / d;
        for (0..QK_K / 32) |ib| {
            const l = std.math.clamp(nearestInt(0.5 * (id * scales[ib] - 1)), 0, 15);
            q2[2 * ib + 1] |= @as(u32, @intCast(l)) << 28;
        }

        // Copy the q2 u32 array into the u16 qs fields
        for (0..QK_K / 16) |i| {
            const word = q2[i];
            blk.qs[2 * i] = @truncate(word);
            blk.qs[2 * i + 1] = @truncate(word >> 16);
        }
    }
}
