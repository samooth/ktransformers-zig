// BF16 Kimi K3 decoder layer: AttnRes + KDA/MLA (BF16) + AttnRes + dense/MoE.
//
// Ports layerForward from kimi_k3_model.zig (f32) but routes the attention
// projection through the BF16 KDA/MLA layers. Dense MLP and LatentMoE stay on
// the f32 resident path (experts stream as MXFP4 in the real model).

const std = @import("std");
const amx = @import("../arch/amx.zig");
const ops = @import("../../kimi/kimi_k3_ops.zig");
const cfgmod = @import("../../kimi/kimi_k3_config.zig");
const kda_bf16 = @import("kimi_k3_kda_bf16.zig");
const mla_bf16 = @import("kimi_k3_mla_bf16.zig");

pub const KimiLayerBf16Weights = struct {
    in_norm: []const f32,
    post_norm: []const f32,
    attn_res_norm: []const f32,
    attn_res_proj: []const f32,
    mlp_res_norm: []const f32,
    mlp_res_proj: []const f32,

    is_mla: bool,
    kda: ?kda_bf16.KdaBf16Weights,
    mla: ?mla_bf16.MlaBf16Weights,

    is_dense: bool,
    dense_inter: usize,
    dense_gate: []const f32,
    dense_up: []const f32,
    dense_down: []const f32,

    moe: ?ops.LatentMoeWeights,
    moe_cfg: ops.LatentMoeCfg,
};

fn matmulF32(y: []f32, x: []const f32, W: []const f32, in: usize, out: usize) void {
    for (0..out) |o| {
        var acc: f64 = 0.0;
        for (0..in) |i| acc += @as(f64, W[o * in + i]) * @as(f64, x[i]);
        y[o] = @floatCast(acc);
    }
}

fn rmsnormF32(y: []f32, x: []const f32, w: []const f32, n: usize, eps: f32) void {
    var ss: f64 = 0.0;
    for (0..n) |i| ss += @as(f64, x[i]) * @as(f64, x[i]);
    const inv: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(n)))) + eps);
    for (0..n) |i| y[i] = w[i] * x[i] * inv;
}

fn situGlu(out: []f32, x: []const f32, n: usize, b1: f32, b2: f32) void {
    ops.situGlu(out, x, n, b1, b2);
}

fn denseMlpSlice(out: []f32, x: []f32, w_gate: []const f32, w_up: []const f32, w_down: []const f32, dense_inter: usize, E: usize, b1: f32, b2: f32) void {
    var dgu = std.heap.page_allocator.alloc(f32, 2 * dense_inter) catch return;
    defer std.heap.page_allocator.free(dgu);
    const act = std.heap.page_allocator.alloc(f32, dense_inter) catch return;
    defer std.heap.page_allocator.free(act);
    matmulF32(dgu[0..dense_inter], x, w_gate, E, dense_inter);
    matmulF32(dgu[dense_inter .. 2 * dense_inter], x, w_up, E, dense_inter);
    ops.situGlu(act, dgu, dense_inter, b1, b2);
    matmulF32(out, act, w_down, dense_inter, E);
}

pub fn layerForward(
    h: []f32,
    block_residual: []f32,
    n_blocks: *usize,
    w: *const KimiLayerBf16Weights,
    c: *const cfgmod.KimiK3Config,
    layer_idx: usize,
    T: usize,
) void {
    const E = c.hidden;
    const maxb = c.n_layers / c.attn_res_block + 2;

    var pref = std.heap.page_allocator.alloc(f32, T * E) catch return;
    defer std.heap.page_allocator.free(pref);
    @memcpy(pref, h);

    var have_prefix: bool = true;

    if (n_blocks.* > 0) {
        var src = std.heap.page_allocator.alloc(f32, (maxb + 1) * E) catch return;
        defer std.heap.page_allocator.free(src);
        var foldA = std.heap.page_allocator.alloc(f32, E) catch return;
        defer std.heap.page_allocator.free(foldA);
        for (0..E) |i| foldA[i] = w.attn_res_norm[i] * w.attn_res_proj[i];
        for (0..T) |t| {
            for (0..n_blocks.*) |b| {
                @memcpy(src[b * E ..][0..E], block_residual[t * maxb * E + b * E ..][0..E]);
            }
            @memcpy(src[n_blocks.* * E ..][0..E], pref[t * E ..][0..E]);
            ops.attnRes(h[t * E ..][0..E], src[0 .. (n_blocks.* + 1) * E], foldA, n_blocks.* + 1, E, c.rms_eps);
        }
    }

    if (layer_idx % c.attn_res_block == 0) {
        for (0..T) |t| {
            @memcpy(block_residual[t * maxb * E + n_blocks.* * E ..][0..E], pref[t * E ..][0..E]);
        }
        n_blocks.* += 1;
        have_prefix = false;
    }

    var hin = std.heap.page_allocator.alloc(f32, T * E) catch return;
    defer std.heap.page_allocator.free(hin);
    var tmp = std.heap.page_allocator.alloc(f32, T * E) catch return;
    defer std.heap.page_allocator.free(tmp);
    for (0..T) |t| rmsnormF32(hin[t * E ..][0..E], h[t * E ..][0..E], w.in_norm, E, c.rms_eps);

    if (w.is_mla) {
        if (w.mla) |mla| {
            mla_bf16.mlaLayerFullBf16(tmp, hin, &mla, c, T);
        }
    } else {
        if (w.kda) |kda| {
            kda_bf16.kdaLayerFullBf16(tmp, hin, &kda, c, T);
        }
    }

    if (have_prefix) {
        for (0..T * E) |i| pref[i] += tmp[i];
    } else {
        @memcpy(pref, tmp);
        have_prefix = true;
    }

    {
        var src = std.heap.page_allocator.alloc(f32, (maxb + 1) * E) catch return;
        defer std.heap.page_allocator.free(src);
        var foldM = std.heap.page_allocator.alloc(f32, E) catch return;
        defer std.heap.page_allocator.free(foldM);
        for (0..E) |i| foldM[i] = w.mlp_res_norm[i] * w.mlp_res_proj[i];
        for (0..T) |t| {
            for (0..n_blocks.*) |b| {
                @memcpy(src[b * E ..][0..E], block_residual[t * maxb * E + b * E ..][0..E]);
            }
            @memcpy(src[n_blocks.* * E ..][0..E], pref[t * E ..][0..E]);
            ops.attnRes(h[t * E ..][0..E], src[0 .. (n_blocks.* + 1) * E], foldM, n_blocks.* + 1, E, c.rms_eps);
        }
    }

    for (0..T) |t| rmsnormF32(hin[t * E ..][0..E], h[t * E ..][0..E], w.post_norm, E, c.rms_eps);

    if (w.is_dense) {
        for (0..T) |t| {
            denseMlpSlice(tmp[t * E ..][0..E], hin[t * E ..][0..E], w.dense_gate, w.dense_up, w.dense_down, w.dense_inter, E, c.situ_b1, c.situ_b2);
        }
    } else {
        if (w.moe) |moe| {
            var moe_copy = moe;
            for (0..T) |t| {
                ops.latentMoe(tmp[t * E ..][0..E], hin[t * E ..][0..E], &moe_copy, &w.moe_cfg);
            }
        }
    }

    for (0..T * E) |i| pref[i] += tmp[i];
    @memcpy(h, pref);
    _ = situGlu;
}