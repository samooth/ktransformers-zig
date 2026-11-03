// BF16 Kimi K3 model: N decoder layers + final RMSNorm.
// Mirrors deepseekv3_model.zig but uses KimiLayerBf16Weights (KDA/MLA BF16).

const std = @import("std");
const amx = @import("../arch/amx.zig");
const layermod = @import("kimi_k3_layer_bf16.zig");
const cfgmod = @import("../../kimi/kimi_k3_config.zig");

pub const KimiModelBf16 = struct {
    allocator: std.mem.Allocator,
    hidden: usize,
    n_layers: usize,
    maxb: usize,
    layers: []layermod.KimiLayerBf16Weights,
    final_norm: []const f32,
    scratch: []f32, // ping-pong [2][max_qlen*hidden]

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: *const cfgmod.KimiK3Config,
        layers: []layermod.KimiLayerBf16Weights,
        final_norm: []const f32,
        max_qlen: usize,
    ) !*KimiModelBf16 {
        const self = try allocator.create(KimiModelBf16);
        const maxb = cfg.n_layers / cfg.attn_res_block + 2;
        const n = max_qlen * cfg.hidden;
        self.* = .{
            .allocator = allocator,
            .hidden = cfg.hidden,
            .n_layers = cfg.n_layers,
            .maxb = maxb,
            .layers = layers,
            .final_norm = final_norm,
            .scratch = try allocator.alloc(f32, 2 * n),
        };
        return self;
    }

    pub fn deinit(self: *KimiModelBf16) void {
        self.allocator.free(self.scratch);
        self.allocator.destroy(self);
    }

    pub fn forward(
        self: *KimiModelBf16,
        cfg: *const cfgmod.KimiK3Config,
        qlen: usize,
        hidden_states: []const f32,
        output: []f32,
    ) void {
        const E = self.hidden;
        const maxb = self.maxb;
        const n = qlen * E;

        const block_residual = self.allocator.alloc(f32, qlen * maxb * E) catch return;
        defer self.allocator.free(block_residual);
        for (block_residual) |*b| b.* = 0.0;

        // running hidden state
        var cur = self.scratch[0..n];
        @memcpy(cur, hidden_states);

        var nb: usize = 0;
        var buf_idx: usize = 1;
        for (0..self.n_layers) |li| {
            const dst = self.scratch[buf_idx * n ..][0..n];
            // layerForward writes in place into h; we use cur as h and write result to dst
            // Actually layerForward modifies h in place. Use a copy.
            @memcpy(dst, cur);
            layermod.layerForward(dst, block_residual, &nb, &self.layers[li], cfg, li, qlen);
            cur = dst;
            buf_idx ^= 1;
        }

        // final RMSNorm
        for (0..qlen) |t| {
            var ss: f64 = 0.0;
            for (0..E) |i| {
                const v = cur[t * E + i];
                ss += @as(f64, v) * @as(f64, v);
            }
            const inv: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(E)))) + cfg.rms_eps);
            for (0..E) |i| output[t * E + i] = self.final_norm[i] * cur[t * E + i] * inv;
        }
    }
};