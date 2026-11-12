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
// ---------------------------------------------------------------------------
// Incremental decode state (Fase 3)
// ---------------------------------------------------------------------------

pub const ModelState = struct {
    allocator: std.mem.Allocator,
    layers: []layermod.LayerState,
    n_blocks: usize,
    block_residual: []f32,
    maxb: usize,
    E: usize,

    pub fn deinit(self: *ModelState) void {
        for (self.layers) |*ls| {
            if (ls.kda) |kd| {
                self.allocator.free(kd.S);
                self.allocator.free(kd.conv);
            }
            if (ls.mla) |ml| {
                self.allocator.free(ml.kv);
                self.allocator.free(ml.rope);
            }
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.block_residual);
    }
};

pub fn initModelState(
    model: *KimiModelBf16,
    cfg: *const cfgmod.KimiK3Config,
    max_seq: usize,
    allocator: std.mem.Allocator,
) ?ModelState {
    const E = model.hidden;
    const maxb = model.maxb;
    const n_layers = model.n_layers;

    var ms = ModelState{
        .allocator = allocator,
        .layers = undefined,
        .n_blocks = 0,
        .block_residual = undefined,
        .maxb = maxb,
        .E = E,
    };

    ms.layers = allocator.alloc(layermod.LayerState, n_layers) catch return null;
    for (0..n_layers) |li| {
        const is_mla = cfgmod.isMla(cfg, li);
        var ls = layermod.LayerState{ .is_mla = is_mla };
        if (is_mla) {
            const H = cfg.n_heads;
            const qn = cfg.qk_nope;
            const qr = cfg.qk_rope;
            const vh = cfg.v_head;
            const kvd = qn + vh;
            const kv = allocator.alloc(f32, max_seq * H * kvd) catch return null;
            for (kv) |*v| v.* = 0.0;
            const rope = allocator.alloc(f32, max_seq * qr) catch return null;
            for (rope) |*v| v.* = 0.0;
            ls.mla = .{ .kv = kv, .rope = rope, .cached = 0, .cap = max_seq };
        } else {
            const H = cfg.kda_heads;
            const D = cfg.kda_head_dim;
            const P = H * D;
            const K = cfg.conv_k;
            const hist = K - 1;
            const S = allocator.alloc(f32, H * D * D) catch return null;
            for (S) |*v| v.* = 0.0;
            const conv_size = 3 * P * hist;
            const conv = if (conv_size == 0)
                allocator.alloc(f32, 0) catch return null
            else blk: {
                const c = allocator.alloc(f32, conv_size) catch return null;
                for (c) |*v| v.* = 0.0;
                break :blk c;
            };
            ls.kda = .{ .S = S, .conv = conv };
        }
        ms.layers[li] = ls;
    }

    ms.block_residual = allocator.alloc(f32, maxb * E) catch return null;
    for (ms.block_residual) |*v| v.* = 0.0;

    return ms;
}

pub fn forwardInc(
    model: *KimiModelBf16,
    cfg: *const cfgmod.KimiK3Config,
    ms: *ModelState,
    input: []const f32,
    output: []f32,
) void {
    const E = model.hidden;
    ms.n_blocks = 0;
    for (ms.block_residual) |*b| b.* = 0.0;

    // input is the hidden state for a single token (embedding done by caller)
    @memcpy(output, input);

    for (0..model.n_layers) |li| {
        layermod.layerForwardInc(output, ms.block_residual, &ms.n_blocks, &model.layers[li], cfg, li, &ms.layers[li]);
    }

    // final RMSNorm
    var ss: f64 = 0.0;
    for (0..E) |i| {
        const v = output[i];
        ss += @as(f64, v) * @as(f64, v);
    }
    const inv: f32 = 1.0 / @sqrt(@as(f32, @floatCast(ss / @as(f64, @floatFromInt(E)))) + cfg.rms_eps);
    for (0..E) |i| output[i] = model.final_norm[i] * output[i] * inv;
}
