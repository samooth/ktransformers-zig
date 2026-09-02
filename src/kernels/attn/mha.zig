// Multi-Head Attention (MHA) engine for non-MLA models.
//
// Ports standard transformer attention (Qwen3, Gemma, etc.) where the
// attention is plain Q/K/V projections + RoPE + scaled dot-product
// attention, optionally with grouped-query attention (GQA) where
// num_kv_heads < num_heads.
//
// Reference: ktransformers/kt-kernel/python/sft/layer.py:100-200
//
// Differences vs MLA (src/mla/mla_core.zig):
//   * No latent-compression absorption — full Q/K/V are materialized.
//   * Per-head Q, K, V tensors are sized [num_heads, max_qlen, head_dim].
//   * KV cache stores per-head K and V separately (continuous KV).
//   * RoPE is applied to Q and K (whole head, not just PE half).
//   * Softmax over [0, kv_len) per (head, query_row).
//   * GQA: num_kv_heads divides num_heads; K/V are repeated per group.

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// BF16 helpers (reused from MLA)
// ============================================================================

pub inline fn bf16ToF32(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

pub inline fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x00008000) >> 16);
}

const Vec16f32 = @Vector(16, f32);

inline fn loadVec16(p: [*]const f32) Vec16f32 {
    return p[0..16].*;
}

inline fn loadVec16Bf16(w: [*]const u16) Vec16f32 {
    var tmp: [16]f32 = undefined;
    for (0..16) |j| tmp[j] = bf16ToF32(w[j]);
    return tmp;
}

// ============================================================================
// RoPE
// ============================================================================

/// Half-split RoPE: x[i] pairs with x[i + head_dim/2]. Matches the
/// Qwen3 / Gemma convention (the rope_size used by these models is the
/// full head_dim; half goes to the rotated-half, the other half stays).
fn ropeRotate(
    v: [*]f32,
    position: usize,
    head_dim: usize,
    theta: f64,
) void {
    const half = head_dim / 2;
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const freq = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim)));
        const angle = @as(f64, @floatFromInt(position)) * freq;
        const cos_v: f32 = @floatCast(@cos(angle));
        const sin_v: f32 = @floatCast(@sin(angle));
        const v0 = v[i];
        const v1 = v[i + half];
        v[i] = v0 * cos_v - v1 * sin_v;
        v[i + half] = v0 * sin_v + v1 * cos_v;
    }
}

/// Softmax over `size` entries (numerically stable, in-place OK).
pub fn softmaxInPlace(buf: [*]f32, size: usize) void {
    var max_val: f32 = -std.math.inf(f32);
    var i: usize = 0;
    while (i < size) : (i += 1) {
        if (buf[i] > max_val) max_val = buf[i];
    }
    var sum: f32 = 0;
    i = 0;
    while (i < size) : (i += 1) {
        const e = @exp(buf[i] - max_val);
        buf[i] = e;
        sum += e;
    }
    const inv = 1.0 / sum;
    i = 0;
    while (i < size) : (i += 1) buf[i] *= inv;
}

// ============================================================================
// MHA Config
// ============================================================================

pub const MhaConfig = struct {
    hidden_size: usize,
    num_heads: usize, // query heads
    num_kv_heads: usize, // KV heads (GQA: num_heads % num_kv_heads == 0)
    head_dim: usize, // per-head Q/K/V dim
    max_qlen: usize,
    max_kvlen: usize,
    rope_theta: f64 = 10000.0,

    // Weight pointers (BF16, caller-owned)
    q_proj: [*]const amx.bf16, // [num_heads * head_dim, hidden_size]
    k_proj: [*]const amx.bf16, // [num_kv_heads * head_dim, hidden_size]
    v_proj: [*]const amx.bf16, // [num_kv_heads * head_dim, hidden_size]
    o_proj: [*]const amx.bf16, // [hidden_size, num_heads * head_dim]

    pub fn qSize(self: MhaConfig) usize {
        return self.num_heads * self.head_dim;
    }
    pub fn kvSize(self: MhaConfig) usize {
        return self.num_kv_heads * self.head_dim;
    }

    /// How many query heads share a single KV head (GQA group size).
    pub fn groupSize(self: MhaConfig) usize {
        return self.num_heads / self.num_kv_heads;
    }
};

// ============================================================================
// KV Cache
// ============================================================================

pub const MhaKvCache = struct {
    num_kv_heads: usize,
    head_dim: usize,
    max_kvlen: usize,
    seq_len: usize = 0,

    // [num_kv_heads, max_kvlen, head_dim] each, contiguous f32
    k_cache: []f32,
    v_cache: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_kv_heads: usize, head_dim: usize, max_kvlen: usize) !MhaKvCache {
        return .{
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .max_kvlen = max_kvlen,
            .k_cache = try allocator.alloc(f32, num_kv_heads * max_kvlen * head_dim),
            .v_cache = try allocator.alloc(f32, num_kv_heads * max_kvlen * head_dim),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MhaKvCache) void {
        self.allocator.free(self.k_cache);
        self.allocator.free(self.v_cache);
    }

    pub fn clear(self: *MhaKvCache) void {
        self.seq_len = 0;
    }

    pub fn getK(self: *MhaKvCache, head: usize, pos: usize) [*]const f32 {
        return self.k_cache.ptr + (head * self.max_kvlen + pos) * self.head_dim;
    }

    pub fn getV(self: *MhaKvCache, head: usize, pos: usize) [*]const f32 {
        return self.v_cache.ptr + (head * self.max_kvlen + pos) * self.head_dim;
    }
};

// ============================================================================
// MHA Engine
// ============================================================================

pub const MhaEngine = struct {
    config: MhaConfig,
    cache: *MhaKvCache,
    allocator: std.mem.Allocator,

    // Q [num_heads, max_qlen, head_dim]
    q: []f32,
    // K/V after projection but before RoPE [num_kv_heads, max_qlen, head_dim]
    k_pre: []f32,
    v: []f32,
    // Attention weights [num_heads, max_qlen, max_kvlen]
    attn_w: []f32,
    // Attention output [num_heads, max_qlen, head_dim]
    attn_out: []f32,
    // Combined heads [max_qlen, num_heads * head_dim]
    combined: []f32,

    pub fn init(allocator: std.mem.Allocator, config: MhaConfig, cache: *MhaKvCache) !MhaEngine {
        const nh = config.num_heads;
        const nkv = config.num_kv_heads;
        const hd = config.head_dim;
        const mq = config.max_qlen;
        const mkv = config.max_kvlen;

        var self = MhaEngine{
            .config = config,
            .cache = cache,
            .allocator = allocator,
            .q = try allocator.alloc(f32, nh * mq * hd),
            .k_pre = try allocator.alloc(f32, nkv * mq * hd),
            .v = try allocator.alloc(f32, nkv * mq * hd),
            .attn_w = try allocator.alloc(f32, nh * mq * mkv),
            .attn_out = try allocator.alloc(f32, nh * mq * hd),
            .combined = try allocator.alloc(f32, mq * nh * hd),
        };
        errdefer self.deinit();
        return self;
    }

    pub fn deinit(self: *MhaEngine) void {
        self.allocator.free(self.q);
        self.allocator.free(self.k_pre);
        self.allocator.free(self.v);
        self.allocator.free(self.attn_w);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.combined);
    }

    pub fn resetCache(self: *MhaEngine) void {
        self.cache.clear();
    }

    // =======================================================================
    // QKV Projection + RoPE on Q and K
    // =======================================================================

    fn projectQ(
        self: *MhaEngine,
        input: [*]const f32, // [qlen, hidden]
        qlen: usize,
        kv_start_pos: usize,
    ) void {
        const cfg = self.config;
        const q_size = cfg.qSize(); // num_heads * head_dim
        // Q: [qlen, hidden] @ [q_size, hidden]^T -> [qlen, q_size]
        matmulF32(
            input, cfg.hidden_size, cfg.q_proj, cfg.hidden_size,
            self.q.ptr, q_size, qlen, q_size, cfg.hidden_size,
        );
        // Per-head RoPE
        for (0..cfg.num_heads) |h| {
            for (0..qlen) |pos| {
                const dst = self.q.ptr + h * qlen * cfg.head_dim + pos * cfg.head_dim;
                ropeRotate(dst, kv_start_pos + pos, cfg.head_dim, cfg.rope_theta);
            }
        }
    }

    fn projectKV(
        self: *MhaEngine,
        input: [*]const f32, // [qlen, hidden]
        qlen: usize,
        kv_start_pos: usize,
    ) void {
        const cfg = self.config;
        const kv_size = cfg.kvSize();
        // K: [qlen, hidden] @ [kv_size, hidden]^T -> [qlen, kv_size]
        matmulF32(
            input, cfg.hidden_size, cfg.k_proj, cfg.hidden_size,
            self.k_pre.ptr, kv_size, qlen, kv_size, cfg.hidden_size,
        );
        // V: [qlen, hidden] @ [kv_size, hidden]^T -> [qlen, kv_size]
        matmulF32(
            input, cfg.hidden_size, cfg.v_proj, cfg.hidden_size,
            self.v.ptr, kv_size, qlen, kv_size, cfg.hidden_size,
        );
        // Per-KV-head RoPE on K (V is not rotated), then write to cache.
        for (0..cfg.num_kv_heads) |h| {
            for (0..qlen) |pos| {
                const k_dst = self.k_pre.ptr + h * qlen * cfg.head_dim + pos * cfg.head_dim;
                ropeRotate(k_dst, kv_start_pos + pos, cfg.head_dim, cfg.rope_theta);
                const v_src = self.v.ptr + h * qlen * cfg.head_dim + pos * cfg.head_dim;
                const cache_pos = kv_start_pos + pos;
                const k_cache_dst = self.cache.k_cache.ptr + (h * cfg.max_kvlen + cache_pos) * cfg.head_dim;
                const v_cache_dst = self.cache.v_cache.ptr + (h * cfg.max_kvlen + cache_pos) * cfg.head_dim;
                @memcpy(k_cache_dst[0..cfg.head_dim], k_dst[0..cfg.head_dim]);
                @memcpy(v_cache_dst[0..cfg.head_dim], v_src[0..cfg.head_dim]);
            }
        }
        const new_seq = kv_start_pos + qlen;
        if (new_seq > self.cache.seq_len) self.cache.seq_len = new_seq;
    }

    // =======================================================================
    // Attention per head (GQA-aware)
    // =======================================================================

    fn attention(
        self: *MhaEngine,
        qlen: usize,
        kvlen: usize,
    ) void {
        const cfg = self.config;
        const grp = cfg.groupSize();
        for (0..cfg.num_heads) |h| {
            const kv_h = h / grp;
            const attn_w_h = self.attn_w.ptr + h * qlen * cfg.max_kvlen;
            const q_h = self.q.ptr + h * qlen * cfg.head_dim;
            const out_h = self.attn_out.ptr + h * qlen * cfg.head_dim;

            // Compute attention scores: Q @ K^T, then scale, then softmax
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
            for (0..qlen) |q_pos| {
                for (0..kvlen) |kv_pos| {
                    const k = self.cache.getK(kv_h, kv_pos);
                    const q_row = q_h + q_pos * cfg.head_dim;
                    var s: f32 = 0;
                    var i: usize = 0;
                    while (i + 16 <= cfg.head_dim) : (i += 16) {
                        const qv: Vec16f32 = loadVec16(q_row + i);
                        const kv: Vec16f32 = loadVec16(k + i);
                        s += @reduce(.Add, qv * kv);
                    }
                    while (i < cfg.head_dim) : (i += 1) {
                        s += q_row[i] * k[i];
                    }
                    attn_w_h[q_pos * cfg.max_kvlen + kv_pos] = s * scale;
                }
                softmaxInPlace(attn_w_h + q_pos * cfg.max_kvlen, kvlen);
            }

            // Output: attn_w @ V
            for (0..qlen) |q_pos| {
                const dst = out_h + q_pos * cfg.head_dim;
                @memset(dst[0..cfg.head_dim], 0);
                for (0..kvlen) |kv_pos| {
                    const w = attn_w_h[q_pos * cfg.max_kvlen + kv_pos];
                    const v = self.cache.getV(kv_h, kv_pos);
                    var i: usize = 0;
                    while (i + 16 <= cfg.head_dim) : (i += 16) {
                        const wv: Vec16f32 = @splat(w);
                        var dvec: Vec16f32 = loadVec16(dst + i);
                        const vvec: Vec16f32 = loadVec16(v + i);
                        dvec += wv * vvec;
                        (dst + i)[0..16].* = dvec;
                    }
                    while (i < cfg.head_dim) : (i += 1) {
                        dst[i] += w * v[i];
                    }
                }
            }
        }
    }

    // =======================================================================
    // Combine heads and O projection
    // =======================================================================

    fn combineAndProject(
        self: *MhaEngine,
        output: [*]f32,
        qlen: usize,
    ) void {
        const cfg = self.config;
        // Concat per-head outputs into [qlen, num_heads * head_dim]
        for (0..qlen) |pos| {
            for (0..cfg.num_heads) |h| {
                const src = self.attn_out.ptr + h * qlen * cfg.head_dim + pos * cfg.head_dim;
                const dst = self.combined.ptr + pos * cfg.num_heads * cfg.head_dim + h * cfg.head_dim;
                @memcpy(dst[0..cfg.head_dim], src[0..cfg.head_dim]);
            }
        }
        // o_proj: [qlen, num_heads*head_dim] @ [hidden, num_heads*head_dim]^T
        matmulF32(
            self.combined.ptr, cfg.num_heads * cfg.head_dim,
            cfg.o_proj, cfg.num_heads * cfg.head_dim,
            output, cfg.hidden_size,
            qlen, cfg.hidden_size, cfg.num_heads * cfg.head_dim,
        );
    }

    // =======================================================================
    // Forward
    // =======================================================================

    pub fn forward(
        self: *MhaEngine,
        input: [*]const f32,
        output: [*]f32,
        qlen: usize,
        kv_start_pos: usize,
    ) void {
        self.projectQ(input, qlen, kv_start_pos);
        self.projectKV(input, qlen, kv_start_pos);
        self.attention(qlen, kv_start_pos + qlen);
        self.combineAndProject(output, qlen);
    }

    pub fn decode(
        self: *MhaEngine,
        input: [*]const f32,
        output: [*]f32,
        position: usize,
    ) void {
        self.forward(input, output, 1, position);
    }
};

// ============================================================================
// Linear projection (reused from MLA; identical math).
// ============================================================================

pub fn matmulF32(
    input: [*]const f32,
    lda: usize,
    weight: [*]const u16, // BF16 [n, k] row-major
    ldb: usize,
    output: [*]f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            var kk: usize = 0;
            while (kk + 16 <= k) : (kk += 16) {
                const a = loadVec16(input + i * lda + kk);
                const b = loadVec16Bf16(weight + j * ldb + kk);
                sum += @reduce(.Add, a * b);
            }
            while (kk < k) : (kk += 1) {
                sum += input[i * lda + kk] * bf16ToF32(weight[j * ldb + kk]);
            }
            output[i * ldc + j] = sum;
        }
    }
}

/// Simple RMSNorm exported for use by model-orchestration layers
/// (Qwen3, etc.) that need to apply it on demand rather than through
/// the MLA engine.
pub fn rmsNormInline(
    x: []const f32,
    weight: [*]const u16, // BF16, [hidden_size]
    out: []f32,
    eps: f32,
) void {
    var sum_sq: f32 = 0;
    for (x) |v| sum_sq += v * v;
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    for (x, 0..) |v, i| {
        out[i] = v * inv_rms * bf16ToF32(weight[i]);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "softmax in-place normalizes" {
    var buf = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    softmaxInPlace(&buf, 4);
    var sum: f32 = 0;
    for (buf) |v| sum += v;
    try std.testing.expectApproxEqAbs(1.0, sum, 1e-5);
    try std.testing.expect(buf[3] > buf[0]);
}

test "rope rotate preserves per-pair norm" {
    var v = [_]f32{ 3, 4, 0, 0, 1, 0, 0, 0 };
    const before = std.math.sqrt(v[0] * v[0] + v[4] * v[4]);
    ropeRotate(&v, 12345, 8, 10000.0);
    const after = std.math.sqrt(v[0] * v[0] + v[4] * v[4]);
    try std.testing.expectApproxEqAbs(before, after, 1e-4);
}

test "bf16 round trip" {
    const vals = [_]f32{ 0.5, 1.0, -2.0, 3.14159, 100.0 };
    for (vals) |v| {
        const b = f32ToBf16(v);
        const r = bf16ToF32(b);
        try std.testing.expectApproxEqRel(v, r, 0.01);
    }
}

test "MHA identity (single head, 1 token) does not crash" {
    const allocator = std.testing.allocator;
    const hidden = 8;
    const nh = 1;
    const nkv = 1;
    const hd = 4;
    const mq = 2;
    const mkv = 4;

    var q_w: [nh * hd * hidden]u16 = undefined;
    var k_w: [nkv * hd * hidden]u16 = undefined;
    var v_w: [nkv * hd * hidden]u16 = undefined;
    var o_w: [hidden * nh * hd]u16 = undefined;
    for (0..q_w.len) |i| q_w[i] = f32ToBf16(0.0);
    for (0..k_w.len) |i| k_w[i] = f32ToBf16(0.0);
    for (0..v_w.len) |i| v_w[i] = f32ToBf16(0.0);
    for (0..o_w.len) |i| o_w[i] = f32ToBf16(0.0);

    var cache = try MhaKvCache.init(allocator, nkv, hd, mkv);
    defer cache.deinit();

    const cfg = MhaConfig{
        .hidden_size = hidden,
        .num_heads = nh,
        .num_kv_heads = nkv,
        .head_dim = hd,
        .max_qlen = mq,
        .max_kvlen = mkv,
        .rope_theta = 10000.0,
        .q_proj = &q_w,
        .k_proj = &k_w,
        .v_proj = &v_w,
        .o_proj = &o_w,
    };
    var engine = try MhaEngine.init(allocator, cfg, &cache);
    defer engine.deinit();

    var input: [hidden]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var output: [hidden]f32 = undefined;
    engine.decode(&input, &output, 0);
    // With all-zero weights, output is zero.
    for (output) |v| try std.testing.expectEqual(0.0, v);
}
