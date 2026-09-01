// MLA (Multi-Head Latent Attention) - Real Implementation
// Based on DeepSeek-V2/V3 architecture
// Reference: ktransformers kt-kernel/operators/llamafile/mla.hpp
//
// Architecture:
//   Q: hidden_size -> q_a_proj -> q_lora_rank -> RMSNorm -> q_b_proj -> [nope_size | rope_size] per head
//   KV: hidden_size -> kv_a_proj_with_mqa -> kv_lora_rank + rope_size -> RMSNorm -> split
//       -> compressed_kv [kv_lora_rank] stored in cache
//       -> k_pe [rope_size] stored in cache (with RoPE applied)
//   Attention:
//     - PE: q_pe @ k_pe^T (standard RoPE attention)
//     - Nope (absorbed): q_nope @ W_UK^T = q_absorb
//                       q_absorb @ compressed_kv^T = attn_weights
//                       o_absorb = attn_weights @ compressed_kv
//                       attention_output = o_absorb @ W_UV
//   Output: concat heads -> o_proj -> hidden_size
//
// Fixes vs the draft port (validated against kt-kernel C++):
//   1. Softmax once, over the SUM of PE + nope scores. The draft softmaxed PE
//      scores alone and then softmaxed again after adding nope scores
//      (double softmax). The C++ computes PE score, adds nope score, then a
//      single softmax per query row.
//   2. RoPE on q_pe uses the absolute position (kv_start_pos + pos), matching
//      the C++ which rotates with rope_angle at each token's absolute cache
//      position. The draft used the within-batch position only.

const std = @import("std");
const MlaConfig = @import("mla_config.zig").MlaConfig;
const MlaKvCache = @import("mla_cache.zig").MlaKvCache;

// ============================================================================
// BF16 helpers (inline for performance)
// ============================================================================

inline fn bf16ToF32(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

inline fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x00008000) >> 16);
}

// ============================================================================
// Vectorized operations
// ============================================================================

const Vec16f32 = @Vector(16, f32);

/// Load 16 f32 from a possibly-unaligned pointer at runtime offset.
inline fn loadVec16(p: [*]const f32) Vec16f32 {
    return p[0..16].*;
}

/// Build a Vec16 from BF16 weights. Vector element stores with a runtime
/// index are illegal (`w[j] = x` requires comptime j) — collect into a
/// normal array first and let array->vector coercion do the work.
inline fn loadVec16Bf16(w: [*]const u16) Vec16f32 {
    var tmp: [16]f32 = undefined;
    for (0..16) |j| {
        tmp[j] = bf16ToF32(w[j]);
    }
    return tmp;
}

/// RMSNorm: x = x / sqrt(mean(x^2) + eps) * weight
pub fn rmsNorm(
    input: [*]const f32,
    weight: [*]const u16, // BF16 weights
    output: [*]f32,
    size: usize,
    eps: f32,
) void {
    // Compute sum of squares
    var sum_sq: f32 = 0;
    var i: usize = 0;
    while (i + 16 <= size) : (i += 16) {
        const v: Vec16f32 = loadVec16(input + i);
        sum_sq += @reduce(.Add, v * v);
    }
    while (i < size) : (i += 1) {
        sum_sq += input[i] * input[i];
    }

    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(size)) + eps);
    const inv_rms = 1.0 / rms;

    // Normalize and scale
    i = 0;
    while (i + 16 <= size) : (i += 16) {
        const v: Vec16f32 = loadVec16(input + i);
        const w = loadVec16Bf16(weight + i);
        const result = v * @as(Vec16f32, @splat(inv_rms)) * w;
        (output + i)[0..16].* = result;
    }
    while (i < size) : (i += 1) {
        output[i] = input[i] * inv_rms * bf16ToF32(weight[i]);
    }
}

/// Linear projection: output = input @ weight^T
/// input: [m, k], weight: [n, k] (row-major BF16), output: [m, n]
pub fn matmulF32(
    input: [*]const f32,
    lda: usize,
    weight: [*]const u16, // BF16 weights, row-major [n, k]
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

/// Linear projection with weight row offset: output = input @ weight[offset..offset+n, :]^T
pub fn matmulF32Offset(
    input: [*]const f32,
    lda: usize,
    weight: [*]const u16,
    ldb: usize,
    offset: usize,
    output: [*]f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                sum += input[i * lda + kk] * bf16ToF32(weight[(offset + j) * ldb + kk]);
            }
            output[i * ldc + j] = sum;
        }
    }
}

// ============================================================================
// RoPE (Rotary Position Embedding)
// ============================================================================

/// Apply RoPE to a separated rope vector (half-split layout: x[i] pairs with x[i+half])
fn ropeRotate(
    v: [*]f32,
    position: usize,
    rope_size: usize,
    theta: f64,
) void {
    const half_rope = rope_size / 2;
    for (0..half_rope) |i| {
        const freq = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rope_size)));
        const angle = @as(f64, @floatFromInt(position)) * freq;
        const cos_val: f32 = @floatCast(@cos(angle));
        const sin_val: f32 = @floatCast(@sin(angle));

        const v0 = v[i];
        const v1 = v[i + half_rope];

        v[i] = v0 * cos_val - v1 * sin_val;
        v[i + half_rope] = v0 * sin_val + v1 * cos_val;
    }
}

/// Apply RoPE to q_pe (already separated into its own buffer)
fn applyRopeKpe(
    k_pe: [*]f32,
    position: usize,
    rope_size: usize,
    theta: f64,
) void {
    ropeRotate(k_pe, position, rope_size, theta);
}

// ============================================================================
// Softmax (in-place over the first `size` entries)
// ============================================================================

fn softmax(
    input: [*]const f32,
    output: [*]f32,
    size: usize,
) void {
    var max_val: f32 = -std.math.inf(f32);
    for (0..size) |i| {
        if (input[i] > max_val) max_val = input[i];
    }

    var sum: f32 = 0;
    for (0..size) |i| {
        output[i] = @exp(input[i] - max_val);
        sum += output[i];
    }

    const inv_sum = 1.0 / sum;
    for (0..size) |i| {
        output[i] *= inv_sum;
    }
}

// ============================================================================
// MLA Core Engine
// ============================================================================

pub const MlaEngine = struct {
    config: MlaConfig,
    cache: *MlaKvCache,
    allocator: std.mem.Allocator,

    // Temporary buffers (allocated once, reused)
    // Q path
    q_a_proj_output: []align(64) f32, // [max_qlen, q_lora_rank]
    q_nope: []align(64) f32, // [num_heads, max_qlen, nope_size]
    q_pe: []align(64) f32, // [num_heads, max_qlen, rope_size]

    // KV path
    kv_a_proj_output: []align(64) f32, // [max_qlen, kv_lora_rank + rope_size]
    compressed_kv: []align(64) f32, // [max_qlen, kv_lora_rank]
    k_pe_buffer: []align(64) f32, // [max_qlen, rope_size]

    // Attention
    q_absorb: []align(64) f32, // [num_heads, max_qlen, kv_lora_rank]
    o_absorb: []align(64) f32, // [num_heads, max_qlen, kv_lora_rank]
    attention_weights: []align(64) f32, // [num_heads, max_qlen, max_kvlen]
    attention_output: []align(64) f32, // [num_heads, max_qlen, nope_size]

    // Final
    combined_output: []align(64) f32, // [max_qlen, num_heads * nope_size]

    pub fn init(allocator: std.mem.Allocator, config: MlaConfig, cache: *MlaKvCache) !MlaEngine {
        const num_heads = config.numHeadsPerTp();
        const max_qlen = config.max_qlen;
        const max_kvlen = config.max_kvlen;

        var self = MlaEngine{
            .config = config,
            .cache = cache,
            .allocator = allocator,
            .q_a_proj_output = try allocator.alignedAlloc(f32, .@"64", max_qlen * config.q_lora_rank),
            .q_nope = try allocator.alignedAlloc(f32, .@"64", num_heads * max_qlen * config.nope_size),
            .q_pe = try allocator.alignedAlloc(f32, .@"64", num_heads * max_qlen * config.rope_size),
            .kv_a_proj_output = try allocator.alignedAlloc(f32, .@"64", max_qlen * (config.kv_lora_rank + config.rope_size)),
            .compressed_kv = try allocator.alignedAlloc(f32, .@"64", max_qlen * config.kv_lora_rank),
            .k_pe_buffer = try allocator.alignedAlloc(f32, .@"64", max_qlen * config.rope_size),
            .q_absorb = try allocator.alignedAlloc(f32, .@"64", num_heads * max_qlen * config.kv_lora_rank),
            .o_absorb = try allocator.alignedAlloc(f32, .@"64", num_heads * max_qlen * config.kv_lora_rank),
            .attention_weights = try allocator.alignedAlloc(f32, .@"64", num_heads * max_qlen * max_kvlen),
            .attention_output = try allocator.alignedAlloc(f32, .@"64", num_heads * max_qlen * config.nope_size),
            .combined_output = try allocator.alignedAlloc(f32, .@"64", max_qlen * num_heads * config.nope_size),
        };
        errdefer self.deinit();

        // Zero attention weights so the per-head accumulate-and-softmax flow
        // (computePeAttention stores, computeNopeAttention accumulates) starts clean.
        @memset(self.attention_weights, 0);
        return self;
    }

    pub fn deinit(self: *MlaEngine) void {
        self.allocator.free(self.q_a_proj_output);
        self.allocator.free(self.q_nope);
        self.allocator.free(self.q_pe);
        self.allocator.free(self.kv_a_proj_output);
        self.allocator.free(self.compressed_kv);
        self.allocator.free(self.k_pe_buffer);
        self.allocator.free(self.q_absorb);
        self.allocator.free(self.o_absorb);
        self.allocator.free(self.attention_weights);
        self.allocator.free(self.attention_output);
        self.allocator.free(self.combined_output);
    }

    // =======================================================================
    // Q Projection: hidden_size -> q_lora_rank -> (nope_size + rope_size) per head
    // Exposed as pub for white-box testing of RoPE absolute-position handling.
    // =======================================================================

    pub fn projectQ(
        self: *MlaEngine,
        input: [*]const f32, // [qlen, hidden_size]
        qlen: usize,
        kv_start_pos: usize,
    ) void {
        const cfg = self.config;
        const num_heads = cfg.numHeadsPerTp();

        // Step 1: q_a_proj: [qlen, hidden_size] x [q_lora_rank, hidden_size]^T -> [qlen, q_lora_rank]
        matmulF32(
            input,
            cfg.hidden_size,
            cfg.q_a_proj,
            cfg.hidden_size,
            self.q_a_proj_output.ptr,
            cfg.q_lora_rank,
            qlen,
            cfg.q_lora_rank,
            cfg.hidden_size,
        );

        // Step 2: RMSNorm on q_lora_rank dimension
        for (0..qlen) |i| {
            rmsNorm(
                self.q_a_proj_output.ptr + i * cfg.q_lora_rank,
                cfg.q_a_norm,
                self.q_a_proj_output.ptr + i * cfg.q_lora_rank,
                cfg.q_lora_rank,
                1e-6,
            );
        }

        // Step 3: q_b_proj per head: [qlen, q_lora_rank] x [nope_size+rope_size, q_lora_rank]^T
        // Split into q_nope and q_pe
        for (0..num_heads) |h| {
            const q_b_proj_head = cfg.q_b_proj + h * (cfg.nope_size + cfg.rope_size) * cfg.q_lora_rank;

            // q_nope: [qlen, nope_size]
            matmulF32(
                self.q_a_proj_output.ptr,
                cfg.q_lora_rank,
                q_b_proj_head,
                cfg.q_lora_rank,
                self.q_nope.ptr + h * qlen * cfg.nope_size,
                cfg.nope_size,
                qlen,
                cfg.nope_size,
                cfg.q_lora_rank,
            );

            // q_pe: [qlen, rope_size]
            matmulF32Offset(
                self.q_a_proj_output.ptr,
                cfg.q_lora_rank,
                q_b_proj_head,
                cfg.q_lora_rank,
                cfg.nope_size,
                self.q_pe.ptr + h * qlen * cfg.rope_size,
                cfg.rope_size,
                qlen,
                cfg.rope_size,
                cfg.q_lora_rank,
            );

            // Apply RoPE to q_pe at the ABSOLUTE position (fix vs draft:
            // rotation angle must reflect the token's cache position, not
            // its offset within the current batch).
            for (0..qlen) |pos| {
                applyRopeKpe(
                    self.q_pe.ptr + h * qlen * cfg.rope_size + pos * cfg.rope_size,
                    kv_start_pos + pos,
                    cfg.rope_size,
                    cfg.rope_theta,
                );
            }
        }
    }

    // =======================================================================
    // KV Projection: hidden_size -> kv_lora_rank + rope_size
    // =======================================================================

    fn projectKV(
        self: *MlaEngine,
        input: [*]const f32, // [qlen, hidden_size]
        qlen: usize,
        kv_start_pos: usize, // Starting position in KV cache
    ) !void {
        const cfg = self.config;

        // Step 1: kv_a_proj_with_mqa: [qlen, hidden_size] x [kv_lora_rank+rope_size, hidden_size]^T
        matmulF32(
            input,
            cfg.hidden_size,
            cfg.kv_a_proj_with_mqa,
            cfg.hidden_size,
            self.kv_a_proj_output.ptr,
            cfg.kv_lora_rank + cfg.rope_size,
            qlen,
            cfg.kv_lora_rank + cfg.rope_size,
            cfg.hidden_size,
        );

        // Step 2: Split into compressed_kv and k_pe, apply RMSNorm to compressed part
        for (0..qlen) |i| {
            const kv_output = self.kv_a_proj_output.ptr + i * (cfg.kv_lora_rank + cfg.rope_size);

            // compressed_kv: first kv_lora_rank elements (RMSNorm'd)
            rmsNorm(
                kv_output,
                cfg.kv_a_norm,
                self.compressed_kv.ptr + i * cfg.kv_lora_rank,
                cfg.kv_lora_rank,
                1e-6,
            );

            // k_pe: last rope_size elements
            const dst = self.k_pe_buffer.ptr + i * cfg.rope_size;
            @memcpy(dst[0..cfg.rope_size], (kv_output + cfg.kv_lora_rank)[0..cfg.rope_size]);

            // Apply RoPE to k_pe
            applyRopeKpe(
                dst,
                kv_start_pos + i,
                cfg.rope_size,
                cfg.rope_theta,
            );
        }

        // Step 3: Store in KV cache
        for (0..qlen) |i| {
            try self.cache.appendToken(
                self.compressed_kv.ptr + i * cfg.kv_lora_rank,
                self.k_pe_buffer.ptr + i * cfg.rope_size,
            );
        }
    }

    // =======================================================================
    // PE Attention scores: q_pe @ k_pe^T (no softmax here)
    // =======================================================================

    fn computePeScores(
        self: *MlaEngine,
        qlen: usize,
        kvlen: usize,
        head_idx: usize,
    ) void {
        const cfg = self.config;
        const q_pe_head = self.q_pe.ptr + head_idx * qlen * cfg.rope_size;
        const attn_weights_head = self.attention_weights.ptr + head_idx * qlen * cfg.max_kvlen;

        for (0..qlen) |q_pos| {
            for (0..kvlen) |kv_pos| {
                const k_pe = self.cache.getRopePtr(kv_pos);

                // Dot product: q_pe[q_pos] . k_pe[kv_pos]
                var score: f32 = 0;
                var i: usize = 0;
                while (i + 16 <= cfg.rope_size) : (i += 16) {
                    const qv: Vec16f32 = loadVec16(q_pe_head + q_pos * cfg.rope_size + i);
                    const kv = loadVec16(k_pe + i);
                    score += @reduce(.Add, qv * kv);
                }
                while (i < cfg.rope_size) : (i += 1) {
                    score += q_pe_head[q_pos * cfg.rope_size + i] * k_pe[i];
                }

                attn_weights_head[q_pos * cfg.max_kvlen + kv_pos] = score;
            }
        }
    }

    // =======================================================================
    // Nope Attention (Absorbed): q_nope @ W_UK^T -> q_absorb
    // =======================================================================

    fn absorbWuk(
        self: *MlaEngine,
        qlen: usize,
        head_idx: usize,
    ) void {
        const cfg = self.config;
        const q_nope_head = self.q_nope.ptr + head_idx * qlen * cfg.nope_size;
        const q_absorb_head = self.q_absorb.ptr + head_idx * qlen * cfg.kv_lora_rank;

        // W_UK is part of kv_b_proj: [head_idx, 0:nope_size, :]
        const w_uk = cfg.kv_b_proj + head_idx * 2 * cfg.nope_size * cfg.kv_lora_rank;

        // q_absorb = q_nope @ W_UK^T
        // [qlen, nope_size] x [kv_lora_rank, nope_size]^T -> [qlen, kv_lora_rank]
        matmulF32(
            q_nope_head,
            cfg.nope_size,
            w_uk,
            cfg.nope_size,
            q_absorb_head,
            cfg.kv_lora_rank,
            qlen,
            cfg.kv_lora_rank,
            cfg.nope_size,
        );
    }

    // =======================================================================
    // Nope Attention scores: q_absorb @ compressed_kv^T (accumulates onto PE scores)
    // =======================================================================

    fn computeNopeScores(
        self: *MlaEngine,
        qlen: usize,
        kvlen: usize,
        head_idx: usize,
    ) void {
        const cfg = self.config;
        const q_absorb_head = self.q_absorb.ptr + head_idx * qlen * cfg.kv_lora_rank;
        const attn_weights_head = self.attention_weights.ptr + head_idx * qlen * cfg.max_kvlen;

        for (0..qlen) |q_pos| {
            const q_abs = q_absorb_head + q_pos * cfg.kv_lora_rank;
            for (0..kvlen) |kv_pos| {
                const ckv = self.cache.getNopePtr(kv_pos);

                var score: f32 = 0;
                var i: usize = 0;
                while (i + 16 <= cfg.kv_lora_rank) : (i += 16) {
                    const qv: Vec16f32 = loadVec16(q_abs + i);
                    const kv = loadVec16(ckv + i);
                    score += @reduce(.Add, qv * kv);
                }
                while (i < cfg.kv_lora_rank) : (i += 1) {
                    score += q_abs[i] * ckv[i];
                }

                // Accumulate with PE scores (no softmax in between — see fix note)
                attn_weights_head[q_pos * cfg.max_kvlen + kv_pos] += score;
            }
        }
    }

    // =======================================================================
    // Single softmax per query row over combined PE+nope scores
    // =======================================================================

    fn softmaxWeights(
        self: *MlaEngine,
        qlen: usize,
        kvlen: usize,
        head_idx: usize,
    ) void {
        const cfg = self.config;
        const attn_weights_head = self.attention_weights.ptr + head_idx * qlen * cfg.max_kvlen;
        for (0..qlen) |q_pos| {
            softmax(
                attn_weights_head + q_pos * cfg.max_kvlen,
                attn_weights_head + q_pos * cfg.max_kvlen,
                kvlen,
            );
        }
    }

    // =======================================================================
    // o_absorb = attention_weights @ compressed_kv
    // =======================================================================

    fn computeOAbsorb(
        self: *MlaEngine,
        qlen: usize,
        kvlen: usize,
        head_idx: usize,
    ) void {
        const cfg = self.config;
        const attn_weights_head = self.attention_weights.ptr + head_idx * qlen * cfg.max_kvlen;
        const o_absorb_head = self.o_absorb.ptr + head_idx * qlen * cfg.kv_lora_rank;

        for (0..qlen) |q_pos| {
            const dst_row = o_absorb_head + q_pos * cfg.kv_lora_rank;
            @memset(dst_row[0..cfg.kv_lora_rank], 0);

            for (0..kvlen) |kv_pos| {
                const weight = attn_weights_head[q_pos * cfg.max_kvlen + kv_pos];
                const ckv = self.cache.getNopePtr(kv_pos);

                var i: usize = 0;
                while (i + 16 <= cfg.kv_lora_rank) : (i += 16) {
                    const w: Vec16f32 = @splat(weight);
                    var dst: Vec16f32 = loadVec16(dst_row + i);
                    const src = loadVec16(ckv + i);
                    dst += w * src;
                    (dst_row + i)[0..16].* = dst;
                }
                while (i < cfg.kv_lora_rank) : (i += 1) {
                    dst_row[i] += weight * ckv[i];
                }
            }
        }
    }

    // =======================================================================
    // attention_output = o_absorb @ W_UV^T
    // =======================================================================

    fn absorbWuv(
        self: *MlaEngine,
        qlen: usize,
        head_idx: usize,
    ) void {
        const cfg = self.config;
        const o_absorb_head = self.o_absorb.ptr + head_idx * qlen * cfg.kv_lora_rank;
        const attn_out_head = self.attention_output.ptr + head_idx * qlen * cfg.nope_size;

        // W_UV is part of kv_b_proj: [head_idx, nope_size:2*nope_size, :]
        const w_uv = cfg.kv_b_proj + (head_idx * 2 + 1) * cfg.nope_size * cfg.kv_lora_rank;

        // attention_output = o_absorb @ W_UV^T
        // [qlen, kv_lora_rank] x [nope_size, kv_lora_rank]^T -> [qlen, nope_size]
        matmulF32(
            o_absorb_head,
            cfg.kv_lora_rank,
            w_uv,
            cfg.kv_lora_rank,
            attn_out_head,
            cfg.nope_size,
            qlen,
            cfg.nope_size,
            cfg.kv_lora_rank,
        );
    }

    // =======================================================================
    // Combine heads and project to hidden_size
    // =======================================================================

    fn combineAndProject(
        self: *MlaEngine,
        output: [*]f32,
        qlen: usize,
    ) void {
        const cfg = self.config;
        const num_heads = cfg.numHeadsPerTp();

        // Concatenate all heads: [qlen, num_heads * nope_size]
        for (0..qlen) |pos| {
            for (0..num_heads) |h| {
                const src = self.attention_output.ptr + h * qlen * cfg.nope_size + pos * cfg.nope_size;
                const dst = self.combined_output.ptr + pos * num_heads * cfg.nope_size + h * cfg.nope_size;
                @memcpy(dst[0..cfg.nope_size], src[0..cfg.nope_size]);
            }
        }

        // o_proj: [qlen, num_heads*nope_size] x [hidden_size, num_heads*nope_size]^T -> [qlen, hidden_size]
        matmulF32(
            self.combined_output.ptr,
            num_heads * cfg.nope_size,
            cfg.o_proj,
            num_heads * cfg.nope_size,
            output,
            cfg.hidden_size,
            qlen,
            cfg.hidden_size,
            num_heads * cfg.nope_size,
        );
    }

    // =======================================================================
    // Main Forward Pass
    // =======================================================================

    /// Forward pass for prefill (multiple query tokens)
    /// input: [qlen, hidden_size] f32
    /// output: [qlen, hidden_size] f32
    /// kv_start_pos: starting position in KV cache (for incremental decoding)
    pub fn forward(
        self: *MlaEngine,
        input: [*]const f32,
        output: [*]f32,
        qlen: usize,
        kv_start_pos: usize,
    ) !void {
        const cfg = self.config;
        const num_heads = cfg.numHeadsPerTp();
        const kvlen = kv_start_pos + qlen;

        // 1. Project Q (RoPE at absolute positions)
        self.projectQ(input, qlen, kv_start_pos);

        // 2. Project KV and store in cache
        try self.projectKV(input, qlen, kv_start_pos);

        // 3. Attention per head
        for (0..num_heads) |h| {
            // 3a. Absorb W_UK: q_nope -> q_absorb
            self.absorbWuk(qlen, h);

            // 3b. PE scores (stores into attention_weights)
            self.computePeScores(qlen, kvlen, h);

            // 3c. Nope scores (accumulates onto PE scores)
            self.computeNopeScores(qlen, kvlen, h);

            // 3d. ONE softmax over combined PE + nope scores (fix vs draft)
            self.softmaxWeights(qlen, kvlen, h);

            // 3e. o_absorb = weights @ compressed_kv
            self.computeOAbsorb(qlen, kvlen, h);

            // 3f. Absorb W_UV: o_absorb -> attention_output
            self.absorbWuv(qlen, h);
        }

        // 4. Combine heads and project
        self.combineAndProject(output, qlen);
    }

    /// Forward pass for single token decode (incremental)
    /// input: [1, hidden_size] f32
    /// output: [1, hidden_size] f32
    pub fn decode(
        self: *MlaEngine,
        input: [*]const f32,
        output: [*]f32,
        position: usize,
    ) !void {
        // Same as forward but with qlen=1 and kvlen=position+1
        try self.forward(input, output, 1, position);
    }

    /// Reset KV cache
    pub fn resetCache(self: *MlaEngine) void {
        self.cache.clear();
    }

    // ===================================================================
    // Paged / batched forward (kt_mla_forward, qlen_count > 1)
    // ===================================================================
    //
    // Mirrors the C++ TP_MLA_Common::forward contract (mla-tp.hpp:84):
    //   forward(qlens, page_tables, kv_lens, input, output)
    // — a batch of sequences, each with its own page table and KV length.
    // Pages are the engine's own MlaKvCache pages (the C++ registers
    // external pages via set_pages; the Zig C API's kt_mla_new pre-allocates
    // config.page_count pages and the page tables index into those).
    //
    // Page-table format (C ABI, c_int per entry):
    //   page_table[logical_pos / token_count_in_page] = physical page idx
    //
    // The new KVs (qlen tokens per sequence) are written at logical
    // positions [kv_len - qlen, kv_len) — the scheduler is expected to have
    // reserved those slots — and attention attends over [0, kv_len).
    //
    // Sequences are processed sequentially through the shared engine
    // (weights and scratch are reused; attention is per-sequence so
    // there is no cross-sequence interference).

    /// Paged variant of computePeScores: q_pe @ k_pe^T with the k_pe read
    /// through the page table.
    fn pagedComputePeScores(
        self: *MlaEngine,
        page_table: [*]const c_int,
        qlen: usize,
        kvlen: usize,
        head_idx: usize,
    ) void {
        const cfg = self.config;
        const q_pe_head = self.q_pe.ptr + head_idx * qlen * cfg.rope_size;
        const attn_weights_head = self.attention_weights.ptr + head_idx * qlen * cfg.max_kvlen;

        for (0..qlen) |q_pos| {
            for (0..kvlen) |kv_pos| {
                const k_pe = self.cache.pagedGetRopePtr(page_table, kv_pos);

                var score: f32 = 0;
                var i: usize = 0;
                while (i + 16 <= cfg.rope_size) : (i += 16) {
                    const qv: Vec16f32 = loadVec16(q_pe_head + q_pos * cfg.rope_size + i);
                    const kv = loadVec16(k_pe + i);
                    score += @reduce(.Add, qv * kv);
                }
                while (i < cfg.rope_size) : (i += 1) {
                    score += q_pe_head[q_pos * cfg.rope_size + i] * k_pe[i];
                }

                attn_weights_head[q_pos * cfg.max_kvlen + kv_pos] = score;
            }
        }
    }

    /// Paged variant of computeNopeScores: q_absorb @ compressed_kv^T
    /// through the page table, accumulating onto the PE scores.
    fn pagedComputeNopeScores(
        self: *MlaEngine,
        page_table: [*]const c_int,
        qlen: usize,
        kvlen: usize,
        head_idx: usize,
    ) void {
        const cfg = self.config;
        const q_absorb_head = self.q_absorb.ptr + head_idx * qlen * cfg.kv_lora_rank;
        const attn_weights_head = self.attention_weights.ptr + head_idx * qlen * cfg.max_kvlen;

        for (0..qlen) |q_pos| {
            const q_abs = q_absorb_head + q_pos * cfg.kv_lora_rank;
            for (0..kvlen) |kv_pos| {
                const ckv = self.cache.pagedGetNopePtr(page_table, kv_pos);

                var score: f32 = 0;
                var i: usize = 0;
                while (i + 16 <= cfg.kv_lora_rank) : (i += 16) {
                    const qv: Vec16f32 = loadVec16(q_abs + i);
                    const kv = loadVec16(ckv + i);
                    score += @reduce(.Add, qv * kv);
                }
                while (i < cfg.kv_lora_rank) : (i += 1) {
                    score += q_abs[i] * ckv[i];
                }

                attn_weights_head[q_pos * cfg.max_kvlen + kv_pos] += score;
            }
        }
    }

    /// Paged variant of computeOAbsorb: o_absorb = weights @ compressed_kv
    /// through the page table.
    fn pagedComputeOAbsorb(
        self: *MlaEngine,
        page_table: [*]const c_int,
        qlen: usize,
        kvlen: usize,
        head_idx: usize,
    ) void {
        const cfg = self.config;
        const attn_weights_head = self.attention_weights.ptr + head_idx * qlen * cfg.max_kvlen;
        const o_absorb_head = self.o_absorb.ptr + head_idx * qlen * cfg.kv_lora_rank;

        for (0..qlen) |q_pos| {
            const dst_row = o_absorb_head + q_pos * cfg.kv_lora_rank;
            @memset(dst_row[0..cfg.kv_lora_rank], 0);

            for (0..kvlen) |kv_pos| {
                const weight = attn_weights_head[q_pos * cfg.max_kvlen + kv_pos];
                const ckv = self.cache.pagedGetNopePtr(page_table, kv_pos);

                var i: usize = 0;
                while (i + 16 <= cfg.kv_lora_rank) : (i += 16) {
                    const w: Vec16f32 = @splat(weight);
                    var dst: Vec16f32 = loadVec16(dst_row + i);
                    const src = loadVec16(ckv + i);
                    dst += w * src;
                    (dst_row + i)[0..16].* = dst;
                }
                while (i < cfg.kv_lora_rank) : (i += 1) {
                    dst_row[i] += weight * ckv[i];
                }
            }
        }
    }

    /// Paged forward for ONE sequence of the batch: projects Q and the new
    /// KVs, writes the KVs through the page table, attends over [0, kv_len),
    /// writes [qlen, hidden_size] into `output`.
    fn forwardPagedSequence(
        self: *MlaEngine,
        input: [*]const f32, // [qlen, hidden_size]
        output: [*]f32, // [qlen, hidden_size]
        page_table: [*]const c_int,
        qlen: usize,
        kv_len: usize,
    ) !void {
        const cfg = self.config;
        const num_heads = cfg.numHeadsPerTp();
        const kv_start_pos = kv_len - qlen;

        // 1. Project Q (RoPE at absolute positions kv_start_pos..kv_len)
        self.projectQ(input, qlen, kv_start_pos);

        // 2. Project the new KVs (kv_a_proj + RMSNorm + RoPE at absolute
        //    positions) into the engine scratch, same as projectKV does,
        //    but write through the page table instead of appendToken.
        matmulF32(
            input,
            cfg.hidden_size,
            cfg.kv_a_proj_with_mqa,
            cfg.hidden_size,
            self.kv_a_proj_output.ptr,
            cfg.kv_lora_rank + cfg.rope_size,
            qlen,
            cfg.kv_lora_rank + cfg.rope_size,
            cfg.hidden_size,
        );
        for (0..qlen) |i| {
            const kv_output = self.kv_a_proj_output.ptr + i * (cfg.kv_lora_rank + cfg.rope_size);
            rmsNorm(
                kv_output,
                cfg.kv_a_norm,
                self.compressed_kv.ptr + i * cfg.kv_lora_rank,
                cfg.kv_lora_rank,
                1e-6,
            );
            const dst = self.k_pe_buffer.ptr + i * cfg.rope_size;
            @memcpy(dst[0..cfg.rope_size], (kv_output + cfg.kv_lora_rank)[0..cfg.rope_size]);
            applyRopeKpe(dst, kv_start_pos + i, cfg.rope_size, cfg.rope_theta);
        }
        // Grow the page pool if the caller's table references pages we
        // haven't allocated yet (kt_mla_new sized it from max_kvlen, but a
        // table could exceed that if the caller re-uses slots).
        const pages_needed: usize = @intCast(page_table[(kv_len - 1) / cfg.token_count_in_page] + 1);
        _ = try self.cache.ensurePageCount(pages_needed);
        for (0..qlen) |i| {
            try self.cache.pagedWriteToken(
                page_table,
                kv_start_pos + i,
                self.compressed_kv.ptr + i * cfg.kv_lora_rank,
                self.k_pe_buffer.ptr + i * cfg.rope_size,
            );
        }

        // 3. Attention per head (paged reads)
        for (0..num_heads) |h| {
            self.absorbWuk(qlen, h);
            self.pagedComputePeScores(page_table, qlen, kv_len, h);
            self.pagedComputeNopeScores(page_table, qlen, kv_len, h);
            self.softmaxWeights(qlen, kv_len, h);
            self.pagedComputeOAbsorb(page_table, qlen, kv_len, h);
            self.absorbWuv(qlen, h);
        }

        // 4. Combine heads and project
        self.combineAndProject(output, qlen);
    }

    /// Batched paged forward — the full kt_mla_forward contract.
    /// input/output are the concatenation of the batch's sequences
    /// ([sum(qlens), hidden_size]); each sequence attends over its own
    /// page table / KV length. Sequences are processed sequentially
    /// through the shared engine (correct, single-threaded semantics).
    pub fn forwardPaged(
        self: *MlaEngine,
        input: [*]const f32,
        output: [*]f32,
        qlens: []const c_int,
        page_tables: []const [*]const c_int,
        kv_lens: []const c_int,
    ) !void {
        var in_off: usize = 0;
        var out_off: usize = 0;
        for (qlens, 0..) |qlen_c, seq| {
            const qlen: usize = @intCast(qlen_c);
            const kv_len: usize = @intCast(kv_lens[seq]);
            if (qlen == 0) continue;
            if (kv_len < qlen) return error.InvalidKvLen;
            try self.forwardPagedSequence(
                input + in_off * self.config.hidden_size,
                output + out_off * self.config.hidden_size,
                page_tables[seq],
                qlen,
                kv_len,
            );
            in_off += qlen;
            out_off += qlen;
        }
    }
};

// ============================================================================
// Tests for internal helpers (BF16 conversion round-trip)
// ============================================================================

test "bf16 round trip" {
    const vals = [_]f32{ 0.5, 1.0, -2.0, 3.14159, 100.0, -0.25 };
    for (vals) |v| {
        const b = f32ToBf16(v);
        const r = bf16ToF32(b);
        try std.testing.expectApproxEqRel(v, r, 0.01);
    }
}

test "rmsNorm known values" {
    var input: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var weight: [4]u16 = .{ 0x3F80, 0x3F80, 0x3F80, 0x3F80 }; // 1.0 in BF16
    var output: [4]f32 = undefined;
    rmsNorm(&input, &weight, &output, 4, 1e-6);
    // RMS of input = sqrt((1+4+9+16)/4) = sqrt(7.5)
    const inv = 1.0 / @sqrt(7.5 + 1e-6);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(input[i] * inv, output[i], 1e-5);
    }
}

test "softmax normalizes" {
    var buf: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    softmax(&buf, &out, 4);
    var sum: f32 = 0;
    for (out) |v| sum += v;
    try std.testing.expectApproxEqAbs(1.0, sum, 1e-5);
    try std.testing.expect(out[3] > out[0]); // monotonic
}

test "rope rotate preserves norm" {
    var v: [8]f32 = undefined;
    for (0..8) |i| v[i] = @floatFromInt(i + 1);
    const norm_before = std.math.sqrt(std.math.pow(f32, v[0], 2) + std.math.pow(f32, v[4], 2));
    ropeRotate(&v, 12345, 8, 10000.0);
    const norm_after = std.math.sqrt(std.math.pow(f32, v[0], 2) + std.math.pow(f32, v[4], 2));
    // Pair (v[0], v[4]) is rotated: norm preserved
    try std.testing.expectApproxEqAbs(norm_before, norm_after, 1e-4);
}
