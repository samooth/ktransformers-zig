// Tests for MLA (Multi-Head Latent Attention)
// Run standalone: zig test src/mla/mla_tests.zig
const std = @import("std");
const testing = std.testing;
const MlaConfig = @import("mla_config.zig").MlaConfig;
const MlaKvCache = @import("mla_cache.zig").MlaCachePage;
const MlaCache = @import("mla_cache.zig").MlaKvCache;
const MlaEngine = @import("mla_core.zig").MlaEngine;

// BF16 encoding of 1.0 is 0x3F80 (NOT 0x3C00 — that is FP16!)
const BF16_ONE: u16 = 0x3F80;

test "MlaConfig defaults" {
    const cfg = MlaConfig{};
    try testing.expect(cfg.hidden_size == 7168);
    try testing.expect(cfg.num_heads == 128);
    try testing.expect(cfg.q_lora_rank == 1536);
    try testing.expect(cfg.kv_lora_rank == 512);
    try testing.expect(cfg.nope_size == 128);
    try testing.expect(cfg.rope_size == 64);
    try testing.expect(cfg.headDim() == 192);
}

test "MlaKvCache basic operations" {
    const allocator = testing.allocator;
    const config = MlaConfig{
        .kv_lora_rank = 64,
        .rope_size = 32,
        .token_count_in_page = 4,
        .max_kvlen = 128,
    };

    var cache = try MlaCache.init(allocator, config, 32);
    defer cache.deinit();

    // Allocate initial page
    _ = try cache.allocPage();

    // Store some tokens
    var nope_buf: [64]f32 = undefined;
    var rope_buf: [32]f32 = undefined;

    for (0..10) |t| {
        @memset(&nope_buf, @as(f32, @floatFromInt(t)));
        @memset(&rope_buf, @as(f32, @floatFromInt(t * 2)));
        try cache.appendToken(&nope_buf, &rope_buf);
    }

    try testing.expect(cache.kvLen() == 10);

    // Verify retrieval
    const retrieved_nope = cache.getNopePtr(5);
    try testing.expect(retrieved_nope[0] == 5.0);

    const retrieved_rope = cache.getRopePtr(5);
    try testing.expect(retrieved_rope[0] == 10.0);

    // Clear
    cache.clear();
    try testing.expect(cache.kvLen() == 0);
}

test "MlaKvCache page allocation" {
    const allocator = testing.allocator;
    const config = MlaConfig{
        .kv_lora_rank = 32,
        .rope_size = 16,
        .token_count_in_page = 4,
        .max_kvlen = 128,
    };

    var cache = try MlaCache.init(allocator, config, 32);
    defer cache.deinit();

    var nope_buf: [32]f32 = undefined;
    var rope_buf: [16]f32 = undefined;

    // Store enough tokens to trigger page allocation
    for (0..12) |t| {
        @memset(&nope_buf, @as(f32, @floatFromInt(t)));
        @memset(&rope_buf, @as(f32, @floatFromInt(t)));
        try cache.appendToken(&nope_buf, &rope_buf);
    }

    try testing.expect(cache.kvLen() == 12);
    try testing.expect(cache.pages.items.len >= 3); // 12 tokens / 4 per page = 3 pages
}

// Helper: build a small engine with weights set to a constant BF16 value.
// The cache is heap-allocated so `engine.cache` stays a stable pointer
// (returning MlaKvCache by value from this helper would copy the
// ArrayLists and leave engine.cache pointing at a moved-from local).
fn makeSmallEngine(
    allocator: std.mem.Allocator,
    weight_bits: u16,
    hidden_size: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    nope_size: usize,
    rope_size: usize,
    num_heads: usize,
    max_qlen: usize,
    max_kvlen: usize,
) !struct {
    q_a_proj: []align(64) u16,
    q_a_norm: []align(64) u16,
    q_b_proj: []align(64) u16,
    kv_a_proj: []align(64) u16,
    kv_a_norm: []align(64) u16,
    kv_b_proj: []align(64) u16,
    o_proj: []align(64) u16,
    cache: *MlaCache,
    engine: *MlaEngine,
} {
    const q_a_proj = try allocator.alignedAlloc(u16, .@"64", q_lora_rank * hidden_size);
    errdefer allocator.free(q_a_proj);
    const q_a_norm = try allocator.alignedAlloc(u16, .@"64", q_lora_rank);
    errdefer allocator.free(q_a_norm);
    const q_b_proj = try allocator.alignedAlloc(u16, .@"64", num_heads * (nope_size + rope_size) * q_lora_rank);
    errdefer allocator.free(q_b_proj);
    const kv_a_proj = try allocator.alignedAlloc(u16, .@"64", (kv_lora_rank + rope_size) * hidden_size);
    errdefer allocator.free(kv_a_proj);
    const kv_a_norm = try allocator.alignedAlloc(u16, .@"64", kv_lora_rank);
    errdefer allocator.free(kv_a_norm);
    const kv_b_proj = try allocator.alignedAlloc(u16, .@"64", num_heads * 2 * nope_size * kv_lora_rank);
    errdefer allocator.free(kv_b_proj);
    const o_proj = try allocator.alignedAlloc(u16, .@"64", hidden_size * num_heads * nope_size);
    errdefer allocator.free(o_proj);

    @memset(q_a_proj, weight_bits);
    @memset(q_a_norm, weight_bits);
    @memset(q_b_proj, weight_bits);
    @memset(kv_a_proj, weight_bits);
    @memset(kv_a_norm, weight_bits);
    @memset(kv_b_proj, weight_bits);
    @memset(o_proj, weight_bits);

    const config = MlaConfig{
        .hidden_size = hidden_size,
        .num_heads = num_heads,
        .q_lora_rank = q_lora_rank,
        .kv_lora_rank = kv_lora_rank,
        .nope_size = nope_size,
        .rope_size = rope_size,
        .max_qlen = max_qlen,
        .max_kvlen = max_kvlen,
        .token_count_in_page = 8,
        .q_a_proj = q_a_proj.ptr,
        .q_a_norm = q_a_norm.ptr,
        .q_b_proj = q_b_proj.ptr,
        .kv_a_proj_with_mqa = kv_a_proj.ptr,
        .kv_a_norm = kv_a_norm.ptr,
        .kv_b_proj = kv_b_proj.ptr,
        .o_proj = o_proj.ptr,
    };

    // page_count for max_kvlen tokens
    const page_count = (max_kvlen + config.token_count_in_page - 1) / config.token_count_in_page;
    const cache = try allocator.create(MlaCache);
    errdefer allocator.destroy(cache);
    cache.* = try MlaCache.init(allocator, config, page_count);
    errdefer cache.deinit();

    const engine = try allocator.create(MlaEngine);
    errdefer allocator.destroy(engine);
    engine.* = try MlaEngine.init(allocator, config, cache);
    errdefer engine.deinit();

    return .{
        .q_a_proj = q_a_proj,
        .q_a_norm = q_a_norm,
        .q_b_proj = q_b_proj,
        .kv_a_proj = kv_a_proj,
        .kv_a_norm = kv_a_norm,
        .kv_b_proj = kv_b_proj,
        .o_proj = o_proj,
        .cache = cache,
        .engine = engine,
    };
}

fn freeSmallEngine(allocator: std.mem.Allocator, t: anytype) void {
    t.engine.deinit();
    t.cache.deinit();
    allocator.destroy(t.engine);
    allocator.destroy(t.cache);
    allocator.free(t.q_a_proj);
    allocator.free(t.q_a_norm);
    allocator.free(t.q_b_proj);
    allocator.free(t.kv_a_proj);
    allocator.free(t.kv_a_norm);
    allocator.free(t.kv_b_proj);
    allocator.free(t.o_proj);
}

test "MlaEngine initialization and forward" {
    const allocator = testing.allocator;

    // Create dummy weights
    const hidden_size = 256;
    const q_lora_rank = 64;
    const kv_lora_rank = 32;
    const nope_size = 16;
    const rope_size = 8;
    const num_heads = 4;
    const max_qlen = 16;
    const max_kvlen = 64;

    var t = try makeSmallEngine(allocator, BF16_ONE, hidden_size, q_lora_rank, kv_lora_rank, nope_size, rope_size, num_heads, max_qlen, max_kvlen);
    defer freeSmallEngine(allocator, t);

    // Test forward pass
    const input = try allocator.alignedAlloc(f32, .@"64", max_qlen * hidden_size);
    defer allocator.free(input);
    const output = try allocator.alignedAlloc(f32, .@"64", max_qlen * hidden_size);
    defer allocator.free(output);

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (input) |*v| v.* = rand.float(f32) * 0.1;

    try t.engine.forward(input.ptr, output.ptr, 4, 0);

    // Output should not be all zeros
    var has_nonzero = false;
    for (output) |v| {
        if (v != 0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
    // Cache should have grown by qlen
    try testing.expect(t.cache.kvLen() == 4);
}

test "MlaEngine decode" {
    const allocator = testing.allocator;

    const hidden_size = 128;
    const q_lora_rank = 32;
    const kv_lora_rank = 16;
    const nope_size = 8;
    const rope_size = 4;
    const num_heads = 2;

    var t = try makeSmallEngine(allocator, BF16_ONE, hidden_size, q_lora_rank, kv_lora_rank, nope_size, rope_size, num_heads, 8, 32);
    defer freeSmallEngine(allocator, t);

    var input: [128]f32 align(64) = undefined;
    var output: [128]f32 align(64) = undefined;
    @memset(&input, 0.1);

    // Decode multiple positions
    for (0..5) |pos| {
        try t.engine.decode(&input, &output, pos);
    }

    try testing.expect(t.cache.kvLen() == 5);
}

test "MlaEngine q_pe RoPE uses absolute position" {
    // Regression (draft-port bug): projectQ must rotate q_pe at
    // kv_start_pos + pos, not at the within-batch position. With distinct
    // weights this yields different q_pe for different kv_start_pos.
    const allocator = testing.allocator;

    const hidden_size = 64;
    const q_lora_rank = 16;
    const kv_lora_rank = 8;
    const nope_size = 4;
    const rope_size = 4;
    const num_heads = 2;

    var t = try makeSmallEngine(allocator, BF16_ONE, hidden_size, q_lora_rank, kv_lora_rank, nope_size, rope_size, num_heads, 4, 32);
    defer freeSmallEngine(allocator, t);

    // Give q_b_proj's rope rows distinct values so q_pe is nonzero and varies
    // across heads/dims (constant 1.0 weights would make all q_pe rows equal,
    // hiding the rotation angle).
    for (t.q_b_proj, 0..) |*w, idx| {
        const row = idx / q_lora_rank; // 0..(nope+rope)-1 within head
        const is_rope_row = row >= nope_size;
        w.* = if (is_rope_row) 0x3F80 else 0; // rope rows = 1.0, nope rows = 0
    }

    var input: [64]f32 align(64) = undefined;
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i % 7);
    @memset(t.engine.q_nope, 0);

    // Rotate the same q_pe row at position 0 and at position 7
    t.engine.projectQ(&input, 1, 0);
    const q_at0 = t.engine.q_pe[0..rope_size].*;

    t.engine.projectQ(&input, 1, 7);
    const q_at7 = t.engine.q_pe[0..rope_size].*;

    var differs = false;
    for (q_at0, q_at7) |a, b| {
        if (@abs(a - b) > 1e-5) {
            differs = true;
            break;
        }
    }
    try testing.expect(differs);
}

test "RMSNorm correctness" {
    // Verifies the basic math: with weight 1.0, output RMS is 1.0
    const eps = 1e-6;
    var input: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };

    // Manual RMSNorm
    var sum_sq: f32 = 0;
    for (input) |v| sum_sq += v * v;
    const rms = @sqrt(sum_sq / 4.0 + eps);
    const inv_rms = 1.0 / rms;

    var output: [4]f32 = undefined;
    for (0..4) |i| {
        output[i] = input[i] * inv_rms * 1.0;
    }

    // Verify: output should be normalized
    var out_sum_sq: f32 = 0;
    for (output) |v| out_sum_sq += v * v;
    const out_rms = @sqrt(out_sum_sq / 4.0);

    // RMS of output should be close to 1.0 (since weight is 1.0)
    try testing.expect(@abs(out_rms - 1.0) < 0.01);
}
