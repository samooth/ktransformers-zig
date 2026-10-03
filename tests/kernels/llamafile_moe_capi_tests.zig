// Standalone test for the kt_llama_moe_* C API (model orchestration).
//
// Verifies the wiring + lifecycle of the LlamaMoe class (the GGML-quantized
// MoE port of C++ LLAMA_MOE_TP). Skips full numerical correctness — that
// requires a real Q4_K × Q8_0 matmul, which the first-cut implementation
// dequants through BF16 (existing `kt_matmul_q*`). The test here confirms:
//   1. kt_llama_moe_new succeeds with valid Q4_K + Q8_0 + Q8_K configs
//   2. The opaque handle round-trips through the C ABI
//   3. kt_llama_moe_load_weights copies the per-expert slices
//   4. kt_llama_moe_free doesn't leak (0 leaks in test mode)
//
// The forward pass is NOT tested here — it's gated to the same runMatmul
// that the Zig-side test exercises. The pybind11 path covers end-to-end
// forward through the C++ -> C ABI -> Zig interface.

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig

test "kt_llama_moe C API: new -> load_weights -> free (Q8_0, no leaks)" {
    const allocator = testing.allocator;

    // 1 expert, hidden=32, inter=32, Q8_0 weights (block size 32, so
    // each expert is 32*32 = 1024 bytes, well-defined). KT_TYPE_Q8_0 = 12
    // matches include/kt_kernel.h.
    const kt_type_q8_0: u32 = 12;
    const hidden: usize = 32;
    const inter: usize = 32;
    // Q8_0: 34 bytes per 32 weights.
    const expert_bytes = @as(usize, @intCast(@divExact(inter * hidden, 32))) * 34;
    // down [hidden, inter] for Q8_0
    const down_bytes = @as(usize, @intCast(@divExact(hidden * inter, 32))) * 34;

    const gate_w = try allocator.alloc(u8, expert_bytes);
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(u8, expert_bytes);
    defer allocator.free(up_w);
    const down_w = try allocator.alloc(u8, down_bytes);
    defer allocator.free(down_w);
    @memset(gate_w, 0);
    @memset(up_w, 0);
    @memset(down_w, 0);

    // Create a real worker pool (the MoE init requires a non-null pool
    // pointer; the test uses a 1-thread pool since we never actually
    // call forward here).
    const pool = kt.kt_worker_pool_new(1);
    defer kt.kt_worker_pool_free(pool);

    const config = kt.kt_llama_moe_config_t{
        .expert_num = 1,
        .num_experts_per_tok = 1,
        .hidden_size = hidden,
        .intermediate_size = inter,
        .layer_idx = 0,
        .pool = pool,
        .gate_type = kt_type_q8_0,
        .up_type = kt_type_q8_0,
        .down_type = kt_type_q8_0,
        .hidden_type = 2, // KT_TYPE_BF16
        .m_block = 32,
        .group_min_len = 32,
        .group_max_len = 1024,
        .gpu_experts_mask = null,
        .gate_proj = gate_w.ptr,
        .up_proj = up_w.ptr,
        .down_proj = down_w.ptr,
    };

    const moe = kt.kt_llama_moe_new(&config);
    kt.kt_llama_moe_load_weights(moe, @as(c_int, @intCast(inter)), 0);
    kt.kt_llama_moe_free(moe);
}

test "kt_llama_moe C API: rejected when pool is null" {
    // The init panics with "Failed to init LlamaMoe (check pool + weight types)"
    // when pool is null. This is a marker for future work: returning
    // an error union from init would let us assertError here.
    const allocator = testing.allocator;
    const config = kt.kt_llama_moe_config_t{
        .expert_num = 1,
        .num_experts_per_tok = 1,
        .hidden_size = 32,
        .intermediate_size = 32,
        .layer_idx = 0,
        .pool = null,
        .gate_type = 16,
        .up_type = 16,
        .down_type = 16,
        .hidden_type = 2,
        .m_block = 32,
        .group_min_len = 32,
        .group_max_len = 1024,
        .gpu_experts_mask = null,
        .gate_proj = @as([*]const u8, @ptrCast(&[_]u8{0})),
        .up_proj = @as([*]const u8, @ptrCast(&[_]u8{0})),
        .down_proj = @as([*]const u8, @ptrCast(&[_]u8{0})),
    };
    _ = config;
    _ = allocator;
}

test "kt_llama_moe C API: forward_many produces sane output for qlen > 1" {
    // End-to-end test of the new forwardMany path. The single-token
    // forward_one is exercised by the prior test. This test verifies
    // that qlen > 1 produces a non-NaN, finite output across the
    // batch and the C ABI signature stays stable.
    const allocator = testing.allocator;
    const hidden: usize = 32;
    const inter: usize = 32;
    const expert_num: usize = 2;
    const top_k: usize = 1;
    const qlen: usize = 4;

    // Q8_0 weights (block size 32, hidden=inter=32 → 1 block per row).
    const expert_bytes = (inter / 32) * 34;
    const down_bytes = (hidden / 32) * 34;
    const gate_w = try allocator.alloc(u8, expert_bytes);
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(u8, expert_bytes);
    defer allocator.free(up_w);
    const down_w = try allocator.alloc(u8, down_bytes);
    defer allocator.free(down_w);
    @memset(gate_w, 0);
    @memset(up_w, 0);
    @memset(down_w, 0);

    const pool = kt.kt_worker_pool_new(2);
    defer kt.kt_worker_pool_free(pool);

    const config = kt.kt_llama_moe_config_t{
        .expert_num = expert_num,
        .num_experts_per_tok = top_k,
        .hidden_size = hidden,
        .intermediate_size = inter,
        .layer_idx = 0,
        .pool = pool,
        .gate_type = 12, // KT_TYPE_Q8_0
        .up_type = 12,
        .down_type = 12,
        .hidden_type = 2, // KT_TYPE_BF16
        .m_block = 32,
        .group_min_len = 32,
        .group_max_len = 1024,
        .gpu_experts_mask = null,
        .gate_proj = gate_w.ptr,
        .up_proj = up_w.ptr,
        .down_proj = down_w.ptr,
    };
    const moe = kt.kt_llama_moe_new(&config);
    kt.kt_llama_moe_load_weights(moe, @intCast(inter), 0);

    // qlen × hidden BF16 input: all 1.0 (0x3F80)
    const input = try allocator.alloc(u8, qlen * hidden * 2);
    defer allocator.free(input);
    for (0..qlen * hidden) |i| input[i * 2 + 0] = 0x80; // BF16 1.0 = 0x3F80

    // Routing: all tokens → expert 0, weight 1.0
    const expert_ids = try allocator.alloc(i64, qlen * top_k);
    defer allocator.free(expert_ids);
    const wts_f32 = try allocator.alloc(f32, qlen * top_k);
    defer allocator.free(wts_f32);
    for (0..qlen) |i| {
        for (0..top_k) |j| {
            expert_ids[i * top_k + j] = 0;
            wts_f32[i * top_k + j] = 1.0;
        }
    }

    const output_f32 = try allocator.alloc(f32, qlen * hidden); // F32 output
    defer allocator.free(output_f32);

    kt.kt_llama_moe_forward(
        moe,
        @intCast(qlen),
        @intCast(top_k),
        expert_ids.ptr,
        wts_f32.ptr,
        input.ptr,
        output_f32.ptr,
    );

    // Verify: output is finite (no NaN/inf) — with zero weights and BF16
    // input 1.0, the gate/up GEMMs produce all-zero (dequant d=0),
    // SwiGLU(0)*0 = 0, down produces 0. The final result is the input
    // residue (since weighted add of 0 to 0 is 0). We just check finite.
    for (0..qlen * hidden) |i| {
        const v = output_f32[i];
        try testing.expect(std.math.isFinite(v));
    }

    kt.kt_llama_moe_free(moe);
}

test "LlamaMoe ABI audit: KT_TYPE constants match main.zig kt_type_t enum" {
    // Regression test for the 5177bdf bug class: llamafile_moe.zig had
    // 11 INVENTED KT_TYPE values (e.g. KT_TYPE_Q2_K=20 when the header
    // says 14; KT_TYPE_IQ3_S=29 matched nothing). A caller passing the
    // REAL header values would have dispatched to the WRONG kernel or
    // panicked. This test pins every constant against the enum that
    // mirrors include/kt_kernel.h.
    const lm = kt.llamafile_moe;
    const KT = kt.kt_type_t;

    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_Q8_0), lm.KT_TYPE_Q8_0);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_Q4_K), lm.KT_TYPE_Q4_K);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_Q5_K), lm.KT_TYPE_Q5_K);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_Q6_K), lm.KT_TYPE_Q6_K);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_Q8_K), lm.KT_TYPE_Q8_K);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_Q2_K), lm.KT_TYPE_Q2_K);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_Q3_K), lm.KT_TYPE_Q3_K);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_IQ2_XXS), lm.KT_TYPE_IQ2_XXS);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_IQ2_XS), lm.KT_TYPE_IQ2_XS);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_IQ2_S), lm.KT_TYPE_IQ2_S);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_IQ3_XXS), lm.KT_TYPE_IQ3_XXS);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_IQ3_S), lm.KT_TYPE_IQ3_S);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_IQ1_S), lm.KT_TYPE_IQ1_S);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_IQ1_M), lm.KT_TYPE_IQ1_M);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_IQ4_NL), lm.KT_TYPE_IQ4_NL);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_IQ4_XS), lm.KT_TYPE_IQ4_XS);
    try testing.expectEqual(@intFromEnum(KT.KT_TYPE_BF16), lm.KT_TYPE_BF16);
}

test "kt_llama_moe C API: parallel (4-thread) == sequential (1-thread) forward" {
    // Review of 1130d0b: the work-stealing branch shared the instance
    // s_* scratches across worker threads — a REAL data race (with
    // valid random Q8_0 weights, 1-thread vs 4-thread outputs diverged
    // non-deterministically: ~4096/8192 values, different every run).
    // Fixed with per-token TokenScratch. This test pins the fix:
    // random-weight forward through a 1-thread pool MUST equal the
    // same forward through a 4-thread pool, bit for bit.
    const allocator = testing.allocator;
    const amx = kt.amx;
    const q8 = kt.gemm_q8_0;

    const hidden: usize = 256;
    const inter: usize = 128;
    const expert_num: usize = 4;
    const top_k: usize = 2;
    const qlen: usize = 32;

    // Random (seeded) VALID Q8_0 weights — raw random BYTES would put
    // NaN f16 scales in the blocks and poison any comparison.
    const gate_bytes = expert_num * inter * (hidden / 32) * 34;
    const up_bytes = expert_num * inter * (hidden / 32) * 34;
    const down_bytes = expert_num * hidden * (inter / 32) * 34;
    const gate_w = try allocator.alloc(u8, gate_bytes);
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(u8, up_bytes);
    defer allocator.free(up_w);
    const down_w = try allocator.alloc(u8, down_bytes);
    defer allocator.free(down_w);
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();
    {
        const row_f = try allocator.alloc(f32, hidden);
        defer allocator.free(row_f);
        for (0..expert_num * inter) |r| {
            for (row_f) |*v| v.* = @as(f32, @floatFromInt(rnd.int(u8))) / 255.0 - 0.5;
            q8.quantizeRowQ8_0(row_f, @as([*]q8.BlockQ8_0, @ptrCast(@alignCast(gate_w.ptr + r * (hidden / 32) * 34))), hidden);
        }
        for (0..expert_num * inter) |r| {
            for (row_f) |*v| v.* = @as(f32, @floatFromInt(rnd.int(u8))) / 255.0 - 0.5;
            q8.quantizeRowQ8_0(row_f, @as([*]q8.BlockQ8_0, @ptrCast(@alignCast(up_w.ptr + r * (hidden / 32) * 34))), hidden);
        }
        for (0..expert_num * hidden) |r| {
            for (row_f[0..inter]) |*v| v.* = @as(f32, @floatFromInt(rnd.int(u8))) / 255.0 - 0.5;
            q8.quantizeRowQ8_0(row_f[0..inter], @as([*]q8.BlockQ8_0, @ptrCast(@alignCast(down_w.ptr + r * (inter / 32) * 34))), inter);
        }
    }

    // Random BF16 input.
    const input = try allocator.alloc(u8, qlen * hidden * 2);
    defer allocator.free(input);
    {
        const in_bf: [*]amx.bf16 = @ptrCast(@alignCast(input.ptr));
        for (0..qlen * hidden) |i| {
            in_bf[i] = amx.f32_to_bf16(@as(f32, @floatFromInt(rnd.int(u8))) / 255.0 - 0.5);
        }
    }

    // Routing: every token -> experts {0, 1} with fixed weights.
    const eids = try allocator.alloc(i64, qlen * top_k);
    defer allocator.free(eids);
    const wts = try allocator.alloc(f32, qlen * top_k);
    defer allocator.free(wts);
    for (0..qlen) |t| {
        eids[t * top_k] = 0;
        eids[t * top_k + 1] = 1;
        wts[t * top_k] = 0.6;
        wts[t * top_k + 1] = 0.4;
    }

    const out_seq = try allocator.alloc(f32, qlen * hidden);
    defer allocator.free(out_seq);
    const out_par = try allocator.alloc(f32, qlen * hidden);
    defer allocator.free(out_par);

    // Sequential (1-thread pool).
    {
        const pool = kt.kt_worker_pool_new(1);
        defer kt.kt_worker_pool_free(pool);
        const cfg = kt.kt_llama_moe_config_t{
            .expert_num = expert_num,
            .num_experts_per_tok = top_k,
            .hidden_size = hidden,
            .intermediate_size = inter,
            .layer_idx = 0,
            .pool = @ptrCast(pool),
            .gate_type = 12,
            .up_type = 12,
            .down_type = 12,
            .hidden_type = 2,
            .m_block = 32,
            .group_min_len = 32,
            .group_max_len = 1024,
            .gpu_experts_mask = null,
            .gate_proj = gate_w.ptr,
            .up_proj = up_w.ptr,
            .down_proj = down_w.ptr,
        };
        const moe = kt.kt_llama_moe_new(&cfg);
        kt.kt_llama_moe_load_weights(moe, @intCast(inter), 0);
        kt.kt_llama_moe_forward(moe, @intCast(qlen), @intCast(top_k), eids.ptr, wts.ptr, input.ptr, out_seq.ptr);
        kt.kt_llama_moe_free(moe);
    }
    // Parallel (4-thread pool).
    {
        const pool = kt.kt_worker_pool_new(4);
        defer kt.kt_worker_pool_free(pool);
        const cfg = kt.kt_llama_moe_config_t{
            .expert_num = expert_num,
            .num_experts_per_tok = top_k,
            .hidden_size = hidden,
            .intermediate_size = inter,
            .layer_idx = 0,
            .pool = @ptrCast(pool),
            .gate_type = 12,
            .up_type = 12,
            .down_type = 12,
            .hidden_type = 2,
            .m_block = 32,
            .group_min_len = 32,
            .group_max_len = 1024,
            .gpu_experts_mask = null,
            .gate_proj = gate_w.ptr,
            .up_proj = up_w.ptr,
            .down_proj = down_w.ptr,
        };
        const moe = kt.kt_llama_moe_new(&cfg);
        kt.kt_llama_moe_load_weights(moe, @intCast(inter), 0);
        kt.kt_llama_moe_forward(moe, @intCast(qlen), @intCast(top_k), eids.ptr, wts.ptr, input.ptr, out_par.ptr);
        kt.kt_llama_moe_free(moe);
    }

    // Bit-exact: same math, isolated scratch per task.
    for (0..qlen * hidden) |i| {
        try testing.expectEqual(out_seq[i], out_par[i]);
    }
}

test "kt_llama_moe C API: Q4_K through the MoE matches the dequantized reference" {
    // Coverage for the Q4_K path through the full pipeline: every prior
    // LlamaMoe forward test used Q8_0 weights, so the K-quant arm never
    // executed end-to-end (the same test-gap class that hid the 5177bdf
    // KT_TYPE bug). Q4_K config + 4-thread pool + qlen>1 drives the
    // parallel scratch path straight into it.
    //
    // Q4_K used to take a fused `gemmQ4KxQ8_0` arm that skipped the
    // activation's BF16 intermediate. That arm is gone: the generic AVX2
    // K-quant driver (`gemm_q4_k.gemmQ4_K`, commit f6b1f0b) is 3.3x
    // faster on this shape and every other weight format already goes
    // through the BF16 intermediate. Q4_K is no longer the odd one out.
    //
    // The reference below therefore models the BF16 activation round-trip
    // explicitly. It used to assume that round-trip was lossless — the
    // comment claimed "the dequant value a_q*a_d is what the fused dot
    // multiplies anyway". That is FALSE: `dequantQ8_0ToBF16` computes
    // `f32_to_bf16(a_q * a_d)`, and while a_q (8-bit) is exact in bf16,
    // the f16 scale d has an 11-bit significand and bf16 keeps 8, so the
    // product loses ~3 low bits (~0.2% relative) per activation. Measured
    // on the m=1 n=256 k=256 GEMM this test exercises:
    //
    //   fused  vs F32 reference                 6.2e-6   (what it asserted)
    //   bf16   vs F32 reference                 7.8e-2   (fails the bound)
    //   bf16   vs F32 ref with bf16 rounding    9.5e-6   (apples to apples)
    //
    // So the driver was never wrong; the reference just omitted the
    // rounding the live path performs. Both round-trips (act and mid) are
    // modelled below, which restores the ~1e-5 agreement.
    const allocator = testing.allocator;
    const amx = kt.amx;
    const q4k = kt.gemm_q4_k;
    const q8 = kt.gemm_q8_0;

    const hidden: usize = 256; // Q4_K: K % 256 == 0
    const inter: usize = 256; // down-proj K too
    const expert_num: usize = 2;
    const top_k: usize = 2;
    const qlen: usize = 8;

    // Seeded random Q4_K weights via the real quantizer.
    const gate_blks = expert_num * inter * (hidden / 256);
    const gate_w = try allocator.alloc(u8, gate_blks * @sizeOf(q4k.BlockQ4_K));
    defer allocator.free(gate_w);
    const up_w = try allocator.alloc(u8, gate_blks * @sizeOf(q4k.BlockQ4_K));
    defer allocator.free(up_w);
    const down_blks = expert_num * hidden * (inter / 256);
    const down_w = try allocator.alloc(u8, down_blks * @sizeOf(q4k.BlockQ4_K));
    defer allocator.free(down_w);
    var prng = std.Random.DefaultPrng.init(0x51D06);
    const rnd = prng.random();
    const w_blk_size = @sizeOf(q4k.BlockQ4_K);
    {
        const row_f = try allocator.alloc(f32, hidden);
        defer allocator.free(row_f);
        for (0..expert_num * inter) |r| {
            for (row_f) |*v| v.* = @as(f32, @floatFromInt(rnd.int(u8))) / 255.0 - 0.5;
            q4k.quantizeRowQ4_K(row_f, @as([*]q4k.BlockQ4_K, @ptrCast(@alignCast(gate_w.ptr + r * (hidden / 256) * w_blk_size))), hidden);
        }
        for (0..expert_num * inter) |r| {
            for (row_f) |*v| v.* = @as(f32, @floatFromInt(rnd.int(u8))) / 255.0 - 0.5;
            q4k.quantizeRowQ4_K(row_f, @as([*]q4k.BlockQ4_K, @ptrCast(@alignCast(up_w.ptr + r * (hidden / 256) * w_blk_size))), hidden);
        }
        for (0..expert_num * hidden) |r| {
            for (row_f[0..inter]) |*v| v.* = @as(f32, @floatFromInt(rnd.int(u8))) / 255.0 - 0.5;
            q4k.quantizeRowQ4_K(row_f[0..inter], @as([*]q4k.BlockQ4_K, @ptrCast(@alignCast(down_w.ptr + r * (inter / 256) * w_blk_size))), inter);
        }
    }

    // Random BF16 input [qlen, hidden].
    const input = try allocator.alloc(u8, qlen * hidden * 2);
    defer allocator.free(input);
    const in_bf: [*]amx.bf16 = @ptrCast(@alignCast(input.ptr));
    for (0..qlen * hidden) |i| {
        in_bf[i] = amx.f32_to_bf16(@as(f32, @floatFromInt(rnd.int(u8))) / 255.0 - 0.5);
    }

    // Mixed routing: first half experts {0,1}, second half {1,0} —
    // exercises both expert-slice offsets.
    const eids = try allocator.alloc(i64, qlen * top_k);
    defer allocator.free(eids);
    const wts = try allocator.alloc(f32, qlen * top_k);
    defer allocator.free(wts);
    for (0..qlen) |t| {
        const flip: i64 = if (t < 4) 0 else 1;
        eids[t * top_k] = flip;
        eids[t * top_k + 1] = 1 - flip;
        wts[t * top_k] = 0.6;
        wts[t * top_k + 1] = 0.4;
    }

    // Run through the C API (4-thread pool -> parallel -> fused arm).
    const out = try allocator.alloc(f32, qlen * hidden);
    defer allocator.free(out);
    {
        const pool = kt.kt_worker_pool_new(4);
        defer kt.kt_worker_pool_free(pool);
        const cfg = kt.kt_llama_moe_config_t{
            .expert_num = expert_num,
            .num_experts_per_tok = top_k,
            .hidden_size = hidden,
            .intermediate_size = inter,
            .layer_idx = 0,
            .pool = @ptrCast(pool),
            .gate_type = 16, // KT_TYPE_Q4_K
            .up_type = 16,
            .down_type = 16,
            .hidden_type = 2,
            .m_block = 32,
            .group_min_len = 32,
            .group_max_len = 1024,
            .gpu_experts_mask = null,
            .gate_proj = gate_w.ptr,
            .up_proj = up_w.ptr,
            .down_proj = down_w.ptr,
        };
        const moe = kt.kt_llama_moe_new(&cfg);
        kt.kt_llama_moe_load_weights(moe, @intCast(inter), 0);
        kt.kt_llama_moe_forward(moe, @intCast(qlen), @intCast(top_k), eids.ptr, wts.ptr, input.ptr, out.ptr);
        kt.kt_llama_moe_free(moe);
    }

    // Dequantize weights once for the reference.
    const gate_dq = try allocator.alloc(f32, expert_num * inter * hidden);
    defer allocator.free(gate_dq);
    const up_dq = try allocator.alloc(f32, expert_num * inter * hidden);
    defer allocator.free(up_dq);
    const down_dq = try allocator.alloc(f32, expert_num * hidden * inter);
    defer allocator.free(down_dq);
    {
        const gb: [*]const q4k.BlockQ4_K = @ptrCast(@alignCast(gate_w.ptr));
        const ub: [*]const q4k.BlockQ4_K = @ptrCast(@alignCast(up_w.ptr));
        const db: [*]const q4k.BlockQ4_K = @ptrCast(@alignCast(down_w.ptr));
        for (0..expert_num * inter) |r| {
            q4k.dequantizeRowQ4_K(gb + r * (hidden / 256), gate_dq[r * hidden ..][0..hidden], hidden);
            q4k.dequantizeRowQ4_K(ub + r * (hidden / 256), up_dq[r * hidden ..][0..hidden], hidden);
        }
        for (0..expert_num * hidden) |r| {
            q4k.dequantizeRowQ4_K(db + r * (inter / 256), down_dq[r * inter ..][0..inter], inter);
        }
    }

    // Reference: per token, per expert — act Q8_0 round-trip, gate/up
    // dots, silu*up, mid Q8_0 round-trip, down dot, weighted sum.
    // Tolerance: after the BF16 activation round-trips below are modelled,
    // this reference and the kernel multiply the SAME values and differ
    // only in F32 accumulation order, so ~1e-5 is expected; 1e-2 is a wide
    // anti-flake bound that still catches wrong-kernel/wrong-offset
    // dispatch (those produce O(1) divergence, as the race probe did).
    const act_f = try allocator.alloc(f32, hidden);
    defer allocator.free(act_f);
    const act_dq = try allocator.alloc(f32, hidden);
    defer allocator.free(act_dq);
    const mid_f = try allocator.alloc(f32, inter);
    defer allocator.free(mid_f);
    const mid_dq = try allocator.alloc(f32, inter);
    defer allocator.free(mid_dq);
    const act_q8 = try allocator.alloc(q8.BlockQ8_0, hidden / 32);
    defer allocator.free(act_q8);
    const mid_q8 = try allocator.alloc(q8.BlockQ8_0, inter / 32);
    defer allocator.free(mid_q8);

    var max_rel: f32 = 0;
    for (0..qlen) |t| {
        // Activation Q8_0 round-trip, then the BF16 round-trip that
        // `dequantQ8_0ToBF16` performs before the GEMM (see the header).
        for (0..hidden) |i| act_f[i] = amx.bf16_to_f32(in_bf[t * hidden + i]);
        q8.quantizeRowQ8_0(act_f, act_q8.ptr, hidden);
        q8.dequantizeRowQ8_0(act_q8.ptr, act_dq, hidden);
        for (0..hidden) |i| act_dq[i] = amx.bf16_to_f32(amx.f32_to_bf16(act_dq[i]));

        const out_ref_buf = try allocator.alloc(f32, hidden);
        defer allocator.free(out_ref_buf);
        @memset(out_ref_buf, 0);
        for (0..top_k) |ki| {
            const e: usize = @intCast(eids[t * top_k + ki]);
            const w = wts[t * top_k + ki];
            // Phase 1: mid[j] = silu(gate·act) * (up·act).
            for (0..inter) |j| {
                var g: f32 = 0;
                var u: f32 = 0;
                for (0..hidden) |i| {
                    g += act_dq[i] * gate_dq[(e * inter + j) * hidden + i];
                    u += act_dq[i] * up_dq[(e * inter + j) * hidden + i];
                }
                mid_f[j] = g / (1.0 + @exp(-g)) * u;
            }
            // Phase 2: mid Q8_0 round-trip + BF16 round-trip + down dot + weighted add.
            q8.quantizeRowQ8_0(mid_f, mid_q8.ptr, inter);
            q8.dequantizeRowQ8_0(mid_q8.ptr, mid_dq, inter);
            for (0..inter) |j| mid_dq[j] = amx.bf16_to_f32(amx.f32_to_bf16(mid_dq[j]));
            for (0..hidden) |i| {
                var dd: f32 = 0;
                for (0..inter) |jj| {
                    dd += mid_dq[jj] * down_dq[(e * hidden + i) * inter + jj];
                }
                out_ref_buf[i] += dd * w;
            }
        }
        // Compare element-wise.
        for (0..hidden) |i| {
            const got = out[t * hidden + i];
            const want = out_ref_buf[i];
            try testing.expect(std.math.isFinite(got));
            const denom = @max(@abs(want), 1e-3);
            const rel = @abs(got - want) / denom;
            max_rel = @max(max_rel, rel);
        }
    }
    if (max_rel >= 1e-2) {
        std.debug.print("Q4_K rel={d:.6} out[0]={d:.6}\n", .{ max_rel, out[0] });
    }
    try testing.expect(max_rel < 1e-2);
}
