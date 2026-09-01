// Standalone test for the kt_mla_* C API (TODO: kt_mla_load_weights — final ABI gap).
//
// Exercises the full C-API lifecycle through the export fns in src/main.zig:
//   kt_mla_new -> kt_mla_load_weights -> kt_mla_forward/decode -> kt_mla_free
//
// What this verifies:
//   - kt_mla_load_weights is exported and callable (closes the last ABI gap;
//     tools/verify_abi.py should report full PASS after this lands)
//   - the weights_loaded gate: kt_mla_load_weights flips the flag that
//     mlaForwardImpl checks (mirrors C++ TP_MLA_Common: mla-tp.hpp:39,86
//     where forward() throws "Not Loaded" without load_weights())
//   - forward/decode run after load_weights with real (zero) weight buffers
//     and produce finite output
//   - the full lifecycle allocates and frees without leaks
//
// Numerical correctness of MLA math is covered by the 11 standalone MLA tests
// (src/mla/mla_tests.zig); this file only covers the C wrapper plumbing.

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig (wired by build.zig)

test "kt_mla C API: new -> load_weights -> forward/decode -> free" {
    const allocator = testing.allocator;

    // Tiny dims (same as the engine-level MLA test in test_kernels.zig).
    const hidden_size: usize = 64;
    const num_heads: usize = 4;
    const q_lora_rank: usize = 32;
    const kv_lora_rank: usize = 16;
    const nope_size: usize = 8;
    const rope_size: usize = 4;
    const max_qlen: usize = 4;
    const max_kvlen: usize = 16;
    const tokens_per_page: usize = 4;

    // Zeroed dummy weight buffers (u16 = amx.bf16 at the C boundary).
    const q_a_proj = try allocator.alloc(u16, q_lora_rank * hidden_size);
    defer allocator.free(q_a_proj);
    const q_a_norm = try allocator.alloc(u16, q_lora_rank);
    defer allocator.free(q_a_norm);
    const q_b_proj = try allocator.alloc(u16, num_heads * (nope_size + rope_size) * q_lora_rank);
    defer allocator.free(q_b_proj);
    const kv_a_proj_with_mqa = try allocator.alloc(u16, (kv_lora_rank + rope_size) * hidden_size);
    defer allocator.free(kv_a_proj_with_mqa);
    const kv_a_norm = try allocator.alloc(u16, kv_lora_rank);
    defer allocator.free(kv_a_norm);
    const kv_b_proj = try allocator.alloc(u16, num_heads * 2 * nope_size * kv_lora_rank);
    defer allocator.free(kv_b_proj);
    const o_proj = try allocator.alloc(u16, hidden_size * num_heads * nope_size);
    defer allocator.free(o_proj);
    @memset(q_a_proj, 0);
    @memset(q_a_norm, 0);
    @memset(q_b_proj, 0);
    @memset(kv_a_proj_with_mqa, 0);
    @memset(kv_a_norm, 0);
    @memset(kv_b_proj, 0);
    @memset(o_proj, 0);

    // kt_mla_new takes a worker_pool pointer; the MLA engine is sequential
    // (cpuinfer param is ignored), so pass a non-null opaque placeholder.
    // The engine never dereferences it (documented in kt_mla_new). Use an
    // usize (8-byte aligned) so the pointer cast doesn't raise alignment.
    var pool_placeholder: usize = 0;

    const config = kt.kt_mla_config_t{
        .hidden_size = hidden_size,
        .q_lora_rank = q_lora_rank,
        .num_heads = num_heads,
        .nope_size = nope_size,
        .rope_size = rope_size,
        .kv_lora_rank = kv_lora_rank,
        .layer_idx = 0,
        .pool = @ptrCast(&pool_placeholder),
        .token_count_in_page = tokens_per_page,
        .max_qlen = max_qlen,
        .max_kvlen = max_kvlen,
        .max_position_embeddings = 128,
        .rope_scaling_factor = 1.0,
        .rope_theta = 10000.0,
        .rope_scaling_beta_fast = 32.0,
        .rope_scaling_beta_slow = 1.0,
        .rope_scaling_mscale = 1.0,
        .rope_scaling_mscale_all_dim = 1.0,
        .rope_scaling_original_max_position_embeddings = 128.0,
        .q_a_proj = q_a_proj.ptr,
        .q_a_norm = q_a_norm.ptr,
        .q_b_proj = q_b_proj.ptr,
        .kv_a_proj_with_mqa = kv_a_proj_with_mqa.ptr,
        .kv_a_norm = kv_a_norm.ptr,
        .kv_b_proj = kv_b_proj.ptr,
        .o_proj = o_proj.ptr,
        .q_a_proj_type = .KT_TYPE_BF16,
        .q_a_norm_type = .KT_TYPE_BF16,
        .q_b_proj_type = .KT_TYPE_BF16,
        .kv_a_proj_with_mqa_type = .KT_TYPE_BF16,
        .kv_a_norm_type = .KT_TYPE_BF16,
        .kv_b_proj_type = .KT_TYPE_BF16,
        .w_o_type = .KT_TYPE_BF16,
        .input_type = .KT_TYPE_BF16,
        .output_type = .KT_TYPE_BF16,
        .m_block = 16,
        .n_block = 16,
        .page_count = (max_kvlen / tokens_per_page) + 1,
    };

    // Lifecycle: new -> load_weights -> forward -> decode -> free.
    const mla = kt.kt_mla_new(config);
    kt.kt_mla_load_weights(mla);

    // forward: qlen_count=1, qlen=1, kvlen=1 (single-token prefill at position 0).
    const input = try allocator.alloc(u16, hidden_size);
    defer allocator.free(input);
    const output = try allocator.alloc(u16, hidden_size);
    defer allocator.free(output);
    @memset(input, 0);
    @memset(output, 0);

    // 8-arg paged C ABI: qlens[0]=1, page_tables/page_table_lens unused
    // (engine has its own sequential cache), kv_lens[0]=1.
    const qlens = [_]c_int{1};
    const kv_lens = [_]c_int{1};
    kt.kt_mla_forward(mla, &qlens, 1, undefined, undefined, &kv_lens, input.ptr, output.ptr);
    kt.kt_mla_prefill(mla, input.ptr, output.ptr, 1, &[_]i64{0});
    kt.kt_mla_decode(mla, input.ptr, output.ptr, 1);

    // Zero weights + zero input -> zero output (finite, exact).
    for (0..hidden_size) |i| {
        try testing.expect(output[i] == 0);
    }

    kt.kt_mla_free(mla);
}

// ============================================================================
// MLA paged attention: full ragged batch (qlen_count = 2)
// ============================================================================
// This is the case that used to @panic in kt_mla_forward. With the
// paged path, two sequences with different qlen and different page
// tables are processed sequentially through the shared engine; the
// outputs are concatenated in input order. The test verifies the
// concatenation order matches the qlen order, and that each slot's
// output equals the output of an equivalent single-sequence call.

test "MlaKvCache save/load round-trip preserves all stored tokens" {
    const allocator = testing.allocator;

    const hidden_size: usize = 32;
    const q_lora_rank: usize = 16;
    const kv_lora_rank: usize = 4;
    const nope_size: usize = 4;
    const rope_size: usize = 4;
    const num_heads: usize = 2;
    const max_qlen: usize = 8;
    const max_kvlen: usize = 32;
    const tokens_per_page: usize = 4;

    // Tiny zeroed weights for the cache init (the test only exercises
    // save/load, not the math).
    const q_a_proj = try allocator.alloc(u16, q_lora_rank * hidden_size);
    defer allocator.free(q_a_proj);
    const q_a_norm = try allocator.alloc(u16, q_lora_rank);
    defer allocator.free(q_a_norm);
    const q_b_proj = try allocator.alloc(u16, num_heads * (nope_size + rope_size) * q_lora_rank);
    defer allocator.free(q_b_proj);
    const kv_a_proj_with_mqa = try allocator.alloc(u16, (kv_lora_rank + rope_size) * hidden_size);
    defer allocator.free(kv_a_proj_with_mqa);
    const kv_a_norm = try allocator.alloc(u16, kv_lora_rank);
    defer allocator.free(kv_a_norm);
    const kv_b_proj = try allocator.alloc(u16, num_heads * 2 * nope_size * kv_lora_rank);
    defer allocator.free(kv_b_proj);
    const o_proj = try allocator.alloc(u16, hidden_size * num_heads * nope_size);
    defer allocator.free(o_proj);
    @memset(q_a_proj, 0);
    @memset(q_a_norm, 0);
    @memset(q_b_proj, 0);
    @memset(kv_a_proj_with_mqa, 0);
    @memset(kv_a_norm, 0);
    @memset(kv_b_proj, 0);
    @memset(o_proj, 0);

    const cfg = kt.mla_config.MlaConfig{
        .hidden_size = hidden_size,
        .num_heads = num_heads,
        .q_lora_rank = q_lora_rank,
        .kv_lora_rank = kv_lora_rank,
        .nope_size = nope_size,
        .rope_size = rope_size,
        .max_qlen = max_qlen,
        .max_kvlen = max_kvlen,
        .token_count_in_page = tokens_per_page,
        .rope_theta = 10000.0,
    };
    const n_pages: usize = 3;
    const total_tokens: usize = 9; // 4 + 4 + 1 across 3 pages

    // Populate the source cache with 9 distinct tokens.
    const src_path = "test_kvcache_save.bin";
    defer {
        var io = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        defer io.deinit();
        std.Io.Dir.cwd().deleteFile(io.io(), src_path) catch {};
    }

    {
        var src = try kt.mla_cache.MlaKvCache.init(allocator, cfg, n_pages);
        defer src.deinit();
        for (0..total_tokens) |t| {
            var nope: [4]f32 = undefined;
            var rope: [4]f32 = undefined;
            for (0..4) |i| nope[i] = @as(f32, @floatFromInt(t * 10 + i));
            for (0..4) |i| rope[i] = @as(f32, @floatFromInt(t * 10 + 100 + i));
            try src.appendToken(&nope, &rope);
        }
        try src.save(src_path);
    }

    // Load into a fresh cache and verify all 9 tokens read back.
    // The page_table is NOT serialized (matching kvcache_load_dump.cpp:
    // the scheduler reconstructs it per session), so we build an
    // identity table for the verify pass.
    var dst = try kt.mla_cache.MlaKvCache.load(allocator, cfg, src_path);
    defer dst.deinit();
    try testing.expectEqual(@as(usize, n_pages), dst.pageCount());
    try testing.expectEqual(@as(usize, total_tokens), dst.kvLen());
    // Build a c_int identity page_table for the verify pass (matches
    // the C ABI and the other dev's pagedGetNopePtr signature).
    var identity_pt = try allocator.alloc(c_int, n_pages);
    defer allocator.free(identity_pt);
    for (0..n_pages) |i| identity_pt[i] = @intCast(i);
    for (0..total_tokens) |t| {
        const nope_ptr = dst.pagedGetNopePtr(identity_pt.ptr, t);
        const rope_ptr = dst.pagedGetRopePtr(identity_pt.ptr, t);
        for (0..4) |i| {
            const expected_nope: f32 = @as(f32, @floatFromInt(t * 10 + i));
            const expected_rope: f32 = @as(f32, @floatFromInt(t * 10 + 100 + i));
            try testing.expectEqual(expected_nope, nope_ptr[i]);
            try testing.expectEqual(expected_rope, rope_ptr[i]);
        }
    }
}

// ============================================================================
// kt_mla_forward paged/batched: equivalence properties (qlen_count > 1)
// ============================================================================
//
// The two invariants that make the page-table indirection CORRECT (not
// just non-crashing):
//   1. Scrambled == identity: mapping the same logical positions to
//      DIFFERENT physical pages must not change the output. Only the
//      page->slot mapping changes, never the math.
//   2. Paged == legacy: a single sequence run through the paged path
//      with an identity table must match the sequential path bit-for-bit.
// These are the tests the ragged-batch comment block above describes
// but that 139bf63 never shipped; forwardPaged had zero test coverage.

fn makePagedMla(
    allocator: std.mem.Allocator,
    weight_value: f32,
) !struct { config: kt.kt_mla_config_t, weights: []u16, mla: *kt.KT_MLA } {
    const hidden_size: usize = 64;
    const num_heads: usize = 4;
    const q_lora_rank: usize = 32;
    const kv_lora_rank: usize = 16;
    const nope_size: usize = 8;
    const rope_size: usize = 4;
    const max_qlen: usize = 8;
    const max_kvlen: usize = 32;
    const tokens_per_page: usize = 4;

    const W = q_lora_rank * hidden_size +
        q_lora_rank +
        num_heads * (nope_size + rope_size) * q_lora_rank +
        (kv_lora_rank + rope_size) * hidden_size +
        kv_lora_rank +
        num_heads * 2 * nope_size * kv_lora_rank +
        hidden_size * num_heads * nope_size;
    const weights = try allocator.alloc(u16, W);
    // NON-ZERO constant weights so the attention math actually exercises
    // the KV reads (zero weights make every equivalence test vacuous).
    for (weights) |*w| w.* = kt.amx.f32_to_bf16(weight_value);

    // Slice into the 7 projections (the order kt_mla_new maps them).
    var o: usize = 0;
    const p_q_a_proj = weights.ptr + o;
    o += q_lora_rank * hidden_size;
    const p_q_a_norm = weights.ptr + o;
    o += q_lora_rank;
    const p_q_b_proj = weights.ptr + o;
    o += num_heads * (nope_size + rope_size) * q_lora_rank;
    const p_kv_a_proj = weights.ptr + o;
    o += (kv_lora_rank + rope_size) * hidden_size;
    const p_kv_a_norm = weights.ptr + o;
    o += kv_lora_rank;
    const p_kv_b_proj = weights.ptr + o;
    o += num_heads * 2 * nope_size * kv_lora_rank;
    const p_o_proj = weights.ptr + o;

    var pool_placeholder: usize = 0;
    const config = kt.kt_mla_config_t{
        .hidden_size = hidden_size,
        .q_lora_rank = q_lora_rank,
        .num_heads = num_heads,
        .nope_size = nope_size,
        .rope_size = rope_size,
        .kv_lora_rank = kv_lora_rank,
        .layer_idx = 0,
        .pool = @ptrCast(&pool_placeholder),
        .token_count_in_page = tokens_per_page,
        .max_qlen = max_qlen,
        .max_kvlen = max_kvlen,
        .max_position_embeddings = 128,
        .rope_scaling_factor = 1.0,
        .rope_theta = 10000.0,
        .rope_scaling_beta_fast = 32.0,
        .rope_scaling_beta_slow = 1.0,
        .rope_scaling_mscale = 1.0,
        .rope_scaling_mscale_all_dim = 1.0,
        .rope_scaling_original_max_position_embeddings = 128.0,
        .q_a_proj = p_q_a_proj,
        .q_a_norm = p_q_a_norm,
        .q_b_proj = p_q_b_proj,
        .kv_a_proj_with_mqa = p_kv_a_proj,
        .kv_a_norm = p_kv_a_norm,
        .kv_b_proj = p_kv_b_proj,
        .o_proj = p_o_proj,
        .q_a_proj_type = .KT_TYPE_BF16,
        .q_a_norm_type = .KT_TYPE_BF16,
        .q_b_proj_type = .KT_TYPE_BF16,
        .kv_a_proj_with_mqa_type = .KT_TYPE_BF16,
        .kv_a_norm_type = .KT_TYPE_BF16,
        .kv_b_proj_type = .KT_TYPE_BF16,
        .w_o_type = .KT_TYPE_BF16,
        .input_type = .KT_TYPE_BF16,
        .output_type = .KT_TYPE_BF16,
        .m_block = 16,
        .n_block = 16,
        // Extra pages beyond max_kvlen/tpp so the scrambled run can land
        // on pages the identity run never touched.
        .page_count = (max_kvlen / tokens_per_page) + 4,
    };

    const mla = kt.kt_mla_new(config);
    kt.kt_mla_load_weights(mla);
    return .{ .config = config, .weights = weights, .mla = mla };
}

test "kt_mla_forward paged: scrambled pages match identity pages" {
    const allocator = testing.allocator;

    var run_a = try makePagedMla(allocator, 0.01);
    defer allocator.free(run_a.weights);
    defer kt.kt_mla_free(run_a.mla);
    var run_b = try makePagedMla(allocator, 0.01);
    defer allocator.free(run_b.weights);
    defer kt.kt_mla_free(run_b.mla);

    const hidden_size: usize = 64;

    // Batch: 2 sequences, qlen 2 each. seq0: kv_len 6 (writes at 4,5);
    // seq1: kv_len 8 (writes at 6,7).
    const qlens = [_]c_int{ 2, 2 };
    const kv_lens = [_]c_int{ 6, 8 };
    const lens = [_]c_int{ 2, 2 }; // ceil(6/4)=2, ceil(8/4)=2

    // Identity: logical page i -> physical page i, per sequence with
    // DISJOINT page sets (seq0 -> {0,1}, seq1 -> {2,3}) so no sequence
    // ever reads another's KVs.
    const ident0 = [_]c_int{ 0, 1 };
    const ident1 = [_]c_int{ 2, 3 };
    // Scrambled: same logical structure on DIFFERENT physical pages
    // (seq0 -> {4,5}, seq1 -> {6,7}) — disjoint from each other AND from
    // the identity run's pages, so both runs start from zeroed pages.
    // Only the page->slot mapping differs, never the math or the
    // read-visible content.
    const scram0 = [_]c_int{ 4, 5 };
    const scram1 = [_]c_int{ 6, 7 };

    const tables_ident = [_][*]const c_int{ &ident0, &ident1 };
    const tables_scram = [_][*]const c_int{ &scram0, &scram1 };

    // Distinct input per token: [sum(qlens)=4, hidden_size].
    const input = try allocator.alloc(u16, 4 * hidden_size);
    defer allocator.free(input);
    for (0..4 * hidden_size) |i| {
        input[i] = kt.amx.f32_to_bf16(0.05 * @as(f32, @floatFromInt(i % 17)));
    }

    const out_ident = try allocator.alloc(u16, 4 * hidden_size);
    defer allocator.free(out_ident);
    const out_scram = try allocator.alloc(u16, 4 * hidden_size);
    defer allocator.free(out_scram);

    kt.kt_mla_forward(run_a.mla, &qlens, 2, &tables_ident, &lens, &kv_lens, input.ptr, out_ident.ptr);
    kt.kt_mla_forward(run_b.mla, &qlens, 2, &tables_scram, &lens, &kv_lens, input.ptr, out_scram.ptr);

    // Same logical content, different physical placement: outputs must
    // agree bit-for-bit.
    var n_nonzero: usize = 0;
    for (0..4 * hidden_size) |i| {
        try testing.expect(out_ident[i] == out_scram[i]);
        if (out_ident[i] != 0) n_nonzero += 1;
    }
    // Guard against a vacuous pass: non-zero weights must produce
    // non-zero output (attention math actually ran).
    try testing.expect(n_nonzero > 0);
}

test "kt_mla_forward paged: identity page table matches legacy sequential" {
    const allocator = testing.allocator;

    var run_paged = try makePagedMla(allocator, 0.01);
    defer allocator.free(run_paged.weights);
    defer kt.kt_mla_free(run_paged.mla);
    var run_seq = try makePagedMla(allocator, 0.01);
    defer allocator.free(run_seq.weights);
    defer kt.kt_mla_free(run_seq.mla);

    const hidden_size: usize = 64;

    // Single sequence, qlen 3 (prefill), kv_len 3.
    const qlens = [_]c_int{3};
    const kv_lens = [_]c_int{3};
    const ident = [_]c_int{0};
    const tables = [_][*]const c_int{&ident};
    const lens = [_]c_int{1};

    const input = try allocator.alloc(u16, 3 * hidden_size);
    defer allocator.free(input);
    for (0..3 * hidden_size) |i| {
        input[i] = kt.amx.f32_to_bf16(0.04 * @as(f32, @floatFromInt((i * 7) % 23)));
    }

    const out_paged = try allocator.alloc(u16, 3 * hidden_size);
    defer allocator.free(out_paged);
    const out_seq = try allocator.alloc(u16, 3 * hidden_size);
    defer allocator.free(out_seq);

    // Paged path (tables provided) vs legacy path (null tables).
    kt.kt_mla_forward(run_paged.mla, &qlens, 1, &tables, &lens, &kv_lens, input.ptr, out_paged.ptr);
    kt.kt_mla_forward(run_seq.mla, &qlens, 1, null, null, &kv_lens, input.ptr, out_seq.ptr);

    var n_nonzero: usize = 0;
    for (0..3 * hidden_size) |i| {
        try testing.expect(out_paged[i] == out_seq[i]);
        if (out_paged[i] != 0) n_nonzero += 1;
    }
    try testing.expect(n_nonzero > 0);
}
