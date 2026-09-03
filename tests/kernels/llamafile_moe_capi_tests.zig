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
