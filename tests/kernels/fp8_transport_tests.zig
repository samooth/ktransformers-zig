// Standalone test for the kt_fp8_* Layerwise Transport C API (TODO #4).
//
// The transport is callback-only plumbing (CPU port): run_producer derives
// per-expert w13/w2 weight+scale pointers from all_rank_host_ptrs using the
// reference layout ([slot][rank][kind], slot = expert % 2) and hands them to
// the caller's callback. This test verifies:
//   - init/new/freeproperty accessors
//   - run_producer fires the callback once per expert with the exact
//     reference pointer math (fp8_layerwise_transport.cpp:409-417)
//   - wait() returns the recorded stats (rank, epoch, layer, expert_count,
//     bytes = expert_count * sum(expert_nbytes), CPU-measured timings)
//   - the epoch-mismatch error channel (poisoned stats)
//   - free() releases the transport

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig (wired by build.zig)

const FP8_CONTROL_BYTES: usize = 8192;
const TP_SIZE: usize = 2;
const TP_SIZE_C: c_int = TP_SIZE;
const NUM_EXPERTS: c_int = 4;
const MAX_EXPERTS_PER_RUN: usize = 8; // upper bound for capture arrays

// Capture state (global, since callconv(.c) callbacks cannot capture).
// Reset to 0 at the start of every test that runs the producer.
var g_calls: usize = 0;
var g_expert_ids: [MAX_EXPERTS_PER_RUN]c_int = undefined;
var g_w13_weight: [MAX_EXPERTS_PER_RUN][TP_SIZE]usize = undefined;
var g_w13_scale: [MAX_EXPERTS_PER_RUN][TP_SIZE]usize = undefined;
var g_w2_weight: [MAX_EXPERTS_PER_RUN][TP_SIZE]usize = undefined;
var g_w2_scale: [MAX_EXPERTS_PER_RUN][TP_SIZE]usize = undefined;

fn captureCallback(
    expert_id: c_int,
    w13_weight_ptrs: [*]const usize,
    w13_scale_ptrs: [*]const usize,
    w2_weight_ptrs: [*]const usize,
    w2_scale_ptrs: [*]const usize,
) callconv(.c) void {
    const i = g_calls;
    g_expert_ids[i] = expert_id;
    for (0..TP_SIZE) |r| {
        g_w13_weight[i][r] = w13_weight_ptrs[r];
        g_w13_scale[i][r] = w13_scale_ptrs[r];
        g_w2_weight[i][r] = w2_weight_ptrs[r];
        g_w2_scale[i][r] = w2_scale_ptrs[r];
    }
    g_calls += 1;
}

fn resetCapture() void {
    g_calls = 0;
    for (0..MAX_EXPERTS_PER_RUN) |i| {
        g_expert_ids[i] = -1;
        for (0..TP_SIZE) |r| {
            g_w13_weight[i][r] = 0;
            g_w13_scale[i][r] = 0;
            g_w2_weight[i][r] = 0;
            g_w2_scale[i][r] = 0;
        }
    }
}

test "fp8 transport: init/new/run_producer/wait/free" {
    resetCapture();

    // 64-byte aligned 8192-byte control region. Stack-allocated (8 KiB is
    // comfortably within the default stack limit).
    var control: [FP8_CONTROL_BYTES]u8 align(64) = undefined;
    kt.kt_fp8_layerwise_init(&control, FP8_CONTROL_BYTES, TP_SIZE_C);

    // 16 distinct host buffer addresses: [slot][rank][kind] = 2*2*4.
    var bufs: [16][256]u8 align(64) = undefined;
    var all_rank: [16]usize = undefined;
    for (0..16) |i| all_rank[i] = @intFromPtr(&bufs[i]);

    // local_host_ptrs = rank-zero [slot][kind] slice. With tp_size=2, the
    // local rank is rank 0, so its 8 entries occupy the kind index 0 slot
    // in each pair: (slot * TP_SIZE + 0) * 4 + kind.
    var local_host: [8]usize = undefined;
    for (0..FP8_HOST_SLOTS_USIZE) |s| for (0..FP8_BUFFER_KINDS_USIZE) |k| {
        local_host[s * FP8_BUFFER_KINDS_USIZE + k] = all_rank[(s * TP_SIZE) * FP8_BUFFER_KINDS_USIZE + k];
    };

    const expert_nbytes = [4]usize{ 128, 64, 128, 64 }; // w13_weight, w13_scale, w2_weight, w2_scale

    const transport = kt.kt_fp8_transport_new(
        &control, FP8_CONTROL_BYTES,
        0, TP_SIZE_C, // rank, tp_size
        0, // cuda_device — ignored (CPU port)
        &local_host,
        null, // local_gpu_ptrs — ignored (CPU port)
        &all_rank,
        &expert_nbytes,
        NUM_EXPERTS,
        60000, // timeout_ms — ignored (in-process)
    );

    // Property accessors
    try testing.expectEqual(@as(c_int, 0), kt.kt_fp8_transport_rank(transport));
    try testing.expectEqual(@as(c_int, TP_SIZE_C), kt.kt_fp8_transport_tp_size(transport));
    try testing.expectEqual(NUM_EXPERTS, kt.kt_fp8_transport_num_experts(transport));
    try testing.expectEqual(@as(c_int, 0), kt.kt_fp8_transport_closed(transport));

    // Producer: two experts. expert 0 -> slot 0, expert 1 -> slot 1.
    kt.kt_fp8_transport_run_producer(transport, 1, 7, 2, captureCallback);
    try testing.expectEqual(@as(usize, 2), g_calls);
    try testing.expectEqual(@as(c_int, 0), g_expert_ids[0]);
    try testing.expectEqual(@as(c_int, 1), g_expert_ids[1]);

    // Exact reference pointer math: all_rank_host_ptrs[(slot*tp + rank)*4 + kind]
    for (0..TP_SIZE) |r| {
        // expert 0 -> slot 0
        try testing.expectEqual(all_rank[(0 * TP_SIZE + r) * 4 + 0], g_w13_weight[0][r]);
        try testing.expectEqual(all_rank[(0 * TP_SIZE + r) * 4 + 1], g_w13_scale[0][r]);
        try testing.expectEqual(all_rank[(0 * TP_SIZE + r) * 4 + 2], g_w2_weight[0][r]);
        try testing.expectEqual(all_rank[(0 * TP_SIZE + r) * 4 + 3], g_w2_scale[0][r]);
        // expert 1 -> slot 1
        try testing.expectEqual(all_rank[(1 * TP_SIZE + r) * 4 + 0], g_w13_weight[1][r]);
        try testing.expectEqual(all_rank[(1 * TP_SIZE + r) * 4 + 1], g_w13_scale[1][r]);
        try testing.expectEqual(all_rank[(1 * TP_SIZE + r) * 4 + 2], g_w2_weight[1][r]);
        try testing.expectEqual(all_rank[(1 * TP_SIZE + r) * 4 + 3], g_w2_scale[1][r]);
    }
    // Pointers are non-null (the bufs are real stack arrays).
    for (0..2) |call| {
        for (0..TP_SIZE) |r| {
            try testing.expect(g_w13_weight[call][r] != 0);
            try testing.expect(g_w13_scale[call][r] != 0);
            try testing.expect(g_w2_weight[call][r] != 0);
            try testing.expect(g_w2_scale[call][r] != 0);
        }
    }
    // Slot alternation: expert 0 and expert 1 received different pointers.
    try testing.expect(g_w13_weight[0][0] != g_w13_weight[1][0]);

    // Wait: success stats
    const stats = kt.kt_fp8_transport_wait(transport, 1);
    try testing.expectEqual(@as(u64, 1), stats.epoch);
    try testing.expectEqual(@as(c_int, 7), stats.layer_id);
    try testing.expectEqual(@as(c_int, 2), stats.expert_count);
    try testing.expectEqual(@as(c_int, 0), stats.rank);
    // bytes = expert_count * sum(expert_nbytes) = 2 * 384 = 768
    try testing.expectEqual(@as(usize, 768), stats.bytes);
    try testing.expect(stats.h2d_ms == 0.0);
    try testing.expect(stats.slot_wait_ms == 0.0);
    try testing.expect(stats.writer_ms >= 0.0);
    try testing.expect(stats.total_ms >= 0.0);
    try testing.expectEqual(@as(c_int, 0), stats.poisoned);
    try testing.expectEqual(@as(c_int, 0), stats.error_code);
    try testing.expectEqual(@as(c_int, -1), stats.error_rank);
    try testing.expectEqual(@as(u8, 0), stats.error_message[0]);

    // Epoch-mismatch error path: wait for a different epoch returns poisoned
    // stats with the protocol error code.
    const err = kt.kt_fp8_transport_wait(transport, 999);
    try testing.expectEqual(@as(c_int, 1), err.poisoned);
    try testing.expectEqual(FP8_ERROR_PROTOCOL_TEST, err.error_code);
    try testing.expectEqual(@as(c_int, 0), err.error_rank);
    try testing.expectEqual(@as(u64, 999), err.epoch);
    try testing.expectEqual(@as(c_int, 0), err.expert_count);
    try testing.expectEqual(@as(usize, 0), err.bytes);

    // close + free
    kt.kt_fp8_transport_close(transport);
    try testing.expectEqual(@as(c_int, 1), kt.kt_fp8_transport_closed(transport));
    kt.kt_fp8_transport_free(transport);

    // free(NULL) is a no-op (matches the C free idiom)
    kt.kt_fp8_transport_free(null);
}

// Sized constants for the host-buffer layout. Mirrors the values in main.zig
// (FP8_HOST_SLOTS = 2, FP8_BUFFER_KINDS = 4).
const FP8_HOST_SLOTS_USIZE: usize = 2;
const FP8_BUFFER_KINDS_USIZE: usize = 4;
const FP8_ERROR_PROTOCOL_TEST: c_int = 1;

test "fp8 transport: getters on null handle return sentinels" {
    // free(null) is already exercised above; this confirms the getter
    // behavior matches the documented contract (-1/0/0/1).
    try testing.expectEqual(@as(c_int, -1), kt.kt_fp8_transport_rank(null));
    try testing.expectEqual(@as(c_int, 0), kt.kt_fp8_transport_tp_size(null));
    try testing.expectEqual(@as(c_int, 0), kt.kt_fp8_transport_num_experts(null));
    try testing.expectEqual(@as(c_int, 1), kt.kt_fp8_transport_closed(null));
    // close(null) is a no-op
    kt.kt_fp8_transport_close(null);
    // wait(null) panics with a clear message — exercised via the
    // null-transport check on every handle operation; we don't catch it
    // here (no `std.testing.expectPanics` in 0.16 stable).
}
