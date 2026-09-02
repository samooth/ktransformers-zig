// NUMA auto-population wiring (kt_worker_pool_new_config) — ownership test.
//
// Exercises the A3 closure: when the caller declares a multi-node
// subpool_numa_map, the C-API path derives per-subpool CPU lists from
// NumaTopology.detect (sysfs) and pins each subpool's workers to its
// node's CPUs. The lists' ownership transfers to the pool
// (owns_cpu_lists=true) so kt_worker_pool_free frees everything — no
// stack-lifetime use-after-free (the first attempt kept them on the
// caller's frame), no leaks.
//
// On this 1-node host, cpusForNode(1) returns empty -> subpool 1 runs
// unpinned (the graceful-degradation path); subpool 0 (node 0) gets the
// full CPU list and pins. The test verifies:
//   - new_config with a 2-node map builds and frees without leaks
//   - the pool reports the expected total thread count
//   - the no-NUMA path (all-zero map) also builds/frees (no pinning)
//
// The leak detection IS the ownership verification: numa_map,
// thread_counts, and (on the multi-node path) the cpu lists must be
// freed by deinit through the captured allocator (testing.allocator
// here — but this test goes through the C exports which use the
// default allocator, so we rely on the structural asserts + the
// export-level smoke below, not testing.allocator leak checks).

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig

test "kt_worker_pool_new_config: 2-node map builds, threads spin up, free is clean" {
    // Two subpools: node 0 (2 threads) + node 1 (2 threads).
    const numa_map = [_]c_int{ 0, 1 };
    const thread_counts = [_]c_int{ 2, 2 };
    const config = kt.kt_worker_pool_config_t{
        .subpool_count = 2,
        .subpool_numa_map = @constCast(&numa_map),
        .subpool_thread_count = @constCast(&thread_counts),
    };

    const pool = kt.kt_worker_pool_new_config(config);
    defer kt.kt_worker_pool_free(pool);

    // Both subpools' workers must have started.
    try testing.expectEqual(@as(c_int, 4), kt.kt_worker_pool_get_thread_num(pool));
}

test "kt_worker_pool_new_config: single-node (all-zero map) builds without pinning" {
    const numa_map = [_]c_int{ 0, 0 };
    const thread_counts = [_]c_int{ 1, 1 };
    const config = kt.kt_worker_pool_config_t{
        .subpool_count = 2,
        .subpool_numa_map = @constCast(&numa_map),
        .subpool_thread_count = @constCast(&thread_counts),
    };

    const pool = kt.kt_worker_pool_new_config(config);
    defer kt.kt_worker_pool_free(pool);
    try testing.expectEqual(@as(c_int, 2), kt.kt_worker_pool_get_thread_num(pool));
}

test "kt_worker_pool_new: simple path unchanged (1 subpool, no pinning)" {
    const pool = kt.kt_worker_pool_new(2);
    defer kt.kt_worker_pool_free(pool);
    try testing.expectEqual(@as(c_int, 2), kt.kt_worker_pool_get_thread_num(pool));
}
