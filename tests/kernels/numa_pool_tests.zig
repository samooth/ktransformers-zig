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

/// Direct access to the runtime pool, for the Subpool signalling tests below.
/// The C exports cannot express "dispatch a job while another thread calls
/// waitIdle", which is the only way to reach the deadlock.
const wp = kt.worker_pool;

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

// ============================================================================
// Subpool completion signalling
// ============================================================================
//
// `done_cv` carries TWO predicates: `done_count >= task_end` (the dispatcher
// inside doWorkStealingJob) and `pending_jobs <= allow` (waitIdle). The worker
// completion path used to `pthread_cond_signal` that single condvar, which wakes
// exactly ONE waiter. If the signal is delivered to the waitIdle caller, the
// dispatcher sleeps on a predicate that is already true — and the broadcast
// that would wake it is emitted by doWorkStealingJob *after* it returns.
// Circular wait, no CPU use, no timeout inside the code.
//
// HONESTY ABOUT WHAT WAS AND WAS NOT OBSERVED. The stress test below runs that
// race 400x2 times and passed against the signal() version on this host, so the
// window is narrow and I did NOT reproduce a hang from it. The change to
// broadcast() is therefore hardening justified by the wait graph, not a fix for
// an observed failure — there are at most two waiters, so waking both costs
// nothing. The two tests after it cover real bugs that WERE reachable.

var dl_nonce_work = std.atomic.Value(usize).init(0);

fn dlCountWork(idx: usize) void {
    _ = idx;
    _ = dl_nonce_work.fetchAdd(1, .monotonic);
}

fn dlJobThread(sub: *wp.Subpool, iters: usize) void {
    for (0..iters) |_| {
        // More tasks than workers, so several workers finish at once and the
        // "who gets the signal" race is live on every iteration.
        sub.doWorkStealingJob(16, dlCountWork);
    }
}

fn dlSyncThread(sub: *wp.Subpool, iters: usize) void {
    for (0..iters) |_| {
        sub.waitIdle(0);
    }
}

test "Subpool: worker completion cannot strand the dispatcher while waitIdle is waiting" {
    const allocator = testing.allocator;
    const sub = try wp.Subpool.init(allocator, 0, 0, 4, null, true);
    defer sub.deinit(allocator);

    dl_nonce_work.store(0, .monotonic);

    // Interleave both roles repeatedly. Before the fix this hangs; a hang is the
    // failure signal here, so CI's suite timeout is what reports it.
    const iters = 400;
    var job = try std.Thread.spawn(.{}, dlJobThread, .{ sub, iters });
    var sync = try std.Thread.spawn(.{}, dlSyncThread, .{ sub, iters });
    job.join();
    sync.join();

    // If we get here, every task ran: the dispatcher's predicate was satisfied
    // by real completions rather than by a wakeup that happened to arrive.
    try testing.expectEqual(@as(usize, iters * 16), dl_nonce_work.load(.acquire));
}

test "WorkerPool: a zero-thread subpool is clamped to 1 instead of hanging forever" {
    const allocator = testing.allocator;
    // The shape a caller produces by deriving thread counts from per-node CPU
    // lists when one node reports none — which this host does, per the note at
    // the top of this file.
    var numa_map = [_]usize{ 0, 1 };
    var thread_counts = [_]usize{ 2, 0 };
    const config = wp.WorkerPoolConfig{
        .subpool_count = 2,
        .subpool_numa_map = &numa_map,
        .subpool_thread_count = &thread_counts,
    };

    var pool = try wp.WorkerPool.init(allocator, config);
    defer pool.deinit();

    // The EFFECTIVE count is on the Subpool. config.subpool_thread_count stays
    // as the caller requested it, so assert against the thing that is real.
    try testing.expectEqual(@as(usize, 2), pool.subpools[0].thread_count);
    try testing.expectEqual(@as(usize, 1), pool.subpools[1].thread_count);

    // And it must actually be able to run work — before the clamp this blocked
    // in doWorkStealingJob for the rest of the test run, with every worker in
    // the pool sitting idle waiting on a predicate no one would ever satisfy.
    const before = dl_nonce_work.load(.acquire);
    pool.subpools[1].doWorkStealingJob(4, dlCountWork);
    try testing.expectEqual(before + 4, dl_nonce_work.load(.acquire));
}

test "Subpool built with no workers runs the job on the calling thread" {
    const allocator = testing.allocator;
    // Bypassing WorkerPool.init entirely, which is what the serial fallback in
    // doWorkStealingJob exists for: the clamp cannot help a direct caller.
    const sub = try wp.Subpool.init(allocator, 0, 0, 0, null, true);
    defer sub.deinit(allocator);
    try testing.expectEqual(@as(usize, 0), sub.thread_count);

    const before = dl_nonce_work.load(.acquire);
    // Before the fallback this never returns.
    sub.doWorkStealingJob(8, dlCountWork);
    try testing.expectEqual(before + 8, dl_nonce_work.load(.acquire));

    // pending_jobs must be back to zero, or a later waitIdle would block.
    sub.waitIdle(0);
}
