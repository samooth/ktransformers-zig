// NUMA-aware worker pool for ktransformers-zig
// Ported from ktransformers kt-kernel/cpu_backend/worker_pool.h

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Worker pool configuration
pub const WorkerPoolConfig = struct {
    subpool_count: usize = 1,
    subpool_numa_map: []usize = &.{},
    subpool_thread_count: []usize = &.{},
    enable_work_stealing: bool = true,

    pub fn default(allocator: Allocator) !WorkerPoolConfig {
        const cpu_count = std.Thread.getCpuCount() orelse 1;
        const config = WorkerPoolConfig{
            .subpool_count = 1,
            .subpool_numa_map = try allocator.alloc(usize, 1),
            .subpool_thread_count = try allocator.alloc(usize, 1),
        };
        config.subpool_numa_map[0] = 0;
        config.subpool_thread_count[0] = cpu_count;
        return config;
    }
};

/// Task function signature
pub const TaskFn = *const fn (usize, *anyopaque) void;

/// Work item for the thread pool
pub const WorkItem = struct {
    callback: TaskFn,
    arg: *anyopaque,
    subpool_idx: usize,
};

/// Subpool (per-NUMA-node thread group)
pub const Subpool = struct {
    threads: []std.Thread,
    queue: *std.Thread.Queue(WorkItem),
    idx: usize,
    numa_id: usize,
    thread_count: usize,
    shutdown: std.atomic.Bool,

    pub fn init(
        allocator: Allocator,
        idx: usize,
        numa_id: usize,
        thread_count: usize,
        _enable_work_stealing: bool,
    ) !Subpool {
        _ = _enable_work_stealing;
        const queue = try allocator.create(std.Thread.Queue(WorkItem));
        var threads = try allocator.alloc(std.Thread, thread_count);

        const subpool = Subpool{
            .threads = threads,
            .queue = queue,
            .idx = idx,
            .numa_id = numa_id,
            .thread_count = thread_count,
            .shutdown = std.atomic.Bool.init(false),
        };

        for (0..thread_count)  | i |  {
            threads[i] = try std.Thread.spawn(.{}, subpool.workerLoop, .{i});
        }

        return subpool;
    }

    fn workerLoop(self: *Subpool, thread_idx: usize) void {
        _ = thread_idx;
        while (!self.shutdown.load(.acquire)) {
            const item = self.queue.pop() orelse {
                std.time.sleep(100_000); // 100µs
                continue;
            };
            item.callback(self.idx, item.arg);
        }
    }

    pub fn submit(self: *Subpool, callback: TaskFn, arg: *anyopaque) void {
        const item = WorkItem{ .callback = callback, .arg = arg, .subpool_idx = self.idx };
        self.queue.push(item) catch @panic("queue full");
    }

    pub fn deinit(self: *Subpool, allocator: Allocator) void {
        self.shutdown.store(true, .release);
        for (self.threads)  | thread |  {
            thread.join() catch {};
        }
        allocator.destroy(self.queue);
        allocator.free(self.threads);
    }
};

/// Main worker pool with NUMA-aware subpools
pub const WorkerPool = struct {
    config: WorkerPoolConfig,
    subpools: []Subpool,
    allocator: Allocator,
    backend: std.Thread.Pool, // fallback for single-pool mode

    pub fn init(allocator: Allocator, config: WorkerPoolConfig) !WorkerPool {
        var subpools = try allocator.alloc(Subpool, config.subpool_count);
        const backend = std.Thread.Pool.init(allocator, config.subpool_thread_count[0]) catch @panic("backend init");

        for (0..config.subpool_count)  | i |  {
            subpools[i] = try Subpool.init(
                allocator,
                i,
                config.subpool_numa_map[i],
                config.subpool_thread_count[i],
                config.enable_work_stealing,
            );
        }

        return WorkerPool{
            .config = config,
            .subpools = subpools,
            .allocator = allocator,
            .backend = backend,
        };
    }

    pub fn initSimple(allocator: Allocator, thread_count: usize) !WorkerPool {
        const config = try WorkerPoolConfig.default(allocator);
        config.subpool_thread_count[0] = thread_count;
        return @This().init(allocator, config);
    }

    /// Get subpool for specific NUMA node
    pub fn getSubpool(self: *WorkerPool, idx: usize) *Subpool {
        return &self.subpools[idx];
    }

    /// Dispatch work to specific NUMA node
    pub fn doNumaJob(
        self: *WorkerPool,
        numa_id: usize,
        callback: TaskFn,
        arg: *anyopaque,
    ) void {
        for (self.subpools) |*subpool| {
            if (subpool.numa_id == numa_id) {
                subpool.submit(callback, arg);
                return;
            }
        }
        // Fallback to first subpool
        self.subpools[0].submit(callback, arg);
    }

    /// Dispatch work to all NUMA nodes (same function)
    pub fn doNumaJobAll(self: *WorkerPool, callback: TaskFn, arg: *anyopaque) void {
        for (self.subpools) |*subpool| {
            subpool.submit(callback, arg);
        }
    }

    /// Work-stealing job distribution
    pub fn doWorkStealingJob(
        self: *WorkerPool,
        count: usize,
        setup_fn: ?fn (usize) void,
        _work_fn: fn (usize, usize) void,
        _cleanup_fn: ?fn (usize) void,
    ) void {
        _ = _work_fn;
        _ = _cleanup_fn;
        _ = setup_fn;
        const total_threads = self.getTotalThreads();
        const items_per_thread = @max(1, count / total_threads);

        // Distribute work across all subpools
        var task_idx: usize = 0;
        for (self.subpools) |*subpool| {
            for (0..subpool.thread_count) |thread_idx| {
                _ = thread_idx;
                const start = task_idx;
                const end = @min(count, start + items_per_thread);
                if (start >= end) break;

                task_idx = end;
                const captured_start = start;
                const captured_end = end;

                subpool.submit(
                    WorkItem{
                        .callback = undefined,
                        .arg = @ptrCast(struct { start: usize, end: usize } { .start = captured_start, .end = captured_end }),
                        .subpool_idx = self.idx,
                    },
                );
            }
        }

        // Wait for completion (simplified - in practice use completion counter)
        std.time.sleep(1_000_000); // 1ms
    }

    fn getTotalThreads(self: *WorkerPool) usize {
        var total: usize = 0;
        for (self.subpools)  | subpool |  {
            total += subpool.thread_count;
        }
        return total;
    }

    /// Dispatch to backend (for compatibility)
    pub fn dispenseBackend(self: *WorkerPool) *std.Thread.Pool {
        return &self.backend;
    }

    pub fn deinit(self: *WorkerPool) void {
        for (self.subpools) |*subpool| {
            subpool.deinit(self.allocator);
        }
        self.allocator.free(self.subpools);
        self.backend.deinit();
    }
};

/// Get NUMA node for current CPU
pub fn numaNodeOfCpu(_cpu: usize) usize {
    _ = _cpu;
    // TODO: Read from /sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id
    return 0;
}

/// Get CPU count per NUMA node
pub fn getCpuCountPerNuma(allocator: Allocator) ![]usize {
    // TODO: Parse /proc/cpuinfo or use hwloc
    const count = std.Thread.getCpuCount() orelse 1;
    const result = try allocator.alloc(usize, 1);
    result[0] = count;
    return result;
}