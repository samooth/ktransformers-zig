// NUMA-aware worker pool for ktransformers-zig
// Ported from ktransformers kt-kernel/cpu_backend/worker_pool.h
//
// Work-stealing design (mirrors C++ InNumaPool):
//   - Persistent worker threads block on a pthread condvar when no job is active
//   - doWorkStealingJob sets task_curr=0, task_end=count, then broadcasts
//   - Each worker steals task indices via task_curr.fetchAdd(1) and runs work_fn
//   - Main thread waits on done_cv until done_count == task_end
//   - No per-task allocation, no queue contention — just an atomic counter
//
// Uses std.c.pthread_mutex_t/cond_t directly because Zig 0.16's std.Thread
// has no blocking mutex/condvar (only std.atomic.Mutex spinlock). The pthread
// primitives block the OS thread when idle — essential for a worker pool.

const std = @import("std");
const Allocator = std.mem.Allocator;
const c = std.c;

/// Worker pool configuration
pub const WorkerPoolConfig = struct {
    subpool_count: usize = 1,
    subpool_numa_map: []usize = &.{}, // owned, freed by deinit
    subpool_thread_count: []usize = &.{}, // owned, freed by deinit
    enable_work_stealing: bool = true,
    allocator: Allocator = std.heap.page_allocator,

    pub fn default(allocator: Allocator) !WorkerPoolConfig {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const config = WorkerPoolConfig{
            .subpool_count = 1,
            .subpool_numa_map = try allocator.alloc(usize, 1),
            .subpool_thread_count = try allocator.alloc(usize, 1),
            .allocator = allocator,
        };
        config.subpool_numa_map[0] = 0;
        config.subpool_thread_count[0] = cpu_count;
        return config;
    }

    pub fn deinit(self: *WorkerPoolConfig) void {
        if (self.subpool_numa_map.len > 0) {
            self.allocator.free(self.subpool_numa_map);
        }
        if (self.subpool_thread_count.len > 0) {
            self.allocator.free(self.subpool_thread_count);
        }
        self.* = undefined;
    }
};

/// Task function signature
pub const TaskFn = *const fn (usize, *anyopaque) void;

/// Subpool (per-NUMA-node thread group) with work-stealing
pub const Subpool = struct {
    threads: []std.Thread,
    allocator: Allocator,
    idx: usize,
    numa_id: usize,
    thread_count: usize,

    // Work-stealing state
    task_curr: std.atomic.Value(usize),
    task_end: usize,
    work_fn: ?*const fn (usize) void,
    active: std.atomic.Value(bool),
    shutdown: std.atomic.Value(bool),
    done_count: std.atomic.Value(usize),

    // Blocking synchronization (pthread — threads actually sleep when idle)
    mutex: c.pthread_mutex_t,
    work_cv: c.pthread_cond_t,
    done_cv: c.pthread_cond_t,

    pub fn init(
        allocator: Allocator,
        idx: usize,
        numa_id: usize,
        thread_count: usize,
        _enable_work_stealing: bool,
    ) !*Subpool {
        _ = _enable_work_stealing;
        const threads = try allocator.alloc(std.Thread, thread_count);

        const subpool = try allocator.create(Subpool);
        subpool.* = Subpool{
            .threads = threads,
            .allocator = allocator,
            .idx = idx,
            .numa_id = numa_id,
            .thread_count = thread_count,
            .task_curr = std.atomic.Value(usize).init(0),
            .task_end = 0,
            .work_fn = null,
            .active = std.atomic.Value(bool).init(false),
            .shutdown = std.atomic.Value(bool).init(false),
            .done_count = std.atomic.Value(usize).init(0),
            .mutex = c.PTHREAD_MUTEX_INITIALIZER,
            .work_cv = c.PTHREAD_COND_INITIALIZER,
            .done_cv = c.PTHREAD_COND_INITIALIZER,
        };

        for (0..thread_count) |t| {
            threads[t] = try std.Thread.spawn(.{}, workerLoop, .{subpool});
        }

        return subpool;
    }

    fn workerLoop(self: *Subpool) void {
        while (true) {
            // Wait for work or shutdown
            _ = c.pthread_mutex_lock(&self.mutex);
            while (!self.shutdown.load(.acquire) and !self.active.load(.acquire)) {
                _ = c.pthread_cond_wait(&self.work_cv, &self.mutex);
            }
            const sd = self.shutdown.load(.acquire);
            _ = c.pthread_mutex_unlock(&self.mutex);
            if (sd) break;

            // Steal and execute tasks (no mutex held — atomic counter distributes work)
            const end = self.task_end;
            const fn_ptr = self.work_fn;
            while (true) {
                const idx = self.task_curr.fetchAdd(1, .monotonic);
                if (idx >= end) break;
                if (fn_ptr) |f| f(idx);
                _ = self.done_count.fetchAdd(1, .monotonic);
            }

            // Signal completion (hold mutex to avoid missed wakeup)
            _ = c.pthread_mutex_lock(&self.mutex);
            _ = c.pthread_cond_signal(&self.done_cv);
            _ = c.pthread_mutex_unlock(&self.mutex);
        }
    }

    /// Dispatch a work-stealing job. Blocks until all `count` tasks complete.
    pub fn doWorkStealingJob(self: *Subpool, count: usize, work_fn: *const fn (usize) void) void {
        _ = c.pthread_mutex_lock(&self.mutex);
        self.task_curr.store(0, .monotonic);
        self.task_end = count;
        self.done_count.store(0, .monotonic);
        self.work_fn = work_fn;
        self.active.store(true, .release);
        _ = c.pthread_cond_broadcast(&self.work_cv);

        // Wait for all tasks to complete
        while (self.done_count.load(.acquire) < count) {
            _ = c.pthread_cond_wait(&self.done_cv, &self.mutex);
        }

        self.active.store(false, .release);
        self.work_fn = null;
        _ = c.pthread_mutex_unlock(&self.mutex);
    }

/// Legacy submit (for C API compatibility). Runs the callback as a single task.
/// Uses a global adapter because Zig function pointers can't capture context.
var g_submit_callback: ?TaskFn = null;
var g_submit_arg: ?*anyopaque = null;
fn submitWrapper(idx: usize) void {
    _ = idx;
    if (g_submit_callback) |cb| {
        if (g_submit_arg) |arg| cb(0, arg);
    }
}
pub fn submit(self: *Subpool, callback: TaskFn, arg: *anyopaque) void {
    g_submit_callback = callback;
    g_submit_arg = arg;
    self.doWorkStealingJob(1, submitWrapper);
}

    pub fn deinit(self: *Subpool, allocator: Allocator) void {
        self.shutdown.store(true, .release);
        _ = c.pthread_cond_broadcast(&self.work_cv);
        for (self.threads) |thread| {
            thread.join();
        }
        // Destroy synchronization primitives BEFORE freeing the Subpool
        // (they live inside the Subpool struct — destroying after free is UB).
        _ = c.pthread_mutex_destroy(&self.mutex);
        _ = c.pthread_cond_destroy(&self.work_cv);
        _ = c.pthread_cond_destroy(&self.done_cv);
        allocator.free(self.threads);
        allocator.destroy(self);
    }
};

/// Main worker pool with NUMA-aware subpools
pub const WorkerPool = struct {
    config: WorkerPoolConfig,
    subpools: []*Subpool,
    allocator: Allocator,
    backend: usize,
    owns_config: bool,

    pub fn init(allocator: Allocator, config: WorkerPoolConfig) !WorkerPool {
        const subpools = try allocator.alloc(*Subpool, config.subpool_count);

        for (0..config.subpool_count) |i| {
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
            .backend = config.subpool_thread_count[0],
            .owns_config = false,
        };
    }

    pub fn initSimple(allocator: Allocator, thread_count: usize) !WorkerPool {
        const config = try WorkerPoolConfig.default(allocator);
        config.subpool_thread_count[0] = thread_count;
        const pool = try @This().init(allocator, config);
        var pool_mut = pool;
        pool_mut.owns_config = true;
        return pool_mut;
    }

    pub fn getSubpool(self: *WorkerPool, idx: usize) *Subpool {
        return self.subpools[idx];
    }

    /// Dispatch a work-stealing job across all subpools' threads.
    pub fn doWorkStealingJob(
        self: *WorkerPool,
        count: usize,
        setup_fn: ?fn (usize) void,
        work_fn: fn (usize, usize) void,
        cleanup_fn: ?fn (usize) void,
    ) void {
        if (count == 0) return;
        const total_threads = self.getTotalThreads();
        if (total_threads == 0) return;

        if (setup_fn) |setup| setup(0);

        // Distribute work across subpools
        const per_thread = @max(1, count / total_threads);
        var dispatched: usize = 0;
        var subpool_jobs: [16]usize = undefined;
        var num_subpools: usize = 0;

        for (self.subpools) |subpool| {
            const share = @min(count - dispatched, per_thread * subpool.thread_count);
            if (share == 0) break;
            subpool_jobs[num_subpools] = share;
            dispatched += share;
            num_subpools += 1;
            if (dispatched >= count) break;
        }

        if (num_subpools == 1) {
            const captured_fn = work_fn;
            const captured_count = subpool_jobs[0];
            self.subpools[0].doWorkStealingJob(captured_count, struct {
                fn f(idx: usize) void {
                    captured_fn(idx, 0);
                }
            }.f);
        } else {
            // Multiple subpools: dispatch each in its own thread
            const Context = struct {
                subpool: *Subpool,
                count: usize,
                work_fn: fn (usize, usize) void,
                subpool_idx: usize,
            };
            var ctxs: [16]Context = undefined;
            var dispatch_threads: [16]std.Thread = undefined;
            for (0..num_subpools) |i| {
                ctxs[i] = .{
                    .subpool = self.subpools[i],
                    .count = subpool_jobs[i],
                    .work_fn = work_fn,
                    .subpool_idx = i,
                };
                dispatch_threads[i] = std.Thread.spawn(.{}, struct {
                    fn run(ctx: Context) void {
                        ctx.subpool.doWorkStealingJob(ctx.count, struct {
                            fn f(idx: usize) void {
                                ctx.work_fn(idx, ctx.subpool_idx);
                            }
                        }.f);
                    }
                }.run, .{ctxs[i]}) catch @panic("dispatch spawn failed");
            }
            for (0..num_subpools) |i| {
                dispatch_threads[i].join();
            }
        }

        if (cleanup_fn) |cleanup| cleanup(0);
    }

    /// Dispatch work to specific NUMA node (legacy compatibility)
    pub fn doNumaJob(
        self: *WorkerPool,
        numa_id: usize,
        callback: *const fn (usize) void,
        arg: *anyopaque,
    ) void {
        _ = arg;
        for (self.subpools) |subpool| {
            if (subpool.numa_id == numa_id) {
                subpool.doWorkStealingJob(1, callback);
                return;
            }
        }
        self.subpools[0].doWorkStealingJob(1, callback);
    }

    /// Dispatch work to all NUMA nodes
    pub fn doNumaJobAll(self: *WorkerPool, callback: *const fn (usize) void, arg: *anyopaque) void {
        _ = arg;
        for (self.subpools) |subpool| {
            subpool.doWorkStealingJob(1, callback);
        }
    }

    pub fn getTotalThreads(self: *WorkerPool) usize {
        var total: usize = 0;
        for (self.subpools) |subpool| {
            total += subpool.thread_count;
        }
        return total;
    }

    pub fn dispenseBackend(self: *WorkerPool) usize {
        return self.backend;
    }

    pub fn deinit(self: *WorkerPool) void {
        for (self.subpools) |subpool| {
            subpool.deinit(self.allocator);
        }
        self.allocator.free(self.subpools);
        if (self.owns_config) {
            self.config.deinit();
        }
    }
};

/// Get NUMA node for a given CPU by reading /sys
pub fn numaNodeOfCpu(cpu: usize) usize {
    var path_buf: [128]u8 = undefined;
    const path = std.fmt.bufPrintZ(&path_buf, "/sys/devices/system/cpu/cpu{}/topology/physical_package_id", .{cpu}) catch return 0;
    const file = c.fopen(path.ptr, "r") orelse return 0;
    defer _ = c.fclose(file);
    var buf: [16]u8 = undefined;
    const n = c.fread(buf[0..].ptr, 1, buf.len, file);
    if (n == 0) return 0;
    const trimmed = std.mem.trimEnd(u8, buf[0..n], "\n");
    return std.fmt.parseInt(usize, trimmed, 10) catch 0;
}

/// Get CPU count per NUMA node by parsing /proc/cpuinfo
pub fn getCpuCountPerNuma(allocator: Allocator) ![]usize {
    const file = c.fopen("/proc/cpuinfo", "r") orelse {
        const result = try allocator.alloc(usize, 1);
        result[0] = std.Thread.getCpuCount() catch 1;
        return result;
    };
    defer _ = c.fclose(file);

    var counts = std.AutoHashMap(usize, usize).init(allocator);
    defer counts.deinit();

    // Read entire file into a buffer
    var buf: [65536]u8 = undefined;
    const n = c.fread(buf[0..].ptr, 1, buf.len, file);
    const content = buf[0..n];

    var current_package: ?usize = null;
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "physical id")) {
            const colon = std.mem.indexOf(u8, line, ":") orelse continue;
            const val = std.mem.trim(u8, line[colon + 1..], " \t");
            const pkg = std.fmt.parseInt(usize, val, 10) catch continue;
            current_package = pkg;
            // Each physical id line corresponds to one logical CPU
            const entry = try counts.getOrPut(pkg);
            if (!entry.found_existing) {
                entry.value_ptr.* = 0;
            }
            entry.value_ptr.* += 1;
        } else if (line.len == 0) {
            current_package = null;
        }
    }

    const num_numa = @max(1, counts.count());
    const result = try allocator.alloc(usize, num_numa);
    @memset(result, 0);
    var it = counts.iterator();
    while (it.next()) |entry| {
        const pkg = entry.key_ptr.*;
        const count = entry.value_ptr.*;
        if (pkg < num_numa) {
            result[pkg] = count;
        } else {
            result[num_numa - 1] += count;
        }
    }

    if (counts.count() == 0) {
        result[0] = std.Thread.getCpuCount() catch 1;
    }

    return result;
}
