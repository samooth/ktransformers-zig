// NUMA-Aware Worker Pool for ktransformers-zig
// Combines thread pinning, local memory allocation, and work distribution

const std = @import("std");
const NumaTopology = @import("numa_topology.zig").NumaTopology;
const NumaMemory = @import("numa_memory.zig");
const NumaNodeMask = NumaMemory.NumaNodeMask;
const NumaPolicy = NumaMemory.NumaPolicy;
const CpuSet = NumaMemory.CpuSet;

/// NUMA-aware worker configuration
pub const NumaWorkerConfig = struct {
    numa_node: usize,
    cpu_start: usize,
    cpu_count: usize,
    memory_policy: NumaPolicy,
    enable_pinning: bool = true,
    enable_local_alloc: bool = true,
};

/// A worker thread bound to a specific NUMA node
pub const NumaWorker = struct {
    thread: std.Thread,
    config: NumaWorkerConfig,
    node_id: usize,
    cpu_id: usize,

    pub fn spawn(
        config: NumaWorkerConfig,
        comptime f: anytype,
        args: anytype,
    ) !NumaWorker {
        // Pin to NUMA node before spawning
        if (config.enable_pinning) {
            var cs = CpuSet.init();
            for (config.cpu_start..config.cpu_start + config.cpu_count) |cpu| {
                cs.set(cpu);
            }
            try NumaMemory.setThreadAffinity(cs.toSlice()[0..config.cpu_start + config.cpu_count]);
        }

        // Set NUMA memory policy
        if (config.enable_local_alloc) {
            var mask = NumaNodeMask.init();
            mask.set(config.numa_node);
            try NumaMemory.setThreadNumaPolicy(mask, config.memory_policy);
        }

        const thread = try std.Thread.spawn(.{}, f, args);

        return NumaWorker{
            .thread = thread,
            .config = config,
            .node_id = config.numa_node,
            .cpu_id = config.cpu_start,
        };
    }

    pub fn join(self: *NumaWorker) void {
        self.thread.join();
    }
};

/// NUMA-aware task for the worker pool
pub const NumaTask = struct {
    callback: *const fn (*anyopaque) void,
    arg: *anyopaque,
    preferred_node: usize,
};

/// NUMA-aware worker pool
/// Each subpool is bound to a specific NUMA node
pub const NumaWorkerPool = struct {
    topology: NumaTopology,
    workers: []std.Thread,
    queues: []std.ArrayList(NumaTask),
    mutexes: []std.Thread.Mutex,
    conds: []std.Thread.Condition,
    shutdown: std.atomic.Value(bool),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, threads_per_node: usize) !NumaWorkerPool {
        var topology = try NumaTopology.detect(allocator);

        const total_threads = topology.num_nodes * threads_per_node;
        const workers = try allocator.alloc(std.Thread, total_threads);
        const queues = try allocator.alloc(std.ArrayList(NumaTask), topology.num_nodes);
        const mutexes = try allocator.alloc(std.Thread.Mutex, topology.num_nodes);
        const conds = try allocator.alloc(std.Thread.Condition, topology.num_nodes);

        var pool = NumaWorkerPool{
            .topology = topology,
            .workers = workers,
            .queues = queues,
            .mutexes = mutexes,
            .conds = conds,
            .shutdown = std.atomic.Value(bool).init(false),
            .allocator = allocator,
        };

        // Initialize queues, mutexes, conds
        for (0..topology.num_nodes) |n| {
            queues[n] = std.ArrayList(NumaTask).init(allocator);
            mutexes[n] = .{};
            conds[n] = .{};
        }

        // Spawn workers per NUMA node
        var thread_idx: usize = 0;
        for (0..topology.num_nodes) |node_id| {
            const node_cpus = topology.cpusForNode(node_id);
            const cpus_per_worker = @max(1, node_cpus.len / threads_per_node);

            for (0..threads_per_node) |t| {
                const cpu_start = t * cpus_per_worker;
                const cpu_count = @min(cpus_per_worker, node_cpus.len - cpu_start);

                const worker_config = NumaWorkerConfig{
                    .numa_node = node_id,
                    .cpu_start = if (node_cpus.len > 0) node_cpus[cpu_start % node_cpus.len] else 0,
                    .cpu_count = cpu_count,
                    .memory_policy = .bind,
                    .enable_pinning = true,
                    .enable_local_alloc = true,
                };

                workers[thread_idx] = try std.Thread.spawn(.{}, workerLoop, .{ &pool, node_id, worker_config });
                thread_idx += 1;
            }
        }

        return pool;
    }

    pub fn deinit(self: *NumaWorkerPool) void {
        self.shutdown.store(true, .release);
        for (self.conds) |*cond| {
            cond.broadcast();
        }
        for (self.workers) |worker| {
            worker.join();
        }
        for (self.queues) |*queue| {
            queue.deinit();
        }
        self.topology.deinit();
        self.allocator.free(self.workers);
        self.allocator.free(self.queues);
        self.allocator.free(self.mutexes);
        self.allocator.free(self.conds);
    }

    fn workerLoop(self: *NumaWorkerPool, node_id: usize, config: NumaWorkerConfig) void {
        // Pin thread to NUMA node
        if (config.enable_pinning) {
            var cs = CpuSet.init();
            const node_cpus = self.topology.cpusForNode(node_id);
            for (node_cpus) |cpu| {
                cs.set(cpu);
            }
            NumaMemory.setThreadAffinity(cs.toSlice()) catch {};
        }

        // Set local NUMA policy
        if (config.enable_local_alloc) {
            var mask = NumaNodeMask.init();
            mask.set(node_id);
            NumaMemory.setThreadNumaPolicy(mask, config.memory_policy) catch {};
        }

        const queue = &self.queues[node_id];
        const mutex = &self.mutexes[node_id];
        const cond = &self.conds[node_id];

        while (true) {
            mutex.lock();
            defer mutex.unlock();

            while (queue.items.len == 0 and !self.shutdown.load(.acquire)) {
                cond.wait(mutex);
            }

            if (self.shutdown.load(.acquire)) {
                return;
            }

            if (queue.items.len > 0) {
                const task = queue.pop();
                mutex.unlock();
                task.callback(task.arg);
                mutex.lock();
            }
        }
    }

    /// Submit task to specific NUMA node
    pub fn submitToNode(self: *NumaWorkerPool, node_id: usize, callback: *const fn (*anyopaque) void, arg: *anyopaque) void {
        const actual_node = @min(node_id, self.topology.num_nodes - 1);
        const task = NumaTask{ .callback = callback, .arg = arg, .preferred_node = actual_node };

        self.mutexes[actual_node].lock();
        defer self.mutexes[actual_node].unlock();
        self.queues[actual_node].append(task) catch @panic("OOM");
        self.conds[actual_node].signal();
    }

    /// Submit task to local NUMA node (based on current CPU)
    pub fn submitLocal(self: *NumaWorkerPool, callback: *const fn (*anyopaque) void, arg: *anyopaque) void {
        // Get current CPU
        const cpu = std.Thread.getCurrentId() % self.topology.num_cpus;
        const node_id = self.topology.nodeForCpu(cpu) orelse 0;
        self.submitToNode(node_id, callback, arg);
    }

    /// Submit task to least-loaded NUMA node
    pub fn submitBalanced(self: *NumaWorkerPool, callback: *const fn (*anyopaque) void, arg: *anyopaque) void {
        var min_size: usize = std.math.maxInt(usize);
        var min_node: usize = 0;

        for (0..self.topology.num_nodes) |n| {
            self.mutexes[n].lock();
            const size = self.queues[n].items.len;
            self.mutexes[n].unlock();
            if (size < min_size) {
                min_size = size;
                min_node = n;
            }
        }

        self.submitToNode(min_node, callback, arg);
    }

    /// Allocate memory on specific NUMA node
    pub fn allocOnNode(self: *NumaWorkerPool, size: usize, alignment: usize, node_id: usize) ![]u8 {
        return NumaMemory.allocNumaAndTouch(self.allocator, size, alignment, node_id);
    }

    /// Allocate memory interleaved across all nodes
    pub fn allocInterleaved(self: *NumaWorkerPool, size: usize, alignment: usize) ![]u8 {
        const mem = try self.allocator.alignedAlloc(u8, @enumFromInt(std.math.log2(alignment)), size);
        try NumaMemory.interleaveMemory(mem.ptr, mem.len, self.topology.num_nodes);
        NumaMemory.touchMemory(mem);
        return mem;
    }

    /// Print pool status
    pub fn printStatus(self: *NumaWorkerPool) void {
        std.debug.print("NUMA Worker Pool Status:\n", .{});
        for (0..self.topology.num_nodes) |n| {
            self.mutexes[n].lock();
            const size = self.queues[n].items.len;
            self.mutexes[n].unlock();
            std.debug.print("  Node {d}: {d} pending tasks\n", .{ n, size });
        }
    }
};
