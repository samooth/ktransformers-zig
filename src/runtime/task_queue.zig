// Task queue for ktransformers-zig
// Ported from ktransformers kt-kernel/cpu_backend/task_queue.h

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Task function type
pub const TaskFn = *const fn (*anyopaque) void;

/// Task queue item
pub const Task = struct {
    callback: TaskFn,
    arg: *anyopaque,
};

/// Lock-free task queue using atomic ring buffer
pub const TaskQueue = struct {
    buffer: []Task,
    head: std.atomic.Usize,
    tail: std.atomic.Usize,
    mask: usize,
    capacity: usize,

    pub fn init(allocator: Allocator, capacity: usize) !TaskQueue {
        // Round up to power of 2
        var cap = capacity;
        cap |= cap >> 1;
        cap |= cap >> 2;
        cap |= cap >> 4;
        cap |= cap >> 8;
        cap |= cap >> 16;
        cap += 1;

        const buffer = try allocator.alloc(Task, cap);
        @memset(buffer, 0);

        return TaskQueue{
            .buffer = buffer,
            .head = std.atomic.Usize.init(0),
            .tail = std.atomic.Usize.init(0),
            .mask = cap - 1,
            .capacity = cap,
        };
    }

    pub fn deinit(self: *TaskQueue, allocator: Allocator) void {
        allocator.free(self.buffer);
        self.buffer = undefined;
    }

    /// Enqueue a task (lock-free, single producer)
    pub fn enqueue(self: *TaskQueue, callback: TaskFn, arg: *anyopaque) bool {
        const head = self.head.load(.relaxed);
        const next_head = (head + 1) & self.mask;

        if (next_head == self.tail.load(.acquire)) {
            return false; // Queue full
        }

        self.buffer[head] = .{ .callback = callback, .arg = arg };
        self.head.store(next_head, .release);
        return true;
    }

    /// Try to dequeue a task (lock-free, single consumer)
    pub fn dequeue(self: *TaskQueue) ?Task {
        const tail = self.tail.load(.relaxed);
        if (tail == self.head.load(.acquire)) {
            return null; // Queue empty
        }

        const task = self.buffer[tail];
        self.tail.store((tail + 1) & self.mask, .release);
        return task;
    }

    /// Get approximate queue size
    pub fn size(self: *TaskQueue) usize {
        const head = self.head.load(.relaxed);
        const tail = self.tail.load(.relaxed);
        return (head + self.capacity - tail) & self.mask;
    }

    /// Check if empty
    pub fn isEmpty(self: *TaskQueue) bool {
        return self.tail.load(.relaxed) == self.head.load(.relaxed);
    }

    /// Check if full
    pub fn isFull(self: *TaskQueue) bool {
        const head = self.head.load(.relaxed);
        const next_head = (head + 1) & self.mask;
        return next_head == self.tail.load(.acquire);
    }
};

/// Multi-producer, multi-consumer task queue using mutex
pub const MpmcTaskQueue = struct {
    mutex: std.Thread.Mutex,
    queue: std.ArrayList(Task),
    cond: std.Thread.Condition,
    is_shutdown: bool,

    pub fn init(allocator: Allocator) MpmcTaskQueue {
        return MpmcTaskQueue{
            .mutex = std.Thread.Mutex.init(),
            .queue = std.ArrayList(Task).init(allocator),
            .cond = std.Thread.Condition.init(),
            .shutdown = false,
        };
    }

    pub fn deinit(self: *MpmcTaskQueue) void {
        self.queue.deinit();
        self.mutex.deinit();
        self.cond.deinit();
    }

    pub fn enqueue(self: *MpmcTaskQueue, callback: TaskFn, arg: *anyopaque) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        _ = self.queue.append(.{ .callback = callback, .arg = arg });
        self.cond.signal();
    }

    pub fn dequeue(self: *MpmcTaskQueue) ?Task {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (self.queue.items.len == 0 and !self.shutdown) {
            self.cond.wait(&self.mutex);
        }

        if (self.queue.items.len == 0) {
            return null; // Shutdown
        }

        const task = self.queue.items[0];
        _ = self.queue.pop();
        return task;
    }

    pub fn shutdown(self: *MpmcTaskQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.shutdown = true;
        self.cond.broadcast();
    }

    pub fn size(self: *MpmcTaskQueue) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.queue.items.len;
    }
};

/// Task submission handle for async execution
pub const TaskHandle = struct {
    queue: *TaskQueue,
    callback: TaskFn,
    arg: *anyopaque,
    done: std.atomic.Bool,

    pub fn submit(queue: *TaskQueue, callback: TaskFn, arg: *anyopaque) !TaskHandle {
        if (!queue.enqueue(callback, arg)) {
            return error.QueueFull;
        }
        return TaskHandle{
            .queue = queue,
            .callback = callback,
            .arg = arg,
            .done = std.atomic.Bool.init(false),
        };
    }

    pub fn wait(self: *TaskHandle) void {
        while (!self.done.load(.acquire)) {
            std.time.sleep(10_000); // 10µs
        }
    }

    pub fn isDone(self: *TaskHandle) bool {
        return self.done.load(.acquire);
    }
};

/// Batched task submission for efficiency
pub const BatchedTaskQueue = struct {
    queue: TaskQueue,
    batch: std.ArrayList(Task),
    batch_size: usize,

    pub fn init(allocator: Allocator, capacity: usize, batch_size: usize) !BatchedTaskQueue {
        return BatchedTaskQueue{
            .queue = try TaskQueue.init(allocator, capacity),
            .batch = std.ArrayList(Task).init(allocator),
            .batch_size = batch_size,
        };
    }

    pub fn deinit(self: *BatchedTaskQueue, allocator: Allocator) void {
        self.queue.deinit(allocator);
        self.batch.deinit();
    }

    pub fn enqueue(self: *BatchedTaskQueue, callback: TaskFn, arg: *anyopaque) !void {
        _ = self.batch.append(.{ .callback = callback, .arg = arg });
        if (self.batch.items.len >= self.batch_size) {
            try self.flush();
        }
    }

    pub fn flush(self: *BatchedTaskQueue) !void {
        for (self.batch.items) |task| {
            if (!self.queue.enqueue(task.callback, task.arg)) {
                return error.QueueFull;
            }
        }
        self.batch.clearRetainingCapacity();
    }
};