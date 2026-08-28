// Memory management for ktransformers-zig
// Provides aligned allocation, NUMA-aware allocation, and huge page support

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Arena allocator with 64-byte alignment for SIMD/AMX buffers
pub const SimdArena = struct {
    arena: std.heap.ArenaAllocator,
    alignment: usize,

    pub fn init(backing: Allocator, alignment: usize) SimdArena {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing),
            .alignment = alignment,
        };
    }

    pub fn deinit(self: *SimdArena) void {
        self.arena.deinit();
    }

    pub fn allocator(self: *SimdArena) Allocator {
        return self.arena.allocator();
    }

    /// Allocate with alignment (default 64 bytes for cache lines)
    pub fn alignedAlloc(self: *SimdArena, comptime T: type, count: usize, alignment: usize) ![]T {
        const bytes = count * @sizeOf(T);
        const ptr = try self.arena.allocator().alignedAlloc(alignment, bytes);
        return @as([]u8, @ptrCast(@alignCast(std.mem.sliceAsBytes(ptr)[0..bytes].ptr)));
    }

    /// Allocate 64-byte aligned buffer
    pub fn alloc64(self: *SimdArena, comptime T: type, count: usize) ![]T {
        return self.alignedAlloc(T, count, 64);
    }

    /// Allocate with specific alignment requirement
    pub fn allocAligned(self: *SimdArena, comptime T: type, count: usize, alignment: usize) ![]T {
        return self.alignedAlloc(T, count, alignment);
    }
};

/// NUMA-aware memory allocation
pub const NumaAllocator = struct {
    allocator: Allocator,
    numa_node: ?usize,

    pub fn init(allocator: Allocator, numa_node: ?usize) NumaAllocator {
        return .{ .allocator = allocator, .numa_node = numa_node };
    }

    pub fn alloc(self: *NumaAllocator, comptime T: type, count: usize) ![]T {
        // TODO: Use numactl/libnuma for actual NUMA allocation
        // For now, fall back to regular aligned alloc
        const mem = self.allocator.alignedAlloc(64, count * @sizeOf(T));
        return @as([*]T, @ptrCast(@alignCast(mem.ptr)))[0..count];
    }

    pub fn free(self: *NumaAllocator, mem: []u8) void {
        self.allocator.free(mem);
    }
};

/// Shared memory buffer for inter-thread communication
pub const SharedBuffer = struct {
    ptr: [*]u8,
    len: usize,
    owner: bool,

    pub fn create(allocator: Allocator, size: usize, alignment: usize) !SharedBuffer {
        const mem = try allocator.alignedAlloc(alignment, size);
        return .{
            .ptr = @ptrCast(mem.ptr),
            .len = size,
            .owner = true,
        };
    }

    pub fn fromRaw(ptr: *u8, len: usize) SharedBuffer {
        return .{ .ptr = ptr, .len = len, .owner = false };
    }

    pub fn slice(self: SharedBuffer) []u8 {
        return self.ptr[0..self.len];
    }

    pub fn sliceAs(self: SharedBuffer, comptime T: type) []T {
        return @as([*]T, @ptrCast(@alignCast(self.ptr)))[0..(self.len / @sizeOf(T))];
    }

    pub fn deinit(self: *SharedBuffer, allocator: Allocator) void {
        if (self.owner) {
            allocator.free(self.slice());
        }
        self.ptr = undefined;
        self.len = 0;
        self.owner = false;
    }
};

/// Buffer pool for reusable aligned buffers
pub const BufferPool = struct {
    allocator: Allocator,
    buffers: std.ArrayList(SharedBuffer),
    buffer_size: usize,
    alignment: usize,

    pub fn init(allocator: Allocator, buffer_size: usize, alignment: usize) BufferPool {
        return .{
            .allocator = allocator,
            .buffers = std.ArrayList(SharedBuffer).init(allocator),
            .buffer_size = buffer_size,
            .alignment = alignment,
        };
    }

    pub fn deinit(self: *BufferPool) void {
        for (self.buffers.items)  | buf |  {
            buf.deinit(self.allocator);
        }
        self.buffers.deinit();
    }

    pub fn acquire(self: *BufferPool) !SharedBuffer {
        if (self.buffers.items.len > 0) {
            return self.buffers.pop();
        }
        return SharedBuffer.create(self.allocator, self.buffer_size, self.alignment);
    }

    pub fn release(self: *BufferPool, buf: SharedBuffer) void {
        _ = self.buffers.append(buf);
    }
};

/// Huge page allocation (Linux)
pub fn allocateHugePages(allocator: Allocator, size: usize, alignment: usize) ![]u8 {
    // Try to allocate with huge pages (2MB or 1GB)
    // This requires hugetlbfs or transparent huge pages
    // Fallback to regular aligned allocation
    return allocator.alignedAlloc(alignment, size);
}

/// Memory advice for kernel buffers
pub fn madviseSequential(ptr: *u8, len: usize) void {
    // TODO: Use posix_madvise with MADV_SEQUENTIAL
    _ = ptr;
    _ = len;
}

pub fn madviseRandom(ptr: *u8, len: usize) void {
    // TODO: Use posix_madvise with MADV_RANDOM
    _ = ptr;
    _ = len;
}

pub fn madviseWillNeed(ptr: *u8, len: usize) void {
    // TODO: Use posix_madvise with MADV_WILLNEED
    _ = ptr;
    _ = len;
}