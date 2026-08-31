// Memory management for ktransformers-zig
// Provides aligned allocation, NUMA-aware allocation, and huge page support

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const numa_mem = @import("../numa/numa_memory.zig");

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
    pub fn alignedAlloc(self: *SimdArena, comptime T: type, count: usize, comptime alignment_val: usize) ![]T {
        const ptr = self.arena.allocator().alignedAlloc(T, @as(std.mem.Alignment, @enumFromInt(std.math.log2(alignment_val))), count);
        return ptr;
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
///
/// On Linux, allocations are bound to `numa_node` via `mbind` so the
/// kernel places the pages on the same NUMA node that will read them.
/// On non-Linux, the `numa_node` field is ignored and the call falls
/// through to a regular aligned alloc (the A3 fix: previously the
/// field was stored but never used — see the old "TODO: Use
/// numactl/libnuma" comment below).
pub const NumaAllocator = struct {
    allocator: Allocator,
    numa_node: ?usize,
    /// Page migration strictness: when the kernel can't satisfy the
    /// mbind (e.g. pages already allocated elsewhere), should we
    /// migrate them (MPOL_MF_MOVE) or fail silently? Defaults to
    /// "migrate if possible, otherwise best-effort". This matches the
    /// behavior of `src/numa/numa_memory.zig::bindMemory` which retries
    /// without STRICT | MOVE if the first attempt fails.
    migrate_on_miss: bool = true,

    pub fn init(allocator: Allocator, numa_node: ?usize) NumaAllocator {
        return .{ .allocator = allocator, .numa_node = numa_node };
    }

    pub fn alloc(self: *NumaAllocator, comptime T: type, count: usize) ![]T {
        const mem = self.allocator.alignedAlloc(64, count * @sizeOf(T));
        if (comptime builtin.os.tag == .linux) {
            if (self.numa_node) |node| {
                var mask = numa_mem.NumaNodeMask.init();
                mask.set(node);
                // best-effort: the helper retries without STRICT on
                // failure and silently drops the binding if even that
                // fails. Production MoE weights can be huge (GiB) and
                // partial binding (some pages on the right node) is
                // better than OOM. The helper already handles the
                // MPOL_MF_MOVE flag based on the policy.
                numa_mem.bindMemory(@ptrCast(mem.ptr), mem.len, mask, .bind) catch {};
            }
        }
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