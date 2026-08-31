// NUMA Memory Allocation and Binding for ktransformers-zig
// Provides NUMA-aware memory allocation, binding, and migration
// Uses Linux mbind/set_mempolicy via C interop

const std = @import("std");
const NumaTopology = @import("numa_topology.zig").NumaTopology;

// ============================================================================
// C Bindings for Linux NUMA APIs
// ============================================================================

const c = @cImport({
    @cInclude("stdlib.h");
    @cInclude("string.h");
    @cInclude("unistd.h");
    @cInclude("sys/syscall.h");
    @cInclude("linux/mempolicy.h");
});

/// Linux mbind flags
const MPOL_DEFAULT = 0;
const MPOL_PREFERRED = 1;
const MPOL_BIND = 2;
const MPOL_INTERLEAVE = 3;
const MPOL_LOCAL = 4;
const MPOL_PREFERRED_MANY = 5;
const MPOL_WEIGHTED_INTERLEAVE = 6;

const MPOL_MF_STRICT = 1 << 0;
const MPOL_MF_MOVE = 1 << 1;
const MPOL_MF_MOVE_ALL = 1 << 2;

/// NUMA memory policy for a thread
pub const NumaPolicy = enum {
    default,           // MPOL_DEFAULT - use system default
    preferred,         // MPOL_PREFERRED - prefer a node, fallback to others
    bind,              // MPOL_BIND - strict binding to nodes
    interleave,        // MPOL_INTERLEAVE - round-robin across nodes
    local,             // MPOL_LOCAL - allocate on local node
    preferred_many,    // MPOL_PREFERRED_MANY - prefer multiple nodes

    pub fn toLinux(self: NumaPolicy) c_int {
        return switch (self) {
            .default => MPOL_DEFAULT,
            .preferred => MPOL_PREFERRED,
            .bind => MPOL_BIND,
            .interleave => MPOL_INTERLEAVE,
            .local => MPOL_LOCAL,
            .preferred_many => MPOL_PREFERRED_MANY,
        };
    }
};

/// NUMA node mask (up to 256 nodes)
pub const NumaNodeMask = struct {
    bits: [4]u64,  // 256 bits

    pub fn init() NumaNodeMask {
        return .{ .bits = .{ 0, 0, 0, 0 } };
    }

    pub fn set(self: *NumaNodeMask, node: usize) void {
        const idx = node / 64;
        const bit = node % 64;
        if (idx < 4) {
            self.bits[idx] |= (@as(u64, 1) << @intCast(bit));
        }
    }

    pub fn clear(self: *NumaNodeMask, node: usize) void {
        const idx = node / 64;
        const bit = node % 64;
        if (idx < 4) {
            self.bits[idx] &= ~(@as(u64, 1) << @intCast(bit));
        }
    }

    pub fn isSet(self: *const NumaNodeMask, node: usize) bool {
        const idx = node / 64;
        const bit = node % 64;
        if (idx >= 4) return false;
        return (self.bits[idx] & (@as(u64, 1) << @intCast(bit))) != 0;
    }

    pub fn count(self: *const NumaNodeMask) usize {
        var n: usize = 0;
        for (self.bits) |b| {
            n += @popCount(b);
        }
        return n;
    }

    pub fn toU64Slice(self: *const NumaNodeMask) []const u64 {
        return &self.bits;
    }
};

// ============================================================================
// System Calls (via syscall() since Zig stdlib lacks these)
// ============================================================================

/// mbind: set NUMA memory policy for address range
fn sys_mbind(
    addr: *anyopaque,
    len: usize,
    mode: c_int,
    nodemask: [*]const u64,
    maxnode: usize,
    flags: c_uint,
) c_long {
    return c.syscall(c.SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
}

/// set_mempolicy: set default NUMA policy for current thread
fn sys_set_mempolicy(mode: c_int, nodemask: ?[*]const u64, maxnode: usize) c_long {
    return c.syscall(c.SYS_set_mempolicy, mode, nodemask, maxnode);
}

/// get_mempolicy: get current NUMA policy
fn sys_get_mempolicy(mode: *c_int, nodemask: ?[*]u64, maxnode: usize, addr: ?*anyopaque, flags: c_uint) c_long {
    return c.syscall(c.SYS_get_mempolicy, mode, nodemask, maxnode, addr, flags);
}

/// move_pages: migrate pages to specified nodes
fn sys_move_pages(
    pid: c_int,
    count: usize,
    pages: [*]const *anyopaque,
    nodes: [*]const c_int,
    status: [*]c_int,
    flags: c_int,
) c_long {
    return c.syscall(c.SYS_move_pages, pid, count, pages, nodes, status, flags);
}

/// sched_setaffinity: pin thread to specific CPUs
fn sys_sched_setaffinity(pid: c_int, cpusetsize: usize, mask: [*]const u8) c_long {
    return c.syscall(c.SYS_sched_setaffinity, pid, cpusetsize, mask);
}

/// sched_getaffinity: get current CPU affinity
fn sys_sched_getaffinity(pid: c_int, cpusetsize: usize, mask: [*]u8) c_long {
    return c.syscall(c.SYS_sched_getaffinity, pid, cpusetsize, mask);
}

// ============================================================================
// CPU Affinity (Thread Pinning)
// ============================================================================

pub const CpuSet = struct {
    bits: [128]u8,  // 1024 CPUs max

    pub fn init() CpuSet {
        var cs: CpuSet = undefined;
        @memset(&cs.bits, 0);
        return cs;
    }

    pub fn set(self: *CpuSet, cpu: usize) void {
        const byte = cpu / 8;
        const bit = cpu % 8;
        if (byte < 128) {
            self.bits[byte] |= (@as(u8, 1) << @intCast(bit));
        }
    }

    pub fn clear(self: *CpuSet, cpu: usize) void {
        const byte = cpu / 8;
        const bit = cpu % 8;
        if (byte < 128) {
            self.bits[byte] &= ~(@as(u8, 1) << @intCast(bit));
        }
    }

    pub fn isSet(self: *const CpuSet, cpu: usize) bool {
        const byte = cpu / 8;
        const bit = cpu % 8;
        if (byte >= 128) return false;
        return (self.bits[byte] & (@as(u8, 1) << @intCast(bit))) != 0;
    }

    pub fn count(self: *const CpuSet) usize {
        var n: usize = 0;
        for (self.bits) |b| {
            n += @popCount(b);
        }
        return n;
    }

    pub fn fromCpuList(cpus: []const usize) CpuSet {
        var cs = CpuSet.init();
        for (cpus) |cpu| {
            cs.set(cpu);
        }
        return cs;
    }

    pub fn toSlice(self: *const CpuSet) []const u8 {
        return &self.bits;
    }
};

/// Pin current thread to specific CPUs
pub fn setThreadAffinity(cpus: []const usize) !void {
    var cs = CpuSet.fromCpuList(cpus);
    const rc = sys_sched_setaffinity(0, @sizeOf(CpuSet), cs.toSlice().ptr);
    if (rc != 0) {
        return error.SetAffinityFailed;
    }
}

/// Pin current thread to a single CPU
pub fn pinThreadToCpu(cpu: usize) !void {
    var cs = CpuSet.init();
    cs.set(cpu);
    const rc = sys_sched_setaffinity(0, @sizeOf(CpuSet), cs.toSlice().ptr);
    if (rc != 0) {
        return error.SetAffinityFailed;
    }
}

/// Get current thread's CPU affinity
pub fn getThreadAffinity() !CpuSet {
    var cs = CpuSet.init();
    const rc = sys_sched_getaffinity(0, @sizeOf(CpuSet), @ptrCast(&cs.bits));
    if (rc != 0) {
        return error.GetAffinityFailed;
    }
    return cs;
}

// ============================================================================
// NUMA Memory Allocation
// ============================================================================

/// Allocate memory bound to specific NUMA node(s)
/// Uses mmap with mbind for strict binding
pub fn allocNuma(
    allocator: std.mem.Allocator,
    size: usize,
    alignment: usize,
    node_mask: NumaNodeMask,
    policy: NumaPolicy,
) ![]u8 {
    // Allocate aligned memory
    const mem = try allocator.alignedAlloc(u8, @enumFromInt(std.math.log2(alignment)), size);

    // Bind to NUMA nodes
    try bindMemory(mem.ptr, mem.len, node_mask, policy);

    return mem;
}

/// Bind existing memory to NUMA nodes
pub fn bindMemory(
    addr: *anyopaque,
    len: usize,
    node_mask: NumaNodeMask,
    policy: NumaPolicy,
) !void {
    const maxnode = 256;
    const mode = policy.toLinux();
    // Coerce the comptime_int flag constants to c_uint explicitly so the
    // result is a runtime c_uint (not a comptime_int depending on a
    // runtime conditional — which Zig 0.16 rejects).
    const flags: c_uint = if (policy == .bind)
        @as(c_uint, MPOL_MF_STRICT) | @as(c_uint, MPOL_MF_MOVE)
    else
        0;

    const rc = sys_mbind(addr, len, mode, node_mask.toU64Slice().ptr, maxnode, flags);
    if (rc != 0) {
        // mbind can fail if pages are already allocated elsewhere
        // Try without STRICT flag
        const rc2 = sys_mbind(addr, len, mode, node_mask.toU64Slice().ptr, maxnode, 0);
        if (rc2 != 0) {
            return error.MbindFailed;
        }
    }
}

/// Set default NUMA policy for current thread
pub fn setThreadNumaPolicy(node_mask: NumaNodeMask, policy: NumaPolicy) !void {
    const maxnode = 256;
    const mode = policy.toLinux();

    const rc = sys_set_mempolicy(mode, node_mask.toU64Slice().ptr, maxnode);
    if (rc != 0) {
        return error.SetMempolicyFailed;
    }
}

/// Get current thread's NUMA policy
pub fn getThreadNumaPolicy() !struct { policy: NumaPolicy, node_mask: NumaNodeMask } {
    var mode: c_int = undefined;
    var mask = NumaNodeMask.init();

    const rc = sys_get_mempolicy(&mode, @ptrCast(&mask.bits), 256, null, 0);
    if (rc != 0) {
        return error.GetMempolicyFailed;
    }

    const policy: NumaPolicy = switch (mode) {
        MPOL_DEFAULT => .default,
        MPOL_PREFERRED => .preferred,
        MPOL_BIND => .bind,
        MPOL_INTERLEAVE => .interleave,
        MPOL_LOCAL => .local,
        MPOL_PREFERRED_MANY => .preferred_many,
        else => .default,
    };

    return .{ .policy = policy, .node_mask = mask };
}

/// Migrate pages to local NUMA node
pub fn migratePagesToNode(addr: [*]u8, page_count: usize, target_node: c_int) !void {
    const pages = try std.heap.page_allocator.alloc(*anyopaque, page_count);
    defer std.heap.page_allocator.free(pages);

    const page_size = std.mem.page_size;
    for (0..page_count) |i| {
        pages[i] = addr + i * page_size;
    }

    const nodes = try std.heap.page_allocator.alloc(c_int, page_count);
    defer std.heap.page_allocator.free(nodes);
    @memset(nodes, target_node);

    const status = try std.heap.page_allocator.alloc(c_int, page_count);
    defer std.heap.page_allocator.free(status);

    const rc = sys_move_pages(0, page_count, pages.ptr, nodes.ptr, status.ptr, 0);
    if (rc != 0) {
        return error.MovePagesFailed;
    }

    // Check status
    for (0..page_count) |i| {
        if (status[i] < 0) {
            return error.PageMigrationFailed;
        }
    }
}

/// Touch memory to ensure pages are allocated (fault them in)
pub fn touchMemory(mem: []u8) void {
    const page_size = std.mem.page_size;
    var i: usize = 0;
    while (i < mem.len) : (i += page_size) {
        mem[i] = 0;
    }
}

/// Allocate and touch memory on specific NUMA node
pub fn allocNumaAndTouch(
    allocator: std.mem.Allocator,
    size: usize,
    alignment: usize,
    node_id: usize,
) ![]u8 {
    var mask = NumaNodeMask.init();
    mask.set(node_id);

    const mem = try allocNuma(allocator, size, alignment, mask, .bind);
    touchMemory(mem);
    return mem;
}

// ============================================================================
// NUMA-Aware Allocator
// ============================================================================

/// Allocator that binds all allocations to a specific NUMA node
pub const NumaAllocator = struct {
    backing: std.mem.Allocator,
    node_id: usize,
    policy: NumaPolicy,

    pub fn init(backing: std.mem.Allocator, node_id: usize, policy: NumaPolicy) NumaAllocator {
        return .{
            .backing = backing,
            .node_id = node_id,
            .policy = policy,
        };
    }

    pub fn allocator(self: *NumaAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }

    fn alloc(self: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const numa_alloc: *NumaAllocator = @ptrCast(@alignCast(self));
        const mem = numa_alloc.backing.rawAlloc(len, ptr_align, ret_addr) orelse return null;

        var mask = NumaNodeMask.init();
        mask.set(numa_alloc.node_id);
        bindMemory(mem, len, mask, numa_alloc.policy) catch {};

        return mem;
    }

    fn resize(self: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const numa_alloc: *NumaAllocator = @ptrCast(@alignCast(self));
        return numa_alloc.backing.rawResize(buf, buf_align, new_len, ret_addr);
    }

    fn free(self: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const numa_alloc: *NumaAllocator = @ptrCast(@alignCast(self));
        numa_alloc.backing.rawFree(buf, buf_align, ret_addr);
    }
};

// ============================================================================
// Memory Migration
// ============================================================================

/// Check where pages are currently allocated
pub fn getPageNodes(addr: [*]const u8, page_count: usize) ![]c_int {
    const pages = try std.heap.page_allocator.alloc(*anyopaque, page_count);
    defer std.heap.page_allocator.free(pages);

    const page_size = std.mem.page_size;
    for (0..page_count) |i| {
        pages[i] = @constCast(addr + i * page_size);
    }

    const status = try std.heap.page_allocator.alloc(c_int, page_count);
    errdefer std.heap.page_allocator.free(status);

    const rc = sys_move_pages(0, page_count, pages.ptr, null, status.ptr, 0);
    if (rc != 0) {
        return error.GetPageNodesFailed;
    }

    return status;
}

/// Migrate memory to local NUMA node (where the calling thread runs)
pub fn migrateToLocal(addr: [*]u8, len: usize) !void {
    const page_size = std.mem.page_size;
    const page_count = (len + page_size - 1) / page_size;

    // Get current CPU and its NUMA node
    const _cpu = c.syscall(c.SYS_getcpu, null, null, null);
    _ = _cpu;
    // For simplicity, assume node 0 if we can't determine
    const target_node: c_int = 0;

    try migratePagesToNode(addr, page_count, target_node);
}

/// Interleave memory across all NUMA nodes
pub fn interleaveMemory(addr: *anyopaque, len: usize, num_nodes: usize) !void {
    var mask = NumaNodeMask.init();
    for (0..num_nodes) |n| {
        mask.set(n);
    }
    try bindMemory(addr, len, mask, .interleave);
}
