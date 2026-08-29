// NUMA Memory Binding for ktransformers-zig
// Root module re-exporting all NUMA functionality

pub const topology = @import("numa_topology.zig");
pub const memory = @import("numa_memory.zig");
pub const worker = @import("numa_worker.zig");

// Re-export commonly used types
pub const NumaTopology = topology.NumaTopology;
pub const NumaNode = topology.NumaNode;
pub const NumaNodeMask = memory.NumaNodeMask;
pub const NumaPolicy = memory.NumaPolicy;
pub const CpuSet = memory.CpuSet;
pub const NumaWorkerPool = worker.NumaWorkerPool;
pub const NumaWorkerConfig = worker.NumaWorkerConfig;

/// Quick-start: detect topology and print it
pub fn detectAndPrint() !void {
    const allocator = std.heap.page_allocator;
    var topo = try NumaTopology.detect(allocator);
    defer topo.deinit();
    topo.print();
}

const std = @import("std");
