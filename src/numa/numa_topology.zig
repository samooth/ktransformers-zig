// NUMA Topology Detection for ktransformers-zig
// Discovers NUMA node layout, CPU affinity, and memory topology
// via /sys/devices/system/node/ and /proc/cpuinfo

const std = @import("std");
const Allocator = std.mem.Allocator;

/// NUMA node information
pub const NumaNode = struct {
    id: usize,
    cpus: []usize,           // CPU indices belonging to this node
    memory_total: usize,      // Total memory in bytes
    memory_free: usize,       // Free memory in bytes
    distance: []usize,        // Distance to other nodes (SLIT table)
};

/// System NUMA topology
pub const NumaTopology = struct {
    nodes: []NumaNode,
    num_nodes: usize,
    num_cpus: usize,
    allocator: Allocator,

    pub fn detect(allocator: Allocator) !NumaTopology {
        // Count NUMA nodes
        var num_nodes: usize = 0;
        while (true) : (num_nodes += 1) {
            const path = try std.fmt.allocPrint(allocator, "/sys/devices/system/node/node{d}", .{num_nodes});
            defer allocator.free(path);
            std.fs.cwd().access(path, .{}) catch break;
        }

        if (num_nodes == 0) {
            // No NUMA - single node system
            num_nodes = 1;
        }

        const nodes = try allocator.alloc(NumaNode, num_nodes);
        errdefer allocator.free(nodes);

        const num_cpus = try countCpus(allocator);

        for (0..num_nodes) |node_id| {
            nodes[node_id] = try detectNode(allocator, node_id, num_nodes, num_cpus);
        }

        return NumaTopology{
            .nodes = nodes,
            .num_nodes = num_nodes,
            .num_cpus = num_cpus,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *NumaTopology) void {
        for (self.nodes) |*node| {
            self.allocator.free(node.cpus);
            self.allocator.free(node.distance);
        }
        self.allocator.free(self.nodes);
    }

    /// Get the NUMA node for a given CPU
    pub fn nodeForCpu(self: *const NumaTopology, cpu: usize) ?usize {
        for (self.nodes, 0..) |node, node_id| {
            for (node.cpus) |c| {
                if (c == cpu) return node_id;
            }
        }
        return null;
    }

    /// Get preferred NUMA node for memory allocation based on CPU
    pub fn preferredNodeForCpu(self: *const NumaTopology, cpu: usize) usize {
        return self.nodeForCpu(cpu) orelse 0;
    }

    /// Get CPUs for a NUMA node
    pub fn cpusForNode(self: *const NumaTopology, node_id: usize) []const usize {
        if (node_id >= self.num_nodes) return &.{};
        return self.nodes[node_id].cpus;
    }

    /// Print topology for debugging
    pub fn print(self: *const NumaTopology) void {
        std.debug.print("NUMA Topology: {d} nodes, {d} CPUs\n", .{ self.num_nodes, self.num_cpus });
        for (self.nodes) |node| {
            std.debug.print("  Node {d}: {d} CPUs, {d}MB total, {d}MB free\n", .{
                node.id,
                node.cpus.len,
                node.memory_total / (1024 * 1024),
                node.memory_free / (1024 * 1024),
            });
            std.debug.print("    CPUs: ", .{});
            for (node.cpus) |cpu| {
                std.debug.print("{d} ", .{cpu});
            }
            std.debug.print("\n", .{});
            std.debug.print("    Distances: ", .{});
            for (node.distance) |d| {
                std.debug.print("{d} ", .{d});
            }
            std.debug.print("\n", .{});
        }
    }
};

fn countCpus(allocator: Allocator) !usize {
    const file = try std.fs.cwd().openFile("/proc/cpuinfo", .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 1_000_000);
    defer allocator.free(content);

    var max_cpu: usize = 0;
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "processor")) {
            if (std.mem.indexOf(u8, line, ":")) |colon| {
                const val = std.mem.trim(u8, line[colon + 1 ..], " \t");
                const cpu = try std.fmt.parseInt(usize, val, 10);
                if (cpu >= max_cpu) max_cpu = cpu + 1;
            }
        }
    }
    return max_cpu;
}

fn detectNode(allocator: Allocator, node_id: usize, num_nodes: usize, num_cpus: usize) !NumaNode {
    // Read cpulist
    const cpulist_path = try std.fmt.allocPrint(allocator, "/sys/devices/system/node/node{d}/cpulist", .{node_id});
    defer allocator.free(cpulist_path);

    var cpus = std.ArrayList(usize).init(allocator);
    errdefer cpus.deinit();

    const cpulist_file = std.fs.cwd().openFile(cpulist_path, .{}) catch {
        // If file doesn't exist, assume all CPUs on node 0
        if (node_id == 0) {
            for (0..num_cpus) |c| {
                try cpus.append(c);
            }
        }
        return NumaNode{
            .id = node_id,
            .cpus = try cpus.toOwnedSlice(),
            .memory_total = 0,
            .memory_free = 0,
            .distance = try allocator.alloc(usize, num_nodes),
        };
    };
    defer cpulist_file.close();

    const cpulist = try cpulist_file.readToEndAlloc(allocator, 4096);
    defer allocator.free(cpulist);

    // Parse cpulist (e.g., "0-15,32-47")
    var ranges = std.mem.splitScalar(u8, std.mem.trim(u8, cpulist, "\n"), ',');
    while (ranges.next()) |range| {
        const trimmed = std.mem.trim(u8, range, " \t");
        if (std.mem.indexOf(u8, trimmed, "-")) |dash| {
            const start = try std.fmt.parseInt(usize, trimmed[0..dash], 10);
            const end = try std.fmt.parseInt(usize, trimmed[dash + 1 ..], 10);
            for (start..end + 1) |c| {
                try cpus.append(c);
            }
        } else if (trimmed.len > 0) {
            const cpu = try std.fmt.parseInt(usize, trimmed, 10);
            try cpus.append(cpu);
        }
    }

    // Read memory info
    const memtotal_path = try std.fmt.allocPrint(allocator, "/sys/devices/system/node/node{d}/meminfo", .{node_id});
    defer allocator.free(memtotal_path);

    var memory_total: usize = 0;
    var memory_free: usize = 0;

    const meminfo_file = std.fs.cwd().openFile(memtotal_path, .{}) catch null;
    if (meminfo_file) |f| {
        defer f.close();
        const meminfo = try f.readToEndAlloc(allocator, 4096);
        defer allocator.free(meminfo);

        var mem_lines = std.mem.splitScalar(u8, meminfo, '\n');
        while (mem_lines.next()) |line| {
            if (std.mem.indexOf(u8, line, "MemTotal")) |_| {
                if (std.mem.indexOf(u8, line, "kB")) |kb| {
                    const val_str = std.mem.trim(u8, line[0..kb], " \tMemTotal:");
                    memory_total = (try std.fmt.parseInt(usize, val_str, 10)) * 1024;
                }
            } else if (std.mem.indexOf(u8, line, "MemFree")) |_| {
                if (std.mem.indexOf(u8, line, "kB")) |kb| {
                    const val_str = std.mem.trim(u8, line[0..kb], " \tMemFree:");
                    memory_free = (try std.fmt.parseInt(usize, val_str, 10)) * 1024;
                }
            }
        }
    }

    // Read distance table (SLIT)
    const distance_path = try std.fmt.allocPrint(allocator, "/sys/devices/system/node/node{d}/distance", .{node_id});
    defer allocator.free(distance_path);

    var distance = try allocator.alloc(usize, num_nodes);
    @memset(distance, 10); // Default distance

    const dist_file = std.fs.cwd().openFile(distance_path, .{}) catch null;
    if (dist_file) |f| {
        defer f.close();
        const dist_content = try f.readToEndAlloc(allocator, 4096);
        defer allocator.free(dist_content);

        var vals = std.mem.splitScalar(u8, std.mem.trim(u8, dist_content, "\n"), ' ');
        var idx: usize = 0;
        while (vals.next()) |val| {
            const trimmed = std.mem.trim(u8, val, " \t");
            if (trimmed.len > 0 and idx < num_nodes) {
                distance[idx] = try std.fmt.parseInt(usize, trimmed, 10);
                idx += 1;
            }
        }
    }

    return NumaNode{
        .id = node_id,
        .cpus = try cpus.toOwnedSlice(),
        .memory_total = memory_total,
        .memory_free = memory_free,
        .distance = distance,
    };
}
