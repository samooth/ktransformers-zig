// NUMA Topology Detection for ktransformers-zig
// Discovers NUMA node layout, CPU affinity, and memory topology
// via /sys/devices/system/node/ and /proc/cpuinfo
//
// Zig 0.16 note: all filesystem access goes through the std.Io interface
// (std.fs.cwd / File.readToEndAlloc are gone in this toolchain). The
// pattern matches runtime/cpu_detect.zig: a std.Io.Threaded instance,
// std.Io.Dir.cwd(), and readStreaming into a fixed buffer (sysfs
// attributes are tiny — a few hundred bytes max).

const std = @import("std");
const Allocator = std.mem.Allocator;

/// NUMA node information
pub const NumaNode = struct {
    id: usize,
    cpus: []usize, // CPU indices belonging to this node
    memory_total: usize, // Total memory in bytes
    memory_free: usize, // Free memory in bytes
    distance: []usize, // Distance to other nodes (SLIT table)
};

/// System NUMA topology
pub const NumaTopology = struct {
    nodes: []NumaNode,
    num_nodes: usize,
    num_cpus: usize,
    allocator: Allocator,

    pub fn detect(allocator: Allocator) !NumaTopology {
        var io = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        defer io.deinit();

        // Count NUMA nodes: probe /sys/devices/system/node/nodeN until
        // the open fails. A single-socket host has only node0.
        var num_nodes: usize = 0;
        while (true) : (num_nodes += 1) {
            var path_buf: [96]u8 = undefined;
            const path = std.fmt.bufPrintZ(&path_buf, "/sys/devices/system/node/node{d}", .{num_nodes}) catch break;
            const dir = std.Io.Dir.cwd();
            const f = dir.openFile(std.Io.Threaded.io(&io), path, .{}) catch break;
            f.close(std.Io.Threaded.io(&io));
        }

        if (num_nodes == 0) {
            // No sysfs NUMA info (container, non-NUMA kernel) — single node.
            num_nodes = 1;
        }

        const nodes = try allocator.alloc(NumaNode, num_nodes);
        errdefer allocator.free(nodes);

        const num_cpus = try countCpus(allocator, &io);

        for (0..num_nodes) |node_id| {
            nodes[node_id] = try detectNode(allocator, &io, node_id, num_nodes, num_cpus);
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
        self.* = undefined;
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

/// Read a small file fully into a fixed buffer; returns the trimmed
/// content. Returns null when the file can't be opened.
fn readSmallFile(io: *std.Io.Threaded, path: [:0]const u8, buf: []u8) ?[]u8 {
    const dir = std.Io.Dir.cwd();
    const f = dir.openFile(std.Io.Threaded.io(io), path, .{}) catch return null;
    defer f.close(std.Io.Threaded.io(io));
    // readStreaming reads until the buffer is full or EOF; returns the
    // number of bytes placed in the (single) iovec.
    const n = std.Io.File.readStreaming(f, std.Io.Threaded.io(io), &.{buf}) catch return null;
    return buf[0..n];
}

fn countCpus(allocator: Allocator, io: *std.Io.Threaded) !usize {
    var buf: [65536]u8 = undefined;
    const content = readSmallFile(io, "/proc/cpuinfo", &buf) orelse return 1;

    var max_cpu: usize = 0;
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "processor")) {
            if (std.mem.indexOf(u8, line, ":")) |colon| {
                const val = std.mem.trim(u8, line[colon + 1 ..], " \t");
                const cpu = std.fmt.parseInt(usize, val, 10) catch continue;
                if (cpu >= max_cpu) max_cpu = cpu + 1;
                _ = allocator;
            }
        }
    }
    return @max(max_cpu, 1);
}

fn detectNode(allocator: Allocator, io: *std.Io.Threaded, node_id: usize, num_nodes: usize, num_cpus: usize) !NumaNode {
    // Zig 0.16: ArrayList is unmanaged (.empty + append(allocator, x)).
    var cpus: std.ArrayList(usize) = .empty;
    errdefer cpus.deinit(allocator);

    // Read cpulist (e.g., "0-15,32-47")
    {
        var path_buf: [96]u8 = undefined;
        const cpulist_path = std.fmt.bufPrintZ(&path_buf, "/sys/devices/system/node/node{d}/cpulist", .{node_id}) catch unreachable;
        var cl_buf: [256]u8 = undefined;
        if (readSmallFile(io, cpulist_path, &cl_buf)) |cpulist| {
            const trimmed = std.mem.trim(u8, cpulist, " \t\n\r");
            var ranges = std.mem.splitScalar(u8, trimmed, ',');
            while (ranges.next()) |range| {
                const r = std.mem.trim(u8, range, " \t");
                if (std.mem.indexOf(u8, r, "-")) |dash| {
                    const start = std.fmt.parseInt(usize, r[0..dash], 10) catch continue;
                    const end = std.fmt.parseInt(usize, r[dash + 1 ..], 10) catch continue;
                    for (start..end + 1) |c| {
                        try cpus.append(allocator, c);
                    }
                } else if (r.len > 0) {
                    const cpu = std.fmt.parseInt(usize, r, 10) catch continue;
                    try cpus.append(allocator, cpu);
                }
            }
        } else if (node_id == 0) {
            // No sysfs cpulist (container): all CPUs on node 0.
            for (0..num_cpus) |c| {
                try cpus.append(allocator, c);
            }
        }
    }

    // Read memory info
    var memory_total: usize = 0;
    var memory_free: usize = 0;
    {
        var path_buf: [96]u8 = undefined;
        const meminfo_path = std.fmt.bufPrintZ(&path_buf, "/sys/devices/system/node/node{d}/meminfo", .{node_id}) catch unreachable;
        var mem_buf: [4096]u8 = undefined;
        if (readSmallFile(io, meminfo_path, &mem_buf)) |meminfo| {
            var mem_lines = std.mem.splitScalar(u8, meminfo, '\n');
            while (mem_lines.next()) |line| {
                if (std.mem.indexOf(u8, line, "MemTotal")) |_| {
                    if (std.mem.indexOf(u8, line, "kB")) |kb| {
                        const val_str = std.mem.trim(u8, line[0..kb], " \tMemTotal:");
                        memory_total = (std.fmt.parseInt(usize, val_str, 10) catch 0) * 1024;
                    }
                } else if (std.mem.indexOf(u8, line, "MemFree")) |_| {
                    if (std.mem.indexOf(u8, line, "kB")) |kb| {
                        const val_str = std.mem.trim(u8, line[0..kb], " \tMemFree:");
                        memory_free = (std.fmt.parseInt(usize, val_str, 10) catch 0) * 1024;
                    }
                }
            }
        }
    }

    // Read distance table (SLIT)
    var distance = try allocator.alloc(usize, num_nodes);
    errdefer allocator.free(distance);
    @memset(distance, 10); // Default distance
    {
        var path_buf: [96]u8 = undefined;
        const dist_path = std.fmt.bufPrintZ(&path_buf, "/sys/devices/system/node/node{d}/distance", .{node_id}) catch unreachable;
        var dist_buf: [512]u8 = undefined;
        if (readSmallFile(io, dist_path, &dist_buf)) |dist_content| {
            const trimmed = std.mem.trim(u8, dist_content, " \t\n\r");
            var vals = std.mem.splitScalar(u8, trimmed, ' ');
            var idx: usize = 0;
            while (vals.next()) |val| {
                const v = std.mem.trim(u8, val, " \t");
                if (v.len > 0 and idx < num_nodes) {
                    distance[idx] = std.fmt.parseInt(usize, v, 10) catch 10;
                    idx += 1;
                }
            }
        }
    }

    return NumaNode{
        .id = node_id,
        .cpus = try cpus.toOwnedSlice(allocator),
        .memory_total = memory_total,
        .memory_free = memory_free,
        .distance = distance,
    };
}
