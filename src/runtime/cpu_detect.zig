// CPU detection for ktransformers-zig
// Ported from ktransformers kt-kernel/python/_cpu_detect.py

const std = @import("std");
const Allocator = std.mem.Allocator;

/// CPU vendor identification
pub const Vendor = enum {
    intel,
    amd,
    arm,
    unknown,
};

/// CPU feature flags
pub const Features = struct {
    avx: bool = false,
    avx2: bool = false,
    fma: bool = false,
    f16c: bool = false,
    avx512f: bool = false,
    avx512bw: bool = false,
    avx512dq: bool = false,
    avx512vl: bool = false,
    avx512vnni: bool = false,
    avx512bf16: bool = false,
    avx512vbmi: bool = false,
    avx512vpopcntdq: bool = false,
    amx_bf16: bool = false,
    amx_int8: bool = false,
    amx_tile: bool = false,
    sve: bool = false,
    sve2: bool = false,
    neon: bool = false,
};

/// CPU information
pub const CpuInfo = struct {
    vendor: Vendor,
    arch: []const u8,
    features: Features,
    flags: std.ArrayList([]const u8),
    model_name: []const u8,
    cpu_family: u32,
    model: u32,
    stepping: u32,
    cache_line_size: usize = 64,
    numa_nodes: usize = 1,
};

/// Detect CPU features and vendor
pub fn detectCpu(allocator: Allocator) !CpuInfo {
    const os = std.Target.current.os.tag;
    switch (os) {
        .linux => return detectCpuLinux(allocator),
        .macos => return detectCpuDarwin(allocator),
        .windows => return detectCpuWindows(allocator),
        else => return CpuInfo{
            .vendor = .unknown,
            .arch = std.Target.current.cpu.arch.name() orelse "unknown",
            .features = Features{},
            .flags = std.ArrayList([]const u8).init(allocator),
            .model_name = "unknown",
            .cpu_family = 0,
            .model = 0,
            .stepping = 0,
        },
    }
}

fn detectCpuLinux(allocator: Allocator) !CpuInfo {
    const file = try std.fs.cwd().openFile("/proc/cpuinfo", .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 1_000_000);
    defer allocator.free(content);

    const lines = std.mem.split(u8, content, "\n");

    const vendor: Vendor = .unknown;
    const model_name: []const u8 = "unknown";
    const cpu_family: u32 = 0;
    const model: u32 = 0;
    const stepping: u32 = 0;
    const flags = std.StringArray.init(allocator);
    const features = Features{};

    for (lines) |line| {
        if (std.mem.startsWith(u8, line, "vendor_id")) {
            const parts = std.mem.split(u8, line, ":");
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    const v = std.mem.trim(u8, value, " \t");
                    if (std.mem.eql(u8, v, "GenuineIntel")) {
                        vendor = .intel;
                    } else if (std.mem.eql(u8, v, "AuthenticAMD")) {
                        vendor = .amd;
                    }
                }
            }
        } else if (std.mem.startsWith(u8, line, "model name")) {
            const parts = std.mem.split(u8, line, ":");
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    model_name = std.mem.trim(u8, value, " \t");
                }
            }
        } else if (std.mem.startsWith(u8, line, "cpu family")) {
            const parts = std.mem.split(u8, line, ":");
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    cpu_family = std.fmt.parseInt(u32, std.mem.trim(u8, value, " \t"), 10) catch 0;
                }
            }
        } else if (std.mem.startsWith(u8, line, "model")) {
            const parts = std.mem.split(u8, line, ":");
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    model = std.fmt.parseInt(u32, std.mem.trim(u8, value, " \t"), 10) catch 0;
                }
            }
        } else if (std.mem.startsWith(u8, line, "stepping")) {
            const parts = std.mem.split(u8, line, ":");
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    stepping = std.fmt.parseInt(u32, std.mem.trim(u8, value, " \t"), 10) catch 0;
                }
            }
        } else if (std.mem.startsWith(u8, line, "flags") || std.mem.startsWith(u8, line, "Features")) {
            const parts = std.mem.split(u8, line, ":");
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    const flag_str = std.mem.trim(u8, value, " \t");
                    for (std.mem.split(u8, flag_str, " ")) |flag| {
                        if (flag.len > 0) {
                            _ = flags.append(allocator, flag) catch {};
                            parseFlag(flag, &features);
                        }
                    }
                }
            }
        }
    }

    // Detect ARM from model name if vendor unknown
    if (vendor == .unknown) {
        const lower = std.ascii.lower(model_name);
        if ((std.mem.indexOf(u8, lower, "aarch64") != null) or
            (std.mem.indexOf(u8, lower, "armv8") != null) or
            (std.mem.indexOf(u8, lower, "arm cortex") != null) or
            (std.mem.indexOf(u8, lower, "kunpeng") != null) or
            (std.mem.indexOf(u8, lower, "kirin") != null) or
            (std.mem.indexOf(u8, lower, "huawei") != null)) {
            vendor = .arm;
        }
    }

    // Get architecture
    const arch = switch (std.Target.current.cpu.arch) {
        .x86_64 => "x86_64",
        .aarch64 => "aarch64",
        .arm => "arm",
        else => "unknown",
    };

    return CpuInfo{
        .vendor = vendor,
        .arch = arch,
        .features = features,
        .flags = flags,
        .model_name = model_name,
        .cpu_family = cpu_family,
        .model = model,
        .stepping = stepping,
    };
}

fn detectCpuDarwin(allocator: Allocator) !CpuInfo {
    const arch = switch (std.Target.current.cpu.arch) {
        .aarch64 => "aarch64",
        .x86_64 => "x86_64",
        else => "unknown",
    };

    const vendor: Vendor = if (std.mem.eql(u8, arch, "aarch64")) .arm else .intel;
    const features = Features{};

    if (std.mem.eql(u8, arch, "aarch64")) {
        features.neon = true;
        features.sve = true; // Apple Silicon has SVE-like features
    }

    return CpuInfo{
        .vendor = vendor,
        .arch = arch,
        .features = features,
        .flags = std.StringArray.init(allocator),
        .model_name = "Apple Silicon",
        .cpu_family = 0,
        .model = 0,
        .stepping = 0,
    };
}

fn detectCpuWindows(allocator: Allocator) !CpuInfo {
    const arch = switch (std.Target.current.cpu.arch) {
        .x86_64 => "x86_64",
        .aarch64 => "aarch64",
        .arm => "arm",
        else => "unknown",
    };

    const vendor: Vendor = if (std.mem.eql(u8, arch, "aarch64")) .arm else .unknown;
    const features = Features{};

    return CpuInfo{
        .vendor = vendor,
        .arch = arch,
        .features = features,
        .flags = std.StringArray.init(allocator),
        .model_name = "Windows CPU",
        .cpu_family = 0,
        .model = 0,
        .stepping = 0,
    };
}

fn parseFlag(flag: []const u8, features: *Features) void {
    const f = std.ascii.lower(flag);
    if (std.mem.eql(u8, f, "avx")) {
        features.avx = true;
    } else if (std.mem.eql(u8, f, "avx2")) {
        features.avx2 = true;
    }
    else if (std.mem.eql(u8, f, "fma")) {
        features.fma = true;
    }
    else if (std.mem.eql(u8, f, "f16c")) {
        features.f16c = true;
    }
    else if (std.mem.eql(u8, f, "avx512f")) {
        features.avx512f = true;
    }
    else if (std.mem.eql(u8, f, "avx512bw")) {
        features.avx512bw = true;
    }
    else if (std.mem.eql(u8, f, "avx512dq")) {
        features.avx512dq = true;
    }
    else if (std.mem.eql(u8, f, "avx512vl")) {
        features.avx512vl = true;
    }
    else if (std.mem.eql(u8, f, "avx512vnni") || std.mem.eql(u8, f, "avx512_vnni")) {
        features.avx512vnni = true;
    }
    else if (std.mem.eql(u8, f, "avx512bf16") || std.mem.eql(u8, f, "avx512_bf16")) {
        features.avx512bf16 = true;
    }
    else if (std.mem.eql(u8, f, "avx512vbmi") || std.mem.eql(u8, f, "avx512_vbmi")) {
        features.avx512vbmi = true;
    }
    else if (std.mem.eql(u8, f, "avx512vpopcntdq") || std.mem.eql(u8, f, "avx512_vpopcntdq")) {
        features.avx512vpopcntdq = true;
    }
    else if (std.mem.eql(u8, f, "amx_bf16") || std.mem.eql(u8, f, "amx-bf16")) {
        features.amx_bf16 = true;
    }
    else if (std.mem.eql(u8, f, "amx_int8") || std.mem.eql(u8, f, "amx-int8")) {
        features.amx_int8 = true;
    }
    else if (std.mem.eql(u8, f, "amx_tile") || std.mem.eql(u8, f, "amx-tile")) {
        features.amx_tile = true;
    }
    else if (std.mem.eql(u8, f, "sve")) {
        features.sve = true;
    }
    else if (std.mem.eql(u8, f, "sve2")) {
        features.sve2 = true;
    }
    else if (std.mem.eql(u8, f, "neon")) {
        features.neon = true;
    }
}

/// Select best CPU variant for current hardware
pub fn selectBestVariant(cpu: CpuInfo) []const u8 {
    const f = cpu.features;

    // Check for AMX support (Sapphire Rapids+)
    if (f.amx_bf16  and  f.amx_int8  and  f.amx_tile) {
        return "amx";
    }

    // Check for AVX512 BF16 (Ice Lake Server, Zen 4+)
    if (f.avx512bf16  and  f.avx512vbmi  and  f.avx512vnni) {
        return "avx512_bf16";
    }

    // Check for AVX512 VBMI (Ice Lake Client)
    if (f.avx512vbmi  and  f.avx512vnni) {
        return "avx512_vbmi";
    }

    // Check for AVX512 VNNI (Cascade Lake+)
    if (f.avx512vnni) {
        return "avx512_vnni";
    }

    // Check for base AVX512 (Skylake-X+)
    if (f.avx512f  and  f.avx512bw  and  f.avx512dq  and  f.avx512vl) {
        return "avx512_base";
    }

    // Default to AVX2 (Haswell+)
    if (f.avx2) {
        return "avx2";
    }

    // Fallback
    return "avx2";
}

/// Print CPU info for debugging
pub fn printCpuInfo(cpu: CpuInfo) void {
    std.debug.print("CPU Info:\n", .{});
    std.debug.print("  Vendor: {s}\n", .{ @tagName(cpu.vendor) });
    std.debug.print("  Arch: {s}\n", .{ cpu.arch });
    std.debug.print("  Model: {s}\n", .{ cpu.model_name });
    std.debug.print("  Family: {d}, Model: {d}, Stepping: {d}\n", .{ cpu.cpu_family, cpu.model, cpu.stepping });
    std.debug.print("  Features:\n", .{});
    std.debug.print("    AVX: {}\n", .{ cpu.features.avx });
    std.debug.print("    AVX2: {}\n", .{ cpu.features.avx2 });
    std.debug.print("    FMA: {}\n", .{ cpu.features.fma });
    std.debug.print("    F16C: {}\n", .{ cpu.features.f16c });
    std.debug.print("    AVX512F: {}\n", .{ cpu.features.avx512f });
    std.debug.print("    AVX512BW: {}\n", .{ cpu.features.avx512bw });
    std.debug.print("    AVX512DQ: {}\n", .{ cpu.features.avx512dq });
    std.debug.print("    AVX512VL: {}\n", .{ cpu.features.avx512vl });
    std.debug.print("    AVX512VNNI: {}\n", .{ cpu.features.avx512vnni });
    std.debug.print("    AVX512BF16: {}\n", .{ cpu.features.avx512bf16 });
    std.debug.print("    AVX512VBMI: {}\n", .{ cpu.features.avx512vbmi });
    std.debug.print("    AMX BF16: {}\n", .{ cpu.features.amx_bf16 });
    std.debug.print("    AMX INT8: {}\n", .{ cpu.features.amx_int8 });
    std.debug.print("    AMX TILE: {}\n", .{ cpu.features.amx_tile });
    std.debug.print("    SVE: {}\n", .{ cpu.features.sve });
    std.debug.print("    NEON: {}\n", .{ cpu.features.neon });
    std.debug.print("  Best variant: {s}\n", .{ selectBestVariant(cpu) });
}

/// Get feature summary string
pub fn getFeatureSummary(cpu: CpuInfo) []const u8 {
    _ = cpu.features;
    const buf: [256]u8 = undefined;
    const writer = std.io.fixedBufferStream(&buf).writer();
    _ = writer;

    // Simplified - just return variant name
    return selectBestVariant(cpu);
}