// CPU detection for ktransformers-zig
// Ported from ktransformers kt-kernel/python/_cpu_detect.py

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const avx512vnni_str = "avx512vnni";
const avx512_vnni_str = "avx512_vnni";

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
    flags: [16]u8,
    model_name: []u8, // owned, freed by deinit
    cpu_family: u32,
    model: u32,
    stepping: u32,
    cache_line_size: usize = 64,
    numa_nodes: usize = 1,

    // A4: cache hierarchy sizes in BYTES, detected from
    // /sys/devices/system/cpu/cpu0/cache/index*/ on Linux. Defaults
    // are conservative fallbacks (32K L1d / 512K L2 / 16M L3) used
    // when detection is unavailable (non-Linux, sysfs hidden, VMs).
    l1d_bytes: usize = 32 * 1024,
    l1i_bytes: usize = 32 * 1024,
    l2_bytes: usize = 512 * 1024,
    l3_bytes: usize = 16 * 1024 * 1024,

    /// Free the model_name slice if it was heap-allocated.
    /// The "unknown" sentinel is the string literal "unknown" which must
    /// not be freed. All other values are heap-allocated by detectCpuLinux.
    pub fn deinit(self: *CpuInfo, allocator: Allocator) void {
        // "unknown" is the only string-literal value. Check by content.
        if (!std.mem.eql(u8, self.model_name, "unknown")) {
            allocator.free(self.model_name);
        }
        self.* = undefined;
    }
};

/// A4: tile/block parameters for the GEMM kernels, derived from the
/// host's actual cache hierarchy. The kernels' current fixed constants
/// (e.g. gemm_224_bf16.zig: N_BLOCK=256, K_BLOCK=1792) target a
/// generic server profile; this helper lets callers size the M/N/K
/// blocks so the working set of one block-step fits in L2 (compute)
/// or L3 (weights streaming).
pub const TileParams = struct {
    /// Recommended n_block: how many output columns to process per
    /// outer step. Sized so one B-panel slice (n_block * k_bytes of
    /// BF16 weights) plus the C output tile stays under ~half of L2.
    n_block: usize,
    /// Recommended k_block: how deep a K-panel may grow before the
    /// A+B working set exceeds L2.
    k_block: usize,
    /// True when the values are the conservative defaults (detection
    /// unavailable) rather than measured-from-sysfs.
    estimated: bool,
};

/// A4: derive GEMM block sizes from the detected cache hierarchy.
/// Heuristic: one block-step's working set is
///   (m_step + n_step) * k_step * 2 bytes (A and B tiles)
/// plus m_step * n_step * 4 bytes (C tile, FP32).
/// We target <= 50% of L2 so two block-steps can be in flight
/// (prefetch of the next panel while computing the current one).
/// m_step/n_step/k_step are the kernel tile granularity (32 for the
/// BF16 GemmKernel224). Returns conservative defaults when the
/// detected sizes look implausible (0 or tiny).
pub fn selectTileParams(cpu: CpuInfo) TileParams {
    const elem_bytes: usize = 2; // BF16
    const m_step: usize = 32;
    const n_step: usize = 32;
    const k_step: usize = 32;

    // A working budget of half the (detected) L2. If L2 is unknown or
    // implausibly small, fall back to the conservative default budget
    // of 256 KiB.
    var l2 = cpu.l2_bytes;
    if (l2 < 16 * 1024) l2 = 256 * 1024;
    const budget = l2 / 2;

    // C tile cost (FP32): m_step * n_step * 4. Subtract from budget.
    const c_cost = m_step * n_step * 4;
    if (budget <= c_cost) {
        return .{ .n_block = 256, .k_block = 1792, .estimated = true };
    }

    // Remaining budget buys K depth: A is m_step * k * 2, B is
    // n_block * k * 2. For a first-cut n_block we use 8*n_step=256
    // (the existing kernel constant) and spend what's left on K.
    const n_block: usize = 8 * n_step;
    const per_k = (m_step + n_block) * elem_bytes;
    var k_block = (budget - c_cost) / per_k;
    // Round down to a multiple of k_step so panels stay tile-aligned.
    k_block = (k_block / k_step) * k_step;
    if (k_block < k_step) k_block = k_step;

    return .{
        .n_block = n_block,
        .k_block = k_block,
        .estimated = cpu.l2_bytes < 16 * 1024,
    };
}

/// Detect CPU features and vendor
pub fn detectCpu(allocator: Allocator) !CpuInfo {
    var io = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    const os = builtin.target.os.tag;
    switch (os) {
        .linux => return detectCpuLinux(allocator, &io),
        .macos => return detectCpuDarwin(allocator, &io),
        .windows => return detectCpuWindows(allocator, &io),
        else => return CpuInfo{
            .vendor = .unknown,
            .arch = @tagName(builtin.target.cpu.arch) orelse "unknown",
            .features = Features{},
            .flags = .{0} ** 16,
            .model_name = @constCast("unknown"),
            .cpu_family = 0,
            .model = 0,
            .stepping = 0,
        },
    }
}

fn detectCpuLinux(allocator: Allocator, io: *std.Io.Threaded) !CpuInfo {
    const dir = std.Io.Dir.cwd();
    const file = try dir.openFile(std.Io.Threaded.io(io), "/proc/cpuinfo", .{});
    defer file.close(std.Io.Threaded.io(io));

    // Read file content using a buffer
    var buffer = try allocator.alloc(u8, 1_000_000);
    defer allocator.free(buffer);
    const n = try file.readPositionalAll(std.Io.Threaded.io(io), buffer, 0);
    const content = buffer[0..n];

    var lines = std.mem.splitScalar(u8, content, 10);

    var vendor: Vendor = .unknown;
    var model_name: []u8 = @constCast("unknown");
    var cpu_family: u32 = 0;
    var model: u32 = 0;
    var stepping: u32 = 0;
    var flags: [16]u8 = undefined;
    var flags_len: usize = 0;
    for (&flags) |*elem| elem.* = 0;
    const features = Features{};

    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "vendor_id")) {
            var parts = std.mem.splitScalar(u8, line, 58);
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
            var parts = std.mem.splitScalar(u8, line, 58);
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    model_name = @constCast(std.mem.trim(u8, value, " \t"));
                }
            }
        } else if (std.mem.startsWith(u8, line, "cpu family")) {
            var parts = std.mem.splitScalar(u8, line, 58);
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    cpu_family = std.fmt.parseInt(u32, std.mem.trim(u8, value, " \t"), 10) catch 0;
                }
            }
        } else if (std.mem.startsWith(u8, line, "model")) {
            var parts = std.mem.splitScalar(u8, line, 58);
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    model = std.fmt.parseInt(u32, std.mem.trim(u8, value, " \t"), 10) catch 0;
                }
            }
        } else if (std.mem.startsWith(u8, line, "stepping")) {
            var parts = std.mem.splitScalar(u8, line, 58);
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    stepping = std.fmt.parseInt(u32, std.mem.trim(u8, value, " \t"), 10) catch 0;
                }
            }
        } else if (std.mem.startsWith(u8, line, "flags") or std.mem.startsWith(u8, line, "Features")) {
            var parts = std.mem.splitScalar(u8, line, 58);
            if (parts.next()) |_| {
                if (parts.next()) |value| {
                    const flag_str = std.mem.trim(u8, value, " \t");
                    var iter = std.mem.splitScalar(u8, flag_str, 32);
                    while (iter.next()) |flag| {
                        if (flag.len > 0) {
                            if (flags_len < 16) {
                                flags[flags_len] = flag[0];
                                flags_len += 1;
                            }
                            try parseFlag(allocator, flag, @constCast(&features));
                        }
                    }
                }
            }
        }
    }

    // Copy model_name to a separate allocation so it survives the
    // `defer allocator.free(buffer)` below. Without this, the returned
    // CpuInfo.model_name is a dangling pointer into the freed buffer.
    model_name = try allocator.dupe(u8, model_name);
    errdefer allocator.free(model_name);

    // Detect ARM from model name if vendor unknown
    if (vendor == .unknown) {
        const lower = std.ascii.allocLowerString(allocator, model_name) catch return error.OutOfMemory;
        if ((std.mem.indexOf(u8, lower, "aarch64") != null) or
            (std.mem.indexOf(u8, lower, "armv8") != null) or
            (std.mem.indexOf(u8, lower, "arm cortex") != null) or
            (std.mem.indexOf(u8, lower, "kunpeng") != null) or
            (std.mem.indexOf(u8, lower, "kirin") != null) or
            (std.mem.indexOf(u8, lower, "huawei") != null))
        {
            vendor = .arm;
        }
    }

    // Get architecture
    const arch = switch (builtin.target.cpu.arch) {
        .x86_64 => "x86_64",
        .aarch64 => "aarch64",
        .arm => "arm",
        else => "unknown",
    };

    // A4: detect the cache hierarchy from sysfs. Best-effort: on
    // failure (no sysfs, container, etc.) the CpuInfo defaults hold.
    const cache = detectCacheLinux(allocator, io);

    return CpuInfo{
        .vendor = vendor,
        .arch = arch,
        .features = features,
        .flags = flags,
        .model_name = model_name,
        .cpu_family = cpu_family,
        .model = model,
        .stepping = stepping,
        .l1d_bytes = cache.l1d,
        .l1i_bytes = cache.l1i,
        .l2_bytes = cache.l2,
        .l3_bytes = cache.l3,
    };
}

/// A4: parsed cache sizes from
/// /sys/devices/system/cpu/cpu0/cache/index{N}/{type,level,size}.
/// `size` is a string like "32K", "512K", "16M" (suffix K/M/G).
const CacheSizes = struct {
    l1d: usize,
    l1i: usize,
    l2: usize,
    l3: usize,
};

fn detectCacheLinux(allocator: Allocator, io: *std.Io.Threaded) CacheSizes {
    var result = CacheSizes{
        .l1d = 32 * 1024,
        .l1i = 32 * 1024,
        .l2 = 512 * 1024,
        .l3 = 16 * 1024 * 1024,
    };
    var saw_real = false;

    // index0..index7 covers all real-world topologies (typically 0..3).
    for (0..8) |idx| {
        var path_buf: [128]u8 = undefined;

        // level
        const level_path = std.fmt.bufPrintZ(&path_buf, "/sys/devices/system/cpu/cpu0/cache/index{d}/level", .{idx}) catch break;
        const level = readSysfsUint(allocator, io, level_path) orelse break;

        var type_buf: [128]u8 = undefined;
        const type_path = std.fmt.bufPrintZ(&type_buf, "/sys/devices/system/cpu/cpu0/cache/index{d}/type", .{idx}) catch break;
        const ctype = readSysfsTrimmed(allocator, io, type_path) orelse break;
        defer allocator.free(ctype);

        var size_buf: [128]u8 = undefined;
        const size_path = std.fmt.bufPrintZ(&size_buf, "/sys/devices/system/cpu/cpu0/cache/index{d}/size", .{idx}) catch break;
        const size_str = readSysfsTrimmed(allocator, io, size_path) orelse break;
        defer allocator.free(size_str);
        const size = parseSizeString(size_str) orelse continue;

        // sysfs "type": Data, Instruction, Unified. L1 splits d/i;
        // L2/L3 are Unified.
        if (level == 1) {
            if (std.mem.eql(u8, ctype, "Data")) {
                result.l1d = size;
                saw_real = true;
            } else if (std.mem.eql(u8, ctype, "Instruction")) {
                result.l1i = size;
                saw_real = true;
            }
        } else if (level == 2) {
            result.l2 = size;
            saw_real = true;
        } else if (level == 3) {
            result.l3 = size;
            saw_real = true;
        }
    }

    if (!saw_real) {
        // No sysfs info at all — keep the conservative defaults
        // (already in `result`) so callers get sane TileParams.
    }
    return result;
}

/// Read a whole small sysfs file, trim the trailing newline.
/// Uses the same std.Io.Dir/File pattern as detectCpuLinux (Zig 0.16:
/// std.fs.cwd is gone; all FS access goes through the Io interface).
fn readSysfsTrimmed(allocator: Allocator, io: *std.Io.Threaded, path: [:0]const u8) ?[]u8 {
    const dir = std.Io.Dir.cwd();
    const f = dir.openFile(std.Io.Threaded.io(io), path, .{}) catch return null;
    defer f.close(std.Io.Threaded.io(io));
    var buf: [64]u8 = undefined;
    // readPositionalAll reads exactly buf.len or up to EOF at offset 0.
    // Sysfs attribute files are tiny (a few bytes), so a 64-byte buffer
    // read at offset 0 captures the whole value including the newline.
    const n = std.Io.File.readPositionalAll(f, std.Io.Threaded.io(io), &buf, 0) catch return null;
    return allocator.dupe(u8, std.mem.trim(u8, buf[0..n], " \t\n\r")) catch null;
}

/// Read a whole small sysfs file as usize (decimal).
fn readSysfsUint(allocator: Allocator, io: *std.Io.Threaded, path: [:0]const u8) ?usize {
    const s = readSysfsTrimmed(allocator, io, path) orelse return null;
    defer allocator.free(s);
    return std.fmt.parseInt(usize, s, 10) catch null;
}

/// Parse sysfs cache size strings: "32K", "512K", "16M", "2048G".
/// Returns bytes. Plain digits ("32768") are also accepted.
fn parseSizeString(s: []const u8) ?usize {
    if (s.len == 0) return null;
    const suffix = s[s.len - 1];
    const mult: usize = switch (suffix) {
        'K' => 1024,
        'M' => 1024 * 1024,
        'G' => 1024 * 1024 * 1024,
        else => 1,
    };
    const num_part = if (mult != 1) s[0 .. s.len - 1] else s;
    const n = std.fmt.parseInt(usize, num_part, 10) catch return null;
    return n * mult;
}

fn detectCpuDarwin(_allocator: Allocator, _io: std.Io) !CpuInfo {
    _ = _io;
    _ = _allocator;
    const arch = switch (builtin.target.cpu.arch) {
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
        .flags = .{0} ** 16,
        .model_name = @constCast("Apple Silicon"),
        .cpu_family = 0,
        .model = 0,
        .stepping = 0,
    };
}

fn detectCpuWindows(_allocator: Allocator, _io: anytype) !CpuInfo {
    _ = _io;
    _ = _io;
    _ = _allocator;
    const arch = switch (builtin.target.cpu.arch) {
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
        .flags = .{0} ** 16,
        .model_name = @constCast("Windows CPU"),
        .cpu_family = 0,
        .model = 0,
        .stepping = 0,
    };
}

fn parseFlag(allocator: std.mem.Allocator, flag: []const u8, features: *Features) error{OutOfMemory}!void {
    const f = try std.ascii.allocLowerString(allocator, flag);
    defer allocator.free(f);
    if (std.mem.eql(u8, f, "avx")) {
        features.avx = true;
    } else if (std.mem.eql(u8, f, "avx2")) {
        features.avx2 = true;
    } else if (std.mem.eql(u8, f, "fma")) {
        features.fma = true;
    } else if (std.mem.eql(u8, f, "f16c")) {
        features.f16c = true;
    } else if (std.mem.eql(u8, f, "avx512f")) {
        features.avx512f = true;
    } else if (std.mem.eql(u8, f, "avx512bw")) {
        features.avx512bw = true;
    } else if (std.mem.eql(u8, f, "avx512dq")) {
        features.avx512dq = true;
    } else if (std.mem.eql(u8, f, "avx512vl")) {
        features.avx512vl = true;
    } else if (std.mem.eql(u8, f, "avx512vnni") or std.mem.eql(u8, f, "avx512_vnni")) {
        features.avx512vnni = true;
    } else if (std.mem.eql(u8, f, "avx512bf16") or std.mem.eql(u8, f, "avx512_bf16")) {
        features.avx512bf16 = true;
    } else if (std.mem.eql(u8, f, "avx512vbmi") or std.mem.eql(u8, f, "avx512_vbmi")) {
        features.avx512vbmi = true;
    } else if (std.mem.eql(u8, f, "avx512vpopcntdq") or std.mem.eql(u8, f, "avx512_vpopcntdq")) {
        features.avx512vpopcntdq = true;
    } else if (std.mem.eql(u8, f, "amx_bf16") or std.mem.eql(u8, f, "amx-bf16")) {
        features.amx_bf16 = true;
    } else if (std.mem.eql(u8, f, "amx_int8") or std.mem.eql(u8, f, "amx-int8")) {
        features.amx_int8 = true;
    } else if (std.mem.eql(u8, f, "amx_tile") or std.mem.eql(u8, f, "amx-tile")) {
        features.amx_tile = true;
    } else if (std.mem.eql(u8, f, "sve")) {
        features.sve = true;
    } else if (std.mem.eql(u8, f, "sve2")) {
        features.sve2 = true;
    } else if (std.mem.eql(u8, f, "neon")) {
        features.neon = true;
    }
}

/// Select best CPU variant for current hardware
pub fn selectBestVariant(cpu: CpuInfo) []const u8 {
    const f = cpu.features;

    // aarch64: NEON is baseline on ARMv8; SVE/SVE2 are optional (Armv8.2+ and
    // Armv8.4+ respectively). On aarch64, prefer SVE2 > SVE > NEON. On Apple
    // Silicon the kernel-space HWCAP reports no SVE even though the M-series
    // microarchitecture has wide SIMD, so fall back to "neon" whenever neither
    // SVE feature is set. We test on cpu.vendor first (set by detectCpuLinux
    // from /proc/cpuinfo model name and by detectCpuDarwin from the arch),
    // falling back to cpu.arch == "aarch64" for hosts where the vendor was
    // not classified.
    if (cpu.vendor == .arm or std.mem.eql(u8, cpu.arch, "aarch64")) {
        if (f.sve2) return "sve2";
        if (f.sve) return "sve";
        return "neon";
    }

    // Check for AMX support (Sapphire Rapids+)
    if (f.amx_bf16 and f.amx_int8 and f.amx_tile) {
        return "amx";
    }

    // Check for AVX512 BF16 (Ice Lake Server, Zen 4+)
    if (f.avx512bf16 and f.avx512vbmi and f.avx512vnni) {
        return "avx512_bf16";
    }

    // Check for AVX512 VBMI (Ice Lake Client)
    if (f.avx512vbmi and f.avx512vnni) {
        return "avx512_vbmi";
    }

    // Check for AVX512 VNNI (Cascade Lake+)
    if (f.avx512vnni) {
        return "avx512_vnni";
    }

    // Check for base AVX512 (Skylake-X+)
    if (f.avx512f and f.avx512bw and f.avx512dq and f.avx512vl) {
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
    std.debug.print("  Vendor: {s}\n", .{@tagName(cpu.vendor)});
    std.debug.print("  Arch: {s}\n", .{cpu.arch});
    std.debug.print("  Model: {s}\n", .{cpu.model_name});
    std.debug.print("  Family: {d}, Model: {d}, Stepping: {d}\n", .{ cpu.cpu_family, cpu.model, cpu.stepping });
    std.debug.print("  Features:\n", .{});
    std.debug.print("    AVX: {}\n", .{cpu.features.avx});
    std.debug.print("    AVX2: {}\n", .{cpu.features.avx2});
    std.debug.print("    FMA: {}\n", .{cpu.features.fma});
    std.debug.print("    F16C: {}\n", .{cpu.features.f16c});
    std.debug.print("    AVX512F: {}\n", .{cpu.features.avx512f});
    std.debug.print("    AVX512BW: {}\n", .{cpu.features.avx512bw});
    std.debug.print("    AVX512DQ: {}\n", .{cpu.features.avx512dq});
    std.debug.print("    AVX512VL: {}\n", .{cpu.features.avx512vl});
    std.debug.print("    AVX512VNNI: {}\n", .{cpu.features.avx512vnni});
    std.debug.print("    AVX512BF16: {}\n", .{cpu.features.avx512bf16});
    std.debug.print("    AVX512VBMI: {}\n", .{cpu.features.avx512vbmi});
    std.debug.print("    AMX BF16: {}\n", .{cpu.features.amx_bf16});
    std.debug.print("    AMX INT8: {}\n", .{cpu.features.amx_int8});
    std.debug.print("    AMX TILE: {}\n", .{cpu.features.amx_tile});
    std.debug.print("    SVE: {}\n", .{cpu.features.sve});
    std.debug.print("    NEON: {}\n", .{cpu.features.neon});
    std.debug.print("  Best variant: {s}\n", .{selectBestVariant(cpu)});
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
