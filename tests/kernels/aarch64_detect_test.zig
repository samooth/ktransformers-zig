// Standalone test for aarch64 detection in src/runtime/cpu_detect.zig.
//
// selectBestVariant() had no aarch64 branch — on an ARM CPU the function
// would fall through every x86 feature check and return "avx2" as a
// fallback (the worst possible mismatch). This test pins the fix: aarch64
// CPUs must be selected as sve2 > sve > neon, and the x86 selection path
// is unchanged.
//
// Avoids the locked tests/kernels/test_kernels.zig by being its own suite;
// wired into build.zig alongside the other standalone suites.

const std = @import("std");
const testing = std.testing;
const cpu_detect = @import("kt").cpu_detect;

const CpuInfo = cpu_detect.CpuInfo;
const Features = cpu_detect.Features;
const Vendor = cpu_detect.Vendor;

fn aarch64Info(features: Features) CpuInfo {
    return .{
        .vendor = .arm,
        .arch = "aarch64",
        .features = features,
        .flags = .{0} ** 16,
        .model_name = @constCast("cortex-a76"), // a placeholder; the function must not read it
        .cpu_family = 0,
        .model = 0,
        .stepping = 0,
    };
}

fn x86Info(features: Features, arch_str: []const u8) CpuInfo {
    return .{
        .vendor = .intel,
        .arch = arch_str,
        .features = features,
        .flags = .{0} ** 16,
        .model_name = @constCast("test"),
        .cpu_family = 0,
        .model = 0,
        .stepping = 0,
    };
}

test "aarch64 with sve2 features picks sve2" {
    var f = Features{};
    f.sve2 = true;
    f.sve = true;
    f.neon = true;
    try testing.expectEqualStrings("sve2", cpu_detect.selectBestVariant(aarch64Info(f)));
}

test "aarch64 with sve (no sve2) picks sve" {
    var f = Features{};
    f.sve = true;
    f.neon = true;
    try testing.expectEqualStrings("sve", cpu_detect.selectBestVariant(aarch64Info(f)));
}

test "aarch64 baseline (neon only) picks neon" {
    var f = Features{};
    f.neon = true;
    try testing.expectEqualStrings("neon", cpu_detect.selectBestVariant(aarch64Info(f)));
}

test "aarch64 with no features (degenerate) still picks neon" {
    // A real aarch64 chip always has NEON; this test covers the vendor-mismatch
    // safety net so a bug in detectCpu* that drops the neon flag cannot strand
    // the selection on a literal "avx2" fallback.
    const f = Features{};
    try testing.expectEqualStrings("neon", cpu_detect.selectBestVariant(aarch64Info(f)));
}

test "aarch64 detection via arch string when vendor classification is missing" {
    // Some hosts may not set vendor = .arm; the function must also accept a
    // bare "aarch64" arch string (detectCpuLinux can fall back to .unknown
    // if the model name grep misses a vendor-classified string).
    var f = Features{};
    f.neon = true;
    const info = CpuInfo{
        .vendor = .unknown,
        .arch = "aarch64",
        .features = f,
        .flags = .{0} ** 16,
        .model_name = @constCast("unknown"),
        .cpu_family = 0,
        .model = 0,
        .stepping = 0,
    };
    try testing.expectEqualStrings("neon", cpu_detect.selectBestVariant(info));
}

test "x86 selection is unaffected: AVX2 still wins when AMX/AVX-512 absent" {
    var f = Features{};
    f.avx2 = true;
    f.fma = true;
    try testing.expectEqualStrings("avx2", cpu_detect.selectBestVariant(x86Info(f, "x86_64")));
}

test "x86 selection precedence: AMX > AVX-512 BF16 > AVX-512 VBMI > VNNI > base > AVX2" {
    // Full-feature Intel: AMX wins.
    {
        var f = Features{};
        f.amx_bf16 = true;
        f.amx_int8 = true;
        f.amx_tile = true;
        f.avx512bf16 = true;
        f.avx2 = true;
        try testing.expectEqualStrings("amx", cpu_detect.selectBestVariant(x86Info(f, "x86_64")));
    }
    // AMX partial: not selected; AVX-512 BF16 wins.
    {
        var f = Features{};
        f.amx_bf16 = true;
        f.amx_int8 = true;
        // no amx_tile
        f.avx512bf16 = true;
        f.avx512vbmi = true;
        f.avx512vnni = true;
        try testing.expectEqualStrings("avx512_bf16", cpu_detect.selectBestVariant(x86Info(f, "x86_64")));
    }
    // Ice Lake Client: avx512vbmi + avx512vnni, no bf16.
    {
        var f = Features{};
        f.avx512vbmi = true;
        f.avx512vnni = true;
        try testing.expectEqualStrings("avx512_vbmi", cpu_detect.selectBestVariant(x86Info(f, "x86_64")));
    }
}
