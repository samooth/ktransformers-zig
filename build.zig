const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // CPU variant to build (default: avx2)
    const variant_opt = b.option([]const u8, "variant", "CPU variant: avx2, avx512_base, avx512_vnni, avx512_vbmi, avx512_bf16, amx") orelse "avx2";

    // Variant configurations
    const Variant = enum {
        avx2,
        avx512_base,
        avx512_vnni,
        avx512_vbmi,
        avx512_bf16,
        amx,
    };

    const variant = std.meta.stringToEnum(Variant, variant_opt).?;

    // Use array lookups
    const cpu_instruct_table = [_][]const u8{ "AVX2", "AVX512", "AVX512", "AVX512", "AVX512", "AVX512" };
    const suffix_table = [_][]const u8{ "avx2", "avx512_base", "avx512_vnni", "avx512_vbmi", "avx512_bf16", "amx" };

    const variant_idx = @intFromEnum(variant);
    const suffix = suffix_table[variant_idx];
    const cpu_instruct = cpu_instruct_table[variant_idx];

    std.debug.print("Building variant: {s} (cpu_instruct={s})\n", .{ suffix, cpu_instruct });

    // Create module for this variant - root.zig re-exports all submodules and includes main.zig
    const mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add C macros for this variant
    mod.addCMacro("std=c++17", "");
    mod.addCMacro("fPIC", "");
    mod.addCMacro("O3", "");
    mod.addCMacro("DNDEBUG", "");
    mod.addCMacro("HAVE_AMX", if (variant == .amx) "1" else "0");
    mod.addCMacro("LLAMA_AVX2", if (variant == .avx2) "1" else "0");
    mod.addCMacro("LLAMA_AVX512", if (variant != .avx2) "1" else "0");
    mod.addCMacro("KTRANSFORMERS_CPU_USE_AMX", if (variant == .amx) "1" else "0");
    mod.addCMacro("KTRANSFORMERS_CPU_USE_AMX_AVX512", if (variant != .avx2) "1" else "0");
    mod.addCMacro("LLAMA_AVX512_VNNI", if (variant == .avx512_vnni or variant == .avx512_vbmi or variant == .avx512_bf16 or variant == .amx) "1" else "0");
    mod.addCMacro("LLAMA_AVX512_BF16", if (variant == .avx512_bf16 or variant == .amx) "1" else "0");
    mod.addCMacro("LLAMA_AVX512_VBMI", if (variant == .avx512_vbmi or variant == .avx512_bf16 or variant == .amx) "1" else "0");
    mod.addCMacro("HAVE_AMX", if (variant == .amx) "1" else "0");

    // Add library for this variant
    const lib = b.addLibrary(.{
        .name = "kt_kernel_ext",
        .linkage = .dynamic,
        .root_module = mod,
    });

    // Install with variant-specific name
    const install_step = b.addInstallArtifact(lib, .{});
    b.getInstallStep().dependOn(&install_step.step);

    // Default step
    const default_step = b.step("build", "Build CPU variant");
    default_step.dependOn(b.getInstallStep());

    // Test step
    const test_step = b.step("test", "Run tests");
    const test_mod = b.createModule(.{
        .root_source_file = b.path("tests/kernels/test_kernels.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("kt", b.createModule(.{ .root_source_file = b.path("src/root.zig"), .target = target, .optimize = optimize }));

    const test_obj = b.addTest(.{ .root_module = test_mod });
    test_step.dependOn(&test_obj.step);

    // MLA test module (src/mla/mla_tests.zig) - second test suite,
    // runs alongside tests/kernels/test_kernels.zig.
    const mla_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mla/mla_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    mla_test_mod.addImport("kt", b.createModule(.{ .root_source_file = b.path("src/root.zig"), .target = target, .optimize = optimize }));
    const mla_test_obj = b.addTest(.{ .root_module = mla_test_mod });
    test_step.dependOn(&mla_test_obj.step);
}
