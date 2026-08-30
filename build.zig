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

    const all_variants = [_]Variant{ .avx2, .avx512_base, .avx512_vnni, .avx512_vbmi, .avx512_bf16, .amx };

    const variant = std.meta.stringToEnum(Variant, variant_opt).?;

    const cpu_instruct_table = [_][]const u8{ "AVX2", "AVX512", "AVX512", "AVX512", "AVX512", "AVX512" };
    const suffix_table = [_][]const u8{ "avx2", "avx512_base", "avx512_vnni", "avx512_vbmi", "avx512_bf16", "amx" };

    // Build one variant's shared library. lib_name determines the installed
    // filename: lib<lib_name>.so. target/optimize are resolved once in the
    // outer build() and passed in (standardTargetOptions can only be called
    // once per build since it registers CLI options).
    const buildLib = struct {
        fn build(builder: *std.Build, variant_kind: Variant, lib_name: []const u8, tgt: anytype, opt: anytype) *std.Build.Step.Compile {
            const variant_idx = @intFromEnum(variant_kind);
            const suffix = suffix_table[variant_idx];
            const cpu_instruct = cpu_instruct_table[variant_idx];

            std.debug.print("Building variant: {s} (cpu_instruct={s})\n", .{ suffix, cpu_instruct });

            const mod = builder.createModule(.{
                .root_source_file = builder.path("src/root.zig"),
                .target = tgt,
                .optimize = opt,
            });

            mod.addCMacro("std=c++17", "");
            mod.addCMacro("fPIC", "");
            mod.addCMacro("O3", "");
            mod.addCMacro("DNDEBUG", "");
            mod.addCMacro("HAVE_AMX", if (variant_kind == .amx) "1" else "0");
            mod.addCMacro("LLAMA_AVX2", if (variant_kind == .avx2) "1" else "0");
            mod.addCMacro("LLAMA_AVX512", if (variant_kind != .avx2) "1" else "0");
            mod.addCMacro("KTRANSFORMERS_CPU_USE_AMX", if (variant_kind == .amx) "1" else "0");
            mod.addCMacro("KTRANSFORMERS_CPU_USE_AMX_AVX512", if (variant_kind != .avx2) "1" else "0");
            mod.addCMacro("LLAMA_AVX512_VNNI", if (variant_kind == .avx512_vnni or variant_kind == .avx512_vbmi or variant_kind == .avx512_bf16 or variant_kind == .amx) "1" else "0");
            mod.addCMacro("LLAMA_AVX512_BF16", if (variant_kind == .avx512_bf16 or variant_kind == .amx) "1" else "0");
            mod.addCMacro("LLAMA_AVX512_VBMI", if (variant_kind == .avx512_vbmi or variant_kind == .avx512_bf16 or variant_kind == .amx) "1" else "0");
            mod.addCMacro("HAVE_AMX", if (variant_kind == .amx) "1" else "0");

            return builder.addLibrary(.{
                .name = lib_name,
                .linkage = .dynamic,
                .root_module = mod,
            });
        }
    }.build;

    // --- Single-variant build (default path) ---
    const lib = buildLib(b, variant, "kt_kernel_ext", target, optimize);
    const install_step = b.addInstallArtifact(lib, .{});
    b.getInstallStep().dependOn(&install_step.step);

    const default_step = b.step("build", "Build CPU variant");
    default_step.dependOn(b.getInstallStep());

    // --- All-variants step: builds all 6 variants with distinct names ---
    // AMX variant compiles on non-AMX hosts (the asm is emitted but guarded
    // at runtime by requestAmxPermission); runtime requires AMX hardware.
    const all_variants_step = b.step("all-variants", "Build all 6 CPU variants");
    for (all_variants) |v| {
        const suffix = suffix_table[@intFromEnum(v)];
        const vlib = buildLib(b, v, b.fmt("kt_kernel_ext_{s}", .{suffix}), target, optimize);
        const vinstall = b.addInstallArtifact(vlib, .{});
        all_variants_step.dependOn(&vinstall.step);
    }

    // --- Test step (runs both suites) ---
    const test_step = b.step("test", "Run tests");
    const test_mod = b.createModule(.{
        .root_source_file = b.path("tests/kernels/test_kernels.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("kt", b.createModule(.{ .root_source_file = b.path("src/root.zig"), .target = target, .optimize = optimize }));

    const test_obj = b.addTest(.{ .root_module = test_mod });
    test_step.dependOn(&b.addRunArtifact(test_obj).step);

    // MLA test module (src/mla/mla_tests.zig) - second test suite,
    // runs alongside tests/kernels/test_kernels.zig.
    const mla_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mla/mla_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    mla_test_mod.addImport("kt", b.createModule(.{ .root_source_file = b.path("src/root.zig"), .target = target, .optimize = optimize }));
    const mla_test_obj = b.addTest(.{ .root_module = mla_test_mod });
    test_step.dependOn(&b.addRunArtifact(mla_test_obj).step);
}
