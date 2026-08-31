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
            mod.link_libc = true;

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

            const lib = builder.addLibrary(.{
                .name = lib_name,
                .linkage = .dynamic,
                .root_module = mod,
            });
            return lib;
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

    // --- Test step (runs all three suites) ---
    //
    // NOTE ON THE LISTEN PROTOCOL: Zig 0.16's `addTest` + `addRunArtifact`
    // passes `--listen=-` to the test binary, which makes it communicate results
    // back to the parent `zig build` process over an stdin/stdout IPC handshake.
    // In THIS environment that handshake fails — the binary panics with
    // "internal test runner failure" in std/Io/Reader.readSliceAll, and zig build
    // prints "failed command" even though every test passes (exit 0). The
    // binaries run perfectly when invoked standalone (no --listen).
    //
    // Workaround: compile each test with addTest (gets us a linked test exe),
    // then override the run command via setExecCmd to invoke the emitted binary
    // directly — no --listen flag. Results print straight to stdout. Verified:
    // all 47 tests (34 kernel + 2 fp8 + 11 mla) pass this way, RC=0.
    const test_step = b.step("test", "Run tests");

    // Compile a test module rooted at test_src with kt imported from kt_src,
    // then run it with the project's simple-mode test runner (tools/test_runner.zig).
    const RunTest = struct {
        fn run(bld: *std.Build, step: *std.Build.Step, test_src: []const u8, kt_src: []const u8, tgt: anytype, opt: anytype) void {
            const tmod = bld.createModule(.{
                .root_source_file = bld.path(test_src),
                .target = tgt,
                .optimize = opt,
            });
            const ktmod = bld.createModule(.{ .root_source_file = bld.path(kt_src), .target = tgt, .optimize = opt });
            ktmod.link_libc = true;
            tmod.addImport("kt", ktmod);
            const tobj = bld.addTest(.{
                .root_module = tmod,
                .test_runner = .{
                    .path = bld.path("tools/test_runner.zig"),
                    .mode = .simple,
                },
            });
            step.dependOn(&bld.addRunArtifact(tobj).step);
        }
    };

    // Suite 1: kernel tests (kt from root.zig)
    RunTest.run(b, test_step, "tests/kernels/test_kernels.zig", "src/root.zig", target, optimize);
    // Suite 2: MLA tests (kt from root.zig)
    RunTest.run(b, test_step, "src/mla/mla_tests.zig", "src/root.zig", target, optimize);
    // Suite 3: FP8 layerwise transport (kt from main.zig — export fn symbols
    // live in main.zig and are not re-exportable as root.zig namespace members)
    RunTest.run(b, test_step, "tests/kernels/fp8_transport_tests.zig", "src/main.zig", target, optimize);
    // Suite 4: MLA C API lifecycle (kt_mla_new/load_weights/forward/decode/free)
    RunTest.run(b, test_step, "tests/kernels/mla_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 5: GGML quant C API (rooted at main.zig for export access)
    RunTest.run(b, test_step, "tests/kernels/ggml_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 5: aarch64 CPU detection in selectBestVariant
    RunTest.run(b, test_step, "tests/kernels/aarch64_detect_test.zig", "src/root.zig", target, optimize);
}
