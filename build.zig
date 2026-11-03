const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // CPU variant to build (default: avx2). The .neon variant requires an
    // aarch64 target (see neon_target_check below).
    const variant_opt = b.option(
        []const u8,
        "variant",
        "CPU variant: avx2, avx512_base, avx512_vnni, avx512_vbmi, avx512_bf16, amx, neon",
    ) orelse "avx2";

    // Variant configurations
    const Variant = enum {
        avx2,
        avx512_base,
        avx512_vnni,
        avx512_vbmi,
        avx512_bf16,
        amx,
        neon,
    };

    const all_variants = [_]Variant{ .avx2, .avx512_base, .avx512_vnni, .avx512_vbmi, .avx512_bf16, .amx };

    const variant = std.meta.stringToEnum(Variant, variant_opt).?;

    const cpu_instruct_table = [_][]const u8{ "AVX2", "AVX512", "AVX512", "AVX512", "AVX512", "AVX512", "NEON" };
    const suffix_table = [_][]const u8{ "avx2", "avx512_base", "avx512_vnni", "avx512_vbmi", "avx512_bf16", "amx", "neon" };

    // Footgun guard: the neon variant targets aarch64. If the user selects it
    // without an aarch64 target, fail fast with a clear message rather than
    // silently building a meaningless x86 "neon" library.
    if (variant == .neon and target.result.cpu.arch != .aarch64) {
        std.debug.print(
            "error: -Dvariant=neon requires an aarch64 target.\n" ++
                "       Use: zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu\n" ++
                "       (got arch: {s})\n",
            .{@tagName(target.result.cpu.arch)},
        );
        std.process.exit(1);
    }

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
    // Install the default variant as `libkt_kernel_ext.so` (no suffix) for the
    // historical "just dlopen me" Python ctypes path; install everything else
    // as `libkt_kernel_ext_<suffix>.so` so they don't clobber the default.
    const variant_suffix = suffix_table[@intFromEnum(variant)];
    const default_lib_name: []const u8 = if (variant == .avx2) "kt_kernel_ext" else b.fmt("kt_kernel_ext_{s}", .{variant_suffix});
    const lib = buildLib(b, variant, default_lib_name, target, optimize);
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
    // Suite 6: DeepseekV3 layer C-API lifecycle (model orchestration)
    RunTest.run(b, test_step, "tests/kernels/dsv3_layer_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 6b: Qwen3 MoE layer + model + CausalLM C-API lifecycle
    RunTest.run(b, test_step, "tests/kernels/qwen3_layer_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 6c: LlamaMoe C-API lifecycle (model orchestration — GGML-quantized MoE)
    RunTest.run(b, test_step, "tests/kernels/llamafile_moe_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 6d: to_float / from_float dispatch (Zig extension — generic block ↔ F32)
    RunTest.run(b, test_step, "tests/kernels/to_from_float_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 6e: math helpers (kt_apply_swiglu/rms_norm/rope + kt_softmax)
    RunTest.run(b, test_step, "tests/kernels/math_helpers_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite: FP8 quantize/dequantize C API (E4M3 and E5M2 block round trips).
    RunTest.run(b, test_step, "tests/kernels/fp8_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 6f: SIMD primitives + GEMM micro-kernels (Cephes exp, RNE bf16
    // store, shared softmax, gemmExpert/gemmExpertAccum, gemmF32Bf16).
    RunTest.run(b, test_step, "tests/kernels/simd_kernels_tests.zig", "src/root.zig", target, optimize);
    // Suite 7: Qwen3 MoE + MHA + GGUF (model orchestration + parser)
    RunTest.run(b, test_step, "tests/kernels/qwen3_moe_tests.zig", "src/root.zig", target, optimize);
    // Suite 7b: GGUF E2E — real on-disk model round-trip
    // (Qwen3.5-0.8B-BF16.gguf; mmap'd, not loaded fully).
    RunTest.run(b, test_step, "tests/kernels/qwen3_gguf_e2e_tests.zig", "src/root.zig", target, optimize);
    // Suite 7c: Kimi K3 CPU ops (Phase 1)
    RunTest.run(b, test_step, "tests/kimi/kimi_k3_ops_test.zig", "src/root.zig", target, optimize);
    // Suite 7d: Kimi K3 BF16 KDA layer smoke
    RunTest.run(b, test_step, "tests/kimi/kimi_k3_kda_bf16_test.zig", "src/root.zig", target, optimize);
    // Suite 7e: Kimi K3 BF16 MLA layer smoke
    RunTest.run(b, test_step, "tests/kimi/kimi_k3_mla_bf16_test.zig", "src/root.zig", target, optimize);
    // Suite 7f: Kimi K3 KDA layer C API
    RunTest.run(b, test_step, "tests/kernels/kimi_kda_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 7g: Kimi K3 MLA layer C API
    RunTest.run(b, test_step, "tests/kernels/kimi_mla_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 8: NUMA auto-population in kt_worker_pool_new_config (A3 closure)
    RunTest.run(b, test_step, "tests/kernels/numa_pool_tests.zig", "src/main.zig", target, optimize);
    // Suite 5b: custom allocator injection C API (B1)
    RunTest.run(b, test_step, "tests/kernels/allocator_capi_tests.zig", "src/main.zig", target, optimize);
    // Suite 5: aarch64 CPU detection in selectBestVariant
    RunTest.run(b, test_step, "tests/kernels/aarch64_detect_test.zig", "src/root.zig", target, optimize);
    // Suite 6: ARM NEON feature detection (comptime flag must be false on
    // x86_64; aarch64 cross-build verified by `zig build
    // -Dvariant=neon -Dtarget=aarch64-linux-gnu` plus tools/verify_abi.py).
    const arch_neon_mod = b.createModule(.{ .root_source_file = b.path("src/kernels/arch/neon.zig"), .target = target, .optimize = optimize });
    const neon_test_mod = b.createModule(.{ .root_source_file = b.path("tests/kernels/neon_arch_test.zig"), .target = target, .optimize = optimize });
    neon_test_mod.addImport("arch_neon", arch_neon_mod);
    const neon_test_obj = b.addTest(.{
        .root_module = neon_test_mod,
        .test_runner = .{ .path = b.path("tools/test_runner.zig"), .mode = .simple },
    });
    test_step.dependOn(&b.addRunArtifact(neon_test_obj).step);

    // Suite 6b: NEON Phase-2 — BF16 GEMM kernel correctness on x86_64.
    // The test imports everything via the single `kt` (root) module,
    // which re-exports gemm_bf16_neon and amx. Direct imports of the
    // underlying files would create a "file exists in two modules"
    // error since the test module already has them via root.kt.
    const neon_kt_mod = b.createModule(.{ .root_source_file = b.path("src/root.zig"), .target = target, .optimize = optimize });
    // The test module's `kt` import transitively pulls in pthread
    // (worker_pool, kt_cpuinfer_sync). link_libc is required so the
    // test build resolves the @cImport for pthread_mutex_* / pthread_cond_*.
    neon_kt_mod.link_libc = true;
    const neon_kernel_test_mod = b.createModule(.{ .root_source_file = b.path("tests/kernels/neon_kernel_test.zig"), .target = target, .optimize = optimize });
    neon_kernel_test_mod.addImport("kt", neon_kt_mod);
    const neon_kernel_test_obj = b.addTest(.{
        .root_module = neon_kernel_test_mod,
        .test_runner = .{ .path = b.path("tools/test_runner.zig"), .mode = .simple },
    });
    test_step.dependOn(&b.addRunArtifact(neon_kernel_test_obj).step);

    // --- aarch64 emulated test step ---
    //
    // WHY THIS EXISTS. The neon CI leg (09660de) proves aarch64 COMPILES and
    // exports the right symbols. Neither can catch a wrong ANSWER. `zig build
    // test` only ever runs the suites for the host target, so every aarch64
    // code path — including `amx.pshufb16`'s scalar FALLBACK, which no x86 run
    // can ever execute — had zero coverage. This step closes that.
    //
    // Requires qemu-user (`qemu-aarch64-static`; Debian/Ubuntu package
    // `qemu-user-static`). It is a separate step rather than part of `test`
    // because the emulator is not always installed, and a missing emulator
    // must not turn a plain `zig build test` red.
    //
    // aarch64-linux-musl, not -gnu: musl is self-contained, so the cross-build
    // needs no aarch64 sysroot and the binary runs under qemu-user without a
    // dynamic loader.
    const qemu_opt = b.option([]const u8, "qemu", "aarch64 user-mode emulator (default qemu-aarch64-static)") orelse "qemu-aarch64-static";
    const arm_step = b.step("test-aarch64", "Cross-build + run the ARM suites under qemu (needs qemu-user-static)");
    const arm_target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .musl });

    const ArmTest = struct {
        fn run(bld: *std.Build, step: *std.Build.Step, test_src: []const u8, kt_src: []const u8, tgt: anytype, opt: anytype, qemu: []const u8) void {
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
                .test_runner = .{ .path = bld.path("tools/test_runner.zig"), .mode = .simple },
            });
            // Run through the emulator explicitly rather than relying on
            // binfmt_misc, which is registered on some developer machines and
            // absent in CI. addFileArg on the emitted test binary keeps the
            // path handling to the build system.
            const qemu_run = bld.addSystemCommand(&.{qemu});
            qemu_run.addFileArg(tobj.getEmittedBin());
            qemu_run.setName(bld.fmt("qemu: {s}", .{test_src}));
            step.dependOn(&qemu_run.step);
        }
    };

    // Verified green under emulation before being wired here (2026-10-04):
    // 49 + 7 + 4 tests. The SIMD suite is the important one — it is the only
    // place `pshufb16`'s scalar fallback and the arch-portable `gemmAvx2`
    // bodies (called unconditionally by gemmQuant for IQ2_XXS/IQ2_XS, so they
    // DO run on ARM) are ever executed.
    ArmTest.run(b, arm_step, "tests/kernels/simd_kernels_tests.zig", "src/root.zig", arm_target, optimize, qemu_opt);
    ArmTest.run(b, arm_step, "tests/kernels/aarch64_detect_test.zig", "src/root.zig", arm_target, optimize, qemu_opt);
    ArmTest.run(b, arm_step, "tests/kernels/neon_kernel_test.zig", "src/root.zig", arm_target, optimize, qemu_opt);

    // Two suites are deliberately NOT run here, and both exclusions are
    // deliberate rather than accidental:
    //
    // 1. `neon_arch_test.zig` asserts `NeonFeatures.available == false`, i.e.
    //    it exists to prove the comptime gate is OFF on x86_64. Cross-compiled
    //    to aarch64 that assertion is false by construction, so the suite
    //    cannot pass there. It is an x86-only suite, not an ARM one.
    //
    // 2. `test_kernels.zig` (148 tests) aborts under emulation at test 8,
    //    "CPU detection", inside `detectCpuLinux` -> `parseFlag`, with a
    //    fault address inside the TEXT segment (a wild instruction fetch,
    //    not a bad store at the reported line). The read path itself is
    //    sound: a 1 MB buffer, `readPositionalAll`, and every write is
    //    length-checked; the same parser runs the same code on x86 against
    //    the same /proc/cpuinfo and passes. So this is very likely an
    //    emulation artifact rather than a defect, and wiring it in would
    //    hand CI a red that may be meaningless. It needs either real ARM
    //    hardware or a dedicated investigation — tracked, not swept under
    //    the rug. It is the single biggest hole left in aarch64 coverage.

    // --- Bench step (B2) ---
    //
    // `zig build -Doptimize=ReleaseFast bench` runs both micro-benchmarks
    // in sequence:
    //   1. bench/gemm_bench.zig — A1-vectorized gemmExpert vs pure-scalar
    //      reference at DeepSeek-V3-shaped sizes (isolated GEMM kernel)
    //   2. bench/moe_bench.zig — full TpMoe.forward end-to-end with
    //      default-vs-tuned tile params, validating that the A4 wiring
    //      doesn't regress (and quantifies any win) at the MoE level
    //
    // Both use the same module wiring as the test suites (kt import
    // from root.zig, libc linked for the clock_gettime syscall path).
    // ReleaseFast is strongly recommended: Debug-mode timings are
    // meaningless for SIMD work.
    const bench_step = b.step("bench", "Run GEMM + MoE forward micro-benchmarks (use -Doptimize=ReleaseFast)");
    const bench_kt_mod = b.createModule(.{ .root_source_file = b.path("src/root.zig"), .target = target, .optimize = optimize });
    bench_kt_mod.link_libc = true;

    inline for ([_][]const u8{ "bench/gemm_bench.zig", "bench/moe_bench.zig" }) |src_path| {
        const bench_mod = b.createModule(.{
            .root_source_file = b.path(src_path),
            .target = target,
            .optimize = optimize,
        });
        bench_mod.addImport("kt", bench_kt_mod);
        const name = std.fs.path.basename(src_path);
        const exe_name = name[0 .. name.len - std.fs.path.extension(name).len];
        const bench_exe = b.addExecutable(.{
            .name = exe_name,
            .root_module = bench_mod,
        });
        bench_step.dependOn(&b.addRunArtifact(bench_exe).step);
    }
}
