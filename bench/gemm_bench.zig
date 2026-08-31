// GEMM micro-benchmarks for ktransformers-zig (B2).
//
// Measures the A1-vectorized `gemm_bf16.gemmExpert` (the hot path of
// MoE/MLP/Linear/Gate) against a pure-scalar reference implementation of
// the same math, at DeepSeek-V3-shaped problem sizes. Reports wall time
// per call, GFLOPS, and the vectorized/scalar speedup.
//
// Run via `zig build bench` (wires this file through build.zig with the
// same module setup as the test suites) or standalone:
//   zig run bench/gemm_bench.zig -lc -O ReleaseFast
//
// Timing uses the same clock_gettime(MONOTONIC) syscall pattern as
// src/main.zig:1297 (std.Timer does not exist in Zig 0.16).

const std = @import("std");
const root = @import("kt");

const amx = root.amx;
const gemm_bf16 = root.gemm_bf16;
const moe = root.moe;
const worker_pool = root.worker_pool;

fn nowNs() u64 {
    var ts: std.os.linux.timespec = .{ .sec = 0, .nsec = 0 };
    _ = std.os.linux.syscall2(.clock_gettime, @intFromEnum(std.os.linux.CLOCK.MONOTONIC), @intFromPtr(&ts));
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Pure-scalar reference GEMM: output[m,n] = input[m,k] @ weight[n,k]^T.
/// Identical math to gemmExpert's inner loop pre-A1 (the triple
/// for-loop), kept here as the baseline the speedup is measured against.
fn gemmExpertScalarRef(
    input: [*]const amx.bf16,
    weight: [*]const amx.bf16,
    output: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    input_ld: usize,
    weight_ld: usize,
    output_ld: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                sum += amx.bf16_to_f32(input[i * input_ld + kk]) *
                    amx.bf16_to_f32(weight[j * weight_ld + kk]);
            }
            output[i * output_ld + j] = sum;
        }
    }
}

const BenchCase = struct {
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    flops: f64,
};

fn runCase(
    alloc: std.mem.Allocator,
    case: BenchCase,
    iters: usize,
) !struct { vec_ns: u64, ref_ns: u64, max_diff: f32 } {
    const m = case.m;
    const n = case.n;
    const k = case.k;

    // Buffers. Input/weight are BF16 (2 bytes); output F32.
    const input = try alloc.alloc(amx.bf16, m * k);
    defer alloc.free(input);
    const weight = try alloc.alloc(amx.bf16, n * k);
    defer alloc.free(weight);
    const out_vec = try alloc.alloc(f32, m * n);
    defer alloc.free(out_vec);
    const out_ref = try alloc.alloc(f32, m * n);
    defer alloc.free(out_ref);

    // Deterministic-ish fill: distinct values, small magnitudes (keeps
    // the F32 accumulation in a sane numeric range for the diff check).
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    for (input) |*v| v.* = amx.f32_to_bf16(rand.float(f32) * 0.2 - 0.1);
    for (weight) |*v| v.* = amx.f32_to_bf16(rand.float(f32) * 0.2 - 0.1);

    // Correctness cross-check first (also serves as warmup).
    gemm_bf16.gemmExpert(input.ptr, weight.ptr, out_vec.ptr, m, n, k, k, k, n);
    gemmExpertScalarRef(input.ptr, weight.ptr, out_ref.ptr, m, n, k, k, k, n);
    var max_diff: f32 = 0;
    for (out_vec, out_ref) |a, b| {
        const d = @abs(a - b);
        if (d > max_diff) max_diff = d;
    }

    // Timed runs: alternate single-call timings, keep the best (min) of
    // the run set per implementation to filter scheduler noise.
    var best_vec: u64 = std.math.maxInt(u64);
    var best_ref: u64 = std.math.maxInt(u64);
    for (0..iters) |_| {
        var t0 = nowNs();
        gemm_bf16.gemmExpert(input.ptr, weight.ptr, out_vec.ptr, m, n, k, k, k, n);
        var dt = nowNs() - t0;
        if (dt < best_vec) best_vec = dt;

        t0 = nowNs();
        gemmExpertScalarRef(input.ptr, weight.ptr, out_ref.ptr, m, n, k, k, k, n);
        dt = nowNs() - t0;
        if (dt < best_ref) best_ref = dt;
    }

    return .{ .vec_ns = best_vec, .ref_ns = best_ref, .max_diff = max_diff };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    _ = init.minimal;

    // Host info: variant string + CPU count (context for the numbers).
    // kt_get_cpu_variant lives in main.zig (not reachable from root.zig),
    // so use cpu_detect directly.
    const cpu: ?root.cpu_detect.CpuInfo = root.cpu_detect.detectCpu(allocator) catch null;
    defer if (cpu != null) {
        var c = cpu.?;
        c.deinit(allocator);
    };

    std.debug.print("ktransformers-zig GEMM bench (B2)\n", .{});
    if (cpu) |c| {
        std.debug.print("  host: {s}\n", .{c.model_name});
        std.debug.print("  variant: {s}\n", .{root.cpu_detect.selectBestVariant(c)});
    }
    std.debug.print("\n", .{});

    const cases = [_]BenchCase{
        // DeepSeek-V3 decode shapes (single token): gate/up is
        // [1, inter] @ [inter, hidden]^T with inter=256 (per-rank
        // 2048/8 in the real model; 256 keeps bench time sane).
        .{ .name = "decode gate/up  [m=1,    n=256, k=7168]", .m = 1, .n = 256, .k = 7168, .flops = 0 },
        // Down projection: [1, hidden] @ [hidden, inter]^T.
        .{ .name = "decode down     [m=1,    n=7168, k=2048]", .m = 1, .n = 7168, .k = 2048, .flops = 0 },
        // Prefill batch: 32 tokens through one expert.
        .{ .name = "prefill batch   [m=32,   n=256, k=7168]", .m = 32, .n = 256, .k = 7168, .flops = 0 },
        // Large square-ish: stresses all tile loops.
        .{ .name = "large square    [m=256,  n=256, k=7168]", .m = 256, .n = 256, .k = 7168, .flops = 0 },
    };

    std.debug.print("{s:<42} {s:>10} {s:>10} {s:>9} {s:>8} {s:>7}\n", .{ "case", "vec ms", "ref ms", "vec GF", "ref GF", "speed" });
    std.debug.print("{s}\n", .{"--------------------------------------------------------------"});

    for (cases) |case| {
        // 2*m*n*k FLOPs per GEMM. Use f64 to avoid overflow at
        // large sizes.
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(case.m)) *
            @as(f64, @floatFromInt(case.n)) * @as(f64, @floatFromInt(case.k));

        const r = try runCase(allocator, case, 5);

        const vec_ms = @as(f64, @floatFromInt(r.vec_ns)) / 1.0e6;
        const ref_ms = @as(f64, @floatFromInt(r.ref_ns)) / 1.0e6;
        const vec_gf = flops / (@as(f64, @floatFromInt(r.vec_ns)) / 1.0e9) / 1.0e9;
        const ref_gf = flops / (@as(f64, @floatFromInt(r.ref_ns)) / 1.0e9) / 1.0e9;
        const speedup = @as(f64, @floatFromInt(r.ref_ns)) / @as(f64, @floatFromInt(r.vec_ns));

        std.debug.print("{s:<42} {d:10.3} {d:10.3} {d:9.2} {d:8.2} {d:6.2}x  (maxdiff {e:.2})\n", .{
            case.name, vec_ms, ref_ms, vec_gf, ref_gf, speedup, r.max_diff,
        });
    }

    std.debug.print("\nvec = gemm_bf16.gemmExpert (A1-vectorized, @Vector(8,f32) K-loop)\n", .{});
    std.debug.print("ref = pure scalar reference (same math, pre-A1 triple for-loop)\n", .{});
    std.debug.print("timings = best (min) of 5 runs, single-call, CLOCK_MONOTONIC\n", .{});
}
