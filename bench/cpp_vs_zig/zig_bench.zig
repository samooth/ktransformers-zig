// Zig side of the C++ <-> Zig kernel comparison. Drives the SAME shapes as
// bench/cpp_vs_zig/cpp_bench.cpp through the Zig port's kernels:
//   src/kernels/amx/gemm_224_bf16.zig  -> gemmExpert, GemmKernel224BF.applySwiGLU
//
// Output format (one line per case), consumed by run.sh:
//   GEMM|<name>|<ms>|<GFLOPs>
//   ACT|<name>|<ms>|<GB/s>
//   GEMM|<name>|<ms>|<GB/s>      <- INT4 rows; the RATE COLUMN IS BYTES, and
//                                   measure.zig's absolute mode prints it as-is
//                                   with its band. Keep that in mind before
//                                   adding a row whose rate is not GFLOP/s:
//                                   the unit lives in the harness, not here.

const std = @import("std");
const root = @import("kt");

const gemm_bf16 = root.gemm_bf16;
const gemm_int4 = root.gemm_int4;
const amx = root.amx;

const KT_HIDDEN: usize = 7168;
const KT_INTER: usize = 2048;

const Shape = struct { name: []const u8, m: usize, n: usize, k: usize };

const gemm_shapes = [_]Shape{
    .{ .name = "decode gate/up", .m = 1, .n = KT_INTER, .k = KT_HIDDEN },
    .{ .name = "decode down", .m = 1, .n = KT_HIDDEN, .k = KT_INTER },
    .{ .name = "small decode", .m = 8, .n = KT_INTER, .k = KT_HIDDEN },
    .{ .name = "prefill 32", .m = 32, .n = KT_INTER, .k = KT_HIDDEN },
    .{ .name = "prefill 128", .m = 128, .n = KT_INTER, .k = KT_HIDDEN },
    .{ .name = "prefill 512", .m = 512, .n = KT_INTER, .k = KT_HIDDEN },
};

const ACT_M: usize = 128;
const ACT_N: usize = KT_INTER;

/// INT4 GPTQ shapes. m = 1 and m = 4 only: this row exists to be an ABSOLUTE
/// measurement (see measure.zig), and it is a bandwidth-bound decode either way,
/// so more m values would cost pass time without changing what it reports.
///
/// INT4 IS THE FIRST ROW HERE WITH i8 ACTIVATIONS, and that is the point of it,
/// not an incidental detail. Every other GEMM row is bf16 activations, which is
/// exactly what `kquant_simd.gemmBF16xQuant` hardcodes (`widen16`) — so the
/// tree currently has a "shared driver" that cannot express INT4 or the INT8
/// path at all. This row is the evidence for why the shared driver must expose
/// a decode slot AND a widening slot rather than be extended until it fits every
/// format: the two things that make INT4 different from the rows above are
/// i8 activations and a per-GROUP scale, and both are outside the driver's
/// signature today.
const int4_shapes = [_]Shape{
    .{ .name = "int4 gptq", .m = 1, .n = KT_INTER, .k = KT_HIDDEN },
    .{ .name = "int4 gptq m4", .m = 4, .n = KT_INTER, .k = KT_HIDDEN },
};

const WARMUP: usize = 2;
const REPS: usize = 9;

fn nowNanos() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.syscall2(.clock_gettime, @intFromEnum(std.os.linux.CLOCK.MONOTONIC), @intFromPtr(&ts));
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn splitmix64(state: *u64) u64 {
    var z = state.* +% 0x9E3779B97F4A7C15;
    state.* = z;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Uniform f32 in [-1, 1) — byte-identical to kt_rand_f32 in common.h.
fn randF32(state: *u64) f32 {
    const bits: u32 = @truncate(splitmix64(state) >> 40);
    return @as(f32, @floatFromInt(bits)) * (1.0 / 8388608.0) - 1.0;
}

/// ggml-impl.h round-to-nearest-even f32->bf16. Used ONLY to generate inputs so
/// both harnesses see identical data; the kernels use amx.f32_to_bf16.
fn f32ToBf16Rne(x: f32) amx.bf16 {
    const u: u32 = @bitCast(x);
    return @truncate((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

fn writeRaw(alloc: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    var io = std.Io.Threaded.init(alloc, .{ .environ = std.process.Environ.empty });
    defer io.deinit();
    const file = try std.Io.Dir.cwd().createFile(io.io(), path, .{ .read = false, .truncate = true });
    defer file.close(io.io());
    try std.Io.File.writePositionalAll(file, io.io(), bytes, 0);
}

fn benchGemm(alloc: std.mem.Allocator, outdir: []const u8, s: Shape) !void {
    var rng: u64 = 0x243F6A8885A308D3;
    const a = try alloc.alloc(amx.bf16, s.m * s.k);
    const b = try alloc.alloc(amx.bf16, s.n * s.k);
    for (a) |*v| v.* = f32ToBf16Rne(randF32(&rng));
    for (b) |*v| v.* = f32ToBf16Rne(randF32(&rng));
    const c = try alloc.alloc(f32, s.m * s.n);

    gemm_bf16.GemmKernel224BF.config();

    // gemmExpert ACCUMULATES into output (+=), the C++ reference assigns. The
    // memset is outside the timed region on both sides.
    for (0..WARMUP) |_| {
        @memset(c, 0);
        gemm_bf16.gemmExpert(a.ptr, b.ptr, c.ptr, s.m, s.n, s.k, s.k, s.k, s.n);
    }
    var best: u64 = std.math.maxInt(u64);
    for (0..REPS) |_| {
        @memset(c, 0);
        const t0 = nowNanos();
        gemm_bf16.gemmExpert(a.ptr, b.ptr, c.ptr, s.m, s.n, s.k, s.k, s.k, s.n);
        const ns = nowNanos() - t0;
        if (ns < best) best = ns;
    }

    const path = try std.fmt.allocPrint(alloc, "{s}/out_zig_gemm_{d}_{d}_{d}.f32", .{ outdir, s.m, s.n, s.k });
    defer alloc.free(path);
    try writeRaw(alloc, path, std.mem.sliceAsBytes(c));

    const ms = @as(f64, @floatFromInt(best)) / 1e6;
    const gflops = 2.0 * @as(f64, @floatFromInt(s.m)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k)) / (ms * 1e6);
    const label = try std.fmt.allocPrint(alloc, "{s} [m={d},n={d},k={d}]", .{ s.name, s.m, s.n, s.k });
    std.debug.print("GEMM|{s}|{d:.4}|{d:.2}\n", .{ label, ms, gflops });
}

/// INT4 GPTQ GEMM, reported as ABSOLUTE throughput over weight bytes.
///
/// No output dump: `compare.py` diffs the `out_zig_gemm_*` dumps against the C++
/// harness's, and there is no C++ row to diff against — writing a dump nothing
/// reads would suggest a comparison that does not exist.
fn benchInt4Gemm(alloc: std.mem.Allocator, s: Shape) !void {
    const kb = s.k / 32;

    var rng: u64 = 0x9E3779B97F4A7C15;
    const act = try alloc.alloc(i8, s.m * s.k);
    for (act) |*v| {
        v.* = @bitCast(@as(u8, @truncate(splitmix64(&rng) >> 24)));
    }
    const w = try alloc.alloc(gemm_int4.BlockQ4_0, s.n * kb);
    for (w) |*blk| {
        // Complete blocks: every one of the 16 codes appears (the zero point 8
        // and both nibble halves), and a bf16 scale of realistic magnitude.
        blk.d = amx.f32_to_bf16(@as(f32, @floatFromInt(@as(i32, @intCast(splitmix64(&rng) % 1024)) + 1)) * 0.0078125);
        for (&blk.qs) |*q| q.* = @truncate(splitmix64(&rng) >> 24);
    }
    const c = try alloc.alloc(f32, s.m * s.n);

    const a_ptr: *const i8 = @ptrCast(act.ptr);
    const b_ptr: *const gemm_int4.BlockQ4_0 = @ptrCast(w.ptr);

    for (0..WARMUP) |_| gemm_int4.GemmKernel224Int4.gemmAvx2(a_ptr, s.k, b_ptr, kb, c.ptr, s.n, s.m, s.n, s.k);
    var best: u64 = std.math.maxInt(u64);
    for (0..REPS) |_| {
        const t0 = nowNanos();
        gemm_int4.GemmKernel224Int4.gemmAvx2(a_ptr, s.k, b_ptr, kb, c.ptr, s.n, s.m, s.n, s.k);
        const ns = nowNanos() - t0;
        if (ns < best) best = ns;
    }

    const ms = @as(f64, @floatFromInt(best)) / 1e6;
    const weight_bytes = w.len * @sizeOf(gemm_int4.BlockQ4_0);
    const gbs = @as(f64, @floatFromInt(weight_bytes)) / (ms * 1e-3) / 1e9;
    const label = try std.fmt.allocPrint(alloc, "{s} [m={d},n={d},k={d}]", .{ s.name, s.m, s.n, s.k });
    std.debug.print("GEMM|{s}|{d:.4}|{d:.2}\n", .{ label, ms, gbs });
}

fn benchAct(alloc: std.mem.Allocator, outdir: []const u8) !void {
    var rng: u64 = 0x13198A2E03707344;
    const g = try alloc.alloc(amx.bf16, ACT_M * ACT_N);
    const u = try alloc.alloc(amx.bf16, ACT_M * ACT_N);
    const d = try alloc.alloc(amx.bf16, ACT_M * ACT_N);
    for (g) |*v| v.* = f32ToBf16Rne(randF32(&rng));
    for (u) |*v| v.* = f32ToBf16Rne(randF32(&rng));

    for (0..WARMUP) |_|
        gemm_bf16.GemmKernel224BF.applySwiGLU(g.ptr, u.ptr, d.ptr, ACT_M, ACT_N, 0, 0);

    var best: u64 = std.math.maxInt(u64);
    for (0..REPS) |_| {
        const t0 = nowNanos();
        gemm_bf16.GemmKernel224BF.applySwiGLU(g.ptr, u.ptr, d.ptr, ACT_M, ACT_N, 0, 0);
        const ns = nowNanos() - t0;
        if (ns < best) best = ns;
    }

    const path = try std.fmt.allocPrint(alloc, "{s}/out_zig_act_{d}_{d}.bf16", .{ outdir, ACT_M, ACT_N });
    defer alloc.free(path);
    try writeRaw(alloc, path, std.mem.sliceAsBytes(d));

    const ms = @as(f64, @floatFromInt(best)) / 1e6;
    const gbs = @as(f64, @floatFromInt(ACT_M * ACT_N)) * 2 * 2 / (ms * 1e6);
    const label = try std.fmt.allocPrint(alloc, "swiglu act [m={d},n={d}]", .{ ACT_M, ACT_N });
    std.debug.print("ACT|{s}|{d:.4}|{d:.2}\n", .{ label, ms, gbs });
}

pub fn main(init: std.process.Init) !void {
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    const alloc = arena_state.allocator();
    const args = try init.minimal.args.toSlice(alloc);
    const outdir = if (args.len > 1) args[1] else ".";

    for (gemm_shapes) |s| try benchGemm(alloc, outdir, s);
    for (int4_shapes) |s| try benchInt4Gemm(alloc, s);
    try benchAct(alloc, outdir);
}
