// Per-format benchmark for the K-quant BF16 x quant GEMMs.
//
// Answers the question docs/perf-roadmap.md poses in P0.1: "Target: each
// format reaches >=60% of the BF16 ceiling bandwidth." The ceiling is the
// vectorized BF16 kernel (`gemm_bf16.gemmExpert`) on the SAME shape — it moves
// 3-4x more bytes than a 4-bit format, so "% of ceiling bandwidth" is the only
// cross-format comparable number. Absolute ms on this shared host are noisy;
// the scalar/AVX2 ratio within a row is stable.
//
// Not wired into `zig build bench` (build.zig is a shared file and this file is
// new), so it is run standalone with the same pattern bench/cpp_vs_zig/run.sh
// uses:
//
//   zig build-exe -OReleaseFast -mcpu=native -lc \
//       --dep kt -Mroot=bench/kquant_bench.zig -Mkt=src/root.zig \
//       -femit-bin=/tmp/kquant_bench
//   taskset -c 9 /tmp/kquant_bench     # KT_PIN: pin it, the host is shared

const std = @import("std");
const root = @import("kt");

const amx = root.amx;
const gemm_bf16 = root.gemm_bf16;
const q2_k = root.gemm_q2_k;
const q3_k = root.gemm_q3_k;
const q4_k = root.gemm_q4_k;
const q5_k = root.gemm_q5_k;
const q6_k = root.gemm_q6_k;
const q8_k = root.gemm_q8_k;

const REPS: usize = 9;
const WARMUP: usize = 1;

fn nowNs() u64 {
    var ts: std.os.linux.timespec = .{ .sec = 0, .nsec = 0 };
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

fn randF32(state: *u64) f32 {
    const bits: u32 = @truncate(splitmix64(state) >> 40);
    return @as(f32, @floatFromInt(bits)) * (1.0 / 8388608.0) - 1.0;
}

const Shape = struct { m: usize, n: usize, k: usize };

const shapes = [_]Shape{
    .{ .m = 1, .n = 2048, .k = 7168 }, // LlamaMoe decode gate/up
    .{ .m = 8, .n = 2048, .k = 7168 }, // small batch
};

/// Per-format descriptor: the module, its block type, its quantizer, and both
/// GEMM entry points (scalar reference vs the AVX2 fused one).
fn Fmt(
    comptime fmt_name: []const u8,
    comptime block: type,
    comptime fmt_qk: usize,
    comptime quantize_fn: fn (src: []const f32, dst: [*]block, k: usize) void,
    comptime gemm_scalar_fn: fn ([*]const amx.bf16, usize, [*]const block, usize, [*]f32, usize, usize, usize, usize) void,
    comptime gemm_vec_fn: fn ([*]const amx.bf16, usize, [*]const block, usize, [*]f32, usize, usize, usize, usize) void,
) type {
    return struct {
        const Self = @This();
        pub const name = fmt_name;
        pub const Block = block;
        pub const QK = fmt_qk;
        pub const quantize = quantize_fn;
        pub const gemmScalar = gemm_scalar_fn;
        pub const gemmVec = gemm_vec_fn;
    };
}

const FmtQ8_K = Fmt("Q8_K", q8_k.BlockQ8_K, q8_k.QK_K, q8_k.quantizeRowQ8_K, q8_k.gemmQ8_KScalar, q8_k.gemmQ8_K);
const FmtQ6_K = Fmt("Q6_K", q6_k.BlockQ6_K, q6_k.QK_K, q6_k.quantizeRowQ6_K, q6_k.gemmQ6_KScalar, q6_k.gemmQ6_K);
const FmtQ5_K = Fmt("Q5_K", q5_k.BlockQ5_K, q5_k.QK_K, q5_k.quantizeRowQ5_K, q5_k.gemmQ5_KScalar, q5_k.gemmQ5_K);
const FmtQ4_K = Fmt("Q4_K", q4_k.BlockQ4_K, q4_k.QK_K, q4_k.quantizeRowQ4_K, q4_k.gemmQ4_KScalar, q4_k.gemmQ4_K);
const FmtQ3_K = Fmt("Q3_K", q3_k.BlockQ3_K, q3_k.QK_K, q3_k.quantizeRowQ3_K, q3_k.gemmQ3_KScalar, q3_k.gemmQ3_K);
const FmtQ2_K = Fmt("Q2_K", q2_k.BlockQ2_K, q2_k.QK_K, q2_k.quantizeRowQ2_K, q2_k.gemmQ2_KScalar, q2_k.gemmQ2_K);

fn benchFormat(comptime F: type, alloc: std.mem.Allocator, shape: Shape) !void {
    const Block = F.Block;
    const kb = shape.k / F.QK;

    // Weights from the format's own quantizer, so every plane / scale field is
    // populated (a zeroed block would leave the unpack on a fast path that
    // real weights never take).
    const b = try alloc.alloc(Block, shape.n * kb);
    defer alloc.free(b);
    var rng: u64 = 0x9E3779B97F4A7C15 ^ F.QK;
    var row: [F.QK]f32 = undefined;
    for (0..shape.n) |j| {
        // EVERY block of the row, not just the first: the first cut of this
        // bench left blocks 1..kb-1 uninitialized, which is invisible in the
        // timings (same instruction path) but shows up as NaN in the
        // agreement column — an f16 `d` of 0x7C01 is NaN, and garbage quants
        // hit that often.
        for (0..kb) |sb| {
            for (&row) |*v| v.* = randF32(&rng);
            F.quantize(&row, @ptrCast(&b[j * kb + sb]), F.QK);
        }
    }

    const act = try alloc.alloc(amx.bf16, shape.m * shape.k);
    defer alloc.free(act);
    for (act) |*v| v.* = amx.f32_to_bf16(randF32(&rng));

    const got = try alloc.alloc(f32, shape.m * shape.n);
    defer alloc.free(got);
    const ref = try alloc.alloc(f32, shape.m * shape.n);
    defer alloc.free(ref);

    const scalar_ms = timeMs(F, act, b, got, kb, shape, F.gemmScalar);
    const vec_ms = timeMs(F, act, b, got, kb, shape, F.gemmVec);

    // Agreement check (not a benchmark, but it must run every time: a "fast"
    // kernel that computes the wrong thing is worthless).
    F.gemmScalar(act.ptr, shape.k, b.ptr, kb, ref.ptr, shape.n, shape.m, shape.n, shape.k);
    F.gemmVec(act.ptr, shape.k, b.ptr, kb, got.ptr, shape.n, shape.m, shape.n, shape.k);

    const bytes = @as(f64, @floatFromInt(b.len * @sizeOf(Block)));
    std.debug.print(
        "{s: <5} m={d:<3} scalar {d:>8.2} ms ({d:>5.2} GB/s)   avx2 {d: >7.3} ms ({d: >6.2} GB/s)   {d: >6.2}x   rel-L2 vs scalar {e:.1}\n",
        .{
            F.name,
            shape.m,
            scalar_ms,
            bytes / (scalar_ms * 1.0e-3) / 1.0e9,
            vec_ms,
            bytes / (vec_ms * 1.0e-3) / 1.0e9,
            scalar_ms / vec_ms,
            maxRelDiff(got, ref),
        },
    );
}

fn timeMs(
    comptime F: type,
    act: []const amx.bf16,
    b: []const F.Block,
    out: []f32,
    kb: usize,
    shape: Shape,
    comptime gemm: fn ([*]const amx.bf16, usize, [*]const F.Block, usize, [*]f32, usize, usize, usize, usize) void,
) f64 {
    const call = struct {
        fn go(
            act2: []const amx.bf16,
            b2: []const F.Block,
            o: []f32,
            kb2: usize,
            sh: Shape,
        ) void {
            gemm(act2.ptr, sh.k, b2.ptr, kb2, o.ptr, sh.n, sh.m, sh.n, sh.k);
        }
    }.go;
    for (0..WARMUP) |_| call(act, b, out, kb, shape);
    var best: u64 = std.math.maxInt(u64);
    for (0..REPS) |_| {
        const t0 = nowNs();
        call(act, b, out, kb, shape);
        const dt = nowNs() - t0;
        if (dt < best) best = dt;
    }
    return @as(f64, @floatFromInt(best)) / 1.0e6;
}

/// The BF16 GEMM in the same shape as a `Fmt`, so it can reuse `timeMs`.
const Bf16AsFormat = struct {
    pub const Block = amx.bf16;
    pub fn gemm(act: [*]const amx.bf16, lda: usize, w: [*]const amx.bf16, ldb: usize, out: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
        _ = ldb;
        gemm_bf16.gemmExpert(act, w, out, m, n, k, lda, k, ldc);
    }
};

/// Max |vectorized - scalar| over the whole output, normalized by the L2 norm
/// of the reference output. That normalization is the honest one for a
/// reduction over K terms: with random-sign data a single output element can
/// be ~0 through cancellation while the row is O(sqrt(K)), so a per-element
/// relative error is dominated by whichever element cancelled. Measured
/// against an f64 ground truth at K=7168 this metric is < 1e-6 for both the
/// scalar and the vectorized path.
fn maxRelDiff(got: []const f32, want: []const f32) f32 {
    var norm: f64 = 0;
    for (want) |w| norm += @as(f64, w) * @as(f64, w);
    norm = @sqrt(norm);
    if (norm == 0) return 0;
    var worst: f64 = 0;
    for (got, want) |g, w| worst = @max(worst, @abs(@as(f64, g) - @as(f64, w)));
    return @as(f32, @floatCast(worst / norm));
}

/// The ceiling: the vectorized BF16 kernel on the same shape.
fn benchBf16Ceiling(alloc: std.mem.Allocator, shape: Shape) !void {
    const act = try alloc.alloc(amx.bf16, shape.m * shape.k);
    defer alloc.free(act);
    const w = try alloc.alloc(amx.bf16, shape.n * shape.k);
    defer alloc.free(w);
    const out = try alloc.alloc(f32, shape.m * shape.n);
    defer alloc.free(out);
    var rng: u64 = 0xfeedface;
    for (act) |*v| v.* = amx.f32_to_bf16(randF32(&rng));
    for (w) |*v| v.* = amx.f32_to_bf16(randF32(&rng));
    gemm_bf16.GemmKernel224BF.config();
    // Same timeMs harness, with the BF16 GEMM adapted to its (activation,
    // weight) signature: weight_ld is k, not "blocks".
    const ms = timeMs(Bf16AsFormat, act, w, out, 0, shape, Bf16AsFormat.gemm);
    const bytes = @as(f64, @floatFromInt(w.len * 2));
    std.debug.print(
        "BF16   m={d:<3} (ceiling) {d: >8.3} ms over {d} MB of weights -> {d: >6.2} GB/s\n",
        .{ shape.m, ms, w.len * 2 / (1024 * 1024), bytes / (ms * 1.0e-3) / 1.0e9 },
    );
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    _ = init.minimal;

    const cpu: ?root.cpu_detect.CpuInfo = root.cpu_detect.detectCpu(alloc) catch null;
    defer if (cpu != null) {
        var c = cpu.?;
        c.deinit(alloc);
    };

    std.debug.print("ktransformers-zig K-quant GEMM bench: scalar vs AVX2 (fused)\n", .{});
    if (cpu) |c| std.debug.print("  host: {s}\n", .{c.model_name});
    std.debug.print("  shapes: n=2048 k=7168 (LlamaMoe gate/up); bandwidth is over WEIGHT bytes only\n\n", .{});

    const formats = [_]type{ FmtQ8_K, FmtQ6_K, FmtQ5_K, FmtQ4_K, FmtQ3_K, FmtQ2_K };
    for (shapes) |shape| {
        try benchBf16Ceiling(alloc, shape);
        inline for (formats) |F| try benchFormat(F, alloc, shape);
        std.debug.print("\n", .{});
    }
}