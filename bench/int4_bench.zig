// INT4 GPTQ GEMM micro-benchmark: the scalar triple loop vs the vectorized
// kernel, plus the two things that decide whether this format is worth more
// work (a decode-free floor, and Q8_0 as the table-free control at almost the
// same bytes-per-weight).
//
// Why standalone: `build.zig` is a shared file, and this is a new file. Same
// pattern as bench/kquant_bench.zig and bench/fp8_bench.zig.
//
//   zig build-exe -OReleaseFast -mcpu=native -lc --dep kt \
//       -Mroot=bench/int4_bench.zig -Mkt=src/root.zig -femit-bin=/tmp/int4bench
//   taskset -c 9 /tmp/int4bench
//
// The host is shared and absolute ms move by up to 3x with the load, so the
// scalar/AVX2 ratio inside one run is the number to read, never the ms.

const std = @import("std");
const root = @import("kt");

const amx = root.amx;
const g4 = root.gemm_int4;
const q8_0 = root.gemm_q8_0;
const K4 = g4.GemmKernel224Int4;

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

/// Full-range i8, which is the only activation distribution that exercises the
/// -128 end and therefore the sign extension. The GEMM is exact in f32 for
/// ordinary data, so this is also the shape where the two paths stop agreeing
/// bit-for-bit (the integer sums leave f32's exact range) - the agreement
/// column is reported for exactly that reason.
fn randI8(state: *u64) i8 {
    return @bitCast(@as(u8, @truncate(splitmix64(state) >> 24)));
}

fn randF32(state: *u64) f32 {
    const bits: u32 = @truncate(splitmix64(state) >> 40);
    return @as(f32, @floatFromInt(bits)) * (1.0 / 8388608.0) - 1.0;
}

const Shape = struct { m: usize, n: usize, k: usize };

const shapes = [_]Shape{
    .{ .m = 1, .n = 2048, .k = 7168 }, // decode
    .{ .m = 4, .n = 2048, .k = 7168 }, // small batch
};

/// Max |avx2 - scalar| over the output, normalized by the L2 norm of the scalar
/// output. The honest metric for a reduction over K terms: with random-sign
/// data one element can be ~0 through cancellation while the row is O(sqrt(K)).
fn maxRelDiff(got: []const f32, want: []const f32) f32 {
    var norm: f64 = 0;
    for (want) |w| norm += @as(f64, w) * @as(f64, w);
    norm = @sqrt(norm);
    if (norm == 0) return 0;
    var worst: f64 = 0;
    for (got, want) |gv, wv| worst = @max(worst, @abs(@as(f64, gv) - @as(f64, wv)));
    return @as(f32, @floatCast(worst / norm));
}

fn timeMs(comptime call: anytype, args: anytype) f64 {
    for (0..WARMUP) |_| @call(.auto, call, args);
    var best: u64 = std.math.maxInt(u64);
    for (0..REPS) |_| {
        const t0 = nowNs();
        @call(.auto, call, args);
        const dt = nowNs() - t0;
        if (dt < best) best = dt;
    }
    return @as(f64, @floatFromInt(best)) / 1.0e6;
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    _ = init.minimal;

    std.debug.print("ktransformers-zig INT4 GPTQ GEMM bench\n", .{});
    std.debug.print("  n=2048 k=7168; bandwidth over WEIGHT bytes ({d} per block / 32 weights)\n", .{@sizeOf(g4.BlockQ4_0)});
    std.debug.print("  scalar = gemmFullTileScalar, the pre-change triple loop\n", .{});
    std.debug.print("  Q8_0 rides along as the ~1 B/weight control (34 B / 32 weights)\n\n", .{});

    for (shapes) |shape| {
        const K = shape.k;
        const kb = K / 32;
        var rng: u64 = 0x1357_9BDF ^ @as(u64, @intCast(K));

        const act = try alloc.alloc(i8, shape.m * K);
        defer alloc.free(act);
        for (act) |*v| v.* = randI8(&rng);

        // Complete blocks: random packed bytes, so all 16 codes appear (the
        // zero point 8 and both nibble halves included), and a per-group bf16
        // scale of realistic magnitude. The format's own quantizer is a method
        // on `Int4BufferB` and would leave the packed bytes correlated with the
        // scale, which is not what the timing should measure.
        const w = try alloc.alloc(g4.BlockQ4_0, shape.n * kb);
        defer alloc.free(w);
        for (w) |*blk| {
            blk.d = amx.f32_to_bf16(@as(f32, @floatFromInt(@as(i32, @intCast(splitmix64(&rng) % 1024)) + 1)) * 0.0078125);
            for (&blk.qs) |*q| q.* = @truncate(splitmix64(&rng) >> 24);
        }

        const out = try alloc.alloc(f32, shape.m * shape.n);
        defer alloc.free(out);
        const ref = try alloc.alloc(f32, shape.m * shape.n);
        defer alloc.free(ref);

        const a_ptr: *const i8 = @ptrCast(act.ptr);
        const b_ptr: *const g4.BlockQ4_0 = @ptrCast(w.ptr);
        const weight_bytes = w.len * @sizeOf(g4.BlockQ4_0);

        std.debug.print("m = {d}\n", .{shape.m});

        const scalar_ms = timeMs(K4.gemmFullTileScalar, .{ a_ptr, K, b_ptr, kb, out.ptr, shape.n, shape.m, shape.n, K });
        K4.gemmFullTileScalar(a_ptr, K, b_ptr, kb, ref.ptr, shape.n, shape.m, shape.n, K);
        std.debug.print(
            "  {s: <26} {d: >8.2} ms  {d: >6.2} GB/s  {d: >6.2}x   rel-L2 {e:.1}\n",
            .{ "gemmFullTileScalar", scalar_ms, @as(f64, @floatFromInt(weight_bytes)) / (scalar_ms * 1e-3) / 1e9, 1.0, maxRelDiff(ref, ref) },
        );

        const avx2_ms = timeMs(K4.gemmAvx2, .{ a_ptr, K, b_ptr, kb, out.ptr, shape.n, shape.m, shape.n, K });
        K4.gemmAvx2(a_ptr, K, b_ptr, kb, out.ptr, shape.n, shape.m, shape.n, K);
        std.debug.print(
            "  {s: <26} {d: >8.3} ms  {d: >6.2} GB/s  {d: >6.2}x   rel-L2 {e:.1}\n",
            .{ "gemmAvx2 (shipped)", avx2_ms, @as(f64, @floatFromInt(weight_bytes)) / (avx2_ms * 1e-3) / 1e9, scalar_ms / avx2_ms, maxRelDiff(out, ref) },
        );

        // Q8_0 control: 1.0625 B/weight against INT4's 0.5625, no lookup table
        // in either. What it costs is the loop and the memory system at the
        // same shape, so the gap between the two rows is the format, not the
        // harness.
        {
            const qkb = K / q8_0.QK8_0;
            const qb = try alloc.alloc(q8_0.BlockQ8_0, shape.n * qkb);
            defer alloc.free(qb);
            var qsrc: [q8_0.QK8_0]f32 = undefined;
            for (0..shape.n) |j| {
                for (0..qkb) |g| {
                    for (&qsrc) |*v| v.* = randF32(&rng);
                    q8_0.quantizeRowQ8_0(qsrc[0..], @ptrCast(&qb[j * qkb + g]), q8_0.QK8_0);
                }
            }
            // Q8_0 takes bf16 activations, so it needs its own copy.
            const bact = try alloc.alloc(amx.bf16, shape.m * K);
            defer alloc.free(bact);
            var arng: u64 = 0x2468_ACE0 ^ @as(u64, @intCast(K));
            for (bact) |*v| v.* = amx.f32_to_bf16(@as(f32, @floatFromInt(randI8(&arng))));
            const ms = timeMs(q8_0.gemmAvx2, .{ bact.ptr, K, qb.ptr, qkb, out.ptr, shape.n, shape.m, shape.n, K });
            const bytes = qb.len * @sizeOf(q8_0.BlockQ8_0);
            std.debug.print(
                "  {s: <26} {d: >8.3} ms  {d: >6.2} GB/s  ({d:.3} B/weight control)\n",
                .{ "Q8_0 gemmAvx2", ms, @as(f64, @floatFromInt(bytes)) / (ms * 1e-3) / 1e9, 34.0 / 32.0 },
            );
        }

        std.debug.print("\n", .{});
    }
}
