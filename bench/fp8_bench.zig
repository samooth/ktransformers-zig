// FP8 E4M3 GEMM micro-benchmark: the scalar triple loop vs the vectorized
// kernel, plus the two decode experiments that decide where the remaining time
// goes (an asm `vgatherdps`, and a table-free arithmetic decode).
//
// Why this file exists. FP8 is the one weight format in the tree whose decode
// needs a 256-ENTRY table, and the C++ reference's own kernel pays for it with
// `_mm256_i32gather_ps`. docs/perf-roadmap.md P0.2 leaves open whether that
// gather is the ceiling ("An asm vgatherdps is the obvious next experiment"),
// and TODO.md Stream E/Stream F record the same open question for the MX
// kernels, where MXFP8 reaches 3.0 GB/s against Q8_0's table-free 7.8 GB/s at
// the same shape. Answering it needs all three decode strategies in ONE binary
// and ONE run, because absolute ms on this shared host move by up to 3x with
// the load while ratios inside a run are stable.
//
// Not wired into `zig build bench` (build.zig is a shared file), so it runs
// standalone, the same way bench/kquant_bench.zig does:
//
//   zig build-exe -OReleaseFast -mcpu=native -lc \
//       --dep kt -Mroot=bench/fp8_bench.zig -Mkt=src/root.zig \
//       -femit-bin=/tmp/fp8bench
//   taskset -c 9 /tmp/fp8bench     # the host is shared; pin it

const std = @import("std");
const builtin = @import("builtin");
const root = @import("kt");

const amx = root.amx;
const fp8 = root.gemm_fp8;
const q8_0 = root.gemm_q8_0;

const REPS: usize = 9;
const WARMUP: usize = 1;

const Vec = @Vector(8, f32);
const VecU = @Vector(8, u32);

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

// ============================================================================
// The three decode strategies under test. All three are EXACT: each one
// produces the same 256 values as `FP8_E4M3_LUT`, so the agreement column at
// the bottom is what makes the timings comparable.
// ============================================================================

/// (A) SHIPPED: eight scalar L1 table loads, the shape LLVM produces for
/// `kquant_simd`-style `inline for (0..8) |i| out[i] = LUT[raw[i]]`.
fn fp8x8_table(p: [*]const fp8.fp8_e4m3) Vec {
    const raw: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(p)).*;
    var out: [8]f32 = undefined;
    inline for (0..8) |i| out[i] = fp8.FP8_E4M3_LUT[raw[i]];
    return out;
}

/// (B) asm gather — the reference's `_mm256_i32gather_ps` spelled directly.
/// NOTE the AT&T operand order, which is the opposite of Intel's here and is
/// the one thing that can silently break: the DESTINATION is the LAST operand
/// and the mask is the first. Written the Intel way it assembles, executes,
/// and returns zeros, because the mask register is then the uninitialised
/// destination and no lane is enabled.
inline fn gather8(lut: [*]const f32, idx: @Vector(8, i32)) Vec {
    var out: Vec = undefined;
    var mask: @Vector(8, i32) = @splat(-1);
    asm volatile (
        \\vgatherdps %[mask], (%[base], %[idx], 4), %[out]
        : [out] "=&x" (out),
          [mask] "+x" (mask),
        : [base] "r" (lut),
          [idx] "x" (idx),
    );
    return out;
}

fn fp8x8_gather(p: [*]const fp8.fp8_e4m3) Vec {
    const raw: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(p)).*;
    return gather8(&fp8.FP8_E4M3_LUT, @intCast(raw));
}

/// (C) table-free arithmetic decode — 15 integer/FP ops per 8 lanes, no
/// memory beyond the weight bytes themselves.
///
/// Exact because the f32 bit pattern of a normal E4M3 is arithmetic on the
/// byte: exponent field `e + 120` (== `(e - 7) + 127`) and mantissa `m << 20`,
/// so `(b & 0x7F) << 20 + 0x3C000000` IS the value. The two non-normal cases
/// need real work: the seven subnormals are `m * 2^-9` (a convert and a
/// multiply by an exact power of two), and 0x7F maps to 0 like the table does.
/// This is what "no table" would look like for FP8 — the input format rule 3
/// of docs/perf-roadmap.md would otherwise use to reject it.
fn fp8x8_arith(p: [*]const fp8.fp8_e4m3) Vec {
    const raw: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(p)).*;
    const b: VecU = @intCast(raw);
    const mag7 = b & @as(VecU, @splat(0x7F));
    const normal: VecU = (mag7 << @as(@Vector(8, u5), @splat(20))) + @as(VecU, @splat(0x3C000000));
    // man * 2^-9: exact, 2^-9 is representable and man <= 7.
    const manf: Vec = @floatFromInt(mag7 & @as(VecU, @splat(7)));
    const subnormal: VecU = @bitCast(manf * @as(Vec, @splat(0.001953125)));
    const is_sub: @Vector(8, bool) = (mag7 & @as(VecU, @splat(0x78))) == @as(VecU, @splat(0));
    var bits: VecU = @select(u32, is_sub, subnormal, normal);
    bits = @select(u32, mag7 == @as(VecU, @splat(0x7F)), @as(VecU, @splat(0)), bits); // the one NaN
    return @bitCast(bits | ((b & @as(VecU, @splat(0x80))) << @as(@Vector(8, u5), @splat(24))));
}

/// (D) 16 lanes per table lookup instead of 8 — the same code, one vector
/// wide, to separate "table is expensive" from "table is expensive per lane".
inline fn widen8(p: [*]const amx.bf16) Vec {
    return amx.widen8(p);
}

/// Every variant shares this skeleton; only `decode` differs. It is a copy of
/// `GemmKernel224FP8.gemmAvx2` (n-outer / m-inner, MR=4, one @reduce per
/// output, scale folded in after the fold) so the measurement is of the DECODE
/// and not of the loop shape.
fn gemmWith(
    comptime decode: fn ([*]const fp8.fp8_e4m3) Vec,
    a: [*]const amx.bf16, lda: usize,
    b: [*]const fp8.fp8_e4m3, b_scales: [*]const f32, ldb: usize,
    c: [*]f32, ldc: usize,
    m: usize, n: usize, k: usize,
) void {
    const MR: usize = 4;
    const kv = k / 8;
    const rem = k - kv * 8;
    var j: usize = 0;
    while (j < n) : (j += 1) {
        const brow = b + j * ldb;
        var i: usize = 0;
        while (i + MR <= m) : (i += MR) {
            var acc = [_]Vec{@splat(0.0)} ** MR;
            for (0..kv) |v| {
                const w = decode(brow + v * 8);
                const kb = v * 8;
                inline for (0..MR) |r| {
                    acc[r] = @mulAdd(Vec, widen8(a + (i + r) * lda + kb), w, acc[r]);
                }
            }
            inline for (0..MR) |r| {
                var sum = @reduce(.Add, acc[r]);
                if (rem != 0) {
                    const arow = a + (i + r) * lda;
                    for (0..rem) |t| {
                        sum += amx.bf16_to_f32(arow[kv * 8 + t]) * fp8.FP8_E4M3_LUT[brow[kv * 8 + t]];
                    }
                }
                c[(i + r) * ldc + j] = sum * b_scales[j];
            }
        }
        while (i < m) : (i += 1) {
            var acc: Vec = @splat(0.0);
            for (0..kv) |v| {
                acc = @mulAdd(Vec, widen8(a + i * lda + v * 8), decode(brow + v * 8), acc);
            }
            var sum = @reduce(.Add, acc);
            if (rem != 0) {
                const arow = a + i * lda;
                for (0..rem) |t| {
                    sum += amx.bf16_to_f32(arow[kv * 8 + t]) * fp8.FP8_E4M3_LUT[brow[kv * 8 + t]];
                }
            }
            c[i * ldc + j] = sum * b_scales[j];
        }
    }
}

// ============================================================================
// Harness
// ============================================================================

const Shape = struct { m: usize, n: usize, k: usize };

const shapes = [_]Shape{
    .{ .m = 1, .n = 2048, .k = 7168 },
    .{ .m = 4, .n = 2048, .k = 7168 },
};

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

fn maxRelDiff(got: []const f32, want: []const f32) f32 {
    var norm: f64 = 0;
    for (want) |w| norm += @as(f64, w) * @as(f64, w);
    norm = @sqrt(norm);
    if (norm == 0) return 0;
    var worst: f64 = 0;
    for (got, want) |g, w| worst = @max(worst, @abs(@as(f64, g) - @as(f64, w)));
    return @as(f32, @floatCast(worst / norm));
}

const Ctx = struct {
    act: []amx.bf16,
    w: []fp8.fp8_e4m3,
    scales: []f32,
    out: []f32,
    ref: []f32,
    shape: Shape,
};

fn report(ctx: Ctx, label: []const u8, ms: f64, scalar_ms: f64, weight_bytes: usize) void {
    const gb = @as(f64, @floatFromInt(weight_bytes)) / (ms * 1.0e-3) / 1.0e9;
    std.debug.print(
        "  {s: <22} {d: >8.3} ms  {d: >6.2} GB/s  {d: >6.2}x   rel-L2 {e:.1}\n",
        .{ label, ms, gb, scalar_ms / ms, maxRelDiff(ctx.out, ctx.ref) },
    );
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    _ = init.minimal;

    std.debug.print("ktransformers-zig FP8 E4M3 GEMM bench ({s})\n", .{@tagName(builtin.cpu.arch)});
    std.debug.print("  n=2048 k=7168; bandwidth over WEIGHT bytes (1 B/weight)\n", .{});
    std.debug.print("  scalar = the pre-change gemmFullTileScalar triple loop\n\n", .{});

    for (shapes) |shape| {
        const K = shape.k;
        var rng: u64 = 0x1234_5678 ^ @as(u64, @intCast(K));

        const act = try alloc.alloc(amx.bf16, shape.m * K);
        defer alloc.free(act);
        for (act) |*v| v.* = amx.f32_to_bf16(randF32(&rng));

        // Random bytes: every one of the 256 encodings appears ~28k times in a
        // 2048x7168 weight matrix, so subnormals and the 0x7F NaN slot are on
        // the measured path and not just on the correctness path.
        const w = try alloc.alloc(fp8.fp8_e4m3, shape.n * K);
        defer alloc.free(w);
        for (w) |*v| v.* = @truncate(splitmix64(&rng) >> 24);

        const scales = try alloc.alloc(f32, shape.n);
        defer alloc.free(scales);
        for (scales) |*v| v.* = randF32(&rng) * 0.01 + 0.5;

        const out = try alloc.alloc(f32, shape.m * shape.n);
        defer alloc.free(out);
        const ref = try alloc.alloc(f32, shape.m * shape.n);
        defer alloc.free(ref);

        const ctx: Ctx = .{ .act = act, .w = w, .scales = scales, .out = out, .ref = ref, .shape = shape };
        const weight_bytes = w.len;

        std.debug.print("m = {d}\n", .{shape.m});

        const scalar_ms = timeMs(fp8.GemmKernel224FP8.gemmFullTileScalar, .{
            act.ptr, K, w.ptr, scales.ptr, K, out.ptr, shape.n, shape.m, shape.n, K,
        });
        // Freeze the oracle before any other variant overwrites `out`.
        fp8.GemmKernel224FP8.gemmFullTileScalar(act.ptr, K, w.ptr, scales.ptr, K, ref.ptr, shape.n, shape.m, shape.n, K);
        report(ctx, "gemmFullTileScalar", scalar_ms, scalar_ms, weight_bytes);

        // (A) the shipped kernel
        const table_ms = timeMs(fp8.GemmKernel224FP8.gemmAvx2, .{
            act.ptr, K, w.ptr, scales.ptr, K, out.ptr, shape.n, shape.m, shape.n, K,
        });
        fp8.GemmKernel224FP8.gemmAvx2(act.ptr, K, w.ptr, scales.ptr, K, out.ptr, shape.n, shape.m, shape.n, K);
        report(ctx, "gemmAvx2 (table loads)", table_ms, scalar_ms, weight_bytes);

        // (B) asm gather
        const gather_ms = timeMs(gemmWith, .{ fp8x8_gather, act.ptr, K, w.ptr, scales.ptr, K, out.ptr, shape.n, shape.m, shape.n, K });
        gemmWith(fp8x8_gather, act.ptr, K, w.ptr, scales.ptr, K, out.ptr, shape.n, shape.m, shape.n, K);
        report(ctx, "gemm + asm vgatherdps", gather_ms, scalar_ms, weight_bytes);

        // (C) table-free arithmetic
        const arith_ms = timeMs(gemmWith, .{ fp8x8_arith, act.ptr, K, w.ptr, scales.ptr, K, out.ptr, shape.n, shape.m, shape.n, K });
        gemmWith(fp8x8_arith, act.ptr, K, w.ptr, scales.ptr, K, out.ptr, shape.n, shape.m, shape.n, K);
        report(ctx, "gemm + arithmetic", arith_ms, scalar_ms, weight_bytes);

        // (D) skeleton-only control: decode replaced by a splat. Bounds how
        // much of the time is the GEMM loop itself rather than the decode.
        const noop_ms = timeMs(gemmWith, .{ fp8x8_noop, act.ptr, K, w.ptr, scales.ptr, K, out.ptr, shape.n, shape.m, shape.n, K });
        report(ctx, "gemm, no decode (floor)", noop_ms, scalar_ms, weight_bytes);

        // (E) The table-free control the roadmap quotes: Q8_0's AVX2 GEMM, whose
        // i8 -> f32 step is pure `vpmovsxbd` + `vcvtdq2ps` with no lookup at
        // all. It moves 34 B per 32 weights (1.0625 B/weight) against FP8's
        // 1.0, so its GB/s is directly comparable and it is the number that
        // decides whether the LUT is the bottleneck.
        {
            const kb = K / q8_0.QK8_0;
            const qb = try alloc.alloc(q8_0.BlockQ8_0, shape.n * kb);
            defer alloc.free(qb);
            var qrng: u64 = 0xfeed_face;
            var row: [q8_0.QK8_0]f32 = undefined;
            for (0..shape.n) |j| {
                for (0..kb) |sb| {
                    for (&row) |*v| v.* = randF32(&qrng);
                    q8_0.quantizeRowQ8_0(row[0..], @ptrCast(&qb[j * kb + sb]), q8_0.QK8_0);
                }
            }
            const qms = timeMs(q8_0.gemmAvx2, .{
                act.ptr, K, qb.ptr, kb, out.ptr, shape.n, shape.m, shape.n, K,
            });
            const qbytes = qb.len * @sizeOf(q8_0.BlockQ8_0);
            std.debug.print(
                "  {s: <22} {d: >8.3} ms  {d: >6.2} GB/s  (table-free control, {d:.3} B/weight)\n",
                .{ "Q8_0 gemmAvx2", qms, @as(f64, @floatFromInt(qbytes)) / (qms * 1.0e-3) / 1.0e9, 34.0 / 32.0 },
            );
        }

        std.debug.print("\n", .{});
    }
}

fn fp8x8_noop(p: [*]const fp8.fp8_e4m3) Vec {
    const raw: @Vector(8, u8) = @as(*align(1) const @Vector(8, u8), @ptrCast(p)).*;
    const idx: VecU = @intCast(raw);
    return @floatFromInt(idx & @as(VecU, @splat(3)));
}