// CLAIM: swiglu-regalloc — IQ anchor bench (IQ2_XXS, IQ2_XS) — 2026-10-04
// — holds bench/cpp_vs_zig/ until committed. Touches no build.zig step.

// bench/cpp_vs_zig/iq_anchor_bench.zig — the two IQ anchor formats, measured.
//
// SCOPE, and why it is only two formats
//
// The C++ reference has no IQ or K-quant AVX2 kernel at all — `operators/avx2/`
// is BF16, FP8, MXFP4, MXFP8, GPTQ-INT4 and raw-INT4, and nothing else. So
// there is no second implementation to diff against and no A/B ratio is
// possible for any IQ format. This bench therefore reports ABSOLUTE GB/s with a
// measured noise floor beside it, which is the existing convention for every
// quantized format in this repo (Q8_0 at 7.8 GB/s, IQ4_XS at ~2.6), not a
// weakened bar.
//
// It measures only IQ2_XXS and IQ2_XS because those two are the *anchors* for
// Stream K's estimate that the unmeasured grid formats (IQ2_S, IQ3_XXS, IQ3_S,
// IQ1_S, IQ1_M) "should land in the same range". His anchors are 1.26 GB/s and
// 1.06 GB/s — a 1.19x spread — and each was measured on a host at load 13-51
// from a "~3 ms" timing. A 1.19x band against a ~1.17x floor is not a band. Until
// the anchors are re-measured on a quiet box, "do the five land in 1.06-1.26?" is
// a question with almost no power to reject anything. So the anchors come first,
// alone; the five follow only if this pass says the interval is real.
//
// SETUP IS ~1700x THE MEASUREMENT, AND THAT IS THE TRAP
//
// Measured on this host, at the bench shape (n=2048, k=7168, 57344 blocks):
//
//     initIq2XsData512          249 ms   once per process (process-cached)
//     quantize all IQ2_XXS     5147 ms   once per process
//     gemmAvx2 at m=1          ~3 ms     what we are trying to measure
//
// So anything that quantizes inside the timed region measures the quantizer.
// Everything below is built once, before timing, and the timed region contains
// exactly one `gemmAvx2` call and a clock read. The ~5.5 s of setup is paid per
// child process and is wall-clock cost, not measurement contamination — but it
// is why the harness's 4 arms x N iterations adds up, and why this bench does
// both formats in ONE process.
//
// BLOCKS ARE REAL, NOT SYNTHESISED
//
// Every block comes from the format's own quantizer on random data, with the
// runtime kmap/kneighbors grid the quantizer requires. That matters: a zeroed or
// randomly-byte-filled block leaves the unpack on a path real weights never take,
// which is the mistake the K-quant bench documents. `quantizeRowIQ2_XXS` in
// gemm_224_iq2_xxs.zig is a @panic stub next to a working
// `quantizeRowIQ2_XXS_WithInit` in iq2_quantize.zig; this uses the latter.
//
// Agreement between `gemmAvx2` and the format's scalar reference is checked on
// every run and printed. A "fast" kernel on blocks that are not representative,
// or that computes the wrong thing, is worth nothing — so the check is part of
// the output, not a comment.
//
// Run it through the control-armed harness, which is the point:
//
//   zig build-exe -OReleaseFast -mcpu=native -lc --cache-dir /tmp/kc \
//       --dep kt -Mroot=bench/cpp_vs_zig/iq_anchor_bench.zig -Mkt=src/root.zig \
//       -femit-bin=/tmp/iq_anchor
//   zig run bench/cpp_vs_zig/measure.zig -- --iters 3 --pin 9 \
//       --cpp /tmp/iq_anchor --zig /tmp/iq_anchor
//
// Both arms are the SAME binary, so every row must come back UNMEASURABLE — that
// run is the noise floor for these shapes, not a speedup measurement. To see the
// absolute numbers, run the bench directly.

const std = @import("std");
const root = @import("kt");
const amx = root.amx;

const N: usize = 2048; // LlamaMoe decode gate/up
const K: usize = 7168;
const QK: usize = 256;
const NB: usize = K / QK;

const REPS: usize = 9;
const WARMUP: usize = 2;

fn nowNs() u64 {
    var ts: std.os.linux.timespec = .{ .sec = 0, .nsec = 0 };
    _ = std.os.linux.syscall2(.clock_gettime, @intFromEnum(std.os.linux.CLOCK.MONOTONIC), @intFromPtr(&ts));
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn randF32(state: *u64) f32 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    const bits: u32 = @truncate(state.* >> 40);
    return @as(f32, @floatFromInt(bits)) / 16777216.0 - 0.5;
}

/// Row of activations, shared by every case so the timed region differs only in
/// the weight format.
const Acts = struct {
    m1: []amx.bf16,
    m8: []amx.bf16,
};

fn makeActs(alloc: std.mem.Allocator) !Acts {
    var rng: u64 = 0x5EED_1234;
    const a8 = try alloc.alloc(amx.bf16, 8 * K);
    for (a8) |*v| v.* = amx.f32_to_bf16(randF32(&rng));
    const a1 = try alloc.alloc(amx.bf16, K);
    @memcpy(a1, a8[0..K]);
    return .{ .m1 = a1, .m8 = a8 };
}

fn timeAvx2(comptime g: anytype, act: []const amx.bf16, b: anytype, c: []f32, m: usize) u64 {
    var best: u64 = std.math.maxInt(u64);
    for (0..REPS + WARMUP) |r| {
        const t0 = nowNs();
        g(act.ptr, b, c.ptr, m, N, K, K, NB, N);
        const dt = nowNs() - t0;
        if (r >= WARMUP and dt < best) best = dt;
    }
    return best;
}

fn timeScalar(comptime g: anytype, act: []const amx.bf16, b: anytype, c: []f32, m: usize) void {
    g(act.ptr, b, c.ptr, m, N, K, K, NB, N);
}

fn relL2(got: []const f32, ref: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, ref) |a, b| {
        const d = @as(f64, a) - @as(f64, b);
        num += d * d;
        den += @as(f64, b) * @as(f64, b);
    }
    if (den == 0) return 0;
    return @sqrt(num / den);
}

pub fn main(init: std.process.Init) !void {
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    const acts = try makeActs(alloc);
    const out1 = try alloc.alloc(f32, N);
    const out8 = try alloc.alloc(f32, 8 * N);
    const ref1 = try alloc.alloc(f32, N);
    const ref8 = try alloc.alloc(f32, 8 * N);

    // ---- SETUP: init + quantize everything, once, OUTSIDE all timing -------
    // ~5.5 s per format on this host. See the header for why this must not be
    // inside a timed region.

    // IQ2_XXS uses the 256-entry grid, IQ2_XS the 512-entry one. Same function
    // name pattern, different grid — initIq2XsData512 is NOT interchangeable
    // with initIq2XsData.
    // Per-format setup timing. Timing both formats together and attributing the
    // total to each is wrong and inflates every ratio by ~2x — which is exactly
    // what the first version of this did.
    const t_xxs0 = nowNs();
    const d_xxs = root.iq2xs_init.initIq2XsData(alloc);

    var rng: u64 = 0xABCD_0001;
    const rowbuf = try alloc.alloc(f32, K);
    for (rowbuf) |*v| v.* = randF32(&rng);

    const b_xxs = try alloc.alloc(root.gemm_iq2_xxs.BlockIQ2_XXS, N * NB);
    for (0..N) |j| root.iq2_quantize.quantizeRowIQ2_XXS_WithInit(d_xxs, rowbuf, @ptrCast(&b_xxs[j * NB]), K, null);
    const setup_xxs = nowNs() - t_xxs0;

    const t_xs0 = nowNs();
    const d_xs = root.iq2xs_init.initIq2XsData512(alloc);
    const b_xs = try alloc.alloc(root.gemm_iq2_xs.BlockIQ2_XS, N * NB);
    for (0..N) |j| root.gemm_iq2_xs.quantizeRowIQ2_XS_WithInit(d_xs, rowbuf, @ptrCast(&b_xs[j * NB]), K, null);
    const setup_xs = nowNs() - t_xs0;

    // ---- IQ2_XXS --------------------------------------------------------
    {
        const bytes = @as(f64, @floatFromInt(b_xxs.len * @sizeOf(root.gemm_iq2_xxs.BlockIQ2_XXS)));
        // Agreement first: if the blocks were not representative the timings
        // would be meaningless, so this is checked before anything is reported.
        timeScalar(root.gemm_iq2_xxs.gemmIQ2_XXSScalar, acts.m1, b_xxs.ptr, ref1, 1);
        root.gemm_iq2_xxs.gemmAvx2(acts.m1.ptr, b_xxs.ptr, out1.ptr, 1, N, K, K, NB, N);
        const rel1 = relL2(out1, ref1);
        timeScalar(root.gemm_iq2_xxs.gemmIQ2_XXSScalar, acts.m8, b_xxs.ptr, ref8, 8);
        root.gemm_iq2_xxs.gemmAvx2(acts.m8.ptr, b_xxs.ptr, out8.ptr, 8, N, K, K, NB, N);
        const rel8 = relL2(out8, ref8);

        const t1 = timeAvx2(root.gemm_iq2_xxs.gemmAvx2, acts.m1, b_xxs.ptr, out1, 1);
        const t8 = timeAvx2(root.gemm_iq2_xxs.gemmAvx2, acts.m8, b_xxs.ptr, out8, 8);
        const ms1 = @as(f64, @floatFromInt(t1)) / 1e6;
        const ms8 = @as(f64, @floatFromInt(t8)) / 1e6;
        std.debug.print("GEMM|IQ2_XXS [m=1,n={d},k={d}]|{d:.4}|{d:.3}\n", .{ N, K, ms1, bytes / (ms1 * 1e-3) / 1e9 });
        std.debug.print("GEMM|IQ2_XXS [m=8,n={d},k={d}]|{d:.4}|{d:.3}\n", .{ N, K, ms8, bytes / (ms8 * 1e-3) / 1e9 });
        std.debug.print("  IQ2_XXS agreement vs scalar: m=1 rel-L2 {d:.2}  m=8 rel-L2 {d:.2}\n", .{ rel1, rel8 });
        // Setup is per-format, not per-m, so both rows get the same figure.
        const sm = @as(f64, @floatFromInt(setup_xxs)) / 1e6;
        std.debug.print("SETUP|IQ2_XXS [m=1,n={d},k={d}]|{d:.2}\n", .{ N, K, sm });
        std.debug.print("SETUP|IQ2_XXS [m=8,n={d},k={d}]|{d:.2}\n", .{ N, K, sm });
    }

    // ---- IQ2_XS ---------------------------------------------------------
    {
        const bytes = @as(f64, @floatFromInt(b_xs.len * @sizeOf(root.gemm_iq2_xs.BlockIQ2_XS)));
        timeScalar(root.gemm_iq2_xs.gemmIQ2_XSScalar, acts.m1, b_xs.ptr, ref1, 1);
        root.gemm_iq2_xs.gemmAvx2(acts.m1.ptr, b_xs.ptr, out1.ptr, 1, N, K, K, NB, N);
        const rel1 = relL2(out1, ref1);
        timeScalar(root.gemm_iq2_xs.gemmIQ2_XSScalar, acts.m8, b_xs.ptr, ref8, 8);
        root.gemm_iq2_xs.gemmAvx2(acts.m8.ptr, b_xs.ptr, out8.ptr, 8, N, K, K, NB, N);
        const rel8 = relL2(out8, ref8);

        const t1 = timeAvx2(root.gemm_iq2_xs.gemmAvx2, acts.m1, b_xs.ptr, out1, 1);
        const t8 = timeAvx2(root.gemm_iq2_xs.gemmAvx2, acts.m8, b_xs.ptr, out8, 8);
        const ms1 = @as(f64, @floatFromInt(t1)) / 1e6;
        const ms8 = @as(f64, @floatFromInt(t8)) / 1e6;
        std.debug.print("GEMM|IQ2_XS [m=1,n={d},k={d}]|{d:.4}|{d:.3}\n", .{ N, K, ms1, bytes / (ms1 * 1e-3) / 1e9 });
        std.debug.print("GEMM|IQ2_XS [m=8,n={d},k={d}]|{d:.4}|{d:.3}\n", .{ N, K, ms8, bytes / (ms8 * 1e-3) / 1e9 });
        std.debug.print("  IQ2_XS agreement vs scalar: m=1 rel-L2 {d:.2}  m=8 rel-L2 {d:.2}\n", .{ rel1, rel8 });
        const sm = @as(f64, @floatFromInt(setup_xs)) / 1e6;
        std.debug.print("SETUP|IQ2_XS [m=1,n={d},k={d}]|{d:.2}\n", .{ N, K, sm });
        std.debug.print("SETUP|IQ2_XS [m=8,n={d},k={d}]|{d:.2}\n", .{ N, K, sm });
    }
}
