// bench/cpp_vs_zig/iq_bench.zig — all seven IQ formats, three predicted buckets.
//
// CLAIM: swiglu-regalloc — IQ format family bench — 2026-10-04
// — holds bench/cpp_vs_zig/ until committed. Touches no build.zig step.
//
// WHY THREE BUCKETS AND NOT TWO
//
// The obvious experiment is "do the five unmeasured formats land at ~0.93 GB/s,
// the same as the two anchors". That is a flat point comparison, and it can only
// confirm. This one has a predicted direction of failure, which is worth more:
//
//   bucket 1  IQ2_S, IQ3_XXS, IQ3_S   unsigned grid8       -> expect ~0.93
//   bucket 2  IQ1_S                   signed grid8s        -> ?
//   bucket 3  IQ1_M                   signed grid8s        -> ?
//                                ... and its scale is structurally unhoistable
//
// IQ1_S and IQ1_M are NOT one bucket, and that is the whole point of reading the
// code before designing the table. Both use the signed `grid8s` (owned by
// gemm_224_iq2_xxs.zig:287, imported by both — "import, never re-declare"). But:
//
//   - iq3_xxs's `dsuper` is commented "per BLOCK, hoisted".
//   - IQ1_M's f16 global scale is "HIDDEN in" the scales-nibble shuffle, the
//     quantizer "ends with a GLOBAL d refit over the quantized scales", and the
//     hot loop recomputes `dl1`/`dl2` per 32-weight group. It cannot be hoisted
//     the way the others' can.
//
// So the prediction is specific: bucket 1 lands together; buckets 2 and 3 differ
// from bucket 1 and possibly from each other, with IQ1_M the most likely to
// diverge because of that per-group refit. If IQ1_S and IQ1_M come in at ~0.93
// anyway, the signed unpack is free and the grid8s split was insurance rather
// than necessity. If they land low, the cost of signed magnitude is localised in
// a two-format experiment.
//
// A table with three buckets can fail usefully. A table with one number cannot.
//
// THE PROCESS-PER-ARM VIABILITY BOUNDARY
//
// Measured setup:measurement ratios for all seven, at n=2048, k=7168:
//
//     IQ3_S   1113x      IQ2_S   1891x      IQ2_XXS 1997x
//     IQ3_XXS 1882x      IQ2_XS  2830x      IQ1_S   3563x
//     IQ1_M   8736x
//
// The spread is almost 8x across one format family, so the ratio is a property of
// the format-plus-bench and has to be printed per row. IQ1_M at 8736x is past the
// point where spawning a process per arm is a defensible sampler: at that ratio
// the per-sample variance is dominated by the quantizer's cache and frequency
// state, not the kernel, so the floor stops describing the thing being measured.
//
// That boundary is a property of the HARNESS, not of any one format, and it is
// the boundary beyond which every future quantized row needs in-process absolute
// mode instead. The harness now names it per row (`setup:meas` plus a
// setup-dominated verdict) precisely so the transition is visible rather than
// silent. The honest response at that ratio is "setup-dominated, with a stated
// ratio" — not a number obtained by quietly changing estimator mid-experiment.
//
// NO C++ ARM, AND THAT IS THE POINT
//
// `operators/avx2/` in the reference is BF16, FP8, MXFP4, MXFP8, GPTQ-INT4 and
// raw-INT4. Zero hits for any IQ or K-quant identifier. So there is nothing to
// diff against for any format here, and absolute GB/s with a measured band is the
// existing convention for this family (Q8_0 at 7.8, IQ4_XS at ~2.6) rather than a
// lowered bar. The K-quant work here has never had a cross-implementation diff for
// the same reason.
//
// ONLY GB/s IS COMPARABLE ACROSS THESE FORMATS
//
// They differ in bits-per-weight, so the byte volume per row differs too: IQ2_XXS
// is 66 B per 256-weight block and IQ2_XS is 74 B. Comparing raw ms across them
// manufactures a gap that is pure byte volume. The two anchors are 1.15x apart in
// ms and 1.01x apart in GB/s — the ms comparison would have hidden a tie behind a
// spurious 15%. Only GB/s is meaningful here.
//
// SETUP IS ~1000x THE MEASUREMENT, AND IT IS REPORTED PER ROW
//
// Measured at n=2048, k=7168 (57344 blocks): ~4.8-11 s to build the grid and
// quantize, against a ~4-6 ms GEMM at m=1. Two consequences, both handled:
//
//   1. All setup is hoisted above the timing loop. Anything that quantizes inside
//      a timed region measures the quantizer.
//   2. Each row emits `SETUP|label|ms`, so the harness computes the
//      setup:measurement ratio per row and can say whether a wide floor is the
//      host's fault or the bench's. The ratio is NOT a constant of the family —
//      IQ2_XXS is ~914x and IQ2_XS ~1845x, because the 512-entry grid init is the
//      expensive one — which is exactly why it is printed per row rather than
//      noted once.
//
// Agreement against each format's own scalar reference is checked on every run
// and printed. Blocks are quantized from real data with the format's own
// `_WithInit` quantizer; a zeroed or randomly-byte-filled block would leave the
// unpack on a path real weights never take.
//
// Run the floor with the control arms (both arms the same binary, so every row
// must read UNMEASURABLE — that run is the band, not a speedup):
//
//   zig build-exe -OReleaseFast -mcpu=native -lc --cache-dir /tmp/kc \
//       --dep kt -Mroot=bench/cpp_vs_zig/iq_bench.zig -Mkt=src/root.zig \
//       -femit-bin=/tmp/iq_bench
//   zig run bench/cpp_vs_zig/measure.zig -- --self-test --iters 3 --pin 9 \
//       --cpp /tmp/iq_bench --zig /tmp/iq_bench
//
// For the absolute figures, run the bench directly: it prints GB/s per row.

const std = @import("std");
const root = @import("kt");
const amx = root.amx;

const N: usize = 2048;
const K: usize = 7168;
const QK: usize = 256;
const NB: usize = K / QK;

const REPS: usize = 60;
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

fn settleMs() void {
    const req = std.os.linux.timespec{ .sec = 0, .nsec = 200 * std.time.ns_per_ms };
    _ = std.os.linux.nanosleep(&req, null);
}

/// Per-format descriptor. Comptime-parameterised, so each instantiation is
/// monomorphic and the differing grid-data types (Iq2GridData vs Iq3GridData)
/// resolve without a runtime tag.
fn Fmt(
    comptime fmt_name: []const u8,
    comptime fmt_bucket: []const u8,
    comptime fmt_block: type,
    comptime fmt_init: anytype,
    comptime fmt_quant: anytype,
    comptime fmt_vec: anytype,
    comptime fmt_scalar: anytype,
) type {
    // Param names differ from field names on purpose: `const name = name;` is a
    // shadowing error, and the K-quant bench already uses this shape.
    return struct {
        const name = fmt_name;
        const bucket = fmt_bucket;
        const Block = fmt_block;
        const initFn = fmt_init;
        const quantFn = fmt_quant;
        const gemmVec = fmt_vec;
        const gemmScalar = fmt_scalar;
    };
}

/// Build every block for one format and report the setup cost, timed separately
/// per format. Timing a shared span and attributing it to each format inflates
/// every row by the number of consumers — which produces a plausible number with
/// no arithmetic error in it.
fn buildBlocks(comptime F: type, alloc: std.mem.Allocator, src: []const f32) !struct { blocks: []F.Block, setup_ns: u64 } {
    const t0 = nowNs();
    const data = F.initFn(alloc);
    const b = try alloc.alloc(F.Block, N * NB);
    for (0..N) |j| F.quantFn(data, src, @ptrCast(&b[j * NB]), K, null);
    return .{ .blocks = b, .setup_ns = nowNs() - t0 };
}

fn timeAvx2(comptime F: type, act: []const amx.bf16, b: []const F.Block, c: []f32, m: usize) u64 {
    settleMs();
    var best: u64 = std.math.maxInt(u64);
    for (0..REPS + WARMUP) |r| {
        const t0 = nowNs();
        F.gemmVec(act.ptr, b.ptr, c.ptr, m, N, K, K, NB, N);
        const dt = nowNs() - t0;
        if (r >= WARMUP and dt < best) best = dt;
    }
    return best;
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

fn runOne(comptime F: type, alloc: std.mem.Allocator, src: []const f32, acts_m1: []const amx.bf16, acts_m8: []const amx.bf16, out1: []f32, out8: []f32, ref1: []f32, ref8: []f32) !void {
    const built = try buildBlocks(F, alloc, src);
    const b = built.blocks;
    const bytes = @as(f64, @floatFromInt(b.len * @sizeOf(F.Block)));
    const bpw = @as(f64, @floatFromInt(@sizeOf(F.Block) * 8)) / @as(f64, @floatFromInt(QK));

    // Correctness before speed: blocks that are not representative make the
    // timings meaningless, so this is checked and printed on every run.
    F.gemmScalar(acts_m1.ptr, b.ptr, ref1.ptr, 1, N, K, K, NB, N);
    F.gemmVec(acts_m1.ptr, b.ptr, out1.ptr, 1, N, K, K, NB, N);
    const rel1 = relL2(out1, ref1);
    F.gemmScalar(acts_m8.ptr, b.ptr, ref8.ptr, 8, N, K, K, NB, N);
    F.gemmVec(acts_m8.ptr, b.ptr, out8.ptr, 8, N, K, K, NB, N);
    const rel8 = relL2(out8, ref8);

    const t1 = timeAvx2(F, acts_m1, b, out1, 1);
    const t8 = timeAvx2(F, acts_m8, b, out8, 8);
    const ms1 = @as(f64, @floatFromInt(t1)) / 1e6;
    const ms8 = @as(f64, @floatFromInt(t8)) / 1e6;
    const sm = @as(f64, @floatFromInt(built.setup_ns)) / 1e6;

    std.debug.print("GEMM|{s} [m=1,n={d},k={d}]|{d:.4}|{d:.3}\n", .{ F.name, N, K, ms1, bytes / (ms1 * 1e-3) / 1e9 });
    std.debug.print("GEMM|{s} [m=8,n={d},k={d}]|{d:.4}|{d:.3}\n", .{ F.name, N, K, ms8, bytes / (ms8 * 1e-3) / 1e9 });
    std.debug.print("SETUP|{s} [m=1,n={d},k={d}]|{d:.2}\n", .{ F.name, N, K, sm });
    std.debug.print("SETUP|{s} [m=8,n={d},k={d}]|{d:.2}\n", .{ F.name, N, K, sm });
    std.debug.print(
        "  {s: <7} bucket={s: <8} block={d: >3}B bpw={d:.3}  setup={d:8.1} ms  setup:meas={d:.0}x  agree m=1 {d:.2} m=8 {d:.2}\n",
        .{ F.name, F.bucket, @sizeOf(F.Block), bpw, sm, sm / ms1, rel1, rel8 },
    );
}

pub fn main(init: std.process.Init) !void {
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    var rng: u64 = 0x5EED_1234;
    const acts8 = try alloc.alloc(amx.bf16, 8 * K);
    for (acts8) |*v| v.* = amx.f32_to_bf16(randF32(&rng));
    const acts1 = try alloc.alloc(amx.bf16, K);
    @memcpy(acts1, acts8[0..K]);

    // The quantizers take f32 source data, not the bf16 activations.
    const src = try alloc.alloc(f32, K);
    for (src) |*v| v.* = randF32(&rng);

    const out1 = try alloc.alloc(f32, N);
    const out8 = try alloc.alloc(f32, 8 * N);
    const ref1 = try alloc.alloc(f32, N);
    const ref8 = try alloc.alloc(f32, 8 * N);

    // Bucket 1: unsigned grid8.
    try runOne(Fmt("IQ2_XXS", "unsigned", root.gemm_iq2_xxs.BlockIQ2_XXS, root.iq2xs_init.initIq2XsData, root.iq2_quantize.quantizeRowIQ2_XXS_WithInit, root.gemm_iq2_xxs.gemmAvx2, root.gemm_iq2_xxs.gemmIQ2_XXSScalar), alloc, src, acts1, acts8, out1, out8, ref1, ref8);
    try runOne(Fmt("IQ2_XS", "unsigned", root.gemm_iq2_xs.BlockIQ2_XS, root.iq2xs_init.initIq2XsData512, root.gemm_iq2_xs.quantizeRowIQ2_XS_WithInit, root.gemm_iq2_xs.gemmAvx2, root.gemm_iq2_xs.gemmIQ2_XSScalar), alloc, src, acts1, acts8, out1, out8, ref1, ref8);
    try runOne(Fmt("IQ2_S", "unsigned", root.gemm_iq2_s.BlockIQ2_S, root.iq2xs_init.initIq2SData, root.gemm_iq2_s.quantizeRowIQ2_S_WithInit, root.gemm_iq2_s.gemmAvx2, root.gemm_iq2_s.gemmIQ2_SScalar), alloc, src, acts1, acts8, out1, out8, ref1, ref8);
    try runOne(Fmt("IQ3_XXS", "unsigned", root.gemm_iq3_xxs.BlockIQ3_XXS, root.iq3xs_init.initIq3XsData, root.iq3_quantize.quantizeRowIQ3_XXS_WithInit, root.gemm_iq3_xxs.gemmAvx2, root.gemm_iq3_xxs.gemmIQ3_XXSScalar), alloc, src, acts1, acts8, out1, out8, ref1, ref8);
    try runOne(Fmt("IQ3_S", "unsigned", root.gemm_iq3_s.BlockIQ3_S, root.iq3xs_init.initIq3SData, root.gemm_iq3_s.quantizeRowIQ3_S_WithInit, root.gemm_iq3_s.gemmAvx2, root.gemm_iq3_s.gemmIQ3_SScalar), alloc, src, acts1, acts8, out1, out8, ref1, ref8);

    // Buckets 2 and 3: both signed (grid8s), but IQ1_M's scale is unhoistable.
    try runOne(Fmt("IQ1_S", "signed", root.gemm_iq1_s.BlockIQ1_S, root.iq2xs_init.initIq1SData, root.gemm_iq1_s.quantizeRowIQ1_S_WithInit, root.gemm_iq1_s.gemmAvx2, root.gemm_iq1_s.gemmIQ1_SScalar), alloc, src, acts1, acts8, out1, out8, ref1, ref8);
    try runOne(Fmt("IQ1_M", "signed-unhoist", root.gemm_iq1_m.BlockIQ1_M, root.iq2xs_init.initIq1SData, root.gemm_iq1_m.quantizeRowIQ1_M_WithInit, root.gemm_iq1_m.gemmAvx2, root.gemm_iq1_m.gemmIQ1_MScalar), alloc, src, acts1, acts8, out1, out8, ref1, ref8);
}