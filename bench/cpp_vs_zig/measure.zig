// CLAIM: bench-harness — the measurement harness (Stream O) — 2026-10-04
// Holds bench/cpp_vs_zig/ until committed. See TODO.md Stream O.
//
// This directory had NO claim while three sessions were editing it, and
// measure.zig was swept by another commit three times — twice leaving HEAD
// not compiling. A marker does not prevent that; tools/check_staged_claims.sh
// does. Until it is wired as a hook, check `git status` for this path before
// you commit anything in it.

// bench/cpp_vs_zig/measure.zig — trustworthy ratios for the C++/Zig comparison.
//
// WHY THIS EXISTS
//
// `run.sh` reports a single number per row: the ratio of the two harnesses'
// best times. On this shared host that number is not trustworthy on its own.
// Measured 2026-10-04: a probe variant that was a BYTE-IDENTICAL copy of the
// baseline differed from it by 14-23% across runs, and the C++ harness's own
// swiglu time moved 0.137 -> 0.190 ms with no code change. A ratio carrying a
// 10% difference is therefore indistinguishable from a ratio carrying none,
// and publishing it invites a decision the data does not support.
//
// So this tool refuses to do that. Every ratio it prints comes with the
// measurement that justifies it:
//
//   1. INTERLEAVED. Arms run round-robin within one pass, not all-A-then-all-B.
//      The host drifts; interleaving stops the drift from being attributed to
//      whichever arm happened to run during it.
//   2. PINNED. `--pin` pins this process, and the harness children inherit the
//      affinity mask across fork/exec, so every arm runs on the same core.
//   3. MIN-OF-N WITH N FIXED ACROSS ARMS. Each harness already does its own
//      warmup and min-of-9 in-process; this adds N process invocations on top.
//      The minimum is the right statistic against interference, which only ever
//      ADDS time -- but the minimum of N is a function of N, so a min-of-3 arm
//      is NOT comparable to a min-of-7 arm. N is therefore fixed for the whole
//      pass and printed.
//   4. A BYTE-IDENTICAL CONTROL ARM IN THE SAME PASS. `zigctl` is the same
//      `zig_bench` binary invoked a second time. Its deviation from 1.0 is a
//      direct, in-run measurement of the noise floor, per row, with no
//      dependence on a number remembered from another day. Each arm's own
//      cross-process spread is folded in too, and the floor is per row because
//      a bandwidth-bound decode and a compute-bound activation do not share one.
//   5. IT REFUSES TO PRINT AN UNRESOLVED RATIO. If the measured ratio falls
//      inside the control's noise band, the verdict is UNMEASURABLE and no
//      ratio is printed at all.
//
// USAGE
//
//   zig run bench/cpp_vs_zig/measure.zig -- [options] [filter]
//
// NOTE the `--` before the options. `zig run` treats everything after the
// script as its own arguments unless the separator is there, so omitting it
// makes the tool silently ignore --iters/--pin/--self-test and run with
// defaults. It fails confusingly rather than loudly.
//
//   filter          only rows whose label contains this substring
//   --iters N       process invocations per arm (default 5, minimum 3)
//   --cpp PATH      C++ harness   (default bench/cpp_vs_zig/cpp_bench)
//   --zig PATH      Zig harness   (default bench/cpp_vs_zig/zig_bench)
//   --out DIR       KT_BENCH_OUT for the harnesses' output dumps
//   --pin CPU       pin to this CPU (inherited by the children)
//   --raw           TSV instead of the markdown table
//   --self-test     run BOTH arms as the same binary, so the true ratio is
//                   exactly 1.0 and every row must come back UNMEASURABLE
//   --cpp-absent    self-test only: run NO C++ arm at all, modelling a format
//                   the reference has no kernel for. Every row then takes the
//                   absolute path, and in --self-test it must print its band
//                   and REFUSE its figure. Without this the absolute mode is
//                   reachable in a self-test only by accident, and an unguarded
//                   mode that emits confident absolute numbers off one control
//                   is the same quiet confidence this tool exists to remove.
//                   Refused outside --self-test: a real pass that silences the
//                   C++ arm has no cross-implementation check at all, and must
//                   not be able to print a figure.
//
// --self-test exists because the control arm should be re-measurable on demand
// rather than borrowed from a previous session's notes. It is also the check
// that the refusal logic still has teeth: if a self-test ever prints a ratio,
// the gate is broken and every other verdict from this binary is suspect.
//
// The harnesses must already be built; `run.sh` builds them. This tool only
// runs them, so it stays out of build.zig entirely.
//
// Cost warning: `prefill 512` is ~1-3 s per repetition inside each harness, so a
// full pass is minutes. Use the filter to skip it when measuring the rest.

const std = @import("std");
const builtin = @import("builtin");

/// Four arms, not three. Each SIDE needs its own same-code control: the
/// reported figure is a quotient of two independent measurements, so a control
/// on only one side cannot bound the error in the ratio. `cpp`/`cppctl` are the
/// same binary run twice; `zig`/`ctl` likewise. Running one control instead of
/// two was a real bug found by --self-test: with a single control the floor was
/// estimated from one side's samples only, and under heavy load a spurious
/// "0.80x, slower" verdict survived on a row whose true ratio is 1.0.
const Arm = enum { cpp, cppctl, zig, ctl };

const ARM_COUNT = 4;

/// Accumulated samples for one harness-reported row, one list per arm.
const Samples = struct {
    label: []const u8 = "",
    ms: [ARM_COUNT]std.ArrayList(f64) = @splat(.empty),
    seen: [ARM_COUNT]usize = @splat(0),
    /// Set-up milliseconds this bench reported for the row, if any. `null` means
    /// the bench did not say, and the tool then says so rather than assuming 1x.
    setup_ms: ?f64 = null,
    /// The harness's own rate for the last sample seen on that arm (GFLOP/s for
    /// a GEMM row, GB/s for an activation row). Carried only because ABSOLUTE
    /// mode prints it; ratio mode has no use for it.
    rate: [ARM_COUNT]f64 = @splat(0),
};

/// One `KIND|label|ms|rate` line, parsed.
const Reading = struct { label: []const u8, ms: f64, rate: f64 };

/// Linear lookup: there are single-digit row counts, and this avoids taking a
/// dependency on a std container whose 0.16 initializer churned under us.
fn slotFor(arena: std.mem.Allocator, table: *std.ArrayList(Samples), label: []const u8) !*Samples {
    for (table.items) |*s| {
        if (std.mem.eql(u8, s.label, label)) return s;
    }
    try table.append(arena, .{});
    return &table.items[table.items.len - 1];
}

const Row = struct {
    /// Harness-reported label, e.g. "swiglu act [m=128,n=2048]".
    label: []const u8,
    /// Best ms per arm, indexed by Arm.
    best: [ARM_COUNT]f64,
    /// min/median/max per arm, indexed by Arm.
    lo: [ARM_COUNT]f64,
    mid: [ARM_COUNT]f64,
    hi: [ARM_COUNT]f64,
    /// Setup reported by the bench for this row, if any. See SETUP_DOMINATED.
    setup_ms: ?f64 = null,
    /// Arena for formatting the setup cell; owned by the caller.
    arena: std.mem.Allocator,
    /// The Zig harness's own rate for this row, carried through untouched.
    rate: f64 = 0,
    /// ABSOLUTE MODE: the C++ harness never reported this label, so there is no
    /// cross-implementation ratio to resolve. The row is a successful
    /// measurement of the Zig side with a band, not a failed comparison.
    absolute: bool = false,
};

const MIN_ITERS: usize = 3;
const DEFAULT_ITERS: usize = 5;

pub fn main(init: std.process.Init) !void {
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    // Without this the arena's doubling blocks are reported as leaks by the
    // Debug allocator, which is how a real leak in this file once hid behind six
    // large ones.
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var iters: usize = DEFAULT_ITERS;
    var cpp_path: []const u8 = "bench/cpp_vs_zig/cpp_bench";
    var zig_path: []const u8 = "bench/cpp_vs_zig/zig_bench";
    var out_dir: []const u8 = "/tmp/kt-measure";
    var pin: ?usize = null;
    var raw = false;
    var self_test = false;
    var cpp_absent = false;
    var filter: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--iters")) {
            i += 1;
            iters = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "--cpp")) {
            i += 1;
            cpp_path = args[i];
        } else if (std.mem.eql(u8, a, "--zig")) {
            i += 1;
            zig_path = args[i];
        } else if (std.mem.eql(u8, a, "--out")) {
            i += 1;
            out_dir = args[i];
        } else if (std.mem.eql(u8, a, "--pin")) {
            i += 1;
            pin = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "--raw")) {
            raw = true;
        } else if (std.mem.eql(u8, a, "--self-test")) {
            self_test = true;
        } else if (std.mem.eql(u8, a, "--cpp-absent")) {
            cpp_absent = true;
        } else if (std.mem.startsWith(u8, a, "--")) {
            std.debug.print("unknown option {s}\n", .{a});
            return error.BadArgs;
        } else {
            filter = a;
        }
    }

    if (self_test) {
        cpp_path = zig_path;
    }

    if (cpp_absent and !self_test) {
        std.debug.print(
            "refusing --cpp-absent without --self-test: silencing the C++ arm in a\n" ++
                "real pass would print absolute figures with nothing to validate them against,\n" ++
                "which is the failure mode this tool exists to prevent. Pass --self-test,\n" ++
                "which withholds every figure and checks the band instead.\n",
            .{},
        );
        return error.BadArgs;
    }

    if (iters < MIN_ITERS) {
        std.debug.print(
            "refusing to run with --iters {d}: below {d} the control spread is not\n" ++
                "an estimate of anything. Use --iters {d} at minimum.\n",
            .{ iters, MIN_ITERS, MIN_ITERS },
        );
        return error.BadArgs;
    }

    if (builtin.os.tag == .linux) {
        if (pin) |cpu| try pinSelf(cpu);
    } else if (pin != null) {
        std.debug.print("--pin is Linux-only on this build; ignoring\n", .{});
    }

    var threaded = std.Io.Threaded.init(init.gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // `Io.Dir.createDirPath` is 0.16's `makeDirPath`; it needs an `Io`, which is
    // why it cannot sit with the rest of the argument handling above.
    try std.Io.Dir.cwd().createDirPath(io, out_dir);

    const loadavg = readLoadAvg();

    // Accumulate one sample per arm per iteration, then reduce.
    var table = std.ArrayList(Samples).empty;
    defer table.deinit(arena);

    var it: usize = 0;
    while (it < iters) : (it += 1) {
        // Interleaved: one invocation of each arm per iteration, in a fixed
        // rotation, so no arm systematically owns a quieter slice of the pass.
        for (std.enums.values(Arm)) |arm| {
            // `--cpp-absent` runs neither C++ arm, so every label is reported
            // by the Zig side alone and takes the absolute path. Skipping the
            // arm (rather than running it and discarding its samples) keeps the
            // pass cheap and makes the condition obvious at the call site.
            if (cpp_absent and (arm == .cpp or arm == .cppctl)) continue;
            const path = switch (arm) {
                .cpp => cpp_path,
                .cppctl => cpp_path, // same-code control for the C++ side
                .zig => zig_path,
                .ctl => zig_path, // same-code control for the Zig side
            };
            const parsed = try runHarness(io, init.gpa, arena, path, out_dir);
            for (parsed.setup) |sv| {
                const sidx: usize = @intFromEnum(arm);
                const slot = try slotFor(arena, &table, sv.label);
                _ = sidx;
                // First arm to report a setup for this row wins; they are
                // measuring the same row and should agree.
                if (slot.setup_ms == null) slot.setup_ms = sv.ms;
            }
            for (parsed.rows) |kv| {
                if (filter) |f| {
                    if (std.mem.indexOf(u8, kv.label, f) == null) continue;
                }
                const idx: usize = @intFromEnum(arm);
                const slot = try slotFor(arena, &table, kv.label);
                if (slot.seen[idx] == 0) {
                    slot.label = try arena.dupe(u8, kv.label);
                    slot.ms = @splat(.empty);
                }
                slot.seen[idx] += 1;
                slot.rate[idx] = kv.rate;
                try slot.ms[idx].append(arena, kv.ms);
            }
        }
        std.debug.print("  iteration {d}/{d} done\n", .{ it + 1, iters });
    }

    var rows = std.ArrayList(Row).empty;
    defer rows.deinit(arena);
    for (table.items) |slot| {
        // The Zig side is what BOTH modes need: the row itself and its same-code
        // control. Without `ctl` there is no floor for either, so the row goes.
        const zig_seen = slot.seen[@intFromEnum(Arm.zig)];
        const ctl_seen = slot.seen[@intFromEnum(Arm.ctl)];
        if (zig_seen != iters or ctl_seen != iters) {
            // A row missing an expected sample would otherwise be compared
            // against a different number of samples -- exactly the
            // min-of-N-vs-min-of-M mistake this tool exists to prevent.
            std.debug.print(
                "skipping {s}: Zig side has zig={d} ctl={d}, expected {d} each\n",
                .{ slot.label, zig_seen, ctl_seen, iters },
            );
            continue;
        }

        // TWO CASES, and the difference matters. `cpp_seen == 0` means the C++
        // harness has no such row at all -- the format has no kernel over there
        // -- which is an ABSOLUTE row, reported with a band and no ratio. A
        // non-zero count that is not `iters` means the two harnesses disagree
        // about the row, and that is the loud skip above; it must never fall
        // through to absolute mode, because "the reference disagrees with us"
        // is not the same finding as "the reference does not have this".
        const cpp_seen = slot.seen[@intFromEnum(Arm.cpp)];
        const cppctl_seen = slot.seen[@intFromEnum(Arm.cppctl)];
        const no_cpp_arm = cpp_seen == 0 and cppctl_seen == 0;
        if (!no_cpp_arm and (cpp_seen != iters or cppctl_seen != iters)) {
            std.debug.print(
                "skipping {s}: C++ side has cpp={d} cppctl={d}, expected {d} each\n",
                .{ slot.label, cpp_seen, cppctl_seen, iters },
            );
            continue;
        }

        var row = Row{
            .label = slot.label,
            .best = @splat(0),
            .lo = @splat(0),
            .mid = @splat(0),
            .hi = @splat(0),
            .rate = slot.rate[@intFromEnum(Arm.zig)],
            .absolute = no_cpp_arm,
            .setup_ms = slot.setup_ms,
            .arena = arena,
        };
        // Only the arms this row actually has; the rest stay 0 and are never
        // read, because `report` branches on `absolute` before touching `cpp`.
        for (std.enums.values(Arm)) |arm| {
            const idx: usize = @intFromEnum(arm);
            if (slot.seen[idx] == 0) continue;
            const s = slot.ms[idx].items;
            std.mem.sort(f64, s, {}, std.sort.asc(f64));
            row.lo[idx] = s[0];
            row.mid[idx] = s[s.len / 2];
            row.hi[idx] = s[s.len - 1];
            row.best[idx] = s[0];
        }
        try rows.append(arena, row);
    }

    try report(arena, rows.items, iters, pin, loadavg, raw, self_test, gitHead(init.gpa, arena, io));
}

/// Noise band for one row, as a multiplicative factor >= 1.
///
/// The reported figure is a QUOTIENT of two independently-measured minima, so
/// each side contributes its own noise and the two compound:
///
///     floor = dev(cpp side) * dev(zig side)
///
/// where a side's `dev` is the widest spread observed across ALL samples of the
/// two same-code arms on that side -- every repetition of `cpp_bench` against
/// every repetition of `cpp_bench`, and likewise for the Zig side.
///
/// Using the full range rather than a min-vs-min comparison is deliberate, and
/// it was arrived at by getting it wrong twice. A first attempt compared the two
/// same-code arms' minima; under load 80 that let a spurious "0.80x, slower"
/// verdict through on a row whose true ratio is 1.0. A second attempt took the
/// worse of min-vs-min and the internal spread; on a QUIET box it still printed
/// "0.96x, slower" twice out of three on the same row. Both failures are the
/// same bug: the minimum is an extreme order statistic, so two small samples of
/// it agree by luck far more often than the underlying quantity justifies, and
/// an estimate built from it understates the noise it is meant to bound.
///
/// The full observed range cannot have that problem. It is a strict bound: no
/// ratio formed from any sample on either side can fall outside it, so if the
/// reported ratio is inside the band, some pair of identical-code measurements
/// differed by at least as much -- and the difference is then not attributable
/// to the code under test.
///
/// The cost is conservatism, which is the correct direction to err. Per-side
/// ranges compound, so two sides each varying 10% give a floor of 1.21x, and a
/// claimed 1.10x win will be refused. That refusal is the feature.
fn sideDev(row: Row, x: Arm, y: Arm) f64 {
    var lo: f64 = std.math.inf(f64);
    var hi: f64 = 0;
    inline for (.{ x, y }) |arm| {
        const idx: usize = @intFromEnum(arm);
        lo = @min(lo, row.lo[idx]);
        hi = @max(hi, row.hi[idx]);
    }
    return if (lo > 0) @max(hi / lo, 1.0) else 1.0;
}

fn floorFor(row: Row) f64 {
    return @max(sideDev(row, .cpp, .cppctl) * sideDev(row, .zig, .ctl), 1.0);
}

/// Band for an ABSOLUTE row: `dev(zig, ctl)` and nothing else.
///
/// WHY ONE CONTROL IS ENOUGH HERE AND NOT IN `floorFor` — the distinction is
/// worth keeping explicit, because collapsing it is the obvious "simplification"
/// and it would be wrong in one direction and look merely tidy in the other.
/// A RATIO is a quotient of two independently-measured minima (C++ and Zig), so
/// each side contributes its own error and the two compound: that is why there
/// are four arms and two controls. An ABSOLUTE figure is ONE measurement, so its
/// only error term is that measurement's own spread, which is precisely what
/// `zig` against `ctl` bounds — the same binary, the same code, the same pass.
/// Adding the C++ side's spread to an absolute number would not make it more
/// trustworthy; there is no C++ number in it to be wrong about.
fn absoluteBand(row: Row) f64 {
    return @max(sideDev(row, .zig, .ctl), 1.0);
}

/// Above this setup:measurement ratio the floor is ours, not the host's.
///
/// Measured on the IQ anchors: 5.1 s of quantization plus 249 ms of grid init
/// against a ~4 ms GEMM, i.e. ~1300x. The first control-armed pass on those rows
/// reported floors of 2.9x-15.8x, which is not host noise -- it is the floor
/// measuring the setup.
///
/// It has to be reported by the bench rather than inferred from the child's wall
/// time, because wall time also contains every rep: 62 reps of a 4.5 ms kernel is
/// ~280 ms and would be indistinguishable from real setup. A bench that does not
/// print `SETUP|label|ms` gets "not reported", not an assumed 1x.
const SETUP_DOMINATED: f64 = 3.0;

/// Band width at which the instrument can no longer see a sub-`x` effect, and
/// the summary says so out loud. Set just under 2.0 because a 2x band already
/// hides the 1.2x class of claim this repo publishes; a tool that refuses should
/// also be loud about being unable to see.
const INSTRUMENT_BLIND: f64 = 1.9;

/// Why a row is unresolvable, and whether that is anybody's fault.
///
/// The distinction matters because the two have opposite remedies. A
/// host-dominated floor is the weather: re-run on a quieter box. A
/// setup-dominated floor is ours: the bench builds inputs that cost far more
/// than the thing it times, so every process-level sample carries the setup's
/// cache and frequency state into the measurement, and the fix is to hoist the
/// setup or to sample in-process instead of spawning a process per arm.
///
/// A floor that is too wide costs power; a floor that is too narrow manufactures
/// findings. Only the first is acceptable in a tool whose job is to refuse -- but
/// "always says unresolvable" is its own failure mode, because a setup-dominated
/// floor will hide a real effect indefinitely and look like rigour while doing
/// it. Hence: say which kind it is.
const Domination = enum {
    /// Setup dominates: fixable in software.
    setup,
    /// The host dominates: not fixable here.
    host,
    /// The bench did not report a setup cost, so this cannot be told.
    unknown,

    fn label(self: Domination) []const u8 {
        return switch (self) {
            .setup => "setup-dominated (fixable: hoist the setup, or sample in-process)",
            .host => "host-dominated (not fixable in software; re-run quieter)",
            .unknown => "setup unreported (cannot tell host from setup)",
        };
    }

    /// Whether process-per-arm is the right sampling strategy for this row.
    ///
    /// Spawning a process per arm is what isolates the arms, so it is correct for
    /// a bench with trivial setup. For a bench whose setup is orders of magnitude
    /// larger than its measurement it is simply the wrong tool, because the arm
    /// boundary and the setup boundary coincide.
    fn sampler(self: Domination) []const u8 {
        return switch (self) {
            .setup => "process-per-arm is the WRONG sampler here; the setup is the measurement",
            .host => "process-per-arm is appropriate; the host is the limit",
            .unknown => "sampler choice undetermined without a reported setup cost",
        };
    }
};

/// The setup:measurement cell. Written inline because it needs `arena`.
fn setupCell(row: Row) []const u8 {
    const r = setupRatio(row) orelse return "not reported";
    return std.fmt.allocPrint(row.arena, "{d:.0}x", .{r}) catch "err";
}

fn setupRatio(row: Row) ?f64 {
    const setup = row.setup_ms orelse return null;
    const m = row.best[@intFromEnum(Arm.zig)];
    if (m <= 0 or setup <= 0) return null;
    return setup / m;
}

fn classify(row: Row) Domination {
    const r = setupRatio(row) orelse return .unknown;
    return if (r >= SETUP_DOMINATED) .setup else .host;
}

const Verdict = enum { faster, slower, unmeasurable };

fn verdictFor(ratio: f64, floor: f64) Verdict {
    if (ratio >= floor) return .faster;
    if (ratio <= 1.0 / floor) return .slower;
    return .unmeasurable;
}

fn report(
    arena: std.mem.Allocator,
    rows: []const Row,
    iters: usize,
    pin: ?usize,
    loadavg: [3]f64,
    raw: bool,
    self_test: bool,
    commit: []const u8,
) !void {
    const cpu_model = readCpuModel();

    var resolved: usize = 0;
    var unresolved: usize = 0;

    if (!raw) {
        if (self_test) {
            std.debug.print("# SELF-TEST — both arms are the same binary\n\n", .{});
            std.debug.print("The true ratio is exactly 1.0, so **every comparison row must read UNMEASURABLE**.\n", .{});
            std.debug.print("The floor printed below is this host's measured noise band for each row,\n", .{});
            std.debug.print("re-derived here and now. A comparison row that prints a ratio means the\n", .{});
            std.debug.print("refusal logic is broken.\n\n", .{});
            std.debug.print("Rows with no C++ arm take the absolute path instead. There the self-test must\n", .{});
            std.debug.print("print the band and REFUSE the figure: a mode that can emit a confident absolute\n", .{});
            std.debug.print("number off one control and no cross-implementation check is the same quiet\n", .{});
            std.debug.print("confidence this tool exists to remove, and it would be untested by the ratio\n", .{});
            std.debug.print("rows above. A self-test that prints an absolute figure means this mode is\n", .{});
            std.debug.print("unguarded.\n\n", .{});
        } else {
            std.debug.print("# C++ reference vs Zig port — resolvable ratios only\n\n", .{});
        }
        std.debug.print("| | |\n|---|---|\n", .{});
        std.debug.print("| commit | `{s}` |\n", .{commit});
        std.debug.print("| host | {s} |\n", .{cpu_model});
        std.debug.print("| load average (1/5/15) | {d:.2} / {d:.2} / {d:.2} |\n", .{ loadavg[0], loadavg[1], loadavg[2] });
        std.debug.print("| pin | {s} |\n", .{if (pin) |c| std.fmt.allocPrint(arena, "cpu {d}", .{c}) catch "?" else "none"});
        std.debug.print("| iterations per arm | {d} (interleaved, min-of-{d} across process invocations; each harness does its own min-of-9 in-process) |\n", .{ iters, iters });
        std.debug.print("| control | `zig_bench` run twice per iteration; its deviation from 1.0 sets the floor |\n\n", .{});
        std.debug.print("`floor` is the multiplicative noise band this pass measured for that row. A ratio is\n", .{});
        std.debug.print("printed ONLY when it falls outside it; otherwise the row reads UNMEASURABLE and no\n", .{});
        std.debug.print("number is offered. Ratio is C++ ms / Zig ms, so >1 means the Zig port is faster.\n\n", .{});
        std.debug.print("| row | C++ ms | Zig ms | setup:meas | floor | ratio | verdict |\n", .{});
        std.debug.print("|---|---|---|---|---|---|---|\n", .{});
    } else {
        std.debug.print("row\tcpp_ms\tzig_ms\tsetup:meas\tfloor\tratio\tverdict\n", .{});
    }

    // ABSOLUTE rows are collected and printed as their own table, AFTER the
    // ratio table, because they are a different KIND OF STATEMENT: not "the
    // port is faster than the reference" but "here is what the port does, with
    // the spread this host produced for it". Mixing the two in one table with
    // empty cells is how a reader ends up treating a missing comparison as a
    // measured one.
    var absolute_rows: usize = 0;

    for (rows) |row| {
        if (row.absolute) {
            absolute_rows += 1;
            continue;
        }
        const cpp = row.best[@intFromEnum(Arm.cpp)];
        const zig = row.best[@intFromEnum(Arm.zig)];
        const floor = floorFor(row);
        const ratio = cpp / zig;
        const v = verdictFor(ratio, floor);

        switch (v) {
            .unmeasurable => {
                unresolved += 1;
                // Deliberately no ratio printed: an unresolved delta is a
                // number that invites a decision the measurement cannot support.
                const dom = classify(row);
                if (raw) {
                    std.debug.print("{s}\t{d:.4}\t{d:.4}\t{s}\t{d:.3}\t-\tUNMEASURABLE:{s}\n", .{
                        row.label, cpp, zig, setupCell(row), floor, @tagName(dom),
                    });
                } else {
                    std.debug.print("| {s} | {d:.4} | {d:.4} | {s} | {d:.3}x | — | **UNMEASURABLE** — {s} |\n", .{
                        row.label, cpp, zig, setupCell(row), floor, dom.label(),
                    });
                }
            },
            .faster, .slower => {
                resolved += 1;
                if (raw) {
                    std.debug.print("{s}\t{d:.4}\t{d:.4}\t{s}\t{d:.3}\t{d:.3}\t{s}\n", .{
                        row.label, cpp, zig, setupCell(row), floor, ratio, @tagName(v),
                    });
                } else {
                    std.debug.print("| {s} | {d:.4} | {d:.4} | {s} | {d:.3}x | **{d:.2}x** | {s} |\n", .{
                        row.label, cpp, zig, setupCell(row), floor, ratio, @tagName(v),
                    });
                }
            },
        }
    }

    if (absolute_rows > 0) {
        std.debug.print("\n", .{});
        if (!raw) {
            std.debug.print("# No C++ arm — absolute figures, with this pass's band\n\n", .{});
            std.debug.print("The C++ harness has no row for these, because it has no kernel for the format:\n", .{});
            std.debug.print("there is nothing to compare against, so there is no ratio and no verdict, and no\n", .{});
            std.debug.print("`UNMEASURABLE` either — that word means a comparison this pass could not resolve, and\n", .{});
            std.debug.print("reusing it here would make a successful measurement read as a failed one.\n\n", .{});
            std.debug.print("`band` is the spread of `zig_bench` against ITSELF in this pass (`dev(zig, ctl)`),\n", .{});
            std.debug.print("which is the whole error budget of a single measurement: one code path, sampled\n", .{});
            std.debug.print("`iters` times. It is NOT the ratio floor, which additionally carries the C++ side's\n", .{});
            std.debug.print("spread because a ratio divides two measurements.\n\n", .{});
            std.debug.print("`setup:meas` is the bench's own input-building time over the reported ms. A\n", .{});
            std.debug.print("high value means this process-per-arm sampler is mostly measuring the SETUP, not\n", .{});
            std.debug.print("the kernel — which is ours to fix (hoist the setup, or sample in-process), not the\n", .{});
            std.debug.print("host's to wait out. See the `why` column.\n\n", .{});
            if (self_test) {
                std.debug.print("**SELF-TEST: the figure column is withheld.** Both arms are the same binary, so\n", .{});
                std.debug.print("nothing here has been validated against an independent implementation. The band\n", .{});
                std.debug.print("is printed because the band is the part that must survive the self-test.\n\n", .{});
            }
            std.debug.print("| row | Zig ms | setup:meas | band | figure | why | note |\n", .{});
            std.debug.print("|---|---|---|---|---|---|---|\n", .{});
        }
        for (rows) |row| {
            if (!row.absolute) continue;
            const zig = row.best[@intFromEnum(Arm.zig)];
            const band = absoluteBand(row);
            const dom = classify(row);
            const figure = if (self_test)
                "withheld (self-test)"
            else if (row.rate > 0)
                try std.fmt.allocPrint(arena, "{d:.2}", .{row.rate})
            else
                "not reported by the harness";
            if (raw) {
                std.debug.print("{s}\t{d:.4}\t{s}\t{d:.3}\t{s}\t{s}\tno C++ arm\n", .{
                    row.label, zig, setupCell(row), band, figure, @tagName(dom),
                });
            } else {
                std.debug.print("| {s} | {d:.4} | {s} | +-{d:.1}% | {s} | {s} | no C++ arm |\n", .{
                    row.label, zig, setupCell(row), (band - 1.0) * 100.0, figure, dom.label(),
                });
            }
        }
    }

    // A floor that has widened far enough to hide the effect being looked for is
    // a BROKEN INSTRUMENT, not a null result, and it has to be able to read that
    // way. Printing the widest band beside the summary is the whole of it: a
    // reader who sees "+-120%" next to a 1.2x claim should not have to work out
    // that the claim was never testable.
    var widest: f64 = 1.0;
    var widest_label: []const u8 = "";
    for (rows) |row| {
        const band = if (row.absolute) absoluteBand(row) else floorFor(row);
        if (band > widest) {
            widest = band;
            widest_label = row.label;
        }
    }
    std.debug.print("\nwidest band this pass: +-{d:.0}% ({s})", .{
        (widest - 1.0) * 100.0, if (widest_label.len == 0) "no rows" else widest_label,
    });
    if (widest >= INSTRUMENT_BLIND) {
        std.debug.print(
            "\n**THE INSTRUMENT IS BLIND at this width.** Nothing below ~{d:.1}x is measurable in this\n" ++
                "pass, so a claim inside that range is not a null result — it is untested. Either the\n" ++
                "box is too busy or the bench's own setup dominates its measurement (read the\n" ++
                "`setup:meas` and `why` columns). Do not quote a ratio from this pass either way.",
            .{widest},
        );
    }
    std.debug.print("\n", .{});

    const ratio_rows = rows.len - absolute_rows;
    std.debug.print("\n{d} of {d} comparison rows resolved above the control's noise floor", .{ resolved, ratio_rows });
    if (absolute_rows > 0) {
        std.debug.print("; {d} row(s) have no C++ arm and are reported above as absolute figures", .{absolute_rows});
    }
    std.debug.print(".\n", .{});
    if (unresolved > 0) {
        std.debug.print(
            "The unresolved rows are not ties. They are rows where this pass could not\n" ++
                "distinguish the two implementations, and the verdict says which kind that is:\n",
            .{},
        );
        var saw_setup = false;
        var saw_host = false;
        for (rows) |row| {
            switch (classify(row)) {
                .setup => saw_setup = true,
                .host => saw_host = true,
                .unknown => {},
            }
        }
        if (saw_setup) {
            std.debug.print(
                "\n  setup-dominated: the bench builds more than it measures, so the setup's\n" ++
                    "  cache and frequency state land inside every sample. More --iters will not\n" ++
                    "  help. Hoist the setup out of the process, or sample in-process instead of\n" ++
                    "  spawning a process per arm. See SETUP_DOMINATED in this file.\n",
                .{},
            );
        }
        if (saw_host) {
            std.debug.print(
                "\n  host-dominated: the measurement is fine and the box is not. No amount of\n" ++
                    "  software changes this; re-run when load is low. Treat any floor above\n" ++
                    "  ~1.15x as evidence that we are the load.\n",
                .{},
            );
        }
        std.debug.print("\nDo not quote a number for any unresolved row.\n", .{});
    }
}

/// Everything one harness invocation reported.
const Parsed = struct {
    rows: []Reading,
    setup: []Reading,
};

/// Run one harness and parse its `KIND|label|ms|rate` lines, plus any optional
/// `SETUP|label|ms` lines.
///
/// The Zig harness prints via std.debug.print (stderr) and the C++ one via
/// printf (stdout), so both streams are scanned.
fn runHarness(
    io: std.Io,
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    path: []const u8,
    out_dir: []const u8,
) !Parsed {
    const argv = [_][]const u8{ path, out_dir };
    var env_map = std.process.Environ.Map.init(arena);
    try env_map.put("KT_BENCH_OUT", out_dir);

    const res = try std.process.run(gpa, io, .{
        .argv = &argv,
        .environ_map = &env_map,
    });
    defer gpa.free(res.stdout);
    defer gpa.free(res.stderr);

    switch (res.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("{s} exited {d}\n{s}\n{s}\n", .{ path, code, res.stdout, res.stderr });
            return error.HarnessFailed;
        },
        else => {
            std.debug.print("{s} did not exit normally\n", .{path});
            return error.HarnessFailed;
        },
    }

    var out = std.ArrayList(Reading).empty;
    var setup = std.ArrayList(Reading).empty;
    try parseInto(arena, res.stdout, &out, &setup);
    try parseInto(arena, res.stderr, &out, &setup);
    return .{ .rows = out.items, .setup = setup.items };
}

fn parseInto(
    arena: std.mem.Allocator,
    text: []const u8,
    out: *std.ArrayList(Reading),
    setup_out: ?*std.ArrayList(Reading),
) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var parts = std.mem.splitScalar(u8, line, '|');
        const kind = parts.next() orelse continue;
        if (std.mem.eql(u8, kind, "SETUP")) {
            // Optional: `SETUP|label|ms`, how long this bench spent building
            // inputs. Benches with trivial setup simply do not print it, and the
            // tool then reports the ratio as unknown instead of inventing 1x.
            const slabel = parts.next() orelse continue;
            const sms = parts.next() orelse continue;
            const v = std.fmt.parseFloat(f64, sms) catch continue;
            try setup_out.?.append(arena, .{ .label = try arena.dupe(u8, slabel), .ms = v, .rate = 0 });
            continue;
        }
        if (!std.mem.eql(u8, kind, "GEMM") and !std.mem.eql(u8, kind, "ACT")) continue;
        const label = parts.next() orelse continue;
        const ms_str = parts.next() orelse continue;
        const ms = std.fmt.parseFloat(f64, ms_str) catch continue;
        // The rate is the 4th field and is OPTIONAL: older harnesses and hand
        // written rows may not print one. Its absence is not a parse failure —
        // it only means absolute mode has no figure to show for that row.
        const rate: f64 = if (parts.next()) |rate_str|
            std.fmt.parseFloat(f64, rate_str) catch 0
        else
            0;
        // Both harnesses use the same labels; if they ever diverge, keying on
        // the label means a missing row is skipped rather than mismatched.
        //
        // The label MUST be copied. It is a slice into `text`, which points at
        // the child's stdout/stderr buffers, and `runHarness` frees those before
        // returning -- so a slice kept here is a dangling pointer. This was not
        // theoretical: the first version did exactly this and a full-table run
        // printed the row as "  ode gate/up [m=1,n=2048,k=7168]", three
        // characters of a label read out of freed memory. A measurement tool
        // that can print a corrupted row label can print a corrupted row VALUE.
        try out.append(arena, .{ .label = try arena.dupe(u8, label), .ms = ms, .rate = rate });
    }
}

/// `git rev-parse --short HEAD`, so a published table names the tree it
/// measured. Without this a table is not reproducible: the whole reason to
/// re-measure on demand is that the kernels move, and a ratio whose commit is
/// unknown cannot be checked against anything.
fn gitHead(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io) []const u8 {
    const argv = [_][]const u8{ "git", "rev-parse", "--short", "HEAD" };
    const res = std.process.run(gpa, io, .{ .argv = &argv }) catch return "unknown";
    defer gpa.free(res.stdout);
    defer gpa.free(res.stderr);
    switch (res.term) {
        .exited => |code| {
            if (code != 0) return "unknown";
            const trimmed = std.mem.trim(u8, res.stdout, " \t\n");
            if (trimmed.len == 0) return "unknown";
            return arena.dupe(u8, trimmed) catch "unknown";
        },
        else => return "unknown",
    }
}

/// Read a small /proc file with raw syscalls.
///
/// `std.fs.cwd()` and `std.fs.Dir.makePath` are both gone in Zig 0.16 -- exactly
/// the API rot AGENTS.md warns about for helpers nobody had exercised yet -- and
/// `Io.Dir` wants an `Io` threaded through every call for what is a three-line
/// read of a kernel-provided file. /proc needs no allocation, so openat/read is
/// both shorter and immune to the next std reshuffle.
fn readProcFile(buf: []u8, comptime path: [:0]const u8) usize {
    if (builtin.os.tag != .linux) return 0;
    const fd = std.os.linux.open(path, .{ .CLOEXEC = true }, 0);
    const signed_fd: isize = @bitCast(fd);
    if (signed_fd < 0) return 0;
    defer _ = std.os.linux.close(@intCast(fd));
    const n = std.os.linux.read(@intCast(fd), buf.ptr, buf.len);
    const signed_n: isize = @bitCast(n);
    return if (signed_n > 0) @intCast(signed_n) else 0;
}

fn readLoadAvg() [3]f64 {
    var out = [3]f64{ 0, 0, 0 };
    var buf: [128]u8 = undefined;
    const n = readProcFile(&buf, "/proc/loadavg");
    var it = std.mem.tokenizeAny(u8, buf[0..n], " \t\n");
    for (&out) |*v| {
        const tok = it.next() orelse break;
        v.* = std.fmt.parseFloat(f64, tok) catch 0;
    }
    return out;
}

fn readCpuModel() []const u8 {
    var buf: [8192]u8 = undefined;
    const n = readProcFile(&buf, "/proc/cpuinfo");
    var lines = std.mem.splitScalar(u8, buf[0..n], '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "model name")) {
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            return std.mem.trim(u8, line[colon + 1 ..], " \t");
        }
    }
    return "unknown";
}

fn pinSelf(cpu: usize) !void {
    var set: std.os.linux.cpu_set_t = @splat(0);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    try std.os.linux.sched_setaffinity(0, &set);
}
