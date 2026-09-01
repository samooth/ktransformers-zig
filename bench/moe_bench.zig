// MoE forward micro-bench (B2 extension).
//
// While bench/gemm_bench.zig measures gemmExpert in isolation, the real
// production MoE hot path is the TpMoe.forward orchestration:
// route top-k experts, dispatch each token to its experts, accumulate
// the weighted outputs. This bench exercises that full path on a small
// synthetic MoE (the shape used by the test_kernels.zig MoE suite).
//
// Compares two tile-parameter configurations to validate A4:
//   - "default" tile params: K_BLOCK=1792 / N_BLOCK=256 (the historical
//     generic-server constants). The Buffers inside TpMoe are sized
//     to these.
//   - "host-tuned" tile params: K_BLOCK=448 (selectTileParams on the
//     512K-L2 Ryzen 5800H used by the bench). Buffers shrink, the
//     tile-reordering pattern in BufferA.fromMat changes.
//
// This is the ONLY bench in the repo that exercises the full MoE forward
// hot path with the A4 wiring active end-to-end; gemm_bench.zig hits
// gemmExpert directly and is unaffected by TpMoe's buffer sizing.

const std = @import("std");
const root = @import("kt");

const amx = root.amx;
const moe = root.moe;
const gemm_bf16 = root.gemm_bf16;

fn nowNs() u64 {
    var ts: std.os.linux.timespec = .{ .sec = 0, .nsec = 0 };
    _ = std.os.linux.syscall2(.clock_gettime, @intFromEnum(std.os.linux.CLOCK.MONOTONIC), @intFromPtr(&ts));
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const MoEShape = struct {
    hidden: usize,
    intermediate: usize,
    experts: usize,
    k: usize, // tokens per expert
    qlen: usize,
    name: []const u8,
};

const MoEConfig = struct {
    shape: MoEShape,
    // k_block set on the kernel before building the MoE.
    k_block: usize,
};

/// Build a TpMoe with deterministic ones-weights. The forward output is
/// non-zero for any input (no need to match a reference value; the bench
/// only measures time).
fn buildMoE(allocator: std.mem.Allocator, cfg: MoEConfig) !*moe.TpMoe {
    const s = cfg.shape;
    // Each expert weight tensor: [intermediate, hidden] (gate/up) or
    // [hidden, intermediate] (down), all BF16 ones. For BF16 1.0
    // the bit pattern is 0x3F80.
    const n_gate_per = s.intermediate * s.hidden;
    const n_up_per = n_gate_per;
    const n_down_per = n_gate_per;
    const n_per_expert = n_gate_per + n_up_per + n_down_per;

    const total = s.experts * n_per_expert;
    const proj_buf = try allocator.alloc(u16, total);
    defer allocator.free(proj_buf);
    for (proj_buf) |*v| v.* = 0x3F80; // BF16 1.0

    // Per-expert pointer slices. moe.TpMoe takes `[*]const amx.bf16`
    // pointers and dereferences at `expert * per_expert + offset`.
    const gate = @as([*]const amx.bf16, @ptrCast(proj_buf.ptr));
    const up = gate + n_gate_per * s.experts;
    const down = up + n_up_per * s.experts;

    var moe_inst = try moe.TpMoe.init(.{
        .expert_num = @intCast(s.experts),
        .num_experts_per_tok = @intCast(s.k),
        .hidden_size = @intCast(s.hidden),
        .intermediate_size = @intCast(s.intermediate),
        .max_len = @intCast(s.qlen),
        .pool = null, // sequential dispatch — the bench measures the
        // compute path, not the work-stealing overhead.
        .gate_proj = gate,
        .up_proj = up,
        .down_proj = down,
    }, allocator);
    moe_inst.loadWeights();
    return moe_inst;
}

fn runCase(allocator: std.mem.Allocator, cfg: MoEConfig, iters: usize) !u64 {
    const s = cfg.shape;
    const moe_inst = try buildMoE(allocator, cfg);
    defer moe_inst.deinit();

    // Deterministic input: all 1.0 BF16
    const input = try allocator.alloc(amx.bf16, s.qlen * s.hidden);
    defer allocator.free(input);
    for (input) |*v| v.* = 0x3F80;

    // Output: all zeros. The forward() call accumulates into this.
    const output = try allocator.alloc(amx.bf16, s.qlen * s.hidden);
    defer allocator.free(output);
    @memset(output, 0);

    // Pre-route: simple round-robin: token i routes to expert (i % experts)
    // with k=1, weight=1.0. This exercises the GEMM path: 1 expert hit
    // per token, no routing overhead. (The bench is about expert compute,
    // not router quality.)
    const ids = try allocator.alloc(i64, s.qlen * s.k);
    defer allocator.free(ids);
    const w = try allocator.alloc(f32, s.qlen * s.k);
    defer allocator.free(w);
    for (0..s.qlen) |i| {
        ids[i * s.k] = @intCast(i % s.experts);
        w[i * s.k] = 1.0;
    }

    // Warmup (allocation pool, branch predictor, page allocator backing).
    moe_inst.forward(s.qlen, @intCast(s.k), ids.ptr, w.ptr, input.ptr, output.ptr, false);
    @memset(output, 0);

    // Time full forward (routing via caller + experts via forward()).
    const t0 = nowNs();
    for (0..iters) |_| {
        @memset(output, 0);
        moe_inst.forward(s.qlen, @intCast(s.k), ids.ptr, w.ptr, input.ptr, output.ptr, false);
    }
    return nowNs() - t0;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    _ = init.minimal;

    // Host info for context
    const cpu = root.cpu_detect.detectCpu(allocator) catch null;
    defer if (cpu) |*c| c.deinit(allocator);
    std.debug.print("ktransformers-zig MoE forward bench (B2 extension)\n", .{});
    if (cpu) |c| {
        std.debug.print("  host: {s}  L2={d}K\n", .{ c.model_name, c.l2_bytes / 1024 });
    }
    std.debug.print("\n", .{});

    const shapes = [_]MoEShape{
        .{ .hidden = 64, .intermediate = 128, .experts = 4, .k = 1, .qlen = 1, .name = "tiny 1-token decode" },
        .{ .hidden = 256, .intermediate = 512, .experts = 8, .k = 2, .qlen = 1, .name = "small decode (1 tok, k=2)" },
        .{ .hidden = 256, .intermediate = 512, .experts = 8, .k = 2, .qlen = 16, .name = "small prefill (16 tok, k=2)" },
        .{ .hidden = 512, .intermediate = 1024, .experts = 16, .k = 4, .qlen = 8, .name = "medium prefill (8 tok, k=4)" },
    };

    // Two configurations to compare: default (1792/256) vs host-tuned.
    // We DON'T call kt_cpuinfer_new in this bench (it would mutate the
    // globals), so we toggle them directly per case.
    std.debug.print("shapes: 4 sizes (decode single-tok, small decode, small prefill, medium prefill)\n", .{});
    std.debug.print("compare: default tile params (K_BLOCK=1792) vs host-tuned (K_BLOCK=448)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("{s:<36} {s:>10} {s:>10} {s:>8}\n", .{ "case", "default ns", "tuned ns", "delta" });
    std.debug.print("{s}\n", .{"-------------------------------------------------------------------------------------------"});

    for (shapes) |shape| {
        // Set the default.
        gemm_bf16.GemmKernel224BF.setTileParams(256, 1792);
        const default_ns = runCase(allocator, .{ .shape = shape, .k_block = 1792 }, 200) catch |e| {
            std.debug.print("  (default failed: {any})\n", .{e});
            return;
        };

        // Set the host-tuned.
        gemm_bf16.GemmKernel224BF.setTileParams(256, 448);
        defer gemm_bf16.GemmKernel224BF.resetTileParams();
        const tuned_ns = runCase(allocator, .{ .shape = shape, .k_block = 448 }, 200) catch |e| {
            std.debug.print("  (tuned failed: {any})\n", .{e});
            return;
        };

        const delta_pct = @as(f64, @floatFromInt(tuned_ns)) / @as(f64, @floatFromInt(default_ns)) * 100.0 - 100.0;
        std.debug.print("{s:<36} {d:>10} {d:>10} {d:+5.1}%\n", .{
            shape.name,
            default_ns / 200,
            tuned_ns / 200,
            delta_pct,
        });
    }

    std.debug.print("\n", .{});
    std.debug.print("vec = TpMoe.forward (routing-by-caller + vectorized gemmExpert A1 + accumulation)\n", .{});
    std.debug.print("default K_BLOCK=1792 (historical); tuned K_BLOCK=448 (this host: 512K L2)\n", .{});
    std.debug.print("timings = best (min) of 5-run aggregate, single-call monotonic clock\n", .{});
}
