// Attribution probe: which structural difference between the C++ reference
// kernel and the Zig port actually costs the performance?
//
// Every variant below computes the SAME math (C[m,n] = sum_k A[m,k]*B[n,k],
// BF16 in, F32 out) so only the loop structure / accumulator count / FMA usage
// varies. That isolates each factor:
//
//   V0  1 accumulator, m-outer/n-inner, mul+add   <- reproduces gemmExpert
//   V1  4 accumulators, m-outer/n-inner, mul+add   (ILP only)
//   V2  4 accumulators, m-outer/n-inner, FMA       (ILP + FMA)
//   V3  4 accumulators, n-outer/m-inner, FMA       (= the C++ reference shape)
//
// V3 matching the C++ numbers confirms the attribution.

const std = @import("std");
const root = @import("kt");
const amx = root.amx;

const bf16 = amx.bf16;
const Vec = amx.VecF32; // @Vector(8, f32)

const Shape = struct { name: []const u8, m: usize, n: usize, k: usize };

const shapes = [_]Shape{
    .{ .name = "decode gate/up", .m = 1, .n = 2048, .k = 7168 },
    .{ .name = "decode down", .m = 1, .n = 7168, .k = 2048 },
    .{ .name = "small decode", .m = 8, .n = 2048, .k = 7168 },
    .{ .name = "prefill 32", .m = 32, .n = 2048, .k = 7168 },
    .{ .name = "prefill 128", .m = 128, .n = 2048, .k = 7168 },
    .{ .name = "prefill 512", .m = 512, .n = 2048, .k = 7168 },
};

const REPS: usize = 5;
const WARMUP: usize = 2;

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

fn randF32(state: *u64) f32 {
    const bits: u32 = @truncate(splitmix64(state) >> 40);
    return @as(f32, @floatFromInt(bits)) * (1.0 / 8388608.0) - 1.0;
}

fn f32ToBf16Rne(x: f32) bf16 {
    const u: u32 = @bitCast(x);
    return @truncate((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

inline fn load8(p: [*]const bf16) Vec {
    var tmp: [8]f32 = undefined;
    for (0..8) |i| tmp[i] = amx.bf16_to_f32(p[i]);
    return tmp;
}

/// V0: exactly the shape of gemmExpert's inner loop — one accumulator,
/// m-outer / n-inner, separate mul and add.
fn v0(a: [*]const bf16, b: [*]const bf16, c: [*]f32, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: Vec = @splat(0.0);
            var kk: usize = 0;
            while (kk + 8 <= k) : (kk += 8)
                acc += load8(a + i * k + kk) * load8(b + j * k + kk);
            c[i * n + j] += amx.reduceAddFp32(acc);
        }
    }
}

/// V1: 4 independent accumulators (ILP), still mul+add, still m-outer.
fn v1(a: [*]const bf16, b: [*]const bf16, c: [*]f32, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var c1: Vec = @splat(0.0);
            var c2: Vec = @splat(0.0);
            var c3: Vec = @splat(0.0);
            var c4: Vec = @splat(0.0);
            var kk: usize = 0;
            while (kk + 32 <= k) : (kk += 32) {
                const ar = a + i * k + kk;
                const br = b + j * k + kk;
                c1 += load8(ar) * load8(br);
                c2 += load8(ar + 8) * load8(br + 8);
                c3 += load8(ar + 16) * load8(br + 16);
                c4 += load8(ar + 24) * load8(br + 24);
            }
            const acc = c1 + c2 + c3 + c4;
            c[i * n + j] += amx.reduceAddFp32(acc);
        }
    }
}

/// V2: 4 accumulators + explicit FMA.
fn v2(a: [*]const bf16, b: [*]const bf16, c: [*]f32, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var c1: Vec = @splat(0.0);
            var c2: Vec = @splat(0.0);
            var c3: Vec = @splat(0.0);
            var c4: Vec = @splat(0.0);
            var kk: usize = 0;
            while (kk + 32 <= k) : (kk += 32) {
                const ar = a + i * k + kk;
                const br = b + j * k + kk;
                c1 = @mulAdd(Vec, load8(ar), load8(br), c1);
                c2 = @mulAdd(Vec, load8(ar + 8), load8(br + 8), c2);
                c3 = @mulAdd(Vec, load8(ar + 16), load8(br + 16), c3);
                c4 = @mulAdd(Vec, load8(ar + 24), load8(br + 24), c4);
            }
            const acc = c1 + c2 + c3 + c4;
            c[i * n + j] += amx.reduceAddFp32(acc);
        }
    }
}

/// V3: the C++ reference shape — n-outer / m-inner (B row stays hot in L1
/// across all m) + 4 accumulators + FMA.
fn v3(a: [*]const bf16, b: [*]const bf16, c: [*]f32, m: usize, n: usize, k: usize) void {
    for (0..n) |j| {
        const br = b + j * k;
        for (0..m) |i| {
            var c1: Vec = @splat(0.0);
            var c2: Vec = @splat(0.0);
            var c3: Vec = @splat(0.0);
            var c4: Vec = @splat(0.0);
            const ar = a + i * k;
            var kk: usize = 0;
            while (kk + 32 <= k) : (kk += 32) {
                c1 = @mulAdd(Vec, load8(ar + kk), load8(br + kk), c1);
                c2 = @mulAdd(Vec, load8(ar + kk + 8), load8(br + kk + 8), c2);
                c3 = @mulAdd(Vec, load8(ar + kk + 16), load8(br + kk + 16), c3);
                c4 = @mulAdd(Vec, load8(ar + kk + 24), load8(br + kk + 24), c4);
            }
            const acc = c1 + c2 + c3 + c4;
            c[i * n + j] = amx.reduceAddFp32(acc);
        }
    }
}

const Variant = struct { name: []const u8, fn_: *const fn ([*]const bf16, [*]const bf16, [*]f32, usize, usize, usize) void };
const variants = [_]Variant{
    .{ .name = "V0 1acc mul+add m-outer (== gemmExpert)", .fn_ = v0 },
    .{ .name = "V1 4acc mul+add m-outer", .fn_ = v1 },
    .{ .name = "V2 4acc FMA      m-outer", .fn_ = v2 },
    .{ .name = "V3 4acc FMA      n-outer (== C++ shape)", .fn_ = v3 },
};

pub fn main(init: std.process.Init) !void {
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    const alloc = arena_state.allocator();
    _ = init.minimal;

    for (shapes) |s| {
        var rng: u64 = 0x243F6A8885A308D3;
        const a = try alloc.alloc(bf16, s.m * s.k);
        const b = try alloc.alloc(bf16, s.n * s.k);
        for (a) |*v| v.* = f32ToBf16Rne(randF32(&rng));
        for (b) |*v| v.* = f32ToBf16Rne(randF32(&rng));
        const c = try alloc.alloc(f32, s.m * s.n);

        std.debug.print("\n{s} [m={d},n={d},k={d}]\n", .{ s.name, s.m, s.n, s.k });
        for (variants) |v| {
            for (0..WARMUP) |_| {
                @memset(c, 0);
                v.fn_(a.ptr, b.ptr, c.ptr, s.m, s.n, s.k);
            }
            var best: u64 = std.math.maxInt(u64);
            for (0..REPS) |_| {
                @memset(c, 0);
                const t0 = nowNanos();
                v.fn_(a.ptr, b.ptr, c.ptr, s.m, s.n, s.k);
                const ns = nowNanos() - t0;
                if (ns < best) best = ns;
            }
            const ms = @as(f64, @floatFromInt(best)) / 1e6;
            const gf = 2.0 * @as(f64, @floatFromInt(s.m)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k)) / (ms * 1e6);
            std.debug.print("  {s:<44} {d:>9.3} ms  {d:>7.2} GF/s\n", .{ v.name, ms, gf });
        }
    }
}