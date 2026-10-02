// Tests for the SIMD work: the Cephes exp, round-to-nearest-even bf16 store,
// the shared softmax, and the restructured GEMM micro-kernels.
//
// Kept in its own file (rather than appended to test_kernels.zig) so the SIMD
// correctness surface has an obvious owner and does not entangle with the
// other kernel suites.
const std = @import("std");
const testing = std.testing;
const kt = @import("kt");

const amx = kt.amx;
const gemm_bf16 = kt.gemm_bf16;
const mla_core = kt.mla_core;

// ============================================================================
// exp
// ============================================================================


// Deterministic pseudo-random f32 in [-mod/2, mod/2) * scale. Shared by every
// test so the data is reproducible without a real RNG.
fn rnd(seed: *u64, mod: u32, scale: f32) f32 {
    seed.* = seed.* *% 6364136223846793005 +% 1442695040888963407;
    const bits: u16 = @truncate(seed.* >> 44);
    const centred: i32 = @as(i32, @intCast(bits % mod)) - @as(i32, @intCast(mod / 2));
    return @as(f32, @floatFromInt(centred)) * scale;
}

fn ulpDiff(a: f32, b: f32) u32 {
    return @intCast(@abs(@as(i32, @bitCast(a)) -% @as(i32, @bitCast(b))));
}

test "expScalar is within 1 ULP of @exp over the activation range" {
    var x: f32 = -87.0;
    while (x <= 87.0) : (x += 0.0031) {
        const got = amx.expScalar(x);
        const want: f32 = @floatCast(@exp(x));
        try testing.expect(std.math.isFinite(got));
        // <= 1 ULP: ~90% of results are bit-identical to @exp, the rest differ
        // by a single ulp of the final rounding.
        try testing.expect(ulpDiff(got, want) <= 1);
    }
}

test "expScalar edge cases" {
    try testing.expectEqual(@as(f32, 1.0), amx.expScalar(0.0));
    // clamp, not overflow/NaN: exp(90) and exp(-90) must stay finite and
    // normal (j is clamped into [-126, 126] so the exponent field is valid).
    const hi = amx.expScalar(90.0);
    const lo = amx.expScalar(-90.0);
    try testing.expect(std.math.isFinite(hi) and hi > 0);
    try testing.expect(std.math.isFinite(lo) and lo > 0);
    try testing.expectEqual(hi, amx.expScalar(87.0));
    try testing.expectEqual(lo, amx.expScalar(-87.0));
    // tiny |x| must be ~1 + x
    try testing.expectApproxEqAbs(@as(f32, 1.0001), amx.expScalar(0.0001), 1e-6);
}

test "expVec agrees with expScalar lane-for-lane, including the clamp" {
    var base: f32 = -90.0;
    while (base <= 90.0) : (base += 0.53) {
        var arr: [8]f32 = undefined;
        for (0..8) |l| arr[l] = base + @as(f32, @floatFromInt(l)) * 0.11;
        const v: amx.VecF32 = arr;
        const got: [8]f32 = amx.expVec(v);
        for (0..8) |l| {
            try testing.expectEqual(amx.expScalar(arr[l]), got[l]);
        }
    }
}

// ============================================================================
// bf16 store: round-to-nearest-even
// ============================================================================

test "f32_to_bf16 round-trips every bf16 pattern exactly" {
    var h: u32 = 0;
    while (h < 65536) : (h += 1) {
        const pattern: amx.bf16 = @truncate(h);
        try testing.expectEqual(pattern, amx.f32_to_bf16(amx.bf16_to_f32(pattern)));
    }
}

test "f32_to_bf16 implements ggml round-to-nearest-even" {
    // Reference is the documented ggml-impl.h formula, transcribed here so the
    // test pins the *rule* rather than hand-picked constants.
    const rne = struct {
        fn f(x: f32) amx.bf16 {
            const u: u32 = @bitCast(x);
            return @truncate((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
        }
    }.f;

    // Exact values are unaffected.
    try testing.expectEqual(@as(amx.bf16, 0x3F80), amx.f32_to_bf16(1.0));
    try testing.expectEqual(@as(amx.bf16, 0xC020), amx.f32_to_bf16(-2.5));
    try testing.expectEqual(@as(amx.bf16, 0x0000), amx.f32_to_bf16(0.0));
    try testing.expectEqual(@as(amx.bf16, 0x8000), amx.f32_to_bf16(-0.0));
    try testing.expectEqual(@as(amx.bf16, 0), amx.f32_to_bf16(1e-42)); // below bf16 min subnormal -> 0
    try testing.expect(amx.f32_to_bf16(1e-30) != 0); // but 1e-30 is a normal bf16

    // The midpoint above 1.0 (bf16 ulp near 1.0 is 2^-7, so the midpoint is
    // 2^-8) must round to EVEN, i.e. back DOWN to 0x3F80. Truncation also
    // gives 0x3F80, so use the odd-mantissa case to prove rounding happens at
    // all: 0x3F81's midpoint up must land on the even neighbour 0x3F82.
    const mid_1_0 = @as(f32, 1.0) + @as(f32, std.math.pow(f32, 2.0, -8));
    try testing.expectEqual(@as(amx.bf16, 0x3F80), amx.f32_to_bf16(mid_1_0)); // tie -> even
    const mid_odd = @as(f32, 1.0) + @as(f32, std.math.pow(f32, 2.0, -7)) + @as(f32, std.math.pow(f32, 2.0, -8));
    try testing.expectEqual(@as(amx.bf16, 0x3F82), amx.f32_to_bf16(mid_odd)); // tie -> even (up)
    try testing.expectEqual(@as(amx.bf16, 0x3F81), amx.f32_to_bf16(mid_odd - 1e-7)); // below -> down

    // Sweep the whole normal range (and below) against the reference rule.
    // Multiply UP so the loop terminates; check both signs each step.
    var x: f32 = 1e-30;
    var steps: usize = 0;
    while (x < 1e30) : (x *= 1.01) {
        try testing.expectEqual(rne(x), amx.f32_to_bf16(x));
        try testing.expectEqual(rne(-x), amx.f32_to_bf16(-x));
        steps += 1;
    }
    try testing.expect(steps > 1000); // guards against a degenerate sweep
}

test "f32_to_bf16_trunc keeps the old truncating behaviour" {
    const mid_up = @as(f32, 1.0) + @as(f32, std.math.pow(f32, 2.0, -8));
    try testing.expectEqual(@as(amx.bf16, 0x3F80), amx.f32_to_bf16_trunc(mid_up));
}

// ============================================================================
// softmax
// ============================================================================

fn softmaxRef(input: []const f32, out: []f32) void {
    var max_val = input[0];
    for (input) |v| if (v > max_val) {
        max_val = v;
    };
    var sum: f32 = 0;
    for (input, 0..) |v, i| {
        const e = @exp(v - max_val);
        out[i] = e;
        sum += e;
    }
    for (out) |*v| v.* /= sum;
}

test "softmaxF32 sums to 1 and matches the scalar reference (all tail cases)" {
    // n = 0..20 covers no-tail, the 8-lane fast path, and every tail length.
    var in_buf: [32]f32 = undefined;
    var got: [32]f32 = undefined;
    var want: [32]f32 = undefined;
    var n: usize = 1;
    while (n <= 20) : (n += 1) {
        for (0..n) |i| {
            in_buf[i] = @as(f32, @floatFromInt(@as(u32, @intCast(i)) * 7 % @as(u32, 19))) * 0.5 - 4.0;
        }
        amx.softmaxF32(in_buf[0..n].ptr, got[0..n].ptr, n);
        softmaxRef(in_buf[0..n], want[0..n]);

        var sum: f32 = 0;
        for (0..n) |i| {
            try testing.expectApproxEqAbs(want[i], got[i], 1e-5);
            sum += got[i];
        }
        try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    }
}

test "softmaxF32 is in-place safe and overflow-free" {
    var buf: [8]f32 = .{ 1000.0, 1001.0, 999.0, 1002.0, 1000.5, 998.0, 1003.0, 1000.25 };
    const want: [8]f32 = buf;
    amx.softmaxF32(&buf, &buf, buf.len);
    for (buf) |v| {
        try testing.expect(std.math.isFinite(v));
        try testing.expect(v > 0.0);
    }
    var sum: f32 = 0;
    for (buf) |v| sum += v;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    try testing.expect(buf[3] > buf[2]); // argmax keeps the largest share
    // in-place must equal out-of-place
    var other: [8]f32 = want;
    amx.softmaxF32(&want, &other, want.len);
    for (0..8) |i| try testing.expectApproxEqAbs(other[i], buf[i], 1e-7);
}

// ============================================================================
// gemmExpert
// ============================================================================

/// Naive triple-loop reference: C[i][j] = sum_k A[i][k] * B[j][k].
fn gemmRef(a: []const amx.bf16, b: []const amx.bf16, c: []f32, m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                sum += amx.bf16_to_f32(a[i * lda + kk]) * amx.bf16_to_f32(b[j * ldb + kk]);
            }
            c[i * ldc + j] = sum;
        }
    }
}

fn fillRow(buf: []amx.bf16, seed: *u64) void {
    for (buf) |*v| v.* = amx.f32_to_bf16(rnd(seed, 21, 0.05));
}

test "gemmExpert matches the naive reference across shapes and tail cases" {
    // m/n/k chosen to exercise: MR tail (m % 4), 32-block tail (k % 32),
    // 8-lane tail (k % 8), and the fully-scalar tail (k % 1).
    const cases = [_][3]usize{
        .{ 1, 1, 8 },
        .{ 1, 3, 32 },
        .{ 4, 5, 64 },
        .{ 5, 7, 40 },
        .{ 8, 9, 33 },
        .{ 13, 11, 71 },
        .{ 16, 16, 128 },
        .{ 3, 2, 1 },
        .{ 7, 1, 7 },
        .{ 17, 13, 97 },
    };
    for (cases) |case| {
        const m = case[0];
        const n = case[1];
        const k = case[2];
        const alloc = testing.allocator;
        const a = try alloc.alloc(amx.bf16, m * k);
        defer alloc.free(a);
        const b = try alloc.alloc(amx.bf16, n * k);
        defer alloc.free(b);
        const got = try alloc.alloc(f32, m * n);
        defer alloc.free(got);
        const want = try alloc.alloc(f32, m * n);
        defer alloc.free(want);

        var seed: u64 = 0x9E3779B97F4A7C15;
        fillRow(a, &seed);
        fillRow(b, &seed);

        gemm_bf16.gemmExpert(a.ptr, b.ptr, got.ptr, m, n, k, k, k, n);
        gemmRef(a, b, want, m, n, k, k, k, n);

        for (want, got) |w, g| {
            // bf16 inputs (8-bit mantissa) summed over k; tolerance scaled to
            // the magnitude of the accumulator.
            try testing.expectApproxEqAbs(w, g, 0.02 * @as(f32, @floatFromInt(@max(k, 1))) * 0.05);
        }
    }
}

test "gemmExpert honours non-default leading dimensions" {
    const m: usize = 5;
    const n: usize = 6;
    const k: usize = 40;
    const lda: usize = k + 7; // padded
    const ldb: usize = k + 3;
    const ldc: usize = n + 5;
    const alloc = testing.allocator;
    const a = try alloc.alloc(amx.bf16, m * lda);
    defer alloc.free(a);
    const b = try alloc.alloc(amx.bf16, n * ldb);
    defer alloc.free(b);
    const got = try alloc.alloc(f32, m * ldc);
    defer alloc.free(got);
    const want = try alloc.alloc(f32, m * ldc);
    defer alloc.free(want);
    @memset(got, 0);
    @memset(want, 0);

    var seed: u64 = 0xDEADBEEF;
    for (0..m) |i| fillRow(a[i * lda ..][0..k], &seed);
    for (0..n) |j| fillRow(b[j * ldb ..][0..k], &seed);

    gemm_bf16.gemmExpert(a.ptr, b.ptr, got.ptr, m, n, k, lda, ldb, ldc);
    gemmRef(a, b, want, m, n, k, lda, ldb, ldc);

    for (0..m) |i| {
        for (0..n) |j| {
            try testing.expectApproxEqAbs(want[i * ldc + j], got[i * ldc + j], 0.2);
        }
        // padding columns must be untouched
        for (n..ldc) |j| try testing.expectEqual(@as(f32, 0), got[i * ldc + j]);
    }
}

test "gemmExpert overwrites (does not accumulate into) its output" {
    const m: usize = 4;
    const n: usize = 4;
    const k: usize = 32;
    const alloc = testing.allocator;
    const a = try alloc.alloc(amx.bf16, m * k);
    defer alloc.free(a);
    const b = try alloc.alloc(amx.bf16, n * k);
    defer alloc.free(b);
    const c = try alloc.alloc(f32, m * n);
    defer alloc.free(c);
    var seed: u64 = 12345;
    fillRow(a, &seed);
    fillRow(b, &seed);

    gemm_bf16.gemmExpert(a.ptr, b.ptr, c.ptr, m, n, k, k, k, n);
    const once = try alloc.dupe(f32, c);
    defer alloc.free(once);
    // Pre-filled with garbage: an overwrite must ignore it, an accumulate
    // would roughly double the result.
    @memset(c, 1000.0);
    gemm_bf16.gemmExpert(a.ptr, b.ptr, c.ptr, m, n, k, k, k, n);
    for (once, c) |x, y| try testing.expectApproxEqAbs(x, y, 1e-4);
}

test "gemmExpertAccum sums partial products (tensor-parallel rank loop)" {
    // Regression guard for the tp>1 bug: forwardDown used to call the
    // overwrite-only entry point once per rank into one shared buffer, so
    // every rank but the last was silently discarded.
    const m: usize = 4;
    const n: usize = 4;
    const k: usize = 64;
    const alloc = testing.allocator;
    const a = try alloc.alloc(amx.bf16, m * k);
    defer alloc.free(a);
    const b = try alloc.alloc(amx.bf16, n * k);
    defer alloc.free(b);
    const c = try alloc.alloc(f32, m * n);
    defer alloc.free(c);
    const whole = try alloc.alloc(f32, m * n);
    defer alloc.free(whole);
    var seed: u64 = 777;
    fillRow(a, &seed);
    fillRow(b, &seed);

    @memset(c, 0);
    // Two ranks, each contributing half of the k axis.
    const k_half = k / 2;
    gemm_bf16.gemmExpertAccum(a.ptr, b.ptr, c.ptr, m, n, k_half, k, k, n);
    gemm_bf16.gemmExpertAccum(a.ptr + k_half, b.ptr + k_half, c.ptr, m, n, k - k_half, k, k, n);

    gemmRef(a, b, whole, m, n, k, k, k, n);
    for (whole, c) |w, g| try testing.expectApproxEqAbs(w, g, 0.2);
}

test "gemmExpert handles degenerate shapes without crashing" {
    var scratch: [64]f32 = undefined;
    const w: [*]const amx.bf16 = @ptrCast(&scratch);
    const x: [*]const amx.bf16 = @ptrCast(&scratch);
    gemm_bf16.gemmExpert(x, w, &scratch, 0, 4, 8, 8, 8, 4);
    gemm_bf16.gemmExpert(x, w, &scratch, 4, 0, 8, 8, 8, 4);
    gemm_bf16.gemmExpert(x, w, &scratch, 4, 4, 0, 8, 8, 4);
}

// ============================================================================
// Shared f32 x BF16 GEMM (MLA matmulF32 / MHA matmulF32)
// ============================================================================

fn f32GemmRef(a: []const f32, b: []const amx.bf16, c: []f32, m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                sum += a[i * lda + kk] * amx.bf16_to_f32(b[j * ldb + kk]);
            }
            c[i * ldc + j] = sum;
        }
    }
}

test "gemmF32Bf16 matches the naive reference (m n k tail cases)" {
    const cases = [_][3]usize{
        .{ 1, 1, 8 },
        .{ 4, 4, 32 },
        .{ 5, 3, 41 },
        .{ 9, 7, 33 },
        .{ 2, 2, 1 },
        .{ 13, 11, 96 },
    };
    for (cases) |case| {
        const m = case[0];
        const n = case[1];
        const k = case[2];
        const alloc = testing.allocator;
        const a = try alloc.alloc(f32, m * k);
        defer alloc.free(a);
        const b = try alloc.alloc(amx.bf16, n * k);
        defer alloc.free(b);
        const got = try alloc.alloc(f32, m * n);
        defer alloc.free(got);
        const want = try alloc.alloc(f32, m * n);
        defer alloc.free(want);

        var seed: u64 = 0xABCDEF;
        for (a) |*v| v.* = rnd(&seed, 17, 0.06);
        fillRow(b, &seed);

        amx.gemmF32Bf16(a.ptr, k, b.ptr, k, 0, got.ptr, n, m, n, k);
        f32GemmRef(a, b, want, m, n, k, k, k, n);
        for (want, got) |w, g| {
            try testing.expectApproxEqAbs(w, g, 0.02 * @as(f32, @floatFromInt(@max(k, 1))) * 0.05);
        }
    }
}

test "matmulF32Offset selects the right weight rows" {
    const m: usize = 3;
    const n: usize = 4;
    const k: usize = 40;
    const total_rows: usize = 9;
    const offset: usize = 5; // rows [5, 9) of the weight matrix
    const alloc = testing.allocator;
    const a = try alloc.alloc(f32, m * k);
    defer alloc.free(a);
    const b = try alloc.alloc(amx.bf16, total_rows * k);
    defer alloc.free(b);
    const got = try alloc.alloc(f32, m * n);
    defer alloc.free(got);
    const want = try alloc.alloc(f32, m * n);
    defer alloc.free(want);

    var seed: u64 = 0x5EED;
    for (a) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        v.* = @as(f32, @floatFromInt(@as(u16, @truncate(seed >> 44)) % @as(u16, 13))) * 0.07 - 0.4;
    }
    fillRow(b, &seed);

    mla_core.matmulF32Offset(a.ptr, k, b.ptr, k, offset, got.ptr, n, m, n, k);
    f32GemmRef(a, b[offset * k ..], want, m, n, k, k, k, n);
    for (want, got) |w, g| try testing.expectApproxEqAbs(w, g, 0.3);
}

test "mla matmulF32 honours leading dimensions" {
    const m: usize = 5;
    const n: usize = 6;
    const k: usize = 40;
    const lda: usize = k + 5;
    const ldb: usize = k + 2;
    const ldc: usize = n + 4;
    const alloc = testing.allocator;
    const a = try alloc.alloc(f32, m * lda);
    defer alloc.free(a);
    const b = try alloc.alloc(amx.bf16, n * ldb);
    defer alloc.free(b);
    const got = try alloc.alloc(f32, m * ldc);
    defer alloc.free(got);
    const want = try alloc.alloc(f32, m * ldc);
    defer alloc.free(want);
    @memset(got, 0);
    @memset(want, 0);

    var seed: u64 = 4242;
    for (0..m) |i| {
        for (0..k) |kk| a[i * lda + kk] = rnd(&seed, 11, 0.09);
    }
    for (0..n) |j| fillRow(b[j * ldb ..][0..k], &seed);

    mla_core.matmulF32(a.ptr, lda, b.ptr, ldb, got.ptr, ldc, m, n, k);
    f32GemmRef(a, b, want, m, n, k, lda, ldb, ldc);
    for (0..m) |i| {
        for (0..n) |j| try testing.expectApproxEqAbs(want[i * ldc + j], got[i * ldc + j], 0.3);
        for (n..ldc) |j| try testing.expectEqual(@as(f32, 0), got[i * ldc + j]);
    }
}

// ============================================================================
// Activation sanity (the exp change flows through these)
// ============================================================================

test "swiglu vector path still matches the scalar path" {
    var gate: [8]f32 = undefined;
    var up: [8]f32 = undefined;
    var seed: u64 = 0xC0FFEE;
    for (&gate) |*v| v.* = rnd(&seed, 9, 0.8) - 0.4;
    for (&up) |*v| v.* = rnd(&seed, 7, 0.6) - 0.3;

    const gv: amx.VecF32 = gate;
    const uv: amx.VecF32 = up;
    const got: [8]f32 = amx.swigluVec(gv, uv);
    for (0..8) |l| {
        try testing.expectApproxEqAbs(amx.swiglu(gate[l], up[l]), got[l], 1e-5);
    }

    // clamped variant
    const lim: amx.VecF32 = @splat(0.5);
    const gc: [8]f32 = amx.swigluClampVec(gv, uv, lim);
    for (0..8) |l| {
        try testing.expectApproxEqAbs(amx.swiglu_clamp(gate[l], up[l], 0.5), gc[l], 1e-5);
    }

    // oai variant
    const alpha: amx.VecF32 = @splat(1.5);
    const go: [8]f32 = amx.swigluOaiVec(gv, uv, alpha, lim);
    for (0..8) |l| {
        try testing.expectApproxEqAbs(amx.swiglu_oai(gate[l], up[l], 1.5, 0.5), go[l], 1e-5);
    }
}
