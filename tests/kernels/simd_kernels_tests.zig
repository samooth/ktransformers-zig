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

// ============================================================================
// K-quant GEMMs (AVX2 path): the invariant is `fused == dequantizeRow + dot`
// ============================================================================
//
// One test body, six format instantiations. The invariant contains NO
// approximation: `dequantizeRowQ*_K` + an f32 dot product is the definition of
// the kernel, so any disagreement is a bug in the vector path (a swapped
// nibble, a wrong scale index, a lane permutation). It is also the only
// invariant that can see those bugs: the older constant-weight tests cannot,
// because a constant weight row is invariant under a permuted lane mapping.
//
// Why this lives here and not in the kernel files: `test` blocks inside
// src/kernels/amx/gemm_224_q*_k.zig are NOT executed by `zig build test` —
// they sit behind the `kt` module boundary, so Zig collects tests only from
// the root file's own module. tests/kernels/test_kernels.zig therefore holds
// its own copies of the per-format K-quant tests (148 of them). Keep the
// vectorized-GEMM coverage here, next to the other SIMD kernels.

const q2_k = kt.gemm_q2_k;
const q3_k = kt.gemm_q3_k;
const q4_k = kt.gemm_q4_k;
const q5_k = kt.gemm_q5_k;
const q6_k = kt.gemm_q6_k;
const q8_k = kt.gemm_q8_k;

const KQ = struct {
    /// deterministic splitmix64, matching bench/kquant_bench.zig's data
    fn rand(state: *u64) u32 {
        var z = state.* +% 0x9E3779B97F4A7C15;
        state.* = z;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        return @truncate(z ^ (z >> 31));
    }

    /// Fuzz the per-super-block scale/min fields with a NON-UNIFORM pattern.
    /// A uniform scale would make the whole block look right even if the
    /// kernel indexed the scale array wrongly (and the reverse: real
    /// quantizers produce near-uniform scales within a super-block, which is
    /// exactly why the old tests could not see it).
    fn vary8K(blk: *q8_k.BlockQ8_K, state: *u64) void {
        blk.d = @as(f32, @floatFromInt(@as(i32, @intCast(rand(state) % 9)) + 1)) * 0.015625;
    }
    fn vary6K(blk: *q6_k.BlockQ6_K, state: *u64) void {
        for (&blk.scales) |*s| s.* = @intCast(@as(i32, @intCast(rand(state) % 121)) - 60);
        blk.d = q6_k.f32_to_f16(@as(f32, @floatFromInt(@as(i32, @intCast(rand(state) % 7)) + 1)) * 0.015625);
    }
    fn vary5K(blk: *q5_k.BlockQ5_K, state: *u64) void {
        for (&blk.scales) |*s| s.* = @truncate(rand(state));
        for (&blk.qh) |*h| h.* = @truncate(rand(state));
        blk.d = q5_k.f32_to_f16(@as(f32, @floatFromInt(@as(i32, @intCast(rand(state) % 7)) + 1)) * 0.015625);
        blk.dmin = q5_k.f32_to_f16(@as(f32, @floatFromInt(@as(i32, @intCast(rand(state) % 5)) + 1)) * 0.03125);
    }
    fn vary4K(blk: *q4_k.BlockQ4_K, state: *u64) void {
        for (&blk.scales) |*s| s.* = @truncate(rand(state));
        blk.d = q4_k.f32_to_f16(@as(f32, @floatFromInt(@as(i32, @intCast(rand(state) % 7)) + 1)) * 0.015625);
        blk.dmin = q4_k.f32_to_f16(@as(f32, @floatFromInt(@as(i32, @intCast(rand(state) % 5)) + 1)) * 0.03125);
    }
    fn vary3K(blk: *q3_k.BlockQ3_K, state: *u64) void {
        for (&blk.scales) |*s| s.* = @truncate(rand(state));
        for (&blk.hmask) |*h| h.* = @truncate(rand(state));
        blk.d = q3_k.f32_to_f16(@as(f32, @floatFromInt(@as(i32, @intCast(rand(state) % 7)) + 1)) * 0.015625);
    }
    fn vary2K(blk: *q2_k.BlockQ2_K, state: *u64) void {
        for (&blk.scales) |*s| s.* = @truncate(rand(state));
        blk.d = q2_k.f32_to_f16(@as(f32, @floatFromInt(@as(i32, @intCast(rand(state) % 7)) + 1)) * 0.015625);
        blk.dmin = q2_k.f32_to_f16(@as(f32, @floatFromInt(@as(i32, @intCast(rand(state) % 5)) + 1)) * 0.03125);
    }

    /// The one test body. `M` covers 1 (the LlamaMoe decode path, which never
    /// fills the MR=4 tile), 4 (exactly one tile) and 5 (tile + row tail);
    /// `KB = 3` super-blocks cover k accumulation across block boundaries.
    fn check(
        comptime block: type,
        comptime QK: usize,
        comptime dequant: fn ([*]const block, []f32, usize) void,
        comptime quantize: fn (src: []const f32, dst: [*]block, k: usize) void,
        comptime vary: fn (*block, *u64) void,
        comptime gemm_scalar: fn ([*]const amx.bf16, usize, [*]const block, usize, [*]f32, usize, usize, usize, usize) void,
        comptime gemm_vec: fn ([*]const amx.bf16, usize, [*]const block, usize, [*]f32, usize, usize, usize, usize) void,
    ) !void {
        const N: usize = 3;
        const KB: usize = 3;
        const K: usize = QK * KB;
        var state: u64 = 0x243F6A8885A308D3;

        const alloc = testing.allocator;
        const a = try alloc.alloc(amx.bf16, 5 * K);
        defer alloc.free(a);
        const b = try alloc.alloc(block, N * KB);
        defer alloc.free(b);
        const got = try alloc.alloc(f32, 5 * N);
        defer alloc.free(got);
        const scal = try alloc.alloc(f32, 5 * N);
        defer alloc.free(scal);
        const w = try alloc.alloc(f32, K);
        defer alloc.free(w);

        // Activations: mixed sign and magnitude, bf16-rounded.
        for (a) |*v| {
            v.* = amx.f32_to_bf16(@as(f32, @floatFromInt(@as(i32, @intCast(rand(&state) % 17)) - 8)) * 0.125);
        }
        // Weights: quantized from varied data so every plane/scale field is
        // populated, then the scale fields are fuzzed (see `vary`).
        var src: [QK]f32 = undefined;
        for (0..N) |j| {
            for (0..KB) |sb| {
                for (&src) |*v| {
                    v.* = @as(f32, @floatFromInt(@as(i32, @intCast(rand(&state) % 23)) - 11)) * 0.0625;
                }
                quantize(&src, @ptrCast(&b[j * KB + sb]), QK);
                vary(&b[j * KB + sb], &state);
            }
        }

        for ([_]usize{ 1, 4, 5 }) |m| {
            gemm_vec(a.ptr, K, b.ptr, KB, got.ptr, N, m, N, K);
            gemm_scalar(a.ptr, K, b.ptr, KB, scal.ptr, N, m, N, K);
            for (0..m) |i| {
                for (0..N) |j| {
                    dequant(@ptrCast(&b[j * KB]), w, K);
                    var expect: f32 = 0;
                    var mag: f32 = 0; // sum |a*w| — cancellation-safe scale
                    for (0..K) |t| {
                        const p = amx.bf16_to_f32(a[i * K + t]) * w[t];
                        expect += p;
                        mag += @abs(p);
                    }
                    const tol = 1e-5 * mag + 1e-6;
                    try testing.expect(@abs(got[i * N + j] - expect) <= tol);
                    try testing.expect(@abs(scal[i * N + j] - expect) <= tol);
                }
            }
        }
    }
};

test "K-quant AVX2 GEMMs == dequantize + dot (Q8_K)" {
    try KQ.check(
        q8_k.BlockQ8_K,
        q8_k.QK_K,
        q8_k.dequantizeRowQ8_K,
        q8_k.quantizeRowQ8_K,
        KQ.vary8K,
        q8_k.gemmQ8_KScalar,
        q8_k.gemmQ8_K,
    );
}

test "K-quant AVX2 GEMMs == dequantize + dot (Q6_K)" {
    try KQ.check(
        q6_k.BlockQ6_K,
        q6_k.QK_K,
        q6_k.dequantizeRowQ6_K,
        q6_k.quantizeRowQ6_K,
        KQ.vary6K,
        q6_k.gemmQ6_KScalar,
        q6_k.gemmQ6_K,
    );
}

test "K-quant AVX2 GEMMs == dequantize + dot (Q5_K)" {
    try KQ.check(
        q5_k.BlockQ5_K,
        q5_k.QK_K,
        q5_k.dequantizeRowQ5_K,
        q5_k.quantizeRowQ5_K,
        KQ.vary5K,
        q5_k.gemmQ5_KScalar,
        q5_k.gemmQ5_K,
    );
}

test "K-quant AVX2 GEMMs == dequantize + dot (Q4_K)" {
    try KQ.check(
        q4_k.BlockQ4_K,
        q4_k.QK_K,
        q4_k.dequantizeRowQ4_K,
        q4_k.quantizeRowQ4_K,
        KQ.vary4K,
        q4_k.gemmQ4_KScalar,
        q4_k.gemmQ4_K,
    );
}

test "K-quant AVX2 GEMMs == dequantize + dot (Q3_K)" {
    try KQ.check(
        q3_k.BlockQ3_K,
        q3_k.QK_K,
        q3_k.dequantizeRowQ3_K,
        q3_k.quantizeRowQ3_K,
        KQ.vary3K,
        q3_k.gemmQ3_KScalar,
        q3_k.gemmQ3_K,
    );
}

test "K-quant AVX2 GEMMs == dequantize + dot (Q2_K)" {
    try KQ.check(
        q2_k.BlockQ2_K,
        q2_k.QK_K,
        q2_k.dequantizeRowQ2_K,
        q2_k.quantizeRowQ2_K,
        KQ.vary2K,
        q2_k.gemmQ2_KScalar,
        q2_k.gemmQ2_K,
    );
}

// ============================================================================
// MX format block layouts (ue8m0)
// ============================================================================
//
// These exist because the MX blocks shipped with `scale: amx.bf16`, which made
// MXFP8 34 bytes and MXFP4 18 bytes against the format's 33 and 17, and used a
// scale encoding MXFP does not have. The formats therefore could not describe a
// real MXFP checkpoint at all. That is the same class of defect as the KT_TYPE
// / block-size regression in eec0c22 (11 invented enum values, 9 wrong block
// sizes), and the existing tests could not catch it: they populated
// `block.scale` through our own field type, so they were self-consistent by
// construction no matter how wrong the layout was.
//
// The invariant these pin is the one that matters for interop: the scale is a
// single BYTE holding an unbiased power of two (ue8m0), matching
// kt-kernel/operators/avx2/mxfp8-moe.hpp ("Scale: ue8m0 per-group
// (group_size=32)") and GGUF's MXFP4/MXFP8 tensor types.

test "MX block sizes: MXFP8 is 33 bytes, MXFP4 is 17" {
    try testing.expectEqual(@as(usize, 33), @sizeOf(kt.gemm_mxfp8.MXFP8Block));
    try testing.expectEqual(@as(usize, 17), @sizeOf(kt.gemm_mxfp4.MXFP4Block));
    // 1 scale byte + payload; a padding regression would show up here too.
    try testing.expectEqual(@as(usize, 1), @sizeOf(@FieldType(kt.gemm_mxfp8.MXFP8Block, "scale")));
    try testing.expectEqual(@as(usize, 1), @sizeOf(@FieldType(kt.gemm_mxfp4.MXFP4Block, "scale")));
}

test "ue8m0 -> f32 matches the reference's b << 23 for all 256 bytes" {
    // Verbatim semantics of convert_ue8m0_to_fp32 in the AVX2 reference,
    // including the two edges it documents rather than fixes: b=0 gives 0.0
    // (not 2^-127) and b=255 overflows the exponent field into NaN.
    const ue = kt.gemm_mxfp8.ue8m0_to_f32;
    for (0..256) |b| {
        const bits: u32 = @as(u32, @intCast(b)) << 23;
        try testing.expectEqual(@as(f32, @bitCast(bits)), ue(@intCast(b)));
    }
    // The identity that matters in practice: 127 -> 1.0, one step per bit.
    try testing.expectEqual(@as(f32, 1.0), ue(127));
    try testing.expectEqual(@as(f32, 2.0), ue(128));
    try testing.expectEqual(@as(f32, 0.5), ue(126));
    try testing.expectEqual(@as(f32, 0.0), ue(0));
}

test "ue8m0 scales are powers of two (what bf16 was not)" {
    // Every representable scale is an exact power of two. A bf16 scale can
    // carry a mantissa, so this loop is the direct regression test for the
    // layout bug: under the old struct, populated values failed here.
    const ue = kt.gemm_mxfp8.ue8m0_to_f32;
    for (1..255) |b| {
        const v = ue(@intCast(b));
        const back = kt.gemm_mxfp8.f32_to_ue8m0(v);
        try testing.expectEqual(@as(u8, @intCast(b)), back);
        // No mantissa: v * 1.5 must land on the next power of two, never
        // strictly between two of them.
        try testing.expect(std.math.isFinite(v * 1.5));
    }
    // f32 -> ue8m0 rounds to nearest power of two.
    try testing.expectEqual(@as(u8, 127), kt.gemm_mxfp8.f32_to_ue8m0(1.0));
    try testing.expectEqual(@as(u8, 127), kt.gemm_mxfp8.f32_to_ue8m0(1.4)); // 1.4 < 1.5 -> down
    try testing.expectEqual(@as(u8, 128), kt.gemm_mxfp8.f32_to_ue8m0(1.5)); // ties -> up
    try testing.expectEqual(@as(u8, 0), kt.gemm_mxfp8.f32_to_ue8m0(0.0));
    try testing.expectEqual(@as(u8, 0), kt.gemm_mxfp8.f32_to_ue8m0(-3.0)); // sign dropped
}

test "E4M3 LUT: subnormals, 0x7F and the finite maximum decode per spec" {
    // The decoder this replaces was pure bit reconstruction and wrong twice
    // over. Pin the three cases that were wrong.
    const dec = kt.gemm_mxfp8.fp8e4m3_to_f32;

    // exp field 0 is SUBNORMAL, not a normal with exponent -7: man * 2^-9.
    // The old decoder returned ~1.0-2.0 here, i.e. up to 1024x too large.
    try testing.expectEqual(@as(f32, 0.0), dec(0x00));
    for (1..8) |man| {
        const want: f32 = @as(f32, @floatFromInt(man)) * 0.001953125; // man * 2^-9
        try testing.expectEqual(want, dec(@intCast(man)));
    }

    // exp=15, man=6 is the largest finite value (448 = 1.75 * 2^8).
    try testing.expectEqual(@as(f32, 448.0), dec(0x7E));
    // exp=15, man=7 (0x7F) is the ONLY NaN encoding in E4M3. Decoding it as
    // 0.0 matches the reference ("Treat as 0 to avoid propagation"); the old
    // decoder returned 480.0, which is not even representable.
    try testing.expectEqual(@as(f32, 0.0), dec(0x7F));

    // Sign symmetry, and the reference's 1.0 / 0.5 anchors.
    try testing.expectEqual(@as(f32, 1.0), dec(0x38));
    try testing.expectEqual(@as(f32, -1.0), dec(0xB8));
    try testing.expectEqual(@as(f32, 0.5), dec(0x30));

    // Monotone in magnitude across every positive encoding, and every value
    // stays within [0, E4M3_MAX]. 0x7F is excluded: it is the NaN hole and
    // decodes to 0, so including it would (correctly) break monotonicity.
    var prev: f32 = -1.0;
    for (0..0x7F) |i| {
        const v = dec(@intCast(i));
        try testing.expect(v >= prev);
        try testing.expect(v <= 448.0);
        prev = v;
    }
    try testing.expectEqual(@as(f32, 448.0), prev); // 0x7E, the finite max
}

test "MXFP8 quantize/dequantize round trip through ue8m0" {
    // The scale is stored as a power of two, so the round trip is exact for
    // values already on that grid and lossy (but bounded) off it.
    var seed: u64 = 0x4D58_0F8;
    for (0..64) |_| {
        const v = rnd(&seed, 64, 8.0); // [-32, 32)
        const q = kt.gemm_mxfp8.fp8e4m3_to_f32(kt.gemm_mxfp8.f32_to_fp8e4m3(v));
        const s = kt.gemm_mxfp8.f32_to_ue8m0(2.0); // scale 2.0
        const deq = q * kt.gemm_mxfp8.ue8m0_to_f32(s);
        // E4M3 keeps ~2 decimal digits; the ue8m0 scale is exact, so the whole
        // error budget is the 3-bit mantissa.
        try testing.expect(std.math.isFinite(deq));
        try testing.expect(@abs(deq - v * 2.0) <= @abs(v * 2.0) * 0.2 + 1e-3);
    }
}

// ============================================================================
// MXFP8 AVX2 GEMM
// ============================================================================

test "MXFP8 AVX2 GEMM == the scalar oracle across shapes, including ragged k" {
    // The oracle is `gemmFullTileScalar`, the readable statement of the math.
    // Shapes chosen to hit every branch in `gemmAvx2`: m % MR (5, 1), n = 1,
    // k a multiple of GROUP_SIZE (32), k with a ragged tail (40, 100), and a
    // single group (32).
    const mxfp8 = kt.gemm_mxfp8;
    var seed: u64 = 0x4D58_0801;
    const shapes = [_][3]usize{
        .{ 1, 1, 32 },
        .{ 4, 8, 32 },
        .{ 5, 7, 64 },
        .{ 8, 32, 96 },
        .{ 1, 33, 40 }, // ragged k (32 + 8)
        .{ 5, 3, 100 }, // ragged k (96 + 4)
        .{ 3, 5, 160 },
    };
    for (shapes) |sh| {
        const m = sh[0];
        const n = sh[1];
        const k = sh[2];
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 256, 4.0));

        const kb = (k + 31) / 32;
        const bw = try testing.allocator.alloc(mxfp8.MXFP8Block, n * kb);
        defer testing.allocator.free(bw);
        for (bw) |*blk| {
            blk.scale = @intCast(120 + (seed >> 8) % 16); // ue8m0, several octaves
            for (&blk.qs) |*q| {
                seed = seed *% 6364136223846793005 +% 1442695040888963407;
                q.* = @truncate(seed >> 24);
            }
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        mxfp8.GemmKernel224MXFP8.gemmAvx2(a.ptr, k, bw.ptr, kb, c_avx.ptr, n, m, n, k);
        mxfp8.GemmKernel224MXFP8.gemmFullTileScalar(a.ptr, k, bw.ptr, kb, c_ref.ptr, n, m, n, k);

        // Both sides are f32 accumulations of the same products in a different
        // order, so agreement is limited by summation order, not by the kernel:
        // the AVX2 path keeps 8 partial lanes and reduces at the end, the
        // oracle sums sequentially. Over 96 terms with cancellation that costs
        // ~1e-5 relative. A wrong kernel or a wrong offset produces O(1)
        // divergence instead (the Q4_K routing test saw rel=4.2 that way), so
        // 1e-4 still catches every dispatch and layout error.
        for (0..m * n) |i| {
            try testing.expect(std.math.isFinite(c_avx[i]));
            const denom = @max(@abs(c_ref[i]), 1e-2);
            try testing.expect(@abs(c_avx[i] - c_ref[i]) / denom < 1e-4);
        }
    }
}

test "MXFP4 AVX2 GEMM == the scalar oracle, and the nibble order is low-first" {
    // Two things are pinned here. The oracle comparison, and — because a
    // reversed nibble order still produces plausible-looking numbers rather
    // than garbage — the ordering itself: `fp4x8` emits the low nibble at the
    // even index, matching `unpackMXFP4`. Swapping the two in `fp4x8` fails
    // this test, which is the only cheap way to notice.
    const mxfp4 = kt.gemm_mxfp4;
    var seed: u64 = 0x4D58_0402;
    const shapes = [_][3]usize{
        .{ 1, 1, 32 },
        .{ 4, 8, 32 },
        .{ 5, 7, 64 },
        .{ 8, 32, 128 },
        .{ 1, 33, 40 }, // ragged k (32 + 8)
        .{ 5, 3, 100 }, // ragged k (96 + 4)
        .{ 3, 5, 224 },
    };
    for (shapes) |sh| {
        const m = sh[0];
        const n = sh[1];
        const k = sh[2];
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 256, 4.0));

        const kb = (k + 31) / 32;
        const bw = try testing.allocator.alloc(mxfp4.MXFP4Block, n * kb);
        defer testing.allocator.free(bw);
        for (bw) |*blk| {
            blk.scale = @intCast(118 + (seed >> 8) % 20); // ue8m0, several octaves
            for (&blk.qs) |*byte| {
                seed = seed *% 6364136223846793005 +% 1442695040888963407;
                // Nibbles drawn from the WHOLE E2M1 range, not just 0x0..0xF in
                // order, so a swapped pair cannot average out.
                byte.* = @truncate((seed >> 20) ^ (seed >> 4));
            }
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        mxfp4.GemmKernel224MXFP4.gemmAvx2(a.ptr, k, bw.ptr, kb, c_avx.ptr, n, m, n, k);
        mxfp4.GemmKernel224MXFP4.gemmFullTileScalar(a.ptr, k, bw.ptr, kb, c_ref.ptr, n, m, n, k);

        // f32 summation-order difference over up to 224 terms with
        // cancellation; a wrong nibble order gives O(1) divergence.
        for (0..m * n) |i| {
            try testing.expect(std.math.isFinite(c_avx[i]));
            const denom = @max(@abs(c_ref[i]), 1e-2);
            try testing.expect(@abs(c_avx[i] - c_ref[i]) / denom < 1e-4);
        }
    }
}

test "MXFP4 nibble order: low nibble is the even weight" {
    // Direct, independent of the GEMM: one block whose bytes carry distinct
    // nibble pairs, checked against the spec-level expectation rather than
    // against another kernel. Low nibble = i, high nibble = (i + 5) & 0xF, so
    // the two differ for every i (a swap is visible) and both stay inside the
    // 4-bit range for all 16 bytes (i + 5 & 0xF wraps rather than overflowing).
    const mxfp4 = kt.gemm_mxfp4;
    var blk: mxfp4.MXFP4Block = undefined;
    blk.scale = 127; // 1.0
    for (&blk.qs) |*b| b.* = 0;
    for (&blk.qs, 0..) |*b, i| {
        b.* = @as(u8, @intCast(i)) | (@as(u8, @intCast((i + 5) & 0xF)) << 4);
    }
    var w: [32]f32 = undefined;
    mxfp4.unpackMXFP4(&blk, &w);
    for (0..16) |i| {
        const lo = mxfp4.fp4e2m1_to_f32(@truncate(@as(u32, @intCast(i))));
        const hi = mxfp4.fp4e2m1_to_f32(@truncate(@as(u32, @intCast((i + 5) & 0xF))));
        try testing.expect(hi != lo); // precondition: the pair is asymmetric
        try testing.expectEqual(lo, w[i * 2]); // even index <- low nibble
        try testing.expectEqual(hi, w[i * 2 + 1]); // odd index <- high nibble
    }
}

test "Q8_0 AVX2 GEMM == the scalar oracle across shapes" {
    // Q8_0 is the GGUF default weight format and the LlamaMoe activation
    // format, so this is the hottest quantized GEMM in the tree. Shapes cover
    // m % MR (1, 5), n = 1, and several block counts. k is always a multiple
    // of QK8_0 = 32 because a Q8_0 row cannot express a ragged tail.
    const q8 = kt.gemm_q8_0;
    var seed: u64 = 0x0800_1234;
    const shapes = [_][3]usize{
        .{ 1, 1, 32 },
        .{ 4, 8, 32 },
        .{ 5, 7, 64 },
        .{ 8, 32, 256 },
        .{ 1, 33, 512 },
        .{ 5, 3, 1024 },
        .{ 3, 5, 7168 },
    };
    for (shapes) |sh| {
        const m = sh[0];
        const n = sh[1];
        const k = sh[2];
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 256, 4.0));

        const kb = k / q8.QK8_0;
        const bw = try testing.allocator.alloc(q8.BlockQ8_0, n * kb);
        defer testing.allocator.free(bw);
        for (bw) |*blk| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            blk.d = q8.f32_to_f16(0.01 + @as(f32, @floatFromInt((seed >> 32) % 64)) / 64.0);
            for (&blk.qs) |*q| {
                seed = seed *% 6364136223846793005 +% 1442695040888963407;
                // Full signed range including the -128 that has no positive
                // counterpart, so a sign-extension slip cannot hide.
                q.* = @bitCast(@as(u8, @truncate(seed >> 24)));
            }
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        // Poison both, including the padding a plain `= 0` memset would leave.
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        q8.gemmAvx2(a.ptr, k, bw.ptr, kb, c_avx.ptr, n, m, n, k);
        q8.gemmQ8_0Scalar(a.ptr, k, bw.ptr, kb, c_ref.ptr, n, m, n, k);

        // Both paths accumulate the same f32 products in a different order (8 partial
        // lanes + one reduce, versus sequential), so they differ by summation
        // error. `rnd` here returns values up to ~512 and the dots run to
        // thousands, so a sum that partly cancels shows up amplified:
        // measured worst is 1.6e-4 relative (an absolute 0.06 on a value of
        // 381, over 256 f32 terms). A wrong scale, a missed sign extension or
        // a bad offset gives O(1) instead - the Q4_K routing test saw rel=4.2
        // that way, and the MXFP4 encoder bug was a clean factor of 2 - so
        // 1e-3 is a loose bound that still fails loudly for every real defect.
        for (0..m * n) |i| {
            try testing.expect(std.math.isFinite(c_avx[i]));
            const denom = @max(@abs(c_ref[i]), 1e-2);
            try testing.expect(@abs(c_avx[i] - c_ref[i]) / denom < 1e-3);
        }
    }
}

test "Q8_0 AVX2 GEMM covers every output element (no uninitialised holes)" {
    // The scalar version zeroes `c` and then accumulates into it; the vector
    // version assigns directly. That is only equivalent if every element of
    // [0,m) x [0,n) really is written, so m values either side of MR are
    // checked for NaN with the output pre-poisoned.
    const q8 = kt.gemm_q8_0;
    for ([_]usize{ 1, 2, 3, 4, 5, 7, 8, 9 }) |m| {
        const n: usize = 3;
        const k: usize = 64;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(0.5);
        const bw = try testing.allocator.alloc(q8.BlockQ8_0, n * (k / q8.QK8_0));
        defer testing.allocator.free(bw);
        for (bw) |*blk| {
            blk.d = q8.f32_to_f16(1.0);
            for (&blk.qs) |*q| q.* = 3; // non-zero, so a written element differs from NaN
        }
        const c = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c);
        @memset(c, std.math.nan(f32));
        q8.gemmAvx2(a.ptr, k, bw.ptr, k / q8.QK8_0, c.ptr, n, m, n, k);
        for (c) |v| try testing.expect(std.math.isFinite(v));
    }
}
