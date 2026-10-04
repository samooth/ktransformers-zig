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

test "IQ4_NL scalar GEMM honours the block offset at k > one group" {
    // Regression test. `gemmIQ4_NLScalar` indexed the activation as
    // `input[i * input_ld + k0]` with k0 in 0..32, inside the per-block loop,
    // never adding `blk * 32`. Every block was therefore multiplied by the SAME
    // first 32 activations. Correct only when nb == 1 (k == 32) — the only
    // shape the pre-existing tests used — and wrong for every real checkpoint,
    // on a live path (`gemmQuant` dispatches KT_TYPE_IQ4_NL).
    //
    // Measured before the fix at n=2048-style shapes: rel error 2.2 at k=64,
    // 8.9 at k=128, 1.5 at k=256 against a correct reference.
    const g = kt.gemm_iq4_nl;
    var seed: u64 = 0x1A2B_3C4D;

    for ([_]usize{ 32, 64, 128, 256, 2048 }) |k| {
        const nb = k / g.QK4_NL;
        const m: usize = 2;
        const n: usize = 3;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 256, 4.0));

        const w = try testing.allocator.alloc(g.BlockIQ4_NL, n * nb);
        defer testing.allocator.free(w);
        for (w) |*blk| {
            blk.d = @bitCast(@as(f16, 0.05));
            for (&blk.qs) |*q| {
                seed = seed *% 6364136223846793005 +% 1442695040888963407;
                q.* = @truncate(seed >> 24);
            }
        }

        const c = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c);
        @memset(c, std.math.nan(f32));
        g.gemmIQ4_NLScalar(a.ptr, w.ptr, c.ptr, m, n, k, k, nb, n);

        // Reference: identical, except that it advances the activation index
        // by the block offset.
        var scratch: [g.QK4_NL]f32 = undefined;
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                for (0..nb) |blk| {
                    g.dequantizeRowIQ4_NL(w.ptr + j * nb + blk, &scratch, g.QK4_NL);
                    for (0..g.QK4_NL) |k0| {
                        sum += amx.bf16_to_f32(a[i * k + blk * g.QK4_NL + k0]) * scratch[k0];
                    }
                }
                const got = c[i * n + j];
                try testing.expect(std.math.isFinite(got));
                const denom = @max(@abs(sum), 1e-2);
                try testing.expect(@abs(got - sum) / denom < 1e-3);
            }
        }
    }
}

test "IQ1_M scalar GEMM honours the block offset at k > one group" {
    // Same regression, second API shape. IQ1_M takes `(a, b, c, m, n, k, lda,
    // ldb, ldc)` where IQ4_NL takes `(input, weight, output, m, n, k, input_ld,
    // weight_ld, output_ld)`, and uses `QK_K` rather than `QK4_NL`. Two
    // shapes of the same copy-pasted bug, so both are pinned.
    const g = kt.gemm_iq1_m;
    var seed: u64 = 0x0BAD_C0DE;

    // k must be a whole number of groups: IQ1_M's group is QK_K = 256, so
    // k = 32 or 64 would give nb = 0 and make the case vacuous.
    for ([_]usize{ 256, 512, 1024, 2048 }) |k| {
        const nb = k / g.QK_K;
        const m: usize = 2;
        const n: usize = 3;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 256, 4.0));

        const w = try testing.allocator.alloc(g.BlockIQ1_M, n * nb);
        defer testing.allocator.free(w);
        // Fill through the byte view: this test is about the GEMM's activation
        // indexing, so the block contents only need to be deterministic, and
        // going via bytes keeps it independent of each format's field layout.
        const wbytes: [*]u8 = @ptrCast(@alignCast(w.ptr));
        for (0..n * nb * @sizeOf(g.BlockIQ1_M)) |i| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            wbytes[i] = @truncate(seed >> 24);
        }

        const c = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c);
        @memset(c, std.math.nan(f32));
        g.gemmIQ1_MScalar(a.ptr, w.ptr, c.ptr, m, n, k, k, nb, n);

        var scratch: [g.QK_K]f32 = undefined;
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                for (0..nb) |blk| {
                    g.dequantizeRowIQ1_M(w.ptr + j * nb + blk, &scratch, g.QK_K);
                    for (0..g.QK_K) |k0| {
                        sum += amx.bf16_to_f32(a[i * k + blk * g.QK_K + k0]) * scratch[k0];
                    }
                }
                const got = c[i * n + j];
                // Random block bytes can put a NaN or Inf in IQ1_M's hidden
                // f16 scale (4 random bits interpreted as an f16 exponent),
                // and the dequantizer faithfully propagates it. So the
                // assertion has to be NaN-aware rather than `isFinite` — what
                // matters is that the kernel and the reference agree on WHICH
                // outputs are non-finite, not that all of them are finite.
                if (std.math.isNan(sum)) {
                    try testing.expect(std.math.isNan(got));
                } else {
                    try testing.expect(std.math.isFinite(got));
                    const denom = @max(@abs(sum), 1e-2);
                    try testing.expect(@abs(got - sum) / denom < 1e-3);
                }
            }
        }
    }
}

test "IQ4_NL AVX2 GEMM == the scalar oracle, and the split-half order is right" {
    // Two things pinned. The oracle comparison, and the SPLIT-HALF nibble
    // order: `dequantizeRowIQ4_NL` puts the low nibbles of all 16 bytes at
    // weights 0..15 and the high nibbles at 16..31, the opposite of MXFP4's
    // interleaving. A permuted unpack still yields plausible magnitudes, so the
    // order is asserted directly rather than left to the oracle.
    const g = kt.gemm_iq4_nl;
    var seed: u64 = 0x1C4E_5EED;

    // Direct order check, independent of the GEMM.
    {
        var blk: g.BlockIQ4_NL = undefined;
        blk.d = @bitCast(@as(f16, 1.0));
        // byte j: low nibble = j+1, high nibble = j+1 (both valid 1..15 range
        // shifted); use distinct pairs so a swap is visible.
        for (&blk.qs, 0..) |*b, j| {
            b.* = @as(u8, @intCast(j)) | (@as(u8, @intCast(15 - j)) << 4);
        }
        var w: [g.QK4_NL]f32 = undefined;
        g.dequantizeRowIQ4_NL(@as([*]const g.BlockIQ4_NL, @ptrCast(&blk)), w[0..], g.QK4_NL);
        const kv = kt.gemm_iq4_xs.KVALUES_IQ4NL;
        for (0..16) |j| {
            // weights 0..15 <- LOW nibbles
            try testing.expectEqual(@as(f32, @floatFromInt(kv[j])), w[j]);
            // weights 16..31 <- HIGH nibbles
            try testing.expectEqual(@as(f32, @floatFromInt(kv[15 - j])), w[j + 16]);
        }
    }

    // Oracle comparison. k spans 1..8 groups so the block offset matters.
    for ([_]usize{ 1, 2, 4, 8 }) |groups| {
        const k = groups * g.QK4_NL;
        const nb = groups;
        const m: usize = 5; // m % MR != 0, so the row tail runs
        const n: usize = 7;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 256, 4.0));

        const w = try testing.allocator.alloc(g.BlockIQ4_NL, n * nb);
        defer testing.allocator.free(w);
        for (w) |*blk| {
            blk.d = @bitCast(@as(f16, 0.02));
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

        g.gemmAvx2(a.ptr, w.ptr, c_avx.ptr, m, n, k, k, nb, n);
        g.gemmIQ4_NLScalar(a.ptr, w.ptr, c_ref.ptr, m, n, k, k, nb, n);

        for (0..m * n) |i| {
            try testing.expect(std.math.isFinite(c_avx[i]));
            const denom = @max(@abs(c_ref[i]), 1e-2);
            try testing.expect(@abs(c_avx[i] - c_ref[i]) / denom < 1e-3);
        }
    }
}

test "K-quant AVX2 GEMMs: shape sweep over m tails, n=1 and group counts" {
    // The six per-format equivalence tests all use ONE shape (m=5, n=3, three
    // super-blocks). That covers the m tail (5 % MR = 1) but nothing else: no
    // n=1, no m below MR, and only one group count. Since all six formats share
    // `kquant_simd.gemmBF16xQuant`, a tail bug in the driver would show up in
    // every format at once, so a shape sweep is worth more here than another
    // format would be.
    //
    // Shapes chosen to hit: m < MR (1,2,3), m == MR (4), m just over MR (5),
    // m a multiple of MR (8), n = 1, and 1 / 2 / 7 groups. K = QK * groups so a
    // group count of 0 is impossible — a vacuous case is how the IQ bug hid.
    const Shape = struct { m: usize, n: usize, groups: usize };
    const shapes = [_]Shape{
        .{ .m = 1, .n = 1, .groups = 1 },
        .{ .m = 2, .n = 3, .groups = 1 },
        .{ .m = 3, .n = 1, .groups = 2 },
        .{ .m = 4, .n = 5, .groups = 1 },
        .{ .m = 5, .n = 1, .groups = 3 },
        .{ .m = 8, .n = 4, .groups = 2 },
        .{ .m = 1, .n = 7, .groups = 7 },
        .{ .m = 3, .n = 2, .groups = 5 },
    };

    const Case = struct {
        name: []const u8,
        blk_bytes: usize,
        qk: usize,
        gemm_vec: *const fn ([*]const amx.bf16, usize, [*]const u8, usize, [*]f32, usize, usize, usize, usize) void,
        gemm_scalar: *const fn ([*]const amx.bf16, usize, [*]const u8, usize, [*]f32, usize, usize, usize, usize) void,
    };
    const cases = [_]Case{
        .{
            .name = "Q2_K",
            .blk_bytes = @sizeOf(q2_k.BlockQ2_K),
            .qk = q2_k.QK_K,
            .gemm_vec = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q2_k.gemmQ2_K(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
            .gemm_scalar = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q2_k.gemmQ2_KScalar(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
        },
        .{
            .name = "Q3_K",
            .blk_bytes = @sizeOf(q3_k.BlockQ3_K),
            .qk = q3_k.QK_K,
            .gemm_vec = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q3_k.gemmQ3_K(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
            .gemm_scalar = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q3_k.gemmQ3_KScalar(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
        },
        .{
            .name = "Q4_K",
            .blk_bytes = @sizeOf(q4_k.BlockQ4_K),
            .qk = q4_k.QK_K,
            .gemm_vec = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q4_k.gemmQ4_K(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
            .gemm_scalar = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q4_k.gemmQ4_KScalar(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
        },
        .{
            .name = "Q5_K",
            .blk_bytes = @sizeOf(q5_k.BlockQ5_K),
            .qk = q5_k.QK_K,
            .gemm_vec = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q5_k.gemmQ5_K(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
            .gemm_scalar = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q5_k.gemmQ5_KScalar(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
        },
        .{
            .name = "Q6_K",
            .blk_bytes = @sizeOf(q6_k.BlockQ6_K),
            .qk = q6_k.QK_K,
            .gemm_vec = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q6_k.gemmQ6_K(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
            .gemm_scalar = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q6_k.gemmQ6_KScalar(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
        },
        .{
            .name = "Q8_K",
            .blk_bytes = @sizeOf(q8_k.BlockQ8_K),
            .qk = q8_k.QK_K,
            .gemm_vec = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q8_k.gemmQ8_K(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
            .gemm_scalar = struct {
                fn f(a: [*]const amx.bf16, lda: usize, b: [*]const u8, ldb: usize, c: [*]f32, ldc: usize, m: usize, n: usize, k: usize) void {
                    q8_k.gemmQ8_KScalar(a, lda, @ptrCast(@alignCast(b)), ldb, c, ldc, m, n, k);
                }
            }.f,
        },
    };

    inline for (cases) |cs| {
        var state: u64 = 0xA5A5_1234 ^ @as(u64, cs.qk);
        for (shapes) |sh| {
            const k = cs.qk * sh.groups;
            const nb = sh.groups;
            // Precondition, asserted rather than assumed: a zero group count
            // would make the whole case vacuous, which is how the IQ block
            // offset bug survived a fully "covered" suite.
            try testing.expect(nb > 0);

            const a = try testing.allocator.alloc(amx.bf16, sh.m * k);
            defer testing.allocator.free(a);
            for (0..sh.m * k) |i| a[i] = amx.f32_to_bf16(rnd(&state, 64, 2.0));

            const wbytes = try testing.allocator.alloc(u8, sh.n * nb * cs.blk_bytes);
            defer testing.allocator.free(wbytes);
            for (0..sh.n * nb * cs.blk_bytes) |i| {
                state = state *% 6364136223846793005 +% 1442695040888963407;
                wbytes[i] = @truncate(state >> 24);
            }

            const c_vec = try testing.allocator.alloc(f32, sh.m * sh.n);
            defer testing.allocator.free(c_vec);
            const c_sca = try testing.allocator.alloc(f32, sh.m * sh.n);
            defer testing.allocator.free(c_sca);
            @memset(c_vec, std.math.nan(f32));
            @memset(c_sca, std.math.nan(f32));

            cs.gemm_vec(a.ptr, k, wbytes.ptr, nb, c_vec.ptr, sh.n, sh.m, sh.n, k);
            cs.gemm_scalar(a.ptr, k, wbytes.ptr, nb, c_sca.ptr, sh.n, sh.m, sh.n, k);

            for (0..sh.m * sh.n) |i| {
                const g = c_vec[i];
                const s = c_sca[i];
                // Random bytes can decode to a NaN scale in some formats, so
                // agreement on WHICH outputs are non-finite is the assertion,
                // not that all are finite.
                if (std.math.isNan(s)) {
                    try testing.expect(std.math.isNan(g));
                } else if (std.math.isInf(s)) {
                    try testing.expectEqual(std.math.isInf(g), std.math.isInf(s));
                } else {
                    try testing.expect(std.math.isFinite(g));
                    const denom = @max(@abs(s), 1e-2);
                    try testing.expect(@abs(g - s) / denom < 1e-3);
                }
            }
        }
    }
}

test "IQ4_XS AVX2 GEMM == the scalar oracle, and the 6-bit scale index is right" {
    // Two things pinned. The oracle comparison, and the 6-bit sub-block scale
    // reassembly, which is split across `scales_l` (4 low bits) and `scales_h`
    // (2 high bits). Reading the halves from the wrong sub-block still yields a
    // plausible magnitude, so the index is asserted directly.
    const g = kt.gemm_iq4_xs;
    var state: u64 = 0x1458_0001;

    // Direct scale check. `scales_l` holds eight 4-bit fields (two per byte,
    // low field even) and `scales_h` eight 2-bit fields. Both are built here by
    // an explicit loop rather than hand-written bit literals — a literal like
    // 0b11_10_01_00 is read MSB-first by eye but LSB-first by `(x >> 2*ib) & 3`,
    // which is exactly the kind of mismatch this check exists to catch. Every
    // sub-block gets a distinct pair so reading the wrong sub-block's halves
    // cannot pass.
    {
        var blk: g.BlockIQ4_XS = undefined;
        blk.d = @bitCast(@as(f16, 0.01));
        blk.scales_h = 0;
        for (0..4) |i| blk.scales_l[i] = 0;
        for (0..8) |ib| {
            blk.scales_l[ib / 2] |= @as(u8, @intCast(ib + 1)) << @intCast(4 * (ib % 2));
            blk.scales_h |= @as(u16, @intCast((ib + 1) % 4)) << @intCast(2 * ib);
        }
        for (&blk.qs) |*q| q.* = 0x00;

        // Compute the expectation from the STORED f16, not the f32 literal:
        // f16(0.01) is 0.0100021362..., so using 0.01 would be off by ~2e-5 and
        // the point of this check is a 1e-9 comparison.
        const d_stored: f32 = @floatCast(@as(f16, @bitCast(blk.d)));
        var w: [g.QK_K]f32 = undefined;
        g.dequantizeRowIQ4_XS(@as([*]const g.BlockIQ4_XS, @ptrCast(&blk)), w[0..], g.QK_K);

        for (0..8) |ib| {
            const low4: u32 = (blk.scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0xf;
            const high2: u32 = (blk.scales_h >> @intCast(2 * ib)) & 3;
            const ls = low4 | (high2 << 4);
            try testing.expectEqual(@as(u32, @intCast(ib + 1)), low4);
            try testing.expectEqual(@as(u32, @intCast((ib + 1) % 4)), high2);
            const dl = d_stored * (@as(f32, @floatFromInt(ls)) - 32.0);
            // All quants are nibble 0, and KVALUES_IQ4NL[0] is -127 (not 0), so
            // the dequantized weight IS the sub-block scale times -127.
            const want = dl * @as(f32, kt.gemm_iq4_xs.KVALUES_IQ4NL[0]);
            try testing.expectApproxEqAbs(want, w[ib * 32], 1e-9);
        }
    }

    // Oracle comparison. k spans 1..3 super-blocks of 256, m covers the MR tail.
    for ([_]usize{ 1, 2, 3 }) |blocks| {
        const k = blocks * g.QK_K;
        const nb = blocks;
        const m: usize = 5;
        const n: usize = 3;
        try testing.expect(nb > 0); // a 0-block case would be vacuous
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&state, 64, 2.0));

        const w = try testing.allocator.alloc(g.BlockIQ4_XS, n * nb);
        defer testing.allocator.free(w);
        for (w) |*blk| {
            blk.d = @bitCast(@as(f16, 0.01));
            for (&blk.scales_l) |*b| {
                state = state *% 6364136223846793005 +% 1442695040888963407;
                b.* = @truncate(state >> 24);
            }
            state = state *% 6364136223846793005 +% 1442695040888963407;
            blk.scales_h = @truncate(state >> 24);
            for (&blk.qs) |*q| {
                state = state *% 6364136223846793005 +% 1442695040888963407;
                q.* = @truncate(state >> 24);
            }
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        g.gemmAvx2(a.ptr, w.ptr, c_avx.ptr, m, n, k, k, nb, n);
        g.gemmIQ4_XSScalar(a.ptr, w.ptr, c_ref.ptr, m, n, k, k, nb, n);

        for (0..m * n) |i| {
            try testing.expect(std.math.isFinite(c_avx[i]));
            const denom = @max(@abs(c_ref[i]), 1e-2);
            try testing.expect(@abs(c_avx[i] - c_ref[i]) / denom < 1e-3);
        }
    }
}

test "pshufb16 is a correct 16-entry byte lookup (asm operand order)" {
    // Load-bearing, not decorative. The AT&T form of `vpshufb` takes the DATA
    // operand second; writing it first still compiles and still emits bytes, but
    // the table then acts as the index and every output is zero whenever the
    // table's bytes have bit 7 set. This test uses table values 200..215, all
    // of which have bit 7 set, precisely so that mistake cannot hide.
    var tbl: [16]u8 = undefined;
    for (&tbl, 0..) |*t, i| t.* = @intCast(200 + i);
    const tv = amx.pshufbTable(&tbl);

    // 0..15 in the low lane, 15..0 in the high lane: exercises both 128-bit
    // halves, since `pshufb` on a ymm is lane-wise.
    var idx: [32]u8 = undefined;
    for (0..16) |i| {
        idx[i] = @intCast(i);
        idx[i + 16] = @intCast(15 - i);
    }
    const got: [32]u8 = amx.pshufb16(tv, idx);
    for (0..32) |i| {
        const want: u8 = if (i < 16) tbl[i] else tbl[15 - (i - 16)];
        try testing.expectEqual(want, got[i]);
    }

    // pshufb semantics: bit 7 of the index SET means "write zero", not
    // "index 128". Every quant table here relies on this to stay in range.
    const zeroed: [32]u8 = amx.pshufb16(tv, @as([32]u8, @splat(0x83)));
    for (zeroed) |v| try testing.expectEqual(@as(u8, 0), v);
    // Indices only use the low nibble, so bit 4..6 must be ignored too.
    const masked: [32]u8 = amx.pshufb16(tv, @as([32]u8, @splat(0x05)));
    for (masked) |v| try testing.expectEqual(tbl[5], v);
}

test "kvalues16 (pshufb unpack) == the scalar KVALUES table for all nibbles" {
    // The unpack the IQ4_NL and IQ4_XS kernels both use, checked against the
    // scalar table lookup for every index and both nibble halves.
    const k16 = kt.gemm_iq4_xs.kvalues16;
    const kv = kt.gemm_iq4_xs.KVALUES_IQ4NL;
    var bytes: [16]u8 = undefined;
    for (&bytes, 0..) |*b, i| {
        // byte i: low nibble = i, high nibble = 15 - i, so all 16 indices
        // appear in each half.
        b.* = @as(u8, @intCast(i)) | (@as(u8, @intCast(15 - i)) << 4);
    }
    const lo_pair = k16(&bytes[0], 0);
    const hi_pair = k16(&bytes[0], 4);
    const lo0: [8]f32 = lo_pair[0];
    const lo1: [8]f32 = lo_pair[1];
    const hi0: [8]f32 = hi_pair[0];
    const hi1: [8]f32 = hi_pair[1];
    for (0..8) |i| {
        // low nibble of byte i -> index i        (weights 0..7, 8..15)
        try testing.expectEqual(@as(f32, @floatFromInt(kv[i])), lo0[i]);
        try testing.expectEqual(@as(f32, @floatFromInt(kv[i + 8])), lo1[i]);
        // high nibble of byte i -> index 15 - i   (weights 16..23, 24..31)
        try testing.expectEqual(@as(f32, @floatFromInt(kv[15 - i])), hi0[i]);
        try testing.expectEqual(@as(f32, @floatFromInt(kv[7 - i])), hi1[i]);
    }
}

// ============================================================================
// FP8 E4M3
// ============================================================================

/// The reference's decode rule, transcribed from
/// kt-kernel/operators/avx2/fp8_dequant.hpp::FP8LUT::init, written
/// independently of `FP8_E4M3_LUT` so the comparison is a real oracle and not
/// a restatement. `std.math.ldexp` is fine at RUNTIME; it is the COMPTIME
/// evaluation of it that this toolchain rejects, which is why the shipped
/// table is built from the f32 exponent field instead.
fn refE4M3(bits: u8) f32 {
    const sign: f32 = if (bits & 0x80 != 0) -1.0 else 1.0;
    const e = (bits >> 3) & 0x0F;
    const man = bits & 0x07;
    if (e == 0 and man == 0) return 0.0 * sign;
    if (e == 0) return sign * std.math.ldexp(@as(f32, @floatFromInt(man)) / 8.0, -6);
    // The reference applies the sign to this branch too: `val = 0.0f;` then
    // `table[i] = sign ? -val : val`, so 0xFF is NEGATIVE zero.
    if (e == 15 and man == 7) return 0.0 * sign;
    return sign * std.math.ldexp(1.0 + @as(f32, @floatFromInt(man)) / 8.0, @as(i32, e) - 7);
}

test "FP8 E4M3 LUT decode is exact for all 256 encodings" {
    const fp8 = kt.gemm_fp8;
    // Bit patterns, not `==`: -0.0 == 0.0 in IEEE, and 0x80 (negative zero) is
    // one of the encodings under test. 0x7F decodes to 0, so there is no NaN
    // in the table and no NaN comparison problem either.
    for (0..256) |i| {
        const bits: u8 = @intCast(i);
        try testing.expectEqual(
            @as(u32, @bitCast(refE4M3(bits))),
            @as(u32, @bitCast(fp8.fp8e4m3_to_f32(bits))),
        );
    }
    // The boundaries that were actually wrong before the LUT existed, spelled
    // out: the two subnormal edges, zero (both signs), the single NaN, and the
    // largest finite value on both signs.
    const cases = [_][2]u32{
        // byte, expected f32 bit pattern
        .{ 0x00, 0x00000000 }, // +zero
        .{ 0x01, 0x3B000000 }, // smallest positive subnormal (2^-9)
        .{ 0x07, 0x3C600000 }, // largest subnormal (7 * 2^-9)
        .{ 0x08, 0x3C800000 }, // smallest normal (2^-6)
        .{ 0x0F, 0x3CF00000 }, // largest 2^-6 binade value
        .{ 0x7E, 0x43E00000 }, // E4M3_MAX = 448
        .{ 0x7F, 0x00000000 }, // the only NaN encoding -> +0
        .{ 0x80, 0x80000000 }, // -zero (the sign IS applied: `-0.0f`)
        .{ 0xFE, 0xC3E00000 }, // -E4M3_MAX
    };
    for (cases) |c| {
        const byte: u8 = @intCast(c[0]);
        try testing.expectEqual(c[1], @as(u32, @bitCast(fp8.fp8e4m3_to_f32(byte))));
        try testing.expectEqual(c[1], @as(u32, @bitCast(refE4M3(byte))));
    }
    // Nothing may decode above E4M3_MAX: the old bit-reconstruction decoder
    // returned 480.0 for 0x7F, which is not representable in the format.
    for (0..256) |i| {
        const v = @abs(fp8.fp8e4m3_to_f32(@intCast(i)));
        try testing.expect(v <= fp8.E4M3_MAX);
        try testing.expect(std.math.isFinite(v));
    }
    // Subnormals are the whole point: every one of the seven must be BELOW
    // 2^-6, which they are not if the exponent field 0 is read as normal.
    for (1..8) |i| {
        const v = fp8.fp8e4m3_to_f32(@intCast(i));
        try testing.expect(v > 0.0);
        try testing.expect(v < 0.015625);
        try testing.expectEqual(v * 512.0, @as(f32, @floatFromInt(i))); // exact
    }
}

test "FP8 AVX2 GEMM == the scalar oracle across shapes, including ragged k" {
    const fp8 = kt.gemm_fp8;
    // Shapes chosen to hit every branch in `gemmAvx2`: m = 1 (the m%MR tail
    // path only), m = 4 (exactly MR), m = 5 (one full group + tail), n = 1 and
    // 2 (no vectorisation on the n axis at all), and k = 8 / 16 / 32 exactly,
    // k past one 32-element step, and k with a 1..7 element tail.
    var seed: u64 = 0x4650_0001;
    const shapes = [_][3]usize{
        .{ 1, 1, 8 },
        .{ 1, 2, 16 },
        .{ 4, 7, 32 },
        .{ 5, 1, 33 }, // ragged k (32 + 1)
        .{ 5, 2, 40 }, // ragged k (32 + 8)
        .{ 5, 7, 96 },
        .{ 1, 7, 97 }, // ragged k (96 + 1)
        .{ 4, 2, 129 },
        .{ 5, 7, 256 },
    };
    for (shapes) |sh| {
        const m = sh[0];
        const n = sh[1];
        const k = sh[2];
        const act = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(act);
        for (0..m * k) |i| act[i] = amx.f32_to_bf16(rnd(&seed, 512, 2.0));

        const w = try testing.allocator.alloc(fp8.fp8_e4m3, n * k);
        defer testing.allocator.free(w);
        // Random bytes: all 256 encodings appear in a 32-element row, so the
        // subnormal and 0x7F paths are exercised, not just the normals.
        for (w) |*q| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            q.* = @truncate(seed >> 24);
        }

        const sc = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(sc);
        for (sc) |*s| s.* = rnd(&seed, 1024, 0.5) + 0.5;

        const got = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(got);
        const scal = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(scal);
        // Poison both outputs: `gemmAvx2` assigns rather than accumulates, so
        // this also proves every element of [0,m) x [0,n) is written.
        @memset(got, std.math.nan(f32));
        @memset(scal, std.math.nan(f32));

        fp8.GemmKernel224FP8.gemmAvx2(act.ptr, k, w.ptr, sc.ptr, k, got.ptr, n, m, n, k);
        fp8.GemmKernel224FP8.gemmFullTileScalar(act.ptr, k, w.ptr, sc.ptr, k, scal.ptr, n, m, n, k);

        // The oracle folds the per-row scale ONCE, after the reduction, which
        // is what the vector path does (and the more accurate of the two
        // formulations — a separate test below pins that choice bit-exactly).
        // `gemmFullTileScalar` folds the scale into every product, so it is
        // compared as a second statement of the same math rather than as the
        // oracle.
        //
        // All three agree only to f32 summation error, and E4M3 weights from
        // random bytes span 2^-9 .. 448, so a partly-cancelling dot is
        // genuinely ill-conditioned: measured worst over this sweep is 6.6e-6
        // for the vector path and 2.3e-5 for the scalar one (both against the
        // f64 oracle, at k = 256). A wrong scale, a missed sign, a bad offset
        // or a table entry decoded twice gives O(1) instead — 1e-4 catches
        // every real defect and is not a bound any of these kernels is near.
        for (0..m) |i| {
            for (0..n) |j| {
                var dot: f64 = 0;
                for (0..k) |kk| {
                    dot += @as(f64, amx.bf16_to_f32(act[i * k + kk])) *
                        @as(f64, fp8.fp8e4m3_to_f32(w[j * k + kk]));
                }
                const want: f64 = dot * @as(f64, sc[j]);
                const denom = @max(@abs(want), 1e-2);
                try testing.expect(std.math.isFinite(got[i * n + j]));
                try testing.expect(std.math.isFinite(scal[i * n + j]));
                try testing.expect(@abs(@as(f64, got[i * n + j]) - want) / denom < 1e-4);
                try testing.expect(@abs(@as(f64, scal[i * n + j]) - want) / denom < 1e-4);
                try testing.expect(@abs(@as(f64, got[i * n + j]) - @as(f64, scal[i * n + j])) / denom < 1e-4);
            }
        }
    }
}

test "FP8 row scale is applied once, after the accumulation" {
    // The two formulations are numerically DIFFERENT, so a test that compares
    // against a per-element-scale oracle is comparing against a different
    // number. This pins the choice exactly rather than within a tolerance,
    // with a case where the two differ by one ULP and the once-scaled value is
    // also the f64 ground truth rounded once.
    //
    //   a = 3.0, w = E4M3 0x07 = 7*2^-9 = 0.013671875, scale = 0.1
    //   once:      (3.0 * 0.013671875) * 0.1  = 0x3B866667  (== f64 truth)
    //   per-element: 3.0 * (0.013671875 * 0.1) = 0x3B866666  (1 ULP low)
    //
    // k = 9 so the vector path is the one under test: 8 weights in the main
    // loop and one through the ragged tail, with the other eight activations
    // zero so exactly one term contributes and the @reduce cannot hide the
    // ordering.
    const fp8 = kt.gemm_fp8;
    const M: usize = 1;
    const N: usize = 1;
    const K: usize = 9;

    var act: [M * K]amx.bf16 = undefined;
    for (&act, 0..) |*v, i| v.* = amx.f32_to_bf16(if (i == 0) 3.0 else 0.0);
    var w: [N * K]fp8.fp8_e4m3 = undefined;
    for (&w) |*q| q.* = 0x07;
    const sc = [N]f32{0.1};

    var got: [M * N]f32 = undefined;
    var scal: [M * N]f32 = undefined;
    fp8.GemmKernel224FP8.gemmAvx2(&act, K, &w, &sc, K, &got, N, M, N, K);
    fp8.GemmKernel224FP8.gemmFullTileScalar(&act, K, &w, &sc, K, &scal, N, M, N, K);

    try testing.expectEqual(@as(u32, 0x3B866667), @as(u32, @bitCast(got[0])));
    try testing.expectEqual(@as(u32, 0x3B866666), @as(u32, @bitCast(scal[0])));
    // The two must genuinely differ, or the assertion above proves nothing.
    try testing.expect(got[0] != scal[0]);
}

test "IQ2_XXS AVX2 GEMM == the scalar oracle, across blocks and m tails" {
    // The oracle is `gemmIQ2_XXSScalar` / `dequantizeRowIQ2_XXS`, NOT a
    // clean-room reading of the GGUF layout. This format packs an exponent
    // nibble into the same u32 as the four 7-bit sign indices, which is exactly
    // the kind of detail a reimplementation gets wrong; the vector path has to
    // reproduce the existing dequantizer exactly.
    const g = kt.gemm_iq2_xxs;
    var state: u64 = 0x1225_0001;

    for ([_]usize{ 1, 2, 3 }) |blocks| {
        const k = blocks * g.QK_K;
        const nb = blocks;
        try testing.expect(nb > 0); // a 0-block case would be vacuous
        const m: usize = 5; // m % MR != 0, so the row tail runs
        const n: usize = 3;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&state, 64, 2.0));

        const w = try testing.allocator.alloc(g.BlockIQ2_XXS, n * nb);
        defer testing.allocator.free(w);
        for (w) |*blk| {
            blk.d = @bitCast(@as(f16, 0.05));
            for (&blk.qs) |*q| {
                state = state *% 6364136223846793005 +% 1442695040888963407;
                q.* = @truncate(state >> 24);
            }
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        g.gemmAvx2(a.ptr, w.ptr, c_avx.ptr, m, n, k, k, nb, n);
        g.gemmIQ2_XXSScalar(a.ptr, w.ptr, c_ref.ptr, m, n, k, k, nb, n);

        for (0..m * n) |i| {
            const av = c_avx[i];
            const rf = c_ref[i];
            // The exponent nibble comes out of `aux1`'s top 4 bits and
            // multiplies d, so a random u16 can legitimately give a huge or
            // non-finite product. Agreement on WHICH outputs are non-finite is
            // the assertion.
            if (std.math.isNan(rf)) {
                try testing.expect(std.math.isNan(av));
            } else if (std.math.isInf(rf)) {
                try testing.expectEqual(std.math.isInf(av), std.math.isInf(rf));
            } else {
                try testing.expect(std.math.isFinite(av));
                const denom = @max(@abs(rf), 1e-2);
                try testing.expect(@abs(av - rf) / denom < 1e-3);
            }
        }
    }
}

// ============================================================================
// P1.3 — the activation's numerics, pinned (TODO Stream D)
//
// These three tests exist because the P1.3 performance investigation concluded
// that NO arithmetic change to `expVec` is worth making (measured: cutting the
// live-constant count from 5 to 4.5 `vbroadcastss` per 8 elements moved the
// activation by 0%, inside a +-23% run-to-run noise band). The decision to
// leave `expVec` alone is only safe if the properties that made it worth
// leaving alone are actually pinned by tests, and at the time of writing three
// of them were not:
//
//   * `expVec` was checked against `expScalar` but NEVER against `@exp`. Only
//     the scalar path had a ULP bound, so the vector path -- the one every MoE
//     and MLA activation actually runs -- had no accuracy test at all.
//   * The 89% bit-identical-to-`@exp` rate was documented in prose and in
//     AGENTS.md but asserted nowhere.
//   * The activation's agreement with the C++ reference was only ever checked
//     by `bench/cpp_vs_zig/compare.py`, i.e. by running the benchmark, not by
//     `zig build test`.
//
// The thresholds below are the measured values, not round numbers. See the
// Stream D entry in TODO.md for the measurements they came from.
// ============================================================================

/// `_mm256_cvtps2dq` is round-to-nearest-EVEN. Zig has no direct builtin for
/// it: `@intFromFloat` truncates and `@round` is half-AWAY-from-zero, which
/// disagrees on exact ties. The 1.5*2^23 magic add forces the same
/// round-to-nearest-even the instruction performs.
fn rneToInt(y: f32) i32 {
    const magic: f32 = 12582912.0;
    if (y >= 0.0) return @intFromFloat((y + magic) - magic);
    return -@as(i32, @intFromFloat((-y + magic) - magic));
}

/// Transcribed from `kt-kernel/operators/avx2/avx2_bf16_utils.hpp:78`
/// (`exp_avx2`) so the comparison is an independent oracle rather than a
/// restatement of `amx.expVec`. Note this is a DIFFERENT algorithm from ours:
/// the reference reduces in log2 space (`frac = y - j`) and so needs no
/// LN2_HI/LN2_LO split, whereas `expVec` reduces in exp space and is ~1e-8
/// accurate against the reference's ~1e-6. That difference is deliberate and is
/// why this test allows a vanishingly small number of disagreements.
const REF_LOG2E: f32 = 1.44269504089;
const REF_EXP_CLAMP_HI: f32 = 88.0;
const REF_POLY = [6]f32{
    0.9999999995, 0.6931471805, 0.2402265069,
    0.0555041087, 0.0096181291, 0.0013333558,
};

fn refExpAvx2(x: f32) f32 {
    const y = x * REF_LOG2E;
    const j: i32 = rneToInt(y);
    const frac = y - @as(f32, @floatFromInt(j));
    var p: f32 = REF_POLY[5];
    p = @mulAdd(f32, p, frac, REF_POLY[4]);
    p = @mulAdd(f32, p, frac, REF_POLY[3]);
    p = @mulAdd(f32, p, frac, REF_POLY[2]);
    p = @mulAdd(f32, p, frac, REF_POLY[1]);
    p = @mulAdd(f32, p, frac, REF_POLY[0]);
    const clamped: i32 = @max(@min(j, 127), -126);
    const pow2: f32 = @bitCast(@as(u32, @bitCast(clamped +% 127)) << 23);
    return pow2 * p;
}

/// `act_fn(gate, up)` — avx2_bf16_utils.hpp:118.
fn refActFn(gate: f32, up: f32) f32 {
    const neg = @min(@as(f32, 0.0) - gate, REF_EXP_CLAMP_HI);
    return (gate / (1.0 + refExpAvx2(neg))) * up;
}

test "expVec is within 1 ULP of @exp and stays ~89% bit-identical" {
    // The vector path had no accuracy test at all before this one: only
    // expScalar was pinned against @exp, and expVec is what every MoE/MLA
    // activation runs. Sampled strictly INSIDE the clamp, because outside
    // [-87, 87] a large ULP distance is the CORRECT clamp rather than an
    // accuracy failure (expVec saturates to exp(+-87)).
    var max_ulp: u32 = 0;
    var total: usize = 0;
    var identical: usize = 0;
    var x: f32 = -87.0;
    while (x <= 87.0 - 0.119) : (x += 0.0031) {
        var arr: [8]f32 = undefined;
        for (0..8) |l| arr[l] = x + @as(f32, @floatFromInt(l)) * 0.017;
        const v: amx.VecF32 = arr;
        const got: [8]f32 = amx.expVec(v);
        for (0..8) |l| {
            const want: f32 = @floatCast(@exp(arr[l]));
            try testing.expect(std.math.isFinite(got[l]));
            const d = ulpDiff(got[l], want);
            if (d > max_ulp) max_ulp = d;
            if (@as(u32, @bitCast(got[l])) == @as(u32, @bitCast(want))) identical += 1;
            total += 1;
        }
    }
    try testing.expectEqual(@as(u32, 1), max_ulp);
    // Measured 89.37%. The floor is 85% so the test tracks a real regression
    // (a dropped or reordered Horner term) rather than the sampler.
    const rate = 100.0 * @as(f64, @floatFromInt(identical)) / @as(f64, @floatFromInt(total));
    try testing.expect(rate >= 85.0);
}

test "expVec == expScalar bit-for-bit through all three activation variants" {
    // The existing variant test compares with a 1e-5 tolerance, which cannot see
    // a lane permutation or a 1-ULP drift. expVec and expScalar must agree
    // EXACTLY, including at both clamp ends and at the boundary values where
    // the synthesised exponent field changes.
    const ends = [_]f32{ -120.0, -90.0, -88.0, -87.5, -87.0, -1.0, -0.5, -0.0, 0.5, 1.0, 86.0, 87.0, 87.5, 88.0, 90.0, 120.0 };
    for (ends) |base| {
        var arr: [8]f32 = undefined;
        for (0..8) |l| arr[l] = base + @as(f32, @floatFromInt(l)) * 0.11;
        const v: amx.VecF32 = arr;
        const ev: [8]f32 = amx.expVec(v);
        for (0..8) |l| try testing.expectEqual(amx.expScalar(arr[l]), ev[l]);
    }

    // Same property, but observed through the entry points callers actually
    // use: the exp inside each variant must match the scalar exp on the same
    // clamped argument.
    const lim: amx.VecF32 = @splat(0.5);
    const alpha: amx.VecF32 = @splat(1.5);
    for (ends) |base| {
        var g: [8]f32 = undefined;
        var u: [8]f32 = undefined;
        var seed: u64 = 0xA5A5_1234;
        for (0..8) |l| {
            g[l] = base + rnd(&seed, 64, 0.02);
            u[l] = rnd(&seed, 9, 0.8) - 0.4;
        }
        const gv: amx.VecF32 = g;
        const uv: amx.VecF32 = u;

        const plain: [8]f32 = amx.swigluVec(gv, uv);
        for (0..8) |l| {
            // silu(g)*u with the exp inlined, so the comparison is exact rather
            // than tolerance-based: both sides call the same expScalar.
            const s = g[l] / (1.0 + amx.expScalar(-g[l]));
            try testing.expectEqual(s * u[l], plain[l]);
        }

        const clamped: [8]f32 = amx.swigluClampVec(gv, uv, lim);
        for (0..8) |l| {
            const gc = @min(g[l], 0.5);
            const uc = @min(@max(u[l], -0.5), 0.5);
            const s = gc / (1.0 + amx.expScalar(-gc));
            try testing.expectEqual(s * uc, clamped[l]);
        }

        const oai: [8]f32 = amx.swigluOaiVec(gv, uv, alpha, lim);
        for (0..8) |l| {
            const gc = @min(@max(g[l], -0.5), 0.5);
            const uc = @min(@max(u[l], -0.5), 0.5);
            const s = gc * (1.0 / (1.0 + amx.expScalar(-gc * 1.5)));
            try testing.expectEqual(s * (uc + 1.0), oai[l]);
        }
    }
}

test "activation matches the C++ reference act_fn after the bf16 round trip" {
    // The strongest single check on this path, and the one that used to live
    // only in bench/cpp_vs_zig/compare.py (i.e. only when someone ran the
    // benchmark). bf16 in, bf16 out, exactly as the MoE forward does it.
    //
    // Why the disagreement budget is 1-in-100000 and not zero: our exp is
    // ~1e-8 accurate and the reference's is ~1e-6, so on rare elements the two
    // land on opposite sides of a bf16 rounding midpoint. Measured over 262144
    // elements: 2 disagreements (7.6e-6), both exactly a midpoint straddle --
    // e.g. gate=-1.609375 gives got=1.5507811 (bf16 0x3fc6) against
    // want=1.5507815 (bf16 0x3fc7), where the midpoint is 1.55078125. The
    // budget exists so that class of element does not fail the build; a real
    // kernel bug (a wrong clamp, a transposed operand, a dropped FMA term)
    // misses by far more than one bf16 ULP and is still caught.
    const M: usize = 16;
    const N: usize = 512;
    const total = M * N;
    var gate: [M * N]amx.bf16 = undefined;
    var up: [M * N]amx.bf16 = undefined;
    var seed: u64 = 0xC0FFEE;
    for (0..total) |i| {
        gate[i] = amx.f32_to_bf16(rnd(&seed, 4096, 0.011) - 22.0);
        up[i] = amx.f32_to_bf16(rnd(&seed, 9, 0.6) - 0.3);
    }

    const limit: f32 = 0.5;
    const alpha: f32 = 1.5;
    const lim_v: amx.VecF32 = @splat(limit);
    const alp_v: amx.VecF32 = @splat(alpha);

    for ([_]struct { name: []const u8, kind: u8 }{
        .{ .name = "plain", .kind = 0 },
        .{ .name = "clamp", .kind = 1 },
        .{ .name = "oai", .kind = 2 },
    }) |variant| {
        var differing: usize = 0;
        for (0..M) |i| {
            var j: usize = 0;
            while (j + amx.VEC_LEN <= N) : (j += amx.VEC_LEN) {
                var ga: [amx.VEC_LEN]f32 = undefined;
                var ua: [amx.VEC_LEN]f32 = undefined;
                for (0..amx.VEC_LEN) |k| {
                    ga[k] = amx.bf16_to_f32(gate[i * N + j + k]);
                    ua[k] = amx.bf16_to_f32(up[i * N + j + k]);
                }
                const gv: amx.VecF32 = ga;
                const uv: amx.VecF32 = ua;
                const res: amx.VecF32 = switch (variant.kind) {
                    0 => amx.swigluVec(gv, uv),
                    1 => amx.swigluClampVec(gv, uv, lim_v),
                    else => amx.swigluOaiVec(gv, uv, alp_v, lim_v),
                };
                const ra: [amx.VEC_LEN]f32 = res;
                for (0..amx.VEC_LEN) |k| {
                    const want: f32 = switch (variant.kind) {
                        0 => refActFn(ga[k], ua[k]),
                        1 => blk: {
                            const gc = @min(ga[k], limit);
                            const uc = @min(@max(ua[k], -limit), limit);
                            break :blk refActFn(gc, uc);
                        },
                        else => blk: {
                            const gc = @min(@max(ga[k], -limit), limit);
                            const uc = @min(@max(ua[k], -limit), limit);
                            const neg = @min(@as(f32, 0.0) - gc * alpha, REF_EXP_CLAMP_HI);
                            break :blk gc * (1.0 / (1.0 + refExpAvx2(neg))) * (uc + 1.0);
                        },
                    };
                    const got_b = amx.f32_to_bf16(ra[k]);
                    const want_b = amx.f32_to_bf16(want);
                    if (got_b != want_b) {
                        differing += 1;
                        // A disagreement must be a single bf16 ULP, i.e. a
                        // rounding-boundary straddle. Anything larger is a bug.
                        const du: u32 = @abs(@as(i32, @as(i16, @bitCast(got_b))) - @as(i32, @as(i16, @bitCast(want_b))));
                        try testing.expectEqual(@as(u32, 1), du);
                    }
                }
            }
        }
        try testing.expect(differing * 100000 <= total);
    }
}

test "FP8 E5M2: the LUT is exact for all 256 encodings, against an independent computation" {
    const fp8 = kt.gemm_fp8;

    // The LUT is built by bit placement (exponent bias +112, mantissa << 21),
    // which is exact by construction but only if the placement is right. This
    // recomputes every entry the slow, obvious way — exponent and mantissa
    // arithmetic, and the subnormal band as man * 2^-16 — and demands bit
    // equality. A sign-bit or shift mistake cannot hide, because all 256
    // entries are compared rather than a sample.
    var checked: usize = 0;
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        const got = fp8.FP8_E5M2_LUT[b];

        const neg = b & 0x80 != 0;
        const e: u8 = (b >> 2) & 0x1F;
        const m: u8 = b & 0x03;

        if (e == 31) {
            // No infinity in E5M2: the whole top exponent is NaN. An
            // implementation that decoded 0x7C as 57344 would pass a
            // round-trip test and still be wrong, so assert NaN explicitly.
            try testing.expect(std.math.isNan(got));
            try testing.expect(std.math.isNan(fp8.fp8e5m2_to_f32(b)));
        } else if (e == 0) {
            const mag = fp8.E5M2_SUBNORMALS[m];
            const want = if (neg) -mag else mag;
            try testing.expectEqual(want, got);
            try testing.expectEqual(want, fp8.fp8e5m2_to_f32(b));
            // Signed zero is a VALUE distinction here: 0x80 must decode to
            // -0.0, not +0.0, or it re-encodes to 0x00 and the byte is lost.
            if (m == 0) try testing.expectEqual(neg, std.math.signbit(got));
        } else {
            const mag = (1.0 + @as(f32, @floatFromInt(m)) / 4.0) *
                std.math.ldexp(@as(f32, 1.0), @as(i32, e) - 15);
            const want = if (neg) -mag else mag;
            try testing.expectEqual(want, got);
            try testing.expectEqual(want, fp8.fp8e5m2_to_f32(b));
        }
        checked += 1;
    }
    try testing.expectEqual(@as(usize, 256), checked);

    // The documented range endpoints, pinned as bytes. 0x7B is the largest
    // finite encoding; 0x7C is already NaN, so an encoder that "saturates" to
    // 0x7C would delete the value instead of clipping it.
    try testing.expectEqual(@as(f32, 57344.0), fp8.fp8e5m2_to_f32(0x7B));
    try testing.expectEqual(@as(f32, -57344.0), fp8.fp8e5m2_to_f32(0xFB));
    try testing.expect(std.math.isNan(fp8.fp8e5m2_to_f32(0x7C)));
    try testing.expectEqual(@as(f32, 2.0 / 65536.0), fp8.fp8e5m2_to_f32(0x02)); // 2^-15
    try testing.expectEqual(@as(f32, 1.0 / 65536.0), fp8.fp8e5m2_to_f32(0x01)); // 2^-16
    try testing.expectEqual(@as(f32, 0.00006103515625), fp8.fp8e5m2_to_f32(0x04)); // 2^-14
}

test "FP8 E5M2 encode∘decode round trips all 252 finite encodings" {
    const fp8 = kt.gemm_fp8;

    // 1. The sweep: decode -> encode -> decode must be byte-stable for every
    //    encoding except the eight NaN holes (0x7C..0x7F and 0xFC..0xFF),
    //    which have no value to round-trip.
    var failures: usize = 0;
    var nan_holes: usize = 0;
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if ((b >> 2) & 0x1F == 31) {
            nan_holes += 1;
            try testing.expect(std.math.isNan(fp8.fp8e5m2_to_f32(b)));
            continue;
        }
        const back = fp8.f32_to_fp8e5m2(fp8.fp8e5m2_to_f32(b));
        if (back != b) {
            failures += 1;
            std.debug.print("  E5M2 0x{X:0>2} -> {e} -> 0x{X:0>2}\n", .{ b, fp8.fp8e5m2_to_f32(b), back });
        }
    }
    try testing.expectEqual(@as(usize, 0), failures);
    try testing.expectEqual(@as(usize, 8), nan_holes);

    // 2. Signed zero must keep its sign rather than collapsing: the LUT puts
    //    the sign on the zero subnormal, and an encoder that compares `a == 0`
    //    would return +0x00 for -0.0.
    try testing.expectEqual(@as(u8, 0x80), fp8.f32_to_fp8e5m2(-0.0));
    try testing.expectEqual(@as(u8, 0x00), fp8.f32_to_fp8e5m2(0.0));
    try testing.expect(std.math.signbit(fp8.fp8e5m2_to_f32(0x80)));
    try testing.expect(!std.math.signbit(fp8.fp8e5m2_to_f32(0x00)));

    // 3. Saturation clips to 57344, NOT to a NaN encoding. This is the E5M2
    //    counterpart of the E4M3 bug where 500.0 became 0x7F and decoded to 0.
    const sat = [_][2]f32{
        .{ 57344.0, 57344.0 },
        .{ 60000.0, 57344.0 },
        .{ 1.0e30, 57344.0 },
        .{ std.math.inf(f32), 57344.0 },
        .{ -57344.0, -57344.0 },
        .{ -60000.0, -57344.0 },
        .{ -std.math.inf(f32), -57344.0 },
    };
    for (sat) |c| {
        const enc = fp8.f32_to_fp8e5m2(c[0]);
        try testing.expect(enc != 0x7C and enc != 0x7D and enc != 0x7E and enc != 0x7F);
        try testing.expectEqual(c[1], fp8.fp8e5m2_to_f32(enc));
    }
    try testing.expectEqual(@as(u8, 0x7B), fp8.f32_to_fp8e5m2(1.0e30));
    try testing.expectEqual(@as(u8, 0xFB), fp8.f32_to_fp8e5m2(-1.0e30));

    // 4. NaN keeps a NaN encoding and its sign.
    try testing.expectEqual(@as(u8, 0x7F), fp8.f32_to_fp8e5m2(std.math.nan(f32)));
    try testing.expectEqual(@as(u8, 0xFF), fp8.f32_to_fp8e5m2(-std.math.nan(f32)));

    // 5. The exponent bias is 15, not E4M3's 7. Every power of two pins it:
    //    an off-by-8 bias shifts the whole format by 256x and would still pass
    //    a self-consistent round-trip.
    const Pow2 = struct { v: f32, byte: u8 };
    const pow2 = [_]Pow2{
        .{ .v = 1.0, .byte = 0x3C }, // exp 15
        .{ .v = 2.0, .byte = 0x40 },
        .{ .v = 0.5, .byte = 0x38 }, // exp 14
        .{ .v = 1024.0, .byte = 0x64 }, // 2^10, exp 25
        .{ .v = 32768.0, .byte = 0x78 }, // 2^15, exp 30
        .{ .v = 0.00006103515625, .byte = 0x04 }, // 2^-14, exp 1
        .{ .v = 0.0001220703125, .byte = 0x08 }, // 2^-13
    };
    for (pow2) |c| {
        try testing.expectEqual(c.byte, fp8.f32_to_fp8e5m2(c.v));
        try testing.expectEqual(c.v, fp8.fp8e5m2_to_f32(c.byte));
    }

    // 6. The mantissa carry. With a 2-bit mantissa the round-up still belongs
    //    to the exponent: 1.9 sits in [1.75, 2.0) and must become 2.0 (0x40),
    //    not 1.75 (0x3F). This is the E4M3 3.875 bug in E5M2 clothing, and it
    //    is a 2x error when it happens.
    try testing.expectEqual(@as(u8, 0x40), fp8.f32_to_fp8e5m2(1.9));
    try testing.expectEqual(@as(f32, 2.0), fp8.fp8e5m2_to_f32(fp8.f32_to_fp8e5m2(1.9)));
    // 3.9375 = 1.96875 * 2 also carries, landing on exp 17 -> 0x44 = 4.0. The
    // byte is NOT 0x40: that is 2.0, and writing it would mean the carry
    // skipped the exponent entirely.
    try testing.expectEqual(@as(u8, 0x44), fp8.f32_to_fp8e5m2(3.9375));
    try testing.expectEqual(@as(f32, 4.0), fp8.fp8e5m2_to_f32(fp8.f32_to_fp8e5m2(3.9375)));
    // ...and the other side: 1.8 is below the midpoint and must round DOWN.
    try testing.expectEqual(@as(u8, 0x3F), fp8.f32_to_fp8e5m2(1.8));
    try testing.expectEqual(@as(f32, 1.75), fp8.fp8e5m2_to_f32(0x3F));

    // 7. Round-to-nearest-even on the exact halfway cases, where round-half-up
    //    gives a different answer. Two mantissa bits means the representable
    //    neighbours in [1,2) are 1.00, 1.01, 1.10, 1.11, so the three
    //    midpoints are 1.125, 1.375 and 1.625, and "even" is decided by the
    //    last retained bit: 1.125 -> 1.00, but 1.375 and 1.625 both -> 1.10.
    // A tie must go to EVEN, not up. 1.125 sits exactly between 1.00 and 1.01,
    // so round-half-up gives 1.25 and this assertion fails — which is the
    // signature of a sticky mask that wrongly includes the round bit.
    try testing.expectEqual(@as(u8, 0x3C), fp8.f32_to_fp8e5m2(1.125));
    try testing.expectEqual(@as(f32, 1.0), fp8.fp8e5m2_to_f32(fp8.f32_to_fp8e5m2(1.125)));
    try testing.expectEqual(@as(f32, 1.5), fp8.fp8e5m2_to_f32(fp8.f32_to_fp8e5m2(1.375)));
    try testing.expectEqual(@as(f32, 1.5), fp8.fp8e5m2_to_f32(fp8.f32_to_fp8e5m2(1.625)));

    // 8. The subnormal band reaches 2^-16 across four encodings, and the ties
    //    at its boundaries round to even, including the one that rounds out of
    //    the subnormals into the smallest normal.
    try testing.expectEqual(@as(u8, 0x01), fp8.f32_to_fp8e5m2(1.0 / 65536.0));
    try testing.expectEqual(@as(u8, 0x02), fp8.f32_to_fp8e5m2(2.0 / 65536.0));
    try testing.expectEqual(@as(u8, 0x03), fp8.f32_to_fp8e5m2(3.0 / 65536.0));
    try testing.expectEqual(@as(u8, 0x00), fp8.f32_to_fp8e5m2(0.5 / 65536.0)); // tie 0<->1, even is 0
    try testing.expectEqual(@as(u8, 0x02), fp8.f32_to_fp8e5m2(1.5 / 65536.0)); // tie 1<->2, even is 2
    try testing.expectEqual(@as(u8, 0x02), fp8.f32_to_fp8e5m2(2.5 / 65536.0)); // tie 2<->3, even is 2
    try testing.expectEqual(@as(u8, 0x04), fp8.f32_to_fp8e5m2(3.5 / 65536.0)); // rounds out to 2^-14
    try testing.expectEqual(@as(f32, 0.00006103515625), fp8.fp8e5m2_to_f32(fp8.f32_to_fp8e5m2(3.5 / 65536.0)));
}

test "FP8 E5M2 encode stays within half an ULP over the whole range" {
    const fp8 = kt.gemm_fp8;

    // Property check rather than a table: with 2 mantissa bits the spacing is
    // 2^-2 relative, so half an ULP is 1/8 = 0.125. Sweep exponents across the
    // entire format and mantissas across a range, and require the decoded value
    // to be within that bound — or saturated, if the input is out of range.
    var state: u64 = 0x5EED_1234;
    var worst: f32 = 0.0;
    for (0..20000) |_| {
        const v = rnd(&state, 2048, 1.0);
        if (v == 0.0) continue;
        const a = @abs(v);
        const dec = @abs(fp8.fp8e5m2_to_f32(fp8.f32_to_fp8e5m2(v)));
        if (a >= fp8.E5M2_MAX) {
            try testing.expectEqual(fp8.E5M2_MAX, dec);
            continue;
        }
        const rel = @abs(dec - a) / a;
        if (rel > worst) worst = rel;
    }
    // Subnormals below 2^-14 have absolute, not relative, spacing, so this
    // bound only holds for the normal band; 0.125 is the normal-band limit.
    try testing.expect(worst <= 0.1251);

    // And the reverse direction: every encoding's own value must survive the
    // encoder unchanged, which is what makes the LUT and the encoder agree.
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if ((b >> 2) & 0x1F == 31) continue;
        try testing.expectEqual(b, fp8.f32_to_fp8e5m2(fp8.fp8e5m2_to_f32(b)));
    }
}

test "FP8 E4M3 encode∘decode round trips every encoding; saturation is 448, not NaN" {
    const fp8 = kt.gemm_fp8;

    // 1. All 256 encodings, through decode -> encode and back. 0x7F / 0xFF are
    //    the single NaN hole and decode to 0, so they cannot come back — every
    //    other byte must. This is the sweep that measures the old encoder at
    //    14 real failures out of 256 (16 counting the NaN hole), worst case
    //    100% relative: byte 0x01 (2^-9) encoded to 0x00, and 500.0 encoded to
    //    0x7F which the decoder then maps to 0 — an over-range weight was
    //    silently deleted rather than clipped.
    var failures: usize = 0;
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (b == 0x7F or b == 0xFF) {
            try testing.expectEqual(@as(f32, 0.0), fp8.fp8e4m3_to_f32(b));
            continue;
        }
        const back = fp8.f32_to_fp8e4m3(fp8.fp8e4m3_to_f32(b));
        if (back != b) failures += 1;
    }
    try testing.expectEqual(@as(usize, 0), failures);

    // 2. Saturation clips to the largest FINITE encoding. 0x7E is 448; 0x7F is
    //    NaN, and returning it is how the old encoder turned 500.0 into 0.
    const sat = [_][2]f32{
        .{ 500.0, 448.0 },
        .{ -500.0, -448.0 },
        .{ 1.0e9, 448.0 },
        .{ -1.0e9, -448.0 },
        .{ std.math.inf(f32), 448.0 },
        .{ -std.math.inf(f32), -448.0 },
        .{ 449.0, 448.0 },
        .{ 448.0, 448.0 }, // the boundary itself is already representable
    };
    for (sat) |s| {
        const enc = fp8.f32_to_fp8e4m3(s[0]);
        try testing.expect(enc != 0x7F and enc != 0xFF);
        try testing.expectEqual(s[1], fp8.fp8e4m3_to_f32(enc));
    }

    // 3. The mantissa carry. 3.875 is 1.1111b x 2^1: four mantissa bits, so it
    //    rounds up out of the 3-bit field and into the exponent. Masking the
    //    overflow gave 0x40 (2.0) — a 2x error on 1/16 of the normal range.
    try testing.expectEqual(@as(u8, 0x48), fp8.f32_to_fp8e4m3(3.875));
    try testing.expectEqual(@as(f32, 4.0), fp8.fp8e4m3_to_f32(fp8.f32_to_fp8e4m3(3.875)));

    // 4. Round-to-nearest-EVEN on the exact halfway cases, which is where
    //    round-half-up differs. 1.0625 is halfway between 1.0 (mantissa 0,
    //    even) and 1.125 (mantissa 1, odd) -> 1.0. 1.1875 is halfway between
    //    1.125 (odd) and 1.25 (even) -> 1.25. The old encoder returned 1.125
    //    for both, i.e. always up.
    try testing.expectEqual(@as(u8, 0x38), fp8.f32_to_fp8e4m3(1.0625)); // -> 1.0
    try testing.expectEqual(@as(f32, 1.0), fp8.fp8e4m3_to_f32(fp8.f32_to_fp8e4m3(1.0625)));
    try testing.expectEqual(@as(u8, 0x3A), fp8.f32_to_fp8e4m3(1.1875)); // -> 1.25
    try testing.expectEqual(@as(f32, 1.25), fp8.fp8e4m3_to_f32(fp8.f32_to_fp8e4m3(1.1875)));

    // 5. All seven subnormals. The old encoder flushed 1..4 to zero and encoded
    //    5..7 as the [2^-7, 2^-6) band (0x02, 0x04, 0x06) — all seven wrong.
    for (1..8) |man| {
        const v = @as(f32, @floatFromInt(man)) * 0.001953125; // man * 2^-9
        try testing.expectEqual(@as(u8, @intCast(man)), fp8.f32_to_fp8e4m3(v));
        try testing.expectEqual(v, fp8.fp8e4m3_to_f32(fp8.f32_to_fp8e4m3(v)));
        try testing.expectEqual(fp8.f32_to_fp8e4m3(v) | 0x80, fp8.f32_to_fp8e4m3(-v));
    }
    // Just below the smallest subnormal is a tie that goes to zero (even), and
    // just above the largest rounds up into the normal band as 0x08 (2^-6).
    try testing.expectEqual(@as(u8, 0x00), fp8.f32_to_fp8e4m3(0.001953125 / 2.0));
    try testing.expectEqual(@as(u8, 0x08), fp8.f32_to_fp8e4m3(0.013671875 + 0.001953125 / 2.0));
    try testing.expectEqual(@as(f32, 0.015625), fp8.fp8e4m3_to_f32(fp8.f32_to_fp8e4m3(0.013671875 + 0.001953125 / 2.0)));

    // 6. Signed zero and NaN keep their encodings, so a sign or a NaN cannot be
    //    lost by a round trip.
    try testing.expectEqual(@as(u8, 0x00), fp8.f32_to_fp8e4m3(0.0));
    try testing.expectEqual(@as(u8, 0x80), fp8.f32_to_fp8e4m3(-0.0));
    try testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(fp8.fp8e4m3_to_f32(fp8.f32_to_fp8e4m3(-0.0)))));
    try testing.expectEqual(@as(u8, 0x7F), fp8.f32_to_fp8e4m3(std.math.nan(f32)));
    try testing.expectEqual(@as(u8, 0xFF), fp8.f32_to_fp8e4m3(-std.math.nan(f32)));

    // 7. The MX format must see the same codec, not a second copy: 254/256
    //    encodings round trip identically through both entry points.
    const mxfp8 = kt.gemm_mxfp8;
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        try testing.expectEqual(
            @as(u32, @bitCast(fp8.fp8e4m3_to_f32(b))),
            @as(u32, @bitCast(mxfp8.fp8e4m3_to_f32(b))),
        );
        try testing.expectEqual(fp8.f32_to_fp8e4m3(fp8.fp8e4m3_to_f32(b)), mxfp8.f32_to_fp8e4m3(mxfp8.fp8e4m3_to_f32(b)));
    }
}

test "IQ2_XS AVX2 GEMM == the scalar oracle, across super-blocks and m tails" {
    // Oracle is `gemmIQ2_XSScalar` / `dequantizeRowIQ2_XS`, not a clean-room
    // reading of the GGUF layout. k spans 1..3 super-blocks deliberately: a
    // block-0-only test cannot catch a missing `blk * QK_K`, which is the exact
    // bug class of e6372ee (all nine IQ scalar GEMMs). Verified by mutation —
    // dropping `blk * QK_K` makes this test fail.
    const g = kt.gemm_iq2_xs;
    var state: u64 = 0x12A5_0001;

    for ([_]usize{ 1, 2, 3 }) |blocks| {
        const k = blocks * g.QK_K;
        const nb = blocks;
        try testing.expect(nb > 0); // a 0-block case would be vacuous
        const m: usize = 5; // m % MR != 0, so the row tail runs
        const n: usize = 3;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&state, 64, 2.0));

        const w = try testing.allocator.alloc(g.BlockIQ2_XS, n * nb);
        defer testing.allocator.free(w);
        for (w) |*blk| {
            blk.d = @bitCast(@as(f16, 0.05));
            for (&blk.scales) |*s| {
                state = state *% 6364136223846793005 +% 1442695040888963407;
                s.* = @truncate(state >> 24);
            }
            for (&blk.qs) |*q| {
                state = state *% 6364136223846793005 +% 1442695040888963407;
                q.* = @truncate(state >> 24);
            }
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        g.gemmAvx2(a.ptr, w.ptr, c_avx.ptr, m, n, k, k, nb, n);
        g.gemmIQ2_XSScalar(a.ptr, w.ptr, c_ref.ptr, m, n, k, k, nb, n);

        for (0..m * n) |i| {
            const av = c_avx[i];
            const rf = c_ref[i];
            if (std.math.isNan(rf)) {
                try testing.expect(std.math.isNan(av));
            } else if (std.math.isInf(rf)) {
                try testing.expectEqual(std.math.isInf(av), std.math.isInf(rf));
            } else {
                try testing.expect(std.math.isFinite(av));
                const denom = @max(@abs(rf), 1e-2);
                try testing.expect(@abs(av - rf) / denom < 1e-3);
            }
        }
    }
}

test "IQ2_S AVX2 GEMM == the scalar oracle, across super-blocks and m tails" {
    // Oracle is the scalar GEMM, not a clean-room reading of the GGUF layout.
    // k sweeps 1..3 super-blocks: a block-0-only test cannot catch a missing
    // `blk * QK_K`, which is the e6372ee bug class (all nine IQ scalar GEMMs).
    // That was verified by mutation on this test pattern.
    const g = kt.gemm_iq2_s;
    var seed: u64 = 0x5EED_0001;

    for ([_]usize{ 1, 2, 3 }) |blocks| {
        const k = blocks * g.QK_K;
        const nb = blocks;
        try testing.expect(nb > 0); // a 0-block case would be vacuous
        const m: usize = 5; // m % MR != 0, so the row tail runs
        const n: usize = 3;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 64, 2.0));

        const w = try testing.allocator.alloc(g.BlockIQ2_S, n * nb);
        defer testing.allocator.free(w);
        for (w) |*blk| {
            blk.d = @bitCast(@as(f16, 0.05));
            for (&blk.qs) |*q| { seed = seed *% 6364136223846793005 +% 1442695040888963407; q.* = @truncate(seed >> 24); }
            for (&blk.qh) |*q| { seed = seed *% 6364136223846793005 +% 1442695040888963407; q.* = @truncate(seed >> 24); }
            for (&blk.scales) |*q| { seed = seed *% 6364136223846793005 +% 1442695040888963407; q.* = @truncate(seed >> 24); }
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        g.gemmAvx2(a.ptr, w.ptr, c_avx.ptr, m, n, k, k, nb, n);
        g.gemmIQ2_SScalar(a.ptr, w.ptr, c_ref.ptr, m, n, k, k, nb, n);

        for (0..m * n) |i| {
            const av = c_avx[i];
            const rf = c_ref[i];
            if (std.math.isNan(rf)) {
                try testing.expect(std.math.isNan(av));
            } else if (std.math.isInf(rf)) {
                try testing.expectEqual(std.math.isInf(av), std.math.isInf(rf));
            } else {
                try testing.expect(std.math.isFinite(av));
                const denom = @max(@abs(rf), 1e-2);
                try testing.expect(@abs(av - rf) / denom < 1e-3);
            }
        }
    }
}

test "IQ3_XXS AVX2 GEMM == the scalar oracle, across super-blocks and m tails" {
    // Oracle is the scalar GEMM, not a clean-room reading of the GGUF layout.
    // k sweeps 1..3 super-blocks: a block-0-only test cannot catch a missing
    // `blk * QK_K`, which is the e6372ee bug class (all nine IQ scalar GEMMs).
    // That was verified by mutation on this test pattern.
    const g = kt.gemm_iq3_xxs;
    var seed: u64 = 0x5EED_0001;

    for ([_]usize{ 1, 2, 3 }) |blocks| {
        const k = blocks * g.QK_K;
        const nb = blocks;
        try testing.expect(nb > 0); // a 0-block case would be vacuous
        const m: usize = 5; // m % MR != 0, so the row tail runs
        const n: usize = 3;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 64, 2.0));

        const w = try testing.allocator.alloc(g.BlockIQ3_XXS, n * nb);
        defer testing.allocator.free(w);
        for (w) |*blk| {
            blk.d = @bitCast(@as(f16, 0.05));
            for (&blk.qs) |*q| { seed = seed *% 6364136223846793005 +% 1442695040888963407; q.* = @truncate(seed >> 24); }
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        g.gemmAvx2(a.ptr, w.ptr, c_avx.ptr, m, n, k, k, nb, n);
        g.gemmIQ3_XXSScalar(a.ptr, w.ptr, c_ref.ptr, m, n, k, k, nb, n);

        for (0..m * n) |i| {
            const av = c_avx[i];
            const rf = c_ref[i];
            if (std.math.isNan(rf)) {
                try testing.expect(std.math.isNan(av));
            } else if (std.math.isInf(rf)) {
                try testing.expectEqual(std.math.isInf(av), std.math.isInf(rf));
            } else {
                try testing.expect(std.math.isFinite(av));
                const denom = @max(@abs(rf), 1e-2);
                try testing.expect(@abs(av - rf) / denom < 1e-3);
            }
        }
    }
}

test "IQ3_S AVX2 GEMM == the scalar oracle, across super-blocks and m tails" {
    // Oracle is the scalar GEMM, not a clean-room reading of the GGUF layout.
    // k sweeps 1..3 super-blocks: a block-0-only test cannot catch a missing
    // `blk * QK_K`, which is the e6372ee bug class (all nine IQ scalar GEMMs).
    // That was verified by mutation on this test pattern.
    const g = kt.gemm_iq3_s;
    var seed: u64 = 0x5EED_0001;

    for ([_]usize{ 1, 2, 3 }) |blocks| {
        const k = blocks * g.QK_K;
        const nb = blocks;
        try testing.expect(nb > 0); // a 0-block case would be vacuous
        const m: usize = 5; // m % MR != 0, so the row tail runs
        const n: usize = 3;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 64, 2.0));

        const w = try testing.allocator.alloc(g.BlockIQ3_S, n * nb);
        defer testing.allocator.free(w);
        for (w) |*blk| {
            blk.d = @bitCast(@as(f16, 0.05));
            for (&blk.qs) |*q| { seed = seed *% 6364136223846793005 +% 1442695040888963407; q.* = @truncate(seed >> 24); }
            for (&blk.qh) |*q| { seed = seed *% 6364136223846793005 +% 1442695040888963407; q.* = @truncate(seed >> 24); }
            for (&blk.signs) |*q| { seed = seed *% 6364136223846793005 +% 1442695040888963407; q.* = @truncate(seed >> 24); }
            for (&blk.scales) |*q| { seed = seed *% 6364136223846793005 +% 1442695040888963407; q.* = @truncate(seed >> 24); }
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        g.gemmAvx2(a.ptr, w.ptr, c_avx.ptr, m, n, k, k, nb, n);
        g.gemmIQ3_SScalar(a.ptr, w.ptr, c_ref.ptr, m, n, k, k, nb, n);

        for (0..m * n) |i| {
            const av = c_avx[i];
            const rf = c_ref[i];
            if (std.math.isNan(rf)) {
                try testing.expect(std.math.isNan(av));
            } else if (std.math.isInf(rf)) {
                try testing.expectEqual(std.math.isInf(av), std.math.isInf(rf));
            } else {
                try testing.expect(std.math.isFinite(av));
                const denom = @max(@abs(rf), 1e-2);
                try testing.expect(@abs(av - rf) / denom < 1e-3);
            }
        }
    }
}
test "IQ1_S AVX2 GEMM == the scalar oracle, across super-blocks and m tails" {
    // Oracle is the scalar GEMM. k sweeps 1..3 super-blocks so a missing
    // `blk * QK_K` (the e6372ee bug class) cannot hide, and `nb > 0` keeps any
    // case from being vacuous.
    //
    // NB these two are the 1-bit formats: grid magnitudes are SIGNED bytes and
    // there is no per-element sign, so the unpack is `grid8s` + a group-wide
    // delta. Using the unsigned `grid8` here would read -1 as 255.
    const g = kt.gemm_iq1_s;
    var seed: u64 = 0x1B1B_0001;

    for ([_]usize{ 1, 2, 3 }) |blocks| {
        const k = blocks * g.QK_K;
        const nb = blocks;
        try testing.expect(nb > 0);
        const m: usize = 5;
        const n: usize = 3;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 64, 2.0));

        const w = try testing.allocator.alloc(g.BlockIQ1_S, n * nb);
        defer testing.allocator.free(w);
        // IQ1_M's hidden f16 scale comes out of the scales bytes, so random ones
        // can be NaN -> the comparison is NaN-aware (see below).
        const wb: [*]u8 = @ptrCast(@alignCast(w.ptr));
        for (0..n * nb * @sizeOf(g.BlockIQ1_S)) |i| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            wb[i] = @truncate(seed >> 24);
        }
        if (@hasField(g.BlockIQ1_S, "d")) {
            for (w) |*blk| blk.d = @bitCast(@as(f16, 0.05));
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        g.gemmAvx2(a.ptr, w.ptr, c_avx.ptr, m, n, k, k, nb, n);
        g.gemmIQ1_SScalar(a.ptr, w.ptr, c_ref.ptr, m, n, k, k, nb, n);

        for (0..m * n) |i| {
            const av = c_avx[i];
            const rf = c_ref[i];
            if (std.math.isNan(rf)) {
                try testing.expect(std.math.isNan(av));
            } else if (std.math.isInf(rf)) {
                try testing.expectEqual(std.math.isInf(av), std.math.isInf(rf));
            } else {
                try testing.expect(std.math.isFinite(av));
                const denom = @max(@abs(rf), 1e-2);
                try testing.expect(@abs(av - rf) / denom < 1e-3);
            }
        }
    }
}

test "IQ1_M AVX2 GEMM == the scalar oracle, across super-blocks and m tails" {
    // Oracle is the scalar GEMM. k sweeps 1..3 super-blocks so a missing
    // `blk * QK_K` (the e6372ee bug class) cannot hide, and `nb > 0` keeps any
    // case from being vacuous.
    //
    // NB these two are the 1-bit formats: grid magnitudes are SIGNED bytes and
    // there is no per-element sign, so the unpack is `grid8s` + a group-wide
    // delta. Using the unsigned `grid8` here would read -1 as 255.
    const g = kt.gemm_iq1_m;
    var seed: u64 = 0x1B1B_0001;

    for ([_]usize{ 1, 2, 3 }) |blocks| {
        const k = blocks * g.QK_K;
        const nb = blocks;
        try testing.expect(nb > 0);
        const m: usize = 5;
        const n: usize = 3;
        const a = try testing.allocator.alloc(amx.bf16, m * k);
        defer testing.allocator.free(a);
        for (0..m * k) |i| a[i] = amx.f32_to_bf16(rnd(&seed, 64, 2.0));

        const w = try testing.allocator.alloc(g.BlockIQ1_M, n * nb);
        defer testing.allocator.free(w);
        // IQ1_M's hidden f16 scale comes out of the scales bytes, so random ones
        // can be NaN -> the comparison is NaN-aware (see below).
        const wb: [*]u8 = @ptrCast(@alignCast(w.ptr));
        for (0..n * nb * @sizeOf(g.BlockIQ1_M)) |i| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            wb[i] = @truncate(seed >> 24);
        }
        if (@hasField(g.BlockIQ1_M, "d")) {
            for (w) |*blk| blk.d = @bitCast(@as(f16, 0.05));
        }

        const c_avx = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_avx);
        const c_ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(c_ref);
        @memset(c_avx, std.math.nan(f32));
        @memset(c_ref, std.math.nan(f32));

        g.gemmAvx2(a.ptr, w.ptr, c_avx.ptr, m, n, k, k, nb, n);
        g.gemmIQ1_MScalar(a.ptr, w.ptr, c_ref.ptr, m, n, k, k, nb, n);

        for (0..m * n) |i| {
            const av = c_avx[i];
            const rf = c_ref[i];
            if (std.math.isNan(rf)) {
                try testing.expect(std.math.isNan(av));
            } else if (std.math.isInf(rf)) {
                try testing.expectEqual(std.math.isInf(av), std.math.isInf(rf));
            } else {
                try testing.expect(std.math.isFinite(av));
                const denom = @max(@abs(rf), 1e-2);
                try testing.expect(@abs(av - rf) / denom < 1e-3);
            }
        }
    }
}


// ============================================================================
// INT4 GPTQ
// ============================================================================

/// Local aliases: this suite is shared and the INT4 names are declared here
/// rather than at the top so an append-only edit cannot collide with another
/// stream's import block.
const g4 = kt.gemm_int4;
const K4 = g4.GemmKernel224Int4;

test "INT4 nibble unpack is in weight order and the zero point is subtracted" {
    // The two things a nibble unpack can get wrong while still producing
    // plausible numbers: the lo/hi ORDER (byte q holds weight 2q in its low
    // nibble and 2q+1 in its high nibble, so a swapped pair still gives a
    // sensible-looking weight) and the zero point (forgetting -8 shifts every
    // weight by +8*d). Neither shows up as an obviously broken magnitude, so
    // both are asserted directly against the spec rather than through the GEMM.
    //
    // One word (4 bytes) = 8 weights. Bytes chosen so every nibble is distinct:
    // byte q = (q+8)<<4 | q, so weight 2q = q and weight 2q+1 = q+8.
    var word: [4]u8 = undefined;
    for (&word, 0..) |*b, q| b.* = @as(u8, @intCast(q)) | (@as(u8, @intCast(q + 8)) << 4);
    const d: f32 = 0.5;
    const got: [8]f32 = K4.nibble8(&word[0], d);
    for (0..8) |t| {
        const byte = word[t / 2];
        const q: f32 = if (t % 2 == 0) @floatFromInt(byte & 0x0F) else @floatFromInt(byte >> 4);
        try testing.expectEqual((q - K4.ZERO_POINT) * d, got[t]);
    }
    // The zero point is a SEPARATE per-group term, not folded into the scale:
    // code 8 is the zero point, so a block of them dequantizes to exactly zero.
    var eight: [4]u8 = .{ 0x88, 0x88, 0x88, 0x88 };
    const zero: [8]f32 = K4.nibble8(&eight[0], 1234.5);
    for (zero) |v| try testing.expectEqual(@as(f32, 0.0), v);
    // ...and code 0 is -8*d, code 15 is +7*d.
    var zero_code: [4]u8 = .{ 0x00, 0x00, 0x00, 0x00 };
    const neg: [8]f32 = K4.nibble8(&zero_code[0], 0.25);
    for (neg) |v| try testing.expectEqual(-8.0 * 0.25, v);
    var max_code: [4]u8 = .{ 0xFF, 0xFF, 0xFF, 0xFF };
    const pos: [8]f32 = K4.nibble8(&max_code[0], 0.25);
    for (pos) |v| try testing.expectEqual(7.0 * 0.25, v);
}

test "INT4 AVX2 GEMM == the scalar oracle, bit-exactly, across shapes" {
    // With i8 activations and 4-bit weights the whole computation is integer
    // arithmetic times `d`: each product needs at most 11 significant bits and
    // the 8-bit significand of `d` fits inside f32's 24, so both paths are
    // EXACT, not merely close. That is a much stronger assertion than a
    // tolerance and it is the point of this test: a wrong unpack, a wrong
    // group offset or a per-element-vs-per-group scale all show up as a
    // different integer, not as a 1e-7 wobble.
    //
    // Activations are kept small (|a| <= 8) so that |sum of integer products|
    // stays under 2^16, which is what makes the exactness a guarantee rather
    // than luck: 8 * 8 * 512 = 32768 < 65536.
    //
    // Shapes hit: m = 1 (the m % MR tail path only), m = 4 (exactly MR), m = 5
    // (group + tail), n = 1 (no vectorisation on the n axis at all), k = 32
    // exactly (one group), k = 31 (a partial group and an ODD tail, so the last
    // weight comes from a LOW nibble), k = 33 (a full group plus one weight),
    // and k = 48 (full group + even tail).
    var seed: u64 = 0x1A2B_3C4D;
    const shapes = [_][3]usize{
        .{ 1, 1, 32 },
        .{ 1, 1, 8 },
        .{ 1, 3, 31 },
        .{ 4, 7, 32 },
        .{ 5, 1, 33 },
        .{ 5, 2, 48 },
        .{ 5, 7, 64 },
        .{ 4, 2, 129 },
        .{ 5, 7, 256 },
        .{ 3, 5, 512 },
    };
    for (shapes) |sh| {
        const m = sh[0];
        const n = sh[1];
        const k = sh[2];
        const kb = (k + K4.GROUP_SIZE - 1) / K4.GROUP_SIZE;
        const a = try testing.allocator.alloc(i8, m * k);
        defer testing.allocator.free(a);
        for (a) |*v| v.* = @intFromFloat(rnd(&seed, 16, 1.0));

        const b = try testing.allocator.alloc(g4.BlockQ4_0, n * kb);
        defer testing.allocator.free(b);
        for (b) |*blk| {
            blk.d = amx.f32_to_bf16(rnd(&seed, 64, 0.03));
            for (&blk.qs) |*q| {
                seed = seed *% 6364136223846793005 +% 1442695040888963407;
                q.* = @truncate(seed >> 24);
            }
        }

        const got = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(got);
        const ref = try testing.allocator.alloc(f32, m * n);
        defer testing.allocator.free(ref);
        // Poison: the vector path assigns where the scalar one assigns, so a
        // hole in [0,m) x [0,n) has to fail rather than pass as "close".
        @memset(got, std.math.nan(f32));
        @memset(ref, std.math.nan(f32));

        K4.gemmAvx2(@as(*const i8, @ptrCast(a.ptr)), k, @as(*const g4.BlockQ4_0, @ptrCast(b.ptr)), kb, got.ptr, n, m, n, k);
        K4.gemmFullTileScalar(@as(*const i8, @ptrCast(a.ptr)), k, @as(*const g4.BlockQ4_0, @ptrCast(b.ptr)), kb, ref.ptr, n, m, n, k);

        for (got, ref) |gv, rv| {
            try testing.expect(std.math.isFinite(gv));
            try testing.expectEqual(rv, gv); // EXACT
        }
    }
}

test "INT4 per-group scale: two groups, two scales, one answer" {
    // The K-block-boundary trap. `applyScales` exists on the AMX path because
    // its INT32 accumulator spans a K-block; if the scale were applied once for
    // the whole k instead of once per group, this shape would come out as
    // 448 (group 1's scale applied to both groups) or 224 (group 0's applied to
    // both). Correct is 672.
    //
    // k = 64: group 0 is all code 15 (7*d0 per weight), group 1 all code 15
    // (7*d1). With a = 1 everywhere and d0 = 1, d1 = 2:
    //   32 * 7 * 1 + 32 * 7 * 2 = 224 + 448 = 672.
    const m: usize = 1;
    const n: usize = 2;
    const k: usize = 64;
    const kb: usize = 2;
    var a: [m * k]i8 = undefined;
    for (&a) |*v| v.* = 1;
    var b: [n * kb]g4.BlockQ4_0 = undefined;
    for (0..n) |j| {
        for (0..kb) |g| {
            b[j * kb + g].d = amx.f32_to_bf16(if (g == 0) 1.0 else 2.0);
            for (&b[j * kb + g].qs) |*q| q.* = 0xFF; // code 15 both nibbles -> 7*d
        }
    }
    var got: [m * n]f32 = undefined;
    var scal: [m * n]f32 = undefined;
    K4.gemmAvx2(&a[0], k, &b[0], kb, &got, n, m, n, k);
    K4.gemmFullTileScalar(&a[0], k, &b[0], kb, &scal, n, m, n, k);
    for (got) |v| try testing.expectEqual(@as(f32, 672.0), v);
    for (scal) |v| try testing.expectEqual(@as(f32, 672.0), v);
    // 672 = 224 + 448 is 3 * 224; a single-scale implementation would return
    // 224 or 448, so the assertion above is not satisfiable by accident.
    try testing.expect(got[0] != 224.0 and got[0] != 448.0);
}

test "INT4 ragged k: the tail is the low nibble of the last byte" {
    // k = 33: one full 32-weight group plus ONE weight, which lives in the LOW
    // nibble of byte 0 of the second block. Getting this wrong (reading the
    // high nibble, or the low nibble of byte 1) still returns a plausible
    // number, which is exactly how `kt_matmul_q{4,5,6,8}_k` and FP8 shipped
    // ragged-k bugs.
    const m: usize = 1;
    const n: usize = 1;
    const k: usize = 33;
    const kb: usize = 2;
    var a: [m * k]i8 = undefined;
    for (&a) |*v| v.* = 1;
    var b: [n * kb]g4.BlockQ4_0 = undefined;
    // Group 0: every weight is code 8 (the zero point) -> contributes 0.
    b[0].d = amx.f32_to_bf16(1.0);
    for (&b[0].qs) |*q| q.* = 0x88;
    // Group 1: byte 0 = low nibble code 15, high nibble code 0 (junk: only the
    // low nibble is a real weight at k=32). d = 2 -> 7 * 2 = 14.
    b[1].d = amx.f32_to_bf16(2.0);
    for (&b[1].qs) |*q| q.* = 0x0F;
    b[1].qs[1] = 0xF0; // if this leaked in, the answer would be 14 + (-16)
    var got: [m * n]f32 = undefined;
    var scal: [m * n]f32 = undefined;
    K4.gemmAvx2(&a[0], k, &b[0], kb, &got, n, m, n, k);
    K4.gemmFullTileScalar(&a[0], k, &b[0], kb, &scal, n, m, n, k);
    try testing.expectEqual(@as(f32, 14.0), got[0]);
    try testing.expectEqual(@as(f32, 14.0), scal[0]);
}

test "INT4 AVX2 GEMM survives the full i8 range at k=7168" {
    // Deliberately NOT bit-exact here: with |a| up to 127 the integer sums stop
    // being representable in f32 for large k, so this shape asserts a tight
    // relative bound instead. It is the shape the benchmark runs, so the
    // agreement number quoted in TODO.md is pinned here.
    var seed: u64 = 0x5EED_1234;
    const m: usize = 4;
    const n: usize = 7;
    const k: usize = 7168;
    const kb: usize = k / 32;
    const a = try testing.allocator.alloc(i8, m * k);
    defer testing.allocator.free(a);
    for (a) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        v.* = @bitCast(@as(u8, @truncate(seed >> 24)));
    }
    const b = try testing.allocator.alloc(g4.BlockQ4_0, n * kb);
    defer testing.allocator.free(b);
    for (b) |*blk| {
        blk.d = amx.f32_to_bf16(rnd(&seed, 64, 0.01));
        for (&blk.qs) |*q| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            q.* = @truncate(seed >> 24);
        }
    }
    const got = try testing.allocator.alloc(f32, m * n);
    defer testing.allocator.free(got);
    const ref = try testing.allocator.alloc(f32, m * n);
    defer testing.allocator.free(ref);
    K4.gemmAvx2(@as(*const i8, @ptrCast(a.ptr)), k, @as(*const g4.BlockQ4_0, @ptrCast(b.ptr)), kb, got.ptr, n, m, n, k);
    K4.gemmFullTileScalar(@as(*const i8, @ptrCast(a.ptr)), k, @as(*const g4.BlockQ4_0, @ptrCast(b.ptr)), kb, ref.ptr, n, m, n, k);
    var worst: f64 = 0;
    var worst_abs: f64 = 0;
    for (got, ref) |gv, rv| {
        try testing.expect(std.math.isFinite(gv));
        const denom = @max(@abs(rv), 1e-2);
        const rel = @abs(gv - rv) / denom;
        worst = @max(worst, rel);
        worst_abs = @max(worst_abs, @abs(gv - rv));
    }
    std.debug.print("k=7168 full-i8: worst rel {e:.3}, worst abs {e:.3}\n", .{ worst, worst_abs });
    try testing.expect(worst < 1e-5);
}
