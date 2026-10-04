// C API tests for the FP8 quantize/dequantize pairs: E4M3 and E5M2.
//
// WHY A NEW SUITE. The codec itself is exhaustively covered in
// tests/kernels/simd_kernels_tests.zig (all 256 encodings, both directions).
// What had NO coverage at all was the C API pair that wraps it — including
// `kt_quantize_fp8_e4m3`, which predates this file and was never tested. That
// layer is where the real mistakes live, and they are different mistakes:
//
//   - the per-block scale must come from THAT block's maximum, not the row's,
//     so a block of small values next to a block of large ones must not share
//     one scale (that silently costs precision everywhere);
//   - the scales array is row-major with ceil(cols / block_size) entries per
//     row, so a wrong blocks_per_row walks off the end of the array;
//   - the last block of a row may be PARTIAL when cols is not a multiple of
//     block_size, and the loops must not run past cols;
//   - and an over-range value must CLIP to the format's max, not become a NaN
//     encoding that decodes to zero. That is a bug this codebase already shipped
//     once, in the E4M3 encoder, where 500.0 encoded to the NaN byte and came
//     back as 0.0 — a weight deleted rather than clipped.
//
// The expected values here are computed with the same codec the API calls. That
// is deliberate: this suite tests the BLOCKING AND SCALE PLUMBING, not the
// codec, and the codec is pinned byte-for-byte elsewhere. Using an independent
// codec copy here would only test that two copies agree.

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig (wired by build.zig)

const amx = kt.amx;
const gemm_fp8 = kt.gemm_fp8;

/// Deterministic pseudo-random f32, matching the SIMD suite's generator so the
/// two suites see the same data.
fn rnd(seed: *u64, mod: u32, scale: f32) f32 {
    seed.* = seed.* *% 6364136223846793005 +% 1442695040888963407;
    const r: u32 = @truncate(seed.* >> 33);
    const v = @as(f32, @floatFromInt(r % mod)) - @as(f32, @floatFromInt(mod)) / 2.0;
    return v * scale;
}

/// One row-major block-quantize/dequantize round trip through the C API, with
/// the expectation recomputed independently from the block layout.
///
/// `max_of` is the format's largest finite value: the API must divide the block
/// maximum by it, and E4M3 (448) and E5M2 (57344) differ by 128x, so the same
/// data produces very different scales and very different round-trip error.
///
/// The error bound has two independent terms and both are needed:
///
///   - `rel_tol` is half an ULP of the format's mantissa (E5M2 1/8, E4M3 1/16),
///     PLUS half an ULP of bf16, because the API returns bf16 and the output
///     rounding is itself a relative error of 2^-9. This term is proportional
///     to the VALUE, not to the block scale.
///   - `abs_scaled_tol` covers values that land in the format's SUBNORMAL band
///     after scaling, where the spacing is absolute (half of 2^-16 for E5M2,
///     half of 2^-9 for E4M3) rather than relative, so it is multiplied by the
///     scale instead. Without this term every small value in a block whose
///     maximum is large fails the check.
fn checkRoundTrip(
    comptime quantize: anytype,
    comptime dequantize: anytype,
    rows: usize,
    cols: usize,
    block_size: usize,
    max_of: f32,
    rel_tol: f32,
    abs_scaled_tol: f32,
) !void {
    var state: u64 = 0xC0FF_EE01;
    const src = try testing.allocator.alloc(amx.bf16, rows * cols);
    defer testing.allocator.free(src);
    for (src) |*v| v.* = amx.f32_to_bf16(rnd(&state, 512, 2.0));

    const blocks_per_row = (cols + block_size - 1) / block_size;

    // GUARD CELLS. Each output buffer gets two extra elements that the API must
    // never touch, and this is not decoration: an over-counted blocks_per_row
    // makes the API walk past the end of the caller's scales array, and when the
    // buffer is allocated to the exact size that overflow is invisible — the
    // suite passed with blocks_per_row deliberately off by one until these
    // guards were added, and a heap overflow that no test can see is the worst
    // kind of test. The sentinel is checked after the call below.
    const GUARD: usize = 2;
    const dst = try testing.allocator.alloc(u8, rows * cols + GUARD);
    defer testing.allocator.free(dst);
    const scales = try testing.allocator.alloc(f32, rows * blocks_per_row + GUARD);
    defer testing.allocator.free(scales);
    const back = try testing.allocator.alloc(amx.bf16, rows * cols + GUARD);
    defer testing.allocator.free(back);

    @memset(scales, -1.0); // so an untouched scale entry is visible as a failure
    @memset(dst, 0xAA);
    @memset(back, 0x5A);

    const scale_sentinel: f32 = -12345.0;
    const dst_sentinel: u8 = 0xC3;
    const back_sentinel: u8 = 0xC3;
    scales[rows * blocks_per_row] = scale_sentinel;
    scales[rows * blocks_per_row + 1] = scale_sentinel;
    dst[rows * cols] = dst_sentinel;
    dst[rows * cols + 1] = dst_sentinel;
    back[rows * cols] = back_sentinel;
    back[rows * cols + 1] = back_sentinel;

    quantize(src.ptr, dst.ptr, scales.ptr, rows, cols, @intCast(block_size));
    dequantize(dst.ptr, scales.ptr, back.ptr, rows, cols, @intCast(block_size));

    // Independent expectation: the scale of a block is that block's maximum
    // divided by the format's max, and every entry of the round trip must sit
    // within `tolerance` of the original.
    for (0..rows) |r| {
        for (0..blocks_per_row) |blk| {
            const c_start = blk * block_size;
            const c_end = @min(cols, c_start + block_size); // partial last block

            var max_val: f32 = 0;
            for (c_start..c_end) |c| {
                const a = @abs(amx.bf16_to_f32(src[r * cols + c]));
                if (a > max_val) max_val = a;
            }
            const want_scale = if (max_val > 0) max_val / max_of else 1.0;
            const got_scale = scales[r * blocks_per_row + blk];
            try testing.expectApproxEqAbs(want_scale, got_scale, 1e-6);

            for (c_start..c_end) |c| {
                const orig = amx.bf16_to_f32(src[r * cols + c]);
                const got = amx.bf16_to_f32(back[r * cols + c]);
                const tol = rel_tol * @abs(orig) + abs_scaled_tol * got_scale;
                try testing.expectApproxEqAbs(orig, got, tol + 1e-30);
            }
        }
    }

    // The API wrote exactly rows*cols outputs and rows*blocks_per_row scales.
    try testing.expectEqual(scale_sentinel, scales[rows * blocks_per_row]);
    try testing.expectEqual(scale_sentinel, scales[rows * blocks_per_row + 1]);
    try testing.expectEqual(dst_sentinel, dst[rows * cols]);
    try testing.expectEqual(dst_sentinel, dst[rows * cols + 1]);
    try testing.expect(back[rows * cols] == back_sentinel and back[rows * cols + 1] == back_sentinel);
}

test "FP8 C API: E5M2 block round trip, including partial trailing blocks" {
    // 3 rows x 70 cols in blocks of 32 -> blocks_per_row = 3, and the last
    // block of each row holds only 6 of the 32 columns.
    try checkRoundTrip(
        kt.kt_quantize_fp8_e5m2,
        kt.kt_dequantize_fp8_e5m2,
        3,
        70,
        32,
        gemm_fp8.E5M2_MAX,
        0.13, // 1/8 half-ULP of a 2-bit mantissa, + bf16's 2^-9
        0.5 * (1.0 / 65536.0), // half of the 2^-16 subnormal step, in scaled units
    );
}

test "FP8 C API: E4M3 block round trip, same shapes (the pair had no coverage)" {
    try checkRoundTrip(
        kt.kt_quantize_fp8_e4m3,
        kt.kt_dequantize_fp8_e4m3,
        3,
        70,
        32,
        gemm_fp8.E4M3_MAX,
        0.07, // 1/16 half-ULP of a 3-bit mantissa, + bf16's 2^-9
        0.5 * (1.0 / 512.0), // half of the 2^-9 subnormal step, in scaled units
    );
}

test "FP8 C API: a single-element block, and cols smaller than one block" {
    // Degenerate block sizes are where an off-by-one in blocks_per_row shows up.
    try checkRoundTrip(kt.kt_quantize_fp8_e5m2, kt.kt_dequantize_fp8_e5m2, 2, 1, 32, gemm_fp8.E5M2_MAX, 0.13, 0.5 * (1.0 / 65536.0));
    try checkRoundTrip(kt.kt_quantize_fp8_e5m2, kt.kt_dequantize_fp8_e5m2, 1, 5, 1, gemm_fp8.E5M2_MAX, 0.13, 0.5 * (1.0 / 65536.0));
}

test "FP8 C API: a block's extremes survive; underflow to zero is by design" {
    const max_of = gemm_fp8.E5M2_MAX;
    const src = [_]amx.bf16{
        amx.f32_to_bf16(max_of), // exactly the format max
        amx.f32_to_bf16(-max_of),
        amx.f32_to_bf16(0.25),
    };
    var dst: [3]u8 = undefined;
    var scales: [1]f32 = undefined;
    var back: [3]amx.bf16 = undefined;

    kt.kt_quantize_fp8_e5m2(&src, &dst, &scales, 1, 3, 3);
    kt.kt_dequantize_fp8_e5m2(&dst, &scales, &back, 1, 3, 3);

    // The block maximum maps to exactly the format's maximum, and comes back
    // as itself (to within one bf16 ULP). This is the assertion that matters:
    // the largest value in a block is the one an encoder is most likely to
    // destroy, because it is the value that saturates if anything does.
    for (back, 0..) |v, i| {
        const got = amx.bf16_to_f32(v);
        try testing.expect(!std.math.isNan(got));
        try testing.expectApproxEqAbs(amx.bf16_to_f32(src[i]), got, @abs(amx.bf16_to_f32(src[i])) * 0.002 + 1e-30);
        try testing.expectEqual(i == 1, got < 0.0); // only the second is negative
    }

    // NOTE ON WHY THERE IS NO SATURATION CASE HERE. Encoder saturation is
    // unreachable through this entry point, and that is structural rather than
    // lucky: the scale is DERIVED from the block maximum as max / E5M2_MAX, so
    // the largest value in the block always divides down to exactly the format
    // max and lands on its encoding rather than past it. Saturation is
    // reachable only for a caller that supplies its own scale — the elementwise
    // `kt_fp8_quantize` path, or an FP8 checkpoint loaded with scales from
    // elsewhere. The encoder's clipping behaviour (clip to 57344, never emit a
    // NaN byte) is therefore pinned exhaustively in
    // tests/kernels/simd_kernels_tests.zig, where 1e30 and infinity are encoded
    // directly. This test covers what only the API can get wrong: the scale
    // derivation and the blocking.

    // The flip side, in two parts, because the boundary is a property of the
    // format's ~30 binades of range and is worth pinning from both sides.
    //
    // A value a few binades below its own block maximum must still round-trip
    // to within the format's precision — the scale derivation must not be
    // throwing away representable data.
    const moderate = [_]amx.bf16{
        amx.f32_to_bf16(1.0),
        amx.f32_to_bf16(0.0001), // ~8 binades down: 5.73 after scaling
    };
    var mdst: [2]u8 = undefined;
    var mscales: [1]f32 = undefined;
    var mback: [2]amx.bf16 = undefined;
    kt.kt_quantize_fp8_e5m2(&moderate, &mdst, &mscales, 1, 2, 2);
    kt.kt_dequantize_fp8_e5m2(&mdst, &mscales, &mback, 1, 2, 2);
    try testing.expectApproxEqAbs(1.0, amx.bf16_to_f32(mback[0]), 0.13);
    try testing.expectApproxEqAbs(0.0001, amx.bf16_to_f32(mback[1]), 0.0001 * 0.13);

    // But a value past the bottom of the format flushes to zero, and that IS the
    // documented behaviour rather than a defect: 1e-12 divides down to ~5.7e-8,
    // below E5M2's smallest subnormal 2^-16, so there is no encoding for it.
    // Asserted explicitly so that a future change cannot quietly alter it.
    const underflow = [_]amx.bf16{
        amx.f32_to_bf16(1.0),
        amx.f32_to_bf16(1e-12),
    };
    var udst: [2]u8 = undefined;
    var uscales: [1]f32 = undefined;
    var uback: [2]amx.bf16 = undefined;
    kt.kt_quantize_fp8_e5m2(&underflow, &udst, &uscales, 1, 2, 2);
    kt.kt_dequantize_fp8_e5m2(&udst, &uscales, &uback, 1, 2, 2);
    try testing.expect(amx.bf16_to_f32(uback[0]) > 0.9);
    try testing.expectEqual(@as(f32, 0.0), amx.bf16_to_f32(uback[1]));
}

test "FP8 C API: E5M2 and E4M3 scales differ by the formats' 128x range ratio" {
    // A cheap structural check that the two APIs are not accidentally sharing a
    // constant: identical input must produce scales in the E5M2:E4M3 ratio
    // 57344:448 = 128 exactly.
    const src = [_]amx.bf16{amx.f32_to_bf16(100.0), amx.f32_to_bf16(-50.0), amx.f32_to_bf16(0.25)};
    var dst: [3]u8 = undefined;
    // Arrays, not scalars: a single f32 has no address to coerce, while
    // &[1]f32 coerces to the [*]f32 the API takes.
    var scales4: [1]f32 = undefined;
    var scales5: [1]f32 = undefined;

    kt.kt_quantize_fp8_e4m3(&src, &dst, &scales4, 1, 3, 3);
    kt.kt_quantize_fp8_e5m2(&src, &dst, &scales5, 1, 3, 3);

    try testing.expectApproxEqAbs(128.0, scales4[0] / scales5[0], 1e-4);
    try testing.expectApproxEqAbs(100.0 / gemm_fp8.E4M3_MAX, scales4[0], 1e-6);
    try testing.expectApproxEqAbs(100.0 / gemm_fp8.E5M2_MAX, scales5[0], 1e-6);
}
