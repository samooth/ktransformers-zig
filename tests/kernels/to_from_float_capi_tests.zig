// Tests for the kt_to_float / kt_from_float / kt_type_row_bytes C API
// (Zig extension — generic block ↔ F32 dispatch).
//
// These ports llama.cpp's to_float / from_float generic dispatch
// (ggml-common.h:336) so the framework's expert code can convert
// activations between formats without knowing the type at compile time
// (e.g. `from_float(s_input_fp32, s_gate_input, hidden, vec_dot_type)`
// in LLAMA_MOE_TP's forward, moe.hpp:269).
//
// Covers the 8 linear (non-IQ) GGML block formats we have C-API
// exports for: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, Q8_K. The IQ
// formats are intentionally not in the dispatch (different per-row
// sizes) — callers use the per-type exports.

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig (wired by build.zig)

const KT_TYPE_F32: c_int = 0;
const KT_TYPE_F16: c_int = 1;
const KT_TYPE_BF16: c_int = 2;
const KT_TYPE_Q8_0: c_int = 12;
const KT_TYPE_Q4_K: c_int = 16;
const KT_TYPE_Q5_K: c_int = 17;
const KT_TYPE_Q6_K: c_int = 18;
const KT_TYPE_Q8_K: c_int = 19;

test "kt_type_row_bytes: known block sizes" {
    // Per-type, n=256 (one full block for all K-quants, 8 blocks for Q8_0).
    try testing.expectEqual(@as(c_int, 8 * 34), kt.kt_type_row_bytes(256, KT_TYPE_Q8_0)); // 8 blocks × 34B
    try testing.expectEqual(@as(c_int, 144), kt.kt_type_row_bytes(256, KT_TYPE_Q4_K));
    try testing.expectEqual(@as(c_int, 176), kt.kt_type_row_bytes(256, KT_TYPE_Q5_K));
    try testing.expectEqual(@as(c_int, 210), kt.kt_type_row_bytes(256, KT_TYPE_Q6_K));
    try testing.expectEqual(@as(c_int, 292), kt.kt_type_row_bytes(256, KT_TYPE_Q8_K));
    // F32, F16, BF16: byte-equivalent to the scalar size.
    try testing.expectEqual(@as(c_int, 32 * 4), kt.kt_type_row_bytes(32, KT_TYPE_F32));
    try testing.expectEqual(@as(c_int, 32 * 2), kt.kt_type_row_bytes(32, KT_TYPE_F16));
    try testing.expectEqual(@as(c_int, 32 * 2), kt.kt_type_row_bytes(32, KT_TYPE_BF16));
    // Unsupported types in the to/from_float dispatch: typeSrcBytes still
    // reports a size (F32/F16/BF16/I8 are pure-scalar formats — 1/2/2/1 byte
    // per element), but the dispatch itself returns -1 (see the
    // unsupported-type test below). The IQ formats (20..28) are out of
    // scope by design — their block layouts have different per-row sizes.
    try testing.expectEqual(@as(c_int, 32), kt.kt_type_row_bytes(32, 3)); // KT_TYPE_I8 (1B/elem)
    try testing.expectEqual(@as(c_int, 0), kt.kt_type_row_bytes(256, 20)); // KT_TYPE_IQ2_XXS (out of scope)
}

test "kt_to_float F32 → F32 is identity" {
    const n: usize = 32;
    const src = [_]f32{ 1.0, -2.5, 3.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0 };
    var dst: [n]f32 = undefined;
    const rc = kt.kt_to_float(@as(*const anyopaque, @ptrCast(&src)), &dst, n, KT_TYPE_F32);
    try testing.expectEqual(@as(c_int, 0), rc);
    for (0..n) |i| try testing.expectEqual(src[i], dst[i]);
}

test "kt_to_float BF16 → F32" {
    // BF16(1.0) = 0x3F80
    const n: usize = 4;
    const src_bf16 = [_]u16{ 0x3F80, 0x4000, 0x4040, 0xC000 }; // 1.0, 2.0, 3.0, -2.0
    var dst: [n]f32 = undefined;
    const rc = kt.kt_to_float(@as(*const anyopaque, @ptrCast(&src_bf16)), &dst, n, KT_TYPE_BF16);
    try testing.expectEqual(@as(c_int, 0), rc);
    try testing.expectEqual(@as(f32, 1.0), dst[0]);
    try testing.expectEqual(@as(f32, 2.0), dst[1]);
    try testing.expectEqual(@as(f32, 3.0), dst[2]);
    try testing.expectEqual(@as(f32, -2.0), dst[3]);
}

test "kt_to_float Q8_0 → F32 round-trip via direct call" {
    // Q8_0: f16 d=0.5 + 32 i8 qs (all 2). Dequant: qs*0.5 = 1.0.
    const n: usize = 32;
    var block_bytes: [n / 32 * 34]u8 = undefined;
    // d = 0.5 in f16 = 0x3800 (LE: bytes 0x00, 0x38)
    block_bytes[0] = 0x00;
    block_bytes[1] = 0x38;
    // 32 i8 qs of value 2
    for (2..34) |i| {
        block_bytes[i] = @bitCast(@as(i8, 2));
    }
    var dst: [n]f32 = undefined;
    const rc = kt.kt_to_float(&block_bytes, &dst, n, KT_TYPE_Q8_0);
    try testing.expectEqual(@as(c_int, 0), rc);
    for (0..n) |i| {
        try testing.expectApproxEqAbs(@as(f32, 1.0), dst[i], 1e-3);
    }
}

test "kt_to_float Q4_K → F32 round-trip" {
    // Q4_K: 144 bytes per 256 weights. Use known-zero content -> all-zero output.
    const n: usize = 256;
    var block_bytes = [_]u8{0} ** 144; // single block
    // d and dmin are 0 in f16 (2 zero bytes each), and all scales/qs zero ->
    // dequant all 0s.
    var dst: [n]f32 = [_]f32{1.0} ** n; // init to 1 to verify they're all overwritten
    const rc = kt.kt_to_float(&block_bytes, &dst, n, KT_TYPE_Q4_K);
    try testing.expectEqual(@as(c_int, 0), rc);
    for (0..n) |i| {
        try testing.expectEqual(@as(f32, 0.0), dst[i]);
    }
}

test "kt_to_float Q5_K → F32 round-trip" {
    const n: usize = 256;
    var block_bytes = [_]u8{0} ** 176;
    var dst: [n]f32 = [_]f32{1.0} ** n;
    const rc = kt.kt_to_float(&block_bytes, &dst, n, KT_TYPE_Q5_K);
    try testing.expectEqual(@as(c_int, 0), rc);
    for (0..n) |i| {
        try testing.expectEqual(@as(f32, 0.0), dst[i]);
    }
}

test "kt_to_float Q6_K → F32 round-trip" {
    const n: usize = 256;
    var block_bytes = [_]u8{0} ** 210;
    var dst: [n]f32 = [_]f32{1.0} ** n;
    const rc = kt.kt_to_float(&block_bytes, &dst, n, KT_TYPE_Q6_K);
    try testing.expectEqual(@as(c_int, 0), rc);
    for (0..n) |i| {
        try testing.expectEqual(@as(f32, 0.0), dst[i]);
    }
}

test "kt_to_float Q8_K → F32 round-trip" {
    const n: usize = 256;
    var block_bytes = [_]u8{0} ** 292;
    var dst: [n]f32 = [_]f32{1.0} ** n;
    const rc = kt.kt_to_float(&block_bytes, &dst, n, KT_TYPE_Q8_K);
    try testing.expectEqual(@as(c_int, 0), rc);
    for (0..n) |i| {
        try testing.expectEqual(@as(f32, 0.0), dst[i]);
    }
}

test "kt_to_float unsupported type returns -1" {
    const n: usize = 32;
    var src: [128]u8 = undefined; // 32 i8s
    var dst: [n]f32 = undefined;
    // KT_TYPE_I8 = 3 (not in the dispatch)
    const rc = kt.kt_to_float(&src, &dst, n, 3);
    try testing.expectEqual(@as(c_int, -1), rc);
    // KT_TYPE_IQ2_XXS = 20 (out of scope by design)
    const rc2 = kt.kt_to_float(&src, &dst, n, 20);
    try testing.expectEqual(@as(c_int, -1), rc2);
}

test "kt_from_float F32 → F32 is identity" {
    const n: usize = 8;
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var dst: [n * 4]u8 = undefined;
    const rc = kt.kt_from_float(@as([*]const f32, @ptrCast(&src)), &dst, n, KT_TYPE_F32);
    try testing.expectEqual(@as(c_int, 0), rc);
    for (0..n) |i| {
        const got: f32 = @bitCast([4]u8{
            dst[i * 4 + 0], dst[i * 4 + 1], dst[i * 4 + 2], dst[i * 4 + 3],
        });
        try testing.expectEqual(src[i], got);
    }
}

test "kt_from_float BF16 → F32 round-trip via dispatch" {
    // F32(1.0) → BF16 → F32(1.0)
    const n: usize = 32;
    const src = [_]f32{1.0} ** n;
    var dst_bf16: [n * 2]u8 = undefined;
    const rc = kt.kt_from_float(@as([*]const f32, @ptrCast(&src)), &dst_bf16, n, KT_TYPE_BF16);
    try testing.expectEqual(@as(c_int, 0), rc);
    // Dequant and check
    var out: [n]f32 = undefined;
    const rc2 = kt.kt_to_float(&dst_bf16, &out, n, KT_TYPE_BF16);
    try testing.expectEqual(@as(c_int, 0), rc2);
    for (0..n) |i| {
        try testing.expectApproxEqAbs(@as(f32, 1.0), out[i], 1e-3);
    }
}

test "kt_from_float Q8_0 round-trip" {
    const n: usize = 32;
    const src = [_]f32{1.0} ** n;
    var dst_q8: [34]u8 = undefined;
    const rc = kt.kt_from_float(@as([*]const f32, @ptrCast(&src)), &dst_q8, n, KT_TYPE_Q8_0);
    try testing.expectEqual(@as(c_int, 0), rc);
    // Dequant
    var out: [n]f32 = undefined;
    const rc2 = kt.kt_to_float(&dst_q8, &out, n, KT_TYPE_Q8_0);
    try testing.expectEqual(@as(c_int, 0), rc2);
    for (0..n) |i| {
        try testing.expectApproxEqAbs(@as(f32, 1.0), out[i], 1e-3);
    }
}

test "kt_from_float Q4_K round-trip" {
    const n: usize = 256;
    const src = [_]f32{1.0} ** n;
    var dst_q4k: [144]u8 = undefined;
    const rc = kt.kt_from_float(@as([*]const f32, @ptrCast(&src)), &dst_q4k, n, KT_TYPE_Q4_K);
    try testing.expectEqual(@as(c_int, 0), rc);
    var out: [n]f32 = undefined;
    const rc2 = kt.kt_to_float(&dst_q4k, &out, n, KT_TYPE_Q4_K);
    try testing.expectEqual(@as(c_int, 0), rc2);
    for (0..n) |i| {
        // Q4_K has a quantization floor (~5-10% on smooth data); 1.0 is
        // an easy case so the error should be tight.
        try testing.expectApproxEqAbs(@as(f32, 1.0), out[i], 0.5);
    }
}

test "kt_from_float unsupported type returns -1" {
    const n: usize = 32;
    const src = [_]f32{1.0} ** n;
    var dst: [128]u8 = undefined;
    const rc = kt.kt_from_float(@as([*]const f32, @ptrCast(&src)), &dst, n, 3); // KT_TYPE_I8
    try testing.expectEqual(@as(c_int, -1), rc);
}
