// Standalone test for the GGML quant C API (TODO: Phase 2 wrappers).
//
// Exercises the one-row quantize/dequantize exports and the matmul exports
// through the actual C ABI (pub export fn in src/main.zig):
//   kt_quantize_qX / kt_dequantize_qX  — one-row round trip per format
//   kt_matmul_qX                       — constant-weight GEMM per format
//
// Verifies the full path: Python-side caller -> C ABI -> kernel-layer
// reference implementation. The kernel-layer math itself is covered by the
// per-format tests in test_kernels.zig; this file covers the wrapper plumbing
// (slice cuts, pointer casts, param marshaling).

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig (wired by build.zig)

test "GGML Q8_0 C API: quantize/dequantize row round trip" {
    const k = 64; // 2 blocks
    var src: [k]f32 = undefined;
    for (0..k) |i| src[i] = @floatFromInt(@as(i32, @intCast(i % 17)) - 8);

    // dst: 2 blocks * 34 bytes, 4-aligned (BlockQ8_0 has u16 first field)
    var dst: [2 * 34]u8 align(@alignOf(kt.gemm_q8_0.BlockQ8_0)) = undefined;
    kt.kt_quantize_q8_0(&src, &dst, k);

    var back: [k]f32 = undefined;
    kt.kt_dequantize_q8_0(&dst, &back, k);

    const d: f32 = 8.0 / 127.0;
    for (0..k) |i| {
        try testing.expectApproxEqAbs(src[i], back[i], d * 0.51);
    }
}

test "GGML Q4_K C API: quantize/dequantize row round trip" {
    const k = 256; // one super-block
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5 + 0.1 * @as(f32, @floatFromInt(i % 7));
    }

    var dst: [144]u8 align(@alignOf(kt.gemm_q4_k.BlockQ4_K)) = undefined;
    kt.kt_quantize_q4_k(&src, &dst, k);

    var back: [k]f32 = undefined;
    kt.kt_dequantize_q4_k(&dst, &back, k);

    var max_abs_err: f32 = 0;
    var sum_abs_x: f32 = 0;
    for (0..k) |i| {
        max_abs_err = @max(max_abs_err, @abs(src[i] - back[i]));
        sum_abs_x += @abs(src[i]);
    }
    try testing.expect(max_abs_err / (sum_abs_x / k) < 0.15);
}

test "GGML Q5_K C API: quantize/dequantize row round trip" {
    const k = 256;
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5 + 0.1 * @as(f32, @floatFromInt(i % 7));
    }

    var dst: [176]u8 align(@alignOf(kt.gemm_q5_k.BlockQ5_K)) = undefined;
    kt.kt_quantize_q5_k(&src, &dst, k);

    var back: [k]f32 = undefined;
    kt.kt_dequantize_q5_k(&dst, &back, k);

    var max_abs_err: f32 = 0;
    var sum_abs_x: f32 = 0;
    for (0..k) |i| {
        max_abs_err = @max(max_abs_err, @abs(src[i] - back[i]));
        sum_abs_x += @abs(src[i]);
    }
    try testing.expect(max_abs_err / (sum_abs_x / k) < 0.12);
}

test "GGML Q6_K C API: quantize/dequantize row round trip" {
    const k = 256;
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5 + 0.1 * @as(f32, @floatFromInt(i % 7));
    }

    var dst: [210]u8 align(@alignOf(kt.gemm_q6_k.BlockQ6_K)) = undefined;
    kt.kt_quantize_q6_k(&src, &dst, k);

    var back: [k]f32 = undefined;
    kt.kt_dequantize_q6_k(&dst, &back, k);

    var max_abs_err: f32 = 0;
    var sum_abs_x: f32 = 0;
    for (0..k) |i| {
        max_abs_err = @max(max_abs_err, @abs(src[i] - back[i]));
        sum_abs_x += @abs(src[i]);
    }
    try testing.expect(max_abs_err / (sum_abs_x / k) < 0.10);
}

test "GGML C API: matmul_q8_0 constant weights" {
    const M = 4;
    const N = 2;
    const K = 32; // one Q8_0 block

    const amx = kt.amx;
    var a: [M * K]kt.amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    // Weight rows quantized from constant 1.0
    var src: [K]f32 = undefined;
    for (&src) |*v| v.* = 1.0;
    var b: [N * 34]u8 align(@alignOf(kt.gemm_q8_0.BlockQ8_0)) = undefined;
    // Row 0 at offset 0, row 1 at offset 34 — BlockQ8_0-sized steps
    const b_blocks: [*]kt.gemm_q8_0.BlockQ8_0 = @ptrCast(@alignCast(&b));
    kt.kt_quantize_q8_0(&src, b_blocks, K);
    kt.kt_quantize_q8_0(&src, b_blocks + 1, K);

    var c: [M * N]f32 = undefined;
    kt.kt_matmul_q8_0(&a, b_blocks, &c, M, N, K, K, 1, N);

    // Constant 1.0 weights x all-1.0 activations => K * (1.0 +- eps)
    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(@as(f32, 32.0), c[i * N + j], 0.5);
        }
    }
}

test "GGML C API: matmul_q4_k constant weights" {
    const M = 4;
    const N = 2;
    const K = 256; // one Q4_K super-block

    const amx = kt.amx;
    var a: [M * K]kt.amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    var src: [K]f32 = undefined;
    for (&src) |*v| v.* = 1.0;
    var b: [N * 144]u8 align(@alignOf(kt.gemm_q4_k.BlockQ4_K)) = undefined;
    const b_blocks: [*]kt.gemm_q4_k.BlockQ4_K = @ptrCast(@alignCast(&b));
    kt.kt_quantize_q4_k(&src, b_blocks, K);
    kt.kt_quantize_q4_k(&src, b_blocks + 1, K);

    var c: [M * N]f32 = undefined;
    kt.kt_matmul_q4_k(&a, b_blocks, &c, M, N, K, K, 1, N);

    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(@as(f32, 256.0), c[i * N + j], 1.5);
        }
    }
}

test "GGML C API: matmul_q6_k constant weights" {
    const M = 4;
    const N = 2;
    const K = 256;

    const amx = kt.amx;
    var a: [M * K]kt.amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    var src: [K]f32 = undefined;
    for (&src) |*v| v.* = 1.0;
    var b: [N * 210]u8 align(@alignOf(kt.gemm_q6_k.BlockQ6_K)) = undefined;
    const b_blocks: [*]kt.gemm_q6_k.BlockQ6_K = @ptrCast(@alignCast(&b));
    kt.kt_quantize_q6_k(&src, b_blocks, K);
    kt.kt_quantize_q6_k(&src, b_blocks + 1, K);

    var c: [M * N]f32 = undefined;
    kt.kt_matmul_q6_k(&a, b_blocks, &c, M, N, K, K, 1, N);

    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(@as(f32, 256.0), c[i * N + j], 1.5);
        }
    }
}

test "GGML Q8_K C API: quantize/dequantize row round trip" {
    const k = 256;
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5 + 0.1 * @as(f32, @floatFromInt(i % 7));
    }

    var dst: [292]u8 align(@alignOf(kt.gemm_q8_k.BlockQ8_K)) = undefined;
    kt.kt_quantize_q8_k(&src, &dst, k);

    var back: [k]f32 = undefined;
    kt.kt_dequantize_q8_k(&dst, &back, k);

    var max_abs_err: f32 = 0;
    var sum_abs_x: f32 = 0;
    for (0..k) |i| {
        max_abs_err = @max(max_abs_err, @abs(src[i] - back[i]));
        sum_abs_x += @abs(src[i]);
    }
    try testing.expect(max_abs_err / (sum_abs_x / k) < 0.02);
}

test "GGML C API: matmul_q8_k constant weights" {
    const M = 4;
    const N = 2;
    const K = 256;

    const amx = kt.amx;
    var a: [M * K]kt.amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    var src: [K]f32 = undefined;
    for (&src) |*v| v.* = 1.0;
    var b: [N * 292]u8 align(@alignOf(kt.gemm_q8_k.BlockQ8_K)) = undefined;
    const b_blocks: [*]kt.gemm_q8_k.BlockQ8_K = @ptrCast(@alignCast(&b));
    kt.kt_quantize_q8_k(&src, b_blocks, K);
    kt.kt_quantize_q8_k(&src, b_blocks + 1, K);

    var c: [M * N]f32 = undefined;
    kt.kt_matmul_q8_k(&a, b_blocks, &c, M, N, K, K, 1, N);

    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(@as(f32, 256.0), c[i * N + j], 1.0);
        }
    }
}
