// GEMM Kernel 224 Q8_0 (GGML quantization) for AMX/AVX512
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK8_0 = 32):
//   block_q8_0 { ggml_half d; int8_t qs[32]; }  — 34 bytes per 32 weights.
//
// Quantize (ggml-quants.c:276 quantize_row_q8_0_ref):
//   d = amax / 127;  qs[j] = round(x[j] / d)
// Dequantize (ggml-quants.c:553 dequantize_row_q8_0):
//   y[j] = qs[j] * d
//
// The scale `d` is IEEE half (f16), NOT bf16 — self-contained conversions are
// included below (amx.zig only provides bf16 helpers).
//
// This file is Phase 1 of the GGML-quant workstream (kernel layer + standalone
// tests only). C-API wiring / root.zig emission is Phase 2.

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// Q8_0 Block Structure (byte-exact vs ggml's block_q8_0)
// ============================================================================

pub const QK8_0: usize = 32;

pub const BlockQ8_0 = extern struct {
    d: u16, // f16 scale (IEEE half, NOT bf16)
    qs: [32]i8, // quants
};

comptime {
    if (@sizeOf(BlockQ8_0) != 34) @compileError("BlockQ8_0 must be 34 bytes (2 f16 + 32 i8)");
}

// ============================================================================
// f16 <-> f32 conversion (IEEE 754 half).
// Zig has a native f16 type with correct IEEE round-to-nearest-even
// semantics for @floatCast, and @bitCast for the u16 storage form —
// matching GGML_FP32_TO_FP16 / GGML_FP16_TO_FP32 exactly (both produce
// IEEE-754 binary16).

/// Convert f32 to IEEE f16 bits (u16 storage form), matching GGML_FP32_TO_FP16.
pub fn f32_to_f16(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

/// Convert IEEE f16 bits (u16 storage form) to f32, matching GGML_FP16_TO_FP32.
pub fn f16_to_f32(x: u16) f32 {
    const h: f16 = @bitCast(x);
    return @floatCast(h);
}

// ============================================================================
// Quantize / Dequantize (byte-exact math vs ggml-quants.c)
// ============================================================================

/// Quantize f32 row to Q8_0 blocks (ggml: d = amax/127, qs = round(x/d)).
/// k must be a multiple of 32. dst has k/32 blocks.
pub fn quantizeRowQ8_0(src: []const f32, dst: [*]BlockQ8_0, k: usize) void {
    std.debug.assert(k % QK8_0 == 0);
    const nb = k / QK8_0;
    for (0..nb) |i| {
        var amax: f32 = 0.0;
        for (0..QK8_0) |j| {
            const v = src[i * QK8_0 + j];
            amax = @max(amax, @abs(v));
        }
        const d = amax / 127.0;
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;
        dst[i].d = f32_to_f16(d);
        for (0..QK8_0) |j| {
            const x0 = src[i * QK8_0 + j] * id;
            const q = @round(x0);
            // ggml stores via int8_t cast; clamp to the i8 range for safety
            const qi: i32 = @intFromFloat(std.math.clamp(q, -127.0, 127.0));
            dst[i].qs[j] = @intCast(qi);
        }
    }
}

/// Dequantize Q8_0 blocks to f32 (ggml: y[j] = qs[j] * d).
pub fn dequantizeRowQ8_0(src: [*]const BlockQ8_0, dst: []f32, k: usize) void {
    std.debug.assert(k % QK8_0 == 0);
    const nb = k / QK8_0;
    for (0..nb) |i| {
        const d = f16_to_f32(src[i].d);
        for (0..QK8_0) |j| {
            dst[i * QK8_0 + j] = @as(f32, @floatFromInt(src[i].qs[j])) * d;
        }
    }
}

/// Dequantize Q8_0 blocks to BF16 (for feeding BF16 GEMM tiles).
pub fn dequantizeRowQ8_0ToBF16(src: [*]const BlockQ8_0, dst: [*]amx.bf16, k: usize) void {
    std.debug.assert(k % QK8_0 == 0);
    const nb = k / QK8_0;
    for (0..nb) |i| {
        const d = f16_to_f32(src[i].d);
        for (0..QK8_0) |j| {
            dst[i * QK8_0 + j] = amx.f32_to_bf16(@as(f32, @floatFromInt(src[i].qs[j])) * d);
        }
    }
}

// ============================================================================
// GEMM (scalar reference; BF16 activations x Q8_0 weights -> F32)
// ============================================================================

/// Scalar GEMM: a [m, k] BF16 activations, b [n, k/32] Q8_0 weights
/// (row-major, ldb in BLOCKS), c [m, n] F32.
/// On-the-fly dequant: each B element becomes qs * d (f32) before the MAC.
pub fn gemmQ8_0Scalar(
    a: [*]const amx.bf16,
    lda: usize,
    b: [*]const BlockQ8_0,
    ldb: usize,
    c: [*]f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) void {
    // Zero output
    for (0..m) |i| {
        for (0..n) |j| c[i * ldc + j] = 0;
    }
    const kb = k / QK8_0;
    for (0..m) |i| {
        const a_row = a + i * lda;
        for (0..n) |j| {
            const b_row = b + j * ldb;
            var acc: f32 = 0;
            for (0..kb) |blk| {
                const d = f16_to_f32(b_row[blk].d);
                const qs = &b_row[blk].qs;
                const a_blk = a_row + blk * QK8_0;
                for (0..QK8_0) |t| {
                    acc += amx.bf16_to_f32(a_blk[t]) * (@as(f32, @floatFromInt(qs[t])) * d);
                }
            }
            c[i * ldc + j] = acc;
        }
    }
}

// ============================================================================
// Tests (standalone: zig test src/kernels/amx/gemm_224_q8_0.zig)
// ============================================================================

const testing = std.testing;

test "f16 <-> f32 round trip known values" {
    // 1.0 = 0x3C00, 0.5 = 0x3800, -2.0 = 0xC000, 65504 (max) = 0x7BFF
    try testing.expectEqual(@as(u16, 0x3C00), f32_to_f16(1.0));
    try testing.expectEqual(@as(u16, 0x3800), f32_to_f16(0.5));
    try testing.expectEqual(@as(u16, 0xC000), f32_to_f16(-2.0));
    try testing.expectEqual(@as(u16, 0x7BFF), f32_to_f16(65504.0));
    try testing.expectEqual(@as(f32, 1.0), f16_to_f32(0x3C00));
    try testing.expectEqual(@as(f32, -2.0), f16_to_f32(0xC000));
    // Inf / zero
    try testing.expectEqual(@as(u16, 0x7C00), f32_to_f16(std.math.inf(f32)));
    try testing.expectEqual(@as(u16, 0x0000), f32_to_f16(0.0));
    // Subnormal f16: 2^-24 = 0x0001
    try testing.expectEqual(@as(f32, 0x1.0p-24), f16_to_f32(0x0001));
}

test "Q8_0 quantize/dequantize round trip exact for integers" {
    // Integer-valued inputs with amax <= 127 round-trip exactly
    // (d = amax/127, qs = round(x/d), y = qs*d reproduces x when x is a
    //  multiple of d — guaranteed for small integers by construction below).
    const k = 64;
    var src: [k]f32 = undefined;
    for (0..k) |i| src[i] = @floatFromInt(@as(i32, @intCast(i % 17)) - 8); // -8..8

    var blocks: [k / QK8_0]BlockQ8_0 = undefined;
    quantizeRowQ8_0(&src, &blocks, k);

    var dst: [k]f32 = undefined;
    dequantizeRowQ8_0(&blocks, &dst, k);

    // Small integers quantize exactly (d = 8/127; qs = x*127/8 is integer
    // for x multiples of... verify: x=1 -> qs = round(127/8) = 16 -> y = 16*8/127 = 1.008
    // NOT exact — so use tolerance based on max quantization error = d/2.
    const amax: f32 = 8.0;
    const d = amax / 127.0;
    for (0..k) |i| {
        try testing.expectApproxEqAbs(src[i], dst[i], d * 0.51);
    }
}

test "Q8_0 block layout is byte-exact (34 bytes)" {
    try testing.expectEqual(@as(usize, 34), @sizeOf(BlockQ8_0));
    // d at offset 0, qs at offset 2
    const blk = BlockQ8_0{ .d = 0x3C00, .qs = [_]i8{1} ** 32 };
    const bytes: [*]const u8 = @ptrCast(&blk);
    try testing.expectEqual(@as(u8, 0x00), bytes[0]); // f16 1.0 little-endian
    try testing.expectEqual(@as(u8, 0x3C), bytes[1]);
    try testing.expectEqual(@as(u8, 1), bytes[2]); // first quant
    try testing.expectEqual(@as(u8, 1), bytes[33]); // last quant
}

test "Q8_0 scalar GEMM exact values" {
    // M=16, N=16, K=32 (one block). A all 1.0; B all qs=1 with d=1.0
    // => C[i][j] = sum over 32 of 1.0*1.0 = 32.
    const M = 16;
    const N = 16;
    const K = 32;

    var a: [M * K]amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    var b: [N]BlockQ8_0 = undefined;
    for (&b) |*blk| {
        blk.d = f32_to_f16(1.0);
        for (&blk.qs) |*q| q.* = 1;
    }

    var c: [M * N]f32 = undefined;
    gemmQ8_0Scalar(&a, K, &b, 1, &c, N, M, N, K);

    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(@as(f32, 32.0), c[i * N + j], 1e-4);
        }
    }
}

test "Q8_0 scalar GEMM with scales" {
    // d = 2.0, qs = 3 => weight = 6.0; a = 2.0 => per-token product 12, K=32 => 384
    const M = 4;
    const N = 4;
    const K = 32;

    var a: [M * K]amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(2.0);

    var b: [N]BlockQ8_0 = undefined;
    for (&b) |*blk| {
        blk.d = f32_to_f16(2.0);
        for (&blk.qs) |*q| q.* = 3;
    }

    var c: [M * N]f32 = undefined;
    gemmQ8_0Scalar(&a, K, &b, 1, &c, N, M, N, K);

    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(@as(f32, 384.0), c[i * N + j], 1e-3);
        }
    }
}
