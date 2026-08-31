// GEMM Kernel 224 Q8_K (GGML K-quant) for AMX/AVX512
//
// Canonical GGUF layout (llama.cpp ggml-common.h, QK_K = 256):
//   block_q8_K {
//     float   d;              // super-block scale — NOTE: f32, not f16!
//     int8_t  qs[QK_K];       // 256 quants
//     int16_t bsums[QK_K/16]; // sum of quants in groups of 16 (dot-product
//                             // sidecar; the scalar path here keeps it exact)
//   } = 292 bytes per 256 weights
//
// Reference math (ggml-quants.c):
//   quantize (:2768): amax over the super-block; iscale = -127/max (signed —
//     max carries its sign); qs = MIN(127, nearest_int(iscale*x));
//     bsums[j] = sum of qs over each 16-group; d = 1/iscale.
//     All-zero fast path: d = 0, qs = 0.
//   dequantize (:2807): y = d * qs[j].
//
// Note: |iscale*x| <= 127 by construction (|x| <= |max|), so MIN(127, v)
// needs no lower clamp — the C int8_t cast is always in range.
//
// Final format of the GGML-quant workstream (after Q8_0, Q4_K, Q5_K, Q6_K).
// C-API wrappers live in main.zig (kt_quantize/dequantize/matmul_q8_k).

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// Q8_K Block Structure (byte-exact vs ggml's block_q8_K)
// ============================================================================

pub const QK_K: usize = 256;
const GROUP: usize = 16; // bsums group size
const N_GROUPS: usize = QK_K / GROUP; // = 16

pub const BlockQ8_K = extern struct {
    d: f32, // super-block scale (f32 — unlike the f16 of the other K-quants)
    qs: [QK_K]i8, // quants
    bsums: [N_GROUPS]i16, // per-16-group quant sums (dot-product sidecar)
};

comptime {
    if (@sizeOf(BlockQ8_K) != 292) @compileError("BlockQ8_K must be 292 bytes");
}

// ============================================================================
// ggml helpers (ported exactly)
// ============================================================================

pub fn nearestInt(fval: f32) i32 {
    std.debug.assert(@abs(fval) <= 4194303.0);
    const val: f32 = fval + 12582912.0;
    const i: i32 = @bitCast(val);
    return (i & 0x007f_ffff) - 0x0040_0000;
}

// ============================================================================
// Quantize / Dequantize (byte-exact vs ggml-quants.c)
// ============================================================================

/// ggml-quants.c:2768 quantize_row_q8_K_ref. k must be a multiple of 256.
pub fn quantizeRowQ8_K(src: []const f32, dst: [*]BlockQ8_K, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    for (0..nb) |i| {
        const x = src[i * QK_K ..][0..QK_K];
        const blk = &dst[i];

        var max: f32 = 0;
        var amax: f32 = 0;
        for (0..QK_K) |j| {
            const ax = @abs(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (amax == 0) {
            // All-zero super-block: d = 0, qs = 0 (bsums also zero)
            blk.* = std.mem.zeroes(BlockQ8_K);
            blk.d = 0;
            continue;
        }
        // Signed iscale (max carries its sign) — the IQ2_XXS-compat choice
        const iscale = -127.0 / max;
        for (0..QK_K) |j| {
            const v = nearestInt(iscale * x[j]);
            // |v| <= 127 by construction; MIN guards the +0.5 rounding edge
            blk.qs[j] = @intCast(@min(127, v));
        }
        for (0..N_GROUPS) |j| {
            var sum: i32 = 0;
            for (0..GROUP) |ii| {
                sum += blk.qs[j * GROUP + ii];
            }
            blk.bsums[j] = @intCast(sum);
        }
        blk.d = 1.0 / iscale;
    }
}

/// ggml-quants.c:2807 dequantize_row_q8_K. k must be a multiple of 256.
pub fn dequantizeRowQ8_K(src: [*]const BlockQ8_K, dst: []f32, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var out: usize = 0;
    for (0..nb) |i| {
        const blk = &src[i];
        for (0..QK_K) |j| {
            dst[out + j] = blk.d * @as(f32, @floatFromInt(blk.qs[j]));
        }
        out += QK_K;
    }
}

/// Dequantize Q8_K to BF16 (for BF16 GEMM tile paths).
pub fn dequantizeRowQ8_KToBF16(src: [*]const BlockQ8_K, dst: [*]amx.bf16, k: usize) void {
    std.debug.assert(k % QK_K == 0);
    const nb = k / QK_K;
    var f: [QK_K]f32 = undefined;
    for (0..nb) |i| {
        dequantizeRowQ8_K(src + i, f[0..], QK_K);
        for (0..QK_K) |j| dst[i * QK_K + j] = amx.f32_to_bf16(f[j]);
    }
}

// ============================================================================
// Scalar GEMM (BF16 activations x Q8_K weights -> F32)
// ============================================================================

pub fn gemmQ8_KScalar(
    a: [*]const amx.bf16,
    lda: usize,
    b: [*]const BlockQ8_K,
    ldb: usize,
    c: [*]f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| c[i * ldc + j] = 0;
    }
    const kb = k / QK_K;
    var bq: [QK_K]f32 = undefined;
    for (0..n) |j| {
        const src = b + j * ldb;
        for (0..kb) |blk| {
            dequantizeRowQ8_K(src + blk, bq[0..], QK_K);
            for (0..m) |i| {
                var acc = c[i * ldc + j];
                const a_row = a + i * lda;
                for (0..QK_K) |t| {
                    acc += amx.bf16_to_f32(a_row[blk * QK_K + t]) * bq[t];
                }
                c[i * ldc + j] = acc;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "Q8_K block layout byte-exact (292 bytes)" {
    try testing.expectEqual(@as(usize, 292), @sizeOf(BlockQ8_K));
    const blk = std.mem.zeroes(BlockQ8_K);
    const bytes: [*]const u8 = @ptrCast(&blk);
    // d [0,4), qs [4,260), bsums [260,292)
    try testing.expectEqual(@as(u8, 0), bytes[0]);
    try testing.expectEqual(@as(u8, 0), bytes[3]);
    try testing.expectEqual(@as(u8, 0), bytes[4]);
    try testing.expectEqual(@as(u8, 0), bytes[259]);
    try testing.expectEqual(@as(u8, 0), bytes[260]);
    try testing.expectEqual(@as(u8, 0), bytes[291]);
}

test "Q8_K quantize/dequantize round trip accuracy" {
    const k = QK_K; // one super-block
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.05) * 0.5 + 0.1 * @as(f32, @floatFromInt(i % 7));
    }

    var blk: [1]BlockQ8_K = undefined;
    quantizeRowQ8_K(&src, &blk, k);

    var dst: [k]f32 = undefined;
    dequantizeRowQ8_K(&blk, &dst, k);

    // ~8.5 bits/weight: the tightest K-quant; error <= d/2 with d = |max|/127
    var max_abs_err: f32 = 0;
    var sum_abs_x: f32 = 0;
    var abs_max: f32 = 0;
    for (0..k) |i| {
        max_abs_err = @max(max_abs_err, @abs(src[i] - dst[i]));
        sum_abs_x += @abs(src[i]);
        abs_max = @max(abs_max, @abs(src[i]));
    }
    const d = abs_max / 127.0;
    // Round-trip error bounded by the quant step half-width
    try testing.expect(max_abs_err <= d * 0.51);
    const rel = max_abs_err / (sum_abs_x / k);
    try testing.expect(rel < 0.02);
}

test "Q8_K bsums are exact group sums" {
    const k = QK_K;
    var src: [k]f32 = undefined;
    for (0..k) |i| {
        src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.37;
    }

    var blk: [1]BlockQ8_K = undefined;
    quantizeRowQ8_K(&src, &blk, k);

    // Each bsums entry must equal the sum of its 16 quants exactly
    for (0..N_GROUPS) |j| {
        var sum: i32 = 0;
        for (0..16) |ii| {
            sum += blk[0].qs[j * 16 + ii];
        }
        try testing.expectEqual(@as(i32, blk[0].bsums[j]), sum);
    }
}

test "Q8_K all-zero super-block fast path" {
    const k = QK_K;
    var src: [k]f32 = [_]f32{0.0} ** k;
    var blk: [1]BlockQ8_K = undefined;
    quantizeRowQ8_K(&src, &blk, k);
    try testing.expectEqual(@as(f32, 0), blk[0].d);

    var dst: [k]f32 = undefined;
    dequantizeRowQ8_K(&blk, &dst, k);
    for (dst) |v| try testing.expectEqual(@as(f32, 0), v);
}

test "Q8_K scalar GEMM vs dequantized reference" {
    const M = 4;
    const N = 4;
    const K = QK_K;

    var a: [M * K]amx.bf16 = undefined;
    for (&a) |*v| v.* = amx.f32_to_bf16(1.0);

    var src: [K]f32 = undefined;
    for (&src) |*v| v.* = 1.0;
    var b: [N]BlockQ8_K = undefined;
    for (0..N) |j| quantizeRowQ8_K(&src, @ptrCast(&b[j]), K);

    var c: [M * N]f32 = undefined;
    gemmQ8_KScalar(&a, K, &b, 1, &c, N, M, N, K);

    var w: [K]f32 = undefined;
    dequantizeRowQ8_K(@ptrCast(&b[0]), &w, K);
    var expected: f32 = 0;
    for (w) |v| expected += v;
    for (0..M) |i| {
        for (0..N) |j| {
            try testing.expectApproxEqAbs(expected, c[i * N + j], 1e-3);
        }
    }
    // Constant 1.0 row quantizes very tightly (d = 1/127 * ... exact)
    try testing.expect(@abs(expected - 256.0) / 256.0 < 0.005);
}
