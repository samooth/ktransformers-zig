// Shared AVX2 primitives for the GGML K-quant BF16 x quant GEMMs.
//
// WHY THIS EXISTS. `gemmQ{2,3,4,5,6,8}_KScalar` in the per-format kernel
// files is a dequantize-to-F32 + scalar dot product: per weight it costs a
// bf16 widen, an integer->float convert, a scale multiply-add and the loop
// bookkeeping — measured at 0.3-0.8 GB/s against the 7.9 GB/s the vectorized
// BF16 kernel sustains on the same host and shape (docs/perf-roadmap.md,
// "Measured headroom, per path"). The kernels are arithmetic-bound, not
// bandwidth-bound, so the fix is SIMD and nothing else.
//
// The shape every K-quant format shares is: a super-block of 256 weights,
// divided into groups that share one scale (`scales[16]`, `scales[8]` packed
// 6-bit, per-16 or per-32), reconstructed as `y = s*q` or `y = s*q - m`. So
// this module provides ONE driver (`gemmBF16xQuant`) parameterized by a
// comptime `Cfg` that describes the per-format scale/unpack step. The five
// per-format files each supply ~20 lines of `prepare`/`unpackVec` (their
// actual bit layout) and a 5-line `gemmQ*_K` entry point.
//
// DESIGN NOTES (all four are measured/verified, not folklore):
//
// 1. LOOP ORDER: n-outer / m-inner, so one weight row (the huge operand)
//    streams from DRAM once instead of once per token row. This was worth
//    3.4-4.0x on the BF16 GEMM — an order of magnitude more than the
//    accumulator count or the FMA.
//
// 2. REGISTER TILING over M (MR = 4 rows of activations in flight). bf16 ->
//    f32 is 2 uops per 8 elements and the quant unpack is ~10 uops per 16,
//    so with MR=4 each unpacked weight chunk feeds four accumulators and the
//    conversion cost per MAC drops ~4x.
//
// 3. FOLD THE SCALE/MIN INTO THE WEIGHT, not into the accumulator. Two
//    formulations are numerically equivalent:
//        (a) acc += a*q      for the whole k, then scale per group
//        (b) acc += a*(s*q - m)          <- this module
//    (a) needs a horizontal reduction per group per row (~6 uops), which at
//    Q6_K/Q3_K/Q2_K granularity (one group = 16 weights) costs MORE than the
//    single broadcast-multiply that (b) needs. (b) also keeps a single
//    accumulator register across the whole k, so there is exactly one
//    `@reduce` per OUTPUT element rather than per group. (b) is what the
//    scalar `dequantizeRow*` + dot reference does, so the vector path is
//    bit-comparable with it (see each format's GEMM-vs-dequant test).
//
// 4. `prepare` runs ONCE per super-block and precomputes the per-group
//    scale/min table (including the f16 -> f32 conversions of the
//    super-block scales). Without it, Q6_K/Q3_K/Q2_K re-do an f16 conversion
//    + an int convert + a multiply per 16 weights, which measured ~30% of the
//    inner-loop uop budget. The table is 64 bytes: L1-resident.
//
// `@exp` never appears here (rule 4 of docs/perf-roadmap.md): there is no
// transcendental in a quantized dot product.

const std = @import("std");
const amx = @import("../arch/amx.zig");

// ============================================================================
// Vector primitives
// ============================================================================

/// Lanes per vector. 16 f32 = 2 ymm registers, which is the width the
/// existing Q4_K AVX2 path (`gemm_224_q4_k.zig::q4kGroupDots`) uses; going
/// wider buys nothing on AVX2 (it would still split into ymm pairs) and
/// narrower doubles the loop bookkeeping per MAC.
pub const VW: usize = 16;
pub const Vec = @Vector(VW, f32);
pub const VecU8 = @Vector(VW, u8);
pub const VecI8 = @Vector(VW, i8);

/// Rows of activations processed per pass. With 4 rows in flight, one
/// unpacked weight vector feeds four accumulators (see note 2 above).
pub const MR: usize = 4;

/// Shift-count vectors for the byte-level bit twiddling every K-quant unpack
/// needs. Zig 0.16 wants the shift amount as a vector of the SHIFT-LOGIC
/// element type (`u3` for `u8`), typed even for comptime counts — so these
/// take a comptime count and hand back the right vector.
pub inline fn shl8(comptime sh: comptime_int) @Vector(VW, u3) {
    return @splat(@as(u3, @intCast(sh)));
}
pub inline fn shr8(comptime sh: comptime_int) @Vector(VW, u3) {
    return @splat(@as(u3, @intCast(sh)));
}

/// bf16 -> f32 for `VW` consecutive values.
///
/// bf16 is the top 16 bits of the f32, so the widening is a zero-extend to
/// u32 plus a shift — which is exactly what LLVM lowers this to
/// (`vpmovzxwd` + `vpslld $16`). F16C is irrelevant here: it converts IEEE
/// f16, and AVX2 has no bf16 conversion instruction at all.
pub inline fn widen16(p: [*]const amx.bf16) Vec {
    const u16x8 = @Vector(8, u16);
    const raw0: u16x8 = @as(*align(1) const u16x8, @ptrCast(p)).*;
    const raw1: u16x8 = @as(*align(1) const u16x8, @ptrCast(p + 8)).*;
    const w0: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), raw0) << @as(@Vector(8, u5), @splat(16)));
    const w1: @Vector(8, f32) = @bitCast(@as(@Vector(8, u32), raw1) << @as(@Vector(8, u5), @splat(16)));
    return std.simd.join(w0, w1);
}

/// Unaligned 16-byte loads. K-quant blocks are packed at strides that are not
/// multiples of 16 (Q8_K 292, Q4_K 144, Q6_K 210, Q5_K 176, Q2_K 84, Q3_K
/// 110) and their quants start at odd offsets inside the block, so EVERY
/// load has to be `align(1)`.
/// NOTE the single-pointer parameter: `&blk.field[i]` inside an extern struct
/// at a non-16-aligned offset has type `*align(1) const u8`, which does NOT
/// coerce to `[*]const u8` (a single pointer cannot become a many pointer).
pub inline fn loadU8(p: *const u8) VecU8 {
    return @as(*align(1) const VecU8, @ptrCast(p)).*;
}
pub inline fn loadI8(p: *const i8) VecI8 {
    return @as(*align(1) const VecI8, @ptrCast(p)).*;
}

/// Zero-extend 16 bytes to 16 f32 (vpmovzxbd + vcvtdq2ps).
pub inline fn u8ToF32(v: VecU8) Vec {
    return @floatFromInt(v);
}
/// Sign-extend 16 i8 to 16 f32 (vpmovsxbw + 2x vcvtdq2ps).
pub inline fn i8ToF32(v: VecI8) Vec {
    return @floatFromInt(v);
}

pub inline fn splatF(x: f32) Vec {
    return @splat(x);
}

/// The two masks every K-quant unpack needs: 4-bit fields (nibbles) and 2-bit
/// fields (Q2_K/Q3_K/Q6_K planes).
pub inline fn mask4() VecU8 {
    return @splat(@as(u8, 0x0F));
}
pub inline fn mask3() VecU8 {
    return @splat(@as(u8, 0x03));
}

// ============================================================================
// The generic driver
// ============================================================================

/// BF16 activations x K-quant weights -> F32 output, vectorized.
///
/// `Cfg` is a comptime description of one GGML format's block layout:
///
///     pub const Cfg = struct {
///         pub const Block: type;      // the extern struct (byte-exact vs ggml)
///         pub const QK: usize;        // weights per super-block (256)
///         pub const GROUP: usize;     // weights sharing one scale (16 / 32 / 256)
///         pub const Scales: type;     // per-super-block scale table
///         pub fn prepare(blk: *const Block) Scales;
///         // `g` and `v` are COMPTIME (see the note on the unrolled
///         // super-block below) — every format derives its byte offsets from
///         // them, so they must fold into load displacements.
///         pub inline fn unpackVec(blk: *const Block, sc: *const Scales, comptime g: usize, comptime v: usize) Vec;
///     };
///
/// Semantics are identical to the per-format `gemmQ*_KScalar`: `c` is
/// OVERWRITTEN (it is written for every (i, j) with no dependence on the old
/// contents), `lda`/`ldb`/`ldc` are strides in activation elements /
/// blocks / output elements respectively, and `k` must be a multiple of `QK`.
pub fn gemmBF16xQuant(
    comptime Cfg: type,
    a: [*]const amx.bf16,
    lda: usize,
    b: [*]const Cfg.Block,
    ldb: usize,
    c: [*]f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) void {
    const QK = Cfg.QK;
    const GROUP = Cfg.GROUP;
    const VECS: usize = GROUP / VW; // vector chunks per scale group
    const N_GROUP: usize = QK / GROUP;

    if (m == 0 or n == 0) return;
    std.debug.assert(k % QK == 0);
    std.debug.assert(GROUP % VW == 0);

    const kb = k / QK;
    const zero: Vec = @splat(0.0);

    var j: usize = 0;
    while (j < n) : (j += 1) {
        const brow = b + j * ldb;

        var i: usize = 0;
        // Main path: MR rows of A share every unpacked weight vector.
        while (i + MR <= m) : (i += MR) {
            var acc = [_]Vec{zero} ** MR;
            var sb: usize = 0;
            while (sb < kb) : (sb += 1) {
                const blk = &brow[sb];
                const scales = Cfg.prepare(blk);
                // COMPTIME group index. The super-block is fully unrolled: a
                // runtime `g` costs ~20 scalar ops of index arithmetic per
                // 16-weight group (decomposing g into chunk/plane/half) plus a
                // 4-way indirect jump for the plane switch — measured at
                // roughly half the inner-loop uop budget for Q6_K/Q3_K/Q2_K.
                // With a comptime `g` the byte offsets fold into the load
                // displacement. Cost: ~16 specializations of the unpack per
                // format (~2 KB of code), which is what every quantized kernel
                // does anyway.
                inline for (0..N_GROUP) |g| {
                    inline for (0..VECS) |v| {
                        const w = Cfg.unpackVec(blk, &scales, g, v);
                        const kbase = sb * QK + g * GROUP + v * VW;
                        inline for (0..MR) |r| {
                            acc[r] = @mulAdd(Vec, widen16(a + (i + r) * lda + kbase), w, acc[r]);
                        }
                    }
                }
            }
            inline for (0..MR) |r| {
                c[(i + r) * ldc + j] = @reduce(.Add, acc[r]);
            }
        }
        // Row tail (m % MR), one row at a time.
        while (i < m) : (i += 1) {
            var acc: Vec = zero;
            var sb: usize = 0;
            while (sb < kb) : (sb += 1) {
                const blk = &brow[sb];
                const scales = Cfg.prepare(blk);
                inline for (0..N_GROUP) |g| {
                    inline for (0..VECS) |v| {
                        const w = Cfg.unpackVec(blk, &scales, g, v);
                        const kbase = sb * QK + g * GROUP + v * VW;
                        acc = @mulAdd(Vec, widen16(a + i * lda + kbase), w, acc);
                    }
                }
            }
            c[i * ldc + j] = @reduce(.Add, acc);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================
//
// These are the readable statement of what each primitive promises. NOTE they
// are NOT executed by `zig build test`: this file belongs to the `kt` module,
// and Zig collects tests from the root file's module only (see LESSONS_ZIG.md
// §8). The LIVE coverage of every primitive below is the six
// `K-quant AVX2 GEMMs == dequantize + dot` tests in
// tests/kernels/simd_kernels_tests.zig, which drive all of them (widen16,
// loadU8/loadI8, the unrolled driver, every format's prepare/unpackVec)
// through the public `gemmQ*_K` entry points on fuzzed scale data — a broken
// `widen16` fails all six.

const testing = std.testing;

test "kquant_simd: widen16 is bf16 -> f32 for 16 lanes" {
    var raw: [16]amx.bf16 = undefined;
    var expect: [16]f32 = undefined;
    for (0..16) |l| {
        const x: f32 = @as(f32, @floatFromInt(l)) * 0.25 - 1.0;
        raw[l] = amx.f32_to_bf16(x);
        expect[l] = amx.bf16_to_f32(raw[l]);
    }
    const got: [16]f32 = widen16(&raw);
    for (0..16) |l| try testing.expectEqual(expect[l], got[l]);
}

test "kquant_simd: unaligned loadU8 works at every byte offset" {
    var buf: [64]u8 = undefined;
    for (&buf, 0..) |*e, idx| e.* = @truncate(idx * 7 + 3);
    for (0..buf.len - VW + 1) |off| {
        const got: [VW]u8 = loadU8(&buf[off]);
        for (0..VW) |l| try testing.expectEqual(buf[off + l], got[l]);
    }
}

test "kquant_simd: scalar GEMM equivalence on a hand-built block" {
    // A toy Cfg: 256 weights as plain 4-bit nibbles with a per-32 scale, which
    // is Q4_K's shape. The point of the test is the DRIVER (stride handling, the
    // m tail, one reduction per output), not the format.
    // 32 bytes per 32-weight group: the low nibble of byte (g*32 + t%32) is
    // weight t, the high nibble is a decoy the mask must discard.
    const ToyBlock = extern struct { s: [256]u8 };
    const ToyCfg = struct {
        pub const Block = ToyBlock;
        pub const QK: usize = 256;
        pub const GROUP: usize = 32;
        pub const Scales = [8]f32;
        pub fn prepare(blk: *const ToyBlock) Scales {
            var t: [8]f32 = undefined;
            for (0..8) |i| t[i] = @floatFromInt(blk.s[i]);
            return t;
        }
        // Lane l of (g, v) is weight g*32 + v*16 + l: consecutive, NOT
        // interleaved. Every lane takes a LOW nibble, so the high nibbles stay
        // decoys for the mask to discard.
        pub fn unpackVec(blk: *const ToyBlock, sc: *const Scales, g: usize, v: usize) Vec {
            const b: VecU8 = loadU8(&blk.s[g * 32 + v * 16]);
            return @as(Vec, u8ToF32(b & mask4())) * @as(Vec, @splat(sc[g]));
        }
    };

    const M: usize = 5; // deliberately not a multiple of MR
    const N: usize = 3;
    const K: usize = ToyCfg.QK;

    var a: [M * K]amx.bf16 = undefined;
    for (&a, 0..) |*e, idx| {
        e.* = amx.f32_to_bf16(@as(f32, @floatFromInt(@as(i32, @intCast(idx % 13)) - 6)) * 0.125);
    }
    var b: [N]ToyBlock = undefined;
    for (&b, 0..) |*e, row| {
        for (&e.s, 0..) |*byte, idx| byte.* = @truncate(row * 31 + idx * 7 + 1);
    }

    var got: [M * N]f32 = undefined;
    gemmBF16xQuant(ToyCfg, &a, K, &b, 1, &got, N, M, N, K);

    // Reference: explicit dequantize + f32 dot (the shape of the per-format
    // GEMM-vs-dequantize tests).
    for (0..M) |i| {
        for (0..N) |j| {
            var expect: f32 = 0;
            for (0..K) |t| {
                const g = t / 32;
                const scale = @as(f32, @floatFromInt(b[j].s[g]));
                const w = scale * @as(f32, @floatFromInt(b[j].s[g * 32 + t % 32] & 0x0F));
                expect += amx.bf16_to_f32(a[i * K + t]) * w;
            }
            try testing.expectApproxEqRel(expect, got[i * N + j], 1e-5);
        }
    }
}
