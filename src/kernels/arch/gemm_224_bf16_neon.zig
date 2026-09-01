// NEON Phase-2 kernel — BF16 activations × BF16 weights, vectorized for ARMv8.
//
// Phase 1 of the NEON workstream (the other dev) added the feature-
// detection flags in src/kernels/arch/neon.zig. This file is Phase 2:
// the first actual NEON-vectorized kernel, paralleling the A1 work in
// src/kernels/amx/gemm_224_bf16.zig (which vectorizes the K-axis inner
// loop with @Vector(8,f32) for AVX2). Here the lane count is 4 (128-bit
// NEON register = 4 f32 lanes).
//
// Why the @Vector(4,f32) lane count matches NEON: NEON registers are
// 128 bits wide (same as SSE, half of AVX2). Four f32 lanes = 64-byte
// per vector op, fully utilizing the register. On x86_64 (where this
// file also compiles, for the x86 scalar-fallback path), @Vector(4,f32)
// lowers to SSE — narrower than AVX2's 8-lane, but still correct and
// benchmarkable. The aarch64 cross-build lowers to NEON fmla / fmul.
//
// Algorithm (NEON-vectorized, same structure as A1's gemmExpert):
//   for i in 0..M_STEP tiles:
//     for j in 0..N_STEP tiles:
//       k_processed = 0
//       while k_processed + K_STEP <= k:
//         A panel: input[i*M..i*M+M_STEP, k..k+K_STEP] (BF16)
//         B panel: weight[j*N..j*N+N_STEP, k..k+K_STEP] (BF16)
//         C tile:  output[i*M..i*M+M_STEP, j*N..j*N+N_STEP] (F32)
//         for p in 0..M_STEP:
//           vec acc = @Vector(4,f32) splat 0
//           for q in 0..N_STEP step 4:
//             vec a = @Vector(4,f32) of input[p][k+0..k+4] (bf16->f32)
//             vec b = @Vector(4,f32) of weight[q..q+3][k+0..k+4] (bf16->f32)
//             acc += a * b   (NEON fmul + fmla, or SSE mulps on x86)
//           output[p][j*N+q] = sum(acc) + output[p][j*N+q]
//         k_processed += K_STEP
//       handle tail (k_processed..k) with the scalar path (same as A1)
//
// BF16 -> F32 conversion: Zig's `@floatCast(@Vector(4,bf16) -> @Vector(4,f32))`
// lowers to NEON `bfcvt` (NEON BF16->F32 conversion instruction). On x86,
// it lowers to a sequence of `pmovzx` + `punpcklwd` + `pslld` (the
// standard BF16 -> F32 promotion pattern, no inherent AVX2 instruction).
//
// The path is gated on `NeonFeatures.available` (comptime switch) so
// the x86 build of the same .zig file doesn't emit dead NEON code.
// Tests verify the algorithm on x86_64 (where the comptime gate fires
// the x86 path); the aarch64 cross-build (`zig build -Dvariant=neon
// -Dtarget=aarch64-linux-gnu`) validates the NEON path emits correctly.

const std = @import("std");
const builtin = @import("builtin");
const amx = @import("amx.zig");
const neon = @import("neon.zig");

// ============================================================================
// Tile geometry (mirrors the x86 kernel's M_STEP / N_STEP / K_STEP)
// ============================================================================

pub const M_STEP: usize = 32;
pub const N_STEP: usize = 32;
pub const K_STEP: usize = 32;

// On NEON: 4 f32 lanes = 128 bits (full NEON register).
// On x86 SSE: 4 f32 lanes = 128 bits (full SSE register, half of AVX2).
// We use 4 lanes here for portability — the algorithm correctness is
// what we test on x86; the lane utilization on NEON is verified by the
// aarch64 cross-build. (The x86 A1 kernel uses 8 lanes for AVX2; the
// NEON register width forces 4.)
const VEC_F32_LANES: usize = 4;

pub const GemmExpertBF16NEON = struct {
    /// 4-f32 vector type — comptime-dispatched: on aarch64 this lowers
    /// to NEON 128-bit f32x4 ops; on x86_64 to SSE 128-bit ops.
    /// Using @Vector(4, f32) directly in a kernel that is gated on
    /// neon.avail means Zig picks the right ABI on the cross-build.
    pub const VecF32 = @Vector(VEC_F32_LANES, f32);

    /// Horizontal sum of a VecF32 into a scalar. Same role as
    /// amx.reduceAddFp32 but for the NEON lane count (4). Inline
    /// unrolled so the compiler emits a single ADDV instruction on
    /// aarch64 (vs the 8-lane reduce which uses separate adds).
    pub fn reduceAdd(v: VecF32) f32 {
        return v[0] + v[1] + v[2] + v[3];
    }

    pub const dt = amx.bf16;
    pub const output_t = f32;

    /// Scalar reference: exact same math as the vectorized path below,
    /// without the @Vector reduction. Used by tests as the oracle.
    /// Also serves as the x86 fallback for the algorithm.
    pub fn gemmExpertScalar(
        input: [*]const dt, // [m, k] BF16 row-major
        weight: [*]const dt, // [n, k] BF16 row-major
        output: [*]output_t, // [m, n] F32 row-major
        m: usize,
        n: usize,
        k: usize,
        input_ld: usize, // row stride of input in elements
        weight_ld: usize, // row stride of weight in elements
        output_ld: usize, // row stride of output in elements
    ) void {
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                for (0..k) |k0| {
                    sum += amx.bf16_to_f32(input[i * input_ld + k0]) *
                        amx.bf16_to_f32(weight[j * weight_ld + k0]);
                }
                output[i * output_ld + j] = sum;
            }
        }
    }

    /// NEON-vectorized path: 4 f32 lanes per accumulator, K_STEP = 32
    /// means 8 vector loads (32/4) per (i,j) pair. Same outer-loop
    /// structure as the scalar version.
    ///
    /// Works correctly on x86_64 too (Zig lowers @Vector(4,f32) to
    /// SSE 128-bit ops); the same code path runs on both, with the
    /// aarch64 cross-build emitting NEON instructions. This is the
    /// cross-platform-testable surface for the algorithm.
    pub fn gemmExpertNeon(
        input: [*]const dt,
        weight: [*]const dt,
        output: [*]output_t,
        m: usize,
        n: usize,
        k: usize,
        input_ld: usize,
        weight_ld: usize,
        output_ld: usize,
    ) void {
        // The K-axis inner loop is a dot product of length K_STEP=32.
        // We process 4 K-elements per vector load; 8 vectors cover
        // the full K_STEP. Zig 0.16 many-pointer rule (no `y += 4`):
        // we use a per-tile index counter `koff` instead.
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                var koff: usize = 0;
                while (koff + K_STEP <= k) : (koff += K_STEP) {
                    // 8 vectors of 4 lanes per K_STEP=32 chunk.
                    var acc: VecF32 = @splat(0.0);
                    var v: usize = 0;
                    while (v < K_STEP) : (v += VEC_F32_LANES) {
                        // Load 4 BF16 from input row i at koff+v and
                        // 4 BF16 from weight row j at koff+v, then
                        // convert to f32 and FMA. The 4-element shape
                        // of the load is enforced by the array-stage
                        // (see the BF16->f32 note in the file header).
                        //
                        // CRITICAL: `const a_f32: VecF32 = a_bf16` does
                        // NOT do the BF16->f32 bit shift; it does
                        // `@floatCast` semantics (so 0x3F80 = 16256.0,
                        // not 1.0). We must go through amx.bf16_to_f32
                        // explicitly to get the BF16-as-f32-bitshift
                        // semantics, just like the scalar path.
                        // The Zig 0.16 vector-indexing rule requires
                        // comptime lane; we stage through a [N]f32
                        // array (the same pattern applySwiGLU uses) and
                        // let the coercion do the rest.
                        var a_f32: [VEC_F32_LANES]f32 = undefined;
                        var b_f32: [VEC_F32_LANES]f32 = undefined;
                        for (0..VEC_F32_LANES) |lane| {
                            a_f32[lane] = amx.bf16_to_f32(input[i * input_ld + koff + v + lane]);
                            b_f32[lane] = amx.bf16_to_f32(weight[j * weight_ld + koff + v + lane]);
                        }
                        const a_v: VecF32 = a_f32;
                        const b_v: VecF32 = b_f32;
                        acc += a_v * b_v;
                    }
                    // Horizontal sum into a scalar.
                    sum += reduceAdd(acc);
                }
                // Tail (koff < k < koff + K_STEP) — handled by the
                // scalar path; the prod cost is amortized into the
                // tile's overhead.
                while (koff < k) : (koff += 1) {
                    sum += amx.bf16_to_f32(input[i * input_ld + koff]) *
                        amx.bf16_to_f32(weight[j * weight_ld + koff]);
                }
                output[i * output_ld + j] = sum;
            }
        }
    }

    /// Public dispatcher: choose the right path at comptime. On
    /// aarch64 the NEON vectorized path is taken; everywhere else,
    /// the scalar fallback. This is the entry point the MoE / MLA
    /// orchestration code would call.
    pub fn gemmExpert(
        input: [*]const dt,
        weight: [*]const dt,
        output: [*]output_t,
        m: usize,
        n: usize,
        k: usize,
        input_ld: usize,
        weight_ld: usize,
        output_ld: usize,
    ) void {
        if (comptime neon.NeonFeatures.available) {
            gemmExpertNeon(input, weight, output, m, n, k, input_ld, weight_ld, output_ld);
        } else {
            gemmExpertScalar(input, weight, output, m, n, k, input_ld, weight_ld, output_ld);
        }
    }
};
