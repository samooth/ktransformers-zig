// LoRA GEMM kernels for SFT backward pass
// Ported from ktransformers kt-kernel/operators/amx/la/avx_kernels.hpp
//
// The C++ reference uses AVX512_BF16 intrinsics (_mm512_dpbf16_ps) for these.
// The AVX512 inline-asm port is future work (requires AVX512_BF16 hardware to
// test). Until then, the scalar fallback below implements the exact same math
// and is the active path on all hosts. This is sufficient for correctness
// (Dev B's gradient-check test validates the math) and the training loop.
//
// When the AVX512 path is ported, the `pub fn` dispatch will switch on a runtime
// CPU-feature check (see AmxFeatures / detectCpu in arch/amx.zig and
// runtime/cpu_detect.zig) and call the intrinsic version instead.

const amx = @import("../arch/amx.zig");
const bf16 = amx.bf16;

// ============================================================================
// loraBf16MatmulT4r4
// output[t][r] = sum_k(input[t][k] * weight[r][k])
// input: [numTokens, kDim] BF16, weight: [rank, kDim] BF16, out: [numTokens, rank] FP32
// Reference: avx_kernels.hpp:49
// ============================================================================

fn loraBf16MatmulT4r4Avx512(
    input: [*]const bf16,
    weight: [*]const bf16,
    out: [*]f32,
    numTokens: usize,
    kDim: usize,
    rank: usize,
) void {
    _ = input;
    _ = weight;
    _ = out;
    _ = numTokens;
    _ = kDim;
    _ = rank;
    @panic("loraBf16MatmulT4r4 AVX512 path not yet ported — scalar fallback is active");
}

pub fn loraBf16MatmulT4r4(
    input: [*]const bf16,
    weight: [*]const bf16,
    out: [*]f32,
    numTokens: usize,
    kDim: usize,
    rank: usize,
) void {
    for (0..numTokens) |t| {
        for (0..rank) |r| {
            var sum: f32 = 0;
            const in_row = input + t * kDim;
            const w_row = weight + r * kDim;
            for (0..kDim) |k| {
                sum += amx.bf16_to_f32(in_row[k]) * amx.bf16_to_f32(w_row[k]);
            }
            out[t * rank + r] = sum;
        }
    }
    _ = loraBf16MatmulT4r4Avx512;
}

// ============================================================================
// loraFp32Bf16FusedAddTransposed
// out[t][i] += scale * sum_r(intermediate[t][r] * weight[r][i])
// intermediate: [numTokens, rank] FP32, weight: [rank, dim] BF16, out: [numTokens, dim] BF16
// Reference: avx_kernels.hpp:1171
// ============================================================================

fn loraFp32Bf16FusedAddTransposedAvx512(
    intermediate: [*]const f32,
    weight: [*]const bf16,
    out: [*]bf16,
    numTokens: usize,
    rank: usize,
    dim: usize,
    scale: f32,
) void {
    _ = intermediate;
    _ = weight;
    _ = out;
    _ = numTokens;
    _ = rank;
    _ = dim;
    _ = scale;
    @panic("loraFp32Bf16FusedAddTransposed AVX512 path not yet ported — scalar fallback is active");
}

pub fn loraFp32Bf16FusedAddTransposed(
    intermediate: [*]const f32,
    weight: [*]const bf16,
    out: [*]bf16,
    numTokens: usize,
    rank: usize,
    dim: usize,
    scale: f32,
) void {
    for (0..numTokens) |t| {
        for (0..dim) |i| {
            var sum: f32 = 0;
            const inter_row = intermediate + t * rank;
            for (0..rank) |r| {
                sum += inter_row[r] * amx.bf16_to_f32(weight[r * dim + i]);
            }
            const idx = t * dim + i;
            out[idx] = amx.f32_to_bf16(amx.bf16_to_f32(out[idx]) + scale * sum);
        }
    }
    _ = loraFp32Bf16FusedAddTransposedAvx512;
}

// ============================================================================
// loraFp32Bf16FusedAddWt  (for grad_A computation)
// out[r][i] += scale * sum_t(intermediate[t][r] * weight[t][i])
// intermediate: [numTokens, rank] FP32, weight: [numTokens, dim] BF16, out: [rank, dim] FP32
// Reference: avx_kernels.hpp:600
// ============================================================================

fn loraFp32Bf16FusedAddWtAvx512(
    intermediate: [*]const f32,
    weight: [*]const bf16,
    out: [*]f32,
    numTokens: usize,
    rank: usize,
    dim: usize,
    scale: f32,
) void {
    _ = intermediate;
    _ = weight;
    _ = out;
    _ = numTokens;
    _ = rank;
    _ = dim;
    _ = scale;
    @panic("loraFp32Bf16FusedAddWt AVX512 path not yet ported — scalar fallback is active");
}

pub fn loraFp32Bf16FusedAddWt(
    intermediate: [*]const f32,
    weight: [*]const bf16,
    out: [*]f32,
    numTokens: usize,
    rank: usize,
    dim: usize,
    scale: f32,
) void {
    for (0..numTokens) |t| {
        for (0..rank) |r| {
            const inter_val = intermediate[t * rank + r];
            for (0..dim) |i| {
                const w_val = amx.bf16_to_f32(weight[t * dim + i]);
                out[r * dim + i] += scale * inter_val * w_val;
            }
        }
    }
    _ = loraFp32Bf16FusedAddWtAvx512;
}
