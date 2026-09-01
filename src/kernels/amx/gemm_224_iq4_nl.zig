// GEMM Kernel 224 IQ4_NL (GGML non-linear 4-bit i-quant, 32-weight super-blocks)
//
// Canonical GGUF layout (llama.cpp ggml-common.h):
//   block_iq4_nl {
//     ggml_half d;                  // super-block scale (f16, single scale — no
//                                  //   per-sub-block scales; this is what makes
//                                  //   IQ4_NL "NL" — the non-linear lookup IS
//                                  //   the per-element structure, no extra scale
//                                  //   bookkeeping needed)
//     uint8_t qs[QK4_NL/2];        // 4-bit quants (16 bytes, 32 nibbles)
//   } = 2 + 16 = 18 bytes per 32 weights (~4.5 bpw)
//
// Dequantization (ggml-quants.c:2725 dequantize_row_iq4_nl):
//   d = f16_to_f32(x[i].d)
//   for j in 0..16:
//     y[j+0 ] = d * kvalues_iq4nl[qs[j] & 0xf]    (low nibble)
//     y[j+16] = d * kvalues_iq4nl[qs[j] >> 4]    (high nibble)
//
// kvalues_iq4nl is the SAME 16-entry non-linear table the other dev's
// IQ4_XS uses (gemm_224_iq4_xs.zig:80). Reusing it here is the
// entire reason IQ4_NL is "the 32-superblock sibling of IQ4_XS" —
// the table is the design choice, the block geometry is just what
// fits. Quantization (ggml-quants.c:4966 quantize_row_iq4_nl_impl)
// is the same shared function; this port implements dequant only
// (the model-loading use case). See gemm_224_iq4_xs.zig:97 for the
// private bestIndexInt8 helper if quantize is later needed.

const std = @import("std");
const amx = @import("../arch/amx.zig");
const gemm_iq4_xs = @import("gemm_224_iq4_xs.zig");

// ============================================================================
// Block Structure (byte-exact vs ggml's block_iq4_nl)
// ============================================================================

pub const QK4_NL: usize = 32;

pub const BlockIQ4_NL = extern struct {
    d: u16, // f16: super-block scale (single scale for all 32 weights)
    qs: [QK4_NL / 2]u8, // 4-bit quants (16 bytes, 32 nibbles)
};

comptime {
    if (@sizeOf(BlockIQ4_NL) != 18) @compileError("BlockIQ4_NL must be 18 bytes");
}

// ============================================================================
// f16 <-> f32 (native f16, same pattern as the rest of the GGML workstream)
// ============================================================================

pub fn f32_to_f16(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

pub fn f16_to_f32(x: u16) f32 {
    const h: f16 = @bitCast(x);
    return @floatCast(h);
}

// ============================================================================
// dequantize_row_iq4_nl — port of dequantize_row_iq4_nl (byte-exact vs C)
// ============================================================================
//
// Reuses KVALUES_IQ4NL (the 16-entry non-linear table) from the IQ4_XS
// kernel. The shared `pub` visibility of KVALUES_IQ4NL in IQ4_XS is
// what makes the IQ4_NL port cheap — only the layout and the dequant
// code differ from IQ4_XS, both of which are dramatically simpler.
pub fn dequantizeRowIQ4_NL(x: [*]const BlockIQ4_NL, y: [*]f32, k: usize) void {
    // k must be a multiple of QK4_NL (32); the caller enforces.
    // Zig 0.16: `y` is a many-pointer ([*]f32); carry an index counter
    // for the per-block write instead of doing `y += QK4_NL` (same
    // pattern applied to IQ4_XS, Q4_K, Q5_K, Q6_K).
    const nb = k / QK4_NL;
    var yoff: usize = 0;

    for (0..nb) |i| {
        const blk = &x[i];
        const d = f16_to_f32(blk.d);

        for (0..QK4_NL / 2) |j| {
            const v_lo = gemm_iq4_xs.KVALUES_IQ4NL[blk.qs[j] & 0xf];
            const v_hi = gemm_iq4_xs.KVALUES_IQ4NL[blk.qs[j] >> 4];
            y[yoff + j + 0] = d * @as(f32, @floatFromInt(v_lo));
            y[yoff + j + QK4_NL / 2] = d * @as(f32, @floatFromInt(v_hi));
        }
        yoff += QK4_NL;
    }
}

// ============================================================================
// dequantize_row_iq4_nl_to_bf16 — convenience for the AMX path / test
// ============================================================================

pub fn dequantizeRowIQ4_NLToBF16(x: [*]const BlockIQ4_NL, y: [*]amx.bf16, k: usize) void {
    const nb = k / QK4_NL;
    var f32_buf: [QK4_NL]f32 = undefined;
    for (0..nb) |i| {
        @memset(&f32_buf, 0);
        dequantizeRowIQ4_NL(x + i, &f32_buf, QK4_NL);
        for (0..QK4_NL) |j| y[i * QK4_NL + j] = amx.f32_to_bf16(f32_buf[j]);
    }
}

// ============================================================================
// scalar GEMM: BF16 activations × IQ4_NL weights -> F32 output
// ============================================================================

pub fn gemmIQ4_NLScalar(
    input: [*]const amx.bf16, // [m, k] BF16 row-major
    weight: [*]const BlockIQ4_NL, // [n, k/QK4_NL] row-major (1 block = 32 weights)
    output: [*]f32, // [m, n] F32 row-major
    m: usize,
    n: usize,
    k: usize,
    input_ld: usize, // row stride of input in elements
    weight_ld: usize, // row stride of weight in BLOCKS
    output_ld: usize, // row stride of output in elements
) void {
    const nb = k / QK4_NL;
    var scratch: [QK4_NL]f32 = undefined;
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..nb) |blk| {
                dequantizeRowIQ4_NL(weight + j * weight_ld + blk, &scratch, QK4_NL);
                for (0..QK4_NL) |k0| {
                    sum += amx.bf16_to_f32(input[i * input_ld + k0]) * scratch[k0];
                }
            }
            output[i * output_ld + j] = sum;
        }
    }
}

// ============================================================================
// quantize — NOT IMPLEMENTED
//
// The C reference calls `quantize_row_iq4_nl_impl` (ggml-quants.c:4966)
// with super_block_size=QK4_NL=32, block_size=32. That function is the
// SAME implementation the other dev's IQ4_XS uses internally (their
// file line 4966 path), parameterized on the super-block size. Their
// helper is currently private (gemm_224_iq4_xs.zig: bestIndexInt8 and
// the shared impl). When quantize becomes a priority, the cleanest
// path is to promote bestIndexInt8 to `pub` in gemm_224_iq4_xs.zig
// (a one-keyword PR) and re-implement the private impl here. Until
// then, the dequant+matmul path is sufficient for LOADING real
// GGUF models whose weights come pre-quantized.
// ============================================================================
pub fn quantizeRowIQ4_NL(_: [*]const f32, _: [*]BlockIQ4_NL, _: usize) void {
    @panic("quantize_row_iq4_nl: not implemented (see comment; the shared impl lives in gemm_224_iq4_xs.zig — would need bestIndexInt8 promoted to pub)");
}
