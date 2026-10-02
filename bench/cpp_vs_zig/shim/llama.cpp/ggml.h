// Minimal stand-in for llama.cpp's ggml.h.
//
// The reference C++ AVX2 kernel headers
//   operators/avx2/avx2_bf16_gemm.hpp
//   operators/avx2/avx2_bf16_utils.hpp
// include "llama.cpp/ggml.h" for exactly four symbols:
//   ggml_bf16_t, ggml_compute_fp32_t, GGML_BF16_TO_FP32, GGML_FP32_TO_BF16
//
// llama.cpp is not vendored in the ktransformers tree, so this shim supplies
// them with bit-identical semantics to ggml-impl.h (round-to-nearest-even on
// the fp32->bf16 direction, plain shift on bf16->fp32). Nothing else from ggml
// is needed, and the kernels themselves are used UNMODIFIED.
#pragma once

#include <stdint.h>
#include <string.h>

typedef uint16_t ggml_bf16_t;
typedef float ggml_compute_fp32_t;

static inline float ggml_bf16_to_fp32(ggml_bf16_t h) {
    float f;
    uint32_t u = ((uint32_t)h) << 16;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// ggml-impl.h semantics: add 0x7fff + ((u >> 16) & 1) then truncate.
static inline ggml_bf16_t ggml_fp32_to_bf16(float s) {
    uint32_t u;
    memcpy(&u, &s, sizeof(u));
    u += 0x7fff + ((u >> 16) & 1);
    return (ggml_bf16_t)(u >> 16);
}

#define GGML_BF16_TO_FP32(x) ggml_bf16_to_fp32(x)
#define GGML_FP32_TO_BF16(x) ggml_fp32_to_bf16(x)