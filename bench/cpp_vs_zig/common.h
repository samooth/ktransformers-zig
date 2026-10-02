// Shared bench definitions for the C++ <-> Zig kernel comparison.
// Included by cpp_bench.cpp; zig_bench.zig mirrors these constants.
#pragma once

#include <stdint.h>

// DeepSeek-V3 MoE: hidden_size = 7168, intermediate_size (per expert, after
// the 1/3 split) = 2048.
#define KT_HIDDEN 7168
#define KT_INTER  2048

// name, m, n, k
#define KT_GEMM_SHAPES(X)              \
    X("decode gate/up",    1, KT_INTER, KT_HIDDEN)  \
    X("decode down",       1, KT_HIDDEN, KT_INTER)  \
    X("small decode",      8, KT_INTER, KT_HIDDEN)  \
    X("prefill 32",       32, KT_INTER, KT_HIDDEN)  \
    X("prefill 128",     128, KT_INTER, KT_HIDDEN)  \
    X("prefill 512",     512, KT_INTER, KT_HIDDEN)

// Activation-only shape (MoE gate/up outputs for a prefill batch).
#define KT_ACT_M 128
#define KT_ACT_N KT_INTER

// splitmix64: identical implementation in both harnesses so both see the
// exact same inputs.
static inline uint64_t kt_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform float in [-1, 1), then rounded to bf16 with ggml's round-to-nearest
// -even (Zig mirrors this so both harnesses get bit-identical inputs).
static inline float kt_rand_f32(uint64_t *state) {
    return (float)(kt_splitmix64(state) >> 40) * (1.0f / 8388608.0f) - 1.0f;
}