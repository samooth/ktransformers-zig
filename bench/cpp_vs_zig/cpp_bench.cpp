// C++ reference benchmark: drives the UNMODIFIED ktransformers AVX2 BF16
// kernels from
//   ktransformers/kt-kernel/operators/avx2/avx2_bf16_gemm.hpp
//   ktransformers/kt-kernel/operators/avx2/avx2_bf16_utils.hpp
//
// Compile with -mavx2 -mfma (this host: Zen3, AVX2+FMA+F16C, no AVX512/AMX).
//
// Output format (one line per case), consumed by run.sh:
//   GEMM|<name>|<ms>|<GFLOPs>
//   ACT|<name>|<ms>|<GFLOPs>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "avx2_bf16_gemm.hpp"
#include "avx2_bf16_utils.hpp"
#include "common.h"

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static constexpr int WARMUP = 2;
static constexpr int REPS = 9;

// SwiGLU over a [m, n] bf16 gate/up pair -> bf16 dst, mirroring what the MoE
// forward does between the gate/up GEMMs and the down GEMM.
static void swiglu_ref(const ggml_bf16_t* gate, const ggml_bf16_t* up,
                       ggml_bf16_t* dst, int m, int n) {
    for (int i = 0; i < m; i++) {
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 g = avx2::load_bf16_to_fp32(gate + i * n + j);
            __m256 u = avx2::load_bf16_to_fp32(up + i * n + j);
            avx2::store_fp32_to_bf16(dst + i * n + j, avx2::act_fn(g, u));
        }
        for (; j < n; j++) {
            dst[i * n + j] = GGML_FP32_TO_BF16(
                GGML_BF16_TO_FP32(gate[i * n + j]) / (1.0f + expf(-GGML_BF16_TO_FP32(gate[i * n + j]))) *
                GGML_BF16_TO_FP32(up[i * n + j]));
        }
    }
}

static void bench_gemm(const char* name, int m, int n, int k) {
    uint64_t rng = 0x243F6A8885A308D3ULL;
    std::vector<ggml_bf16_t> a((size_t)m * k), b((size_t)n * k);
    for (auto& v : a) v = GGML_FP32_TO_BF16(kt_rand_f32(&rng));
    for (auto& v : b) v = GGML_FP32_TO_BF16(kt_rand_f32(&rng));
    std::vector<float> c((size_t)m * n);

    // Buffer wrappers are pure pass-through views on row-major memory: no
    // packing, no transpose, no staging copies.
    avx2::GemmKernelAVX2BF16::BufferA ba((size_t)m, (size_t)k, a.data());
    avx2::GemmKernelAVX2BF16::BufferB bb((size_t)n, (size_t)k, b.data());
    avx2::GemmKernelAVX2BF16::BufferC bc((size_t)m, (size_t)n, c.data());

    avx2::GemmKernelAVX2BF16::config();
    for (int i = 0; i < WARMUP; i++)
        avx2::gemm_bf16(m, n, k, ba, bb, bc, 0, 1);

    double best = 1e30;
    for (int r = 0; r < REPS; r++) {
        double t0 = now_ms();
        avx2::gemm_bf16(m, n, k, ba, bb, bc, 0, 1);
        best = std::min(best, now_ms() - t0);
    }

    char path[256];
    snprintf(path, sizeof(path), "%s/out_cpp_gemm_%d_%d_%d.f32", getenv("KT_BENCH_OUT") ? getenv("KT_BENCH_OUT") : ".", m, n, k);
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(c.data(), sizeof(float), c.size(), f); fclose(f); }

    double gflops = 2.0 * (double)m * n * k / (best * 1e6);
    char label[160];
    snprintf(label, sizeof(label), "%s [m=%d,n=%d,k=%d]", name, m, n, k);
    printf("GEMM|%s|%.4f|%.2f\n", label, best, gflops);
}

static void bench_act(const char* name, int m, int n) {
    uint64_t rng = 0x13198A2E03707344ULL;
    std::vector<ggml_bf16_t> g((size_t)m * n), u((size_t)m * n), d((size_t)m * n);
    for (auto& v : g) v = GGML_FP32_TO_BF16(kt_rand_f32(&rng));
    for (auto& v : u) v = GGML_FP32_TO_BF16(kt_rand_f32(&rng));

    for (int i = 0; i < WARMUP; i++) swiglu_ref(g.data(), u.data(), d.data(), m, n);

    double best = 1e30;
    for (int r = 0; r < REPS; r++) {
        double t0 = now_ms();
        swiglu_ref(g.data(), u.data(), d.data(), m, n);
        best = std::min(best, now_ms() - t0);
    }

    const char* outdir = getenv("KT_BENCH_OUT") ? getenv("KT_BENCH_OUT") : ".";
    char path[256];
    snprintf(path, sizeof(path), "%s/out_cpp_act_%d_%d.bf16", outdir, m, n);
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(d.data(), sizeof(ggml_bf16_t), d.size(), f); fclose(f); }

    // 3 transcendentals + 3 mul/div per element; report GB/s of the bf16 I/O.
    double gbs = (double)m * n * 2 * 2 / (best * 1e6);
    char label[160];
    snprintf(label, sizeof(label), "%s [m=%d,n=%d]", name, m, n);
    printf("ACT|%s|%.4f|%.2f\n", label, best, gbs);
}

int main() {
    avx2::GemmKernelAVX2BF16::config();
#define RUN_X(nm, m, n, k) bench_gemm(nm, m, n, k);
    KT_GEMM_SHAPES(RUN_X)
#undef RUN_X
    bench_act("swiglu act", KT_ACT_M, KT_ACT_N);
    return 0;
}