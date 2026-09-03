/**
 * ktransformers-zig C API Header
 * Compatible with existing kt_kernel_ext.so Python bindings
 */

#ifndef KTRANSFORMERS_C_API_H
#define KTRANSFORMERS_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Types and Constants
// ============================================================================

/// Quantization methods
typedef enum {
    KT_QUANT_NONE = 0,
    KT_QUANT_FP8 = 1,
    KT_QUANT_INT8 = 2,
    KT_QUANT_INT4 = 3,
    KT_QUANT_GPTQ = 4,
    KT_QUANT_AWQ = 5,
    KT_QUANT_MXFP4 = 6,
    KT_QUANT_MXFP8 = 7,
    KT_QUANT_BF16 = 8,
} kt_quant_method_t;

/// GGML-compatible type identifiers
typedef enum {
    KT_TYPE_F32 = 0,
    KT_TYPE_F16 = 1,
    KT_TYPE_BF16 = 2,
    KT_TYPE_I8 = 3,
    KT_TYPE_I4 = 4,
    KT_TYPE_Q4_0 = 8,
    KT_TYPE_Q4_1 = 9,
    KT_TYPE_Q5_0 = 10,
    KT_TYPE_Q5_1 = 11,
    KT_TYPE_Q8_0 = 12,
    KT_TYPE_Q8_1 = 13,
    KT_TYPE_Q2_K = 14,
    KT_TYPE_Q3_K = 15,
    KT_TYPE_Q4_K = 16,
    KT_TYPE_Q5_K = 17,
    KT_TYPE_Q6_K = 18,
    KT_TYPE_Q8_K = 19,
    KT_TYPE_IQ2_XXS = 20,
    KT_TYPE_IQ2_XS = 21,
    KT_TYPE_IQ3_XXS = 22,
    KT_TYPE_IQ1_S = 23,
    KT_TYPE_IQ4_NL = 24,
    KT_TYPE_IQ3_S = 25,
    KT_TYPE_IQ2_S = 26,
    KT_TYPE_IQ4_XS = 27,
    KT_TYPE_IQ1_M = 28,
    KT_TYPE_MXFP4 = 29,
    KT_TYPE_MXFP8 = 30,
} kt_type_t;

/// CPU variant
typedef enum {
    KT_VARIANT_AVX2 = 0,
    KT_VARIANT_AVX512_BASE = 1,
    KT_VARIANT_AVX512_VNNI = 2,
    KT_VARIANT_AVX512_VBMI = 3,
    KT_VARIANT_AVX512_BF16 = 4,
    KT_VARIANT_AMX = 5,
} kt_variant_t;

/// Opaque handles
typedef struct KT_CPUInfer KT_CPUInfer;
typedef struct KT_WorkerPool KT_WorkerPool;
typedef struct KT_MOE KT_MOE;
typedef struct KT_MLA KT_MLA;
typedef struct KT_DSV3Layer KT_DSV3Layer;
typedef struct KT_Linear KT_Linear;
typedef struct KT_MLP KT_MLP;
typedef struct KT_Gate KT_Gate;

/// Quantization configuration
typedef struct {
    kt_quant_method_t quant_method;
    int bits;
    int group_size;
    int zero_point;
    int per_channel;
} kt_quant_config_t;

/// General configuration (model-level)
typedef struct {
    size_t vocab_size;
    size_t hidden_size;
    size_t num_experts_per_tok;
    size_t n_routed_experts;
    size_t n_shared_experts;
    size_t max_qlen;

    // Weight pointers (void* for opaque data)
    void* lm_heads_ptr;
    kt_type_t lm_heads_type;
    void* norm_weights_ptr;
    kt_type_t norm_weights_type;
    void* token_embd_ptr;
    kt_type_t token_embd_type;

    KT_WorkerPool* pool;
} kt_general_config_t;

/// MLA configuration
typedef struct {
    size_t hidden_size;
    size_t q_lora_rank;
    size_t num_heads;
    size_t nope_size;
    size_t rope_size;
    size_t kv_lora_rank;

    int layer_idx;
    KT_WorkerPool* pool;
    size_t token_count_in_page;
    size_t max_qlen;
    size_t max_kvlen;

    // RoPE
    size_t max_position_embeddings;
    double rope_scaling_factor;
    double rope_theta;
    double rope_scaling_beta_fast;
    double rope_scaling_beta_slow;
    double rope_scaling_mscale;
    double rope_scaling_mscale_all_dim;
    double rope_scaling_original_max_position_embeddings;

    // Projections
    void* q_a_proj;
    void* q_a_norm;
    void* q_b_proj;
    void* kv_a_proj_with_mqa;
    void* kv_a_norm;
    void* kv_b_proj;
    void* o_proj;

    // Types
    kt_type_t q_a_proj_type;
    kt_type_t q_a_norm_type;
    kt_type_t q_b_proj_type;
    kt_type_t kv_a_proj_with_mqa_type;
    kt_type_t kv_a_norm_type;
    kt_type_t kv_b_proj_type;
    kt_type_t w_o_type;
    kt_type_t input_type;
    kt_type_t output_type;

    size_t m_block;
    size_t n_block;
    size_t page_count;
} kt_mla_config_t;

/// MoE configuration
typedef struct {
    int expert_num;
    int num_experts_per_tok;
    int hidden_size;
    int intermediate_size;

    int layer_idx;
    KT_WorkerPool* pool;

    // SGLang offload
    int num_gpu_experts;
    uint8_t* gpu_experts_mask;
    void* physical_to_logical_map;

    // Weights
    void* gate_proj;
    void* up_proj;
    void* down_proj;

    void* gate_scale;
    void* up_scale;
    void* down_scale;

    void* gate_zero;
    void* up_zero;
    void* down_zero;

    kt_quant_config_t quant_config;

    // For AMX/TP
    int max_len;
    void** gate_projs;
    void** up_projs;
    void** down_projs;
    void** gate_scales;
    void** up_scales;
    void** down_scales;
    void** gate_zeros;
    void** up_zeros;
    void** down_zeros;

    // Backward weights
    void** gate_bwd_projs;
    void** up_bwd_projs;
    void** down_bwd_projs;
    void** gate_bwd_scales;
    void** up_bwd_scales;
    void** down_bwd_scales;

    char* path;
    int save;
    int load;
    int share_backward_bb;
    int share_cache_pool;
    int m_block;
    int group_min_len;
    int group_max_len;

    kt_type_t gate_type;
    kt_type_t up_type;
    kt_type_t down_type;
    kt_type_t hidden_type;

    int max_cache_depth;
    float swiglu_limit;
    float swiglu_alpha;
} kt_moe_config_t;

/// MoE SFT (LoRA) configuration
typedef struct {
    // Base config
    kt_moe_config_t base;

    // LoRA
    int lora_rank;
    float lora_alpha;
    float lora_dropout;

    void* gate_lora_a;
    void* gate_lora_b;
    void* up_lora_a;
    void* up_lora_b;
    void* down_lora_a;
    void* down_lora_b;

    int full_weight_grad;
    int authoritative_optimizer_grads;

    void* grad_gate_proj;
    void* grad_up_proj;
    void* grad_down_proj;
} kt_moe_sft_config_t;

/// Gate configuration
typedef struct {
    size_t hidden_size;
    size_t num_experts_per_tok;
    size_t n_routed_experts;
    size_t n_group;
    size_t topk_group;

    int norm_topk_prob;
    float routed_scaling_factor;
    char* scoring_func;
    char* topk_method;

    int layer_idx;
    KT_WorkerPool* pool;

    void* weight;
    kt_type_t weight_type;
    void* e_score_correction_bias;
    kt_type_t e_score_correction_bias_type;

    size_t max_seqlen;
} kt_gate_config_t;

/// Linear configuration
typedef struct {
    int hidden_size;
    int intermediate_size;
    int stride;
    int group_max_len;
    void* proj;
    kt_type_t proj_type;
    kt_type_t hidden_type;
} kt_linear_config_t;

/// MLP configuration
typedef struct {
    int hidden_size;
    int intermediate_size;
    int stride;
    int group_max_len;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    kt_type_t gate_type;
    kt_type_t up_type;
    kt_type_t down_type;
    kt_type_t hidden_type;
} kt_mlp_config_t;

// ============================================================================
// Worker Pool
// ============================================================================

/// Worker pool configuration
typedef struct {
    int subpool_count;
    int* subpool_numa_map;
    int* subpool_thread_count;
} kt_worker_pool_config_t;

/// Create worker pool
KT_WorkerPool* kt_worker_pool_new(int thread_count);
KT_WorkerPool* kt_worker_pool_new_config(kt_worker_pool_config_t config);
void kt_worker_pool_free(KT_WorkerPool* pool);

// ============================================================================
// CPUInfer
// ============================================================================

/// Create CPUInfer instance
KT_CPUInfer* kt_cpuinfer_new(int thread_count);
KT_CPUInfer* kt_cpuinfer_new_config(kt_worker_pool_config_t config);
void kt_cpuinfer_free(KT_CPUInfer* cpuinfer);

/// Submit task
void kt_cpuinfer_submit(KT_CPUInfer* cpuinfer, void (*func)(void*), void* arg);

/// Sync
void kt_cpuinfer_sync(KT_CPUInfer* cpuinfer, size_t allow_n_pending);

/// Get backend worker pool
KT_WorkerPool* kt_cpuinfer_get_backend(KT_CPUInfer* cpuinfer);

// ============================================================================
// MoE
// ============================================================================

/// Create MoE instance
KT_MOE* kt_moe_new(KT_CPUInfer* cpuinfer, kt_moe_config_t config);
KT_MOE* kt_moe_new_sft(KT_CPUInfer* cpuinfer, kt_moe_sft_config_t config);
void kt_moe_free(KT_MOE* moe);

/// SFT forward (with optional backward cache)
void kt_moe_forward_sft(KT_MOE* moe, int qlen, int k, const int64_t* expert_ids,
                        const float* weights, const void* input, void* output, int save_for_backward);

/// Update LoRA weight pointers (call after LoRA weights change)
void kt_moe_update_lora_weights(KT_MOE* moe, void* gate_lora_a, void* gate_lora_b,
                                void* up_lora_a, void* up_lora_b, void* down_lora_a, void* down_lora_b);

/// Warm up
void kt_moe_warm_up(KT_MOE* moe);

/// Load weights
void kt_moe_load_weights(KT_MOE* moe);
void kt_moe_load_weights_with_map(KT_MOE* moe, uint64_t* physical_to_logical_map);

/// Forward
void kt_moe_forward(KT_MOE* moe, int qlen, int k, const int64_t* expert_ids,
                     const float* weights, const void* input, void* output, int incremental);

/// Backward (SFT/LoRA training). Computes grad_input, 6 LoRA grads (gate/up/down x A/B),
/// optional base-weight grads (grad_gate/up/down_proj), and routing-weight grads.
/// Matches C++ backward_binding at moe-sft-tp.hpp:1249.
void kt_moe_backward(KT_MOE* moe, const void* grad_output, void* grad_input,
                     void* grad_gate_lora_a, void* grad_gate_lora_b, void* grad_up_lora_a, void* grad_up_lora_b,
                     void* grad_down_lora_a, void* grad_down_lora_b, void* grad_weights,
                     void* grad_gate_proj, void* grad_up_proj, void* grad_down_proj,
                     int accumulate_optimizer_grads, float optimizer_grad_scale);

/// Get CPU variant string
const char* kt_get_cpu_variant();

// ============================================================================
// MLA
// ============================================================================

KT_MLA* kt_mla_new(kt_mla_config_t config);
void kt_mla_free(KT_MLA* mla);
void kt_mla_load_weights(KT_MLA* mla);
void kt_mla_forward(KT_MLA* mla, const int* qlens, int qlen_count,
                     const int** page_tables, int* page_table_lens,
                     const int* kv_lens, const void* input, void* output);

// ============================================================================
// Gate
// ============================================================================

KT_Gate* kt_gate_new(kt_gate_config_t config);
void kt_gate_free(KT_Gate* gate);
// kt_gate_forward: Zig extension beyond the 4-arg C++ minimum (input/output/qlen).
// The Zig port additionally returns the routing decisions in (topk_ids,
// topk_weights) and exposes batch_size for callers that batch multiple
// sequences through one gate instance. The ctypes wrapper (and the
// existing tests) bind this 7-arg form; the 4-arg form is not exported.
void kt_gate_forward(KT_Gate* gate, const void* input, void* output,
                     int64_t* topk_ids, float* topk_weights,
                     int batch_size, int qlen);

// ============================================================================
// Linear / MLP
// ============================================================================

KT_Linear* kt_linear_new(kt_linear_config_t config);
void kt_linear_free(KT_Linear* linear);
void kt_linear_forward(KT_Linear* linear, const void* input, void* output, int qlen);

KT_MLP* kt_mlp_new(kt_mlp_config_t config);
void kt_mlp_free(KT_MLP* mlp);
void kt_mlp_forward(KT_MLP* mlp, const void* input, void* output, int qlen);

// ============================================================================
// Quantization utilities
// ============================================================================

/// Convert BF16 to FP32
void kt_bf16_to_f32(const void* src, float* dst, size_t count);

/// Convert FP32 to BF16
void kt_f32_to_bf16(const float* src, void* dst, size_t count);

/// Quantize BF16 to INT8 (per-row)
void kt_quantize_int8_per_row(const void* src, int8_t* dst, float* scales, size_t rows, size_t cols);

/// Dequantize INT8 to BF16
void kt_dequantize_int8(const int8_t* src, const float* scales, void* dst, size_t rows, size_t cols);

/// Quantize BF16 to INT4 (GPTQ style)
void kt_quantize_int4_gptq(const void* src, uint8_t* dst, float* scales, uint8_t* zeros, size_t rows, size_t cols, int group_size);

/// Dequantize INT4 to BF16
void kt_dequantize_int4_gptq(const uint8_t* src, const float* scales, const uint8_t* zeros, void* dst, size_t rows, size_t cols, int group_size);

/// Quantize BF16 to FP8 E4M3
void kt_quantize_fp8_e4m3(const void* src, uint8_t* dst, float* scales, size_t rows, size_t cols, int block_size);

/// Dequantize FP8 E4M3 to BF16
void kt_dequantize_fp8_e4m3(const uint8_t* src, const float* scales, void* dst, size_t rows, size_t cols, int block_size);

// ============================================================================
// GGML Quantization Types (byte-exact vs llama.cpp ggml-common.h)
// ============================================================================
// Block sizes: Q8_0 = 34 B / 32 weights, Q4_K = 144 B / 256,
//              Q5_K = 176 B / 256, Q6_K = 210 B / 256.
// Quantize: src is F32 [k], dst is BlockX[k/QK]; Dequantize: the reverse.
// Matmul: a [m,k] BF16 activations, b [n, k/QK] blocks (row-major, ldb in
// BLOCKS), c [m,n] F32 — on-the-fly dequant.

/// Q8_0: quantize/dequantize one row (k must be a multiple of 32)
void kt_quantize_q8_0(const float* src, void* dst, size_t k);
void kt_dequantize_q8_0(const void* src, float* dst, size_t k);

/// Q4_K: quantize/dequantize one row (k must be a multiple of 256)
void kt_quantize_q4_k(const float* src, void* dst, size_t k);
void kt_dequantize_q4_k(const void* src, float* dst, size_t k);

/// Q5_K: quantize/dequantize one row (k must be a multiple of 256)
void kt_quantize_q5_k(const float* src, void* dst, size_t k);
void kt_dequantize_q5_k(const void* src, float* dst, size_t k);

/// Q6_K: quantize/dequantize one row (k must be a multiple of 256)
void kt_quantize_q6_k(const float* src, void* dst, size_t k);
void kt_dequantize_q6_k(const void* src, float* dst, size_t k);

/// Q8_K: quantize/dequantize one row (k must be a multiple of 256).
/// Block = f32 d + 256 x i8 qs + 16 x i16 bsums (292 bytes) — note d is
/// f32 here, unlike the f16 of the other K-quants.
void kt_quantize_q8_k(const float* src, void* dst, size_t k);
void kt_dequantize_q8_k(const void* src, float* dst, size_t k);

/// GGML GEMMs: BF16 activations x quantized weights -> F32 output
void kt_matmul_q8_0(const void* a, const void* b, float* c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void kt_matmul_q4_k(const void* a, const void* b, float* c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void kt_matmul_q5_k(const void* a, const void* b, float* c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void kt_matmul_q6_k(const void* a, const void* b, float* c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void kt_matmul_q8_k(const void* a, const void* b, float* c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);

// ============================================================================
// Utility
// ============================================================================

/// Initialize GGML
void kt_ggml_init();

/// Get version string
const char* kt_version();

// ============================================================================
// FP8 Layerwise Transport
// ============================================================================

typedef struct KT_FP8LayerwiseTransport KT_FP8LayerwiseTransport;

/// Initialize FP8 layerwise control
void kt_fp8_layerwise_init(void* control_ptr, size_t control_size, int tp_size);

/// Create transport
KT_FP8LayerwiseTransport* kt_fp8_transport_new(void* control_ptr, size_t control_size,
                                                 int rank, int tp_size, int cuda_device,
                                                 const uintptr_t* local_host_ptrs,
                                                 const uintptr_t* local_gpu_ptrs,
                                                 const uintptr_t* all_rank_host_ptrs,
                                                 const size_t* expert_nbytes,
                                                 int num_experts, int timeout_ms);

void kt_fp8_transport_free(KT_FP8LayerwiseTransport* transport);

/// Run producer
void kt_fp8_transport_run_producer(KT_FP8LayerwiseTransport* transport,
                                     uint64_t epoch, int layer_id, int expert_count,
                                     void (*callback)(int expert_id,
                                                      const uintptr_t* w13_weight_ptrs,
                                                      const uintptr_t* w13_scale_ptrs,
                                                      const uintptr_t* w2_weight_ptrs,
                                                      const uintptr_t* w2_scale_ptrs));

/// Wait for completion
typedef struct {
    uint64_t epoch;
    int layer_id;
    int expert_count;
    int rank;
    double writer_ms;
    double slot_wait_ms;
    double h2d_ms;
    double total_ms;
    size_t bytes;
    int poisoned;
    int error_code;
    int error_rank;
    const char* error_message;
} kt_fp8_stats_t;

kt_fp8_stats_t kt_fp8_transport_wait(KT_FP8LayerwiseTransport* transport, uint64_t epoch);

/// Close transport
void kt_fp8_transport_close(KT_FP8LayerwiseTransport* transport);

/// Properties
int kt_fp8_transport_rank(KT_FP8LayerwiseTransport* transport);
int kt_fp8_transport_tp_size(KT_FP8LayerwiseTransport* transport);
int kt_fp8_transport_num_experts(KT_FP8LayerwiseTransport* transport);
int kt_fp8_transport_closed(KT_FP8LayerwiseTransport* transport);

#ifdef __cplusplus
}
#endif

// ============================================================================
// Zig extensions (not in the C++ kt-kernel, exported by the Zig .so)
// ============================================================================
// These are stable additions made by the Zig port. They are part of the .so
// surface and gated by tools/verify_abi.py. The C++ pybind11 wrapper does not
// expose them; only the ctypes path (python/kt_kernel/__init__.py) uses them.

// --- MoE per-expert step APIs (allow splitting gate+up and down projections
//     for callers that want to interleave work with other ops) ---
void kt_moe_forward_gate_up(KT_MOE* moe, int expert_idx, int qlen,
                            const void* input, void* gate_output, void* up_output);
void kt_moe_forward_down(KT_MOE* moe, int expert_idx, int qlen,
                         const void* input, void* output);

// --- MLA single-sequence conveniences (the paged/batched kt_mla_forward is
//     the canonical contract; these are wrappers for the prefill / decode
//     shapes that the C++ reference exposes via paged-attention queues) ---
void kt_mla_prefill(KT_MLA* mla, const void* input, void* output,
                    int qlen, const int64_t* position_ids);
void kt_mla_decode(KT_MLA* mla, const void* input, void* output,
                   int64_t position_id);
// external kv_cache is accepted for API symmetry with the C++ MLA binding
// but is currently ignored (the engine uses its own internal MlaKvCache).
void kt_mla_update_kv_cache(KT_MLA* mla, void* kv_cache,
                            const void* new_kv, int64_t position);

// --- FP8 E4M3 quantize/dequantize (per-tensor and per-block variants) ---
void kt_fp8_quantize(const void* config, const float* src, uint8_t* dst, size_t count);
void kt_fp8_dequantize(const void* config, const uint8_t* src, float* dst, size_t count);
void kt_fp8_quantize_block(const float* src, uint8_t* dst, float* scales,
                           size_t num_blocks, int block_size);
void kt_fp8_dequantize_block(const uint8_t* src, const float* scales,
                             float* dst, size_t num_blocks, int block_size);

// --- Scalar GEMM helpers (BF16 / INT8 / INT4 GPTQ / FP8) ---
void kt_matmul_bf16(const void* a, const void* b, float* c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void kt_matmul_int8(const void* a, const void* b, int* c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void kt_matmul_int4(const void* a, const void* b, float* c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);
void kt_matmul_fp8(const void* a, const void* b, const float* b_scales, float* c,
                    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc);

// --- Generic block ↔ F32 dispatch (ports llama.cpp to_float / from_float) ---
// One C entry point per direction, dispatching on `kt_type_t`. Used by the
// framework's expert code to convert activations between formats without
// knowing the type at compile time. Covers the 8 linear (non-IQ) GGML
// block formats we have C-API exports for: F32, F16, BF16, Q8_0, Q4_K,
// Q5_K, Q6_K, Q8_K. IQ formats (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS, IQ4_NL,
// IQ1_S, IQ1_M, IQ2_S, IQ3_S) are intentionally NOT dispatched here —
// their block layouts have different per-row sizes and dispatching on a
// single `n` would need different "n" semantics per type. Callers should
// use the per-type `kt_dequantize_iq*` / `kt_quantize_iq*` exports for those.
//
// Returns 0 on success, -1 on unsupported type (caller should fall back to
// the per-type exports).
int kt_to_float(const void* src, float* dst, size_t n, int type);
int kt_from_float(const float* src, void* dst, size_t n, int type);
// Source byte size for one row of `n` elements in `type` format. 0 if unsupported.
int kt_type_row_bytes(size_t n, int type);

// --- WorkerPool thread count (useful for the C API consumer to size resources) ---
int kt_worker_pool_get_thread_num(KT_WorkerPool* pool);

// --- Math helper primitives (SwiGLU, RMSNorm, RoPE, Softmax) ---
void kt_apply_swiglu(const void* gate, const void* up, void* dst,
                     size_t count, float limit, float alpha);
void kt_apply_rms_norm(const void* input, const void* weight, void* output,
                       size_t hidden_size, float eps);
void kt_apply_rope(void* q, void* k, int64_t position, size_t head_dim, double rope_theta);
void kt_softmax(const float* input, float* output, size_t size);

// --- Custom allocator injection (call BEFORE any kt_*_new) ---
// Installs a C-ABI allocator used by every subsequent kt_* construction
// and per-call scratch. Objects created earlier keep the allocator they
// were built with; their *_free paths use that captured allocator, so
// swapping the default between new and free is safe. Pass NULL to
// restore the built-in default (mmap-backed page allocator).
typedef struct kt_allocator_vtable {
    void* userdata;                                   /* passed to every callback */
    uint8_t* (*alloc)(void* userdata, size_t size, size_t alignment);   /* NULL on failure */
    void (*free)(void* userdata, uint8_t* ptr, size_t size, size_t alignment);
    int (*resize)(void* userdata, uint8_t* ptr, size_t old_size,
                  size_t new_size, size_t alignment); /* 0 ok, -1 unsupported */
} kt_allocator_vtable;
void kt_set_default_allocator(const kt_allocator_vtable* vtable);

// ============================================================================
// DeepseekV3 Decoder Layer — model orchestration (Zig extension)
// ============================================================================
// Ports examples/modeling_deepseek_v3.py's DeepseekV3DecoderLayer: single
// sequence, continuous KV. Weight pointers are caller-owned (borrowed).

typedef struct kt_dsv3_layer_config_t {
    /* dims */
    size_t hidden_size;
    size_t q_lora_rank;
    size_t num_heads;
    size_t nope_size;
    size_t rope_size;
    size_t kv_lora_rank;
    size_t max_qlen;
    size_t max_kvlen;
    size_t token_count_in_page;
    double rope_theta;
    size_t expert_num;
    size_t num_experts_per_tok;
    size_t intermediate_size;
    size_t n_group;
    size_t topk_group;
    int norm_topk_prob;
    float routed_scaling_factor;
    void* pool;                       /* KT_WorkerPool* or NULL */
    /* weights (BF16 unless noted) */
    const void* q_a_proj;
    const void* q_a_norm;
    const void* q_b_proj;
    const void* kv_a_proj_with_mqa;
    const void* kv_a_norm;
    const void* kv_b_proj;
    const void* o_proj;
    const void* attn_norm_weight;
    const void* ffn_norm_weight;
    const void* gate_weight;
    const void* e_score_correction_bias;  /* f32 [expert_num]; NULL = no bias */
    const void* gate_proj;
    const void* up_proj;
    const void* down_proj;
} kt_dsv3_layer_config_t;

KT_DSV3Layer* kt_dsv3_layer_new(const kt_dsv3_layer_config_t* config);
void kt_dsv3_layer_free(KT_DSV3Layer* layer);
/* qlen tokens at kv_start_pos context positions; input/output BF16 [qlen, hidden] */
void kt_dsv3_layer_forward(KT_DSV3Layer* layer, size_t qlen, size_t kv_start_pos,
                           const void* input, void* output);

// --- DeepseekV3Model (N-layer loop + final RMSNorm) / ForCausalLM (+ lm_head) ---
typedef struct kt_dsv3_model_config_t {
    size_t num_layers;
    kt_dsv3_layer_config_t layer;       /* per-layer template */
    const void* final_norm_weight;     /* BF16 [hidden_size] */
    const void* lm_head;               /* BF16 [vocab_size, hidden_size]; CausalLM only */
    size_t vocab_size;                 /* 0 = model-only */
} kt_dsv3_model_config_t;

typedef struct KT_DSV3Model KT_DSV3Model;
typedef struct KT_DSV3CausalLM KT_DSV3CausalLM;

KT_DSV3Model* kt_dsv3_model_new(const kt_dsv3_model_config_t* config);
void kt_dsv3_model_forward(KT_DSV3Model* model, size_t qlen, size_t kv_start_pos,
                            const void* input, void* output);
void kt_dsv3_model_free(KT_DSV3Model* model);

KT_DSV3CausalLM* kt_dsv3_causallm_new(const kt_dsv3_model_config_t* config);
/* logits: f32 [qlen, vocab_size] */
void kt_dsv3_causallm_forward(KT_DSV3CausalLM* clm, size_t qlen, size_t kv_start_pos,
                              const void* input, float* logits);
void kt_dsv3_causallm_free(KT_DSV3CausalLM* clm);

// ============================================================================
// Qwen3 MoE (model orchestration — Zig extension)
// ============================================================================
// Ports the Qwen3 MoE forward pass: standard MHA + GQA + RoPE + pre-norm
// + vanilla softmax top-k gate (no group routing, no e_score_correction_bias)
// + MoE FFN. Weight pointers are caller-owned (borrowed).

typedef struct kt_qwen3moe_layer_config_t {
    /* dims */
    size_t hidden_size;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    size_t max_qlen;
    size_t max_kvlen;
    double rope_theta;
    size_t expert_num;
    size_t num_experts_per_tok;
    size_t intermediate_size;
    void* pool;                       /* KT_WorkerPool* or NULL */
    /* weights (BF16) */
    const void* q_proj;               /* [num_heads*head_dim, hidden_size] */
    const void* k_proj;               /* [num_kv_heads*head_dim, hidden_size] */
    const void* v_proj;               /* [num_kv_heads*head_dim, hidden_size] */
    const void* o_proj;               /* [hidden_size, num_heads*head_dim] */
    const void* attn_norm_weight;     /* BF16 [hidden_size] */
    const void* ffn_norm_weight;      /* BF16 [hidden_size] */
    const void* gate_weight;          /* BF16 [expert_num, hidden_size] */
    const void* gate_proj;            /* BF16 [expert_num, inter, hidden] */
    const void* up_proj;
    const void* down_proj;
} kt_qwen3moe_layer_config_t;

typedef struct KT_Qwen3MoeLayer KT_Qwen3MoeLayer;
typedef struct KT_Qwen3MoeModel KT_Qwen3MoeModel;
typedef struct KT_Qwen3MoeCausalLM KT_Qwen3MoeCausalLM;

KT_Qwen3MoeLayer* kt_qwen3moe_layer_new(const kt_qwen3moe_layer_config_t* config);
void kt_qwen3moe_layer_forward(KT_Qwen3MoeLayer* layer, size_t qlen, size_t kv_start_pos,
                                const void* input, void* output);
void kt_qwen3moe_layer_free(KT_Qwen3MoeLayer* layer);

typedef struct kt_qwen3moe_model_config_t {
    size_t num_layers;
    kt_qwen3moe_layer_config_t layer;     /* per-layer template */
    const void* final_norm_weight;        /* BF16 [hidden_size] */
    const void* lm_head;                  /* BF16 [vocab_size, hidden_size]; CausalLM only */
    size_t vocab_size;                    /* 0 = model-only */
} kt_qwen3moe_model_config_t;

KT_Qwen3MoeModel* kt_qwen3moe_model_new(const kt_qwen3moe_model_config_t* config);
void kt_qwen3moe_model_forward(KT_Qwen3MoeModel* model, size_t qlen, size_t kv_start_pos,
                                const void* input, void* output);
void kt_qwen3moe_model_free(KT_Qwen3MoeModel* model);

KT_Qwen3MoeCausalLM* kt_qwen3moe_causallm_new(const kt_qwen3moe_model_config_t* config);
void kt_qwen3moe_causallm_forward(KT_Qwen3MoeCausalLM* clm, size_t qlen, size_t kv_start_pos,
                                  const void* input, float* logits);
void kt_qwen3moe_causallm_free(KT_Qwen3MoeCausalLM* clm);

// ============================================================================
// LlamaMoe (Zig extension) — GGML-quantized MoE for GGUF checkpoints
// ============================================================================
// Ports the C++ LLAMA_MOE_TP from ktransformers/kt-kernel/operators/llamafile/moe.hpp.
// The framework's archive/experts.py uses this as the default backend for
// GGUF-loaded checkpoints (the .gguf weights are passed in directly, no
// re-quantization needed). Supported weight types: Q8_0, Q4_K, Q5_K, Q6_K, Q8_K.
// hidden_type must be KT_TYPE_BF16. The activation vec_dot_type is Q8_0.
//
// First cut: forward_one path (single-token decode). The qlen > 1 path
// falls back to per-token forward_one (correct, not optimal — see
// src/kernels/moe/llamafile_moe.zig for the followup).
//
// Pool requirement: a KT_WorkerPool* must be passed in (this differs from
// the C++ which uses a NUMA subpool index). Pass NULL to use the default
// subpool (the one created by kt_worker_pool_new).
typedef struct kt_llama_moe_config_t {
    /* dims */
    size_t expert_num;
    size_t num_experts_per_tok;
    size_t hidden_size;
    size_t intermediate_size;

    size_t layer_idx;
    void* pool;                       /* KT_WorkerPool* or NULL */

    /* GGML types (kt_type_t): {KT_TYPE_Q8_0, KT_TYPE_Q4_K, KT_TYPE_Q5_K, KT_TYPE_Q6_K, KT_TYPE_Q8_K} */
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    uint32_t hidden_type;              /* must be KT_TYPE_BF16 */

    /* tile geometry */
    size_t m_block;
    size_t group_min_len;
    size_t group_max_len;

    /* optional SGLang-style GPU offload mask (1 byte per expert, 1 = skip) */
    const uint8_t* gpu_experts_mask;

    /* weights: raw GGML blocks, byte-exact vs llama.cpp.
     *   gate_proj / up_proj:  [expert_num, intermediate, hidden] in <type> blocks
     *   down_proj:            [expert_num, hidden, intermediate] in <type> blocks */
    const void* gate_proj;
    const void* up_proj;
    const void* down_proj;
} kt_llama_moe_config_t;

typedef struct KT_LlamaMoe KT_LlamaMoe;

KT_LlamaMoe* kt_llama_moe_new(const kt_llama_moe_config_t* config);
void kt_llama_moe_free(KT_LlamaMoe* moe);
/* Copy weights from the caller's GGUF buffer into the per-expert
 * replicated storage. complete_intermediate_size is the full
 * intermediate dim across all TP ranks (the Zig port processes one
 * tp_part_idx = 0 for now). offset is the per-rank offset in
 * intermediate_size units. */
void kt_llama_moe_load_weights(KT_LlamaMoe* moe, int complete_intermediate_size, int offset);
/* Forward: qlen tokens, each with k expert_ids and weights.
 * input is BF16 [qlen, hidden], output is F32 [qlen, hidden]. */
void kt_llama_moe_forward(KT_LlamaMoe* moe, int qlen, int k, const int64_t* expert_ids,
                         const float* weights, const void* input, void* output);

#endif // KTRANSFORMERS_C_API_H