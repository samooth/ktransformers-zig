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
void kt_gate_forward(KT_Gate* gate, const void* input, void* output, int qlen);

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

#endif // KTRANSFORMERS_C_API_H