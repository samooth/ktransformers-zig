# C API Reference

The shared library `libkt_kernel_ext*.so` is the integration surface for
ktransformers-zig. The C header **`include/kt_kernel.h` is the contract** — this
document is a quick-reference map to it, organized by family. Full type
definitions (`kt_moe_config_t`, `kt_fp8_stats_t`, etc.) live in the header; the
function signatures here are signatures-of-record, not a replacement for the
header.

As of `tools/verify_abi.py` last PASS, the API is **105 symbols × 7 built
.so files** (6 x86 variants + 1 aarch64 `neon` cross-build). Any change to
a symbol's name, signature, or call convention is an ABI break and must
be gated behind the verifier.

## Family map

| Family | Symbols | What it does |
|---|---|---|
| [Version / detection](#version--detection) | 2 | `kt_version`, `kt_get_cpu_variant` |
| [BF16 conversion](#bf16-conversion) | 2 | `kt_bf16_to_f32`, `kt_f32_to_bf16` |
| [Worker pool](#worker-pool) | 3 | `kt_worker_pool_new(_config)`, `kt_worker_pool_free` |
| [CPUInfer](#cpuinfer) | 6 | `kt_cpuinfer_new(_config)`, `_free`, `_submit`, `_sync`, `_get_backend` |
| [Gate](#gate) | 3 | `kt_gate_new`, `kt_gate_free`, `kt_gate_forward` |
| [Linear](#linear) | 3 | `kt_linear_new`, `kt_linear_free`, `kt_linear_forward` |
| [MLP](#mlp) | 3 | `kt_mlp_new`, `kt_mlp_free`, `kt_mlp_forward` |
| [MoE (inference + SFT)](#moe) | 10 | `kt_moe_new(_sft)`, `_free`, `_warm_up`, `_load_weights(_with_map)`, `_forward`, `_forward_sft`, `_backward`, `_update_lora_weights` |
| [MLA (attention)](#mla) | 4 | `kt_mla_new`, `_load_weights`, `_free`, `_forward` |
| [Qwen3 MoE (model orchestration)](#qwen3-moe-model-orchestration) | 9 | `kt_qwen3moe_{layer,model,causallm}_{new,forward,free}` |
| [DeepseekV3 (model orchestration)](#deepseekv3-model-orchestration) | 9 | `kt_dsv3_{layer,model,causallm}_{new,forward,free}` |
| [Quantize / dequantize (per-row)](#quantize--dequantize) | 6 | `kt_quantize/dequantize_{int8, int4_gptq, fp8_e4m3}` |
| [GGML block formats (Q8_0 / Q4_K / Q5_K / Q6_K / Q8_K)](#ggml-block-formats) | 10 | `kt_quantize/dequantize_q{8_0,4_k,5_k,6_k,8_k}` |
| [GGML block matmul](#ggml-block-matmul) | 5 | `kt_matmul_q{8_0,4_k,5_k,6_k,8_k}` |
| [FP8 layerwise transport](#fp8-layerwise-transport) | 10 | `kt_fp8_layerwise_init`, `kt_fp8_transport_{new,free,close,closed,rank,tp_size,num_experts,run_producer,wait}` |
| [GGML init (no-op stub)](#ggml-init) | 1 | `kt_ggml_init` |

**105 symbols total** in the C ABI (header is the contract, verified by
`tools/verify_abi.py`). The math primitives
(`kt_apply_swiglu` / `kt_apply_rms_norm` / `kt_apply_rope` / `kt_softmax`)
exist as `export fn` in `src/main.zig` and are emitted into the .so, but are
**not declared in `include/kt_kernel.h`** — they're undocumented
auxiliaries. The verify_abi gate only checks header-declared symbols, so
they won't be caught if removed silently. Adding them to the header is a
follow-up TODO. They're listed in [Math helpers](#math-helpers) for
reference; treat them as advisory until the header is updated.

All `KT_*` opaque types are created by `kt_*_new` / `kt_*_new_config` calls
and must be released with the matching `kt_*_free` to avoid leaks. The
opaque-handle contract is: *null is the only valid value you may hand to
`*_free` for a no-op*; all other entry points expect a valid handle (null
is a programmer error and aborts the process via `@panic`).

---

## Version / detection

```c
const char* kt_version();            // → "0.6.1-zig" (semver + suffix)
const char* kt_get_cpu_variant();    // → "avx2" | "avx512_base" | "avx512_vnni" | "avx512_vbmi" | "avx512_bf16" | "amx" | "neon"
```

`kt_get_cpu_variant` matches the *built* variant, not the *best* one — call
`kt_version` first and dlopen the matching `libkt_kernel_ext_<variant>.so` from
`zig-out/lib/`. The Python ctypes wrapper in `python/kt_kernel/__init__.py`
does this autodetection.

---

## BF16 conversion

```c
void kt_bf16_to_f32(const void* src, float* dst, size_t count);
void kt_f32_to_bf16(const float* src, void* dst, size_t count);
```

The `void*` on the BF16 side is the standard "we treat them as opaque 16-bit
values" pattern; the byte layout is the BF16 half of an f32 (high 16 bits,
big-endian bit order). `count` is the number of elements (not bytes).

---

## Worker pool

```c
KT_WorkerPool* kt_worker_pool_new(int thread_count);
KT_WorkerPool* kt_worker_pool_new_config(kt_worker_pool_config_t config);  // NUMA-aware
void kt_worker_pool_free(KT_WorkerPool* pool);
```

`thread_count` ≤ 0 means "use the hardware parallelism". For NUMA topology
control (sub-pools pinned to specific nodes), use `new_config` with
`kt_worker_pool_config_t.subpool_count` / `subpool_numa_map` /
`subpool_thread_count` arrays. Worker pools are reference-counted by the
CPUInfer that owns them; free via the CPUInfer, not directly.

---

## CPUInfer

```c
KT_CPUInfer* kt_cpuinfer_new(int thread_count);
KT_CPUInfer* kt_cpuinfer_new_config(kt_worker_pool_config_t config);
void kt_cpuinfer_free(KT_CPUInfer* cpuinfer);

void kt_cpuinfer_submit(KT_CPUInfer* cpuinfer, void (*func)(void*), void* arg);
void kt_cpuinfer_sync(KT_CPUInfer* cpuinfer, size_t allow_n_pending);
KT_WorkerPool* kt_cpuinfer_get_backend(KT_CPUInfer* cpuinfer);  // escape hatch to the worker pool
```

Fire-and-forget task submission. `func` runs on a worker thread; the
caller's `arg` is a single pointer (use a struct to pass more). `kt_cpuinfer_sync`
waits for at most `allow_n_pending` in-flight tasks to complete (a low number
keeps tail latency bounded; pass `SIZE_MAX` to wait for all). `get_backend`
returns the internal worker pool — useful when you want to submit custom
work between CPUInfer calls.

---

## Gate

```c
KT_Gate* kt_gate_new(kt_gate_config_t config);
void kt_gate_free(KT_Gate* gate);
void kt_gate_forward(KT_Gate* gate, const void* input, void* output, int qlen);
```

Expert routing: scores each token against all experts, picks top-k,
returns per-expert routing weights. The current implementation is scalar
(`gemm_bf16.gemmExpert` for the scoring GEMM + naive O(n·k) top-k), BF16 in/out.

---

## Linear

```c
KT_Linear* kt_linear_new(kt_linear_config_t config);
void kt_linear_free(KT_Linear* linear);
void kt_linear_forward(KT_Linear* linear, const void* input, void* output, int qlen);
```

`out[m, n] = in[m, k] @ weight[n, k]^T` — single-projection GEMM. BF16
activations × BF16 weight → F32 output. `hidden_type` must be
`KT_TYPE_BF16` (other dtypes panic; the rest of the API surface for
non-BF16 is a Phase-2 follow-up).

---

## MLP

```c
KT_MLP* kt_mlp_new(kt_mlp_config_t config);
void kt_mlp_free(KT_MLP* mlp);
void kt_mlp_forward(KT_MLP* mlp, const void* input, void* output, int qlen);
```

SwiGLU MLP: gate GEMM + up GEMM + F32 SwiGLU + BF16 round-trip + down
GEMM. Single forward call; the layer's three projections come from the
config (gate_proj / up_proj / down_proj).

---

## MoE

```c
KT_MOE* kt_moe_new(KT_CPUInfer* cpuinfer, kt_moe_config_t config);
KT_MOE* kt_moe_new_sft(KT_CPUInfer* cpuinfer, kt_moe_sft_config_t config);  // SFT/LoRA path
void kt_moe_free(KT_MOE* moe);

void kt_moe_warm_up(KT_MOE* moe);                                          // no-op placeholder
void kt_moe_load_weights(KT_MOE* moe);
void kt_moe_load_weights_with_map(KT_MOE* moe, uint64_t* physical_to_logical_map);  // logical-slot remap

void kt_moe_forward(KT_MOE* moe, int qlen, int k,
                    const int64_t* expert_ids, const float* weights,
                    const void* input, void* output, int incremental);

void kt_moe_forward_sft(KT_MOE* moe, int qlen, int k,
                        const int64_t* expert_ids, const float* weights,
                        const void* input, void* output, int save_for_backward);
void kt_moe_update_lora_weights(KT_MOE* moe,
                                void* gate_lora_a, void* gate_lora_b,
                                void* up_lora_a,   void* up_lora_b,
                                void* down_lora_a, void* down_lora_b);
void kt_moe_backward(KT_MOE* moe,
                     const void* grad_output, void* grad_input,
                     void* grad_gate_lora_a, void* grad_gate_lora_b,
                     void* grad_up_lora_a,   void* grad_up_lora_b,
                     void* grad_down_lora_a, void* grad_down_lora_b,
                     void* grad_weights,
                     void* grad_gate_proj, void* grad_up_proj, void* grad_down_proj,
                     int accumulate_optimizer_grads, float optimizer_grad_scale);
```

**Lifecycle**: `new` (or `new_sft` for training) → `load_weights` (or
`load_weights_with_map` for the physical→logical expert remap used by the K2
reference) → `forward` (any number of times) → `free`. `warm_up` is a no-op
placeholder; load weights implicitly warms up the GEMM tiles.

**Forward**: `expert_ids` is `[qlen, k]` (row-major; the k experts each token
is routed to), `weights` is `[qlen, k]` (the routing weights), `input` is
`[qlen, hidden]` BF16, `output` is `[qlen, hidden]` BF16. Set `incremental`
non-zero to do an in-place accumulate (avoid clearing the output first).

**SFT**: `forward_sft` with `save_for_backward=1` populates an internal
cache; `backward` then consumes it to compute the six LoRA grads plus base
weight grads. `update_lora_weights` swaps the active LoRA matrices without
rebuilding the MoE.

---

## Qwen3 MoE (model orchestration)

Layer / model / CausalLM orchestration for Qwen3-style MoE
(Qwen3-30B-A3B, Qwen3.5-35B-A3B, Qwen3-Coder-Next). Standard MHA + GQA
+ RoPE + pre-norm + **vanilla softmax top-k gate** (no group-top2, no
e_score_correction_bias — those are DeepSeek-V3-specific). Targets
the same MoE-FFN as DeepSeek-V3 (gate/up/down, SwiGLU, BF16 weights).

```c
KT_Qwen3MoeLayer* kt_qwen3moe_layer_new(const kt_qwen3moe_layer_config_t* config);
void  kt_qwen3moe_layer_forward(KT_Qwen3MoeLayer* layer, size_t qlen, size_t kv_start_pos,
                                 const void* input, void* output);
void  kt_qwen3moe_layer_free(KT_Qwen3MoeLayer* layer);

KT_Qwen3MoeModel* kt_qwen3moe_model_new(const kt_qwen3moe_model_config_t* config);
void  kt_qwen3moe_model_forward(KT_Qwen3MoeModel* model, size_t qlen, size_t kv_start_pos,
                                 const void* input, void* output);
void  kt_qwen3moe_model_free(KT_Qwen3MoeModel* model);

KT_Qwen3MoeCausalLM* kt_qwen3moe_causallm_new(const kt_qwen3moe_model_config_t* config);
/* logits: f32 [qlen, vocab_size] */
void  kt_qwen3moe_causallm_forward(KT_Qwen3MoeCausalLM* clm, size_t qlen, size_t kv_start_pos,
                                    const void* input, float* logits);
void  kt_qwen3moe_causallm_free(KT_Qwen3MoeCausalLM* clm);
```

**Config (pointer-passed, not by-value)**: every weight pointer is
caller-owned (borrowed). The model and CausalLM share the layer config
struct — the CausalLM is just `Model + lm_head` (the layer config is
embedded by value in `kt_qwen3moe_model_config_t.layer`; `final_norm_weight`
and `lm_head` are the model-level additions).

**RoPE default**: Qwen3 uses `rope_theta = 1,000,000.0` (vs DeepSeek-V3's
10,000). The default in the C-API struct is 1,000,000.0; you only need to
override it if you're running a Qwen3 checkpoint that was trained with a
different value.

**GQA**: `num_kv_heads` may be less than `num_heads` (GQA). The KV cache is
sized for `num_kv_heads`; K and V tensors are shared across query-head
groups in the attention layer (every `num_heads/num_kv_heads`-th query
head reads the same K/V).

**Forward**: input/output are BF16 `[qlen, hidden]`. `kv_start_pos` is
the starting position in the (continuous) KV cache for the new query
tokens — pass 0 for the first decode step, then 1, 2, 3, ... for
subsequent decode steps; pass `(N, qlen=N)` for prefill in one shot. The
layer handles both single-token decode (`qlen=1`) and multi-token prefill
(`qlen>1`).

**Lifecycle** (same pattern for all three):

```c
kt_qwen3moe_layer_config_t cfg = {
    .hidden_size = 4096, .num_heads = 32, .num_kv_heads = 4,
    .head_dim = 128, .max_qlen = 1, .max_kvlen = 32768,
    .rope_theta = 1000000.0,
    .expert_num = 128, .num_experts_per_tok = 8,
    .intermediate_size = 12288, .pool = NULL,
    .q_proj = q_weight_ptr, .k_proj = ..., /* etc. */
};
KT_Qwen3MoeLayer* layer = kt_qwen3moe_layer_new(&cfg);
uint16_t input[4096] = {...};  // BF16 hidden state (1 token)
uint16_t output[4096];
kt_qwen3moe_layer_forward(layer, 1, /*kv_start_pos=*/0, input, output);
kt_qwen3moe_layer_free(layer);
```

**Differences vs DeepSeek-V3** (`kt_dsv3_*`):

| Feature | Qwen3 MoE | DeepSeek-V3 |
|---|---|---|
| Attention | Standard MHA + GQA + RoPE | MLA (latent compression) |
| Routing | softmax over experts, top-k, no bias, no grouping | sigmoid + e_score_correction_bias + group-top2 + scale |
| Normalization | pre-norm (RMSNorm → MHA → residual → RMSNorm → MoE → residual) | pre-norm (same pattern) |
| FFN | SwiGLU (gate × up, clamped) | SwiGLU (same) |
| RoPE θ | 1,000,000 | 10,000 |
| Layer config | by-pointer (single `kt_qwen3moe_layer_config_t*`) | by-pointer (same) |

**pybind11 surface** (Tier 1): `from kt_kernel_ext.qwen3moe import
Qwen3MoeDecoderLayer, Qwen3MoeModel, Qwen3MoeForCausalLM, LayerConfig,
ModelConfig`. The shim's mirror structs are validated against the Zig
extern structs by `tools/audit_layout.py` (5/5 model-orchestration
configs, 0 divergences).

**Why not Qwen3-Next hybrid attention (DCA/DeltaNet)?** That's a
separate workstream — see `TODO_QWEN.md` for the open questions. The
Qwen3 standard-MHA path covers Qwen3-30B-A3B, Qwen3.5-35B-A3B, and
Qwen3-Coder-Next (which uses the same vanilla attention).

---

## MHA (reusable for any non-MLA model)

The `MhaEngine` in `src/kernels/attn/mha.zig` is the standard multi-head
attention with GQA + RoPE + continuous KV cache. It's wired into the
`Qwen3MoeDecoderLayer` (`kt_qwen3moe_layer_*`) but is not exposed as a
top-level C API surface — the model-orchestration layer is the entry
point. The lower-level math helpers (rmsNormInline, matmulF32,
softmaxInPlace) ARE exported in the .so (see
[Math helpers](#math-helpers)); call them directly if you need to
assemble a custom layer outside the Qwen3 surface.

**MHA engine contract** (for callers that bypass the model orchestration):

```zig
const cfg = mha.MhaConfig{
    .hidden_size = 4096, .num_heads = 32, .num_kv_heads = 8, // GQA: 4 query heads per KV
    .head_dim = 128, .max_qlen = 1, .max_kvlen = 32768,
    .rope_theta = 1000000.0,                              // Qwen3 default
    .q_proj = @ptrCast(q_weight), .k_proj = @ptrCast(k_weight),
    .v_proj = @ptrCast(v_weight), .o_proj = @ptrCast(o_weight),
};
var cache = try mha.MhaKvCache.init(allocator, cfg.num_kv_heads, cfg.head_dim, cfg.max_kvlen);
defer cache.deinit();
var engine = try mha.MhaEngine.init(allocator, cfg, &cache);
defer engine.deinit();

var input: [cfg.hidden_size]f32 = ...;
var output: [cfg.hidden_size]f32 = ...;
engine.decode(&input, &output, /*position=*/0);  // single-token decode
// For multi-token prefill:
engine.forward(&input, &output, /*qlen=*/N, /*kv_start_pos=*/0);
```

**Helpers** (in `src/kernels/attn/mha.zig`, exposed via the C extension
section in `src/main.zig`):

| Function | Signature | Purpose |
|---|---|---|
| `mha.matmulF32` | `(input, lda, weight, ldb, output, ldc, m, n, k)` | BF16 weight matmul → F32 output |
| `mha.rmsNormInline` | `(x, weight, out, eps)` | In-place RMSNorm with BF16 weight |
| `mha.softmaxInPlace` | `(buf, size)` | Numerically stable in-place softmax |

These are also useful for assembling a Gemma or non-Qwen3 dense layer
on top of the same primitives (see the "When NOT to Add a New Model
Layer" section in `docs/porting-guide.md`).

---

## DeepseekV3 (model orchestration)

Layer / model / CausalLM orchestration for DeepSeek-V3 (and V2 — same
schema). MLA attention + sigmoid+group-top2 routing + standard MoE FFN.

```c
KT_DSV3Layer* kt_dsv3_layer_new(const kt_dsv3_layer_config_t* config);
void kt_dsv3_layer_forward(KT_DSV3Layer* layer, size_t qlen, size_t kv_start_pos,
                           const void* input, void* output);
void kt_dsv3_layer_free(KT_DSV3Layer* layer);

KT_DSV3Model* kt_dsv3_model_new(const kt_dsv3_model_config_t* config);
void kt_dsv3_model_forward(KT_DSV3Model* model, size_t qlen, size_t kv_start_pos,
                           const void* input, void* output);
void kt_dsv3_model_free(KT_DSV3Model* model);

KT_DSV3CausalLM* kt_dsv3_causallm_new(const kt_dsv3_model_config_t* config);
void kt_dsv3_causallm_forward(KT_DSV3CausalLM* clm, size_t qlen, size_t kv_start_pos,
                             const void* input, float* logits);
void kt_dsv3_causallm_free(KT_DSV3CausalLM* clm);
```

Same pointer-passed config pattern as Qwen3 (see above). The
Differences-vs-Qwen3 table in the Qwen3 section summarizes the
architectural deltas (MLA vs MHA, sigmoid+group-top2 vs softmax
top-k, RoPE θ=10000 vs 1000000).

---

## MLA

```c
KT_MLA* kt_mla_new(kt_mla_config_t config);
void kt_mla_free(KT_MLA* mla);
void kt_mla_load_weights(KT_MLA* mla);  // must precede forward — reference: "Not Loaded"
void kt_mla_forward(KT_MLA* mla, const int* qlens, int qlen_count,
                    const void* input, void* output);
```

Multi-head Latent Attention (DeepSeek-V2/V3). The forward signature takes
a list of per-batch sequence lengths (`qlens[qlen_count]`) for variable-length
batched prefill; pass `qlen_count=1, qlens[0]=N` for uniform batches or
decode. The engine holds its own KV cache internally; the external `kv_cache`
parameter that some signatures have is accepted for ABI stability but ignored
(the engine's internal cache is the source of truth).

**Failure mode**: calling `forward` before `load_weights` aborts — this mirrors
the C++ reference where `forward()` throws "Not Loaded".

---

## Quantize / dequantize (per-row)

```c
void kt_quantize_int8_per_row(const void* src, int8_t* dst, float* scales, size_t rows, size_t cols);
void kt_dequantize_int8(const int8_t* src, const float* scales, void* dst, size_t rows, size_t cols);
void kt_quantize_int4_gptq(const void* src, uint8_t* dst, float* scales, uint8_t* zeros, size_t rows, size_t cols, int group_size);
void kt_dequantize_int4_gptq(const uint8_t* src, const float* scales, const uint8_t* zeros, void* dst, size_t rows, size_t cols, int group_size);
void kt_quantize_fp8_e4m3(const void* src, uint8_t* dst, float* scales, size_t rows, size_t cols, int block_size);
void kt_dequantize_fp8_e4m3(const uint8_t* src, const float* scales, void* dst, size_t rows, size_t cols, int block_size);
```

BF16 in/out (the `void*` on the BF16 side). INT8: one scale per row. INT4 GPTQ:
one scale/zero per `group_size` elements. FP8 E4M3: one scale per
`block_size` elements (commonly 128, matching llama.cpp).

---

## GGML block formats

These match the byte layouts in `llama.cpp ggml-common.h` exactly, so
weights loaded from a `.gguf` file can be passed straight in.

```c
void kt_quantize_q8_0(const float* src, void* dst, size_t k);
void kt_dequantize_q8_0(const void* src, float* dst, size_t k);
void kt_quantize_q4_k(const float* src, void* dst, size_t k);
void kt_dequantize_q4_k(const void* src, float* dst, size_t k);
void kt_quantize_q5_k(const float* src, void* dst, size_t k);
void kt_dequantize_q5_k(const void* src, float* dst, size_t k);
void kt_quantize_q6_k(const float* src, void* dst, size_t k);
void kt_dequantize_q6_k(const void* src, float* dst, size_t k);
void kt_quantize_q8_k(const float* src, void* dst, size_t k);  // Q8_K
void kt_dequantize_q8_k(const void* src, float* dst, size_t k);  // Q8_K
```

`k` must be a multiple of the format's super-block width (32 for Q8_0, 256
for the K-quants). The block sizes:

| Format | Bytes per block | Weights per block | Notes |
|---|---|---|---|
| Q8_0  | 34   | 32  | f16 d, 32×i8 qs |
| Q4_K  | 144  | 256 | f16 d+dmin, 12-byte 6-bit scales, 128-byte 4-bit qs |
| Q5_K  | 176  | 256 | f16 d+dmin, 12-byte 6-bit scales, 32-byte qh, 128-byte qs |
| Q6_K  | 210  | 256 | ql[128] + qh[64] + scales[16] + f16 d |
| Q8_K  | 292  | 256 | **f32 d** (the only K-quant with f32 super-block scale), 256×i8 qs, 16×i16 bsums |

Quantize one row at a time (`k` is the row length in elements). The dequant
functions reconstruct an F32 row.

---

## GGML block matmul

```c
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
```

`C[m, n] = A[m, k] × B[n, k]ᵀ` where:
- `A` is `[m, k]` BF16 activations (row-major, leading dim `lda`)
- `B` is `[n, k/format_block_size]` quantized weights in **row-major with leading dim in BLOCKS** (e.g. `ldb = k/256` for the K-quants) — this matches how GGUF stores them
- `C` is `[m, n]` F32 output (row-major, leading dim `ldc`)

On-the-fly dequantization happens inside the matmul (scalar kernels today;
AMX-vectorized on Sapphire Rapids+ on the `amx` variant). `k` must be a
multiple of the format's super-block width.

**AMX INT4 partial-group support**: `gemm_224_int4.zig::gemmFullTile`
iterates by `GROUP_SIZE` (32 K elements) and applies the per-group
scale immediately, so any K value is supported (not just multiples of
32). Before this fix, K values that didn't divide evenly by 32
silently returned 0 because the trailing partial group was dropped.
The scalar fallback (`gemmFullTileScalar`) handles the partial
trailing group explicitly (both even and odd `k_tail`).

### I-quants (grid-based, sub-3-bit)

The non-linear I-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL,
IQ4_XS) are also byte-exact vs llama.cpp and live in
`src/kernels/amx/gemm_224_iq*.zig`. They use grid-based magnitude
lookups (256–512 entry precomputed tables, fingerprint-matched via the
2-bit kmap from `src/kernels/amx/iq2xs_init.zig`) plus per-block
scales and per-weight sign bytes. Quantize is real for the 2-bit
formats (IQ2_XXS, IQ2_XS); the rest have quantize stubbed in
llama.cpp too and only dequant + GEMM are ported here. The byte
layouts match `ggml-common.h` so a `.gguf` with I-quant weights
drops in without preprocessing. Useful when you need to load a
sub-3-bpw GGUFs (e.g. the IQ2_XXS GGUFs in `/ai/models/`) and want
to do the dequant inside the matmul.

---

## FP8 layerwise transport

CPU-port, callback-only plumbing. The transport stores per-rank host-buffer
state and, in `run_producer`, hands the caller's callback the per-expert
FP8 weight pointers — it does **not** run GEMMs itself (the caller does,
via `kt_matmul_fp8`). `cuda_device` / `local_gpu_ptrs` are accepted for ABI
compatibility and ignored (no GPU in the CPU port).

```c
void kt_fp8_layerwise_init(void* control_ptr, size_t control_size, int tp_size);

KT_FP8LayerwiseTransport* kt_fp8_transport_new(
    void* control_ptr, size_t control_size,
    int rank, int tp_size, int cuda_device,                // cuda_device: ignored
    const uintptr_t* local_host_ptrs,                      // 8 entries: [slot][kind]
    const uintptr_t* local_gpu_ptrs,                      // ignored (CPU port)
    const uintptr_t* all_rank_host_ptrs,                   // rank 0 only: [slot][rank][kind] = 2*tp_size*4
    const size_t* expert_nbytes,                          // 4 per-kind byte sizes
    int num_experts, int timeout_ms);                      // timeout_ms: ignored (in-process)

void kt_fp8_transport_free(KT_FP8LayerwiseTransport* transport);
void kt_fp8_transport_close(KT_FP8LayerwiseTransport* transport);
void kt_fp8_transport_run_producer(KT_FP8LayerwiseTransport* transport,
                                     uint64_t epoch, int layer_id, int expert_count,
                                     void (*callback)(int expert_id,
                                                      const uintptr_t* w13_weight_ptrs,
                                                      const uintptr_t* w13_scale_ptrs,
                                                      const uintptr_t* w2_weight_ptrs,
                                                      const uintptr_t* w2_scale_ptrs));
kt_fp8_stats_t kt_fp8_transport_wait(KT_FP8LayerwiseTransport* transport, uint64_t epoch);
int kt_fp8_transport_rank(KT_FP8LayerwiseTransport* transport);
int kt_fp8_transport_tp_size(KT_FP8LayerwiseTransport* transport);
int kt_fp8_transport_num_experts(KT_FP8LayerwiseTransport* transport);
int kt_fp8_transport_closed(KT_FP8LayerwiseTransport* transport);
```

**Control region**: an opaque 8192-byte buffer the caller allocates and
hands in. `kt_fp8_layerwise_init` validates (64-byte alignment, size ≥ 8192,
tp_size ∈ [1, 8]) and stamps a magic/version header. `kt_fp8_transport_new`
re-validates the header; the two values must match. Control region shape and
validation match the C++ reference (`fp8_layerwise_transport.cpp`) for ABI
compatibility — the full multi-process slot/ack/poison protocol isn't used in
the CPU port.

**Lifecycle**: `init` → `new` (rank 0 only; per-rank transports are creatable
but the producer is rank-zero only) → `run_producer(epoch, layer_id, count, cb)`
→ `wait(epoch)` returns the recorded stats → `close` → `free`.

**Per-expert pointer math** (in `run_producer`, slot = `expert_id % 2`,
4 kinds = w13_weight, w13_scale, w2_weight, w2_scale):

```
all_rank_host_ptrs[(slot * tp_size + rank) * 4 + kind]   // slot/rank/kind index
```

The callback receives four arrays of `tp_size` host pointer values, one per
(kind, rank) pair. Slot=0 receives even experts, slot=1 receives odd ones
(consumer is expected to drain the slot before the next reuse).

**Stats (`kt_fp8_stats_t`)** records CPU-measured timings (`writer_ms`,
`slot_wait_ms=0`, `total_ms`; `h2d_ms=0` since no GPU), `bytes = expert_count × Σ expert_nbytes`,
and error fields (`poisoned`, `error_code`, `error_rank`, `error_message`)
populated on protocol violations (e.g. `wait` called for a never-produced
epoch). See `tests/kernels/fp8_transport_tests.zig` for a worked example.

---

## GGUF parser

`src/io/gguf.zig` — read-only GGUF v3 header parser. Byte-exact vs the
llama.cpp `ggml.h` format. Not exposed through the C API (callers use
the C bindings below to consume the tensors). Useful for loading
weights from a `.gguf` file and passing them to the matmul functions.

**Zig usage** (the Python ctypes wrapper in `python/kt_kernel/__init__.py`
exposes the higher-level `matmul_quantized` API; for raw file loading
use the Zig module directly):

```zig
const gguf = @import("io/gguf.zig");

// Read the file (mmap recommended for large files; the parser
// only walks 0..data_offset).
const file_bytes = try allocator.alignedAlloc(u8, .@"64", file_size);
const fd = try std.posix.openat(AT.FDCWD, path, .{.RDONLY}, 0);
defer std.posix.close(fd);
_ = std.posix.read(fd, file_bytes);

var h = try gguf.parse(file_bytes, file_size, allocator);
defer h.deinit();

// Find a tensor by name (linear search; GGUF doesn't require
// sorted tensors).
const idx = h.findTensor("blk.0.attn_q.weight") orelse return;
// Tensors carry: name, n_dims, dims[0..n_dims], tensor_type, offset.
// `offset` is the byte offset into the file; the bytes at
// [offset, offset+block_bytes) are the packed tensor data
// (block_bytes depends on tensor_type + dims).
```

**KV metadata**: `h.getKv("general.architecture")` returns the metadata
string (or `null` if missing). Supports `string`, `u32`/`i32`/`u8`/`i16`,
and `f32`/`f64` types.

**Tensor types** (`gguf.TensorType`): byte-exact vs llama.cpp —
F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9, Q2_K=10,
Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15, IQ2_XXS=16, IQ2_XS=17,
IQ3_XXS=18, IQ1_S=19, IQ4_NL=20, IQ3_S=21, IQ2_S=22, IQ4_XS=23, IQ1_M=24,
BF16=25, MXFP4=28, MXFP8=29. Values match the `kt_type_t` enum in
`include/kt_kernel.h` for direct comparison.

**Real-model E2E test**: `tests/kernels/qwen3_gguf_e2e_tests.zig`
opens `/ai/models/Qwen3.5-0.8B-BF16.gguf` via `mmap(2)` and verifies
the parser produces a sensible `Header` (version=3, ≥150 tensors,
`token_embd.weight` present, per-block `attn_*`/`ffn_*` names decoded
correctly). This is the only test in the suite that exercises a real
on-disk file.

---

## GGUF init

```c
void kt_ggml_init();  // no-op stub
```

Placeholder for the C++ `ggml_init` ABI. Calling it is a no-op; it's there
so existing ktransformers Python code that always calls `kt_ggml_init()` at
import time doesn't break.

---

## Math helpers

The internal kernels (BF16 round-trip, SwiGLU, RMSNorm, RoPE, softmax)
are also exposed as exports in the .so:

```c
void kt_apply_swiglu(void* gate, void* up, void* dst, size_t count, float limit, float alpha);
void kt_apply_rms_norm(void* input, void* weight, void* output, size_t hidden_size, float eps);
void kt_apply_rope(void* q, void* k, int64_t position, size_t head_dim, double rope_theta);
void kt_softmax(const float* input, float* output, size_t size);
```

These are **not** declared in `include/kt_kernel.h` and therefore not
covered by the ABI verifier. They exist as `export fn` in `src/main.zig`
and are present in the .so, but treat them as undocumented — they may
change or disappear between versions. The GEMM entry points
(`kt_matmul_q*`) use these internally; you only need them if you're
assembling a custom layer outside the C API surface.

---

## Patterns

### Minimal MoE forward (Python ctypes flavor)

```python
import ctypes
lib = ctypes.CDLL("libkt_kernel_ext.so")
lib.kt_version.restype = ctypes.c_char_p
print(lib.kt_version())  # b"0.6.1-zig"

cpu = lib.kt_cpuinfer_new(0)  # 0 = auto thread count
moe = lib.kt_moe_new(cpu, moe_config)
lib.kt_moe_load_weights(moe)

qlen, k = 8, 2
expert_ids = (ctypes.c_int64 * (qlen * k))(...)  # [qlen, k] row-major
weights    = (ctypes.c_float  * (qlen * k))(...)  # [qlen, k] row-major
input      = (ctypes.c_uint16 * (qlen * hidden))(...)  # BF16 raw
output     = (ctypes.c_uint16 * (qlen * hidden))(...)

lib.kt_moe_forward(moe, qlen, k, expert_ids, weights, input, output, 0)
lib.kt_moe_free(moe)
lib.kt_cpuinfer_free(cpu)
```

### Variant selection

The `.so` you load must match your CPU. The Python ctypes wrapper in
`python/kt_kernel/__init__.py` auto-detects via `kt_get_cpu_variant()` and
picks the right `libkt_kernel_ext_<variant>.so` from `zig-out/lib/`. For
manual dlopen:

```python
import os, ctypes
v = ctypes.CDLL("libkt_kernel_ext.so").kt_get_cpu_variant().decode()
# "avx2" | "avx512_vnni" | ... | "neon"
lib = ctypes.CDLL(f"libkt_kernel_ext_{v}.so")
```

For aarch64, only the `neon` variant is available (scalar GEMM fallbacks;
NEON-vectorized kernels are Phase 2).

### Cross-builds

```bash
zig build -Dvariant=avx2                 # x86_64 fallback (default)
zig build -Dvariant=amx                  # Sapphire Rapids+ AMX
zig build all-variants                   # all 6 x86 .so into zig-out/lib/
zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu
                                          # libkt_kernel_ext_neon.so (ARM64 ELF)
```

The neon variant requires an aarch64 target — passing `-Dvariant=neon` without
`-Dtarget=aarch64-linux-gnu` exits with a clear error rather than silently
building a meaningless x86 "neon" library.

---

## ABI contract

- **Header is the contract**: `include/kt_kernel.h`. Don't change symbol names,
  signatures, or call conventions without bumping the version and coordinating.
- **`tools/verify_abi.py`** parses the header and checks every declared `kt_*`
  is exported (via `nm -D`) by every built `.so`. CI gate. `tools/verify_abi.py`
  alone is the source of truth — keep it green.
- **Multi-variant layout**: 6 x86 variants (`avx2`, `avx512_base`, `avx512_vnni`,
  `avx512_vbmi`, `avx512_bf16`, `amx`) plus 1 aarch64 `neon` cross-build. Each
  installs as `libkt_kernel_ext_<variant>.so` except the default `avx2` which
  installs as `libkt_kernel_ext.so` (the historical un-suffixed name).
- **Opaque handles** (`KT_MOE`, `KT_MLA`, etc.) are file-private structs in
  Zig cast to opaque pointers. Treat them as opaque from C; the only
  valid operation on a null handle is passing it to the matching `*_free`
  for a no-op. Any other use of a null handle is a programmer error.

---

## See also

- `include/kt_kernel.h` — full type definitions + signatures-of-record.
- `docs/porting-guide.md` — internal C++ → Zig file map; for contributors.
- `LESSONS_ZIG.md` — verified Zig 0.16 gotchas (architecture, inline asm,
  allocator patterns, std.simd, etc.).
- `AGENTS.md` — repo conventions for new contributors.
- `TODO.md` — open work; read `#101`, `#105`, `#106` for context on
  what's intentionally out of scope or staged.

## Known gaps (this doc, not the C header)

- The math primitives in [Math helpers](#math-helpers) are emitted into
  the .so but not declared in `include/kt_kernel.h`. Adding them to the
  header would close the gap and let `tools/verify_abi.py` gate them too.
- `tools/verify_abi.py` only scans `zig-out/lib/`. If you build a `.so`
  by hand and copy it in, rename it to match the `libkt_kernel_ext_*`
  pattern, or remove it before running the verifier.
