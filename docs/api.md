# C API Reference

The shared library `libkt_kernel_ext*.so` is the integration surface for
ktransformers-zig. The C header **`include/kt_kernel.h` is the contract** — this
document is a quick-reference map to it, organized by family. Full type
definitions (`kt_moe_config_t`, `kt_fp8_stats_t`, etc.) live in the header; the
function signatures here are signatures-of-record, not a replacement for the
header.

As of `tools/verify_abi.py` last PASS, the API is **68 symbols × 7 built .so
files** (6 x86 variants + 1 aarch64 `neon` cross-build). Any change to a symbol's
name, signature, or call convention is an ABI break and must be gated behind
the verifier.

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
| [Quantize / dequantize (per-row)](#quantize--dequantize) | 6 | `kt_quantize/dequantize_{int8, int4_gptq, fp8_e4m3}` |
| [GGML block formats (Q8_0 / Q4_K / Q5_K / Q6_K / Q8_K)](#ggml-block-formats) | 10 | `kt_quantize/dequantize_q{8_0,4_k,5_k,6_k,8_k}` |
| [GGML block matmul](#ggml-block-matmul) | 5 | `kt_matmul_q{8_0,4_k,5_k,6_k,8_k}` |
| [FP8 layerwise transport](#fp8-layerwise-transport) | 10 | `kt_fp8_layerwise_init`, `kt_fp8_transport_{new,free,close,closed,rank,tp_size,num_experts,run_producer,wait}` |
| [GGML init (no-op stub)](#ggml-init) | 1 | `kt_ggml_init` |

**68 symbols total** in the C ABI (header is the contract, verified by
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

## GGML init

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
