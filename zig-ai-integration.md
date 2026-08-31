# zig-ai × ktransformers-zig Integration Plan

**Confirmed direction:** build a **layered hybrid engine** where **zig-ai is the host**
(GPU attention + speculative decode) and **ktransformers-zig is a CPU kernel backend**
(delivered as `libkt_kernel_ext_*.so`, consumed via its C API).

This document is the actionable plan for that integration. Every file reference below
was verified by inspection of both repos.

---

## 1. Does it make sense?

Yes — the two projects are complementary halves of a hybrid inference engine.

| | ktransformers-zig | zig-ai |
|---|---|---|
| **Role** | CPU-offload kernel library | Full GPU inference engine |
| **Language** | Zig → `.so` | Zig + CUDA |
| **What it owns** | MoE expert GEMMs, MLA attention, FP8 transport, quantized GEMM (BF16/INT8/INT4/FP8/MXFP), linear/mlp/gate | Paged attention (decode/prefill), KV quantization, speculative decoding (MTP), benchmark harness |
| **Value prop** | Offload the huge MoE expert weights + MLA to CPU; attention elsewhere | Fast GPU decode + MTP; measures TTFT / tok/s |

ktransformers' original story is **hybrid CPU/GPU**: offload MoE experts to CPU while
attention runs on GPU. `ktransformers-zig` is a drop-in `.so` for that exact role. `zig-ai`
is a Zig GPU engine with paged attention and speculative decoding. **Layering them —
GPU attention+MTP on zig-ai, CPU-offloaded MoE+MLA on the ktransformers-zig `.so` — is the
natural hybrid engine.** The C API is already the integration seam.

**When it doesn't make sense:** a *source merge* is the wrong move right now. The CPU
port isn't finished (`moe_sft` and `kt_fp8_*` are in flight), and the two codebases have
different build systems and memory models. Integrate at the `.so` boundary first; merge
source only much later.

---

## 2. Architecture

```
┌───────────────────────────── zig-ai (host) ─────────────────────────────┐
│  PagedAttention (decode/prefill)  │  MTP speculative decode (CUDA)      │
│  KV quantization (q4/q8)          │  Benchmark harness (TTFT, tok/s)    │
│                                   │  build.zig (Zig + CUDA toolchain)   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ dlopen / FFI
                    ┌───────────┴───────────┐  ← the seam: ktransformers-zig C API
                    │  libkt_kernel_ext_*.so │    (53–59 exported kt_* symbols)
                    │  MoE forward/backward │    built by ktransformers-zig
                    │  MLA attention        │
                    │  FP8 layerwise xport  │
                    │  Quantized GEMM       │
                    │  Linear / MLP / Gate  │
                    └───────────────────────┘
```

- **ktransformers-zig** produces a set of variant `.so`s (`avx2`, `avx512_base`,
  `avx512_vnni`, `avx512_vbmi`, `avx512_bf16`, `amx`). At runtime the matching variant
  is loaded (the existing Python wrapper already auto-detects the CPU variant from
  `/proc/cpuinfo`).
- **zig-ai** hosts the model: it owns attention and speculative decode, and calls into the
  `.so` for MoE layers, MLA projections, and optionally FP8 weight transport.
- The dependency points **one way**: zig-ai depends on the `.so`; ktransformers-zig knows
  nothing about zig-ai.

---

## 3. The seam — ktransformers-zig C API

The `.so` exports a stable C API (declared in `include/kt_kernel.h`). Verified by
`tools/verify_abi.py`: **53 `kt_*` symbols**, all exported by every variant `.so`. The
relevant families for the integration:

- **MoE** — `kt_moe_new` / `kt_moe_new_with_pool` / `kt_moe_forward` / `kt_moe_forward_sft`
  / `kt_moe_backward` / `kt_moe_update_lora_weights` / `kt_moe_free`
- **MLA attention** — `kt_mla_new` / `kt_mla_forward` / `kt_mla_prefill` / `kt_mla_decode` /
  `kt_mla_update_kv_cache` / `kt_mla_free` *(MLA bodies are still placeholder in this
  port — see §6)*
- **Linear / MLP / Gate** — `kt_linear_forward`, `kt_mlp_forward`, `kt_gate_forward`
- **FP8** — `kt_fp8_quantize` / `kt_fp8_dequantize` / `kt_fp8_quantize_block` /
  `kt_fp8_dequantize_block` / `kt_matmul_fp8` / `kt_quantize_fp8_e4m3` /
  `kt_dequantize_fp8_e4m3` / `kt_fp8_layerwise_init` / `kt_fp8_transport_*`
- **Utilities** — `kt_version`, `kt_get_cpu_variant`, CPU-detect, NUMA topology

The full authoritative list is `include/kt_kernel.h` (511 lines); `tools/verify_abi.py`
asserts every declared `kt_*` is exported. **This C API is the integration contract — keep
it stable.**

---

## 4. Phased plan

### Phase 0 — Measure (lowest risk, do now)

Use **zig-ai's benchmark harness** to measure the `ktransformers-zig` `.so`. This produces
real TTFT / tok/s numbers for the Zig CPU kernels and validates the port — no source merge.

This also answers the original "what is the Time to first token" question: **neither repo
ships a committed numeric TTFT** — both ship *harnesses* that measure it. To get a number,
you run them.

zig-ai harness entry points (verified):
- `benchmarks/bench_baseline.sh` — greps `prefill` ms + tok/s from a run log.
- `benchmarks/bench_sweep.sh` — prefill×decode sweep, parses prefill ms and tok/s.
- `benchmarks/bench_paged_attention.zig` — measures prefill latency/throughput and
  decode latency (ms/token) + throughput (tok/s) vs batch size.
- `python/cli/commands/chat.py` — computes `ttft = first_token_time - start_time` and
  prints `TTFT: {ttft*1000:.0f}ms`.

Concrete first step: write a small driver that loads the ktransformers-zig `.so` (via its
existing `python/kt_kernel` ctypes wrapper or via `zig-ai`'s `std.DynLib`), runs a MoE/MLA
forward, and reports latency/throughput in the same format as `bench_paged_attention.zig`.

### Phase 1 — Pin the seam

Stabilize the `.so` as a loadable backend for zig-agi:
- Freeze the C API subset zig-ai will call (MoE + MLA + FP8 + linear/mlp/gate).
- Add a `build.zig` step in zig-ai that **installs the matching variant `.so`** next to the
  zig-ai binary (or builds it via `zig build -Dvariant=...` on the ktransformers-zig tree).
- Expose a thin Zig FFI shim in zig-ai (`src/backend/cpu_kernels.zig` or similar) that
  `dlopen`s the `.so`, resolves the `kt_*` symbols, and wraps them in idiomatic Zig calls.
- Handle variant selection: zig-ai calls `kt_get_cpu_variant()` (or reads `/proc/cpuinfo`)
  and loads the corresponding `libkt_kernel_ext_{variant}.so`.

### Phase 2 — Layer (the hybrid engine)

Wire the `.so` into the zig-ai model forward pass:
- **Attention + MTP**: keep in zig-ai (CUDA paged attention, speculative decode).
- **MoE layers**: zig-ai allocates/routes as today, but calls `kt_moe_forward` (or
  `kt_moe_forward_sft` for training) for the expert GEMMs, executed on CPU via the `.so`.
  Use `kt_moe_new_with_pool` + NUMA-aware `WorkerPool` (work-stealing, now wired at
  `moe.zig:552`) for per-expert parallelism.
- **MLA projections / LoRA**: optionally call `kt_mla_*` for CPU MLA (note: MLA bodies are
  still placeholder in this port — §6).
- **FP8 weights**: use `kt_fp8_transport_run_producer` / `kt_fp8_transport_wait` for the
  layerwise FP8 weight transport across TP ranks, with the actual dequant+GEMM done in the
  callback (matching the reference design).

The result: **GPU attention+MTP on zig-ai, CPU-offloaded MoE/MLA on ktransformers-zig** — a
unified hybrid engine.

---

## 5. Key files (verified)

### zig-ai
- `benchmarks/bench_baseline.sh` — TTFT (prefill ms) + tok/s harness.
- `benchmarks/bench_sweep.sh` — prefill×decode sweep.
- `benchmarks/bench_paged_attention.zig` — prefill/decode latency & throughput vs batch.
- `examples/moe_bench.zig` — existing GPU-only MoE pipeline (the thing the `.so` augments).
- `python/cli/commands/chat.py` — runtime TTFT computation (`ttft = first_token_time - start_time`).
- `build.zig` — Zig + CUDA toolchain; where the `.so` install step goes (Phase 1).

### ktransformers-zig
- `include/kt_kernel.h` — the C API contract (511 lines).
- `src/main.zig` — all `export fn kt_*` (53 symbols).
- `tools/verify_abi.py` — asserts every declared symbol is exported by all 6 variants.
- `src/kernels/moe/moe.zig` — MoE forward; `config.pool` branch at `:552` (work-stealing,
  per-expert parallelism).
- `src/kernels/moe/moe_sft.zig` — SFT training forward (`forward_sft` + `save_for_backward`).
- `src/kernels/arch/amx.zig` — AMX inline asm + XFEATURE enable.
- `src/runtime/worker_pool.zig` — NUMA-aware work-stealing worker pool.
- `python/kt_kernel/__init__.py` — ctypes wrapper (auto-detects variant, dlopens `.so`).

### ktransformers reference (original C++)
- `kt-kernel/operators/amx/fp8-moe.hpp` + `mxfp8-moe.hpp` — MoE expert layout & buffer writes.
- `kt-kernel/operators/amx/k2-moe.hpp` — `write_weight_scale_to_buffer` (the CPU-offload path).
- `kt-kernel/fp8_layerwise_transport.hpp` / `fp8_layerwise_transport.cpp` — layerwise FP8 transport.
- `kt-kernel/python/utils/amx.py` — Python-side orchestration (model to copy the pattern from).

---

## 6. Risks & open decisions

- **CPU port not finished.** `moe_sft` (TODO #5) and `kt_fp8_*` (TODO #4) are in flight.
  `kt_mla_*` bodies are still placeholder. Integration Phase 0–1 is safe now; Phase 2 MoE
  layering should wait until `moe_sft` + `kt_fp8_*` land, and MLA layering until the MLA
  C API is backed by a real engine (not placeholder).
- **Build system.** zig-ai's `build.zig` must install/build the matching variant `.so`.
  Two options: (a) shell out to `zig build -Dvariant=...` in the ktransformers-zig tree, or
  (b) install prebuilt `.so`s. Decide in Phase 1.
- **C API stability.** The `.so` is the contract. Any change to `include/kt_kernel.h` or the
  exported signatures breaks zig-ai. Gate changes behind the ABI check (`tools/verify_abi.py`).
- **Memory model.** zig-ai is CUDA; the `.so` is NUMA CPU. Keep ownership clean: zig-ai owns
  GPU buffers, the `.so` owns CPU/NUMA memory. For MoE, zig-ai passes weight pointers into
  the `.so`; the `.so` returns CPU-computed outputs that zig-ai copies back to GPU.
- **Speculative decode (MTP).** Currently zig-ai-only. Whether MTP ever moves behind the C
  API is a later decision — out of scope for now.
- **Overlap.** Both projects implement attention (MLA vs paged) and MoE (CPU vs GPU). The
  chosen split: **zig-ai owns attention+MTP; ktransformers-zig owns MoE+MLA+quantized-GEMM.**
  Reconcile the GPU-MoE path in `examples/moe_bench.zig` vs the CPU-MoE path deliberately,
  not by merging.

---

## 7. Current status (blockers)

| Item | Status | Blocks |
|------|--------|--------|
| Work-stealing MoE (`moe.zig`) | **Done** (`b762a2e`, 44/44 green) | Phase 2 MoE layering (unblocked) |
| `moe_sft` port (`moe_sft.zig`) | In flight (TODO #5) | SFT/Training layering in Phase 2 |
| `kt_fp8_*` C API (`main.zig`) | In flight (TODO #4) | FP8 transport in Phase 2 |
| MLA C API real implementation | Placeholder bodies | MLA layering in Phase 2 |

Phases 0–1 can proceed immediately. Phase 2 MoE layering is unblocked now; the MLA and FP8
pieces join as they land.

---

## 8. Recommended next step

Start **Phase 0** immediately: write a measurement driver using zig-ai's
`bench_paged_attention.zig` / `bench_baseline.sh` harness to benchmark the ktransformers-zig
`.so`. This gives you real TTFT/tok/s numbers for the Zig port and de-risks the later
layering — all without touching zig-ai's engine source.
