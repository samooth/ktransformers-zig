# Guía de porting: Kimi K3 → ktransformers-zig

> Port del modelo Kimi K3 (KDA + Gated MLA + AttnRes + LatentMoE) a ktransformers-zig.
> Origen: zig-ai (port ya completado, Fases 0-7).
> Destino: ktransformers-zig (CPU kernel library, C API, .so variants).

## Qué es Kimi K3
MoE de 2.78T params (104B activos/token): 93 capas (69 KDA + 24 Gated MLA, capa 0 densa),
hidden 7168, 96 heads, 896 experts top-16 + 2 shared, latent 3584, SiTU-GLU, experts MXFP4.

## Módulos a portar (de zig-ai)
| zig-ai | ktransformers-zig destino |
|---|---|
| `kimi_k3_config.zig` | `src/kimi/kimi_k3_config.zig` (nuevo) |
| `kimi_k3_ops.zig` | `src/kimi/kimi_k3_ops.zig` (nuevo, CPU SIMD) |
| `kimi_k3_model.zig` | `src/kernels/moe/kimi_k3_model.zig` (nuevo) |
| `kimi_k3_layer.zig` | `src/kernels/moe/kimi_k3_layer.zig` (nuevo) |
| `kimi_tiktoken.zig` | `src/tokenizer/kimi_tiktoken.zig` (nuevo) |
| `kimi_k3_chat.zig` | `src/chat/kimi_k3_chat.zig` (nuevo) |
| `kimi_k3_sampler.zig` | `src/sampler/kimi_k3_sampler.zig` (nuevo) |
| `kimi_k3_state.zig` | `src/state/kimi_k3_state.zig` (nuevo) |

## Reuso en ktransformers-zig
- MLA: `src/mla/mla_core.zig` (adaptar para Gated MLA + NoPE)
- MoE: `src/kernels/moe/moe.zig` (adaptar para LatentMoE)
- GEMM: `src/kernels/amx/gemm_224_bf16.zig` (reusar)
- MXFP4: `src/kernels/amx/gemm_224_mxfp4.zig` (reusar)
- RMSNorm: `src/kernels/amx/` (reusar)
- ShortConv: nuevo (LFM2 tiene shortconv en zig-ai, portar)

## Fases
0. Config + helpers (isMla/isKda/isDense)
1. CPU ops (SiTU, KDA, AttnRes, router, LatentMoE) — scalar + SIMD
2. Model (full recompute)
3. Incremental decode + state
4. Tokenizer + chat + sampler
5. C API + Python bindings
6. Tests + ABI verification

## Invariants (no negociables)
1. A_log por head.
2. MLA NoPE: rope slots existen y se cachean; scale sobre 192.
3. Routing bias solo selección.
4. KDA recurrence order.
5. SiTU: sigmoid sobre gate sin capar.
6. AttnRes boundary: snapshot + clear.
7. MXFP4 nibble order: low nibble = even element.
8. Config sin defaults.
MDEOF
echo written