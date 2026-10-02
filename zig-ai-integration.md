# zig-ai × ktransformers-zig Integration Plan

**Confirmed direction:** **layered hybrid engine — zig-ai is the host** (GPU attention,
PagedAttention, speculative decode, KV quant, E2E pipeline), **ktransformers-zig is an
optional CPU kernel backend** delivered as `libkt_kernel_ext_{variant}.so`, consumed at
runtime via `dlopen` behind an A/B flag. **Never a source merge.**

This document supersedes the 2026-09-01 draft. Status data re-verified against both
trees on **2026-10-03**: ktransformers-zig is now **112 C symbols / 244 tests green /
8 installed variants**, model orchestration (`kt_dsv3_*`, `kt_qwen3moe_*`,
`kt_llama_moe_*`) is **complete**, and zig-ai has since grown **Lane F** (its own CPU
hybrid executor for MoE experts), which changes the overlap analysis fundamentally.

---

## 1. Does it make sense?

**Yes — as a swappable CPU backend behind a flag, not as a replacement for anything in
zig-ai.** The reasoning has two halves.

### 1.1 What the two projects are now

| | ktransformers-zig | zig-ai (lane F + engine) |
|---|---|---|
| **Role** | CPU kernel library (MoE, MLA, GEMM, FP8 transport) | Full GPU inference engine |
| **Artifacts** | 6 x86 `.so` variants + aarch64 NEON (AVX2 … AVX512_BF16, AMX) | monolithic engine binary + CUDA kernels |
| **CPU MoE path** | `kt_llama_moe_*`: full MoE forward (gate/up/down + weighted sum), work-stealing NUMA pool, 16 GGML formats, tile tuning (K_BLOCK), AMX/AVX512 kernels | `src/moe/`: GEMV (qlen=1, decode), per-physical-core workers, deterministic row-split, 12 formats, zero-copy over pinned GGUF bytes |
| **Weights model** | **copies** (`kt_llama_moe_load_weights` replicates into per-expert storage) | **zero-copy** (GEMV walks raw pinned GGUF bytes directly; copy-once HostBank) |
| **What it uniquely has** | AVX512/AMX GEMM, 6 formats lane F lacks entirely (q8_k, iq2_xs, iq3_xxs, iq4_xs, iq1_s, iq1_m — see §2.2), SFT/LoRA backward, FP8 layerwise transport, NUMA subpool pinning | attention+decode E2E, PagedAttention, KV quant, MTP, VRAM budget, layer streaming, benchmark + A/B harness (`moe-bench`) |

### 1.2 The overlap nobody saw coming (lane F)

The 2026-09-01 draft predates lane F. zig-ai **already has** a CPU executor for MoE
experts (`src/moe/cpu_gemv.zig` + `cpu_executor.zig`, TODO 4.6/4.7/4.8/4.9 shipped),
with properties kt-zig's LlamaMoe path does not have:

- **Zero-copy**: GEMV operates directly on the pinned HostBank bytes (TODO 4.3' copy-once;
  gemma-4-26B IQ2_XXS at 9.6GB RSS in 30GB RAM). kt-zig's `kt_llama_moe_load_weights`
  **duplicates every expert bank** — on a 30GB-RAM budget this is disqualifying.
- **Determinism by construction**: static row-split, output independent of worker count;
  verified greedy-parity methodology already in place (TODO 4.5).
- **Deployed A/B**: hybrid split q\*, PCIe fetch fraction, LRU slot cache, async DMA
  primitive (4.8), prefetch (4.9) — the surrounding machinery exists and is measured.

Conversely, kt-zig has things lane F does not:

- **Better ISA scaling**: 6 variant builds with AVX512/VNNI/VBMI/BF16 and AMX dispatch
  (bench: 2.8×–5.3× vectorized vs scalar at DSV3 shapes, on AMX-class hosts). lane F's
  GEMV is auto/scalar only — no AVX512, no AMX.
- **16 GGML formats** vs lane F's 12 (kernel layer) / 8 (wired into the hybrid
  executor), and critically: **its GEMM path handles qlen>1 (prefill)** — lane F's
  GEMV is decode-only (qlen=1).
- **SFT/LoRA backward**, FP8 transport, MLA, model orchestration (DSV3/Qwen3MoE).

### 1.3 Verdict

1. **It's worth it, scoped**: kt-zig as an *optional* CPU backend selected by ISA
   capability (AVX512/AMX hosts) or format need, behind a flag. lane F stays the
   default (zero-copy + determinism + RAM budget).
2. **No merge, no build coupling**: consume the `.so` at runtime. zig-ai's build
   never compiles kt-zig source (avoids lane contention on build.zig, protocol P1-P5).
3. **The one structural blocker**: `kt_llama_moe_load_weights` copies weights
   (§6.1). This contradicts the whole copy-once/zero-copy premise of lane F. Either
   kt-zig grows a borrowed-weights constructor, or the kt backend is limited to
   RAM-rich hosts.

**When it does NOT make sense:** source merge (different memory models, allocators,
release cadences); replacing lane F with kt-zig wholesale (loses zero-copy and the
   determinism guarantees the greedy-parity methodology depends on); wiring kt-zig into
   the build graph (contention surface between lanes).

---

## 2. Verified seam inventory (2026-09-04)

All verified by reading `include/kt_kernel.h` (847 lines, 112 symbols gated by
`tools/verify_abi.py` + `audit_layout.py` + `audit_arity.py` in CI).

### 2.1 The symbol subset zig-ai would need

**Track A — LlamaMoe backend (full MoE forward, decode+prefill):**

```c
kt_llama_moe_config_t {
    size_t expert_num, num_experts_per_tok, hidden_size, intermediate_size, layer_idx;
    void* pool;                    // KT_WorkerPool* or NULL
    uint32_t gate_type, up_type, down_type;   // kt_type_t: 16 GGML values
    uint32_t hidden_type;          // must be KT_TYPE_BF16
    size_t m_block, group_min_len, group_max_len;
    const uint8_t* gpu_experts_mask;         // 1 byte/expert, 1 = skip (GPU)
    const void* gate_proj, up_proj, down_proj;
}
KT_LlamaMoe* kt_llama_moe_new(const kt_llama_moe_config_t*);
void kt_llama_moe_load_weights(KT_LlamaMoe*, int complete_intermediate_size, int offset);
void kt_llama_moe_forward(KT_LlamaMoe*, int qlen, int k, const int64_t* expert_ids,
                          const float* weights, const void* input /*BF16*/, void* output /*F32*/);
void kt_llama_moe_free(KT_LlamaMoe*);
```

**Track B — GEMM primitives (fills lane F's format/qlen gaps):**

```c
// BF16 activations × quantized weights → F32. m, n, k, lda, ldb, ldc.
kt_matmul_q8_0 / kt_matmul_q4_k / kt_matmul_q5_k / kt_matmul_q6_k / kt_matmul_q8_k
kt_f32_to_bf16(const float* src, void* dst, size_t count);
```

**Track C — runtime & pool:**

```c
kt_worker_pool_new(int) / kt_worker_pool_free(...)   // or NULL pool in config
kt_get_cpu_variant()   // → "avx2" | "avx512_base" | ... picks the .so variant
kt_version()           // "0.6.1-zig" — log at load
```

**Out of scope for zig-ai** (decided): `kt_dsv3_*` / `kt_qwen3moe_*` (zig-ai owns model
orchestration), `kt_mla_*` (zig-ai owns attention), SFT/LoRA (no training path in
zig-ai), FP8 transport producer/consumer (reference material for TODO 4.10 only — the
zig-ai implementation should stay native on FetchStream, not cross the FFI for
latency-critical DMA orchestration).

### 2.2 Format mapping (zig-ai → kt_type_t)

Two layers on the zig-ai side, both verified:
- `cpu_gemv.Format` enum (cpu_gemv.zig:39): **12 formats** — q4_0, q8_0, q6_k, q4_1,
  q4_k, q5_k, q3_k, q2_k, iq3_s, iq2_s, iq4_nl, iq2_xxs (kernel layer).
- `gemvFormatOf` (moe_layer.zig:370) **wires only 8 of them into the hybrid
  executor**: q4_0, q4_1, q5_k, q6_k, q4_k, q8_0, iq4_nl, iq2_xxs. The other 4
  (q3_k, q2_k, iq3_s, iq2_s) exist in the GEMV but return null at dispatch (dtype
  "sin soporte CPU" → overflow warning/skip).

kt's LlamaMoe accepts 16 GGML types via `gate_type/up_type/down_type`. kt_type_t
(kt_kernel.h:34) is GGML-numbered — values match GGUF's `GgmlType` tags exactly
(Q4_0=8 … IQ1_M=28), so mapping is mechanical. **Formats kt adds over the wired-8**:
q8_k, q2_k, q3_k, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_xs, iq1_s, iq1_m. Practical
shortlist for real models: **q8_k** (some Qwen variants), **iq4_xs**, **q2_k/q3_k**
(dense 2-3bpw GGUFs). Of lane F's wired-8, kt lacks only **q4_0 and q4_1** (the other
six — q4_k, q5_k, q6_k, q8_0, iq4_nl, iq2_xxs — are all in kt's 16); models relying on
q4_0/q4_1 CPU offload keep needing lane F. gemma-4's down=iq4_nl works on both paths.

### 2.3 Hardware reality check

Development host: **Ryzen 7 5800H — AVX2 only, no AVX512, no AMX** (verified
/proc/cpuinfo). The variant system dispatches to `avx2` here. The AMX/AVX512 payoff
is real but only measurable on other machines — on this host a Phase-0 A/B validates
**plumbing and correctness**, not peak performance. The A/B must be interpreted
accordingly: a tie or modest loss on AVX2 is NOT a kill signal; a plumbing failure or
correctness divergence IS.

---

## 3. Architecture

```
┌─────────────────────────── zig-ai (host) ─────────────────────────────┐
│ GPU: attention / PagedAttention / MTP / KV-quant / CUDA graphs        │
│ CPU: src/moe/ (lane F)  ←──────────── default, zero-copy, deterministic│
│        │                                                               │
│        └── src/moe/kt_backend.zig (NEW, ~300 L)                        │
│              · std.DynLib (pattern: src/cuda/cudaz_ext_sync.zig:61)   │
│              · ZIG_AI_MOE_BACKEND=kt|native (default native)          │
│              · f32→BF16 activation shim, kt_type_t mapping            │
│              · breadcrumbs [cpu_executor] gated DEBUG_LEVEL           │
└──────────────────────────┬────────────────────────────────────────────┘
                           │ dlopen("libkt_kernel_ext_{variant}.so")
                ┌──────────┴───────────┐
                │ ktransformers-zig .so│  built in ITS OWN tree:
                │  kt_llama_moe_*      │    zig build -Dvariant=<v> install
                │  kt_matmul_q*        │  (no zig-ai build coupling)
                │  kt_worker_pool_*    │
                └──────────────────────┘
```

- **Dependency is one-way**: zig-ai → `.so`. kt-zig never learns about zig-ai. The C
  API (`include/kt_kernel.h`, triple-gated: exports + arity + layout) is the contract.
- **Variant selection**: `kt_get_cpu_variant()` from the loaded `.so`, or read
  /proc/cpuinfo before load — same string either way.
- **Allocation**: kt-zig has an injectable allocator (`kt_set_default_allocator`) —
  Phase 2 wires it to zig-ai's tracking allocator so expert storage shows up in leak
  gates. Config structs are passed by pointer (`kt_llama_moe_config_t*`) — the
  u8-vs-c_int enum bug found in the pybind shim (2026-09-01 lessons) is avoided by
  transcribing the `extern struct` field-for-field in Zig (same language, same ABI).

---

## 4. Phased plan (updated)

### Phase 0 — A/B driver (risk-free, do first)

Do NOT touch the engine. Write `benchmarks/bench_kt_moe.zig` (or extend the moe-bench
harness) following the established protocol (same as TODO 4.5's session):

- Build: `zig build -Dvariant=avx2` in the kt-zig tree → `libkt_kernel_ext_avx2.so`.
- Driver: `std.DynLib.open("libkt_kernel_ext_avx2.so")`, resolve §2.1 Track A+C
  symbols, run `kt_llama_moe_forward` on the same synthetic MoE fixture zig-ai's
  `moe-make-fixture` produces (valid GGUF MoE, fused or `.experts.i` naming).
- Compare: ms/capa-llamada vs lane F `cpu_executor` on identical shapes/steps/skew
  (`MOE_STEPS=40 MOE_SKEW=zipf COPYONCE=1`), plus output-parity (F32 out vs lane F
  GEMV path, tolerance = BF16 conversion epsilon — inputs are converted f32→BF16).
- Report in the same format the TODO table already uses (GATHER/STAGING/caché
  columns), so results slot into TODO 4.x directly.

**Decision gates:**
1. kt wins or ties on AVX2 → backend worth shipping behind flag even on modest hosts
   (gets qlen>1 prefill + the §2.2 extra formats).
2. kt loses on AVX2 → backend auto-disabled on non-AVX512 hosts; keep for AMX/AVX512.
3. Output divergence beyond BF16 epsilon → kill signal for track A until kt-zig
   resolves.

### Phase 1 — Backend behind flag (seam, ~1 day of work)

`src/moe/kt_backend.zig` in zig-ai:

1. `std.DynLib` lifecycle, variant auto-detect, `kt_version()` breadcrumb at load.
2. `extern struct KtLlamaMoeConfig` — field-for-field transcription of
   `kt_llama_moe_config_t` (kt_kernel.h:798-830).
3. `ktTypeOf(dtype: gguf.GgmlType) u32` — mirror of `qtypeOf`/`gemvFormatOf` semantics.
4. Activation shim: f32 [qlen, hidden] → BF16 via `kt_f32_to_bf16` (or in-Zig; either
   is one pass over hidden-sized rows — trivial next to the GEMMs).
5. Pool: start with `pool = NULL` (default subpool); NUMA subpool config is a Phase-3
   refinement.
6. Env: `ZIG_AI_MOE_BACKEND=kt|native` (default native — same opt-in pattern as
   `ZIG_AI_HYBRID`); `ZIG_AI_KT_SO=<path>` override; breadcrumbs through
   `debugz.dbg.printLevel` under `[cpu_executor]`/`[moe_layer]` scope (NO prints in
   the .so contract; wrapper owns logging — same rule as CUDA kernels).
7. Tests: `tests/test_kt_backend.zig` gated on `.so` presence (skip pattern like
   `GGUF_MODEL_PATH`-gated tests): load, forward, parity vs lane F on the fixture,
   free — zero leaks through the tracking allocator (if kt's allocator injection is
   wired; else assert RSS delta returns to baseline after `kt_llama_moe_free`).

### Phase 2 — RAM fix (the load_weights blocker)

`kt_llama_moe_load_weights` **copies** all expert banks into per-expert replicated
storage. On a 30GB host with a 9.9GB IQ2_XXS model, lane F's copy-once wins by
construction (kt path would need ~2× model RAM). Options, in preference order:

1. **Upstream change (preferred)**: ask kt-zig for
   `kt_llama_moe_new_borrowed(gate, up, down)` — the GEMMs already iterate GGML
   blocks; borrowing the caller's pointers is a local change in
   `src/kernels/moe/llamafile_moe.zig`. Until it lands, Phase-1 backend is
   **degraded**: usable only when `free RAM > 1.2 × expert-bank bytes` (checked at
   backend init, breadcrumb + auto-fallback to native).
2. **zig-ai-side workaround**: none clean. Do not duplicate HostBank into kt storage
   per-model; that defeats copy-once. Accept the constraint instead.

### Phase 3 — Refinements (only if Phases 0-2 said yes)

- NUMA subpool pinning via `kt_worker_pool_new_config` + kt-zig's auto-populated
  thread lists (mirrors lane F's physical-core pinning).
- Track B GEMM primitives as *auxiliary* fallback for the formats lane F lacks
  (§2.2 shortlist), keeping zero-copy lane F as the first choice.
- `kt_get_cpu_variant()`-gated auto-enable: backend ON only for `avx512_*`/`amx`
  variants, unless user forces `ZIG_AI_MOE_BACKEND=kt`.
- Prefill (qlen>1): kt's forward handles it; lane F's GEMV cannot — route
  CPU-offloaded *prefill* expert GEMMs through kt when attached.

### Phase 4 — Reference mining (no code coupling)

- **TODO 4.10 (expert streaming 2-slot)**: `kt_fp8_transport_run_producer` /
   `fp8_layerwise_transport` is the pattern the ticket already cites. Read the Zig
   port (`kt-zig tree`) as the reference impl alongside the C++ original — but
   implement native on zig-ai's FetchStream; don't FFI the DMA orchestration.
- SFT/LoRA: revisit only if zig-ai ever grows a training path.

---

## 5. Key files (verified 2026-09-04)

### zig-ai (host side — where work happens)
- `src/moe/cpu_gemv.zig` — lane F GEMV, 12 formats, `Format` enum (the incumbent).
- `src/moe/cpu_executor.zig` — worker pool (physical cores, sched_setaffinity).
- `src/moe/hybrid.zig` / `bandwidth.zig` — q\* split arithmetic (4.7 B1 verdict: cap=1
  until overlap lands — kt backend does not change this).
- `src/moe/offload_cache.zig` — LRU expert slot cache.
- `src/transformer/moe_layer.zig` — `gemvFormatOf` (:370), `qtypeOf` (:70) — the two
  dispatch points a kt backend plugs alongside.
- `src/cuda/cudaz_ext_sync.zig:61` — existing `std.DynLib` pattern to copy for kt.
- `benchmarks/` + `zig build moe-bench` + `moe-make-fixture` — A/B protocol.
- `TODO.md` §4 (4.3'-4.13) — the plan this integrates with (§7 below).

### ktransformers-zig (backend side)
- `include/kt_kernel.h` — 112-symbol contract; llama_moe family at :788-846,
  matmul_q family at :480-489, worker pool at :320-322, variant/version at :596-603.
- `src/main.zig` — all `export fn kt_*` (122 exports incl. ABI probes).
- `src/kernels/moe/llamafile_moe.zig` — LlamaMoe: 16-format `gemmQuant` dispatch,
  `gpu_experts_mask`, work-stealing pool branch; **where a borrowed-weights
  constructor would go (Phase 2)**.
- `src/runtime/worker_pool.zig` — NUMA-aware work-stealing pool.
- `tools/verify_abi.py` + `audit_layout.py` + `audit_arity.py` — the ABI gates (CI
  runs them; zig-ai can pin its expected symbol list against them).
- `REMAINING.md` — their open items (TP subpool routing, forward_many batching,
  Q4_K×Q8_0 direct GEMM) — relevant to Phase-3 performance, none block Phase 0-1.

### Reference (upstream C++)
- `ktransformers/kt-kernel/operators/llamafile/moe.hpp` — LLAMA_MOE_TP original.
- `ktransformers/kt-kernel/fp8_layerwise_transport.cpp:209-710` — TODO 4.10 pattern.

---

## 6. Risks & open decisions

1. **`load_weights` copy (HARD blocker for RAM-tight hosts)** — §4 Phase 2. Preferred
   fix is upstream (`new_borrowed`); until then backend auto-falls-back when the RAM
   check fails. **Decision owner: lane F + kt-zig maintainers.**
2. **BF16 activation seam** — kt takes BF16 input, lane F takes f32. Greedy-parity
   (the 4.5 methodology) must tolerate a one-roundtrip BF16 epsilon, or the backend is
   restricted to non-parity-critical paths (overflow only). Test this explicitly in
   Phase 0.
3. **AVX2-only dev host** — §2.3. Phase 0 cannot see the AMX/AVX512 payoff; do not
   kill the backend on an AVX2 tie/loss. Kill only on plumbing/correctness failure.
4. **ABI drift** — kt-zig moves fast (68→86→90→105→112 symbols in ~2 weeks). zig-ai's
   shim must (a) resolve only the ~12 symbols it uses, (b) tolerate `null` lookup
   gracefully, (c) log `kt_version()` so A/B results name the exact .so. Never link
   the header at build time.
5. **Determinism** — kt's work-stealing changes reduction order across runs. The
   greedy-parity tests that gate zig-ai releases must run with backend=native; backend
   runs get their own tolerance class. Document in the test header.
6. **Pool double-spawn** — lane F pins its own workers; kt's pool spawns more threads.
   Enforce: when backend=kt is active, lane F's executor stays idle (they alternate,
   never mix within one layer call), or kt pool = NULL. Measure; then enforce.
7. **Lane protocol** — Phase 1 touches `src/moe/` (lane F surface) and `build.zig`
   (test step). Follow P1 (owner-commented import lines), P2 (sweep before commit),
   P6 (TODO.md/TODO.en.md updated in the closing commit).

---

## 7. Integration with the existing zig-ai plan (TODO §4)

The kt backend slots into the existing 4.x tickets **without restructuring them**:

| Existing ticket | Interaction with kt backend |
|---|---|
| **4.3' copy-once** (shipped) | The kt backend must NOT break copy-once: it can borrow the pinned GGUF buffer pointers for `gate/up/down_proj` fields **only after Phase 2's borrowed constructor**; the copy constructor contradicts 4.3' (§6.1). |
| **4.5 A/B hybrid acceptance** (blocked by wiring) | Phase 0's A/B driver reuses this exact protocol (same fixture/steps/skew/env) — results are directly comparable and land in the same TODO table. |
| **4.6 GEMV formats** (shipped) | kt's Track B GEMMs are the *complement*, not replacement: 6 genuinely-new formats (§2.2), zero-copy stays lane F. |
| **4.7 q\* split** (B1 negative verdict) | **Unchanged.** The B1 verdict (q\* 1.65× slower without overlap) is about PCIe fetch vs CPU compute balance — backend choice doesn't alter it. kt backend inherits cap=1 default. |
| **4.8 async DMA / 4.9 prefetch** (shipped primitives) | Orthogonal — they move *experts to GPU*; kt backend computes *experts on CPU*. Both can coexist (fetch some, compute others), but don't re-open 4.7 until 4.8/4.9 deliver overlap, per the B1 verdict. |
| **4.10 expert streaming 2-slot** (open) | Uses kt-zig's FP8 transport **as reference only** (§4 Phase 4). Native impl on FetchStream. |
| **4.11 KV spill / 4.12 rebuild** (open) | Independent. No interaction. |
| **4.13 io_uring copy-once read** (bench shipped) | Unchanged; kt backend consumes the same pinned buffer post-Phase-2. |

**Sequencing:** Phase 0 (driver) can run immediately and lands as a new TODO 4.x row
(e.g. "4.14 kt-backend A/B"). Phase 1 (seam) is the only engine-side change and it is
opt-in by default-off flag — zero risk to existing lanes. Phase 2 depends on
conversation with kt-zig. Phases 3-4 are gated on 0-2 results.

---

## 8. Recommended next step

**Phase 0, today:** build the avx2 `.so`, write the DynLib A/B driver against the
moe-make-fixture output, run it with the 4.5 protocol (MOE_STEPS=40 MOE_SKEW=zipf),
record GATHER/STAGING-equivalent numbers + output-parity delta vs lane F, and add the
row to TODO.md (4.14) with the verdict. That single run decides whether Phase 1 gets
a green light — no engine code changes needed to get the answer.
