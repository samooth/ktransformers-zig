# ktransformers-zig TODO

## Status: Beta (All workstreams closed; 15/15 GGML formats + 200+ tests)

Last updated: 2026-09-03

### Coordination protocol (multi-agent)
**CLAIM before you code.** Before starting a new task, add a claim line
in the relevant section: `- [ ] <task> — CLAIMED by <agent> since <date>`.
This prevents parallel collisions (we hit one with LlamaMoe: two agents
independently ported the same LLAMA_MOE_TP to Zig in the same session).
Claims are removed when the task is done; abandoned claims can be reclaimed.

**Build: WORKING** - `zig build` produces `zig-out/lib/libkt_kernel_ext.so`; 6 x86 variants + aarch64 neon cross-build.
**Tests: WORKING** - `zig build test` runs all suites: 73 kernels + 11 MLA + 9 MLA C-API + 9 Qwen3 MoE + 2 FP8 + 9 GGML C API + 7 aarch64 + 4 NEON kernel + 2 allocator C API = **156 total pass, 0 leaks, exit 0**.
**Multi-variant: WORKING** - `zig build all-variants` produces 6 `.so` + neon, each exporting **105 C API symbols** (double-gated: exports + arity via `tools/verify_abi.py`; layout via `tools/audit_layout.py`). The 9 kt_qwen3moe_* symbols are exported from all 8 variants.
**Bench: WORKING** - `zig build -Doptimize=ReleaseFast bench` (B2 + moe_bench extensions): gemmExpert vectorized vs scalar ref at DeepSeek-V3 shapes — **2.8x (down, memory-bound) to 5.3x (prefill) measured**, maxdiff ≤ 1.7e-6; MoE forward end-to-end with default-vs-tuned tile params.
**Runtime: WORKING** - work-stealing worker pool + **sched_setaffinity pinning (A3)** + **waitIdle/kt_cpuinfer_sync drain (B3)**; NUMA topology via /sys and /proc/cpuinfo + **mbind in NumaAllocator (A3)**; **L1/L2/L3 sysfs detection + selectTileParams (A4)**.
**SFT/LoRA: FORWARD+BACKWARD COMPLETE** - training path done; C API exports (forward_sft/backward/update_lora_weights); smoke test passes.
**Allocator: INJECTABLE (B1)** - `kt_set_default_allocator` C-ABI vtable; all 9 context types capture-and-free symmetrically.
**Model Orchestration: COMPLETE** - DeepseekV3DecoderLayer + DeepseekV3Model + DeepseekV3ForCausalLM (C-API `kt_dsv3_*`) AND Qwen3MoeDecoderLayer + Qwen3MoeModel + Qwen3MoeForCausalLM (C-API `kt_qwen3moe_*`).
**GGML: 10/10 STANDARD FORMATS COMPLETE** - Q8_0, Q4_K, Q5_K, Q6_K, Q8_K, Q2_K, Q3_K, IQ4_XS, IQ2_XXS (full quantize with kmap init), IQ3_XXS, IQ4_NL — all with quantize + dequantize + GEMM + tests.
**MHA engine + GGUF parser (Qwen3 dev)**: vanilla MHA `MhaEngine` + GGUF v3 parser in `src/io/gguf.zig` — committed in `a61271e`.

---

## ✅ Completed (2026-08-31 — dev-feedback audit, IMPROVE.md P0-P3 ALL RESOLVED)

Verified audit of external dev feedback (see IMPROVE.md: which claims were
true vs false, with file:line evidence). All real items fixed:

- [x] **A2 — AMX guard was comptime-broken (P0)** — `AmxFeatures.available`
  used `@hasField(std.Target.Cpu.Feature.Set, "amx_int8")` which is ALWAYS
  false in Zig 0.16 (Feature.Set is a packed bitmask, not a named-field
  struct) — every AMX intrinsic silently no-op'd even on AMX hardware and
  on `-Dvariant=amx` builds. Fixed with comptime
  `std.Target.x86.featureSetHas(.amx_tile)` + runtime `detectAmxSupport()`
  (CPUID leaf 7 EBX bits 22/25 + arch_prctl XTILEDATA permission, cached in
  an atomic). All intrinsic guards updated. (commit ff57125)
- [x] **D1/D2/D3 — MoE latent buffer/index bugs (P0)** — D1: sequential
  forward allocated `expert_down_out` as `hidden` (1 token) but
  forwardDown writes `count*hidden` (overflow when an expert got >1
  token; hidden because tests used qlen=1). D2: `forwardParallel` used
  `gi*h+h` — loop index as stride (corruption for qlen>1; hidden for the
  same reason). D3: `forwardGateUp/Down` passed `ldc=inter` while the
  scratch is `m*n` with `n=inter/tp_count` (overflow for tp>1). All
  fixed + 2 regression tests exercising qlen=3 (multi-token-per-expert and
  cross-expert). (commit ff57125)
- [x] **A1 — gemmExpert was pure scalar (P1)** — hot path of
  MoE/MLP/Linear/Gate was a triple for-loop. Vectorized the K axis with
  `@Vector(8,f32)` (AVX2 `vmulps` verified in disassembly) + scalar tail
  for k%8. New `amx.reduceAddFp32` helper. 2 regression tests (aligned k,
  k=37 tail boundary) vs hand-computed reference. **Measured (B2): 5.2x
  decode gate/up, 5.3x prefill, 2.8x decode down (memory-bound), 5.0x
  large square; maxdiff ≤ 1.7e-6.** (commit 7515468)
- [x] **A3 — worker pool had no NUMA pinning (P1)** — `Subpool.init` now
  takes `?[]const usize` CPU list; every worker calls
  `numa.setThreadAffinity` (sched_setaffinity) as its FIRST action.
  `WorkerPoolConfig.subpool_thread_cpus` plumbs lists through.
  `NumaAllocator.alloc` calls real `mbind` (closes the old "TODO:
  numactl/libnuma"). Fixed Zig 0.16 comptime_int|runtime rot in
  `bindMemory` en route. root.zig comptime fn-refs force the NUMA
  helpers into the .so (`nm` verified). Regression test: spawned worker
  observes itself on the pinned CPU set via getcpu. (commit c35a526)
- [x] **D4 — kt_gate_forward ignored DeepSeek-V3 routing fields (P1)** —
  n_group/topk_group/norm_topk_prob/routed_scaling_factor/e_score_correction_bias
  were accepted and discarded; topk_weights were raw logits (not
  probabilities). Implemented the full reference algorithm
  (kt-kernel python/sft/layer.py:696-728): sigmoid scores + bias
  correction + group-top2 sum + topk_group selection + masked final
  top-k + sigmoid weights + optional normalize + scaling. Malformed
  configs fall back to the legacy naive path (no panic on model load).
  GateContext captures the routing config. 2 regression tests mirroring
  the Python reference test (hand-computed ids+weights, normalized sum
  == routed_scaling_factor). (commit f5443f3)
- [x] **B2 — benchmark suite (P2)** — `bench/gemm_bench.zig` + `zig build
  bench` step: gemmExpert vs pure-scalar reference at 4 DeepSeek-V3
  shapes, best-of-5 CLOCK_MONOTONIC, GFLOPS + speedup + maxdiff
  cross-check. Zig 0.16 notes: std.Timer doesn't exist (clock_gettime
  syscall pattern); `{e:.2}` is the scientific-notation format.
  (commit 7d2db9c)
- [x] **A4 — no L1/L2/L3 detection (P2)** — `CpuInfo` gains
  l1d/l1i/l2/l3_bytes populated from
  `/sys/devices/system/cpu/cpu0/cache/index*/` (Zig 0.16 Io API:
  `std.Io.Dir.cwd` + `readPositionalAll`; `std.fs.cwd` is gone). New
  `selectTileParams(cpu)` derives GEMM n_block/k_block from real L2
  (50%-of-L2 working-set budget, tile-aligned). Host measured:
  L1d=32K L2=512K L3=16M → k_block=448 (evidence the fixed
  K_BLOCK=1792 constant exceeds this host's L2). Kernels still use the
  fixed constants — wiring selectTileParams in is the follow-up. 2 tests.
  (commit ca0485a)
- [x] **B3 — kt_cpuinfer_sync was a no-op (P2)** — `Subpool.pending_jobs`
  atomic (incr at doWorkStealingJob entry, decr under mutex + done_cv
  broadcast at exit) + `waitIdle(allow_n)`. `kt_cpuinfer_sync` drains
  every subpool to the caller's allowance. Contract now holds even if
  submit becomes fire-and-forget. Test: two job bursts drain to 0
  pending (accounting bugs fail by deadlock). (commit ca0485a)
- [x] **B1 — page_allocator hardcoded everywhere (P2)** — new
  `kt_set_default_allocator(vtable|null)` C-ABI export (symbol 87):
  userdata + alloc/free/resize fn pointers, malloc semantics. Adapter
  bridges to std.mem.Allocator (layouts are NOT compatible — real
  conversion, not a cast). ALL 9 context types (WorkerPool, TpMoe,
  TpMoeSft, MlaContext, Gate, Linear, MLP, Fp8Transport) capture the
  allocator at `*_new` and free through the CAPTURED one — swapping the
  default between new and free is safe. Per-call scratch (MLA/Linear/
  MLP) uses the context allocator. Bonus bug fixed: TpMoeSft.deinit
  freed grad_inter with hardcoded page_allocator (invalid-free). 2-test
  suite `tests/kernels/allocator_capi_tests.zig` with a tracking vtable:
  1 alloc/1 free balanced, 0 live bytes, even after mid-lifecycle
  default swap. (commit 15a8ea5)
- [x] **Rejected feedback (documented in IMPROVE.md C1-C5)** —
  kt_mla_forward is NOT a placeholder; amx.zig HAS real instructions;
  loadWeights is NOT a no-op; there ARE 15+ numerical correctness
  tests; the dev's kt-*.tar.gz drop-in files do NOT exist on this
  system.

---

## 🟡 High Priority (Next Steps)

### Runtime (follow-ups from the audit)

- [x] **Wire `selectTileParams` into the kernels — CLOSED (ce849ec, validated by 196a6c4)** — the tune hook (`tuneTileParamsForHost` in main.zig) installs measured cache-derived K_BLOCK/N_BLOCK into the BF16 kernel. Two fixes landed in ce849ec: (1) **idempotent guard** — detectCpu allocates, so a process-wide static makes the first call win (was re-running the sysfs probe on every kt_cpuinfer_new); (2) **defensive wiring at every operator _new** — kt_moe_new/kt_moe_new_sft/kt_dsv3_{layer,model,causallm}_new all construct TpMoe (whose init reads GemmKernel224BF.K_BLOCK for BufferA/B/C sizing) and now call the tune first, so callers that build their own pool still get host-sized expert buffers. **Measured validation** (moe_bench, 196a6c4): host-derived K_BLOCK=448 on a 512K-L2 Ryzen gives **-13.2% / -20.3%** on 16-tok / 8-tok prefill shapes vs the 1792 default (working set per block-step drops 9MB -> 2.3MB), noise-level on decode shapes — the A4 design intent, delivered. Note: INT8/MXFP kernels still use fixed K_BLOCK=3584 comptime consts (only the BF16 kernel has the runtime override) — extending the `pub var` pattern to them is optional follow-up if a host ever shows pressure on their shapes.
- [x] **NUMA helpers + kt_worker_pool auto-population — CLOSED (bd7e712 + 476a96e)** — bd7e712 (other dev, 2026-09-01) revived the rotted helpers: numa_topology full IO-layer rewrite (std.Io.Dir.cwd, readSmallFile fixed-buffer, node-probe via ENOENT loop), allocNuma via rawAlloc (comptime-enum workaround), getThreadAffinity return-convention BUG FIX (kernel returns bytes-copied, not errno — every success was misread as failure). 476a96e (this session) wired the remaining piece: kt_worker_pool_new_config auto-populates `subpool_thread_cpus` from NumaTopology.detect when the caller declares a multi-node subpool_numa_map (empty node -> unpinned subpool; single-node/detect-fail -> no pinning). Ownership: WorkerPoolConfig.owns_cpu_lists frees outer+inner slices at deinit; the C path now sets owns_config=true (fixes a pre-existing leak: inline-built numa_map/thread_counts were never freed). 3 C-API tests in tests/kernels/numa_pool_tests.zig (suite 8). Remaining rot: numa_worker.zig (unreferenced, unowned — fix on first use).
- [x] **Route internal scratch through the MoE allocator — CLOSED (58b0cb1)** — `routeExperts`/`routeExpertsDeepSeek`/`routeExpertsWithOpts` scratch (logits + score rows + group scores/mask) now allocates through the caller-threaded allocator: `GateContext.allocator` at kt_gate_forward (B1-captured), `self.allocator` in DeepseekV3DecoderLayer.forward, testing-allocator at test sites. Buffers no longer escape a kt_set_default_allocator-installed tracker. No ABI change (internal API). Routing tests green: Gate+MoE e2e, group-top2 vs Python reference, both fallbacks.
- [x] **`kt_mla_forward` qlen_count > 1 — CLOSED (139bf63 + 2e8ce43)** — paged/batched path: qlen_count==1 keeps the legacy sequential route; >1 loops per sequence through `MlaEngine.forwardPaged` with per-sequence page tables (mla-tp.hpp:84 contract), page_table_lens validated (>= ceil(kv_len/tokens_per_page)). `MlaKvCache` gained paged accessors (`pagedGetPageInfo/pagedWriteToken/pagedGetNopePtr/pagedGetRopePtr/ensurePageCount`) + save/load/dump/fromDump (32-byte "KVCA" header; page_table NOT serialized — scheduler's job, same as kvcache_load_dump.cpp). **Equivalence tests (2e8ce43 — the coverage 139bf63 shipped without)**: (a) scrambled==identity — same logical content on different physical pages, bit-for-bit (non-zero weights + non-zero-output guard = non-vacuous); (b) paged==legacy — single sequence through identity table == sequential path bit-for-bit. **Test (a) caught a real OOB on first run**: forwardPagedSequence sized the page pool from the LAST table entry only; a scrambled table with its max page index earlier (e.g. {7,6}) read past the pool (`index 7, len 7` in pagedGetRopePtr). Fixed: scan all n_pt_entries for max_page. 143 tests, 0 leaks, ABI 96/96 triple gate PASS.

### Python Packaging
- [x] **Minimal pybind11 wrapper — DONE (4040b5b)** — see the Tier-1 entry
  under Completed GGML/pybind11 below. (`python/kt_kernel/` ctypes wrapper
  remains the zero-dependency surface.)

### CI/CD
- [x] **Run the ABI gates in CI** — `wheels.yml` has `abi-gate` job
  (builds all variants, runs `verify_abi.py` + `audit_layout.py` +
  `zig build test`; other dev added in `7db2936`).
- [x] **Python binding for `kt_set_default_allocator`** — symbol 87 has
  ctypes wrapper (lines 542-566 of `python/kt_kernel/__init__.py`):
  `kt_allocator_vtable` Structure (userdata + alloc/free/resize
  CFUNCTYPEs) + `kt_set_default_allocator(POINTER(vtable))` call
  (None restores default). End-to-end test at
  `tests/test_allocator.py`: installs a tracker allocator, runs
  kt_worker_pool_new(0) + kt_worker_pool_free() round-trip, asserts
  5 allocs/5 frees/408 bytes (matching the WorkerPool's internal
  buffer setup), then asserts kt_set_default_allocator(None) actually
  stops the tracker (counts stay frozen across a second round-trip).

---

## ✅ Completed (2026-08-29)

### Critical Build Fixes
- [x] **Module System**: `build.zig` now uses `src/root.zig` as root source file
- [x] **`src/root.zig`**: Created with proper submodule re-exports
- [x] **`zig build`**: Produces `zig-out/lib/libkt_kernel_ext.so`
- [x] **`zig build test`**: All tests compile successfully

### Zig 0.16 Migration Fixes
- [x] **`@intFromFloat` → `@intFromFloat` with typed cast**: Fixed in `buffers.zig`
- [x] **`std.atomic.Bool` → `std.atomic.Value(bool)`**: Fixed in `worker_pool.zig`
- [x] **`std.Thread.Pool` removed**: Replaced with placeholder in `WorkerPool`
- [x] **`std.DoublyLinkedList(T)` → `std.DoublyLinkedList`**: Fixed API call
- [x] **`std.time.sleep()` removed**: Replaced with spin-wait
- [x] **`alignedAlloc` signature change**: Updated to new 3-arg API
- [x] **Pointer types need optional**: Updated `[*]T = null` to `?[*]T = null`
- [x] **`packed struct` with arrays**: Changed to `extern struct`
- [x] **Packed struct fix in `arch/amx.zig`**: TileConfig now uses extern struct
- [x] **AMX features check**: Use `@hasField` for comptime safety
- [x] **moe.zig indentation**: Fixed ternary indentation
- [x] **main.zig forward references**: Removed all undeclared function references
- [x] **cpu_detect.zig `||` → `or`**: Fixed all string comparisons
- [x] **Test infrastructure**: Tests now compile and run

---

## 🟡 High Priority (Next Steps)

### Task assignments (2026-08-30, two agents working concurrently)

**Dev A — MLA C API wiring + warmUp** (files: `src/main.zig`, `src/kernels/moe/moe.zig` warmUp only):
- [x] **Wire `kt_mla_*` C API to the real MLA engine** — replaced 5 placeholders (kt_mla_new/free/forward/prefill/decode) + kt_mla_update_kv_cache. MlaContext wrapper struct holds MlaEngine + MlaKvCache. BF16↔F32 conversion at C/Zig boundary. `*anyopaque` cast to `[*]const u16` for BF16 weights. External kv_cache param documented as ignored. Engine end-to-end test added.
- [x] **Upgrade `TpMoe.warmUp` to real pre-touch** — replaced no-op with first-byte touch of gate/up/down BF16 storage. Guarded by weights_loaded and bf16_weights_alloced. Touches first page of each storage to prime the TLB.

**Dev B — MXFP4/MXFP8 kernels** (files: `gemm_224_mxfp{4,8}.zig`, `src/root.zig` re-exports, `tests` append-only, docs):
- [x] **Complete MXFP4/MXFP8 kernels** — both files now in root.zig's import graph (re-exports + comptime fn-refs force analysis per the LAZY-ANALYSIS lesson). mxfp4 bogus `unpackMXFP4(block, &sum)` line removed; mxfp8 u8 cast in `f32_to_fp8e4m3` fixed. Per-32-block scale folded into the dequant (FP8 pattern adapted to block scale). AMX tile path mirrors `gemm_224_fp8.zig` (config/cleanC/storeC/loadA/runTile/loadB + gemmFullTile dispatch, scalar fallback preserved). Two exact-value tests in `test_kernels.zig`; 27 kernels + 11 MLA = 38 total pass, 0 leaks. References: `operators/amx/mxfp8-moe.hpp`, `fp4-moe.hpp`, `avx2/mxfp4-moe.hpp`.

**Exclusivity notes (2026-08-30, live two-agent work)**: Dev A owns main.zig + moe.zig; Dev B owns mxfp files + root.zig. Shared files only by the stated constraints (tests append-only, docs append/mark). `zig build test` currently exits 0 with 0 leaks (38/38) — keep it that way.

### Kernels
- [x] **Complete INT4 GPTQ kernel (`gemm_224_int4.zig`)** — replaced scalar fallback with AMX tile path (`tileint8dpd` + on-the-fly INT4→INT8 dequantization). Lo/hi nibble pattern from C++ `GemmKernel224Int4` (amx_kernels.hpp:1559-1848). Per-group scales applied at K-block boundary via `applyScales` (INT32 → FP32). Correctness test added to `tests/kernels/test_kernels.zig`. Non-AMX hosts fall back to the preserved scalar implementation.
- [x] **Complete FP8 E4M3 kernel (`gemm_224_fp8.zig`)** — replaced scalar fallback with AMX tile path that reuses the BF16 GEMM tiles (`tilebf16dpd` accumulates into FP32, no INT32 scratch). On-the-fly FP8→BF16 dequantization in `loadB` (byte-level, 1:1: each FP8 byte becomes one BF16 short). Per-row scale applied at end of each `(m, n)` block (not per K-step). Tile config matches BF16 (TILE_M=16, TILE_K=32, TILE_N=16, VNNI_BLK=2). Correctness test added (skips on non-AMX). Non-AMX hosts fall back to scalar.
- [x] **Complete MXFP4/MXFP8 kernels** — both wired into root.zig's import graph; AMX tile path + scalar fallback. 2 exact-value tests; 27 kernels + 11 MLA = 38 total pass, 0 leaks.
- [x] **Vectorize `applySwiGLU` with `std.simd`** (Dev B) — VecF32=8 vectorized helpers (`swigluVec`/`swigluClampVec`/`swigluOaiVec`) in `arch/amx.zig`; `applySwiGLU` refactored to branch-once on variant + vector inner loop (8-wide) + scalar tail for `n % 8`. Verified bit-exact vs scalar across all three variants and `n`=5..17 (covers tail, exact-width, multi-vector); 1 new correctness test in test_kernels.zig.
- [x] **Add actual AMX inline assembly (previously placeholder)** — `ldtilecfg`, `tilerelease`, `tileloadd`, `tilestored`, `tilezero`, `tilebf16dpd`, `tileint8dpd` all wired in `src/kernels/arch/amx.zig` with proper comptime tile-register immediates. Includes XFEATURE_XTILEDATA permission request via `arch_prctl`. Correctness tests added to `tests/kernels/test_kernels.zig` (skip on non-AMX hardware).

### MoE Layer
- [x] **Implement `TpMoe.loadWeights()` with online quantization (BF16 → INT8/INT4)** — BF16 path now packs all 3 projections (gate, up, down) with per-TP-rank slicing. INT8 path unchanged. Module-level BF16 weight storage used to avoid struct field changes.
- [x] **Complete the MoE compute path** — `loadWeights` BF16 packing (all 3 projections), `forwardGateUp`/`forwardDown` real bodies (was no-ops with TODOs), `forward` expert GEMM enabled (gates, SwiGLU, down, routing weight accumulation). Forward equivalence test passes.
- [x] **Implement `merge_results()` with AVX512 FP32 add + BF16 convert**
- [x] **Expert routing with top-k selection (GEMM-based, naive top-k)** — `routeExperts` upgraded from scalar dot product per (token, expert) to one batched BF16 GEMM (`gemm_bf16.gemmExpert`) + naive O(n*k) top-k. Pool parameter made optional (`?*WorkerPool`) so the C API (no pool) and MoE forward (real pool) both work. Naive top-k is fine for k=2-8; SIMD/heap optimization is a follow-up.
- [x] **Replace 5 placeholder TpMoe methods** — `deinit` (frees the two init-allocated slices, page_allocator convention; buffer structs are non-owning POD views), `warmUp` (no-op + TODO until loadWeights populates ptrs), `loadWeightsWithMap` (double-load + logical-slot remap), `forwardGateUp`/`forwardDown` (guarded no-ops with real bodies in comments, blocked on the loadWeights BF16 bug). End-to-end test with zero inputs passes; 5 TpMoe placeholder methods now safe.
- [x] **Fix `gemmExpert` `weight_ld` OOB in MoE expert compute** — `moe.computeExpert` (3 sites) and `TpMoe.forwardGateUp`/`forwardDown` (3 sites) were passing `weight_ld = n` instead of `k`, causing OOB reads of up to `n^2 - n + k - 1` past the end of `n*k`-element weight buffers. The OOB happened to land in page allocator zero-fill on Linux so existing tests passed by accident. All 6 sites now use `weight_ld = k` matching the `[n, k]` row-major layout. See LESSONS_ZIG.md §"gemmExpert weight layout".

### C API Completeness
- [x] **MLA attention — code complete in `src/mla/`, re-exported from `root.zig`** (config, cache, core modules wired into the library; 11/11 standalone tests passing). C API integration (`kt_mla_*` in `main.zig`) assigned Dev A — see Task assignments above.
- [x] **Gate (`kt_gate_*` functions)** — `kt_gate_new` allocates a `GateContext` (BF16 weight pointer + dims); `kt_gate_free` destroys it; `kt_gate_forward` calls `moe.routeExperts` with the stored weights and `null` pool. BF16-only (`@panic` on other dtypes, matching the rest of the C API).
- [x] **Linear/MLP (`kt_linear_*`, `kt_mlp_*`)** — both wrap a real GEMM (BF16 weight + BF16 input → F32 output → BF16 output). Linear = 1 GEMM; MLP = gate GEMM + up GEMM + F32 SwiGLU + BF16 round-trip + down GEMM. Also fixed `kt_linear_config_t` / `kt_mlp_config_t` / `kt_gate_config_t` to match the C header field names (was: `weight_ptr`/`dtype`; now: `weight`/`weight_type` for Gate, `proj`/`proj_type`/`hidden_type` for Linear, `gate_proj`/`up_proj`/`down_proj`/`*_type`/`hidden_type` for MLP). Linear+MLP end-to-end test in test_kernels.zig exercises the same pipeline via `gemm_bf16` directly. 25/25 tests pass.
- [x] **SFT/LoRA backward pass** (Dev B) — `TpMoeSft.backward` computes grad_input, 6 LoRA grads (gate/up/down × A/B), base-weight grads, routing-weight grads. `kt_moe_backward` C API export. Smoke test passes (43/43 total).
- [x] **FP8 layerwise transport (`kt_fp8_*`) — #4 DONE (2026-08-31, verified by coordinator)** — all 10 symbols implemented in `src/main.zig` (+351 lines): `kt_fp8_layerwise_init`, `kt_fp8_transport_{new,free,close,closed,rank,tp_size,num_experts,run_producer,wait}`. Callback-only plumbing per spec (real dequant+GEMM stays in the caller via `kt_matmul_fp8`); `cuda_device`/`local_gpu_ptrs`/`timeout_ms` documented no-ops (CPU port); pointer math matches reference layout `(slot * tp_size + rank) * 4 + kind`, slot = expert % 2 (fp8_layerwise_transport.cpp:409-417); `Fp8ControlHeader` extern struct mirrors fp8_layerwise_transport.cpp:49-56. Verification (evidence standard): 6 variants build exit 0; `zig build test` 47/47 (34 kernels + 11 MLA + 2 new FP8 transport tests), 0 leaks, exit 0; `tools/verify_abi.py` confirms all 10 transport symbols exported by every variant — only `kt_mla_load_weights` remains missing (next item). Tests: `tests/kernels/fp8_transport_tests.zig` (lifecycle + exact reference pointer-math assertions + null-handle sentinels), wired as third suite in `build.zig`.
- [x] **`kt_mla_load_weights` C API — DONE (2026-08-31) — ABI is now 100% complete against the header.** Mirrors the C++ reference semantics (TP_MLA_Common::load_weights sets `weights_loaded`; forward() throws "Not Loaded" without it — mla-tp.hpp:39,86): `MlaContext.weights_loaded` flag set by `kt_mla_load_weights`; `mlaForwardImpl` (covers forward/prefill/decode) panics with a clear message if called before load_weights. `kt_mla_new` additionally validates all 7 weight pointers non-null up front (C header types them non-nullable `*anyopaque`, but a caller bug passing 0 would otherwise surface as a far-away segfault). All 7 `kt_mla_*` exports marked `pub` for Zig-level test access (no ABI change). Verification: `tools/verify_abi.py` **full PASS — all 53 declared symbols exported by all 6 variants** (first time); `zig build test` 48/48 (34 kernels + 11 MLA + 2 fp8 + 1 new MLA C-API lifecycle test `tests/kernels/mla_capi_tests.zig`, fourth suite), 0 leaks, exit 0; 6 variants build exit 0.

---

## 🟢 Medium Priority

### Build System
- [x] **Test runner actually runs tests** — replaced `test_step.dependOn(&test_obj.step)` with `test_step.dependOn(&b.addRunArtifact(test_obj).step)` in build.zig for both test modules. Fixed use-after-free in `src/runtime/cpu_detect.zig::detectCpuLinux` (model_name pointed into a soon-to-be-freed buffer). Fixed 3 `.flags = std.ArrayList(u8).init(allocator)` type mismatches.
- [x] **Multi-variant build** — `zig build all-variants` builds all 6 variants with distinct names (`libkt_kernel_ext_{variant}.so`). Per-variant C macros preserved; single-variant `-Dvariant=` path kept for dev.
- [x] **Variant-specific library names** — each variant installs as `libkt_kernel_ext_{suffix}.so`.

### Runtime
- [x] **Worker pool work-stealing** — replaced busy-wait spin + broken queue with atomic-counter work-stealing via `std.c.pthread_mutex_t`/`pthread_cond_t`. Workers block on a condvar when idle (no CPU burn). `doWorkStealingJob(count, fn)` distributes `count` tasks across all subpool threads. Verified: 1000 tasks across 4 threads complete correctly.
- [x] **NUMA topology** — `numaNodeOfCpu()` reads `/sys/devices/system/cpu/cpu{N}/topology/physical_package_id`; `getCpuCountPerNuma()` parses `/proc/cpuinfo` (correctly returns 16 CPUs on 1 NUMA for Ryzen 5800H).
- [x] **Verify worker_pool.zig** — work-stealing pool with NUMA subpools compiles and runs.
- [x] **Wire work-stealing into MoE kernels** (Dev A+b, commit b762a2e) — `TpMoe.forward` now gates on `config.pool`: parallel path uses `doWorkStealingJob` with per-expert private BF16 scratch (zeroed, sequential reduction); sequential fallback unchanged. 44/44 tests pass including pool-vs-sequential equivalence. Review gate (data-race-free reduction, allocator symmetry, sequential fallback, BF16-scratch semantics) confirmed.
- [x] **Wire work-stealing into SFT/LoRA path** (Dev A+b, commit d1e0adb, reviewed 2026-08-30) — `TpMoeSft.forward_sft` mirrors the MoE pattern: branches on `self.moe.config.pool`, dispatches per-expert tasks via `doWorkStealingJob` on `subpool[0]`, each task owns its BF16 scratch (`[][]amx.bf16`), writes a disjoint `save_for_backward` cache slice via precomputed `expert_token_offset[e]`, sequential reduction after join (BF16 → F32 via `bf16_to_f32`, weighted by routing w). SFT-specific `g_parallel` (moe_sft.zig:67) is module-scoped — decoupled from moe.zig's context. Review gate (a–e + backward-cache) confirmed: 6 variants build, 45/45 tests pass (34 kernels incl. new "SFT forward_sft: work-stealing pool matches sequential (equivalence)" + 11 MLA), 0 leaks. ABI verifier PASS on all non-FP8 symbols (10 `kt_fp8_transport_*` + `kt_mla_load_weights` missing are pre-existing #4 carve-outs, not #5 defects).

---

## 🔵 Low Priority / Nice to Have

### Python Packaging
- [x] **Minimal ctypes wrapper** (`python/kt_kernel/__init__.py`) — pure-Python `ctypes` wrapper that `dlopen`s the variant `.so` (auto-detected from `/proc/cpuinfo`), exposes `kt_version`, `kt_get_cpu_variant`, worker pool, CPUInfer, Linear, Gate, MLP, and bf16 conversion functions. Smoke-tested: `import kt_kernel; print(kt_kernel.kt_version())` works.
- [x] **`pyproject.toml` + `setup.py` for `kt-kernel` wheel** — setuptools with package-data for the 6 variant `.so` files; custom `build_ext` runs `zig build all-variants` and bundles them into `python/kt_kernel/`. Verified: `python3 setup.py build_ext --inplace` → `import kt_kernel` → `kt_version()` returns `b'0.6.1-zig'`.
- [x] **CI/CD for multi-variant wheel building** — `.github/workflows/wheels.yml` builds all 6 variants and packages the wheel.
- [x] **Minimal pybind11 wrapper in C++ that loads Zig `.so`** — DONE (4040b5b): `bindings/kt_kernel_pybind.cpp` + `build.sh`, module named `kt_kernel_ext` (C++ reference drop-in), gated by `tools/audit_layout.py`. Full entry under GGML/pybind11 Tier-1 below.

### Advanced Features
- [x] **SFT/LoRA backward pass** — forward_sft (Dev A) + backward (Dev B) complete with LoRA kernels; kt_moe_forward_sft/kt_moe_backward/kt_moe_update_lora_weights C API exports. 43/43 tests pass.
- [x] **Speculative decoding (MTP head) — RESOLVED: out of scope for this .so** (verified 2026-08-31) — MTP/NextN speculative decoding in ktransformers is a **serving-engine feature that runs on the GPU via SGLang/vLLM EAGLE flags** (`--speculative-algorithm EAGLE`, doc/en/DeepSeek-V4-Flash.md:188-198), NOT a CPU kernel in kt-kernel. Evidence: (1) zero MTP/nextn implementation anywhere in kt-kernel C++ or the kernel-side Python — only the config field `num_nextn_predict_layers=1` + weight-checkpoint skipping of the extra layer (check.py:67, sft/artifacts.py:319-350 reject MTP tensors); (2) LESSONS_KTRANSFORMERS.md:123 lists it as a feature of the *framework*, not the kernel library; (3) the archive's modeling_deepseek_v3.py has no NextN class. The draft head itself is a small decoder (embed×2 → linear → decoder layer → norm → lm_head) that the serving engine orchestrates on-GPU against the CPU-offloaded main model via the FP8 layerwise transport (#4, landed). If CPU-side drafting is ever wanted, every primitive already exists here (DecoderLayer + lm_head + our model-orchestration tower) — but that would be new design work, not a port. Closing the TODO as reference-absent, same verdict as the operators/kml/ discovery.
- [x] ARM/KML backend (NEON/SVE) — RESOLVED: 2026-09-02
  - [x] **aarch64 variant detection** — `selectBestVariant` picks `sve2` > `sve` > `neon` for `cpu.vendor == .arm` or `cpu.arch == "aarch64"`. 7 tests in `tests/kernels/aarch64_detect_test.zig`.
  - [x] **NEON Phase-2: BF16 GEMM kernel** — `src/kernels/arch/gemm_224_bf16_neon.zig` (205 lines) parallels the A1 work in `src/kernels/amx/gemm_224_bf16.zig` with `@Vector(4,f32)` (4 f32 lanes = 128-bit NEON register). Comptime-dispatched: x86_64 builds hit the SSE fallback; aarch64 cross-build emits NEON fmla/fmul. Dispatcher (`gemmExpert`) routes to scalar or vectorized based on the NEON comptime gate. 4 tests in `tests/kernels/neon_kernel_test.zig`: vecF32 layout, dispatcher routing, vectorized-vs-scalar reference, correctness on x86_64 fallback.
  - [x] **NEON feature detection** — `src/kernels/arch/neon.zig` exposes `NeonFeatures.available` (comptime flag), `selectBestVariant` honors it. 1 test in `tests/kernels/neon_arch_test.zig`.
  - [x] **Cross-compiled aarch64 `.so`** — `zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu` produces `zig-out/lib/libkt_kernel_ext_neon.so` (5.4 MB ARM ELF, 112 symbols). `tools/verify_abi.py` gates it PASS: 105/105 header prototypes exported, arity audit clean. File header verified: ELF 64-bit LSB aarch64, dynamically linked, with debug_info. Build-time footgun guard: passing `-Dvariant=neon` without `-Dtarget=aarch64-linux-gnu` exits with a clear error (no silent x86 "neon" build).
  - **Remaining follow-ups (out of scope for this workstream)**: SVE/SVE2 GEMM kernels (the dispatcher already routes to `sve2 > sve > neon` but the actual vectorized kernels beyond the BF16 one are not yet ported; further work only justified when a real aarch64 host is in the dev rotation).
- [x] **GGML Q8_0 kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q8_0.zig`: byte-exact `BlockQ8_0` (34 bytes: f16 `d` + 32×i8 `qs`, `@sizeOf` compile-checked) vs ggml-common.h; quantize/dequantize matching ggml-quants.c:276/553 (`d = amax/127`, `qs = round(x·1/d)`, `y = qs·d`); f16↔f32 via Zig's **native f16** `@floatCast`/`@bitCast` (first attempt hand-rolled the bit math — buggy subnormal path, replaced by native casts, lesson for LESSONS); scalar GEMM (BF16 activations × Q8_0 weights → F32, on-the-fly dequant). 5 exact-value tests in test_kernels.zig (f16 round-trip, 34-byte layout, quantize/dequant tolerance, 2 GEMM constants). Wired into root.zig with comptime fn-refs (emission verified via nm). Remaining formats: Q4_K → Q6_K → Q5_K (defer Q8_K — super-block scale/min layout is the fiddliest and least used); C-API quantize/dequantize exports are Phase 2.
- [x] **GGML Q4_K kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q4_k.zig`: byte-exact `BlockQ4_K` (144 bytes: f16 `d`+`dmin`, 12-byte 6-bit packed `scales`, 128-byte 4-bit `qs`, `@sizeOf` compile-checked) vs ggml-common.h; `nearestInt` (magic-number trick, ggml-quants.c:621), `getScaleMinK4` (6-bit scale/min unpack, :880), full `makeQkx2Quants` error-minimizing search (:799, 20-step), `quantizeRowQ4_K` (:1457), `dequantizeRowQ4_K` (:1529); scalar GEMM (BF16 × Q4_K → F32). 5 tests in test_kernels.zig (nearest_int known values, 144-byte layout, scale/min unpack scheme, round-trip accuracy <15% rel on smooth data, GEMM constant 256.0). Wired into root.zig (emission verified via nm).
- [x] **GGML Q6_K kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q6_k.zig`: byte-exact `BlockQ6_K` (210 bytes: `ql[128]` low nibbles + `qh[64]` 2-bit highs + 16×i8 `scales` + f16 `d`, compile-checked); `nearestInt`, `makeQxQuants` (rmse_type=1 weighted search + ±9 refine loop, ggml-quants.c:628), `quantizeRowQ6_K` (:1869 incl. all-zero fast path), `dequantizeRowQ6_K` (:1939 — 6-bit quants = ql nibble | qh 2-bit<<4, minus 32); scalar GEMM. Zig gotcha found: C packs L as `uint8_t` — Zig `i8 >> 4` is arithmetic and `(3<<6)` overflows i8, must cast to u8 before the qh packing. 4 tests in test_kernels.zig (210-byte layout offsets, round-trip rel err < 10% on smooth data, all-zero block → d=0 → all-zero output, GEMM vs dequantized reference). Wired into root.zig; emission verified via nm.
- [x] **GGML Q5_K kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q5_k.zig`: byte-exact `BlockQ5_K` (176 bytes: f16 `d`+`dmin`, 12-byte 6-bit packed `scales` [same scheme as Q4_K], `qh[32]` high-bit plane, `qs[128]` low nibbles, compile-checked); quantize reuses `makeQkx2Quants` with q5_K's parameters (nmax=31, rmin=-0.5, nstep=15); dequant: `y = d·sc·((qs nibble) + (qh bit?16:0)) − dmin·m`. Two subtle reference behaviors caught by tests: (1) the `qh` pointer NEVER advances in the C code — all 4 chunks of 64 weights reuse the same 32 qh bytes, separated by shifting bit masks (m1=1,m2=2, <<2 per chunk); (2) the output pointer `y` advances every chunk (×64), unlike the block-strided intuition. 5 tests in test_kernels.zig (176-byte layout offsets, round-trip rel err < 12%, all-zero input → exact zero, qh high-bit plane exercised by wide-range data, GEMM vs dequantized reference <1% quant error). Wired into root.zig; emission verified via nm. Remaining: Q8_K (DONE — see the Q8_K entry below).
- [x] **GGML C-API wrappers — Phase 2 DONE (2026-08-31)** — 12 new exports in `src/main.zig` + 12 declarations in `include/kt_kernel.h`: `kt_quantize/dequantize_q{8_0,4_k,5_k,6_k}` (one-row F32↔block conversions) and `kt_matmul_q{8_0,4_k,5_k,6_k}` (BF16 activations × quantized weights → F32, ldb in BLOCKS). Since the block layouts are byte-exact vs GGUF, Python can pass weights loaded from .gguf files straight into the matmuls. `verify_abi.py` now gates **65 symbols × 7 variants, full PASS**. 7 C-API lifecycle tests in `tests/kernels/ggml_capi_tests.zig` (fifth suite, rooted at main.zig): per-format round trips through the real exports + constant-weight matmuls. main.zig GGML/amx aliases marked `pub` for test access (no ABI effect). **GGML workstream COMPLETE for the 4 workhorse formats** (Q8_0, Q4_K, Q5_K, Q6_K); Q8_K landed after (see next entries).
- [x] **GGML Q8_K kernel — DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q8_k.zig`: byte-exact `BlockQ8_K` (292 bytes: **f32** `d` [not f16!], 256×i8 `qs`, 16×i16 `bsums` — the dot-product sidecar, compile-checked). The simplest K-quant math: signed `iscale = -127/max` (max carries sign — the IQ2_XXS-compat choice), `qs = MIN(127, nearestInt(iscale·x))` with **no lower clamp needed** (|iscale·x| ≤ 127 by construction), `bsums[j] = Σ qs` per 16-group, all-zero fast path. `kt_quantize/dequantize/matmul_q8_k` C-API + header declarations — ABI now **68 symbols**. 5 kernel tests + 2 C-API tests (292-byte layout, round-trip ≤ d/2 exact bound + rel <2%, bsums exact-group-sum invariant, zero fast path, GEMM vs dequantized reference <0.5%). **GGML workstream COMPLETE — all 5 formats.**
- [x] **GGML Q2_K kernel — DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q2_k.zig`: byte-exact `BlockQ2_K` (84 bytes: 16×u8 `scales` [4-bit scale+min, ONE byte per sub-block — unlike Q4_K/Q5_K's 6-bit scheme], 64×u8 `qs` [2-bit quants, 4/byte], f16 `d`+`dmin`, compile-checked). Quantize (:891): per-16 `makeQkx2Quants(nmax=3, rmin=-0.5, nstep=15, use_mad=TRUE)` with |x| weights (the q2_K-specific params), 4-bit scale/min packing with q4scale=15, clamp [0,3], 2-bit plane packing. Dequant (:961): per-128 chunk, 4 shift planes × 2 sub-blocks (q[l], q[l+16]) — `y = d·sc_lo·((qs>>shift)&3) − dmin·sc_hi`. Two Zig gotchas caught by tests: (1) `u3` shift loop counter wraps at 8 (6+2→0) and re-enters → scales[] OOB; use a usize plane counter and `@intCast` at the use site. (2) The honest round-trip floor for 2.625 bits/weight is rel ~0.39 on smooth data (verified against a Python port of the ggml reference) — threshold set to 45% with that evidence, not a guess. 4 tests (84-byte layout, round trip < 45% rel, all-zero, GEMM vs dequantized ref). Wired into root.zig; emission verified via nm. 111 tests total.
- [x] **GGML Q3_K kernel — DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q3_k.zig`: byte-exact `BlockQ3_K` (110 bytes: `hmask[32]` high-bit plane, `qs[64]` 2-bit quants, `scales[12]` packed 6-bit, f16 `d`, compile-checked). The most layout-intricate format: 16 int8 scales packed into 12 bytes via a 4-way nibble+2-bit crisscross; dequant unpacks via the aux[kmask1/kmask2] bit-shuffle. `make_q3_quants` (nmax=4, 5-try coordinate descent). The high bit SUBTRACTS 4 (inverting the +16 pattern of Q5_K). Three bugs caught by tests+debug: (1) my unpackScales masked aux[2] top byte (scales[11] lives in bits 24..31) — no effect on this data but latent; (2) `u3` shift counter wraps (same as Q2_K); (3) the `m` hmask-bit counter must advance across ALL 8 planes (2 chunks x 4 planes) — resetting it per chunk made chunk-1 weights read wrong hmask bits (rel 2.37 → 0.33 after fix, debug-verified). Honest threshold: 35% (measured 0.33 with the Python-port cross-check). 4 tests (110-byte layout, round trip < 35% rel, all-zero, GEMM vs dequantized ref). 115 tests total. **All 7 standard K-quant formats + Q8_0 now complete.**
- [x] **GGML IQ4_XS kernel — DONE (2026-08-31)** — `src/kernels/amx/gemm_224_iq4_xs.zig`: byte-exact `BlockIQ4_XS` (136 bytes: f16 `d`, u16 `scales_h` [8×2-bit highs crisscrossed], `scales_l[4]` [8×4-bit lows, 2/byte], `qs[128]` [4-bit indices], compile-checked). **The first non-linear format**: quant values are INDICES into `kvalues_iq4nl` {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113} looked up via `bestIndexInt8` (binary search). Quantizer: shared `quantizeRowIQ4NLImpl` (per-32 weighted least-squares with ntry=7 refinement + the d=-max_scale/32 negative-scale convention). Honest threshold: 25% (measured 0.21 — the non-linear table + 6-bit per-32 scales have their own noise floor; sub-block scales come out tiny on smooth sinusoids, debug-verified). 5 tests (136-byte layout, kvalues table, round trip < 25% rel, all-zero, GEMM vs dequantized ref). 120 tests total. **First IQ-family format landed.**
- [x] **GGML IQ2_XXS kernel — DONE (2026-09-01)** — `src/kernels/amx/gemm_224_iq2_xxs.zig`: byte-exact `BlockIQ2_XXS` (66 bytes: f16 `d` + `qs[32]u16`, compile-checked). **Grid-based 2.0625 bpw**: each u16 encodes a grid-index byte + 7-bit sign-selector; per-32 group a 4-bit db factor in the top nibble of `aux32[1]`. Three byte-exact lookup tables transcribed: `IQ2XXS_GRID` (256×u64, 8 magnitude bytes each from the {0x08,0x19,0x2b} alphabet), `KSIGNS_IQ2XS` (128×u8), `KMASK_IQ2XS` (8×u8). Quantize: **stubbed with @panic** (requires runtime kmap/kneighbors init via `iq2xs_init_impl` — separate workstream, documented). Dequant + scalar GEMM: real, byte-exact vs ggml-quants.c:2488. LE byte-order gotcha caught by the hand-crafted-value test: grid64 byte 0 = LSB (= 0x2b for grid[1]), not the printed hex MSB (= 0x08). 5 tests (66-byte layout, 3 lookup tables vs ggml, zero block, hand-crafted known values, GEMM with 2 weight columns). 125 tests total.
- [x] **GGML IQ3_XXS kernel — DONE (2026-09-01)** — `src/kernels/amx/gemm_224_iq3_xxs.zig`: byte-exact `BlockIQ3_XXS` (98 bytes: f16 `d` + `qs[96]` [64 grid indices + 32 scales+signs], compile-checked). **Grid-based 3.0625 bpw**: per 32-weight group, 4 sub-blocks of 8; each sub-block reads TWO u32 grid entries (4 magnitude bytes each, alphabet {0x04..0x3e}); db = d·(0.5+(aux32>>28))·0.5 from the top nibble of the group's scale area; sign palette shared with IQ2_XXS (KSIGNS_IQ2XS). 256×u32 `IQ3XXS_GRID` table byte-exact from ggml-common.h. Quantize: **DONE (2026-09-01, follow-up commit) — the @panic stub was replaced** by the wrapper in `gemm_224_iq3_xxs.zig:quantizeRowIQ3_XXS` (lazy-init global kmap + mutex) delegating to `iq3_quantize.quantizeRowIQ3_XXS_WithInit` (see the IQ3 quantize entry below). Dequant + scalar GEMM: real, byte-exact vs ggml-quants.c:2575. 5 tests (98-byte layout, grid tables vs ggml, zero block, hand-crafted known values, GEMM 2-column) + 3 new (kmap exact+correspondence, off-grid validity, quantize round-trip).
- [x] **IQ2_XXS kmap/kneighbors init — DONE (2026-09-01)** — `src/kernels/amx/iq2xs_init.zig`: port of `iq2xs_init_impl` (ggml-quants.c:2853-3255). `KGRID_2BIT_256` (256×u16 packed 2-bit fingerprints) byte-exact from the reference. `initIq2XsData`: builds the 8-byte position grid (pos[k]=2·nibble+1), the exact-match kmap (all 256 grid entries map to themselves — verified), and the kneighbors table for the 43436 off-grid fingerprints (2-pass: count + fill, nwant=2 nearest via Hamming distance, sorted). 2 tests verify: (a) all 256 exact matches land correctly; (b) all 43436 off-grid entries have valid negative-encoded offsets pointing into kneighbors with count>0. This unblocks the full IQ2_XXS quantize (the @panic stub). 132 tests total. **IQ2_XXS quantize now unblocked — the remaining work is the nearest-grid search loop (quantize_row_iq2_xxs_impl) using these tables.** **IQ2_XXS full quantize — DONE (2026-09-01)**: `src/kernels/amx/iq2_quantize.zig` — port of `quantize_row_iq2_xxs_impl` (ggml-quants.c:3294-3465): `makeQPQuants` (weighted LS + 5-try coordinate descent), `findBestNeighbour` (on/off-grid refinement), sign extraction with odd-parity flip, ±6-scale grid-search refinement via the kmap, d=max_scale/31 global scale, 4-bit per-32 scale in the top nibble. Constant-0.5 debug verification confirms correct algorithm (values map to the {8,25,43} magnitude alphabet at the right scale; dst[1]=0.4686 ≈ 25·db). Sinusoid test threshold is 150% rel — algorithm-correctness gate, not precision (the reference always uses importance weights; without them, low-|x| values get poorly quantized BY DESIGN). 133 tests. **IQ2_XXS fully complete: quantize + dequantize + GEMM + kmap init.** **All standard GGML quant formats now complete** (7 linear K-quants + Q8_0, IQ4_XS, IQ2_XXS, IQ3_XXS = 10 formats). **IQ3_XXS full quantize — DONE (2026-09-01)**: `src/kernels/amx/iq3xs_init.zig` — port of `iq3xs_init_impl` (ggml-quants.c:3703-3902, grid_size=256): `KGRID_Q3XS_256` byte-exact, kmap **4096** entries (4× 3-bit nibbles, not the 43692 of IQ2's 4×2-bit), positions 4 bytes (odd 1..15), kneighbors nwant=2, 2-pass count+fill. `src/kernels/amx/iq3_quantize.zig` — port of `quantize_row_iq3_xxs_impl` (:3938-4150): **NO makeQPQuants initial fit** (unlike IQ2) — the ±15-scale grid search itself produces the levels (kMaxQ=8, 3-bit), sigma2 = **2·**sumx2/QK_K (differs from IQ2), GROUP_MAX_EPS = 1e-8, off-grid redo pass at the final scale, d = max_scale/31 × **1.0125 fudge factor** (:4137), 7-bit sign packing at strides {0,7,14,21}, 4-bit per-32 scale in the top nibble. **Grid-index correspondence verified with evidence** (Python cross-check vs the C tables): IQ3XXS_GRID[gi] bytes = ALPHABET[nibble(grid[gi])] bijection {0..7}→{4,12,20,28,36,44,52,62}, and both Zig tables byte-exact vs the C — so a kmap grid_index is a valid IQ3XXS_GRID dequant index (round-trip correct by construction). Verification: constant-0.5 → mean 0.4917, max err 0.139 (3-bit grid expected); sinusoid → RMS rel err 20.8% (null importance weights, same rationale as IQ2). 140 tests total. **IQ3_XXS fully complete: quantize + dequantize + GEMM + kmap init — the quantize-with-kmap recipe now proven twice (IQ2_XXS, IQ3_XXS).** **IQ2_XS — DONE (2026-09-02, c6f41e6)**: `src/kernels/amx/gemm_224_iq2_xs.zig` + `iq2xs_init.zig` generalized (initIq2GridData core + KGRID_2BIT_512 byte-exact + initIq2XsData512, nwant=2). 2.3125 bpw, 74-byte block: **9-bit** 512-grid index | signs<<9 in u16 qs + **separate** u8 scales[8] (4-bit dual-nibble per 32 weights — l=0,1→lo nibble, l=2,3→hi). db = d·(0.5+nibble)·**0.25** (XXS uses 0.5). Quantize (:3472-3650): 16-weight groups, ±9-scale search, off-grid redo, d=max_scale/31 (no fudge). Round-trip evidence: const-0.5 → max err 0.037 (XXS: 0.149 — the denser grid measurably finer); sinusoid RMS rel 0.367 (XXS 0.42), 150% max-rel bound (null importance weights, same rationale). 5 tests. Tables byte-exact vs C (Python cross-check).  **IQ2_S — DONE (2026-09-02, ede64ea)**: `src/kernels/amx/gemm_224_iq2_s.zig` + `iq2xs_init.zig` KGRID_2BIT_1024 + initIq2SData (**nwant=1** — the only 2-bit format refining off-grid to the single nearest neighbor). 82-byte block: **10-bit** grid index split qs-lo(8)+qh-hi(2, 4/byte @2-bit stride); **RAW 8-bit sign bytes** (no KSIGNS palette, no parity trick — the S format spends its extra bits buying back the sign bit); quantize: sigma2=2·sumx2/QK_K, null-weight fallback **0.25·sigma2+x²**, no initial fit, d=max_scale/31·**0.9875**. 6 tests green on FIRST run (incl. qh-hi-bits selecting grid [256] — the 10-bit split proof).  **IQ3_S — DONE (2026-09-02, a543876)**: `src/kernels/amx/gemm_224_iq3_s.zig` + `iq3xs_init.zig` 3-bit core generalized + KGRID_Q3XS_512 + initIq3SData (**nwant=3**, ggml:3762). 110-byte block: **9-bit** index qs-lo+qh-hi (1 bit/sub-block-of-4, 8/byte @slot%8), **RAW signs** [32]u8, scales[4] own array with the NEW scheme **db = d·(1+2·nibble)** (2-bit family: 0.25·(0.5+n)). Quantize (:4169-4350): 8×4-weight sub-blocks/group, is_on_grid starts **FALSE** + redo pass runs for ALL sub-blocks (ggml:4283 commented-out `continue`), d=max_scale/31·**1.033**. 5 tests green on FIRST run; const-0.5 probe → mean 0.5038 (finest grid format).  **IQ1_S — DONE (2026-09-02, 127bbef)**: `src/kernels/amx/gemm_224_iq1_s.zig` + `iq2xs_init.zig` KGRID_1BIT_2048 + initIq1SData (**nwant=3**, same 43692 kmap — nibble {0,1,2}→byte {ff(-1),00(0),01(+1)}). 50-byte block: qs[32] lo + qh[8]u16 packing 4×3-bit index-his(12b) + **3-bit scale**(bits 12-14) + **delta-sign bit**(15) — no separate scales array. Dequant (:2651): dl=d·(2n+1), delta=±0.125, y=dl·(grid[j]+delta) — the delta shifts the 1-bit code to effective {−1.125, 0.125, 1.125}. Quantize (:4508-4670) **EXACT — no grid search**: qsort (value,idx) pairs + prefix sums + exhaustive (i1,i2) split search against BOTH delta shifts; **d=max_scale/15** (3-bit scale, not 31) × 1.125 fudge ('Don\'t ask' — ggml:4661). findBestNeighbour2 with the **exhaustive 2048-entry fallback** (empty-neighbor fingerprints). **Round-trip test caught a real bug on first run**: the refine pass must read the runtime POSITION grid ({1,3,5}) not the dequant grid ({−1,0,1}) — (pg−1)/2 over −1 overflows the xg index. Gotchas: 'i8' AND 'i1' shadow primitives. 6 tests; const-0.5 → mean 0.5624 (format floor — 0.5 is unreachable at 3 levels).  **IQ1_M — DONE (2026-09-02, d0b0211) — ALL 15 GGML FORMATS COMPLETE**: `src/kernels/amx/gemm_224_iq1_m.zig`. 56-byte block: qs[32]+qh[16]+scales[8], **NO d field** — the f16 scale hides in the top nibbles of sc[0..3] (extractScale reconstruction). Delta flips INDEPENDENTLY per group-half (4 mask variants 0x00/0x80/0x08/0x88); quantize = SSD split search against all 4 delta combos simultaneously + negative-scale k-involution (0↔3, 1↔2) + **GLOBAL d refit** over quantized 3-bit scales + 1.1125 fudge. Alignment lesson: extern struct of u8 arrays = natural alignment 1 — the C's u16* scales access is silent UB; Zig panics; all access byte-composed (sc16 helper). **Plus the kmap-init 22x speedup** (d0b0211): qsort-then-scan replaced by bounded HISTOGRAM (d2 ≤ 8·36=288) single-pass — same neighbor set + same order; iq1s 130.6s→6.0s, iq2s 51.7s→3.0s. Process-wide kmap CACHE (mutex + page_allocator, *Owned variants for old semantics). Fixed a latent landmine: gemm_224_iq3_xxs.zig:241 used nonexistent std.Thread.Mutex (never compiled — lazy analysis hid it; std.atomic.Mutex is the 0.16 name). 134 kernels tests, 186 OK/0 leaks total. **THE GGML FORMAT SET IS COMPLETE: 15/15** (Q8_0 + 7 K-quants + IQ4_XS + IQ4_NL + IQ2_XXS/XS/S + IQ3_XXS/S + IQ1_S/M) — every kt_type_t enum value now has a real kernel with quantize+dequantize+GEMM+tests.
- [x] **DeepseekV3DecoderLayer — model-orchestration layer DONE (2026-08-31)** — `src/kernels/moe/deepseekv3_layer.zig`: the Zig-native port of the Python reference layer forward (kt-kernel/examples/modeling_deepseek_v3.py:1196-1226 + MoEGate:394-461). Single-sequence orchestration wiring the existing kernels: residual → RMSNorm(attn_norm_w) → MlaEngine.forward (continuous KV, kv_start_pos param) → residual add → RMSNorm(ffn_norm_w) → routeExpertsDeepSeek (sigmoid + e_score_correction_bias + group-top2 noaux_tc + norm_topk_prob + routed_scaling_factor) → TpMoe.forward (work-stealing when pool configured) → residual add → BF16 out. Owned contexts: MlaEngine + MlaKvCache + TpMoe created in init, freed in deinit; weight pointers borrowed (caller-owned, standard C-API convention); per-call scratch allocated symmetric. NOTE: the C++ reference's operators/kml/deepseekv3.hpp does NOT exist in the checkout (ARM/KML-only include, dir absent) — the Python modeling file was the real spec. Fixed a stray root.zig import left by the parallel session (operators/deepseekv3_layer.zig — nonexistent path). Test: 2-step decode (kv_start_pos 0 then 1) with zero weights asserting exact residual semantics (output == input through BF16), 0 leaks. **Model-orchestration: DecoderLayer + C-API + pybind11 done.** C-API: kt_dsv3_layer_new/forward/free + kt_dsv3_layer_config_t extern struct (34 fields, config by POINTER — pointer-passing, not by-value; ABI 87 -> 90 symbols, double-gated). New suite tests/kernels/dsv3_layer_capi_tests.zig (2 decode steps through the extern-struct boundary, exact residual semantics). pybind shim: m.dsv3.DeepseekV3DecoderLayer + LayerConfig (size_t pointer fields with per-field conversion; safe defaults — rope_theta=10000 because 0.0 produces NaN in RoPE, found live via pybind e2e). **DeepseekV3Model + ForCausalLM DONE (follow-up commit)** — src/kernels/moe/deepseekv3_model.zig: Model = N × DeepseekV3DecoderLayer (ping-pong hidden buffers) + final RMSNorm (inline f32 helper); ForCausalLM = Model + lm_head GEMM (gemmExpert, [qlen,hidden]×[vocab,hidden]^T → f32 logits). C-API: kt_dsv3_model_new/forward/free + kt_dsv3_causallm_new/forward/free + kt_dsv3_model_config_t (embeds kt_dsv3_layer_config_t as per-layer template + num_layers/final_norm/lm_head/vocab_size). ABI 90 → 96 symbols, double-gated. 2 kernel tests (2-layer residual chain + final-norm=1 → exact 1.0; lm_head weights=1 → logits=64 exact), 106 total.
- [x] **Python ctypes GGML bindings — DONE (2026-08-31)** — `python/kt_kernel/__init__.py`: block-size constants (`Q8_0_BLOCK_BYTES`=34 … `Q8_K_BLOCK_BYTES`=292), raw bindings for all 15 `kt_quantize/dequantize/matmul_q{8_0,4_k,5_k,6_k,8_k}` exports, plus convenience API: `quantize_row(fmt, src)` (list→packed block bytes), `dequantize_row(fmt, blocks, k)` (bytes→c_float array, with expected-size validation), `matmul_quantized(fmt, a_f32, b_blocks, m, n, k)` (auto f32→BF16 activation conversion via kt_f32_to_bf16). Footgun guards: ValueError on non-multiple-of-QK lengths, unknown formats, and buffer-size mismatches. Verified end-to-end: all 5 formats round-trip within kernel tolerances, q8_0 GEMM ≈32.0 exact, error paths raise. Bundled variant .so files refreshed to 68-symbol builds (the stale bundle lacked the GGML symbols — found via the import failing with undefined symbol).
- [x] **ABI signature parity audit — Tier 0 DONE (2026-08-31)** — full 86-symbol sweep of header prototypes vs Zig export signatures. Found and fixed 3 real divergences (stack-corruption risk for header-following callers): `kt_mla_new` (dropped the extra cpuinfer param; header contract is 1 arg), `kt_mla_forward` (rewritten to the 8-arg paged/batched C contract: qlens/page_tables/kv_lens arrays; qlen_count==1 fully supported, >1 panics with a clear message — paged-attention indirection is future work; page_tables accepted-but-unused documented), `kt_gate_forward` (header updated to the 7-arg Zig form with topk_ids/topk_weights outputs — the ctypes wrapper + tests already used it; 4-arg C++ minimum noted as not exported). Also gated 18 previously-unheadered .so exports (4 math helpers kt_apply_swiglu/rms_norm/rope + kt_softmax, MLA prefill/decode/update_kv_cache conveniences, MoE forward_gate_up/down, FP8 quantize helpers, scalar matmuls bf16/int8/int4/fp8, worker_pool_get_thread_num) in a clearly-marked 'Zig extensions' header section — ABI: 68 → 86 symbols. **New gates**: `tools/audit_arity.py` (header/Zig arity parser with fn-ptr + multi-line + trailing-comma handling) + wired into `verify_abi.py` as a second gate (FAIL on any name or arity divergence). Side effect: the audit surfaced two latent compile errors in `kt_matmul_int4` (b_scratch pointer-type mismatch, applyScales b/c types) — fixed; INT4 AMX-path tests now actually run (previously silently un-analyzed under lazy compilation). Python wrapper: 18 new raw bindings + argtypes, 61 names in __all__.
- [x] **pybind11 drop-in wrapper — Tier 1 DONE (2026-08-31)** — `bindings/kt_kernel_pybind.cpp` + `bindings/build.sh`: a pybind11 module named `kt_kernel_ext` (matching the C++ reference's PYBIND11_MODULE name, so `from kt_kernel_ext.moe import MOE, MOEConfig` works unchanged) that links against the Zig `.so` and routes every method through the C API. Surface: `moe.MOE` (construct-from-config, load_weights, warm_up, forward, forward_sft), `moe.MOEConfig` (full field set incl. embedded quant_config), `mla.MLA` + `mla.MLAConfig` (paged forward, qlen_count==1), `kvcache.ggml_type` enum, `WorkerPool`/`CPUInfer` (submit is synchronous; the Zig work-stealing pool is reached via the MoE pool config), `kt_version`/`kt_get_cpu_variant`. **The hard part was the ABI**: config structs are passed BY VALUE (~340/272 bytes) — a C++ mirror struct with wrong field order silently corrupts the callee (found live: the shim originally missed `num_gpu_experts`/`gpu_experts_mask`/`physical_to_logical_map` + embedded `quant_config`, and typed `kt_type_t` enums as u8 instead of c_int). Three-layer fix: (1) shim structs transcribed field-for-field from the Zig `extern struct`s; (2) new Zig ABI probes `kt_abi_size_*`/`kt_abi_field_offset` (via `@offsetOf`, test-only exports); (3) new gate `tools/audit_layout.py` — loads the probes from the .so, computes the C++ shim's Itanium-ABI offsets, FAILs on any sizeof/offset divergence. Layout gate caught the MLA u8-vs-int enum bug live. End-to-end verified: MOE construct→load_weights→forward (zero weights→zero output exact) and MLA construct→load_weights→forward through the real pybind11 path with numpy buffers. Third Python surface alongside the ctypes wrapper; build via bindings/build.sh (pybind11 from torch include).

- [x] **LlamaMoe (Zig extension) — first cut DONE (2026-09-02)** — ports
  the C++ `LLAMA_MOE_TP` from `ktransformers/kt-kernel/operators/llamafile/moe.hpp`
  (the framework's default backend for GGUF-loaded checkpoints). The
  Zig class (`src/kernels/moe/llamafile_moe.zig`, 622 lines) implements
  the full per-expert algorithm: BF16 input → Q8_0 quant → SwiGLU
  → Q4_K/Q5_K/Q6_K/Q8_0/Q8_K × Q8_0 dot products → F32 weighted sum.
  Supported weight types: Q8_0, Q4_K, Q5_K, Q6_K, Q8_K. C ABI:
  `kt_llama_moe_{new,free,load_weights,forward}` (4 symbols) +
  `kt_llama_moe_config_t` extern struct. ABI: 105 → 109 symbols,
  verify_abi.py PASS on all 8 variants. The first-cut matmul path
  dequantizes Q8_0 → BF16 and uses the existing `kt_matmul_q*`
  (byte-exact vs llama.cpp, but does an extra dequant/quant round
  trip on the activation). Tests: `tests/kernels/llamafile_moe_capi_tests.zig`
  (2 tests: new+load_weights+free lifecycle, pool-null rejection).
  **Limitations (follow-ups)**: (1) add a true Q4_K × Q8_0 matmul
  to skip the BF16 round-trip; (2) `forward_many` (batched per-m-block
  work-stealing) is a follow-up — the C ABI calls per-token `forward_one`
  for qlen > 1 (correct, not optimal); (3) TP path is `tp_part_idx=0`
  only (the C++ exposes a subpool index; the Zig port uses the pool's
  default subpool for now). **Closes the §1 gap** from the other dev's
  "qué falta" analysis.

### Documentation
- [x] **API documentation** — `docs/api.md` (500 lines, hand-written; toolchain is Zig 0.16.0-dev.2535 which has **no `zig doc` subcommand** and no `-femit-docs` flag, so the doc is not auto-generated). Integrator's quick-reference: 68 kt_* symbols grouped by family (version/detection, BF16, workers, MoE/MLA/Gate/Linear/MLP, GGML block formats + matmuls, FP8 layerwise transport, math helpers), pointers + minimal Python ctypes example, ABI contract (header is the source of truth; `tools/verify_abi.py` is the gate; multi-variant layout), known gaps. Also flags the 4 math-helper exports (`kt_apply_swiglu` / `kt_apply_rms_norm` / `kt_apply_rope` / `kt_softmax`) that exist in the .so but are missing from `include/kt_kernel.h` and therefore not ABI-gated — follow-up to add them to the header. The "zig doc" wording in this TODO is misleading; the toolchain simply does not have it.
- [x] **Porting guide from C++** — `docs/porting-guide.md` (181 lines, committed in `e2c900d`): C++ reference file map → Zig counterpart map, lazy-analysis wiring pattern, comptime fn-ref pattern for `.so` symbol emission, AMX arch-gate, the bf16 f16 cast (Zig 0.16 native f16 vs hand-rolled bit math), and the `@Vector` indexing gotchas. Linked from AGENTS.md.

---

## 📊 Progress Tracking
| Component | Status | % |
|-----------|--------|---|
| Runtime (pool/queue/memory/cpu) | Work-stealing pool + sched_setaffinity pinning (A3) + waitIdle sync (B3) + mbind NumaAllocator + L1/L2/L3 detection (A4); MoE + SFT paths parallel | 100% |
| AMX Intrinsics | Real inline asm + runtime detectAmxSupport guard (A2 fix: comptime featureSetHas + CPUID + arch_prctl) | 95% |
| Buffer Packing | Working | 80% |
| BF16 GEMM | gemmExpert vectorized (@Vector(8,f32), A1) — 5.2x decode / 5.3x prefill measured (B2); AMX tile path verified | 85% |
| INT8 GEMM | Working | 70% |
| INT4/FP8/MXFP4/8 GEMM | INT4, FP8, MXFP4, MXFP8 all done (AMX + scalar fallback) | 100% |
| GGML Quant | All 5 formats (Q8_0/Q4_K/Q5_K/Q6_K/Q8_K) byte-exact + C-API matmuls + Python bindings | 100% |
| MoE Orchestration | Forward + work-stealing parallel + DeepSeek-V3 group-top2 routing (D4); D1/D2/D3 buffer bugs fixed with qlen>1/tp>1 regression tests | 100% |
| SFT/LoRA Training | Forward + backward complete + work-stealing parallel forward; 4 C API exports | 100% |
| C API | 87/87 header symbols exported by all variants; TRIPLE GATE: exports + arity (verify_abi.py) + layout (audit_layout.py); injectable allocator (B1) | 100% |
| Build System | Multi-variant (6 x86 variants + aarch64 neon, distinct .so names) + bench step (B2) | 90% |
| Tests | 68 kernels + 11 MLA + 2 FP8 + 1 MLA C API + 9 GGML C API + 7 aarch64 + 1 neon + 2 allocator C API = 101 total pass, 0 leaks, exit 0 | 100% |
| Python Integration | ctypes wrapper (61+ names incl. GGML) + pybind11 drop-in (4040b5b, layout-gated) + wheel + CI; kt_set_default_allocator binding pending | 85% |

---

## Notes

- **Reference**: Original C++ in `/ai/repos/2026/ktransformers/ktransformers/kt-kernel/`
- **Zig Version**: 0.16.0-dev.2535+b5bd49460
- **Target**: Drop-in replacement for `kt_kernel_ext.so` (6 CPU variants)
- **Python Compat**: Match `kt_kernel_ext.so` C API exactly for pybind11
- **Key Files Modified**:
  - `build.zig`: Uses `src/root.zig` as root, test step enabled
  - `src/root.zig`: Re-exports all submodules
  - `src/main.zig`: Cleaned up forward references
  - `src/kernels/amx/buffers.zig`: Fixed @intFromFloat
  - `src/kernels/arch/amx.zig`: Fixed packed struct, AMX feature check
  - `src/kernels/moe/moe.zig`: Fixed indentation, pointer optionals
  - `src/runtime/cpu_detect.zig`: Fixed || → or
  - `src/runtime/memory.zig`: Fixed alignedAlloc API
  - `src/runtime/worker_pool.zig`: Fixed atomic, DoublyLinkedList, Thread.Pool
  - `tests/kernels/test_kernels.zig`: Fixed import path, type casts

---

## ✅ Additional Fixes (2026-08-29 - Session 2)

### Build Error Resolution
- [x] **Fixed all 31 build errors** in Zig 0.16
- [x] **Fixed 6 main.zig errors** (parentheses, pointer types, opaque types)
- [x] **Fixed 18 moe.zig errors** (type casting, pointer arithmetic, @memset)
- [x] **Fixed 4 gemm_224_int8 errors** (pointer casting, setTile, quantization)
- [x] **Fixed 2 worker_pool.zig errors** (std.Io.Mutex changes)
- [x] **Added 5 missing TpMoe methods** (deinit, warmUp, loadWeightsWithMap, forwardGateUp, forwardDown)
- [x] **Fixed @memset API** (now 2-arg version in Zig 0.16)
- [x] **Fixed all setTile calls** with proper u8/u16 casting
- [x] **Fixed pointer arithmetic** throughout kernels with @ptrCast patterns
- [x] **Fixed f32↔int conversions** (@intFromFloat, @floatFromInt)

### Documentation
- [x] **Updated LESSONS_ZIG.md** with 30+ Zig 0.16 lessons learned
- [x] **Added build success summary** to LESSONS_ZIG.md

## Reference Repos

  The reference repo are is at /ai/repos/2026/ktransformers


### Current Status (2026-08-31)
- **Dev A (2026-08-30)**: Linear/MLP C API wired; kt_*_config_t structs
  synced with C header; MoE gemmExpert weight_ld OOB fixed (6 sites);
  multi-variant build (`zig build all-variants` → 6 .so); minimal Python
  ctypes wrapper; kt_get_cpu_variant lazy-init fixed; SFT/LoRA forward half
  (LoRA kernels, TpMoeSft, forward_sft, C API); runtime hardening (work-stealing
  worker pool, NUMA topology); allocator safety fix (TpMoe stores allocator).
- **Dev B (2026-08-30)**: MXFP4/MXFP8 kernels complete; vectorized applySwiGLU
  complete (bit-exact vs scalar); SFT/LoRA backward half complete (backward
  method, kt_moe_backward export).
- **Audit session (2026-08-31)**: external-dev feedback verified against the
  code (IMPROVE.md) and every real item fixed: P0 guard AMX + MoE buffer bugs
  (ff57125), gemmExpert vectorization A1 (7515468, 5.2x measured), NUMA
  pinning A3 (c35a526), DeepSeek-V3 routing D4 (f5443f3), benchmark suite B2
  (7d2db9c), cache detection A4 + cpuinfer_sync B3 (ca0485a), injectable
  allocator B1 (15a8ea5). ABI 86→87 symbols; tests 90→101; all P0-P3 closed.
- **Last fully-green**: 101 tests pass, 0 leaks, `zig build test` exit 0;
  `verify_abi.py` 87/87 + arity PASS on all variants; `zig build bench` green.
