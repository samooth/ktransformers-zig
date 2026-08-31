# ktransformers-zig TODO

## Status: Alpha (Build + Tests Working)

Last updated: 2026-08-31

**Build: WORKING** - `zig build` produces `zig-out/lib/libkt_kernel_ext.so`
**Tests: WORKING** - `zig build test` RUNS all suites: 53 kernels + 11 MLA + 2 FP8 + 1 MLA C API + 7 aarch64 + 7 GGML C API = 81 total pass, 0 leaks, exit 0. (Test harness now uses a simple-mode runner, `tools/test_runner.zig` — the default Zig 0.16 `--listen=-` IPC handshake fails in this environment; see LESSONS note in build.zig.)
**Multi-variant: WORKING** - `zig build all-variants` produces 6 `.so`, each exporting 59 C API symbols.
**Runtime: WORKING** - work-stealing worker pool (pthread mutex/cond, threads block when idle); NUMA topology via /sys and /proc/cpuinfo.
**SFT/LoRA: FORWARD+BACKWARD COMPLETE** - training path done; C API exports (forward_sft/backward/update_lora_weights); smoke test passes.

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
- [ ] Minimal pybind11 wrapper in C++ that loads Zig `.so` (optional; ctypes wrapper is functional)

### Advanced Features
- [x] **SFT/LoRA backward pass** — forward_sft (Dev A) + backward (Dev B) complete with LoRA kernels; kt_moe_forward_sft/kt_moe_backward/kt_moe_update_lora_weights C API exports. 43/43 tests pass.
- [ ] Speculative decoding (MTP head)
- [ ] ARM/KML backend (NEON/SVE)
  - [x] **aarch64 variant detection** — `selectBestVariant` now picks `sve2` > `sve` > `neon` for `cpu.vendor == .arm` or `cpu.arch == "aarch64"` (was falling through to the x86 default and returning `"avx2"`). 7 tests in `tests/kernels/aarch64_detect_test.zig` (sve2/sve/neon precedence, vendor-mismatch safety net, x86 selection unchanged). Bundle of an aarch64 GEMM kernel and a cross-compiled variant `.so` still pending.
- [x] **GGML Q8_0 kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q8_0.zig`: byte-exact `BlockQ8_0` (34 bytes: f16 `d` + 32×i8 `qs`, `@sizeOf` compile-checked) vs ggml-common.h; quantize/dequantize matching ggml-quants.c:276/553 (`d = amax/127`, `qs = round(x·1/d)`, `y = qs·d`); f16↔f32 via Zig's **native f16** `@floatCast`/`@bitCast` (first attempt hand-rolled the bit math — buggy subnormal path, replaced by native casts, lesson for LESSONS); scalar GEMM (BF16 activations × Q8_0 weights → F32, on-the-fly dequant). 5 exact-value tests in test_kernels.zig (f16 round-trip, 34-byte layout, quantize/dequant tolerance, 2 GEMM constants). Wired into root.zig with comptime fn-refs (emission verified via nm). Remaining formats: Q4_K → Q6_K → Q5_K (defer Q8_K — super-block scale/min layout is the fiddliest and least used); C-API quantize/dequantize exports are Phase 2.
- [x] **GGML Q4_K kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q4_k.zig`: byte-exact `BlockQ4_K` (144 bytes: f16 `d`+`dmin`, 12-byte 6-bit packed `scales`, 128-byte 4-bit `qs`, `@sizeOf` compile-checked) vs ggml-common.h; `nearestInt` (magic-number trick, ggml-quants.c:621), `getScaleMinK4` (6-bit scale/min unpack, :880), full `makeQkx2Quants` error-minimizing search (:799, 20-step), `quantizeRowQ4_K` (:1457), `dequantizeRowQ4_K` (:1529); scalar GEMM (BF16 × Q4_K → F32). 5 tests in test_kernels.zig (nearest_int known values, 144-byte layout, scale/min unpack scheme, round-trip accuracy <15% rel on smooth data, GEMM constant 256.0). Wired into root.zig (emission verified via nm).
- [x] **GGML Q6_K kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q6_k.zig`: byte-exact `BlockQ6_K` (210 bytes: `ql[128]` low nibbles + `qh[64]` 2-bit highs + 16×i8 `scales` + f16 `d`, compile-checked); `nearestInt`, `makeQxQuants` (rmse_type=1 weighted search + ±9 refine loop, ggml-quants.c:628), `quantizeRowQ6_K` (:1869 incl. all-zero fast path), `dequantizeRowQ6_K` (:1939 — 6-bit quants = ql nibble | qh 2-bit<<4, minus 32); scalar GEMM. Zig gotcha found: C packs L as `uint8_t` — Zig `i8 >> 4` is arithmetic and `(3<<6)` overflows i8, must cast to u8 before the qh packing. 4 tests in test_kernels.zig (210-byte layout offsets, round-trip rel err < 10% on smooth data, all-zero block → d=0 → all-zero output, GEMM vs dequantized reference). Wired into root.zig; emission verified via nm.
- [x] **GGML Q5_K kernel — Phase 1 DONE (2026-08-31)** — `src/kernels/amx/gemm_224_q5_k.zig`: byte-exact `BlockQ5_K` (176 bytes: f16 `d`+`dmin`, 12-byte 6-bit packed `scales` [same scheme as Q4_K], `qh[32]` high-bit plane, `qs[128]` low nibbles, compile-checked); quantize reuses `makeQkx2Quants` with q5_K's parameters (nmax=31, rmin=-0.5, nstep=15); dequant: `y = d·sc·((qs nibble) + (qh bit?16:0)) − dmin·m`. Two subtle reference behaviors caught by tests: (1) the `qh` pointer NEVER advances in the C code — all 4 chunks of 64 weights reuse the same 32 qh bytes, separated by shifting bit masks (m1=1,m2=2, <<2 per chunk); (2) the output pointer `y` advances every chunk (×64), unlike the block-strided intuition. 5 tests in test_kernels.zig (176-byte layout offsets, round-trip rel err < 12%, all-zero input → exact zero, qh high-bit plane exercised by wide-range data, GEMM vs dequantized reference <1% quant error). Wired into root.zig; emission verified via nm. Remaining: Q8_K (deferred — fiddliest, least used); C-API wrappers Phase 2.
- [x] **GGML C-API wrappers — Phase 2 DONE (2026-08-31)** — 12 new exports in `src/main.zig` + 12 declarations in `include/kt_kernel.h`: `kt_quantize/dequantize_q{8_0,4_k,5_k,6_k}` (one-row F32↔block conversions) and `kt_matmul_q{8_0,4_k,5_k,6_k}` (BF16 activations × quantized weights → F32, ldb in BLOCKS). Since the block layouts are byte-exact vs GGUF, Python can pass weights loaded from .gguf files straight into the matmuls. `verify_abi.py` now gates **65 symbols × 7 variants, full PASS**. 7 C-API lifecycle tests in `tests/kernels/ggml_capi_tests.zig` (fifth suite, rooted at main.zig): per-format round trips through the real exports + constant-weight matmuls. main.zig GGML/amx aliases marked `pub` for test access (no ABI effect). **GGML workstream COMPLETE for the 4 workhorse formats** (Q8_0, Q4_K, Q5_K, Q6_K); Q8_K remains deferred.

### Documentation
- [ ] API documentation (zig doc)
- [x] **Porting guide from C++** — `docs/porting-guide.md` (181 lines, committed in `e2c900d`): C++ reference file map → Zig counterpart map, lazy-analysis wiring pattern, comptime fn-ref pattern for `.so` symbol emission, AMX arch-gate, the bf16 f16 cast (Zig 0.16 native f16 vs hand-rolled bit math), and the `@Vector` indexing gotchas. Linked from AGENTS.md.

---

## 📊 Progress Tracking
| Component | Status | % |
|-----------|--------|---|
| Runtime (pool/queue/memory/cpu) | Work-stealing pool + NUMA topology + kernel wiring done (MoE + SFT paths parallel) | 100% |
| AMX Intrinsics | Real inline asm (tile_loadconfig/loadd/stored/zero/dpbf16ps/dpbssd) | 90% |
| Buffer Packing | Working | 80% |
| BF16 GEMM | Working | 70% |
| INT8 GEMM | Working | 70% |
| INT4/FP8/MXFP4/8 GEMM | INT4, FP8, MXFP4, MXFP8 all done (AMX + scalar fallback) | 100% |
| MoE Orchestration | Forward + work-stealing parallel path complete (per-expert parallelism, sequential reduction); Gate C API wired; weight_ld OOB fixed | 100% |
| SFT/LoRA Training | Forward + backward complete + work-stealing parallel forward (d1e0adb: pool-vs-sequential equivalence + backward round-trip); 4 C API exports (new_sft/forward_sft/backward/update_lora) | 100% |
| C API | 53/53 header symbols exported by all 6 variants (verify_abi.py full PASS); MLA + MoE + Gate + Linear + MLP + SFT backward + FP8 transport + load_weights complete | 100% |
| Build System | Multi-variant (6 variants, distinct .so names) | 85% |
| Tests | 53 kernels + 11 MLA + 2 FP8 + 1 MLA C API + 7 aarch64 + 7 GGML C API = 81 total pass, 0 leaks, zig build test exits 0 (simple-mode runner) | 100% |
| Python Integration | ctypes wrapper + pyproject.toml + CI workflow; wheel build verified | 60% |

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


### Current Status (2026-08-30)
- **Dev A (this session)**: Linear/MLP C API wired; kt_*_config_t structs
  synced with C header; MoE gemmExpert weight_ld OOB fixed (6 sites);
  multi-variant build (`zig build all-variants` → 6 .so); minimal Python
  ctypes wrapper; kt_get_cpu_variant lazy-init fixed; SFT/LoRA forward half
  (LoRA kernels, TpMoeSft, forward_sft, C API); runtime hardening (work-stealing
  worker pool, NUMA topology); allocator safety fix (TpMoe stores allocator).
- **Dev B (concurrent)**: MXFP4/MXFP8 kernels complete; vectorized applySwiGLU
  complete (bit-exact vs scalar); SFT/LoRA backward half complete (backward
  method, kt_moe_backward export).
- **Last fully-green**: 32/32 kernels + 11/11 MLA = 43/43 total pass, 0 leaks,
  `zig build test` exits 0.
