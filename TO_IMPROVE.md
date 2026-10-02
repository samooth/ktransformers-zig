# TO_IMPROVE — ktransformers-zig

> Auditoría externa de mejoras (2026-09-09). Autor: lane-c (dev de
> zig-ai, `/home/t0m4s/repos/2026/zig-ai`), realizada durante la
> evaluación KT-1 (docs/PLAN_KT_ZIG.md en zig-ai: veredicto NO GO como
> dependencia — esta lista es el feedback constructivo que quedó fuera
> de aquella decisión).
>
> Contexto de la auditoría: build verificado con Zig 0.16.0 **stable**
> (`~/.zvm/0.16.0/zig`, no solo el dev-2535 del LESSONS), 6 variantes
> x86 + neon, `verify_abi.py` 112/112 PASS, `zig build bench`
> ReleaseFast corrido (gemmExpert A1: 1.31× m=1 / 4.4× m=32; el
> down-projection m=1 sale **0.78× — más lento que su propio scalar**).
> Hardware host: AMD Ryzen 7 5800H (Zen3, AVX2 — sin AMX/AVX-512).
>
> Prioridades: P0 = suelo/higiene (riesgo latente), P1 = rendimiento
> con ROI demostrado, P2 = alcance/confianza. Cada item cita evidencia
> (file:line o dato medido). Nada de esto bloquea el uso actual del .so
> como drop-in de pybind11.

---

## P0 — Suelo / higiene (riesgo latente)

### P0.1 — Purgar dead-code y snapshots del árbol
- `src/kernels/amx/gemm_224_{bf16,fp8,int4,int8}_avx512.zig` son
  código muerto documentado en AGENTS.md:85-87: importan
  `arch/amx_intrinsics.zig` que **no existe** — no compilarían si el
  análisis perezoso de Zig los referenciara. Cualquier `git add -A`
  futuro o un import descuidado los arrastra al build.
- `main.zig.backup*` son snapshots obsoletos en el árbol.
- **Acción**: `git rm` de los 4 _avx512 muertos + backups, o moverlos
  a `attic/` con un README de por qué murieron (el patrón
  `gemm_224_*_avx512` vs los vivos `gemm_224_*` sin sufijo es fácil de
  confundir).
- **Riesgo si no**: colisión de doble-port (mismo bug que su TODO
  protocolo ya documentó: dos agentes portaron LLAMA_MOE_TP en
  paralelo — confundir el vivo con el muerto es la misma clase).

### P0.2 — `src/numa/numa_worker.zig`: API rot "fix on first use"
- TODO.md:124 lo deja explícito: "Remaining rot: numa_worker.zig
  (unreferenced, unowned — fix on first use)". Rot 0.16-dev (std.fs.cwd,
  page_size, alignedAlloc) y sin dueño.
- **Acción**: o se repara en un commit de higiene (30 min con el
  patrón de bd7e712 que ya revivió numa_topology) o se borra. Hoy es
  una trampa: el próximo import fallará con errores confusos.
- **Nota**: el resto de `src/numa/` (topology, memory, bindMemory) ya
  está revivido y testeado — solo queda este fichero.

### P0.3 — `posix_madvise` TODOs sin implementar (runtime/memory.zig:174-186)
- Comentarios `TODO: Use posix_madvise with MADV_SEQUENTIAL/RANDOM/
  WILLNEED` presentes, cero llamadas. Con pesos de decenas de GB
  mapeados, el hint de acceso es la diferencia entre page-fault storms
  y streaming sostenido en el load path.
- **Acción**: implementar los tres hints en NumaAllocator
  (SECUENTIAL para loadWeights, RANDOM para random-access GEMM,
  WILLNEED para warmUp). Gate: medir cold-load de un GGUF 4-8GB antes
  y después.

---

## P1 — Rendimiento (ROI ya demostrado en su propio código)

### P1.1 — Extender `selectTileParams` runtime a INT8/MXFP kernels
- Solo `GemmKernel224BF` tiene el override runtime de K_BLOCK (`pub
  var` + `tuneTileParamsForHost`). INT8/MXFP siguen con la constante
  comptime 3584. El TODO propio (línea 123) lo marca "optional
  follow-up if a host ever shows pressure" — pero la presión YA está
  demostrada por analogía: la medición BF16 dio **−13.2%/−20.3%**
  prefill con K=448 (working set 9MB→2.3MB por block-step).
- **Acción**: replicar el patrón `pub var` en los 4 kernels INT8 y
  2 MXFP + wire en el mismo hook tune. Es mecánico, el diseño está
  hecho, y el win de L2 ya está validado en la familia.

### P1.2 — down-projection m=1 a 0.78×: cuello memory-bound real
- Su propio bench (gemm_bench, B2): `decode down [m=1, n=7168, k=2048]
  18.47 vs 14.40 ms = 0.78×` — el gemmExpert A1 es MÁS LENTO que su
  scalar en EL shape de decode MoE. m=1 es exactamente el caso de uso
  del .so (llama.cpp-style MoE decode CPU-offload).
- **Causas probables**: (a) sin GEMV dedicado — el A1 vectoriza el
  K-loop del GEMM pero en m=1 todo el coste es el STREAM del peso B
  (7168×2048) y el tile/pack overhead no se amortiza; (b) el
  accumulator/scalar-tail domina.
- **Acción**: camino GEMV M=1 dedicado (dot por fila con dequant
  inline y streaming lineal del peso — ver zig-ai `src/moe/cpu_gemv.zig`
  dotScalar familia, 24 formatos, o ggml `vec_dot` como referencia
  canónica). Gate: ≥1.0× vs su scalar como mínimo, objetivo 1.3×
  (nuestro gemv_anda a BW de memoria en m=1).

### P1.3 — Kmap init: revisar thread-safety del cache process-global
- `iq2xs_init/iq3xs_init` son histogram+cached process-wide (22× vs
  qsort, d0b0211 — excelente técnica). Pero su worker pool es
  work-stealing multi-thread: si dos workers tocan el primer GEMM de
  iq2/iq3 en paralelo, la inicialización lazy puede correr dos veces
  o leer a medias.
- **Acción**: audit de la garantía (¿once? ¿mutex? ¿atomic flag?) +
  test de carrera (dos threads × primer-forward simultáneo, 100
  iteraciones). Si falta, un `std.once`-pattern en el init lo cierra.

### P1.4 — Faltan formatos GGML nuevos (enum desactualizado)
- "15/15 complete" es contra un enum que no tiene **tq1_0/tq2_0**
  (ternary 2-bit, landed en llama.cpp 2025) ni los tipos GGUF
  recientes (>39). zig-ai ya los mapea (nuestro TODO 6.3 tq1_0 con
  append GPU + cuarentena E2E documentada).
- **Acción**: sync del KT_TYPE enum contra `ggml-type` HEAD +
  añadir tq1_0/tq2_0 (quantize+dequantize+GEMM+test, mismo molde que
  iq1_s). Mantener el comptime `@sizeOf` audit de KT_TYPE que ya
  tienen (eec0c22 fue la lección).

---

## P2 — Alcance / confianza

### P2.1 — Cero validación E2E con modelo REAL
- 226 tests, todos unit/synth-shapes; los benches usan shapes DeepSeek
  sintéticas. Ningún smoke E2E contra un GGUF real de HuggingFace con
  oráculo (llama.cpp greedy parity) — es la clase de validación que
  en zig-ai encontró 6 bugs numéricos encadenados que TODOS los tests
  unit pasaban por alto (nuestro 7.2, @3470e7d).
- **Acción**: 1 test E2E (Qwen3-MoE-0.5B o gemma-moe small): cargar
  con su GGUF parser + forward + argmax greedy −n 8, comparar tokens
  vs llama-cli con el mismo GGUF. Solo eso eleva la confianza de las
  226 pruebas restantes.

### P2.2 — Load path sin io_uring
- La carga de pesos va por read serial/paged. zig-ai midió **7.4×
  en cold NVMe** con io_uring QD32 chunk-4M en el mismo tipo de carga
  (benches 4.13; trampas documentadas ahí: register_buffers
  RLIMIT_MEMLOCK aparte, O_DIRECT alignment, fadvise para medir
  cold). Combinado con P0.3 (madvise) es el load path completo.
- **Acción**: `tools/bench_iouring.zig` primero (medir antes de
  integrar), luego wiring en loadWeights si el win replica.

### P2.3 — README de arranque de 10 minutos
- El entry-point real (build → variantes → pybind drop-in → smoke
  mínimo con fixture) está enterrado en AGENTS.md. `docs/api.md`
  cubre los 112 símbolos pero no hay "primer contacto" para un
  dev nuevo. `README.md` es de packaging.
- **Acción**: quickstart en README (o docs/QUICKSTART.md): prereqs,
  `zig build -Dvariant=...`, elegir variante por CPU,
  `LD_PRELOAD`/pybind smoke, dónde mirar cuando falla el ABI gate.
  1 página.

### P2.4 — CLAIM protocol sin TTL
- Su protocolo multi-agente dice "abandoned claims can be reclaimed"
  pero no define CUÁNDO. En zig-ai un claim huérfano bloqueó una
  tarea 2 días (asignación sin ACK — documentado como antipatrón).
- **Acción**: TTL explícito en TODO.md ("claim sin commit ni mensaje
  en 48h = reclivable") — 1 línea en el header del protocolo, cero
  código.

---

## No-mejoras (explícitamente fuera)

- **AMX**: NO es una mejora aquí — el kernel AMX está completo y con
  guard correcto (ff57125); el problema es que no hay hardware AMX
  ni en dev-hosts comunes (Ryzen/Intel consumer). No gastar más en
  AMX hasta tener un target Sapphire Rapids real.
- **MLA**: completo y testeado; el gap es de CONSUMIDOR (zig-ai no
  soporta arch DeepSeek hoy), no de implementación. No tocar.
- **Re-architectura del C ABI**: los 112 símbolos con triple gate
  funcionan; el costo de re-encapsular no compra nada.

## Referencias cruzadas zig-ai (por qué lo sabemos)

- GEMV M=1 quant 24-formatos: `zig-ai/src/moe/cpu_gemv.zig` (dotScalar
  familia, mapeo gemvFormatOf) + tests/test_cpu_gemv.zig 24/24.
- io_uring load: zig-ai TODO 4.13 (7.4× cold, trampas register_buffers/
  O_DIRECT/fadvise) — herramienta `tools/bench_copyonce.zig` reutilizable.
- E2E-oracle pattern: llama.cpp greedy parity (zig-ai 7.2: 6 bugs
  numéricos que los unit tests pasaban por alto).
- Dead-code trampa: el mismo patrón de doble-port colisionado está
  documentado en SU protocolo de claims (llamafile_moe colisión) —
  P0.1 existe para prevenir la recurrencia.
