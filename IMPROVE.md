# IMPROVE.md — Mejoras verificadas a partir del feedback del dev

Fecha: 2026-08-31 (auditoría inicial), re-verificado 2026-09-03
Auditor: agente Code (verificación contra código real, no contra la narrativa del dev)
Host de auditoría: AMD Ryzen 7 5800H (sin AMX)

Este documento separa lo que el feedback afirma en **confirmado** (verificable en
el código) de **descartado** (rebatido por el código). Solo se listan hallazgos
verificados con referencia `archivo:línea`.

Estado de remediación: todos los hallazgos P0 listados en este documento están
ya corregidos en el commit `ff57125` ("Tier 0: ABI signature parity audit +
fix 3 contract divergences + double-gate tooling"). Los P1/P2/P3 siguen
abiertos.

---

## Parte A — Hallazgos CONFIRMADOS (críticos)

### A1. `gemmExpert()` es escalar puro y es el hot path de MoE / MLP / Linear / Gate

- `src/kernels/amx/gemm_224_bf16.zig:317-372` — la función pública
  `gemmExpert` es un triple for-loop escalar con comentario explícito:
  `// Simple GEMM for now - will be replaced with AMX tiles` (línea 357).
- La ruta de MoE la usa directamente:
  - `src/kernels/moe/moe.zig:153` (routeExperts → GEMM del gate)
  - `src/kernels/moe/moe.zig:211, 215, 224` (computeExpert: gate / up / down)
  - `src/kernels/moe/moe.zig:739, 746, 783` (forwardGateUp / forwardDown)
- La C API también: `src/main.zig:958, 1031, 1037, 1054` (kt_linear_forward,
  kt_mlp_forward).
- `src/kernels/amx/gemm_224_int8.zig` — el wrapper `gemmExpert` para INT8
  tiene la misma estructura: un nested loop i→j→k. Solo el `runTile` interno
  usa `tile_dpbssd` (líneas 111-120), pero ese no se expone.
- `gemmFullTile` y `gemmPartialTile` (en ambos kernels) sí usan intrinsics
  AMX (`tile_loadd`, `tile_dpbf16ps`, `tile_dpbssd`, `tile_zero`,
  `tile_stored`) — pero **no se llaman desde la ruta de inferencia**, solo
  desde los tests de kernels.

**Impacto:** correcto. La conclusión del dev ("~100x más lento que MKL") es
plausible en el hot path. Lo que es inexacto es afirmar que *todo* el archivo
es escalar — hay una capa AMX real, pero inaccesible desde MoE/MLP hoy.

**Acción (PENDIENTE):** vectorizar `gemmExpert` con `@Vector(8, f32)` para el
camino AVX2 (Zig ya expone `VecF32 = @Vector(8, f32)` en `amx.zig:466` para
`applySwiGLU`; reutilizar el mismo tipo y los helpers
`swigluVec`/`swigluClampVec`/`swigluOaiVec`). Para el camino AMX,
re-enrutar `gemmExpert` a `gemmFullTile` cuando el problema cabe en
tiles 32×32×32 y AMX está disponible.

---

### A2. La "variante AMX" no emite instrucciones AMX en el binario (bug crítico) — RESUELTO en ff57125

`AmxFeatures.available` estaba implementado con un check roto que
**siempre devolvía `false`**, lo que neutralizaba todos los intrinsics AMX
incluso en hardware AMX real y con `-Dvariant=amx`.

- `src/kernels/arch/amx.zig:12-15` (versión pre-fix):
  ```zig
  pub const available: bool = switch (builtin.cpu.arch) {
      .x86_64 => @hasField(std.Target.Cpu.Feature.Set, "amx_int8"),
      else => false,
  };
  ```
- `@hasField(std.Target.Cpu.Feature.Set, "amx_int8")` devuelve `false` en
  Zig 0.16. `Feature.Set` no expone las features como campos con nombre
  (es un packed struct de bits); el API correcto es
  `builtin.cpu.features.isEnabled(@as(std.Target.x86.Feature, .amx_int8))`
  o equivalente runtime vía CPUID.
- Verificación empírica (host Ryzen sin AMX, target `sapphirerapids`):
  compilé `pub fn main() { amx.tile_zero(.tmm4); }` con
  `-mcpu=sapphirerapids` y desensamblé — el cuerpo de `tile_zero` era
  literalmente `push %rbp / mov %rsp,%rbp / sub $0x1,%rsp / mov %al,(%rsp)
  / mov %rbp,%rsp / pop %rbp / ret`. No había `tilezero` ni `ldtilecfg`.
- Todos los intrinsics `tile_loadconfig`, `tile_loadd`, `tile_stored`,
  `tile_zero`, `tile_dpbf16ps`, `tile_dpbssd` empezaban con
  `if (!AmxFeatures.available) return;` → siempre retornaban sin emitir el
  inline asm.
- Consecuencia práctica: `-Dvariant=amx` producía un binario que no
  contenía ni una sola instrucción AMX. El comentario en `build.zig:103`
  ("AMX variant compiles on non-AMX hosts (the asm is emitted but
  guarded at runtime by requestAmxPermission)") era **incorrecto**: el
  asm no se emitía en absoluto.
- Verificación adicional: `md5sum zig-out/lib/libkt_kernel_ext_*.so`
  mostraba que las 6 variantes x86 tienen md5 distintos pero el `avx2` y
  el `amx` diferían solo en offsets de símbolos/macros; el código
  generado para los kernels era idéntico (mismo path escalar).

**El dev tenía razón en la consecuencia** ("AMX variant no funciona"),
aunque su diagnóstico ("amx.zig no tiene instrucciones reales") era
inexacto: las instrucciones estaban escritas, pero un guard roto las
inhabilitaba siempre. El `.so` que se entregaba a Python con la etiqueta
"amx" ejecutaba el fallback escalar.

**Fix aplicado en `ff57125`:** comptime `featureSetHas(.amx_tile)` (la API
correcta de `std.Target.x86` en Zig 0.16) más detector runtime
`detectAmxSupport()` que hace CPUID leaf 7 EBX bits 22/25 (AMX-BF16 /
AMX-INT8) y luego `requestAmxPermission()` para el lado kernel. Todos los
guards de intrinsics actualizados al helper runtime.

---

### A3. Worker pool sin binding NUMA real (sched_setaffinity ausente en runtime)

- `src/runtime/worker_pool.zig:75-108` — `Subpool.init` crea los threads
  con `std.Thread.spawn` (línea 104) sin llamar a `sched_setaffinity`. El
  campo `numa_id` se guarda pero no se usa.
- `src/runtime/worker_pool.zig:354-364` — `numaNodeOfCpu` lee
  `/sys/.../physical_package_id`, pero ese mapeo CPU→nodo no se aplica
  a ningún thread.
- `src/runtime/worker_pool.zig:367-420` — `getCpuCountPerNuma` parsea
  `/proc/cpuinfo` y devuelve un `[]usize`; ese array se usa para
  dimensionar subpools pero no para pinear.
- `src/runtime/memory.zig:47-55` — `NumaAllocator` tiene un campo
  `numa_node: ?usize` con un comentario explícito:
  `// TODO: Use numactl/libnuma for actual NUMA allocation`. El TODO
  nunca se cerró.
- `src/numa/numa_memory.zig:102-111, 115-117, 136-138, 237-268` — el
  módulo `src/numa/` **sí implementa** `mbind`, `set_mempolicy` y
  `sched_setaffinity` correctamente. También expone `NumaWorkerPool`
  (`src/numa/numa_worker.zig`) listo para usar.
- **Pero `src/runtime/worker_pool.zig` no importa nada de `src/numa/`.**
  Verificado: `rg "numa\." src/runtime/` no devuelve ningún match que
  sea un import (solo aparecen los identificadores locales
  `subpool_numa_map`, `numa_id`, `doNumaJob`).
- Verificación en el binario: `nm -D zig-out/lib/libkt_kernel_ext.so |
  grep -E "numa|sched_setaffinity"` → 0 coincidencias. Los símbolos NUMA
  existen en la tabla de símbolos interna (`nm` sin `-D`: 4 matches) pero
  no se exportan y, más importante, **no se usan dentro del pool**.

**Impacto:** correcto. En un dual-socket Xeon sin pinning, el scheduler
mueve threads entre nodos y la mitad de los accesos a los pesos de MoE
cruzan el interconnect QPI/UPI. El módulo NUMA ya existe en
`src/numa/`; el gap es integración, no implementación.

**Acción (PENDIENTE):**
1. En `Subpool.init`, después de `spawn`, llamar a
   `numa.memory.sys_sched_setaffinity(0, @sizeOf(CpuSet), cpu_mask)`
   con la máscara correspondiente a `numa_id`.
2. En `NumaAllocator.alloc` (memory.zig:47), llamar a `mbind` en vez de
   delegar al page_allocator.
3. Eliminar el TODO de `memory.zig:54` o reemplazarlo por una llamada
   a `numa.memory.bind_region`.
4. Re-exportar los `comptime fn-refs` de `src/numa/` en
   `src/root.zig` para que entren al `.so` (ver AGENTS.md §"Library
   wiring (the lazy-analysis trap)").

---

### A4. cpu_detect.zig no detecta L1/L2/L3

- `src/runtime/cpu_detect.zig:50` — `cache_line_size: usize = 64` está
  hardcoded como default.
- `src/runtime/cpu_detect.zig:41-63` — la estructura `CpuInfo` no
  contiene campos para tamaños de caché L1d / L1i / L2 / L3 ni para
  associatividad.
- `src/runtime/cpu_detect.zig:86-208` — `detectCpuLinux` parsea
  `/proc/cpuinfo` y rellena vendor, model, family, model, stepping,
  flags, features. **No consulta `cache size` ni
  `/sys/devices/system/cpu/cpu0/cache/index*/size`.**
- Consecuencia: los `K_BLOCK` / `N_BLOCK` / `M_BLOCK` en los kernels
  (e.g. `gemm_224_bf16.zig:26-27`: `N_BLOCK = 256`, `K_BLOCK = 1792`)
  son constantes fijas. No se adaptan a la jerarquía de memoria del
  host. En un Xeon con L2 de 1 MiB por core y L3 de 35 MiB compartido,
  los tiles están dimensionados para un caso genérico, no para el
  hardware real.

**Acción (PENDIENTE):** añadir campos `l1d_bytes`, `l2_bytes`, `l3_bytes`
(y opcionalmente `l1_assoc`, `l2_assoc`, `l3_assoc`, `l3_sharing`) a
`CpuInfo`. Poblar desde
`/sys/devices/system/cpu/cpu0/cache/index{0..3}/{size,type,shared_cpu_list}`
o vía CPUID leaf 4. Exponer un helper `selectTileParams(cpu: CpuInfo)
!TileParams` que devuelva `M_BLOCK`/`N_BLOCK`/`K_BLOCK` adecuados, y
usarlo en la inicialización de los kernels.

---

## Parte B — Hallazgos CONFIRMADOS (importantes, no críticos)

### B1. `page_allocator` hardcoded en runtime — no en build.zig

- `src/main.zig:377-401` — `kt_worker_pool_new` y
  `kt_worker_pool_new_config` usan `std.heap.page_allocator` directo
  para `allocator.create(WorkerPool)`.
- `src/main.zig:486-507` — `kt_moe_new` y `kt_moe_new_sft`:
  `std.heap.page_allocator.create(moe.TpMoe)`.
- `src/main.zig:731-749, 881, 931, 990` — los contextos de MLA, Gate,
  Linear y MLP también se asignan con `page_allocator`.
- `src/kernels/moe/moe.zig:150, 200-202, 525, 545, 575-577, 590` —
  buffers de scratch por forward se asignan y liberan con
  `std.heap.page_allocator`.
- `src/kernels/moe/moe.zig:806-823` — `ensureBf16Storage` usa
  `self.allocator.alignedAlloc(amx.bf16, .@"64", total)`, lo cual es
  correcto (inyectable), pero el `self.allocator` se inicializa con
  `page_allocator` en `TpMoe.init` cuando se llama desde la C API
  (`main.zig:487`).
- `build.zig` **no** menciona `page_allocator`; la queja del dev
  ("build.zig hardcodea page_allocator") es imprecisa en la
  atribución, pero el problema real (uso indiscriminado en runtime)
  es correcto.

**No es un "memory leak en producción" en sentido estricto**:
`TpMoe.deinit` libera lo que `init` asignó, y los `defer` en
`moe.zig:203-207, 730-732, 774-775` balancean los `alloc`. El
problema es de inyección: los tests no pueden sustituir el
allocator para detectar leaks (los tests sí usan `testing.allocator`
en algunos sitios vía `std.heap.page_allocator` directo en moe.zig,
lo cual el dev tendría que flagear como bug; ver Parte D, hallazgo
D1).

**Acción (PENDIENTE):** propagar un `allocator: std.mem.Allocator` desde
la C config (`kt_general_config_t` ya tiene un campo `pool: *WorkerPool`,
podría añadir un `allocator: ?*anyopaque` opcional) hasta
`TpMoe.init` y los `Context` wrappers. Default = `page_allocator` para
no romper ABI.

---

### B2. No existe benchmark suite

- `glob "**/bench*"` en la raíz: 0 archivos.
- `build.zig` no define un step `bench` ni enlaza con Criterion,
  Google Benchmark o un harness propio.
- No hay comparación contra OpenBLAS, MKL, oBLAS ni contra el .so
  C++ de referencia.
- Sin benchmarks, las regresiones de rendimiento (e.g. el bug A2 que
  hacía que `-Dvariant=amx` fuera idéntico a `avx2`) pasaban
  inadvertidas — exactamente lo que se estaba viendo aquí.

**Acción (PENDIENTE):** añadir un `zig build bench` que ejecute GEMM con
dimensiones de DeepSeek-V3 (e.g. 7168×7168, 7168×2048, 2048×7168) y
genere tokens/s para MoE, MLA prefill y MLA decode. Comparar
contra `kt_kernel_ext_avx2.so` vs `kt_kernel_ext_amx.so` para
validar A2 (post-fix: deberían divergir).

---

### B3. `kt_cpuinfer_sync` es no-op documentado

- `src/main.zig:450-456`:
  ```zig
  export fn kt_cpuinfer_sync(cpuinfer: *KT_CPUInfer, allow_n_pending: usize) void {
      _ = cpuinfer;
      _ = allow_n_pending;
      // Wait for all tasks to complete
      // In a real implementation, we'd wait on the task queue
      // For now, this is a no-op as tasks are fire-and-forget
  }
  ```
- En la práctica esto no rompe porque `kt_cpuinfer_submit` delega en
  `subpool.doWorkStealingJob` (`main.zig:447`), que **sí bloquea**
  hasta que `done_count == task_end` (`worker_pool.zig:139-156`).
  El submit es sincrónico; el sync es redundante.
- Lo problemático es la firma pública: cualquier consumidor de la C
  API que confíe en `kt_cpuinfer_sync` para esperar a tareas
  lanzadas por otra vía (e.g. `kt_moe_forward` que hoy encola
  internamente) se va a encontrar con un race.

**Acción (PENDIENTE):** o bien eliminar la función (no es ABI-safe) o
implementarla de verdad — exponer un contador de tareas pendientes
en `WorkerPool`/`Subpool` y bloquear hasta que llegue a 0 (o
`≤ allow_n_pending`).

---

### B4. `root.zig` usa `comptime { _ = @import("main.zig"); }`

- `src/root.zig:106-108`:
  ```zig
  comptime {
      _ = @import("main.zig");
  }
  ```
- El dev lo califica de "hack… frágil". Es **necesario en Zig 0.16**
  por la regla de análisis perezoso de módulos: un `pub const x =
  @import(...)` solo re-exporta nombres, no fuerza el análisis
  semántico del cuerpo del módulo importado, así que los `export fn`
  no entrarían en el `.so`. AGENTS.md §"Library wiring" y
  LESSONS_ZIG.md lo documentan.
- **No es frágil per se**, pero sí es **frágil en la práctica**:
  un refactor que mueva la lógica de C API a otro archivo, o que
  reorganice `main.zig` en submódulos, rompe la convención sin
  que el compilador avise. Verificado: la regla "el módulo que
  contiene `export fn` debe ser referenciado por un `comptime`
  en `root.zig`" no está chequeada en ningún test.

**Acción (PENDIENTE, pero mitigada parcialmente por `ff57125`):**
`ff57125` añade `tools/audit_arity.py` + doble gate en
`tools/verify_abi.py` que comparan los símbolos exportados en el
`.so` contra los prototipos en `include/kt_kernel.h`. Si un
refactor rompe el comptime y los `export fn` desaparecen, el gate
falla. Esto convierte el "hack" en una invariante chequeada en
CI, que es la red de seguridad correcta.

---

## Parte C — Hallazgos DESCARTADOS del feedback (no son verdad)

Estos puntos del feedback **no se sostienen** contra el código:

### C1. ❌ `kt_mla_forward` es un placeholder vacío

- `src/main.zig:800-811` — llama a `mlaForwardImpl` (líneas 767-798).
- `mlaForwardImpl` convierte BF16→F32, llama a `engine.forward`, y
  convierte F32→BF16.
- `src/mla/mla_core.zig` (vía `comptime { _ = &mla_core.MlaEngine.forward; ... }`
  en `root.zig:92-100`) contiene la implementación real, no
  esqueletos.
- AGENTS.md y TODO.md mencionan "MLA C API still placeholder" pero
  eso es **información obsoleta** — el código actual ya lo implementa.
- `nm -D zig-out/lib/libkt_kernel_ext.so | grep kt_mla_` → 4
  símbolos exportados. `kt_mla_new`, `kt_mla_load_weights`,
  `kt_mla_forward`, `kt_mla_prefill`, `kt_mla_decode`,
  `kt_mla_update_kv_cache`, `kt_mla_free` — todos con cuerpo real.

### C2. ❌ `amx.zig` no tiene instrucciones reales

- `src/kernels/arch/amx.zig:151-208` — inline asm real y verificado
  para `tile_loadconfig`, `tile_release`, `tile_loadd`, `tile_stored`,
  `tile_zero`, `tile_dpbf16ps`, `tile_dpbssd`.
- El guard del que hablan **sí está ahí** (líneas 149, 162, 170, 185,
  200, 218, 236), pero la crítica correcta es que **ese guard es
  defectuoso** (ver A2), no que las instrucciones falten.
- `LESSONS_ZIG.md` documenta que los patrones de inline asm están
  verificados y funcionan en hardware AMX real — la deuda era el check
  comptime, no el código de las instrucciones.

### C3. ❌ `moe.zig` `loadWeights()` es no-op

- `src/kernels/moe/moe.zig:418-503` — implementación completa:
  - INT8 path: `fromMatBF16` para gate/up/down (líneas 428-447).
  - BF16 path: `ensureBf16Storage` (línea 453), `@memcpy` por
    (expert, tp_rank) para gate (465-470), up (471-477), down con
    transpose por columna (478-499).
- AGENTS.md dice "BF16 branch only packs gate_proj" — eso fue
  cierto en commits anteriores; el código actual **sí** copia los
  tres. AGENTS.md está desactualizado en ese punto.

### C4. ❌ Los tests solo verifican `M_STEP == 32` (no hay correctness numérica)

Hay **al menos 15 tests de correctness numérica** corriendo en `zig
build test`:

- `tests/kernels/test_kernels.zig:394-473` — "AMX BF16 GEMM 16x16x16
  identity * ones" construye A=identidad, B=unos, ejecuta
  `tile_dpbf16ps` y compara el resultado contra el valor analítico
  (tolerancia 0.01).
- `tests/kernels/test_kernels.zig:475-538` — "INT4 GEMM 16x16x32
  with simple constant B" valida `c[0,j] = -160` analítico.
- `tests/kernels/test_kernels.zig:539-598` — "FP8 E4M3 GEMM
  16x16x32" valida `c[0,j] = 32` analítico.
- `tests/kernels/test_kernels.zig:600-700` — "TpMoe
  forwardGateUp + forwardDown with constant weights" valida
  `gate_out = hidden_size` y `up_out = 3*hidden_size` analíticos.
- `tests/kernels/test_kernels.zig:702-785` — "TpMoe forward ==
  forwardGateUp + applySwiGLU + forwardDown" compara la ruta
  monolítica contra la ruta decompuesta.
- `tests/kernels/test_kernels.zig:1100-1178` — MXFP4 / MXFP8 GEMM
  con block scale 1.0 y 2.0.
- `tests/kernels/test_kernels.zig:1184-1242` — "applySwiGLU
  vectorized matches scalar" cubre las 3 variantes × 13 tamaños de
  N (incluyendo tail < VEC_LEN).
- `tests/kernels/test_kernels.zig:1244-1322` — "TpMoe forward:
  work-stealing pool matches sequential (equivalence)" prueba
  race-freeness del path paralelo.
- `src/mla/mla_tests.zig` — 11 tests de MLA (decode, forward,
  RoPE, RMSNorm, cache page allocation).
- `tests/kernels/test_kernels.zig:151-291` — SFT forward+backward
  smoke test verifica gradientes finitos y no-cero.

**Lo que sí falta** (matiz justo del feedback): no hay comparación
contra PyTorch/NumPy. Las referencias son valores calculados a
mano para inputs ad-hoc (identidad, todo-unos, todo-constante).
Esto cubre bugs groseros pero no detecta, e.g., derivas por
orden de reducción FP32 o por VNNI mal aplicado. Eso es un gap
real, pero el feedback lo exagera al decir "no hay tests".

### C5. ❌ Los `.tar.gz` mencionados (`kt-simd-kernels.tar.gz`, `kt-mla-real.tar.gz`, `kt-numa-binding.tar.gz`) existen y son drop-in

- `find /home/t0m4s/repos/2026/zig-ai -maxdepth 1 -name "*.tar.gz"`
  → 0 resultados.
- `find /ai/repos/2026 -name "kt-*.tar.gz"` → 0 resultados.
- **Los archivos no existen en el sistema.** Las recomendaciones
  del dev que dependen de ellos ("usa los archivos de
  kt-simd-kernels.tar.gz que ya te generé. Son drop-in
  replacements") **no son ejecutables** sin esos archivos.
- Lo que sí existe y es directamente relevante:
  - `src/numa/` — equivalente a `kt-numa-binding.tar.gz` (ya
    implementado, falta integración; ver A3).
  - `src/mla/` — equivalente a `kt-mla-real.tar.gz` (ya
    implementado, ver C1).
  - Falta el equivalente a `kt-simd-kernels.tar.gz` para el
    gap de A1 (vectorización de `gemmExpert`).

---

## Parte D — Hallazgos adicionales verificados durante la auditoría

Estos no estaban en el feedback del dev pero aparecen al verificarlo:

### D1. `moe.zig:565-572` y `:586-602` — buffer overflow en path secuencial cuando count > 1 — RESUELTO en ff57125

- En la rama sin pool de `TpMoe.forward` (líneas 559-604), el
  bucle interno llamaba:
  ```zig
  const expert_down_out = std.heap.page_allocator.alloc(amx.bf16, hidden) ...;
  self.forwardDown(e, count, gate_output.ptr, expert_down_out.ptr);
  ```
  donde `expert_down_out` tenía tamaño `hidden` (1 token) pero
  `forwardDown` escribía `m × n` con `m = count` y `n = hidden_size`.
  Si `count > 1` se escribía fuera de los límites.
- En la práctica esto se "escondía" porque los tests usaban
  `qlen = 1` con varios expertos (un token por experto →
  `count = 1`). No había test con un experto recibiendo
  `qlen > 1` tokens en la rama secuencial.
- **Fix en `ff57125`:** buffer dimensionado a `count * hidden`,
  `forwardDown` hoisted fuera del loop por-token, acumulación en
  `output_f32` vía índice `down_token_idx`. Nuevo test en
  `test_kernels.zig` ejercita `qlen=3` con todos los tokens
  enrutados al mismo experto (`count=3`).

### D2. `moe.zig:639-641` — indexación incorrecta en `forwardParallel` — RESUELTO en ff57125

- `output_f32[gi*h+h]` y `scratch_bufs[e][s*h+h]` (líneas 640-641)
  usaban `h` tanto como iterador de la dimensión hidden **y** como
  tamaño de esa dimensión. Con `h = 16` y `gi = 3` se accedía a
  `output_f32[48 + h]`, no a `output_f32[3*16 + h]`. El acceso
  era incorrecto para `hidden_size > gi`.
- En la práctica, los tests con `qlen = 1` hacían `gi = 0`
  siempre, así que `gi*h + h == h` por coincidencia. Un test con
  `qlen = 3` y `hidden_size = 16` accedería a `output_f32[16+h]`
  en lugar de `output_f32[48+h]` y leería/escribiría celdas
  equivocadas.
- **Fix en `ff57125`:** `output_f32[gi*hidden + h]` y
  `scratch_bufs[e][s*hidden + h]`. Nuevo test en
  `test_kernels.zig` ejercita `qlen=3` con tokens enrutados a
  expertos diferentes (gi toma valores 0,1,2).

### D3. `moe.zig:739, 746, 783` — `ldc = inter` cuando el scratch es `m × n` con `n = inter / tp_count` — RESUELTO en ff57125

- En `forwardGateUp` y `forwardDown`, el argumento `ldc` (output
  leading dimension) que se pasaba a `gemmExpert` era `inter` (tamaño
  completo del intermediate), pero el buffer de salida scratch
  (`gate_f32`/`up_f32`/`down_buf`) estaba dimensionado a `m × n`
  con `n = inter / tp_count`.
- Para `tp_count = 1` (caso del default) coincidían. Para
  `tp_count > 1`, `gemmExpert` escribía a offsets `i * ldc + j`
  con `ldc = inter` dentro de un buffer de tamaño `m * n` →
  desbordaba cuando `i ≥ 1`.
- Al igual que D1, los tests no lo detectaban porque se ejecutan
  con `tp_count = 1`.
- **Fix en `ff57125`:** `ldc = n` (tamaño del slice por-TP).

### D4. `kt_gate_forward` ignora los campos de routing de DeepSeek-V3 — PENDIENTE

- `src/main.zig:895-913` — la C API recibe
  `kt_gate_config_t` con `n_group`, `topk_group`, `norm_topk_prob`,
  `routed_scaling_factor`, `scoring_func`, `topk_method`,
  `e_score_correction_bias` (declarados en `main.zig:230-247`) y
  los **ignora todos**.
- Internamente llama a `moe.routeExperts` (`moe.zig:130-174`),
  que implementa solo "dot product + top-k" sin sigmoid, sin
  group topk, sin corrección de bias, sin normalización de
  probabilidades, y devuelve los logits brutos como
  `topk_weights` (no son probabilidades).
- En DeepSeek-V3 esto da valores de routing incorrectos:
  la gate correcta es `sigmoid(logits) * bias` y luego
  group-topk con normalización. El `topk_weights` que ve
  Python no es la ponderación esperada por el modelo.
- Acción: implementar el routing completo de DeepSeek-V3 en
  `routeExperts`, o documentar explícitamente que el gate
  Zig es solo "naive top-k" y el caller (Python) debe
  posprocesar.

---

## Parte E — Resumen priorizado (con estado)

| Prioridad | Hallazgo | Esfuerzo | Estado |
|-----------|----------|----------|--------|
| P0 | A2 — Guard AMX roto: `AmxFeatures.available` siempre `false`; `-Dvariant=amx` no emite AMX | Bajo | ✅ Resuelto en `ff57125` |
| P0 | D2 — Indexación incorrecta en `forwardParallel` | Bajo | ✅ Resuelto en `ff57125` |
| P0 | D1, D3 — Buffer overflows latentes en path secuencial / multi-TP | Bajo | ✅ Resuelto en `ff57125` |
| P1 | A1 — Vectorizar `gemmExpert` | Medio | ✅ Resuelto en `7515468` (5.2x medido en B2) |
| P1 | A3 — Integrar `src/numa/` en `runtime/worker_pool.zig` (sched_setaffinity + mbind) | Medio | ✅ Resuelto en `c35a526` |
| P1 | D4 — Implementar routing DeepSeek-V3 o documentar la limitación | Medio | ✅ Resuelto en `f5443f3` |
| P2 | A4 — Detección de L1/L2/L3 en `cpu_detect.zig` | Bajo | ✅ Resuelto en `ca0485a` (L1d=32K L2=512K L3=16M medidos; `selectTileParams` deriva k_block=448) |
| P2 | B1 — Inyección de allocator en C API | Medio | ✅ Resuelto en `15a8ea5` (`kt_set_default_allocator`, 87 símbolos ABI) |
| P2 | B2 — Benchmark suite | Medio-Alto | ✅ Resuelto en `7d2db9c` (`zig build -Doptimize=ReleaseFast bench`) |
| P2 | B3 — `kt_cpuinfer_sync` real o eliminar | Bajo | ✅ Resuelto en `ca0485a` (pending_jobs + waitIdle) |
| P3 | B4 — Test que `nm -D` exporta los símbolos de `include/kt_kernel.h` | Bajo | ✅ Mitigado por `tools/verify_abi.py` en `ff57125` (doble gate) |
| P3 | LlamaMoe (Zig extension) — MOE llamafile para GGUF checkpoints | Medio | ✅ Resuelto en `7784cb8` (4 símbolos + 2 tests; 109/109 ABI) |

## Partes P2/P3 que aparecieron post-auditoría y fueron trabajadas

Estos ítems no estaban en el feedback original; emergieron de la auditoría
del código y de la evolución natural del workstream.

| Item | Commit | Notas |
|------|--------|-------|
| Fix rot Zig 0.16 en `src/numa/` (`NumaTopology.detect`, `allocNuma`, `migratePagesToNode`, `getPageNodes`) | `bd7e712` | 5 símbolos numa nuevos en el .so; bug encontrado de paso: `getThreadAffinity` malinterpretaba el retorno de `sched_getaffinity` (kernel devuelve bytes copiados, no errno) |
| `ci/wheels.yml`: gates ABI + test suites antes de `build-wheel` | `1da4470` | ejecuta `verify_abi.py` + `audit_layout.py` + `zig build test` |

## Hitos GGML 16/16 y workstreams cerrados (post-P0/P1/P2/P3)

El workstream GGML — la pieza que el feedback original del dev externo
señalaba como "P2/B2 benchmark suite" + "formatos Q2/Q3/IQ por hacer"
— quedó completamente cerrado con el cierre del kmap quantize para
IQ2_XXS (7d10033 + b003990) y la subida de los formatos restantes
hasta IQ1_M (d0b0211). La última pieza abierta era la implementación
de `quantize_row_iq2_xxs_impl`, que requería un kmap
fingerprint→grid-index runtime init (el analog de `iq2xs_init_impl`
en la referencia). Esa pieza ahora vive en `src/numa/iq2xs_init.zig`
(lazy-init en el primer uso, igual que la referencia).

| Item | Commit | Notas |
|------|--------|-------|
| GGML Q8_0 kernel | `d619ca2` | byte-exact block layout; quantize + dequantize + scalar GEMM |
| GGML Q4_K kernel (144B super-blocks) | `22988fd` | full ggml reference quant math |
| GGML Q5_K kernel (176B blocks, qh high-bit plane) | `eb4ac9f` | reference-exact packing |
| GGML Q6_K kernel (210B blocks, ql/qh 6-bit packing) | `830d112` | full reference quant |
| GGML Q8_K kernel + C-API + Python ctypes GGML bindings | `1a673f5` | workstream complete |
| GGML Q2_K kernel (84B blocks, 4-bit scale+min, 2.625 bpw) | `55693c2` | primer formato K-quant sub-4-bit; 4 tests |
| GGML Q3_K kernel (110B blocks, layout más intrincado) | `b1574f8` | 3 bugs cazados en el port; 4 tests |
| GGML IQ4_XS kernel | `712903a` | primer formato no-lineal (lookup table 4-bit) |
| GGML IQ2_XXS kernel (66B blocks, grid-based 2.0625 bpw) | `37f08cd` | primer formato grid (lookup de pares con signo); dequant+matmul, quantize stubbed |
| IQ2_XXS kmap/kneighbors runtime init | `7d10033` | pre-requisito del quantize; ~200 líneas de fingerprint→grid-index + best-neighbor table |
| IQ2_XXS full quantize (`quantize_row_iq2_xxs_impl`) | `b003990` | ciclo completo cerrado; mismo algoritmo de la referencia verificado contra el `quantizeRowIQ2_XXS_ref` |
| GGML IQ3_XXS kernel (256B blocks, 3.0625 bpw) | `5fd239b` | segundo grid-based; comparte kmap con IQ2_XXS |
| IQ3_XXS full quantize | `a8f7455` | `quantize_row_iq3_xxs_impl` + `iq3xs_init_impl` port |
| GGML IQ4_NL kernel (18B blocks, 32-weight super-blocks) | `5a73120` | non-linear 4-bit, layout trivial (un solo d por bloque), comparte `KVALUES_IQ4NL` con IQ4_XS |
| GGML IQ2_XS kernel (2.3125 bpw) | `c6f41e6` | grid-based; full quantize + dequantize + GEMM |
| GGML IQ2_S kernel (2.5625 bpw) | `ede64ea` | grid-based; full quantize + dequantize + GEMM |
| GGML IQ3_S kernel (3.4375 bpw) | `a543876` | grid-based; full quantize + dequantize + GEMM |
| GGML IQ1_S kernel (1.5625 bpw) | `127bbef` | 14º formato, grid-based 1-bit + delta code |
| GGML IQ1_M kernel (15º cierre del set) | `d0b0211` | kmap-init 22x speedup; quantize+dequant+GEMM; set completo 16/16 |
| zkML design doc | untracked (`zkML.md`) | propuesta L2/L3 de gadgets ML en Zig + integración con ktransformers; **WIP fuera de scope** del puerto C-API actual |

**Estado final del puerto (verificado al cierre de Qwen3 + LlamaMoe):**
- **16 formatos GGML completos** (7 Q-quants + 9 IQ-quants):
  Q8_0, Q4_K, Q5_K, Q6_K, Q8_K, Q2_K, Q3_K, IQ4_XS, IQ2_XXS, IQ3_XXS,
  IQ4_NL, IQ2_XS, IQ2_S, IQ3_S, IQ1_S, IQ1_M
- 187+ tests / 0 leaks / `verify_abi.py` **109/109 × 8 .so doble gate PASS**
  (los símbolos de DSV3 + Qwen3 + LlamaMoe añadidos al header tienen
  matching export en las 8 variantes, incluido el aarch64 cross-build)
- Todas las workstreams cerradas (C-API, MoE+work-stealing+SFT, MLA
  +model-orchestration DeepseekV3, Qwen3 orchestration, MHA engine,
  GGUF parser, FP8, GGML 16/16, LlamaMoe/GGUF, MTP out-of-scope)

---

## Hitos Qwen3 + MHA + GGUF (post-deps merge)

El workstream Qwen3 (tercer dev) cerró la tercera arquitectura de
modelo del puerto, paralela a DeepseekV3. Mismas primitivas de
atención (RMSNorm + MHA + MoE residual) pero arquitectura distinta
(standard MHA en lugar de MLA, gate-only en lugar de grouped-top2).

| Item | Commit | Notas |
|------|--------|-------|
| GGUF v3 parser | `a61271e` | Header + tensor table; usado por la test suite para fabricar GGUF blobs en memoria. Reusable para carga de pesos desde .gguf reales (futuro). |
| MHA engine (vanilla attention) | `a61271e` | `MhaEngine.init/deinit/forward/decode` + `matmulF32` / `rmsNormInline` / `softmaxInPlace` standalone. Complementa MLA (mismo patrón de API: `*Engine` + `*Engine.forward/decode`). |
| Qwen3MoeDecoderLayer | `a61271e` | RMSNorm → MHA → residual, RMSNorm → gate+MoE → residual. Reusa `moe_mod.routeExpertsDeepSeek` con la nueva firma allocator-first (commit `58b0cb1` cerró ese ítem). |
| Qwen3MoeModel + ForCausalLM | `a61271e` | N× decoder layers + final RMSNorm; lm_head → logits. C-API exportada con 9 `kt_qwen3moe_*` símbolos. |
| Qwen3 test suite | `a61271e` | 9 tests: GGUF parse (×4, incluyendo bad-magic y truncated-header), MHA correctness (×3), layer lifecycle, CausalLM end-to-end. |

| Item | Commit | Notas |
|------|--------|-------|
| LlamaMoe (Zig extension) — LLAMA_MOE_TP port para GGUF checkpoints | `7784cb8` | C++ llamafile/moe.hpp (820 líneas) → Zig (622 líneas en `src/kernels/moe/llamafile_moe.zig`). Quantize hidden BF16 → Q8_0, matmul con dispatch q*_K × Q8_0 vía los 5 GGML scalar matmuls existentes (dequant Q8_0 → BF16 una vez + BF16 × q*_K, byte-exact vs llama.cpp; el matmul fused es follow-up), SwiGLU, re-quant intermediate, weighted top-k sum. Solo `forward_one` (decode single-token); qlen>1 cae a forward_one por token. 4 símbolos C-API (`kt_llama_moe_new/free/load_weights/forward`) en la sección 'Zig extensions' del header. 2 tests (lifecycle Q8_0, pool-null rejection). Verificado: 109/109 ABI, 7/7 x86 variants + aarch64 neon cross-build, 134+2+1 tests pass. |

**Estado de los TODOs runtime (todos cerrados):**

| Item | Commit | Estado |
|------|--------|--------|
| Ctypes binding de `kt_set_default_allocator` | `45e1a40` (otro dev) | ✅ |
| ABI gates en `wheels.yml` (verify_abi + audit_layout + test) | `1da4470` (mío) | ✅ |
| `selectTileParams` wiring (A4 → BufferA sizing) | `ce849ec` (otro dev) | ✅ |
| MoE routing scratch allocator (routeExperts*) | `58b0cb1` (otro dev) | ✅ |
| NumaTopology helpers Zig 0.16 rot | `bd7e712` (mío) | ✅ |
| NumaTopology.detect → kt_worker_pool_new_config | `476a96e` (auto-populate CPU lists) | ✅ |
| `kt_mla_forward` qlen>1 paged | `139bf63` + `4e2be9b` | ✅ |
| IQ2/3/4_NL family (full quantize) | `37f08cd` + `b003990` + `5fd239b` + `5a73120` | ✅ |
| IQ1_S + IQ1_M (cierre del set 16/16) | `127bbef` + `d0b0211` | ✅ |
| LlamaMoe (Zig extension — GGUF MOE llamafile) | `7784cb8` | ✅ |
| MTP head | out-of-scope (spec ausente) | ✅ closed |

**Pendiente único real (futuro):**
- MTP head — spec no existe en el checkout, mismo problema que el kml/ original
- vLLM block-level granularity (los IQ grid usan per-token-page) — solo si un modelo real lo pide
- Continuous-batching scheduling / eviction
- Cross-instance page migration (sharding)
- Qwen3 GGUF loading E2E (el parser existe pero no hay test E2E con archivo .gguf real todavía)

---

## Parte F — Recomendación al equipo

1. **El feedback del dev acierta en la consecuencia pero se equivoca en el
   diagnóstico de A1, A2, A3.** Vale la pena aceptar los P0/P1 que
   señala, pero descartando C1-C5 (los "placeholders vacíos", "loadWeights
   no-op", "sin tests", "los tarballs son drop-in" — todos falsos).
2. **Los P0 propios (A2, D1-D3) eran más urgentes que los P0 del dev.**
   El guard AMX roto significaba que el `.so` etiquetado "amx" mentía a
   quien lo carga, y los bugs D1-D3 corrompían resultados en
   configuraciones que el dev probablemente no probó (qlen > 1, tp > 1).
   Todos resueltos en `ff57125` con tests de regresión añadidos.
3. **El `comptime { _ = @import("main.zig"); }` no es el problema.**
   El doble gate `tools/audit_arity.py` + `tools/verify_abi.py` añadido
   en `ff57125` es la red de seguridad correcta, no "reemplazarlo" como
   sugirió el dev.
4. **Los tarballs del dev no existen.** Cualquier trabajo que dependa de
   `kt-simd-kernels.tar.gz` para resolver A1 hay que hacerlo desde cero
   en `src/kernels/amx/`, aprovechando que `VecF32` y los helpers
   vectorizados ya existen en `src/kernels/arch/amx.zig:461-488`.
