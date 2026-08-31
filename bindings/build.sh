#!/usr/bin/env bash
# Build the minimal pybind11 drop-in wrapper (Tier 1) against the Zig .so.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/bindings/kt_kernel_pybind.cpp"
OUT="${REPO_ROOT}/python/kt_kernel/kt_kernel_ext${PYBIND11_MODULE_SUFFIX:-.so}"

# Locate the Zig shared library (the C API we wrap).
SO_DIR="${REPO_ROOT}/zig-out/lib"
if [ -f "${SO_DIR}/libkt_kernel_ext.so" ]; then
    ZIG_SO="${SO_DIR}/libkt_kernel_ext.so"
elif [ -f "${SO_DIR}/libkt_kernel_ext_avx2.so" ]; then
    ZIG_SO="${SO_DIR}/libkt_kernel_ext_avx2.so"
else
    echo "ERROR: no libkt_kernel_ext*.so in ${SO_DIR}. Run 'zig build all-variants' first." >&2
    exit 1
fi
echo "Wrapping: ${ZIG_SO}"

# Python + pybind11 include paths.
PYINC="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PYSFX="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="${REPO_ROOT}/python/kt_kernel/kt_kernel_ext${PYSFX}"

PB_INC="$(python3 -c '
import os
for base in [
    os.path.join(os.__file__.rsplit("/", 2)[0], "site-packages", "pybind11", "include"),
]:
    pass
# fall back to torch-bundled pybind11
import glob
cands = glob.glob("/home/t0m4s/**/pybind11/include", recursive=True)
print(cands[0] if cands else "")
' 2>/dev/null || true)"
if [ -z "${PB_INC}" ] || [ ! -d "${PB_INC}" ]; then
    # final fallback: torch include (contains pybind11/)
    for c in \
        "/home/t0m4s/repos/2026/ulises/data/local/lib/python3.13/site-packages/torch/include" \
        "/home/t0m4s/.unsloth/studio/unsloth_studio/lib/python3.13/site-packages/torch/include" \
        ; do
        if [ -f "${c}/pybind11/pybind11.h" ]; then PB_INC="${c}"; break; fi
    done
fi

if [ -z "${PB_INC}" ] || [ ! -f "${PB_INC}/pybind11/pybind11.h" ]; then
    echo "ERROR: cannot locate pybind11 headers (looked for torch include). " >&2
    echo "       pip install pybind11, or set PB_INC manually." >&2
    exit 1
fi
echo "pybind11: ${PB_INC}"

echo "Compiling ${SRC} -> ${OUT}"
g++ -O2 -shared -std=c++17 -fPIC \
    -I"${PYINC}" -I"${PB_INC}" \
    -DPYBIND11_MODULE_SUFFIX=\"${PYSFX}\" \
    "${SRC}" \
    -L"$(dirname "${ZIG_SO}")" -lkt_kernel_ext \
    -o "${OUT}"

echo "Built: ${OUT}"
echo ""
echo "Smoke test:"
cd "${REPO_ROOT}/python" && KT_KERNEL_LIB_PATH="${ZIG_SO}" python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('kt_kernel_ext', 'kt_kernel/kt_kernel_ext' + __import__('sysconfig').get_config_var('EXT_SUFFIX'))
kt_kernel_ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kt_kernel_ext)
print('module:', kt_kernel_ext)
print('kt_version:', kt_kernel_ext.kt_version())
print('variant:', kt_kernel_ext.kt_get_cpu_variant())
print('moe submodule:', kt_kernel_ext.moe)
print('mla submodule:', kt_kernel_ext.mla)
print('kvcache submodule:', kt_kernel_ext.kvcache)
print('ggml_type.Q8_0 =', int(kt_kernel_ext.kvcache.ggml_type.Q8_0))
print('WorkerPool / CPUInfer:', kt_kernel_ext.WorkerPool, kt_kernel_ext.CPUInfer)
print('MOEConfig fields:', sorted(kt_kernel_ext.moe.MOEConfig().expert_num.__class__.__name__ for _ in [0]) or 'ok')
print('MLAConfig ok:', hasattr(kt_kernel_ext.mla, 'MLAConfig'))
print('PYBIND11 DROP-IN SMOKE: OK')
"
