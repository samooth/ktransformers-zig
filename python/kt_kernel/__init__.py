"""
kt_zig — pure-Python ctypes wrapper for the ktransformers-zig C API.

Usage:
    import kt_zig
    print(kt_zig.kt_version())

    # Small BF16 linear forward
    linear = kt_zig.kt_linear_new(config)
    kt_zig.kt_linear_forward(linear, input_bf16, output_bf16, batch_size)
    kt_zig.kt_linear_free(linear)
"""

import ctypes
import os
import platform

# ---------------------------------------------------------------------------
# Select the variant .so to load.
# ---------------------------------------------------------------------------

def _detect_variant_name():
    """Return the best variant suffix for this CPU by checking /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
    except OSError:
        return "avx2"

    if "amx" in cpuinfo:
        return "amx"
    if "avx512_vbmi" in cpuinfo or "avx512vbmi" in cpuinfo:
        return "avx512_vbmi"
    if "avx512_bf16" in cpuinfo or "avx512bf16" in cpuinfo:
        return "avx512_bf16"
    if "avx512_vnni" in cpuinfo or "avx512vnni" in cpuinfo:
        return "avx512_vnni"
    if "avx512f" in cpuinfo or "avx512" in cpuinfo:
        return "avx512_base"
    return "avx2"


def _find_so():
    """Locate the variant .so, searching next to this file then zig-out/lib."""
    variant = _detect_variant_name()
    name = f"libkt_kernel_ext_{variant}.so"

    # Next to this package (installed alongside).
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(pkg_dir, name)
    if os.path.isfile(candidate):
        return candidate

    # Fall back to the non-suffixed name (single-variant build).
    fallback = os.path.join(pkg_dir, "libkt_kernel_ext.so")
    if os.path.isfile(fallback):
        return fallback

    # Walk up to the workspace root and check zig-out/lib.
    root = pkg_dir
    for _ in range(4):
        root = os.path.dirname(root)
        candidate = os.path.join(root, "zig-out", "lib", name)
        if os.path.isfile(candidate):
            return candidate
        fallback = os.path.join(root, "zig-out", "lib", "libkt_kernel_ext.so")
        if os.path.isfile(fallback):
            return fallback

    raise RuntimeError(
        f"Could not find {name} (variant={variant}). "
        "Run `zig build all-variants` first."
    )


_so_path = _find_so()
_so = ctypes.CDLL(_so_path)


# ---------------------------------------------------------------------------
# Opaque handles
# ---------------------------------------------------------------------------

class KT_WorkerPool(ctypes.Structure):
    pass

class KT_CPUInfer(ctypes.Structure):
    pass

class KT_MOE(ctypes.Structure):
    pass

class KT_MLA(ctypes.Structure):
    pass

class KT_Linear(ctypes.Structure):
    pass

class KT_MLP(ctypes.Structure):
    pass

class KT_Gate(ctypes.Structure):
    pass


# ---------------------------------------------------------------------------
# Exported functions
# ---------------------------------------------------------------------------

kt_version = _so.kt_version
kt_version.argtypes = []
kt_version.restype = ctypes.c_char_p

kt_get_cpu_variant = _so.kt_get_cpu_variant
kt_get_cpu_variant.argtypes = []
kt_get_cpu_variant.restype = ctypes.c_char_p

kt_ggml_init = _so.kt_ggml_init
kt_ggml_init.argtypes = []
kt_ggml_init.restype = None

kt_bf16_to_f32 = _so.kt_bf16_to_f32
kt_bf16_to_f32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
kt_bf16_to_f32.restype = None

kt_f32_to_bf16 = _so.kt_f32_to_bf16
kt_f32_to_bf16.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_size_t]
kt_f32_to_bf16.restype = None

# Worker pool
kt_worker_pool_new = _so.kt_worker_pool_new
kt_worker_pool_new.argtypes = [ctypes.c_int]
kt_worker_pool_new.restype = ctypes.POINTER(KT_WorkerPool)

kt_worker_pool_free = _so.kt_worker_pool_free
kt_worker_pool_free.argtypes = [ctypes.POINTER(KT_WorkerPool)]
kt_worker_pool_free.restype = None

# CPUInfer
kt_cpuinfer_new = _so.kt_cpuinfer_new
kt_cpuinfer_new.argtypes = [ctypes.c_int]
kt_cpuinfer_new.restype = ctypes.POINTER(KT_CPUInfer)

kt_cpuinfer_free = _so.kt_cpuinfer_free
kt_cpuinfer_free.argtypes = [ctypes.POINTER(KT_CPUInfer)]
kt_cpuinfer_free.restype = None

# Linear
kt_linear_new = _so.kt_linear_new
kt_linear_new.argtypes = [ctypes.c_void_p]
kt_linear_new.restype = ctypes.POINTER(KT_Linear)

kt_linear_free = _so.kt_linear_free
kt_linear_free.argtypes = [ctypes.POINTER(KT_Linear)]
kt_linear_free.restype = None

kt_linear_forward = _so.kt_linear_forward
kt_linear_forward.argtypes = [
    ctypes.POINTER(KT_Linear),
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]
kt_linear_forward.restype = None

# Gate
kt_gate_new = _so.kt_gate_new
kt_gate_new.argtypes = [ctypes.c_void_p]
kt_gate_new.restype = ctypes.POINTER(KT_Gate)

kt_gate_free = _so.kt_gate_free
kt_gate_free.argtypes = [ctypes.POINTER(KT_Gate)]
kt_gate_free.restype = None

kt_gate_forward = _so.kt_gate_forward
kt_gate_forward.argtypes = [
    ctypes.POINTER(KT_Gate),
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
]
kt_gate_forward.restype = None

# MLP
kt_mlp_new = _so.kt_mlp_new
kt_mlp_new.argtypes = [ctypes.c_void_p]
kt_mlp_new.restype = ctypes.POINTER(KT_MLP)

kt_mlp_free = _so.kt_mlp_free
kt_mlp_free.argtypes = [ctypes.POINTER(KT_MLP)]
kt_mlp_free.restype = None

kt_mlp_forward = _so.kt_mlp_forward
kt_mlp_forward.argtypes = [
    ctypes.POINTER(KT_MLP),
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]
kt_mlp_forward.restype = None

__all__ = [
    "kt_version",
    "kt_get_cpu_variant",
    "kt_ggml_init",
    "kt_bf16_to_f32",
    "kt_f32_to_bf16",
    "kt_worker_pool_new",
    "kt_worker_pool_free",
    "kt_cpuinfer_new",
    "kt_cpuinfer_free",
    "kt_linear_new",
    "kt_linear_free",
    "kt_linear_forward",
    "kt_gate_new",
    "kt_gate_free",
    "kt_gate_forward",
    "kt_mlp_new",
    "kt_mlp_free",
    "kt_mlp_forward",
    "_so_path",
]
