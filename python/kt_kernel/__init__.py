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

# ---------------------------------------------------------------------------
# GGML quantization types (Q8_0, Q4_K, Q5_K, Q6_K, Q8_K)
# ---------------------------------------------------------------------------
# Block layouts are byte-exact vs llama.cpp ggml-common.h: blocks produced by
# quantize_row() feed the kt_matmul_q* functions directly, and weights loaded
# from .gguf files can be passed to the matmuls without any conversion.

Q8_0_QK = 32            # weights per Q8_0 block
Q8_0_BLOCK_BYTES = 34   # f16 d + 32 x i8
Q4_K_QK = 256           # weights per K-quant super-block
Q4_K_BLOCK_BYTES = 144  # f16 d+dmin, 12B 6-bit scales, 128B 4-bit quants
Q5_K_BLOCK_BYTES = 176  # f16 d+dmin, 12B scales, 32B high bits, 128B 4-bit
Q6_K_BLOCK_BYTES = 210  # 128B ql, 64B qh, 16 x i8 scales, f16 d
Q8_K_BLOCK_BYTES = 292  # f32 d (NOT f16), 256 x i8, 16 x i16 bsums

_GGML_FORMATS = ("q8_0", "q4_k", "q5_k", "q6_k", "q8_k")
_GGML_LAYOUT = {
    "q8_0": (Q8_0_QK, Q8_0_BLOCK_BYTES),
    "q4_k": (Q4_K_QK, Q4_K_BLOCK_BYTES),
    "q5_k": (Q4_K_QK, Q5_K_BLOCK_BYTES),
    "q6_k": (Q4_K_QK, Q6_K_BLOCK_BYTES),
    "q8_k": (Q4_K_QK, Q8_K_BLOCK_BYTES),
}

# Raw one-row bindings (kept public for zero-copy users who manage buffers):
#   kt_quantize_<fmt>(f32* src, void* dst, k)      k multiple of QK
#   kt_dequantize_<fmt>(void* src, f32* dst, k)
# Raw matmuls:
#   kt_matmul_<fmt>(bf16* a, void* b, f32* c, m, n, k, lda, ldb, ldc)
#   b holds n rows of k/QK contiguous blocks (ldb = 1), a is BF16.
_GGML_QUANT = {}
_GGML_DEQUANT = {}
_GGML_MATMUL = {}
for _fmt in _GGML_FORMATS:
    _q = getattr(_so, "kt_quantize_" + _fmt)
    _q.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_size_t]
    _q.restype = None
    _d = getattr(_so, "kt_dequantize_" + _fmt)
    _d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    _d.restype = None
    _m = getattr(_so, "kt_matmul_" + _fmt)
    _m.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    _m.restype = None
    globals()["kt_quantize_" + _fmt] = _q
    globals()["kt_dequantize_" + _fmt] = _d
    globals()["kt_matmul_" + _fmt] = _m
    _GGML_QUANT[_fmt] = _q
    _GGML_DEQUANT[_fmt] = _d
    _GGML_MATMUL[_fmt] = _m


def _ggml_layout(fmt):
    try:
        return _GGML_LAYOUT[fmt]
    except KeyError:
        raise ValueError(
            "unknown GGML format %r (expected one of %s)" % (fmt, ", ".join(_GGML_FORMATS))
        ) from None


def quantize_row(fmt, src):
    """Quantize one row of float32 values into packed GGML block bytes.

    fmt: one of "q8_0", "q4_k", "q5_k", "q6_k", "q8_k".
    src: sequence of numbers; length must be a multiple of the format's
         weights-per-block (32 for q8_0, 256 for the K-quants).
    Returns bytes of length len(src)//QK * BLOCK_BYTES.
    """
    qk, block_bytes = _ggml_layout(fmt)
    k = len(src)
    if k % qk:
        raise ValueError("%s: row length %d is not a multiple of %d" % (fmt, k, qk))
    src_buf = (ctypes.c_float * k)(*src)
    dst = ctypes.create_string_buffer((k // qk) * block_bytes)
    _GGML_QUANT[fmt](src_buf, dst, k)
    return dst.raw


def dequantize_row(fmt, blocks, k):
    """Dequantize GGML blocks back to float32.

    blocks: bytes (or equally-sized buffer) holding k//QK blocks.
    Returns a ctypes c_float array of length k (list() it or index directly).
    """
    qk, block_bytes = _ggml_layout(fmt)
    if k % qk:
        raise ValueError("%s: k=%d is not a multiple of %d" % (fmt, k, qk))
    expected = (k // qk) * block_bytes
    if len(blocks) != expected:
        raise ValueError(
            "%s: expected %d block bytes for k=%d, got %d" % (fmt, expected, k, len(blocks))
        )
    out = (ctypes.c_float * k)()
    _GGML_DEQUANT[fmt](blocks, out, k)
    return out


def matmul_quantized(fmt, a_f32, b_blocks, m, n, k):
    """Quantized GEMM with a float32 convenience interface.

    a_f32: flat sequence of m*k activation floats (converted to BF16 here —
           for repeated use, convert once with kt_f32_to_bf16 and call the
           raw kt_matmul_<fmt> instead).
    b_blocks: packed bytes of n rows x (k//QK) blocks each, contiguous.
    Returns a ctypes c_float array of m*n (row-major output).
    """
    qk, block_bytes = _ggml_layout(fmt)
    if k % qk:
        raise ValueError("%s: k=%d is not a multiple of %d" % (fmt, k, qk))
    if len(a_f32) != m * k:
        raise ValueError("a_f32: expected %d values, got %d" % (m * k, len(a_f32)))
    expected = n * (k // qk) * block_bytes
    if len(b_blocks) != expected:
        raise ValueError(
            "%s: expected %d block bytes, got %d" % (fmt, expected, len(b_blocks))
        )
    a = (ctypes.c_float * (m * k))(*a_f32)
    a_bf16 = ctypes.create_string_buffer(m * k * 2)
    kt_f32_to_bf16(a, a_bf16, m * k)
    c = (ctypes.c_float * (m * n))()
    _GGML_MATMUL[fmt](a_bf16, b_blocks, c, m, n, k, k, 1, n)
    return c


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
    # GGML quantization
    "Q8_0_QK", "Q8_0_BLOCK_BYTES", "Q4_K_QK",
    "Q4_K_BLOCK_BYTES", "Q5_K_BLOCK_BYTES", "Q6_K_BLOCK_BYTES", "Q8_K_BLOCK_BYTES",
    "quantize_row", "dequantize_row", "matmul_quantized",
] + [
    "kt_quantize_" + _f for _f in _GGML_FORMATS
] + [
    "kt_dequantize_" + _f for _f in _GGML_FORMATS
] + [
    "kt_matmul_" + _f for _f in _GGML_FORMATS
]
