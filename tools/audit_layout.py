#!/usr/bin/env python3
"""Audit the pybind11 shim's C++ struct layouts against the Zig .so.

The C ABI passes large config structs BY VALUE (kt_moe_config_t is ~370
bytes). If the C++ mirror struct's field order or padding differs from the
Zig extern struct, the callee silently reads garbage — the arity audit
cannot catch this class of bug. This tool loads the Zig .so, calls the
kt_abi_size_* / kt_abi_field_offset probes, and compares against the C++
shim's offsetof/sizeof, parsed from the source. Any divergence is a FAIL.

Usage:
    python3 tools/audit_layout.py [--so zig-out/lib/libkt_kernel_ext.so]
"""
from __future__ import annotations

import argparse
import ctypes
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIM = REPO_ROOT / "bindings" / "kt_kernel_pybind.cpp"

# C++ shim field names, in declaration order, per struct (struct_id in the
# Zig probe). Kept in one place so the parser + probe agree.
LAYOUT_TABLES = {
    "kt_moe_config_t": (0, [
        "expert_num", "num_experts_per_tok", "hidden_size", "intermediate_size",
        "layer_idx", "pool", "num_gpu_experts", "gate_proj", "up_proj",
        "down_proj", "gate_scale", "up_scale", "down_scale", "max_len",
        "path", "save", "load", "share_cache_pool", "gate_type", "up_type",
        "down_type", "hidden_type", "swiglu_limit", "swiglu_alpha",
    ]),
    "kt_mla_config_t": (1, [
        "hidden_size", "q_lora_rank", "num_heads", "nope_size", "rope_size",
        "kv_lora_rank", "layer_idx", "pool", "q_a_proj", "q_a_norm",
        "q_b_proj", "kv_a_proj_with_mqa", "kv_a_norm", "kv_b_proj",
        "o_proj", "m_block", "n_block", "page_count",
    ]),
}


def parse_cpp_offsets(text: str, struct_name: str, fields: list[str]) -> dict[str, int]:
    """Parse the C++ shim struct and compute field offsets the way the
    Itanium x86-64 ABI would (scan declaration order, natural alignment).

    This mirrors what the compiler does for an extern "C" struct with
    default alignment: each field is placed at the next multiple of its
    natural alignment, growing the struct. C++ default-constructed member
    initializers do not affect layout.
    """
    m = re.search(rf"struct {struct_name}\s*\{{(.*?)\}};", text, re.S)
    if not m:
        raise RuntimeError(f"struct {struct_name} not found in shim source")
    body = m.group(1)
    # strip comments
    body = re.sub(r"//[^\n]*", "", body)
    decls = []
    for line in body.splitlines():
        line = line.strip().rstrip(";")
        if not line or line in ("};", "{"):
            continue
        # e.g. "int expert_num = 0", "void* pool = nullptr",
        #      "kt_quant_config_t quant_config", "const char* path = nullptr"
        mm = re.match(r"^(.*?)\s*\**\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=.*)?$", line)
        if not mm:
            continue
        decls.append((mm.group(1).strip(), mm.group(2)))

    sizes = {
        "int": 4, "uint8_t": 1, "float": 4, "double": 8,
        "void": 8, "const char": 8, "char": 1,
        "size_t": 8, "kt_quant_config_t": 20,  # u8 + 4*int with padding
    }
    # The Itanium ABI packs a u8 followed by ints with natural alignment:
    # quant_config = { u8; pad 3; int; int; int; int } = 20 bytes, align 4.
    offsets: dict[str, int] = {}
    pos = 0
    max_align = 1
    for ctype, fname in decls:
        if ctype.startswith("kt_quant_config_t"):
            size, align = 20, 4
        elif ctype in ("void", "const char") or ctype.endswith("*"):
            base = ctype.rstrip("*").strip()
            size, align = sizes.get(base, 8), 8
        else:
            size = sizes.get(ctype, 4)
            align = size  # natural alignment == size for scalar types
            if ctype == "uint8_t":
                align = 1
        # Align pos to the field's natural alignment BEFORE placing it
        pos = (pos + align - 1) & ~(align - 1)
        offsets[fname] = pos
        pos += size
        max_align = max(max_align, align)
    # tail padding to struct alignment
    total = (pos + max_align - 1) & ~(max_align - 1)
    offsets["__size__"] = total
    return offsets


def load_zig_probes(so_path: Path):
    lib = ctypes.CDLL(str(so_path))
    lib.kt_abi_size_moe_config.restype = ctypes.c_size_t
    lib.kt_abi_size_mla_config.restype = ctypes.c_size_t
    lib.kt_abi_field_offset.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.kt_abi_field_offset.restype = ctypes.c_size_t
    return lib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=Path,
                    default=REPO_ROOT / "zig-out" / "lib" / "libkt_kernel_ext.so")
    args = ap.parse_args()
    if not args.so.exists():
        sys.stderr.write(f"error: {args.so} not found (build first)\n")
        return 2

    lib = load_zig_probes(args.so)
    shim_src = SHIM.read_text()
    ok = True

    size_fns = {
        "kt_moe_config_t": lib.kt_abi_size_moe_config,
        "kt_mla_config_t": lib.kt_abi_size_mla_config,
    }

    for struct_name, (struct_id, fields) in LAYOUT_TABLES.items():
        cpp = parse_cpp_offsets(shim_src, struct_name, fields)
        zig_size = size_fns[struct_name]()
        cpp_size = cpp["__size__"]
        print(f"=== {struct_name} ===")
        print(f"  zig sizeof : {zig_size}")
        print(f"  cpp sizeof : {cpp_size}")
        if zig_size != cpp_size:
            print(f"  SIZE MISMATCH (+{abs(int(zig_size) - int(cpp_size))}) — FAIL")
            ok = False
        for idx, fname in enumerate(fields):
            z = lib.kt_abi_field_offset(struct_id, idx)
            c = cpp.get(fname)
            status = "OK" if z == c else f"MISMATCH (zig {z} vs cpp {c})"
            if z != c:
                ok = False
            print(f"  {fname:28s} zig={z:<4} cpp={c:<4} {status}")
        print()

    if ok:
        print("PASS: pybind11 shim struct layouts match the Zig extern structs.")
        return 0
    print("FAIL: struct layout divergence — the C++ shim would corrupt the C ABI.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
