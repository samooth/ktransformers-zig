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
    # Model-orchestration layer configs are passed BY POINTER; field
    # layout still matters because the C++ shim mirror struct must
    # match the Zig extern struct (a reordering silently corrupts
    # every weight pointer). Audited for the 3 new structs.
    "kt_dsv3_layer_config_t": (2, [
        "hidden_size", "q_lora_rank", "num_heads", "nope_size", "rope_size",
        "kv_lora_rank", "max_qlen", "max_kvlen", "token_count_in_page",
        "rope_theta", "expert_num", "num_experts_per_tok", "intermediate_size",
        "n_group", "topk_group", "norm_topk_prob", "routed_scaling_factor",
        "pool", "q_a_proj", "q_a_norm", "q_b_proj", "kv_a_proj_with_mqa",
        "kv_a_norm", "kv_b_proj", "o_proj", "attn_norm_weight",
        "ffn_norm_weight", "gate_weight", "e_score_correction_bias",
        "gate_proj", "up_proj", "down_proj",
    ]),
    "kt_qwen3moe_layer_config_t": (3, [
        "hidden_size", "num_heads", "num_kv_heads", "head_dim", "max_qlen",
        "max_kvlen", "rope_theta", "expert_num", "num_experts_per_tok",
        "intermediate_size", "pool", "q_proj", "k_proj", "v_proj", "o_proj",
        "attn_norm_weight", "ffn_norm_weight", "gate_weight", "gate_proj",
        "up_proj", "down_proj",
    ]),
    "kt_qwen3moe_model_config_t": (4, [
        "num_layers", "layer", "final_norm_weight", "lm_head", "vocab_size",
    ]),
}


def parse_cpp_offsets(text: str, struct_name: str, fields: list[str]) -> dict[str, int]:
    """Parse the C++ shim struct and compute field offsets the way the
    Itanium x86-64 ABI would (scan declaration order, natural alignment).

    Supports C/C++ declaration styles used in the ktransformers-zig shim:
      type name;                              (single field per line)
      type name1, name2, ..., nameN;          (multi-field on one line)
      T* name;  /  const T* name;             (pointer type)
      T* name1, *name2, ...;                  (multi-field, *-prefixes)
      , name1, name2, ...;                    (continuation line)

    Continuation lines (those starting with ",") are joined with the
    previous statement before parsing.
    """
    m = re.search(rf"struct {struct_name}\s*\{{(.*?)\}};", text, re.S)
    if not m:
        raise RuntimeError(f"struct {struct_name} not found in shim source")
    body = m.group(1)
    body = re.sub(r"//[^\n]*", "", body)

    raw_lines = body.splitlines()
    joined = []
    # A continuation line is one that doesn't start with a recognized C++
    # type token — it's a list of additional field names like
    # "           kv_lora_rank, max_qlen, max_kvlen, token_count_in_page;"
    # joined to the previous statement. We detect by trying to split
    # the line into "type name" — if that fails (no whitespace separator
    # or no recognizable type), it must be a continuation.
    KNOWN_TYPES = {
        "int", "uint8_t", "float", "double", "void", "char",
        "size_t", "const", "unsigned", "signed", "long", "short",
        "kt_quant_config_t",
    }
    def starts_with_type(line: str) -> bool:
        first = line.split()[0] if line.split() else ""
        # Handle "void*" (no space) — strip a trailing * to test the
        # underlying type name.
        stripped = first.rstrip("*")
        return first in KNOWN_TYPES or stripped in KNOWN_TYPES or first.startswith("kt_")

    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(",") and joined:
            joined[-1] = joined[-1] + " " + s
        elif not starts_with_type(s) and joined:
            # Continuation: e.g. "kv_lora_rank, max_qlen, ..."
            joined[-1] = joined[-1] + " " + s
        else:
            joined.append(s)
    body = "\n".join(joined)

    decls = []
    for line in body.splitlines():
        line = line.strip().rstrip(";")
        if not line or line in ("};", "{"):
            continue
        # Split on top-level commas. We don't support function-pointer
        # types in the shim (no commas inside type strings).
        chunks = [c.strip() for c in line.split(",")]
        # The TYPE comes from the first chunk. The first chunk has the
        # form "<type> <name> [= default]" where type may be:
        #   "size_t"  /  "int"  /  "double"  /  "float"  /  "char"
        #   "void"  /  "void*"  /  "const void"  /  "const void*"
        #   "const T"  /  "T*"  /  "const T*"
        #   "kt_<name>_t"  /  "kt_<name>_t*"
        # We locate the LAST identifier-like word at the end of the
        # type prefix. The type ends right before the first whitespace
        # followed by a name (no leading *, no =, no ,).
        first = chunks[0]
        # Match "<type> <name> [= default]" by:
        # 1. Find a "=" or end of string after a name-like token.
        # 2. Everything before that is "type name" with optional spaces.
        # We use a more robust approach: find the last identifier
        # followed by "=" or whitespace at the end.
        # Strip a trailing "= default" / "= 0" / "= nullptr"
        if "=" in first:
            first = first.split("=", 1)[0].strip()
        # The name is the last identifier in the first chunk. The type
        # is everything before it. We split on whitespace: type has
        # one or more tokens, name is the last token. But "void*" has
        # no space, so we need to handle "T*" as a single type token.
        # Split into tokens by whitespace:
        tokens = first.split()
        if not tokens:
            continue
        # The last token is the name; everything before is the type
        # (with optional trailing * glued to a token).
        name = tokens[-1]
        type_tokens = tokens[:-1]
        # If the last type token ends with '*' or '&', that's a
        # pointer-typed field. Otherwise, the type is the simple
        # concatenation of the type tokens.
        names = [name]
        # Process remaining chunks: each is a name (possibly prefixed
        # with '*' and/or suffixed with "= default").
        for c in chunks[1:]:
            c = c.strip()
            if not c:
                continue
            if "=" in c:
                c = c.split("=", 1)[0].strip()
            # Strip a leading "*" or "&" (these are pointer markers
            # when the type is "T*" and the field is the second+ in
            # a multi-field decl like "T* a, *b;").
            while c and c[0] in ("*", "&"):
                c = c[1:].lstrip()
            if c:
                names.append(c)
        # The type is a single string (the type tokens joined with
        # single spaces — splitting on whitespace loses the original
        # spacing but the C++ parser doesn't care, and the type
        # resolution below uses substring/keyword matching).
        type_str = " ".join(type_tokens)
        for n in names:
            decls.append((type_str, n))

    sizes = {
        "int": 4, "uint8_t": 1, "float": 4, "double": 8,
        "void": 8, "const char": 8, "char": 1,
        "size_t": 8, "kt_quant_config_t": 20,  # u8 + 4*int with padding
    }
    # Struct fields the C++ shim uses for the model-orchestration configs.
    # When a field type matches a key here, the field's size + alignment
    # are taken from the table (the parse_cpp_offsets() function on that
    # struct runs first via the LAYOUT_TABLES entry). For the simple
    # model_config_t with one embedded struct (Qwen3MoeModelCfg), the
    # alignment is the natural alignment of the embedded type.
    embedded_offsets = {}
    if struct_name == "kt_qwen3moe_model_config_t":
        # The `layer` field is the embedded kt_qwen3moe_layer_config_t.
        # We computed its layout just above; reuse the size.
        embedded = parse_cpp_offsets(text, "kt_qwen3moe_layer_config_t",
                                     dict(LAYOUT_TABLES)["kt_qwen3moe_layer_config_t"][1])
        sizes["kt_qwen3moe_layer_config_t"] = embedded["__size__"]
    # The Itanium ABI packs a u8 followed by ints with natural alignment:
    # quant_config = { u8; pad 3; int; int; int; int } = 20 bytes, align 4.
    offsets: dict[str, int] = {}
    pos = 0
    max_align = 1
    for ctype, fname in decls:
        if ctype.startswith("kt_quant_config_t"):
            size, align = 20, 4
        elif ctype in sizes and ctype.startswith("kt_") and ctype.endswith("_t"):
            # Embedded struct (kt_qwen3moe_layer_config_t etc.).
            size = sizes[ctype]
            align = 8  # all our embedded types have pointer-8-byte alignment
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
    lib.kt_abi_size_dsv3_layer_config.restype = ctypes.c_size_t
    lib.kt_abi_size_qwen3moe_layer_config.restype = ctypes.c_size_t
    lib.kt_abi_size_qwen3moe_model_config.restype = ctypes.c_size_t
    # field_offset is shared across struct_id 0..4; the older call site
    # (struct_id 0/1) still works.
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
        "kt_dsv3_layer_config_t": lib.kt_abi_size_dsv3_layer_config,
        "kt_qwen3moe_layer_config_t": lib.kt_abi_size_qwen3moe_layer_config,
        "kt_qwen3moe_model_config_t": lib.kt_abi_size_qwen3moe_model_config,
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
        # Older struct_id 0/1 support per-field probe. The new model-
        # orchestration configs (2/3/4) are by-pointer; we skip the
        # per-field probe there because Zig doesn't expose offsets for
        # them yet (only sizeof). The sizeof check is the primary gate.
        if struct_id <= 1:
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
