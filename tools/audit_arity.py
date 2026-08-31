#!/usr/bin/env python3
"""Audit the C API ABI signature parity of ktransformers-zig.

Parses include/kt_kernel.h to discover every declared `kt_*` function with
its full prototype, then parses the matching `pub export fn` in
src/main.zig and reports any divergences in name OR arity.

This catches the bugs the name-only verify_abi.py cannot: a Zig export with
the right name but a wrong signature would silently corrupt header-following
callers' stack/registers.

Usage:
    python3 tools/audit_arity.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER = REPO_ROOT / "include" / "kt_kernel.h"
MAIN_ZIG = REPO_ROOT / "src" / "main.zig"


def _strip_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return text


def _find_balanced(text: str, start: int, open_c: str, close_c: str) -> int:
    """Return the index of the matching close char, or -1 if not found."""
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def parse_header_prototypes(path: Path) -> dict[str, int]:
    """Return {function_name: arg_count} for every `kt_xxx(...)` declaration."""
    text = _strip_comments(path.read_text())
    proto: dict[str, int] = {}
    for m in re.finditer(r"\b(kt_[A-Za-z0-9_]+)\s*\(", text):
        name = m.group(1)
        if name in proto:
            continue  # dedupe (typedef forward-decls etc.)
        close = _find_balanced(text, m.end() - 1, "(", ")")
        if close == -1:
            continue
        # Require `;` (or `;` then optional newline) within 200 chars to
        # confirm this is a function prototype and not e.g. a forward decl
        # inside a struct (we already strip // and /* */ so structs should
        # not contain kt_xxx(); but kt_forward_decl style might).
        tail = text[close + 1 : close + 200]
        if not re.match(r"\s*;", tail):
            continue
        body = text[m.end() : close]
        proto[name] = _count_args(body)
    return proto


def parse_zig_exports(path: Path) -> dict[str, int]:
    """Return {function_name: arg_count} for every `pub export fn kt_xxx(...)`."""
    text = path.read_text()
    exports: dict[str, int] = {}
    for m in re.finditer(r"(?:pub\s+)?export\s+fn\s+(kt_[A-Za-z0-9_]+)\s*\(", text):
        name = m.group(1)
        if name in exports:
            continue
        close = _find_balanced(text, m.end() - 1, "(", ")")
        if close == -1:
            continue
        body = text[m.end() : close]
        exports[name] = _count_args(body)
    return exports


def _count_args(args_text: str) -> int:
    """Count top-level comma-separated parameter tokens.

    Skips the interior of balanced `(` / `[` / `{` and, crucially, the
    interior of function-pointer parameter types like
    `void (*callback)(int, const uintptr_t*, ...)` — those inner commas
    are NOT separate parameters of the outer function. Also tolerates Zig
    optional prefixes (`?*anyopaque`, `?[*]const u8`) and the pointer-
    follows-paren marker (`(*fn)(...)`) by walking balanced brackets from
    any `(` that follows an identifier or `*`.
    """
    s = args_text.strip()
    # Strip a trailing comma (header / Zig style may include it after the
    # last parameter in a multi-line declaration).
    if s.endswith(","):
        s = s[:-1].rstrip()
    if not s or s == "void":
        return 0
    n = 1
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            # Skip past any fn-ptr parameter type. A fn-ptr in Zig looks like
            # `fntype (*name)(args)`; in C/header: `ret (*name)(args)`.
            # Heuristic: if the char before `(` is `*` or `:` or `]`, or the
            # next non-space char after `(` is `*`, it's a fn-ptr.
            j = i + 1
            k = j
            while k < len(s) and s[k].isspace():
                k += 1
            prev = s[i - 1] if i > 0 else " "
            is_fnptr = (prev == "*" or prev == ":" or prev == "]"
                        or (k < len(s) and s[k] == "*"))
            if is_fnptr:
                close = _find_balanced(s, i, "(", ")")
                if close != -1:
                    i = close + 1
                    continue
            depth += 1
            i += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                break
            i += 1
        elif ch == "," and depth == 0:
            n += 1
            i += 1
        else:
            i += 1
    return n


def main() -> int:
    header = parse_header_prototypes(HEADER)
    zig = parse_zig_exports(MAIN_ZIG)
    common = sorted(set(header) | set(zig))

    name_mismatch = []
    arity_mismatch = []
    for name in common:
        h, z = header.get(name), zig.get(name)
        if h is None:
            name_mismatch.append((name, "only in Zig (.so)", z))
            continue
        if z is None:
            name_mismatch.append((name, "only in header", h))
            continue
        if h != z:
            arity_mismatch.append((name, h, z))

    print(f"Header prototypes : {len(header)}")
    print(f"Zig exports      : {len(zig)}")
    print()

    if name_mismatch:
        print("=== NAME MISMATCHES (header vs Zig) ===")
        for name, where, arity in name_mismatch:
            print(f"  {name:35s} {where:25s} arity={arity}")
        print()

    if arity_mismatch:
        print("=== ARITY MISMATCHES (header vs Zig) ===")
        for name, h, z in arity_mismatch:
            print(f"  {name:35s} header={h}  zig={z}")
        print()
        ok = False
    else:
        ok = True

    # Cross-checks worth reporting even on success
    only_in_header = sorted(n for n in header if n not in zig)
    only_in_zig = sorted(n for n in zig if n not in header)
    if only_in_header:
        print(f"=== {len(only_in_header)} NAMES DECLARED IN HEADER, NOT IN ZIG .SO (truly missing) ===")
        for n in only_in_header:
            print(f"  - {n}")
        ok = False
    if only_in_zig:
        print(f"=== {len(only_in_zig)} ZIG EXPORTS NOT IN HEADER (ungated extras) ===")
        for n in only_in_zig:
            print(f"  + {n} (arity {zig[n]})")

    print()
    if ok:
        print(f"PASS: all {len(common)} shared names agree in arity, "
              f"{len(header)} header decls + {len(zig)} Zig exports are in sync.")
        return 0
    print("FAIL: ABI signature parity gap — see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
