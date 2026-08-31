#!/usr/bin/env python3
"""Verify the C API ABI of the ktransformers-zig shared libraries.

Parses include/kt_kernel.h to discover the public `kt_*` function API, then
checks that every declared symbol is actually exported by each built variant
shared object (libkt_kernel_ext_*.so). This is the `nm -D` check used to
validate dev reports, automated.

It deliberately lives outside the Zig build (no build.zig / root.zig wiring)
and touches no project source, so it cannot interfere with in-flight work.

Usage:
    python3 tools/verify_abi.py
    python3 tools/verify_abi.py --so-dir zig-out/lib --header include/kt_kernel.h
    python3 tools/verify_abi.py --prefix libkt_kernel_ext   # default
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER_DEFAULT = REPO_ROOT / "include" / "kt_kernel.h"
SO_DIR_DEFAULT = REPO_ROOT / "zig-out" / "lib"
SO_PREFIX_DEFAULT = "libkt_kernel_ext"

# Captures `kt_xxx(` occurrences -> function names declared in the C header.
SYMBOL_RE = re.compile(r"\b(kt_[A-Za-z0-9_]+)\s*\(")

# A symbol is "exported" if nm reports it as global/weak text (T or W).
EXPORTED_RE = re.compile(r"^[0-9a-f]*\s+[TWt]\s+(\S+)$")


def discover_expected_symbols(header: Path) -> List[str]:
    text = header.read_text(encoding="utf-8", errors="ignore")
    found = SYMBOL_RE.findall(text)
    # Dedupe, keep sorted, drop obvious non-functions (none expected, but be safe).
    return sorted(set(found))


def exported_symbols(so: Path) -> Set[str]:
    try:
        out = subprocess.run(
            ["nm", "-D", str(so)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except FileNotFoundError:
        sys.stderr.write("error: `nm` not found on PATH\n")
        sys.exit(2)
    symbols: Set[str] = set()
    for line in out.splitlines():
        m = EXPORTED_RE.match(line.strip())
        if m:
            symbols.add(m.group(1))
    return symbols


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify ktransformers-zig C API ABI.")
    ap.add_argument("--header", type=Path, default=HEADER_DEFAULT)
    ap.add_argument("--so-dir", type=Path, default=SO_DIR_DEFAULT)
    ap.add_argument("--prefix", default=SO_PREFIX_DEFAULT)
    args = ap.parse_args()

    if not args.header.exists():
        sys.stderr.write(f"error: header not found: {args.header}\n")
        return 2
    if not args.so_dir.is_dir():
        sys.stderr.write(
            f"warning: so dir not found: {args.so_dir} (build first?)\n"
            f"         expected symbols from header are still listed below.\n"
        )

    expected = discover_expected_symbols(args.header)
    if not expected:
        sys.stderr.write("error: no kt_* symbols discovered in header\n")
        return 2

    # Find variant .so files.
    sos = sorted(args.so_dir.glob(f"{args.prefix}*.so")) if args.so_dir.is_dir() else []
    if not sos:
        sys.stderr.write("warning: no matching .so files; cannot check exports.\n")

    print(f"Header        : {args.header}")
    print(f"Expected API  : {len(expected)} kt_* symbols")
    print(f"Variants found: {len(sos)}")
    print("=" * 64)

    overall_ok = True
    for so in sos:
        exported = exported_symbols(so)
        missing = [s for s in expected if s not in exported]
        status = "OK" if not missing else f"MISSING {len(missing)}"
        if missing:
            overall_ok = False
        print(f"{so.name:40s} {status}")
        for s in missing:
            print(f"    - {s}")

    # Symbols declared in the header but absent from EVERY variant are likely
    # false positives (e.g. inside a comment) rather than real gaps.
    if sos:
        union_exported: Set[str] = set()
        for so in sos:
            union_exported |= exported_symbols(so)
        never = [s for s in expected if s not in union_exported]
        if never:
            print("-" * 64)
            print("Declared in header but exported by NO variant (verify manually):")
            for s in never:
                print(f"    ? {s}")

    print("=" * 64)
    if not sos:
        # No binaries to check against; not a failure of the tool.
        print("No .so files checked (build the project to verify exports).")
        return 0
    if overall_ok:
        print(f"PASS: all {len(expected)} declared symbols exported by all variants.")
        return 0
    print("FAIL: some declared symbols are not exported by every variant.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
