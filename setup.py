"""
setuptools build for kt-kernel.

Builds all 6 CPU-variant shared libraries via `zig build all-variants`
and bundles them into the kt_kernel package directory so the runtime
variant auto-detect (_find_so) can locate them at import time.
"""

import os
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext

VARIANTS = [
    "avx2",
    "avx512_base",
    "avx512_vnni",
    "avx512_vbmi",
    "avx512_bf16",
    "amx",
]


class ZigBuildExt(build_ext):
    """Custom build step: compile all variants with Zig and copy .sos into the package."""

    def run(self):
        # Build all variants in-tree.
        # OPTIMIZE controls the Zig build mode (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall).
        # Default is Debug for build reliability; set OPTIMIZE=ReleaseSafe for distribution.
        optimize = os.environ.get("KT_OPTIMIZE", "Debug")
        print(f"[kt-kernel] Building all CPU variants via Zig (optimize={optimize})...")
        subprocess.check_call(
            ["zig", "build", f"-Doptimize={optimize}", "all-variants"],
            cwd=self._project_root(),
        )

        # Locate built .so files.
        zig_out = os.path.join(self._project_root(), "zig-out", "lib")
        if not os.path.isdir(zig_out):
            raise RuntimeError(f"zig-out/lib not found at {zig_out}")

        # Destination: the kt_kernel package directory (next to __init__.py).
        pkg_dir = self._package_dir()
        os.makedirs(pkg_dir, exist_ok=True)

        copied = 0
        for variant in VARIANTS:
            name = f"libkt_kernel_ext_{variant}.so"
            src = os.path.join(zig_out, name)
            dst = os.path.join(pkg_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied += 1
                print(f"  [kt-kernel] bundled {name}")
            else:
                print(f"  [kt-kernel] WARNING: {name} not found in zig-out/lib (skipped)")

        if copied == 0:
            raise RuntimeError(
                "No variant .so files found. Check that `zig build all-variants` succeeded."
            )
        print(f"[kt-kernel] Bundled {copied} variant .so files into {pkg_dir}")

    def _project_root(self):
        # This file sits at the repo root (same level as build.zig).
        return os.path.dirname(os.path.abspath(__file__))

    def _package_dir(self):
        return os.path.join(self._project_root(), "python", "kt_kernel")


setup(
    cmdclass={"build_ext": ZigBuildExt},
    # Disable the default C extension build (we use Zig, not Cython).
    ext_modules=[],
    script_args=["build_ext", "--inplace"],
)
