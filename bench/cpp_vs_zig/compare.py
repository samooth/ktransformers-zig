#!/usr/bin/env python3
"""Join the C++/Zig bench outputs, print a comparison table, and verify the
two implementations compute the same thing."""
import pathlib
import struct
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out")


def load(path):
    rows = {}
    for line in path.read_text().splitlines():
        kind, name, ms, rate = line.split("|")
        rows[name] = (kind, float(ms), float(rate))
    return rows


cpp = load(OUT / "cpp.txt")
zig = load(OUT / "zig.txt")

host = ""
for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
    if line.startswith("model name"):
        host = line.split(":", 1)[1].strip()
        break
flags = set()
for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
    if line.startswith("flags"):
        flags = set(line.split(":", 1)[1].split())
        break
feat = [f for f in ("avx", "avx2", "fma", "f16c", "avx512f", "amx_bf16") if f in flags]
print(f"host: {host}")
print(f"cpu : {' '.join(feat)}")
print()

hdr = f"{'case':<20} {'shape':<19} {'C++ ms':>9} {'Zig ms':>9} {'C++ GF/s':>9} {'Zig GF/s':>9} {'Zig vs C++':>11}"
print(hdr)
print("-" * len(hdr))

for name, (kind, cpp_ms, cpp_rate) in cpp.items():
    if kind != "GEMM":
        continue
    label, _, shape = name.rpartition(" [")
    shape = shape.rstrip("]")
    zig_ms, zig_rate = zig[name][1], zig[name][2]
    print(
        f"{label:<20} {shape:<18} {cpp_ms:>9.3f} {zig_ms:>9.3f} "
        f"{cpp_rate:>9.2f} {zig_rate:>9.2f} {cpp_ms / zig_ms:>10.2f}x"
    )

print()
act_key = next(k for k in cpp if k.startswith("swiglu"))
print(f"SwiGLU activation {act_key} (GB/s of bf16 I/O)")
_, cpp_ms, cpp_rate = cpp[act_key]
_, zig_ms, zig_rate = zig[act_key]
print(f"  C++ {cpp_ms:.3f} ms ({cpp_rate:.2f} GB/s)   Zig {zig_ms:.3f} ms ({zig_rate:.2f} GB/s)   Zig/C++ = {cpp_ms / zig_ms:.2f}x")

# ---- numerical cross-check -----------------------------------------------
print()
print("numerical check (max |C++ - Zig| over the output matrix):")
ok = True
for p in sorted(OUT.glob("out_cpp_gemm_*.f32")):
    z = OUT / p.name.replace("out_cpp_", "out_zig_")
    if not z.exists():
        continue
    a = struct.unpack(f"{p.stat().st_size // 4}f", p.read_bytes())
    b = struct.unpack(f"{p.stat().st_size // 4}f", z.read_bytes())
    diff = max(abs(x - y) for x, y in zip(a, b))
    scale = max(abs(x) for x in a) or 1.0
    rel = diff / scale
    good = rel < 2e-3
    ok &= good
    print(f"  {p.name[13:-4]:<20} maxdiff {diff:.3e}  rel {rel:.2e}  {'OK' if good else 'FAIL'}")

p = OUT / "out_cpp_act_128_2048.bf16"
z = OUT / "out_zig_act_128_2048.bf16"
if p.exists() and z.exists():
    a = struct.unpack(f"{p.stat().st_size // 2}H", p.read_bytes())
    b = struct.unpack(f"{z.stat().st_size // 2}H", z.read_bytes())
    # Both are bf16; compare after normalising to float.
    n_diff = sum(1 for x, y in zip(a, b) if x != y)
    worst = max(abs((x << 16) - (y << 16)) / max(abs(x << 16), 1.0) for x, y in zip(a, b))
    print(
        f"  {'act 128x2048':<20} differing elements {n_diff}/{len(a)} "
        f"({100 * n_diff / len(a):.2f}%), worst rel {worst:.2e}"
    )

sys.exit(0 if ok else 1)