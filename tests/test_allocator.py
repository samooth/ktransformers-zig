"""
Smoke test for the kt_set_default_allocator ctypes binding.

Verifies that the userland can install a custom allocator (alloc +
free + resize-stub) via the kt_kernel/__init__.py ctypes wrapper, that
it actually intercepts allocations from a kt_worker_pool_new +
kt_worker_pool_free round-trip, and that kt_set_default_allocator(None)
restores the default (process-allocator) behavior.

Run:  python3 tests/test_allocator.py
"""

import ctypes
import os
import sys

# Locate the .so without depending on the installed package being on
# sys.path. The wrapper at python/kt_kernel/__init__.py does this for
# us; we just need to ensure the ctypes.CDLL load is reachable.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

import kt_kernel


class Counters(ctypes.Structure):
    _fields_ = [
        ("allocs", ctypes.c_size_t),
        ("frees", ctypes.c_size_t),
        ("bytes", ctypes.c_size_t),
    ]


ALLOC = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
)
FREE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
)
RESIZE = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
)


counters = Counters()
counters.allocs = 0
counters.frees = 0
counters.bytes = 0


@ALLOC
def _alloc(_ud, size, _alignment):
    counters.allocs += 1
    counters.bytes += size
    # ctypes.create_string_buffer gives us a C-malloc'd block via the
    # CPython API; the pointer we return to the C shim is the address
    # of that block. We store the buffer in a list to keep it alive
    # (otherwise the GC may free the bytes before our free() callback
    # is called, which would crash).
    buf = ctypes.create_string_buffer(size)
    _alloc.buffers.append(buf)
    return ctypes.cast(buf, ctypes.c_void_p).value


_alloc.buffers = []


@FREE
def _free(_ud, _ptr, _size, _alignment):
    counters.frees += 1


@RESIZE
def _resize(_ud, _ptr, _old, _new, _alignment):
    return -1  # not supported


def main():
    vtable = kt_kernel.kt_allocator_vtable()
    vtable.userdata = ctypes.cast(ctypes.pointer(counters), ctypes.c_void_p).value
    vtable.alloc = _alloc
    vtable.free = _free
    vtable.resize = _resize

    kt_kernel.kt_set_default_allocator(ctypes.pointer(vtable))

    pool = kt_kernel.kt_worker_pool_new(0)
    kt_kernel.kt_worker_pool_free(pool)

    assert counters.allocs > 0, "tracker should have been called"
    assert counters.allocs == counters.frees, (
        f"allocs ({counters.allocs}) != frees ({counters.frees})"
    )
    assert counters.bytes > 0, "no bytes allocated"

    # Restore the default (process) allocator. After this, the
    # counter MUST stay frozen across another worker_pool round-trip.
    frozen = (counters.allocs, counters.frees, counters.bytes)
    kt_kernel.kt_set_default_allocator(None)

    pool2 = kt_kernel.kt_worker_pool_new(0)
    kt_kernel.kt_worker_pool_free(pool2)
    after = (counters.allocs, counters.frees, counters.bytes)
    assert frozen == after, (
        f"tracker was still active after kt_set_default_allocator(None): {frozen} -> {after}"
    )

    print(
        f"OK: tracker allocs={counters.allocs} frees={counters.frees} bytes={counters.bytes}"
    )


if __name__ == "__main__":
    main()
