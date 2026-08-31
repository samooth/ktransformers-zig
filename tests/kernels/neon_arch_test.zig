// Standalone test for src/kernels/arch/neon.zig feature-detection flags.
//
// The full aarch64 cross-build verification (linking the whole library for
// aarch64 and exporting all 68 C API symbols) is the authoritative
// smoke test, exercised via `zig build -Dvariant=neon -Dtarget=aarch64-linux-gnu`
// and confirmed by tools/verify_abi.py. This suite only covers the comptime
// flags here, which is the part that runs in CI (native x86_64 test runs).
//
// Mirrors tests/kernels/aarch64_detect_test.zig in spirit — small,
// focused on the new module, wired in build.zig as its own suite so
// tests/kernels/test_kernels.zig is untouched (per the file-lock
// convention during parallel dev work).

const std = @import("std");
const testing = std.testing;
const neon = @import("arch_neon");

test "NeonFeatures.available is false on x86_64 target" {
    // This test runs natively on the dev host (x86_64), so the comptime
    // switch in neon.zig must take the `else => false` branch for every
    // flag. If the arch switch is dropped or the aarch64 arm accidentally
    // becomes unconditional, this test catches it.
    try testing.expect(!neon.NeonFeatures.available);
    try testing.expect(!neon.SveFeatures.available);
    try testing.expect(!neon.Sve2Features.available);
}
