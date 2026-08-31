// ARM NEON / SVE feature detection for ktransformers-zig
//
// Mirrors src/kernels/arch/amx.zig's structure (a compile-time feature flag
// + per-host stubs) so that future NEON/SVE kernels can gate on
// `NeonFeatures.available` / `SveFeatures.available` the same way the AMX
// GEMMs gate on `AmxFeatures.available`.
//
// ARMv8 (aarch64) baseline: NEON (ASIMD). NEON is mandatory on all
// aarch64 cores — there is no aarch64 host without NEON. The comptime
// branch below simply checks that the build target is aarch64.
//
// SVE is optional (ARMv8.2-A, configurable vector length 128-2048 bits).
// SVE2 is ARMv8.4-A. Both are queried at runtime via
// getauxval(AT_HWCAP/AT_HWCAP2) on Linux, but for the Phase 1 cross-
// compile gate we use the comptime arch check (always-available is the
// same decision kernel authors need at compile time).
//
// Phase 1 scope: this file declares the feature flags and the cross-build
// metadata only — the actual NEON GEMM kernels (analogous to
// gemm_224_*.zig on x86) are Phase 2 work. The scalar GEMM fallbacks
// run on aarch64 today.

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Feature detection
// ============================================================================

/// NEON (ASIMD) is the baseline SIMD ISA on all ARMv8 (aarch64) cores —
/// there is no aarch64 hardware without it. The flag exists so that
/// future NEON-specific code can gate on `if (NeonFeatures.available)`
/// without repeating the arch check; the comptime `true` on aarch64 also
/// means the `else` branch of such `if`s is dead-pruned (no per-instruction
/// runtime cost on the target hardware).
pub const NeonFeatures = struct {
    pub const available: bool = switch (builtin.cpu.arch) {
        .aarch64 => true,
        else => false,
    };
};

/// SVE is optional on ARMv8.2-A+. The comptime check below only verifies
/// the target arch — runtime SVE width detection (VL) and HWCAP lookup
/// land in Phase 2 alongside the first SVE kernel. The SVE kernel would
/// be the right place to query VL via `cntb` at runtime.
pub const SveFeatures = struct {
    pub const available: bool = switch (builtin.cpu.arch) {
        .aarch64 => true, // assume available on aarch64 build target; runtime
        // confirmation would require HWCAP for ARMv8.2
        // compliance, deferred to Phase 2.
        else => false,
    };
};

/// SVE2 (ARMv8.4-A). Same caveat as SveFeatures: target-arch check only.
pub const Sve2Features = struct {
    pub const available: bool = switch (builtin.cpu.arch) {
        .aarch64 => true,
        else => false,
    };
};
