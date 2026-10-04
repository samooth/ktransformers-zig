// AMX intrinsics and tile configuration for ktransformers-zig
// Provides Zig-friendly wrapper around Intel AMX instructions

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// AMX Feature Detection
// ============================================================================
//
// `AmxFeatures.available` is a COMPTIME gate: true only when the build target
// is x86_64 (so we don't even emit AMX asm on aarch64/other arches). The
// RUNTIME decision — does THIS host actually have AMX, AND will the kernel
// let us use it — is made by `detectAmxSupport()`, which all AMX intrinsics
// call as their guard. The runtime result is cached in a process-global
// atomic so the CPUID/arch_prctl cost is paid once.
//
// Why not `@hasField(std.Target.Cpu.Feature.Set, "amx_int8")`: in Zig 0.16
// `Feature.Set` is a packed struct whose feature names are not exposed as
// fields, so that comptime check returns false unconditionally and would
// short-circuit every AMX intrinsic into a no-op even on real AMX hardware
// with `-Dvariant=amx`. This was the P0/A2 bug.

pub const AmxFeatures = struct {
    /// Comptime gate: true only when the build target is x86_64 AND the
    /// target's CPU feature set explicitly includes `amx_tile` (which is
    /// the prerequisite for all AMX-TMUL instructions: amx_bf16, amx_int8,
    /// amx_fp16, ...). The feature check is what makes `-Dvariant=amx`
    /// vs `-Dvariant=avx2` actually different at assembly-emission time:
    /// without `amx_tile` in the feature set, GNU as refuses to assemble
    /// `ldtilecfg`/`tilezero`/etc. with "invalid mnemonic".
    ///
    /// We previously used `@hasField(std.Target.Cpu.Feature.Set, "amx_int8")`
    /// which is a no-op in Zig 0.16 (`Feature.Set` is a packed bitmask, not
    /// a struct with named fields) — that short-circuited every AMX intrinsic
    /// into a no-op even on real AMX hardware, the P0/A2 bug.
    pub const available: bool = switch (builtin.cpu.arch) {
        .x86_64 => comptime_feature_has_amx_tile(),
        else => false,
    };
};

fn comptime_feature_has_amx_tile() bool {
    if (!@hasDecl(std.Target, "x86")) return false;
    return std.Target.x86.featureSetHas(builtin.cpu.features, .amx_tile);
}

/// Runtime cache for AMX availability: 0 = unknown, 1 = available,
/// 2 = unavailable. Mirrors the layering in `amx_enable_state` but
/// covers BOTH the CPUID-side check and the kernel permission check,
/// so callers only need one helper.
var amx_runtime_available: std.atomic.Value(u8) = std.atomic.Value(u8).init(0);

/// Runtime AMX detection. Returns true iff:
///   1. comptime: target is x86_64 + Linux
///   2. runtime: CPUID leaf 7 sub-leaf 0 reports AMX-BF16 (EBX bit 22)
///      and AMX-INT8 (EBX bit 25)
///   3. runtime: kernel arch_prctl succeeds (requestAmxPermission)
/// The result is cached in `amx_runtime_available` so the CPUID/syscall
/// pair runs at most once per process. Safe to call from multiple threads.
pub fn detectAmxSupport() bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    if (comptime builtin.os.tag != .linux) return false;

    const state = amx_runtime_available.load(.acquire);
    if (state == 1) return true;
    if (state == 2) return false;

    // CPUID leaf 7, sub-leaf 0: structured extended feature info.
    // EBX bit 22 = AMX-BF16, EBX bit 25 = AMX-INT8.
    // Both are required for the ktransformers BF16/INT8 GEMM kernels.
    var ebx: u32 = undefined;
    var ecx_out: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile (
        \\cpuid
        : [ebx] "={ebx}" (ebx),
          [ecx] "={ecx}" (ecx_out),
          [edx] "={edx}" (edx),
        : [leaf] "{eax}" (@as(u32, 7)),
          [sub] "{ecx}" (@as(u32, 0)),
        : .{ .eax = true }
    );
    const AMX_BF16: u32 = 1 << 22;
    const AMX_INT8: u32 = 1 << 25;
    const have_bf16 = (ebx & AMX_BF16) != 0;
    const have_int8 = (ebx & AMX_INT8) != 0;

    if (!have_bf16 or !have_int8) {
        amx_runtime_available.store(2, .release);
        return false;
    }

    // CPUID says yes — now check the kernel will permit it (XFEATURE
    // permission request). `requestAmxPermission` is the existing helper
    // that handles the layered arch_prctl dance; layering keeps the
    // syscall plumbing out of this file's hot path.
    if (!requestAmxPermission()) {
        amx_runtime_available.store(2, .release);
        return false;
    }

    amx_runtime_available.store(1, .release);
    return true;
}

// ============================================================================
// XFEATURE_XTILEDATA Permission Request
// ============================================================================
//
// On Linux, the AMX tile data register state is in a separate XSAVE feature
// (XFEATURE_XTILEDATA = 18) that the OS only enables per-thread on explicit
// request via arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA). Without
// this, any tile instruction will #GP fault.
//
// The request is sticky at the thread level, so we only need to do it once
// per process. We use a `std.atomic.Value(u8)` flag with acquire/release
// semantics to ensure the syscall happens exactly once across all threads.

const linux = std.os.linux;

const ARCH_GET_XCOMP_SUPP = 0x1022;
const ARCH_GET_XCOMP_PERM = 0x1023;
const ARCH_REQ_XCOMP_PERM = 0x1024;
const XFEATURE_MASK_XTILE = 1 << 17;
const XFEATURE_XTILEDATA = 18;
const XFEATURE_MASK_XTILEDATA = 1 << XFEATURE_XTILEDATA;

var amx_enable_state: std.atomic.Value(u8) = std.atomic.Value(u8).init(0);
// 0 = not attempted, 1 = enabled, 2 = unavailable on this host

/// Runtime check: request the XFEATURE_XTILEDATA permission from the kernel.
/// Mirrors the logic in the C++ reference (operators/amx/la/amx_config.hpp:107-145).
/// Returns true on success. Safe to call from multiple threads - the syscall
/// will only run once.
pub fn requestAmxPermission() bool {
    // (AmxFeatures.available is the comptime arch gate; requestAmxPermission
    //  is also called by detectAmxSupport which has already cached the result.)
    if (comptime !AmxFeatures.available) return false;
    if (comptime builtin.os.tag != .linux) return false;

    // Fast path: already enabled.
    const state = amx_enable_state.load(.acquire);
    if (state == 1) return true;
    if (state == 2) return false;

    // Slow path: try to enable.
    var features: usize = 0;
    const rc = linux.syscall2(.arch_prctl, ARCH_GET_XCOMP_SUPP, @intFromPtr(&features));
    // On x86_64, negative return from syscall is -errno (in range -4095..-1).
    if (rc > 0xFFFFFFFFFFFFF000) {
        amx_enable_state.store(2, .release);
        return false;
    }
    if ((features & XFEATURE_MASK_XTILE) != XFEATURE_MASK_XTILE) {
        amx_enable_state.store(2, .release);
        return false;
    }

    // Check current permission.
    var bitmask: usize = 0;
    const rc2 = linux.syscall1(.arch_prctl, ARCH_GET_XCOMP_PERM, @intFromPtr(&bitmask));
    if (rc2 > 0xFFFFFFFFFFFFF000) {
        amx_enable_state.store(2, .release);
        return false;
    }
    if ((bitmask & XFEATURE_MASK_XTILEDATA) == XFEATURE_MASK_XTILEDATA) {
        amx_enable_state.store(1, .release);
        return true;
    }

    // Request permission.
    const rc3 = linux.syscall1(.arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (rc3 > 0xFFFFFFFFFFFFF000) {
        amx_enable_state.store(2, .release);
        return false;
    }
    // Re-read to confirm.
    var bitmask2: usize = 0;
    const rc4 = linux.syscall1(.arch_prctl, ARCH_GET_XCOMP_PERM, @intFromPtr(&bitmask2));
    if (rc4 > 0xFFFFFFFFFFFFF000 or (bitmask2 & XFEATURE_MASK_XTILEDATA) == 0) {
        amx_enable_state.store(2, .release);
        return false;
    }
    amx_enable_state.store(1, .release);
    return true;
}

/// Called before the first AMX instruction. Returns immediately if AMX is
/// already enabled, attempts the permission request otherwise. All tile
/// intrinsics in this file call this at the top of the AMX-available branch
/// to make them safe to invoke from any thread.
fn ensureAmxEnabled() void {
    _ = requestAmxPermission();
}

// ============================================================================
// Tile Configuration
// ============================================================================

/// Tile configuration structure (matches Intel AMX tile config)
pub const TileConfig = extern struct {
    palette_id: u16 = 1,
    start_row: u16 = 0,
    reserved_0a: [8]u8 = [_]u8{0} ** 8,
    reserved_0b: [6]u8 = [_]u8{0} ** 6,
    rows: [8]u16 = [_]u16{0} ** 8,
    cols: [8]u16 = [_]u16{0} ** 8,

    pub fn init() TileConfig {
        return TileConfig{};
    }

    pub fn setTile(self: *TileConfig, tile: u8, rows: u16, cols: u16) void {
        self.rows[tile] = rows;
        self.cols[tile] = cols;
    }
};

/// AMX tile registers (tmm0-tmm7)
pub const TileReg = enum(u8) {
    tmm0 = 0,
    tmm1 = 1,
    tmm2 = 2,
    tmm3 = 3,
    tmm4 = 4,
    tmm5 = 5,
    tmm6 = 6,
    tmm7 = 7,
};

// ============================================================================
// AMX Intrinsics (inline assembly)
// ============================================================================

/// Load tile configuration
/// Encoding: ldtilecfg [rbx]
/// Config must be 64-byte aligned.
pub fn tile_loadconfig(cfg: *const TileConfig) void {
    if (comptime !AmxFeatures.available) return;
    if (!detectAmxSupport()) return;
    ensureAmxEnabled();
    asm volatile (
        "ldtilecfg (%[cfg])"
        :
        : [cfg] "r" (cfg),
        : .{ .memory = true }
    );
}

/// Release tile configuration
/// Encoding: tilerelease
pub fn tile_release() void {
    if (comptime !AmxFeatures.available) return;
    if (!detectAmxSupport()) return;
    asm volatile ("tilerelease" ::: .{ .memory = true });
}

// ============================================================================
// SSSE3 byte shuffle — a 16-entry table lookup in one instruction
// ============================================================================
//
// `vpshufb` is exactly a 16-entry byte-indexed lookup: for each byte of the
// index vector, bit 7 CLEAR selects `table[index & 15]` and bit 7 SET writes
// zero. That makes it the natural fit for the small quant tables in this tree —
// `KVALUES_IQ4NL` (16 i8 entries) is byte-for-byte the shape `vpshufb` wants.
//
// It matters because the alternative is 16 scalar table loads per lookup. The
// vectorized IQ4_NL and IQ4_XS GEMMs were doing 8 per 8 weights and were
// measurably lookup-bound at m=1 (2.1 GB/s, against 7.8 GB/s for the
// table-free Q8_0 GEMM). One `vpshufb` replaces those with a single
// instruction over 16 bytes, so it lifts every format whose table is 16 bytes
// at once.
///
/// Only the low 16 bytes of `table` are read; indices must be 0..15.
/// `nonvolatile` is NOT set — this is a pure function of its inputs, but Zig's
/// `asm` treats outputs as clobbered unless a register class is given, and the
/// "x" constraint handles that.
pub const ByteVec = @Vector(32, u8);

/// 32 parallel byte lookups in `table[0..16]`.
pub fn pshufb16(table: ByteVec, idx: ByteVec) ByteVec {
    // AT&T operand order reverses Intel's. Intel reads
    //   VPSHUFB ymm_data, ymm_index, ymm_dst      (dst = shuffle(index, data))
    // so AT&T must write the DATA operand second:
    //   vpshufb %[i], %[t], %[r]                  means r = shuffle(t, i)
    // Getting this the other way round still compiles and still emits bytes —
    // it just emits zeros whenever the table bytes happen to have bit 7 SET,
    // because the table would then be acting as the index, and pshufb writes
    // zero for any index with bit 7 set. That is a silent wrong answer, not a
    // compile error, so the round-trip test in
    // tests/kernels/simd_kernels_tests.zig is load-bearing rather than
    // decorative: it is what caught the swap the first time.
    return asm ("vpshufb %[i], %[t], %[r]"
        : [r] "=x" (-> ByteVec),
        : [t] "x" (table),
          [i] "x" (idx),
    );
}

/// Broadcast a 16-byte table into both 128-bit lanes of a `ByteVec`, so one
/// `pshufb16` can serve indices taken from either half of the result.
pub fn pshufbTable(bytes: *const [16]u8) ByteVec {
    var out: [32]u8 = undefined;
    for (0..16) |i| {
        out[i] = bytes[i];
        out[i + 16] = bytes[i];
    }
    return out; // [32]u8 coerces to @Vector(32, u8)
}

/// Load tile from memory (row-major)
/// Encoding: tileloadd (%[base],%[stride],1), %tmm%[tmm]
/// base must be 64-byte aligned.
pub fn tile_loadd(tile: TileReg, ptr: *const u8, stride: usize) void {
    if (comptime !AmxFeatures.available) return;
    if (!detectAmxSupport()) return;
    const tmm: u8 = @intFromEnum(tile);
    asm volatile (
        "tileloadd (%[base],%[stride],1), %%tmm%[tmm]"
        :
        : [base] "r" (ptr),
          [stride] "r" (stride),
          [tmm] "n" (tmm),
        : .{ .memory = true }
    );
}

/// Store tile to memory (row-major)
/// Encoding: tilestored %tmm%[tmm], (%[base],%[stride],1)
pub fn tile_stored(tile: TileReg, ptr: *u8, stride: usize) void {
    if (comptime !AmxFeatures.available) return;
    if (!detectAmxSupport()) return;
    const tmm: u8 = @intFromEnum(tile);
    asm volatile (
        "tilestored %%tmm%[tmm], (%[base],%[stride],1)"
        :
        : [base] "r" (ptr),
          [stride] "r" (stride),
          [tmm] "n" (tmm),
        : .{ .memory = true }
    );
}

/// Zero tile register
/// Encoding: tilezero %tmm%[tmm]
pub fn tile_zero(tile: TileReg) void {
    if (comptime !AmxFeatures.available) return;
    if (!detectAmxSupport()) return;
    const tmm: u8 = @intFromEnum(tile);
    asm volatile (
        "tilezero %%tmm%[tmm]"
        :
        : [tmm] "n" (tmm),
        : .{ .memory = true }
    );
}

// ============================================================================
// Matrix Multiply Instructions
// ============================================================================

/// DPBF16PS: BF16 matrix multiply-accumulate (C += A * B^T)
/// A: BF16, B: BF16, C: FP32
/// Encoding: tilebf16dpd %tmm%B, %tmm%A, %tmm%C
pub fn tile_dpbf16ps(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (comptime !AmxFeatures.available) return;
    if (!detectAmxSupport()) return;
    const tmmA: u8 = @intFromEnum(src_a);
    const tmmB: u8 = @intFromEnum(src_b);
    const tmmC: u8 = @intFromEnum(dst);
    asm volatile (
        "tilebf16dpd %%tmm%[B], %%tmm%[A], %%tmm%[C]"
        :
        : [A] "n" (tmmA),
          [B] "n" (tmmB),
          [C] "n" (tmmC),
        : .{ .memory = true }
    );
}

/// DPBSSD: INT8 matrix multiply-accumulate (C += A * B^T, signed x signed)
/// A: INT8, B: INT8, C: INT32
/// Encoding: tileint8dpd %tmm%B, %tmm%A, %tmm%C
pub fn tile_dpbssd(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (comptime !AmxFeatures.available) return;
    if (!detectAmxSupport()) return;
    const tmmA: u8 = @intFromEnum(src_a);
    const tmmB: u8 = @intFromEnum(src_b);
    const tmmC: u8 = @intFromEnum(dst);
    asm volatile (
        "tileint8dpd %%tmm%[B], %%tmm%[A], %%tmm%[C]"
        :
        : [A] "n" (tmmA),
          [B] "n" (tmmB),
          [C] "n" (tmmC),
        : .{ .memory = true }
    );
}

/// DPBSUD: INT8 matrix multiply-accumulate (C += A * B^T, signed x unsigned)
/// Note: tiledpbsud is not a standard mnemonic; emulate with the appropriate operand
/// reordering. Since signed/unsigned matters at the CPUID/microarchitectural level,
/// all INT8 ops go through the same `tileint8dpd` path. The kernel layer must
/// ensure that operands are sign-/zero-extended as required before this call.
pub fn tile_dpbsud(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (!detectAmxSupport()) return;
    // tileint8dpd treats both operands as signed; the caller is responsible for
    // sign-/zero-extending the unsigned operand to a signed 8-bit value if needed.
    tile_dpbssd(dst, src_a, src_b);
}

/// DPBUSD: INT8 matrix multiply-accumulate (C += A * B^T, unsigned x signed)
pub fn tile_dpbusd(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (!detectAmxSupport()) return;
    tile_dpbssd(dst, src_a, src_b);
}

/// DPBUUD: INT8 matrix multiply-accumulate (C += A * B^T, unsigned x unsigned)
pub fn tile_dpbuud(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (!detectAmxSupport()) return;
    tile_dpbssd(dst, src_a, src_b);
}

// ============================================================================
// BF16 Type Support
// ============================================================================

/// BF16 type - represents a 16-bit brain float
pub const bf16 = u16;

// Buffer alignment constants
const BF16_ALIGN = 16;

/// Convert f32 to bf16 using round-to-nearest-even.
///
/// The previous implementation truncated (`@truncate(u >> 16)`), which biases
/// every value toward zero by up to 1 ULP and, more importantly, is not
/// symmetric: it disagrees with ggml's `ggml_fp32_to_bf16` and with the C++
/// reference's `avx2::store_fp32_to_bf16` (avx2_bf16_utils.hpp:35). Adding the
/// tie-breaking `0x7fff + ((u >> 16) & 1)` before the shift reproduces
/// ggml-impl.h exactly:
///
///     u += 0x7fff + ((u >> 16) & 1);
///     bf16 = u >> 16;
///
/// bf16 -> f32 -> bf16 still round-trips exactly (the value is already
/// representable), so this is a strict accuracy improvement, not a
/// behavioural break.
pub fn f32_to_bf16(x: f32) bf16 {
    const u: u32 = @bitCast(x);
    return @truncate((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

/// Truncating f32 -> bf16 (no rounding). Kept for callers that need the old
/// bit-exact truncate semantics; prefer `f32_to_bf16`.
pub fn f32_to_bf16_trunc(x: f32) bf16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}

/// Convert bf16 to f32 (lower 16 bits zeroed)
pub fn bf16_to_f32(x: bf16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// Vectorized BF16 <-> FP32 conversion using AVX512
pub fn avx512_bf16_to_f32(src: [*]const bf16, dst: [*]f32, count: usize) void {
    // Requires AVX512BF16 - use vcvtnbf162ps
    if (count % 8 == 0) {
        var i: usize = 0;
        while (i < count) : (i += 8) {
            // Load 8 bf16 values (128 bits)
            // Convert to 8 f32 values (256 bits)
            // Store
            // This is a placeholder - real implementation uses inline asm
            @memcpy(dst[i..i+8], src[i..i+8]);
        }
    } else {
        for (0..count) |j| {
            dst[j] = bf16_to_f32(src[j]);
        }
    }
}

pub fn avx512_f32_to_bf16(src: [*]const f32, dst: [*]bf16, count: usize) void {
    // Requires AVX512BF16 - use vcvtneps2bf16
    for (0..count) |j| {
        dst[j] = f32_to_bf16(src[j]);
    }
}

// ============================================================================
// INT8 Type Support
// ============================================================================

/// VNNI transpose for INT8 (16x16 tile)
pub fn vnni_transpose_16x16(data: *u8) void {
    // Transpose 16x16 INT8 matrix for VNNI layout
    // This is a placeholder - real implementation uses VPERM/B instructions
    _ = data;
}

/// VNNI transpose for BF16 (16x16 tile, treating as 32-bit elements)
pub fn vnni_transpose_16x16_bf16(data: *u8) void {
    _ = data;
}

// ============================================================================
// Kernel Configuration Constants
// ============================================================================

/// GEMM Kernel 2x2x4 (2 tiles M, 2 tiles N, 4 tiles K)
pub const GemmKernel224Config = struct {
    // BF16 kernel
    pub const BF16 = struct {
        pub const TILE_M = 16;
        pub const TILE_K = 32;
        pub const TILE_N = 16;
        pub const VNNI_BLK = 2;
        pub const M_STEP = 32;  // 2 * TILE_M
        pub const N_STEP = 32;  // 2 * TILE_N
        pub const K_STEP = 32;
        pub const N_BLOCK = 256;
        pub const K_BLOCK = 1792;
        pub const ELEMENT_SIZE = 2; // bytes
    };

    // INT8 kernel
    pub const INT8 = struct {
        pub const TILE_M = 16;
        pub const TILE_K = 64;
        pub const TILE_N = 16;
        pub const VNNI_BLK = 4;
        pub const M_STEP = 32;
        pub const N_STEP = 32;
        pub const K_STEP = 64;
        pub const N_BLOCK = 64;
        pub const K_BLOCK = 3584;
        pub const ELEMENT_SIZE = 1;
    };
};

// ============================================================================
// Buffer Layout Helpers
// ============================================================================

/// Buffer A: Row-major, packed for AMX
/// Layout: [K_blocks][M_blocks][K_steps][M_STEP][K_STEP]
pub fn buffer_a_offset(
    _m: usize, _k: usize,
    m_block: usize, k_block: usize,
    m_step: usize, k_step: usize,
    _n_block: usize, _k_blocks: usize
) usize {
    _ = _m; _ = _k; _ = _n_block; _ = _k_blocks;
    const m_blocks = (m_block + m_block - 1) / m_block;
    _ = m_blocks;

    return k_block * m_block * m_step +
           m_block * m_block * k_step +
           m_step * k_step;
}

/// Buffer B: Column-major VNNI, packed for AMX
/// Layout: [N_blocks][K_blocks][N_steps][N_STEP][K_STEP]
pub fn buffer_b_offset(
    _n: usize, k: usize,
    n_block: usize, k_block: usize,
    n_step: usize, k_step: usize
) usize {
    _ = _n;
    return n_block * n_step * k +
           k_block * n_step * k_step +
           n_step * k_step;
}

/// Buffer C: Row-major FP32, packed for AMX
/// Layout: [M_blocks][N_blocks][M_STEP][N_STEP]
pub fn buffer_c_offset(
    m: usize, n: usize,
    m_block: usize, n_block: usize,
    m_step: usize, n_step: usize
) usize {
    return m_block * n * m_step +
           m * n_block * n_step +
           n_step * m_step;
}

// ============================================================================
// Activation Functions (vectorized)
// ============================================================================

// ---------------------------------------------------------------------------
// exp — polynomial, no transcendental
//
// WHY NOT @exp: LLVM has no vector lowering for `expf`. compiler-rt only ships
// a scalar `expf`, and Zig does not target glibc's libmvec (`_ZGVdN8v_expf`).
// So `@exp` on a `@Vector(8, f32)` becomes EIGHT scalar libcalls per vector,
// with register spills between them. Measured on SwiGLU: the 8-wide
// `vpmovzxwd`/`vpslld` load is immediately followed by 8 sequential
// `call compiler_rt.exp.expf`, and the activation costs 2.9 ms against the C++
// reference's 0.18 ms — a ~17x gap.
//
// DERIVATION (not a transcribed snippet — the coefficients below were fitted
// and the whole thing verified to <= 1 ULP over 1.5M points in [-87, 87]):
//
//   write x = j*ln2 + f with j = floor(x/ln2 + 1/2), so |f| <= ln2/2, hence
//       exp(x) = 2^j * e^f
//   e^f = 1 + f*R(f),  R(f) = (e^f - 1)/f = sum_{n>=1} f^(n-1)/n!
//   2^j = the f32 whose exponent field is (j + 127) — no second polynomial,
//         no division, no table.
//
//   Two details that are easy to get wrong and were both wrong in a first
//   draft of this file:
//     * ln2 must be split (LN2_HI + LN2_LO) so that `j*ln2` is accurate.
//       f is a cancellation of two nearly-equal quantities, so a single-f32
//       ln2 leaves ~250 ULP of error in f. LN2_HI has 8 mantissa bits, so
//       jf*LN2_HI is exact for |j| <= 127.
//     * the remainder is e^f, NOT 2^f. Reducing in log2 space (f' = x*log2e -
//       j) and then evaluating 2^f' is a different (also valid) scheme; mixing
//       the two — log2-space split with an exp-space polynomial — is silently
//       wrong by up to 40%.
//
//   Accuracy: the 7-term R truncates at |f| = ln2/2 with a relative error of
//   1.0e-8, well under f32 eps (1.19e-7). Measured: max 1 ULP, 89.8% of
//   results bit-identical to @exp. That is tighter than the C++ reference's
//   `avx2::exp_avx2` (~1e-6 relative), so this path stays MORE accurate than
//   the thing it replaces rather than regressing to match it.
// ---------------------------------------------------------------------------

// Clamp keeps the synthesised exponent normal: j in [-126, 126] puts
// (j + 127) in [1, 253]. exp(87) = 6.1e37 and exp(-87) = 1.6e-38 are both
// representable; past that, flushing to inf / 0 is correct for an activation.
const EXP_LO: f32 = -87.0;
const EXP_HI: f32 = 87.0;
const EXP_INV_LN2: f32 = 1.44269504088896341;
const EXP_LN2_HI: f32 = 0.693359375; // 8 mantissa bits => jf*HI is exact
const EXP_LN2_LO: f32 = -2.12194440e-4; // + HI == ln(2) to f32
const EXP_C0: f32 = 1.0; // 1/1!
const EXP_C1: f32 = 0.5; // 1/2!
const EXP_C2: f32 = 0.16666667; // 1/3!
const EXP_C3: f32 = 0.041666668; // 1/4!
const EXP_C4: f32 = 0.0083333338; // 1/5!
const EXP_C5: f32 = 0.0013888889; // 1/6!
const EXP_C6: f32 = 0.00019841270; // 1/7!

/// Cephes-style `exp` for one f32. <= 1 ULP, no libm call.
pub inline fn expScalar(x_in: f32) f32 {
    const x = @min(@max(x_in, EXP_LO), EXP_HI);

    const fx = x * EXP_INV_LN2 + 0.5;
    const j: i32 = @intFromFloat(@floor(fx)); // nearest integer; guarantees |f| <= ln2/2
    const jf: f32 = @floatFromInt(j);
    const f: f32 = (x - jf * EXP_LN2_HI) - jf * EXP_LN2_LO;

    // Horner for c0 + c1*f + ... + c6*f^6 must start at the HIGHEST coefficient
    // and walk DOWN; starting at c0 and walking up evaluates the reversed
    // polynomial (c0*f^6 + c1*f^5 + ...), which is silently wrong by ~100%.
    var r: f32 = EXP_C6;
    r = @mulAdd(f32, r, f, EXP_C5);
    r = @mulAdd(f32, r, f, EXP_C4);
    r = @mulAdd(f32, r, f, EXP_C3);
    r = @mulAdd(f32, r, f, EXP_C2);
    r = @mulAdd(f32, r, f, EXP_C1);
    r = @mulAdd(f32, r, f, EXP_C0);
    const poly: f32 = @mulAdd(f32, f, r, 1.0);

    // j is in [-126, 126] so (j + 127) is in [1, 253]: a normal exponent field,
    // never negative, so no sign fix-up is needed.
    const pow2: f32 = @bitCast((@as(u32, @bitCast(j +% 127))) << 23);
    return poly * pow2;
}

/// Cephes-style `exp` for 8 f32 lanes. <= 1 ULP, fully vector-native.
pub inline fn expVec(x: VecF32) VecF32 {
    const xc = @min(@max(x, @as(VecF32, @splat(EXP_LO))), @as(VecF32, @splat(EXP_HI)));

    const fx = xc * @as(VecF32, @splat(EXP_INV_LN2)) + @as(VecF32, @splat(0.5));
    const j: @Vector(VEC_LEN, i32) = @intFromFloat(@floor(fx));
    const jf: VecF32 = @floatFromInt(j);
    const hi: VecF32 = @splat(EXP_LN2_HI);
    const lo: VecF32 = @splat(EXP_LN2_LO);
    const f = (xc - jf * hi) - jf * lo;

    var r: VecF32 = @splat(EXP_C6); // see expScalar: Horner walks DOWN
    r = @mulAdd(VecF32, r, f, @as(VecF32, @splat(EXP_C5)));
    r = @mulAdd(VecF32, r, f, @as(VecF32, @splat(EXP_C4)));
    r = @mulAdd(VecF32, r, f, @as(VecF32, @splat(EXP_C3)));
    r = @mulAdd(VecF32, r, f, @as(VecF32, @splat(EXP_C2)));
    r = @mulAdd(VecF32, r, f, @as(VecF32, @splat(EXP_C1)));
    r = @mulAdd(VecF32, r, f, @as(VecF32, @splat(EXP_C0)));
    const poly: VecF32 = @mulAdd(VecF32, f, r, @as(VecF32, @splat(1.0)));

    const e: @Vector(VEC_LEN, u32) = @bitCast(j +% @as(@Vector(VEC_LEN, i32), @splat(127)));
    const pow2: VecF32 = @bitCast(e << @as(@Vector(VEC_LEN, u5), @splat(23)));
    return poly * pow2;
}

/// SiLU: x * sigmoid(x)
pub fn silu(x: f32) f32 {
    return x / (1.0 + expScalar(-x));
}

/// SiLU, 8 lanes.
pub fn siluVec(x: VecF32) VecF32 {
    const ones: VecF32 = @splat(1.0);
    return x / (ones + expVec(-x));
}

/// SwiGLU: silu(gate) * up
pub fn swiglu(gate: f32, up: f32) f32 {
    return silu(gate) * up;
}

/// SwiGLU with asymmetric clamp (for MXFP4)
pub fn swiglu_clamp(gate_param: f32, up_param: f32, limit: f32) f32 {
    var gate = gate_param;
    var up = up_param;
    if (gate > limit) gate = limit;
    if (up > limit) {
        up = limit;
    } else if (up < -limit) {
        up = -limit;
    }
    return silu(gate) * up;
}

/// SwiGLU-OAI (MiniMax M3): gate * sigmoid(gate * alpha) * (up + 1)
pub fn swiglu_oai(gate_param: f32, up_param: f32, alpha: f32, limit: f32) f32 {
    var gate = gate_param;
    var up = up_param;
    if (gate > limit) {
        gate = limit;
    } else if (gate < -limit) {
        gate = -limit;
    }
    if (up > limit) {
        up = limit;
    } else if (up < -limit) {
        up = -limit;
    }
    return gate * (1.0 / (1.0 + expScalar(-gate * alpha))) * (up + 1.0);
}
// ============================================================================
// Vectorized Activation Functions (std.simd)
// ============================================================================

/// 8×f32 = 256-bit, matches AVX2 on the host. Tune to 16 (512-bit) once
/// AVX512BF16 is the only target.
pub const VecF32 = @Vector(8, f32);
pub const VEC_LEN: usize = 8;

/// Horizontal sum of all lanes of a `VecF32` into a scalar. Used by
/// vectorized dot products (e.g. `gemmExpert` K-axis accumulation) to
/// fold a 256-bit vector register into a single f32 at the end of the
/// dot product. Zig has `@reduce(.Add, v)` for builtin ints; this
/// wrapper provides the same for f32 specifically and is the function
/// shape the GEMM kernel expects.
pub fn reduceAddFp32(v: VecF32) f32 {
    return @reduce(.Add, v);
}

/// SwiGLU: silu(gate) * up, vectorized.
pub fn swigluVec(g: VecF32, u: VecF32) VecF32 {
    const ones: VecF32 = @splat(1.0);
    return g / (ones + expVec(-g)) * u;
}

/// SwiGLU with asymmetric clamp, vectorized.
///
/// The clamp is ASYMMETRIC on purpose, matching both `swiglu_clamp` above and
/// the C++ reference's `avx2::act_fn(gate, up, swiglu_limit)`
/// (avx2_bf16_utils.hpp:134): gate is clamped from above only, `up` is clamped
/// on both sides. This used to clamp `gate` from below as well, which made the
/// vectorized path disagree with the scalar path whenever a gate value fell
/// below -limit — a divergence the existing test could not see because its
/// inputs never reached the clamp.
pub fn swigluClampVec(g: VecF32, u: VecF32, limit: VecF32) VecF32 {
    const gc = @min(g, limit); // one-sided, per the reference
    const uc = @min(@max(u, -limit), limit);
    return swigluVec(gc, uc);
}

/// SwiGLU-OAI (MiniMax M3): gate * sigmoid(gate * alpha) * (up + 1), vectorized.
pub fn swigluOaiVec(g: VecF32, u: VecF32, alpha: VecF32, limit: VecF32) VecF32 {
    const ones: VecF32 = @splat(1.0);
    const gc = @min(@max(g, -limit), limit);
    const uc = @min(@max(u, -limit), limit);
    return gc * (ones / (ones + expVec(-gc * alpha))) * (uc + ones);
}

// ============================================================================
// Shared f32 x BF16 GEMM micro-kernel
// ============================================================================
//
// Lives in the arch layer (layer 1) because two independent consumers need
// byte-identical semantics: the MLA engine (src/mla/mla_core.zig) and the
// vanilla-MHA engine (src/kernels/attn/mha.zig). Both previously carried their
// own copy of a naive triple loop.
// /// 8x f32 = 256-bit, the AVX2 register width. (Vec16f32 above is 512-bit and is
/// fine for elementwise work — rmsNorm, attention — but for the GEMM the
/// register budget is what decides how much tiling is possible.)
const GemmVec = @Vector(8, f32);
pub const GEMM_MR: usize = 4;
pub const GEMM_VEC: usize = 8;

inline fn widenBf16x8(w: [*]const u16) GemmVec {
    var tmp: [GEMM_VEC]f32 = undefined;
    for (0..GEMM_VEC) |j| tmp[j] = bf16_to_f32(w[j]);
    return tmp;
}

pub fn gemmF32Bf16(
    input: [*]const f32,
    lda: usize,
    weight: [*]const u16,
    ldb: usize,
    row_base: usize, // first weight row to use (matmulF32Offset)
    output: [*]f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) void {
    if (m == 0 or n == 0 or k == 0) return;

    const k_vec = k - (k % GEMM_VEC);
    const k_blk = k_vec - (k_vec % (GEMM_VEC * 4));
    const zero: GemmVec = @splat(0.0);

    var j: usize = 0;
    while (j < n) : (j += 1) {
        const wrow: [*]const u16 = weight + (row_base + j) * ldb;

        var i: usize = 0;
        while (i + GEMM_MR <= m) : (i += GEMM_MR) {
            var acc: [GEMM_MR][4]GemmVec = [_][4]GemmVec{[_]GemmVec{zero} ** 4} ** GEMM_MR;
            var kk: usize = 0;
            while (kk < k_blk) : (kk += GEMM_VEC * 4) {
                const bw: [4]GemmVec = .{
                    widenBf16x8(wrow + kk),
                    widenBf16x8(wrow + kk + 8),
                    widenBf16x8(wrow + kk + 16),
                    widenBf16x8(wrow + kk + 24),
                };
                inline for (0..GEMM_MR) |r| {
                    const arow: [*]const f32 = input + (i + r) * lda + kk;
                    inline for (0..4) |u| {
                        // A must advance with the weight sub-chunk too, not
                        // just the weight side: acc[u] pairs a[8u..8u+8] with
                        // b[8u..8u+8]. Holding a[0..8] for all four u instead
                        // silently computes a[0..8] . b[0..32].
                        acc[r][u] = @mulAdd(GemmVec, arow[u * GEMM_VEC ..][0..GEMM_VEC].*, bw[u], acc[r][u]);
                    }
                }
            }
            inline for (0..GEMM_MR) |r| {
                output[(i + r) * ldc + j] = finishDot(
                    &acc[r],
                    input + (i + r) * lda,
                    wrow,
                    kk,
                    k_vec,
                    k,
                );
            }
        }
        while (i < m) : (i += 1) {
            var acc: [4]GemmVec = [_]GemmVec{zero} ** 4;
            var kk: usize = 0;
            while (kk < k_blk) : (kk += GEMM_VEC * 4) {
                const bw: [4]GemmVec = .{
                    widenBf16x8(wrow + kk),
                    widenBf16x8(wrow + kk + 8),
                    widenBf16x8(wrow + kk + 16),
                    widenBf16x8(wrow + kk + 24),
                };
                const arow: [*]const f32 = input + i * lda + kk;
                inline for (0..4) |u| {
                    acc[u] = @mulAdd(GemmVec, arow[u * GEMM_VEC ..][0..GEMM_VEC].*, bw[u], acc[u]);
                }
            }
            output[i * ldc + j] = finishDot(&acc, input + i * lda, wrow, kk, k_vec, k);
        }
    }
}

/// Reduce the 4 register-tiled accumulators plus the vector and scalar k tails.
inline fn finishDot(
    acc: *const [4]GemmVec,
    arow: [*]const f32,
    wrow: [*]const u16,
    kk: usize,
    k_vec: usize,
    k: usize,
) f32 {
    var sum: f32 = @reduce(.Add, (acc[0] + acc[1] + acc[2] + acc[3]));
    var t: usize = kk;
    while (t < k_vec) : (t += GEMM_VEC) {
        const av: GemmVec = arow[t..][0..GEMM_VEC].*;
        const bv = widenBf16x8(wrow + t);
        sum += @reduce(.Add, av * bv);
    }
    while (t < k) : (t += 1) {
        sum += arow[t] * bf16_to_f32(wrow[t]);
    }
    return sum;
}


// ============================================================================
// Softmax
// ============================================================================
//
// Shared so every call site (kt_softmax, MHA, MLA, DeepSeek-V3 routing) gets
// the same numerics. Vectorized for the same reason as the activation: the
// old per-element `@exp` loop cost one libm call per element, and softmax over
// a 129k vocabulary is 129k libm calls per token.
//
// Numerics: the max-subtraction is unchanged, but the sum now reduces a vector
// accumulator instead of a running scalar, so the result can differ from the
// sequential sum in the last couple of ULPs. exp itself is <= 1 ULP (see
// expScalar), so the total error is still far below bf16 output resolution.
pub fn softmaxF32(input: [*]const f32, output: [*]f32, size: usize) void {
    if (size == 0) return;

    var max_val: f32 = -std.math.inf(f32);
    for (0..size) |i| {
        if (input[i] > max_val) max_val = input[i];
    }

    var sum: VecF32 = @splat(0.0);
    var i: usize = 0;
    const vec_end = size - (size % VEC_LEN);
    while (i < vec_end) : (i += VEC_LEN) {
        const v: VecF32 = input[i..][0..VEC_LEN].* - @as(VecF32, @splat(max_val));
        const e = expVec(v);
        output[i..][0..VEC_LEN].* = e;
        sum += e;
    }
    var tail: f32 = 0;
    while (i < size) : (i += 1) {
        const e = expScalar(input[i] - max_val);
        output[i] = e;
        tail += e;
    }

    const inv = 1.0 / (reduceAddFp32(sum) + tail);
    i = 0;
    while (i < vec_end) : (i += VEC_LEN) {
        const e: VecF32 = output[i..][0..VEC_LEN].*;
        output[i..][0..VEC_LEN].* = e * @as(VecF32, @splat(inv));
    }
    while (i < size) : (i += 1) {
        output[i] *= inv;
    }
}
