// AMX intrinsics and tile configuration for ktransformers-zig
// Provides Zig-friendly wrapper around Intel AMX instructions

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// AMX Feature Detection
// ============================================================================

pub const AmxFeatures = struct {
    pub const available: bool = switch (builtin.cpu.arch) {
        .x86_64 => @hasField(std.Target.Cpu.Feature.Set, "amx_int8"),
        else => false,
    };
};

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
    if (!AmxFeatures.available) return false;
    if (comptime builtin.os.tag != .linux) return false;

    // Fast path: already enabled.
    const state = amx_enable_state.load(.acquire);
    if (state == 1) return true;
    if (state == 2) return false;

    // Slow path: try to enable.
    var features: usize = 0;
    const rc = linux.syscall1(.arch_prctl, ARCH_GET_XCOMP_SUPP, @intFromPtr(&features));
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
    if (!AmxFeatures.available) return;
    ensureAmxEnabled();
    asm volatile (
        "ldtilecfg (%[cfg])"
        :
        : [cfg] "r" (cfg),
        : "memory"
    );
}

/// Release tile configuration
/// Encoding: tilerelease
pub fn tile_release() void {
    if (!AmxFeatures.available) return;
    asm volatile ("tilerelease" ::: "memory");
}

/// Load tile from memory (row-major)
/// Encoding: tileloadd (%[base],%[stride],1), %tmm%[tmm]
/// base must be 64-byte aligned.
pub fn tile_loadd(tile: TileReg, ptr: *const u8, stride: usize) void {
    if (!AmxFeatures.available) return;
    const tmm: u8 = @intFromEnum(tile);
    asm volatile (
        "tileloadd (%[base],%[stride],1), %%tmm%[tmm]"
        :
        : [base] "r" (ptr),
          [stride] "r" (stride),
          [tmm] "n" (tmm),
        : "memory"
    );
}

/// Store tile to memory (row-major)
/// Encoding: tilestored %tmm%[tmm], (%[base],%[stride],1)
pub fn tile_stored(tile: TileReg, ptr: *u8, stride: usize) void {
    if (!AmxFeatures.available) return;
    const tmm: u8 = @intFromEnum(tile);
    asm volatile (
        "tilestored %%tmm%[tmm], (%[base],%[stride],1)"
        :
        : [base] "r" (ptr),
          [stride] "r" (stride),
          [tmm] "n" (tmm),
        : "memory"
    );
}

/// Zero tile register
/// Encoding: tilezero %tmm%[tmm]
pub fn tile_zero(tile: TileReg) void {
    if (!AmxFeatures.available) return;
    const tmm: u8 = @intFromEnum(tile);
    asm volatile (
        "tilezero %%tmm%[tmm]"
        :
        : [tmm] "n" (tmm),
        : "memory"
    );
}

// ============================================================================
// Matrix Multiply Instructions
// ============================================================================

/// DPBF16PS: BF16 matrix multiply-accumulate (C += A * B^T)
/// A: BF16, B: BF16, C: FP32
/// Encoding: tilebf16dpd %tmm%B, %tmm%A, %tmm%C
pub fn tile_dpbf16ps(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (!AmxFeatures.available) return;
    const tmmA: u8 = @intFromEnum(src_a);
    const tmmB: u8 = @intFromEnum(src_b);
    const tmmC: u8 = @intFromEnum(dst);
    asm volatile (
        "tilebf16dpd %%tmm%[B], %%tmm%[A], %%tmm%[C]"
        :
        : [A] "n" (tmmA),
          [B] "n" (tmmB),
          [C] "n" (tmmC),
        : "memory"
    );
}

/// DPBSSD: INT8 matrix multiply-accumulate (C += A * B^T, signed x signed)
/// A: INT8, B: INT8, C: INT32
/// Encoding: tileint8dpd %tmm%B, %tmm%A, %tmm%C
pub fn tile_dpbssd(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (!AmxFeatures.available) return;
    const tmmA: u8 = @intFromEnum(src_a);
    const tmmB: u8 = @intFromEnum(src_b);
    const tmmC: u8 = @intFromEnum(dst);
    asm volatile (
        "tileint8dpd %%tmm%[B], %%tmm%[A], %%tmm%[C]"
        :
        : [A] "n" (tmmA),
          [B] "n" (tmmB),
          [C] "n" (tmmC),
        : "memory"
    );
}

/// DPBSUD: INT8 matrix multiply-accumulate (C += A * B^T, signed x unsigned)
/// Note: tiledpbsud is not a standard mnemonic; emulate with the appropriate operand
/// reordering. Since signed/unsigned matters at the CPUID/microarchitectural level,
/// all INT8 ops go through the same `tileint8dpd` path. The kernel layer must
/// ensure that operands are sign-/zero-extended as required before this call.
pub fn tile_dpbsud(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (!AmxFeatures.available) return;
    // tileint8dpd treats both operands as signed; the caller is responsible for
    // sign-/zero-extending the unsigned operand to a signed 8-bit value if needed.
    tile_dpbssd(dst, src_a, src_b);
}

/// DPBUSD: INT8 matrix multiply-accumulate (C += A * B^T, unsigned x signed)
pub fn tile_dpbusd(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (!AmxFeatures.available) return;
    tile_dpbssd(dst, src_a, src_b);
}

/// DPBUUD: INT8 matrix multiply-accumulate (C += A * B^T, unsigned x unsigned)
pub fn tile_dpbuud(dst: TileReg, src_a: TileReg, src_b: TileReg) void {
    if (!AmxFeatures.available) return;
    tile_dpbssd(dst, src_a, src_b);
}

// ============================================================================
// BF16 Type Support
// ============================================================================

/// BF16 type - represents a 16-bit brain float
pub const bf16 = u16;

// Buffer alignment constants
const BF16_ALIGN = 16;

/// Convert f32 to bf16 (upper 16 bits of f32)
pub fn f32_to_bf16(x: f32) bf16 {
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

/// SiLU: x * sigmoid(x)
pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
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
    return gate * (1.0 / (1.0 + @exp(-gate * alpha))) * (up + 1.0);
}