// GEMM Kernel 224 BF16 for AMX
// Ported from ktransformers kt-kernel/operators/amx/la/amx_kernels.hpp GemmKernel224BF

const std = @import("std");
const amx = @import("../arch/amx.zig");
const buffers = @import("buffers.zig");

// ============================================================================
// Kernel Configuration
// ============================================================================

pub const GemmKernel224BF = struct {
    pub const dt = amx.bf16;
    pub const output_t = f32;
    pub const ELEMENT_SIZE = 2;

    pub const TILE_M = 16;
    pub const TILE_K = 32;
    pub const TILE_N = 16;
    pub const VNNI_BLK = 2;

    pub const M_STEP = 32;  // TILE_M * 2
    pub const N_STEP = 32;  // TILE_N * 2
    pub const K_STEP = 32;  // TILE_K

    pub const N_BLOCK = 256;
    pub const K_BLOCK = 1792;

    pub fn name() []const u8 { return "BF16"; }

    pub fn recommended_nth(n: usize) usize {
        return (n + N_BLOCK - 1) / N_BLOCK;
    }

    pub inline fn split_range_n(ith: usize) struct { n_start: usize, n_end: usize } {
        const n_start = N_BLOCK * ith;
        const n_end = @min(N_BLOCK, N_BLOCK * (ith + 1));
        return .{ .n_start = n_start, .n_end = n_end };
    }

    // ========================================================================
    // Tile Configuration
    // ========================================================================

    pub fn config() void {
        var tile_config = amx.TileConfig.init();

        // Tile 0,1: A matrices (16 x 32 BF16)
        for (0..2) |i| {
            tile_config.setTile(i, TILE_M, TILE_K * @sizeOf(dt));
        }

        // Tile 2,3: B matrices (16 x 32 BF16 VNNI)
        for (2..4) |i| {
            tile_config.setTile(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * @sizeOf(dt));
        }

        // Tile 4,5,6,7: C matrices (16 x 16 FP32)
        for (4..8) |i| {
            tile_config.setTile(i, TILE_M, TILE_N * @sizeOf(output_t));
        }

        amx.tile_loadconfig(&tile_config);
    }

    // ========================================================================
    // Core GEMM Operations
    // ========================================================================

    /// Load A tiles (tile 0,1)
    pub fn loadA(a: *const dt, lda: usize) void {
        amx.tile_loadd(amx.TileReg.tmm0, @ptrCast(a), lda);
        amx.tile_loadd(amx.TileReg.tmm1, @ptrCast(a + TILE_M * lda), lda);
    }

    /// Load B tiles (tile 2,3) - VNNI layout
    pub fn loadB(b: *const dt, ldb: usize) void {
        amx.tile_loadd(amx.TileReg.tmm2, @ptrCast(b), ldb);
        amx.tile_loadd(amx.TileReg.tmm3, @ptrCast(b + TILE_N * ldb), ldb);
    }

    /// Zero C tiles (tile 4,5,6,7)
    pub fn cleanC() void {
        amx.tile_zero(amx.TileReg.tmm4);
        amx.tile_zero(amx.TileReg.tmm5);
        amx.tile_zero(amx.TileReg.tmm6);
        amx.tile_zero(amx.TileReg.tmm7);
    }

    /// Load C tiles
    pub fn loadC(c: *output_t, ldc: usize) void {
        amx.tile_loadd(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_loadd(amx.TileReg.tmm5, @ptrCast(c + TILE_N), ldc);
        amx.tile_loadd(amx.TileReg.tmm6, @ptrCast(c + ldc * TILE_M), ldc);
        amx.tile_loadd(amx.TileReg.tmm7, @ptrCast(c + ldc * TILE_M + TILE_N), ldc);
    }

    /// Store C tiles
    pub fn storeC(c: *output_t, ldc: usize) void {
        amx.tile_stored(amx.TileReg.tmm4, @ptrCast(c), ldc);
        amx.tile_stored(amx.TileReg.tmm5, @ptrCast(c + TILE_N), ldc);
        amx.tile_stored(amx.TileReg.tmm6, @ptrCast(c + ldc * TILE_M), ldc);
        amx.tile_stored(amx.TileReg.tmm7, @ptrCast(c + ldc * TILE_M + TILE_N), ldc);
    }

    /// Execute DPBF16PS on all 4 C tiles
    pub fn runTile() void {
        // C[0] += A[0] * B[0]
        amx.tile_dpbf16ps(amx.TileReg.tmm4, amx.TileReg.tmm0, amx.TileReg.tmm2);
        // C[1] += A[0] * B[1]
        amx.tile_dpbf16ps(amx.TileReg.tmm5, amx.TileReg.tmm0, amx.TileReg.tmm3);
        // C[2] += A[1] * B[0]
        amx.tile_dpbf16ps(amx.TileReg.tmm6, amx.TileReg.tmm1, amx.TileReg.tmm2);
        // C[3] += A[1] * B[1]
        amx.tile_dpbf16ps(amx.TileReg.tmm7, amx.TileReg.tmm1, amx.TileReg.tmm3);
    }

    // ========================================================================
    // Buffer Types
    // ========================================================================

    pub const BufferA = buffers.BufferA(dt);
    pub const BufferB = buffers.BufferB(dt);
    pub const BufferC = buffers.BufferC(output_t);

    // ========================================================================
    // Full Tile GEMM
    // ========================================================================

    /// Compute full 32x32 tile GEMM using AMX
    /// a: [32, k] row-major, b: [32, k] col-major VNNI, c: [32, 32] row-major FP32
    pub fn gemmFullTile(
        a: *const dt, lda: usize,
        b: *const dt, ldb: usize,
        c: *output_t, ldc: usize,
        k: usize
    ) void {
        config();

        var k_processed: usize = 0;
        while (k_processed < k) : (k_processed += K_STEP) {
            const k_remain = k - k_processed;
            const k_step = @min(K_STEP, k_remain);

            if (k_step == K_STEP) {
                // Full K_STEP - use AMX tiles directly
                loadA(a + k_processed, lda);
                loadB(b + k_processed, ldb);

                if (k_processed == 0) {
                    cleanC();
                } else {
                    loadC(c, ldc);
                }

                runTile();
                storeC(c, ldc);
            } else {
                // Partial K_STEP - need masked load or fallback
                // For now, fall back to scalar
                gemmPartialTile(a + k_processed, lda, b + k_processed, ldb, c, ldc, k_step);
            }
        }
    }

    /// Partial tile GEMM (scalar fallback)
    fn gemmPartialTile(
        a: *const dt, lda: usize,
        b: *const dt, ldb: usize,
        c: *output_t, ldc: usize,
        k_step: usize
    ) void {
        for (0..M_STEP) |m| {
            for (0..N_STEP) |n| {
                var sum: f32 = 0;
                for (0..k_step) |k| {
                    sum += amx.bf16_to_f32(a[m * lda + k]) * amx.bf16_to_f32(b[n * ldb + k]);
                }
                c[m * ldc + n] += sum;
            }
        }
    }

    // ========================================================================
    // Batched GEMM for MoE
    // ========================================================================

    /// Batched GEMM for multiple experts
    /// experts: array of expert data pointers
    pub fn batchedGemm(
        experts: []struct { a: *const dt, b: *const dt, c: *output_t },
        _m: usize, _n: usize, k: usize,
        lda: usize, ldb: usize, ldc: usize
    ) void {
        _ = _m; _ = _n;
        for (experts) |exp| {
            gemmFullTile(exp.a, lda, exp.b, ldb, exp.c, ldc, k);
        }
    }

    // ========================================================================
    // Vectorized Activation Functions
    // ========================================================================

    /// Apply SwiGLU activation to gate and up outputs
    /// gate, up: [m, n] BF16, dst: [m, n] BF16
    pub fn applySwiGLU(
        gate: [*]dt, up: [*]dt, dst: [*]dt,
        m: usize, n: usize,
        limit: f32, alpha: f32
    ) void {
        for (0..m) |i| {
            for (0..n) |j| {
                const gate_val = amx.bf16_to_f32(gate[i * n + j]);
                const up_val = amx.bf16_to_f32(up[i * n + j]);

                var result: f32 = 0;
                if (alpha > 0) {
                    // SwiGLU-OAI
                    result = amx.swiglu_oai(gate_val, up_val, alpha, limit);
                } else if (limit > 0) {
                    // SwiGLU with clamp
                    result = amx.swiglu_clamp(gate_val, up_val, limit);
                } else {
                    // Standard SwiGLU
                    result = amx.swiglu(gate_val, up_val);
                }
                dst[i * n + j] = amx.f32_to_bf16(result);
            }
        }
    }
};

// ============================================================================
// High-level GEMM Interface
// ============================================================================

/// GEMM wrapper for MoE expert computation
/// Computes: output = input * weight^T for a single expert
pub fn gemmExpert(
    input: [*]const amx.bf16,  // [m, k]
    weight: [*]const amx.bf16, // [n, k] (stored transposed in BufferB)
    output: [*]f32,            // [m, n]
    m: usize, n: usize, k: usize,
    input_ld: usize, weight_ld: usize, output_ld: usize
) void {
    GemmKernel224BF.config();

    const m_tiles = (m + GemmKernel224BF.M_STEP - 1) / GemmKernel224BF.M_STEP;
    const n_tiles = (n + GemmKernel224BF.N_STEP - 1) / GemmKernel224BF.N_STEP;

    for (0..m_tiles) |mt| {
        const m_start = mt * GemmKernel224BF.M_STEP;
        const m_end = @min(m, m_start + GemmKernel224BF.M_STEP);
        const m_actual = m_end - m_start;

        for (0..n_tiles) |nt| {
            const n_start = nt * GemmKernel224BF.N_STEP;
            const n_end = @min(n, n_start + GemmKernel224BF.N_STEP);
            const n_actual = n_end - n_start;

            // Process K dimension in chunks
            var k_processed: usize = 0;
            while (k_processed < k) : (k_processed += GemmKernel224BF.K_STEP) {
                const k_step = @min(GemmKernel224BF.K_STEP, k - k_processed);

                // Load A tile (m_actual x k_step)
                // Load B tile (n_actual x k_step) - VNNI
                // Compute and accumulate to C tile

                if (k_processed == 0) {
                    // Zero output tile
                    for (0..m_actual) |i| {
                        for (0..n_actual) |j| {
                            output[(m_start + i) * output_ld + n_start + j] = 0;
                        }
                    }
                }

                // Simple GEMM for now - will be replaced with AMX tiles
                for (0..m_actual) |i| {
                    for (0..n_actual) |j| {
                        var sum: f32 = 0;
                        for (0..k_step) |kk| {
                            const a_val = amx.bf16_to_f32(input[(m_start + i) * input_ld + k_processed + kk]);
                            const b_val = amx.bf16_to_f32(weight[(n_start + j) * weight_ld + k_processed + kk]);
                            sum += a_val * b_val;
                        }
                        output[(m_start + i) * output_ld + n_start + j] += sum;
                    }
                }
            }
        }
    }
}