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

    // A4: runtime-overridable block sizes. Defaults preserve the
    // historical constants exactly (1792/256 — a generic-server profile
    // assuming >=1 MiB L2). On hosts with a smaller measured L2 (e.g.
    // Ryzen 5800H: 512K), `setTileParams` swaps K_BLOCK so a block-step's
    // A+B+C working set stays inside 50% of L2 (see
    // runtime/cpu_detect.selectTileParams). Must be called BEFORE any
    // BufferA/expert-buffer allocation (TpMoe.init reads these).
    pub var N_BLOCK: usize = 256;
    pub var K_BLOCK: usize = 1792;

    pub const DEFAULT_N_BLOCK: usize = 256;
    pub const DEFAULT_K_BLOCK: usize = 1792;

    /// Override the block sizes from measured cache hierarchy. Invalid
    /// or out-of-range values keep the defaults (never a bad-config
    /// crash). Tile alignment is enforced (multiples of K_STEP/N_STEP).
    pub fn setTileParams(n_block_in: usize, k_block_in: usize) void {
        if (n_block_in >= N_STEP and n_block_in % N_STEP == 0) {
            N_BLOCK = n_block_in;
        }
        if (k_block_in >= K_STEP and k_block_in % K_STEP == 0) {
            K_BLOCK = k_block_in;
        }
    }

    /// Restore the compiled-in defaults.
    pub fn resetTileParams() void {
        N_BLOCK = DEFAULT_N_BLOCK;
        K_BLOCK = DEFAULT_K_BLOCK;
    }

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
            tile_config.setTile(@as(u8, @intCast(i)), TILE_M, @as(u16, @intCast(TILE_K * @sizeOf(dt))));
        }

        // Tile 2,3: B matrices (16 x 32 BF16 VNNI)
        for (2..4) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), @as(u16, @intCast(TILE_K / VNNI_BLK)), @as(u16, @intCast(TILE_N * VNNI_BLK * @sizeOf(dt))));
        }

        // Tile 4,5,6,7: C matrices (16 x 16 FP32)
        for (4..8) |i| {
            tile_config.setTile(@as(u8, @intCast(i)), TILE_M, @as(u16, @intCast(TILE_N * @sizeOf(output_t))));
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
    ///
    /// Vectorized over `n` with `amx.VecF32` (8×f32 / 256-bit, matches AVX2 on
    /// the host). Three variants (OAI / clamp / standard) are chosen once
    /// outside the loops so the hot path has no per-element branching. The
    /// tail (`n % VecF32.len`) is handled with a scalar loop.
    pub fn applySwiGLU(
        gate: [*]dt, up: [*]dt, dst: [*]dt,
        m: usize, n: usize,
        limit: f32, alpha: f32
    ) void {
        const VEC_LEN: usize = amx.VEC_LEN;
        const main_end: usize = n - (n % VEC_LEN);

        if (alpha > 0) {
            // SwiGLU-OAI (MiniMax M3)
            const alpha_vec: amx.VecF32 = @splat(alpha);
            const limit_vec: amx.VecF32 = @splat(limit);
            for (0..m) |i| {
                const gate_row: [*]dt = gate + i * n;
                const up_row: [*]dt = up + i * n;
                const dst_row: [*]dt = dst + i * n;
                var j: usize = 0;
                while (j < main_end) : (j += VEC_LEN) {
                    var g_arr: [VEC_LEN]f32 = undefined;
                    var u_arr: [VEC_LEN]f32 = undefined;
                    for (0..VEC_LEN) |k| {
                        g_arr[k] = amx.bf16_to_f32(gate_row[j + k]);
                        u_arr[k] = amx.bf16_to_f32(up_row[j + k]);
                    }
                    const gv: amx.VecF32 = g_arr;
                    const uv: amx.VecF32 = u_arr;
                    const res: amx.VecF32 = amx.swigluOaiVec(gv, uv, alpha_vec, limit_vec);
                    const res_arr: [VEC_LEN]f32 = res;
                    for (0..VEC_LEN) |k| {
                        dst_row[j + k] = amx.f32_to_bf16(res_arr[k]);
                    }
                }
                while (j < n) : (j += 1) {
                    const g = amx.bf16_to_f32(gate_row[j]);
                    const u = amx.bf16_to_f32(up_row[j]);
                    dst_row[j] = amx.f32_to_bf16(amx.swiglu_oai(g, u, alpha, limit));
                }
            }
        } else if (limit > 0) {
            // SwiGLU with clamp (MXFP4 path)
            const limit_vec: amx.VecF32 = @splat(limit);
            for (0..m) |i| {
                const gate_row: [*]dt = gate + i * n;
                const up_row: [*]dt = up + i * n;
                const dst_row: [*]dt = dst + i * n;
                var j: usize = 0;
                while (j < main_end) : (j += VEC_LEN) {
                    var g_arr: [VEC_LEN]f32 = undefined;
                    var u_arr: [VEC_LEN]f32 = undefined;
                    for (0..VEC_LEN) |k| {
                        g_arr[k] = amx.bf16_to_f32(gate_row[j + k]);
                        u_arr[k] = amx.bf16_to_f32(up_row[j + k]);
                    }
                    const gv: amx.VecF32 = g_arr;
                    const uv: amx.VecF32 = u_arr;
                    const res: amx.VecF32 = amx.swigluClampVec(gv, uv, limit_vec);
                    const res_arr: [VEC_LEN]f32 = res;
                    for (0..VEC_LEN) |k| {
                        dst_row[j + k] = amx.f32_to_bf16(res_arr[k]);
                    }
                }
                while (j < n) : (j += 1) {
                    const g = amx.bf16_to_f32(gate_row[j]);
                    const u = amx.bf16_to_f32(up_row[j]);
                    dst_row[j] = amx.f32_to_bf16(amx.swiglu_clamp(g, u, limit));
                }
            }
        } else {
            // Standard SwiGLU (the common case — linear, branch-free inner loop)
            for (0..m) |i| {
                const gate_row: [*]dt = gate + i * n;
                const up_row: [*]dt = up + i * n;
                const dst_row: [*]dt = dst + i * n;
                var j: usize = 0;
                while (j < main_end) : (j += VEC_LEN) {
                    var g_arr: [VEC_LEN]f32 = undefined;
                    var u_arr: [VEC_LEN]f32 = undefined;
                    for (0..VEC_LEN) |k| {
                        g_arr[k] = amx.bf16_to_f32(gate_row[j + k]);
                        u_arr[k] = amx.bf16_to_f32(up_row[j + k]);
                    }
                    const gv: amx.VecF32 = g_arr;
                    const uv: amx.VecF32 = u_arr;
                    const res: amx.VecF32 = amx.swigluVec(gv, uv);
                    const res_arr: [VEC_LEN]f32 = res;
                    for (0..VEC_LEN) |k| {
                        dst_row[j + k] = amx.f32_to_bf16(res_arr[k]);
                    }
                }
                while (j < n) : (j += 1) {
                    const g = amx.bf16_to_f32(gate_row[j]);
                    const u = amx.bf16_to_f32(up_row[j]);
                    dst_row[j] = amx.f32_to_bf16(amx.swiglu(g, u));
                }
            }
        }
    }
};

// ============================================================================
// High-level GEMM Interface
// ============================================================================
//
// Layout contract (unchanged): output[m, n] = sum_k input[m, k] * weight[n, k]
// with `weight` row-major as [n, k] (one output neuron per row) and `input`
// row-major as [m, k]. Leading dimensions are explicit so tensor-parallel rank
// slicing can pass offset pointers with ld == k.
//
// LOOP ORDER — this is the whole performance story on non-AMX hosts.
//
// The obvious nest is m-outer / n-inner, and that is what this function used
// to do. It is wrong for a GEMM where one operand dwarfs the other: here
// `weight` is n*k (29 MB for a DeepSeek-V3 gate/up projection, well past this
// host's L3), so an m-outer nest re-streams the ENTIRE weight matrix from DRAM
// once per token row — m passes over 29 MB. Measured on Zen3 at m=128 that is
// 3.7 GB from DRAM at ~7 GB/s.
//
// n-outer / m-inner instead keeps one row of `weight` (k*2 bytes = 14 KB,
// L1-resident) hot across all m, so the weights are read from DRAM exactly ONCE
// and only the small activation matrix is re-read — out of L2/L3. Same 3.7 GB
// of logical traffic, but at ~20 GB/s. Measured 3.4-4.0x, and it lands within
// 5% of the C++ reference's numbers.
//
// REGISTER TILING. bf16 -> f32 is 2 uops per 8 elements per operand and a GEMM
// needs 2 operands per 8 MACs, so the conversion — not the FMA — dominates the
// uop budget (0.5 vs 0.125 uops/MAC). With `MR` rows of A in flight, each
// converted 32-element chunk of `weight` feeds MR accumulators, so MR=4 cuts
// the conversions per k-block from 32 to 20. MR*4 accumulators + 4 weight
// temporaries = 8 of 16 ymm, which fits without spilling.
//
// ACCUMULATION. The two entry points are deliberately separate: `gemmExpert`
// OVERWRITES `output` (so callers may hand it uninitialised memory — several
// do), `gemmExpertAccum` ADDS to it (so a tensor-parallel rank loop can
// accumulate partial sums into one buffer). The old single entry point zeroed
// its output tile on entry and then accumulated within its own k-tiles, so it
// behaved as OVERWRITE — but `TpMoe.forwardDown` relied on it ADDING across
// tp ranks, which silently lost every rank but the last. Making the contract
// explicit fixes that instead of preserving it.

/// Rows of `input` processed per pass. Chosen so the 4 weight temporaries and
/// MR*4 accumulators stay inside 16 ymm registers.
const MR: usize = 4;

/// bf16 -> f32 for 8 consecutive values. LLVM lowers this to
/// `vpmovzxwd` + `vpslld $16` (the AVX2 bf16 widening trick); there is no F16C
/// instruction for bf16, which is why F16C is irrelevant to this path.
inline fn widen8(p: [*]const amx.bf16) amx.VecF32 {
    return amx.widen8(p);
}

/// Dot the 32-element k-block at `a_k` against the already-widened weight chunk
/// `bw[0..4]`, accumulating into `acc`. 4 independent FMAs keep the FMA pipes
/// busy; explicit `@mulAdd` is required because Zig defaults `fp-contract` to
/// `off` (emitting separate vmulps/vaddps instead of vfmadd231ps).
inline fn fmaBlock(
    a_k: [*]const amx.bf16,
    bw: [4]amx.VecF32,
    acc: *[4]amx.VecF32,
) void {
    inline for (0..4) |u| {
        acc[u] = @mulAdd(amx.VecF32, widen8(a_k + u * amx.VEC_LEN), bw[u], acc[u]);
    }
}

fn gemmExpertImpl(
    input: [*]const amx.bf16,
    weight: [*]const amx.bf16,
    output: [*]f32,
    m: usize, n: usize, k: usize,
    input_ld: usize, weight_ld: usize, output_ld: usize,
    accum: bool,
) void {
    GemmKernel224BF.config();
    if (m == 0 or n == 0) return;

    const k_vec = k - (k % amx.VEC_LEN); // 8-element fast-path prefix
    const k_blk = k_vec - (k_vec % 32);  // 32-element register-tiled prefix
    const zero: amx.VecF32 = @splat(0.0);

    var j: usize = 0;
    while (j < n) : (j += 1) {
        const brow: [*]const amx.bf16 = weight + j * weight_ld;

        var i: usize = 0;
        // Main path: MR rows of A share each widened weight chunk.
        while (i + MR <= m) : (i += MR) {
            var acc: [MR][4]amx.VecF32 = [_][4]amx.VecF32{[_]amx.VecF32{zero} ** 4} ** MR;
            var kk: usize = 0;
            while (kk < k_blk) : (kk += 32) {
                const bw: [4]amx.VecF32 = .{
                    widen8(brow + kk),
                    widen8(brow + kk + 8),
                    widen8(brow + kk + 16),
                    widen8(brow + kk + 24),
                };
                inline for (0..MR) |r| {
                    fmaBlock(input + (i + r) * input_ld + kk, bw, &acc[r]);
                }
            }
            inline for (0..MR) |r| {
                storeDot(output + (i + r) * output_ld + j, &acc[r], input + (i + r) * input_ld, brow, kk, k_vec, k, accum);
            }
        }
        // Row tail (m % MR), one row at a time.
        while (i < m) : (i += 1) {
            var acc: [1][4]amx.VecF32 = [_][4]amx.VecF32{[_]amx.VecF32{zero} ** 4};
            var kk: usize = 0;
            while (kk < k_blk) : (kk += 32) {
                const bw: [4]amx.VecF32 = .{
                    widen8(brow + kk),
                    widen8(brow + kk + 8),
                    widen8(brow + kk + 16),
                    widen8(brow + kk + 24),
                };
                fmaBlock(input + i * input_ld + kk, bw, &acc[0]);
            }
            storeDot(output + i * output_ld + j, &acc[0], input + i * input_ld, brow, kk, k_vec, k, accum);
        }
    }
}

/// Reduce `acc` (4 vectors spanning k_blk..k_vec) plus the vector/scalar k tails
/// into one f32 and write it to `out`.
inline fn storeDot(
    out: [*]f32,
    acc: *const [4]amx.VecF32,
    a_row: [*]const amx.bf16,
    brow: [*]const amx.bf16,
    kk: usize, k_vec: usize, k: usize,
    accum: bool,
) void {
    var sum = amx.reduceAddFp32(acc[0] + acc[1] + acc[2] + acc[3]);
    var t: usize = kk;
    while (t < k_vec) : (t += amx.VEC_LEN) {
        sum += amx.reduceAddFp32(widen8(a_row + t) * widen8(brow + t));
    }
    while (t < k) : (t += 1) {
        sum += amx.bf16_to_f32(a_row[t]) * amx.bf16_to_f32(brow[t]);
    }
    if (accum) {
        out[0] += sum;
    } else {
        out[0] = sum;
    }
}

/// GEMM for MoE expert computation: **overwrites** `output`.
/// Computes `output = input * weight^T` for a single expert.
/// `output` may point at uninitialised memory.
pub fn gemmExpert(
    input: [*]const amx.bf16,
    weight: [*]const amx.bf16,
    output: [*]f32,
    m: usize, n: usize, k: usize,
    input_ld: usize, weight_ld: usize, output_ld: usize,
) void {
    gemmExpertImpl(input, weight, output, m, n, k, input_ld, weight_ld, output_ld, false);
}

/// GEMM for MoE expert computation: **accumulates into** `output`.
/// `output += input * weight^T`. Use this when splitting a projection across
/// tensor-parallel ranks into one shared FP32 buffer.
pub fn gemmExpertAccum(
    input: [*]const amx.bf16,
    weight: [*]const amx.bf16,
    output: [*]f32,
    m: usize, n: usize, k: usize,
    input_ld: usize, weight_ld: usize, output_ld: usize,
) void {
    gemmExpertImpl(input, weight, output, m, n, k, input_ld, weight_ld, output_ld, true);
}
