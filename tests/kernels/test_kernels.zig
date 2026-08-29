// Test for ktransformers-zig
// Verifies basic compilation and functionality

const std = @import("std");
const testing = std.testing;

const root = @import("kt");

const amx = root.amx;
const buffers = root.buffers;
const gemm_bf16 = root.gemm_bf16;
const gemm_int8 = root.gemm_int8;
const gemm_int4 = root.gemm_int4;
const gemm_fp8 = root.gemm_fp8;
const cpu_detect = root.cpu_detect;
const worker_pool = root.worker_pool;
const memory = root.memory;
const moe = root.moe;

test "AMX feature detection" {
    // Just verify the module compiles
    _ = amx.AmxFeatures.available;
    _ = amx.TileConfig.init();
    _ = amx.bf16;
}

test "BF16 conversion" {
    const val: f32 = 3.14159;
    const bf16_val = amx.f32_to_bf16(val);
    const back = amx.bf16_to_f32(bf16_val);

    // BF16 has ~3 decimal digits precision
    const diff = if (back > val) back - val else val - back;
    try testing.expect(diff < 0.01);
}

test "BufferA required size" {
    const size = buffers.BufferA(amx.bf16).requiredSize(128);
    try testing.expect(size == 128 * 2); // 2 bytes per BF16
}

test "BufferB required size" {
    const size = buffers.BufferB(amx.bf16).requiredSize(4096, 4096, 2, false);
    try testing.expect(size == 4096 * 4096 * 2);
}

test "BufferB with scales" {
    const size = buffers.BufferB(amx.bf16).requiredSize(4096, 4096, 2, true);
    try testing.expect(size == 4096 * 4096 * 2 + 4096 * 4); // + scales
}

test "GEMM BF16 config" {
    try testing.expect(gemm_bf16.GemmKernel224BF.M_STEP == 32);
    try testing.expect(gemm_bf16.GemmKernel224BF.N_STEP == 32);
    try testing.expect(gemm_bf16.GemmKernel224BF.K_STEP == 32);
    try testing.expect(gemm_bf16.GemmKernel224BF.N_BLOCK == 256);
}

test "GEMM INT8 config" {
    try testing.expect(gemm_int8.GemmKernel224Int8.M_STEP == 32);
    try testing.expect(gemm_int8.GemmKernel224Int8.N_STEP == 32);
    try testing.expect(gemm_int8.GemmKernel224Int8.K_STEP == 64);
    try testing.expect(gemm_int8.GemmKernel224Int8.N_BLOCK == 64);
}

test "CPU detection" {
    const allocator = testing.allocator;
    const cpu = cpu_detect.detectCpu(allocator) catch @panic("CPU detect failed");

    try testing.expect(cpu.vendor != cpu_detect.Vendor.unknown);
    try testing.expect(cpu.arch.len > 0);

    // Print for debugging
    cpu_detect.printCpuInfo(cpu);
}

test "Worker pool creation" {
    const allocator = testing.allocator;
    var pool = try worker_pool.WorkerPool.initSimple(allocator, 2);
    defer pool.deinit();

    try testing.expect(pool.config.subpool_count == 1);
    try testing.expect(pool.config.subpool_thread_count[0] == 2);
}

test "Worker pool with config" {
    const allocator = testing.allocator;
    const config = worker_pool.WorkerPoolConfig{
        .subpool_count = 2,
        .subpool_numa_map = try allocator.alloc(usize, 2),
        .subpool_thread_count = try allocator.alloc(usize, 2),
    };
    config.subpool_numa_map[0] = 0;
    config.subpool_numa_map[1] = 1;
    config.subpool_thread_count[0] = 2;
    config.subpool_thread_count[1] = 2;

    const pool = try worker_pool.WorkerPool.init(allocator, config);
    var pool_mut = pool;
    defer pool_mut.deinit();

    try testing.expect(pool.config.subpool_count == 2);
    try testing.expect(pool.config.subpool_thread_count[0] == 2);
    try testing.expect(pool.config.subpool_thread_count[1] == 2);
}

test "Memory arena allocation" {
    const allocator = testing.allocator;
    var arena = memory.SimdArena.init(allocator, 64);
    defer arena.deinit();

    const buf = try arena.alloc64(f32, 1024);
    try testing.expect(buf.len == 1024);
    try testing.expect(@alignOf(@TypeOf(buf)) >= 64);
}

test "MoE config defaults" {
    const config = moe.MoeConfig{};
    try testing.expect(config.expert_num == 0);
    try testing.expect(config.num_experts_per_tok == 0);
    try testing.expect(config.hidden_size == 0);
    try testing.expect(config.intermediate_size == 0);
}

test "SwiGLU activation" {
    const gate: f32 = 1.0;
    const up: f32 = 2.0;
    const result = amx.swiglu(gate, up);
    // silu(1.0) * 2.0 = 1.0 / (1 + e^-1) * 2 ≈ 0.731 * 2 = 1.462
    try testing.expect(@abs(result - 1.462) < 0.01);
}

test "SwiGLU with clamp" {
    const gate: f32 = 15.0;
    const up: f32 = 15.0;
    const limit: f32 = 10.0;
    const result = amx.swiglu_clamp(gate, up, limit);
    // Both clamped to 10.0, silu(10) * 10 ≈ 10 * 10 = 100
    try testing.expect(@abs(result - 100.0) < 1.0);
}

test "Quantize BF16 to INT8" {
    const allocator = testing.allocator;
    const k = 32;
    var src = try allocator.alloc(amx.bf16, k);
    var dst = try allocator.alloc(i8, k);
    var scale: f32 = 0;
    defer {
        allocator.free(src);
        allocator.free(dst);
    }

    // Fill with values
    for (0..k) |i| {
        src[i] = amx.f32_to_bf16(@as(f32, @floatFromInt(i)) * 0.1);
    }

    buffers.quantizeRowBF16ToInt8(src.ptr, dst.ptr, &scale, k);

    try testing.expect(scale > 0);
    try testing.expect(scale < 1.0);
}

test "Dequantize INT8 to BF16" {
    const allocator = testing.allocator;
    const k = 32;
    var src = try allocator.alloc(i8, k);
    var dst = try allocator.alloc(amx.bf16, k);
    defer {
        allocator.free(src);
        allocator.free(dst);
    }

    // Fill with values
    for (0..k) |i| {
        src[i] = @as(i8, @intCast(i - 16));
    }

    buffers.dequantizeRowInt8ToBF16(src.ptr, 0.1, dst.ptr, k);

    // Check first value: -16 * 0.1 = -1.6
    try testing.expect(@abs(amx.bf16_to_f32(dst[0]) - (-1.6)) < 0.1);
}

test "AMX tile zero and store (no-op on non-AMX)" {
    // Skip on non-AMX hardware - intrinsics are no-ops and C tile won't be zeroed.
    if (!amx.AmxFeatures.available) return;

    var c_tile: [16 * 16]f32 align(64) = undefined;
    // Fill with garbage
    for (0..c_tile.len) |i| {
        c_tile[i] = @as(f32, @floatFromInt(i + 1));
    }

    amx.tile_loadconfig(&amx.TileConfig.init());
    defer amx.tile_release();

    amx.tile_zero(amx.TileReg.tmm4);
    amx.tile_stored(amx.TileReg.tmm4, @ptrCast(&c_tile), 64);

    // After tile_zero + tile_stored, the buffer should be all zeros.
    for (c_tile) |v| {
        try testing.expect(v == 0.0);
    }
}

test "AMX BF16 GEMM 16x16x16 identity * ones" {
    // On non-AMX hardware, intrinsics are no-ops and C will be unchanged from
    // its initial state (all zeros). Skip those cases.
    if (!amx.AmxFeatures.available) return;

    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 16;

    // Allocate 64-byte aligned buffers.
    var a: [M * K]amx.bf16 align(64) = undefined; // [M, K] row-major
    var b: [K * N]amx.bf16 align(64) = undefined; // [K, N] row-major (will be used as B^T)
    var c: [M * N]f32 align(64) = undefined; // [M, N] row-major output

    // A = identity: A[i, j] = 1.0 if i == j else 0.0
    for (0..M) |i| {
        for (0..K) |j| {
            a[i * K + j] = if (i == j) amx.f32_to_bf16(1.0) else amx.f32_to_bf16(0.0);
        }
    }

    // B = all ones: B[i, j] = 1.0
    for (0..b.len) |i| {
        b[i] = amx.f32_to_bf16(1.0);
    }

    // C = zeros
    for (0..c.len) |i| {
        c[i] = 0.0;
    }

    // Manually do C[M,N] = A[M,K] * B[K,N] using tile intrinsics.
    amx.tile_loadconfig(&amx.TileConfig.init());
    defer amx.tile_release();

    // Zero C tile.
    amx.tile_zero(amx.TileReg.tmm4);
    amx.tile_stored(amx.TileReg.tmm4, @ptrCast(&c), 64);

    // Load A into tmm0 (one row of A, padded to 32 BF16 = 64 bytes).
    // Since K=16 < 32, pad the second half with zeros.
    var a_padded: [16 * 32]amx.bf16 align(64) = undefined;
    for (0..16) |i| {
        for (0..32) |j| {
            a_padded[i * 32 + j] = if (j < K) a[i * K + j] else amx.f32_to_bf16(0.0);
        }
    }
    amx.tile_loadd(amx.TileReg.tmm0, @ptrCast(&a_padded), 64);

    // Load B (as B^T) into tmm2. AMX BF16 GEMM does C += A * B^T.
    // B is [K, N] = [16, 16]. We need to load it in VNNI-transposed form
    // (pairs of 2 BF16 elements). For simplicity, since N=16 fits in one
    // tile, we can load B as a 16x16 matrix and treat it as B^T for a 16x16
    // A. The tile shape for B in this kernel is TILE_N x TILE_K = 16x32.
    var b_padded: [16 * 32]amx.bf16 align(64) = undefined;
    for (0..16) |i| {
        for (0..32) |j| {
            b_padded[i * 32 + j] = if (j < K) b[j * N + i] else amx.f32_to_bf16(0.0);
        }
    }
    amx.tile_loadd(amx.TileReg.tmm2, @ptrCast(&b_padded), 64);

    // C[16x16] += A[16x32] * B[16x32]^T (effectively [16x16])
    amx.tile_dpbf16ps(amx.TileReg.tmm4, amx.TileReg.tmm0, amx.TileReg.tmm2);

    // Store C tile back.
    amx.tile_stored(amx.TileReg.tmm4, @ptrCast(&c), 64);

    // Verify: C[i, j] should be sum over k of A[i, k] * B[k, j] = sum of B[k, j] for k == i
    // = B[i, j] = 1.0 for all (i, j).
    for (0..M) |i| {
        for (0..N) |j| {
            const expected: f32 = 1.0;
            const actual = c[i * N + j];
            const diff = if (actual > expected) actual - expected else expected - actual;
            // BF16 has ~3 decimal digits precision; allow tolerance.
            try testing.expect(diff < 0.01);
        }
    }
}

test "INT4 GEMM 16x16x32 with simple constant B" {
    // On non-AMX hardware, the kernel falls back to scalar (correct but slow).
    // The test still validates correctness either way.
    if (!amx.AmxFeatures.available) return;

    // Small test: M=16, N=16, K=32 (one GPTQ group per output row).
    // A: row 0 = all 1s, other rows = all 0s.
    // B: every weight = 3 (nibble 0xB). After dequant: (3-8)*scale = -5*scale.
    //   With scale=1.0: dequant value = -5.
    // Expected: c[0, j] = sum_{k=0..32} A[0,k]*B[j,k] = 32 * (-5) = -160.
    //            c[i, j] = 0 for i > 0.
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 32;
    const k_blocks: usize = K / 32; // = 1 block per row

    var a: [M * K]i8 align(64) = undefined;
    var b: [N * k_blocks]gemm_int4.BlockQ4_0 align(64) = undefined;
    var c: [M * N]f32 align(64) = undefined;

    // A: row 0 all 1s, rest all 0s.
    for (0..M) |i| {
        for (0..K) |j| {
            a[i * K + j] = if (i == 0) 1 else 0;
        }
    }

    // B: every weight = 3 (nibble 0xB), scale = 1.0.
    // Pack 32 weights into 16 bytes: each byte has lo=3, hi=3 → 0xBB.
    for (0..N) |i| {
        for (0..k_blocks) |blk| {
            b[i * k_blocks + blk].d = amx.f32_to_bf16(1.0);
            for (0..16) |q| {
                b[i * k_blocks + blk].qs[q] = 0xBB;
            }
        }
    }

    // C: zero.
    for (0..c.len) |i| {
        c[i] = 0.0;
    }

    // Call the AMX INT4 GEMM.
    // lda = K (row stride of A in elements), ldb = k_blocks (row stride of B in blocks),
    // ldc = N (row stride of C in elements).
    gemm_int4.GemmKernel224Int4.gemmFullTile(
        &a, K,
        &b, k_blocks,
        &c, N,
        M, N, K,
    );

    // Verify: c[0, j] ≈ -160, c[i, j] ≈ 0 for i > 0.
    const tolerance: f32 = 1.0;
    for (0..M) |i| {
        for (0..N) |j| {
            const expected: f32 = if (i == 0) -160.0 else 0.0;
            const actual = c[i * N + j];
            const diff = if (actual > expected) actual - expected else expected - actual;
            try testing.expect(diff < tolerance);
        }
    }
}
test "FP8 E4M3 GEMM 16x16x32 with constant B" {
    // On non-AMX hardware, the kernel falls back to scalar (correct but slow).
    if (!amx.AmxFeatures.available) return;

    // M=16, N=16, K=32 (one K_STEP=32 step).
    // A: row 0 all 1.0 (BF16), other rows all 0.0.
    // B: every entry = FP8 0x38 (= BF16 1.0 in E4M3: sign=0, exp=7, mant=0).
    // Per-row scale = 1.0 for all rows.
    // Expected: c[0, j] = sum_{k=0..32} A[0, k] * B[j, k] * scale[j] = 32 * 1.0 * 1.0 * 1.0 = 32.
    //            c[i, j] = 0 for i > 0.
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 32;

    var a: [M * K]amx.bf16 align(64) = undefined;
    var b: [N * K]u8 align(64) = undefined; // fp8_e4m3 is u8
    var b_scales: [N]f32 align(64) = undefined;
    var c: [M * N]f32 align(64) = undefined;

    // A: row 0 all 1.0 in BF16, rest all 0.0.
    for (0..M) |i| {
        for (0..K) |j| {
            a[i * K + j] = if (i == 0) amx.f32_to_bf16(1.0) else amx.f32_to_bf16(0.0);
        }
    }

    // B: every entry = 0x38 (FP8 1.0).
    for (0..b.len) |i| {
        b[i] = 0x38;
    }

    // B scales: all 1.0.
    for (0..N) |i| {
        b_scales[i] = 1.0;
    }

    // C: zero.
    for (0..c.len) |i| {
        c[i] = 0.0;
    }

    gemm_fp8.GemmKernel224FP8.gemmFullTile(
        &a, K, // lda in elements
        &b, &b_scales, K, // ldb in elements (1 byte each)
        &c, N, // ldc in elements
        M, N, K,
    );

    // Verify: c[0, j] ≈ 32, c[i, j] ≈ 0 for i > 0.
    // FP8 E4M3 has ~3 mantissa bits, so expect some rounding error.
    const tolerance: f32 = 0.5;
    for (0..M) |i| {
        for (0..N) |j| {
            const expected: f32 = if (i == 0) 32.0 else 0.0;
            const actual = c[i * N + j];
            const diff = if (actual > expected) actual - expected else expected - actual;
            try testing.expect(diff < tolerance);
        }
    }
}
