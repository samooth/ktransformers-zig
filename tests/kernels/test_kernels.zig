// Test for ktransformers-zig
// Verifies basic compilation and functionality

const std = @import("std");
const testing = std.testing;

const root = @import("kt");

const amx = root.amx;
const buffers = root.buffers;
const gemm_bf16 = root.gemm_bf16;
const gemm_int8 = root.gemm_int8;
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