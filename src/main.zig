// Main Zig entry point for ktransformers-zig library
// Exports C-compatible API

const std = @import("std");
const root = @import("root.zig");

const memory = root.memory;
const worker_pool = root.worker_pool;
const task_queue = root.task_queue;
const cpu_detect = root.cpu_detect;
const amx = root.amx;
const buffers = root.buffers;
const gemm_bf16 = root.gemm_bf16;
const gemm_int8 = root.gemm_int8;
const moe = root.moe;

// C API types
pub const kt_quant_method_t = enum(c_int) {
    KT_QUANT_NONE = 0,
    KT_QUANT_FP8 = 1,
    KT_QUANT_INT8 = 2,
    KT_QUANT_INT4 = 3,
    KT_QUANT_GPTQ = 4,
    KT_QUANT_AWQ = 5,
    KT_QUANT_MXFP4 = 6,
    KT_QUANT_MXFP8 = 7,
    KT_QUANT_BF16 = 8,
};

pub const kt_type_t = enum(c_int) {
    KT_TYPE_F32 = 0,
    KT_TYPE_F16 = 1,
    KT_TYPE_BF16 = 2,
    KT_TYPE_I8 = 3,
    KT_TYPE_I4 = 4,
    KT_TYPE_Q4_0 = 8,
    KT_TYPE_Q4_1 = 9,
    KT_TYPE_Q5_0 = 10,
    KT_TYPE_Q5_1 = 11,
    KT_TYPE_Q8_0 = 12,
    KT_TYPE_Q8_1 = 13,
    KT_TYPE_Q2_K = 14,
    KT_TYPE_Q3_K = 15,
    KT_TYPE_Q4_K = 16,
    KT_TYPE_Q5_K = 17,
    KT_TYPE_Q6_K = 18,
    KT_TYPE_Q8_K = 19,
    KT_TYPE_IQ2_XXS = 20,
    KT_TYPE_IQ2_XS = 21,
    KT_TYPE_IQ3_XXS = 22,
    KT_TYPE_IQ1_S = 23,
    KT_TYPE_IQ4_NL = 24,
    KT_TYPE_IQ3_S = 25,
    KT_TYPE_IQ2_S = 26,
    KT_TYPE_IQ4_XS = 27,
    KT_TYPE_IQ1_M = 28,
    KT_TYPE_MXFP4 = 29,
    KT_TYPE_MXFP8 = 30,
};

pub const kt_variant_t = enum(c_int) {
    KT_VARIANT_AVX2 = 0,
    KT_VARIANT_AVX512_BASE = 1,
    KT_VARIANT_AVX512_VNNI = 2,
    KT_VARIANT_AVX512_VBMI = 3,
    KT_VARIANT_AVX512_BF16 = 4,
    KT_VARIANT_AMX = 5,
};

pub const kt_quant_config_t = extern struct {
    quant_method: kt_quant_method_t,
    bits: c_int,
    group_size: c_int,
    zero_point: c_int,
    per_channel: c_int,
};

pub const kt_worker_pool_config_t = extern struct {
    subpool_count: c_int,
    subpool_numa_map: [*]c_int,
    subpool_thread_count: [*]c_int,
};

pub const kt_general_config_t = extern struct {
    vocab_size: usize,
    hidden_size: usize,
    num_experts_per_tok: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    max_qlen: usize,
    lm_heads_ptr: *anyopaque,
    lm_heads_type: kt_type_t,
    norm_weights_ptr: *anyopaque,
    norm_weights_type: kt_type_t,
    token_embd_ptr: *anyopaque,
    token_embd_type: kt_type_t,
    pool: *worker_pool.WorkerPool,
};

pub const kt_mla_config_t = extern struct {
    hidden_size: usize,
    q_lora_rank: usize,
    num_heads: usize,
    nope_size: usize,
    rope_size: usize,
    kv_lora_rank: usize,
    layer_idx: c_int,
    pool: *worker_pool.WorkerPool,
    token_count_in_page: usize,
    max_qlen: usize,
    max_kvlen: usize,
    max_position_embeddings: usize,
    rope_scaling_factor: f64,
    rope_theta: f64,
    rope_scaling_beta_fast: f64,
    rope_scaling_beta_slow: f64,
    rope_scaling_mscale: f64,
    rope_scaling_mscale_all_dim: f64,
    rope_scaling_original_max_position_embeddings: f64,
    q_a_proj: *anyopaque,
    q_a_norm: *anyopaque,
    q_b_proj: *anyopaque,
    kv_a_proj_with_mqa: *anyopaque,
    kv_a_norm: *anyopaque,
    kv_b_proj: *anyopaque,
    o_proj: *anyopaque,
    q_a_proj_type: kt_type_t,
    q_a_norm_type: kt_type_t,
    q_b_proj_type: kt_type_t,
    kv_a_proj_with_mqa_type: kt_type_t,
    kv_a_norm_type: kt_type_t,
    kv_b_proj_type: kt_type_t,
    w_o_type: kt_type_t,
    input_type: kt_type_t,
    output_type: kt_type_t,
    m_block: usize,
    n_block: usize,
    page_count: usize,
};

pub const kt_moe_config_t = extern struct {
    expert_num: c_int,
    num_experts_per_tok: c_int,
    hidden_size: c_int,
    intermediate_size: c_int,
    layer_idx: c_int,
    pool: *worker_pool.WorkerPool,
    num_gpu_experts: c_int,
    gpu_experts_mask: [*]u8,
    physical_to_logical_map: *anyopaque,
    gate_proj: *amx.bf16,
    up_proj: *amx.bf16,
    down_proj: *amx.bf16,
    gate_scale: *f32,
    up_scale: *f32,
    down_scale: *f32,
    gate_zero: *anyopaque,
    up_zero: *anyopaque,
    down_zero: *anyopaque,
    quant_config: kt_quant_config_t,
    max_len: c_int,
    gate_projs: [*]*[*]amx.bf16,
    up_projs: [*]*[*]amx.bf16,
    down_projs: [*]*[*]amx.bf16,
    gate_scales: [*]*[*]f32,
    up_scales: [*]*[*]f32,
    down_scales: [*]*[*]f32,
    gate_zeros: ?*anyopaque,
    up_zeros: ?*anyopaque,
    down_zeros: ?*anyopaque,
    gate_bwd_projs: [*]*[*]amx.bf16,
    up_bwd_projs: [*]*[*]amx.bf16,
    down_bwd_projs: [*]*[*]amx.bf16,
    gate_bwd_scales: [*]*[*]f32,
    up_bwd_scales: [*]*[*]f32,
    down_bwd_scales: [*]*[*]f32,
    path: [*]const u8,
    save: c_int,
    load: c_int,
    share_backward_bb: c_int,
    share_cache_pool: c_int,
    m_block: c_int,
    group_min_len: c_int,
    group_max_len: c_int,
    gate_type: kt_type_t,
    up_type: kt_type_t,
    down_type: kt_type_t,
    hidden_type: kt_type_t,
    max_cache_depth: c_int,
    swiglu_limit: f32,
    swiglu_alpha: f32,
};

pub const KT_CPUInfer = opaque {};
pub const KT_WorkerPool = opaque {};
pub const KT_MOE = opaque {};
pub const KT_MLA = opaque {};
pub const KT_Gate = opaque {};
pub const KT_Linear = opaque {};
pub const KT_MLP = opaque {};

var g_cpu_variant: [16]u8 = undefined;
var g_cpu_variant_ptr: [*]const u8 = @ptrCast(&g_cpu_variant[0]);

// ============================================================================
// C API Implementation
// ============================================================================

export fn kt_version() [*]const u8 {
    return "0.6.1-zig";
}

export fn kt_get_cpu_variant() [*]const u8 {
    return g_cpu_variant_ptr;
}

fn detect_cpu_variant() void {
    const allocator = std.heap.page_allocator;
    const cpu = cpu_detect.detectCpu(allocator) catch {
        g_cpu_variant_ptr = "unknown";
        return;
    };

    const variant = cpu_detect.selectBestVariant(cpu);
    // Copy to global buffer
    @memcpy(g_cpu_variant[0..variant.len], variant);
    g_cpu_variant[variant.len] = 0;
}

export fn kt_ggml_init() void {
    // No-op in Zig version
}

export fn kt_bf16_to_f32(src: [*]const amx.bf16, dst: [*]f32, count: usize) void {
    for (0..count) |i| {
        dst[i] = amx.bf16_to_f32(src[i]);
    }
}

export fn kt_f32_to_bf16(src: [*]const f32, dst: [*]amx.bf16, count: usize) void {
    for (0..count) |i| {
        dst[i] = amx.f32_to_bf16(src[i]);
    }
}

export fn kt_worker_pool_new(thread_count: c_int) *KT_WorkerPool {
    const allocator = std.heap.page_allocator;
    const pool = allocator.create(worker_pool.WorkerPool) catch @panic("OOM");
    pool.* = worker_pool.WorkerPool.initSimple(allocator, @intCast(thread_count)) catch @panic("pool init");
    return @ptrCast(pool);
}

export fn kt_worker_pool_new_config(config: kt_worker_pool_config_t) *KT_WorkerPool {
    return kt_worker_pool_new(config.subpool_thread_count[0]);
}

export fn kt_worker_pool_free(pool: *KT_WorkerPool) void {
    const wp: *worker_pool.WorkerPool = @ptrCast(@alignCast(pool));
    wp.deinit();
    std.heap.page_allocator.destroy(wp);
}

export fn kt_cpuinfer_new(thread_count: c_int) *KT_CPUInfer {
    detect_cpu_variant();
    kt_ggml_init();

    const allocator = std.heap.page_allocator;
    const pool = allocator.create(worker_pool.WorkerPool) catch @panic("OOM");
    pool.* = worker_pool.WorkerPool.initSimple(allocator, @intCast(thread_count)) catch @panic("pool init");
    return @ptrCast(pool);
}

export fn kt_cpuinfer_new_config(config: kt_worker_pool_config_t) *KT_CPUInfer {
    return kt_cpuinfer_new(config.subpool_thread_count[0]);
}

export fn kt_cpuinfer_free(cpuinfer: *KT_CPUInfer) void {
    const pool: *worker_pool.WorkerPool = @ptrCast(@alignCast(cpuinfer));
    pool.deinit();
    std.heap.page_allocator.destroy(pool);
}

var g_submit_fn: ?*const fn (*anyopaque) callconv(.c) void = null;

fn submitAdapter(_: usize, arg: *anyopaque) void {
    if (g_submit_fn) |f| f(arg);
}

export fn kt_cpuinfer_submit(cpuinfer: *KT_CPUInfer, func: *const fn (*anyopaque) callconv(.c) void, arg: *anyopaque) void {
    const pool: *worker_pool.WorkerPool = @ptrCast(@alignCast(cpuinfer));
    g_submit_fn = func;
    pool.subpools[0].submit(submitAdapter, arg);
}

export fn kt_cpuinfer_sync(cpuinfer: *KT_CPUInfer, allow_n_pending: usize) void {
    // Wait for all tasks to complete
    _ = cpuinfer;
    _ = allow_n_pending;
    // In a real implementation, we'd wait on the task queue
}

export fn kt_cpuinfer_get_backend(cpuinfer: *KT_CPUInfer) *KT_WorkerPool {
    return @ptrCast(cpuinfer);
}

fn toMoeConfig(config: kt_moe_config_t) moe.MoeConfig {
    return .{
        .expert_num = config.expert_num,
        .num_experts_per_tok = config.num_experts_per_tok,
        .hidden_size = config.hidden_size,
        .intermediate_size = config.intermediate_size,
        .layer_idx = config.layer_idx,
        .pool = null,
        .gate_proj = @ptrCast(config.gate_proj),
        .up_proj = @ptrCast(config.up_proj),
        .down_proj = @ptrCast(config.down_proj),
        .gate_scale = @ptrCast(config.gate_scale),
        .up_scale = @ptrCast(config.up_scale),
        .down_scale = @ptrCast(config.down_scale),
        .max_len = config.max_len,
        .swiglu_limit = config.swiglu_limit,
        .swiglu_alpha = config.swiglu_alpha,
    };
}

export fn kt_moe_new(cpuinfer: *KT_CPUInfer, config: kt_moe_config_t) *KT_MOE {
    const pool: *worker_pool.WorkerPool = @ptrCast(@alignCast(cpuinfer));
    var cfg = toMoeConfig(config);
    cfg.pool = pool;
    const moe_inst = std.heap.page_allocator.create(moe.TpMoe) catch @panic("OOM");
    moe_inst.* = moe.TpMoe.init(cfg, std.heap.page_allocator) catch @panic("Failed to init MoE");
    return @ptrCast(moe_inst);
}

export fn kt_moe_free(moe_ptr: *KT_MOE) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    std.heap.page_allocator.destroy(m);
}

export fn kt_moe_warm_up(moe_ptr: *KT_MOE) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    // Warm up implementation would go here
    _ = m;
}

export fn kt_moe_load_weights(moe_ptr: *KT_MOE) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    m.loadWeights();
}

export fn kt_moe_load_weights_with_map(moe_ptr: *KT_MOE, physical_to_logical_map: [*]u64) void {
    kt_moe_load_weights(moe_ptr);
    _ = physical_to_logical_map;
}

export fn kt_moe_forward(
    moe_ptr: *KT_MOE,
    qlen: c_int,
    k: c_int,
    expert_ids: [*]const i64,
    weights: [*]const f32,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    incremental: c_int
) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    m.forward(
        @intCast(qlen),
        @intCast(k),
        expert_ids,
        weights,
        input,
        output,
        incremental != 0
    );
}

export fn kt_quantize_int8_per_row(src: [*]const amx.bf16, dst: [*]i8, scales: [*]f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        buffers.quantizeRowBF16ToInt8(src[r * cols ..][0..cols].ptr, dst[r * cols ..][0..cols].ptr, &scales[r], cols);
    }
}

export fn kt_dequantize_int8(src: [*]const i8, scales: [*]const f32, dst: [*]amx.bf16, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        buffers.dequantizeRowInt8ToBF16(src[r * cols ..][0..cols].ptr, scales[r], dst[r * cols ..][0..cols].ptr, cols);
    }
}

export fn kt_quantize_int4_gptq(src: [*]const amx.bf16, dst: [*]u8, scales: [*]f32, zeros: [*]u8, rows: usize, cols: usize, group_size: c_int) void {
    _ = src; _ = dst; _ = scales; _ = zeros; _ = rows; _ = cols; _ = group_size;
    // Implementation would go here
}

export fn kt_dequantize_int4_gptq(src: [*]const u8, scales: [*]const f32, zeros: [*]const u8, dst: [*]amx.bf16, rows: usize, cols: usize, group_size: c_int) void {
    _ = src; _ = scales; _ = zeros; _ = dst; _ = rows; _ = cols; _ = group_size;
    // Implementation would go here
}

export fn kt_quantize_fp8_e4m3(src: [*]const amx.bf16, dst: [*]u8, scales: [*]f32, rows: usize, cols: usize, block_size: c_int) void {
    _ = src; _ = dst; _ = scales; _ = rows; _ = cols; _ = block_size;
    // Implementation would go here
}

export fn kt_dequantize_fp8_e4m3(src: [*]const u8, scales: [*]const f32, dst: [*]amx.bf16, rows: usize, cols: usize, block_size: c_int) void {
    _ = src; _ = scales; _ = dst; _ = rows; _ = cols; _ = block_size;
    // Implementation would go here
}
