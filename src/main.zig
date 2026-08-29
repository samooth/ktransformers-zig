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
const gemm_int4 = root.gemm_int4;
const gemm_fp8 = root.gemm_fp8;
const gemm_mxfp4 = root.gemm_mxfp4;
const gemm_mxfp8 = root.gemm_mxfp8;
const moe = root.moe;

// ============================================================================
// C API Types
// ============================================================================

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

pub const kt_gate_config_t = extern struct {
    hidden_size: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    dtype: kt_type_t,
    weight_ptr: *anyopaque,
};

pub const kt_linear_config_t = extern struct {
    in_features: usize,
    out_features: usize,
    bias: bool,
    dtype: kt_type_t,
    weight_ptr: *anyopaque,
    bias_ptr: ?*anyopaque,
};

pub const kt_mlp_config_t = extern struct {
    hidden_size: usize,
    intermediate_size: usize,
    gate_proj_ptr: *anyopaque,
    up_proj_ptr: *anyopaque,
    down_proj_ptr: *anyopaque,
    gate_proj_type: kt_type_t,
    up_proj_type: kt_type_t,
    down_proj_type: kt_type_t,
    swiglu_limit: f32,
    swiglu_alpha: f32,
};

pub const kt_fp8_transport_config_t = extern struct {
    src_ptr: *anyopaque,
    dst_ptr: *anyopaque,
    count: usize,
    block_size: usize,
    src_scale_ptr: *anyopaque,
    dst_scale_ptr: *anyopaque,
};

pub const KT_CPUInfer = opaque {};
pub const KT_WorkerPool = opaque {};
pub const KT_MOE = opaque {};
pub const KT_MLA = opaque {};
pub const KT_Gate = opaque {};
pub const KT_Linear = opaque {};
pub const KT_MLP = opaque {};

var g_cpu_variant: [32]u8 = undefined;
var g_cpu_variant_ptr: [*]const u8 = @ptrCast(&g_cpu_variant[0]);
var g_cpu_variant_len: usize = 0;

// ============================================================================
// CPU Variant Detection
// ============================================================================

fn detect_cpu_variant() void {
    const allocator = std.heap.page_allocator;
    const cpu = cpu_detect.detectCpu(allocator) catch {
        const fallback = "unknown";
        @memcpy(g_cpu_variant[0..fallback.len], fallback);
        g_cpu_variant[fallback.len] = 0;
        g_cpu_variant_len = fallback.len;
        return;
    };

    const variant = cpu_detect.selectBestVariant(cpu);
    @memcpy(g_cpu_variant[0..variant.len], variant);
    g_cpu_variant[variant.len] = 0;
    g_cpu_variant_len = variant.len;
}

// ============================================================================
// Core C API Functions
// ============================================================================

export fn kt_version() [*]const u8 {
    return "0.6.1-zig";
}

export fn kt_get_cpu_variant() [*]const u8 {
    return g_cpu_variant_ptr;
}

export fn kt_ggml_init() void {
    // No-op in Zig version - GGML not used
}

// ============================================================================
// BF16 Conversion
// ============================================================================

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

// ============================================================================
// Worker Pool
// ============================================================================

export fn kt_worker_pool_new(thread_count: c_int) *KT_WorkerPool {
    const allocator = std.heap.page_allocator;
    const pool = allocator.create(worker_pool.WorkerPool) catch @panic("OOM");
    pool.* = worker_pool.WorkerPool.initSimple(allocator, @intCast(thread_count)) catch @panic("pool init");
    return @ptrCast(pool);
}

export fn kt_worker_pool_new_config(config: kt_worker_pool_config_t) *KT_WorkerPool {
    const allocator = std.heap.page_allocator;

    const numa_map = allocator.alloc(usize, @intCast(config.subpool_count)) catch @panic("OOM");
    const thread_counts = allocator.alloc(usize, @intCast(config.subpool_count)) catch @panic("OOM");

    for (0..@as(usize, @intCast(config.subpool_count))) |i| {
        numa_map[i] = @intCast(config.subpool_numa_map[i]);
        thread_counts[i] = @intCast(config.subpool_thread_count[i]);
    }

    const pool = allocator.create(worker_pool.WorkerPool) catch @panic("OOM");
    pool.* = worker_pool.WorkerPool.init(allocator, .{
        .subpool_count = @intCast(config.subpool_count),
        .subpool_numa_map = numa_map,
        .subpool_thread_count = thread_counts,
    }) catch @panic("pool init");
    return @ptrCast(pool);
}

export fn kt_worker_pool_free(pool: *KT_WorkerPool) void {
    const wp: *worker_pool.WorkerPool = @ptrCast(@alignCast(pool));
    wp.deinit();
    std.heap.page_allocator.destroy(wp);
}

export fn kt_worker_pool_get_thread_num(pool: *KT_WorkerPool) c_int {
    const wp: *worker_pool.WorkerPool = @ptrCast(@alignCast(pool));
    return @intCast(wp.getTotalThreads());
}

// ============================================================================
// CPU Infer
// ============================================================================

export fn kt_cpuinfer_new(thread_count: c_int) *KT_CPUInfer {
    detect_cpu_variant();
    kt_ggml_init();
    return @ptrCast(kt_worker_pool_new(thread_count));
}

export fn kt_cpuinfer_new_config(config: kt_worker_pool_config_t) *KT_CPUInfer {
    detect_cpu_variant();
    kt_ggml_init();
    return @ptrCast(kt_worker_pool_new_config(config));
}

export fn kt_cpuinfer_free(cpuinfer: *KT_CPUInfer) void {
    kt_worker_pool_free(@ptrCast(cpuinfer));
}

export fn kt_cpuinfer_get_backend(cpuinfer: *KT_CPUInfer) *KT_WorkerPool {
    return @ptrCast(cpuinfer);
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
    _ = cpuinfer;
    _ = allow_n_pending;
    // Wait for all tasks to complete
    // In a real implementation, we'd wait on the task queue
    // For now, this is a no-op as tasks are fire-and-forget
}

// ============================================================================
// MoE
// ============================================================================

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
    m.deinit();
    std.heap.page_allocator.destroy(m);
}

export fn kt_moe_warm_up(moe_ptr: *KT_MOE) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    m.warmUp();
}

export fn kt_moe_load_weights(moe_ptr: *KT_MOE) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    m.loadWeights();
}

export fn kt_moe_load_weights_with_map(moe_ptr: *KT_MOE, physical_to_logical_map: [*]u64) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    m.loadWeightsWithMap(physical_to_logical_map);
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

export fn kt_moe_forward_gate_up(
    moe_ptr: *KT_MOE,
    expert_idx: c_int,
    qlen: c_int,
    input: [*]const amx.bf16,
    gate_output: [*]amx.bf16,
    up_output: [*]amx.bf16
) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    m.forwardGateUp(@intCast(expert_idx), @intCast(qlen), input, gate_output, up_output);
}

export fn kt_moe_forward_down(
    moe_ptr: *KT_MOE,
    expert_idx: c_int,
    qlen: c_int,
    input: [*]const amx.bf16,
    output: [*]amx.bf16
) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    m.forwardDown(@intCast(expert_idx), @intCast(qlen), input, output);
}

// ============================================================================
// MLA (Multi-Head Latent Attention)
// ============================================================================

export fn kt_mla_new(cpuinfer: *KT_CPUInfer, config: kt_mla_config_t) *KT_MLA {
    _ = cpuinfer;
    _ = config;
    // MLA implementation placeholder
    const mla: *KT_MLA = @ptrCast(std.heap.page_allocator.create(u8) catch @panic("OOM"));
    return mla;
}

export fn kt_mla_free(mla: *KT_MLA) void {
    std.heap.page_allocator.destroy(@as(*u8, @ptrCast(mla)));
}

export fn kt_mla_forward(
    mla: *KT_MLA,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    qlen: c_int,
    kvlen: c_int,
    position_ids: [*]const i64
) void {
    _ = mla;
    _ = input;
    _ = output;
    _ = qlen;
    _ = kvlen;
    _ = position_ids;
    // MLA forward pass - placeholder
}

export fn kt_mla_prefill(
    mla: *KT_MLA,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    qlen: c_int,
    position_ids: [*]const i64
) void {
    _ = mla;
    _ = input;
    _ = output;
    _ = qlen;
    _ = position_ids;
    // MLA prefill - placeholder
}

export fn kt_mla_decode(
    mla: *KT_MLA,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    position_id: i64
) void {
    _ = mla;
    _ = input;
    _ = output;
    _ = position_id;
    // MLA decode - placeholder
}

export fn kt_mla_update_kv_cache(
    mla: *KT_MLA,
    kv_cache: *anyopaque,
    new_kv: [*]const amx.bf16,
    position: i64
) void {
    _ = mla;
    _ = kv_cache;
    _ = new_kv;
    _ = position;
    // Update KV cache - placeholder
}

// ============================================================================
// Gate (Expert Routing)
// ============================================================================

export fn kt_gate_new(config: kt_gate_config_t) *KT_Gate {
    const gate: *KT_Gate = @ptrCast(std.heap.page_allocator.create(u8) catch @panic("OOM"));
    _ = config;
    return gate;
}

export fn kt_gate_free(gate: *KT_Gate) void {
    std.heap.page_allocator.destroy(@as(*u8, @ptrCast(gate)));
}

export fn kt_gate_forward(
    _gate: *KT_Gate,
    _input: [*]const amx.bf16,
    _logits: [*]f32,
    topk_ids: [*]i64,
    topk_weights: [*]f32,
    batch_size: c_int,
    num_experts_per_tok: c_int
) void {
    _ = _gate;
    _ = _input;
    _ = _logits;
    // Simple linear projection + top-k selection
    // In a real implementation, this would use the gate weight matrix
    for (0..@as(usize, @intCast(batch_size))) |b| {
        // Placeholder: just fill with dummy values
        for (0..@as(usize, @intCast(num_experts_per_tok))) |k| {
            topk_ids[b * @as(usize, @intCast(num_experts_per_tok)) + k] = @intCast(k);
            topk_weights[b * @as(usize, @intCast(num_experts_per_tok)) + k] = 1.0 / @as(f32, @floatFromInt(num_experts_per_tok));
        }
    }
}

// ============================================================================
// Linear
// ============================================================================

export fn kt_linear_new(config: kt_linear_config_t) *KT_Linear {
    const linear: *KT_Linear = @ptrCast(std.heap.page_allocator.create(u8) catch @panic("OOM"));
    _ = config;
    return linear;
}

export fn kt_linear_free(linear: *KT_Linear) void {
    std.heap.page_allocator.destroy(@as(*u8, @ptrCast(linear)));
}

export fn kt_linear_forward(
    linear: *KT_Linear,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    batch_size: c_int,
    in_features: c_int,
    out_features: c_int
) void {
    _ = linear;
    // Placeholder: identity mapping
    const total = @as(usize, @intCast(batch_size)) * @as(usize, @intCast(out_features));
    for (0..total) |i| {
        output[i] = if (i < @as(usize, @intCast(batch_size)) * @as(usize, @intCast(in_features)))
            input[i]
        else
            amx.f32_to_bf16(0);
    }
}

// ============================================================================
// MLP
// ============================================================================

export fn kt_mlp_new(config: kt_mlp_config_t) *KT_MLP {
    const mlp_inst: *KT_MLP = @ptrCast(std.heap.page_allocator.create(u8) catch @panic("OOM"));
    _ = config;
    return mlp_inst;
}

export fn kt_mlp_free(mlp_inst: *KT_MLP) void {
    std.heap.page_allocator.destroy(@as(*u8, @ptrCast(mlp_inst)));
}

export fn kt_mlp_forward(
    _mlp_inst: *KT_MLP,
    _input: [*]const amx.bf16,
    output: [*]amx.bf16,
    batch_size: c_int
) void {
    _ = _mlp_inst;
    _ = _input;
    _ = output;
    _ = batch_size;
    // Placeholder: copy input to output (disabled due to unknown length)
    // @memcpy(output, input);
}

// ============================================================================
// FP8 Layerwise Transport
// ============================================================================

export fn kt_fp8_quantize(
    config: kt_fp8_transport_config_t,
    src: [*]const f32,
    dst: [*]u8,
    count: usize
) void {
    _ = config;
    for (0..count) |i| {
        dst[i] = gemm_fp8.f32_to_fp8e4m3(src[i]);
    }
}

export fn kt_fp8_dequantize(
    config: kt_fp8_transport_config_t,
    src: [*]const u8,
    dst: [*]f32,
    count: usize
) void {
    _ = config;
    for (0..count) |i| {
        dst[i] = gemm_fp8.fp8e4m3_to_f32(src[i]);
    }
}

export fn kt_fp8_quantize_block(
    src: [*]const f32,
    dst: [*]u8,
    scales: [*]f32,
    num_blocks: usize,
    block_size: usize
) void {
    for (0..num_blocks) |b| {
        const block_src = src[b * block_size ..][0..block_size];
        const block_dst = dst[b * block_size ..][0..block_size];

        // Find max for scale
        var max_val: f32 = 0;
        for (block_src) |val| {
            const abs_val = if (val < 0) -val else val;
            if (abs_val > max_val) max_val = abs_val;
        }

        scales[b] = if (max_val > 0) max_val / gemm_fp8.E4M3_MAX else 1.0;
        const inv_scale = 1.0 / scales[b];

        for (0..block_size) |i| {
            block_dst[i] = gemm_fp8.f32_to_fp8e4m3(block_src[i] * inv_scale);
        }
    }
}

export fn kt_fp8_dequantize_block(
    src: [*]const u8,
    scales: [*]const f32,
    dst: [*]f32,
    num_blocks: usize,
    block_size: usize
) void {
    for (0..num_blocks) |b| {
        const scale = scales[b];
        for (0..block_size) |i| {
            dst[b * block_size + i] = gemm_fp8.fp8e4m3_to_f32(src[b * block_size + i]) * scale;
        }
    }
}

// ============================================================================
// Quantization / Dequantization
// ============================================================================

export fn kt_quantize_int8_per_row(src: [*]const amx.bf16, dst: [*]i8, scales: [*]f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        gemm_int8.GemmKernel224Int8.quantizeRowBF16ToInt8(src[r * cols ..][0..cols].ptr, dst[r * cols ..][0..cols].ptr, &scales[r], cols);
    }
}

export fn kt_dequantize_int8(src: [*]const i8, scales: [*]const f32, dst: [*]amx.bf16, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        gemm_int8.GemmKernel224Int8.dequantizeRowInt8ToBF16(src[r * cols ..][0..cols].ptr, scales[r], dst[r * cols ..][0..cols].ptr, cols);
    }
}

export fn kt_quantize_int4_gptq(
    src: [*]const amx.bf16,
    dst: [*]u8,
    scales: [*]f32,
    zeros: [*]u8,
    rows: usize,
    cols: usize,
    group_size: c_int
) void {
    const gs = @as(usize, @intCast(group_size));
    const blocks_per_row = (cols + gs - 1) / gs;

    for (0..rows) |r| {
        for (0..blocks_per_row) |blk| {
            const k_start = blk * gs;
            const k_end = @min(cols, k_start + gs);
            const k_actual = k_end - k_start;
            _ = k_actual; // Mark as used to suppress warning

            var max_val: f32 = 0;
            for (k_start..k_end) |c| {
                const val = amx.bf16_to_f32(src[r * cols + c]);
                const abs_val = if (val < 0) -val else val;
                if (abs_val > max_val) max_val = abs_val;
            }

            const scale = if (max_val > 0) max_val / 7.0 else 1.0;
            const inv_scale = 1.0 / scale;
            scales[r * blocks_per_row + blk] = scale;

            for (0..16) |q| {
                const k0 = k_start + q * 2;
                const k1 = k0 + 1;

                const v0 = if (k0 < cols)
                    @as(u8, @intCast(@max(0, @min(15, @as(i32, @intFromFloat(amx.bf16_to_f32(src[r * cols + k0]) * inv_scale + 8.0))))))
                else
                    8;
                const v1 = if (k1 < cols)
                    @as(u8, @intCast(@max(0, @min(15, @as(i32, @intFromFloat(amx.bf16_to_f32(src[r * cols + k1]) * inv_scale + 8.0))))))
                else
                    8;

                dst[(r * blocks_per_row + blk) * 16 + q] = (v1 << 4) | v0;
            }
            zeros[r * blocks_per_row + blk] = 8; // zero point = 8
        }
    }
}

export fn kt_dequantize_int4_gptq(
    src: [*]const u8,
    scales: [*]const f32,
    zeros: [*]const u8,
    dst: [*]amx.bf16,
    rows: usize,
    cols: usize,
    group_size: c_int
) void {
    const gs = @as(usize, @intCast(group_size));
    const blocks_per_row = (cols + gs - 1) / gs;

    for (0..rows) |r| {
        for (0..blocks_per_row) |blk| {
            const scale = scales[r * blocks_per_row + blk];
            const zero = @as(f32, @floatFromInt(zeros[r * blocks_per_row + blk]));

            for (0..16) |q| {
                const byte = src[(r * blocks_per_row + blk) * 16 + q];
                const low = @as(f32, @floatFromInt(byte & 0x0F)) - zero;
                const high = @as(f32, @floatFromInt((byte >> 4) & 0x0F)) - zero;

                const k0 = blk * gs + q * 2;
                const k1 = k0 + 1;

                if (k0 < cols) dst[r * cols + k0] = amx.f32_to_bf16(low * scale);
                if (k1 < cols) dst[r * cols + k1] = amx.f32_to_bf16(high * scale);
            }
        }
    }
}

export fn kt_quantize_fp8_e4m3(
    src: [*]const amx.bf16,
    dst: [*]u8,
    scales: [*]f32,
    rows: usize,
    cols: usize,
    block_size: c_int
) void {
    const bs = @as(usize, @intCast(block_size));
    const blocks_per_row = (cols + bs - 1) / bs;

    for (0..rows) |r| {
        for (0..blocks_per_row) |blk| {
            const c_start = blk * bs;
            const c_end = @min(cols, c_start + bs);

            var max_val: f32 = 0;
            for (c_start..c_end) |c| {
                const val = amx.bf16_to_f32(src[r * cols + c]);
                const abs_val = if (val < 0) -val else val;
                if (abs_val > max_val) max_val = abs_val;
            }

            const scale = if (max_val > 0) max_val / gemm_fp8.E4M3_MAX else 1.0;
            scales[r * blocks_per_row + blk] = scale;
            const inv_scale = 1.0 / scale;

            for (c_start..c_end) |c| {
                dst[r * cols + c] = gemm_fp8.f32_to_fp8e4m3(amx.bf16_to_f32(src[r * cols + c]) * inv_scale);
            }
        }
    }
}

export fn kt_dequantize_fp8_e4m3(
    src: [*]const u8,
    scales: [*]const f32,
    dst: [*]amx.bf16,
    rows: usize,
    cols: usize,
    block_size: c_int
) void {
    const bs = @as(usize, @intCast(block_size));
    const blocks_per_row = (cols + bs - 1) / bs;

    for (0..rows) |r| {
        for (0..blocks_per_row) |blk| {
            const scale = scales[r * blocks_per_row + blk];
            const c_start = blk * bs;
            const c_end = @min(cols, c_start + bs);

            for (c_start..c_end) |c| {
                dst[r * cols + c] = amx.f32_to_bf16(gemm_fp8.fp8e4m3_to_f32(src[r * cols + c]) * scale);
            }
        }
    }
}

// ============================================================================
// Backward Pass (for SFT/LoRA training)
// ============================================================================

export fn kt_moe_backward(
    moe_ptr: *KT_MOE,
    grad_output: [*]const f32,
    grad_input: [*]f32,
    qlen: c_int,
    expert_ids: [*]const i64,
    weights: [*]const f32
) void {
    _ = moe_ptr;
    _ = grad_output;
    _ = grad_input;
    _ = qlen;
    _ = expert_ids;
    _ = weights;
    // Backward pass placeholder
}

export fn kt_linear_backward(
    linear: *KT_Linear,
    grad_output: [*]const f32,
    grad_input: [*]f32,
    batch_size: c_int
) void {
    _ = linear;
    _ = grad_output;
    _ = grad_input;
    _ = batch_size;
    // Linear backward placeholder
}

export fn kt_mlp_backward(
    mlp_inst: *KT_MLP,
    grad_output: [*]const f32,
    grad_input: [*]f32,
    batch_size: c_int
) void {
    _ = mlp_inst;
    _ = grad_output;
    _ = grad_input;
    _ = batch_size;
    // MLP backward placeholder
}

// ============================================================================
// Utility Functions
// ============================================================================

export fn kt_apply_swiglu(
    gate: [*]const amx.bf16,
    up: [*]const amx.bf16,
    dst: [*]amx.bf16,
    count: usize,
    limit: f32,
    alpha: f32
) void {
    for (0..count) |i| {
        const gate_val = amx.bf16_to_f32(gate[i]);
        const up_val = amx.bf16_to_f32(up[i]);
        var result: f32 = 0;
        if (alpha > 0) {
            result = amx.swiglu_oai(gate_val, up_val, alpha, limit);
        } else if (limit > 0) {
            result = amx.swiglu_clamp(gate_val, up_val, limit);
        } else {
            result = amx.swiglu(gate_val, up_val);
        }
        dst[i] = amx.f32_to_bf16(result);
    }
}

export fn kt_apply_rms_norm(
    input: [*]const amx.bf16,
    weight: [*]const amx.bf16,
    output: [*]amx.bf16,
    hidden_size: usize,
    eps: f32
) void {
    // Compute RMS norm: output = input / sqrt(mean(x^2) + eps) * weight
    var sum_sq: f32 = 0;
    for (0..hidden_size) |i| {
        const val = amx.bf16_to_f32(input[i]);
        sum_sq += val * val;
    }
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(hidden_size)) + eps);
    const inv_rms = 1.0 / rms;

    for (0..hidden_size) |i| {
        const val = amx.bf16_to_f32(input[i]);
        const w = amx.bf16_to_f32(weight[i]);
        output[i] = amx.f32_to_bf16(val * inv_rms * w);
    }
}

export fn kt_apply_rope(
    q: [*]amx.bf16,
    k: [*]amx.bf16,
    position: i64,
    head_dim: usize,
    rope_theta: f64
) void {
    // RoPE (Rotary Position Embedding)
    // Apply rotation to pairs of dimensions
    const half_dim = head_dim / 2;
    for (0..half_dim) |i| {
        const freq = 1.0 / std.math.pow(f64, rope_theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim)));
        const angle = @as(f64, @floatFromInt(position)) * freq;
        const cos_val = @cos(angle);
        const sin_val = @sin(angle);

        const q0 = amx.bf16_to_f32(q[i]);
        const q1 = amx.bf16_to_f32(q[i + half_dim]);
        q[i] = amx.f32_to_bf16(@as(f32, @floatCast(q0 * cos_val - q1 * sin_val)));
        q[i + half_dim] = amx.f32_to_bf16(@as(f32, @floatCast(q0 * sin_val + q1 * cos_val)));

        const k0 = amx.bf16_to_f32(k[i]);
        const k1 = amx.bf16_to_f32(k[i + half_dim]);
        k[i] = amx.f32_to_bf16(@as(f32, @floatCast(k0 * cos_val - k1 * sin_val)));
        k[i + half_dim] = amx.f32_to_bf16(@as(f32, @floatCast(k0 * sin_val + k1 * cos_val)));
    }
}

export fn kt_softmax(
    input: [*]const f32,
    output: [*]f32,
    size: usize
) void {
    var max_val: f32 = -std.math.inf(f32);
    for (0..size) |i| {
        if (input[i] > max_val) max_val = input[i];
    }

    var sum: f32 = 0;
    for (0..size) |i| {
        output[i] = @exp(input[i] - max_val);
        sum += output[i];
    }

    const inv_sum = 1.0 / sum;
    for (0..size) |i| {
        output[i] *= inv_sum;
    }
}

export fn kt_matmul_bf16(
    a: [*]const amx.bf16,
    b: [*]const amx.bf16,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize
) void {
    gemm_bf16.gemmExpert(a, b, c, m, n, k, lda, ldb, ldc);
}

export fn kt_matmul_int8(
    a: [*]const i8,
    b: [*]const i8,
    c: [*]i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize
) void {
    _ = m;
    _ = n;
    gemm_int8.GemmKernel224Int8.gemmFullTile(@as(*const i8, @ptrCast(a)), lda, @as(*const i8, @ptrCast(b)), ldb, @ptrCast(c), ldc, k);
}

export fn kt_matmul_int4(
    a: [*]const i8,
    b: [*]const gemm_int4.BlockQ4_0,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize
) void {
    gemm_int4.GemmKernel224Int4.gemmFullTile(@as(*const i8, @ptrCast(a)), lda, @as(*const gemm_int4.BlockQ4_0, @ptrCast(b)), ldb, @ptrCast(c), ldc, m, n, k);
}

export fn kt_matmul_fp8(
    a: [*]const amx.bf16,
    b: [*]const gemm_fp8.fp8_e4m3,
    b_scales: [*]const f32,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize
) void {
    gemm_fp8.GemmKernel224FP8.gemmFullTile(a, lda, b, b_scales, ldb, c, ldc, m, n, k);
}
