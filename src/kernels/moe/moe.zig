// MoE (Mixture of Experts) implementation for ktransformers-zig
// Integrates AMX kernels with tensor-parallel MoE

const std = @import("std");
const amx = @import("../arch/amx.zig");
const buffers = @import("../amx/buffers.zig");
const gemm_bf16 = @import("../amx/gemm_224_bf16.zig");
const gemm_int8 = @import("../amx/gemm_224_int8.zig");
const worker_pool = @import("../../runtime/worker_pool.zig");
const task_queue = @import("../../runtime/task_queue.zig");
const memory = @import("../../runtime/memory.zig");

// ============================================================================
// Configuration Structures
// ============================================================================

pub const QuantConfig = struct {
    quant_method: []const u8 = "",
    bits: i32 = 0,
    group_size: i32 = 0,
    zero_point: bool = false,
    per_channel: bool = false,
};

pub const MoeConfig = struct {
    expert_num: i32 = 0,
    num_experts_per_tok: i32 = 0,
    hidden_size: i32 = 0,
    intermediate_size: i32 = 0,

    layer_idx: i32 = 0,
    pool: ?*worker_pool.WorkerPool = null,

    // SGLang offload
    num_gpu_experts: i32 = 0,
    gpu_experts_mask: ?[*]u8 = null,
    physical_to_logical_map: ?*anyopaque = null,

    // Weights (BF16 format)
    gate_proj: ?[*]amx.bf16 = null,
    up_proj: ?[*]amx.bf16 = null,
    down_proj: ?[*]amx.bf16 = null,

    // Quantized weights (INT8)
    gate_proj_int8: ?[*]i8 = null,
    up_proj_int8: ?[*]i8 = null,
    down_proj_int8: ?[*]i8 = null,
    gate_scale: ?[*]f32 = null,
    up_scale: ?[*]f32 = null,
    down_scale: ?[*]f32 = null,

    // Quantization config
    quant_config: QuantConfig = QuantConfig{},

    // For TP/AMX
    max_len: i32 = 0,
    gate_projs: []*[*]amx.bf16 = &.{},
    up_projs: []*[*]amx.bf16 = &.{},
    down_projs: []*[*]amx.bf16 = &.{},
    gate_scales: []*[*]f32 = &.{},
    up_scales: []*[*]f32 = &.{},
    down_scales: []*[*]f32 = &.{},

    // Backward weights
    gate_bwd_projs: []*[*]amx.bf16 = &.{},
    up_bwd_projs: []*[*]amx.bf16 = &.{},
    down_bwd_projs: []*[*]amx.bf16 = &.{},
    gate_bwd_scales: []*[*]f32 = &.{},
    up_bwd_scales: []*[*]f32 = &.{},
    down_bwd_scales: []*[*]f32 = &.{},

    path: []const u8 = "",
    save: bool = false,
    load: bool = false,
    share_backward_bb: bool = false,
    share_cache_pool: bool = false,
    m_block: i32 = 4,
    group_min_len: i32 = 0,
    group_max_len: i32 = 0,

    swiglu_limit: f32 = 0.0,
    swiglu_alpha: f32 = 0.0,

    pub fn maxPossibleQlen(self: MoeConfig) i32 {
        return @max(self.max_len, self.group_max_len);
    }

    pub fn shouldSkipExpert(self: MoeConfig, expert_id: i64) bool {
        return expert_id < 0 or expert_id >= self.expert_num or
            (self.gpu_experts_mask != null and self.gpu_experts_mask[@intCast(expert_id)] != 0);
    }
};

// ============================================================================
// Expert Router
// ============================================================================

pub fn routeExperts(
    input: [*]const amx.bf16,  // [qlen, hidden_size]
    gate_proj: [*]const amx.bf16,  // [expert_num, hidden_size]
    qlen: usize,
    hidden_size: usize,
    expert_num: usize,
    num_experts_per_tok: usize,
    topk_ids: [*]i64,
    topk_weights: [*]f32,
    pool: *worker_pool.WorkerPool,
) void {
    // Compute gate scores: input @ gate_proj^T
    // Then select top-k experts per token

    const total_work = qlen;
    _ = @max(1, total_work / pool.getTotalThreads());

    // Simple sequential implementation for now
    for (0..qlen)  | i |  {
        var scores = std.heap.page_allocator.alloc(f32, expert_num) catch @panic("OOM");
        defer std.heap.page_allocator.free(scores);

        for (0..expert_num)  | e |  {
            var sum: f32 = 0;
            for (0..hidden_size)  | h |  {
                sum += amx.bf16_to_f32(input[i * hidden_size + h]) *
                       amx.bf16_to_f32(gate_proj[e * hidden_size + h]);
            }
            scores[e] = sum;
        }

        // Find top-k
        for (0..num_experts_per_tok)  | k |  {
            var best_idx: i64 = -1;
            var best_val: f32 = -std.math.inf(f32);
            for (0..expert_num)  | e |  {
                if (scores[e] > best_val) {
                    best_val = scores[e];
                    best_idx = @intCast(e);
                }
            }
            topk_ids[i * num_experts_per_tok + k] = best_idx;
            topk_weights[i * num_experts_per_tok + k] = best_val;
            if (best_idx >= 0) scores[@intCast(best_idx)] = -std.math.inf(f32);
        }
    }
}

// ============================================================================
// MoE Expert Computation
// ============================================================================

/// Compute single expert forward pass
pub fn computeExpert(
    input: [*]const amx.bf16,    // [m, hidden_size]
    gate_proj: [*]const amx.bf16, // [intermediate_size, hidden_size]
    up_proj: [*]const amx.bf16,   // [intermediate_size, hidden_size]
    down_proj: [*]const amx.bf16, // [hidden_size, intermediate_size]
    _gate_scale: ?*const f32,
    _up_scale: ?*const f32,
    _down_scale: ?*const f32,
    m: usize,
    hidden_size: usize,
    intermediate_size: usize,
    output: [*]f32,              // [m, hidden_size] (FP32 accumulation)
    swiglu_limit: f32,
    swiglu_alpha: f32,
) void {
    // Silence unused parameter warnings
    _ = _gate_scale; _ = _up_scale; _ = _down_scale;

    // Allocate intermediate buffers
    const gate_output = std.heap.page_allocator.alloc(amx.bf16, m * intermediate_size) catch @panic("OOM");
    const up_output = std.heap.page_allocator.alloc(amx.bf16, m * intermediate_size) catch @panic("OOM");
    const down_output = std.heap.page_allocator.alloc(amx.bf16, m * hidden_size) catch @panic("OOM");
    defer {
        std.heap.page_allocator.free(gate_output);
        std.heap.page_allocator.free(up_output);
        std.heap.page_allocator.free(down_output);
    }

    // Gate projection: input @ gate_proj^T -> [m, intermediate_size]
    gemm_bf16.gemmExpert(input, gate_proj, gate_output, m, intermediate_size, hidden_size,
        hidden_size, intermediate_size, intermediate_size);

    // Up projection: input @ up_proj^T -> [m, intermediate_size]
    gemm_bf16.gemmExpert(input, up_proj, up_output, m, intermediate_size, hidden_size,
        hidden_size, intermediate_size, intermediate_size);

    // SwiGLU activation
    gemm_bf16.GemmKernel224BF.applySwiGLU(gate_output, up_output, gate_output, m, intermediate_size,
        swiglu_limit, swiglu_alpha);

    // Down projection: gate_output @ down_proj^T -> [m, hidden_size]
    gemm_bf16.gemmExpert(gate_output, down_proj, down_output, m, hidden_size, intermediate_size,
        intermediate_size, hidden_size, hidden_size);

    // Accumulate to output (FP32)
    for (0..m)  | i |  {
        for (0..hidden_size)  | h |  {
            output[i * hidden_size + h] += amx.bf16_to_f32(down_output[i * hidden_size + h]);
        }
    }
}

// ============================================================================
// Tensor-Parallel MoE
// ============================================================================

pub const TpMoe = struct {
    config: MoeConfig,
    tp_configs: []MoeConfig,
    tp_count: usize,
    experts: []ExpertData,
    weights_loaded: bool = false,

    pub const ExpertData = struct {
        // Packed weight buffers
        gate_bf16: buffers.BufferA(amx.bf16),
        up_bf16: buffers.BufferA(amx.bf16),
        down_bf16: buffers.BufferA(amx.bf16),

        // INT8 quantized
        gate_int8: gemm_int8.Int8BufferB,
        up_int8: gemm_int8.Int8BufferB,
        down_int8: gemm_int8.Int8BufferB,

        // Temporary buffers
        gate_buf: buffers.BufferC(f32),
        up_buf: buffers.BufferC(f32),
        down_buf: buffers.BufferC(f32),
    };

    pub fn init(config: MoeConfig, allocator: std.mem.Allocator) !TpMoe {
        const pool = config.pool orelse return error.NoWorkerPool;
        const tp_count = pool.config.subpool_count;

        // Validate intermediate_size divisible by tp_count
        if (@as(usize, @intCast(config.intermediate_size)) % tp_count != 0) {
            return error.InvalidConfig;
        }

        var tp_configs = try allocator.alloc(MoeConfig, tp_count);
        var experts = try allocator.alloc(ExpertData, @intCast(config.expert_num));

        // Create TP configs
        for (0..tp_count)  | i |  {
            tp_configs[i] = config;
            tp_configs[i].intermediate_size = @divTrunc(tp_configs[i].intermediate_size, @as(i32, @intCast(tp_count)));
        }

        // Allocate expert buffers
        for (0..@as(usize, @intCast(config.expert_num))) |e| {
            const expert_config = tp_configs[0]; // Use first TP config for sizing

            experts[e] = ExpertData{
                .gate_bf16 = buffers.BufferA(amx.bf16).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(config.hidden_size)),
                    undefined, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.K_STEP,
                    gemm_bf16.GemmKernel224BF.K_BLOCK,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .up_bf16 = buffers.BufferA(amx.bf16).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(config.hidden_size)),
                    undefined, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.K_STEP,
                    gemm_bf16.GemmKernel224BF.K_BLOCK,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .down_bf16 = buffers.BufferA(amx.bf16).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(expert_config.intermediate_size)),
                    undefined, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.K_STEP,
                    gemm_bf16.GemmKernel224BF.K_BLOCK,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .gate_int8 = gemm_int8.Int8BufferB.init(
                    @as(usize, @intCast(expert_config.intermediate_size)), @as(usize, @intCast(config.hidden_size)),
                    undefined, gemm_int8.GemmKernel224Int8.N_STEP,
                    gemm_int8.GemmKernel224Int8.K_STEP,
                    gemm_int8.GemmKernel224Int8.K_BLOCK,
                    gemm_int8.GemmKernel224Int8.N_BLOCK
                ),
                .up_int8 = gemm_int8.Int8BufferB.init(
                    @as(usize, @intCast(expert_config.intermediate_size)), @as(usize, @intCast(config.hidden_size)),
                    undefined, gemm_int8.GemmKernel224Int8.N_STEP,
                    gemm_int8.GemmKernel224Int8.K_STEP,
                    gemm_int8.GemmKernel224Int8.K_BLOCK,
                    gemm_int8.GemmKernel224Int8.N_BLOCK
                ),
                .down_int8 = gemm_int8.Int8BufferB.init(
                    @as(usize, @intCast(config.hidden_size)), @as(usize, @intCast(expert_config.intermediate_size)),
                    undefined, gemm_int8.GemmKernel224Int8.N_STEP,
                    gemm_int8.GemmKernel224Int8.K_STEP,
                    gemm_int8.GemmKernel224Int8.K_BLOCK,
                    gemm_int8.GemmKernel224Int8.N_BLOCK
                ),
                .gate_buf = buffers.BufferC(f32).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(expert_config.intermediate_size)),
                    undefined, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.N_STEP,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .up_buf = buffers.BufferC(f32).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(expert_config.intermediate_size)),
                    undefined, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.N_STEP,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .down_buf = buffers.BufferC(f32).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(config.hidden_size)),
                    undefined, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.N_STEP,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
            };
        }

        return TpMoe{
            .config = config,
            .tp_configs = tp_configs,
            .tp_count = tp_count,
            .experts = experts,
        };
    }

    pub fn loadWeights(self: *TpMoe) void {
        // Load weights into expert buffers
        for (0..@as(usize, @intCast(self.config.expert_num)))  | e |  {
            const expert = &self.experts[e];

            if (self.config.gate_proj_int8 != null) {
                // Load INT8 quantized weights
                if (self.config.gate_proj) |gp| {
                    expert.gate_int8.fromMatBF16(
                        gp + e * @as(usize, @intCast(self.config.intermediate_size)) * @as(usize, @intCast(self.config.hidden_size)),
                        @intCast(self.config.hidden_size)
                    );
                }
                if (self.config.up_proj) |up| {
                    expert.up_int8.fromMatBF16(
                        up + e * @as(usize, @intCast(self.config.intermediate_size)) * @as(usize, @intCast(self.config.hidden_size)),
                        @intCast(self.config.hidden_size)
                    );
                }
                if (self.config.down_proj) |dp| {
                    expert.down_int8.fromMatBF16(
                        dp + e * @as(usize, @intCast(self.config.hidden_size)) * @as(usize, @intCast(self.config.intermediate_size)),
                        @intCast(self.config.intermediate_size)
                    );
                }
            } else if (self.config.gate_proj != null) {
                // Load BF16 weights and pack
                for (0..self.tp_count)  | tp |  {
                    const tp_config = self.tp_configs[tp];
                    const _gate_start = e * @as(usize, @intCast(self.config.intermediate_size)) * @as(usize, @intCast(self.config.hidden_size)) + tp * @as(usize, @intCast(tp_config.intermediate_size)) * @as(usize, @intCast(tp_config.hidden_size));
                    const _up_start = e * @as(usize, @intCast(self.config.intermediate_size)) * @as(usize, @intCast(self.config.hidden_size)) + tp * @as(usize, @intCast(tp_config.intermediate_size)) * @as(usize, @intCast(tp_config.hidden_size));
                    const _down_start = e * @as(usize, @intCast(self.config.hidden_size)) * @as(usize, @intCast(self.config.intermediate_size)) + tp * @as(usize, @intCast(tp_config.hidden_size)) * @as(usize, @intCast(tp_config.intermediate_size));
                    _ = _up_start; _ = _down_start;

                    expert.gate_bf16.fromMat(@as(usize, @intCast(tp_config.maxPossibleQlen())),
                        self.config.gate_proj orelse @as([*]amx.bf16, @ptrFromInt(_gate_start)), @as(usize, @intCast(tp_config.hidden_size)),
                        gemm_bf16.GemmKernel224BF.M_STEP, gemm_bf16.GemmKernel224BF.K_STEP,
                        gemm_bf16.GemmKernel224BF.K_BLOCK);
                    // Similar for up_proj, down_proj
                }
            }
        }
        self.weights_loaded = true;
    }

    pub fn forward(
        self: *TpMoe,
        qlen: usize,
        k: i32,
        expert_ids: [*]const i64,
        _weights: [*]const f32,
        input: [*]const amx.bf16,
        output: [*]amx.bf16,
        _incremental: bool
    ) void {
        _ = _weights; _ = _incremental;
        if (!self.weights_loaded) {
            @panic("Weights not loaded");
        }

        // Route tokens to experts
        // For each expert, gather tokens and compute
        // Merge results across TP ranks

        // Simplified implementation - route and compute per expert
        var expert_tokens = std.heap.page_allocator.alloc(usize, @intCast(self.config.expert_num)) catch @panic("OOM");
        defer std.heap.page_allocator.free(expert_tokens);

        // Count tokens per expert
        {
            var i: usize = 0;
            while (i < expert_tokens.len) : (i += 1) {
                expert_tokens[i] = @as(usize, 0);
            }
        }
        for (0..qlen)  | i |  {
            for (0..@as(usize, @intCast(k)))  | j |  {
                const eid = expert_ids[i * @as(usize, @intCast(k)) + j];
                if (eid >= 0 and eid < @as(i64, self.config.expert_num)) {
                    expert_tokens[@intCast(eid)] += 1;
                }
            }
        }

        // Allocate output buffer
        const output_f32 = std.heap.page_allocator.alloc(f32, qlen * @as(usize, @intCast(self.config.hidden_size))) catch @panic("OOM");
        defer std.heap.page_allocator.free(output_f32);
        @memset(output_f32, 0);

        // Process each expert
        for (0..@as(usize, @intCast(self.config.expert_num)))  | e |  {
            const count = expert_tokens[e];
            if (count == 0) continue;

            const _expert = &self.experts[e];
            _ = _expert;

            // Gather input tokens for this expert
            const expert_input = std.heap.page_allocator.alloc(amx.bf16, count * @as(usize, @intCast(self.config.hidden_size))) catch @panic("OOM");
            defer std.heap.page_allocator.free(expert_input);

            var token_idx: usize = 0;
            for (0..qlen)  | i |  {
                for (0..@as(usize, @intCast(k)))  | j |  {
                    const eid = expert_ids[i * @as(usize, @intCast(k)) + j];
                    if (eid == @as(i64, @intCast(e))) {
                        @memcpy(expert_input[token_idx * @as(usize, @intCast(self.config.hidden_size)) ..][0..@as(usize, @intCast(self.config.hidden_size))],
                            input[i * @as(usize, @intCast(self.config.hidden_size)) ..][0..@as(usize, @intCast(self.config.hidden_size))]);
                        token_idx += 1;
                    }
                }
            }

            // Compute expert
            const gate_scale_ptr = if (self.config.gate_scale) |gs| &gs[e * @as(usize, @intCast(self.config.intermediate_size))] else null;
            const up_scale_ptr = if (self.config.up_scale) |us| &us[e * @as(usize, @intCast(self.config.intermediate_size))] else null;
            const down_scale_ptr = if (self.config.down_scale) |ds| &ds[e * @as(usize, @intCast(self.config.hidden_size))] else null;
                // Placeholder: expert computation disabled for now
                // expert_input is reserved for future use
                _ = gate_scale_ptr;
                _ = up_scale_ptr;
                _ = down_scale_ptr;
        }

        // Convert FP32 output to BF16
        for (0..qlen * @as(usize, @intCast(self.config.hidden_size))) |i| {
            output[i] = amx.f32_to_bf16(output_f32[i]);
        }
    }

    pub fn deinit(self: *TpMoe) void {
        // Placeholder: cleanup resources
        _ = self;
    }

    pub fn warmUp(self: *TpMoe) void {
        // Placeholder: warm up the model
        _ = self;
    }

    pub fn loadWeightsWithMap(self: *TpMoe, physical_to_logical_map: [*]u64) void {
        // Placeholder: load weights with physical-to-logical mapping
        _ = self;
        _ = physical_to_logical_map;
    }

    pub fn forwardGateUp(
        self: *TpMoe,
        expert_idx: usize,
        qlen: usize,
        input: [*]const amx.bf16,
        gate_output: [*]amx.bf16,
        up_output: [*]amx.bf16
    ) void {
        // Placeholder: forward pass for gate and up projections
        _ = self;
        _ = expert_idx;
        _ = qlen;
        _ = input;
        _ = gate_output;
        _ = up_output;
    }

    pub fn forwardDown(
        self: *TpMoe,
        expert_idx: usize,
        qlen: usize,
        input: [*]const amx.bf16,
        output: [*]amx.bf16
    ) void {
        // Placeholder: forward pass for down projection
        _ = self;
        _ = expert_idx;
        _ = qlen;
        _ = input;
        _ = output;
    }
};