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
// BF16 Weight Storage (per-expert, per-TP-rank)
// ============================================================================
//
// The ExpertData struct has no field for BF16 weights (only Int8BufferB for
// INT8 quantized weights and BufferA for activations). To avoid changing the
// struct, we store BF16 weight pointers in module-level arrays indexed by
// (expert_idx, tp_rank). Allocated in loadWeights, freed in deinit.
//
// Layout per expert per TP rank:
//   gate_bf16_storage[expert * tp_count + tp] -> [*]amx.bf16, [intermediate_size/tp_count, hidden_size]
//   up_bf16_storage[expert * tp_count + tp]   -> [*]amx.bf16, [intermediate_size/tp_count, hidden_size]
//   down_bf16_storage[expert * tp_count + tp] -> [*]amx.bf16, [hidden_size, intermediate_size/tp_count]

var gate_bf16_storage: [*]align(64) amx.bf16 = @alignCast(@as([*]align(64) amx.bf16, @ptrFromInt(64)));
var up_bf16_storage: [*]align(64) amx.bf16 = @alignCast(@as([*]align(64) amx.bf16, @ptrFromInt(64)));
var down_bf16_storage: [*]align(64) amx.bf16 = @alignCast(@as([*]align(64) amx.bf16, @ptrFromInt(64)));
var gate_bf16_alloc: []align(64) amx.bf16 = &[_]amx.bf16{};
var up_bf16_alloc: []align(64) amx.bf16 = &[_]amx.bf16{};
var down_bf16_alloc: []align(64) amx.bf16 = &[_]amx.bf16{};
var bf16_weights_alloced: usize = 0;
var bf16_storage_expert_count: usize = 0;
var bf16_storage_tp_count: usize = 0;

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
                    null, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.K_STEP,
                    gemm_bf16.GemmKernel224BF.K_BLOCK,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .up_bf16 = buffers.BufferA(amx.bf16).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(config.hidden_size)),
                    null, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.K_STEP,
                    gemm_bf16.GemmKernel224BF.K_BLOCK,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .down_bf16 = buffers.BufferA(amx.bf16).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(expert_config.intermediate_size)),
                    null, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.K_STEP,
                    gemm_bf16.GemmKernel224BF.K_BLOCK,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .gate_int8 = gemm_int8.Int8BufferB.init(
                    @as(usize, @intCast(expert_config.intermediate_size)), @as(usize, @intCast(config.hidden_size)),
                    null, gemm_int8.GemmKernel224Int8.N_STEP,
                    gemm_int8.GemmKernel224Int8.K_STEP,
                    gemm_int8.GemmKernel224Int8.K_BLOCK,
                    gemm_int8.GemmKernel224Int8.N_BLOCK
                ),
                .up_int8 = gemm_int8.Int8BufferB.init(
                    @as(usize, @intCast(expert_config.intermediate_size)), @as(usize, @intCast(config.hidden_size)),
                    null, gemm_int8.GemmKernel224Int8.N_STEP,
                    gemm_int8.GemmKernel224Int8.K_STEP,
                    gemm_int8.GemmKernel224Int8.K_BLOCK,
                    gemm_int8.GemmKernel224Int8.N_BLOCK
                ),
                .down_int8 = gemm_int8.Int8BufferB.init(
                    @as(usize, @intCast(config.hidden_size)), @as(usize, @intCast(expert_config.intermediate_size)),
                    null, gemm_int8.GemmKernel224Int8.N_STEP,
                    gemm_int8.GemmKernel224Int8.K_STEP,
                    gemm_int8.GemmKernel224Int8.K_BLOCK,
                    gemm_int8.GemmKernel224Int8.N_BLOCK
                ),
                .gate_buf = buffers.BufferC(f32).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(expert_config.intermediate_size)),
                    null, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.N_STEP,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .up_buf = buffers.BufferC(f32).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(expert_config.intermediate_size)),
                    null, gemm_bf16.GemmKernel224BF.M_STEP,
                    gemm_bf16.GemmKernel224BF.N_STEP,
                    gemm_bf16.GemmKernel224BF.N_BLOCK
                ),
                .down_buf = buffers.BufferC(f32).init(
                    @as(usize, @intCast(config.maxPossibleQlen())), @as(usize, @intCast(config.hidden_size)),
                    null, gemm_bf16.GemmKernel224BF.M_STEP,
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
        const expert_count = @as(usize, @intCast(self.config.expert_num));
        const hidden_size = @as(usize, @intCast(self.config.hidden_size));
        const inter = @as(usize, @intCast(self.config.intermediate_size));
        const inter_per_tp = inter / self.tp_count;

        for (0..expert_count)  | e |  {
            const expert = &self.experts[e];

            if (self.config.gate_proj_int8 != null) {
                // Load INT8 quantized weights
                if (self.config.gate_proj) |gp| {
                    expert.gate_int8.fromMatBF16(
                        gp + e * inter * hidden_size,
                        @intCast(hidden_size)
                    );
                }
                if (self.config.up_proj) |up| {
                    expert.up_int8.fromMatBF16(
                        up + e * inter * hidden_size,
                        @intCast(hidden_size)
                    );
                }
                if (self.config.down_proj) |dp| {
                    expert.down_int8.fromMatBF16(
                        dp + e * hidden_size * inter,
                        @intCast(inter)
                    );
                }
            } else if (self.config.gate_proj != null) {
                // Allocate per-expert per-TP-rank BF16 weight storage (once)
                // Uses page_allocator (C API convention: init allocates with
                // the passed allocator, deinit frees with page_allocator).
                if (e == 0) {
                    ensureBf16Storage(std.heap.page_allocator, expert_count, self.tp_count, hidden_size, inter);
                }

                // Load BF16 weights: allocate per-expert per-TP-rank slices
                // and copy from the config arrays. The config gate/up arrays
                // are [expert_num, intermediate_size, hidden_size] row-major
                // (gate_proj[i] = weight matrix for expert i).
                // The config down array is [expert_num, hidden_size, intermediate_size].
                //
                // For each TP rank t, we take a contiguous slice of
                // intermediate_size/tp_count rows from the weight matrix.
                if (self.config.gate_proj) |gp| {
                    for (0..self.tp_count) |t| {
                        const slice = gate_bf16_storage + (e * self.tp_count + t) * inter_per_tp * hidden_size;
                        const src = gp + e * inter * hidden_size + t * inter_per_tp * hidden_size;
                        @memcpy(slice[0..inter_per_tp * hidden_size], src[0..inter_per_tp * hidden_size]);
                    }
                }
                if (self.config.up_proj) |up| {
                    for (0..self.tp_count) |t| {
                        const slice = up_bf16_storage + (e * self.tp_count + t) * inter_per_tp * hidden_size;
                        const src = up + e * inter * hidden_size + t * inter_per_tp * hidden_size;
                        @memcpy(slice[0..inter_per_tp * hidden_size], src[0..inter_per_tp * hidden_size]);
                    }
                }
                if (self.config.down_proj) |dp| {
                    for (0..self.tp_count) |t| {
                        // Down projection: [hidden_size, intermediate_size].
                        // For each column, we take a slice of
                        // intermediate_size/tp_count elements.
                        // Storage layout: per-TP, all rows of down for that TP.
                        // We store as [inter_per_tp, hidden_size] row-major
                        // (transposed for the GEMM which expects [n, k] = [inter_per_tp, hidden_size] input).
                        // Actually the down GEMM is C[hidden, inter] = A[hidden, inter] @ W[inter, hidden]
                        // so we need W stored as [inter_per_tp, hidden_size] row-major.
                        // The config stores down as [hidden, inter] row-major.
                        // For TP rank t, we need columns [t*inter_per_tp, (t+1)*inter_per_tp) of W.
                        // Since W is stored transposed in the config, we extract
                        // rows [t*inter_per_tp, (t+1)*inter_per_tp) and store them
                        // contiguously as [inter_per_tp, hidden_size].
                        const slice = down_bf16_storage + (e * self.tp_count + t) * inter_per_tp * hidden_size;
                        for (0..hidden_size) |col| {
                            const src = dp + e * hidden_size * inter + col * inter + t * inter_per_tp;
                            @memcpy(slice[col * inter_per_tp ..][0..inter_per_tp], src[0..inter_per_tp]);
                        }
                    }
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
        weights: [*]const f32,
        input: [*]const amx.bf16,
        output: [*]amx.bf16,
        _incremental: bool
    ) void {
        _ = _incremental;
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

            // Compute expert: gate+up GEMMs, SwiGLU activation, down GEMM
            const inter = @as(usize, @intCast(self.config.intermediate_size));
            const hidden = @as(usize, @intCast(self.config.hidden_size));
            const gate_output = std.heap.page_allocator.alloc(amx.bf16, count * inter) catch @panic("OOM");
            defer std.heap.page_allocator.free(gate_output);
            const up_output = std.heap.page_allocator.alloc(amx.bf16, count * inter) catch @panic("OOM");
            defer std.heap.page_allocator.free(up_output);

            // Per-expert gate + up projections
            self.forwardGateUp(e, count, expert_input.ptr, gate_output.ptr, up_output.ptr);

            // Apply SwiGLU: silu(gate) * up, in-place into gate_output
            for (0..count * inter) |idx| {
                const g = amx.bf16_to_f32(gate_output[idx]);
                const u = amx.bf16_to_f32(up_output[idx]);
                gate_output[idx] = amx.f32_to_bf16(amx.swiglu(g, u));
            }

            // Per-expert down projection, accumulates into output_f32
            // For each token routed to this expert, find its weight and accumulate
            token_idx = 0;
            for (0..qlen) |i| {
                for (0..@as(usize, @intCast(k))) |j| {
                    const eid = expert_ids[i * @as(usize, @intCast(k)) + j];
                    if (eid == @as(i64, @intCast(e))) {
                        // Compute this token's down projection
                        const expert_down_out = std.heap.page_allocator.alloc(amx.bf16, hidden) catch @panic("OOM");
                        defer std.heap.page_allocator.free(expert_down_out);
                        // Zero the output (forwardDown accumulates)
                        @memset(expert_down_out, 0);

                        self.forwardDown(e, count, gate_output.ptr, expert_down_out.ptr);

                        // Scale by routing weight and accumulate into output_f32
                        const w = weights[i * @as(usize, @intCast(k)) + j];
                        for (0..hidden) |h| {
                            const val = amx.bf16_to_f32(expert_down_out[h]);
                            output_f32[i * hidden + h] += val * w;
                        }
                        token_idx += 1;
                    }
                }
            }
        }

        // Convert FP32 output to BF16
        for (0..qlen * @as(usize, @intCast(self.config.hidden_size))) |i| {
            output[i] = amx.f32_to_bf16(output_f32[i]);
        }
    }

    pub fn deinit(self: *TpMoe) void {
        // Free the two slices allocated in init(). The 9 buffer structs inside
        // each ExpertData (BufferA/BufferC/Int8BufferB) are non-owning POD
        // views: they hold only a `ptr: [*]K` into externally-owned weight
        // memory plus layout metadata, and expose no deinit(). Nothing else
        // is owned by TpMoe, so freeing the outer slices is complete.
        //
        // Also free the BF16 weight storage if it was allocated.
        //
        // NOTE: deinit has no allocator parameter, so it must use the same
        // allocator init received. The only caller is the C API wrapper
        // (kt_moe_free in main.zig), which passes std.heap.page_allocator to
        // both create() and TpMoe.init() — therefore page_allocator here.
        const allocator = std.heap.page_allocator;
        freeBf16Storage(allocator);
        allocator.free(self.experts);
        allocator.free(self.tp_configs);
        self.* = undefined;
    }

    pub fn warmUp(self: *TpMoe) void {
        // Pre-touch weight pages to avoid first-forward page faults.
        // With loadWeights now populating the module-level BF16 storage
        // (commit 246f66e), we can safely touch the first byte of each
        // storage. We pre-touch the first page of each storage (gate/up/down)
        // which catches the common case where the entire storage fits in
        // the first few pages. A more thorough implementation would touch
        // every page, but the first-page touch is the highest-impact one
        // (it primes the TLB and brings the most-recently-accessed pages in).
        if (!self.weights_loaded) return;
        if (bf16_weights_alloced > 0) {
            _ = gate_bf16_storage[0];
            _ = up_bf16_storage[0];
            _ = down_bf16_storage[0];
        }
    }

    pub fn loadWeightsWithMap(self: *TpMoe, physical_to_logical_map: [*]u64) void {
        // Load all experts in physical order first (double-load + remap
        // approach: wasteful — every weight is loaded/packed once for the
        // physical layout — but correct, and loadWeights is a one-time
        // setup cost. Optimization is deliberately out of scope here).
        self.loadWeights();

        // Then remap: physical slot p holds the weights of logical expert
        // physical_to_logical_map[p]; copy it into logical slot p.
        // ExpertData is a POD view struct (no owned heap pointers), so a
        // shallow struct copy is safe — no aliasing hazards because the
        // packed buffers are never freed through ExpertData.
        const physical_num: usize = @intCast(self.config.expert_num);
        for (0..physical_num) |p| {
            const logical = physical_to_logical_map[p];
            if (logical >= physical_num) continue; // skip invalid map entries
            self.experts[logical] = self.experts[p];
        }
        self.weights_loaded = true;
    }

    pub fn forwardGateUp(
        self: *TpMoe,
        expert_idx: usize,
        qlen: usize,
        input: [*]const amx.bf16,
        gate_output: [*]amx.bf16,
        up_output: [*]amx.bf16
    ) void {
        const cfg = self.config;
        const m = qlen;
        const k: usize = @intCast(cfg.hidden_size);
        const inter: usize = @intCast(cfg.intermediate_size);
        const n = inter / self.tp_count;
        const ldc: usize = inter;

        // Use FP32 scratch buffers (aligned) for GEMM output, then convert
        // to BF16. The gate_output/up_output params are BF16 arrays that may
        // not be 4-byte aligned (amx.bf16 = u16), so we can't pass them
        // directly to gemmExpert which writes f32.
        var gate_f32 = std.heap.page_allocator.alloc(f32, m * n) catch @panic("OOM");
        defer std.heap.page_allocator.free(gate_f32);
        var up_f32 = std.heap.page_allocator.alloc(f32, m * n) catch @panic("OOM");
        defer std.heap.page_allocator.free(up_f32);

        for (0..self.tp_count) |t| {
            const weight_off = (expert_idx * self.tp_count + t) * n * k;
            const out_off_t = t * m * n;
            // Gate projection: input [m, k] @ gate_weight [n, k]^T -> gate_f32 [m, n]
            gemm_bf16.gemmExpert(
                input,
                gate_bf16_storage + weight_off,
                gate_f32.ptr,
                m, n, k, k, n, ldc
            );
            // Up projection: input [m, k] @ up_weight [n, k]^T -> up_f32 [m, n]
            gemm_bf16.gemmExpert(
                input,
                up_bf16_storage + weight_off,
                up_f32.ptr,
                m, n, k, k, n, ldc
            );
            // Convert FP32 -> BF16 and store into the per-rank output slice
            for (0..m * n) |idx| {
                gate_output[out_off_t + idx] = amx.f32_to_bf16(gate_f32[idx]);
                up_output[out_off_t + idx] = amx.f32_to_bf16(up_f32[idx]);
            }
        }
    }

    pub fn forwardDown(
        self: *TpMoe,
        expert_idx: usize,
        qlen: usize,
        input: [*]const amx.bf16,
        output: [*]amx.bf16
    ) void {
        const cfg = self.config;
        const m = qlen;
        const k = @as(usize, @intCast(cfg.intermediate_size)) / self.tp_count;
        const n: usize = @intCast(cfg.hidden_size);
        const ldc = n;

        // Accumulate FP32 down output into the BF16 output (additive).
        var down_buf = std.heap.page_allocator.alloc(f32, m * n) catch @panic("OOM");
        defer std.heap.page_allocator.free(down_buf);
        @memset(down_buf, 0);

        for (0..self.tp_count) |t| {
            const weight_off = (expert_idx * self.tp_count + t) * n * k;
            const in_off_t = t * m * k;
            gemm_bf16.gemmExpert(
                input + in_off_t,
                down_bf16_storage + weight_off,
                down_buf.ptr,
                m, n, k, k, n, n
            );
        }

        // Accumulate into output (additive, matching TpMoe.forward's FP32 accumulation)
        for (0..m) |i| {
            for (0..n) |j| {
                const prev = amx.bf16_to_f32(output[i * ldc + j]);
                output[i * ldc + j] = amx.f32_to_bf16(prev + down_buf[i * n + j]);
            }
        }
    }
};

// ============================================================================
// BF16 Weight Storage Helpers
// ============================================================================

/// Ensure BF16 weight storage is allocated for the given expert/tp counts.
fn ensureBf16Storage(allocator: std.mem.Allocator, expert_count: usize, tp_count: usize, hidden_size: usize, inter: usize) void {
    const inter_per_tp = inter / tp_count;
    const per_expert = inter_per_tp * hidden_size;
    const total = expert_count * tp_count * per_expert;
    if (bf16_storage_expert_count == expert_count and bf16_storage_tp_count == tp_count and bf16_weights_alloced == total) {
        return;
    }
    freeBf16Storage(allocator);
    gate_bf16_alloc = allocator.alignedAlloc(amx.bf16, .@"64", total) catch @panic("OOM");
    up_bf16_alloc = allocator.alignedAlloc(amx.bf16, .@"64", total) catch @panic("OOM");
    down_bf16_alloc = allocator.alignedAlloc(amx.bf16, .@"64", total) catch @panic("OOM");
    gate_bf16_storage = gate_bf16_alloc.ptr;
    up_bf16_storage = up_bf16_alloc.ptr;
    down_bf16_storage = down_bf16_alloc.ptr;
    bf16_weights_alloced = total;
    bf16_storage_expert_count = expert_count;
    bf16_storage_tp_count = tp_count;
}

/// Free BF16 weight storage.
fn freeBf16Storage(allocator: std.mem.Allocator) void {
    if (gate_bf16_alloc.len > 0) {
        allocator.free(gate_bf16_alloc);
        allocator.free(up_bf16_alloc);
        allocator.free(down_bf16_alloc);
        gate_bf16_storage = @alignCast(@as([*]align(64) amx.bf16, @ptrFromInt(64)));
        up_bf16_storage = @alignCast(@as([*]align(64) amx.bf16, @ptrFromInt(64)));
        down_bf16_storage = @alignCast(@as([*]align(64) amx.bf16, @ptrFromInt(64)));
        gate_bf16_alloc = &[_]amx.bf16{};
        up_bf16_alloc = &[_]amx.bf16{};
        down_bf16_alloc = &[_]amx.bf16{};
        bf16_weights_alloced = 0;
    }
}
