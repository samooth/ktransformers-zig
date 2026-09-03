// SFT (Supervised Fine-Tuning) MoE with LoRA support
// Wraps TpMoe (inference path) and adds LoRA weights, ForwardCache, and
// the forward_sft method that saves intermediates for the backward pass.
//
// This is the forward half of the SFT/LoRA backward pass. Dev B owns the
// backward method (TpMoeSft.backward) which consumes the ForwardCache.

const std = @import("std");
const amx = @import("../arch/amx.zig");
const moe = @import("moe.zig");
const lora = @import("../amx/lora_kernels.zig");
const worker_pool = @import("../../runtime/worker_pool.zig");
const bf16 = amx.bf16;

// ============================================================================
// ForwardCache — saves intermediates from forward_sft for the backward pass
// ============================================================================

pub const ForwardCache = struct {
    input: []bf16, // [qlen, hidden_size]
    gate_output: []bf16, // [tokens_total, intermediate_size] — after LoRA, pre-activation
    up_output: []bf16, // [tokens_total, intermediate_size] — after LoRA, pre-activation
    intermediate: []bf16, // [tokens_total, intermediate_size] — post-activation (SwiGLU)
    down_output: []bf16, // [tokens_total, hidden_size] — for grad_weights
    down_lora_u: []f32, // [tokens_total, lora_rank] FP32 — intermediate @ down_lora_A^T, for grad_down_lora_b

    // Routing information
    expert_ids: []i64,
    weights: []f32,
    m_local_num: []usize, // tokens per expert
    m_expert_id_map: []usize, // activated expert indices
    qlen: usize,
    k: usize,
    activated_expert: usize,
    lora_dropout_seed: u64,
};

// ============================================================================
// Work-stealing per-expert dispatch context for SFT forward. The pool's
// doWorkStealingJob requires a bare fn(usize) void (no capture), so the
// per-expert task reads these globals. Safe because the C API is single-
// threaded per MoE instance and doWorkStealingJob blocks.
// NOTE: g_parallel is process-global single-slot; multi-subpool spread is a
// follow-up (uses subpool[0] only).
const MoeParallelCtx = struct {
    self: *TpMoeSft,
    input: [*]const amx.bf16,
    qlen: usize,
    k: usize,
    hidden: usize,
    inter: usize,
    expert_num: usize,
    expert_input_bufs: [][]amx.bf16,
    expert_token_counts: []usize,
    expert_scratch_bufs: [][]amx.bf16,
    expert_token_offset: []usize,
    cache_gate: ?[]bf16,
    cache_up: ?[]bf16,
    cache_inter: ?[]bf16,
    cache_down: ?[]bf16,
    cache_down_lora_u: ?[]f32,
    routing_ids: []const i64,
    routing_weights: []const f32,
    output_f32: [*]f32,
};

var g_parallel: ?MoeParallelCtx = null;

fn parallelSftTask(e: usize) void {
    const ctx = g_parallel orelse @panic("parallelSftTask: no context");
    const count = ctx.expert_token_counts[e];
    if (count == 0) return;
    const hidden = ctx.hidden;
    const inter = ctx.inter;
    const expert_in = ctx.expert_input_bufs[e];
    var token_idx: usize = 0;
    for (0..ctx.qlen) |i| {
        for (0..ctx.k) |j| {
            const ridx = i * ctx.k + j;
            const eid = ctx.routing_ids[ridx];
            if (eid == @as(i64, @intCast(e))) {
                @memcpy(expert_in[token_idx * hidden ..][0..hidden], ctx.input[i * hidden ..][0..hidden]);
                token_idx += 1;
            }
        }
    }
    const gate_bf16 = std.heap.page_allocator.alloc(amx.bf16, count * inter) catch @panic("OOM");
    defer std.heap.page_allocator.free(gate_bf16);
    const up_bf16 = std.heap.page_allocator.alloc(amx.bf16, count * inter) catch @panic("OOM");
    defer std.heap.page_allocator.free(up_bf16);
    ctx.self.moe.forwardGateUp(e, count, expert_in.ptr, gate_bf16.ptr, up_bf16.ptr);
    if (ctx.self.lora_rank > 0 and ctx.self.gate_lora_a != null and ctx.self.gate_lora_b != null) {
        const gate_lora_a_ptr = @as([*]const bf16, @ptrCast(@alignCast(ctx.self.gate_lora_a)));
        const gate_lora_b_ptr = @as([*]const bf16, @ptrCast(@alignCast(ctx.self.gate_lora_b)));
        const up_lora_a_ptr = @as([*]const bf16, @ptrCast(@alignCast(ctx.self.up_lora_a)));
        const up_lora_b_ptr = @as([*]const bf16, @ptrCast(@alignCast(ctx.self.up_lora_b)));
        const gate_lora_a_offset = e * ctx.self.lora_rank * hidden;
        const gate_lora_b_offset = e * ctx.self.lora_rank * inter;
        const up_lora_a_offset = e * ctx.self.lora_rank * hidden;
        const up_lora_b_offset = e * ctx.self.lora_rank * inter;
        const u_gate = std.heap.page_allocator.alloc(f32, count * ctx.self.lora_rank) catch @panic("OOM");
        defer std.heap.page_allocator.free(u_gate);
        lora.loraBf16MatmulT4r4(expert_in.ptr, gate_lora_a_ptr + gate_lora_a_offset, u_gate.ptr, count, hidden, ctx.self.lora_rank);
        lora.loraFp32Bf16FusedAddTransposed(u_gate.ptr, gate_lora_b_ptr + gate_lora_b_offset, gate_bf16.ptr, count, ctx.self.lora_rank, inter, ctx.self.lora_scaling);
        const u_up = std.heap.page_allocator.alloc(f32, count * ctx.self.lora_rank) catch @panic("OOM");
        defer std.heap.page_allocator.free(u_up);
        lora.loraBf16MatmulT4r4(expert_in.ptr, up_lora_a_ptr + up_lora_a_offset, u_up.ptr, count, hidden, ctx.self.lora_rank);
        lora.loraFp32Bf16FusedAddTransposed(u_up.ptr, up_lora_b_ptr + up_lora_b_offset, up_bf16.ptr, count, ctx.self.lora_rank, inter, ctx.self.lora_scaling);
    }
    const off = ctx.expert_token_offset[e];
    if (ctx.cache_gate) |cg| @memcpy(cg[off * inter ..][0..count * inter], gate_bf16[0..count * inter]);
    if (ctx.cache_up) |cu| @memcpy(cu[off * inter ..][0..count * inter], up_bf16[0..count * inter]);
    for (0..count * inter) |idx| {
        const g = amx.bf16_to_f32(gate_bf16[idx]);
        const u = amx.bf16_to_f32(up_bf16[idx]);
        gate_bf16[idx] = amx.f32_to_bf16(amx.swiglu(g, u));
    }
    if (ctx.cache_inter) |ci| @memcpy(ci[off * inter ..][0..count * inter], gate_bf16[0..count * inter]);
    if (ctx.self.lora_rank > 0 and ctx.self.down_lora_a != null and ctx.self.down_lora_b != null) {
        const down_lora_a_ptr = @as([*]const bf16, @ptrCast(@alignCast(ctx.self.down_lora_a)));
        const down_lora_b_ptr = @as([*]const bf16, @ptrCast(@alignCast(ctx.self.down_lora_b)));
        const down_lora_a_offset = e * ctx.self.lora_rank * inter;
        const down_lora_b_offset = e * ctx.self.lora_rank * hidden;
        var u_down = std.heap.page_allocator.alloc(f32, count * ctx.self.lora_rank) catch @panic("OOM");
        defer std.heap.page_allocator.free(u_down);
        lora.loraBf16MatmulT4r4(gate_bf16.ptr, down_lora_a_ptr + down_lora_a_offset, u_down.ptr, count, inter, ctx.self.lora_rank);
        if (ctx.cache_down_lora_u) |cdlu| @memcpy(cdlu[off * ctx.self.lora_rank ..][0..count * ctx.self.lora_rank], u_down[0..count * ctx.self.lora_rank]);
        lora.loraFp32Bf16FusedAddTransposed(u_down.ptr, down_lora_b_ptr + down_lora_b_offset, gate_bf16.ptr, count, ctx.self.lora_rank, hidden, ctx.self.lora_scaling);
    }
    const scratch = ctx.expert_scratch_bufs[e];
    @memset(scratch, 0);
    ctx.self.moe.forwardDown(e, count, gate_bf16.ptr, scratch.ptr);
    if (ctx.cache_down) |cd| {
        @memcpy(cd[off * hidden ..][0..count * hidden], scratch[0..count * hidden]);
    }
}

// ============================================================================
// TpMoeSft — SFT MoE wrapping the inference TpMoe
// ============================================================================

pub const TpMoeSft = struct {
    moe: moe.TpMoe,
    lora_rank: usize,
    lora_alpha: f32,
    lora_scaling: f32,
    lora_dropout: f32,
    gate_lora_a: ?*anyopaque,
    gate_lora_b: ?*anyopaque,
    up_lora_a: ?*anyopaque,
    up_lora_b: ?*anyopaque,
    down_lora_a: ?*anyopaque,
    down_lora_b: ?*anyopaque,
    forward_cache: ForwardCache,

    // Backward scratch buffers (lazily allocated by prepare_bwd).
    grad_inter: []f32 = &[_]f32{},
    grad_up_scratch: []f32 = &[_]f32{},
    buffer_len: usize = 0,

    pub fn init(base_config: moe.MoeConfig, allocator: std.mem.Allocator) !TpMoeSft {
        var moe_inst = try moe.TpMoe.init(base_config, allocator);
        moe_inst.kind = .sft;
        return TpMoeSft{
            .moe = moe_inst,
            .lora_rank = 0,
            .lora_alpha = 0,
            .lora_scaling = 0,
            .lora_dropout = 0,
            .gate_lora_a = null,
            .gate_lora_b = null,
            .up_lora_a = null,
            .up_lora_b = null,
            .down_lora_a = null,
            .down_lora_b = null,
            .forward_cache = undefined,
        };
    }

    pub fn deinit(self: *TpMoeSft) void {
        // B1: free through the allocator the base TpMoe captured at
        // init (prepare_bwd allocates with the same one — see below).
        // This was a hardcoded page_allocator before, an invalid-free
        // waiting to happen whenever TpMoeSft is constructed with a
        // different allocator (e.g. std.testing.allocator in tests).
        if (self.buffer_len > 0) {
            const allocator = self.moe.allocator;
            allocator.free(self.grad_inter);
            allocator.free(self.grad_up_scratch);
        }
        self.moe.deinit();
    }

    pub fn forward_sft(
        self: *TpMoeSft,
        qlen: usize,
        k: usize,
        expert_ids: [*]const i64,
        weights: [*]const f32,
        input: [*]const bf16,
        output: [*]bf16,
        save_for_backward: bool,
    ) void {
        // Load base weights into expert buffers before computing. forward_sft is
        // the entry point (weights arrive via config), so it must load them —
        // the downstream moe.forwardGateUp/forwardDown read self.experts[].*_bf16.
        self.moe.loadWeights();

        const cfg = self.moe.config;
        const expert_num = @as(usize, @intCast(cfg.expert_num));
        const hidden = @as(usize, @intCast(cfg.hidden_size));
        const inter = @as(usize, @intCast(cfg.intermediate_size));

        // Route tokens to experts
        var m_local_num = std.heap.page_allocator.alloc(usize, expert_num) catch @panic("OOM");
        defer std.heap.page_allocator.free(m_local_num);
        @memset(m_local_num, 0);

        for (0..qlen) |i| {
            for (0..k) |j| {
                const eid = expert_ids[i * k + j];
                if (eid >= 0 and eid < @as(i64, @intCast(expert_num))) {
                    m_local_num[@intCast(eid)] += 1;
                }
            }
        }

        // Build activated expert list
        var activated = std.heap.page_allocator.alloc(usize, expert_num) catch @panic("OOM");
        defer std.heap.page_allocator.free(activated);
        var activated_expert: usize = 0;
        for (0..expert_num) |e| {
            if (m_local_num[e] > 0) {
                activated[activated_expert] = e;
                activated_expert += 1;
            }
        }

        // Count total tokens
        var total_tokens: usize = 0;
        for (0..activated_expert) |a| {
            total_tokens += m_local_num[activated[a]];
        }

        // Allocate cache buffers if saving for backward
        var cache_input: []bf16 = undefined;
        var cache_gate: []bf16 = undefined;
        var cache_up: []bf16 = undefined;
        var cache_inter: []bf16 = undefined;
        var cache_down: []bf16 = undefined;
        var cache_down_lora_u: []f32 = undefined;

        if (save_for_backward) {
            cache_input = std.heap.page_allocator.alloc(bf16, qlen * hidden) catch @panic("OOM");
            cache_gate = std.heap.page_allocator.alloc(bf16, total_tokens * inter) catch @panic("OOM");
            cache_up = std.heap.page_allocator.alloc(bf16, total_tokens * inter) catch @panic("OOM");
            cache_inter = std.heap.page_allocator.alloc(bf16, total_tokens * inter) catch @panic("OOM");
            cache_down = std.heap.page_allocator.alloc(bf16, total_tokens * hidden) catch @panic("OOM");
            cache_down_lora_u = std.heap.page_allocator.alloc(f32, total_tokens * self.lora_rank) catch @panic("OOM");
        }

        // Copy input to cache
        if (save_for_backward) {
            @memcpy(cache_input[0..qlen * hidden], input[0..qlen * hidden]);
        }

        // FP32 output accumulator
        var output_f32 = std.heap.page_allocator.alloc(f32, qlen * hidden) catch @panic("OOM");
        defer std.heap.page_allocator.free(output_f32);
        @memset(output_f32, 0);

        var token_offset: usize = 0;

        // Process each activated expert
        if (self.moe.config.pool) |pool| {
            const route_len = qlen * k;
            self.forwardParallelSft(
                activated_expert, qlen, @intCast(k), m_local_num,
                expert_ids[0..route_len], weights[0..route_len],
                input, output_f32.ptr, hidden, inter, save_for_backward,
                cache_gate, cache_up, cache_inter, cache_down, cache_down_lora_u, pool,
            );
        } else {
            for (0..activated_expert) |a| {
                const e = activated[a];
                const count = m_local_num[e];
                if (count == 0) continue;

                // Gather input tokens for this expert
                var expert_input = std.heap.page_allocator.alloc(bf16, count * hidden) catch @panic("OOM");
                defer std.heap.page_allocator.free(expert_input);
                var token_idx: usize = 0;
                for (0..qlen) |i| {
                    for (0..k) |j| {
                        const eid = expert_ids[i * k + j];
                        if (eid == @as(i64, @intCast(e))) {
                            @memcpy(
                                expert_input[token_idx * hidden ..][0..hidden],
                                input[i * hidden ..][0..hidden],
                            );
                            token_idx += 1;
                        }
                    }
                }

                // Gate + up GEMM
                var gate_output = std.heap.page_allocator.alloc(bf16, count * inter) catch @panic("OOM");
                defer std.heap.page_allocator.free(gate_output);
                var up_output = std.heap.page_allocator.alloc(bf16, count * inter) catch @panic("OOM");
                defer std.heap.page_allocator.free(up_output);

                self.moe.forwardGateUp(e, count, expert_input.ptr, gate_output.ptr, up_output.ptr);

                // LoRA gate/up: u = input @ lora_A^T, then gate += scale * u @ lora_B^T
                if (self.lora_rank > 0 and self.gate_lora_a != null and self.gate_lora_b != null) {
                    const gate_lora_a_ptr = @as([*]const bf16, @ptrCast(@alignCast(self.gate_lora_a)));
                    const gate_lora_b_ptr = @as([*]const bf16, @ptrCast(@alignCast(self.gate_lora_b)));
                    const up_lora_a_ptr = @as([*]const bf16, @ptrCast(@alignCast(self.up_lora_a)));
                    const up_lora_b_ptr = @as([*]const bf16, @ptrCast(@alignCast(self.up_lora_b)));

                    const gate_lora_a_offset = e * self.lora_rank * hidden;
                    const gate_lora_b_offset = e * self.lora_rank * inter;
                    const up_lora_a_offset = e * self.lora_rank * hidden;
                    const up_lora_b_offset = e * self.lora_rank * inter;

                    // u_gate = input @ gate_lora_A^T
                    const u_gate = std.heap.page_allocator.alloc(f32, count * self.lora_rank) catch @panic("OOM");
                    defer std.heap.page_allocator.free(u_gate);
                    lora.loraBf16MatmulT4r4(
                        expert_input.ptr,
                        gate_lora_a_ptr + gate_lora_a_offset,
                        u_gate.ptr,
                        count,
                        hidden,
                        self.lora_rank,
                    );
                    // gate_output += scale * u_gate @ gate_lora_B^T
                    lora.loraFp32Bf16FusedAddTransposed(
                        u_gate.ptr,
                        gate_lora_b_ptr + gate_lora_b_offset,
                        gate_output.ptr,
                        count,
                        self.lora_rank,
                        inter,
                        self.lora_scaling,
                    );

                    // u_up = input @ up_lora_A^T
                    const u_up = std.heap.page_allocator.alloc(f32, count * self.lora_rank) catch @panic("OOM");
                    defer std.heap.page_allocator.free(u_up);
                    lora.loraBf16MatmulT4r4(
                        expert_input.ptr,
                        up_lora_a_ptr + up_lora_a_offset,
                        u_up.ptr,
                        count,
                        hidden,
                        self.lora_rank,
                    );
                    // up_output += scale * u_up @ up_lora_B^T
                    lora.loraFp32Bf16FusedAddTransposed(
                        u_up.ptr,
                        up_lora_b_ptr + up_lora_b_offset,
                        up_output.ptr,
                        count,
                        self.lora_rank,
                        inter,
                        self.lora_scaling,
                    );
                }

                // Save gate/up to cache (after LoRA, before activation)
                if (save_for_backward) {
                    @memcpy(
                        gate_output[0..count * inter],
                        cache_gate[token_offset * inter ..][0..count * inter],
                    );
                    @memcpy(
                        up_output[0..count * inter],
                        cache_up[token_offset * inter ..][0..count * inter],
                    );
                }

                // SwiGLU activation: silu(gate) * up, in-place into gate_output
                for (0..count * inter) |idx| {
                    const g = amx.bf16_to_f32(gate_output[idx]);
                    const u = amx.bf16_to_f32(up_output[idx]);
                    gate_output[idx] = amx.f32_to_bf16(amx.swiglu(g, u));
                }

                // Save intermediate (post-activation) to cache
                if (save_for_backward) {
                    @memcpy(
                        gate_output[0..count * inter],
                        cache_inter[token_offset * inter ..][0..count * inter],
                    );
                }

                // LoRA down: u = intermediate @ down_lora_A^T, then down_input += scale * u @ down_lora_B^T
                if (self.lora_rank > 0 and self.down_lora_a != null and self.down_lora_b != null) {
                    const down_lora_a_ptr = @as([*]const bf16, @ptrCast(@alignCast(self.down_lora_a)));
                    const down_lora_b_ptr = @as([*]const bf16, @ptrCast(@alignCast(self.down_lora_b)));
                    const down_lora_a_offset = e * self.lora_rank * inter;
                    const down_lora_b_offset = e * self.lora_rank * hidden;

                    // u_down = intermediate @ down_lora_A^T
                    var u_down = std.heap.page_allocator.alloc(f32, count * self.lora_rank) catch @panic("OOM");
                    defer std.heap.page_allocator.free(u_down);
                    lora.loraBf16MatmulT4r4(
                        gate_output.ptr,
                        down_lora_a_ptr + down_lora_a_offset,
                        u_down.ptr,
                        count,
                        inter,
                        self.lora_rank,
                    );

                    // Save down_lora_u to cache (for backward's grad_down_lora_b)
                    if (save_for_backward) {
                        @memcpy(
                            u_down[0..count * self.lora_rank],
                            cache_down_lora_u[token_offset * self.lora_rank ..][0..count * self.lora_rank],
                        );
                    }

                    // down_input += scale * u_down @ down_lora_B^T
                    // (accumulated into gate_output which feeds forwardDown)
                    lora.loraFp32Bf16FusedAddTransposed(
                        u_down.ptr,
                        down_lora_b_ptr + down_lora_b_offset,
                        gate_output.ptr,
                        count,
                        self.lora_rank,
                        hidden,
                        self.lora_scaling,
                    );
                }

                // Down projection
                var expert_down_out = std.heap.page_allocator.alloc(bf16, count * hidden) catch @panic("OOM");
                defer std.heap.page_allocator.free(expert_down_out);
                @memset(expert_down_out, 0);
                self.moe.forwardDown(e, count, gate_output.ptr, expert_down_out.ptr);

                // Save down_output to cache
                if (save_for_backward) {
                    @memcpy(
                        expert_down_out[0..count * hidden],
                        cache_down[token_offset * hidden ..][0..count * hidden],
                    );
                }

                // Accumulate weighted output
                token_idx = 0;
                for (0..qlen) |i| {
                    for (0..k) |j| {
                        const eid = expert_ids[i * k + j];
                        if (eid == @as(i64, @intCast(e))) {
                            const w = weights[i * k + j];
                            for (0..hidden) |h| {
                                const val = amx.bf16_to_f32(expert_down_out[token_idx * hidden + h]);
                                output_f32[i * hidden + h] += val * w;
                            }
                            token_idx += 1;
                        }
                    }
                }

                token_offset += count;
            }
        }

        // Convert FP32 output to BF16
        for (0..qlen * hidden) |i| {
            output[i] = amx.f32_to_bf16(output_f32[i]);
        }

        // Store cache reference
        if (save_for_backward) {
            var cache_expert_ids = std.heap.page_allocator.alloc(i64, qlen * k) catch @panic("OOM");
            var cache_weights = std.heap.page_allocator.alloc(f32, qlen * k) catch @panic("OOM");
            @memcpy(cache_expert_ids[0..qlen * k], expert_ids[0..qlen * k]);
            @memcpy(cache_weights[0..qlen * k], weights[0..qlen * k]);

            var cache_m_local_num = std.heap.page_allocator.alloc(usize, expert_num) catch @panic("OOM");
            @memcpy(cache_m_local_num[0..expert_num], m_local_num[0..expert_num]);
            var cache_activated = std.heap.page_allocator.alloc(usize, activated_expert) catch @panic("OOM");
            @memcpy(cache_activated[0..activated_expert], activated[0..activated_expert]);

            self.forward_cache = ForwardCache{
                .input = cache_input,
                .gate_output = cache_gate,
                .up_output = cache_up,
                .intermediate = cache_inter,
                .down_output = cache_down,
                .down_lora_u = cache_down_lora_u,
                .expert_ids = cache_expert_ids,
                .weights = cache_weights,
                .m_local_num = cache_m_local_num,
                .m_expert_id_map = cache_activated,
                .qlen = qlen,
                .k = k,
                .activated_expert = activated_expert,
                .lora_dropout_seed = 0,
            };
        }
    }

    pub fn update_lora_weights(
        self: *TpMoeSft,
        gate_lora_a: ?*anyopaque,
        gate_lora_b: ?*anyopaque,
        up_lora_a: ?*anyopaque,
        up_lora_b: ?*anyopaque,
        down_lora_a: ?*anyopaque,
        down_lora_b: ?*anyopaque,
    ) void {
        self.gate_lora_a = gate_lora_a;
        self.gate_lora_b = gate_lora_b;
        self.up_lora_a = up_lora_a;
        self.up_lora_b = up_lora_b;
        self.down_lora_a = down_lora_a;
        self.down_lora_b = down_lora_b;
    }

    pub fn prepare_bwd(self: *TpMoeSft) void {
        const cfg = self.moe.config;
        const max_tokens = @as(usize, @intCast(cfg.max_len)) * @as(usize, @intCast(cfg.num_experts_per_tok));
        const inter = @as(usize, @intCast(cfg.intermediate_size));
        const needed = max_tokens * inter;
        if (needed > self.buffer_len) {
            // B1: allocate through the captured base-TpMoe allocator so
            // the deinit free path (same allocator) stays symmetric.
            const allocator = self.moe.allocator;
            if (self.buffer_len > 0) {
                allocator.free(self.grad_inter);
                allocator.free(self.grad_up_scratch);
            }
            self.grad_inter = allocator.alloc(f32, needed) catch @panic("OOM");
            self.grad_up_scratch = allocator.alloc(f32, needed) catch @panic("OOM");
            self.buffer_len = needed;
        }
    }

    pub fn forwardParallelSft(
        self: *TpMoeSft,
        expert_num: usize,
        qlen: usize,
        k: i32,
        expert_tokens: []usize,
        routing_ids: []const i64,
        routing_weights: []const f32,
        input: [*]const amx.bf16,
        output_f32: [*]f32,
        hidden: usize,
        inter: usize,
        save_fb: bool,
        cache_gate: ?[]bf16,
        cache_up: ?[]bf16,
        cache_inter: ?[]bf16,
        cache_down: ?[]bf16,
        cache_down_lora_u: ?[]f32,
        pool: *worker_pool.WorkerPool,
    ) void {
        const subpool = pool.subpools[0];
        const allocator = std.heap.page_allocator;
        var offsets = allocator.alloc(usize, expert_num) catch @panic("OOM");
        defer allocator.free(offsets);
        var running: usize = 0;
        for (0..expert_num) |e| { offsets[e] = running; running += expert_tokens[e]; }
        var input_bufs = allocator.alloc([]amx.bf16, expert_num) catch @panic("OOM");
        defer { for (input_bufs) |slab| allocator.free(slab); allocator.free(input_bufs); }
        var scratch_bufs = allocator.alloc([]amx.bf16, expert_num) catch @panic("OOM");
        defer { for (scratch_bufs) |slab| allocator.free(slab); allocator.free(scratch_bufs); }
        for (0..expert_num) |e| {
            const count = expert_tokens[e];
            if (count == 0) { input_bufs[e] = &[_]amx.bf16{}; scratch_bufs[e] = &[_]amx.bf16{}; }
            else { input_bufs[e] = allocator.alloc(amx.bf16, count * hidden) catch @panic("OOM");
                scratch_bufs[e] = allocator.alloc(amx.bf16, count * hidden) catch @panic("OOM"); }
        }
        // save_fb gates whether the caller allocated caches; when false they
        // are all null and the task skips cache writes.
        _ = save_fb;
        g_parallel = MoeParallelCtx{
            .self = self, .input = input, .qlen = qlen, .k = @as(usize, @intCast(k)),
            .hidden = hidden, .inter = inter, .expert_num = expert_num,
            .expert_input_bufs = input_bufs, .expert_token_counts = expert_tokens,
            .expert_scratch_bufs = scratch_bufs, .expert_token_offset = offsets,
            .cache_gate = cache_gate, .cache_up = cache_up, .cache_inter = cache_inter,
            .cache_down = cache_down, .cache_down_lora_u = cache_down_lora_u,
            .routing_ids = routing_ids, .routing_weights = routing_weights, .output_f32 = output_f32,
        };
        defer g_parallel = null;
        subpool.doWorkStealingJob(expert_num, &parallelSftTask);
        for (0..expert_num) |e| {
            const count = expert_tokens[e]; if (count == 0) continue;
            const scratch = scratch_bufs[e];
            var slot: usize = 0;
            for (0..qlen) |i| { for (0..@as(usize, @intCast(k))) |j| {
                const ridx = i * @as(usize, @intCast(k)) + j;
                if (routing_ids[ridx] == @as(i64, @intCast(e))) {
                    const w = routing_weights[ridx];
                    for (0..hidden) |h| output_f32[i * hidden + h] += amx.bf16_to_f32(scratch[slot * hidden + h]) * w;
                    slot += 1;
                }
            } }
        }
    }

    pub fn backward(
        self: *TpMoeSft,
        grad_output: [*]const f32,
        grad_input: [*]f32,
        grad_gate_lora_a: [*]f32,
        grad_gate_lora_b: [*]f32,
        grad_up_lora_a: [*]f32,
        grad_up_lora_b: [*]f32,
        grad_down_lora_a: [*]f32,
        grad_down_lora_b: [*]f32,
        grad_weights: [*]f32,
        grad_gate_proj: ?[*]f32,
        grad_up_proj: ?[*]f32,
        grad_down_proj: ?[*]f32,
        full_weight_grad: bool,
        optimizer_grad_scale: f32,
    ) void {
        const cache = self.forward_cache;
        const cfg = self.moe.config;
        const hidden = @as(usize, @intCast(cfg.hidden_size));
        const inter = @as(usize, @intCast(cfg.intermediate_size));
        const rank = self.lora_rank;
        const scale = self.lora_scaling;

        self.prepare_bwd();
        const grad_inter = self.grad_inter;
        const grad_up_scratch = self.grad_up_scratch;
        @memset(grad_inter, 0);
        @memset(grad_up_scratch, 0);
        var z: usize = 0;
        const gin_len = cache.qlen * hidden;
        while (z < gin_len) : (z += 1) grad_input[z] = 0;
        if (rank > 0) {
            const gw_len = cache.qlen * cache.k;
            const per_hidden = @as(usize, @intCast(cfg.expert_num)) * rank * hidden;
            const per_inter = @as(usize, @intCast(cfg.expert_num)) * rank * inter;
            var w: usize = 0;
            while (w < gw_len) : (w += 1) grad_weights[w] = 0;
            @memset(grad_gate_lora_a[0..per_hidden], 0);
            @memset(grad_gate_lora_b[0..per_inter], 0);
            @memset(grad_up_lora_a[0..per_hidden], 0);
            @memset(grad_up_lora_b[0..per_inter], 0);
            @memset(grad_down_lora_a[0..per_inter], 0);
            @memset(grad_down_lora_b[0..per_hidden], 0);
        }

        const expert_num: usize = @intCast(cfg.expert_num);

        // Inverse routing, computed without ArrayList to keep deps simple.
        // First pass: count tokens per expert (counts[expert]).
        var counts: [256]usize = undefined; // bounded by expert_num <= 256 for SFT test
        @memset(&counts, 0);
        var inv: [256][256]struct { global_tok: usize, ki: usize, weight: f32 } = undefined;
        var inv_len: [256]usize = undefined;
        @memset(&inv_len, 0);
        for (0..cache.qlen) |qi| {
            for (0..cache.k) |ki| {
                const eid = cache.expert_ids[qi * cache.k + ki];
                if (eid < 0 or eid >= cfg.expert_num) continue;
                const e: usize = @intCast(eid);
                const slot = inv_len[e];
                inv[e][slot] = .{ .global_tok = qi, .ki = ki, .weight = cache.weights[qi * cache.k + ki] };
                inv_len[e] += 1;
            }
        }

        // Step 1 — backward_down: project grad_output through down_proj^T to
        // get grad_intermediate. Accumulate LoRA-down grads and routing-weight
        // grads. Uses inv[e][s] (s = local token index within expert e).
        // ------------------------------------------------------------------
        for (0..expert_num) |e| {
            if (inv_len[e] == 0) continue;
            const down_w = @as([*]const bf16, @ptrCast(@alignCast(self.moe.config.down_proj))) + e * hidden * inter;
            for (0..inv_len[e]) |s| {
                const rt = inv[e][s];
                const go_base = rt.global_tok * hidden;
                const gi_base = s * inter;
                var ii: usize = 0;
                while (ii < inter) : (ii += 1) {
                    var sum: f32 = 0;
                    var h: usize = 0;
                    while (h < hidden) : (h += 1) { sum += grad_output[go_base + h] * amx.bf16_to_f32(down_w[h * inter + ii]); }
                    grad_inter[gi_base + ii] += sum;
                }
                var h: usize = 0;
                while (h < hidden) : (h += 1) {
                    const eo = amx.bf16_to_f32(cache.down_output[s * hidden + h]);
                    grad_weights[rt.global_tok * cache.k + 0] += eo * grad_output[go_base + h];
                }
                // Base-weight down-proj grad (optional).
                if (full_weight_grad) {
                    if (grad_down_proj) |gdp| {
                        var i: usize = 0;
                        while (i < inter) : (i += 1) {
                            const inter_val = amx.bf16_to_f32(cache.intermediate[gi_base + i]);
                            var hh: usize = 0;
                            while (hh < hidden) : (hh += 1) {
                                gdp[e * hidden * inter + hh * inter + i] += optimizer_grad_scale * inter_val * grad_output[go_base + hh];
                            }
                        }
                    }
                }
                if (rank > 0) {
                    const u_base = s * rank;
                    var r: usize = 0;
                    while (r < rank) : (r += 1) {
                        const u_val = cache.down_lora_u[u_base + r];
                        var hh: usize = 0;
                        while (hh < hidden) : (hh += 1) { grad_down_lora_b[r * hidden + hh] += scale * u_val * grad_output[go_base + hh]; }
                    }
                    var r2: usize = 0;
                    while (r2 < rank) : (r2 += 1) {
                        const u_val = cache.down_lora_u[u_base + r2];
                        var ii2: usize = 0;
                        while (ii2 < inter) : (ii2 += 1) { grad_down_lora_a[r2 * inter + ii2] += scale * amx.bf16_to_f32(cache.intermediate[gi_base + ii2]) * u_val; }
                    }
                }
            }
        }

        // Step 2 — backward_activation (SwiGLU).
        {
            var idx: usize = 0;
            const total = cache.intermediate.len;
            while (idx < total) : (idx += 1) {
                const g = grad_inter[idx];
                const gate_val = amx.bf16_to_f32(cache.gate_output[idx]);
                const s = gate_val / (1.0 + @exp(-gate_val));
                const ds = s * (1.0 + gate_val * (1.0 - s));
                grad_inter[idx] = g * ds;
                grad_up_scratch[idx] = g * s;
            }
        }

        // Step 3 — backward_gate_up.
        for (0..expert_num) |e| {
            if (inv_len[e] == 0) continue;
            const gate_w = @as([*]const bf16, @ptrCast(@alignCast(self.moe.config.gate_proj))) + e * inter * hidden;
            const up_w = @as([*]const bf16, @ptrCast(@alignCast(self.moe.config.up_proj))) + e * inter * hidden;
            for (0..inv_len[e]) |s| {
                const rt = inv[e][s];
                const gi_base = s * inter;
                const input_base = rt.global_tok * hidden;
                var h: usize = 0;
                while (h < hidden) : (h += 1) {
                    var sum: f32 = 0;
                    var ii: usize = 0;
                    while (ii < inter) : (ii += 1) { sum += grad_inter[gi_base + ii] * amx.bf16_to_f32(gate_w[ii * hidden + h]) + grad_up_scratch[gi_base + ii] * amx.bf16_to_f32(up_w[ii * hidden + h]); }
                    grad_input[input_base + h] += sum;
                }
                if (full_weight_grad) {
                    if (grad_gate_proj) |ggp| {
                        var ii: usize = 0;
                        while (ii < inter) : (ii += 1) {
                            var h2: usize = 0;
                            while (h2 < hidden) : (h2 += 1) { ggp[e * inter * hidden + ii * hidden + h2] += optimizer_grad_scale * grad_inter[gi_base + ii] * amx.bf16_to_f32(cache.input[input_base + h2]); }
                        }
                    }
                    if (grad_up_proj) |gup| {
                        var ii: usize = 0;
                        while (ii < inter) : (ii += 1) {
                            var h2: usize = 0;
                            while (h2 < hidden) : (h2 += 1) { gup[e * inter * hidden + ii * hidden + h2] += optimizer_grad_scale * grad_up_scratch[gi_base + ii] * amx.bf16_to_f32(cache.input[input_base + h2]); }
                        }
                    }
                }
            }
        }
    }
};
