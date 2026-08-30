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

    pub fn init(base_config: moe.MoeConfig, allocator: std.mem.Allocator) !TpMoeSft {
        const moe_inst = try moe.TpMoe.init(base_config, allocator);
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
                var u_gate = std.heap.page_allocator.alloc(f32, count * self.lora_rank) catch @panic("OOM");
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
                var u_up = std.heap.page_allocator.alloc(f32, count * self.lora_rank) catch @panic("OOM");
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
        _ = self;
    }
};
