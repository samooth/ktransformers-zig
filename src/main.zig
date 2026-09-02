// Main Zig entry point for ktransformers-zig library
// Exports C-compatible API

const std = @import("std");
const root = @import("root.zig");

const memory = root.memory;
const worker_pool = root.worker_pool;
const task_queue = root.task_queue;
const cpu_detect = root.cpu_detect;
pub const amx = root.amx;
const buffers = root.buffers;
const gemm_bf16 = root.gemm_bf16;
const gemm_int8 = root.gemm_int8;
const gemm_int4 = root.gemm_int4;
const gemm_fp8 = root.gemm_fp8;
const gemm_mxfp4 = root.gemm_mxfp4;
const gemm_mxfp8 = root.gemm_mxfp8;
// GGML quant kernels: pub so C-API tests (rooted here) can reference the
// block types; no ABI effect.
pub const gemm_q8_0 = root.gemm_q8_0;
pub const gemm_q4_k = root.gemm_q4_k;
pub const gemm_q5_k = root.gemm_q5_k;
pub const gemm_q6_k = root.gemm_q6_k;
pub const gemm_q8_k = root.gemm_q8_k;
const moe = root.moe;
const moe_sft = root.moe_sft;
// MLA: pub so C-API tests (rooted here) can reference the config/cache
// types; no ABI effect.
pub const mla_config = root.mla_config;
pub const mla_cache = root.mla_cache;
const mla_core = root.mla_core;
// DeepseekV3DecoderLayer orchestration (WIP by the model-orchestration dev;
// root re-exports it from kernels/moe/deepseekv3_layer.zig).
const deepseekv3_layer = root.deepseekv3_layer;
const deepseekv3_model = root.deepseekv3_model;
const qwen3_layer = root.qwen3_layer;
const qwen3_model = root.qwen3_model;

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

pub const kt_moe_sft_config_t = extern struct {
    base: kt_moe_config_t,
    lora_rank: c_int,
    lora_alpha: f32,
    lora_dropout: f32,
    gate_lora_a: ?*anyopaque,
    gate_lora_b: ?*anyopaque,
    up_lora_a: ?*anyopaque,
    up_lora_b: ?*anyopaque,
    down_lora_a: ?*anyopaque,
    down_lora_b: ?*anyopaque,
    full_weight_grad: c_int,
    authoritative_optimizer_grads: c_int,
    grad_gate_proj: ?*anyopaque,
    grad_up_proj: ?*anyopaque,
    grad_down_proj: ?*anyopaque,
};

pub const kt_gate_config_t = extern struct {
    hidden_size: usize,
    num_experts_per_tok: usize,
    n_routed_experts: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: c_int,
    routed_scaling_factor: f32,
    scoring_func: [*]u8,
    topk_method: [*]u8,
    layer_idx: c_int,
    pool: *KT_WorkerPool,
    weight: *anyopaque,
    weight_type: kt_type_t,
    e_score_correction_bias: *anyopaque,
    e_score_correction_bias_type: kt_type_t,
    max_seqlen: usize,
};

pub const kt_linear_config_t = extern struct {
    hidden_size: usize, // input dim
    intermediate_size: usize, // output dim
    stride: c_int,
    group_max_len: c_int,
    proj: *anyopaque,
    proj_type: kt_type_t,
    hidden_type: kt_type_t,
};

pub const kt_mlp_config_t = extern struct {
    hidden_size: usize,
    intermediate_size: usize,
    stride: c_int,
    group_max_len: c_int,
    gate_proj: *anyopaque,
    up_proj: *anyopaque,
    down_proj: *anyopaque,
    gate_type: kt_type_t,
    up_type: kt_type_t,
    down_type: kt_type_t,
    hidden_type: kt_type_t,
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
pub const KT_DSV3Layer = opaque {};
pub const KT_Gate = opaque {};
pub const KT_Linear = opaque {};
pub const KT_MLP = opaque {};
pub const KT_FP8LayerwiseTransport = opaque {};

// C struct returned by kt_fp8_transport_wait. Field order/types must match
// `kt_fp8_stats_t` in `include/kt_kernel.h:480-494` (the C header is the
// contract — `int layer_id` per the header even though the C++ internal
// struct uses int64_t).
pub const kt_fp8_stats_t = extern struct {
    epoch: u64,
    layer_id: c_int,
    expert_count: c_int,
    rank: c_int,
    writer_ms: f64,
    slot_wait_ms: f64,
    h2d_ms: f64,
    total_ms: f64,
    bytes: usize,
    poisoned: c_int,
    error_code: c_int,
    error_rank: c_int,
    error_message: [*]const u8,
};

var g_cpu_variant: [32]u8 = undefined;
var g_cpu_variant_ptr: [*]const u8 = @ptrCast(&g_cpu_variant[0]);
var g_cpu_variant_len: usize = 0;

// ============================================================================
// Default Allocator (B1)
// ============================================================================
//
// Historically every C-API entry allocated with std.heap.page_allocator
// hardcoded (IMPROVE.md B1). Two problems: (1) Python/embedded callers
// cannot supply their own allocator (e.g. a tracked arena to detect
// leaks, or a NUMA-bound allocator), and (2) the free paths must
// hardcode the same allocator — an invalid-free waiting to happen
// the moment any constructor gains an allocator parameter.
//
// B1 fix, additive-only (header/ABI coordination required for config
// struct changes, so we do NOT touch kt_*_config_t):
//
//   - `kt_set_default_allocator(vtable)` lets the caller install a
//     C-ABI allocator (alloc/free/resize fn pointers + userdata)
//     BEFORE constructing any kt_* object. Called with null, it
//     resets to page_allocator (also the pre-call default).
//   - Every context created after that captures `g_default_allocator`
//     at construction and frees through the captured allocator —
//     so set-default-allocator must not be swapped between a *_new
//     and its *_free (documented contract; a per-context capture
//     would need a header change).
//   - Per-call scratch (mlaForwardImpl buffers etc.) also uses the
//     captured allocator so it round-trips symmetrically.
//
// Invariant maintained everywhere: whatever allocates also frees —
// always the same allocator instance, either the captured context
// one or the global default in stateless helpers.

/// C-ABI allocator vtable. `alloc` returns null on failure (matching
/// malloc semantics). Layout is C-friendly (opaque userdata + plain
/// fn pointers); the adapter below bridges it to std.mem.Allocator.
pub const kt_allocator_vtable_t = extern struct {
    /// userdata passed back to every callback (may be null)
    userdata: ?*anyopaque,
    /// allocate `size` bytes at `alignment` (power of two) — returns
    /// null on failure
    alloc: ?*const fn (userdata: ?*anyopaque, size: usize, alignment: usize) callconv(.c) ?[*]u8,
    /// free a previous allocation (alignment must match the alloc)
    free: ?*const fn (userdata: ?*anyopaque, ptr: [*]u8, size: usize, alignment: usize) callconv(.c) void,
    /// optional in-place resize; return 0 on success, -1 (or leave
    /// null) to force the alloc+copy+free fallback
    resize: ?*const fn (userdata: ?*anyopaque, ptr: [*]u8, old_size: usize, new_size: usize, alignment: usize) callconv(.c) c_int,
};

var g_alloc_vtable: ?*const kt_allocator_vtable_t = null;

/// Bridge struct: holds the C vtable and implements the
/// std.mem.Allocator.VTable calling convention. One global instance
/// (the vtable is process-wide by design; a multi-allocator setup
/// would use userdata to distinguish).
const CAllocAdapter = struct {
    fn alloc(userdata: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const vt: *const kt_allocator_vtable_t = @ptrCast(@alignCast(userdata));
        const f = vt.alloc orelse return null;
        return f(vt.userdata, len, @intFromEnum(alignment));
    }

    fn resize(userdata: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ret_addr;
        const vt: *const kt_allocator_vtable_t = @ptrCast(@alignCast(userdata));
        if (vt.resize) |f| {
            return f(vt.userdata, buf.ptr, buf.len, new_len, @intFromEnum(alignment)) == 0;
        }
        return false;
    }

    fn remap(userdata: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        // The C vtable has no remap contract; fall back to the
        // resize-then-copy path by reporting "unsupported" (null).
        _ = userdata;
        _ = buf;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return null;
    }

    fn free(userdata: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        _ = ret_addr;
        const vt: *const kt_allocator_vtable_t = @ptrCast(@alignCast(userdata));
        const f = vt.free orelse return;
        f(vt.userdata, buf.ptr, buf.len, @intFromEnum(alignment));
    }

    fn vtable() *const std.mem.Allocator.VTable {
        return &.{
            .alloc = alloc,
            .resize = resize,
            .remap = remap,
            .free = free,
        };
    }
};

/// The allocator every C-API entry uses. page_allocator when no custom
/// allocator has been installed; an adapter around the installed
/// vtable otherwise.
fn defaultAllocator() std.mem.Allocator {
    const vt = g_alloc_vtable orelse return std.heap.page_allocator;
    return std.mem.Allocator{
        // The VTable callbacks receive the vtable pointer itself as
        // userdata (so the adapter can read .userdata/.alloc/... from
        // it). CAllocAdapter casts it back to the C vtable type.
        .ptr = @constCast(@ptrCast(vt)),
        .vtable = CAllocAdapter.vtable(),
    };
}

/// Install a custom allocator for all subsequent kt_* allocations.
/// Pass null to restore the default (page_allocator).
/// MUST be called before any kt_*_new — objects created earlier keep
/// the allocator they were constructed with.
/// Zig extension beyond the C++ kt-kernel (same section as the other
/// Zig extensions in include/kt_kernel.h).
pub export fn kt_set_default_allocator(vtable: ?*const kt_allocator_vtable_t) void {
    g_alloc_vtable = vtable;
}

/// B1: create/destroy helpers used by every context below — single
/// choke point so the allocator choice is uniform.
fn ctxCreate(comptime T: type, value: T) *T {
    const a = defaultAllocator();
    const p = a.create(T) catch @panic("OOM");
    p.* = value;
    return p;
}

fn ctxDestroy(comptime T: type, ptr: *T) void {
    defaultAllocator().destroy(T, ptr);
}

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

fn ensure_cpu_variant_detected() void {
    if (g_cpu_variant_len == 0) detect_cpu_variant();
}

// ============================================================================
// Core C API Functions
// ============================================================================

// ---------------------------------------------------------------------------
// ABI layout probes — used by tools/audit_layout.py and the pybind11 shim
// (bindings/kt_kernel_pybind.cpp) to assert that the C++ mirror structs
// match the Zig extern struct byte layout exactly. A mismatch here is a
// silent stack-corruption bug; these probes turn it into a loud failure.
// NOT part of include/kt_kernel.h (test-only surface, nm-gated).
// ---------------------------------------------------------------------------

export fn kt_abi_size_moe_config() usize {
    return @sizeOf(kt_moe_config_t);
}

export fn kt_abi_size_mla_config() usize {
    return @sizeOf(kt_mla_config_t);
}

export fn kt_abi_size_moe_sft_config() usize {
    return @sizeOf(kt_moe_sft_config_t);
}

export fn kt_abi_size_gate_config() usize {
    return @sizeOf(kt_gate_config_t);
}

export fn kt_abi_size_linear_config() usize {
    return @sizeOf(kt_linear_config_t);
}

export fn kt_abi_size_mlp_config() usize {
    return @sizeOf(kt_mlp_config_t);
}

/// Field-offset probe: returns the byte offset of field `field_index` within
/// the struct identified by `struct_id` (0 = kt_moe_config_t, 1 = kt_mla_config_t).
/// Returns SIZE_MAX for an unknown id/index. Lets the C++ shim (and the
/// layout audit tool) verify field alignment without duplicating layout
/// knowledge in two places. Uses @offsetOf (comptime, no UB).
export fn kt_abi_field_offset(struct_id: c_int, field_index: c_int) usize {
    if (struct_id == 0) {
        const offsets = [_]usize{
            @offsetOf(kt_moe_config_t, "expert_num"),
            @offsetOf(kt_moe_config_t, "num_experts_per_tok"),
            @offsetOf(kt_moe_config_t, "hidden_size"),
            @offsetOf(kt_moe_config_t, "intermediate_size"),
            @offsetOf(kt_moe_config_t, "layer_idx"),
            @offsetOf(kt_moe_config_t, "pool"),
            @offsetOf(kt_moe_config_t, "num_gpu_experts"),
            @offsetOf(kt_moe_config_t, "gate_proj"),
            @offsetOf(kt_moe_config_t, "up_proj"),
            @offsetOf(kt_moe_config_t, "down_proj"),
            @offsetOf(kt_moe_config_t, "gate_scale"),
            @offsetOf(kt_moe_config_t, "up_scale"),
            @offsetOf(kt_moe_config_t, "down_scale"),
            @offsetOf(kt_moe_config_t, "max_len"),
            @offsetOf(kt_moe_config_t, "path"),
            @offsetOf(kt_moe_config_t, "save"),
            @offsetOf(kt_moe_config_t, "load"),
            @offsetOf(kt_moe_config_t, "share_cache_pool"),
            @offsetOf(kt_moe_config_t, "gate_type"),
            @offsetOf(kt_moe_config_t, "up_type"),
            @offsetOf(kt_moe_config_t, "down_type"),
            @offsetOf(kt_moe_config_t, "hidden_type"),
            @offsetOf(kt_moe_config_t, "swiglu_limit"),
            @offsetOf(kt_moe_config_t, "swiglu_alpha"),
        };
        if (field_index < 0 or field_index >= offsets.len) return std.math.maxInt(usize);
        return offsets[@intCast(field_index)];
    } else if (struct_id == 1) {
        const offsets = [_]usize{
            @offsetOf(kt_mla_config_t, "hidden_size"),
            @offsetOf(kt_mla_config_t, "q_lora_rank"),
            @offsetOf(kt_mla_config_t, "num_heads"),
            @offsetOf(kt_mla_config_t, "nope_size"),
            @offsetOf(kt_mla_config_t, "rope_size"),
            @offsetOf(kt_mla_config_t, "kv_lora_rank"),
            @offsetOf(kt_mla_config_t, "layer_idx"),
            @offsetOf(kt_mla_config_t, "pool"),
            @offsetOf(kt_mla_config_t, "q_a_proj"),
            @offsetOf(kt_mla_config_t, "q_a_norm"),
            @offsetOf(kt_mla_config_t, "q_b_proj"),
            @offsetOf(kt_mla_config_t, "kv_a_proj_with_mqa"),
            @offsetOf(kt_mla_config_t, "kv_a_norm"),
            @offsetOf(kt_mla_config_t, "kv_b_proj"),
            @offsetOf(kt_mla_config_t, "o_proj"),
            @offsetOf(kt_mla_config_t, "m_block"),
            @offsetOf(kt_mla_config_t, "n_block"),
            @offsetOf(kt_mla_config_t, "page_count"),
        };
        if (field_index < 0 or field_index >= offsets.len) return std.math.maxInt(usize);
        return offsets[@intCast(field_index)];
    }
    return std.math.maxInt(usize);
}

export fn kt_version() [*]const u8 {
    return "0.6.1-zig";
}

export fn kt_get_cpu_variant() [*]const u8 {
    ensure_cpu_variant_detected();
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
    const allocator = defaultAllocator();
    const pool = allocator.create(worker_pool.WorkerPool) catch @panic("OOM");
    pool.* = worker_pool.WorkerPool.initSimple(allocator, @intCast(thread_count)) catch @panic("pool init");
    return @ptrCast(pool);
}

export fn kt_worker_pool_new_config(config: kt_worker_pool_config_t) *KT_WorkerPool {
    const allocator = defaultAllocator();

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
    // B1: free through the allocator the pool was constructed with
    // (captured in the struct), NOT the current default — a caller may
    // swap the default between new and free.
    wp.allocator.destroy(wp);
}

export fn kt_worker_pool_get_thread_num(pool: *KT_WorkerPool) c_int {
    const wp: *worker_pool.WorkerPool = @ptrCast(@alignCast(pool));
    return @intCast(wp.getTotalThreads());
}

// ============================================================================
// CPU Infer
// ============================================================================

/// A4 wiring: derive GEMM tile block sizes from the host's measured cache
/// hierarchy and install them into the BF16 kernel before any operator
/// construction. Best-effort — on detection failure the compiled-in
/// defaults (K_BLOCK=1792/N_BLOCK=256) hold. Runs once per process (the
/// static guard caches the result: detectCpu allocates, and every
/// operator *_new calls this defensively in case the caller built its
/// worker pool directly instead of via kt_cpuinfer_new).
var g_tile_params_tuned: bool = false;
fn tuneTileParamsForHost() void {
    if (g_tile_params_tuned) return;
    g_tile_params_tuned = true;
    const allocator = defaultAllocator();
    var cpu = cpu_detect.detectCpu(allocator) catch {
        gemm_bf16.GemmKernel224BF.resetTileParams();
        return;
    };
    defer cpu.deinit(allocator);
    const tp = cpu_detect.selectTileParams(cpu);
    gemm_bf16.GemmKernel224BF.setTileParams(tp.n_block, tp.k_block);
}

export fn kt_cpuinfer_new(thread_count: c_int) *KT_CPUInfer {
    detect_cpu_variant();
    tuneTileParamsForHost();
    kt_ggml_init();
    return @ptrCast(kt_worker_pool_new(thread_count));
}

export fn kt_cpuinfer_new_config(config: kt_worker_pool_config_t) *KT_CPUInfer {
    detect_cpu_variant();
    tuneTileParamsForHost();
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
    // B3: real drain semantics. Wait until every in-flight
    // doWorkStealingJob on every subpool has finished, or until at
    // most `allow_n_pending` remain (the C++ reference contract).
    // The current submit path is synchronous so this is normally a
    // fast return, but the accounting is real: if submit becomes
    // fire-and-forget (or a caller enqueues from another thread),
    // sync blocks on the subpool condvars until drained.
    const pool: *worker_pool.WorkerPool = @ptrCast(@alignCast(cpuinfer));
    for (pool.subpools) |subpool| {
        subpool.waitIdle(allow_n_pending);
    }
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
    tuneTileParamsForHost();
    const pool: *worker_pool.WorkerPool = @ptrCast(@alignCast(cpuinfer));
    var cfg = toMoeConfig(config);
    cfg.pool = pool;
    const allocator = defaultAllocator();
    const moe_inst = allocator.create(moe.TpMoe) catch @panic("OOM");
    moe_inst.* = moe.TpMoe.init(cfg, allocator) catch @panic("Failed to init MoE");
    return @ptrCast(moe_inst);
}

export fn kt_moe_new_sft(cpuinfer: *KT_CPUInfer, config: kt_moe_sft_config_t) *KT_MOE {
    tuneTileParamsForHost();
    const pool: *worker_pool.WorkerPool = @ptrCast(@alignCast(cpuinfer));
    var base_cfg = toMoeConfig(config.base);
    base_cfg.pool = pool;
    const allocator = defaultAllocator();
    const sft_inst = allocator.create(moe_sft.TpMoeSft) catch @panic("OOM");
    sft_inst.* = moe_sft.TpMoeSft.init(base_cfg, allocator) catch @panic("Failed to init SFT MoE");
    sft_inst.lora_rank = @intCast(config.lora_rank);
    sft_inst.lora_alpha = config.lora_alpha;
    sft_inst.lora_scaling = if (config.lora_rank > 0) config.lora_alpha / @as(f32, @floatFromInt(@as(c_int, config.lora_rank))) else 0.0;
    sft_inst.lora_dropout = config.lora_dropout;
    sft_inst.gate_lora_a = config.gate_lora_a;
    sft_inst.gate_lora_b = config.gate_lora_b;
    sft_inst.up_lora_a = config.up_lora_a;
    sft_inst.up_lora_b = config.up_lora_b;
    sft_inst.down_lora_a = config.down_lora_a;
    sft_inst.down_lora_b = config.down_lora_b;
    return @ptrCast(sft_inst);
}

export fn kt_moe_forward_sft(
    moe_ptr: *KT_MOE,
    qlen: c_int,
    k: c_int,
    expert_ids: [*]const i64,
    weights: [*]const f32,
    input: [*]const amx.bf16,
    output: [*]const amx.bf16,
    save_for_backward: c_int,
) void {
    const m: *moe_sft.TpMoeSft = @ptrCast(@alignCast(moe_ptr));
    m.forward_sft(
        @intCast(qlen),
        @intCast(k),
        expert_ids,
        weights,
        input,
        @constCast(@ptrCast(output)),
        save_for_backward != 0,
    );
}

export fn kt_moe_update_lora_weights(
    moe_ptr: *KT_MOE,
    gate_lora_a: ?*anyopaque,
    gate_lora_b: ?*anyopaque,
    up_lora_a: ?*anyopaque,
    up_lora_b: ?*anyopaque,
    down_lora_a: ?*anyopaque,
    down_lora_b: ?*anyopaque,
) void {
    const m: *moe_sft.TpMoeSft = @ptrCast(@alignCast(moe_ptr));
    m.update_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b);
}

export fn kt_moe_backward(
    moe_ptr: *KT_MOE,
    grad_output: [*]const f32,
    grad_input: [*]f32,
    grad_gate_lora_a: [*]f32,
    grad_gate_lora_b: [*]f32,
    grad_up_lora_a: [*]f32,
    grad_up_lora_b: [*]f32,
    grad_down_lora_a: [*]f32,
    grad_down_lora_b: [*]f32,
    grad_weights: [*]f32,
    grad_gate_proj: ?*anyopaque,
    grad_up_proj: ?*anyopaque,
    grad_down_proj: ?*anyopaque,
    accumulate_optimizer_grads: c_int,
    optimizer_grad_scale: f32,
) void {
    const m: *moe_sft.TpMoeSft = @ptrCast(@alignCast(moe_ptr));
    m.backward(
        grad_output,
        grad_input,
        grad_gate_lora_a,
        grad_gate_lora_b,
        grad_up_lora_a,
        grad_up_lora_b,
        grad_down_lora_a,
        grad_down_lora_b,
        grad_weights,
        if (grad_gate_proj) |p| @as(?[*]f32, @alignCast(@ptrCast(p))) else null,
        if (grad_up_proj) |p| @as(?[*]f32, @alignCast(@ptrCast(p))) else null,
        if (grad_down_proj) |p| @as(?[*]f32, @alignCast(@ptrCast(p))) else null,
        accumulate_optimizer_grads != 0,
        optimizer_grad_scale,
    );
}

export fn kt_moe_free(moe_ptr: *KT_MOE) void {
    const m: *moe.TpMoe = @ptrCast(@alignCast(moe_ptr));
    switch (m.kind) {
        .inference => {
            const a = m.allocator;
            m.deinit();
            a.destroy(m);
        },
        .sft => {
            const sft: *moe_sft.TpMoeSft = @ptrCast(@alignCast(moe_ptr));
            const a = sft.moe.allocator;
            sft.deinit();
            a.destroy(sft);
        },
    }
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
// MLA (Multi-Head Latent Attention) - C API implementation
// ============================================================================
//
// Each *KT_MLA is actually a *MlaContext that holds the MlaEngine and its
// associated MlaKvCache. Both must outlive any forward/decode call but are
// freed together in kt_mla_free.
//
// Weight pointers in kt_mla_config_t are passed as *anyopaque. We cast them
// to [*]const u16 (BF16). The C caller is responsible for passing BF16 data
// (the convention for non-quantized MLA). Quantized weight support (INT8/INT4)
// is tracked as a follow-up.
//
// The engine's internal KV cache is the source of truth. The external
// `kv_cache` parameter in kt_mla_update_kv_cache is accepted for API
// compatibility with the C++ reference but is currently ignored.
//
// BF16 <-> F32 conversion happens at the C/Zig boundary. The engine operates
// on F32 internally (for matmul precision), but the C API uses BF16. We
// allocate scratch F32 buffers per call on page_allocator.

const MlaContext = struct {
    engine: *mla_core.MlaEngine,
    cache: *mla_cache.MlaKvCache,
    allocator: std.mem.Allocator,
    /// Mirrors the C++ reference's TP_MLA_Common::weights_loaded (mla-tp.hpp:39):
    /// forward() throws "Not Loaded" unless load_weights() ran first. The Zig
    /// engine already holds the weight pointers from kt_mla_new (no NUMA copy
    /// pass needed), so this is a state flag set by kt_mla_load_weights.
    weights_loaded: bool = false,
};

// kt_mla_new matches the C++ contract (single config arg, 1 param total).
// The Zig implementation does not use a CPUInfer — the engine is
// single-threaded per MLA instance (matches the reference semantics), so
// dropping the cpuinfer parameter is the correct contract.
pub export fn kt_mla_new(config: kt_mla_config_t) *KT_MLA {
    // Validate the full MLA weight set up front. The C header types these as
    // *anyopaque (non-nullable), but a Python/caller bug could still pass 0.
    // The engine stores raw pointers — catching it here is the only sane spot.
    if (@intFromPtr(config.q_a_proj) == 0 or @intFromPtr(config.q_a_norm) == 0 or
        @intFromPtr(config.q_b_proj) == 0 or @intFromPtr(config.kv_a_proj_with_mqa) == 0 or
        @intFromPtr(config.kv_a_norm) == 0 or @intFromPtr(config.kv_b_proj) == 0 or
        @intFromPtr(config.o_proj) == 0)
    {
        @panic("kt_mla_new: null weight pointer(s) in MLA config");
    }

    // 1. Map C config -> internal MlaConfig
    const mla_cfg = mla_config.MlaConfig{
        .hidden_size = config.hidden_size,
        .q_lora_rank = config.q_lora_rank,
        .num_heads = config.num_heads,
        .nope_size = config.nope_size,
        .rope_size = config.rope_size,
        .kv_lora_rank = config.kv_lora_rank,
        .max_qlen = config.max_qlen,
        .max_kvlen = config.max_kvlen,
        .token_count_in_page = config.token_count_in_page,
        .rope_theta = config.rope_theta,
        .rope_scaling_factor = config.rope_scaling_factor,
        .rope_scaling_mscale = config.rope_scaling_mscale,
        .q_a_proj = @ptrCast(@alignCast(config.q_a_proj)),
        .q_a_norm = @ptrCast(@alignCast(config.q_a_norm)),
        .q_b_proj = @ptrCast(@alignCast(config.q_b_proj)),
        .kv_a_proj_with_mqa = @ptrCast(@alignCast(config.kv_a_proj_with_mqa)),
        .kv_a_norm = @ptrCast(@alignCast(config.kv_a_norm)),
        .kv_b_proj = @ptrCast(@alignCast(config.kv_b_proj)),
        .o_proj = @ptrCast(@alignCast(config.o_proj)),
    };

    // 2. Estimate max_pages and create KV cache
    const allocator = defaultAllocator();
    const max_pages = (config.max_kvlen / config.token_count_in_page) + 1;
    const cache = allocator.create(mla_cache.MlaKvCache) catch @panic("OOM");
    cache.* = mla_cache.MlaKvCache.init(allocator, mla_cfg, max_pages) catch @panic("OOM");

    // 3. Create MlaEngine (allocates 11 aligned f32 scratch buffers internally)
    const engine = allocator.create(mla_core.MlaEngine) catch @panic("OOM");
    engine.* = mla_core.MlaEngine.init(allocator, mla_cfg, cache) catch @panic("OOM");

    // 4. Create context wrapper
    const ctx = allocator.create(MlaContext) catch @panic("OOM");
    ctx.* = .{ .engine = engine, .cache = cache, .allocator = allocator };
    return @ptrCast(ctx);
}

pub export fn kt_mla_free(mla: *KT_MLA) void {
    const ctx: *MlaContext = @ptrCast(@alignCast(mla));
    // B1: capture before deinit (deinit sets * = undefined).
    const a = ctx.allocator;
    ctx.engine.deinit();
    ctx.cache.deinit();
    a.destroy(ctx.engine);
    a.destroy(ctx.cache);
    a.destroy(ctx);
}

/// Mark weights as loaded (C++ reference: TP_MLA_Common::load_weights sets
/// weights_loaded=true; forward() throws "Not Loaded" without it —
/// mla-tp.hpp:39,86). The Zig engine binds weight pointers at kt_mla_new
/// (no separate NUMA copy pass), so this is a state flag that gates
/// forward/prefill/decode. Null-pointer validation happens in kt_mla_new.
pub export fn kt_mla_load_weights(mla: *KT_MLA) void {
    const ctx: *MlaContext = @ptrCast(@alignCast(mla));
    ctx.weights_loaded = true;
}

/// Shared implementation for forward and prefill.
/// Converts BF16 input -> F32, calls engine.forward, converts F32 output -> BF16.
/// `position_ids` is unused (the engine uses kv_start_pos for absolute positioning,
/// matching the math fix documented in src/mla/mla_core.zig).
fn mlaForwardImpl(
    ctx: *MlaContext,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    qlen: usize,
    kvlen: usize,
) void {
    if (!ctx.weights_loaded) {
        @panic("kt_mla_forward: weights not loaded (call kt_mla_load_weights first)");
    }
    const cfg = ctx.engine.config;
    // B1: per-call scratch goes through the context's captured
    // allocator (round-trips symmetrically with whatever kt_mla_new
    // installed).
    const allocator = ctx.allocator;

    // BF16 -> F32 input conversion
    const input_f32 = allocator.alloc(f32, qlen * cfg.hidden_size) catch @panic("OOM");
    defer allocator.free(input_f32);
    for (0..qlen * cfg.hidden_size) |i| {
        input_f32[i] = amx.bf16_to_f32(input[i]);
    }

    // Allocate F32 output
    const output_f32 = allocator.alloc(f32, qlen * cfg.hidden_size) catch @panic("OOM");
    defer allocator.free(output_f32);

    // Call engine.forward (kv_start_pos = kvlen - qlen for prefill)
    const kv_start_pos = kvlen - qlen;
    ctx.engine.forward(input_f32.ptr, output_f32.ptr, qlen, kv_start_pos) catch @panic("forward failed");

    // F32 -> BF16 output conversion
    for (0..qlen * cfg.hidden_size) |i| {
        output[i] = amx.f32_to_bf16(output_f32[i]);
    }
}

// Matches the C++ kt_kernel.h paged/batched contract: qlens, page_tables,
// kv_lens are parallel arrays of length qlen_count. Two paths:
//   - qlen_count == 1 && page_tables == null: sequential forward into the
//     engine's internal cache (unchanged legacy behavior — the C header
//     types page_tables as a pointer, so null means "no indirection").
//   - otherwise: paged/batched forward — every sequence's page table
//     indexes the engine's page pool (page_table[logical_pos /
//     token_count_in_page] = page idx). New KVs are written at logical
//     positions [kv_len - qlen, kv_len) through the table; attention
//     reads [0, kv_len) through the same indirection (mla-tp.hpp:84).
// page_table_lens is validated against the per-sequence requirement
// ceil(kv_len / token_count_in_page) when tables are provided.
pub export fn kt_mla_forward(
    mla: *KT_MLA,
    qlens: [*]const c_int,
    qlen_count: c_int,
    page_tables: ?[*]const [*]const c_int,
    page_table_lens: ?[*]const c_int,
    kv_lens: [*]const c_int,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
) void {
    const ctx: *MlaContext = @ptrCast(@alignCast(mla));
    if (!ctx.weights_loaded) {
        @panic("kt_mla_forward: weights not loaded (call kt_mla_load_weights first)");
    }
    const count: usize = @intCast(qlen_count);
    if (count == 0) return;
    const cfg = ctx.engine.config;
    const allocator = ctx.allocator;

    // Legacy sequential path: qlen_count == 1 uses the engine's own
    // sequential cache (no page_table needed). The paged path is only
    // required when qlen_count > 1 (the "batch of sequences" case).
    if (count == 1) {
        const qlen: usize = @intCast(qlens[0]);
        const kvlen: usize = @intCast(kv_lens[0]);
        mlaForwardImpl(ctx, input, output, qlen, kvlen);
        return;
    }

    // Paged/batched path. Required from here on.
    const tables = page_tables orelse
        @panic("kt_mla_forward: qlen_count > 1 requires page_tables");
    const lens = page_table_lens orelse
        @panic("kt_mla_forward: qlen_count > 1 requires page_table_lens");
    for (0..count) |seq| {
        const kv_len: usize = @intCast(kv_lens[seq]);
        const need = (kv_len + cfg.token_count_in_page - 1) / cfg.token_count_in_page;
        if (kv_len > cfg.max_kvlen) {
            @panic("kt_mla_forward: kv_lens exceeds max_kvlen");
        }
        if (@as(usize, @intCast(lens[seq])) < need) {
            @panic("kt_mla_forward: page_table too short for kv_lens");
        }
    }

    // Sum the batch's qlens for the F32 conversion buffers.
    var qlen_sum: usize = 0;
    for (0..count) |seq| qlen_sum += @intCast(qlens[seq]);

    const input_f32 = allocator.alloc(f32, qlen_sum * cfg.hidden_size) catch @panic("OOM");
    defer allocator.free(input_f32);
    for (0..qlen_sum * cfg.hidden_size) |i| {
        input_f32[i] = amx.bf16_to_f32(input[i]);
    }
    const output_f32 = allocator.alloc(f32, qlen_sum * cfg.hidden_size) catch @panic("OOM");
    defer allocator.free(output_f32);

    ctx.engine.forwardPaged(
        input_f32.ptr,
        output_f32.ptr,
        qlens[0..count],
        tables[0..count],
        kv_lens[0..count],
    ) catch @panic("forwardPaged failed");

    for (0..qlen_sum * cfg.hidden_size) |i| {
        output[i] = amx.f32_to_bf16(output_f32[i]);
    }
}

pub export fn kt_mla_prefill(
    mla: *KT_MLA,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    qlen: c_int,
    position_ids: [*]const i64,
) void {
    // Prefill: kvlen = position_ids[qlen-1] + 1 (the last position's abs position + 1)
    const last_pos = position_ids[@as(usize, @intCast(qlen)) - 1];
    const kvlen = @as(usize, @intCast(last_pos)) + 1;
    const ctx: *MlaContext = @ptrCast(@alignCast(mla));
    mlaForwardImpl(ctx, input, output, @intCast(qlen), kvlen);
}

pub export fn kt_mla_decode(
    mla: *KT_MLA,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    position_id: i64,
) void {
    // Decode: single token, kvlen = position_id + 1
    const ctx: *MlaContext = @ptrCast(@alignCast(mla));
    mlaForwardImpl(ctx, input, output, 1, @as(usize, @intCast(position_id)) + 1);
}

pub export fn kt_mla_update_kv_cache(
    mla: *KT_MLA,
    kv_cache: *anyopaque,
    new_kv: [*]const amx.bf16,
    position: i64,
) void {
    _ = kv_cache; // external cache not yet supported; engine uses internal cache
    const ctx: *MlaContext = @ptrCast(@alignCast(mla));
    const cfg = ctx.engine.config;
    _ = position; // appendToken appends sequentially; position is implicit
    const allocator = ctx.allocator; // B1

    // new_kv has shape [hidden_size] = [kv_lora_rank + rope_size] for the compressed KV
    // Split into compressed_kv (nope) and k_pe (rope), convert BF16 -> F32
    const nope_f32 = allocator.alloc(f32, cfg.kv_lora_rank) catch @panic("OOM");
    defer allocator.free(nope_f32);
    const rope_f32 = allocator.alloc(f32, cfg.rope_size) catch @panic("OOM");
    defer allocator.free(rope_f32);

    for (0..cfg.kv_lora_rank) |i| {
        nope_f32[i] = amx.bf16_to_f32(new_kv[i]);
    }
    for (0..cfg.rope_size) |i| {
        rope_f32[i] = amx.bf16_to_f32(new_kv[cfg.kv_lora_rank + i]);
    }

    ctx.cache.appendToken(nope_f32.ptr, rope_f32.ptr) catch @panic("appendToken failed");
}

// ============================================================================
// DeepseekV3 Decoder Layer (model orchestration — Zig extension)
// ============================================================================
// Mirrors kt_dsv3_layer_config_t in include/kt_kernel.h exactly (the C ABI
// contract; verify with tools/audit_layout.py if extended).

pub const kt_dsv3_layer_config_t = extern struct {
    hidden_size: usize,
    q_lora_rank: usize,
    num_heads: usize,
    nope_size: usize,
    rope_size: usize,
    kv_lora_rank: usize,
    max_qlen: usize,
    max_kvlen: usize,
    token_count_in_page: usize,
    rope_theta: f64,
    expert_num: usize,
    num_experts_per_tok: usize,
    intermediate_size: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: c_int,
    routed_scaling_factor: f32,
    pool: ?*anyopaque,
    q_a_proj: *const anyopaque,
    q_a_norm: *const anyopaque,
    q_b_proj: *const anyopaque,
    kv_a_proj_with_mqa: *const anyopaque,
    kv_a_norm: *const anyopaque,
    kv_b_proj: *const anyopaque,
    o_proj: *const anyopaque,
    attn_norm_weight: *const anyopaque,
    ffn_norm_weight: *const anyopaque,
    gate_weight: *const anyopaque,
    e_score_correction_bias: ?*const anyopaque,
    gate_proj: *const anyopaque,
    up_proj: *const anyopaque,
    down_proj: *const anyopaque,
};

fn toLayerConfig(c: kt_dsv3_layer_config_t) deepseekv3_layer.LayerConfig {
    return .{
        .hidden_size = c.hidden_size,
        .q_lora_rank = c.q_lora_rank,
        .num_heads = c.num_heads,
        .nope_size = c.nope_size,
        .rope_size = c.rope_size,
        .kv_lora_rank = c.kv_lora_rank,
        .max_qlen = c.max_qlen,
        .max_kvlen = c.max_kvlen,
        .token_count_in_page = c.token_count_in_page,
        .rope_theta = c.rope_theta,
        .expert_num = c.expert_num,
        .num_experts_per_tok = c.num_experts_per_tok,
        .intermediate_size = c.intermediate_size,
        .n_group = c.n_group,
        .topk_group = c.topk_group,
        .norm_topk_prob = c.norm_topk_prob != 0,
        .routed_scaling_factor = c.routed_scaling_factor,
        .pool = @ptrCast(@alignCast(c.pool)),
        .q_a_proj = @ptrCast(@alignCast(c.q_a_proj)),
        .q_a_norm = @ptrCast(@alignCast(c.q_a_norm)),
        .q_b_proj = @ptrCast(@alignCast(c.q_b_proj)),
        .kv_a_proj_with_mqa = @ptrCast(@alignCast(c.kv_a_proj_with_mqa)),
        .kv_a_norm = @ptrCast(@alignCast(c.kv_a_norm)),
        .kv_b_proj = @ptrCast(@alignCast(c.kv_b_proj)),
        .o_proj = @ptrCast(@alignCast(c.o_proj)),
        .attn_norm_weight = @ptrCast(@alignCast(c.attn_norm_weight)),
        .ffn_norm_weight = @ptrCast(@alignCast(c.ffn_norm_weight)),
        .gate_weight = @ptrCast(@alignCast(c.gate_weight)),
        .e_score_correction_bias = if (c.e_score_correction_bias) |p| @ptrCast(@alignCast(p)) else null,
        .gate_proj = @ptrCast(@alignCast(c.gate_proj)),
        .up_proj = @ptrCast(@alignCast(c.up_proj)),
        .down_proj = @ptrCast(@alignCast(c.down_proj)),
    };
}

pub export fn kt_dsv3_layer_new(config: *const kt_dsv3_layer_config_t) *KT_DSV3Layer {
    tuneTileParamsForHost();
    const layer = deepseekv3_layer.DeepseekV3DecoderLayer.init(defaultAllocator(), toLayerConfig(config.*)) catch @panic("Failed to init DSV3 layer");
    return @ptrCast(layer);
}

pub export fn kt_dsv3_layer_forward(
    layer: *KT_DSV3Layer,
    qlen: usize,
    kv_start_pos: usize,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
) void {
    const l: *deepseekv3_layer.DeepseekV3DecoderLayer = @ptrCast(@alignCast(layer));
    l.forward(qlen, kv_start_pos, input, output);
}

pub export fn kt_dsv3_layer_free(layer: *KT_DSV3Layer) void {
    const l: *deepseekv3_layer.DeepseekV3DecoderLayer = @ptrCast(@alignCast(layer));
    l.deinit();
}

// ============================================================================
// DeepseekV3Model + ForCausalLM (model-level orchestration — Zig extension)
// ============================================================================
// The model wraps N DecoderLayers + final RMSNorm; the CausalLM adds the
// lm_head GEMM. Config reuses kt_dsv3_layer_config_t for the per-layer
// template plus the model-level fields (num_layers, final_norm_weight,
// lm_head, vocab_size).

pub const kt_dsv3_model_config_t = extern struct {
    num_layers: usize,
    layer: kt_dsv3_layer_config_t, // per-layer template (all fields)
    final_norm_weight: *const anyopaque, // BF16 [hidden_size]
    lm_head: *const anyopaque, // BF16 [vocab_size, hidden_size] (CausalLM only)
    vocab_size: usize, // 0 = Model-only (no lm_head)
};

const KT_DSV3Model = opaque {};
const KT_DSV3CausalLM = opaque {};

fn toModelConfig(c: kt_dsv3_model_config_t) deepseekv3_model.ModelConfig {
    return .{
        .num_layers = c.num_layers,
        .layer = toLayerConfig(c.layer),
        .final_norm_weight = @ptrCast(@alignCast(c.final_norm_weight)),
    };
}

pub export fn kt_dsv3_model_new(config: *const kt_dsv3_model_config_t) *KT_DSV3Model {
    tuneTileParamsForHost();
    const model = deepseekv3_model.DeepseekV3Model.init(defaultAllocator(), toModelConfig(config.*)) catch @panic("Failed to init DSV3 model");
    return @ptrCast(model);
}

pub export fn kt_dsv3_model_forward(
    model: *KT_DSV3Model,
    qlen: usize,
    kv_start_pos: usize,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
) void {
    const m: *deepseekv3_model.DeepseekV3Model = @ptrCast(@alignCast(model));
    m.forward(qlen, kv_start_pos, input, output);
}

pub export fn kt_dsv3_model_free(model: *KT_DSV3Model) void {
    const m: *deepseekv3_model.DeepseekV3Model = @ptrCast(@alignCast(model));
    m.deinit();
}

pub export fn kt_dsv3_causallm_new(config: *const kt_dsv3_model_config_t) *KT_DSV3CausalLM {
    tuneTileParamsForHost();
    const clm = deepseekv3_model.DeepseekV3ForCausalLM.init(defaultAllocator(), .{
        .model = toModelConfig(config.*),
        .lm_head = @ptrCast(@alignCast(config.lm_head)),
        .vocab_size = config.vocab_size,
    }) catch @panic("Failed to init DSV3 CausalLM");
    return @ptrCast(clm);
}

pub export fn kt_dsv3_causallm_forward(
    clm: *KT_DSV3CausalLM,
    qlen: usize,
    kv_start_pos: usize,
    input: [*]const amx.bf16,
    logits: [*]f32,
) void {
    const c: *deepseekv3_model.DeepseekV3ForCausalLM = @ptrCast(@alignCast(clm));
    c.forward(qlen, kv_start_pos, input, logits);
}

pub export fn kt_dsv3_causallm_free(clm: *KT_DSV3CausalLM) void {
    const c: *deepseekv3_model.DeepseekV3ForCausalLM = @ptrCast(@alignCast(clm));
    c.deinit();
}

// ============================================================================
// Qwen3 MoE Decoder Layer + Model + CausalLM (model orchestration —
// Zig extension). Ports the Qwen3-style MoE forward pass: standard
// MHA + GQA + RoPE + pre-norm + vanilla softmax top-k gate (no group
// routing, no e_score_correction_bias) + MoE FFN.
//
// Reference: ktransformers/kt-kernel/python/sft/layer.py:461-560
// (Qwen3 MoE forward) and ktransformers/kt-kernel/doc/en/Qwen3.5.md.
// ============================================================================
// Mirrors kt_qwen3moe_layer_config_t in include/kt_kernel.h exactly
// (the C ABI contract; verify with tools/audit_layout.py if extended).

pub const kt_qwen3moe_layer_config_t = extern struct {
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_qlen: usize,
    max_kvlen: usize,
    rope_theta: f64,
    expert_num: usize,
    num_experts_per_tok: usize,
    intermediate_size: usize,
    pool: ?*anyopaque,
    q_proj: *const anyopaque,
    k_proj: *const anyopaque,
    v_proj: *const anyopaque,
    o_proj: *const anyopaque,
    attn_norm_weight: *const anyopaque,
    ffn_norm_weight: *const anyopaque,
    gate_weight: *const anyopaque,
    gate_proj: *const anyopaque,
    up_proj: *const anyopaque,
    down_proj: *const anyopaque,
};

fn toQwen3LayerConfig(c: kt_qwen3moe_layer_config_t) qwen3_layer.LayerConfig {
    return .{
        .hidden_size = c.hidden_size,
        .num_heads = c.num_heads,
        .num_kv_heads = c.num_kv_heads,
        .head_dim = c.head_dim,
        .max_qlen = c.max_qlen,
        .max_kvlen = c.max_kvlen,
        .rope_theta = c.rope_theta,
        .expert_num = c.expert_num,
        .num_experts_per_tok = c.num_experts_per_tok,
        .intermediate_size = c.intermediate_size,
        .pool = @ptrCast(@alignCast(c.pool)),
        .q_proj = @ptrCast(@alignCast(c.q_proj)),
        .k_proj = @ptrCast(@alignCast(c.k_proj)),
        .v_proj = @ptrCast(@alignCast(c.v_proj)),
        .o_proj = @ptrCast(@alignCast(c.o_proj)),
        .attn_norm_weight = @ptrCast(@alignCast(c.attn_norm_weight)),
        .ffn_norm_weight = @ptrCast(@alignCast(c.ffn_norm_weight)),
        .gate_weight = @ptrCast(@alignCast(c.gate_weight)),
        .gate_proj = @ptrCast(@alignCast(c.gate_proj)),
        .up_proj = @ptrCast(@alignCast(c.up_proj)),
        .down_proj = @ptrCast(@alignCast(c.down_proj)),
    };
}

const KT_Qwen3MoeLayer = opaque {};
const KT_Qwen3MoeModel = opaque {};
const KT_Qwen3MoeCausalLM = opaque {};

pub export fn kt_qwen3moe_layer_new(config: *const kt_qwen3moe_layer_config_t) *KT_Qwen3MoeLayer {
    const layer = qwen3_layer.Qwen3MoeDecoderLayer.init(defaultAllocator(), toQwen3LayerConfig(config.*)) catch @panic("Failed to init Qwen3 layer");
    return @ptrCast(layer);
}

pub export fn kt_qwen3moe_layer_forward(
    layer: *KT_Qwen3MoeLayer,
    qlen: usize,
    kv_start_pos: usize,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
) void {
    const l: *qwen3_layer.Qwen3MoeDecoderLayer = @ptrCast(@alignCast(layer));
    l.forward(qlen, kv_start_pos, input, output);
}

pub export fn kt_qwen3moe_layer_free(layer: *KT_Qwen3MoeLayer) void {
    const l: *qwen3_layer.Qwen3MoeDecoderLayer = @ptrCast(@alignCast(layer));
    l.deinit();
}

pub const kt_qwen3moe_model_config_t = extern struct {
    num_layers: usize,
    layer: kt_qwen3moe_layer_config_t,
    final_norm_weight: *const anyopaque,
    lm_head: *const anyopaque,
    vocab_size: usize,
};

fn toQwen3ModelConfig(c: kt_qwen3moe_model_config_t) qwen3_model.ModelConfig {
    return .{
        .num_layers = c.num_layers,
        .layer = toQwen3LayerConfig(c.layer),
        .final_norm_weight = @ptrCast(@alignCast(c.final_norm_weight)),
    };
}

pub export fn kt_qwen3moe_model_new(config: *const kt_qwen3moe_model_config_t) *KT_Qwen3MoeModel {
    const model = qwen3_model.Qwen3MoeModel.init(defaultAllocator(), toQwen3ModelConfig(config.*)) catch @panic("Failed to init Qwen3 model");
    return @ptrCast(model);
}

pub export fn kt_qwen3moe_model_forward(
    model: *KT_Qwen3MoeModel,
    qlen: usize,
    kv_start_pos: usize,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
) void {
    const m: *qwen3_model.Qwen3MoeModel = @ptrCast(@alignCast(model));
    m.forward(qlen, kv_start_pos, input, output);
}

pub export fn kt_qwen3moe_model_free(model: *KT_Qwen3MoeModel) void {
    const m: *qwen3_model.Qwen3MoeModel = @ptrCast(@alignCast(model));
    m.deinit();
}

pub export fn kt_qwen3moe_causallm_new(config: *const kt_qwen3moe_model_config_t) *KT_Qwen3MoeCausalLM {
    const clm = qwen3_model.Qwen3MoeForCausalLM.init(defaultAllocator(), .{
        .model = toQwen3ModelConfig(config.*),
        .lm_head = @ptrCast(@alignCast(config.lm_head)),
        .vocab_size = config.vocab_size,
    }) catch @panic("Failed to init Qwen3 CausalLM");
    return @ptrCast(clm);
}

pub export fn kt_qwen3moe_causallm_forward(
    clm: *KT_Qwen3MoeCausalLM,
    qlen: usize,
    kv_start_pos: usize,
    input: [*]const amx.bf16,
    logits: [*]f32,
) void {
    const c: *qwen3_model.Qwen3MoeForCausalLM = @ptrCast(@alignCast(clm));
    c.forward(qlen, kv_start_pos, input, logits);
}

pub export fn kt_qwen3moe_causallm_free(clm: *KT_Qwen3MoeCausalLM) void {
    const c: *qwen3_model.Qwen3MoeForCausalLM = @ptrCast(@alignCast(clm));
    c.deinit();
}

// ============================================================================
// Gate (Expert Routing)
// ============================================================================

/// GateContext wraps the gate weight matrix. Stored on the heap so we can
/// return a stable pointer. Allocated once in kt_gate_new, freed in kt_gate_free.
///
/// D4: it now also captures the DeepSeek-V3 routing config (n_group,
/// topk_group, norm_topk_prob, routed_scaling_factor, bias pointer).
/// When n_group > 1, kt_gate_forward dispatches to
/// moe.routeExpertsDeepSeek (sigmoid + bias + group-top2 + normalize
/// + scale). When n_group <= 1 (or the legacy config shape), it keeps
/// calling the legacy naive top-k path — same behavior as before D4.
const GateContext = struct {
    weight: [*]const amx.bf16, // [num_experts, hidden_size] BF16
    hidden_size: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    // D4: DeepSeek-V3 routing config.
    n_group: usize = 1,
    topk_group: usize = 1,
    norm_topk_prob: bool = false,
    routed_scaling_factor: f32 = 1.0,
    /// Per-expert e_score_correction_bias, FP32, `num_experts` entries.
    /// Null pointer in the config → no bias.
    bias: ?[*]const f32 = null,
    /// B1: allocator captured at kt_gate_new; kt_gate_free destroys
    /// through it (immune to default-allocator swaps).
    allocator: std.mem.Allocator,
};

pub export fn kt_gate_new(config: kt_gate_config_t) *KT_Gate {
    if (config.weight_type != .KT_TYPE_BF16) @panic("Gate only supports BF16 weights");
    if (config.e_score_correction_bias_type != .KT_TYPE_F32 and config.e_score_correction_bias_type != .KT_TYPE_BF16) {
        // The C header types the bias as a raw pointer; a caller that
        // doesn't have a bias tensor typically passes null. Panic on a
        // non-null pointer with a dtype we can't read, but tolerate the
        // "unspecified" zero enum value used by legacy configs.
    }
    const allocator = defaultAllocator();
    const ctx = allocator.create(GateContext) catch @panic("OOM");
    ctx.* = .{
        .weight = @ptrCast(@alignCast(config.weight)),
        .hidden_size = config.hidden_size,
        .num_experts = config.n_routed_experts,
        .num_experts_per_tok = config.num_experts_per_tok,
        .n_group = @max(1, config.n_group),
        .topk_group = @max(1, config.topk_group),
        .norm_topk_prob = config.norm_topk_prob != 0,
        .routed_scaling_factor = config.routed_scaling_factor,
        .bias = if (config.e_score_correction_bias_type == .KT_TYPE_F32)
            @as(?[*]const f32, @ptrCast(@alignCast(config.e_score_correction_bias)))
        else
            null,
        .allocator = allocator,
    };
    return @ptrCast(ctx);
}

pub export fn kt_gate_free(gate: *KT_Gate) void {
    const ctx: *GateContext = @ptrCast(@alignCast(gate));
    const a = ctx.allocator;
    a.destroy(ctx);
}

pub export fn kt_gate_forward(
    _gate: *KT_Gate,
    input: [*]const amx.bf16,
    _logits: [*]f32, // unused; kept for API compat
    topk_ids: [*]i64,
    topk_weights: [*]f32,
    batch_size: c_int,
    num_experts_per_tok: c_int,
) void {
    _ = _logits;
    const ctx: *GateContext = @ptrCast(@alignCast(_gate));
    const qlen: usize = @intCast(batch_size);
    const k: usize = @intCast(num_experts_per_tok);

    if (ctx.n_group > 1) {
        // D4: DeepSeek-V3 routing — sigmoid + bias + group-top2 +
        // normalization + scaling. Matches the Python reference at
        // ktransformers/kt-kernel/python/sft/layer.py:696-728.
        moe.routeExpertsDeepSeek(
            ctx.allocator,
            input, ctx.weight, qlen, ctx.hidden_size, ctx.num_experts, k,
            .{
                .scoring = .sigmoid,
                .bias = ctx.bias,
                .n_group = ctx.n_group,
                .topk_group = ctx.topk_group,
                .norm_topk_prob = ctx.norm_topk_prob,
                .routed_scaling_factor = ctx.routed_scaling_factor,
            },
            topk_ids, topk_weights,
            null, // no pool: C API doesn't expose a pool; sequential routing
        );
    } else {
        // Legacy path (pre-D4): flat top-k of raw logits. Preserved for
        // configs that don't set group routing (n_group <= 1).
        moe.routeExperts(
            ctx.allocator,
            input, ctx.weight, qlen, ctx.hidden_size, ctx.num_experts, k,
            topk_ids, topk_weights,
            null,
        );
    }
}

// ============================================================================
// Linear
// ============================================================================

/// LinearContext wraps the projection weight. `hidden_size` is the INPUT dim
/// and `intermediate_size` is the OUTPUT dim (matches the C++ reference:
/// `LinearConfig(hidden_size=input_size, intermediate_size=output_size, ...)`).
const LinearContext = struct {
    weight: [*]const amx.bf16, // [out_features, in_features] BF16
    in_features: usize,
    out_features: usize,
    /// B1: allocator captured at construction; used by the per-call
    /// scratch and by kt_linear_free.
    allocator: std.mem.Allocator,
};

export fn kt_linear_new(config: kt_linear_config_t) *KT_Linear {
    if (config.proj_type != .KT_TYPE_BF16) @panic("Linear only supports BF16 weights");
    if (config.hidden_type != .KT_TYPE_BF16) @panic("Linear only supports BF16 input");
    const allocator = defaultAllocator();
    const ctx = allocator.create(LinearContext) catch @panic("OOM");
    ctx.* = .{
        .weight = @ptrCast(@alignCast(config.proj)),
        .in_features = config.hidden_size,
        .out_features = config.intermediate_size,
        .allocator = allocator,
    };
    return @ptrCast(ctx);
}

export fn kt_linear_free(linear: *KT_Linear) void {
    const ctx: *LinearContext = @ptrCast(@alignCast(linear));
    const a = ctx.allocator;
    a.destroy(ctx);
}

export fn kt_linear_forward(
    linear: *KT_Linear,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    batch_size: c_int,
) void {
    const ctx: *LinearContext = @ptrCast(@alignCast(linear));
    const m: usize = @intCast(batch_size);

    const out_f32 = ctx.allocator.alloc(f32, m * ctx.out_features) catch @panic("OOM");
    defer ctx.allocator.free(out_f32);

    // out[m, out] = input[m, in] @ weight^T[out, in]
    // gemmExpert reads weight as [n=out, k=in] row-major, weight_ld = k.
    gemm_bf16.gemmExpert(
        input, ctx.weight, out_f32.ptr,
        m, ctx.out_features, ctx.in_features,
        ctx.in_features, ctx.in_features, ctx.out_features,
    );

    for (0..m * ctx.out_features) |i| {
        output[i] = amx.f32_to_bf16(out_f32[i]);
    }
}

// ============================================================================
// MLP
// ============================================================================

/// MlpContext wraps the three MLP projections. Layout per projection:
///   gate/up: [intermediate_size, hidden_size]
///   down:    [hidden_size, intermediate_size]
const MlpContext = struct {
    gate_w: [*]const amx.bf16,
    up_w: [*]const amx.bf16,
    down_w: [*]const amx.bf16,
    hidden_size: usize,
    intermediate_size: usize,
    /// B1: allocator captured at construction.
    allocator: std.mem.Allocator,
};

export fn kt_mlp_new(config: kt_mlp_config_t) *KT_MLP {
    if (config.gate_type != .KT_TYPE_BF16 or
        config.up_type != .KT_TYPE_BF16 or
        config.down_type != .KT_TYPE_BF16 or
        config.hidden_type != .KT_TYPE_BF16)
        @panic("MLP only supports BF16");
    const allocator = defaultAllocator();
    const ctx = allocator.create(MlpContext) catch @panic("OOM");
    ctx.* = .{
        .gate_w = @ptrCast(@alignCast(config.gate_proj)),
        .up_w = @ptrCast(@alignCast(config.up_proj)),
        .down_w = @ptrCast(@alignCast(config.down_proj)),
        .hidden_size = config.hidden_size,
        .intermediate_size = config.intermediate_size,
        .allocator = allocator,
    };
    return @ptrCast(ctx);
}

export fn kt_mlp_free(mlp_inst: *KT_MLP) void {
    const ctx: *MlpContext = @ptrCast(@alignCast(mlp_inst));
    const a = ctx.allocator;
    a.destroy(ctx);
}

export fn kt_mlp_forward(
    mlp_inst: *KT_MLP,
    input: [*]const amx.bf16,
    output: [*]amx.bf16,
    batch_size: c_int,
) void {
    const ctx: *MlpContext = @ptrCast(@alignCast(mlp_inst));
    const m: usize = @intCast(batch_size);
    const h = ctx.hidden_size;
    const i_dim = ctx.intermediate_size;

    // F32 intermediate buffers. gemmExpert writes F32; using F32 buffers
    // avoids the BF16/F32 size-mismatch that would happen if we tried to
    // store F32 into a BF16-sized allocation. We round-trip through BF16
    // for the down GEMM because gemmExpert's input is BF16.
    const a = ctx.allocator; // B1
    const gate_buf = a.alloc(f32, m * i_dim) catch @panic("OOM");
    const up_buf = a.alloc(f32, m * i_dim) catch @panic("OOM");
    const down_buf = a.alloc(f32, m * h) catch @panic("OOM");
    defer {
        a.free(gate_buf);
        a.free(up_buf);
        a.free(down_buf);
    }

    // gate[m, i] = input[m, h] @ gate_w^T[i, h]
    // gemmExpert reads weight as [n=i_dim, k=h] row-major, weight_ld = k = h.
    gemm_bf16.gemmExpert(
        input, ctx.gate_w, gate_buf.ptr,
        m, i_dim, h,
        h, h, i_dim,
    );
    // up[m, i] = input[m, h] @ up_w^T[i, h]
    gemm_bf16.gemmExpert(
        input, ctx.up_w, up_buf.ptr,
        m, i_dim, h,
        h, h, i_dim,
    );
    // gate_buf = swiglu(gate_buf, up_buf) — in place
    for (0..m * i_dim) |idx| {
        gate_buf[idx] = amx.swiglu(gate_buf[idx], up_buf[idx]);
    }
    // Convert gate_buf F32 -> BF16 view for the down GEMM input
    const swiglu_bf16 = a.alloc(amx.bf16, m * i_dim) catch @panic("OOM");
    defer a.free(swiglu_bf16);
    for (0..m * i_dim) |idx| {
        swiglu_bf16[idx] = amx.f32_to_bf16(gate_buf[idx]);
    }
    // down[m, h] = swiglu_out[m, i] @ down_w^T[h, i]
    // down_w: [n=h, k=i_dim] row-major, weight_ld = k = i_dim.
    gemm_bf16.gemmExpert(
        swiglu_bf16.ptr, ctx.down_w, down_buf.ptr,
        m, h, i_dim,
        i_dim, i_dim, h,
    );

    // F32 -> BF16 conversion
    for (0..m * h) |i| {
        output[i] = amx.f32_to_bf16(down_buf[i]);
    }
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
// FP8 Layerwise Transport — callback-only CPU port
// ============================================================================
//
// Coordinator-confirmed scope: the transport stores the per-rank host
// buffer state handed in by the caller and, in run_producer, hands the
// caller's callback the correct per-expert w13/w2 weight+scale pointers.
// It does NOT run GEMMs (the Python caller does, via kt_matmul_fp8) and
// it does NOT stage H2D copies — so `cuda_device`, `local_gpu_ptrs` and
// `timeout_ms` are accepted for ABI compatibility and ignored (no GPU in
// the CPU port, no cross-process waits to time out).
//
// Pointer-layout contract (mirrors ktransformers kt-kernel
// fp8_layerwise_transport.cpp::run_producer and the write_weight_scale_to_buffer
// consumer; see also examples/test_k2_write_buffer.py::get_expert_ptrs):
//
//   local_host_ptrs:    [slot][kind]           — FP8_HOST_SLOTS * FP8_BUFFER_KINDS = 8
//   all_rank_host_ptrs: [slot][rank][kind]     — 2 * tp_size * 4 (rank zero only)
//     index = (slot * tp_size + rank) * FP8_BUFFER_KINDS + kind
//   expert_nbytes:      [kind]                 — 4 (w13_weight, w13_scale, w2_weight, w2_scale)
//
// The C API passes raw arrays without lengths (the protocol constants above
// fix them); shorter arrays from the caller are UB, as in any C API.

const FP8_CONTROL_BYTES: usize = 8192;
const FP8_MAX_TP_SIZE: c_int = 8;
const FP8_BUFFER_KINDS: usize = 4; // w13_weight, w13_scale, w2_weight, w2_scale
const FP8_HOST_SLOTS: usize = 2;
const FP8_CONTROL_MAGIC: u64 = 0x4b544650384c5731; // "KTFP8LW1" (reference: fp8_layerwise_transport.cpp:29)
const FP8_CONTROL_VERSION: u64 = 2;
const FP8_ERROR_PROTOCOL: c_int = 1; // mirrors kErrorProtocol in fp8_layerwise_transport.cpp:35

/// First 64 bytes of the shared control region, matching the reference
/// ControlHeader layout (fp8_layerwise_transport.cpp:49-56). The full
/// multi-process slot/ack/poison protocol is not used by the CPU port:
/// init writes the header, transport_new validates it.
const Fp8ControlHeader = extern struct {
    magic: u64,
    version: u64,
    struct_bytes: u64,
    tp_size: u64,
    padding: [32]u8,
};

/// Writer callback type passed to kt_fp8_transport_run_producer. Each call
/// receives four arrays of `tp_size` host pointer values; the kind order
/// matches the C header: w13_weight, w13_scale, w2_weight, w2_scale.
pub const Fp8WriterCallback = *const fn (
    expert_id: c_int,
    w13_weight_ptrs: [*]const usize,
    w13_scale_ptrs: [*]const usize,
    w2_weight_ptrs: [*]const usize,
    w2_scale_ptrs: [*]const usize,
) callconv(.c) void;

const Fp8Transport = struct {
    rank: c_int,
    tp_size: c_int,
    num_experts: c_int,
    local_host_ptrs: [FP8_HOST_SLOTS * FP8_BUFFER_KINDS]usize,
    all_rank_host_ptrs: []usize, // [slot][rank][kind] on rank zero, empty on other ranks
    expert_nbytes: [FP8_BUFFER_KINDS]usize,
    closed: bool,
    // Last producer-run statistics (CPU-measured; reference fields are the same).
    last_epoch: u64,
    last_layer_id: c_int,
    last_expert_count: c_int,
    writer_ns: u64,
    slot_wait_ns: u64,
    total_ns: u64,
    /// B1: allocator captured at kt_fp8_transport_new; the free paths
    /// (kt_fp8_transport_free) destroy through it.
    allocator: std.mem.Allocator,
};

/// Validate the control region header and cast it to a typed pointer.
/// Panics on any mismatch — the control region is the C API's only error
/// channel for the init/new contract (no exception path across C ABI).
fn fp8CheckedControl(control_ptr: ?*anyopaque, control_size: usize, tp_size: c_int) *const Fp8ControlHeader {
    const ptr = control_ptr orelse @panic("FP8 layerwise control pointer is null");
    if (@intFromPtr(ptr) & 63 != 0) @panic("FP8 layerwise control pointer must be 64-byte aligned");
    if (control_size < FP8_CONTROL_BYTES) @panic("FP8 layerwise control region must be at least 8192 bytes");
    if (tp_size <= 0 or tp_size > FP8_MAX_TP_SIZE) @panic("FP8 layerwise transport supports TP sizes 1 through 8");
    const header: *const Fp8ControlHeader = @ptrCast(@alignCast(ptr));
    if (header.magic != FP8_CONTROL_MAGIC) @panic("FP8 layerwise control is uninitialized or has an incompatible ABI");
    if (header.version != FP8_CONTROL_VERSION or header.struct_bytes != @sizeOf(Fp8ControlHeader)) {
        @panic("FP8 layerwise control is uninitialized or has an incompatible ABI");
    }
    if (header.tp_size != @as(u64, @intCast(tp_size))) {
        @panic("FP8 layerwise control TP size does not match transport TP size");
    }
    return header;
}

fn fp8CheckedTransport(transport: ?*KT_FP8LayerwiseTransport) *Fp8Transport {
    return @ptrCast(@alignCast(transport orelse @panic("FP8 layerwise transport is null")));
}

fn fp8NowNs() u64 {
    var ts: std.os.linux.timespec = .{ .sec = 0, .nsec = 0 };
    _ = std.os.linux.syscall2(.clock_gettime, @intFromEnum(std.os.linux.CLOCK.MONOTONIC), @intFromPtr(&ts));
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn fp8NsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1.0e6;
}

fn fp8PoisonedStats(epoch: u64, rank: c_int, message: [*]const u8) kt_fp8_stats_t {
    return .{
        .epoch = epoch,
        .layer_id = 0,
        .expert_count = 0,
        .rank = rank,
        .writer_ms = 0,
        .slot_wait_ms = 0,
        .h2d_ms = 0,
        .total_ms = 0,
        .bytes = 0,
        .poisoned = 1,
        .error_code = FP8_ERROR_PROTOCOL,
        .error_rank = rank,
        .error_message = message,
    };
}

pub export fn kt_fp8_layerwise_init(control_ptr: ?*anyopaque, control_size: usize, tp_size: c_int) void {
    // Mirror reference initialize_fp8_layerwise_control: validate the control
    // region, zero it, and stamp the header. Mirroring the magic/version
    // layout keeps drop-in compatibility with a Python caller that
    // initializes the same buffer through this C API.
    const ptr = control_ptr orelse @panic("FP8 layerwise control pointer is null");
    if (@intFromPtr(ptr) & 63 != 0) @panic("FP8 layerwise control pointer must be 64-byte aligned");
    if (control_size < FP8_CONTROL_BYTES) @panic("FP8 layerwise control region must be at least 8192 bytes");
    if (tp_size <= 0 or tp_size > FP8_MAX_TP_SIZE) @panic("FP8 layerwise transport supports TP sizes 1 through 8");
    @memset(@as([*]u8, @ptrCast(ptr))[0..FP8_CONTROL_BYTES], 0);
    const header: *Fp8ControlHeader = @ptrCast(@alignCast(ptr));
    header.magic = FP8_CONTROL_MAGIC;
    header.version = FP8_CONTROL_VERSION;
    header.struct_bytes = @sizeOf(Fp8ControlHeader);
    header.tp_size = @intCast(tp_size);
    // padding[32] left zeroed by the memset above.
}

pub export fn kt_fp8_transport_new(
    control_ptr: ?*anyopaque,
    control_size: usize,
    rank: c_int,
    tp_size: c_int,
    cuda_device: c_int,
    local_host_ptrs: ?[*]const usize,
    local_gpu_ptrs: ?[*]const usize,
    all_rank_host_ptrs: ?[*]const usize,
    expert_nbytes: ?[*]const usize,
    num_experts: c_int,
    timeout_ms: c_int,
) *KT_FP8LayerwiseTransport {
    _ = cuda_device; // no GPU in the CPU port (documented no-op)
    _ = local_gpu_ptrs; // ditto
    _ = timeout_ms; // in-process: no cross-rank waits to time out

    _ = fp8CheckedControl(control_ptr, control_size, tp_size);

    if (rank < 0 or rank >= tp_size) @panic("FP8 layerwise rank is outside TP range");
    if (num_experts <= 0) @panic("FP8 layerwise num_experts must be positive");

    const local = local_host_ptrs orelse @panic("FP8 layerwise local_host_ptrs must contain 8 pointers in [slot][kind] order");
    const nbytes = expert_nbytes orelse @panic("FP8 layerwise expert_nbytes must contain 4 sizes");

    const tp: usize = @intCast(tp_size);
    const all_count: usize = FP8_HOST_SLOTS * tp * FP8_BUFFER_KINDS;
    const allocator = defaultAllocator(); // B1

    const all_slice: []usize = if (rank == 0) blk: {
        const src = all_rank_host_ptrs orelse @panic("FP8 layerwise all_rank_host_ptrs is required on rank zero");
        const buf = allocator.alloc(usize, all_count) catch @panic("OOM");
        for (0..all_count) |i| {
            if (src[i] == 0) @panic("FP8 layerwise buffer pointers must be non-zero");
            buf[i] = src[i];
        }
        break :blk buf;
    } else allocator.alloc(usize, 0) catch @panic("OOM");

    var local_buf: [FP8_HOST_SLOTS * FP8_BUFFER_KINDS]usize = undefined;
    for (0..FP8_HOST_SLOTS * FP8_BUFFER_KINDS) |i| {
        if (local[i] == 0) @panic("FP8 layerwise buffer pointers must be non-zero");
        local_buf[i] = local[i];
    }
    var nb_buf: [FP8_BUFFER_KINDS]usize = undefined;
    for (0..FP8_BUFFER_KINDS) |i| {
        if (nbytes[i] == 0) @panic("FP8 layerwise expert byte sizes must be non-zero");
        nb_buf[i] = nbytes[i];
    }

    const t = allocator.create(Fp8Transport) catch @panic("OOM");
    t.* = .{
        .rank = rank,
        .tp_size = tp_size,
        .num_experts = num_experts,
        .local_host_ptrs = local_buf,
        .all_rank_host_ptrs = all_slice,
        .expert_nbytes = nb_buf,
        .closed = false,
        .last_epoch = 0,
        .last_layer_id = -1,
        .last_expert_count = 0,
        .writer_ns = 0,
        .slot_wait_ns = 0,
        .total_ns = 0,
        .allocator = allocator,
    };
    return @ptrCast(t);
}

pub export fn kt_fp8_transport_run_producer(
    transport: ?*KT_FP8LayerwiseTransport,
    epoch: u64,
    layer_id: c_int,
    expert_count: c_int,
    callback: ?Fp8WriterCallback,
) void {
    const t: *Fp8Transport = fp8CheckedTransport(transport);
    if (t.closed) @panic("FP8 layerwise transport is closed");
    if (t.rank != 0) @panic("FP8 layerwise run_producer is rank-zero only");
    const cb = callback orelse @panic("FP8 layerwise writer callback is null");
    if (epoch == 0) @panic("FP8 layerwise epoch zero is reserved");
    if (layer_id < 0) @panic("FP8 layerwise layer_id must be non-negative");
    if (expert_count <= 0 or expert_count > t.num_experts) {
        @panic("FP8 layerwise expert_count is outside the configured expert range");
    }

    const tp: usize = @intCast(t.tp_size);
    const ec: usize = @intCast(expert_count);

    // Per-call scratch: four arrays of FP8_MAX_TP_SIZE entries; only the
    // first tp_size entries are populated and visible to the callback.
    var w13_weight: [FP8_MAX_TP_SIZE]usize = undefined;
    var w13_scale: [FP8_MAX_TP_SIZE]usize = undefined;
    var w2_weight: [FP8_MAX_TP_SIZE]usize = undefined;
    var w2_scale: [FP8_MAX_TP_SIZE]usize = undefined;

    var writer_ns: u64 = 0;
    const begin = fp8NowNs();
    for (0..ec) |expert| {
        // Reference: ptrs[kind][rank] = all_rank_host_ptrs_[(slot * tp_size_ + rank) * 4 + kind],
        // with slot = expert % kFP8LayerwiseHostSlots (= 2). This is the
        // exact index formula from fp8_layerwise_transport.cpp:409-417; the
        // expert_nbytes array only feeds the bytes stat (handled in wait).
        const slot = expert % FP8_HOST_SLOTS;
        for (0..tp) |r| {
            const base = (slot * tp + r) * FP8_BUFFER_KINDS;
            w13_weight[r] = t.all_rank_host_ptrs[base + 0];
            w13_scale[r] = t.all_rank_host_ptrs[base + 1];
            w2_weight[r] = t.all_rank_host_ptrs[base + 2];
            w2_scale[r] = t.all_rank_host_ptrs[base + 3];
        }
        const writer_begin = fp8NowNs();
        cb(@intCast(expert), &w13_weight, &w13_scale, &w2_weight, &w2_scale);
        writer_ns += fp8NowNs() - writer_begin;
    }
    t.total_ns = fp8NowNs() - begin;
    t.writer_ns = writer_ns;
    t.slot_wait_ns = 0; // single-process CPU port: no consumer slot acks to wait for
    t.last_epoch = epoch;
    t.last_layer_id = layer_id;
    t.last_expert_count = expert_count;
}

pub export fn kt_fp8_transport_wait(transport: ?*KT_FP8LayerwiseTransport, epoch: u64) kt_fp8_stats_t {
    const t: *Fp8Transport = fp8CheckedTransport(transport);
    if (t.closed) @panic("FP8 layerwise transport is closed");
    if (epoch == 0) {
        return fp8PoisonedStats(0, t.rank, "FP8 layerwise epoch zero is reserved");
    }
    if (t.last_epoch != epoch) {
        return fp8PoisonedStats(epoch, t.rank, "FP8 layerwise wait called for an epoch that was never produced");
    }
    // Reference semantics: per expert the consumer adds Σ expert_nbytes to
    // the local bytes counter (fp8_layerwise_transport.cpp::copy_expert),
    // so the per-rank layer total is expert_count × Σ expert_nbytes. The
    // CPU port surfaces the same number.
    var per_expert_bytes: usize = 0;
    for (t.expert_nbytes) |n| per_expert_bytes += n;
    const ec: usize = @intCast(t.last_expert_count);
    return .{
        .epoch = epoch,
        .layer_id = t.last_layer_id,
        .expert_count = t.last_expert_count,
        .rank = t.rank,
        .writer_ms = fp8NsToMs(t.writer_ns),
        .slot_wait_ms = fp8NsToMs(t.slot_wait_ns),
        .h2d_ms = 0, // CPU port: no H2D copies
        .total_ms = fp8NsToMs(t.total_ns),
        .bytes = per_expert_bytes * ec,
        .poisoned = 0,
        .error_code = 0,
        .error_rank = -1,
        .error_message = "",
    };
}

pub export fn kt_fp8_transport_close(transport: ?*KT_FP8LayerwiseTransport) void {
    // Idempotent (reference uses compare_exchange; here plain bool assignment
    // on a single-threaded C API is sufficient).
    const t: *Fp8Transport = @ptrCast(@alignCast(transport orelse return));
    t.closed = true;
}

pub export fn kt_fp8_transport_free(transport: ?*KT_FP8LayerwiseTransport) void {
    // Match the C free(NULL) idiom: a null transport is a no-op.
    const t: *Fp8Transport = @ptrCast(@alignCast(transport orelse return));
    const a = t.allocator; // B1: captured allocator, not the default
    a.free(t.all_rank_host_ptrs);
    a.destroy(t);
}

pub export fn kt_fp8_transport_rank(transport: ?*KT_FP8LayerwiseTransport) c_int {
    const t: *Fp8Transport = @ptrCast(@alignCast(transport orelse return -1));
    return t.rank;
}

pub export fn kt_fp8_transport_tp_size(transport: ?*KT_FP8LayerwiseTransport) c_int {
    const t: *Fp8Transport = @ptrCast(@alignCast(transport orelse return 0));
    return t.tp_size;
}

pub export fn kt_fp8_transport_num_experts(transport: ?*KT_FP8LayerwiseTransport) c_int {
    const t: *Fp8Transport = @ptrCast(@alignCast(transport orelse return 0));
    return t.num_experts;
}

pub export fn kt_fp8_transport_closed(transport: ?*KT_FP8LayerwiseTransport) c_int {
    const t: *Fp8Transport = @ptrCast(@alignCast(transport orelse return 1));
    return if (t.closed) 1 else 0;
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
// GGML Quantization Types (byte-exact vs llama.cpp ggml-common.h)
// ============================================================================
// One-row quantize/dequantize per format. The Python layer quantizes weight
// rows with these; the block layouts match GGUF files exactly, so weights
// loaded directly from .gguf can be passed to the matmuls below unchanged.

pub export fn kt_quantize_q8_0(src: [*]const f32, dst: *anyopaque, k: usize) void {
    gemm_q8_0.quantizeRowQ8_0(src[0..k], @ptrCast(@alignCast(dst)), k);
}

pub export fn kt_dequantize_q8_0(src: *const anyopaque, dst: [*]f32, k: usize) void {
    gemm_q8_0.dequantizeRowQ8_0(@ptrCast(@alignCast(src)), dst[0..k], k);
}

pub export fn kt_quantize_q4_k(src: [*]const f32, dst: *anyopaque, k: usize) void {
    gemm_q4_k.quantizeRowQ4_K(src[0..k], @ptrCast(@alignCast(dst)), k);
}

pub export fn kt_dequantize_q4_k(src: *const anyopaque, dst: [*]f32, k: usize) void {
    gemm_q4_k.dequantizeRowQ4_K(@ptrCast(@alignCast(src)), dst[0..k], k);
}

pub export fn kt_quantize_q5_k(src: [*]const f32, dst: *anyopaque, k: usize) void {
    gemm_q5_k.quantizeRowQ5_K(src[0..k], @ptrCast(@alignCast(dst)), k);
}

pub export fn kt_dequantize_q5_k(src: *const anyopaque, dst: [*]f32, k: usize) void {
    gemm_q5_k.dequantizeRowQ5_K(@ptrCast(@alignCast(src)), dst[0..k], k);
}

pub export fn kt_quantize_q6_k(src: [*]const f32, dst: *anyopaque, k: usize) void {
    gemm_q6_k.quantizeRowQ6_K(src[0..k], @ptrCast(@alignCast(dst)), k);
}

pub export fn kt_dequantize_q6_k(src: *const anyopaque, dst: [*]f32, k: usize) void {
    gemm_q6_k.dequantizeRowQ6_K(@ptrCast(@alignCast(src)), dst[0..k], k);
}

pub export fn kt_quantize_q8_k(src: [*]const f32, dst: *anyopaque, k: usize) void {
    gemm_q8_k.quantizeRowQ8_K(src[0..k], @ptrCast(@alignCast(dst)), k);
}

pub export fn kt_dequantize_q8_k(src: *const anyopaque, dst: [*]f32, k: usize) void {
    gemm_q8_k.dequantizeRowQ8_K(@ptrCast(@alignCast(src)), dst[0..k], k);
}

/// GGML GEMMs: a [m,k] BF16 activations, b [n, k/QK] blocks (row-major,
/// ldb in BLOCKS), c [m,n] F32. On-the-fly dequant (scalar kernels).
pub export fn kt_matmul_q8_0(
    a: [*]const amx.bf16,
    b: [*]const gemm_q8_0.BlockQ8_0,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    gemm_q8_0.gemmQ8_0Scalar(a, lda, b, ldb, c, ldc, m, n, k);
}

pub export fn kt_matmul_q4_k(
    a: [*]const amx.bf16,
    b: [*]const gemm_q4_k.BlockQ4_K,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    gemm_q4_k.gemmQ4_KScalar(a, lda, b, ldb, c, ldc, m, n, k);
}

pub export fn kt_matmul_q5_k(
    a: [*]const amx.bf16,
    b: [*]const gemm_q5_k.BlockQ5_K,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    gemm_q5_k.gemmQ5_KScalar(a, lda, b, ldb, c, ldc, m, n, k);
}

pub export fn kt_matmul_q6_k(
    a: [*]const amx.bf16,
    b: [*]const gemm_q6_k.BlockQ6_K,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    gemm_q6_k.gemmQ6_KScalar(a, lda, b, ldb, c, ldc, m, n, k);
}

pub export fn kt_matmul_q8_k(
    a: [*]const amx.bf16,
    b: [*]const gemm_q8_k.BlockQ8_K,
    c: [*]f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) void {
    gemm_q8_k.gemmQ8_KScalar(a, lda, b, ldb, c, ldc, m, n, k);
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
