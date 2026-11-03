//! Kimi K3 model configuration for ktransformers-zig.
//! Port of zig-ai's kimi_k3_config.zig.

const std = @import("std");

pub const K3_MAX_TOPK: usize = 64;

pub const KimiK3Config = struct {
    hidden: usize,
    n_layers: usize,
    vocab: usize,
    rms_eps: f32,

    kda_heads: usize,
    kda_head_dim: usize,
    conv_k: usize,
    gate_lb: f32,

    n_heads: usize,
    q_lora: usize,
    kv_lora: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head: usize,
    mla_out_gate: bool,

    n_experts: usize,
    topk: usize,
    n_shared: usize,
    latent: usize,
    moe_inter: usize,
    routed_scale: f32,
    moe_renorm: bool,
    latent_norm: bool,

    first_dense: usize,
    dense_inter: usize,

    attn_res_block: usize,

    situ_b1: f32,
    situ_b2: f32,

    n_full_attn: usize,
    full_attn: [256]i32 = [_]i32{0} ** 256,
};

pub fn isMla(cfg: *const KimiK3Config, layer: usize) bool {
    const one: i32 = @intCast(layer + 1);
    for (cfg.full_attn[0..cfg.n_full_attn]) |v| {
        if (v == one) return true;
    }
    return false;
}

pub fn isKda(cfg: *const KimiK3Config, layer: usize) bool {
    return !isMla(cfg, layer);
}

pub fn isDense(cfg: *const KimiK3Config, layer: usize) bool {
    return layer < cfg.first_dense;
}
