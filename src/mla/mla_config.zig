// MLA (Multi-Head Latent Attention) Configuration
// Based on DeepSeek-V2/V3 architecture
// Reference: ktransformers kt-kernel/operators/common.hpp (GeneralMLAConfig)
//            ktransformers kt-kernel/operators/llamafile/mla.hpp

const std = @import("std");

/// MLA configuration matching DeepSeek-V2/V3
pub const MlaConfig = struct {
    // Model dimensions
    hidden_size: usize = 7168, // e.g., 7168 for DeepSeek-V3
    num_heads: usize = 128, // Number of attention heads
    q_lora_rank: usize = 1536, // Q compression rank (q_lora_rank << hidden_size)
    kv_lora_rank: usize = 512, // KV compression rank (kv_lora_rank << hidden_size)
    nope_size: usize = 128, // Non-positional head dimension
    rope_size: usize = 64, // Positional (RoPE) head dimension
    // head_dim = nope_size + rope_size = 192 for DeepSeek-V3

    // Sequence lengths
    max_qlen: usize = 4096, // Max query length
    max_kvlen: usize = 16384, // Max KV cache length
    token_count_in_page: usize = 64, // Tokens per KV cache page

    // RoPE parameters
    rope_theta: f64 = 10000.0,
    rope_scaling_factor: f64 = 1.0,
    rope_scaling_mscale: f64 = 1.0,

    // Weights (pointers to BF16 data)
    q_a_proj: [*]const u16 = undefined, // [q_lora_rank, hidden_size] BF16
    q_a_norm: [*]const u16 = undefined, // [q_lora_rank] BF16
    q_b_proj: [*]const u16 = undefined, // [num_heads, nope_size+rope_size, q_lora_rank] BF16

    kv_a_proj_with_mqa: [*]const u16 = undefined, // [kv_lora_rank+rope_size, hidden_size] BF16
    kv_a_norm: [*]const u16 = undefined, // [kv_lora_rank] BF16
    kv_b_proj: [*]const u16 = undefined, // [num_heads, 2*nope_size, kv_lora_rank] BF16
    // Note: kv_b_proj contains both W_UK (up-projection for keys) and W_UV (up-projection for values)
    // In DeepSeek-V2, kv_b_proj has shape [num_heads, 2*nope_size, kv_lora_rank]
    // where [:, 0:nope_size, :] = W_UK and [:, nope_size:2*nope_size, :] = W_UV

    o_proj: [*]const u16 = undefined, // [hidden_size, num_heads*nope_size] BF16

    // Weight types for quantization
    q_a_proj_type: u8 = 0, // 0=BF16, 1=INT8, 2=INT4, etc.
    q_b_proj_type: u8 = 0,
    kv_a_proj_type: u8 = 0,
    kv_b_proj_type: u8 = 0,
    o_proj_type: u8 = 0,

    // TP (Tensor Parallelism)
    tp_part_idx: usize = 0,
    tp_count: usize = 1,

    pub fn headDim(self: MlaConfig) usize {
        return self.nope_size + self.rope_size;
    }

    pub fn numHeadsPerTp(self: MlaConfig) usize {
        return self.num_heads / self.tp_count;
    }

    pub fn qLoraRank(self: MlaConfig) usize {
        return self.q_lora_rank;
    }

    pub fn kvLoraRank(self: MlaConfig) usize {
        return self.kv_lora_rank;
    }
};

/// Page table entry for MLA KV cache
pub const MlaPageTable = struct {
    page_indices: []usize, // Maps token position -> page index
    kv_len: usize, // Current KV length

    pub fn pageForToken(self: MlaPageTable, token_pos: usize, tokens_per_page: usize) usize {
        return self.page_indices[token_pos / tokens_per_page];
    }

    pub fn tokenInPage(self: MlaPageTable, token_pos: usize, tokens_per_page: usize) usize {
        _ = self;
        return token_pos % tokens_per_page;
    }
};
