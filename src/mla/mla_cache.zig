// MLA Compressed KV Cache
// Stores compressed latent vectors instead of full K/V
// Memory: kv_lora_rank per token instead of num_heads * head_dim
//
// Zig 0.16 note: std.ArrayList is unmanaged here — init with `.empty`,
// pass the allocator to every mutating call, `deinit(allocator)`.

const std = @import("std");
const MlaConfig = @import("mla_config.zig").MlaConfig;

/// Compressed KV cache page
/// Each page stores `token_count_in_page` tokens, each with:
/// - `kv_lora_rank` floats: compressed KV (nope part)
/// - `rope_size` floats: positional embeddings (rope part)
pub const MlaCachePage = struct {
    nope_data: []align(64) f32, // [token_count_in_page, kv_lora_rank] compressed KV
    rope_data: []align(64) f32, // [token_count_in_page, rope_size] positional embeddings
    tokens_used: usize, // How many tokens are stored in this page

    pub fn init(allocator: std.mem.Allocator, tokens_per_page: usize, kv_lora_rank: usize, rope_size: usize) !MlaCachePage {
        const nope = try allocator.alignedAlloc(f32, .@"64", tokens_per_page * kv_lora_rank);
        const rope = try allocator.alignedAlloc(f32, .@"64", tokens_per_page * rope_size);
        @memset(nope, 0);
        @memset(rope, 0);
        return MlaCachePage{
            .nope_data = nope,
            .rope_data = rope,
            .tokens_used = 0,
        };
    }

    pub fn deinit(self: *MlaCachePage, allocator: std.mem.Allocator) void {
        allocator.free(self.nope_data);
        allocator.free(self.rope_data);
    }

    /// Get pointer to nope vector for a specific token in the page
    pub fn nopePtr(self: *const MlaCachePage, token_in_page: usize, kv_lora_rank: usize) [*]f32 {
        return self.nope_data.ptr + token_in_page * kv_lora_rank;
    }

    /// Get pointer to rope vector for a specific token in the page
    pub fn ropePtr(self: *const MlaCachePage, token_in_page: usize, rope_size: usize) [*]f32 {
        return self.rope_data.ptr + token_in_page * rope_size;
    }

    /// Store compressed KV for a token
    pub fn storeNope(self: *MlaCachePage, token_in_page: usize, kv_lora_rank: usize, data: [*]const f32) void {
        const dst = self.nopePtr(token_in_page, kv_lora_rank);
        @memcpy(dst[0..kv_lora_rank], data[0..kv_lora_rank]);
    }

    /// Store RoPE embeddings for a token
    pub fn storeRope(self: *MlaCachePage, token_in_page: usize, rope_size: usize, data: [*]const f32) void {
        const dst = self.ropePtr(token_in_page, rope_size);
        @memcpy(dst[0..rope_size], data[0..rope_size]);
    }
};

/// MLA KV Cache Manager
/// Manages paged compressed KV cache
pub const MlaKvCache = struct {
    pages: std.ArrayList(MlaCachePage),
    page_table: std.ArrayList(usize), // token_pos -> page_index
    allocator: std.mem.Allocator,
    config: MlaConfig,
    total_tokens: usize,

    pub fn init(allocator: std.mem.Allocator, config: MlaConfig, max_pages: usize) !MlaKvCache {
        var pages = try std.ArrayList(MlaCachePage).initCapacity(allocator, max_pages);
        errdefer pages.deinit(allocator);
        var page_table = try std.ArrayList(usize).initCapacity(allocator, config.max_kvlen);
        errdefer page_table.deinit(allocator);

        return MlaKvCache{
            .pages = pages,
            .page_table = page_table,
            .allocator = allocator,
            .config = config,
            .total_tokens = 0,
        };
    }

    pub fn deinit(self: *MlaKvCache) void {
        for (self.pages.items) |*page| {
            page.deinit(self.allocator);
        }
        self.pages.deinit(self.allocator);
        self.page_table.deinit(self.allocator);
    }

    /// Allocate a new page
    pub fn allocPage(self: *MlaKvCache) !usize {
        const page_idx = self.pages.items.len;
        const page = try MlaCachePage.init(
            self.allocator,
            self.config.token_count_in_page,
            self.config.kv_lora_rank,
            self.config.rope_size,
        );
        try self.pages.append(self.allocator, page);
        return page_idx;
    }

    /// Append a token to the cache
    pub fn appendToken(self: *MlaKvCache, nope: [*]const f32, rope: [*]const f32) !void {
        const token_pos = self.total_tokens;
        const page_idx = token_pos / self.config.token_count_in_page;
        const token_in_page = token_pos % self.config.token_count_in_page;

        // Allocate new page if needed
        if (page_idx >= self.pages.items.len) {
            _ = try self.allocPage();
        }

        // Update page table
        if (token_pos >= self.page_table.items.len) {
            try self.page_table.append(self.allocator, page_idx);
        } else {
            self.page_table.items[token_pos] = page_idx;
        }

        // Store data
        const page = &self.pages.items[page_idx];
        page.storeNope(token_in_page, self.config.kv_lora_rank, nope);
        page.storeRope(token_in_page, self.config.rope_size, rope);
        page.tokens_used = token_in_page + 1;

        self.total_tokens += 1;
    }

    /// Get page and offset for a token position
    pub fn getPageInfo(self: *const MlaKvCache, token_pos: usize) struct { page_idx: usize, token_in_page: usize } {
        const page_idx = self.page_table.items[token_pos];
        const token_in_page = token_pos % self.config.token_count_in_page;
        return .{ .page_idx = page_idx, .token_in_page = token_in_page };
    }

    /// Get pointer to nope data for a token
    pub fn getNopePtr(self: *const MlaKvCache, token_pos: usize) [*]f32 {
        const info = self.getPageInfo(token_pos);
        return self.pages.items[info.page_idx].nopePtr(info.token_in_page, self.config.kv_lora_rank);
    }

    /// Get pointer to rope data for a token
    pub fn getRopePtr(self: *const MlaKvCache, token_pos: usize) [*]f32 {
        const info = self.getPageInfo(token_pos);
        return self.pages.items[info.page_idx].ropePtr(info.token_in_page, self.config.rope_size);
    }

    /// Get total KV length
    pub fn kvLen(self: *const MlaKvCache) usize {
        return self.total_tokens;
    }

    /// Clear cache (pages are retained for reuse)
    pub fn clear(self: *MlaKvCache) void {
        self.total_tokens = 0;
        for (self.pages.items) |*page| {
            page.tokens_used = 0;
        }
        self.page_table.clearRetainingCapacity();
    }
};
