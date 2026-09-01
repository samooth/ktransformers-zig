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

    /// Append a token to the cache under a custom page_table. The
    /// `page_table[token_pos]` slot must already point at a valid
    /// page (use `allocPage()` first if not). This is the paged
    /// counterpart to the existing `appendToken`.
    pub fn appendTokenPaged(
        self: *MlaKvCache,
        page_table: [*]usize,
        token_pos: usize,
        nope: [*]const f32,
        rope: [*]const f32,
    ) !void {
        const page_idx = page_table[token_pos];
        const token_in_page = token_pos % self.config.token_count_in_page;
        const page = &self.pages.items[page_idx];
        page.storeNope(token_in_page, self.config.kv_lora_rank, nope);
        page.storeRope(token_in_page, self.config.rope_size, rope);
        page.tokens_used = @max(page.tokens_used, token_in_page + 1);
    }

    /// Get total KV length
    pub fn kvLen(self: *const MlaKvCache) usize {
        return self.total_tokens;
    }

    // ===================================================================
    // Paged access (caller-supplied page tables — kt_mla_forward path)
    // ===================================================================
    //
    // The paged C API (kt_mla_forward with qlen_count>1) passes per-sequence
    // page tables: `page_table[logical_pos / token_count_in_page]` = index
    // into the cache's `pages` array (mirroring the C++ mla-tp.hpp contract
    // where pages are registered via set_pages and indexed per query).
    // The engine writes each query's new KVs through the same tables.

    /// Resolve (page_idx, token_in_page) for a logical position through a
    /// caller-supplied page table (c_int array, per the C ABI).
    pub fn pagedGetPageInfo(
        self: *const MlaKvCache,
        page_table: [*]const c_int,
        logical_pos: usize,
    ) struct { page_idx: usize, token_in_page: usize } {
        const tpp = self.config.token_count_in_page;
        const page_idx: usize = @intCast(page_table[logical_pos / tpp]);
        const token_in_page = logical_pos % tpp;
        return .{ .page_idx = page_idx, .token_in_page = token_in_page };
    }

    /// Ensure the cache has at least `n` physical pages, allocating zeroed
    /// pages as needed. Returns the new page count.
    pub fn ensurePageCount(self: *MlaKvCache, n: usize) !usize {
        while (self.pages.items.len < n) _ = try self.allocPage();
        return self.pages.items.len;
    }

    /// Write a token's (nope, rope) at an arbitrary logical position,
    /// routed through the caller's page table. The page must already
    /// exist (ensurePageCount). Does NOT touch total_tokens (the paged
    /// path manages lengths per-sequence via kv_lens).
    pub fn pagedWriteToken(
        self: *MlaKvCache,
        page_table: [*]const c_int,
        logical_pos: usize,
        nope: [*]const f32,
        rope: [*]const f32,
    ) !void {
        const info = self.pagedGetPageInfo(page_table, logical_pos);
        const page = &self.pages.items[info.page_idx];
        page.storeNope(info.token_in_page, self.config.kv_lora_rank, nope);
        page.storeRope(info.token_in_page, self.config.rope_size, rope);
        page.tokens_used = @max(page.tokens_used, info.token_in_page + 1);
    }

    /// Get pointer to nope data for a logical position via the page table.
    pub fn pagedGetNopePtr(self: *const MlaKvCache, page_table: [*]const c_int, logical_pos: usize) [*]f32 {
        const info = self.pagedGetPageInfo(page_table, logical_pos);
        return self.pages.items[info.page_idx].nopePtr(info.token_in_page, self.config.kv_lora_rank);
    }

    /// Get pointer to rope data for a logical position via the page table.
    pub fn pagedGetRopePtr(self: *const MlaKvCache, page_table: [*]const c_int, logical_pos: usize) [*]f32 {
        const info = self.pagedGetPageInfo(page_table, logical_pos);
        return self.pages.items[info.page_idx].ropePtr(info.token_in_page, self.config.rope_size);
    }

    /// Number of physical pages currently allocated.
    pub fn pageCount(self: *const MlaKvCache) usize {
        return self.pages.items.len;
    }

    /// Clear cache (pages are retained for reuse)
    pub fn clear(self: *MlaKvCache) void {
        self.total_tokens = 0;
        for (self.pages.items) |*page| {
            page.tokens_used = 0;
        }
        self.page_table.clearRetainingCapacity();
    }

    // ===================================================================
    // Serialization: save/load (binary file) + dump/fromDump (in-memory
    // snapshot). Mirrors the C++ kvcache_load_dump.cpp API surface
    // (kvcache.h: KVCache::load_kvcache / save_kvcache) for the MLA
    // paged cache. Format is little-endian binary, contiguous bytes —
    // a 32-byte header followed by all page contents in order.
    //
    // File format (matches the standalone kvcache.zig pool format so
    // dumps are interchangeable between the two representations):
    //   header (32 bytes):
    //     magic   u32 = 0x4B56'4341  // "KVCA" little-endian
    //     version u32 = 1
    //     npages  u32
    //     tokens_per_page u32
    //     kv_lora_rank u32
    //     rope_size u32
    //     total_tokens u32
    //     reserved u32 (must be 0)
    //   per page (kv_lora_rank + rope_size) * tokens_per_page * 4 bytes:
    //     nope_data[tokens_per_page][kv_lora_rank] f32
    //     rope_data[tokens_per_page][rope_size]   f32
    //
    // The page_table (logical_pos -> page_id) is NOT serialized: the
    // scheduler reconstructs it per session, matching the C++
    // kvcache_load_dump.cpp semantics. The total_tokens header field
    // is informational (read on load, written on save) so a session
    // can checkpoint/resume the count without re-walking all pages.
    // ===================================================================

    pub const MAGIC: u32 = 0x4B56_4341; // 'K''V''C''A'
    pub const VERSION: u32 = 1;
    pub const HEADER_BYTES: usize = 32;

    /// Size in bytes of the serialized form of this cache.
    pub fn serializedSize(self: *const MlaKvCache) usize {
        const cfg = self.config;
        const per_page = (cfg.kv_lora_rank + cfg.rope_size) * cfg.token_count_in_page * @sizeOf(f32);
        return HEADER_BYTES + per_page * self.pages.items.len;
    }

    /// Write the full cache to `out_buf` (must be >= serializedSize).
    /// Returns the number of bytes written.
    pub fn dump(self: *const MlaKvCache, out_buf: []u8) usize {
        const cfg = self.config;
        std.debug.assert(out_buf.len >= self.serializedSize());
        std.mem.writeInt(u32, out_buf[0..4], MAGIC, .little);
        std.mem.writeInt(u32, out_buf[4..8], VERSION, .little);
        std.mem.writeInt(u32, out_buf[8..12], @intCast(self.pages.items.len), .little);
        std.mem.writeInt(u32, out_buf[12..16], @intCast(cfg.token_count_in_page), .little);
        std.mem.writeInt(u32, out_buf[16..20], @intCast(cfg.kv_lora_rank), .little);
        std.mem.writeInt(u32, out_buf[20..24], @intCast(cfg.rope_size), .little);
        std.mem.writeInt(u32, out_buf[24..28], @intCast(self.total_tokens), .little);
        std.mem.writeInt(u32, out_buf[28..32], 0, .little);

        // MlaCachePage layout: flat nope_data[tokens_per_page][kv_lora_rank]
        // and rope_data[tokens_per_page][rope_size] (see mla_cache.zig:15).
        // Serialize nope first, then rope, per token — matches the
        // in-memory layout so the file is a direct memory dump.
        var off: usize = HEADER_BYTES;
        for (self.pages.items) |*page| {
            const nope_bytes_len = cfg.token_count_in_page * cfg.kv_lora_rank * @sizeOf(f32);
            @memcpy(out_buf[off..][0..nope_bytes_len], @as(*const [8192]u8, @ptrCast(page.nope_data.ptr))[0..nope_bytes_len]);
            off += nope_bytes_len;
            const rope_bytes_len = cfg.token_count_in_page * cfg.rope_size * @sizeOf(f32);
            @memcpy(out_buf[off..][0..rope_bytes_len], @as(*const [8192]u8, @ptrCast(page.rope_data.ptr))[0..rope_bytes_len]);
            off += rope_bytes_len;
        }
        return off;
    }

    /// Reconstruct a fresh MlaKvCache from a serialized buffer.
    pub fn fromDump(allocator: std.mem.Allocator, config: MlaConfig, in_buf: []const u8) !MlaKvCache {
        if (in_buf.len < HEADER_BYTES) return error.BufferTooSmall;
        const magic = std.mem.readInt(u32, in_buf[0..4], .little);
        const version = std.mem.readInt(u32, in_buf[4..8], .little);
        const npages = std.mem.readInt(u32, in_buf[8..12], .little);
        const k_tpp = std.mem.readInt(u32, in_buf[12..16], .little);
        const k_kv_lora_rank = std.mem.readInt(u32, in_buf[16..20], .little);
        const k_rope_size = std.mem.readInt(u32, in_buf[20..24], .little);
        const k_total = std.mem.readInt(u32, in_buf[24..28], .little);
        if (magic != MAGIC) return error.BadMagic;
        if (version != VERSION) return error.BadVersion;
        if (k_tpp != config.token_count_in_page) return error.DimMismatch;
        if (k_kv_lora_rank != config.kv_lora_rank) return error.DimMismatch;
        if (k_rope_size != config.rope_size) return error.DimMismatch;

        var cache = try MlaKvCache.init(allocator, config, npages);
        errdefer cache.deinit();
        for (0..npages) |_| _ = try cache.allocPage();

        var off: usize = HEADER_BYTES;
        for (0..npages) |p| {
            const page = &cache.pages.items[p];
            const nope_bytes_len = config.token_count_in_page * config.kv_lora_rank * @sizeOf(f32);
            @memcpy(@as(*[8192]u8, @ptrCast(page.nope_data.ptr))[0..nope_bytes_len], in_buf[off..][0..nope_bytes_len]);
            off += nope_bytes_len;
            const rope_bytes_len = config.token_count_in_page * config.rope_size * @sizeOf(f32);
            @memcpy(@as(*[8192]u8, @ptrCast(page.rope_data.ptr))[0..rope_bytes_len], in_buf[off..][0..rope_bytes_len]);
            off += rope_bytes_len;
        }
        cache.total_tokens = k_total;
        return cache;
    }

    /// Save to a file (binary, overwrite). Allocates one buffer of
    /// `serializedSize()` for the dump. The file is created if absent
    /// and truncated to zero if present.
    pub fn save(self: *const MlaKvCache, file_path: [*:0]const u8) !void {
        var io = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io.deinit();
        const dir = std.Io.Dir.cwd();
        const path: []const u8 = std.mem.span(file_path);
        const file = try dir.createFile(io.io(), path, .{ .read = false, .truncate = true });
        defer file.close(io.io());
        var buf = try self.allocator.alloc(u8, self.serializedSize());
        defer self.allocator.free(buf);
        const n = self.dump(buf);
        try std.Io.File.writePositionalAll(file, io.io(), buf[0..n], 0);
    }

    /// Load from a file (binary). The cache is constructed with `npages`
    /// physical pages and `total_tokens` set to the header value; the
    /// page_table (logical_pos -> page_id) is left empty — the caller
    /// schedules it (matching the C++ kvcache_load_dump.cpp semantics).
    pub fn load(allocator: std.mem.Allocator, config: MlaConfig, file_path: [*:0]const u8) !MlaKvCache {
        var io = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        defer io.deinit();
        const dir = std.Io.Dir.cwd();
        const path: []const u8 = std.mem.span(file_path);
        const file = try dir.openFile(io.io(), path, .{});
        defer file.close(io.io());
        const stat = try file.stat(io.io());
        const buf = try allocator.alloc(u8, @intCast(stat.size));
        defer allocator.free(buf);
        _ = try std.Io.File.readPositionalAll(file, io.io(), buf, 0);
        return MlaKvCache.fromDump(allocator, config, buf);
    }
};
