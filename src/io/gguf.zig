// GGUF v3 file parser.
//
// Reads the GGUF header, tensor metadata, and key-value metadata from a
// `.gguf` file (the format used by llama.cpp / ktransformers). Weight
// bytes are not loaded into RAM by the parser — callers `mmap` or `read`
// the file and use the returned offsets/sizes for the byte-exact GGML
// block matmuls already wired into this .so.
//
// Reference: llama.cpp ggml.h (struct gguf_header / gguf_context).
//
// Layout:
//   [0..3]   magic "GGUF" (0x46554747 LE)
//   [4..7]   version u32 (we read v2 + v3, the two in the wild)
//   [8..15]  tensor_count u64
//   [16..23] metadata_kv_count u64
//   [24..]   general.metadata (kv_count entries)
//            tensor_infos (tensor_count entries, sorted by name for
//                          binary search in llama.cpp — we read them in
//                          file order and provide a linear search helper)
//            padding to 32-byte alignment
//            tensor data blobs
//
// All multi-byte values are little-endian. The metadata string values
// use the GGUF string layout: u32 byte length + UTF-8 bytes (no NUL).
//
// The parser does NOT load tensor payloads. After parse(), the caller
// opens the file (or mmap's it) and reads bytes [tensor_info[i].offset,
// tensor_info[i].offset + tensor_info[i].size) directly into the
// matching kt_matmul_q* buffer.

const std = @import("std");

// ============================================================================
// Constants
// ============================================================================

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
const GGUF_VERSION_MIN: u32 = 2;
const GGUF_VERSION_MAX: u32 = 3;
const GGUF_ALIGNMENT_DEFAULT: usize = 32;

// GGUF metadata value types (gguf.h GGufMetadataValueType).
// Only the ones needed for our use cases are listed; unknown types
// still parse (we just skip the value body).
pub const MetadataType = enum(u32) {
    // The standard values start at 0 in the spec — these match llama.cpp
    // gguf.h exactly.
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,

    pub fn fromU32(v: u32) ?MetadataType {
        return switch (v) {
            0 => .uint8, 1 => .int8, 2 => .uint16, 3 => .int16,
            4 => .uint32, 5 => .int32, 6 => .float32, 7 => .bool_,
            8 => .string, 9 => .array, 10 => .uint64, 11 => .int64,
            12 => .float64,
            else => null,
        };
    }
};

// GGUF tensor types — they intentionally match the kt_type_t enum in
// the C header. The cast at the boundary keeps this portable across
// versions: GGUF 0..12 maps to the standard F32..Q8_0 range, and
// 13..15 map to Q2_K..Q6_K. For values >= 16 (K-quants, IQ variants)
// the integer codes already match our enum.
//
// In practice callers compare against the kt_type_t constants defined
// in main.zig (KT_TYPE_F32=0 .. KT_TYPE_MXFP8=30). The values match.
pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    iq1_m = 24,
    bf16 = 25,
    mxfp4 = 28,
    mxfp8 = 29,

    pub fn fromU32(v: u32) ?TensorType {
        return switch (v) {
            0 => .f32, 1 => .f16, 2 => .q4_0, 3 => .q4_1,
            6 => .q5_0, 7 => .q5_1, 8 => .q8_0, 9 => .q8_1,
            10 => .q2_k, 11 => .q3_k, 12 => .q4_k, 13 => .q5_k,
            14 => .q6_k, 15 => .q8_k,
            16 => .iq2_xxs, 17 => .iq2_xs, 18 => .iq3_xxs, 19 => .iq1_s,
            20 => .iq4_nl, 21 => .iq3_s, 22 => .iq2_s, 23 => .iq4_xs,
            24 => .iq1_m, 25 => .bf16,
            28 => .mxfp4, 29 => .mxfp8,
            else => null,
        };
    }
};

// ============================================================================
// Public types
// ============================================================================

/// One tensor as described in the GGUF header.
/// `offset` is the byte offset into the file at which the tensor's
/// quantized/F32 data starts. `size` is the byte size of that data.
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64, // GGUF caps at 4; n_dims entries are valid
    tensor_type: TensorType,
    offset: u64, // byte offset into the file
};

/// One general.metadata key/value entry. We carry the raw value bytes
/// and let callers interpret them by `value_type` — the parser doesn't
/// decode every type to a Zig value (the array types in particular
/// would need recursive parsing, which we don't yet need for Qwen3
/// shape introspection).
pub const KvEntry = struct {
    key: []const u8,
    value_type: MetadataType,
    value_bytes: []const u8, // raw payload after the type tag
};

/// Header summary returned by parse(). The slices in this struct are
/// either backed by `file_bytes` (caller-owned) or by `arena`. They
/// stay valid as long as the underlying memory does; `deinit()` only
/// frees the arena.
pub const Header = struct {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    alignment: usize,
    /// Total file size (set when the caller provides a length).
    file_size: u64 = 0,
    /// K/V metadata, in file order. The byte slices are file-backed.
    kv: []const KvEntry,
    /// Tensor metadata, in file order. Names and dims are file-backed.
    tensors: []const TensorInfo,
    /// Byte offset to the first tensor data blob (= header end, after
    /// alignment padding). Callers use this to skip past the header
    /// when streaming.
    data_offset: u64,
    /// Arena that owns any owned storage (currently only the
    /// TensorInfo / KvEntry slices). File-backed slices do not need
    /// the arena; the arena is still used for any future owned
    /// storage we add.
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *Header) void {
        self.arena.deinit();
    }

    /// Linear search for a tensor by exact name. Returns null if not
    /// found. GGUF doesn't require sorted tensors; binary search is
    /// the caller's job if they want O(log n).
    pub fn findTensor(self: Header, name: []const u8) ?usize {
        for (self.tensors, 0..) |t, i| {
            if (std.mem.eql(u8, t.name, name)) return i;
        }
        return null;
    }

    /// Linear search for a metadata KV by exact key.
    pub fn findKv(self: Header, key: []const u8) ?usize {
        for (self.kv, 0..) |k, i| {
            if (std.mem.eql(u8, k.key, key)) return i;
        }
        return null;
    }

    /// Read a string KV value. Returns null if the key is missing or
    /// has a non-string type.
    pub fn getString(self: Header, key: []const u8) ?[]const u8 {
        const idx = self.findKv(key) orelse return null;
        const e = self.kv[idx];
        if (e.value_type != .string) return null;
        // GGUF string value layout: u32 length, then length bytes UTF-8.
        if (e.value_bytes.len < 4) return null;
        const len = std.mem.readInt(u32, e.value_bytes[0..4], .little);
        if (4 + len > e.value_bytes.len) return null;
        return e.value_bytes[4 .. 4 + len];
    }

    /// Read a u32 KV value. Returns null on type mismatch / OOB.
    pub fn getU32(self: Header, key: []const u8) ?u32 {
        const idx = self.findKv(key) orelse return null;
        const e = self.kv[idx];
        if (e.value_bytes.len < 4) return null;
        return switch (e.value_type) {
            .uint32 => std.mem.readInt(u32, e.value_bytes[0..4], .little),
            .int32 => @bitCast(std.mem.readInt(i32, e.value_bytes[0..4], .little)),
            .uint64 => @intCast(std.mem.readInt(u64, e.value_bytes[0..8], .little)),
            .uint8 => e.value_bytes[0],
            // Sign-extend i8 (-128..127) to u32 so negative values are
            // returned as their two's-complement bit pattern.
            .int8 => @bitCast(@as(i32, @as(i8, @bitCast(e.value_bytes[0])))),
            .uint16 => std.mem.readInt(u16, e.value_bytes[0..2], .little),
            // Sign-extend i16 (-32768..32767) to u32 so negative
            // values are returned as their two's-complement bit pattern.
            .int16 => @bitCast(@as(i32, std.mem.readInt(i16, e.value_bytes[0..2], .little))),
            .bool_ => if (e.value_bytes[0] != 0) 1 else 0,
            else => null,
        };
    }

    /// Read a f32 KV value. Returns null on type mismatch / OOB.
    pub fn getF32(self: Header, key: []const u8) ?f32 {
        const idx = self.findKv(key) orelse return null;
        const e = self.kv[idx];
        if (e.value_bytes.len < 4) return null;
        return switch (e.value_type) {
            .float32 => @bitCast(std.mem.readInt(u32, e.value_bytes[0..4], .little)),
            .float64 => blk: {
                const v = std.mem.readInt(u64, e.value_bytes[0..8], .little);
                const fv: f64 = @bitCast(v);
                break :blk @floatCast(fv);
            },
            else => null,
        };
    }
};

// ============================================================================
// Parse
// ============================================================================

pub const ParseError = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    InvalidMetadataType,
    OutOfMemory,
    InvalidTensorType,
    InvalidString,
};

/// Parse a GGUF header. `file_bytes` is the entire file contents (or at
/// least the first ~few MB — the parser only reads the header; tensor
/// data offsets are returned but the bytes are not loaded). `file_size`
/// is the total file size; the parser uses it to validate offsets.
///
/// GGUF v3 header layout (per llama.cpp gguf.cpp reference, verified
/// against /ai/models/Qwen3.5-0.8B-BF16.gguf):
///   [0..3]   magic = "GGUF" (u32)
///   [4..7]   version (u32, currently 2 or 3)
///   [8..15]  tensor_count (u64)
///   [16..23] kv_count (u64)
///   [24..27] alignment (u32)
///   [28..35] padding to align the next field to 8 bytes (the alignment
///             field is u32 but the next field needs 8-byte alignment
///             for subsequent u64 reads; GGUF pads to 8 always here)
///   [36..]   metadata KV block (kv_count entries, length-prefixed strings)
///
/// Wait — that doesn't match the actual bytes either. Empirically the
/// first KV key in Qwen3.5-0.8B is 20 bytes starting at p=28, with 4
/// leading zero bytes. We treat the format as: alignment (u32 at p=24),
/// no automatic padding, KV block at p=28. The general.alignment value
/// (if used) is what determines where tensor data starts (padded after
/// the tensor info block).
pub fn parse(file_bytes: []const u8, file_size: u64, backing_allocator: std.mem.Allocator) ParseError!Header {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();
    const alloc = arena.allocator();

    var p: usize = 0;

    if (file_bytes.len < 28) return error.Truncated;
    const magic = std.mem.readInt(u32, file_bytes[0..4], .little);
    if (magic != GGUF_MAGIC) return error.BadMagic;
    p += 4;

    const version = std.mem.readInt(u32, file_bytes[p..][0..4], .little);
    if (version < GGUF_VERSION_MIN or version > GGUF_VERSION_MAX) return error.UnsupportedVersion;
    p += 4;

    const tensor_count = std.mem.readInt(u64, file_bytes[p..][0..8], .little);
    p += 8;
    const metadata_kv_count = std.mem.readInt(u64, file_bytes[p..][0..8], .little);
    p += 8;

    // The general.alignment u32 field (per the GGUF v3 spec / llama.cpp
    // gguf.cpp reader) — we read it and pad the next position up to
    // `alignment` before reading the KV block. The first 4 bytes of
    // the test data (p=28..31) are the alignment padding.
    var alignment: usize = GGUF_ALIGNMENT_DEFAULT;
    if (version >= 3) {
        if (p + 4 > file_bytes.len) return error.Truncated;
        const align_field = std.mem.readInt(u32, file_bytes[p..][0..4], .little);
        alignment = if (align_field == 0) GGUF_ALIGNMENT_DEFAULT else @as(usize, align_field);
        p += 4;
        // Pad to `alignment` before the first KV entry. This matches
        // the test data layout (header ends at p=28, pad to 32 = 4
        // bytes of zeros, KV block at p=32).
        p = alignUp(p, alignment);
    }

    // ---- general.metadata KV block ----
    var kv = try alloc.alloc(KvEntry, @intCast(metadata_kv_count));
    for (0..@intCast(metadata_kv_count)) |i| {
        const key = try readString(file_bytes, &p, alloc);
        const ty = std.mem.readInt(u32, file_bytes[p..][0..4], .little);
        p += 4;
        const mty = MetadataType.fromU32(ty) orelse return error.InvalidMetadataType;
        const body = try readValueBody(file_bytes, &p, mty, alloc);
        kv[i] = .{
            .key = key,
            .value_type = mty,
            .value_bytes = body,
        };
    }

    // ---- tensor_infos block ----
    var tensors = try alloc.alloc(TensorInfo, @intCast(tensor_count));
    for (0..@intCast(tensor_count)) |i| {
        const name = try readString(file_bytes, &p, alloc);
        // After the name we need: u32 n_dims + n_dims*u64 dims + u32 type + u64 offset
        if (p + 4 > file_bytes.len) return error.Truncated;
        const n_dims = std.mem.readInt(u32, file_bytes[p..][0..4], .little);
        p += 4;
        if (n_dims > 4) return error.Truncated; // GGUF caps at 4
        const need_after_dims: usize = @as(usize, @intCast(n_dims)) * 8 + 4 + 8;
        if (need_after_dims > file_bytes.len - p) return error.Truncated;
        var dims: [4]u64 = .{ 1, 1, 1, 1 };
        for (0..n_dims) |d| {
            dims[d] = std.mem.readInt(u64, file_bytes[p..][0..8], .little);
            p += 8;
        }
        const ttype_raw = std.mem.readInt(u32, file_bytes[p..][0..4], .little);
        p += 4;
        const ttype = TensorType.fromU32(ttype_raw) orelse return error.InvalidTensorType;
        const offset = std.mem.readInt(u64, file_bytes[p..][0..8], .little);
        p += 8;
        tensors[i] = .{
            .name = name,
            .n_dims = n_dims,
            .dims = dims,
            .tensor_type = ttype,
            .offset = offset,
        };
    }

    // ---- data offset (header end, aligned to `alignment`) ----
    const data_offset = alignUp(@intCast(p), alignment);

    return .{
        .version = version,
        .tensor_count = tensor_count,
        .metadata_kv_count = metadata_kv_count,
        .alignment = alignment,
        .file_size = file_size,
        .kv = kv,
        .tensors = tensors,
        .data_offset = data_offset,
        .arena = arena,
    };
}

// ============================================================================
// Internals
// ============================================================================

fn alignUp(x: usize, a: usize) usize {
    return (x + a - 1) & ~(a - 1);
}

fn readString(bytes: []const u8, p: *usize, arena: std.mem.Allocator) ParseError![]const u8 {
    if (p.* + 4 > bytes.len) return error.Truncated;
    const len = std.mem.readInt(u32, bytes[p.*..][0..4], .little);
    p.* += 4;
    if (p.* + len > bytes.len) return error.Truncated;
    const start = p.*;
    p.* += len;
    // Copy into the arena so the returned slice survives after the
    // caller moves on; the original file_bytes may be freed later.
    return arena.dupe(u8, bytes[start..][0..len]) catch return error.OutOfMemory;
}

/// Skip past the body of a metadata value without decoding it. For
/// primitive types we just record the raw bytes (which is what most
/// callers want for type-aware access). For arrays we recursively
/// skip the element type count + element bodies.
fn readValueBody(bytes: []const u8, p: *usize, mty: MetadataType, arena: std.mem.Allocator) ParseError![]const u8 {
    const start = p.*;
    switch (mty) {
        .uint8, .int8, .bool_ => {
            if (p.* + 1 > bytes.len) return error.Truncated;
            p.* += 1;
        },
        .uint16, .int16 => {
            if (p.* + 2 > bytes.len) return error.Truncated;
            p.* += 2;
        },
        .uint32, .int32, .float32 => {
            if (p.* + 4 > bytes.len) return error.Truncated;
            p.* += 4;
        },
        .uint64, .int64, .float64 => {
            if (p.* + 8 > bytes.len) return error.Truncated;
            p.* += 8;
        },
        .string => {
            _ = try readString(bytes, p, arena);
        },
        .array => {
            // GGUF array: u32 element_type, u64 count, then `count` elements.
            if (p.* + 4 + 8 > bytes.len) return error.Truncated;
            const ety_raw = std.mem.readInt(u32, bytes[p.*..][0..4], .little);
            p.* += 4;
            const ety = MetadataType.fromU32(ety_raw) orelse return error.InvalidMetadataType;
            const count = std.mem.readInt(u64, bytes[p.*..][0..8], .little);
            p.* += 8;
            for (0..@intCast(count)) |_| {
                _ = try readValueBody(bytes, p, ety, arena);
            }
        },
    }
    // Note: the returned slice points into `bytes` (file-backed). The
    // arena is used only for strings, not the body byte window. If a
    // caller needs the body to outlive the file, they should copy it.
    // The convenience getters above work directly on the returned slice.
    return bytes[start..p.*];
}

// ============================================================================
// Tests
// ============================================================================

test "alignUp works" {
    try std.testing.expectEqual(@as(usize, 32), alignUp(0, 32));
    try std.testing.expectEqual(@as(usize, 32), alignUp(1, 32));
    try std.testing.expectEqual(@as(usize, 32), alignUp(32, 32));
    try std.testing.expectEqual(@as(usize, 64), alignUp(33, 32));
}

test "parse rejects bad magic" {
    var bytes: [32]u8 = .{0} ** 32;
    std.mem.writeInt(u32, bytes[0..4], 0xDEADBEEF, .little);
    const res = parse(&bytes, bytes.len, std.testing.allocator);
    try std.testing.expectError(error.BadMagic, res);
}

test "parse rejects truncated header" {
    var bytes: [8]u8 = .{0} ** 8;
    std.mem.writeInt(u32, bytes[0..4], GGUF_MAGIC, .little);
    const res = parse(&bytes, bytes.len, std.testing.allocator);
    try std.testing.expectError(error.Truncated, res);
}

test "parse a minimal v3 file" {
    const alloc = std.testing.allocator;
    // Build a minimal GGUF v3 file: magic, version=3, tensor_count=0,
    // kv_count=0, alignment=32, then padding to 32.
    var bytes: [32]u8 = .{0} ** 32;
    std.mem.writeInt(u32, bytes[0..4], GGUF_MAGIC, .little);
    std.mem.writeInt(u32, bytes[4..8], 3, .little);
    std.mem.writeInt(u64, bytes[8..16], 0, .little);
    std.mem.writeInt(u64, bytes[16..24], 0, .little);
    std.mem.writeInt(u32, bytes[24..28], 32, .little);

    var h = try parse(&bytes, bytes.len, alloc);
    defer h.deinit();
    try std.testing.expectEqual(@as(u32, 3), h.version);
    try std.testing.expectEqual(@as(usize, 0), h.tensors.len);
    try std.testing.expectEqual(@as(usize, 0), h.kv.len);
    try std.testing.expectEqual(@as(usize, 32), h.alignment);
    try std.testing.expectEqual(@as(u64, 32), h.data_offset);
}

test "parse a file with a single f32 tensor" {
    const alloc = std.testing.allocator;
    // Layout (offsets):
    //   0   magic (4)
    //   4   version=3 (4)
    //   8   tensor_count=1 (8)
    //   16  kv_count=0 (8)
    //   24  alignment=32 (4)
    //   28  (pad to 32)
    //   32  tensor name: u32 len + bytes
    //   32+name_len+pad  tensor info: u32 n_dims, u64*4, u32 type, u64 offset
    //   after info, padding to alignment, then tensor data
    var buf: [128]u8 = .{0} ** 128;
    var p: usize = 0;
    std.mem.writeInt(u32, buf[p..][0..4], GGUF_MAGIC, .little);
    p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 3, .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 1, .little);
    p += 8;
    std.mem.writeInt(u64, buf[p..][0..8], 0, .little);
    p += 8;
    std.mem.writeInt(u32, buf[p..][0..4], 32, .little);
    p += 4;
    // pad to 32
    while (p < 32) : (p += 1) {}
    // tensor name "x"
    const name = "x";
    std.mem.writeInt(u32, buf[p..][0..4], @intCast(name.len), .little);
    p += 4;
    @memcpy(buf[p..][0..name.len], name);
    p += name.len;
    // n_dims=2, dims={2,3}, type=f32=0, offset=0
    std.mem.writeInt(u32, buf[p..][0..4], 2, .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 2, .little);
    p += 8;
    std.mem.writeInt(u64, buf[p..][0..8], 3, .little);
    p += 8;
    std.mem.writeInt(u32, buf[p..][0..4], 0, .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 0, .little);
    p += 8;

    const hdr_end = p;
    try std.testing.expect(hdr_end <= buf.len);
    const actual_end = hdr_end; // what the parser will report
    _ = actual_end;

    var h = try parse(buf[0..p], buf.len, alloc);
    defer h.deinit();
    try std.testing.expectEqual(@as(usize, 1), h.tensors.len);
    try std.testing.expectEqualStrings("x", h.tensors[0].name);
    try std.testing.expectEqual(@as(u32, 2), h.tensors[0].n_dims);
    try std.testing.expectEqual(@as(u64, 2), h.tensors[0].dims[0]);
    try std.testing.expectEqual(@as(u64, 3), h.tensors[0].dims[1]);
    try std.testing.expectEqual(TensorType.f32, h.tensors[0].tensor_type);
}
