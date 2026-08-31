// Standalone test for kt_set_default_allocator (B1: injectable allocator).
//
// Exercises the C-ABI allocator vtable through src/main.zig exports:
//   kt_set_default_allocator(tracking) -> kt_gate_new -> kt_gate_free
//   -> kt_set_default_allocator(null)
//
// What this verifies:
//   - The adapter bridges the C vtable (userdata + alloc/free/resize
//     fn pointers) to std.mem.Allocator faithfully
//   - Every context allocation the C API performs during a Gate
//     lifecycle goes through the injected allocator
//   - The free path uses the CAPTURED allocator: even after we reset
//     the default back to page_allocator (simulating a caller swap
//     between new and free), kt_gate_free still frees through the
//     injected one — the bookkeeping balances to exactly zero
//   - The no-allocator-installed path (default page_allocator) is
//     unaffected: a lifecycle with the vtable installed-then-reset
//     before construction never touches the tracker

const std = @import("std");
const testing = std.testing;
const kt = @import("kt"); // module rooted at src/main.zig (wired by build.zig)

/// Tracking allocator state: counts live bytes and live allocations.
const Tracker = struct {
    live_bytes: std.atomic.Value(usize),
    live_allocs: std.atomic.Value(usize),
    total_allocs: std.atomic.Value(usize),
    total_frees: std.atomic.Value(usize),

    fn init() Tracker {
        return .{
            .live_bytes = std.atomic.Value(usize).init(0),
            .live_allocs = std.atomic.Value(usize).init(0),
            .total_allocs = std.atomic.Value(usize).init(0),
            .total_frees = std.atomic.Value(usize).init(0),
        };
    }
};

// The C vtable callbacks receive the Tracker via userdata. We back
// every allocation with page_allocator and store a 64-byte header
// (raw_len + requested size) ahead of the returned pointer so free()
// can rebuild the original slice and balance the byte counter.
fn trkAlloc(userdata: ?*anyopaque, size: usize, alignment: usize) callconv(.c) ?[*]u8 {
    _ = alignment; // payload is 64-byte aligned (see hdr below)
    const trk: *Tracker = @ptrCast(@alignCast(userdata orelse return null));
    const hdr: usize = 64;
    const raw = std.heap.page_allocator.alignedAlloc(u8, .@"64", size + hdr) catch return null;
    const raw_bytes: *[hdr]u8 = @ptrCast(raw.ptr);
    std.mem.writeInt(usize, raw_bytes[0..8], raw.len, .little);
    std.mem.writeInt(usize, raw_bytes[8..16], size, .little);
    const payload = raw.ptr + hdr;

    _ = trk.live_bytes.fetchAdd(size, .monotonic);
    _ = trk.live_allocs.fetchAdd(1, .monotonic);
    _ = trk.total_allocs.fetchAdd(1, .monotonic);
    return payload;
}

fn trkFree(userdata: ?*anyopaque, ptr: [*]u8, size: usize, alignment: usize) callconv(.c) void {
    _ = size;
    _ = alignment;
    const trk: *Tracker = @ptrCast(@alignCast(userdata orelse return));
    const hdr: usize = 64;
    const raw_start = ptr - hdr;
    const raw_bytes: *[hdr]u8 = @ptrCast(raw_start);
    const raw_len = std.mem.readInt(usize, raw_bytes[0..8], .little);
    const req_size = std.mem.readInt(usize, raw_bytes[8..16], .little);
    const raw: []u8 = raw_start[0..raw_len];
    std.heap.page_allocator.free(raw);

    _ = trk.live_bytes.fetchSub(req_size, .monotonic);
    _ = trk.live_allocs.fetchSub(1, .monotonic);
    _ = trk.total_frees.fetchAdd(1, .monotonic);
}

fn trkResize(userdata: ?*anyopaque, ptr: [*]u8, old_size: usize, new_size: usize, alignment: usize) callconv(.c) c_int {
    _ = userdata;
    _ = ptr;
    _ = old_size;
    _ = new_size;
    _ = alignment;
    return -1; // unsupported: force the alloc+copy+free fallback
}

var g_tracker: Tracker = undefined;

/// Shared helper: build a minimal legacy-shape Gate config over the
/// given weights (kt.kt_gate_config_t is the real main.zig type).
fn makeGateConfig(
    gate_w: []u16,
    hidden: usize,
    expert_num: usize,
) kt.kt_gate_config_t {
    const KT_TYPE_BF16: c_int = 2;
    const KT_TYPE_NONE: c_int = 0;
    return .{
        .hidden_size = hidden,
        .num_experts_per_tok = 1,
        .n_routed_experts = expert_num,
        .n_group = 1,
        .topk_group = 1,
        .norm_topk_prob = 0,
        .routed_scaling_factor = 1.0,
        .scoring_func = @ptrCast(@constCast("sigmoid")),
        .topk_method = @ptrCast(@constCast("naive")),
        .layer_idx = 0,
        // The gate's routing path passes pool=null internally to
        // routeExperts (kt_gate_forward doesn't forward the C pool to
        // the sequential router), so any non-null dummy satisfies the
        // non-optional field type here.
        .pool = @ptrFromInt(@alignOf(usize)),
        .weight = @ptrCast(gate_w.ptr),
        .weight_type = @enumFromInt(KT_TYPE_BF16),
        // Non-optional *anyopaque in the header type; the Zig side
        // treats bias as "absent" when the dtype field is not
        // KT_TYPE_F32, so a dummy non-null pointer + KT_TYPE_NONE is
        // the legacy-config idiom.
        .e_score_correction_bias = @ptrFromInt(@alignOf(usize)),
        .e_score_correction_bias_type = @enumFromInt(KT_TYPE_NONE),
        .max_seqlen = 1,
    };
}

test "kt_set_default_allocator: full Gate lifecycle through injected allocator" {
    g_tracker = Tracker.init();
    const vtable = kt.kt_allocator_vtable_t{
        .userdata = &g_tracker,
        .alloc = trkAlloc,
        .free = trkFree,
        .resize = trkResize,
    };

    // Install BEFORE any kt_*_new (documented contract).
    kt.kt_set_default_allocator(&vtable);
    // Restore the default at every exit path so later suites are
    // unaffected even if an assertion fails mid-test.
    defer kt.kt_set_default_allocator(null);

    const allocator = testing.allocator;

    // Gate weight: [8 experts, 4 hidden] BF16 ones.
    const gate_w = try allocator.alloc(u16, 8 * 4);
    defer allocator.free(gate_w);
    for (gate_w) |*v| v.* = 0x3F80; // BF16 1.0

    const gate = kt.kt_gate_new(makeGateConfig(gate_w, 4, 8));
    try testing.expect(@intFromPtr(gate) != 0);

    // The context allocation must have gone through the tracker.
    try testing.expect(g_tracker.total_allocs.load(.monotonic) >= 1);
    try testing.expect(g_tracker.live_allocs.load(.monotonic) >= 1);

    // Swap the default back to page_allocator BEFORE freeing — the
    // context captured the injected allocator at construction, so the
    // free must still go through the tracker (the B1 invariant).
    kt.kt_set_default_allocator(null);
    kt.kt_gate_free(gate);

    // Everything allocated through the tracker must be freed through it.
    try testing.expectEqual(@as(usize, 0), g_tracker.live_allocs.load(.monotonic));
    try testing.expectEqual(@as(usize, 0), g_tracker.live_bytes.load(.monotonic));
    try testing.expectEqual(g_tracker.total_allocs.load(.monotonic), g_tracker.total_frees.load(.monotonic));

    std.debug.print("  tracker: {d} allocs / {d} frees balanced, 0 live bytes\n", .{
        g_tracker.total_allocs.load(.monotonic),
        g_tracker.total_frees.load(.monotonic),
    });
}

test "kt_set_default_allocator(null) restores the built-in default" {
    // After a null reset, a Gate lifecycle must run WITHOUT touching
    // any previously installed tracker (fresh zeroed tracker here
    // must stay untouched).
    g_tracker = Tracker.init();
    const vtable = kt.kt_allocator_vtable_t{
        .userdata = &g_tracker,
        .alloc = trkAlloc,
        .free = trkFree,
        .resize = trkResize,
    };
    kt.kt_set_default_allocator(&vtable);
    kt.kt_set_default_allocator(null); // immediately reset

    const allocator = testing.allocator;
    const gate_w = try allocator.alloc(u16, 4 * 4);
    defer allocator.free(gate_w);
    for (gate_w) |*v| v.* = 0x3F80;

    const gate = kt.kt_gate_new(makeGateConfig(gate_w, 4, 4));
    kt.kt_gate_free(gate);

    // The reset happened before kt_gate_new, so the tracker must be
    // COMPLETELY untouched.
    try testing.expectEqual(@as(usize, 0), g_tracker.total_allocs.load(.monotonic));
    try testing.expectEqual(@as(usize, 0), g_tracker.total_frees.load(.monotonic));
    try testing.expectEqual(@as(usize, 0), g_tracker.live_bytes.load(.monotonic));
}
