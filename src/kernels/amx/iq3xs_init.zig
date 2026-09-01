// IQ3_XXS runtime kmap + kneighbors initialization.
//
// Port of ggml-quants.c:iq3xs_init_impl (lines 3703-3902), grid_size=256
// (the IQ3_XXS case; grid_size=512 is IQ3_S, not ported).
//
// Differences vs the IQ2 init (iq2xs_init.zig):
//   - kmap_size = 4096 (8^4: four 3-bit nibbles packed in a u16 index),
//     not 43692 (8^4 four 2-bit nibbles of a u16).
//   - The grid entries are 4-byte "positions" pos[k] = 2*l + 1 with
//     l = 3-bit nibble, i.e. byte values from {1, 3, ..., 15}.
//   - nwant = 2 neighbors for grid_size=256 (same as IQ2_XXS).
//
// NOTE on grid values: the runtime positions here (odd bytes 1..15) are
// NOT the IQ3XXS_GRID dequant table bytes ({0x04..0x3e} alphabet). The
// quantizer writes qs[.] = grid_index (a 0..255 index into IQ3XXS_GRID);
// both tables enumerate the same 256-entry fingerprint set, so an index
// produced by this kmap is a valid IQ3XXS_GRID index.
// Verification of that correspondence lives in the tests.

const std = @import("std");

/// kgrid_256: the 256 packed 3-bit-per-nibble "fingerprints"
/// (byte-exact from ggml-quants.c:3708-3725). Each u16 packs 4 nibbles
/// of 3 bits (the 4 high bits are always 0).
pub const KGRID_Q3XS_256 = [256]u16{
    0,     2,     4,     9,     11,    15,    16,    18,    25,    34,    59,    61,
    65,    67,    72,    74,    81,    85,    88,    90,    97,    108,   120,   128,
    130,   132,   137,   144,   146,   153,   155,   159,   169,   175,   189,   193,
    199,   200,   202,   213,   248,   267,   287,   292,   303,   315,   317,   321,
    327,   346,   362,   413,   436,   456,   460,   462,   483,   497,   513,   515,
    520,   522,   529,   531,   536,   538,   540,   551,   552,   576,   578,   585,
    592,   594,   641,   643,   648,   650,   657,   664,   698,   704,   706,   720,
    729,   742,   758,   769,   773,   808,   848,   852,   870,   889,   901,   978,
    992,   1024,  1026,  1033,  1035,  1040,  1042,  1046,  1049,  1058,  1089,  1091,
    1093,  1096,  1098,  1105,  1112,  1139,  1143,  1144,  1152,  1154,  1161,  1167,
    1168,  1170,  1183,  1184,  1197,  1217,  1224,  1228,  1272,  1276,  1309,  1323,
    1347,  1367,  1377,  1404,  1473,  1475,  1486,  1509,  1537,  1544,  1546,  1553,
    1555,  1576,  1589,  1594,  1600,  1602,  1616,  1625,  1636,  1638,  1665,  1667,
    1672,  1685,  1706,  1722,  1737,  1755,  1816,  1831,  1850,  1856,  1862,  1874,
    1901,  1932,  1950,  1971,  2011,  2032,  2052,  2063,  2077,  2079,  2091,  2095,
    2172,  2192,  2207,  2208,  2224,  2230,  2247,  2277,  2308,  2345,  2356,  2389,
    2403,  2424,  2501,  2504,  2506,  2520,  2570,  2593,  2616,  2624,  2630,  2646,
    2669,  2700,  2714,  2746,  2754,  2795,  2824,  2835,  2839,  2874,  2882,  2905,
    2984,  3028,  3042,  3092,  3108,  3110,  3124,  3153,  3185,  3215,  3252,  3288,
    3294,  3364,  3397,  3434,  3483,  3523,  3537,  3587,  3589,  3591,  3592,  3610,
    3626,  3670,  3680,  3722,  3749,  3754,  3776,  3789,  3803,  3824,  3857,  3873,
    3904,  3906,  3924,  3992,
};

pub const KMAP_SIZE: usize = 4096;

/// The grid: each entry is 4 positions pos[k] = 2*l + 1 (l = 3-bit nibble).
pub const Iq3GridData = struct {
    /// grid_size entries of 4 bytes each (the "positions", values 1..15 odd).
    grid: [][4]i32,
    /// kmap[4096]: >=0 = exact grid index; <0 = -(offset+1) into kneighbors.
    kmap: []i32,
    /// For each off-grid fingerprint: a u16 count followed by that many
    /// grid indices.
    kneighbors: []u16,
};

/// Build the grid from the packed u16 fingerprints. pos[k] = 2*nibble+1.
fn buildGrid(allocator: std.mem.Allocator, grid_size: usize) [][4]i32 {
    var grid = allocator.alloc([4]i32, grid_size) catch @panic("OOM");
    for (0..grid_size) |k| {
        const packed_val = KGRID_Q3XS_256[k];
        for (0..4) |i| {
            const l: i32 = @intCast((packed_val >> @intCast(3 * i)) & 0x7);
            grid[k][i] = 2 * l + 1;
        }
    }
    return grid;
}

/// Build the exact-match kmap: for each grid entry, compute its
/// fingerprint and store kmap[fingerprint] = grid_index.
fn buildKmapExact(kmap: []i32, grid: []const [4]i32, grid_size: usize) void {
    for (0..KMAP_SIZE) |i| kmap[i] = -1;
    for (0..grid_size) |i| {
        var index: u16 = 0;
        for (0..4) |k| {
            const q: u16 = @intCast(@divTrunc(grid[i][k] - 1, 2));
            index |= (q << @intCast(3 * k));
        }
        kmap[index] = @intCast(i);
    }
}

/// For an off-grid fingerprint index, compute the positions and find its
/// nwant nearest grid entries. dist2 is [2*grid_size] (dist, index) pairs.
fn findNeighbors(
    grid: []const [4]i32,
    grid_size: usize,
    fingerprint: usize,
    dist2: []i32,
    nwant: usize,
) usize {
    var pos: [4]i32 = undefined;
    for (0..4) |k| {
        const l: i32 = @intCast((fingerprint >> @intCast(3 * k)) & 0x7);
        pos[k] = 2 * l + 1;
    }
    for (0..grid_size) |j| {
        var d2: i32 = 0;
        for (0..4) |k| {
            const diff = grid[j][k] - pos[k];
            d2 += diff * diff;
        }
        dist2[2 * j] = d2;
        dist2[2 * j + 1] = @intCast(j);
    }
    // qsort by dist2 then index (iq3_compare_func)
    std.mem.sort([2]i32, @as([][2]i32, @ptrCast(@alignCast(dist2[0 .. 2 * grid_size]))), {}, struct {
        fn lt(_: void, a: [2]i32, b: [2]i32) bool {
            if (a[0] != b[0]) return a[0] < b[0];
            return a[1] < b[1];
        }
    }.lt);
    // Count neighbors at nwant distinct distance levels
    var n: usize = 0;
    var d2 = dist2[0];
    var nhave: usize = 1;
    for (0..grid_size) |j| {
        if (dist2[2 * j] > d2) {
            if (nhave == nwant) break;
            d2 = dist2[2 * j];
            nhave += 1;
        }
        n += 1;
    }
    return n;
}

/// Full init of the IQ3_XXS kmap + kneighbors. Call before quantizeRowIQ3_XXS.
/// This is expensive (~4096 × 256 × 4 = ~4M distance evals) — only for quantize.
pub fn initIq3XsData(allocator: std.mem.Allocator) *Iq3GridData {
    var self = allocator.create(Iq3GridData) catch @panic("OOM");
    const grid_size: usize = 256; // IQ3_XXS
    const nwant: usize = 2;

    self.grid = buildGrid(allocator, grid_size);
    self.kmap = allocator.alloc(i32, KMAP_SIZE) catch @panic("OOM");
    buildKmapExact(self.kmap, self.grid, grid_size);

    // Pass 1: count neighbors for each off-grid fingerprint
    var n_per_i = allocator.alloc(usize, KMAP_SIZE) catch @panic("OOM");
    defer allocator.free(n_per_i);
    var num_neighbors: usize = 0;
    var num_not_in_map: usize = 0;
    var dist2 = allocator.alloc(i32, 2 * grid_size) catch @panic("OOM");
    defer allocator.free(dist2);
    for (0..KMAP_SIZE) |i| {
        if (self.kmap[i] >= 0) {
            n_per_i[i] = 0;
            continue;
        }
        num_not_in_map += 1;
        n_per_i[i] = findNeighbors(self.grid, grid_size, i, dist2, nwant);
        num_neighbors += n_per_i[i];
    }

    // Allocate kneighbors
    self.kneighbors = allocator.alloc(u16, num_neighbors + num_not_in_map) catch @panic("OOM");

    // Build the offsets and fill
    var offsets = allocator.alloc(i32, KMAP_SIZE) catch @panic("OOM");
    defer allocator.free(offsets);
    var counter: i32 = 0;
    for (0..KMAP_SIZE) |i| {
        if (self.kmap[i] >= 0) {
            offsets[i] = -1;
            continue;
        }
        offsets[i] = counter;
        counter += @intCast(1 + n_per_i[i]);
    }

    for (0..KMAP_SIZE) |i| {
        if (self.kmap[i] >= 0) continue;
        var pos: [4]i32 = undefined;
        for (0..4) |k| {
            const l: i32 = @intCast((i >> @intCast(3 * k)) & 0x7);
            pos[k] = 2 * l + 1;
        }
        for (0..grid_size) |j| {
            var d2: i32 = 0;
            for (0..4) |k| {
                const diff = self.grid[j][k] - pos[k];
                d2 += diff * diff;
            }
            dist2[2 * j] = d2;
            dist2[2 * j + 1] = @intCast(j);
        }
        std.mem.sort([2]i32, @as([][2]i32, @ptrCast(@alignCast(dist2[0 .. 2 * grid_size]))), {}, struct {
            fn lt(_: void, a: [2]i32, b: [2]i32) bool {
                if (a[0] != b[0]) return a[0] < b[0];
                return a[1] < b[1];
            }
        }.lt);

        const local_counter: usize = @intCast(offsets[i]);
        self.kmap[i] = -@as(i32, @intCast(local_counter + 1));
        var lc = local_counter;
        self.kneighbors[lc] = 0; // placeholder count — filled below
        lc += 1;
        var d2 = dist2[0];
        var n: u16 = 0;
        var nhave: usize = 1;
        for (0..grid_size) |j| {
            if (dist2[2 * j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2 * j];
                nhave += 1;
            }
            self.kneighbors[lc] = @intCast(dist2[2 * j + 1]);
            lc += 1;
            n += 1;
        }
        self.kneighbors[local_counter] = n;
    }
    return self;
}

pub fn freeIq3XsData(allocator: std.mem.Allocator, data: *Iq3GridData) void {
    allocator.free(data.grid);
    allocator.free(data.kmap);
    allocator.free(data.kneighbors);
    allocator.destroy(data);
}
