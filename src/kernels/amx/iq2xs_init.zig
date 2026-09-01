// IQ2_XXS / IQ3_XXS runtime kmap + kneighbors initialization.
//
// Port of ggml-quants.c:iq2xs_init_impl (lines 2853-3255).
//
// The quantizer needs two runtime-built tables:
//   - kmap[43692]: maps every 8-nibble "fingerprint" (packed 2-bit x 8
//     = a u16 index) to either its exact grid index, or -(offset+1) when
//     not on-grid, pointing into the kneighbors table for refinement.
//   - kneighbors[]: for each off-grid fingerprint, nwant (=2) nearest
//     grid entries (Hamming-distance via the mapped 3-bit byte values).
//
// The fingerprints come from the kgrid_2bit_256 table (256 entries of
// packed 2-bit values). The "positions" are pos[k] = 2*l + 1 where
// l = nibble k, giving byte values from the alphabet {1, 3, 5, 7} —
// the same alphabet as the IQ2XXS_GRID / IQ3XXS_GRID magnitude bytes.
//
// This is shared between IQ2_XXS, IQ2_XS, IQ2_S and IQ1 formats.
// We only port the IQ2_XXS case (grid_size=256) for now.

const std = @import("std");

/// kgrid_2bit_256: the 256 packed 2-bit "fingerprints" (byte-exact from
/// ggml-quants.c:2856). Each u16 packs 8 nibbles of 2 bits.
pub const KGRID_2BIT_256 = [256]u16{
    0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
    100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
    1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
    1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
    2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
    4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
    5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
    8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
};

pub const KMAP_SIZE: usize = 43692;

/// The grid as u64: each entry has pos[k] = 2*l + 1 packed as 8 bytes
/// (same layout as IQ2XXS_GRID entries — the magnitude-byte alphabet).
pub const Iq2GridData = struct {
    /// grid_size entries of 8 bytes each (the "positions").
    grid: [][8]i32,
    /// kmap[43692]: >=0 = exact grid index; <0 = -(offset+1) into kneighbors.
    kmap: []i32,
    /// For each off-grid fingerprint: a u16 count followed by that many
    /// grid indices.
    kneighbors: []u16,
};

fn iq2CompareFunc(a: [*]const i32, b: [*]const i32) c_int {
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    if (a[1] < b[1]) return -1;
    if (a[1] > b[1]) return 1;
    return 0;
}

/// Build the grid from the packed u16 fingerprints. pos[k] = 2*nibble+1.
fn buildGrid(allocator: std.mem.Allocator, grid_size: usize) [][8]i32 {
    var grid = allocator.alloc([8]i32, grid_size) catch @panic("OOM");
    for (0..grid_size) |k| {
        const packed_val = KGRID_2BIT_256[k];
        for (0..8) |i| {
            const l: i32 = @intCast((packed_val >> @intCast(2 * i)) & 0x3);
            grid[k][i] = 2 * l + 1;
        }
    }
    return grid;
}

/// Build the exact-match kmap: for each grid entry, compute its
/// fingerprint and store kmap[fingerprint] = grid_index.
fn buildKmapExact(kmap: []i32, grid: []const [8]i32, grid_size: usize) void {
    for (0..KMAP_SIZE) |i| kmap[i] = -1;
    for (0..grid_size) |i| {
        var index: u16 = 0;
        for (0..8) |k| {
            const q: u16 = @intCast(@divTrunc(grid[i][k] - 1, 2));
            index |= (q << @intCast(2 * k));
        }
        kmap[index] = @intCast(i);
    }
}

/// For an off-grid fingerprint index, compute the Hamming-positions
/// and find its nwant nearest grid entries. dist2 is [2*grid_size].
fn findNeighbors(
    grid: []const [8]i32,
    grid_size: usize,
    fingerprint: usize,
    dist2: []i32,
    nwant: usize,
) usize {
    var pos: [8]i32 = undefined;
    for (0..8) |k| {
        const l: i32 = @intCast((fingerprint >> @intCast(2 * k)) & 0x3);
        pos[k] = 2 * l + 1;
    }
    for (0..grid_size) |j| {
        var d2: i32 = 0;
        for (0..8) |k| {
            const diff = grid[j][k] - pos[k];
            d2 += diff * diff;
        }
        dist2[2 * j] = d2;
        dist2[2 * j + 1] = @intCast(j);
    }
    // qsort by dist2 then index
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

/// Full lazy-init of the IQ2_XXS kmap + kneighbors. Call before quantizeRowIQ2_XXS.
/// This is expensive (~43692 × 256 × 8 = ~89M distance evals) — only for quantize.
pub fn initIq2XsData(allocator: std.mem.Allocator) *Iq2GridData {
    var self = allocator.create(Iq2GridData) catch @panic("OOM");
    const grid_size: usize = 256; // IQ2_XXS
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
        var pos: [8]i32 = undefined;
        for (0..8) |k| {
            const l: i32 = @intCast((i >> @intCast(2 * k)) & 0x3);
            pos[k] = 2 * l + 1;
        }
        for (0..grid_size) |j| {
            var d2: i32 = 0;
            for (0..8) |k| {
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

pub fn freeIq2XsData(allocator: std.mem.Allocator, data: *Iq2GridData) void {
    allocator.free(data.grid);
    allocator.free(data.kmap);
    allocator.free(data.kneighbors);
    allocator.destroy(data);
}
