//! Kimi K3 sampler (port): deterministic PCG32 temperature/top-p/top-k.
//!
//! Fase 6c: PCG32 PRNG with temperature scaling and top-p sampling.

const std = @import("std");

pub const Sampler = struct {
    state: u64,
    inc: u64,

    pub fn init(seed: u64) Sampler {
        var s = Sampler{ .state = 0, .inc = (seed << 1) | 1 };
        _ = s.next();
        s.state +%= seed;
        _ = s.next();
        return s;
    }

    pub fn next(self: *Sampler) u32 {
        const oldstate = self.state;
        self.state = oldstate *% 6364136223846793005 +% self.inc;
        const xorshifted: u32 = @truncate(((oldstate >> 18) ^ oldstate) >> 27);
        const rot: u5 = @truncate(oldstate >> 59);
        return (xorshifted >> rot) | (xorshifted << @as(u5, @truncate(32 - @as(u6, rot))));
    }

    pub fn nextF32(self: *Sampler) f32 {
        return @as(f32, @floatFromInt(self.next())) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
    }

    /// Greedy: argmax over logits.
    pub fn greedy(logits: []const f32) u32 {
        var best: u32 = 0;
        var best_val: f32 = logits[0];
        for (logits, 0..) |v, i| {
            if (v > best_val) {
                best_val = v;
                best = @intCast(i);
            }
        }
        return best;
    }

    /// Temperature + top-p sampling. Returns sampled token id.
    pub fn sample(self: *Sampler, logits: []const f32, temperature: f32, top_p: f32, top_k: usize) u32 {
        if (temperature <= 0.0) return greedy(logits);

        const n = logits.len;
        // scale by temperature
        const scaled = std.heap.page_allocator.alloc(f32, n) catch return greedy(logits);
        defer std.heap.page_allocator.free(scaled);
        for (logits, 0..) |v, i| scaled[i] = v / temperature;

        // softmax
        var maxv: f32 = scaled[0];
        for (scaled) |v| if (v > maxv) {
            maxv = v;
        };
        var sum: f64 = 0.0;
        for (scaled) |v| sum += @exp(@as(f64, v - maxv));
        for (scaled, 0..) |v, i| scaled[i] = @floatCast(@exp(@as(f64, v - maxv)) / sum);

        // sort indices by probability descending (simple insertion sort)
        var idx = std.heap.page_allocator.alloc(usize, n) catch return greedy(logits);
        defer std.heap.page_allocator.free(idx);
        for (0..n) |i| idx[i] = i;
        for (1..n) |i| {
            const key = idx[i];
            var j = i;
            while (j > 0 and scaled[idx[j - 1]] < scaled[key]) {
                idx[j] = idx[j - 1];
                j -= 1;
            }
            idx[j] = key;
        }

        // top-p: accumulate until p
        var cum: f32 = 0.0;
        var cutoff: usize = n;
        for (0..n) |i| {
            cum += scaled[idx[i]];
            if (cum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        if (cutoff == 0) cutoff = 1;

        // top-k: further truncate to top_k most probable (if top_k > 0)
        if (top_k > 0 and top_k < cutoff) cutoff = top_k;
        if (cutoff == 0) cutoff = 1;

        // renormalize over cutoff
        var s: f64 = 0.0;
        for (0..cutoff) |i| s += @as(f64, scaled[idx[i]]);
        const inv: f32 = @floatCast(1.0 / s);

        // sample
        const r = self.nextF32();
        var acc: f32 = 0.0;
        for (0..cutoff) |i| {
            acc += scaled[idx[i]] * inv;
            if (r < acc) return @intCast(idx[i]);
        }
        return @intCast(idx[cutoff - 1]);
    }
};
