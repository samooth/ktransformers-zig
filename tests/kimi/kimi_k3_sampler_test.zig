const std = @import("std");
const testing = std.testing;
const sampler = @import("kt").kimi_k3_sampler;

test "kimi_k3_sampler: greedy returns argmax" {
    const logits = [_]f32{ 0.1, 0.9, 0.3 };
    try testing.expectEqual(@as(u32, 1), sampler.Sampler.greedy(&logits));
}

test "kimi_k3_sampler: deterministic with same seed" {
    var s1 = sampler.Sampler.init(42);
    var s2 = sampler.Sampler.init(42);
    try testing.expectEqual(s1.next(), s2.next());
    try testing.expectEqual(s1.next(), s2.next());
}

test "kimi_k3_sampler: sample returns valid index" {
    const allocator = std.testing.allocator;
    var s = sampler.Sampler.init(1);
    const logits = try allocator.alloc(f32, 5);
    defer allocator.free(logits);
    for (logits) |*v| v.* = 1.0;
    const tok = s.sample(logits, 1.0, 0.9, 0);
    try testing.expect(tok < 5);
}

test "kimi_k3_sampler: temperature 0 is greedy" {
    var s = sampler.Sampler.init(1);
    const logits = [_]f32{ 0.1, 0.9, 0.3 };
    const tok = s.sample(&logits, 0.0, 1.0, 0);
    try testing.expectEqual(@as(u32, 1), tok);
}
