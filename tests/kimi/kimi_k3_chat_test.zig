const std = @import("std");
const testing = std.testing;
const chat = @import("kt").kimi_k3_chat;

test "kimi_k3_chat: render XTML user segment" {
    const allocator = std.testing.allocator;
    var t = chat.Transcript.init(allocator);
    defer t.deinit();
    try t.add(.user, "hello");
    const seg = try chat.Transcript.renderXtml(allocator, &t.messages.items[0]);
    defer allocator.free(seg);
    try testing.expect(std.mem.indexOf(u8, seg, "<|user|>") != null);
    try testing.expect(std.mem.indexOf(u8, seg, "hello") != null);
}

test "kimi_k3_chat: JSONL roundtrip" {
    const allocator = std.testing.allocator;
    const path = "/tmp/opencode/k3_chat_test.jsonl";
    // clean up
    std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    var t1 = chat.Transcript.init(allocator);
    defer t1.deinit();
    try t1.add(.system, "you are helpful");
    try t1.add(.user, "hi");
    try t1.saveJsonl(path);
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    var t2 = try chat.Transcript.loadJsonl(allocator, path);
    defer t2.deinit();
    try testing.expectEqual(@as(usize, 2), t2.messages.items.len);
    try testing.expectEqual(chat.Role.system, t2.messages.items[0].role);
    try testing.expectEqualStrings("you are helpful", t2.messages.items[0].content);
    try testing.expectEqual(chat.Role.user, t2.messages.items[1].role);
}

test "kimi_k3_chat: renderAll produces XTML" {
    const allocator = std.testing.allocator;
    var t = chat.Transcript.init(allocator);
    defer t.deinit();
    try t.add(.assistant, "answer");
    const all = try t.renderAll(allocator);
    defer allocator.free(all);
    try testing.expect(std.mem.indexOf(u8, all, "<|assistant|>") != null);
}
