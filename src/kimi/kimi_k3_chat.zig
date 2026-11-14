//! Kimi K3 chat (port of k3_chat.c): XTML segment rendering + JSONL transcripts.
//!
//! Fase 6b: renders official XTML segments for system/user/assistant/thinking,
//! and saves/loads strict JSONL transcripts.

const std = @import("std");

pub const Role = enum { system, user, assistant, thinking };

pub const Message = struct {
    role: Role,
    content: []const u8,
};

pub const Transcript = struct {
    allocator: std.mem.Allocator,
    messages: std.ArrayList(Message),

    pub fn init(allocator: std.mem.Allocator) Transcript {
        return .{ .allocator = allocator, .messages = .empty };
    }

    pub fn deinit(self: *Transcript) void {
        for (self.messages.items) |m| {
            self.allocator.free(m.content);
        }
        self.messages.deinit(self.allocator);
    }

    pub fn add(self: *Transcript, role: Role, content: []const u8) !void {
        const copy = try self.allocator.dupe(u8, content);
        try self.messages.append(self.allocator, .{ .role = role, .content = copy });
    }

    /// Render XTML segment for a message.
    pub fn renderXtml(allocator: std.mem.Allocator, msg: *const Message) ![]u8 {
        return switch (msg.role) {
            .system => std.fmt.allocPrint(allocator, "<|system|>\n{s}\n", .{msg.content}),
            .user => std.fmt.allocPrint(allocator, "<|user|>\n{s}\n", .{msg.content}),
            .assistant => std.fmt.allocPrint(allocator, "<|assistant|>\n{s}\n", .{msg.content}),
            .thinking => std.fmt.allocPrint(allocator, "<think>\n{s}\n</think>\n", .{msg.content}),
        };
    }

    /// Render full transcript as XTML.
    pub fn renderAll(self: *Transcript, allocator: std.mem.Allocator) ![]u8 {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(allocator);
        for (self.messages.items) |*m| {
            const seg = try renderXtml(allocator, m);
            defer allocator.free(seg);
            try buf.appendSlice(allocator, seg);
        }
        return try buf.toOwnedSlice(allocator);
    }

    /// Save transcript as JSONL (one JSON object per line).
    pub fn saveJsonl(self: *Transcript, path: []const u8) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        for (self.messages.items) |m| {
            const role_str = switch (m.role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
                .thinking => "thinking",
            };
            const line = try std.fmt.allocPrint(self.allocator, "{{\"role\":\"{s}\",\"content\":\"{s}\"}}\n", .{ role_str, m.content });
            defer self.allocator.free(line);
            try file.writeStreamingAll(io, line);
        }
    }

    /// Load transcript from JSONL.
    pub fn loadJsonl(allocator: std.mem.Allocator, path: []const u8) !Transcript {
        const io = std.Io.Threaded.global_single_threaded.io();
        const data = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 20));
        defer allocator.free(data);

        var t = Transcript.init(allocator);
        var lines = std.mem.splitScalar(u8, data, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch continue;
            defer parsed.deinit();
            const obj = switch (parsed.value) {
                .object => |o| o,
                else => continue,
            };
            const role_v = obj.get("role") orelse continue;
            const content_v = obj.get("content") orelse continue;
            if (role_v != .string or content_v != .string) continue;
            const role: Role = if (std.mem.eql(u8, role_v.string, "system")) .system else if (std.mem.eql(u8, role_v.string, "user")) .user else if (std.mem.eql(u8, role_v.string, "assistant")) .assistant else .thinking;
            try t.add(role, content_v.string);
        }
        return t;
    }
};
