// Root module re-exports all submodules for proper Zig module resolution

pub const memory = @import("runtime/memory.zig");
pub const worker_pool = @import("runtime/worker_pool.zig");
pub const task_queue = @import("runtime/task_queue.zig");
pub const cpu_detect = @import("runtime/cpu_detect.zig");
pub const amx = @import("kernels/arch/amx.zig");
pub const buffers = @import("kernels/amx/buffers.zig");
pub const gemm_bf16 = @import("kernels/amx/gemm_224_bf16.zig");
pub const gemm_int8 = @import("kernels/amx/gemm_224_int8.zig");
pub const moe = @import("kernels/moe/moe.zig");