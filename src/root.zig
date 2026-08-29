// Root module re-exports all submodules for proper Zig module resolution
// Also exports C API functions from main.zig so they are linked into the library

pub const memory = @import("runtime/memory.zig");
pub const worker_pool = @import("runtime/worker_pool.zig");
pub const task_queue = @import("runtime/task_queue.zig");
pub const cpu_detect = @import("runtime/cpu_detect.zig");
pub const numa = @import("numa/numa.zig");
pub const amx = @import("kernels/arch/amx.zig");
pub const buffers = @import("kernels/amx/buffers.zig");
pub const gemm_bf16 = @import("kernels/amx/gemm_224_bf16.zig");
pub const gemm_int4 = @import("kernels/amx/gemm_224_int4.zig");
pub const gemm_fp8 = @import("kernels/amx/gemm_224_fp8.zig");
pub const gemm_int8 = @import("kernels/amx/gemm_224_int8.zig");
pub const moe = @import("kernels/moe/moe.zig");

// Force main.zig (C API: export fn kt_*) to be semantically analyzed and
// exported into the shared library. A plain `const _ = @import(...)` is
// lazily stripped by the compiler and produces zero exported symbols;
// wrapping in comptime forces the analysis. Verified experimentally.
comptime {
    _ = @import("main.zig");
}
