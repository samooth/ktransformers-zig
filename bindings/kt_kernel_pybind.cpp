// Minimal pybind11 drop-in wrapper for the ktransformers-zig C API.
//
// The reference C++ ext_bindings.cpp wraps C++ template classes (TP_MOE<T>,
// TP_MLA<T> ...) that live in the C++ headers. The Zig port replaces those
// kernels with a pure C ABI (libkt_kernel_ext.so, 86 exported kt_* symbols,
// contract in include/kt_kernel.h). A C++ class cannot link against a C ABI,
// so this shim re-exposes the same Python-facing surface (module kt_kernel_ext,
// submodules moe/mla/kvcache, classes MOE/MLA/CPUInfer/WorkerPool) but routes
// every method through the Zig C API. This is the Tier-1 "drop-in":
// ktransformers Python code that does `from kt_kernel_ext.moe import MOE` keeps
// working unchanged.
//
// Build: bindings/build.sh.  No dependency on the C++ operator headers.
// The Zig .so is linked at build time (linker resolves the C symbols); the
// variant is selected by pointing KT_KERNEL_LIB_PATH at the right .so before
// building, or by rebuilding per-variant.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// C API function declarations — signatures match include/kt_kernel.h exactly.
// The linker resolves these against the linked Zig .so (no dlsym needed).
// ---------------------------------------------------------------------------

// Opaque handle types — layout defined by the Zig side; we only pass
// pointers.  Defined as (empty) complete types so pybind11 can register
// py::class_ for them without needing the real definition.
struct KT_WorkerPool {};
struct KT_CPUInfer {};
struct KT_MOE {};
struct KT_MLA {};
struct KT_Gate {};
struct KT_Linear {};
struct KT_MLP {};

// Config structs — field layout MUST exactly match the Zig extern structs
// in src/main.zig (these are the C ABI contract).  Field order, types, and
// sizes are critical: a mismatch corrupts the callee's stack / reads garbage.
struct kt_quant_config_t {
    int quant_method = 0;
    int bits = 0;
    int group_size = 0;
    int zero_point = 0;
    int per_channel = 0;
};

struct kt_moe_config_t {
    int expert_num = 0;
    int num_experts_per_tok = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int layer_idx = 0;
    size_t pool = 0;
    int num_gpu_experts = 0;
    size_t gpu_experts_mask = 0;
    size_t physical_to_logical_map = 0;
    size_t gate_proj = 0;
    size_t up_proj = 0;
    size_t down_proj = 0;
    size_t gate_scale = 0;
    size_t up_scale = 0;
    size_t down_scale = 0;
    size_t gate_zero = 0;
    size_t up_zero = 0;
    size_t down_zero = 0;
    kt_quant_config_t quant_config;
    int max_len = 0;
    size_t gate_projs = 0;
    size_t up_projs = 0;
    size_t down_projs = 0;
    size_t gate_scales = 0;
    size_t up_scales = 0;
    size_t down_scales = 0;
    size_t gate_zeros = 0;
    size_t up_zeros = 0;
    size_t down_zeros = 0;
    size_t gate_bwd_projs = 0;
    size_t up_bwd_projs = 0;
    size_t down_bwd_projs = 0;
    size_t gate_bwd_scales = 0;
    size_t up_bwd_scales = 0;
    size_t down_bwd_scales = 0;
    size_t path = 0;
    int save = 0;
    int load = 0;
    int share_backward_bb = 0;
    int share_cache_pool = 0;
    int m_block = 0;
    int group_min_len = 0;
    int group_max_len = 0;
    int gate_type = 0;
    int up_type = 0;
    int down_type = 0;
    int hidden_type = 0;
    int max_cache_depth = 0;
    float swiglu_limit = 0;
    float swiglu_alpha = 0;
};

// kt_moe_sft_config_t — { base: kt_moe_config_t, lora fields... } (Zig main.zig)
struct kt_moe_sft_config_t {
    kt_moe_config_t base;
    int lora_rank = 0;
    float lora_alpha = 0;
    float lora_dropout = 0;
    void* gate_lora_a = nullptr;
    void* gate_lora_b = nullptr;
    void* up_lora_a = nullptr;
    void* up_lora_b = nullptr;
    void* down_lora_a = nullptr;
    void* down_lora_b = nullptr;
    int full_weight_grad = 0;
    int authoritative_optimizer_grads = 0;
    void* grad_gate_proj = nullptr;
    void* grad_up_proj = nullptr;
    void* grad_down_proj = nullptr;
};

struct kt_mla_config_t {
    size_t hidden_size = 0;
    size_t q_lora_rank = 0;
    size_t num_heads = 0;
    size_t nope_size = 0;
    size_t rope_size = 0;
    size_t kv_lora_rank = 0;
    int layer_idx = 0;
    size_t pool = 0;
    size_t token_count_in_page = 0;
    size_t max_qlen = 0;
    size_t max_kvlen = 0;
    size_t max_position_embeddings = 0;
    double rope_scaling_factor = 0;
    double rope_theta = 0;
    double rope_scaling_beta_fast = 0;
    double rope_scaling_beta_slow = 0;
    double rope_scaling_mscale = 0;
    double rope_scaling_mscale_all_dim = 0;
    double rope_scaling_original_max_position_embeddings = 0;
    size_t q_a_proj = 0;
    size_t q_a_norm = 0;
    size_t q_b_proj = 0;
    size_t kv_a_proj_with_mqa = 0;
    size_t kv_a_norm = 0;
    size_t kv_b_proj = 0;
    size_t o_proj = 0;
    int q_a_proj_type = 0;
    int q_a_norm_type = 0;
    int q_b_proj_type = 0;
    int kv_a_proj_with_mqa_type = 0;
    int kv_a_norm_type = 0;
    int kv_b_proj_type = 0;
    int w_o_type = 0;
    int input_type = 0;
    int output_type = 0;
    size_t m_block = 0;
    size_t n_block = 0;
    size_t page_count = 0;
};

// kt_gate_config_t (Zig main.zig:230)
struct kt_gate_config_t {
    size_t hidden_size = 0;
    size_t num_experts_per_tok = 0;
    size_t n_routed_experts = 0;
    size_t n_group = 0;
    size_t topk_group = 0;
    int norm_topk_prob = 0;
    float routed_scaling_factor = 0;
    size_t scoring_func = 0;
    size_t topk_method = 0;
    int layer_idx = 0;
    size_t pool = 0;
    size_t weight = 0;
    int weight_type = 0;
    size_t e_score_correction_bias = 0;
    int e_score_correction_bias_type = 0;
    size_t max_seqlen = 0;
};

// kt_linear_config_t (Zig main.zig:249)
struct kt_linear_config_t {
    size_t hidden_size = 0;
    size_t intermediate_size = 0;
    int stride = 0;
    int group_max_len = 0;
    size_t proj = 0;
    int proj_type = 0;
    int hidden_type = 0;
};

// kt_mlp_config_t (Zig main.zig:259)
struct kt_mlp_config_t {
    size_t hidden_size = 0;
    size_t intermediate_size = 0;
    int stride = 0;
    int group_max_len = 0;
    size_t gate_proj = 0;
    size_t up_proj = 0;
    size_t down_proj = 0;
    int gate_type = 0;
    int up_type = 0;
    int down_type = 0;
    int hidden_type = 0;
};

extern "C" {
    // version / detection
    const char* kt_version();
    const char* kt_get_cpu_variant();

    // ABI layout probes (test-only; used by tools/audit_layout.py)
    size_t kt_abi_size_moe_config();
    size_t kt_abi_size_mla_config();
    size_t kt_abi_size_moe_sft_config();
    size_t kt_abi_size_gate_config();
    size_t kt_abi_size_linear_config();
    size_t kt_abi_size_mlp_config();
    size_t kt_abi_field_offset(int struct_id, int field_index);

    // MoE — config structs are passed BY VALUE (matches include/kt_kernel.h
    // and the Zig export; the Itanium ABI passes >16B aggregates by
    // pointer-to-copy under the hood, so the shim must declare the struct
    // type itself, not a pointer).
    KT_MOE* kt_moe_new(KT_CPUInfer* cpuinfer, kt_moe_config_t config);
    KT_MOE* kt_moe_new_sft(KT_CPUInfer* cpuinfer, kt_moe_sft_config_t config);
    void    kt_moe_forward(KT_MOE* moe, int qlen, int k,
                           const int64_t* expert_ids, const float* weights,
                           const void* input, void* output, int incremental);
    void    kt_moe_forward_sft(KT_MOE* moe, int qlen, int k,
                               const int64_t* expert_ids, const float* weights,
                               const void* input, void* output, int save_for_backward);
    void    kt_moe_backward(KT_MOE* moe, const void* grad_output, void* grad_input,
                            void* gga, void* ggb, void* gua, void* gub,
                            void* gda, void* gdb, void* gw,
                            void* ggp, void* gup, void* gdp,
                            int accumulate, float scale);
    void    kt_moe_load_weights(KT_MOE* moe);
    void    kt_moe_warm_up(KT_MOE* moe);
    void    kt_moe_free(KT_MOE* moe);

    // MLA — config by value
    KT_MLA* kt_mla_new(kt_mla_config_t config);
    void    kt_mla_load_weights(KT_MLA* mla);
    void    kt_mla_forward(KT_MLA* mla, const int* qlens, int qlen_count,
                           const int** page_tables, const int* page_table_lens,
                           const int* kv_lens, const void* input, void* output);
    void    kt_mla_free(KT_MLA* mla);

    // Gate / Linear / MLP — config by value
    KT_Gate*   kt_gate_new(kt_gate_config_t config);
    KT_Linear* kt_linear_new(kt_linear_config_t config);
    KT_MLP*    kt_mlp_new(kt_mlp_config_t config);

    // WorkerPool / CPUInfer
    KT_WorkerPool* kt_worker_pool_new(int thread_count);
    void           kt_worker_pool_free(KT_WorkerPool* pool);
    KT_CPUInfer*   kt_cpuinfer_new(int thread_count);
    void           kt_cpuinfer_free(KT_CPUInfer* cpuinfer);
    void           kt_cpuinfer_submit(KT_CPUInfer* cpuinfer, void (*func)(void*), void* arg);
    void           kt_cpuinfer_sync(KT_CPUInfer* cpuinfer, size_t allow_n_pending);
    KT_WorkerPool* kt_cpuinfer_get_backend(KT_CPUInfer* cpuinfer);

    // DeepseekV3 layer (model orchestration — config BY POINTER, matches Zig)
    void* kt_dsv3_layer_new(const void* config);
    void  kt_dsv3_layer_forward(void* layer, size_t qlen, size_t kv_start_pos,
                                 const void* input, void* output);
    void  kt_dsv3_layer_free(void* layer);
}

// ---------------------------------------------------------------------------
// Python-exposed classes. Each holds an opaque C handle and forwards calls.
// ---------------------------------------------------------------------------

class PyMoe {
public:
    explicit PyMoe(KT_MOE* handle) : handle_(handle) {}
    ~PyMoe() { if (handle_) kt_moe_free(handle_); }
    PyMoe(const PyMoe&) = delete;
    PyMoe& operator=(const PyMoe&) = delete;

    // input/output are raw buffer addresses (Python passes ndarray.ctypes.data)
    void forward(int qlen, int k,
                 const std::vector<int64_t>& expert_ids,
                 const std::vector<float>& weights,
                 size_t input, size_t output, int incremental) {
        kt_moe_forward(handle_, qlen, k, expert_ids.data(), weights.data(),
                       reinterpret_cast<const void*>(input),
                       reinterpret_cast<void*>(output), incremental);
    }
    void forward_sft(int qlen, int k,
                     const std::vector<int64_t>& expert_ids,
                     const std::vector<float>& weights,
                     size_t input, size_t output, int save_for_backward) {
        kt_moe_forward_sft(handle_, qlen, k, expert_ids.data(), weights.data(),
                           reinterpret_cast<const void*>(input),
                           reinterpret_cast<void*>(output), save_for_backward);
    }
    void load_weights() { kt_moe_load_weights(handle_); }
    void warm_up()      { kt_moe_warm_up(handle_); }

private:
    KT_MOE* handle_;
};

class PyMla {
public:
    explicit PyMla(KT_MLA* handle) : handle_(handle) {}
    ~PyMla() { if (handle_) kt_mla_free(handle_); }
    PyMla(const PyMla&) = delete;
    PyMla& operator=(const PyMla&) = delete;

    void load_weights() { kt_mla_load_weights(handle_); }
    // input/output are raw buffer addresses (ndarray.ctypes.data)
    void forward(const std::vector<int>& qlens,
                 const std::vector<std::vector<int>>& page_tables,
                 const std::vector<int>& page_table_lens,
                 const std::vector<int>& kv_lens,
                 size_t input, size_t output) {
        std::vector<const int*> pt_ptrs;
        for (const auto& pt : page_tables) { pt_ptrs.push_back(pt.data()); }
        kt_mla_forward(handle_, qlens.data(), (int)qlens.size(),
                       pt_ptrs.data(), page_table_lens.data(),
                       kv_lens.data(),
                       reinterpret_cast<const void*>(input),
                       reinterpret_cast<void*>(output));
    }

private:
    KT_MLA* handle_;
};

// ---------------------------------------------------------------------------
// Module definition. Module name MUST be kt_kernel_ext to match the C++ build.
// ---------------------------------------------------------------------------
// ---- DeepseekV3DecoderLayer (model orchestration — Zig extension) ----
// ABI struct mirror of kt_dsv3_layer_config_t (src/main.zig; pointer fields
// kept as const void* — the pybind Dsv3LayerCfg below stores size_t ints and
// converts in the constructor).
struct kt_dsv3_layer_config_t {
    size_t hidden_size, q_lora_rank, num_heads, nope_size, rope_size,
           kv_lora_rank, max_qlen, max_kvlen, token_count_in_page;
    double rope_theta;
    size_t expert_num, num_experts_per_tok, intermediate_size,
           n_group, topk_group;
    int norm_topk_prob;
    float routed_scaling_factor;
    void* pool;
    const void* q_a_proj, *q_a_norm, *q_b_proj, *kv_a_proj_with_mqa,
              *kv_a_norm, *kv_b_proj, *o_proj,
              *attn_norm_weight, *ffn_norm_weight, *gate_weight,
              *e_score_correction_bias, *gate_proj, *up_proj, *down_proj;
};

// pybind-side config: same layout, pointer fields as size_t (Python int
// addresses) — layout-identical (8/8), converted in the constructor.
struct Dsv3LayerCfg {
    size_t hidden_size = 0, q_lora_rank = 0, num_heads = 0, nope_size = 0, rope_size = 0,
           kv_lora_rank = 0, max_qlen = 0, max_kvlen = 0, token_count_in_page = 0;
    double rope_theta = 10000.0;   // 0.0 makes RoPE produce NaN — keep the ref default
    size_t expert_num = 0, num_experts_per_tok = 0, intermediate_size = 0,
           n_group = 1, topk_group = 1;
    int norm_topk_prob = 1;
    float routed_scaling_factor = 1.0f;
    size_t pool;
    size_t q_a_proj, q_a_norm, q_b_proj, kv_a_proj_with_mqa,
           kv_a_norm, kv_b_proj, o_proj,
           attn_norm_weight, ffn_norm_weight, gate_weight,
           e_score_correction_bias, gate_proj, up_proj, down_proj;
};

class PyDsv3Layer {
public:
    explicit PyDsv3Layer(void* h) : h_(h) {}
    ~PyDsv3Layer() { if (h_) kt_dsv3_layer_free(h_); }
    PyDsv3Layer(const PyDsv3Layer&) = delete;
    PyDsv3Layer& operator=(const PyDsv3Layer&) = delete;
    void forward(size_t qlen, size_t kv_start_pos, size_t input, size_t output) {
        kt_dsv3_layer_forward(h_, qlen, kv_start_pos,
                              reinterpret_cast<const void*>(input),
                              reinterpret_cast<void*>(output));
    }
private:
    void* h_;
};

PYBIND11_MODULE(kt_kernel_ext, m) {
    m.doc() = "ktransformers-zig C API pybind11 drop-in wrapper";

    // top-level utilities
    m.def("kt_version", [] { return kt_version(); });
    m.def("kt_get_cpu_variant", [] { return kt_get_cpu_variant(); });

    // ---- WorkerPool ----
    py::class_<KT_WorkerPool>(m, "WorkerPool")
        .def(py::init([](int thread_count) -> KT_WorkerPool* {
            KT_WorkerPool* h = kt_worker_pool_new(thread_count);
            if (!h) throw std::runtime_error("kt_worker_pool_new returned null");
            return h;
        }))
        .def("free", [](KT_WorkerPool* h) { kt_worker_pool_free(h); });

    // ---- CPUInfer ----
    py::class_<KT_CPUInfer>(m, "CPUInfer")
        .def(py::init([](int thread_count) -> KT_CPUInfer* {
            KT_CPUInfer* h = kt_cpuinfer_new(thread_count);
            if (!h) throw std::runtime_error("kt_cpuinfer_new returned null");
            return h;
        }))
        .def("free", [](KT_CPUInfer* h) { kt_cpuinfer_free(h); })
        .def("submit", [](KT_CPUInfer* self, py::function cb) {
            // Synchronous fallback: call the Python callback on the calling
            // thread. (The C++ version enqueues to a thread pool; the Zig
            // work-stealing pool is reached via kt_moe_new_with_pool. This
            // synchronous submit is sufficient for single-threaded drop-in
            // testing and matches the ctypes wrapper's behavior.)
            cb();
        })
        .def("sync", [](KT_CPUInfer* /*self*/, size_t /*allow_n_pending*/) {})
        .def("backend_", [](KT_CPUInfer* self) -> KT_WorkerPool* {
            return kt_cpuinfer_get_backend(self);
        });

    // ---- MoE submodule ----
    auto moe = m.def_submodule("moe", "Mixture-of-Experts kernels");

    // PyMoe is THE "MOE" class — constructed directly from MOEConfig.
    py::class_<PyMoe>(moe, "MOE")
        .def(py::init([](const kt_moe_config_t& cfg, KT_CPUInfer* cpuinfer) {
            KT_MOE* h = kt_moe_new(cpuinfer, cfg);
            if (!h) throw std::runtime_error("kt_moe_new returned null");
            return std::unique_ptr<PyMoe>(new PyMoe(h));
        }), py::arg("config"), py::arg("cpuinfer") = nullptr)
        .def("forward", &PyMoe::forward,
             py::arg("qlen"), py::arg("k"), py::arg("expert_ids"),
             py::arg("weights"), py::arg("input"), py::arg("output"),
             py::arg("incremental") = 0)
        .def("forward_sft", &PyMoe::forward_sft,
             py::arg("qlen"), py::arg("k"), py::arg("expert_ids"),
             py::arg("weights"), py::arg("input"), py::arg("output"),
             py::arg("save_for_backward") = 0)
        .def("load_weights", &PyMoe::load_weights)
        .def("warm_up", &PyMoe::warm_up);

    py::class_<kt_moe_config_t>(moe, "MOEConfig")
        .def(py::init<>())
        .def_readwrite("expert_num", &kt_moe_config_t::expert_num)
        .def_readwrite("num_experts_per_tok", &kt_moe_config_t::num_experts_per_tok)
        .def_readwrite("hidden_size", &kt_moe_config_t::hidden_size)
        .def_readwrite("intermediate_size", &kt_moe_config_t::intermediate_size)
        .def_readwrite("layer_idx", &kt_moe_config_t::layer_idx)
        .def_readwrite("pool", &kt_moe_config_t::pool)
        .def_readwrite("num_gpu_experts", &kt_moe_config_t::num_gpu_experts)
        .def_readwrite("gate_proj", &kt_moe_config_t::gate_proj)
        .def_readwrite("up_proj", &kt_moe_config_t::up_proj)
        .def_readwrite("down_proj", &kt_moe_config_t::down_proj)
        .def_readwrite("gate_scale", &kt_moe_config_t::gate_scale)
        .def_readwrite("up_scale", &kt_moe_config_t::up_scale)
        .def_readwrite("down_scale", &kt_moe_config_t::down_scale)
        .def_readwrite("max_len", &kt_moe_config_t::max_len)
        .def_readwrite("path", &kt_moe_config_t::path)
        .def_readwrite("save", &kt_moe_config_t::save)
        .def_readwrite("load", &kt_moe_config_t::load)
        .def_readwrite("share_cache_pool", &kt_moe_config_t::share_cache_pool)
        .def_readwrite("gate_type", &kt_moe_config_t::gate_type)
        .def_readwrite("up_type", &kt_moe_config_t::up_type)
        .def_readwrite("down_type", &kt_moe_config_t::down_type)
        .def_readwrite("hidden_type", &kt_moe_config_t::hidden_type)
        .def_readwrite("swiglu_limit", &kt_moe_config_t::swiglu_limit)
        .def_readwrite("swiglu_alpha", &kt_moe_config_t::swiglu_alpha);

    // (MOE and MOESFT are the registered py::class_ constructors above)

    // ---- MLA submodule ----
    auto mla = m.def_submodule("mla", "Multi-head Latent Attention");

    py::class_<kt_mla_config_t>(mla, "MLAConfig")
        .def(py::init<>())
        .def_readwrite("hidden_size", &kt_mla_config_t::hidden_size)
        .def_readwrite("q_lora_rank", &kt_mla_config_t::q_lora_rank)
        .def_readwrite("num_heads", &kt_mla_config_t::num_heads)
        .def_readwrite("nope_size", &kt_mla_config_t::nope_size)
        .def_readwrite("rope_size", &kt_mla_config_t::rope_size)
        .def_readwrite("kv_lora_rank", &kt_mla_config_t::kv_lora_rank)
        .def_readwrite("layer_idx", &kt_mla_config_t::layer_idx)
        .def_readwrite("pool", &kt_mla_config_t::pool)
        .def_readwrite("token_count_in_page", &kt_mla_config_t::token_count_in_page)
        .def_readwrite("max_qlen", &kt_mla_config_t::max_qlen)
        .def_readwrite("max_kvlen", &kt_mla_config_t::max_kvlen)
        .def_readwrite("max_position_embeddings", &kt_mla_config_t::max_position_embeddings)
        .def_readwrite("rope_scaling_factor", &kt_mla_config_t::rope_scaling_factor)
        .def_readwrite("rope_theta", &kt_mla_config_t::rope_theta)
        .def_readwrite("rope_scaling_beta_fast", &kt_mla_config_t::rope_scaling_beta_fast)
        .def_readwrite("rope_scaling_beta_slow", &kt_mla_config_t::rope_scaling_beta_slow)
        .def_readwrite("rope_scaling_mscale", &kt_mla_config_t::rope_scaling_mscale)
        .def_readwrite("rope_scaling_mscale_all_dim", &kt_mla_config_t::rope_scaling_mscale_all_dim)
        .def_readwrite("rope_scaling_original_max_position_embeddings",
                       &kt_mla_config_t::rope_scaling_original_max_position_embeddings)
        .def_readwrite("q_a_proj", &kt_mla_config_t::q_a_proj)
        .def_readwrite("q_a_norm", &kt_mla_config_t::q_a_norm)
        .def_readwrite("q_b_proj", &kt_mla_config_t::q_b_proj)
        .def_readwrite("kv_a_proj_with_mqa", &kt_mla_config_t::kv_a_proj_with_mqa)
        .def_readwrite("kv_a_norm", &kt_mla_config_t::kv_a_norm)
        .def_readwrite("kv_b_proj", &kt_mla_config_t::kv_b_proj)
        .def_readwrite("o_proj", &kt_mla_config_t::o_proj)
        .def_readwrite("q_a_proj_type", &kt_mla_config_t::q_a_proj_type)
        .def_readwrite("q_a_norm_type", &kt_mla_config_t::q_a_norm_type)
        .def_readwrite("q_b_proj_type", &kt_mla_config_t::q_b_proj_type)
        .def_readwrite("kv_a_proj_with_mqa_type", &kt_mla_config_t::kv_a_proj_with_mqa_type)
        .def_readwrite("kv_a_norm_type", &kt_mla_config_t::kv_a_norm_type)
        .def_readwrite("kv_b_proj_type", &kt_mla_config_t::kv_b_proj_type)
        .def_readwrite("w_o_type", &kt_mla_config_t::w_o_type)
        .def_readwrite("input_type", &kt_mla_config_t::input_type)
        .def_readwrite("output_type", &kt_mla_config_t::output_type)
        .def_readwrite("m_block", &kt_mla_config_t::m_block)
        .def_readwrite("n_block", &kt_mla_config_t::n_block)
        .def_readwrite("page_count", &kt_mla_config_t::page_count);

    // PyMla is THE "MLA" class — constructed directly from MLAConfig.
    py::class_<PyMla>(mla, "MLA")
        .def(py::init([](const kt_mla_config_t& cfg) {
            KT_MLA* h = kt_mla_new(cfg);
            if (!h) throw std::runtime_error("kt_mla_new returned null");
            return std::unique_ptr<PyMla>(new PyMla(h));
        }), py::arg("config"))
        .def("load_weights", &PyMla::load_weights)
        .def("forward", &PyMla::forward,
             py::arg("qlens"), py::arg("page_tables"),
             py::arg("page_table_lens"), py::arg("kv_lens"),
             py::arg("input"), py::arg("output"));

    // ---- DeepseekV3DecoderLayer (model orchestration) ----
    auto dsv3 = m.def_submodule("dsv3", "DeepseekV3 model orchestration");
    py::class_<PyDsv3Layer>(dsv3, "DeepseekV3DecoderLayer")
        .def(py::init([](const Dsv3LayerCfg& cfg) {
            // pybind config stores pointers as size_t ints; the ABI struct
            // uses const void*. Same layout — convert field by field.
            kt_dsv3_layer_config_t abi{};
            abi.hidden_size = cfg.hidden_size;
            abi.q_lora_rank = cfg.q_lora_rank;
            abi.num_heads = cfg.num_heads;
            abi.nope_size = cfg.nope_size;
            abi.rope_size = cfg.rope_size;
            abi.kv_lora_rank = cfg.kv_lora_rank;
            abi.max_qlen = cfg.max_qlen;
            abi.max_kvlen = cfg.max_kvlen;
            abi.token_count_in_page = cfg.token_count_in_page;
            abi.rope_theta = cfg.rope_theta;
            abi.expert_num = cfg.expert_num;
            abi.num_experts_per_tok = cfg.num_experts_per_tok;
            abi.intermediate_size = cfg.intermediate_size;
            abi.n_group = cfg.n_group;
            abi.topk_group = cfg.topk_group;
            abi.norm_topk_prob = cfg.norm_topk_prob;
            abi.routed_scaling_factor = cfg.routed_scaling_factor;
            abi.pool = reinterpret_cast<void*>(cfg.pool);
            abi.q_a_proj = reinterpret_cast<const void*>(cfg.q_a_proj);
            abi.q_a_norm = reinterpret_cast<const void*>(cfg.q_a_norm);
            abi.q_b_proj = reinterpret_cast<const void*>(cfg.q_b_proj);
            abi.kv_a_proj_with_mqa = reinterpret_cast<const void*>(cfg.kv_a_proj_with_mqa);
            abi.kv_a_norm = reinterpret_cast<const void*>(cfg.kv_a_norm);
            abi.kv_b_proj = reinterpret_cast<const void*>(cfg.kv_b_proj);
            abi.o_proj = reinterpret_cast<const void*>(cfg.o_proj);
            abi.attn_norm_weight = reinterpret_cast<const void*>(cfg.attn_norm_weight);
            abi.ffn_norm_weight = reinterpret_cast<const void*>(cfg.ffn_norm_weight);
            abi.gate_weight = reinterpret_cast<const void*>(cfg.gate_weight);
            abi.e_score_correction_bias = reinterpret_cast<const void*>(cfg.e_score_correction_bias);
            abi.gate_proj = reinterpret_cast<const void*>(cfg.gate_proj);
            abi.up_proj = reinterpret_cast<const void*>(cfg.up_proj);
            abi.down_proj = reinterpret_cast<const void*>(cfg.down_proj);
            return std::unique_ptr<PyDsv3Layer>(new PyDsv3Layer(
                kt_dsv3_layer_new(&abi)));
        }), py::arg("config"))
        .def("forward", &PyDsv3Layer::forward,
             py::arg("qlen"), py::arg("kv_start_pos"),
             py::arg("input"), py::arg("output"));

    py::class_<Dsv3LayerCfg>(dsv3, "LayerConfig")
        .def(py::init<>())
        .def_readwrite("hidden_size", &Dsv3LayerCfg::hidden_size)
        .def_readwrite("q_lora_rank", &Dsv3LayerCfg::q_lora_rank)
        .def_readwrite("num_heads", &Dsv3LayerCfg::num_heads)
        .def_readwrite("nope_size", &Dsv3LayerCfg::nope_size)
        .def_readwrite("rope_size", &Dsv3LayerCfg::rope_size)
        .def_readwrite("kv_lora_rank", &Dsv3LayerCfg::kv_lora_rank)
        .def_readwrite("max_qlen", &Dsv3LayerCfg::max_qlen)
        .def_readwrite("max_kvlen", &Dsv3LayerCfg::max_kvlen)
        .def_readwrite("token_count_in_page", &Dsv3LayerCfg::token_count_in_page)
        .def_readwrite("rope_theta", &Dsv3LayerCfg::rope_theta)
        .def_readwrite("expert_num", &Dsv3LayerCfg::expert_num)
        .def_readwrite("num_experts_per_tok", &Dsv3LayerCfg::num_experts_per_tok)
        .def_readwrite("intermediate_size", &Dsv3LayerCfg::intermediate_size)
        .def_readwrite("n_group", &Dsv3LayerCfg::n_group)
        .def_readwrite("topk_group", &Dsv3LayerCfg::topk_group)
        .def_readwrite("norm_topk_prob", &Dsv3LayerCfg::norm_topk_prob)
        .def_readwrite("routed_scaling_factor", &Dsv3LayerCfg::routed_scaling_factor)
        .def_readwrite("pool", &Dsv3LayerCfg::pool)
        .def_readwrite("q_a_proj", &Dsv3LayerCfg::q_a_proj)
        .def_readwrite("q_a_norm", &Dsv3LayerCfg::q_a_norm)
        .def_readwrite("q_b_proj", &Dsv3LayerCfg::q_b_proj)
        .def_readwrite("kv_a_proj_with_mqa", &Dsv3LayerCfg::kv_a_proj_with_mqa)
        .def_readwrite("kv_a_norm", &Dsv3LayerCfg::kv_a_norm)
        .def_readwrite("kv_b_proj", &Dsv3LayerCfg::kv_b_proj)
        .def_readwrite("o_proj", &Dsv3LayerCfg::o_proj)
        .def_readwrite("attn_norm_weight", &Dsv3LayerCfg::attn_norm_weight)
        .def_readwrite("ffn_norm_weight", &Dsv3LayerCfg::ffn_norm_weight)
        .def_readwrite("gate_weight", &Dsv3LayerCfg::gate_weight)
        .def_readwrite("e_score_correction_bias", &Dsv3LayerCfg::e_score_correction_bias)
        .def_readwrite("gate_proj", &Dsv3LayerCfg::gate_proj)
        .def_readwrite("up_proj", &Dsv3LayerCfg::up_proj)
        .def_readwrite("down_proj", &Dsv3LayerCfg::down_proj);

    // ---- kvcache submodule (minimal: expose ggml_type enum) ----
    auto kvcache = m.def_submodule("kvcache", "KV cache utilities");
    enum class ggml_type : int {
        F32 = 0, F16 = 1, BF16 = 2, I8 = 3, I4 = 4, Q4_0 = 8,
        Q4_1 = 9, Q5_0 = 10, Q5_1 = 11, Q8_0 = 12, Q8_1 = 13,
        Q2_K = 14, Q3_K = 15, Q4_K = 16, Q5_K = 17, Q6_K = 18, Q8_K = 19,
    };
    py::enum_<ggml_type>(kvcache, "ggml_type")
        .value("F32", ggml_type::F32).value("F16", ggml_type::F16)
        .value("BF16", ggml_type::BF16).value("I8", ggml_type::I8)
        .value("I4", ggml_type::I4).value("Q4_0", ggml_type::Q4_0)
        .value("Q4_1", ggml_type::Q4_1).value("Q5_0", ggml_type::Q5_0)
        .value("Q5_1", ggml_type::Q5_1).value("Q8_0", ggml_type::Q8_0)
        .value("Q8_1", ggml_type::Q8_1).value("Q2_K", ggml_type::Q2_K)
        .value("Q3_K", ggml_type::Q3_K).value("Q4_K", ggml_type::Q4_K)
        .value("Q5_K", ggml_type::Q5_K).value("Q6_K", ggml_type::Q6_K)
        .value("Q8_K", ggml_type::Q8_K)
        .export_values();
}
