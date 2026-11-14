"""High-level Python bindings for the Kimi K3 C API (ktransformers-zig).

Wraps kt_kimi_* (layer, model, causal LM, incremental state) with ctypes.
Weights are caller-owned (zero-copy): pass numpy arrays / pointers.

Example:
    import ctypes, numpy as np
    from kt_kernel.kimi import KimiModel, KimiModelState

    cfg = KimiModelConfig(num_layers=..., hidden=..., ...)
    # ... fill layer_configs array ...
    model = KimiModel(cfg)
    state = model.new_state(max_seq=4096)
    out = model.forward_inc(state, x_hidden)   # x_hidden: np.ndarray [hidden]
    model.free_state(state)
    model.free()
"""

import ctypes
import os

# Reuse the already-loaded library from the package __init__.
try:
    from . import _so as _lib
except Exception:  # pragma: no cover - fallback when imported standalone
    import ctypes as _ct
    _here = os.path.dirname(os.path.abspath(__file__))
    _lib = _ct.CDLL(os.path.join(_here, "libkt_kernel_ext.so"))


# ---------------------------------------------------------------------------
# Opaque handles
# ---------------------------------------------------------------------------

class KT_KimiKdaLayer(ctypes.Structure):
    pass

class KT_KimiMlaLayer(ctypes.Structure):
    pass

class KT_KimiLayer(ctypes.Structure):
    pass

class KT_KimiModel(ctypes.Structure):
    pass

class KT_KimiCausalLM(ctypes.Structure):
    pass

class KT_KimiModelState(ctypes.Structure):
    pass


# ---------------------------------------------------------------------------
# Config structs (mirror include/kt_kernel.h)
# ---------------------------------------------------------------------------

class kt_kimi_kda_layer_config_t(ctypes.Structure):
    _fields_ = [
        ("hidden", ctypes.c_size_t),
        ("kda_heads", ctypes.c_size_t),
        ("kda_head_dim", ctypes.c_size_t),
        ("conv_k", ctypes.c_size_t),
        ("gate_lb", ctypes.c_float),
        ("rms_eps", ctypes.c_float),
        ("q", ctypes.c_void_p),
        ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p),
        ("g", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("f_a", ctypes.c_void_p),
        ("f_b", ctypes.c_void_p),
        ("o", ctypes.c_void_p),
        ("q_conv", ctypes.c_void_p),
        ("k_conv", ctypes.c_void_p),
        ("v_conv", ctypes.c_void_p),
        ("A_log", ctypes.c_void_p),
        ("dt_bias", ctypes.c_void_p),
        ("o_norm", ctypes.c_void_p),
    ]


class kt_kimi_mla_layer_config_t(ctypes.Structure):
    _fields_ = [
        ("hidden", ctypes.c_size_t),
        ("n_heads", ctypes.c_size_t),
        ("q_lora", ctypes.c_size_t),
        ("kv_lora", ctypes.c_size_t),
        ("qk_nope", ctypes.c_size_t),
        ("qk_rope", ctypes.c_size_t),
        ("v_head", ctypes.c_size_t),
        ("rms_eps", ctypes.c_float),
        ("q_a", ctypes.c_void_p),
        ("q_b", ctypes.c_void_p),
        ("kv_a", ctypes.c_void_p),
        ("kv_b", ctypes.c_void_p),
        ("o", ctypes.c_void_p),
        ("g", ctypes.c_void_p),
        ("q_a_norm", ctypes.c_void_p),
        ("kv_a_norm", ctypes.c_void_p),
    ]


class kt_kimi_layer_config_t(ctypes.Structure):
    _fields_ = [
        ("is_mla", ctypes.c_int),
        ("hidden", ctypes.c_size_t),
        ("rms_eps", ctypes.c_float),
        ("kda_heads", ctypes.c_size_t),
        ("kda_head_dim", ctypes.c_size_t),
        ("conv_k", ctypes.c_size_t),
        ("gate_lb", ctypes.c_float),
        ("kda_q", ctypes.c_void_p),
        ("kda_k", ctypes.c_void_p),
        ("kda_v", ctypes.c_void_p),
        ("kda_g", ctypes.c_void_p),
        ("kda_b", ctypes.c_void_p),
        ("kda_f_a", ctypes.c_void_p),
        ("kda_f_b", ctypes.c_void_p),
        ("kda_o", ctypes.c_void_p),
        ("kda_q_conv", ctypes.c_void_p),
        ("kda_k_conv", ctypes.c_void_p),
        ("kda_v_conv", ctypes.c_void_p),
        ("kda_A_log", ctypes.c_void_p),
        ("kda_dt_bias", ctypes.c_void_p),
        ("kda_o_norm", ctypes.c_void_p),
        ("n_heads", ctypes.c_size_t),
        ("q_lora", ctypes.c_size_t),
        ("kv_lora", ctypes.c_size_t),
        ("qk_nope", ctypes.c_size_t),
        ("qk_rope", ctypes.c_size_t),
        ("v_head", ctypes.c_size_t),
        ("mla_q_a", ctypes.c_void_p),
        ("mla_q_b", ctypes.c_void_p),
        ("mla_kv_a", ctypes.c_void_p),
        ("mla_kv_b", ctypes.c_void_p),
        ("mla_o", ctypes.c_void_p),
        ("mla_g", ctypes.c_void_p),
        ("mla_q_a_norm", ctypes.c_void_p),
        ("mla_kv_a_norm", ctypes.c_void_p),
        ("in_norm", ctypes.c_void_p),
        ("post_norm", ctypes.c_void_p),
        ("attn_res_norm", ctypes.c_void_p),
        ("attn_res_proj", ctypes.c_void_p),
        ("mlp_res_norm", ctypes.c_void_p),
        ("mlp_res_proj", ctypes.c_void_p),
        ("is_dense", ctypes.c_int),
        ("dense_inter", ctypes.c_size_t),
        ("dense_gate", ctypes.c_void_p),
        ("dense_up", ctypes.c_void_p),
        ("dense_down", ctypes.c_void_p),
    ]


class kt_kimi_model_config_t(ctypes.Structure):
    _fields_ = [
        ("num_layers", ctypes.c_size_t),
        ("hidden", ctypes.c_size_t),
        ("attn_res_block", ctypes.c_size_t),
        ("layers", ctypes.c_void_p),   # const kt_kimi_layer_config_t*
        ("final_norm", ctypes.c_void_p),
    ]


class kt_kimi_causallm_config_t(ctypes.Structure):
    _fields_ = [
        ("model", kt_kimi_model_config_t),
        ("lm_head", ctypes.c_void_p),
        ("vocab_size", ctypes.c_size_t),
    ]


# ---------------------------------------------------------------------------
# Function bindings
# ---------------------------------------------------------------------------

def _bind(name, argtypes, restype):
    fn_ = getattr(_lib, name)
    fn_.argtypes = argtypes
    fn_.restype = restype
    return fn_

kt_kimi_model_new = _bind("kt_kimi_model_new", [ctypes.POINTER(kt_kimi_model_config_t)], ctypes.POINTER(KT_KimiModel))
kt_kimi_model_free = _bind("kt_kimi_model_free", [ctypes.POINTER(KT_KimiModel)], None)
kt_kimi_model_forward = _bind("kt_kimi_model_forward", [ctypes.POINTER(KT_KimiModel), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)], None)

kt_kimi_model_state_new = _bind("kt_kimi_model_state_new", [ctypes.POINTER(KT_KimiModel), ctypes.c_size_t], ctypes.POINTER(KT_KimiModelState))
kt_kimi_model_state_free = _bind("kt_kimi_model_state_free", [ctypes.POINTER(KT_KimiModelState)], None)
kt_kimi_model_forward_inc = _bind("kt_kimi_model_forward_inc", [ctypes.POINTER(KT_KimiModel), ctypes.POINTER(KT_KimiModelState), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)], None)

kt_kimi_causallm_new = _bind("kt_kimi_causallm_new", [ctypes.POINTER(kt_kimi_causallm_config_t)], ctypes.POINTER(KT_KimiCausalLM))
kt_kimi_causallm_free = _bind("kt_kimi_causallm_free", [ctypes.POINTER(KT_KimiCausalLM)], None)
kt_kimi_causallm_forward = _bind("kt_kimi_causallm_forward", [ctypes.POINTER(KT_KimiCausalLM), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)], None)


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------

import numpy as np


class KimiModelState:
    def __init__(self, ptr):
        self._ptr = ptr

    @property
    def ptr(self):
        return self._ptr

    def free(self):
        if self._ptr is not None:
            kt_kimi_model_state_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.free()


class KimiModel:
    def __init__(self, cfg):
        self._cfg = cfg
        self._ptr = kt_kimi_model_new(cfg)

    @property
    def ptr(self):
        return self._ptr

    def forward(self, qlen, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty(qlen * self._cfg.hidden, dtype=np.float32)
        kt_kimi_model_forward(self._ptr, qlen, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out.reshape(qlen, self._cfg.hidden)

    def new_state(self, max_seq=4096):
        return KimiModelState(kt_kimi_model_state_new(self._ptr, max_seq))

    def forward_inc(self, state, x):
        x = np.ascontiguousarray(x, dtype=np.float32).ravel()
        out = np.empty_like(x)
        kt_kimi_model_forward_inc(self._ptr, state.ptr, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out

    def free(self):
        if self._ptr is not None:
            kt_kimi_model_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.free()


class KimiCausalLM:
    def __init__(self, cfg, vocab_size):
        self._cfg = cfg
        self._vocab_size = vocab_size
        self._ptr = kt_kimi_causallm_new(cfg)

    def forward(self, qlen, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty(qlen * self._vocab_size, dtype=np.float32)
        kt_kimi_causallm_forward(self._ptr, qlen, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out.reshape(qlen, self._vocab_size)

    def free(self):
        if self._ptr is not None:
            kt_kimi_causallm_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.free()
