"""Kimi-Linear model compatibility and recurrent-state cache hooks."""

from __future__ import annotations

import sys

import torch

from lores.backends.hf.cache import install_state_dtype_hooks, make_state_quant_hook


def ensure_kimi_loadable() -> None:
    """Restore the OutputRecorder import expected by Kimi remote code before loading."""
    import transformers.utils.generic as G

    if not hasattr(G, "OutputRecorder"):
        from transformers.modeling_utils import OutputRecorder

        G.OutputRecorder = OutputRecorder


def patch_kimi_model(model) -> None:
    """Adapt loaded Kimi cache, masking, gate, and generation APIs."""
    M = sys.modules[type(model).__module__]

    def get_mask_sizes(self, cache_position, layer_idx: int):
        q_len = cache_position if isinstance(cache_position, int) else cache_position.shape[0]
        return q_len + self.get_seq_length(layer_idx), 0

    M.KimiDynamicCache.get_mask_sizes = get_mask_sizes
    if not hasattr(M.KimiDynamicCache, "get_query_offset"):  # transformers >= 5.14 masking_utils
        M.KimiDynamicCache.get_query_offset = lambda self, layer_idx: self.get_seq_length(layer_idx)

    # Transformers 5.x expects a dict; Kimi has no tied weights.
    if isinstance(getattr(type(model), "_tied_weights_keys", None), list):
        type(model)._tied_weights_keys = {}

    if not hasattr(M, "_orig_create_causal_mask"):
        # Transformers >= 5.9 renamed input_embeds and dropped cache_position.
        M._orig_create_causal_mask = M.create_causal_mask

        def create_causal_mask(**kw):
            if "input_embeds" in kw:
                kw["inputs_embeds"] = kw.pop("input_embeds")
            kw.pop("cache_position", None)
            return M._orig_create_causal_mask(**kw)

        M.create_causal_mask = create_causal_mask

    if not hasattr(M, "_orig_fused_kda_gate"):
        M._orig_fused_kda_gate = M.fused_kda_gate

        def fused_kda_gate(*a, g_bias=None, **kw):
            ints = [x for x in a if isinstance(x, int)]  # head_dim passed positionally by Kimi's remote code
            ts = [x for x in a if not isinstance(x, int)]
            g = ts[0]
            # FLA's fused_kda_gate expects [..., H, K], not flattened [..., H*K].
            if ints and g.shape[-1] != ints[0]:
                g = g.reshape(*g.shape[:-1], -1, ints[0])
            if g_bias is not None:
                kw["dt_bias"] = g_bias
            return M._orig_fused_kda_gate(g, *ts[1:], **kw)

        M.fused_kda_gate = fused_kda_gate

    # Let Kimi create its own cache instead of receiving a standard DynamicCache.
    import types

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *a, **kw):
        model_kwargs.pop("past_key_values", None)

    model._prepare_cache_for_generation = types.MethodType(_prepare_cache_for_generation, model)


def iter_kimi_kda_mixers(model):
    """Yield each ``KimiDeltaAttention`` (``layer.self_attn`` of the ``is_linear_attn`` layers)."""
    for layer in model.model.layers:
        if getattr(layer, "is_linear_attn", False):
            yield layer.self_attn


_kimi_cache_accessors = (
    lambda args, kwargs: kwargs.get("cache_params"),
    lambda cp, i: cp.recurrent_states[i] if getattr(cp, "recurrent_states", None) is not None else None,
    lambda cp, i, t: cp.recurrent_states.__setitem__(i, t),
)


_kimi_state_quant_hook = make_state_quant_hook(*_kimi_cache_accessors)


def install_kimi_state_quant(model, *, mode: str, block_size: int = 128) -> int:
    """Install post-readout KDA state quantization; MLA layers are untouched. Returns mixer count."""
    from lores.backends.hf.cache import install_state_quant_hook
    from lores.fake_quant.state import make_state_quant

    n = 0
    for mix in iter_kimi_kda_mixers(model):
        q = None if mode == "off" else make_state_quant(mode, block_size)  # off removes quantization only; dtype hooks stay
        install_state_quant_hook(mix, q, _kimi_state_quant_hook)
        n += 1
    return n


def install_kimi_fp32_state_cache(model, *, strict=False) -> int:
    """Pin KDA state storage to FP32; strict mode rejects already-rounded states."""
    return install_state_dtype_hooks(iter_kimi_kda_mixers(model), torch.float32, *_kimi_cache_accessors, strict=strict)
