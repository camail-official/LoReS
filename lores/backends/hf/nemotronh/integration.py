"""Nemotron-H model compatibility and recurrent-state cache hooks."""

from __future__ import annotations

import torch

from lores.backends.hf.cache import get_recurrent_state, install_state_dtype_hooks, make_state_quant_hook, set_recurrent_state


def _patch_hybrid_cache_attn_layer_lookup() -> None:
    """Resolve default attention-cache lookups from initialized attention layers.

    Unused MLP/MoE placeholders must not determine past length or decode masks.
    Keep explicit layer indices and the original lookup before prefill.
    """
    # Transformers >= 5.9 requires explicit placeholder mappings for MLP/MoE layers.
    from transformers import cache_utils as _cu
    from transformers.cache_utils import Cache, CacheLayerMixin

    for _map_name in ("DYNAMIC_LAYER_TYPE_MAPPING", "STATIC_LAYER_TYPE_MAPPING"):
        _map = getattr(_cu, _map_name, None)
        if _map is not None:
            _default = _cu.DynamicLayer if "DYNAMIC" in _map_name else _cu.StaticLayer
            _map.setdefault("mlp", _default)
            _map.setdefault("moe", _default)

    if getattr(Cache, "_lores_attn_lookup_patched", False):
        return

    def attn_idx(cache, layer_idx):
        if layer_idx not in (0, None):
            return layer_idx  # explicit index: caller knows which layer it wants
        for i, layer in enumerate(cache.layers):
            if isinstance(layer, CacheLayerMixin) and layer.is_initialized:
                return i
        return layer_idx

    seq_length, mask_sizes = Cache.get_seq_length, Cache.get_mask_sizes
    Cache.get_seq_length = lambda self, layer_idx=0: seq_length(self, attn_idx(self, layer_idx))
    Cache.get_mask_sizes = lambda self, query_length, layer_idx: mask_sizes(self, query_length, attn_idx(self, layer_idx))
    Cache._lores_attn_lookup_patched = True


_patch_hybrid_cache_attn_layer_lookup()


def iter_nemotronh_layers(model):
    """Yield ``block.mixer`` for each ``NemotronHBlock`` (detection / introspection)."""
    if getattr(getattr(model, "config", None), "model_type", "") != "nemotron_h":
        return
    for block in model.model.layers:
        yield block.mixer


_nemotronh_cache_accessors = (
    lambda args, kwargs: kwargs.get("cache_params", args[1] if len(args) > 1 else None),
    get_recurrent_state,
    set_recurrent_state,
)


_nemotronh_state_quant_hook = make_state_quant_hook(*_nemotronh_cache_accessors)


def install_nemotronh_fp32_state_cache(model, *, strict=False) -> int:
    """Pin recurrent storage to FP32 before cache writes; strict mode rejects already-rounded caches."""
    mixers = (mix for mix in iter_nemotronh_layers(model) if hasattr(mix, "in_proj"))
    return install_state_dtype_hooks(mixers, torch.float32, *_nemotronh_cache_accessors, strict=strict)


def install_nemotronh_state_quant(model, *, mode: str, block_size: int = 128) -> int:
    """Install post-readout state quantization on NemotronH mamba mixers.

    State layout is ``[B, heads, head_dim, ssm_state_size]``; block grouping
    follows the last axis. Attention/MLP blocks are skipped.
    Replaces existing hooks; returns the number of patched mixers.
    """
    from lores.backends.hf.cache import install_state_quant_hook
    from lores.fake_quant.state import make_state_quant

    n = 0
    for mix in iter_nemotronh_layers(model):
        if not hasattr(mix, "in_proj"):  # mamba mixers only
            continue
        q = None if mode == "off" else make_state_quant(mode, block_size)  # off removes quantization only; dtype hooks stay
        install_state_quant_hook(mix, q, _nemotronh_state_quant_hook)
        n += 1
    return n
