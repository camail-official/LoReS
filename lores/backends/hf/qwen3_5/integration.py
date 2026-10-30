"""Qwen3.5 text-backbone integration and recurrent-state cache hooks."""

from __future__ import annotations

import torch

from lores.backends.hf.cache import get_recurrent_state, install_state_dtype_hooks, make_state_quant_hook, set_recurrent_state


def text_backbone(model):
    """Return the text backbone from an omni wrapper or text-only model."""
    inner = model.model
    return inner.language_model if hasattr(inner, "language_model") else inner


def is_linear_layer(layer) -> bool:
    """True iff the layer is a GDN (``linear_attention``) layer; layer_type -> block_type on transformers >= 5.9."""
    lt = getattr(layer, "layer_type", None) or getattr(layer, "block_type", None)
    return lt == "linear_attention" if lt is not None else hasattr(layer, "linear_attn")


def layer_mixer(layer):
    """The token mixer of a ``Qwen3_5DecoderLayer`` (``linear_attn`` for GDN layers, ``self_attn`` for attention)."""
    return layer.linear_attn if is_linear_layer(layer) else layer.self_attn


def iter_qwen3_5_gdn_mixers(model):
    """Yield each ``Qwen3_5GatedDeltaNet`` (the ``linear_attention`` layers' ``layer.linear_attn``)."""
    for layer in text_backbone(model).layers:
        mix = getattr(layer, "linear_attn", None)
        if mix is not None:
            yield mix


_qwen3_5_cache_accessors = (
    lambda args, kwargs: kwargs.get("cache_params", args[1] if len(args) > 1 else None),
    get_recurrent_state,  # dict-aware ({0: tensor} on transformers >= 5.9, bare tensor on 5.8)
    set_recurrent_state,
)


_qwen3_5_state_quant_hook = make_state_quant_hook(*_qwen3_5_cache_accessors)


def install_qwen3_5_state_quant(model, *, mode: str, block_size: int = 128) -> int:
    """Install post-readout GDN state quantization; attention/MLP layers are untouched. Returns mixer count."""
    from lores.backends.hf.cache import install_state_quant_hook
    from lores.fake_quant.state import make_state_quant

    n = 0
    for mix in iter_qwen3_5_gdn_mixers(model):
        q = None if mode == "off" else make_state_quant(mode, block_size)  # off removes quantization only; dtype hooks stay
        install_state_quant_hook(mix, q, _qwen3_5_state_quant_hook)
        n += 1
    return n


def install_qwen3_5_fp32_state_cache(model, *, strict=False) -> int:
    """Pin recurrent storage to FP32 before cache writes.

    Strict mode rejects already-rounded states and non-FP32 results.
    Weights and convolution caches retain their configured dtypes.
    """
    return install_state_dtype_hooks(iter_qwen3_5_gdn_mixers(model), torch.float32, *_qwen3_5_cache_accessors, strict=strict)
