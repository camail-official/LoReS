"""Recurrent-cache access and dtype hooks; models supply their cache layout and mixer selection."""

import functools

import torch

FP32_STATE_PROTOCOL = {"dtype": "float32", "order": "update_readout_quantize_store"}


def get_recurrent_state(cp, i):
    """Read a tensor or dict-backed recurrent cache slot."""
    rs = getattr(cp.layers[i], "recurrent_states", None)
    return rs.get(0) if isinstance(rs, dict) else rs


def set_recurrent_state(cp, i, t):
    """Replace a tensor or dict-backed recurrent cache slot."""
    rs = getattr(cp.layers[i], "recurrent_states", None)
    if isinstance(rs, dict):
        rs[0] = t
    else:
        cp.layers[i].recurrent_states = t


def install_state_dtype_hooks(mixers, dtype: torch.dtype, get_cache, get_state, set_state, *, strict=False) -> int:
    """Pin cache allocation BEFORE its first copy, and initialized states before forward.

    Reinstallation replaces only our dtype hook. Already-correct tensors retain their
    identity. Lazy zero buffers are promoted before the kernel result is copied
    into them, without changing convolution-cache dtype. Strict FP32 evaluation
    rejects already-rounded nonzero state and non-FP32 kernel/cache results.
    """

    def cast(module, args, kwargs):
        cp = get_cache(args, kwargs)
        if cp is None:
            return
        if hasattr(cp, "layers"):
            layer = cp.layers[module.layer_idx]
            if hasattr(layer, "update_recurrent_state") and not hasattr(layer, "_lq_update_original"):
                layer._lq_update_original = layer.update_recurrent_state

                @functools.wraps(layer._lq_update_original)
                def update(recurrent_states, *a, **kw):
                    if module._lq_strict_state and recurrent_states.dtype != torch.float32:
                        raise RuntimeError("Kernel returned non-FP32 recurrent state before cache copy")
                    return layer._lq_update_original(recurrent_states, *a, **kw)

                layer.update_recurrent_state = update
            if hasattr(layer, "lazy_initialization") and not hasattr(layer, "_lq_lazy_original"):
                layer._lq_lazy_original = layer.lazy_initialization

                @functools.wraps(layer._lq_lazy_original)
                def initialize(*a, **kw):
                    result = layer._lq_lazy_original(*a, **kw)
                    state = get_state(cp, module.layer_idx)
                    target = module._lq_cache_dtype
                    if isinstance(state, torch.Tensor) and state.numel() and state.dtype != target:
                        if module._lq_strict_state and bool(state.count_nonzero()):
                            raise RuntimeError("Cache initializer rounded nonzero recurrent state before the FP32 pin")
                        set_state(cp, module.layer_idx, state.to(target))
                    return result

                layer.lazy_initialization = initialize
        rs = get_state(cp, module.layer_idx)
        if isinstance(rs, torch.Tensor) and rs.numel() and rs.dtype != dtype:
            if strict and bool(rs.count_nonzero()):
                raise RuntimeError("Cannot recover FP32 recurrent state from an already-rounded nonzero cache")
            set_state(cp, module.layer_idx, rs.to(dtype))

    def check(module, args, kwargs, output):
        cp = get_cache(args, kwargs)
        if cp is None:
            return
        rs = get_state(cp, module.layer_idx)
        if not isinstance(rs, torch.Tensor) or not rs.numel() or rs.dtype != torch.float32:
            raise RuntimeError("FP32-state evaluation requires a nonempty FP32 recurrent state on every cache write")

    n = 0
    for mix in mixers:
        mix._lq_cache_dtype, mix._lq_strict_state = dtype, strict
        if getattr(mix, "_fp32s_handle", None) is not None:
            mix._fp32s_handle.remove()
        if getattr(mix, "_fp32s_check_handle", None) is not None:
            mix._fp32s_check_handle.remove()
        mix._fp32s_check_handle = None
        mix._fp32s_handle = mix.register_forward_pre_hook(cast, with_kwargs=True)
        if strict:
            if dtype != torch.float32:
                raise ValueError("Strict reference evaluation requires FP32 recurrent state")
            # Check BEFORE a fake-quant hook could mask a bad kernel/cache dtype.
            mix._fp32s_check_handle = mix.register_forward_hook(check, with_kwargs=True, prepend=True)
        n += 1
    return n


def make_state_quant_hook(get_cache, get_state, set_state):
    """Quantize after readout for the next decode step, storing FP32 representatives.

    Prefill resets sequence state and quantizes its final store once. No-cache
    forwards reset only; architecture callbacks access the recurrent cache.
    """

    def hook(module, args, kwargs, output):
        cp = get_cache(args, kwargs)
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if cp is None or hs is None or hs.dim() < 2:
            if hasattr(module._state_quant, "reset"):
                module._state_quant.reset()  # no cache → new sequence: drop per-sequence quantizer state
            return
        if hs.shape[1] != 1:
            if hasattr(module._state_quant, "reset"):
                module._state_quant.reset()
        rs = get_state(cp, module.layer_idx)
        if rs is None:
            return
        q = module._state_quant(rs.to(torch.float32))
        if rs.dtype == torch.float32:
            rs.copy_(q)  # in place: keep tensor identity for the next step's cache read
        else:
            set_state(cp, module.layer_idx, q)  # promote the cache slot to fp32

    return hook


def install_state_quant_hook(mix, q, hook) -> None:
    """Replace a mixer's quantization hook; q=None removes it without changing the cache."""
    if getattr(mix, "_sq_handle", None) is not None:
        mix._sq_handle.remove()
        mix._sq_handle = None
    mix._state_quant = q
    if q is None:
        return
    mix._sq_handle = mix.register_forward_hook(hook, with_kwargs=True)
