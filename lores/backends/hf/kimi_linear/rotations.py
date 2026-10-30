"""Kimi-Linear carried value rotations; diagonal decay forbids carried key rotations."""

from __future__ import annotations

import sys

import torch

from lores.rotations.wrappers import rotate_operand
from lores.utils.hadamard_utils import hadamard_matrix

from .integration import iter_kimi_kda_mixers


def install_kimi_rv(model, *, seed: int = 0) -> int:
    """Rotate KDA values by per-layer Hadamards and inversely rotate kernel output.

    Pre-hooks publish each layer's rotation to shared wrappers; calls must be
    sequential. Carried R_k is invalid through KDA's per-channel decay.
    Idempotent; returns ``dv``.
    """
    mixers = list(iter_kimi_kda_mixers(model))
    if not mixers:
        raise RuntimeError("no KDA layers found — not a Kimi-Linear model?")
    M = sys.modules[type(mixers[0]).__module__]  # the dynamically-loaded modeling_kimi module

    dv = mixers[0].head_dim
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    for mix in mixers:
        mix._rv_R_layer = hadamard_matrix(dv, dev).to(torch.float32)  # fresh per layer
        if getattr(mix, "_rv_pre_handle", None) is not None:
            mix._rv_pre_handle.remove()

        def _publish(module, args, _M=M):
            _M._rv_R = module._rv_R_layer

        mix._rv_pre_handle = mix.register_forward_pre_hook(_publish)
    M._rv_R = mixers[0]._rv_R_layer

    if not hasattr(M, "_rv_orig_kernels"):
        M._rv_orig_kernels = (M.chunk_kda, M.fused_recurrent_kda)
    chunk0, rec0 = M._rv_orig_kernels

    def _mk(kernel):
        def wrapped(*a, **kw):  # called as kernel(q=..., k=..., v=..., g=..., beta=..., ...)
            kw["v"] = rotate_operand(kw["v"], M._rv_R)
            o, state = kernel(*a, **kw)
            return rotate_operand(o, M._rv_R.T), state

        return wrapped

    M.chunk_kda = _mk(chunk0)
    M.fused_recurrent_kda = _mk(rec0)
    return dv
