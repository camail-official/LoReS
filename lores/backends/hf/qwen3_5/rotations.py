"""Qwen3.5 recurrent Q/K and V rotations around GDN kernels."""

from __future__ import annotations

import torch

from lores.rotations.wrappers import make_fused_rot

from .integration import iter_qwen3_5_gdn_mixers


def _wrap_gdn_kernels(mix, fq, fk, fv, fo):
    """Wrap GDN kernels with operand/output transforms, rebuilding from saved originals."""
    if not hasattr(mix, "_rot_orig_kernels"):
        mix._rot_orig_kernels = (mix.chunk_gated_delta_rule, mix.recurrent_gated_delta_rule)
    chunk0, rec0 = mix._rot_orig_kernels

    def _mk(kernel):
        def wrapped(q, k, v, *a, **kw):
            out, state = kernel(fq(q), fk(k), fv(v), *a, **kw)
            return fo(out), state

        return wrapped

    mix.chunk_gated_delta_rule = _mk(chunk0)
    mix.recurrent_gated_delta_rule = _mk(rec0)


def _install_rv_norm_unrotate(mix, R: torch.Tensor) -> None:
    """Undo R_v before the gated norm, using a fused kernel when R factors.

    Reinstallation unwraps the previous norm wrapper.
    """
    from lores.rotations.wrappers import UnrotateWrapper, try_factor_and_register

    mix.R_v = R
    fused = try_factor_and_register(mix, attr="R_v", block_dtype=torch.float16, device=R.device)
    if isinstance(mix.norm, UnrotateWrapper):
        mix.norm = mix.norm.norm
    mix.norm = UnrotateWrapper(mix.norm, layer=mix, rot_attr="R_v", use_fused=fused)


def install_qwen3_5_rk(model, *, seed: int = 0) -> int:
    """Rotate Q/K by per-layer Hadamards; store state as R_k.T @ S.

    Scalar head decay and Q/K L2 normalization commute with this rotation.
    Returns key-head dimension.
    """
    from lores.utils.hadamard_utils import hadamard_matrix

    dk = getattr(model.config, "text_config", model.config).linear_key_head_dim
    dev = next(model.parameters()).device
    torch.manual_seed(seed)

    ident = lambda t: t  # noqa: E731
    for mix in iter_qwen3_5_gdn_mixers(model):
        rot = make_fused_rot(hadamard_matrix(dk, dev).to(torch.float32))  # fresh Hadamard per layer
        _wrap_gdn_kernels(mix, rot, rot, ident, ident)
    return dk


def install_qwen3_5_rv(model, *, seed: int = 0) -> int:
    """Rotate V by per-layer Hadamards and undo the rotation before the gated norm.

    Store state as S @ R_v. Returns value-head dimension.
    """
    from lores.utils.hadamard_utils import hadamard_matrix

    dv = getattr(model.config, "text_config", model.config).linear_value_head_dim
    dev = next(model.parameters()).device
    torch.manual_seed(seed)

    ident = lambda t: t  # noqa: E731
    for mix in iter_qwen3_5_gdn_mixers(model):
        R = hadamard_matrix(dv, dev).to(torch.float32)  # fresh Hadamard per layer
        _wrap_gdn_kernels(mix, ident, ident, make_fused_rot(R), ident)
        _install_rv_norm_unrotate(mix, R)  # inverse fused into the gated norm (o @ Rᵀ + RMSNorm, one launch)
    return dv
