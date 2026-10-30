"""Nemotron-H recurrent key/value rotations around SSM kernels."""

from __future__ import annotations

import torch

from .integration import iter_nemotronh_layers


def install_nemotronh_rotations(model, *, rk: bool = False, rv: bool = False, seed: int = 0) -> int:
    """Install per-layer recurrent rotations through loaded kernel wrappers.

    Forward pre-hooks publish each layer's rotation; calls must be sequential.
    R_k rotates B/C on the state axis. R_v rotates x and inversely rotates output;
    intervening terms must be per-head scalars, so in-kernel z gating is rejected.
    Load kernels with a forward pass first. Idempotent; returns mixer count.
    """
    import transformers.models.nemotron_h.modeling_nemotron_h as M

    from lores.rotations.wrappers import make_fused_rot
    from lores.utils.hadamard_utils import hadamard_matrix

    if M.selective_state_update is None or M.mamba_chunk_scan_combined is None:
        raise RuntimeError("NemotronH SSM kernels not loaded — run one forward before installing rotations.")

    N = model.config.ssm_state_size
    dh = model.config.mamba_head_dim
    dev = next(model.parameters()).device
    torch.manual_seed(seed)

    n = 0
    for mix in iter_nemotronh_layers(model):
        if not hasattr(mix, "in_proj"):
            continue
        mix._rot_rk = hadamard_matrix(N, dev).to(torch.float32) if rk else None
        mix._rot_rk_apply = make_fused_rot(mix._rot_rk) if mix._rot_rk is not None else None
        mix._rot_rv = hadamard_matrix(dh, dev).to(torch.float32) if rv else None
        mix._rot_rv_apply = mix._rot_rv_inv_apply = None
        if mix._rot_rv is not None:
            mix._rot_rv_apply = make_fused_rot(mix._rot_rv)
            mix._rot_rv_inv_apply = make_fused_rot(mix._rot_rv.T.contiguous())
        if getattr(mix, "_rot_pre_handle", None) is not None:
            mix._rot_pre_handle.remove()

        def _publish(module, args, _M=M):
            _M._cur_rk_apply = module._rot_rk_apply
            _M._cur_rv_apply = module._rot_rv_apply
            _M._cur_rv_inv_apply = module._rot_rv_inv_apply

        mix._rot_pre_handle = mix.register_forward_pre_hook(_publish)
        n += 1

    if not hasattr(M, "_rk_orig_ssu"):
        M._rk_orig_ssu = M.selective_state_update
        M._rk_orig_csc = M.mamba_chunk_scan_combined
    ssu, csc = M._rk_orig_ssu, M._rk_orig_csc

    def _check_rv_safe(k):
        if k.get("z", None) is not None:
            raise RuntimeError("nemotronh R_v: kernel received z (in-kernel gate) — breaks invariance.")

    def _rk(t):
        # `None` applier ⇔ rk off for this layer ⇒ pass-through.
        ap = getattr(M, "_cur_rk_apply", None)
        return ap(t) if ap is not None and torch.is_tensor(t) else t

    def ssu_wrap(state, x, dt, A, B, C, *a, **k):  # selective_state_update(state,x,dt,A,B,C,...)
        rv_f = getattr(M, "_cur_rv_apply", None)
        if rv_f is None:
            return ssu(state, x, dt, A, _rk(B), _rk(C), *a, **k)
        _check_rv_safe(k)
        y = ssu(state, rv_f(x), dt, A, _rk(B), _rk(C), *a, **k)
        return M._cur_rv_inv_apply(y)

    def csc_wrap(x, dt, A, B, C, *a, **k):  # mamba_chunk_scan_combined(x,dt,A,B,C,...)
        rv_f = getattr(M, "_cur_rv_apply", None)
        if rv_f is None:
            return csc(x, dt, A, _rk(B), _rk(C), *a, **k)
        _check_rv_safe(k)
        out = csc(rv_f(x), dt, A, _rk(B), _rk(C), *a, **k)
        if isinstance(out, tuple):  # (y, final_state) when return_final_states — state stays rotated
            return (M._cur_rv_inv_apply(out[0]), *out[1:])
        return M._cur_rv_inv_apply(out)

    M.selective_state_update = ssu_wrap
    M.mamba_chunk_scan_combined = csc_wrap
    return n
