"""Single-token MixerZoo recurrence with FP32 state and quantization after readout.

Multi-token calls retain the original chunked forward; recurrent cells store the
quantized state between steps.
"""

from __future__ import annotations

from types import MethodType

import torch
from einops import rearrange
from fla.modules.l2norm import l2norm

from lores.fake_quant.state import make_state_quant

from .integration import iter_mixer_zoo_layers, unified_mixer_cls

__all__ = ["install_mixer_zoo_stepping", "reset_mixer_zoo_state", "remove_mixer_zoo_stepping"]


class _StepState:
    """Per-mixer decode-time buffers: conv caches, the recurrent state, and ``t``."""

    __slots__ = ("conv_q", "conv_k", "conv_v", "S", "t")

    def __init__(self, attn, batch_size: int, device, dtype):
        W = attn.q_conv1d.kernel_size[0]
        self.conv_q = torch.zeros(batch_size, attn.key_dim, W, device=device, dtype=dtype)
        self.conv_k = torch.zeros(batch_size, attn.key_dim, W, device=device, dtype=dtype)
        self.conv_v = torch.zeros(batch_size, attn.value_dim, W, device=device, dtype=dtype)
        self.S = torch.zeros(batch_size, attn.num_heads, attn.head_k_dim, attn.head_v_dim, device=device, dtype=torch.float32)
        self.t = 0


def _step(attn, hidden_states: torch.Tensor):
    """One decode step: ``hidden_states`` is ``[B, 1, D]``; returns the mixer output ``[B, 1, D]``."""
    x = hidden_states
    B = x.shape[0]
    st = attn._mz_step_state
    if st is None:
        st = attn._mz_step_state = _StepState(attn, B, x.device, x.dtype)

    # proj + conv step (ShortConvolution's native single-token path, cache in-place)
    q, _ = attn.q_conv1d(attn.q_proj(x), cache=st.conv_q, output_final_state=True)
    k, _ = attn.k_conv1d(attn.k_proj(x), cache=st.conv_k, output_final_state=True)
    v, _ = attn.v_conv1d(attn.v_proj(x), cache=st.conv_v, output_final_state=True)

    q = rearrange(q, "... (h d) -> ... h d", d=attn.head_k_dim)
    k = rearrange(k, "... (h d) -> ... h d", d=attn.head_k_dim)
    v = rearrange(v, "... (h d) -> ... h d", d=attn.head_v_dim)

    # qk featurization norm + RoPE at the current position (mirrors UnifiedMixer.forward)
    if attn.qk_norm == "rms":
        q, k = attn.q_norm(q), attn.k_norm(k)
    else:
        q, k = l2norm(q), l2norm(k)
    q, k = attn.rotary(q, k, seqlen_offset=st.t)

    g = attn._decay(x)  # [B, 1, H] or [B, 1, H, dk], log-decay in fp32
    beta = attn.b_proj(x).sigmoid() if attn._use_beta else None
    q32, k32, v32 = q[:, 0].float(), k[:, 0].float(), v[:, 0].float()  # [B, H, d]
    if attn.mixer_family == "linear" and attn.linear_write_gate:
        v32 = v32 * beta[:, 0].float().unsqueeze(-1)
    R = attn._mz_rk  # [dk, dk] orthonormal or None: quantize S in a rotated dk-basis
    if R is not None:
        q32, k32 = q32 @ R, k32 @ R  # state stored as Rᵀ S; reads/writes rotate with it
    Rv = attn._mz_rv  # [dv, dv] orthonormal or None: rotated dv-basis
    if Rv is not None:
        # state stored as S·Rv. Both decay forms act on dk (scalar per head, or diagonal along
        # dk), so a dv-axis rotation commutes with the recurrence exactly — the value operand
        # rotates on write and the read is un-rotated below.
        v32 = v32 @ Rv
    S = st.S
    # decay FIRST, then the write / delta update (the naive references' order)
    if attn.decay == "scalar":
        S = S * g[:, 0].float().exp()[..., None, None]
    elif attn.decay == "diagonal":
        a = g[:, 0].float().exp()  # [B, H, dk] — acts in the ORIGINAL dk basis
        if R is not None:
            # diagonal decay does not commute with R: sandwich back to the original basis,
            # decay, re-rotate. Exact in fp.
            S = torch.einsum("ji,bhjv->bhiv", R, a[..., None] * torch.einsum("ij,bhjv->bhiv", R, S))
        else:
            S = S * a[..., None]  # decay along dk
    if attn.is_delta:
        b32 = beta[:, 0].float().unsqueeze(-1)  # [B, H, 1]
        kTs = torch.einsum("bhk,bhkv->bhv", k32, S)
        S = S + torch.einsum("bhk,bhv->bhkv", k32, b32 * (v32 - kTs))
    else:
        S = S + torch.einsum("bhk,bhv->bhkv", k32, v32)
    if getattr(attn, "_mz_stash_ops", False):
        attn._mz_last_ops = (
            q32,
            k32,
            v32,
            None if g is None else g[:, 0].float(),
            None if beta is None else beta[:, 0].float(),
        )
    # Read the updated FP32 state BEFORE storage quantization. Rounding at t
    # first affects the recurrence/readout at t+1; there is no alternate order.
    o = torch.einsum("bhk,bhkv->bhv", q32 * attn.head_k_dim**-0.5, S)
    st.S = attn._mz_state_quant(S) if attn._mz_state_quant is not None else S
    if Rv is not None:
        o = o @ Rv.T  # read back out of the rotated dv-basis (exact in fp)
    o = o.unsqueeze(1).to(x.dtype)  # [B, 1, H, dv]

    gate = rearrange(attn.g_proj(x), "... (h d) -> ... h d", d=attn.head_v_dim)
    o = attn.o_norm(o, gate)
    o = rearrange(o, "... h d -> ... (h d)")
    st.t += 1
    return attn.o_proj(o), None, None


def _routed_forward(
    self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False, **kwargs
):
    if hidden_states.shape[1] == 1:
        return _step(self, hidden_states)
    return unified_mixer_cls().forward(
        self,
        hidden_states,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        **kwargs,
    )


def install_mixer_zoo_stepping(model, state_quant=None, rk: torch.Tensor | None = None, rv: torch.Tensor | None = None):
    """Route single-token mixer forwards through the recurrent step path.

    ``state_quant`` is a mode or per-layer quantizer factory. ``rv`` stores
    ``S @ Rv``; ``rk`` uses a carried basis where valid, or a per-step sandwich
    for diagonal decay. Both matrices must be orthonormal FP32.
    """
    for idx, attn in enumerate(iter_mixer_zoo_layers(model)):
        if attn.is_softmax:
            raise ValueError("MixerZoo stepping covers recurrent cells only")
        if state_quant is None:
            attn._mz_state_quant = None
        elif callable(state_quant):
            attn._mz_state_quant = state_quant(idx)
        else:
            attn._mz_state_quant = make_state_quant(state_quant)
        attn._mz_rk = rk
        attn._mz_rv = rv
        attn._mz_step_state = None
        attn.forward = MethodType(_routed_forward, attn)
    return model


def reset_mixer_zoo_state(model):
    """Drop all per-mixer step buffers (call between sequences)."""
    for attn in iter_mixer_zoo_layers(model):
        attn._mz_step_state = None


def remove_mixer_zoo_stepping(model):
    """Restore the original chunked forward on every mixer."""
    for attn in iter_mixer_zoo_layers(model):
        for name in ("_mz_step_state", "_mz_state_quant", "_mz_rk", "_mz_rv"):
            if hasattr(attn, name):
                delattr(attn, name)
        if "forward" in attn.__dict__:
            del attn.forward
