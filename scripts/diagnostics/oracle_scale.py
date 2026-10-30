"""Closed-loop state quantization with oracle scales (``--oracle-scale`` of the Delta-ell runners).

Every scale group takes its scale from an FP32 copy of the state that is advanced with the same inputs as the
quantized state but never rounded; rounding is unchanged (symmetric grid, RTN or SR, ``int{b}_block[_dk][_sr]``).
MixerZoo advances the copy from the step path's operands; the HF hybrids advance it through the mixer's own forward in a
pre-hook (swap the copy into the cache's recurrent slot, run the step, keep the result, restore the cache).
"""

import re

import torch

from lores.fake_quant.quantizers import _saturate
from scripts.diagnostics.rotation_scale_feedback import blocks, unblocks


def parse(mode):
    """``(qmax, value_axis, sr)`` of a symmetric integer block mode."""
    m = re.fullmatch(r"int(\d+)_block(_dk)?(_sr)?", mode)
    if m is None:
        raise ValueError(f"oracle scales need a symmetric integer block mode, got {mode!r}")
    return (1 << (int(m.group(1)) - 1)) - 1, bool(m.group(2)), bool(m.group(3))


def round_at(xb, scale, qmax, sr):
    """Round scale groups ``xb`` on the symmetric grid of ``scale``."""
    v = xb / scale
    if sr:
        fl = torch.floor(v)
        q = fl + (torch.rand_like(v) < (v - fl)).to(v.dtype)
    else:
        q = torch.round(v)
    return q.clamp_(-qmax, qmax) * scale


def advance(attn, S, k32, v32, g, beta):
    """Decay then write or delta update in the stored basis, as the MixerZoo step path."""
    R = attn._mz_rk
    if attn.decay == "scalar":
        S = S * g.exp()[..., None, None]
    elif attn.decay == "diagonal":
        a = g.exp()
        if R is None:
            S = S * a[..., None]
        else:
            S = torch.einsum("ji,bhjv->bhiv", R, a[..., None] * torch.einsum("ij,bhjv->bhiv", R, S))
    if attn.is_delta:
        kTs = torch.einsum("bhk,bhkv->bhv", k32, S)
        return S + torch.einsum("bhk,bhv->bhkv", k32, beta.unsqueeze(-1) * (v32 - kTs))
    return S + torch.einsum("bhk,bhv->bhkv", k32, v32)


class MixerZooOracleQuant:
    """Per-mixer quantizer for the MixerZoo step path."""

    def __init__(self, attn, mode):
        self.qmax, value_axis, self.sr = parse(mode)
        self.axis = "value" if value_axis else "key"
        self.attn = attn
        self.shadow = None
        self.last_t = -1
        attn._mz_stash_ops = True

    def __call__(self, S):
        t = self.attn._mz_step_state.t
        if self.shadow is None or t <= self.last_t or self.shadow.shape != S.shape:
            self.shadow = torch.zeros_like(S, dtype=torch.float32)
        self.last_t = t
        _, k32, v32, g, beta = self.attn._mz_last_ops
        self.shadow = advance(self.attn, self.shadow, k32, v32, g, beta)
        xb, n = blocks(_saturate(S.to(torch.float32)), self.axis)
        xb = xb.contiguous()  # SR noise is drawn on a contiguous transposed tensor, as in quantize_state
        scale = blocks(self.shadow, self.axis)[0].abs().amax(-1, keepdim=True).clamp_min(1e-12) / self.qmax
        return unblocks(round_at(xb, scale, self.qmax, self.sr), n, self.axis)


class HybridOracleQuant:
    """Quantizer for the HF state hooks; ``busy`` marks the nested forward that advances the copy."""

    def __init__(self, mode, block_size=128):
        self.qmax, self.dk, self.sr = parse(mode)
        self.block = block_size
        self.shadow = None
        self.busy = False

    def reset(self):
        self.shadow = None

    def __call__(self, S):
        if self.busy:
            return S
        S = S.to(torch.float32)
        if self.shadow is None:  # prefill store: the first stored state is clean
            self.shadow = S.detach().clone()
        x, xs = _saturate(S), self.shadow
        if self.dk:
            x, xs = x.transpose(-1, -2).contiguous(), xs.transpose(-1, -2).contiguous()
        *lead, n = x.shape
        pad = (self.block - n % self.block) % self.block
        if pad:
            x, xs = torch.nn.functional.pad(x, (0, pad)), torch.nn.functional.pad(xs, (0, pad))
        xb, sb = x.reshape(*lead, -1, self.block), xs.reshape(*lead, -1, self.block)
        scale = sb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / self.qmax
        out = round_at(xb, scale, self.qmax, self.sr).reshape(*lead, x.shape[-1])[..., :n]
        return out.transpose(-1, -2).contiguous() if self.dk else out


def _conv_slot(cp, i):
    layer = getattr(cp, "layers", None)
    if layer is not None:  # transformers LinearAttentionCacheLayer
        return layer[i].conv_states, 0
    return cp.conv_states, i


def _shadow_hook(get_cache, get_state, set_state):
    """Pre-hook: advance the mixer's FP32 copy through its own forward, then restore the cache slots."""

    def hook(module, args, kwargs):
        q = module._oracle
        if q.busy or q.shadow is None:
            return
        cp = get_cache(args, kwargs)
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if cp is None or hs is None or hs.dim() < 2 or hs.shape[1] != 1:
            return
        rs = get_state(cp, module.layer_idx)
        if rs is None:
            return
        holder, key = _conv_slot(cp, module.layer_idx)
        conv_saved = holder[key].clone()
        rs_saved = rs.detach().clone()
        rs.copy_(q.shadow.to(rs.dtype))
        q.busy = True
        try:
            with torch.no_grad():
                module(*args, **kwargs)
        finally:
            q.busy = False
        now = get_state(cp, module.layer_idx)
        q.shadow = now.detach().to(torch.float32).clone()
        if now.shape == rs_saved.shape and now.dtype == rs_saved.dtype:
            now.copy_(rs_saved)
        else:
            set_state(cp, module.layer_idx, rs_saved)
        holder[key] = conv_saved

    return hook


def install_oracle_scale(model, arch, mode, *, block_size=128):
    """Replace every installed state quantizer of Nemotron-H or Qwen3.5 by the oracle-scale one."""
    from lores.backends.hf.state import state_quant_mixers

    if arch == "nemotronh":
        from lores.backends.hf.nemotronh.integration import _nemotronh_cache_accessors as accessors
    elif arch == "qwen3_5":
        from lores.backends.hf.qwen3_5.integration import _qwen3_5_cache_accessors as accessors
    else:
        raise ValueError(f"oracle scales are wired for nemotronh and qwen3_5, not {arch}")
    mixers = state_quant_mixers(model)
    if not mixers:
        raise ValueError("no state quantizers installed")
    for mix in mixers:
        mix._state_quant = mix._oracle = HybridOracleQuant(mode, block_size)
        mix.register_forward_pre_hook(_shadow_hook(*accessors), with_kwargs=True)
    return len(mixers)
