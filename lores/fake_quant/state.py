"""Recurrent-state policies and shared hooks for fake-quantized cache writes.

Architecture modules supply cache accessors and explicit state layouts. Quantizer
callables return FP32 representatives and may expose reset() for sequence state.
"""

from __future__ import annotations

import re

import torch

from .quantizers import quantize_state


def _quantize_allocated_bits(x: torch.Tensor, bits: int, gran: str, block_size: int) -> torch.Tensor:
    """Apply one allocation's precision; FP32 protection must never round through FP16."""
    if bits >= 32:
        return x
    if bits >= 16:
        return x.to(torch.float16).to(torch.float32)
    return quantize_state(x, f"int{bits}_{gran}", block_size)


def _axis_granularities(layout: str) -> tuple[str, str]:
    """Return (value-axis, key-axis) grouping suffixes for the state layout."""
    return ("block", "block_dk") if layout == "vk" else ("block_dk", "block")


class HeadBitsQuant:
    """Allocate integer precision per head, with one shared grouping granularity."""

    def __init__(self, gran: str, bits: torch.Tensor, block_size: int = 128):
        self.gran, self.bits, self.bs = gran, bits.long(), block_size

    def __call__(self, h_fp32: torch.Tensor) -> torch.Tensor:
        h = h_fp32.to(torch.float32)
        out = h.clone()
        for b in self.bits.unique().tolist():
            m = (self.bits == b).to(h.device)
            out[:, m] = _quantize_allocated_bits(h[:, m], b, self.gran, self.bs)
        return out


class AxisBitsQuant:
    """Allocate grouping axis and bits per head [H] or per key channel [H,K].

    Channel allocation uses full-group grids before selecting state rows.
    Rounding/symmetry come from the actual mode, not from the allocation width.
    """

    def __init__(
        self,
        value_mask: torch.Tensor,
        bits: torch.Tensor,
        block_size: int = 128,
        layout: str = "kv",
        *,
        mode: str = "int4_block",
    ):
        match = re.fullmatch(r"int[2-8]_block(?:_dk)?(_asym_sr|_sr|_asym)?", mode)
        if match is None:
            raise ValueError(f"Axis bit allocation requires an integer block mode, got {mode!r}")
        self.variant = match.group(1) or ""
        self.m, self.bits, self.bs = value_mask.bool(), bits.long(), block_size
        if self.m.shape != self.bits.shape or self.bits.ndim not in (1, 2):
            raise ValueError("Axis and bit masks must have identical [H] or [H,K] shapes")
        if self.bits.ndim == 2 and layout != "kv":
            raise ValueError("Channel-axis allocation requires the [B,H,K,V] layout")
        self.val_gran, self.key_gran = _axis_granularities(layout)

    def __call__(self, h_fp32: torch.Tensor) -> torch.Tensor:
        x = h_fp32.to(torch.float32)
        out = x.clone()
        for b in self.bits.unique().tolist():
            for m, g in ((self.m, self.val_gran), (~self.m, self.key_gran)):
                sel = (m & (self.bits == b)).to(x.device)
                if not bool(sel.any()):
                    continue
                if self.bits.ndim == 2:
                    if tuple(self.bits.shape) != tuple(x.shape[1:3]):
                        raise ValueError("Channel mask does not match state [H,K]")
                    q = _quantize_allocated_bits(x, b, g + self.variant, self.bs)
                    out = torch.where(sel[None, ..., None], q, out)
                else:
                    out[:, sel] = _quantize_allocated_bits(x[:, sel], b, g + self.variant, self.bs)
        return out


def make_state_quant(mode: str, block_size: int = 128):
    """Factory for a per-mixer state quantizer: a stateless :func:`quantize_state` partial."""
    from functools import partial

    return partial(quantize_state, mode=mode, block_size=block_size)


def quant_every(inner, C):
    """Quantize every C-th call; reset() restarts the phase and resets the inner quantizer."""
    box = {"n": 0}

    def q(S):
        box["n"] += 1
        return inner(S) if box["n"] % C == 0 else S

    def _reset():
        box["n"] = 0
        if hasattr(inner, "reset"):
            inner.reset()

    q.reset = _reset
    return q
