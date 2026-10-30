"""Naive PyTorch reference of the int8 state pool: uint8 codes (q + 127; 0..255 on the affine
grid) + one fp32 scale per row over the last axis. Same quantizer math as the fake path (``_saturate``, then
``absmax.clamp_min(1e-12) / 127``), so ``unpack_state(pack_state(h))`` equals the fake representative bit for bit.
Used by the kernel tests and benches to build and check slot pools.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from lores.fake_quant.quantizers import _saturate

__all__ = ["PackedState", "pack_state", "unpack_state"]


class PackedState(NamedTuple):
    """One layer's recurrent state as codes + scales. ``unpack`` always returns fp32: the int grid is exact there,
    and a bf16 cache dtype would re-round every element on every read.
    """

    codes: torch.Tensor  # uint8
    scales: torch.Tensor  # fp32 per row; [..., 2] = (scale, min) on the affine grid
    shape: tuple  # original state shape
    asym: bool = False  # affine grid: scales[..., 0] = (max-min)/255, scales[..., 1] = min; codes 0..255


@torch.no_grad()
def pack_state(h: torch.Tensor, asym: bool = False) -> PackedState:
    """Quantize state ``h`` to int8 codes + one fp32 scale per row (same math/saturation as the fake path).

    ``asym``: the affine per-row grid of ``quantizers.quant_intn_block_asym``: scale = (max - min) / 255 clamped
    at 1e-12, zero-point = min, codes round((x - min) / scale) in 0..255; ``scales`` is ``[..., 2]`` = (scale, min).
    """
    shape = tuple(h.shape)
    x = _saturate(h.to(torch.float32))
    if asym:
        lo = x.amin(dim=-1, keepdim=True)
        hi = x.amax(dim=-1, keepdim=True)
        scale = ((hi - lo) / 255).clamp_min(1e-12)
        u = torch.round((x - lo) / scale).clamp_(0, 255).to(torch.uint8)
        return PackedState(u.contiguous(), torch.cat([scale, lo], dim=-1).contiguous(), shape, True)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127
    q = torch.round(x / scale).clamp_(-127, 127)  # signed codes, extra negative dropped
    u = (q.to(torch.int16) + 127).to(torch.uint8)
    return PackedState(u.contiguous(), scale.squeeze(-1).contiguous(), shape)


@torch.no_grad()
def unpack_state(p: PackedState) -> torch.Tensor:
    """Dequantize to the fp32 tensor the next decode step consumes (== the fake representative)."""
    if p.asym:
        return (p.codes.to(torch.float32) * p.scales[..., 0:1] + p.scales[..., 1:2]).reshape(p.shape)
    return ((p.codes.to(torch.float32) - 127) * p.scales.unsqueeze(-1)).reshape(p.shape)
