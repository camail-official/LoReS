"""PyTorch key-axis INT8 packer for the prefill/decode handoff (Mamba-2, GDN and KDA).

Value-major codes have shape ``[slots, HV, V, K]``; scales are shared over V for
each K channel: ``[slots, HV, K]`` or ``[slots, HV, K, 2]`` for (scale, minimum).
Uses the same symmetric/affine grids as the key-axis decode kernel.
"""

import torch


def quantize_key_axis(h: torch.Tensor, asym: bool, sr_seed: int | None = None, group: int | None = None):
    """``h`` fp32 ``[n, HV, V, K]`` -> (codes uint8 ``[n, HV, V, K]``, scales ``[n, HV, K(, 2)]``).

    ``group``: one scale per ``group`` consecutive value rows instead of all V; scales are then
    ``[n, HV, V // group, K(, 2)]``.
    """
    assert h.dim() == 4 and h.dtype == torch.float32, (h.shape, h.dtype)
    n, HV, V, K = h.shape
    G = group or V
    x = h.reshape(n, HV, V // G, G, K)
    if sr_seed is not None:
        gen = torch.Generator(device=h.device).manual_seed(int(sr_seed))
        u = torch.rand(x.shape, generator=gen, device=h.device, dtype=torch.float32)
    if asym:
        mn = x.amin(dim=3)  # over the group's value rows -> [n, HV, V//G, K]
        mx = x.amax(dim=3)
        scale = ((mx - mn) / 255.0).clamp_min(1e-12)
        q = (x - mn[:, :, :, None, :]) / scale[:, :, :, None, :]
        q = torch.floor(q + u) if sr_seed is not None else torch.round(q)
        codes = q.clamp_(0, 255).to(torch.uint8).reshape(h.shape)
        scales = torch.stack([scale, mn], dim=-1)
    else:
        scale = (x.abs().amax(dim=3) / 127.0).clamp_min(1e-12)
        q = x / scale[:, :, :, None, :]
        q = torch.floor(q + u) if sr_seed is not None else torch.round(q)
        codes = (q.clamp_(-127, 127) + 127).to(torch.uint8).reshape(h.shape)
        scales = scale
    return codes, (scales.squeeze(2) if group is None else scales)


def dequantize_key_axis(codes: torch.Tensor, scales: torch.Tensor, asym: bool) -> torch.Tensor:
    """Inverse of :func:`quantize_key_axis`; returns fp32 ``[n, HV, V, K]``."""
    n, HV, V, K = codes.shape
    grouped = scales.ndim == (5 if asym else 4)
    G = V // scales.shape[2] if grouped else V
    c = codes.to(torch.float32).reshape(n, HV, V // G, G, K)
    s = scales if grouped else scales.unsqueeze(2)
    if asym:
        out = c * s[..., 0][:, :, :, None, :] + s[..., 1][:, :, :, None, :]
    else:
        out = (c - 127.0) * s[:, :, :, None, :]
    return out.reshape(codes.shape)


@torch.no_grad()
def pack_key_axis_to_slots(states, codes, scales, slots, asym: bool, sr_seed=None) -> None:
    """Quantize ``states`` fp32 ``[n, HV, V, K]`` into the pools at ``slots`` (int64/int32 ``[n]``)."""
    c, s = quantize_key_axis(states.contiguous().to(torch.float32), asym, sr_seed)
    idx = slots.long()
    codes[idx] = c
    scales[idx] = s.to(scales.dtype)


@torch.no_grad()
def unpack_key_axis_from_slots(out, codes, scales, slots, asym: bool):
    """Dequantize the pools at ``slots`` into ``out`` fp32 ``[n, HV, V, K]``; returns ``out``."""
    idx = slots.long()
    out.copy_(dequantize_key_axis(codes[idx], scales[idx], asym))
    return out
