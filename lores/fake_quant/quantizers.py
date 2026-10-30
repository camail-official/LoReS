"""Tensor-only fake quantization: grids, scaling, rounding, and numerical guards.

Quantized values are represented in floating point; this module does not pack a
cache or provide integer-storage memory savings. No model or runner imports.
"""

from __future__ import annotations

import functools
import re

import torch

_SAT_BOUND = 1e30
_sat_counts: dict = {}  # device -> GPU int64 counter; synchronized only at report time


def _validate_int_bits(bits: int) -> None:
    """Integer state grids use 2–8 bits; FP16/FP32 baselines are separate modes."""
    if not 2 <= bits <= 8:
        raise ValueError(f"integer state quantization requires 2–8 bits, got {bits}")


def _quant_int_block(x_fp32: torch.Tensor, bits: int, block_size: int, *, asym: bool, stochastic: bool) -> torch.Tensor:
    """Shared integer grid, rounding, and reconstruction; keep scale-clamp order format-specific."""
    _validate_int_bits(bits)
    x_fp32 = x_fp32.to(torch.float32)
    *lead, n = x_fp32.shape
    pad = (block_size - n % block_size) % block_size
    xp = torch.nn.functional.pad(x_fp32, (0, pad)) if pad else x_fp32
    if pad and asym:
        xp[..., n:] = x_fp32[..., -1:]  # Repeating a valid value leaves min/max unchanged.
    xb = xp.reshape(*lead, -1, block_size)
    if asym:
        qmin, qmax = 0, (1 << bits) - 1
        lo = xb.amin(dim=-1, keepdim=True)
        hi = xb.amax(dim=-1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp_min(1e-12)
        v = (xb - lo) / scale
    else:
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax
        scale = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
        v = xb / scale
    if stochastic:
        floor = torch.floor(v)
        q = floor + (torch.rand_like(v) < (v - floor)).to(v.dtype)
    else:
        q = torch.round(v)
    q = q.clamp_(qmin, qmax)
    deq = q * scale
    if asym:
        deq = deq + lo
    return deq.reshape(*lead, xp.shape[-1])[..., :n]


def quant_intn_block(x_fp32: torch.Tensor, bits: int, block_size: int = 128) -> torch.Tensor:
    """Round to a symmetric 2–8-bit grid with FP32 block scales along the last axis.

    Uses ``[-qmax, qmax]``, where ``qmax = 2**(bits - 1) - 1``.
    """
    return _quant_int_block(x_fp32, bits, block_size, asym=False, stochastic=False)


def quant_intn_block_sr(x_fp32: torch.Tensor, bits: int, block_size: int = 128) -> torch.Tensor:
    """Use the symmetric INT-n block grid with stochastic floor/Bernoulli rounding."""
    return _quant_int_block(x_fp32, bits, block_size, asym=False, stochastic=True)


def quant_intn_block_asym_sr(x_fp32: torch.Tensor, bits: int, block_size: int = 128) -> torch.Tensor:
    """Use an affine INT-n grid per last-axis block with stochastic rounding."""
    return _quant_int_block(x_fp32, bits, block_size, asym=True, stochastic=True)


def quant_intn_block_asym(x_fp32: torch.Tensor, bits: int, block_size: int = 128) -> torch.Tensor:
    """Use an affine INT-n grid per last-axis block with round-to-nearest."""
    return _quant_int_block(x_fp32, bits, block_size, asym=True, stochastic=False)


@functools.lru_cache(maxsize=32)
def _fp_grid(e: int, m: int) -> torch.Tensor:
    """Ascending non-negative representable magnitudes of a ``(sign, e, m)`` float code.

    E4M3 follows the E4M3FN finite grid (the all-ones code is NaN, so the largest
    magnitude is 448); other ``(e, m)`` pairs use an all-finite grid.
    """
    bias = 2 ** (e - 1) - 1
    vals = {0.0}
    for f in range(1, 2**m):
        vals.add(2.0 ** (1 - bias) * (f / 2**m))
    for E in range(1, 2**e):
        for f in range(2**m):
            if (e, m, E, f) == (4, 3, 15, 7):
                continue  # E4M3FN reserves this pattern for NaN.
            vals.add(2.0 ** (E - bias) * (1 + f / 2**m))
    return torch.tensor(sorted(vals), dtype=torch.float32)


def quant_fp_block(x_fp32: torch.Tensor, e: int, m: int, block_size: int = 128, stochastic: bool = False) -> torch.Tensor:
    """Fake-quantize to a scaled ``1+e+m``-bit float grid along the last axis.

    Each block's maximum magnitude maps to the grid maximum. Uses nearest
    or stochastic rounding; format conventions are defined by :func:`_fp_grid`.
    """
    x = x_fp32.to(torch.float32)
    lead, last = x.shape[:-1], x.shape[-1]
    bs = min(block_size, last)
    assert last % bs == 0, f"last dim {last} not divisible by block size {bs}"
    xb = x.reshape(*lead, last // bs, bs)
    grid = _fp_grid(e, m).to(x.device)
    scale = (xb.abs().amax(dim=-1, keepdim=True) / grid[-1]).clamp_min(1e-12)
    y = (xb / scale).abs()
    idx = torch.bucketize(y, grid)
    lo = grid[(idx - 1).clamp_min(0)]
    hi = grid[idx.clamp_max(grid.numel() - 1)]
    if stochastic:
        # Round up with probability given by the position between neighboring grid values.
        gap = hi - lo
        pr = torch.where(gap > 0, (y - lo) / gap.clamp_min(1e-30), torch.zeros_like(y))
        q = torch.where(torch.rand_like(y) < pr, hi, lo)
    else:
        lower = (y - lo) <= (hi - y)
        if (e, m) == (4, 3):
            # E4M3FN round-to-nearest, ties-to-even significand (code LSB).
            tie = (y - lo) == (hi - y)
            lower = torch.where(tie, (idx - 1).clamp_min(0).remainder(2) == 0, lower)
        q = torch.where(lower, lo, hi)
    return (torch.sign(xb) * q * scale).reshape(*lead, last)


def _saturate(h: torch.Tensor) -> torch.Tensor:
    """Clamp state values to ±1e30 and replace NaNs with zero before quantization.

    Counts affected elements per device for :func:`saturation_events`.
    """
    over = (~torch.isfinite(h)) | (h.abs() > _SAT_BOUND)
    n = over.sum()
    c = _sat_counts.get(h.device)
    if c is None:
        _sat_counts[h.device] = n.to(torch.int64)
    else:
        c += n
    return torch.nan_to_num(h, nan=0.0, posinf=_SAT_BOUND, neginf=-_SAT_BOUND).clamp_(-_SAT_BOUND, _SAT_BOUND)


def saturation_events() -> int:
    """Total saturated state elements so far in this process (syncs; call at report time)."""
    return int(sum(int(c.item()) for c in _sat_counts.values()))


def quantize_state(h_fp32: torch.Tensor, mode: str, block_size: int = 128) -> torch.Tensor:
    """Quantize one recurrent state ``h`` (``[B, num_heads, head_dim, state_size]``).

    Returns an FP32 representative of the quantized state. ``mode`` ∈
    ``off``, integer block/tensor modes (2–8 bits), or the
    ``fpE{e}M{m}_block`` float grammar.
    All non-``off`` modes run through the saturating guard :func:`_saturate` first.
    """
    if mode == "off":
        return h_fp32
    h_fp32 = _saturate(h_fp32)
    # "_block" groups the last axis; "_dk" transposes the final two axes first.
    # "_tensor" uses one symmetric scale per state matrix (per head).
    mt = re.fullmatch(r"int(\d+)_tensor", mode)
    if mt:
        bits = int(mt.group(1))
        _validate_int_bits(bits)
        qmax = (1 << (bits - 1)) - 1
        x = h_fp32.to(torch.float32)
        scale = x.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12) / qmax
        return torch.round(x / scale).clamp_(-qmax, qmax) * scale
    mf = re.fullmatch(r"fpE(\d+)M(\d+)_block(_dk)?(_sr)?", mode)
    if mf:
        e, mm = int(mf.group(1)), int(mf.group(2))
        sr = bool(mf.group(4))
        if mf.group(3):
            xt = h_fp32.to(torch.float32).transpose(-1, -2).contiguous()
            return quant_fp_block(xt, e, mm, block_size, stochastic=sr).transpose(-1, -2).contiguous()
        return quant_fp_block(h_fp32, e, mm, block_size, stochastic=sr).to(torch.float32)
    m = re.fullmatch(r"int(\d+)_block(_dk)?(_sr|_asym|_asym_sr)?", mode)
    if m:
        bits = int(m.group(1))
        fn = {"_sr": quant_intn_block_sr, "_asym": quant_intn_block_asym, "_asym_sr": quant_intn_block_asym_sr}.get(
            m.group(3), quant_intn_block
        )
        if m.group(2):
            xt = h_fp32.to(torch.float32).transpose(-1, -2).contiguous()
            return fn(xt, bits, block_size).transpose(-1, -2).contiguous()
        return fn(h_fp32, bits, block_size).to(torch.float32)
    raise ValueError(f"unknown state_quant mode {mode!r}")
