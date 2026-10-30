"""Pack chunk-final states into decode slots with per-row FP32 scales in one launch.

Matches the naive packer's saturation and scale-clamp ordering; symmetric codes
are biased by +127, while affine rows store scale and minimum.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from lores.kernels.state_int.quant import (
    requant_int8_asym_sr,
    requant_int8_asym_sr_div,
    requant_int8_sr,
    requant_int8_sr_div,
    sr_hash,
    sr_u1,
)

_SAT_BOUND = 1e30  # lores.fake_quant.quantizers._SAT_BOUND


@triton.jit(do_not_specialize=["sr_seed"])  # a seed of 0/1 must stay a runtime int (sr_hash calls .to on it)
def _pack_state_kernel(
    states,  # fp [n, H, D, N], contiguous
    codes,  # uint8 pool [slots, H, D, N]
    scales,  # fp32 pool [slots, H, D] or [slots, H, D, 2]
    slot_ptr,  # int32 [n]
    stride_codes_slot,
    stride_scales_slot,
    sr_seed,  # int32 scalar, one per pack call (SR only): the handoff draws fresh noise like a decode step
    HD: tl.constexpr,  # H * D rows per sequence
    N: tl.constexpr,
    QMAX: tl.constexpr,  # 127, or 255 on the affine grid
    SAT: tl.constexpr,
    FAST: tl.constexpr,
    ASYM: tl.constexpr,  # affine grid: scales row = [scale, lo]
    SR: tl.constexpr,  # stochastic rounding floor(v/s + u) (quant.py), same noise as the decode requant
):
    pid = tl.program_id(0).to(tl.int64)  # one program per (sequence, head, value row)
    i_n = pid // HD
    i_hd = pid % HD
    slot = tl.load(slot_ptr + i_n).to(tl.int64)

    o_n = tl.arange(0, N)
    x = tl.load(states + pid * N + o_n).to(tl.float32)
    # saturating guard: non-finite -> 0/+-SAT, then clamp (real int caches saturate, not blow up)
    x = tl.where(x != x, 0.0, x)
    x = tl.minimum(tl.maximum(x, -SAT), SAT)
    base = codes + slot * stride_codes_slot + i_hd * N + o_n
    if SR:
        hp = sr_hash(((slot * HD + i_hd) * (N // 2) + tl.arange(0, N // 2)).to(tl.int32), sr_seed)  # per pair
        u1 = tl.reshape(tl.join(sr_u1(hp, 0), sr_u1(hp, 1)), (N,)) + 7.62939453125e-06  # half LSB, as the decode kernels

    if ASYM:  # packed_state_kernel_naive.pack_state(asym=True): scale = (max - min) / 255 clamped, lo = min
        lo = tl.min(x, 0)
        hi = tl.max(x, 0)
        sb1 = tl.maximum((hi - lo) / QMAX, 1e-12)
        tl.store(scales + slot * stride_scales_slot + i_hd * 2, sb1)
        tl.store(scales + slot * stride_scales_slot + i_hd * 2 + 1, lo)
        if SR:
            if FAST:
                ua = requant_int8_asym_sr(x, QMAX / tl.maximum(hi - lo, 1e-12), lo, u1)
            else:
                ua = requant_int8_asym_sr_div(x, sb1, lo, u1)
            tl.store(base, ua)
            return
        if FAST:
            qa = libdevice.float2int_rn((x - lo) * (QMAX / tl.maximum(hi - lo, 1e-12))).to(tl.float32)
        else:
            qa = libdevice.rint((x - lo) / sb1)
        tl.store(base, tl.minimum(tl.maximum(qa, 0.0), QMAX).to(tl.uint8))
        return

    absmax = tl.maximum(tl.max(tl.abs(x), 0), 1e-12)
    scale = absmax / QMAX
    tl.store(scales + slot * stride_scales_slot + i_hd, scale)
    # Byte-for-byte the decode requant (quant.py): FAST is reciprocal-mul + float2int_rn,
    # else rint over Triton's plain divide -- approximate, so it can flip an exact tie versus
    # torch (24.5 -> 24.500002 -> 25). Kept plain on purpose: prefill must round like decode.
    if SR:
        if FAST:
            tl.store(base, requant_int8_sr(x, QMAX / absmax, u1))
        else:
            tl.store(base, requant_int8_sr_div(x, scale, u1))
        return
    if FAST:
        q = libdevice.float2int_rn(x * (QMAX / absmax)).to(tl.float32)
    else:
        q = libdevice.rint(x / scale)
    tl.store(base, (tl.minimum(tl.maximum(q, -QMAX), QMAX) + QMAX).to(tl.uint8))


@triton.jit
def _unpack_state_kernel(
    states,  # fp32 out [n, H, D, N], contiguous
    codes,
    scales,
    slot_ptr,
    stride_codes_slot,
    stride_scales_slot,
    HD: tl.constexpr,
    N: tl.constexpr,
    ASYM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    i_n = pid // HD
    i_hd = pid % HD
    slot = tl.load(slot_ptr + i_n).to(tl.int64)
    u = tl.load(codes + slot * stride_codes_slot + i_hd * N + tl.arange(0, N)).to(tl.float32)
    if ASYM:
        sb1 = tl.load(scales + slot * stride_scales_slot + i_hd * 2)
        lo = tl.load(scales + slot * stride_scales_slot + i_hd * 2 + 1)
        tl.store(states + pid * N + tl.arange(0, N), u * sb1 + lo)
    else:
        scale = tl.load(scales + slot * stride_scales_slot + i_hd)
        tl.store(states + pid * N + tl.arange(0, N), (u - 127.0) * scale)


def _check_pools(states, codes, scales, slots, asym):
    assert states.dim() == 4 and states.is_contiguous(), (states.shape, states.is_contiguous())
    n, H, D, N = states.shape
    assert codes.dtype == torch.uint8 and codes.shape[1:] == (H, D, N), codes.shape
    assert codes.stride()[1:] == (D * N, N, 1), "codes must be dense under the slot axis"
    sw = 2 if asym else 1  # fp32 per row in the scales pool
    assert scales.dtype == torch.float32 and scales.numel() // scales.shape[0] == H * D * sw, (scales.shape, sw)
    assert scales.is_contiguous() or scales.stride(0) >= H * D * sw, "scales slot-strided only"
    assert slots.shape == (n,), (slots.shape, n)
    return n * H * D, H * D, N


def unpack_state_from_slots(states, codes, scales, slots, asym=False):
    """Dequantize the pools at ``slots`` into ``states`` [n, H, D, N]."""
    assert states.dtype == torch.float32
    grid, HD, N = _check_pools(states, codes, scales, slots, asym)
    _unpack_state_kernel[(grid,)](
        states, codes, scales, slots.to(torch.int32), codes.stride(0), scales.stride(0), HD=HD, N=N, ASYM=asym, num_warps=4
    )
    return states


def pack_state_to_slots(states, codes, scales, slots, asym=False, sr_seed=None, *, fast=False):
    """Quantize ``states`` [n, H, D, N] into the pools at ``slots``.

    ``sr_seed`` (python int, fresh per call) switches the handoff to stochastic rounding -- the HF hook
    quantizes the prefill-final state with the recipe's SR mode too; None keeps RTN.
    """
    grid, HD, N = _check_pools(states, codes, scales, slots, asym)
    _pack_state_kernel[(grid,)](
        states,
        codes,
        scales,
        slots.to(torch.int32),
        codes.stride(0),
        scales.stride(0),
        0 if sr_seed is None else int(sr_seed) & 0x7FFFFFFF,
        HD=HD,
        N=N,
        QMAX=255 if asym else 127,
        SAT=_SAT_BOUND,
        FAST=fast,
        ASYM=asym,
        SR=sr_seed is not None,
        num_warps=4,
    )
