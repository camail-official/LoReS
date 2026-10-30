"""Register-level INT8 state codecs for uint8 codes and per-row FP32 scales.

Symmetric codes are biased by +127; affine rows store (scale, minimum). Exact
codecs divide and round ties-to-even; FAST uses reciprocals and can differ by one
code at a boundary. SR seeds must advance every step to avoid fixed dithering.
"""

import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def dequant_int8(u8, scale):
    """uint8 code (value = q+127) -> fp32."""
    return (u8.to(tl.float32) - 127.0) * scale


@triton.jit
def requant_int8(x, scale):
    """fp32 -> uint8 code (q+127), RTN (half-to-even), grid ±127."""
    q = libdevice.rint(x / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    return (q + 127.0).to(tl.int32).to(tl.uint8)


@triton.jit
def block_scale(absmax, qmax):
    """fake-path scale: absmax.clamp_min(1e-12) / qmax (fp32)."""
    return tl.maximum(absmax, 1e-12) / qmax


@triton.jit
def dequant_int8_fast(u8, scale, offs):
    """`offs` = 127 * scale, precomputed per row. convert + FMA."""
    return u8.to(tl.float32) * scale - offs


@triton.jit
def requant_int8_fast(x, inv_scale):
    """`inv_scale` = 127 / clamped absmax, per row. mul + one cvt.rni + int clamp."""
    q = libdevice.float2int_rn(x * inv_scale)
    q = tl.minimum(tl.maximum(q, -127), 127)
    return (q + 127).to(tl.uint8)


@triton.jit
def sr_hash(idx, seed):
    """uint32 hash word per (int32 element-pair index, per-row int32 seed).

    The seed enters by addition: XOR with a seed below 2**17 only permutes a fixed index block, which makes the
    rounding noise of a value row (and of a head) coherent over steps.
    """
    x = (idx.to(tl.uint32) + seed.to(tl.uint32) * 2654435761) * 2654435761  # 0x9E3779B1
    x = x ^ (x >> 15)
    x = x * 2246822519  # 0x85EBCA77
    return x ^ (x >> 13)


# The GDN/KDA kernels add half an LSB (2**-17 = 7.62939453125e-06, a literal inside the kernels: Triton cannot read
# python globals) so the noise mean is exactly 0.5.


@triton.jit
def sr_u1(h, LANE: tl.constexpr):
    """fp32 in [1, 2) from 16-bit half LANE of hash word h, moved into the mantissa window with one shift + one mask."""
    if LANE == 0:
        m = (h << 7) & 0x007FFF80  # bits 0..15 -> mantissa 7..22
    else:
        m = (h >> 9) & 0x007FFF80  # bits 16..31
    return (m | 1065353216).to(tl.float32, bitcast=True)  # 0x3F800000 | mantissa


@triton.jit
def requant_int8_sr(x, inv_scale, u1):
    """SR int8: floor(x * inv_scale + u1) with u1 in [1, 2) -> code q+127 == q1+126."""
    q1 = tl.minimum(tl.maximum(libdevice.float2int_rd(x * inv_scale + u1), -126), 128)
    return (q1 + 126).to(tl.uint8)


@triton.jit
def requant_int8_sr_div(x, scale, u1):
    """SR int8, exact division (the HF arithmetic): floor(x / scale + u1), u1 in [1, 2) -> code q1 + 126."""
    q1 = tl.minimum(tl.maximum(libdevice.float2int_rd(x / scale + u1), -126), 128)
    return (q1 + 126).to(tl.uint8)


@triton.jit
def dequant_int8_asym(u8, scale, lo):
    """uint8 code 0..255 -> fp32 on the affine grid."""
    return u8.to(tl.float32) * scale + lo


@triton.jit
def requant_int8_asym(x, scale, lo):
    """RTN onto the affine grid: rint((x - lo) / scale), clamp 0..255 (exact codec)."""
    q = libdevice.rint((x - lo) / scale)
    return tl.minimum(tl.maximum(q, 0.0), 255.0).to(tl.int32).to(tl.uint8)


@triton.jit
def requant_int8_asym_fast(x, inv_scale, lo):
    """FAST RTN: reciprocal-mul + float2int_rn (boundary codes may flip vs the exact codec)."""
    q = libdevice.float2int_rn((x - lo) * inv_scale)
    return tl.minimum(tl.maximum(q, 0), 255).to(tl.uint8)


@triton.jit
def requant_int8_asym_sr(x, inv_scale, lo, u1):
    """SR onto the affine grid: floor((x - lo) * inv_scale + u1), u1 in [1, 2) -> code = q1 - 1."""
    q1 = tl.minimum(tl.maximum(libdevice.float2int_rd((x - lo) * inv_scale + u1), 1), 256)
    return (q1 - 1).to(tl.uint8)


@triton.jit
def requant_int8_asym_sr_div(x, scale, lo, u1):
    """SR int8 affine, exact division: floor((x - lo) / scale + u1) -> code q1 - 1."""
    q1 = tl.minimum(tl.maximum(libdevice.float2int_rd((x - lo) / scale + u1), 1), 256)
    return (q1 - 1).to(tl.uint8)


@triton.jit
def _int8_tile_load(
    codes,
    scales,
    slot,
    stride_codes_slot,
    stride_scales_slot,
    i_hv,
    o_v,
    o_k,
    mask_v,
    mask_h,
    V: tl.constexpr,
    K: tl.constexpr,
    ASYM: tl.constexpr,
):
    """Load + dequantise the [BV, K] tile (value rows ``o_v`` of head ``i_hv`` in ``slot``) -- both kernels."""
    p_c = codes + slot * stride_codes_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    if ASYM:
        p_s = scales + slot * stride_scales_slot + (i_hv * V + o_v) * 2
        b_sc = tl.load(p_s, mask=mask_v, other=1.0)
        b_lo = tl.load(p_s + 1, mask=mask_v, other=0.0)
        b_h = dequant_int8_asym(tl.load(p_c, mask=mask_h, other=0), b_sc[:, None], b_lo[:, None])
    else:
        p_s = scales + slot * stride_scales_slot + i_hv * V + o_v
        b_sc = tl.load(p_s, mask=mask_v, other=1.0)
        b_h = dequant_int8(tl.load(p_c, mask=mask_h, other=127), b_sc[:, None])
    return b_h


@triton.jit
def _int8_tile_store(
    codes,
    scales,
    slot,
    stride_codes_slot,
    stride_scales_slot,
    i_hv,
    o_v,
    o_k,
    mask_v,
    mask_h,
    b_h,
    seed,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    ASYM: tl.constexpr,
    SR: tl.constexpr,
    FAST: tl.constexpr,
):
    """Requantise the updated tile (one scale per value row over all K keys), store codes + scales into ``slot``."""
    p_c = codes + slot * stride_codes_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    sr_rows = (i_hv * V + o_v)[:, None]
    if ASYM:
        mn = tl.min(tl.where(mask_h, b_h, float("inf")), 1)
        mx = tl.max(tl.where(mask_h, b_h, float("-inf")), 1)
        new_scale = tl.maximum((mx - mn) / 255.0, 1e-12)
        if SR:  # 16-bit lanes + half LSB (unbiased)
            hp = sr_hash(sr_rows * (K // 2) + tl.arange(0, K // 2)[None, :], seed)
            u1 = tl.reshape(tl.join(sr_u1(hp, 0), sr_u1(hp, 1)), (BV, K)) + 7.62939453125e-06
            if FAST:
                u8 = requant_int8_asym_sr(b_h, (255.0 / tl.maximum(mx - mn, 1e-12))[:, None], mn[:, None], u1)
            else:
                u8 = requant_int8_asym_sr_div(b_h, new_scale[:, None], mn[:, None], u1)
        elif FAST:
            u8 = requant_int8_asym_fast(b_h, (255.0 / tl.maximum(mx - mn, 1e-12))[:, None], mn[:, None])
        else:
            u8 = requant_int8_asym(b_h, new_scale[:, None], mn[:, None])
        p_s = scales + slot * stride_scales_slot + (i_hv * V + o_v) * 2
        tl.store(p_s, new_scale, mask=mask_v)
        tl.store(p_s + 1, mn, mask=mask_v)
    else:
        new_scale = block_scale(tl.max(tl.abs(tl.where(mask_h, b_h, 0.0)), 1), 127.0)
        if SR:
            hp = sr_hash(sr_rows * (K // 2) + tl.arange(0, K // 2)[None, :], seed)
            u1 = tl.reshape(tl.join(sr_u1(hp, 0), sr_u1(hp, 1)), (BV, K)) + 7.62939453125e-06
            if FAST:
                u8 = requant_int8_sr(b_h, (1.0 / new_scale)[:, None], u1)
            else:
                u8 = requant_int8_sr_div(b_h, new_scale[:, None], u1)
        elif FAST:
            u8 = requant_int8_fast(b_h, (1.0 / new_scale)[:, None])
        else:
            u8 = requant_int8(b_h, new_scale[:, None])
        p_s = scales + slot * stride_scales_slot + i_hv * V + o_v
        tl.store(p_s, new_scale, mask=mask_v)
    tl.store(p_c, u8, mask=mask_h)


@triton.jit
def _int8_tile_load_key(
    codes,
    scales,
    slot,
    stride_codes_slot,
    stride_scales_slot,
    i_hv,
    i_v,
    o_v,
    o_k,
    mask_k,
    mask_h,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    ASYM: tl.constexpr,
):
    """Key axis: load + dequantise the [BV, K] tile with one scale per key channel (column) of the tile."""
    p_c = codes + slot * stride_codes_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    if ASYM:
        p_s = scales + slot * stride_scales_slot + ((i_hv * (V // BV) + i_v) * K + o_k) * 2
        b_sc = tl.load(p_s, mask=mask_k, other=1.0)
        b_lo = tl.load(p_s + 1, mask=mask_k, other=0.0)
        b_h = dequant_int8_asym(tl.load(p_c, mask=mask_h, other=0), b_sc[None, :], b_lo[None, :])
    else:
        p_s = scales + slot * stride_scales_slot + (i_hv * (V // BV) + i_v) * K + o_k
        b_sc = tl.load(p_s, mask=mask_k, other=1.0)
        b_h = dequant_int8(tl.load(p_c, mask=mask_h, other=127), b_sc[None, :])
    return b_h


@triton.jit
def _int8_tile_store_key(
    codes,
    scales,
    slot,
    stride_codes_slot,
    stride_scales_slot,
    i_hv,
    i_v,
    o_v,
    o_k,
    mask_k,
    mask_h,
    b_h,
    seed,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    ASYM: tl.constexpr,
    SR: tl.constexpr,
    FAST: tl.constexpr,
):
    """Key axis: requantise the [BV, K] tile per key channel (one scale per column over the tile's value rows)."""
    p_c = codes + slot * stride_codes_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    if SR:
        sr_rows = (i_hv * V + o_v)[:, None]
        hp = sr_hash(sr_rows * (K // 2) + tl.arange(0, K // 2)[None, :], seed)
        u1 = tl.reshape(tl.join(sr_u1(hp, 0), sr_u1(hp, 1)), (BV, K)) + 7.62939453125e-06
    if ASYM:
        mn = tl.min(tl.where(mask_h, b_h, float("inf")), 0)
        mx = tl.max(tl.where(mask_h, b_h, float("-inf")), 0)
        new_scale = tl.maximum((mx - mn) / 255.0, 1e-12)
        if SR:
            if FAST:
                u8 = requant_int8_asym_sr(b_h, (255.0 / tl.maximum(mx - mn, 1e-12))[None, :], mn[None, :], u1)
            else:
                u8 = requant_int8_asym_sr_div(b_h, new_scale[None, :], mn[None, :], u1)
        elif FAST:
            u8 = requant_int8_asym_fast(b_h, (255.0 / tl.maximum(mx - mn, 1e-12))[None, :], mn[None, :])
        else:
            u8 = requant_int8_asym(b_h, new_scale[None, :], mn[None, :])
        p_s = scales + slot * stride_scales_slot + ((i_hv * (V // BV) + i_v) * K + o_k) * 2
        tl.store(p_s, new_scale, mask=mask_k)
        tl.store(p_s + 1, mn, mask=mask_k)
    else:
        new_scale = block_scale(tl.max(tl.abs(tl.where(mask_h, b_h, 0.0)), 0), 127.0)
        if SR:
            if FAST:
                u8 = requant_int8_sr(b_h, (1.0 / new_scale)[None, :], u1)
            else:
                u8 = requant_int8_sr_div(b_h, new_scale[None, :], u1)
        elif FAST:
            u8 = requant_int8_fast(b_h, (1.0 / new_scale)[None, :])
        else:
            u8 = requant_int8(b_h, new_scale[None, :])
        p_s = scales + slot * stride_scales_slot + (i_hv * (V // BV) + i_v) * K + o_k
        tl.store(p_s, new_scale, mask=mask_k)
    tl.store(p_c, u8, mask=mask_h)
