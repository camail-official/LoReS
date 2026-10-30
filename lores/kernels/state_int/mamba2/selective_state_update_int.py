# Copied from mamba_ssm 2.3.1 `mamba_ssm/ops/triton/selective_state_update.py`
# (Copyright (c) 2024, Tri Dao, Albert Gu) and adapted for LoReS's int8 state cache. Edits are marked
# `# INT-STATE:`. Decode-only; the chunked prefill kernels are never touched.
"""``selective_state_update`` with the state held in HBM as int8 codes.

The fp state tensor never exists in HBM: each program loads its rows' codes + fp32 scales, dequantizes in
registers, runs the (unchanged) update + readout math in fp32, re-quantizes with fresh per-row absmax scales,
and stores codes + scales back. ``dstate == 128`` (asserted in the wrapper): one scale per state row.
"""

import torch
import triton
import triton.language as tl

from lores.kernels.state_int.quant import (
    block_scale,
    dequant_int8,
    dequant_int8_asym,
    dequant_int8_fast,
    requant_int8,
    requant_int8_asym,
    requant_int8_asym_fast,
    requant_int8_asym_sr,
    requant_int8_asym_sr_div,
    requant_int8_fast,
    requant_int8_sr,
    requant_int8_sr_div,
    sr_hash,
    sr_u1,
)

try:  # softplus helper from the same wheel the original kernel ships in
    from mamba_ssm.ops.triton.softplus import softplus
except ImportError:  # mamba_ssm absent: byte-identical local copy of mamba_ssm 2.3.1's triton>=3 branch

    @triton.jit
    def softplus(dt):
        """Byte-identical local copy of mamba_ssm's triton>=3 softplus; MUST stay log(exp(dt) + 1), not log1p."""
        return tl.math.log(tl.math.exp(dt) + 1)


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit(do_not_specialize=["sr_salt"])
def _selective_scan_update_int_kernel(
    # INT-STATE: state_ptr (fp) replaced by codes + per-row fp32 scales
    codes_ptr,
    scales_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    null_block_id,
    sr_seed_ptr,  # int32[batch] per-row seed, advanced by the caller every step (SR only)
    sr_salt,  # int32 per-LAYER salt (SR only): layers share slot ids and seq_lens, their noise must not
    batch,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # INT-STATE: codes strides (uint8 tensor [B, H, dim, dstate]) + scales strides ([B, H, dim])
    stride_codes_batch,
    stride_codes_head,
    stride_codes_dim,
    stride_scales_batch,
    stride_scales_head,
    stride_scales_dim,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    FAST: tl.constexpr,  # reciprocal codecs when fast=True (boundary codes may flip 1 step)
    SR: tl.constexpr,  # stochastic rounding floor(v/s + u), reciprocal codec (quant.py)
    ASYM: tl.constexpr,  # affine per-row grid, scales row = [scale, lo] (quant.py)
    KEY_AXIS: tl.constexpr,  # one scale per state column (dstate channel) over all dim rows; BLOCK_SIZE_M covers dim
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    out_ptrs = out_ptr + offs_m * stride_out_dim

    if HAS_STATE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        # int64: slot_id x page-stride overflows int32 in pooled caches (vLLM pages are ~5 MB).
        # null_block_id skips scheduler padding rows (vLLM pads decode batches with the reserved
        # null block 0; -1 keeps the HF pad convention).
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        if (state_batch_idx < 0) | (state_batch_idx == null_block_id):
            tl.store(out_ptrs, 0.0, mask=offs_m < dim)
            return
        codes_ptr += state_batch_idx * stride_codes_batch + pid_h * stride_codes_head  # INT-STATE
        scales_ptr += state_batch_idx * stride_scales_batch + pid_h * stride_scales_head  # INT-STATE
        sr_row0 = ((state_batch_idx * nheads + pid_h) * dim).to(tl.int32)
    else:
        codes_ptr += pid_b * stride_codes_batch + pid_h * stride_codes_head  # INT-STATE
        scales_ptr += pid_b * stride_scales_batch + pid_h * stride_scales_head  # INT-STATE
        sr_row0 = (pid_b * nheads + pid_h) * dim
    if SR:
        # hash-group index = global state row * dstate + group; unique per group, wraps harmlessly
        sr_seed = tl.load(sr_seed_ptr + pid_b) ^ (sr_salt * 1103515245)  # per-layer decorrelation (int32 odd multiplier)
        sr_rows = (sr_row0 + offs_m)[:, None] * BLOCK_SIZE_DSTATE

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head

    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim

    mask_m = offs_m < dim
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    # INT-STATE: one fp32 scale per state row (asym: [scale, lo]); key axis: per state column
    if KEY_AXIS:
        sc_ptrs = scales_ptr + offs_n * stride_scales_dim
        sc = tl.load(sc_ptrs)
        if ASYM:
            lo_ptrs = sc_ptrs + 1
            lo = tl.load(lo_ptrs)
    else:
        sc_ptrs = scales_ptr + offs_m * stride_scales_dim
        sc = tl.load(sc_ptrs, mask=mask_m, other=1.0)
        if ASYM:
            lo_ptrs = sc_ptrs + 1
            lo = tl.load(lo_ptrs, mask=mask_m, other=0.0)

    x = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    if not TIE_HDIM:
        dt = tl.load(dt_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)
    else:
        dt = tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = tl.exp(A * dt)  # scalar

    if HAS_D:
        D = tl.load(D_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    # INT-STATE: dequantize the rows' codes in registers
    mask_2d = mask_m[:, None]  # state axis is provably full (dstate == BLOCK_SIZE_DSTATE == 128)
    codes_ptrs = codes_ptr + offs_m[:, None] * stride_codes_dim + offs_n[None, :]
    u8 = tl.load(codes_ptrs, mask=mask_2d, other=0)
    if KEY_AXIS:
        if ASYM:
            state = dequant_int8_asym(u8, sc[None, :], lo[None, :])
        elif FAST:
            state = dequant_int8_fast(u8, sc[None, :], 127.0 * sc[None, :])
        else:
            state = dequant_int8(u8, sc[None, :])
    elif ASYM:
        state = dequant_int8_asym(u8, sc[:, None], lo[:, None])
    elif FAST:
        state = dequant_int8_fast(u8, sc[:, None], 127.0 * sc[:, None])
    else:
        state = dequant_int8(u8, sc[:, None])

    B = tl.load(B_ptr + offs_n * stride_B_dstate).to(tl.float32)
    C = tl.load(C_ptr + offs_n * stride_C_dstate).to(tl.float32)
    if not TIE_HDIM:
        A = tl.load(A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate, mask=mask_2d, other=0.0).to(
            tl.float32
        )
        dA2 = tl.exp(A * dt[:, None])
        dB = B[None, :] * dt[:, None]
    else:
        dA2 = dA
        dB = B * dt

    state = state * dA2 + dB * x[:, None]

    out = tl.sum(state * C[None, :], axis=1)

    # INT-STATE: requantize with fresh scales and store codes + scales
    if SR:
        hp = sr_hash(sr_rows + tl.arange(0, BLOCK_SIZE_DSTATE // 2)[None, :], sr_seed)  # one word per pair
        u1 = tl.reshape(tl.join(sr_u1(hp, 0), sr_u1(hp, 1)), (BLOCK_SIZE_M, BLOCK_SIZE_DSTATE)) + 7.62939453125e-06
    if KEY_AXIS:  # per state column over the valid dim rows; same codecs, other axis
        if ASYM:
            mn = tl.min(tl.where(mask_2d, state, float("inf")), axis=0)
            mx = tl.max(tl.where(mask_2d, state, float("-inf")), axis=0)
            new_scale = tl.maximum((mx - mn) / 255.0, 1e-12)
            if SR:
                if FAST:
                    new_codes = requant_int8_asym_sr(state, (255.0 / tl.maximum(mx - mn, 1e-12))[None, :], mn[None, :], u1)
                else:
                    new_codes = requant_int8_asym_sr_div(state, new_scale[None, :], mn[None, :], u1)
            elif FAST:
                new_codes = requant_int8_asym_fast(state, (255.0 / tl.maximum(mx - mn, 1e-12))[None, :], mn[None, :])
            else:
                new_codes = requant_int8_asym(state, new_scale[None, :], mn[None, :])
            tl.store(lo_ptrs, mn)
        else:
            absmax = tl.max(tl.abs(tl.where(mask_2d, state, 0.0)), axis=0)
            new_scale = block_scale(absmax, 127.0)
            if SR:
                if FAST:
                    new_codes = requant_int8_sr(state, (127.0 / tl.maximum(absmax, 1e-12))[None, :], u1)
                else:
                    new_codes = requant_int8_sr_div(state, new_scale[None, :], u1)
            elif FAST:
                new_codes = requant_int8_fast(state, (127.0 / tl.maximum(absmax, 1e-12))[None, :])
            else:
                new_codes = requant_int8(state, new_scale[None, :])
        tl.store(codes_ptrs, new_codes, mask=mask_2d)
        tl.store(sc_ptrs, new_scale)
    elif ASYM:
        mn = tl.min(state, axis=1)
        mx = tl.max(state, axis=1)
        new_scale = tl.maximum((mx - mn) / 255.0, 1e-12)
        if SR:
            if FAST:
                new_codes = requant_int8_asym_sr(state, (255.0 / tl.maximum(mx - mn, 1e-12))[:, None], mn[:, None], u1)
            else:  # exact division, the HF arithmetic
                new_codes = requant_int8_asym_sr_div(state, new_scale[:, None], mn[:, None], u1)
        elif FAST:
            new_codes = requant_int8_asym_fast(state, (255.0 / tl.maximum(mx - mn, 1e-12))[:, None], mn[:, None])
        else:
            new_codes = requant_int8_asym(state, new_scale[:, None], mn[:, None])
        tl.store(lo_ptrs, mn, mask=mask_m)
    else:
        absmax = tl.max(tl.abs(state), axis=1)
        new_scale = block_scale(absmax, 127.0)
        if SR:
            if FAST:
                new_codes = requant_int8_sr(state, (127.0 / tl.maximum(absmax, 1e-12))[:, None], u1)
            else:  # exact division, the HF arithmetic
                new_codes = requant_int8_sr_div(state, new_scale[:, None], u1)
        elif FAST:
            new_codes = requant_int8_fast(state, (127.0 / tl.maximum(absmax, 1e-12))[:, None])
        else:
            new_codes = requant_int8(state, new_scale[:, None])
    if not KEY_AXIS:
        tl.store(codes_ptrs, new_codes, mask=mask_2d)
        tl.store(sc_ptrs, new_scale, mask=mask_m)

    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=mask_m)


# batch -> (BLOCK_SIZE_M, num_warps), cold-L2 H100 sweep. The large-batch default (64, 4) launches only
# B*nheads programs at BM=64 -- at B=1 that is half the machine idle; B > 8 keeps (64, 4).
_TUNED_SMALL_BATCH = {1: (8, 4), 2: (8, 2), 4: (8, 2), 8: (8, 1)}


def selective_state_update_int(
    codes,
    scales,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    out=None,
    null_block_id=-1,
    sr_seed=None,
    asym=False,
    sr_salt=0,
    block_m=None,
    num_warps=None,
    fast=False,
    axis="value",
):
    """Int8-state decode step. Mirrors ``selective_state_update``'s contract with the fp ``state`` replaced by
    ``codes`` (uint8 ``[batch, nheads, dim, dstate]``, updated in place) + ``scales`` (fp32 ``[batch, nheads, dim]``,
    one per row; ``[..., 2]`` = (scale, lo) when ``asym``).

    ``sr_seed`` (int32[>= batch] device tensor, row b seeds batch row b; the caller advances each row every
    step -- vLLM passes its seq_lens buffer UNSLICED, a per-call view costs ~4 us of host time per layer)
    switches the requant to stochastic rounding; None keeps RTN. ``block_m`` / ``num_warps`` pin the launch
    config (serving table in lores.backends.vllm.state); None = the small-batch table. Returns ``out`` like the original.
    ``axis="key"``: one scale per state column (dstate channel) over all ``dim`` rows (``scales`` ``[batch, nheads,
    dstate(, 2)]``; one program covers a head's rows).
    """
    assert axis in ("value", "key"), axis
    key_axis = axis == "key"
    if asym:
        assert scales.dim() == 4 and scales.shape[-1] == 2 and scales.stride(-1) == 1, (scales.shape, scales.stride())
    else:
        assert scales.dim() == 3, scales.shape
    if sr_seed is not None:
        assert sr_seed.dtype == torch.int32 and sr_seed.shape[0] >= x.shape[0], (sr_seed.dtype, sr_seed.shape)
    has_heads = x.dim() > 2
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    _, nheads, dim, dstate = codes.shape
    assert dstate == 128, f"int state kernel requires dstate == 128 (one scale per row), got {dstate}"
    assert scales.shape[:3] == (*codes.shape[:2], dstate if key_axis else dim), (scales.shape, codes.shape, axis)
    batch = x.shape[0]
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    if out is None:
        out = torch.empty_like(x)
    else:  # caller-preallocated (vLLM passes its own buffer); kernel writes via out strides
        assert out.shape == x.shape and out.dtype == x.dtype, (out.shape, out.dtype, x.shape, x.dtype)
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)  # noqa: E731
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    BLOCK_SIZE_M, nwarps = 64, 4
    if batch <= 8:
        for b in (8, 4, 2, 1):
            if b <= batch:
                BLOCK_SIZE_M, nwarps = _TUNED_SMALL_BATCH[b]
                break
    if block_m is not None:
        BLOCK_SIZE_M = block_m
    if num_warps is not None:
        nwarps = num_warps
    if key_axis:  # a column's scale needs all of its dim rows in one program
        BLOCK_SIZE_M, nwarps = triton.next_power_of_2(dim), num_warps or 4
    tie_hdim = A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and (dt_bias is None or dt_bias.stride(-1) == 0)
    with torch.cuda.device(x.device.index):
        _selective_scan_update_int_kernel[grid](
            codes,
            scales,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            state_batch_indices,
            null_block_id,
            sr_seed,
            int(sr_salt),
            batch,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            codes.stride(0),
            codes.stride(1),
            codes.stride(2),
            scales.stride(0),
            scales.stride(1),
            scales.stride(2),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else (0, 0),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            *(D.stride(0), D.stride(1)) if D is not None else (0, 0),
            z_strides[0],
            z_strides[1],
            z_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            FAST=fast,
            SR=sr_seed is not None,
            ASYM=asym,
            KEY_AXIS=key_axis,
            num_warps=nwarps,
        )
    if not has_heads:
        out = out.squeeze(1)
    return out
