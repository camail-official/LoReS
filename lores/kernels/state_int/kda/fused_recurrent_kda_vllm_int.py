# Copied from vLLM 0.26 `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`
# (`fused_recurrent_gated_delta_rule_fwd_kernel`, the kernel `ops/kda.py::fused_recurrent_kda`
# launches with IS_KDA=True for Kimi-Linear decode; Apache-2.0, itself derived from
# flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, Yu Zhang, MIT) and adapted for
# LoReS's packed integer state cache. Edits are marked `# INT-STATE:`.
"""KDA recurrence with uint8 state codes and FP32 scales.

Codes are [slots, HV, V, K]. Value-axis scales are [slots, HV, V];
key-axis scales are [slots, HV, K] and require a whole-head tile (BV=V).
Affine grids append a final (scale, lo) axis. Readout precedes quantization.
Continuous batching only; null slot 0 is skipped. No speculative decoding.
"""

import torch
import triton
import triton.language as tl

from lores.kernels.state_int.quant import (
    _int8_tile_load,
    _int8_tile_store,
    block_scale,
    dequant_int8,
    dequant_int8_asym,
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

try:  # vLLM's exp (exp2-based) so the decay is bit-identical to the vendor kernel; fla's is the same op
    from vllm.third_party.flash_linear_attention.ops.op import exp as _vexp
except ImportError:  # pragma: no cover - HF venv
    from fla.ops.utils.op import exp as _vexp


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T", "sr_salt"])
def fused_recurrent_kda_int_kernel(  # noqa: D103 — vLLM kernel body, see module docstring
    q,
    k,
    v,
    g,
    beta,
    o,
    codes,  # INT-STATE: uint8 [slots, HV, V, K] replaces h0/ht
    scales,  # INT-STATE: fp32 [slots, HV, V] (sym) / [slots, HV, V, 2] = (scale, lo) (asym); KEY_AXIS: V -> K
    cu_seqlens,
    ssm_state_indices,
    sr_seed,  # INT-STATE: int32[N] per-sequence seed (vLLM: seq_lens), SR only
    sr_salt,  # INT-STATE: int32 per-layer salt, SR only
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_codes_slot: tl.constexpr,  # INT-STATE
    stride_scales_slot: tl.constexpr,  # INT-STATE
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SR: tl.constexpr,  # INT-STATE
    ASYM: tl.constexpr,  # INT-STATE
    FAST: tl.constexpr,  # INT-STATE
    KEY_AXIS: tl.constexpr,  # INT-STATE: scales per key channel, one per [BV, K] tile (BV == V: whole channel)
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv

    p_gk = g + (bos * HV + i_hv) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # INT-STATE: the state lives in the slot pool; load + dequantise instead of h0.
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    # Skip if state index is invalid (NULL_BLOCK_ID=0)
    if state_idx <= 0:
        return
    p_c = codes + state_idx * stride_codes_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    if KEY_AXIS:  # INT-STATE: one scale per key channel (column), broadcast down the value rows
        if ASYM:
            p_s = scales + state_idx * stride_scales_slot + ((i_hv * (V // BV) + i_v) * K + o_k) * 2
            b_sc = tl.load(p_s, mask=mask_k, other=1.0)
            b_lo = tl.load(p_s + 1, mask=mask_k, other=0.0)
            b_h = dequant_int8_asym(tl.load(p_c, mask=mask_h, other=0), b_sc[None, :], b_lo[None, :])
        else:
            p_s = scales + state_idx * stride_scales_slot + (i_hv * (V // BV) + i_v) * K + o_k
            b_sc = tl.load(p_s, mask=mask_k, other=1.0)
            b_h = dequant_int8(tl.load(p_c, mask=mask_h, other=127), b_sc[None, :])
    elif ASYM:
        p_s = scales + state_idx * stride_scales_slot + (i_hv * V + o_v) * 2
        b_sc = tl.load(p_s, mask=mask_v, other=1.0)
        b_lo = tl.load(p_s + 1, mask=mask_v, other=0.0)
        b_h = dequant_int8_asym(tl.load(p_c, mask=mask_h, other=0), b_sc[:, None], b_lo[:, None])
    else:
        p_s = scales + state_idx * stride_scales_slot + i_hv * V + o_v
        b_sc = tl.load(p_s, mask=mask_v, other=1.0)
        b_h = dequant_int8(tl.load(p_c, mask=mask_h, other=127), b_sc[:, None])
    if SR:
        sr_seed_v = tl.load(sr_seed + i_n) ^ (sr_salt * 1103515245)  # per-layer decorrelation, as the GDN copy
        sr_rows = (i_hv * V + o_v)[:, None]

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BV, BK]
        b_gk = tl.load(p_gk).to(tl.float32)
        b_h *= _vexp(b_gk[None, :])
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # INT-STATE: requantise the updated tile (per value row over all K keys, or per key channel on the
        # key axis) and store codes + scales into the slot instead of the fp32 state.
        final_state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok).to(tl.int64)
        if final_state_idx > 0:
            p_ct = codes + final_state_idx * stride_codes_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            if KEY_AXIS:  # INT-STATE: per-column (key channel) grid over all V rows; same codecs, other axis
                if SR:
                    hp = sr_hash(sr_rows * (K // 2) + tl.arange(0, K // 2)[None, :], sr_seed_v + i_t)
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
                    p_st = scales + final_state_idx * stride_scales_slot + ((i_hv * (V // BV) + i_v) * K + o_k) * 2
                    tl.store(p_st, new_scale, mask=mask_k)
                    tl.store(p_st + 1, mn, mask=mask_k)
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
                    p_st = scales + final_state_idx * stride_scales_slot + (i_hv * (V // BV) + i_v) * K + o_k
                    tl.store(p_st, new_scale, mask=mask_k)
                tl.store(p_ct, u8, mask=mask_h)
            elif ASYM:
                mn = tl.min(tl.where(mask_h, b_h, float("inf")), 1)
                mx = tl.max(tl.where(mask_h, b_h, float("-inf")), 1)
                new_scale = tl.maximum((mx - mn) / 255.0, 1e-12)
                if SR:
                    hp = sr_hash(sr_rows * (K // 2) + tl.arange(0, K // 2)[None, :], sr_seed_v + i_t)
                    u1 = tl.reshape(tl.join(sr_u1(hp, 0), sr_u1(hp, 1)), (BV, K)) + 7.62939453125e-06
                    if FAST:
                        u8 = requant_int8_asym_sr(b_h, (255.0 / tl.maximum(mx - mn, 1e-12))[:, None], mn[:, None], u1)
                    else:
                        u8 = requant_int8_asym_sr_div(b_h, new_scale[:, None], mn[:, None], u1)
                elif FAST:
                    u8 = requant_int8_asym_fast(b_h, (255.0 / tl.maximum(mx - mn, 1e-12))[:, None], mn[:, None])
                else:
                    u8 = requant_int8_asym(b_h, new_scale[:, None], mn[:, None])
                p_st = scales + final_state_idx * stride_scales_slot + (i_hv * V + o_v) * 2
                tl.store(p_st, new_scale, mask=mask_v)
                tl.store(p_st + 1, mn, mask=mask_v)
            else:
                new_scale = block_scale(tl.max(tl.abs(tl.where(mask_h, b_h, 0.0)), 1), 127.0)
                if SR:  # 16-bit lanes + half LSB (unbiased)
                    hp = sr_hash(sr_rows * (K // 2) + tl.arange(0, K // 2)[None, :], sr_seed_v + i_t)
                    u1 = tl.reshape(tl.join(sr_u1(hp, 0), sr_u1(hp, 1)), (BV, K)) + 7.62939453125e-06
                    if FAST:
                        u8 = requant_int8_sr(b_h, (1.0 / new_scale)[:, None], u1)
                    else:
                        u8 = requant_int8_sr_div(b_h, new_scale[:, None], u1)
                elif FAST:
                    u8 = requant_int8_fast(b_h, (1.0 / new_scale)[:, None])
                else:
                    u8 = requant_int8(b_h, new_scale[:, None])
                p_st = scales + final_state_idx * stride_scales_slot + i_hv * V + o_v
                tl.store(p_st, new_scale, mask=mask_v)
            tl.store(p_ct, u8, mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_kda_int(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    codes: torch.Tensor,
    scales: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    sr_seed: torch.Tensor | None = None,
    sr_salt: int = 0,
    asym: bool = False,
    fast: bool = True,
    launch: tuple[int, int, int] | None = None,
    axis: str = "value",
    decode: bool | None = None,
) -> torch.Tensor:
    """VLLM's ``fused_recurrent_kda`` decode step on a packed-int8 slot pool; returns ``o`` [1, T, HV, V].

    One-token steps (plain decode) run :func:`fused_recurrent_kda_decode_int` unless ``decode=False``.

    Same call convention as ``fused_recurrent_kda`` for the non-speculative continuous-batching
    path (``initial_state``/``inplace_final_state`` replaced by the ``codes``/``scales`` pools).
    ``g`` is the per-(token, head, key channel) log gate as produced by ``fused_kda_gate``.
    ``launch`` = (BV, num_warps, num_stages); default = vLLM's (min(V, 8), 1, 3).
    ``axis``: "value" = one scale per value row over all keys; "key" = one scale per key channel per
    [BV, K] tile (scales [slots, HV, V // BV, K(, 2)]; BV == V, the default (V, 4, 2), is one scale per whole
    key channel).
    """
    assert axis in ("value", "key"), axis
    key_axis = axis == "key"
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q, k, v, g, beta = q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous()
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    default = (V, 4, 2) if key_axis else (min(triton.next_power_of_2(V), 8), 1, 3)
    BV, num_warps, num_stages = [x if x is not None else d for x, d in zip(launch or default, default)]
    if key_axis:
        assert V % BV == 0 and triton.next_power_of_2(BV) == BV, "key axis: BV must be a power of two dividing V"
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    assert K % 2 == 0 and BK == K, "int8 SR lanes pack 2 keys per hash word; K must be a power of two"
    assert g.shape == (B, T, HV, K), (g.shape, (B, T, HV, K))  # KDA: one gate per key channel
    assert ssm_state_indices is not None, "INT-STATE: the packed pool needs slot indices"
    NS = codes.shape[0]
    assert codes.dtype == torch.uint8 and codes.shape == (NS, HV, V, K), (codes.shape, (NS, HV, V, K))
    _check_scales(scales, NS, HV, K, V, BV, key_axis, asym)
    if sr_seed is not None:
        assert sr_seed.dtype == torch.int32 and sr_seed.shape[0] >= N, (sr_seed.dtype, sr_seed.shape, N)

    one_token = (cu_seqlens is None and T == 1) or (cu_seqlens is not None and cu_seqlens.numel() - 1 == T)
    if decode is not False and one_token and ssm_state_indices.ndim == 1 and beta.ndim == v.ndim - 1:
        return fused_recurrent_kda_decode_int(
            q.reshape(N, H, K),
            k.reshape(N, H, K),
            v.reshape(N, HV, V),
            g.reshape(N, HV, K),
            beta.reshape(N, HV),
            scale,
            codes,
            scales,
            ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            sr_seed=sr_seed,
            sr_salt=sr_salt,
            asym=asym,
            fast=fast,
            launch=launch,
            axis=axis,
        ).reshape(v.shape)

    o = torch.empty_like(v)
    if ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_recurrent_kda_int_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        codes=codes,
        scales=scales,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        sr_seed=sr_seed if sr_seed is not None else ssm_state_indices,  # dummy pointer when SR is off
        sr_salt=int(sr_salt),
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_codes_slot=codes.stride(0),
        stride_scales_slot=scales.stride(0),
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        SR=sr_seed is not None,
        ASYM=asym,
        FAST=fast,
        KEY_AXIS=key_axis,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def _check_scales(scales, NS, HV, K, V, BV, key_axis, asym):
    """Scale pool layout: value axis [NS, HV, V(,2)]; key axis [NS, HV, V // BV, K(,2)], or [NS, HV, K(,2)] when BV == V."""
    width = 2 if asym else 1
    if key_axis:
        groups = V // BV
        lead = (NS, HV, K) if groups == 1 and scales.ndim == 3 + (asym and 1) else (NS, HV, groups, K)
        count = NS * HV * groups * K * width
    else:
        lead, count = (NS, HV, V), NS * HV * V * width
    assert scales.dtype == torch.float32 and tuple(scales.shape[: len(lead)]) == lead, (scales.shape, lead)
    assert scales.numel() == count and scales[0].is_contiguous(), (scales.shape, asym)  # slots may be page-strided


# ----------------------------------------------------------------------------- decode-only kernel
# vLLM has no decode-only KDA kernel (its GDN path has `fused_recurrent_gated_delta_rule_packed_decode`); Kimi
# decode runs the general varlen kernel above with T == 1. Its token loop, per-token slot reload, guarded
# store and 3-D grid cost nothing in fp32 (memory-bound) but 40% at int8. This is the same step with those
# removed: one token, one slot lookup, straight-line body. Bit-identical to the general kernel.


@triton.jit(do_not_specialize=["sr_salt"])
def fused_recurrent_kda_decode_int_kernel(  # noqa: D103
    q,
    k,
    v,
    g,
    beta,
    o,
    codes,
    scales,
    ssm_state_indices,
    sr_seed,
    sr_salt,
    scale,
    stride_codes_slot: tl.constexpr,
    stride_scales_slot: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    SR: tl.constexpr,
    ASYM: tl.constexpr,
    FAST: tl.constexpr,
    KEY_AXIS: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    if state_idx <= 0:
        return
    p_c = codes + state_idx * stride_codes_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    if KEY_AXIS:
        if ASYM:
            p_s = scales + state_idx * stride_scales_slot + ((i_hv * (V // BV) + i_v) * K + o_k) * 2
            b_sc = tl.load(p_s, mask=mask_k, other=1.0)
            b_lo = tl.load(p_s + 1, mask=mask_k, other=0.0)
            b_h = dequant_int8_asym(tl.load(p_c, mask=mask_h, other=0), b_sc[None, :], b_lo[None, :])
        else:
            p_s = scales + state_idx * stride_scales_slot + (i_hv * (V // BV) + i_v) * K + o_k
            b_sc = tl.load(p_s, mask=mask_k, other=1.0)
            b_h = dequant_int8(tl.load(p_c, mask=mask_h, other=127), b_sc[None, :])
    else:
        b_h = _int8_tile_load(
            codes, scales, state_idx, stride_codes_slot, stride_scales_slot, i_hv, o_v, o_k, mask_v, mask_h, V, K, ASYM
        )

    b_q = tl.load(q + (i_n * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(k + (i_n * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(v + (i_n * HV + i_hv) * V + o_v, mask=mask_v, other=0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale
    b_gk = tl.load(g + (i_n * HV + i_hv) * K + o_k).to(tl.float32)
    b_h *= _vexp(b_gk[None, :])
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_beta = tl.load(beta + i_n * HV + i_hv).to(tl.float32)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    sr_seed_v = 0
    if SR:
        sr_seed_v = tl.load(sr_seed + i_n) ^ (sr_salt * 1103515245)
    if KEY_AXIS:
        if SR:
            sr_rows = (i_hv * V + o_v)[:, None]
            hp = sr_hash(sr_rows * (K // 2) + tl.arange(0, K // 2)[None, :], sr_seed_v)
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
            p_st = scales + state_idx * stride_scales_slot + ((i_hv * (V // BV) + i_v) * K + o_k) * 2
            tl.store(p_st, new_scale, mask=mask_k)
            tl.store(p_st + 1, mn, mask=mask_k)
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
            p_st = scales + state_idx * stride_scales_slot + (i_hv * (V // BV) + i_v) * K + o_k
            tl.store(p_st, new_scale, mask=mask_k)
        tl.store(p_c, u8, mask=mask_h)
    else:
        _int8_tile_store(
            codes,
            scales,
            state_idx,
            stride_codes_slot,
            stride_scales_slot,
            i_hv,
            o_v,
            o_k,
            mask_v,
            mask_h,
            b_h,
            sr_seed_v,
            V,
            K,
            BV,
            ASYM,
            SR,
            FAST,
        )


def fused_recurrent_kda_decode_int(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    codes: torch.Tensor,
    scales: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
    sr_seed: torch.Tensor | None = None,
    sr_salt: int = 0,
    asym: bool = False,
    fast: bool = True,
    launch: tuple[int, int, int] | None = None,
    axis: str = "value",
) -> torch.Tensor:
    """One KDA decode token per sequence on the packed-int8 slot pool; returns ``o`` [N, HV, V].

    ``q``/``k`` [N, H, K], ``v`` [N, HV, V], ``g`` [N, HV, K] (per-key log gate), ``beta`` [N, HV],
    ``ssm_state_indices`` [N] (slot 0 = vLLM's null block, skipped). ``launch``/``axis`` as in
    :func:`fused_recurrent_kda_int`.
    """
    assert axis in ("value", "key"), axis
    key_axis = axis == "key"
    N, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    assert k.shape == (N, H, K) and g.shape == (N, HV, K) and beta.shape == (N, HV), (k.shape, g.shape, beta.shape)
    assert ssm_state_indices.ndim == 1 and ssm_state_indices.shape[0] >= N
    if scale is None:
        scale = K**-0.5
    q, k, v, g, beta = q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous()
    BK = triton.next_power_of_2(K)
    assert BK == K and K % 2 == 0, "K must be a power of two (NK == 1; int8 SR lanes pack 2 keys per hash word)"
    default = (V, 4, 2) if key_axis else (min(triton.next_power_of_2(V), 8), 1, 3)
    BV, num_warps, num_stages = [x if x is not None else d for x, d in zip(launch or default, default)]
    if key_axis:
        assert V % BV == 0 and triton.next_power_of_2(BV) == BV, "key axis: BV must be a power of two dividing V"
    NS = codes.shape[0]
    assert codes.dtype == torch.uint8 and codes.shape == (NS, HV, V, K), (codes.shape, (NS, HV, V, K))
    assert codes.stride()[1:] == (V * K, K, 1)
    _check_scales(scales, NS, HV, K, V, BV, key_axis, asym)
    if sr_seed is not None:
        assert sr_seed.dtype == torch.int32 and sr_seed.shape[0] >= N, (sr_seed.dtype, sr_seed.shape, N)
    o = torch.empty_like(v)
    fused_recurrent_kda_decode_int_kernel[(triton.cdiv(V, BV), N * HV)](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        codes=codes,
        scales=scales,
        ssm_state_indices=ssm_state_indices,
        sr_seed=sr_seed if sr_seed is not None else ssm_state_indices,
        sr_salt=int(sr_salt),
        scale=scale,
        stride_codes_slot=codes.stride(0),
        stride_scales_slot=scales.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        SR=sr_seed is not None,
        ASYM=asym,
        FAST=fast,
        KEY_AXIS=key_axis,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o
