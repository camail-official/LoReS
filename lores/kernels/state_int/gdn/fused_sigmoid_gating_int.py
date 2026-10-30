# Copied from vLLM 0.26 `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py`
# (Apache-2.0; itself derived from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang,
# Yu Zhang, MIT) and adapted for LoReS's packed integer state cache. Edits are marked
# `# INT-STATE:`. Serves the vLLM Qwen3.5 GDN decode path.
"""``fused_sigmoid_gating_delta_rule_update`` with the state held in HBM as packed int8.

Same kernel structure as vLLM's: one program per (value tile, sequence x value head), the
sigmoid gate (softplus(a + dt_bias) * -exp(A_log)) and beta = sigmoid(b) computed in-kernel,
q/k L2-normalised in-kernel, T tokens per sequence processed in registers. The only differences
are at the state boundary: the fp32 tile is dequantised from uint8 codes + one fp32 scale per
value channel (spanning all K keys) at load, and requantised (RTN or stochastic rounding,
symmetric or affine grid, fast or exact codec) at every token before the store.

int8 only. Continuous batching only (``ssm_state_indices`` required; the null block 0 is skipped
exactly as in vLLM). No speculative decoding (``num_accepted_tokens`` must be None).
"""

import torch
import triton
import triton.language as tl

from lores.kernels.state_int.quant import _int8_tile_load, _int8_tile_load_key, _int8_tile_store, _int8_tile_store_key


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T", "sr_salt"])
def fused_sigmoid_gating_delta_rule_update_int_kernel(  # noqa: D103 — vLLM kernel body, see module docstring
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    codes,  # INT-STATE: uint8 [slots, HV, V, K] replaces h0/ht
    scales,  # INT-STATE: fp32 [slots, HV, V] (sym) or [slots, HV, V, 2] = (scale, lo) (asym)
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
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SR: tl.constexpr,  # INT-STATE: stochastic rounding
    ASYM: tl.constexpr,  # INT-STATE: affine per-row grid
    FAST: tl.constexpr,  # INT-STATE: reciprocal codec (else exact division)
    KEY_AXIS: tl.constexpr,  # INT-STATE: one scale per key channel of the [BV, K] tile (BV == V: whole channel)
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

    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # INT-STATE: the state lives in the slot pool; load + dequantise instead of h0.
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    # Skip if state index is invalid (NULL_BLOCK_ID=0)
    if state_idx <= 0:
        return
    if KEY_AXIS:
        b_h = _int8_tile_load_key(
            codes, scales, state_idx, stride_codes_slot, stride_scales_slot, i_hv, i_v, o_v, o_k, mask_k, mask_h, V, K, BV, ASYM
        )
    else:
        b_h = _int8_tile_load(
            codes, scales, state_idx, stride_codes_slot, stride_scales_slot, i_hv, o_v, o_k, mask_v, mask_h, V, K, ASYM
        )
    sr_seed_v = 0
    if SR:
        sr_seed_v = tl.load(sr_seed + i_n) ^ (sr_salt * 1103515245)  # per-layer decorrelation

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x

        # compute beta_output = sigmoid(b)
        b_beta = tl.sigmoid(b_b.to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # [BV, BK]
        b_h *= tl.exp(b_g)
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # INT-STATE: requantise the updated tile (one scale per value row over all K keys, or per key
        # channel on the key axis) and store codes + scales into the slot instead of the fp32 state.
        final_state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok).to(tl.int64)
        if final_state_idx > 0:
            if KEY_AXIS:
                _int8_tile_store_key(
                    codes,
                    scales,
                    final_state_idx,
                    stride_codes_slot,
                    stride_scales_slot,
                    i_hv,
                    i_v,
                    o_v,
                    o_k,
                    mask_k,
                    mask_h,
                    b_h,
                    sr_seed_v + i_t,
                    V,
                    K,
                    BV,
                    ASYM,
                    SR,
                    FAST,
                )
            else:
                _int8_tile_store(
                    codes,
                    scales,
                    final_state_idx,
                    stride_codes_slot,
                    stride_scales_slot,
                    i_hv,
                    o_v,
                    o_k,
                    mask_v,
                    mask_h,
                    b_h,
                    sr_seed_v + i_t,
                    V,
                    K,
                    BV,
                    ASYM,
                    SR,
                    FAST,
                )

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV


def fused_sigmoid_gating_delta_rule_update_int(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    sr_seed: torch.Tensor | None = None,
    sr_salt: int = 0,
    asym: bool = False,
    fast: bool = True,
    axis: str = "value",
):
    """VLLM's fused sigmoid-gating delta-rule decode step on a packed-int8 slot pool.

    Same call convention as ``fused_sigmoid_gating_delta_rule_update`` for the non-speculative
    continuous-batching path (``initial_state``/``inplace_final_state`` replaced by the
    ``codes``/``scales`` pools; ``num_accepted_tokens`` unsupported). Returns ``o``.
    ``axis="key"``: one scale per key channel over all V value rows (scales [slots, HV, K(, 2)]; whole-head tile).
    """
    assert axis in ("value", "key"), axis
    key_axis = axis == "key"
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), (V if key_axis else min(triton.next_power_of_2(V), 32))
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    assert K % 2 == 0 and BK == K, "int8 SR lanes pack 2 keys per hash word; K must be a power of two"

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    assert ssm_state_indices is not None, "INT-STATE: the packed pool needs slot indices"
    NS = codes.shape[0]
    assert codes.dtype == torch.uint8 and codes.shape == (NS, HV, V, K), (codes.shape, (NS, HV, V, K))
    _check_scales(scales, NS, HV, V, K, key_axis, asym)
    assert codes.stride()[1:] == (V * K, K, 1)
    if sr_seed is not None:
        assert sr_seed.dtype == torch.int32 and sr_seed.shape[0] >= N, (sr_seed.dtype, sr_seed.shape, N)

    o = q.new_empty(NK, *v.shape)
    if ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_int_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
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
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        SR=sr_seed is not None,
        ASYM=asym,
        FAST=fast,
        KEY_AXIS=key_axis,
        num_warps=4,
        num_stages=3,
    )
    return o.squeeze(0)


def _check_scales(scales, NS, HV, V, K, key_axis, asym):
    """Scale pool: value axis [NS, HV, V(, 2)]; key axis [NS, HV, K(, 2)] (one scale per key channel, whole head)."""
    width = 2 if asym else 1
    lead = (NS, HV, K) if key_axis else (NS, HV, V)
    assert scales.dtype == torch.float32 and tuple(scales.shape[:3]) == lead, (scales.shape, lead)
    assert scales.numel() == NS * HV * lead[2] * width and scales[0].is_contiguous(), (scales.shape, asym)


# ---------------------------------------------------------------------------------------------
# Copied from vLLM 0.26 `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`
# (`fused_recurrent_gated_delta_rule_packed_decode`): the decode-only fast path that reads the
# packed [q | k | v] buffer straight from the conv. Same INT-STATE edits as above.
try:  # vLLM's exp (exp2-based) so the decay is bit-identical to the vendor kernel; fla's is the same op
    from vllm.third_party.flash_linear_attention.ops.op import exp as _vexp
except ImportError:  # pragma: no cover - HF venv
    from fla.ops.utils.op import exp as _vexp


@triton.jit(do_not_specialize=["sr_salt"])
def fused_recurrent_gated_delta_rule_packed_decode_int_kernel(  # noqa: D103 — vLLM kernel body, see module docstring
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    codes,  # INT-STATE: uint8 [slots, HV, V, K] replaces h0/ht
    scales,  # INT-STATE: fp32 [slots, HV, V] (sym) or [slots, HV, V, 2] = (scale, lo) (asym)
    ssm_state_indices,
    sr_seed,  # INT-STATE: int32[B] per-sequence seed (vLLM: seq_lens), SR only
    sr_salt,  # INT-STATE: int32 per-layer salt, SR only
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_codes_slot: tl.constexpr,  # INT-STATE
    stride_scales_slot: tl.constexpr,  # INT-STATE
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    SR: tl.constexpr,  # INT-STATE
    ASYM: tl.constexpr,  # INT-STATE
    FAST: tl.constexpr,  # INT-STATE
    KEY_AXIS: tl.constexpr,  # INT-STATE
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
    p_o = o + (i_n * HV + i_hv) * V + o_v

    # Skip if state index is invalid (NULL_BLOCK_ID=0)
    if state_idx <= 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    # INT-STATE: load + dequantise the slot instead of h0
    if KEY_AXIS:
        b_h = _int8_tile_load_key(
            codes, scales, state_idx, stride_codes_slot, stride_scales_slot, i_hv, i_v, o_v, o_k, mask_k, mask_h, V, K, BV, ASYM
        )
    else:
        b_h = _int8_tile_load(
            codes, scales, state_idx, stride_codes_slot, stride_scales_slot, i_hv, o_v, o_k, mask_v, mask_h, V, K, ASYM
        )

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h *= _vexp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    # INT-STATE: requantise (one scale per value row over all K keys, or per key channel) and store codes + scales
    sr_seed_v = 0
    if SR:
        sr_seed_v = tl.load(sr_seed + i_n) ^ (sr_salt * 1103515245)
    if KEY_AXIS:
        _int8_tile_store_key(
            codes,
            scales,
            state_idx,
            stride_codes_slot,
            stride_scales_slot,
            i_hv,
            i_v,
            o_v,
            o_k,
            mask_k,
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
        return
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


def fused_recurrent_gated_delta_rule_packed_decode_int(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    codes: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    sr_seed: torch.Tensor | None = None,
    sr_salt: int = 0,
    asym: bool = False,
    fast: bool = True,
    launch: tuple[int, int, int] | None = None,
    axis: str = "value",
) -> torch.Tensor:
    """VLLM's packed decode-only GDN step on the packed-int8 slot pool; writes ``out`` [B, 1, HV, V].

    ``launch`` = (BV, num_warps, num_stages); default = vLLM's (min(V, 32), 1, 3), key axis (V, 4, 2).
    ``axis`` as in :func:`fused_sigmoid_gating_delta_rule_update_int`.
    """
    assert axis in ("value", "key"), axis
    key_axis = axis == "key"
    assert mixed_qkv.ndim == 2 and mixed_qkv.stride(-1) == 1, mixed_qkv.shape
    assert a.ndim == 2 and b.ndim == 2 and a.stride(-1) == 1 and b.stride(-1) == 1
    assert ssm_state_indices.ndim == 1 and out.is_contiguous()
    B = mixed_qkv.shape[0]
    NS, HV, V, K = codes.shape
    assert a.shape == (B, HV) and b.shape == (B, HV) and ssm_state_indices.shape[0] == B
    assert codes.dtype == torch.uint8 and codes.stride()[1:] == (V * K, K, 1)
    _check_scales(scales, NS, HV, V, K, key_axis, asym)
    assert out.shape == (B, 1, HV, V), out.shape
    qk_dim = mixed_qkv.shape[1] - HV * V
    H = (qk_dim // 2) // K
    assert H > 0 and HV % H == 0 and (qk_dim // 2) % K == 0, (mixed_qkv.shape, HV, V, K)
    BK = triton.next_power_of_2(K)
    assert triton.cdiv(K, BK) == 1 and K % 2 == 0, "NK == 1; int8 SR lanes pack 2 keys per hash word"
    BV, num_warps, num_stages = launch or ((V, 4, 2) if key_axis else (min(triton.next_power_of_2(V), 32), 1, 3))
    assert not key_axis or BV == V, "key axis: one scale per whole key channel needs BV == V"
    if sr_seed is not None:
        assert sr_seed.dtype == torch.int32 and sr_seed.shape[0] >= B, (sr_seed.dtype, sr_seed.shape, B)
    grid = (triton.cdiv(V, BV), B * HV)
    fused_recurrent_gated_delta_rule_packed_decode_int_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        codes=codes,
        scales=scales,
        ssm_state_indices=ssm_state_indices,
        sr_seed=sr_seed if sr_seed is not None else ssm_state_indices,
        sr_salt=int(sr_salt),
        scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_codes_slot=codes.stride(0),
        stride_scales_slot=scales.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        SR=sr_seed is not None,
        ASYM=asym,
        FAST=fast,
        KEY_AXIS=key_axis,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
