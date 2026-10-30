"""Per-head inverse Hadamard followed by weighted RMSNorm and gating."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

__all__ = ["unrotate_rmsnorm_gated_fwd"]


@triton.autotune(
    configs=[triton.Config({"BT": bt}, num_warps=nw) for bt in (16, 32, 64) for nw in (2, 4, 8)],
    key=["D", "ACTIVATION"],
)
@triton.jit(do_not_specialize=["T"])
def _unrotate_rmsnorm_gated_kernel(
    o_ptr,
    g_ptr,
    y_ptr,
    signs_ptr,
    H_block_ptr,
    weight_ptr,
    eps,
    T,
    D: tl.constexpr,
    BS: tl.constexpr,
    STRIDE_T: tl.constexpr,
    KRON_SCALE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BT: tl.constexpr,
):
    """Unrotate and normalize one head for a tile of tokens."""
    i_chunk, i_head = tl.program_id(0), tl.program_id(1)
    head_off = i_head * D
    p_H = tl.make_block_ptr(H_block_ptr, (BS, BS), (BS, 1), (0, 0), (BS, BS), (1, 0))
    b_H = tl.load(p_H)
    s0 = tl.load(signs_ptr + tl.arange(0, BS)).to(tl.float32)
    s1 = tl.load(signs_ptr + BS + tl.arange(0, BS)).to(tl.float32)

    p_o0 = tl.make_block_ptr(o_ptr + head_off, (T, BS), (STRIDE_T, 1), (i_chunk * BT, 0), (BT, BS), (1, 0))
    p_o1 = tl.make_block_ptr(o_ptr + head_off, (T, BS), (STRIDE_T, 1), (i_chunk * BT, BS), (BT, BS), (1, 0))
    o0 = tl.load(p_o0, boundary_check=(0,), padding_option="zero").to(tl.float32)
    o1 = tl.load(p_o1, boundary_check=(0,), padding_option="zero").to(tl.float32)
    w0 = tl.dot(o0.to(b_H.dtype), b_H).to(tl.float32)
    w1 = tl.dot(o1.to(b_H.dtype), b_H).to(tl.float32)
    u0 = (w0 + w1) * KRON_SCALE * s0[None, :]
    u1 = (w0 - w1) * KRON_SCALE * s1[None, :]
    sumsq = tl.zeros((BT,), dtype=tl.float32)
    sumsq += tl.sum(u0 * u0, axis=1) + tl.sum(u1 * u1, axis=1)
    rstd = 1.0 / tl.sqrt(sumsq / D + eps)

    p_g0 = tl.make_block_ptr(g_ptr + head_off, (T, BS), (STRIDE_T, 1), (i_chunk * BT, 0), (BT, BS), (1, 0))
    p_g1 = tl.make_block_ptr(g_ptr + head_off, (T, BS), (STRIDE_T, 1), (i_chunk * BT, BS), (BT, BS), (1, 0))
    g0 = tl.load(p_g0, boundary_check=(0,), padding_option="zero").to(tl.float32)
    g1 = tl.load(p_g1, boundary_check=(0,), padding_option="zero").to(tl.float32)
    u0 = u0 * rstd[:, None]
    u1 = u1 * rstd[:, None]
    wlo = tl.load(weight_ptr + tl.arange(0, BS)).to(tl.float32)
    whi = tl.load(weight_ptr + BS + tl.arange(0, BS)).to(tl.float32)
    u0 = u0 * wlo[None, :]
    u1 = u1 * whi[None, :]
    if ACTIVATION == "swish":
        u0 = u0 * g0 * tl.sigmoid(g0)
        u1 = u1 * g1 * tl.sigmoid(g1)
    else:
        u0 = u0 * tl.sigmoid(g0)
        u1 = u1 * tl.sigmoid(g1)

    p_y0 = tl.make_block_ptr(y_ptr + head_off, (T, BS), (STRIDE_T, 1), (i_chunk * BT, 0), (BT, BS), (1, 0))
    p_y1 = tl.make_block_ptr(y_ptr + head_off, (T, BS), (STRIDE_T, 1), (i_chunk * BT, BS), (BT, BS), (1, 0))
    tl.store(p_y0, u0.to(y_ptr.dtype.element_ty), boundary_check=(0,))
    tl.store(p_y1, u1.to(y_ptr.dtype.element_ty), boundary_check=(0,))


def unrotate_rmsnorm_gated_fwd(
    o: torch.Tensor,
    g: torch.Tensor,
    signs: torch.Tensor,
    H_block: torch.Tensor,
    weight: torch.Tensor,
    *,
    n_groups: int,
    head_dim: int,
    eps: float = 1e-5,
    activation: str = "swish",
) -> torch.Tensor:
    """Unrotate each head, normalize, apply shared head weights, then gate.

    Inputs are [..., n_groups * head_dim]; weight has head_dim elements.
    The half-size Hadamard dot uses H_block's dtype; other arithmetic is FP32.
    """
    if head_dim < 32 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(f"head_dim must be a power of two >= 32, got {head_dim}")
    if o.shape != g.shape or o.shape[-1] != n_groups * head_dim:
        raise ValueError("o and g must have matching shapes [..., n_groups * head_dim]")
    if signs.numel() != head_dim:
        raise ValueError(f"signs must have {head_dim} elements")
    bs = head_dim // 2
    if tuple(H_block.shape) != (bs, bs):
        raise ValueError(f"H_block must have shape {(bs, bs)}")
    if weight.numel() != head_dim:
        raise ValueError(f"weight must have {head_dim} elements")
    if activation == "silu":
        activation = "swish"
    if activation not in ("swish", "sigmoid"):
        raise ValueError(f"unsupported activation {activation!r}")

    o_flat = o.contiguous().view(-1, n_groups * head_dim)
    g_flat = g.contiguous().view_as(o_flat)
    out = torch.empty_like(o_flat)
    T = o_flat.shape[0]

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), n_groups)

    _unrotate_rmsnorm_gated_kernel[grid](
        o_flat,
        g_flat,
        out,
        signs.contiguous(),
        H_block.contiguous(),
        weight.contiguous(),
        float(eps),
        T,
        D=head_dim,
        BS=bs,
        STRIDE_T=n_groups * head_dim,
        KRON_SCALE=1.0 / math.sqrt(2.0),
        ACTIVATION=activation,
    )
    return out.view(o.shape)
