"""Per-head Hadamard rotation: row vectors map to (x * signs) @ H_D / sqrt(D)."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

__all__ = ["rotate_fwd"]


@triton.autotune(
    configs=[triton.Config({"BT": bt}, num_warps=nw) for bt in (16, 32, 64) for nw in (2, 4, 8)],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def _rotate_kernel(
    y_ptr,
    out_ptr,
    signs_ptr,
    H_block_ptr,
    T,
    D: tl.constexpr,
    BS: tl.constexpr,  # D // 2
    STRIDE_T: tl.constexpr,  # num_heads * head_dim (= row stride in elements)
    KRON_SCALE: tl.constexpr,  # 1/√2 (the residual half of 1/√D after H_block already has 1/√(D/2))
    BT: tl.constexpr,
):
    """One program handles ``[BT, D]`` for one (T-chunk, head)."""
    i_chunk = tl.program_id(0)
    i_h = tl.program_id(1)

    head_off = i_h * D

    # Load the two half-head tiles.
    p0 = tl.make_block_ptr(
        y_ptr + head_off,
        (T, BS),
        (STRIDE_T, 1),
        (i_chunk * BT, 0),
        (BT, BS),
        (1, 0),
    )
    p1 = tl.make_block_ptr(
        y_ptr + head_off,
        (T, BS),
        (STRIDE_T, 1),
        (i_chunk * BT, BS),
        (BT, BS),
        (1, 0),
    )
    b0 = tl.load(p0, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b1 = tl.load(p1, boundary_check=(0,), padding_option="zero").to(tl.float32)

    # H_D = H_2 ⊗ H_(D/2).
    s0 = tl.load(signs_ptr + tl.arange(0, BS)).to(tl.float32)
    s1 = tl.load(signs_ptr + BS + tl.arange(0, BS)).to(tl.float32)
    b0 = b0 * s0[None, :]
    b1 = b1 * s1[None, :]

    p_H = tl.make_block_ptr(H_block_ptr, (BS, BS), (BS, 1), (0, 0), (BS, BS), (1, 0))
    b_H = tl.load(p_H)

    w0 = tl.dot(b0.to(b_H.dtype), b_H).to(tl.float32)  # (b0 * s0) @ H_{D/2} / √(D/2)
    w1 = tl.dot(b1.to(b_H.dtype), b_H).to(tl.float32)

    y0 = (w0 + w1) * KRON_SCALE  # = (y * signs) @ H_D / √D , lower half
    y1 = (w0 - w1) * KRON_SCALE  # = (y * signs) @ H_D / √D , upper half

    o0 = tl.make_block_ptr(
        out_ptr + head_off,
        (T, BS),
        (STRIDE_T, 1),
        (i_chunk * BT, 0),
        (BT, BS),
        (1, 0),
    )
    o1 = tl.make_block_ptr(
        out_ptr + head_off,
        (T, BS),
        (STRIDE_T, 1),
        (i_chunk * BT, BS),
        (BT, BS),
        (1, 0),
    )
    tl.store(o0, y0.to(out_ptr.dtype.element_ty), boundary_check=(0,))
    tl.store(o1, y1.to(out_ptr.dtype.element_ty), boundary_check=(0,))


def rotate_fwd(
    y: torch.Tensor,
    signs: torch.Tensor,
    H_block: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Rotate each head; y is [..., num_heads * head_dim], head_dim a power of two >= 32.

    signs has length head_dim; H_block is the normalized half-size Hadamard.
    Arithmetic is FP32 except the tensor-core dot in H_block's dtype.
    Output preserves y's shape and dtype.
    """
    if head_dim < 32 or (head_dim & (head_dim - 1)) != 0:
        # tl.dot requires K ≥ 16 ⇒ BS = head_dim // 2 ≥ 16 ⇒ head_dim ≥ 32. Realistic GDN / KDA /
        # Mamba-2 head dims are ≥ 64; caller should fall back to the eager path below that.
        raise ValueError(f"head_dim must be a power of two ≥ 32, got {head_dim}")
    if y.shape[-1] != num_heads * head_dim:
        raise ValueError(f"y.shape[-1] ({y.shape[-1]}) != num_heads * head_dim ({num_heads * head_dim})")
    if signs.numel() != head_dim:
        raise ValueError(f"signs must have {head_dim} elements, got {signs.numel()}")
    bs = head_dim // 2
    if tuple(H_block.shape) != (bs, bs):
        raise ValueError(f"H_block must be [{bs}, {bs}], got {tuple(H_block.shape)}")

    out_shape = y.shape
    y_flat = y.contiguous().view(-1, num_heads, head_dim)
    T_flat = y_flat.shape[0]
    out_flat = torch.empty_like(y_flat)

    def grid(meta):
        return (triton.cdiv(T_flat, meta["BT"]), num_heads)

    _rotate_kernel[grid](
        y_flat,
        out_flat,
        signs.contiguous(),
        H_block.contiguous(),
        T_flat,
        D=head_dim,
        BS=bs,
        STRIDE_T=num_heads * head_dim,
        KRON_SCALE=1.0 / math.sqrt(2.0),
    )
    return out_flat.view(out_shape)
