"""Online operand rotations and inverse-rotation/norm wrappers, with eager and fused paths."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from lores.kernels.rotations.rotate import rotate_fwd
from lores.kernels.rotations.unrotate_norm import unrotate_rmsnorm_gated_fwd
from lores.rotations.matrices import factor_random_hadamard

log = logging.getLogger(__name__)

__all__ = [
    "SignedInputWrapper",
    "UnrotateWrapper",
    "fused_rotation_buffers",
    "make_fused_rot",
    "rotate_operand",
    "try_factor_and_register",
]


def rotate_operand(t: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Right-rotate the last axis in FP32, preserving operand dtype and device."""
    return torch.matmul(t.float(), R.to(t.device)).to(t.dtype).contiguous()


class UnrotateWrapper(nn.Module):
    """Apply R_v.T per head before gated RMSNorm.

    Fused mode requires CUDA, factored buffers and power-of-two head_dim >= 32;
    residual/prenorm calls are unsupported. Invalid fused inputs raise.
    Eager mode uses FP32 matrix multiplication followed by the original norm.
    """

    def __init__(
        self,
        norm: nn.Module,
        *,
        layer: nn.Module,
        rot_attr: str = "R_v",
        use_fused: bool = False,
    ) -> None:
        super().__init__()
        self.norm = norm
        self._layer_ref = [layer]  # list-wrapped so nn.Module doesn't register `layer` as a submodule
        self.rot_attr = rot_attr
        self.use_fused = use_fused

    @property
    def rotation(self) -> torch.Tensor:
        """The current ``R_v`` rotation (read fresh from the owning layer)."""
        return getattr(self._layer_ref[0], self.rot_attr)

    def _fused_buffers(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Look up ``(signs, H_block)`` on the owning layer; ``None`` if either is missing."""
        return fused_rotation_buffers(self._layer_ref[0], self.rot_attr)

    def forward(self, o: torch.Tensor, *args, **kwargs):
        """Right-apply ``R_vᵀ`` (per head) to ``o``, then run the gated RMSNorm."""
        R = self.rotation
        head_dim = R.shape[0]
        # Requested fused execution must not silently fall back.
        if self.use_fused:
            checks = {
                "cuda input": o.is_cuda,
                "head_dim power-of-two >= 32": head_dim >= 32 and (head_dim & (head_dim - 1)) == 0,
                "gate tensor arg": len(args) >= 1 and isinstance(args[0], torch.Tensor),
                "no residual/prenorm": "residual" not in kwargs and not kwargs.get("prenorm", False),
                "last dim == head_dim": o.shape[-1] == head_dim,
            }
            bad = [k for k, ok in checks.items() if not ok]
            if bad:
                raise RuntimeError(f"UnrotateWrapper(use_fused=True): precondition failed: {bad}")
            bufs = self._fused_buffers()
            if bufs is None:
                raise RuntimeError("UnrotateWrapper(use_fused=True): missing (signs, H_block) buffers")
            signs, H_block = bufs
            g = args[0]
            # Flatten heads; Qwen's [rows, head_dim] layout has one group.
            n_groups = o.shape[-2] if o.ndim >= 3 else 1
            o_flat = o.reshape(*o.shape[:-2], n_groups * head_dim) if o.ndim >= 3 else o
            g_flat = g.reshape(o_flat.shape)
            out_flat = unrotate_rmsnorm_gated_fwd(
                o_flat,
                g_flat,
                signs,
                H_block,
                self.norm.weight,
                n_groups=n_groups,
                head_dim=head_dim,
                eps=self.norm.eps,
                activation=self.norm.activation,
            )
            return out_flat.reshape(o.shape)

        # Eager fallback: separate matmul + gated norm.
        o = torch.matmul(o.float(), R.float().t()).to(o.dtype)  # per-head: o_h <- o_h @ R_v^T
        return self.norm(o, *args, **kwargs)


class SignedInputWrapper(nn.Module):
    """Apply the exported per-channel signs before the inner module's online Hadamard.

    Keep the inner/sign attributes unchanged: both contribute to state-dict keys.
    """

    def __init__(self, inner: nn.Module, sign: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("sign", sign)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.inner(x * self.sign.to(x.dtype), *args, **kwargs)


def make_fused_rot(R: torch.Tensor):
    """Build a last-axis t @ R callable for [..., heads, d] or [..., d] inputs.

    Factor R once; unsupported matrices, dimensions or devices use eager FP32 matmul.
    """
    d = R.shape[0]
    signs = hblock = None
    if d >= 32 and (d & (d - 1)) == 0:
        try:
            signs, hblock = factor_random_hadamard(R, block_dtype=torch.float16)
        except ValueError:
            pass

    def apply(t: torch.Tensor) -> torch.Tensor:
        if signs is not None and t.is_cuda and t.shape[-1] == d:
            heads = t.shape[-2] if t.ndim >= 3 else 1
            flat = t.reshape(*t.shape[:-2], heads * d) if t.ndim >= 3 else t
            out = rotate_fwd(flat, signs.to(t.device), hblock.to(t.device), num_heads=heads, head_dim=d)
            return out.view(t.shape)
        return torch.matmul(t.to(torch.float32), R.to(t.device)).to(t.dtype).contiguous()

    return apply


def fused_rotation_buffers(owner, rot_attr: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(signs, H_block)`` fused-kernel buffers stored on ``owner`` for ``rot_attr``; ``None`` if absent."""
    signs = getattr(owner, f"{rot_attr}_signs", None)
    hblock = getattr(owner, f"{rot_attr}_hblock", None)
    if signs is None or hblock is None:
        return None
    return signs, hblock


def try_factor_and_register(
    owner: nn.Module,
    *,
    attr: str,
    block_dtype: torch.dtype,
    device: torch.device | str,
    owner_name: str = "",
) -> bool:
    """Register factored Hadamard buffers, or return False for an eager fallback."""
    R = getattr(owner, attr)
    try:
        signs, H_block = factor_random_hadamard(R.data, block_dtype=block_dtype, signs_dtype=block_dtype)
    except ValueError as e:
        log.warning("fused rotation disabled for %s.%s: %s", owner_name, attr, e)
        return False
    owner.register_buffer(f"{attr}_signs", signs.to(device))
    owner.register_buffer(f"{attr}_hblock", H_block.to(device))
    return True
