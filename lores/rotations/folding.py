"""Architecture-independent norm fusion and orthogonal weight transformations."""

from __future__ import annotations

import torch


def fuse_rmsnorm_into(norm: torch.nn.Module, linears) -> None:
    """Fold an RMSNorm's learnable per-channel scale into the linear(s) it feeds, then set the scale to 1.

    ``RMSNorm_w(x) = (x / rms(x)) ⊙ w`` and ``linear(RMSNorm_w(x)) = ((x/rms) ⊙ w) @ Wᵀ = (x/rms) @ (W ⊙ w)ᵀ`` —
    so ``W ← W ⊙ w`` (scale each input column) makes the linear absorb the scale. (RMSNorm has no bias.)

    ``w`` is moved to each linear's device: under ``device_map="auto"`` (kimi-48B) a norm can be
    sharded away from the linears it feeds. No-op when they are co-located.
    """
    w = norm.weight.data.double()
    for lin in linears:
        lin.weight.data = (lin.weight.data.double() * w.to(lin.weight.device)).to(lin.weight.dtype)
    norm.weight.data = torch.ones_like(norm.weight.data)


def rotate_input(linear: torch.nn.Module, R1: torch.Tensor) -> None:
    """``W ← W @ R1`` — the linear now expects an R1-rotated input and returns its original output."""
    linear.weight.data = (linear.weight.data.double() @ R1.double()).to(linear.weight.dtype)


def rotate_output(linear: torch.nn.Module, R1: torch.Tensor) -> None:
    """``W ← R1ᵀ @ W`` (and the bias) — the linear's output now lands in the R1 basis."""
    linear.weight.data = (R1.double().t() @ linear.weight.data.double()).to(linear.weight.dtype)
    if getattr(linear, "bias", None) is not None:
        linear.bias.data = (R1.double().t() @ linear.bias.data.double()).to(linear.bias.dtype)


def fold_hadamard_input(down_proj: torch.nn.Module) -> None:
    """Fold an inverse input Hadamard into a linear projection: ``R4`` on a down-projection
    or ``R_o`` on a recurrent output projection. Both use :func:`apply_exact_had_to_linear`
    with ``had_dim=-1, output=False`` and need a matching online activation transform.
    ``down_proj.in_features`` must be ``K·2^m`` for a Hadamard order ``K`` (``get_hadK`` raises otherwise — e.g.
    ``intermediate_size = 5632 = 44·128`` → ``get_hadK`` returns ``(H₄₄, K=44)``).
    """
    from lores.utils.hadamard_utils import apply_exact_had_to_linear

    apply_exact_had_to_linear(down_proj, had_dim=-1, output=False)
