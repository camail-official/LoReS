"""Seeded random-Hadamard rotations and their fused-kernel factorization, built on SpinQuant."""

from __future__ import annotations

import math

import torch

__all__ = ["get_orthogonal_matrix", "r4_signs", "sylvester_hadamard", "factor_random_hadamard"]


def r4_signs(n: int, layer_index: int, *, dtype: torch.dtype = torch.float32, device="cpu") -> torch.Tensor:
    """R4 signs shared by weight preparation and serving: CPU draw seeded by 1000 + layer index.

    The explicit CPU device also works under vLLM's CUDA default-device context.
    A local generator keeps this checkpoint convention independent of global RNG state.
    """
    g = torch.Generator(device="cpu").manual_seed(1000 + layer_index)
    signs = torch.randint(0, 2, (n,), generator=g, device="cpu") * 2 - 1
    return signs.to(device=device, dtype=dtype)


# From SpinQuant (https://github.com/facebookresearch/SpinQuant): eval_utils/rotation_utils.py.
def get_orthogonal_matrix(size, device="cuda"):
    """Random signed Hadamard of order ``size``."""
    from lores.utils.hadamard_utils import random_hadamard_matrix

    return random_hadamard_matrix(size, device)


def sylvester_hadamard(d: int, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """Return the unnormalized Sylvester Hadamard of power-of-two order d."""
    if d < 1 or (d & (d - 1)) != 0:
        raise ValueError(f"Hadamard size must be a positive power of two, got d={d}")
    H = torch.ones((1, 1), dtype=torch.float32, device=device)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H.to(dtype)


def factor_random_hadamard(
    R: torch.Tensor,
    *,
    atol: float = 1e-3,
    block_dtype: torch.dtype = torch.float16,
    signs_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor a random Hadamard rotation into ``(signs, H_block)``.

    Returns ``[D]`` signs and ``H_(D/2) / sqrt(D/2)`` in the requested dtypes.
    Callers apply the remaining ``1/sqrt(2)`` butterfly normalization.
    Requires power-of-two ``D >= 2`` and verifies reconstruction in FP32
    within ``atol``; invalid rotations raise ValueError.
    """
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be a square 2-D matrix, got shape {tuple(R.shape)}")
    D = R.shape[0]
    if D < 2 or (D & (D - 1)) != 0:
        raise ValueError(f"Kronecker factor requires D a power of two ≥ 2, got D={D}")

    R_f32 = R.detach().to(torch.float32)
    sqrt_D = math.sqrt(D)

    # H_D[:, 0] is all ones ⇒ R[:, 0] = signs / √D.
    signs_raw = R_f32[:, 0] * sqrt_D
    signs = signs_raw.round()
    if not torch.all(signs.abs() == 1.0):
        max_dev = (signs_raw.abs() - 1.0).abs().max().item()
        raise ValueError(
            f"R[:, 0] * √D is not in {{±1}} (max deviation {max_dev:.4g}); R does not look like a random-Hadamard matrix."
        )

    H_D = sylvester_hadamard(D, dtype=torch.float32, device=R.device)
    reconstructed = torch.diag(signs) @ H_D / sqrt_D
    err = (reconstructed - R_f32).abs().max().item()
    if err > atol:
        raise ValueError(
            f"Factorization residual ‖diag(signs) @ H_D / √D - R‖∞ = {err:.4g} > atol={atol:.4g}; "
            "R is not (within tolerance) a random-Hadamard matrix."
        )

    half = D // 2
    H_block = sylvester_hadamard(half, dtype=torch.float32, device=R.device) / math.sqrt(half)
    return signs.to(signs_dtype).contiguous(), H_block.to(block_dtype).contiguous()
