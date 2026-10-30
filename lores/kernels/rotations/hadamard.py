"""Compile-compatible FP32 CUDA Hadamard for online W/A rotations."""

import torch


@torch.library.custom_op("lores::hadamard", mutates_args=(), device_types="cuda")
def hadamard(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized FP32 Sylvester transform using Dao's CUDA implementation."""
    from fast_hadamard_transform import hadamard_transform

    n = x.shape[-1]
    assert x.dtype == torch.float32, "Online Hadamard requires FP32 inputs"
    assert 0 < n <= 32768 and n & (n - 1) == 0, "Hadamard block width must be a power of two <= 32768"
    return hadamard_transform(x).contiguous()


@hadamard.register_fake
def _hadamard_fake(x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
