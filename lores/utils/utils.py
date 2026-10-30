# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
# From SpinQuant (https://github.com/facebookresearch/SpinQuant): utils/utils.py.

import logging

import torch

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None

def _hadamard_transform_torch(u: torch.Tensor) -> torch.Tensor:
    """Unnormalized last-axis Sylvester Hadamard; width must be a power of two.

    BF16/FP16 inputs accumulate in FP32 and return their original dtype."""
    n = u.shape[-1]
    if n & (n - 1) != 0 or n < 1:
        raise ValueError(f"Hadamard transform requires a power-of-two last dim, got {n}")
    orig_dtype = u.dtype
    batch = u.shape[:-1]
    work = u.reshape(-1, n)
    work = work.to(torch.float32) if orig_dtype not in (torch.float32, torch.float64) else work.clone()
    h = 1
    while h < n:
        v = work.reshape(work.shape[0], n // (2 * h), 2, h)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        work = torch.stack([a + b, a - b], dim=2).reshape(work.shape[0], n)
        h *= 2
    return work.reshape(*batch, n).to(orig_dtype)


class HadamardTransform(torch.autograd.Function):
    """Unnormalized Hadamard: CUDA extension when installed, otherwise PyTorch."""

    @staticmethod
    def forward(ctx, u):
        if hadamard_transform is None:
            return _hadamard_transform_torch(u)
        return hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        if hadamard_transform is None:
            return _hadamard_transform_torch(grad)
        return hadamard_transform(grad)



def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect

    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(
            torch.cuda.memory_reserved(device=i)
            for i in range(torch.cuda.device_count())
        )

    memory_before = total_reserved_mem()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
