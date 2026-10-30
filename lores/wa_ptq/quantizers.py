# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
# From SpinQuant (https://github.com/facebookresearch/SpinQuant): utils/quant_utils.py.

"""Weight and activation fake quantization for W/A preparation."""

import torch

from lores.utils import hadamard_utils


def get_minq_maxq(bits, sym):
    """Return the integer activation grid bounds."""
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def sym_quant(x, scale, maxq):
    """Round onto the signed integer grid."""
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    """Reconstruct signed integers using their scale."""
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    """Apply symmetric fake quantization."""
    return sym_dequant(*sym_quant(x, scale, maxq))


class STEQuantize(torch.autograd.Function):
    """Symmetric fake quantization with a straight-through gradient."""

    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    """Affine fake quantization with a straight-through gradient."""

    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(torch.nn.Module):
    """Per-token or grouped activation quantization."""

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16

    def free(self) -> None:
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)

    def configure(self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0) -> None:
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, "Clip ratio should be in (0, 1]"

    def find_params_per_token_groupwise(self, x) -> None:
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)


class ActQuantWrapper(torch.nn.Module):
    """Optional input Hadamard and activation quantization around a linear."""

    def __init__(self, module: torch.nn.Linear) -> None:
        super().__init__()
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.register_buffer("had_K", None)
        self.K = 1
        self.online_full_had = False
        self.fp32_had = False

    def extra_repr(self) -> str:
        return f"Input Quantizer Bits: {self.quantizer.bits}"

    def forward(self, x):
        x_dtype = x.dtype
        if self.online_full_had:
            x = hadamard_utils.matmul_hadU_cuda(x.float() if self.fp32_had else x, self.had_K, self.K).to(x_dtype)
        if self.quantizer.bits < 16:
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
        return self.module(x).to(x_dtype)


class WeightQuantizer(torch.nn.Module):
    """Symmetric weight quantization with optional clipping search."""

    def __init__(self, shape: int = 1) -> None:
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel: bool = False,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        weight_groupsize: int = -1,
    ) -> None:
        self.bits = bits
        self.perchannel = perchannel
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize
        self.maxq = torch.tensor(2 ** (bits - 1) - 1)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return
        self.maxq = self.maxq.to(x.device)
        shape = x.shape
        grouped = self.weight_groupsize > 0
        if grouped:
            x = x.reshape(x.shape[-2], x.shape[-1] // self.weight_groupsize, self.weight_groupsize)
        elif self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        xmax = x.abs().amax(dim=-1)
        if not grouped:
            # Preserve SpinQuant's FP32 range calculation for per-channel weights.
            xmax = torch.maximum(xmax, torch.zeros(x.shape[0], device=x.device))
        xmax = xmax.clamp(min=1e-5)
        scale = xmax / self.maxq
        if self.mse:
            best = torch.full_like(xmax, float("inf")) if grouped else torch.full([x.shape[0]], float("inf"), device=x.device)
            for i in range(int(self.maxshrink * self.grid)):
                scale1 = (1 - i / self.grid) * xmax / self.maxq
                q = sym_quant_dequant(x, scale1.unsqueeze(-1), self.maxq)
                err = (q - x).abs_().pow_(self.norm).sum(-1)
                better = err < best
                best[better] = err[better]
                scale[better] = scale1[better]

        if grouped:
            self.scale = scale.unsqueeze(-1).repeat(1, 1, self.weight_groupsize).reshape(shape)
        else:
            if not self.perchannel:
                scale = scale.repeat(shape[0])
            self.scale = scale.reshape([-1] + [1] * (len(shape) - 1))

    def fake_quantize(self, x):
        if self.bits >= 16:
            return x
        if not self.ready():
            raise RuntimeError("Find weight quantization parameters before quantizing")
        return sym_quant_dequant(x, self.scale, self.maxq).to(x.dtype)

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(module) -> None:
    """Wrap registered linears once, preserving shared module aliases."""
    wrappers = {}

    def wrap(parent):
        if isinstance(parent, ActQuantWrapper):
            return
        for name, child in list(parent._modules.items()):
            if isinstance(child, torch.nn.Linear):
                if id(child) not in wrappers:
                    wrappers[id(child)] = ActQuantWrapper(child)
                setattr(parent, name, wrappers[id(child)])
            elif child is not None:
                wrap(child)

    wrap(module)


def find_qlayers(
    module,
    layers=[torch.nn.Linear, ActQuantWrapper],
    name: str = "",
):
    """Find target modules recursively, preserving their qualified names."""
    if type(module) in [torch.nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res
