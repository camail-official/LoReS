# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
# From SpinQuant (https://github.com/facebookresearch/SpinQuant): eval_utils/gptq_utils.py.

"""Layer-wise GPTQ calibration; architecture adapters supply the target ordering."""

from __future__ import annotations

import logging
import math

import torch
import tqdm

from lores.utils import utils
from lores.wa_ptq import quantizers

log = logging.getLogger("lores")


class GPTQ:
    """Accumulate a linear layer's Hessian and quantize its weights."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.columns = layer.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        # Escalate damping for near-singular calibration Hessians.
        for _ in range(8):
            try:
                H = torch.linalg.cholesky(H)
                break
            except torch._C._LinAlgError:
                H[diag, diag] += 10 * damp
                damp *= 10
        else:
            raise RuntimeError(
                f"GPTQ Hessian still not positive-definite after damp escalation (final damp={damp.item():.4g}); "
                "calibration may be degenerate."
            )
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)])

                q = self.quantizer.fake_quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in %s-bit quantized weights", self.quantizer.bits)
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


def _gptq_impl(model, dataloader, dev, args, layers, seqlen, dtype, sequential, what: str):
    """Run GPTQ capture, Hessian accumulation, and layer-by-layer replay over ``model.model.layers``."""
    root = model.model
    hidden = model.config.hidden_size

    log.info("(GPTQ/%s) calibrating on %d samples ...", what, len(dataloader))
    root.embeddings = root.embeddings.to(dev)
    layers[0] = layers[0].to(dev)
    inps = torch.zeros((len(dataloader), seqlen, hidden), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            inps[cache["i"]] = hidden_states
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            raise ValueError  # we only need the input to layer 0 — abort the rest of the forward

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module.cpu()
    root.embeddings = root.embeddings.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    outs = torch.zeros_like(inps)

    for i in tqdm.tqdm(range(len(layers)), desc=f"(GPTQ/{what}) Layers"):
        layer = layers[i].to(dev)
        full = quantizers.find_qlayers(layer, layers=[torch.nn.Linear])
        gptq = {}
        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quantizers.WeightQuantizer()
                # fasterquant groups columns itself; each column needs [rows, 1] scales.
                gptq[name].quantizer.configure(args.w_bits, perchannel=True, mse=args.w_clip)

            def add_batch(name):
                def tmp(_, inp, _out):
                    gptq[name].add_batch(inp[0].data)  # noqa: F821

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
            for j in range(len(dataloader)):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()
            for name in subset:
                gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.w_groupsize)
                gptq[name].free()
        for j in range(len(dataloader)):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer, gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    utils.cleanup_memory(verbos=True)
    log.info("(GPTQ/%s) done.", what)
