#!/usr/bin/env python3
"""Residual of one INT-n RTN state quantization injected at a single decode step, against an FP32 shadow recurrence."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lores.backends.hf.loading import load_mixerzoo
from lores.evaluation.teacher_forcing import make_sequences

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class InjectOnceQuant:
    """One-shot injection probe: identity pass-through except at step ``t_inj``, where the
    stored state is value-axis INT-n RTN quantized ONCE. A clean fp32 shadow (same stashed
    ops) then measures the residual error energy per head each step — for non-delta cells
    the residual decays exactly as the head's realized ``prod a_t^2``. Also records the
    per-step per-head log-decay ``g`` (mean over dk for diagonal decay)."""

    def __init__(self, bits: int, t_inj: int):
        self.qmax = (1 << (bits - 1)) - 1
        self.t_inj = t_inj
        self.attn = None
        self.shadow = None
        self.last_t = -1
        self.err2: list[torch.Tensor] = []  # per step: [H] residual energy
        self.ref2: list[torch.Tensor] = []  # per step: [H] shadow energy
        self.logdecay: list[torch.Tensor] = []  # per step: [H]
        self.pred: list[torch.Tensor] = []  # per step: [H] decay-only predicted residual,
        # channel-exact: sum_i ||E0[i,:]||^2 exp(2 sum_s g_{s,i}) (per-channel for diagonal
        # decay, scalar collapses to ||E0||^2 prod a^2)
        self.r0 = None  # [B,H,dk] injected-error row energies
        self.cum_g = None  # [B,H] or [B,H,dk] running post-injection log-decay

    def __call__(self, S: torch.Tensor) -> torch.Tensor:
        t = self.attn._mz_step_state.t
        if self.shadow is None or t <= self.last_t:
            self.shadow = torch.zeros_like(S, dtype=torch.float32)
            self.r0 = self.cum_g = None
        self.last_t = t
        self.shadow = shadow_step(self.attn, self.shadow)
        g_full = self.attn._mz_last_ops[3]  # log-decay [B, H] or [B, H, dk] or None
        g = torch.zeros(S.shape[:2], device=S.device) if g_full is None else (g_full.mean(-1) if g_full.dim() > 2 else g_full)
        self.logdecay.append(g[0].float().cpu())
        if self.r0 is not None and g_full is not None:
            self.cum_g = self.cum_g + g_full.float()
        x = S.to(torch.float32)
        if t == self.t_inj:
            s = x.abs().amax(dim=-2, keepdim=True).clamp_min(1e-12) / self.qmax
            deq = torch.round(x / s).clamp_(-self.qmax, self.qmax) * s
            self.r0 = (deq - x).pow(2).sum(-1)  # [B,H,dk]
            self.cum_g = None if g_full is None else torch.zeros_like(g_full.float())
            x = deq
        if self.r0 is None:
            self.pred.append(torch.zeros(S.shape[1]))
        elif self.cum_g is None:
            self.pred.append(self.r0.sum(-1)[0].cpu())
        else:
            w = (2 * self.cum_g).exp()
            p = (self.r0 * w).sum(-1) if w.dim() == 3 else self.r0.sum(-1) * w
            self.pred.append(p[0].cpu())
        self.err2.append(((x - self.shadow) ** 2).sum(dim=(-1, -2))[0].cpu())
        self.ref2.append((self.shadow**2).sum(dim=(-1, -2))[0].cpu())
        return x


def shadow_step(attn, shadow: torch.Tensor) -> torch.Tensor:
    """Update an independent FP32 shadow from the current step's operators."""
    _, k32, v32, g, beta = attn._mz_last_ops
    if attn.decay == "scalar":
        shadow = shadow * g.exp()[..., None, None]
    elif attn.decay == "diagonal":
        shadow = shadow * g.exp()[..., None]
    if attn.is_delta:
        kTs = torch.einsum("bhk,bhkv->bhv", k32, shadow)
        return shadow + torch.einsum("bhk,bhv->bhkv", k32, beta.unsqueeze(-1) * (v32 - kTs))
    return shadow + torch.einsum("bhk,bhv->bhkv", k32, v32)


def run_inject(model, ids: torch.Tensor, bits: int, t_inj: int):
    from lores.backends.hf.mixer_zoo.integration import iter_mixer_zoo_layers
    from lores.backends.hf.mixer_zoo.step import install_mixer_zoo_stepping, remove_mixer_zoo_stepping, reset_mixer_zoo_state

    quants: list[InjectOnceQuant] = []
    install_mixer_zoo_stepping(model, state_quant=lambda idx: quants.append(InjectOnceQuant(bits, t_inj)) or quants[-1])
    for q, attn in zip(quants, iter_mixer_zoo_layers(model)):
        q.attn = attn
        attn._mz_stash_ops = True
    L, T = len(quants), ids.shape[1]
    err2 = ref2 = logdecay = pred = None  # allocated once H is known (first row)
    t0 = time.time()
    for r in range(len(ids)):
        reset_mixer_zoo_state(model)
        for q in quants:
            q.err2, q.ref2, q.logdecay, q.pred = [], [], [], []
            q.shadow, q.last_t, q.r0, q.cum_g = None, -1, None, None
        for t in range(T):
            model(input_ids=ids[r : r + 1, t : t + 1])
            if t % 512 == 0:
                print(f"  [inj row {r} step {t}/{T}] {time.time() - t0:.0f}s", flush=True)
        if err2 is None:
            H = quants[0].err2[0].shape[0]
            err2 = torch.zeros(len(ids), L, H, T)
            ref2 = torch.zeros(len(ids), L, H, T)
            logdecay = torch.zeros(len(ids), L, H, T)
            pred = torch.zeros(len(ids), L, H, T)
        for li, q in enumerate(quants):
            err2[r, li] = torch.stack(q.err2, dim=-1)
            ref2[r, li] = torch.stack(q.ref2, dim=-1)
            logdecay[r, li] = torch.stack(q.logdecay, dim=-1)
            pred[r, li] = torch.stack(q.pred, dim=-1)
    reset_mixer_zoo_state(model)
    remove_mixer_zoo_stepping(model)
    return err2, ref2, logdecay, pred


def main() -> None:
    torch.set_grad_enabled(False)  # shadow chains the graph across steps otherwise -> OOM
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="linear_none")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--models-root", default="models/mixerzoo_1.3B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--inject-at",
        type=int,
        required=True,
        help="quantize the stored state once at this step and run clean otherwise; saves per-(seq, layer, head) residual curves",
    )
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()
    from lores.evaluation.output_store import create_run

    out_dir = create_run(
        "diagnostics", args.cell, "state-error-law", vars(args), namespace=("state-error-law",), root=args.output_root
    )

    model, tok = load_mixerzoo(args.cell, args.models_root, args.device)
    ids = make_sequences(tok, args.seqlen, args.seqs).to(args.device)
    err2, ref2, logdecay, pred = run_inject(model, ids, args.bits, args.inject_at)
    stem = f"inj_int{args.bits}_t{args.inject_at}"
    torch.save(
        {
            "cell": args.cell,
            "bits": args.bits,
            "t_inj": args.inject_at,
            "seqs": args.seqs,
            "seqlen": args.seqlen,
            "err2": err2,
            "ref2": ref2,
            "logdecay": logdecay,
            "pred": pred,
        },
        os.path.join(out_dir, f"{stem}.pt"),
    )
    e0 = err2[..., args.inject_at]  # [seqs, L, H] injected energy
    print(
        f"{stem}: injected energy mean={float(e0.mean()):.4g} "
        f"pre-inject floor={float(err2[..., args.inject_at - 1].mean()):.4g} "
        f"final residual/injected={float((err2[..., -1] / e0.clamp_min(1e-30)).median()):.4g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
