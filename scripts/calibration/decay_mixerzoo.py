#!/usr/bin/env python3
"""Measure train-split log-decay scores for MixerZoo precision allocation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lores.evaluation.output_store import create_run
from lores.evaluation.provenance import (
    experiment_identity,
    local_checkpoint_identity,
    save_tensor_artifact,
)
from lores.evaluation.runs import _snapshot_sources, complete_run

CELLS = ("mamba2_beta", "gated_deltanet", "gla_beta", "kda")


@torch.no_grad()
def measure_decay(model, ids_cal: torch.Tensor):
    """Return mean log-decay [layer, head] or [layer, head, key channel]."""
    from lores.backends.hf.mixer_zoo.integration import iter_mixer_zoo_layers
    from lores.backends.hf.mixer_zoo.step import (
        install_mixer_zoo_stepping,
        remove_mixer_zoo_stepping,
        reset_mixer_zoo_state,
    )

    attns = list(iter_mixer_zoo_layers(model))
    sums, ns, origs = [None] * len(attns), [0] * len(attns), []
    for i, a in enumerate(attns):
        orig = a._decay

        def _wrap(fn, idx):
            def w(x):
                g = fn(x)
                if g is not None:
                    m = g.to(torch.float32).mean(dim=(0, 1)).cpu()
                    sums[idx] = m if sums[idx] is None else sums[idx] + m
                    ns[idx] += 1
                return g

            return w

        origs.append(orig)
        a._decay = _wrap(orig, i)

    try:
        install_mixer_zoo_stepping(model)
        reset_mixer_zoo_state(model)
        for t in range(ids_cal.shape[1]):
            model(input_ids=ids_cal[:, t : t + 1])
    finally:
        remove_mixer_zoo_stepping(model)
        for a, o in zip(attns, origs):
            a._decay = o
    assert all(n > 0 for n in ns), "decay capture saw nothing — is this a decayed cell?"
    return torch.stack([s / n for s, n in zip(sums, ns)])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--models-root", default="models/mixerzoo_1.3B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()
    if args.steps < 1:
        ap.error("--steps must be positive")
    from lores.backends.hf.loading import load_mixerzoo
    from lores.evaluation.data import get_wikitext2

    model, tokenizer = load_mixerzoo(args.cell, args.models_root, args.device)
    calibration = get_wikitext2(nsamples=1, seed=args.seed, seqlen=args.steps, tokenizer=tokenizer, eval_mode=False)
    ids = calibration[0][0].to(args.device)
    scores = measure_decay(model, ids)
    diagonal = model.config.decay == "diagonal"
    if scores.ndim != (3 if diagonal else 2) or not torch.isfinite(scores).all():
        raise ValueError("Captured decay does not match the model's allocation granularity")
    config = experiment_identity(args, ids, local_checkpoint_identity(Path(args.models_root) / args.cell))
    config.update(
        allocation_unit="key_channel" if diagonal else "head",
        calibration_seed=args.seed,
        calibration_tokens=args.steps,
        score_units="mean_log_decay",
        split="train",
    )
    run = create_run("calibration", args.cell, "decay-scores", config, root=args.output_root)
    _snapshot_sources(run, config["source"])
    save_tensor_artifact(run / "decay_scores.pt", {"scores": scores.float().cpu(), "token_ids": ids.cpu()})
    complete_run(run, ["decay_scores.pt"])
    print(f"WROTE {run}; allocation_unit={config['allocation_unit']}")


if __name__ == "__main__":
    main()
