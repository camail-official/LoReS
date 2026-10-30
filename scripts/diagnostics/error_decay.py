"""Extract slow/mid/fast error-decay curves from saved single-injection traces."""

import argparse

import numpy as np
import torch

from lores.evaluation.output_store import create_run, sha256, source_path, write_json_once
from lores.evaluation.runs import complete_run

PICKS = (("slow", 0.98), ("mid", 0.85), ("fast", 0.6))


def decay_curves(trace, sequence=0):
    """Select heads by realized mean decay and normalize by injected energy."""
    t0 = trace["t_inj"]
    if not 0 <= sequence < trace["err2"].shape[0] or not 0 <= t0 < trace["err2"].shape[-1] - 1:
        raise ValueError("Invalid sequence or injection position")
    e = trace["err2"].double().numpy()[sequence, ..., t0:]
    g = trace["logdecay"].double().numpy()[sequence, ..., t0 + 1 :]
    _, heads, n = e.shape
    abar = np.exp(g.mean(-1)).reshape(-1)
    rel = (e / np.clip(e[..., :1], 1e-30, None)).reshape(-1, n)
    order = np.argsort(abar)
    ref0 = trace["ref2"].double().numpy()[sequence, ..., t0].reshape(-1)
    result = {"t0": int(t0), "heads": {}, "rel_inj": (e[..., 0].reshape(-1) / np.clip(ref0, 1e-30, None)).tolist()}
    for name, q in PICKS:
        idx = int(order[int(q * (len(order) - 1))])
        result["heads"][name] = {
            "layer": idx // heads,
            "head": idx % heads,
            "abar": float(abar[idx]),
            "surviving": rel[idx].tolist(),
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", nargs="+", required=True, help="single-injection .pt files from state_error_law")
    ap.add_argument("--sequence", type=int, default=0)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()
    if args.sequence < 0:
        ap.error("--sequence must be nonnegative")
    paths = [source_path(path) for path in args.traces]
    config = vars(args) | {"inputs": [{"path": str(path), "sha256": sha256(path)} for path in paths]}
    out = {
        "source": "state_error_law.py, single-injection value-axis RTN",
        "sequence": args.sequence,
        "picks": dict(PICKS),
        "cells": {},
    }
    for path in paths:
        trace = torch.load(path, weights_only=True, map_location="cpu")
        cell = trace["cell"]
        if cell in out["cells"]:
            raise ValueError(f"Duplicate cell: {cell}")
        if trace["bits"] != 4:
            raise ValueError(f"Expected the INT4 single-injection diagnostic: {path}")
        out["cells"][cell] = decay_curves(trace, args.sequence)
    run = create_run("diagnostics", "shared", "decay-heads", config, namespace=("state-error-derived",), root=args.output_root)
    write_json_once(run / "decay_heads.json", out)
    complete_run(run, ["decay_heads.json"])
    print(f"WROTE {run / 'decay_heads.json'}")


if __name__ == "__main__":
    main()
