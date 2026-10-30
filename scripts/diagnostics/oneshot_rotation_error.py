r"""One-step state quantization error with and without the matched within-group rotation, SR.

For every snapshot state S (layer, decode step, head) of ``state_geometry --dump-states`` and every (bits, axis):
plain Q(S), rotated R^T Q(R S) with R_k along d_k for value-axis grouping and R_v along d_v for key-axis grouping.
Reports the median of |S_hat - S|_F / |S|_F over layers, steps and heads.

    python -m scripts.diagnostics.oneshot_rotation_error --model gla_beta --snapshot gla_beta_states.pt --output-root DIR
"""

import argparse
import json

import torch

from lores.fake_quant.quantizers import quantize_state
from lores.utils.hadamard_utils import hadamard_matrix


def main():
    """Entry point."""
    from lores.evaluation.output_store import create_run

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()
    snap = torch.load(args.snapshot, map_location="cpu", mmap=True, weights_only=False)
    # Snapshots are [T, H, dv, dk]; Nemotron-H stores [.., dv, dk], the others [.., dk, dv].
    vk = args.model == "nemotronh"
    mode_of = {"value": "int{b}_block", "key": "int{b}_block_dk"} if vk else {"value": "int{b}_block_dk", "key": "int{b}_block"}
    rot_last = {"value": vk, "key": not vk}  # whether the rotated axis (d_k for value, d_v for key grouping) is the last
    configs = [(b, axis, rot) for b in (8, 6, 4) for axis in ("value", "key") for rot in ("none", "rot")]
    errors = {c: [] for c in configs}
    for key, d in sorted(snap.items(), key=lambda kv: int(kv[0])):
        S = d["S"].to(args.device).float()
        S = S if vk else S.transpose(-1, -2).contiguous()
        for b, axis, rot in configs:
            mode = mode_of[axis].format(b=b) + "_sr"
            torch.manual_seed(0)
            if rot == "none":
                S_hat = quantize_state(S, mode)
            else:
                last = rot_last[axis]
                R = hadamard_matrix(S.shape[-1 if last else -2], args.device).to(torch.float32)
                S_hat = quantize_state(S @ R if last else R.T @ S, mode)
                S_hat = S_hat @ R.T if last else R @ S_hat
            errors[b, axis, rot].append(
                ((S_hat - S).pow(2).sum((-1, -2)).sqrt() / S.pow(2).sum((-1, -2)).sqrt().clamp_min(1e-30)).cpu()
            )
        print(f"layer {key} done", flush=True)
    res = {"model": args.model, "snapshot": args.snapshot, "rounding": "SR", "cells": {}}
    for (b, axis, rot), e in errors.items():
        res["cells"][f"INT{b}-{axis}-{rot}"] = torch.stack(e).median().item()
        print(f"INT{b} {axis} {rot}: median relative error {res['cells'][f'INT{b}-{axis}-{rot}']:.4f}", flush=True)
    run = create_run(
        "diagnostics", args.model, "oneshot-rotation", vars(args), namespace=("oneshot-rotation",), root=args.output_root
    )
    (run / "oneshot.json").write_text(json.dumps(res, indent=1))
    print(f"WROTE {run / 'oneshot.json'}")


if __name__ == "__main__":
    main()
