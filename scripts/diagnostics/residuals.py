"""Residual-input kurtosis and per-layer activation dumps for MixerZoo figures."""

import argparse

from lores.evaluation.output_store import create_run, sha256, write_json_once
from lores.models.registry import MIXER_ZOO_CELLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=MIXER_ZOO_CELLS)
    parser.add_argument("--models-root", default="models/mixerzoo_1.3B")
    parser.add_argument("--dump-activations", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    import numpy as np
    import torch

    from lores.backends.hf.loading import load_mixerzoo
    from lores.evaluation.data import get_wikitext2

    config = vars(args) | {
        "seqlen": 2048,
        "samples": 6,
        "stat_captures": 48,
        "dump_tokens": 128,
        "producer_sha256": sha256(__file__),
    }
    run = create_run("diagnostics", args.cell, "residuals", config, namespace=("residuals",), root=args.output_root)
    print(f"OUTPUT {run}", flush=True)
    model, tokenizer = load_mixerzoo(args.cell, args.models_root, device=args.device)
    calib = get_wikitext2(tokenizer=tokenizer, nsamples=6, seed=args.seed, seqlen=2048)
    captures, snapshots = [], {}

    def capture(module, inputs, *, layer):
        value = inputs[0].detach()
        # Kurtosis pools the first 48 sequence/layer captures, in forward order.
        if len(captures) < 48:
            captures.append(value.cpu())
        if args.dump_activations and layer not in snapshots:
            snapshots[layer] = value[0, :128].float().abs().cpu()

    from functools import partial

    hooks = [block.attn.q_proj.register_forward_pre_hook(partial(capture, layer=i)) for i, block in enumerate(model.model.layers)]
    try:
        with torch.no_grad():
            for ids, _ in calib:
                model(ids.to(args.device))
    finally:
        for hook in hooks:
            hook.remove()
    value = torch.cat(captures, dim=0).to(args.device).float()
    kurtosis = round((((value - value.mean()) / (value.std() + 1e-9)) ** 4).mean().item(), 1)
    write_json_once(run / "statistics.json", {args.cell: {"resid_in": {"kurt": kurtosis}}})
    artifacts = {"statistics.json": sha256(run / "statistics.json")}
    if args.dump_activations:
        matrix = np.stack([snapshots[i].numpy() for i in range(len(model.model.layers))])
        with (run / f"{args.cell}.npz").open("xb") as output:
            np.savez_compressed(output, M=matrix, cell=args.cell)
        artifacts[f"{args.cell}.npz"] = sha256(run / f"{args.cell}.npz")
    write_json_once(run / "complete.json", {"artifacts": artifacts})
    print(f"{args.cell}: Pearson kurtosis={kurtosis}", flush=True)


if __name__ == "__main__":
    main()
