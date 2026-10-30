"""Allocate decay-ranked and matched random precision masks from saved scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lores.evaluation.loss_traces import read_delta_ell_summary
from lores.evaluation.output_store import create_run, sha256, write_json_once
from lores.evaluation.provenance import source_identity
from lores.evaluation.runs import _snapshot_sources, complete_run


def build_promotion_masks(scores, fraction, base_axis, promoted_axis, seed=0, *, budget_scope, base_bits=4, promoted_bits=6):
    """Preserve head/channel granularity and match the random control's budget."""
    if scores.ndim not in (2, 3) or not scores.numel() or not torch.isfinite(scores).all():
        raise ValueError("Expected finite nonempty decay scores [L,H] or [L,H,K]")
    if not 0 <= fraction <= 1 or not 2 <= base_bits < promoted_bits <= 8:
        raise ValueError("Require fraction in [0,1] and 2 <= base-bits < promoted-bits <= 8")
    if base_axis not in ("value", "key") or promoted_axis not in ("value", "key"):
        raise ValueError("Invalid grouping axis")
    if budget_scope not in ("global", "per-layer"):
        raise ValueError("Budget scope must be global or per-layer")
    scores = scores.detach().cpu()
    groups = 1 if budget_scope == "global" else scores.shape[0]
    flat = scores.numpy().reshape(groups, -1)
    n = round(fraction * flat.shape[1])
    orders = {"decay": np.argsort(-flat, axis=1, kind="stable")}
    # Seeded random controls.
    if budget_scope == "global":
        orders["random"] = np.random.default_rng(seed).permutation(flat.size).reshape(1, -1)
    else:
        random = torch.rand(scores.shape, generator=torch.Generator().manual_seed(seed)).numpy().reshape(flat.shape)
        orders["random"] = np.argsort(-random, axis=1, kind="stable")
    result = {}
    for name, order in orders.items():
        promote = np.zeros(flat.shape, dtype=bool)
        np.put_along_axis(promote, order[:, :n], True, axis=1)
        promote = promote.reshape(scores.shape)
        bits = np.where(promote, promoted_bits, base_bits)
        axes = np.where(promote, promoted_axis == "value", base_axis == "value")
        result[name] = {
            kind: {str(i): row.tolist() for i, row in enumerate(array)} for kind, array in (("bits", bits), ("axes", axes))
        }
    return result


def select_uniform_axes(model, checkpoint, runs, base_bits, promoted_bits):
    """Choose each precision's axis using verified, unrotated uniform runs."""
    medians, inputs, axes = {}, {}, {}
    for level, bits in (("base", base_bits), ("promoted", promoted_bits)):
        medians[bits] = {}
        for axis in ("value", "key"):
            run = runs[f"{level}_{axis}"].resolve(strict=True)
            identity = json.loads((run / "result.json").read_text())["experiment_identity"]
            settings = identity["settings"]
            dk = (axis == "value") != (model == "nemotronh")
            expected_mode = f"int{bits}_block" + ("_dk" if dk else "")
            if (
                settings.get("model", settings.get("cell")),
                settings["mode"],
                settings.get("rot", "none"),
                settings.get("rk", False),
                settings.get("rv", False),
                settings.get("n_seqs", settings.get("seqs")),
                settings["seqlen"],
                settings["seed"],
                settings["quant_every"],
                identity["inputs"],
            ) != (model, expected_mode, "none", False, False, 8, 2048, 0, 1, {}):
                raise ValueError(f"Not the required full uniform configuration: {run}")
            if identity["checkpoint"] != checkpoint:
                raise ValueError("Calibration and evaluation checkpoints differ")
            medians[bits][axis] = read_delta_ell_summary(run / "summary.json")
            inputs[f"int{bits}_{axis}"] = {"path": str(run), "summary_sha256": sha256(run / "summary.json")}
        axes[bits] = min(("value", "key"), key=medians[bits].get)
    return axes, inputs, medians


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--base-bits", type=int, default=4, choices=range(2, 9))
    parser.add_argument("--promoted-bits", type=int, default=6, choices=range(2, 9))
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-scope", required=True, choices=("global", "per-layer"))
    parser.add_argument("--base-axis", choices=("value", "key"), help="explicit axis; supply both axes or all four uniform runs")
    parser.add_argument("--promoted-axis", choices=("value", "key"))
    for level in ("base", "promoted"):
        for axis in ("value", "key"):
            parser.add_argument(f"--{level}-{axis}", type=Path, help="verified uniform delta-ell run directory")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if not 0 <= args.fraction <= 1 or args.base_bits >= args.promoted_bits:
        parser.error("require fraction in [0,1] and base-bits < promoted-bits")
    runs = {f"{level}_{axis}": getattr(args, f"{level}_{axis}") for level in ("base", "promoted") for axis in ("value", "key")}
    explicit_axes = (args.base_axis, args.promoted_axis)
    if any(explicit_axes):
        if not all(explicit_axes) or any(runs.values()):
            parser.error("supply both explicit axes, without uniform runs")
    elif not all(runs.values()):
        parser.error("supply both explicit axes or all four uniform runs")
    calibration = args.calibration.resolve(strict=True)
    stamp = json.loads((calibration / "run.json").read_text())
    model = stamp["model"]
    receipt = json.loads((calibration / "complete.json").read_text())
    if "decay_scores.pt" not in receipt["artifacts"] or any(
        sha256(calibration / name) != digest for name, digest in receipt["artifacts"].items()
    ):
        raise ValueError("Calibration artifact checksum mismatch")
    scores = torch.load(calibration / "decay_scores.pt", map_location="cpu", weights_only=True)["scores"]
    if scores.ndim != (3 if model in ("kimi", "gla_beta", "kda") else 2) or not torch.isfinite(scores).all():
        raise ValueError("Wrong decay-score granularity or non-finite scores")
    if all(explicit_axes):
        axes = {args.base_bits: args.base_axis, args.promoted_bits: args.promoted_axis}
        inputs, medians = {}, {}
    else:
        axes, inputs, medians = select_uniform_axes(
            model, stamp["config"]["checkpoint"], runs, args.base_bits, args.promoted_bits
        )
    source = source_identity()
    config = {
        "model": model,
        "seed": args.seed,
        "fraction": args.fraction,
        "base_bits": args.base_bits,
        "promoted_bits": args.promoted_bits,
        "base_axis": axes[args.base_bits],
        "promoted_axis": axes[args.promoted_bits],
        "allocation_unit": "key_channel" if scores.ndim == 3 else "head",
        "budget_scope": args.budget_scope,
        "axis_selection": "explicit" if all(explicit_axes) else "uniform_results",
        "calibration": str(calibration),
        "scores_sha256": sha256(calibration / "decay_scores.pt"),
        "score_units": "mean_log_decay",
        "uniform_inputs": inputs,
        "uniform_medians": medians,
        "source": source,
        "numpy": np.__version__,
        "torch": str(torch.__version__),
    }
    masks = build_promotion_masks(
        scores,
        args.fraction,
        axes[args.base_bits],
        axes[args.promoted_bits],
        args.seed,
        budget_scope=args.budget_scope,
        base_bits=args.base_bits,
        promoted_bits=args.promoted_bits,
    )
    run = create_run("calibration", model, "decay-promotion", config, root=args.output_root)
    _snapshot_sources(run, source)
    artifacts = []
    for ranking, allocation in masks.items():
        for kind, data in allocation.items():
            name = f"{ranking}_{kind}.json"
            write_json_once(run / name, data)
            artifacts.append(name)
    write_json_once(run / "allocation.json", {key: value for key, value in config.items() if key != "source"})
    complete_run(run, artifacts + ["allocation.json"])
    print(f"ALLOCATION {run}", flush=True)
    print(
        f"{model}: {config['base_axis']} -> {config['promoted_axis']}, "
        f"{config['allocation_unit']}, {args.budget_scope} promotion fraction {args.fraction}",
        flush=True,
    )


if __name__ == "__main__":
    main()
