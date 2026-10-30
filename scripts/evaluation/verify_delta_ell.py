"""Verify explicitly selected Delta-ell runs and collect their global medians."""

import argparse
import csv
import json
from pathlib import Path

import torch

from lores.evaluation.loss_traces import load_loss_pair, read_delta_ell_summary, validate_loss_trace
from lores.evaluation.output_store import create_run, sha256, write_json_once


def inspect(path, sequences, seqlen):
    path = Path(path).resolve()
    row = {"run": str(path)}
    if not (path / "complete.json").exists():
        return row | {"status": "incomplete"}
    metadata = json.loads((path / "run.json").read_text())
    family = metadata["family"]
    if family not in ("references", "delta_ell"):
        raise ValueError(f"Not a loss-trace run: {path}")
    receipt = json.loads((path / "complete.json").read_text())
    required = {"traces.pt"} if family == "references" else {"traces.pt", "result.json", "summary.json"}
    if not required.issubset(receipt["artifacts"]):
        raise ValueError(f"Incomplete artifact receipt: {path}")
    for name, digest in receipt["artifacts"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {path / name}")
    trace = torch.load(path / "traces.pt", map_location="cpu", weights_only=True)
    validate_loss_trace(trace)
    if (trace["expected_sequences"], trace["expected_seqlen"]) != (sequences, seqlen):
        raise ValueError(f"Unexpected requested coverage: {path}")
    row.update(
        status="verified_trace",
        model=metadata["model"],
        shape=list(trace["ce"].shape),
        ce_mean=trace["ce"].mean().item(),
        trace_sha256=sha256(path / "traces.pt"),
    )
    if (path / "summary.json").exists():
        row["dce_median"] = read_delta_ell_summary(path / "summary.json")
        summary = json.loads((path / "summary.json").read_text())
        quant, reference = [path / summary[f"{kind}_trace"] for kind in ("quant", "reference")]
        if quant.resolve() != (path / "traces.pt").resolve():
            raise ValueError(f"Summary belongs to a different run: {path}")
        q, b = load_loss_pair(quant, reference)
        delta = q["ce"] - b["ce"]
        row.update(
            mean_abs_difference=delta.abs().mean().item(),
            max_abs_difference=delta.abs().max().item(),
            first_token_differences=delta[:, 0].tolist(),
            reference=summary["reference_trace"],
        )
        result = json.loads((path / "result.json").read_text())
        settings = result["experiment_identity"]["settings"]
        clipping_allowed = settings.get("allow_state_clipping", False)
        clipping_events = result.get("sq_saturation_events", 0)
        if clipping_events and not clipping_allowed:
            raise ValueError(f"Clipping was not explicitly allowed: {path}")
        if clipping_allowed and (
            result.get("allow_state_clipping") is not True
            or trace.get("allow_state_clipping") is not True
            or trace.get("sq_saturation_events") != clipping_events
        ):
            raise ValueError(f"Clipping metadata does not match the saved trace: {path}")
        row.update(allow_state_clipping=clipping_allowed, sq_saturation_events=clipping_events)
        rotated = bool(settings.get("rk") or settings.get("rv") or settings.get("rot", "none") != "none")
        if not rotated and bool(delta[:, 0].count_nonzero()):
            raise ValueError(f"First-token loss changed before storage quantization: {path}")
        interval = settings.get("quant_every", 1)
        if not rotated and bool(delta[:, :interval].count_nonzero()):
            raise ValueError(f"Loss changed before the first storage boundary: {path}")
        if not rotated and settings["mode"] == "off" and bool(delta.count_nonzero()):
            raise ValueError(f"Off/off losses differ: {path}")
        if result.get("rollout") is not None:
            raise ValueError(f"Unexpected generation in a Delta-ell run: {path}")
        row["parity"] = result.get("parity")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--selection", type=Path, help="TSV of configuration ID and exact run directory, no header")
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()
    ids = []
    if args.selection:
        if args.runs:
            parser.error("supply run directories or --selection, not both")
        with args.selection.open() as f:
            selected = list(csv.reader(f, delimiter="\t"))
        if any(len(row) != 2 for row in selected):
            parser.error("selection rows must have two columns")
        ids = [row[0] for row in selected]
        args.runs = [Path(row[1]) for row in selected]
        if len(ids) != len(set(ids)):
            parser.error("duplicate configuration IDs")
    if not args.runs:
        parser.error("provide explicit run directories or --selection")
    rows = []
    for path in args.runs:
        try:
            row = inspect(path, args.sequences, args.seqlen)
        except Exception as error:
            row = {"run": str(path), "status": "invalid", "error": str(error)}
        rows.append(row)
        if ids:
            row["id"] = ids[len(rows) - 1]
        print(json.dumps(row), flush=True)
    if args.save_report:
        run = create_run(
            "diagnostics",
            "all-models",
            "trace-verification",
            {
                "runs": [str(path.resolve()) for path in args.runs],
                "sequences": args.sequences,
                "seqlen": args.seqlen,
                "verifier_sha256": sha256(__file__),
            },
            namespace=("delta-ell",),
            root=args.output_root,
        )
        write_json_once(run / "verification.json", {"runs": rows})
        if ids:
            write_json_once(run / "inventory.json", {"records": rows})
        print(f"VERIFICATION {run}")
    if any(row["status"] != "verified_trace" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
