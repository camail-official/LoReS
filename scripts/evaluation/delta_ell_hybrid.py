"""Save teacher-forced loss traces against a verified, shared FP32-state reference.

Both arms use FP32 recurrent state from the first store; weights/activations remain
BF16. Reset quantizers per sequence and begin with ``--prefill 1``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lores.backends.hf.cache import FP32_STATE_PROTOCOL
from lores.evaluation.loading import DELTA_ELL_MODELS as REGISTRY
from lores.evaluation.state_quant import bind_mask_quantizers
from lores.evaluation.teacher_forcing import decode_per_position, make_sequences

BIN_EDGES = [0, 256, 512, 1024, 2048, 4096]


def check_baseline_stamp(b: dict, args, n_seqs: int, seqs_sha: str, *, checkpoint_identity: dict) -> None:
    """Reject references with different checkpoints, tokens, or evaluation settings."""
    for k, v in (
        ("model", args.model),
        ("seqlen", args.seqlen),
        ("prefill", args.prefill),
        ("n_seqs", n_seqs),
        ("seed", args.seed),
        ("seqs_sha", seqs_sha),
        ("checkpoint_identity", checkpoint_identity),
        ("state_protocol", FP32_STATE_PROTOCOL),
    ):
        actual = b.get(k)
        if k == "state_protocol" and isinstance(actual, dict):
            actual = {key: value for key, value in actual.items() if key != "version"}
        if actual != v:
            raise ValueError(f"baseline/protocol mismatch on {k!r}: baseline {actual!r} != cell {v!r}")


def cell_stem(args):
    """Human-readable cell label and stem; complete numerical identity is recorded separately."""
    mixtag = f"_mixhead-{Path(args.mixhead).stem}" if args.mixhead else ""
    tags = (
        mixtag
        + (f"_headbits-{Path(args.headbits).stem}" if args.headbits else "")
        + (f"_bs{args.block_size}" if args.block_size != 128 else "")
        + ("_oracle" if args.oracle_scale else "")
    )
    return f"{args.model}_{args.rot}_{args.mode}{tags}", args.mode + tags


def bucket(x, edges, n_dec):
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], min(edges[i + 1], n_dec)
        if lo >= n_dec:
            out.append({"lo": edges[i], "hi": edges[i + 1], "mean": None})
        else:
            out.append({"lo": edges[i], "hi": edges[i + 1], "mean": float(x[:, lo:hi].float().mean())})
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--revision", help="explicit checkpoint revision")
    ap.add_argument("--rot", default="none")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--prefill", type=int, default=1, help="prefill tokens; default 1 quantizes the first recurrent store")
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--allow-state-clipping",
        action="store_true",
        help="accept the 1e30 state guard and record its event count; complete finite FP32 loss traces are still required",
    )
    ap.add_argument(
        "--headbits",
        default=None,
        metavar="BITS_JSON",
        help="per-(layer, head) BIT allocation json {layer: [bits]} (HeadBitsQuant at the "
        "mode's grouping) — the retention-promotion arm for hybrids",
    )
    ap.add_argument(
        "--mixhead", default=None, metavar="MASK_JSON", help="axis mask: true = value, false = key; overrides mode grouping"
    )
    ap.add_argument("--block-size", type=int, default=128, help="scale-group length along the mode's grouping axis")
    ap.add_argument(
        "--oracle-scale",
        action="store_true",
        help="scale every group from an FP32 copy of the state advanced with the same inputs (diagnostic); "
        "plain integer block modes only",
    )
    ap.add_argument("--output-root", default=None, help="external raw-output store")
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="run the vendor-shipped baseline (--rot none --mode off), write "
        "ce/am + protocol stamp to the baseline file, and exit",
    )
    ap.add_argument(
        "--baseline-file", default=None, help="explicit verified reference traces.pt; required except with --write-baseline"
    )
    return ap


def main():
    args = build_parser().parse_args()
    from lores.evaluation.output_store import source_path

    for name in ("mixhead", "headbits", "baseline_file"):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, str(source_path(value)))
    n_flags = sum(bool(x) for x in (args.headbits, args.mixhead))
    if args.oracle_scale and n_flags:
        raise ValueError("--oracle-scale takes a plain integer block mode")

    if args.write_baseline:
        if args.baseline_file or args.rot != "none" or args.mode != "off" or n_flags:
            raise ValueError("--write-baseline requires --rot none --mode off and no baseline/mask flags")
        bfile = None  # the new reference run's traces.pt, set in _evaluate
    else:
        if not args.baseline_file:
            raise ValueError("An explicit --baseline-file is required; generate an FP32 reference first")
        bfile = Path(args.baseline_file).resolve(strict=True)

    build, rots = REGISTRY[args.model]
    assert args.rot in rots, f"{args.model}: rot must be one of {rots}"
    tok, model, set_mode = build(args.rot, revision=args.revision, block_size=args.block_size)
    seqs = make_sequences(tok, args.seqlen, args.n_seqs)
    if tuple(seqs.shape) != (args.n_seqs, args.seqlen):
        raise ValueError("Dataset did not provide the full requested sequence coverage")
    from lores.evaluation.provenance import (
        environment_identity,
        experiment_identity,
        reference_identity,
        source_identity,
    )

    source, environment = source_identity(), environment_identity()
    identity = experiment_identity(
        args,
        seqs,
        model._lq_checkpoint_identity,
        source=source,
        environment=environment,
        reference=None if args.write_baseline else bfile,
    )
    ref_identity = reference_identity(seqs, args.prefill, model._lq_checkpoint_identity, source, environment)
    from lores.evaluation.runs import evaluation_run

    _, label = cell_stem(args)
    family = "references" if args.write_baseline else "delta_ell"
    with evaluation_run(family, args.model, label, identity, root=args.output_root) as run:
        _evaluate(args, model, set_mode, seqs, identity, ref_identity, bfile, run)


def _evaluate(args, model, set_mode, seqs, identity, ref_identity, bfile, run):
    from lores.evaluation.loss_traces import make_loss_trace, save_loss_trace, summarize_delta_ell, validate_loss_trace
    from lores.evaluation.output_store import write_json_once
    from lores.evaluation.runs import complete_run

    n_dec = args.seqlen - args.prefill
    edges = [e for e in BIN_EDGES if e < n_dec] + [n_dec]
    print(f"{args.model} rot={args.rot} mode={args.mode} seqs={seqs.shape[0]} n_dec={n_dec}", flush=True)

    def set_mode_bind(mode):
        set_mode(mode)
        bind_mask_quantizers(model, args, mode)

    import hashlib

    seqs_sha = hashlib.sha1(seqs.cpu().numpy().tobytes()).hexdigest()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if args.write_baseline:
        bfile = run / "traces.pt"
        assert args.rot == "none" and args.mode == "off", "--write-baseline is the vendor-shipped config: --rot none --mode off"
        set_mode_bind("off")
        off_ce, off_am = decode_per_position(model, seqs, args.prefill, args.seed)
        trace = make_loss_trace(
            off_ce,
            seqs,
            prefill=args.prefill,
            expected_sequences=args.n_seqs,
            expected_seqlen=args.seqlen,
            metadata={
                "am": off_am,
                "model": args.model,
                "seqlen": args.seqlen,
                "prefill": args.prefill,
                "n_seqs": int(seqs.shape[0]),
                "seed": args.seed,
                "seqs_sha": seqs_sha,
                "gpu": gpu,
                "checkpoint_identity": model._lq_checkpoint_identity,
                "state_protocol": FP32_STATE_PROTOCOL,
                "reference_identity": ref_identity,
                "experiment_identity": identity,
            },
        )
        save_loss_trace(bfile, trace)
        complete_run(run, ["traces.pt"])
        print(f"WROTE BASELINE {bfile} off_ce_mean={float(off_ce.mean()):.6f}", flush=True)
        return
    b = torch.load(bfile, map_location="cpu", weights_only=True)
    validate_loss_trace(b)
    check_baseline_stamp(b, args, int(seqs.shape[0]), seqs_sha, checkpoint_identity=model._lq_checkpoint_identity)
    baseline_identity = dict(b.get("reference_identity") or {})
    protocol = baseline_identity.get("state_protocol")
    if isinstance(protocol, dict):
        baseline_identity["state_protocol"] = {key: value for key, value in protocol.items() if key != "version"}
    if baseline_identity != ref_identity:
        raise ValueError("Reference producer/code/input/environment identity mismatch")
    off_ce, off_am = b["ce"], b["am"]
    if b.get("gpu") != gpu:
        # Record cross-device kernel differences alongside the shared reference.
        print(f"[warn] baseline gpu {b.get('gpu')!r} != cell gpu {gpu!r}", flush=True)
    print("=== quantized run (reference: vendor baseline) ===", flush=True)
    set_mode_bind(args.mode)
    if args.oracle_scale:
        from scripts.diagnostics.oracle_scale import install_oracle_scale

        n = install_oracle_scale(model, args.model, args.mode, block_size=args.block_size)
        print(f"oracle scales on {n} mixers", flush=True)
    q_ce, q_am = decode_per_position(model, seqs, args.prefill, args.seed, allow_state_clipping=args.allow_state_clipping)

    dce = q_ce - off_ce
    flips = (q_am != off_am).float()
    rec = {
        "baseline_file": str(bfile),
        "checkpoint_identity": model._lq_checkpoint_identity,
        "state_protocol": FP32_STATE_PROTOCOL,
        "experiment_identity": identity,
        "baseline_gpu": b.get("gpu"),  # != this cell's gpu ⇒ dce carries a cross-device fp offset
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": args.model,
        "rot": args.rot,
        "mode": args.mode,
        "seqlen": args.seqlen,
        "prefill": args.prefill,
        "n_seqs": int(seqs.shape[0]),
        "off_ce_mean": float(off_ce.mean()),
        "q_ce_mean": float(q_ce.mean()),
        "dce_mean": float(dce.mean()),
        "dce_per_seq": dce.mean(dim=1).tolist(),
        "off_ce_per_seq": off_ce.mean(dim=1).tolist(),
        "flip_rate": float(flips.mean()),
        "off_ce_bins": bucket(off_ce, edges, n_dec),
        "dce_bins": bucket(dce, edges, n_dec),
        "flip_bins": bucket(flips, edges, n_dec),
    }
    from lores.fake_quant.quantizers import saturation_events

    rec["sq_saturation_events"] = saturation_events()  # saturating-guard audit (0 = clean run)
    rec["allow_state_clipping"] = args.allow_state_clipping
    if rec["sq_saturation_events"] and not args.allow_state_clipping:
        raise ValueError("State saturation occurred; refusing to publish a clean loss trace")
    out = run / "result.json"
    trace_path = run / "traces.pt"
    trace = make_loss_trace(
        q_ce,
        seqs,
        prefill=args.prefill,
        expected_sequences=args.n_seqs,
        expected_seqlen=args.seqlen,
        metadata={
            "am": q_am,
            "experiment_identity": identity,
            "allow_state_clipping": args.allow_state_clipping,
            "sq_saturation_events": rec["sq_saturation_events"],
        },
    )
    save_loss_trace(trace_path, trace)
    rec.update(summarize_delta_ell(trace_path, bfile, run / "summary.json"))
    write_json_once(out, rec)
    print("WROTE", out, flush=True)
    complete_run(run, ["traces.pt", "summary.json", "result.json"])
    print(json.dumps({k: rec[k] for k in ("dce_mean", "flip_rate", "dce_bins")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
