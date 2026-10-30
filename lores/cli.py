"""Command-line PTQ and lm-eval evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from lores.config import EvalConfig, ModelConfig, PTQConfig, QuantConfig, RotationConfig
from lores.evaluation.output_store import create_run, sha256, write_json_once


def _add_common(p: argparse.ArgumentParser) -> None:
    # model
    p.add_argument("--input_model", default=ModelConfig.name_or_path, help="HF model name or path")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--model_max_length", type=int, default=2048)
    p.add_argument("--access_token", default=None)
    # Unset rotations use architecture defaults; explicit flags override them.
    p.add_argument(
        "--rotate_o",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Output-projection input Hadamard; default on for MixerZoo, off for hybrids.",
    )
    p.add_argument(
        "--rotate_residual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fold residual-stream R1; default on for MixerZoo, off for hybrids.",
    )
    p.add_argument(
        "--rotate_r4",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="MLP/MoE down-projection input Hadamard; follows R1 unless explicitly set.",
    )
    p.add_argument("--rotation_seed", type=int, default=0)
    # weight quant
    p.add_argument("--w_bits", type=int, default=16)
    p.add_argument("--w_groupsize", type=int, default=-1)
    p.add_argument("--w_clip", action=argparse.BooleanOptionalAction, default=False, help="grid-search the weight clip range")
    p.add_argument("--nsamples", type=int, default=128, help="GPTQ calibration samples")
    # linear-input activation quant (the SpinQuant "a_bits": inputs to all nn.Linears)
    p.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="bits for linear inputs (q/k/v/g/b/gk projections, o_proj, MLP); lm_head remains unquantized",
    )
    p.add_argument("--a_groupsize", type=int, default=-1, help="-1 = per-token; >0 = group-wise per-token for the linear inputs")
    p.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use asymmetric activation grids instead of the symmetric default.",
    )
    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")


def _config_from_args(a: argparse.Namespace) -> PTQConfig:
    return PTQConfig(
        model=ModelConfig(
            name_or_path=a.input_model,
            dtype=a.dtype,
            seqlen=a.model_max_length,
            access_token=a.access_token,
        ),
        rotation=RotationConfig(
            o=a.rotate_o,
            residual=a.rotate_residual,
            r4=a.rotate_r4,
            seed=a.rotation_seed,
        ),
        quant=QuantConfig(
            w_bits=a.w_bits,
            w_groupsize=a.w_groupsize,
            w_clip=a.w_clip,
            a_bits=a.a_bits,
            a_groupsize=a.a_groupsize,
            a_sym=not a.a_asym,
            calib_nsamples=a.nsamples,
        ),
        eval=EvalConfig(),
        device=a.device,
        seed=a.seed,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse ``ptq`` args into a :class:`PTQConfig` and run the pipeline."""
    parser = argparse.ArgumentParser(
        "lores", description="LoReS: post-training quantization of recurrent states, weights and activations"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ptq = sub.add_parser("ptq", help="rotate + quantize + lm-eval (default: WikiText-2 word perplexity)")
    _add_common(p_ptq)
    p_ptq.add_argument(
        "--eval_tasks",
        default="wikitext",
        help="comma-separated lm-eval task names; default 'wikitext' reports WikiText-2 word perplexity",
    )
    p_ptq.add_argument("--eval_batch_size", type=int, default=16, help="lm-eval batch size")
    p_ptq.add_argument("--eval_limit", type=int, default=None, help="lm-eval document limit per task (debug)")
    p_ptq.add_argument("--output-root", help="external raw-data root (default: output/ under the checkout)")
    p_ptq.add_argument("--dry-run", action="store_true", help="print configuration without loading models or writing results")

    args = parser.parse_args(argv)
    if args.eval_batch_size < 1:
        parser.error("--eval_batch_size must be positive")
    logging.basicConfig(level=logging.INFO, format="[lores] %(message)s")
    cfg = _config_from_args(args)
    cfg.eval.tasks = args.eval_tasks
    cfg.eval.limit, cfg.eval.batch_size = args.eval_limit, args.eval_batch_size
    config = asdict(cfg)
    config["model"].pop("access_token")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    from lores.wa_ptq.pipeline import run_ptq

    config["producer_sha256"] = sha256(__file__)
    run = create_run(
        "diagnostics",
        Path(cfg.model.name_or_path).name,
        f"W{cfg.quant.w_bits}A{cfg.quant.a_bits}",
        config,
        namespace=("ptq",),
        root=args.output_root,
    )
    print(f"OUTPUT {run}", flush=True)
    metrics = run_ptq(cfg)
    write_json_once(run / "result.json", {"config": config, "metrics": metrics})
    write_json_once(run / "complete.json", {"artifacts": {"result.json": sha256(run / "result.json")}})


if __name__ == "__main__":
    main()
