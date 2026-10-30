"""Capture train-split log-decay traces for hybrid precision allocation."""

import argparse
import sys
from pathlib import Path

import torch

from lores.backends.hf.cache import FP32_STATE_PROTOCOL
from lores.evaluation.data import train_tokens
from lores.evaluation.loading import DELTA_ELL_MODELS as REGISTRY
from lores.evaluation.loss_traces import make_loss_trace, save_loss_trace
from lores.evaluation.output_store import create_run, fingerprint, sha256, write_json_once
from lores.evaluation.provenance import experiment_identity, save_tensor_artifact, source_identity
from lores.evaluation.runs import _snapshot_sources, complete_run
from lores.evaluation.teacher_forcing import decode_per_position


def nemo_step_logdecay(dt, A, dt_bias, dt_softplus):
    """Per-head log-decay for one nemotron selective_state_update call (batch 0).

    Decode call site expands dt to [B, H, P], dt_bias to [H, P] and A to [H, P, N]
    (constant over the expanded dims); reduce each back to per-head form first.
    """
    d = dt[0].float()
    if d.dim() > 1:
        d = d.reshape(d.shape[0], -1).mean(dim=-1) if d.shape[0] < d.numel() else d.reshape(-1)
    if dt_bias is not None:
        b = dt_bias.float()
        b = b.reshape(b.shape[0], -1)[:, 0] if b.dim() > 1 else b.reshape(-1)
        assert b.numel() == d.numel(), (
            f"dt_bias layout {tuple(dt_bias.shape)} does not reduce to one bias per head "
            f"({d.numel()} heads) — refusing to truncate"
        )
        d = d + b
    if dt_softplus:
        d = torch.nn.functional.softplus(d)
    Ah = A.float().reshape(A.shape[0], -1)[:, 0] if A.dim() > 1 else A.float().reshape(-1)
    return Ah[: d.numel()] * d


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen3_5", "nemotronh", "kimi"], required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    tok, model, set_mode = REGISTRY[args.model][0]("none")
    set_mode("off")
    ids = train_tokens(tok, args.steps + 1).cpu()
    captured = {}
    restore = []

    def record(index, value):
        captured.setdefault(index, []).append(value.detach().float().cpu().clone())

    if args.model == "qwen3_5":
        from lores.backends.hf.qwen3_5.integration import iter_qwen3_5_gdn_mixers

        mixers = list(iter_qwen3_5_gdn_mixers(model))
        for index, mixer in enumerate(mixers):
            original = mixer.recurrent_gated_delta_rule

            def wrap(kernel, index):
                def call(q, k, v, *a, **kw):
                    g = kw["g"][0].float().reshape(-1, kw["g"].shape[-1])
                    if g.shape[0] != 1:
                        raise ValueError("Expected a single decode position")
                    record(index, g[0])
                    return kernel(q, k, v, *a, **kw)

                return call

            restore.append((mixer, "recurrent_gated_delta_rule", original))
            mixer.recurrent_gated_delta_rule = wrap(original, index)
    elif args.model == "kimi":
        from lores.backends.hf.kimi_linear.integration import iter_kimi_kda_mixers

        mixers = list(iter_kimi_kda_mixers(model))
        owner = sys.modules[type(mixers[0]).__module__]
        original = owner.fused_recurrent_kda
        calls = 0

        def call(*a, **kw):
            nonlocal calls
            index = calls % len(mixers)
            calls += 1
            g = kw.get("g", a[3] if len(a) > 3 else None)
            if g is None or g.shape[0:2] != (1, 1):
                raise ValueError("Expected one Kimi decode position")
            record(index, g[0, 0])
            return original(*a, **kw)

        restore.append((owner, "fused_recurrent_kda", original))
        owner.fused_recurrent_kda = call
    else:
        import transformers.models.nemotron_h.modeling_nemotron_h as owner

        from lores.backends.hf.nemotronh.integration import iter_nemotronh_layers

        model(input_ids=ids[:, :3].cuda(), use_cache=True)
        mixers = [m for m in iter_nemotronh_layers(model) if hasattr(m, "in_proj")]
        original = owner.selective_state_update
        calls = 0

        def call(state, x, dt, A, B, C, *a, **kw):
            nonlocal calls
            index = calls % len(mixers)
            calls += 1
            record(index, nemo_step_logdecay(dt, A, kw.get("dt_bias"), kw.get("dt_softplus", False)))
            return original(state, x, dt, A, B, C, *a, **kw)

        restore.append((owner, "selective_state_update", original))
        owner.selective_state_update = call

    source = source_identity()
    relative = Path(__file__).resolve().relative_to(Path.cwd()).as_posix()
    source["files"][relative] = sha256(__file__)
    source["sha256"] = fingerprint(source["files"])
    identity = experiment_identity(args, ids, model._lq_checkpoint_identity, source=source)
    identity.update(split="train", token_selection="first contiguous tokens", state_protocol=FP32_STATE_PROTOCOL)
    run = create_run("calibration", args.model, "fp32-decay-traces", identity, root=args.output_root)
    _snapshot_sources(run, source)
    print(f"CALIBRATION {run}", flush=True)
    try:
        ce, am = decode_per_position(model, ids, 1, args.seed)
    finally:
        for owner, name, original in restore:
            setattr(owner, name, original)
    if sorted(captured) != list(range(len(mixers))):
        raise ValueError("Missing recurrent layers in calibration")
    lengths = {len(values) for values in captured.values()}
    if len(lengths) != 1 or min(lengths) < args.steps:
        raise ValueError(f"Incomplete calibration positions: {lengths}")
    traces = torch.stack([torch.stack(captured[i]) for i in range(len(mixers))])
    if not torch.isfinite(traces).all():
        raise ValueError("Non-finite decay trace")
    scores = traces.mean(dim=1)
    save_tensor_artifact(run / "log_decay.pt", {"log_decay": traces, "input_ids": ids})
    save_tensor_artifact(run / "decay_scores.pt", {"scores": scores, "input_ids": ids})
    save_loss_trace(
        run / "traces.pt",
        make_loss_trace(
            ce,
            ids,
            prefill=1,
            expected_sequences=1,
            expected_seqlen=ids.shape[1],
            metadata={"experiment_identity": identity, "am": am},
        ),
    )
    write_json_once(
        run / "summary.json",
        {
            "model": args.model,
            "score_shape": list(scores.shape),
            "decay_positions_per_layer": traces.shape[1],
            "units": "mean_log_decay",
            "allocation_unit": "key_channel" if scores.ndim == 3 else "head",
        },
    )
    complete_run(run, ["log_decay.pt", "decay_scores.pt", "traces.pt", "summary.json"])
    print(f"CALIBRATION COMPLETE {run}", flush=True)


if __name__ == "__main__":
    main()
