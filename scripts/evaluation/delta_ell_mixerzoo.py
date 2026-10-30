#!/usr/bin/env python3
"""MixerZoo stepped/chunked parity and teacher-forced loss traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from lores.backends.hf.loading import MODELS_ROOT, load_mixerzoo


@torch.no_grad()
def stepped_ce(model, ids: torch.Tensor, collect_logits: bool = False):
    """Run ``ids`` [B, T] teacher-forced one token at a time through the (patched) model.

    Returns per-position CE [B, T-1] (fp32); optionally the full logits [B, T, V] (parity only).
    Caller must have installed stepping and is responsible for reset before/after.
    """
    from lores.backends.hf.mixer_zoo.step import reset_mixer_zoo_state

    reset_mixer_zoo_state(model)
    B, T = ids.shape
    ce = torch.zeros(B, T - 1, dtype=torch.float32)
    logits_all = [] if collect_logits else None
    t0 = time.time()
    for t in range(T):
        logits = model(input_ids=ids[:, t : t + 1]).logits[:, -1].float()
        if collect_logits:
            logits_all.append(logits.cpu())
        if t < T - 1:
            ce[:, t] = F.cross_entropy(logits, ids[:, t + 1], reduction="none").cpu()
        if t % 512 == 0:
            print(f"  [step {t}/{T}] {time.time() - t0:.0f}s", flush=True)
    reset_mixer_zoo_state(model)
    if collect_logits:
        return ce, torch.stack(logits_all, dim=1)
    return ce


def binned(x: torch.Tensor, nbins: int = 16) -> list[float]:
    """Mean of ``x`` [B, T] over B and over T split into ``nbins`` equal position bins."""
    T = x.shape[1]
    edges = [round(i * T / nbins) for i in range(nbins + 1)]
    return [x[:, a:b].mean().item() for a, b in zip(edges[:-1], edges[1:])]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--models-root", default=MODELS_ROOT)
    ap.add_argument("--mode", required=True, help="state quantization mode, e.g. int6_block / int4_block_dk")
    ap.add_argument(
        "--headbits",
        default=None,
        metavar="BITS_JSON",
        help="per-(layer, head) BIT allocation json {layer: [bits]}; quantizes each head "
        "at its own precision with the mode's grouping (HeadBitsQuant)",
    )
    ap.add_argument(
        "--chanbits",
        default=None,
        metavar="BITS_JSON",
        help="per-(layer, head, key-channel) bits for diagonal-decay GLA/KDA; requires --mixhead",
    )
    ap.add_argument(
        "--mixhead",
        default=None,
        metavar="MASK_JSON",
        help="per-(layer, head) grouping-axis mask from the readout probe: heads marked "
        "True quantize value-axis (block_dk), the rest key-axis (block)",
    )
    ap.add_argument("--rk", action="store_true", help="quantize the state in a Hadamard-rotated dk basis")
    ap.add_argument("--rv", action="store_true", help="quantize the state in a Hadamard-rotated dv (value) basis")
    ap.add_argument(
        "--oracle-scale",
        action="store_true",
        help="scale every group from an FP32 copy of the state advanced with the same operands (diagnostic)",
    )
    ap.add_argument(
        "--quant-every",
        type=int,
        default=1,
        metavar="C",
        help="quantize the state only every C steps (= chunk-boundary quantization with "
        "chunk size C; the state evolves in fp in between). Sweeps the "
        "re-quantization COUNT for the damage-vs-events fit.",
    )
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="explicit per-pass stochastic-rounding seed")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--parity-len", type=int, default=512)
    ap.add_argument("--output-root", default=None, help="external raw-output store")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None, help="override output stem")
    args = ap.parse_args()
    from lores.evaluation.output_store import source_path

    for name in ("mixhead", "headbits", "chanbits"):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, str(source_path(value)))

    assert not (args.headbits and args.chanbits), "--headbits and --chanbits are exclusive"
    assert not (args.chanbits and not args.mixhead), "--chanbits requires --mixhead"
    assert not (args.mixhead and not (args.headbits or args.chanbits)), "--mixhead requires --headbits or --chanbits"
    assert not (args.chanbits and (args.rk or args.rv)), (
        "channel promotion is in the physical decay basis; rotations are not supported"
    )
    assert not (args.oracle_scale and (args.headbits or args.chanbits or args.quant_every != 1)), (
        "--oracle-scale takes a plain integer block mode"
    )
    stem = args.tag or (
        args.mode
        + ("_headbits" if args.headbits else "")
        + ("_chanbits" if args.chanbits else "")
        + ("_mixhead" if args.mixhead else "")
        + ("_rk" if args.rk else "")
        + ("_rv" if args.rv else "")
        + ("_oracle" if args.oracle_scale else "")
        + "_qar"
        + (f"_q{args.quant_every}" if args.quant_every != 1 else "")
    )
    t0 = time.time()
    model, tokenizer = load_mixerzoo(args.cell, args.models_root, args.device)

    from lores.evaluation.data import get_wikitext2
    from lores.evaluation.teacher_forcing import to_rows

    testenc = get_wikitext2(tokenizer=tokenizer, seqlen=args.seqlen, eval_mode=True)
    flat = testenc.input_ids[0]
    need = args.seqs * args.seqlen
    assert flat.numel() >= need, f"wt2 eval too short: {flat.numel()} < {need}"
    ids = to_rows(testenc, args.seqs, args.seqlen).to(args.device)
    from lores.backends.hf.cache import FP32_STATE_PROTOCOL
    from lores.evaluation.provenance import (
        experiment_identity,
        local_checkpoint_identity,
    )
    from lores.evaluation.runs import evaluation_run

    checkpoint = local_checkpoint_identity(os.path.join(args.models_root, args.cell))
    identity = experiment_identity(args, ids, checkpoint)
    identity["state_protocol"] = FP32_STATE_PROTOCOL
    with evaluation_run("references", args.cell, "fp32-" + stem, identity, root=args.output_root) as ref_run:
        with evaluation_run("delta_ell", args.cell, stem, identity, root=args.output_root) as run:
            _evaluate(args, model, ids, identity, run, ref_run, t0)


def _evaluate(args, model, ids, identity, run, ref_run, t0):
    from lores.backends.hf.mixer_zoo.step import (
        install_mixer_zoo_stepping,
        remove_mixer_zoo_stepping,
    )
    from lores.evaluation.loss_traces import (
        make_loss_trace,
        save_loss_trace,
        summarize_delta_ell,
    )
    from lores.evaluation.output_store import write_json_once
    from lores.evaluation.runs import complete_run

    out_path = run / "result.json"
    reference_path = ref_run / "traces.pt"
    trace_path = run / "traces.pt"

    # ---- parity record: stepped fp vs chunked forward on one short sequence ----
    par_ids = ids[:1, : args.parity_len]
    chunk_logits = model(input_ids=par_ids).logits.float().cpu()  # chunked (T>1) path, pre-patch
    install_mixer_zoo_stepping(model)
    ce_par, step_logits = stepped_ce(model, par_ids, collect_logits=True)
    agree = (chunk_logits.argmax(-1) == step_logits.argmax(-1)).float().mean().item()
    ce_chunk = F.cross_entropy(chunk_logits[0, :-1], par_ids[0, 1:].cpu(), reduction="none")
    parity = {
        "top1_agree": agree,
        "dce_mean_abs": (ce_par[0] - ce_chunk).abs().mean().item(),
        "ce_chunk_mean": ce_chunk.mean().item(),
        "ce_step_mean": ce_par[0].mean().item(),
    }
    print(f"[parity] top1_agree={agree:.4f} dce_mean_abs={parity['dce_mean_abs']:.5f}", flush=True)
    remove_mixer_zoo_stepping(model)

    # ---- state quantization, bit allocation, and rotations ----
    state_quant = args.mode
    if args.chanbits:
        from lores.fake_quant.state import AxisBitsQuant

        if model.config.decay != "diagonal":
            raise ValueError("--chanbits requires diagonal channel decay")
        cb = json.load(open(args.chanbits))
        axes = json.load(open(args.mixhead))

        def state_quant(idx):
            return AxisBitsQuant(torch.tensor(axes[str(idx)]), torch.tensor(cb[str(idx)]), mode=args.mode)
    elif args.headbits and args.mixhead:
        import json as _json

        from lores.fake_quant.state import AxisBitsQuant

        _hb = _json.load(open(args.headbits))
        _masks = _json.load(open(args.mixhead))
        state_quant = lambda idx: AxisBitsQuant(  # noqa: E731
            torch.tensor(_masks[str(idx)]), torch.tensor(_hb[str(idx)]), mode=args.mode
        )
    elif args.headbits:
        import json as _json

        from lores.fake_quant.state import HeadBitsQuant

        _hb = _json.load(open(args.headbits))
        from lores.evaluation.state_quant import headbits_gran

        _gran = headbits_gran(args.mode)
        state_quant = lambda idx: HeadBitsQuant(_gran, torch.tensor(_hb[str(idx)]))  # noqa: E731
    elif args.oracle_scale:
        from lores.backends.hf.mixer_zoo.integration import iter_mixer_zoo_layers
        from scripts.diagnostics.oracle_scale import MixerZooOracleQuant

        _attns = list(iter_mixer_zoo_layers(model))
        state_quant = lambda idx: MixerZooOracleQuant(_attns[idx], args.mode)  # noqa: E731
    rv = None
    if args.rv:
        from lores.utils.hadamard_utils import hadamard_matrix

        rv = hadamard_matrix(model.config.head_dim, args.device).to(torch.float32)
    rk = None
    if args.rk:
        from lores.utils.hadamard_utils import hadamard_matrix

        rk = hadamard_matrix(model.config.head_dim, args.device).to(torch.float32)  # already orthonormal!

    if args.quant_every > 1:  # quantize only every C-th step == chunk-boundary quantization
        _base = state_quant

        def _every_c(idx, _base=_base, C=args.quant_every):
            from lores.fake_quant.state import make_state_quant, quant_every

            inner = _base(idx) if callable(_base) else make_state_quant(_base)
            return quant_every(inner, C)  # shared wrapper: reset() forwarding included

        state_quant = _every_c

    def install_quant():
        install_mixer_zoo_stepping(model, state_quant=state_quant, rk=rk, rv=rv)

    # ---- teacher-forced ΔCE(t), fp then quant ----
    install_mixer_zoo_stepping(model)
    torch.manual_seed(args.seed)
    ce_fp = stepped_ce(model, ids)
    save_loss_trace(
        reference_path,
        make_loss_trace(
            ce_fp,
            ids,
            prefill=1,
            expected_sequences=args.seqs,
            expected_seqlen=args.seqlen,
            metadata={"role": "reference", "experiment_identity": identity},
        ),
    )
    complete_run(ref_run, ["traces.pt"])
    print(f"[tf] fp ce={ce_fp.mean().item():.5f} ({time.time() - t0:.0f}s)", flush=True)
    remove_mixer_zoo_stepping(model)

    install_quant()
    torch.manual_seed(args.seed)
    ce_q = stepped_ce(model, ids)
    from lores.fake_quant.quantizers import saturation_events

    if saturation_events():
        raise ValueError("State saturation occurred; refusing to publish a clean loss trace")
    save_loss_trace(
        trace_path,
        make_loss_trace(
            ce_q,
            ids,
            prefill=1,
            expected_sequences=args.seqs,
            expected_seqlen=args.seqlen,
            metadata={"role": "quantized", "experiment_identity": identity},
        ),
    )
    summary = summarize_delta_ell(trace_path, reference_path, run / "summary.json")
    print(f"[tf] quant ce={ce_q.mean().item():.5f} ({time.time() - t0:.0f}s)", flush=True)
    remove_mixer_zoo_stepping(model)

    dce = ce_q - ce_fp
    tf = {
        "ce_fp_mean": ce_fp.mean().item(),
        "ce_q_mean": ce_q.mean().item(),
        "dce_mean": dce.mean().item(),
        "ce_fp_per_seq": ce_fp.mean(dim=1).tolist(),
        "ce_q_per_seq": ce_q.mean(dim=1).tolist(),
        "dce_per_seq": dce.mean(dim=1).tolist(),
        "ce_fp_bins": binned(ce_fp),
        "ce_q_bins": binned(ce_q),
        "dce_bins": binned(dce),
        **summary,
    }

    result = {
        "cell": args.cell,
        "mode": args.mode,
        "argv": sys.argv[1:],
        "quant_after_read": True,
        "experiment_identity": identity,
        "mixhead": args.mixhead,
        "headbits": args.headbits,
        "chanbits": args.chanbits,
        "rk": args.rk,
        "rv": args.rv,
        "oracle_scale": args.oracle_scale,
        "quant_every": args.quant_every,
        "seqs": args.seqs,
        "seqlen": args.seqlen,
        "parity": parity,
        "tf": tf,
        "runtime_s": round(time.time() - t0, 1),
    }
    write_json_once(Path(out_path), result)
    complete_run(run, ["traces.pt", "summary.json", "result.json"])
    print(f"[done] {out_path} ({result['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
