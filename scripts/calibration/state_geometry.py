"""Capture paired state/query snapshots and per-head grouping-range statistics."""

import argparse
import sys
from pathlib import Path

import torch

from lores.backends.hf.loading import MODELS_ROOT, load_mixerzoo
from lores.backends.hf.state import state_quant_mixers
from lores.evaluation.data import train_tokens
from lores.evaluation.loading import DELTA_ELL_MODELS
from lores.evaluation.output_store import create_run, writable_output, write_json_once
from lores.evaluation.provenance import save_tensor_artifact, stamp
from scripts.calibration.rho import rho

ZOO = ("linear_none_beta", "deltanet", "gla_beta", "mamba2_beta", "gated_deltanet", "kda")


@torch.no_grad()
def capture_decode(model_key, make_capture, *, ids, prepare=None):
    """Capture recurrent stores on batched train sequences without quantizing."""
    build, _ = DELTA_ELL_MODELS[model_key]
    tok, model, set_mode = build("none")
    set_mode("int8_block")  # install the state hooks; quantizers replaced below before any state math
    for i, mix in enumerate(state_quant_mixers(model)):
        cap = make_capture(i, mix)
        mix._state_quant = cap
    if prepare is not None:
        prepare(tok, model)
    ids = ids(tok).cuda()
    out = model(input_ids=ids[:, :1], use_cache=True, return_dict=True)
    cache = out.past_key_values
    for t in range(1, ids.shape[1]):
        out = model(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values


def record_snapshot(index, state, query, diag, states, queries, dump_states):
    # State layout is [H, dv, dk]; queries retain the kernel's head/group layout.
    rows = state.abs().amax(dim=-1).square().mean(dim=-1)
    cols = state.abs().amax(dim=-2).square().mean(dim=-1)
    diag.setdefault(index, []).append(torch.stack((rows, cols), dim=-1))
    if dump_states:
        states.setdefault(index, []).append(state.bfloat16())
        queries.setdefault(index, []).append(query.cpu().bfloat16())


def finalize(name, diag, states, queries, output, *, seqs, seqlen, stride, first_step=None):
    first_step = stride - 1 if first_step is None else first_step
    counts = {len(v) for v in diag.values()}
    if counts != {seqs * len(range(first_step, seqlen, stride))}:
        raise ValueError(f"Incomplete state snapshots: {counts}")
    ranges = {}
    for i, values in sorted(diag.items()):
        mean = torch.stack(values).mean(dim=0)
        ranges[str(i)] = {"P_val": mean[:, 0].tolist(), "P_key": mean[:, 1].tolist(), "n_snapshots": len(values)}
    output = Path(output)
    write_json_once(output / f"{name}_rangediag.json", ranges)
    if states:
        save_tensor_artifact(
            output / f"{name}_states.pt",
            {str(i): {"S": torch.stack(values), "q": torch.stack(queries[i])} for i, values in sorted(states.items())},
        )
    write_json_once(
        output / f"{name}_meta.json",
        stamp(
            {
                "model": name,
                "calib": "wikitext2-train",
                "seqs": seqs,
                "steps": seqlen,
                "stride": stride,
                "first_step": first_step,
                "state_layout": "vk",
                "snapshot_dtype": "bfloat16",
                "range_dtype": "float32",
                "rho": rho(ranges),
            }
        ),
    )
    print(f"WROTE {output}", flush=True)


@torch.no_grad()
def zoo_main(cell, seqs=8, seqlen=2048, *, output, dump_states=False, stride=64, first_step=None, models_root=MODELS_ROOT):
    from lores.backends.hf.mixer_zoo.integration import iter_mixer_zoo_layers
    from lores.backends.hf.mixer_zoo.step import install_mixer_zoo_stepping, remove_mixer_zoo_stepping

    model, tok = load_mixerzoo(cell, models_root)
    ids = train_tokens(tok, seqs * seqlen).reshape(seqs, seqlen).cuda()
    diag, states, queries = {}, {}, {}
    attns = []
    first_step = stride - 1 if first_step is None else first_step

    def factory(index):
        calls = 0

        def capture(state):
            nonlocal calls
            calls += 1
            if calls - 1 >= first_step and (calls - 1 - first_step) % stride == 0:
                query = attns[index]._mz_last_ops[0].float()
                for b in range(state.shape[0]):
                    record_snapshot(index, state[b].float().transpose(-1, -2).cpu(), query[b], diag, states, queries, dump_states)
            return state

        return capture

    install_mixer_zoo_stepping(model, state_quant=factory)
    for attn in iter_mixer_zoo_layers(model):
        attn._mz_stash_ops = True
        attns.append(attn)
    try:
        for t in range(seqlen):
            model(input_ids=ids[:, t : t + 1])
    finally:
        remove_mixer_zoo_stepping(model)
    finalize(cell, diag, states, queries, output, seqs=seqs, seqlen=seqlen, stride=stride, first_step=first_step)


@torch.no_grad()
def hyb_main(name, n_seqs=8, seqlen=2048, *, output, dump_states=False, stride=64):
    qslot, diag, states, queries, counts = {}, {}, {}, {}, {}
    restore = []

    def prepare(tok, model):
        """Capture the queries actually passed to each recurrent kernel."""
        if name == "nemotronh":
            import transformers.models.nemotron_h.modeling_nemotron_h as M

            from lores.backends.hf.nemotronh.integration import iter_nemotronh_layers as it

            model(torch.tensor([[1, 2, 3]], device="cuda"))  # bind the lazy SSM kernels
            mixers = [m for m in it(model) if hasattr(m, "in_proj")]
            orig = M.selective_state_update
            i_call = {"i": 0}

            def wrapped(state, x, dt, A, B, C, *a, **k):
                i = i_call["i"] % len(mixers)
                i_call["i"] += 1
                qslot[i] = C.float().cpu()  # [B, G, dk] — all batch items
                return orig(state, x, dt, A, B, C, *a, **k)

            restore.append((M, "selective_state_update", orig))
            M.selective_state_update = wrapped
        elif name == "qwen3_5":
            from lores.backends.hf.qwen3_5.integration import iter_qwen3_5_gdn_mixers as it

            for i, m in enumerate(it(model)):
                o0 = m.recurrent_gated_delta_rule

                def _mk(kernel, i=i):
                    def w(q, k, v, *a, **kw):
                        # T=1 decode -> [B,H,dk]; l2norm to match the kernel's internal qk-norm
                        # (zoo parity: the zoo stepper stashes q POST-l2norm)
                        qs = q.float().reshape(q.shape[0], -1, q.shape[-1])
                        qslot[i] = (qs / qs.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu()
                        return kernel(q, k, v, *a, **kw)

                    return w

                restore.append((m, "recurrent_gated_delta_rule", o0))
                m.recurrent_gated_delta_rule = _mk(o0)
        else:
            from lores.backends.hf.kimi_linear.integration import iter_kimi_kda_mixers as it

            mixers = list(it(model))
            M = sys.modules[type(mixers[0]).__module__]
            orig = M.fused_recurrent_kda
            i_call = {"i": 0}

            def wrapped(*a, **kw):
                i = i_call["i"] % len(mixers)
                i_call["i"] += 1
                q = kw.get("q", a[0] if a else None)
                # T=1 decode -> [B,H,dk]; l2norm to match the kernel's internal qk-norm (zoo parity)
                qs = q.float().reshape(q.shape[0], -1, q.shape[-1])
                qslot[i] = (qs / qs.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu()
                return orig(*a, **kw)

            restore.append((M, "fused_recurrent_kda", orig))
            M.fused_recurrent_kda = wrapped

    def mk_rec(index, mix):
        def capture(state):
            counts[index] = counts.get(index, 0) + 1
            if counts[index] % stride == 0 and index in qslot:
                for b in range(state.shape[0]):
                    value = state[b].float().cpu()
                    if name != "nemotronh":
                        value = value.transpose(-1, -2)
                    record_snapshot(index, value, qslot[index][b], diag, states, queries, dump_states)
            return state

        return capture

    try:
        capture_decode(
            name, mk_rec, prepare=prepare, ids=lambda tok: train_tokens(tok, n_seqs * (seqlen + 1)).reshape(n_seqs, seqlen + 1)
        )
    finally:
        for owner, attr, original in restore:
            setattr(owner, attr, original)
    finalize(name, diag, states, queries, output, seqs=n_seqs, seqlen=seqlen, stride=stride)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ZOO + ("qwen3_5", "nemotronh", "kimi"))
    parser.add_argument("--seqs", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--first-step", type=int, help="MixerZoo snapshot phase; default stride - 1")
    parser.add_argument("--models-root", default=MODELS_ROOT)
    parser.add_argument("--dump-states", action="store_true", help="save paired BF16 state/query snapshots")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if min(args.seqs, args.seqlen, args.stride) < 1 or args.stride > args.seqlen:
        parser.error("require positive seqs and 1 <= stride <= seqlen")
    if args.first_step is not None and (args.model not in ZOO or not 0 <= args.first_step < args.seqlen):
        parser.error("--first-step requires MixerZoo and 0 <= first-step < seqlen")
    output = (
        writable_output(args.outdir, root=args.output_root)
        if args.outdir
        else create_run("calibration", args.model, "state-geometry", vars(args), root=args.output_root)
    )
    options = dict(output=output, dump_states=args.dump_states, stride=args.stride)
    if args.model in ZOO:
        zoo_main(args.model, args.seqs, args.seqlen, models_root=args.models_root, first_step=args.first_step, **options)
    else:
        hyb_main(args.model, args.seqs, args.seqlen, **options)


if __name__ == "__main__":
    main()
