r"""Offline replay of the recurrent state under store-time quantization, with the quantizer's or oracle scales.

Records the per-step operands of one WikiText-2 evaluation row through the FP32 MixerZoo step path (or loads operands
recorded by ``record_qwen_operands``) and replays every layer: an FP32 copy of the state against quantized trajectories
for INT6 SR, INT4 SR and INT6 RTN, plain and rotated, each taking its scales from the stored state (``loop``, as the
state quantizer does) or from the FP32 copy in the same basis (``oracle``). ``--axis value`` groups each value row along
the key axis and rotates with R_k; ``--axis key`` groups each key column along the value axis and rotates with R_v.
Stochastic-rounding draws are shared within a (bits, rounding) group.

Per step and head: fresh rounding energy ``inj2``, accumulated error energy before the store ``acc2``, their inner
product ``cross``, the decay retention of the stored error ``ret_decay`` and the mean over scale groups of the scale
over the oracle scale ``scale_ratio``; ``S2`` is the FP32 state energy. ``--snapshot-step`` stores the pre-rounding
and FP32 states (quantization basis) and the query at those steps.

    python -m scripts.diagnostics.rotation_scale_feedback --cell mamba2_beta --output-root DIR [--axis key]
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from lores.fake_quant.quantizers import _saturate
from lores.utils.hadamard_utils import hadamard_matrix

BLOCK = 128
GROUPS = [(6, "SR"), (4, "SR"), (6, "RTN")]
SOURCES = ("loop", "oracle")
ROT = {"value": "rk", "key": "rv"}
NAMES = ("inj2", "acc2", "cross", "ret_decay", "scale_ratio")


def configs(axis):
    """Config tuples (bits, rounding, rotation, scale source) for one grouping axis."""
    return [(b, rnd, rot, src) for b, rnd in GROUPS for rot in ("none", ROT[axis]) for src in SOURCES]


def blocks(x, axis):
    """[..., dk, dv] -> scale groups [..., rows, nb, BLOCK]: along dk per value row (``value``) or along dv per key column."""
    xt = x.transpose(-1, -2) if axis == "value" else x
    n = xt.shape[-1]
    pad = (-n) % BLOCK
    if pad:
        xt = F.pad(xt, (0, pad))
    return xt.reshape(*xt.shape[:-1], -1, BLOCK), n


def unblocks(xb, n, axis):
    """Inverse of :func:`blocks`."""
    x = xb.reshape(*xb.shape[:-2], -1)[..., :n]
    return x.transpose(-1, -2) if axis == "value" else x


def scales(xb, qmax):
    """Quantizer scale: absmax per scale group over ``qmax``."""
    return xb.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax


def quant(xb, scale, qmax, u):
    """Symmetric grid; ``u`` uniform draws for SR (None = RTN)."""
    v = xb / scale
    if u is None:
        q = torch.round(v)
    else:
        fl = torch.floor(v)
        q = fl + (u < (v - fl)).to(v.dtype)
    return q.clamp(-qmax, qmax) * scale


def record_operands(model, ids):
    """Step the FP32 model over ``ids`` [1, T]; return per-layer stacked (q, k, v, g, beta) and layer metadata."""
    from lores.backends.hf.mixer_zoo.integration import iter_mixer_zoo_layers
    from lores.backends.hf.mixer_zoo.step import install_mixer_zoo_stepping, remove_mixer_zoo_stepping, reset_mixer_zoo_state

    install_mixer_zoo_stepping(model)
    layers = list(iter_mixer_zoo_layers(model))
    for attn in layers:
        attn._mz_stash_ops = True
    buf = [[] for _ in layers]
    reset_mixer_zoo_state(model)
    with torch.no_grad():
        for t in range(ids.shape[1]):
            model(input_ids=ids[:, t : t + 1])
            for i, attn in enumerate(layers):
                buf[i].append(tuple(None if x is None else x[0].detach().cpu() for x in attn._mz_last_ops))
    reset_mixer_zoo_state(model)
    remove_mixer_zoo_stepping(model)
    out = []
    for attn, steps in zip(layers, buf):
        q, k, v, g, beta = (None if steps[0][j] is None else torch.stack([s[j] for s in steps]) for j in range(5))
        out.append({"decay": attn.decay, "is_delta": bool(attn.is_delta), "q": q, "k": k, "v": v, "g": g, "beta": beta})
    return out


def replay(layer, R, device, axis="value", snapshot_steps=(), seed=0):
    """Replay one layer; return per-step metrics [T, C, H], the FP32 state energy [T, H] and the snapshots."""
    CONFIGS = configs(axis)
    q, k, v = (layer[x].to(device) for x in ("q", "k", "v"))
    g = None if layer["g"] is None else layer["g"].to(device)
    beta = None if layer["beta"] is None else layer["beta"].to(device)
    T, H, dk = k.shape
    dv = v.shape[-1]
    C = len(CONFIGS)
    rot = torch.tensor([c[2] != "none" for c in CONFIGS], device=device)
    fwd = (lambda S: R.T @ S) if axis == "value" else (lambda S: S @ R)
    back = (lambda X: R @ X) if axis == "value" else (lambda X: X @ R.T)
    S_fp = torch.zeros(H, dk, dv, device=device)
    S_q = torch.zeros(C, H, dk, dv, device=device)
    M = {n: torch.full((T, C, H), float("nan"), device=device) for n in NAMES}
    S2 = torch.full((T, H), float("nan"), device=device)
    groups = [(b, rnd, [ci for ci, c in enumerate(CONFIGS) if c[:2] == (b, rnd)]) for b, rnd in GROUPS]
    snaps = []

    def decay(S, gt):
        if layer["decay"] == "scalar":
            return S * gt.exp()[..., None, None]
        if layer["decay"] == "diagonal":
            return S * gt.exp()[..., None]
        return S

    def advance(S, kt, vt, gt, bt):
        S = decay(S, gt)
        if layer["is_delta"]:
            kTs = torch.einsum("hk,...hkv->...hv", kt, S)
            return S + torch.einsum("hk,...hv->...hkv", kt, bt[:, None] * (vt - kTs))
        return S + torch.einsum("hk,hv->hkv", kt, vt)

    with torch.no_grad():
        for t in range(T):
            kt, vt = k[t], v[t]
            gt = None if g is None else g[t]
            bt = None if beta is None else beta[t]
            if t:
                Ep = S_q - S_fp[None]
                M["ret_decay"][t] = decay(Ep, gt).pow(2).sum((-1, -2)) / Ep.pow(2).sum((-1, -2)).clamp_min(1e-30)
            S_fp = advance(S_fp, kt, vt, gt, bt)
            S_tilde = advance(S_q, kt, vt, gt, bt)
            E = S_tilde - S_fp[None]
            M["acc2"][t] = E.pow(2).sum((-1, -2))
            S2[t] = S_fp.pow(2).sum((-1, -2))
            X = _saturate(torch.where(rot[:, None, None, None], fwd(S_tilde), S_tilde))
            Xfp = torch.where(rot[:, None, None, None], fwd(S_fp)[None], S_fp[None])
            if t in snapshot_steps:
                snaps.append({"t": t, "X": X.cpu(), "Xfp": Xfp.expand_as(X).cpu(), "q": q[t].cpu()})
            Xq = torch.empty_like(X)
            torch.manual_seed(seed + 1000 * t + 7)
            for b, rnd, idx in groups:
                qmax = (1 << (b - 1)) - 1
                xb, n = blocks(X[idx], axis)
                s_loop = scales(xb, qmax)
                s_or = scales(blocks(Xfp[idx], axis)[0], qmax)
                use_loop = torch.tensor([CONFIGS[ci][3] == "loop" for ci in idx], device=device)
                M["scale_ratio"][t, idx] = (s_loop / s_or).mean((-1, -2, -3))
                u = torch.rand(xb.shape[1:], device=device) if rnd == "SR" else None
                Xq[idx] = unblocks(quant(xb, torch.where(use_loop[:, None, None, None, None], s_loop, s_or), qmax, u), n, axis)
            S_new = torch.where(rot[:, None, None, None], back(Xq), Xq)
            e = S_new - S_tilde
            M["inj2"][t] = e.pow(2).sum((-1, -2))
            M["cross"][t] = (E * e).sum((-1, -2))
            S_q = S_new
    return {n: x.cpu() for n, x in M.items()}, S2.cpu(), snaps


def main():
    """Entry point."""
    from lores.backends.hf.loading import MODELS_ROOT
    from lores.evaluation.output_store import create_run

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, help="MixerZoo cell, or the model name of --operands")
    ap.add_argument("--models-root", default=MODELS_ROOT)
    ap.add_argument("--operands", default=None, help="recorded operands (record_qwen_operands) instead of a MixerZoo cell")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=None, help="replay only the first N layers")
    ap.add_argument("--axis", choices=("value", "key"), default="value")
    ap.add_argument("--snapshot-step", type=int, nargs="*", default=(), help="store the states at these steps")
    ap.add_argument("--seed", type=int, default=0, help="offset of the per-step SR draws")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()
    out_dir = create_run(
        "diagnostics", args.cell, f"scale-feedback-{args.axis}", vars(args), namespace=("scale-feedback",), root=args.output_root
    )
    t0 = time.time()
    if args.operands:
        layers = torch.load(args.operands, map_location="cpu", weights_only=False)["layers"]
    else:
        from lores.backends.hf.loading import load_mixerzoo
        from lores.evaluation.teacher_forcing import make_sequences

        model, tokenizer = load_mixerzoo(args.cell, args.models_root, args.device)
        layers = record_operands(model, make_sequences(tokenizer, args.seqlen, 8)[:1].to(args.device))
        del model
        torch.cuda.empty_cache()
    print(f"{len(layers)} layers of operands in {time.time() - t0:.0f}s", flush=True)
    R = hadamard_matrix(layers[0]["k" if args.axis == "value" else "v"].shape[-1], args.device).to(torch.float32)
    per_layer = []
    for li, layer in enumerate(layers[: args.layers]):
        M, S2, snaps = replay(layer, R, args.device, args.axis, set(args.snapshot_step), seed=args.seed)
        per_layer.append({"metrics": M, "S2": S2, "snapshots": snaps})
        print(f"layer {li} replayed {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(out_dir, f"{args.cell}_scale.pt")
    torch.save({"cell": args.cell, "axis": args.axis, "configs": configs(args.axis), "layers": per_layer}, path)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
