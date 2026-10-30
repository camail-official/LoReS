"""Relative readout error of a value row against the row's magnitude relative to the largest row of its head.

For each snapshot (state S [H, dv, dk], query q) every value row i contributes x = ||S_i|| / max_j ||S_j|| (within the
head) and, per grouping, (o_hat_i - o_i)^2 and o_i^2. Rows are binned in log x; the per-bin error is
sqrt(sum err^2 / sum o^2) (RTN, INT8 by default). The proposition predicts a flat line for value grouping (the row's own
crest factor) and a slope of -1 for key grouping (the step is set by the head's largest rows).

    python -m scripts.diagnostics.row_error_vs_magnitude --model gated_deltanet --snapshot <model>_states.pt --out DIR

Snapshots come from ``scripts.calibration.state_geometry --dump-states`` (per layer S [T, H, dv, dk], q [T, Hq, dk]).
"""

import argparse
import json
from pathlib import Path

import torch


def quantize(x, dim, bits=8):
    """Symmetric absmax INT grid along ``dim`` (one scale per slice orthogonal to ``dim``), round to nearest."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12) / qmax
    return torch.round(x / scale).clamp_(-qmax, qmax) * scale


def readout(s, q):
    """Readout of ``s`` [H, K, V] with query ``q`` [H, K] -> [H, V]."""
    return torch.einsum("hkv,hk->hv", s, q)


def expand_query(q, heads):
    """Mamba-2 stores C per group (or per head); repeat groups over their heads."""
    if q.shape[0] == heads:
        return q
    assert heads % q.shape[0] == 0, (q.shape, heads)
    return q.repeat_interleave(heads // q.shape[0], dim=0)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    edges = torch.logspace(-4, 0, args.bins + 1)
    arms = ("value", "key")
    err = {g: torch.zeros(args.bins) for g in arms}
    sig = torch.zeros(args.bins)
    count = torch.zeros(args.bins)
    snap = torch.load(args.snapshot, map_location="cpu", mmap=True, weights_only=False)
    for key, d in sorted(snap.items(), key=lambda kv: int(kv[0])):
        for t in range(d["S"].shape[0]):
            s = d["S"][t].float().transpose(-1, -2)  # [H, K, V]: value rows along the last axis
            q = expand_query(d["q"][t].float(), s.shape[0])
            o = readout(s, q)
            norms = s.norm(dim=1)  # [H, V] row magnitudes over the keys
            x = (norms / norms.amax(dim=1, keepdim=True).clamp_min(1e-30)).clamp(1e-4, 1.0)
            b = torch.bucketize(x.flatten(), edges[1:-1])
            sig.index_add_(0, b, o.flatten().pow(2))
            count.index_add_(0, b, torch.ones_like(b, dtype=torch.float32))
            for g in arms:
                s_hat = quantize(s, dim=1 if g == "value" else 2, bits=args.bits)
                e = readout(s_hat, q) - o
                err[g].index_add_(0, b, e.flatten().pow(2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "x_mid": (edges[:-1] * edges[1:]).sqrt().tolist(),
        "count": count.tolist(),
        **{g: (e / sig.clamp_min(1e-30)).sqrt().tolist() for g, e in err.items()},
    }
    json.dump(res, open(out / f"row_error_vs_magnitude_{args.model}.json", "w"))
    print(args.model, "rows", int(count.sum()), "nonempty bins", int((count > 0).sum()))


if __name__ == "__main__":
    main()
