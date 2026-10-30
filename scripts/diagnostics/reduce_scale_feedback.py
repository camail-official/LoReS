r"""Reduce ``rotation_scale_feedback`` replays to the figure inputs.

``inflation``: 24-layer replays of the MixerZoo cells on both axes, INT4 SR, quantizer scales; per head the median over
the second half of the sequence of the per-step scale over the oracle scale (mean over scale groups), and per 64-step
bin and head the median over the bin, then the 90th percentile over heads.

``alignment``: value-axis replays with snapshots, INT4 SR, oracle scales. At every snapshot, each run's own state before
rounding gives the exact SR mean and second moment of every entry at the oracle scale (outcomes floor and floor + 1,
clamped to the grid), in the run's own basis with the query in that basis. Per head, averaged over the snapshots: the
RMS fresh readout error R_k / plain, the RMS fresh error ratio, and the query alignment
A = d_k sum_j q_j^2 c_j / (|q|^2 sum_j c_j) with c_j the expected error energy of key column j; ``dev`` is the largest
relative deviation over the snapshots of the readout ratio from (error ratio x alignment ratio).

``rounding``: Mamba-2 value-axis replays with different SR draws; plain, quantizer scales. Per head and step the
remaining error |E_t|^2 - sum_tau gamma_{tau,t}^2 |eps_tau|^2 (exact under scalar decay), SR averaged over the replays,
RTN from the first; per head its mean over time.

    python -m scripts.diagnostics.reduce_scale_feedback --kind inflation --files REPLAY.pt ... [--output-root DIR]
"""

import argparse
import json

import numpy as np
import torch

from lores.utils.hadamard_utils import hadamard_matrix
from scripts.diagnostics.rotation_scale_feedback import ROT, blocks, scales, unblocks

BIN = 64


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def metric(d, name):
    """[T, C, L * H] over all replayed layers."""
    return torch.cat([pl["metrics"][name] for pl in d["layers"]], -1)


def inflation(files):
    out = {"value": {}, "key": {}}
    for f in files:
        d = load(f)
        C, rot = d["configs"], ROT[d["axis"]]
        sr = metric(d, "scale_ratio")
        T = sr.shape[0]
        binned = sr[: T - T % BIN].reshape(-1, BIN, *sr.shape[1:])
        r = {"t": [BIN // 2 + BIN * i for i in range(binned.shape[0])]}
        for label, name in (("plain", "none"), ("rot", rot)):
            c = C.index((4, "SR", name, "loop"))
            r[f"inflation_{label}"] = sr[T // 2 :, c].nanmedian(0).values.tolist()
            r[f"{label}_p90"] = torch.nanquantile(binned[:, :, c].nanmedian(1).values, 0.9, dim=1).tolist()
        out[d["axis"]][d["cell"]] = r
        p90 = [np.nanpercentile(r[f"inflation_{label}"], 90) for label in ("plain", "rot")]
        print(f"{d['cell']} {d['axis']}: p90 over heads, plain {p90[0]:.2f}, rotated {p90[1]:.2f}", flush=True)
    return out


def moments(X, Xscale, qmax):
    xb, n = blocks(X, "value")
    s = scales(blocks(Xscale, "value")[0], qmax)
    v = xb / s
    fl = torch.floor(v)
    f = v - fl
    e0 = fl.clamp(-qmax, qmax) * s - xb
    e1 = (fl + 1).clamp(-qmax, qmax) * s - xb
    return unblocks((1 - f) * e0 + f * e1, n, "value"), unblocks((1 - f) * e0**2 + f * e1**2, n, "value")


def readout(q, mu, m2):
    """Expected squared readout error, expected error energy and query alignment per head."""
    r = torch.einsum("hk,hkv->hv", q, mu).pow(2).sum(-1) + (q.pow(2)[..., None] * (m2 - mu**2)).sum((-1, -2))
    c = m2.sum(-1)
    return r, c.sum(-1), q.shape[-1] * (q.pow(2) * c).sum(-1) / (q.pow(2).sum(-1) * c.sum(-1))


def alignment(files):
    qmax = 7
    out = {}
    for f in files:
        d = load(f)
        C = d["configs"]
        po, ro = C.index((4, "SR", "none", "oracle")), C.index((4, "SR", "rk", "oracle"))
        R = None
        acc = {k: [] for k in ("ratio", "norm", "Ap", "Ar", "dev")}
        for pl in d["layers"]:
            if not pl["snapshots"]:
                continue
            a = dict.fromkeys(acc, 0)
            for sn in pl["snapshots"]:
                S, q = sn["Xfp"][po], sn["q"]
                R = hadamard_matrix(S.shape[-2], "cpu").float() if R is None else R
                rp, cp, ap = readout(q, *moments(sn["X"][po], S, qmax))
                rr, cr, ar = readout(q @ R, *moments(sn["X"][ro], R.T @ S, qmax))
                a["ratio"] = a["ratio"] + (rr / rp).sqrt()
                a["norm"] = a["norm"] + (cr / cp).sqrt()
                a["Ap"] = a["Ap"] + ap
                a["Ar"] = a["Ar"] + ar
                a["dev"] = torch.maximum(torch.as_tensor(a["dev"]), ((rr / rp) - (cr / cp) * (ar / ap)).abs() / (rr / rp))
            ns = len(pl["snapshots"])
            for k in acc:
                acc[k].append(a[k] if k == "dev" else a[k] / ns)
        v = {k: torch.cat(x).numpy() for k, x in acc.items()}
        worse, high = v["ratio"] > 1, v["Ap"] >= 0.7
        summary = {
            "heads": len(worse),
            "worse": int(worse.sum()),
            "Ap_median": float(np.median(v["Ap"])),
            "Ap_below_half": float(np.mean(v["Ap"] < 0.5)),
            "Ar_q10_q50_q90": np.quantile(v["Ar"], [0.1, 0.5, 0.9]).tolist(),
            "norm_median": float(np.median(v["norm"])),
            "norm_median_worse": float(np.median(v["norm"][worse])) if worse.any() else None,
            "norm_median_other": float(np.median(v["norm"][~worse])),
            "worse_share_Ap_ge_0.7": float(worse[high].mean()) if high.any() else None,
            "worse_share_Ap_lt_0.7": float(worse[~high].mean()) if (~high).any() else None,
            "identity_within_5pct": float(np.mean(v["dev"] <= 0.05)),
        }
        print(d["cell"], json.dumps(summary), flush=True)
        out[d["cell"]] = {"summary": summary, **{k: x.tolist() for k, x in v.items()}}
    return out


def rounding(files):
    res = {"cell": "mamba2_beta", "bits": 6, "axis": "value", "rotation": "none", "scales": "loop", "sr_replays": len(files)}
    for rnd, use in (("RTN", files[:1]), ("SR", files)):
        num = den = 0
        for f in use:
            d = load(f)
            if (d["cell"], d["axis"]) != ("mamba2_beta", "value"):
                raise ValueError(f"Expected Mamba-2 value-axis replays: {f}")
            ci = d["configs"].index((6, rnd, "none", "loop"))
            acc2, inj2, cross, a2 = (metric(d, n)[:, ci].double() for n in ("acc2", "inj2", "cross", "ret_decay"))
            D = torch.zeros_like(inj2)
            D[0] = inj2[0]
            for t in range(1, len(D)):
                D[t] = a2[t] * D[t - 1] + inj2[t]
            num, den = num + (acc2 + 2 * cross + inj2), den + D
        diff = (num / len(use) - den / len(use)).numpy()
        res[rnd] = {"raw_head_time_mean": np.nanmean(diff, 0).tolist()}
        m = np.array(res[rnd]["raw_head_time_mean"])
        print(f"{rnd}: heads with negative remaining error {np.mean(m < 0):.3f}, median {np.median(m):.3e}", flush=True)
    return res


def main():
    from lores.evaluation.output_store import create_run

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", required=True, choices=("inflation", "alignment", "rounding"))
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()
    torch.set_num_threads(8)
    res = {"inflation": inflation, "alignment": alignment, "rounding": rounding}[args.kind](args.files)
    run = create_run(
        "diagnostics",
        "all-models",
        f"scale-feedback-{args.kind}",
        {"files": args.files},
        namespace=("scale-feedback",),
        root=args.output_root,
    )
    path = run / f"{args.kind}.json"
    path.write_text(json.dumps(res))
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
