"""Record the per-step GDN operands of Qwen3.5 for ``rotation_scale_feedback --operands``.

Runs the FP32-state model on WikiText-2 evaluation row 0 (prefill one token, then one token per step) and captures what
each GDN mixer hands its recurrent kernel: q and k (L2-normalised as the kernel does, repeated to the value heads), v,
the log decay g and the write gate beta, stored per layer as [T, H, ...] FP32.

    python -m scripts.diagnostics.record_qwen_operands --revision REV --output-root DIR
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F


def main():
    """Entry point."""
    from lores.backends.hf.qwen3_5.integration import iter_qwen3_5_gdn_mixers
    from lores.evaluation.loading import load_delta_ell_model
    from lores.evaluation.output_store import create_run
    from lores.evaluation.teacher_forcing import make_sequences

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()
    out_dir = create_run("diagnostics", "qwen3_5", "operands", vars(args), namespace=("scale-feedback",), root=args.output_root)
    t0 = time.time()
    tok, model, _ = load_delta_ell_model("qwen3_5", "none", revision=args.revision)
    ids = make_sequences(tok, args.seqlen, 8)[:1].to("cuda")
    mixers = list(iter_qwen3_5_gdn_mixers(model))
    buf = [[] for _ in mixers]

    def recorder(i, fn):
        def wrapped(query, key, value, *a, g=None, beta=None, **kw):
            q = F.normalize(query.float(), dim=-1)[0, -1]
            k = F.normalize(key.float(), dim=-1)[0, -1]
            buf[i].append((q.cpu(), k.cpu(), value.float()[0, -1].cpu(), g.float()[0, -1].cpu(), beta.float()[0, -1].cpu()))
            return fn(query, key, value, *a, g=g, beta=beta, **kw)

        return wrapped

    for i, mix in enumerate(mixers):
        mix.recurrent_gated_delta_rule = recorder(i, mix.recurrent_gated_delta_rule)
        mix.chunk_gated_delta_rule = recorder(i, mix.chunk_gated_delta_rule)
    with torch.no_grad():
        cache = model(input_ids=ids[:, :1], use_cache=True, return_dict=True).past_key_values
        for t in range(1, args.seqlen):
            cache = model(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True, return_dict=True).past_key_values
    layers = []
    for steps in buf:
        assert len(steps) == args.seqlen, (len(steps), args.seqlen)
        q, k, v, g, beta = (torch.stack([s[j] for s in steps]) for j in range(5))
        layers.append({"decay": "scalar", "is_delta": True, "q": q, "k": k, "v": v, "g": g, "beta": beta})
    path = os.path.join(out_dir, "qwen3_5_operands.pt")
    torch.save({"cell": "qwen3_5", "seqlen": args.seqlen, "layers": layers}, path)
    print(f"WROTE {path}: {len(layers)} layers, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
