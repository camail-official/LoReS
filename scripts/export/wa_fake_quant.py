"""Export rotation-aware llm-compressor fake-quant weights as a .pt artifact."""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lores.models.registry import HYBRID_MODELS, checkpoint_for
from lores.wa_ptq.export import CALLEN, GROUPSIZE, NCAL, calibrate, prepared_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="qwen3_5", choices=list(HYBRID_MODELS))
    ap.add_argument("--w-bits", type=int, required=True, choices=[4, 8])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    scheme = f"W{args.w_bits}A16"
    out = args.out or f"models/{args.arch}_w{args.w_bits}g{GROUPSIZE}_r1_llmc.pt"

    from transformers import AutoTokenizer

    from lores.wa_ptq.recipes import BEST_STACK

    tok = AutoTokenizer.from_pretrained(checkpoint_for(args.arch))
    model = prepared_model(args.arch)
    print(f"[wa_export] prepared {type(model).__name__} with {BEST_STACK[args.arch]}", flush=True)

    calibrate(model, tok, scheme, ignore=["lm_head", "re:.*lm_head.*", "re:.*visual.*", "re:.*vision.*"])

    # Ensure GPTQ baked its quantized values into weights, not only a QDQ hook.
    probe = next(m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n and "vis" not in n)
    nuniq = int(torch.unique(probe.weight[0, :GROUPSIZE].float()).numel())
    print(f"[wa_export] guard: first group has {nuniq} distinct values (<= {2**args.w_bits})", flush=True)
    assert nuniq <= 2**args.w_bits, f"weights not actually quantized ({nuniq} distinct values)"

    from lores.wa_ptq.weights import _weights_only

    stamp = {
        "quantizer": "llmcompressor",
        "scheme": scheme,
        "groupsize": GROUPSIZE,
        "folded": BEST_STACK[args.arch],
        "ncal": NCAL,
        "callen": CALLEN,
        "arch": args.arch,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.tmp{os.getpid()}"  # atomic: a preempted job must not leave a loadable partial
    # Quantized values are baked into weights; discard runtime quantizer metadata.
    sd = {
        k: v
        for k, v in _weights_only(model.state_dict()).items()
        if not k.endswith(("weight_scale", "weight_zero_point", "weight_g_idx", "weight_global_scale"))
    }
    torch.save({"stamp": stamp, "state_dict": sd}, tmp)
    os.replace(tmp, out)
    print(f"[wa_export] saved {scheme} g{GROUPSIZE} -> {out}\n[wa_export] stamp {stamp}", flush=True)


if __name__ == "__main__":
    main()
