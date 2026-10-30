"""Export rotation-aware compressed-tensors checkpoints for vLLM."""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lores.models.registry import HYBRID_MODELS, checkpoint_for, model_info
from lores.wa_ptq.export import CALLEN, NCAL, calibrate, prepared_model  # noqa: E402

SCHEMES = ("W4A16", "W8A16", "W8A8", "W4A8")


def unwrap(model) -> int:
    """Replace every ActQuantWrapper / SignedInputWrapper by the Linear it wraps; returns the count."""
    from lores.rotations.wrappers import SignedInputWrapper
    from lores.wa_ptq.quantizers import ActQuantWrapper

    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            inner = child
            while isinstance(inner, (ActQuantWrapper, SignedInputWrapper)):
                inner = inner.module if isinstance(inner, ActQuantWrapper) else inner.inner
            if inner is not child:
                setattr(parent, name, inner)
                n += 1
    return n


def _kimi_calibrate_all_experts(model) -> None:
    """Calibrate every Kimi expert on all tokens while retaining routed output weights."""
    moe = next(m for m in model.modules() if type(m).__name__ == "KimiSparseMoeBlock")

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        out = torch.zeros_like(x, dtype=torch.float32)
        for e, expert in enumerate(self.experts):
            w = (topk_weight * (topk_ids == e)).sum(-1)  # [T], 0 for tokens not routed here
            out += expert(x).float() * w.float()[:, None]
        return out.to(x.dtype)

    type(moe).moe_infer = moe_infer
    print(f"[wa_export_vllm] kimi: all-token expert calibration on {type(moe).__name__}", flush=True)


def main():
    """Prepare, GPTQ-quantize with the preset, unwrap, save compressed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(HYBRID_MODELS))
    ap.add_argument("--scheme", required=True, choices=SCHEMES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=None, help="override the base checkpoint (tiny-model dry runs)")
    ap.add_argument("--tok", default=None, help="tokenizer id/path when it differs from the base (tiny dry runs)")
    args = ap.parse_args()
    checkpoint = args.base or checkpoint_for(args.arch)
    tag = model_info(args.arch)["prefix"]
    out = args.out or f"models/hf/{tag}-rot-{args.scheme.lower()}"
    from transformers import AutoTokenizer

    from lores.wa_ptq.recipes import BEST_STACK

    kimi = args.arch == "kimi_linear"
    tok = AutoTokenizer.from_pretrained(args.tok or checkpoint, trust_remote_code=kimi)
    model = prepared_model(args.arch, device="cpu" if kimi else "cuda", checkpoint=checkpoint)
    if kimi:
        _kimi_calibrate_all_experts(model)
    folded = BEST_STACK[args.arch]
    print(f"[wa_export_vllm] prepared {type(model).__name__} with {folded}", flush=True)
    calibrate(model, tok, args.scheme, ignore=["lm_head", "re:.*lm_head.*", "re:.*visual.*", "re:.*vision.*"])
    n = unwrap(model)
    print(f"[wa_export_vllm] unwrapped {n} activation wrappers", flush=True)
    assert not any(type(m).__name__ in ("ActQuantWrapper", "SignedInputWrapper") for m in model.modules())
    if args.arch == "qwen3_5":
        # vLLM only knows the stock multimodal container: transplant the quantized text backbone
        # (module objects keep their quantization schemes) into Qwen3_5ForConditionalGeneration.
        from llmcompressor.transformers.compression.compressed_tensors_utils import (
            modify_save_pretrained,
        )
        from transformers import Qwen3_5ForConditionalGeneration

        full = Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.bfloat16)
        full.model.language_model = model.model  # ForCausalLM.model IS the text model; keys get the prefix on save
        full.lm_head = model.lm_head
        modify_save_pretrained(full)
        model = full
    model.config.linq_wa_rot = "r4s,ro" if args.arch == "nemotronh" else "r4,ro"
    os.makedirs(out, exist_ok=True)
    # vLLM requires packed W4 and Kimi W8A16 experts; W8A8 stays int-quantized.
    fmt = {"quantization_format": "pack-quantized"} if args.scheme.startswith("W4") or (kimi and args.scheme == "W8A16") else {}
    model.save_pretrained(out, save_compressed=True, safe_serialization=True, **fmt)
    tok.save_pretrained(out)
    if kimi:  # remote code + tiktoken vocab + chat template: save_pretrained writes none of them
        import shutil

        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(checkpoint, local_files_only=True))
        for f in list(src.glob("*.py")) + list(src.glob("*.jinja")) + [src / "tiktoken.model", src / "generation_config.json"]:
            if f.exists() and not (Path(out) / f.name).exists():
                shutil.copy(f, Path(out) / f.name)
    if args.arch == "qwen3_5":  # vLLM's multimodal loader needs the (stock) image/video processor configs too
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(checkpoint).save_pretrained(out)
    stamp = {
        "quantizer": "llmcompressor",
        "scheme": args.scheme,
        "folded": folded,
        "ncal": NCAL,
        "callen": CALLEN,
        "arch": args.arch,
        "base": checkpoint,
        "online": model.config.linq_wa_rot,
    }
    (Path(out) / "lores_stamp.json").write_text(__import__("json").dumps(stamp, indent=1))
    print(f"[wa_export_vllm] saved {args.scheme} -> {out}\n[wa_export_vllm] stamp {stamp}", flush=True)


if __name__ == "__main__":
    main()
