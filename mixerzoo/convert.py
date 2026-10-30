"""Convert each selected MixerZoo cell's latest DCP checkpoint to Hugging Face format."""

import argparse
import io
import os
import re
import tempfile
from datetime import timedelta

import fla  # noqa: F401
import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import mixerzoo  # noqa: F401  -- registers model_type `mixer_zoo` with AutoModel
from lores.models.registry import MIXER_ZOO_CELLS

HERE = os.path.dirname(__file__)
OUT_ROOT = "models/mixerzoo_1.3B"


def _final_step_dir(cell: str, checkpoint_root: str) -> tuple[int, str]:
    ckpt = f"{checkpoint_root}/mixerzoo_{cell}_1.3B/checkpoint"
    steps = []
    for name in os.listdir(ckpt):
        m = re.fullmatch(r"step-(\d+)", name)
        if m and os.path.isdir(os.path.join(ckpt, name)):
            steps.append(int(m.group(1)))
    assert steps, f"{cell}: no step-* dirs under {ckpt}"
    step = max(steps)
    return step, os.path.join(ckpt, f"step-{step}")


def convert(cell: str, *, checkpoint_root: str, tokenizer: str, out_root: str = OUT_ROOT, tmp_root: str | None = None) -> None:
    cfg = AutoConfig.from_pretrained(f"{HERE}/configs/{cell}.json", trust_remote_code=True)
    step, dcp = _final_step_dir(cell, checkpoint_root)
    out = f"{out_root}/{cell}"
    assert os.path.isdir(dcp), f"{cell}: missing DCP {dcp}"
    os.makedirs(out, exist_ok=True)

    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        pt = os.path.join(tmp, "checkpoint.pt")
        dcp_to_torch_save(dcp, pt)
        model = AutoModelForCausalLM.from_config(cfg)
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        state = torch.load(pt, map_location="cpu")["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f"{cell}: unexpected keys {unexpected[:5]}"
        assert all("lm_head" in k for k in missing), f"{cell}: missing non-lm_head keys {missing[:5]}"

        # break the shared-storage tie so safetensors can serialise; loader re-ties via config.
        emb = model.get_input_embeddings().weight
        if model.lm_head.weight.data_ptr() == emb.data_ptr():
            model.lm_head.weight = torch.nn.Parameter(emb.detach().clone())

        # Probe a BF16 copy: recurrent kernels require CUDA; save the original FP32 weights.
        if torch.cuda.is_available():
            import copy

            probe = copy.deepcopy(model).to(torch.bfloat16).eval().cuda()
            with torch.no_grad():
                ids = (torch.arange(16).unsqueeze(0) % cfg.vocab_size).cuda()
                logits = probe(ids).logits
            assert torch.isfinite(logits).all(), f"{cell}: non-finite logits"
            del logits, probe
            torch.cuda.empty_cache()

        model.save_pretrained(out)

    AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True).save_pretrained(out)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"OK {cell:18s} step-{step:<7d} {cfg.mixer_family}/{cfg.decay:8s} {n:.1f}M -> {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert final-step MixerZoo checkpoints to HF.")
    parser.add_argument("cells", nargs="*")
    parser.add_argument("--checkpoint-root", required=True, help="directory holding mixerzoo_<cell>_1.3B/checkpoint/step-*")
    parser.add_argument("--tokenizer", required=True, help="tokenizer directory saved with the training run")
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--tmp-root", default=None, help="scratch directory for the torch checkpoint (default: system temp)")
    args = parser.parse_args()
    if set(args.cells) - set(MIXER_ZOO_CELLS):
        parser.error(f"unknown cells: {sorted(set(args.cells) - set(MIXER_ZOO_CELLS))}")
    for c in args.cells or MIXER_ZOO_CELLS:
        convert(c, checkpoint_root=args.checkpoint_root, tokenizer=args.tokenizer, out_root=args.out_root, tmp_root=args.tmp_root)
