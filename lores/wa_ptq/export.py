"""Shared rotation-aware llm-compressor preparation and calibration."""

import sys

import torch

NCAL, CALLEN, SEED, GROUPSIZE = 128, 2048, 0, 128


def prepared_model(arch, *, device="cuda", checkpoint=None, rotations=True):
    """Install rotations and activation wrappers before weight calibration."""
    from lores.backends.hf.loading import load_export_model
    from lores.wa_ptq.prepare import prepare_model
    from lores.wa_ptq.recipes import make_wa_config

    model = load_export_model(arch, device=device, checkpoint=checkpoint)
    cfg = make_wa_config(arch, device="cuda" if device == "auto" else device)  # rotations/wrappers only
    if not rotations:
        from lores.config import RotationConfig

        cfg.rotation = RotationConfig(residual=False, r4=False, o=False, r4_signs=False)
    prepare_model(cfg, model)
    return model


def calibrate(model, tokenizer, scheme, *, ignore):
    """Run sequential GPTQ on the fixed WikiText-2 train calibration sample."""
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from typing_extensions import Unpack

    from lores.evaluation.data import get_wikitext2

    calibration = get_wikitext2(
        tokenizer=tokenizer,
        nsamples=NCAL,
        seed=SEED,
        seqlen=CALLEN,
        eval_mode=False,
    )
    rows = [{"input_ids": sample[0].reshape(-1).tolist(), "attention_mask": [1] * CALLEN} for sample in calibration]
    # Re-executed transformer source needs Unpack in its module globals.
    for name in {type(module).__module__ for module in model.modules()}:
        module = sys.modules.get(name)
        if module is not None and not hasattr(module, "Unpack"):
            module.Unpack = Unpack
    recipe = GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore)
    model.cpu()
    torch.cuda.empty_cache()
    oneshot(
        model=model,
        recipe=recipe,
        dataset=Dataset.from_list(rows),
        processor=tokenizer,
        max_seq_length=CALLEN,
        num_calibration_samples=NCAL,
        shuffle_calibration_samples=False,
        pipeline="sequential",
    )
