"""HF checkpoint loading for evaluation, W/A preparation, and exports."""

import logging

import torch

from lores.config import PTQConfig
from lores.models.registry import checkpoint_for

log = logging.getLogger("lores")
MODELS_ROOT = "models/mixerzoo_1.3B"
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_hybrid(arch, *, device="cuda", device_map=None, pin_revision=False, revision=None, attn_implementation=None):
    """Load BF16 checkpoints with Kimi compatibility and optional revision pinning."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if arch == "kimi":
        from lores.backends.hf.kimi_linear.integration import ensure_kimi_loadable

        ensure_kimi_loadable()
    name = checkpoint_for(arch)
    options = {"trust_remote_code": arch == "kimi"}
    if pin_revision:
        from transformers.utils.hub import cached_file, extract_commit_hash

        revision = extract_commit_hash(cached_file(name, "config.json", revision=revision), None)
        if not revision:
            raise ValueError("Evaluation requires the resolved checkpoint revision")
    if revision is not None:
        options["revision"] = revision
    tok = AutoTokenizer.from_pretrained(name, **options)
    if attn_implementation is not None:
        options["attn_implementation"] = attn_implementation
    device_map = "auto" if arch == "kimi" else device_map
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device_map, **options).eval()
    if device_map is None:
        model = model.to(device)
    if arch == "kimi":
        from lores.backends.hf.kimi_linear.integration import patch_kimi_model

        patch_kimi_model(model)
    model.config.use_cache = True
    if pin_revision:
        model._lq_checkpoint_identity = {"name_or_path": name, "revision": revision}
    return tok, model


def load_mixerzoo(cell, models_root=MODELS_ROOT, device="cuda"):
    """Register MixerZoo and load its local BF16 checkpoint and tokenizer."""
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lores.backends.hf.mixer_zoo.integration import unified_mixer_cls

    unified_mixer_cls()
    path = str(Path(models_root) / cell)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    return model, AutoTokenizer.from_pretrained(path)


def load_ptq_model(cfg: PTQConfig):
    """Load a checkpoint/tokenizer for PTQ; untie embeddings before folding rotations."""
    import transformers

    tok = cfg.model.access_token
    # Register custom architectures before AutoConfig resolves them.
    cfg_dict, _ = transformers.PretrainedConfig.get_config_dict(cfg.model.name_or_path, token=tok)
    if cfg_dict.get("model_type") == "mixer_zoo":
        from lores.backends.hf.mixer_zoo.integration import unified_mixer_cls

        unified_mixer_cls()  # imports mixerzoo (registers mixer_zoo)
    config = transformers.AutoConfig.from_pretrained(cfg.model.name_or_path, token=tok)
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, config=config, dtype=_DTYPES[cfg.model.dtype], token=tok
    )
    if process_word_embeddings:
        # Separate storage prevents folding R1 twice into tied weights.
        model.lm_head.weight = torch.nn.Parameter(model.get_input_embeddings().weight.data.clone(), requires_grad=False)
        log.info("untied lm_head from embeddings (tie_word_embeddings=True in checkpoint); cloned weight")
    model = model.to(cfg.device).eval()
    model.config.use_cache = False
    model.seqlen = cfg.model.seqlen
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.name_or_path, token=tok, use_fast=True)
    log.info(
        "loaded %s (%s, %d params) + tokenizer",
        cfg.model.name_or_path,
        cfg.model.dtype,
        sum(p.numel() for p in model.parameters()),
    )
    return model, tokenizer


def load_export_model(arch, *, device="cuda", checkpoint=None):
    """Load the BF16 model used by llm-compressor exports and parity checks."""
    import transformers

    if arch == "kimi_linear":
        # Kimi loads on CPU for sequential calibration, or shards for parity checks.
        from lores.backends.hf.kimi_linear.integration import ensure_kimi_loadable, patch_kimi_model

        ensure_kimi_loadable()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint or checkpoint_for(arch),
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=("auto" if device == "auto" else None),
        ).eval()
        patch_kimi_model(model)
    else:
        model = (
            transformers.AutoModelForCausalLM.from_pretrained(checkpoint or checkpoint_for(arch), dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
    return model
