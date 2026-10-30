"""Model identities and architecture detection, without importing integrations."""

from __future__ import annotations

_ARCH_BY_MODEL_TYPE = {
    "nemotron_h": "nemotronh",
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "kimi_linear": "kimi_linear",
    "mixer_zoo": "mixer_zoo",
}


HYBRID_MODELS = {
    "nemotronh": {
        "checkpoint": "nvidia/Nemotron-H-8B-Reasoning-128K",
        "prefix": "nemotronh-8b",
    },
    "qwen3_5": {"checkpoint": "Qwen/Qwen3.5-9B", "prefix": "qwen35-9b"},
    "kimi_linear": {"checkpoint": "moonshotai/Kimi-Linear-48B-A3B-Instruct", "prefix": "kimi-48b"},
}
_ALIASES = {"nemo": "nemotronh", "qwen": "qwen3_5", "kimi": "kimi_linear"}

MIXER_ZOO_CELLS = (
    "softmax",
    "softmax_qknorm",
    "linear_none",
    "linear_none_beta",
    "deltanet",
    "mamba2",
    "mamba2_beta",
    "gated_deltanet",
    "gla",
    "gla_beta",
    "kda",
)


def model_info(name):
    """Resolve a model name or short alias (nemo, qwen, kimi) to one hybrid model identity."""
    return HYBRID_MODELS[_ALIASES.get(name, name)]


def checkpoint_for(name):
    """Return the default HF checkpoint for a hybrid model or runner alias."""
    return model_info(name)["checkpoint"]


def detect_arch(model):
    """Return the supported architecture name, or raise."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    arch = _ARCH_BY_MODEL_TYPE.get(model_type)
    if arch is None:
        raise NotImplementedError(
            f"prepare_model supports NemotronH, Qwen3.5, Kimi-Linear and MixerZoo models (got {type(model).__name__})"
        )
    return arch
