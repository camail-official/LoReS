"""Direct configuration of the paper's weight/activation rotation recipes."""

from __future__ import annotations

from dataclasses import replace

from lores.config import PTQConfig, QuantConfig, RotationConfig

W_GROUPSIZE = 128

# Exported weights record this flag list, in this order, in their stamp (``folded``).
_ROTATION_FLAGS = (
    ("residual", "--rotate_residual"),
    ("r4", "--rotate_r4"),
    ("o", "--rotate_o"),
    ("r4_signs", "--r4_signs"),
)
_WA_ROTATIONS = {
    "qwen3_5": RotationConfig(residual=True, r4=True, o=True),
    "nemotronh": RotationConfig(residual=True, r4=True, o=True, r4_signs=True),
    "kimi_linear": RotationConfig(residual=True, r4=True, o=True),
}
BEST_STACK = {
    arch: [flag for field, flag in _ROTATION_FLAGS if getattr(rotation, field)] for arch, rotation in _WA_ROTATIONS.items()
}


def make_wa_config(arch: str, *, a_bits: int = 16, device: str = "cuda") -> PTQConfig:
    """Build a fresh W/A recipe without depending on the command-line parser.

    Copies rotation settings because preparation resolves their tri-state defaults
    in place. Weights come from llm-compressor exports, so only activations are configured.
    """
    return PTQConfig(rotation=replace(_WA_ROTATIONS[arch]), quant=QuantConfig(a_bits=a_bits), device=device)
