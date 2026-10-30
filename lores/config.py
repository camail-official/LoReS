"""PTQ configuration. Bit-width 16 disables quantization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Checkpoint loading."""

    name_or_path: str = "nvidia/Nemotron-H-8B-Reasoning-128K"
    dtype: str = "bfloat16"
    seqlen: int = 2048
    access_token: str | None = None


@dataclass
class RotationConfig:
    """None selects architecture defaults; an unset R4 follows residual rotation."""

    o: bool | None = None  # output-projection input Hadamard
    residual: bool | None = None  # R1 residual-stream rotation
    r4: bool | None = None  # down-projection input Hadamard
    r4_signs: bool = False

    seed: int = 0


@dataclass
class QuantConfig:
    """Weight and activation quantization."""

    w_bits: int = 16
    w_groupsize: int = -1  # -1 = per-output-channel
    w_clip: bool = False

    a_bits: int = 16
    a_groupsize: int = -1  # -1 = per-token; >0 = group size
    a_sym: bool = True

    calib_nsamples: int = 128  # WikiText-2 training samples


@dataclass
class EvalConfig:
    """lm-eval settings."""

    tasks: str = "wikitext"
    batch_size: int | str = 16
    limit: int | None = None


@dataclass
class PTQConfig:
    """PTQ pipeline settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    device: str = "cuda"
    seed: int = 0
