"""vLLM state and W/A rotation setup; exported W/A rotations remain independent of state settings."""

import re

import torch
from torch import nn

from lores.backends.vllm.config import current as linq_config
from lores.kernels.rotations.hadamard import hadamard
from lores.rotations.matrices import r4_signs
from lores.utils import hadamard_utils


class OnlineHadamard(nn.Module):
    """FP32 DAO Hadamard with setup outside compiled forwards."""

    def __init__(self, n: int, sign: torch.Tensor | None = None):
        super().__init__()
        with torch.device("cpu"):  # hadamard_utils builds on the default device (CUDA under vLLM)
            had_K, K = hadamard_utils.get_hadK(n)
        dev = torch.get_default_device()
        if dev.type != "cuda" and torch.cuda.is_available():
            dev = torch.device("cuda")
        self.K = K
        self.n = n
        self._sqrt_n = float(torch.tensor(n).sqrt())
        self.register_buffer("had_K", (had_K.float() if had_K is not None else torch.zeros(0)).to(dev), persistent=False)
        self.register_buffer("sign", (sign if sign is not None else torch.zeros(0)).to(dev), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sign.numel():
            x = x * self.sign.to(x.dtype)
        X = x.float()
        n = X.shape[-1]
        if self.K == 1:
            out = hadamard(X.contiguous()) / self._sqrt_n
        else:
            inp = hadamard(X.reshape(-1, self.K, n // self.K).contiguous()) / self._sqrt_n
            out = (self.had_K @ inp).reshape(X.shape)
        return out.to(x.dtype)


def wa_rot(kind: str) -> bool:
    """True when 'r4' (either flavour) or 'ro' is enabled."""
    wa = linq_config().wa_rot
    return kind in wa or (kind == "r4" and "r4s" in wa)


def layer_index(prefix: str) -> int:
    """The layer index in a serving layer prefix ('...layers.N...')."""
    m = re.search(r"layers\.(\d+)", prefix)
    assert m, f"LINQ: no layer index in prefix {prefix!r}"
    return int(m.group(1))


def r4_sign(n: int, prefix: str) -> torch.Tensor:
    """Resolve the serving layer prefix to the same R4 signs folded into its checkpoint."""
    return r4_signs(n, layer_index(prefix))


def _no_tp() -> None:
    from vllm.distributed import get_tensor_model_parallel_world_size

    assert get_tensor_model_parallel_world_size() == 1, "linq.wa_rot: the Hadamard needs the unsharded input (TP=1)"


def make_r4(n: int, prefix: str) -> OnlineHadamard | None:
    """The down_proj-input Hadamard for LinqConfig.wa_rot, or None when off."""
    if not wa_rot("r4"):
        return None
    _no_tp()
    return OnlineHadamard(n, r4_sign(n, prefix) if wa_rot("r4s") else None)


def make_ro(n: int) -> OnlineHadamard | None:
    """The recurrent-mixer out_proj-input Hadamard for LinqConfig.wa_rot, or None when off."""
    if not wa_rot("ro"):
        return None
    _no_tp()
    return OnlineHadamard(n)


# Routed experts share an unsigned R4; per-layer signed R4 is not supported here.
_LINQ_R4 = None


def linq_set_expert_r4(h) -> None:
    """Bind the unsigned Hadamard shared by routed experts of the same width."""
    global _LINQ_R4
    _LINQ_R4 = h


def apply_expert_r4(output: torch.Tensor) -> None:
    """Rotate routed-expert activations in place before their second GEMM, when enabled."""
    if _LINQ_R4 is not None and output.shape[-1] == _LINQ_R4.n:
        output.copy_(_LINQ_R4(output))


def make_channel_spec(segments, dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-channel signs/log2(head_dim) from (offset, width, head_dim, signs) segments."""
    signs = torch.zeros(dim, dtype=torch.float32, device=device)
    l2d = torch.zeros(dim, dtype=torch.int8, device=device)
    for off, width, head_dim, seg_signs in segments:
        assert head_dim & (head_dim - 1) == 0 and 32 <= head_dim <= 128
        assert off % head_dim == 0, "heads must be 2^k-aligned in the channel axis"
        assert width % head_dim == 0
        k = head_dim.bit_length() - 1
        l2d[off : off + width] = k
        s = seg_signs.to(torch.float32).flatten()
        assert s.numel() == head_dim
        signs[off : off + width] = s.repeat(width // head_dim)
    return signs, l2d


def linq_rot_any() -> bool:
    """True when the state-side rotation (rk) is enabled."""
    return bool(linq_config().rot)


def factor(head_dim: int, device):
    """Sign vector for R = diag(signs) @ H_d / sqrt(d), seeded by head_dim."""
    g = torch.Generator(device="cpu").manual_seed(head_dim)
    # vLLM may set a CUDA default device; this draw uses a CPU generator.
    bits = torch.randint(0, 2, (head_dim,), generator=g, dtype=torch.int8, device="cpu")
    signs = bits * 2 - 1
    return signs.to(device=device, dtype=torch.float32)


def mamba2_spec(mixer):
    """Cache rotation spec for x || B || C: the same R_k on B and C, x untouched."""
    spec = getattr(mixer, "_linq_conv_spec", None)
    if spec is not None:
        return spec

    dim = mixer.conv1d.weight.size(0)
    dev = mixer.conv1d.weight.device
    inter = mixer.intermediate_size // mixer.tp_size
    bc = mixer.n_groups * mixer.ssm_state_size // mixer.tp_size
    n = mixer.ssm_state_size
    signs, log2dim = make_channel_spec([(inter, 2 * bc, n, factor(n, dev))], dim, dev)
    spec = (signs, log2dim, 0)  # rot_d = 0: the x channels are not rotated, so no uniform fast path
    mixer._linq_conv_spec = spec
    return spec
