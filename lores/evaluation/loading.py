"""Delta-ell model setup: pinned checkpoints, FP32 state, and selected rotations."""

from functools import partial

import torch

from lores.backends.hf.loading import load_hybrid
from lores.backends.hf.state import ROTATIONS, install_rotations, pin_fp32_state, state_installer


def load_delta_ell_model(arch, rotation="none", *, revision=None, block_size=128):
    """Load a revision-pinned model with FP32 state from its first store."""
    tok, model = load_hybrid(arch, pin_revision=True, revision=revision)
    pin_fp32_state(model, arch, strict=True)
    if arch == "nemotronh" and rotation != "none":
        with torch.no_grad():
            model(torch.tensor([[1, 2, 3]], device="cuda"))
    install_rotations(model, arch, rotation)
    return tok, model, state_installer(model, arch, block_size=block_size)


DELTA_ELL_MODELS = {arch: (partial(load_delta_ell_model, arch), rots) for arch, rots in ROTATIONS.items()}
