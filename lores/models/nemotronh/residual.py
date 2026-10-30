"""Nemotron-H residual-stream rotation and norm folding."""

from __future__ import annotations

import torch

_MAMBA_IN, _MAMBA_OUT = ("in_proj",), ("out_proj",)


_MLP_IN, _MLP_OUT = ("up_proj",), ("down_proj",)


_ATTN_IN, _ATTN_OUT = ("q_proj", "k_proj", "v_proj"), ("o_proj",)


def _block_io(block):
    """``(in_paths, out_paths)`` for a ``NemotronHBlock`` by inspecting which linears its mixer carries."""
    mix = block.mixer
    if hasattr(mix, "in_proj"):
        return _MAMBA_IN, _MAMBA_OUT
    if hasattr(mix, "up_proj"):
        return _MLP_IN, _MLP_OUT
    if hasattr(mix, "q_proj"):
        return _ATTN_IN, _ATTN_OUT
    raise NotImplementedError(f"NemotronH: unrecognized mixer {type(mix).__name__}")


@torch.no_grad()
def fuse_nemotronh_norms(model) -> None:
    """Fold residual norms into their readers and mean-center embeddings.

    Leave internal Mamba gated norms untouched. Run before residual rotation.
    """
    from lores.rotations.folding import fuse_rmsnorm_into

    bb = model.model
    emb = bb.embeddings.weight.data.double()
    bb.embeddings.weight.data = (emb - emb.mean(dim=-1, keepdim=True)).to(bb.embeddings.weight.dtype)
    for block in bb.layers:
        in_paths, _ = _block_io(block)
        fuse_rmsnorm_into(block.norm, [getattr(block.mixer, p) for p in in_paths])
    fuse_rmsnorm_into(bb.norm_f, [model.lm_head])


@torch.no_grad()
def rotate_nemotronh_residual(model, R1: torch.Tensor, *, fold_r1: bool = True) -> None:
    """Fold R1 into embeddings, lm_head, and residual-facing projections after norm fusion.

    R4 and R_o are handled separately; fold_r1=False leaves weights unchanged.
    """
    if not fold_r1:
        return
    from lores.rotations.folding import rotate_input, rotate_output

    bb = model.model
    R1 = R1.to(device=bb.embeddings.weight.device)
    rotate_input(bb.embeddings, R1)  # Embedding.weight [vocab, hidden] folds like a Linear's [out, in]
    rotate_input(model.lm_head, R1)  # norm_f already folded into it by fuse_nemotronh_norms
    for block in bb.layers:
        in_paths, out_paths = _block_io(block)
        for p in in_paths:
            rotate_input(getattr(block.mixer, p), R1)
        for p in out_paths:
            rotate_output(getattr(block.mixer, p), R1)
