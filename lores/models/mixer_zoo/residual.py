"""Residual-stream rotation and norm folding for the MixerZoo block tree."""

from __future__ import annotations

import torch

from lores.rotations.folding import (
    fold_hadamard_input,
    fuse_rmsnorm_into,
    rotate_input,
    rotate_output,
)


def _mixer_inputs(mix):
    """Residual-facing projections in UnifiedMixer's fixed layout."""
    linears = [mix.q_proj, mix.k_proj, mix.v_proj, mix.g_proj]
    if mix.decay != "none":
        linears.append(mix.gk_proj[0] if mix.decay == "diagonal" else mix.gk_proj)
    if mix._use_beta:
        linears.append(mix.b_proj)
    return linears


def _check_attn_class(model, what: str) -> None:
    """Every ``block.attn`` must be a ``UnifiedMixer``."""
    from lores.backends.hf.mixer_zoo.integration import unified_mixer_cls

    attn_cls = unified_mixer_cls()
    for block in model.model.layers:
        if not isinstance(block.attn, attn_cls):
            raise NotImplementedError(
                f"{what}: block.attn is {type(block.attn).__name__}, not {attn_cls.__name__} — "
                f"hybrid attention layers are not handled",
            )


@torch.no_grad()
def fuse_norms(model) -> None:
    """Fold norm scales and mean-center embeddings (the latter changes the model)."""
    _check_attn_class(model, "residual rotation (norm fusion)")
    emb = model.model.embeddings.weight.data.double()
    model.model.embeddings.weight.data = (emb - emb.mean(dim=-1, keepdim=True)).to(model.model.embeddings.weight.dtype)
    for block in model.model.layers:
        fuse_rmsnorm_into(block.attn_norm, _mixer_inputs(block.attn))
        fuse_rmsnorm_into(block.mlp_norm, (block.mlp.gate_proj, block.mlp.up_proj))
    fuse_rmsnorm_into(model.model.norm, [model.lm_head])


@torch.no_grad()
def rotate_residual(
    model,
    R1: torch.Tensor | None,
    *,
    rotate_r4: bool = True,
    rotate_o: bool = False,
    fold_r1: bool = True,
) -> None:
    """Fold R1 into residual-stream weights in place, after ``fuse_norms``.

    Optional R4/R_o folds act on down_proj/o_proj inputs and require matching
    online activation Hadamards, installed separately after quantization wrapping.
    ``fold_r1=False`` applies only those input folds; ``R1`` may then be None.
    """
    _check_attn_class(model, "residual rotation (R1 fold)")
    if fold_r1:
        assert R1 is not None
        R1 = R1.to(device=model.model.embeddings.weight.device)
        rotate_input(model.model.embeddings, R1)  # Embedding.weight is [vocab, hidden], like a Linear's [out, in]
        rotate_input(model.lm_head, R1)  # after model.norm was folded into it by fuse_norms
    for block in model.model.layers:
        if fold_r1:
            for lin in _mixer_inputs(block.attn):
                rotate_input(lin, R1)
            for lin in (block.mlp.gate_proj, block.mlp.up_proj):
                rotate_input(lin, R1)
        for lin in (block.attn.o_proj,):  # o_proj: R1 on the output dim, R_o on the input dim
            if fold_r1:
                rotate_output(lin, R1)
            if rotate_o:
                fold_hadamard_input(lin)
        for lin in (block.mlp.down_proj,):
            if fold_r1:
                rotate_output(lin, R1)
            if rotate_r4:
                fold_hadamard_input(lin)
