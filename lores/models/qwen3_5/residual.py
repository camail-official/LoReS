"""Qwen3.5 residual-stream rotation and norm folding; the vision tower is untouched."""

from __future__ import annotations

import torch

from lores.backends.hf.qwen3_5.integration import is_linear_layer, layer_mixer, text_backbone

_GDN_IN, _GDN_OUT = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), ("out_proj",)


_ATTN_IN, _ATTN_OUT = ("q_proj", "k_proj", "v_proj"), ("o_proj",)


_MLP_IN, _MLP_OUT = ("gate_proj", "up_proj"), ("down_proj",)


def _mixer_io(layer):
    """``(in_projection_names, out_projection_names)`` of a layer's mixer, by type."""
    return (_GDN_IN, _GDN_OUT) if is_linear_layer(layer) else (_ATTN_IN, _ATTN_OUT)


def _fuse_qwen3_5_rmsnorm_into(norm: torch.nn.Module, linears) -> None:
    """Fold Qwen RMSNorm's (1 + weight) scale into its readers, then zero the weight."""
    s = 1.0 + norm.weight.data.double()
    for lin in linears:
        lin.weight.data = (lin.weight.data.double() * s).to(lin.weight.dtype)
    norm.weight.data = torch.zeros_like(norm.weight.data)


@torch.no_grad()
def fuse_qwen3_5_norms(model) -> None:
    """Fold both residual norms per layer and mean-center embeddings.

    Leave the vision tower, internal gated norms, and attention Q/K norms untouched.
    """
    bb = text_backbone(model)
    emb = bb.embed_tokens.weight.data.double()
    bb.embed_tokens.weight.data = (emb - emb.mean(dim=-1, keepdim=True)).to(bb.embed_tokens.weight.dtype)
    for layer in bb.layers:
        mix = layer_mixer(layer)
        in_paths, _ = _mixer_io(layer)
        _fuse_qwen3_5_rmsnorm_into(layer.input_layernorm, [getattr(mix, p) for p in in_paths])
        _fuse_qwen3_5_rmsnorm_into(layer.post_attention_layernorm, [getattr(layer.mlp, p) for p in _MLP_IN])
    _fuse_qwen3_5_rmsnorm_into(bb.norm, [model.lm_head])


@torch.no_grad()
def rotate_qwen3_5_residual(model, R1: torch.Tensor, *, fold_r1: bool = True) -> None:
    """Fold R1 into text embeddings, lm_head, mixer projections, and MLP projections.

    Run after norm fusion. R4 and R_o are separate; fold_r1=False is a no-op.
    """
    if not fold_r1:
        return
    from lores.rotations.folding import rotate_input, rotate_output

    bb = text_backbone(model)
    R1 = R1.to(device=bb.embed_tokens.weight.device)
    rotate_input(bb.embed_tokens, R1)  # Embedding.weight [vocab, hidden] folds like a Linear's [out, in]
    rotate_input(model.lm_head, R1)  # bb.norm already folded into it by fuse_qwen3_5_norms
    for layer in bb.layers:
        mix = layer_mixer(layer)
        in_paths, out_paths = _mixer_io(layer)
        for p in in_paths:
            rotate_input(getattr(mix, p), R1)
        for p in out_paths:
            rotate_output(getattr(mix, p), R1)
        for p in _MLP_IN:
            rotate_input(getattr(layer.mlp, p), R1)
        for p in _MLP_OUT:
            rotate_output(getattr(layer.mlp, p), R1)
