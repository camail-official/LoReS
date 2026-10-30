"""Kimi-Linear residual-stream rotation and norm folding."""

from __future__ import annotations

import torch


def _kimi_ffn(layer):
    """Return the layer's MoE or dense feed-forward block."""
    return getattr(layer, "block_sparse_moe", None) or layer.mlp


def _reads_residual(mod, hidden: int) -> bool:
    """Select weights whose input dimension equals the residual width."""
    w = getattr(mod, "weight", None)
    return w is not None and w.shape[1] == hidden


def _writes_residual(mod, hidden: int) -> bool:
    """Select weights whose output dimension equals the residual width."""
    w = getattr(mod, "weight", None)
    return w is not None and w.shape[0] == hidden


def _attn_linears(attn):
    """Return the attention mixer's linear projections."""
    return [m for m in attn.modules() if isinstance(m, torch.nn.Linear)]


def _ffn_weight_modules(ffn):
    """Return FFN linears and the MoE router, which owns a weight but is not nn.Linear."""
    mods = [m for m in ffn.modules() if isinstance(m, torch.nn.Linear)]
    gate = getattr(ffn, "gate", None)  # MoE router (weight-only module; reads the residual)
    if gate is not None:
        mods.append(gate)
    return mods


@torch.no_grad()
def fuse_kimi_norms(model) -> None:
    """Fold residual norms into their readers and mean-center embeddings.

    Kimi uses plain weight scaling, not Qwen's (1 + weight). Include the MoE
    router; leave internal KDA/MLA norms untouched. Run before residual rotation.
    """
    bb = model.model
    hidden = model.config.hidden_size
    emb = bb.embed_tokens.weight.data.double()
    bb.embed_tokens.weight.data = (emb - emb.mean(dim=-1, keepdim=True)).to(bb.embed_tokens.weight.dtype)

    from lores.rotations.folding import fuse_rmsnorm_into

    fuse = fuse_rmsnorm_into

    for layer in bb.layers:
        attn_readers = [m for m in _attn_linears(layer.self_attn) if _reads_residual(m, hidden)]
        ffn_readers = [m for m in _ffn_weight_modules(_kimi_ffn(layer)) if _reads_residual(m, hidden)]
        fuse(layer.input_layernorm, attn_readers)
        fuse(layer.post_attention_layernorm, ffn_readers)
    fuse(bb.norm, [model.lm_head])


@torch.no_grad()
def rotate_kimi_residual(model, R1: torch.Tensor, *, fold_r1: bool = True) -> None:
    """Fold R1 into residual-facing weights after norm fusion.

    Select projections by input/output width; leave internal projections alone.
    Move R1 per weight for sharded models. R4 and R_o are handled separately.
    """
    if not fold_r1:
        return
    from lores.rotations.folding import rotate_input, rotate_output

    bb = model.model
    hidden = model.config.hidden_size
    R1 = R1.to(torch.float32)

    # Sharded weights require R1 on each target device.
    def _in(m):
        rotate_input(m, R1.to(m.weight.device))

    def _out(m):
        rotate_output(m, R1.to(m.weight.device))

    _in(bb.embed_tokens)  # Embedding.weight [vocab, hidden] folds like a Linear's [out, in]
    _in(model.lm_head)  # bb.norm already folded into it by fuse_kimi_norms
    for layer in bb.layers:
        for m in _attn_linears(layer.self_attn) + _ffn_weight_modules(_kimi_ffn(layer)):
            if _reads_residual(m, hidden):
                _in(m)
            elif _writes_residual(m, hidden):
                _out(m)
