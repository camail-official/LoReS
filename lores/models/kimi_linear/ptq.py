"""kimi_linear W/A preparation targets."""

from __future__ import annotations

from lores.wa_ptq.spec import HybridSpec


def build_ptq_spec(model):
    """Define KDA/MLA and dense/MoE rotation and activation targets; exclude lm_head and the router."""
    from lores.models.kimi_linear.residual import fuse_kimi_norms, rotate_kimi_residual

    bb = model.model

    def down_projs():
        gi = 0  # Stable per-projection sign seed.
        for layer in bb.layers:
            ffn = getattr(layer, "block_sparse_moe", None) or layer.mlp
            experts = list(getattr(ffn, "experts", []))
            holders = [(e, "w2") for e in experts]
            shared = getattr(ffn, "shared_experts", None)
            if shared is not None:
                holders.append((shared, "down_proj"))
            if not experts:
                holders = [(ffn, "down_proj")]
            for parent, attr in holders:
                yield gi, parent, attr
                gi += 1

    def o_projs():
        for layer in bb.layers:
            if getattr(layer, "is_linear_attn", False):
                yield layer.self_attn.o_proj

    return HybridSpec(
        label="Kimi-Linear",
        hidden=model.config.hidden_size,
        embed=bb.embed_tokens,
        actquant_root=bb,
        fuse_norms=fuse_kimi_norms,
        rotate_residual=rotate_kimi_residual,
        down_projs=down_projs,
        o_projs=o_projs,
    )
