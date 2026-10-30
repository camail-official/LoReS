"""nemotronh W/A preparation targets."""

from __future__ import annotations

from lores.wa_ptq.spec import HybridSpec


def build_ptq_spec(model):
    """Define Mamba/attention/MLP rotation and activation targets, excluding lm_head."""
    from lores.models.nemotronh.residual import fuse_nemotronh_norms, rotate_nemotronh_residual

    layers = model.model.layers

    def down_projs():
        for li, block in enumerate(layers):
            if getattr(block.mixer, "down_proj", None) is not None:
                yield li, block.mixer, "down_proj"

    def o_projs():
        for block in layers:
            op = getattr(block.mixer, "out_proj", None)
            if op is not None:
                yield op

    return HybridSpec(
        label="NemotronH",
        hidden=model.config.hidden_size,
        embed=model.model.embeddings,
        actquant_root=model,
        fuse_norms=fuse_nemotronh_norms,
        rotate_residual=rotate_nemotronh_residual,
        down_projs=down_projs,
        o_projs=o_projs,
    )
