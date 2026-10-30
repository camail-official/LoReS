"""qwen3_5 W/A preparation targets."""

from __future__ import annotations

from lores.wa_ptq.spec import HybridSpec


def build_ptq_spec(model):
    """Define text-backbone rotation and activation targets, excluding lm_head and the vision tower."""
    from lores.backends.hf.qwen3_5.integration import is_linear_layer, layer_mixer, text_backbone
    from lores.models.qwen3_5.residual import fuse_qwen3_5_norms, rotate_qwen3_5_residual

    bb = text_backbone(model)

    def down_projs():
        for li, layer in enumerate(bb.layers):
            yield li, layer.mlp, "down_proj"

    def o_projs():
        for layer in bb.layers:
            if is_linear_layer(layer):
                yield layer_mixer(layer).out_proj

    return HybridSpec(
        label="Qwen3.5",
        hidden=model.config.get_text_config().hidden_size,
        embed=bb.embed_tokens,
        actquant_root=bb,
        fuse_norms=fuse_qwen3_5_norms,
        rotate_residual=rotate_qwen3_5_residual,
        down_projs=down_projs,
        o_projs=o_projs,
    )
