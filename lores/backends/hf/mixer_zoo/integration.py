"""MixerZoo registration and layer iteration."""

from __future__ import annotations

__all__ = ["iter_mixer_zoo_layers", "unified_mixer_cls"]


def unified_mixer_cls():
    """The vendored ``UnifiedMixer`` class (importing ``mixerzoo`` registers model_type ``mixer_zoo`` with AutoModel)."""
    from mixerzoo.models import UnifiedMixer

    return UnifiedMixer


def iter_mixer_zoo_layers(model):
    """Yield every ``UnifiedMixer`` module in ``model`` (i.e. the ``.attn`` of each ``MixerZooBlock``)."""
    cls = unified_mixer_cls()
    for module in model.modules():
        if isinstance(module, cls):
            yield module
