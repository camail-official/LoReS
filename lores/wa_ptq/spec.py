"""The contract between W/A preparation and architecture adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from torch import nn


@dataclass
class HybridSpec:
    """Model targets and callbacks needed by the shared hybrid preparation algorithm."""

    label: str
    hidden: int
    embed: nn.Module
    actquant_root: nn.Module
    fuse_norms: Callable[..., None]
    rotate_residual: Callable[..., None]
    down_projs: Callable[[], Iterable[tuple[int, nn.Module, str]]]
    o_projs: Callable[[], Iterable[nn.Module]]
