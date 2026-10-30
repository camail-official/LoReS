"""Explicitly selected serving launch tables (FP32 GDN decode, Mamba-2 and GDN INT8 decode)."""

import json
from functools import cache
from pathlib import Path

from lores.backends.vllm.config import current


@cache
def _load(path):
    return json.loads(Path(path).read_text())


def override_table(kind):
    """Read an explicitly selected serving launch table."""
    path = current().decode_tuning
    return ({int(k): v for k, v in _load(path).get(kind, {}).items()} or None) if path else None


def floor_entry(table, batch):
    """The entry at the largest batch key <= ``batch``, or None."""
    for b in sorted(table, reverse=True):
        if b <= batch:
            return tuple(table[b])
    return None


def gdn_decode_config(batch: int, default: tuple[int, int, int]) -> tuple[int, int, int]:
    """(BV, warps, stages) at the batch floor of the selected table; ``default`` (vLLM's launch) without one."""
    table = override_table("gdn_fp")
    return (floor_entry(table, batch) if table else None) or default
