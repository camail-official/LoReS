"""Paper Delta-ell values from a verified Delta-ell inventory.

Each value is rederived from its complete saved FP32 loss pair before plotting.
"""

import json
from functools import cache
from pathlib import Path

from lores.evaluation.loss_traces import read_delta_ell_summary


@cache
def records(inventory):
    return {row["id"]: row for row in json.loads(Path(inventory).read_text())["records"]}


@cache
def median(config_id, inventory):
    row = records(inventory)[config_id]
    if row["status"] != "verified_trace":
        raise ValueError(f"No verified numerical result for {config_id}: {row['status']}")
    value = read_delta_ell_summary(Path(row["run"]) / "summary.json")
    if value != row["dce_median"]:
        raise ValueError(f"Inventory differs from saved traces: {config_id}")
    return value


def promotion_grid(model, inventory):
    """The same nine fixed-mask SR configurations for GSM8K, GPQA and AIME."""
    rounding = "SR" if model == "qwen3_5" else "asymSR"
    mask_rounding = "symSR" if model == "qwen3_5" else "asymSR"
    values = {float(bits): median(f"{model}/INT{bits}-value-{rounding}-none", inventory) for bits in (4, 6, 8)}
    for bits in (4, 6):
        for fraction in (25, 50, 75):
            config = f"{model}/INT{bits}to{bits + 2}-{fraction}pct-value-{mask_rounding}"
            values[bits + 2 * fraction / 100] = median(config, inventory)
    return values
