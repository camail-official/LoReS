"""Load completed timing runs for serving plots and tables."""

import json
import statistics
from pathlib import Path

import numpy as np

from lores.evaluation.output_store import sha256

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def load_timings(srcs, sweep, aggregation="ratio-of-medians"):
    paths = [Path(src) for src in ([srcs] if isinstance(srcs, (str, Path)) else srcs)]
    if not paths:
        raise ValueError("Provide completed timing run directories")
    if aggregation not in ("ratio-of-medians", "median-paired"):
        raise ValueError("Unknown timing aggregation")
    data = {}
    timing_mode = None
    for path in paths:
        if (path / "invalid.json").exists():
            raise ValueError(f"Invalid timing run: {path}")
        config = json.loads((path / "run.json").read_text())["config"]
        mode = (config.get("state_fast", False), config.get("async_scheduling", False), bool(config.get("no_wa_rot")))
        if timing_mode is not None and mode != timing_mode:
            raise ValueError("Do not combine different FAST, scheduling, or W/A rotation modes")
        timing_mode = mode
        complete = json.loads((path / "complete.json").read_text())
        if sha256(path / "summary.json") != complete["summary_sha256"]:
            raise ValueError(f"Summary checksum mismatch: {path}")
        if (
            config["sweep"] != sweep
            or any(config.get(k, False) for k in ("pilot", "functional_check", "preflight"))
            or config["repetitions"] != 7
        ):
            raise ValueError(f"Not a complete seven-repetition {sweep} measurement: {path}")
        weight = config["weight"] + ("p" if config.get("pinned", False) else "")
        for row in json.loads((path / "summary.json").read_text()):
            if sweep == "concurrency" and row["length"] != 2:
                raise ValueError(f"Unexpected concurrency-sweep prompt length: {path}")
            for arm in ("base", "int8"):
                if aggregation == "ratio-of-medians":
                    value = row["median_tpot_ms"][arm]
                else:
                    repetitions = [
                        json.loads((path / f"{arm}_r{rep}" / f"T{row['length']}_c{row['concurrency']}.json").read_text())[
                            "mean_tpot_ms"
                        ]
                        for rep in range(1, 8)
                    ]
                    value = repetitions if aggregation == "median-paired" else statistics.mean(repetitions)
                key = config["model"], weight, arm, row["length"], row["concurrency"]
                if key in data:
                    raise ValueError(f"Duplicate timing cell: {key}")
                data[key] = value
    return data


def load_concurrency(srcs, aggregation="ratio-of-medians"):
    runs = load_timings(srcs, "concurrency", aggregation)
    return {(f"{model}_{weight}", arm, c): value for (model, weight, arm, _, c), value in runs.items()}


def load_context(srcs, aggregation="ratio-of-medians"):
    return load_timings(srcs, "context", aggregation)


def gain(base, quantized, aggregation="ratio-of-medians"):
    base, quantized = np.asarray(base), np.asarray(quantized)
    ratio = base / quantized
    if aggregation == "median-paired":
        ratio = np.median(ratio, axis=-1)
    return (ratio - 1) * 100
