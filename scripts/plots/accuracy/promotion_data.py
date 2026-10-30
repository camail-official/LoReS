"""Read explicit grading-summary manifests for promotion and rotation figures."""

import json
from pathlib import Path

from scripts.plots.shared.style import PALETTE

C_GSM, C_GPQA, C_AIME = PALETTE[2], PALETTE[1], PALETTE[0]
COLORS = {"GSM8K": C_GSM, "GPQA": C_GPQA, "AIME": C_AIME}


def read_accuracy(path, parent):
    """Read one score; relative summary paths are relative to their manifest."""
    return json.loads((parent / path).read_text())["accuracy"]


def load_summaries(path):
    """Read scores and derive minimum-bit markers at 95% of FP32 accuracy."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    data, thresholds = {}, {}
    for model in ("Qwen-3.5", "Nemotron-H"):
        data[model], thresholds[model] = {}, {}
        for task, color in COLORS.items():
            scores = {key: read_accuracy(value, path.parent) for key, value in manifest[model][task].items()}
            reference = scores.pop("fp32")
            anchors = {float(key): value for key, value in scores.items()}
            if set(anchors) != {4 + i / 2 for i in range(9)}:
                raise ValueError(f"Incomplete promotion grid: {model}/{task}")
            data[model][task] = color, anchors, reference
            thresholds[model][task] = next(
                (x for x in sorted(anchors) if all(v >= 0.95 * reference for b, v in anchors.items() if b >= x)), None
            )
    return data, thresholds
