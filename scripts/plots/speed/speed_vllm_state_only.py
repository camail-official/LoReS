"""INT8-state TPOT speedup over an FP32 state with BF16 weights by batch size (two-token prefill).

python -m scripts.plots.speed.speed_vllm_state_only --src RUN [RUN ...] --output-dir DIR
"""

import argparse

import numpy as np

from scripts.plots.shared.style import PALETTE, figure, save, speedup_axis
from scripts.plots.speed.timing_data import BATCHES, gain
from scripts.plots.speed.timing_data import load_concurrency as load

PANELS = [("nemo", "Nemotron-H"), ("qwen", "Qwen-3.5")]
COLOUR = PALETTE[5]  # the INT8-state colour of fig:speed-vllm-joint


def panels(bot):
    """Two 1.01 in panel boxes side by side."""
    L, R, GAP, TOP, PW = 0.42, 0.03, 0.28, 0.24, 1.01
    FW, FH = L + 2 * PW + GAP + R, PW + TOP + bot
    fig, axes = figure(FW, FH, ncols=2, layout="none")
    fig.set_layout_engine("none")
    fig.set_size_inches(FW, FH)
    for i, ax in enumerate(axes):
        ax.set_position([(L + i * (PW + GAP)) / FW, bot / FH, PW / FW, PW / FH])
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", visible=False)
    return fig, axes, (L + PW + GAP / 2) / FW


def plot(ms, out, *, output_dir, max_batch=256):
    """One bar per concurrency."""
    bs = [b for b in BATCHES if b <= max_batch and all((f"{p}_bf16", "base", b) in ms for p, _ in PANELS)]
    fig, axes, cx = panels(0.62)  # same boxes and legend band as the stacked panels
    x = np.arange(len(bs))
    for ax, (pre, title) in zip(axes, PANELS):
        g = lambda arm: np.array([ms[(f"{pre}_bf16", arm, b)] for b in bs])  # noqa: E731
        ax.bar(x, gain(g("base"), g("int8")), 0.6, color=COLOUR, edgecolor="none", label="INT8 state")
        ax.set_xticks(x, [str(b) if i % 2 == 0 else "" for i, b in enumerate(bs)])  # every other label, as the kernel figure
        ax.set_xlabel("batch size")
        ax.set_title(title, fontsize=8)
        speedup_axis(ax, "speedup" if ax is axes[0] else None)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(cx, 0.0),
        handlelength=1.1,
        handletextpad=0.4,
        borderaxespad=0,
    )
    save(fig, out, output_dir=output_dir)


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", nargs="+", required=True, help="completed fixed-batch concurrency timing runs")
    ap.add_argument("--out", default="speed_vllm_state_only")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    plot(load(a.src), a.out, output_dir=a.output_dir)


if __name__ == "__main__":
    main()
