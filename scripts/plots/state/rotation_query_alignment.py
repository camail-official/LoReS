"""Distribution over heads of the query alignment of plain's fresh rounding error (value axis, SR, oracle scales), each bin
split by whether R_k's expected fresh readout error is larger or smaller than plain's. Alignment
A = d_k sum_j q_j^2 E|eps_{:,j}|^2 / (|q|^2 E|eps|_F^2); 1 = flat over key channels.

    python -m scripts.plots.state.rotation_query_alignment --alignment alignment.json --output-dir DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

from scripts.plots.shared.style import FULL, PALETTE, figure, save

MODELS = {"mamba2_beta": "Mamba-2", "gla_beta": "GLA", "gated_deltanet": "GDN", "kda": "KDA", "qwen3_5": "Qwen-3.5"}
MAIN = ("qwen3_5", "mamba2_beta")
WORSE, BETTER = PALETTE[1], PALETTE[0]  # magenta hurts, teal helps, as the ratio colour map of fig:sq-rotation-effect
EDGES = np.arange(0, 1.21, 0.1)  # alignments above 1.2 fall in the last bin
BOX, GAP, GAP_FREE, L, R, TOP, BOT, LEG, TICKS = 1.01, 0.13, 0.3, 0.38, 0.03, 0.17, 0.34, 0.2, 0.2  # inches


def counts(ph):
    """Heads per alignment bin with R_k worse / better."""
    a = np.clip(np.array(ph["Ap"]), 0, EDGES[-1] - 1e-9)
    worse = np.array(ph["ratio"]) > 1
    return np.histogram(a[worse], EDGES)[0], np.histogram(a[~worse], EDGES)[0]


def render(d, rows, name, sharey, output_dir, box=BOX):
    """Rows of panels (shared or own y axis), one x label and the legend below."""
    ncol = max(len(r) for r in rows)
    gap = GAP if sharey else GAP_FREE
    rowh = TOP + box + TICKS
    W, H = L + ncol * box + (ncol - 1) * gap + R, LEG + BOT + (len(rows) - 1) * rowh + box + TOP
    fig, axes = figure(W, H, nrows=len(rows), ncols=ncol, squeeze=False, layout="none")
    fig.set_layout_engine("none")
    fig.set_size_inches(W, H)
    hs = {c: counts(d[c]) for r in rows for c in r}
    ymax = max((w + b).max() for w, b in hs.values())
    for ri, row in enumerate(rows):
        yb = LEG + BOT + (len(rows) - 1 - ri) * rowh
        for i, ax in enumerate(axes[ri]):
            if i >= len(row):
                ax.remove()
                continue
            c = row[i]
            ax.set_position([(L + i * (box + gap)) / W, yb / H, box / W, box / H])
            ax.grid(axis="x", visible=False)
            worse, better = hs[c]
            ctr = EDGES[:-1] + 0.05
            ax.bar(ctr, worse, 0.085, color=WORSE, lw=0)
            ax.bar(ctr, better, 0.085, bottom=worse, color=BETTER, lw=0)
            ax.set_xlim(0, EDGES[-1])
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_ylim(0, 1.05 * (ymax if sharey else (worse + better).max()))
            ax.set_title(MODELS[c], pad=3)
            if i and sharey:
                ax.tick_params(axis="y", labelleft=False)
        axes[ri][0].set_ylabel("count")
    xc = (L + (ncol * box + (ncol - 1) * gap) / 2) / W
    fig.text(xc, (LEG + 0.02) / H, "query alignment", ha="center", va="bottom")
    handles = [
        Patch(color=WORSE, label=r"$\mathbf{R}_k$ worse"),
        Patch(color=BETTER, label=r"$\mathbf{R}_k$ better"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(xc, 0),
        ncol=2,
        handlelength=1.1,
        handletextpad=0.3,
        columnspacing=0.9,
        borderaxespad=0,
        borderpad=0,
    )
    save(fig, name, output_dir=output_dir)


def main():
    """Render the two-model main-text figure and the five-model appendix figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True, help="reduce_scale_feedback --kind alignment output")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.alignment.read_text())
    render(d, [MAIN], "sq_rot_query_alignment", True, args.output_dir)
    cells = [c for c in MODELS if c in d]
    box = min(BOX, (FULL - L - R - (len(cells) - 1) * GAP) / len(cells))
    render(d, [cells], "sq_rot_query_alignment_all", True, args.output_dir, box)


if __name__ == "__main__":
    main()
