"""Scale inflation against decode step: per 64-step bin and head the median of the per-step mean-over-groups
quantization scale over the oracle-copy scale, then the 90th percentile over heads; INT4 SR loop runs. Colour by
grouping axis, solid rotated (R_k on value rows, R_v on key columns), dashed plain. Input: inflation.json of
``reduce_scale_feedback --kind inflation``.

    python -m scripts.plots.state.rotation_inflation_time --inflation inflation.json --output-dir DIR
    python -m scripts.plots.state.rotation_inflation_time ... --cell gla_beta --out sq_rotation_feedback_b

The first writes all four models in one row of 1.01 in boxes (fig:sq-rot-inflation-time), the second one compact
panel (fig:sq-rotation-effect b).
"""

import argparse
import json

from scripts.plots.shared.style import HALF, PALETTE, figure, save

# model order of the neighbouring figures
NAMES = {"mamba2_beta": "Mamba-2", "gla_beta": "GLA", "gated_deltanet": "GDN", "kda": "KDA"}
COLOURS = {"value": PALETTE[2], "key": PALETTE[1]}
ROT = {"value": r"$\mathbf{R}_k$", "key": r"$\mathbf{R}_v$"}
LEGEND = dict(fontsize=6, labelspacing=0.2, handlelength=1.6, handletextpad=0.4, borderaxespad=0.2, borderpad=0.2)


def panel(ax, data, cell, lw=None):
    """Plain (dashed) and rotated (solid) per grouping axis."""
    for axis in ("value", "key"):
        r = data[axis][cell]
        ax.plot(r["t"], r["plain_p90"], "--", color=COLOURS[axis], lw=lw, label=axis)
        ax.plot(r["t"], r["rot_p90"], "-", color=COLOURS[axis], lw=lw, label=f"{axis} + {ROT[axis]}")
    ax.set_xlim(0, 2048)


def grid(data, *, output_dir, out):
    """All four models in one row of 1.01 in boxes, one common y axis, legend below."""
    PW, L, R, GAP, TOP, BOT = 1.01, 0.42, 0.08, 0.26, 0.18, 0.62
    W, H = L + 4 * PW + 3 * GAP + R, TOP + PW + BOT
    fig, axes = figure(W, H, ncols=4, sharey=True, layout="none")
    fig.set_layout_engine("none")
    fig.set_size_inches(W, H)
    for i, (ax, cell) in enumerate(zip(axes, NAMES)):
        ax.set_position([(L + i * (PW + GAP)) / W, BOT / H, PW / W, PW / H])
        panel(ax, data, cell, lw=1.1)
        ax.set_title(NAMES[cell], fontsize=8, pad=3)
        ax.set_xticks([0, 1000, 2000])
        ax.set_xlabel("$t$")
        if i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("scale / oracle scale")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=((L + (4 * PW + 3 * GAP) / 2) / W, (BOT - 0.40) / H),
        ncol=4,
        fontsize=8,
        frameon=False,
        handlelength=1.6,  # long enough to show the dashes
        handletextpad=0.3,
        columnspacing=1.0,
        borderaxespad=0,
    )
    save(fig, out, output_dir=output_dir)


def single(data, cell, *, width, output_dir, out):
    """One panel without title: the subcaption carries the model. The axes box spans the heatmap band of
    fig:sq-rotation-effect (a) and (c), so t sits inside the plot.
    """
    H, L, R, BOT, TOP = 0.53 * HALF, 0.34, 0.10, 0.145, 0.53 * HALF - 1.230533
    fig, ax = figure(width, H, layout="none")
    fig.set_layout_engine("none")
    ax.set_position([L / width, BOT / H, (width - L - R) / width, (H - BOT - TOP) / H])
    panel(ax, data, cell)
    ax.set_xticks([0, 1000, 2000])
    ax.set_ylim(0, 8)  # headroom for the legend
    ax.set_yticks([0, 2, 4, 6])
    ax.tick_params(labelsize=7, length=0, pad=2)  # as the heatmap labels
    ax.set_ylabel("scale / oracle scale", fontsize=7, labelpad=2)
    ax.text(0.97, 0.03, "$t$", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.legend(loc="upper left", ncol=2, columnspacing=0.6, **{**LEGEND, "fontsize": 5, "handlelength": 1.2})
    save(fig, out, output_dir=output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inflation", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--out", default="sq_rotation_inflation_time")
    ap.add_argument("--cell", choices=list(NAMES), help="one compact panel of this model instead of the grid")
    ap.add_argument("--width", type=float, default=1.35)
    a = ap.parse_args()
    data = json.load(open(a.inflation))
    if a.cell:
        single(data, a.cell, width=a.width, output_dir=a.output_dir, out=a.out)
    else:
        grid(data, output_dir=a.output_dir, out=a.out)
