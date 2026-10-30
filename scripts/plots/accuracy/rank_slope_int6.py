"""Compare INT6 global-median delta-ell with GSM8K accuracy."""

import argparse
import json
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

from scripts.plots.accuracy.promotion_data import read_accuracy
from scripts.plots.shared.delta_ell_data import median
from scripts.plots.shared.style import (
    FULL,
    PALETTE,
    figure,
    fix_box,
    parse_plot_args,
    save,
)

BOX = 1.1  # inches; identical coordinate box across the three fig:e2e-transfer panels

C_DCE, C_GEN = PALETTE[0], PALETTE[1]
MS = 5.0


def split(marker):
    """Split the marker diagonally, rotating a shared transform only once."""
    ms = MarkerStyle(marker, fillstyle="left")
    shared = ms._alt_transform is ms._transform
    ms._transform = ms._transform.rotate_deg(-45)
    ms._alt_transform = ms._transform if shared else ms._alt_transform.rotate_deg(-45)
    return ms


FACE = dict(markerfacecolor=C_GEN, markerfacecoloralt=C_DCE, markeredgecolor="none")

CONFIGS = [("value", "o"), (r"value+$\mathbf{R}_k$", "s"), ("key", "^"), (r"key+$\mathbf{R}_v$", "D")]

MODELS = {
    "nemo": {
        "value": "nemotronh/INT6-value-RTN-none",
        "value+$\\mathbf{R}_k$": "nemotronh/INT6-value-RTN-rk",
        "key": "nemotronh/INT6-key-RTN-none",
        "key+$\\mathbf{R}_v$": "nemotronh/INT6-key-RTN-rv",
    },
    "app": {
        "value": "qwen3_5/INT6-value-RTN-none",
        "value+$\\mathbf{R}_k$": "qwen3_5/INT6-value-RTN-rk",
        "key": "qwen3_5/INT6-key-RTN-none",
        "key+$\\mathbf{R}_v$": "qwen3_5/INT6-key-RTN-rv",
    },
}


def draw(ax, rows):
    for label, marker in CONFIGS:
        val, gen = rows[label]
        ax.plot([val], [gen], marker=split(marker), ls="none", ms=MS, zorder=3, **FACE)


def finish(ax, vals):
    ax.set_xscale("log")
    # at this width matplotlib drops all but one decade label; place them explicitly
    lo, hi = min(vals), max(vals)
    ticks, d = [], 0.1
    while d <= 1000:
        if lo / 3 <= d <= hi * 3:
            ticks.append(d)
        d *= 10
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.invert_xaxis()
    ax.set_xlabel(r"$\Delta\ell$ (nats)", color=C_DCE)
    ax.set_ylabel(r"GSM8K (\%)", color=C_GEN)
    ax.tick_params(axis="x", colors=C_DCE, labelcolor=C_DCE)
    ax.tick_params(axis="y", colors=C_GEN, labelcolor=C_GEN)
    ax.spines["bottom"].set_color(C_DCE)
    ax.spines["left"].set_color(C_GEN)


def panel(tag, accuracy, inventory, *, output_dir):
    rows = {label: (median(config_id, inventory), accuracy[config_id]) for label, config_id in MODELS[tag].items()}
    fig, ax = figure(0.29 * FULL, 1.75)
    draw(ax, rows)
    finish(ax, [v for v, _ in rows.values()])
    # above the axes, two rows of three: every in-axes corner is occupied in one model
    # or the other, and a side legend eats too much of a 0.29\textwidth panel
    ax.legend(
        handles=[Line2D([], [], ls="none", marker=split(m), ms=MS - 1, label=lab, **FACE) for lab, m in CONFIGS],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        fontsize=8,
        labelspacing=0.25,
        columnspacing=0.8,
        handletextpad=0.2,
        handlelength=0.8,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    fix_box(fig, ax, BOX)
    save(fig, "rank_slope_int6_" + tag, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="*", choices=list(MODELS))
    parser.add_argument("--accuracy-manifest", required=True, help="Delta-ell configuration ID -> GSM8K grading-summary path")
    parser.add_argument("--delta-ell-inventory", required=True)
    args = parse_plot_args(parser)
    path = Path(args.accuracy_manifest)
    accuracy = {k: read_accuracy(p, path.parent) for k, p in json.loads(path.read_text()).items()}
    for tag in args.tags or ["nemo", "app"]:
        panel(tag, accuracy, args.delta_ell_inventory, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
