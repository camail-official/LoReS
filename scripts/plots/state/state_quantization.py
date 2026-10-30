"""Render chunk interval, grouping, decay and rounding from explicit data inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from lores.evaluation.output_store import source_path
from scripts.plots.shared.delta_ell_data import median
from scripts.plots.shared.style import INK_SOFT, PALETTE, figure, save
from scripts.plots.state import grouping_rows

CELLS = [("mamba2_beta", PALETTE[1]), ("gated_deltanet", PALETTE[2])]
ROUNDING_COLOURS = {"RTN": PALETTE[5], "SR": PALETTE[6]}  # salmon, plum: unused elsewhere in the figure
E0_D = -7  # (d): |remaining error| < 10^E0_D falls in the zero bin; one bin per decade on each side
GREY = "0.35"

# Four panels across the paper's 5.5-inch text width.
PW, LT, RP, TAIL = 0.79, 0.55, 0.01, 0.10  # TAIL: overhang of the last x tick label
SW = LT + PW + RP  # 1.35 -> 4 slots + TAIL = 5.50 in
TOP, BOT = 0.08, 0.94  # top pad; xlabel + legend + caption band
FW, FH = 4 * SW + TAIL, TOP + PW + BOT


CHUNKS = [1, 4, 16, 64, 256]
FLOOR = 0.001  # symlog linear threshold: below this the axis is linear, so 0 is drawable

# Ordered by C=1 damage, worst first.
CHUNK_MODELS = [
    ("linear_none_beta", "Linear", "key", "o"),
    ("mamba2_beta", "Mamba-2", "value", "s"),
    ("gated_deltanet", "GDN", "value", "^"),
    ("kda", "KDA", "value", "D"),
    ("gla_beta", "GLA", "value", "v"),
]


def draw_chunks(ax, inventory):
    for (cell, label, axis, marker), colour in zip(CHUNK_MODELS, PALETTE):
        ys = [median(f"{cell}/INT6-{axis}-RTN-none", inventory)]
        ys += [median(f"{cell}/INT6-chunk{c}-RTN", inventory) for c in CHUNKS[1:]]
        ax.plot(CHUNKS, ys, marker=marker, color=colour, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CHUNKS)
    ax.set_xticklabels([str(c) for c in CHUNKS])
    ax.set_xlim(0.8, 320)
    # symlog so exact zeros are representable; below FLOOR the axis is linear.
    ax.set_yscale("symlog", linthresh=FLOOR, linscale=0.4)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.001, 0.01, 0.1, 1])
    ax.set_yticklabels(["$0$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
    ax.grid(axis="x", visible=False)
    ax.tick_params(colors=INK_SOFT)


def panel_b(ax, j):
    grouping_rows.draw(ax, *grouping_rows.parse(j))
    pos = ax.get_position()  # the two-row legend is drawn here; the shared loop draws none
    grouping_rows.legend(ax.figure, pos.x0 + pos.width / 2, (BOT - 0.40) / FH)
    return "relative error", [], 0


def panel_c(ax, j):
    styles = (("slow", "-"), ("mid", "--"), ("fast", (0, (3, 1, 1, 1))))
    for cell, col in CELLS:
        c = j["cells"][cell]
        for name, ls in styles:
            y = np.array(c["heads"][name]["surviving"])
            ax.plot(c["t0"] + np.arange(len(y)), y, color=col, ls=ls)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 3)
    ax.set_yticks([1e0, 1e-4, 1e-8])  # explicit strings: the pgf preamble has no \mathdefault
    ax.set_yticklabels(["$10^{0}$", "$10^{-4}$", "$10^{-8}$"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1024, 1500, 2000])
    ax.set_xticklabels([r"$t_0$", "1500", "2000"])
    return "surviving error", [Line2D([], [], color=GREY, ls=ls, label=n) for n, ls in styles], 3


def signed_decade(v):
    a = np.abs(v)
    k = np.where(a < 10.0**E0_D, 0, np.floor(np.log10(np.maximum(a, 1e-300))) - E0_D + 1)
    return (np.sign(v) * k).astype(int)


def panel_d(ax, rounding):
    ks = {r: signed_decade(np.array(rounding[r]["raw_head_time_mean"])) for r in ROUNDING_COLOURS}
    bins = np.arange(min(k.min() for k in ks.values()), max(k.max() for k in ks.values()) + 1)
    w = 0.42
    for j, (r, colour) in enumerate(ROUNDING_COLOURS.items()):
        ax.bar(bins + (j - 0.5) * w, [(ks[r] == b).mean() for b in bins], w, color=colour, lw=0)
    ax.axvline(0, color=INK_SOFT, lw=0.7, ls=":", zorder=0)
    # the decade boundary 10^e sits between bins e - E0_D and e - E0_D + 1
    ax.set_xticks([-(-2 - E0_D + 0.5), 0, -E0_D + 0.5], [r"$-10^{-2}$", "$0$", r"$10^{0}$"])
    ax.set_xlim(bins[0] - 0.7, bins[-1] + 0.7)
    return "fraction of heads", [Patch(color=c, label=r) for r, c in ROUNDING_COLOURS.items()], 2


def model_handles(names):
    return [Line2D([], [], color=col, marker=m, label=lab) for (_, lab, _, m), col in zip(CHUNK_MODELS, PALETTE) if lab in names]


def panel_a(ax, inventory):
    draw_chunks(ax, inventory)
    return r"$\Delta\ell$", model_handles({"Linear", "Mamba-2", "GDN", "KDA", "GLA"}), 3


def plot(inventory, grouping, decay, rounding):
    fig, axes = figure(FW, FH, ncols=4, layout="none")
    fig.set_layout_engine("none")
    fig.set_size_inches(FW, FH)
    panels = (
        (panel_a, inventory, "chunk size $C$", "(a) chunk size"),
        (panel_b, grouping, "row size / head max", "(b) grouping"),
        (panel_c, decay, "$t$", "(c) decay"),
        (panel_d, rounding, "remaining error", "(d) rounding"),
    )
    for i, (ax, (draw, data, xlabel, cap)) in enumerate(zip(axes, panels)):
        x0 = i * SW
        ax.set_position([(x0 + LT) / FW, BOT / FH, PW / FW, PW / FH])
        ax.grid(axis="x", visible=False)
        ax.tick_params(colors=INK_SOFT)
        ylabel, handles, ncol = draw(ax, data)
        cx = (x0 + LT + PW / 2) / FW
        ax.set_ylabel(ylabel)
        fig.text(cx, (BOT - 0.26) / FH, xlabel, ha="center", va="center")
        if handles:  # panel (b) draws its own two-row legend
            fig.legend(
                handles=handles,
                loc="upper center",
                ncol=ncol,
                fontsize=8,
                bbox_to_anchor=(cx, (BOT - 0.40) / FH),
                labelspacing=0.25,
                columnspacing=0.5,
                handlelength=1.1,
                handletextpad=0.3,
                borderaxespad=0,
                borderpad=0,
            )
        fig.text(cx, 0.06 / FH, cap, ha="center", va="bottom", fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-ell-inventory", type=Path, required=True)
    parser.add_argument("--grouping", type=Path, required=True, help="row_error_vs_magnitude_gated_deltanet.json")
    parser.add_argument("--decay", type=Path, required=True, help="decay_heads.json")
    parser.add_argument("--rounding", type=Path, required=True, help="Mamba-2 INT6 rounding_diff JSON (RTN and SR)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    grouping = json.loads(source_path(args.grouping).read_text())
    decay = json.loads(source_path(args.decay).read_text())
    rounding = json.loads(source_path(args.rounding).read_text())
    want = {"cell": "mamba2_beta", "bits": 6, "axis": "value", "rotation": "none", "scales": "loop"}
    if {k: rounding.get(k) for k in want} != want or not {"RTN", "SR"} <= set(rounding):
        parser.error(f"Expected the Mamba-2 INT6 value-axis RTN/SR rounding difference: {args.rounding}")
    fig = plot(source_path(args.delta_ell_inventory), grouping, decay, rounding)
    save(fig, "sec4_overview", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
