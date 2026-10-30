"""Plot relative one-shot rounding-error CDFs over layer/head pairs."""

import argparse
import json

import numpy as np
from matplotlib.ticker import NullLocator

from lores.evaluation.output_store import source_path
from scripts.plots.shared.style import INK_SOFT, PALETTE, figure, parse_plot_args, save

CELLS = [("mamba2_beta", "Mamba-2", PALETTE[1]), ("gated_deltanet", "GDN", PALETTE[2])]


def main(source, *, output_dir, out="inject_rel_cdf"):
    L, R, TOP, BOT, PW = 0.54, 0.04, 0.08, 0.64, 1.01
    FW, FH = L + PW + R, TOP + PW + BOT
    fig, ax = figure(FW, FH, layout="none")
    fig.set_layout_engine("none")
    fig.set_size_inches(FW, FH)
    ax.set_position([L / FW, BOT / FH, PW / FW, PW / FH])
    j = json.loads(source_path(source).read_text())
    for cell, label, col in CELLS:
        xs = np.sort(np.array(j["cells"][cell]["rel_inj"]))
        ax.plot(xs, np.linspace(0, 1, len(xs)), color=col, label=label)
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.05, 0.1])
    ax.set_xticklabels(["0.02", "0.05", "0.1"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("CDF")
    ax.grid(axis="x", visible=False)
    ax.tick_params(colors=INK_SOFT)
    cx = (L + PW / 2) / FW
    fig.text(
        cx,
        (BOT - 0.30) / FH,
        r"$\lVert\boldsymbol{\varepsilon}_{t_0}\rVert_F^2\,/\,\lVert\mathbf{S}_{t_0}\rVert_F^2$",
        ha="center",
        va="center",
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(cx, 0.0),
        labelspacing=0.25,
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.3,
        borderaxespad=0,
        borderpad=0,
    )
    save(fig, out, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parse_plot_args(parser)
    main(source=args.input, output_dir=args.output_dir)
