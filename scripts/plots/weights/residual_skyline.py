"""Plot token/channel residual magnitudes for Transformer and KDA with shared height/color scales."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from lores.evaluation.output_store import source_path
from scripts.plots.shared.style import FULL, INK_SOFT, parse_plot_args, save

LAYER, TOK_STRIDE = 12, 2

CMAP = LinearSegmentedColormap.from_list("resid", ["#045275", "#089099", "#7CCBA2", "#FCDE9C"])
CELLS = ("softmax", "kda")


def load(stem, layers, sources):
    M = np.load(source_path(sources[stem]))["M"].astype(np.float32)
    return np.abs(M).max(1) if layers else np.abs(M[LAYER, ::TOK_STRIDE])


def panel(sources, *, output_dir, layers=False):
    mats = {stem: load(stem, layers, sources) for stem in CELLS}
    ZMAX = max(a.max() for a in mats.values())  # shared across both figures
    NORM = PowerNorm(0.4, 0, ZMAX)

    # One pgf with both panels, authored at the 0.58\textwidth wrap it sits in (no \resizebox,
    # text set by LaTeX at 9pt/8pt).
    fig = plt.figure(figsize=(0.58 * FULL, 1.75), dpi=600, layout="none")  # dpi sets the raster resolution
    for i, stem in enumerate(CELLS):
        title = "KDA" if stem == "kda" else ("Transformer" if layers else "Softmax attention")
        a = mats[stem]
        ntok, nchan = a.shape
        X, Y = np.meshgrid(np.arange(nchan), np.arange(ntok) * (1 if layers else TOK_STRIDE))
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.patch.set_alpha(0)  # transparent axes background: panel 2 must not cover panel 1's z-label in the gutter
        surf = ax.plot_surface(
            X, Y, a, rcount=ntok, ccount=nchan, facecolors=CMAP(NORM(a)), linewidth=0, antialiased=False, shade=False
        )
        surf.set_rasterized(True)
        ax.set_zlim(0, ZMAX * 1.02)
        ax.set_xlabel("channel", labelpad=-5)
        ax.set_ylabel("layer" if layers else "token", labelpad=-5)
        # mpl places the 3-D z-label ~0.3in right of the box, which at this panel width lands
        # under the neighbouring panel / off the figure; pin it by hand next to the z ticks.
        ax.text2D(
            1.06, 0.50, r"$\max_t |x|$" if layers else r"$|x|$", transform=ax.transAxes, rotation=90, ha="left", va="center"
        )
        ax.set_xticks([0, 1024, 2048])
        ax.set_yticks([0, 8, 16, 23] if layers else [0, 64, 128])
        ax.set_zticks([0, 5, 10, 15])
        ax.tick_params(colors=INK_SOFT, pad=-3)
        ax.set_title(title, y=0.92)
        ax.view_init(elev=32, azim=-58)
        ax.set_box_aspect((1.5, 1.0, 0.95), zoom=0.83)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((1, 1, 1, 0))
            pane._axinfo["grid"]["color"] = (0.88, 0.88, 0.86, 1)
            pane._axinfo["grid"]["linewidth"] = 0.4
            pane.line.set_color(INK_SOFT)
            pane.line.set_linewidth(0.5)
    fig.subplots_adjust(left=-0.045, right=0.93, bottom=-0.02, top=1.06, wspace=0.10)
    save(fig, "residual_skyline_layers" if layers else "residual_skyline", output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for cell in CELLS:
        parser.add_argument(f"--{cell}", required=True)
    args = parse_plot_args(parser)
    sources = {cell: getattr(args, cell) for cell in CELLS}
    panel(sources, output_dir=args.output_dir)
    panel(sources, layers=True, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
