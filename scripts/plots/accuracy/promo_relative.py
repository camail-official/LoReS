"""Plot task accuracy relative to FP32 across promotion settings."""

import argparse

from matplotlib.lines import Line2D

from scripts.plots.accuracy.promotion_data import C_AIME, C_GPQA, C_GSM, load_summaries
from scripts.plots.shared.style import FULL, INK_SOFT, figure, parse_plot_args, save


def main(data, thresholds, *, output_dir, out="promo_relative", width=0.50):
    fig, axes = figure(width * FULL, 1.75, ncols=2, layout="none")
    fig.set_layout_engine("none")
    # Place the panel boxes explicitly in inches (square boxes, labels sized around them).
    W = width * FULL
    L, R, GAP = 0.42, 0.03, 0.28  # inches: y-label band, right pad, inter-panel gap
    PW = (W - L - R - GAP) / 2  # square panels
    BOT, TOP = 0.56, 0.24  # inches: ticks+xlabel+legend band, title band
    H = PW + BOT + TOP
    fig.set_size_inches(W, H)
    for i, ax in enumerate(axes):
        ax.set_position([(L + i * (PW + GAP)) / W, BOT / H, PW / W, PW / H])
    for ax, (model, benches) in zip(axes, data.items()):
        ax.axhline(1.0, color=INK_SOFT, lw=0.8, ls=":", zorder=0)
        for bench, (col, a, ref) in benches.items():
            if thresholds[model][bench] is not None:
                ax.axvline(thresholds[model][bench], color=col, lw=0.7, ls="--", alpha=0.8, zorder=0)
            xs = sorted(a)
            ax.plot(xs, [a[x] / ref for x in xs], ls="-", marker="o", ms=2.2, lw=1.1, color=col)
        ax.set_xlim(3.85, 8.15)
        ax.set_ylim(0, 1.1)
        ax.set_xticks([4, 5, 6, 7, 8], ["4", "5", "6", "7", "8"])
        ax.set_title(model, fontsize=8)
        ax.tick_params(axis="x", colors=INK_SOFT)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("accuracy / FP32")
    lr_mid = (L + PW + GAP / 2) / W
    fig.text(lr_mid, (BOT - 0.31) / H, "effective state bits", ha="center", va="center")
    handles = [
        Line2D([], [], color=c, lw=1.2, marker="o", ms=2.2, label=b)
        for b, c in (("GSM8K", C_GSM), ("GPQA", C_GPQA), ("AIME", C_AIME))
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(lr_mid, 0.0),
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0,
    )
    save(fig, out, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-manifest", required=True)
    args = parse_plot_args(parser)
    data, thresholds = load_summaries(args.accuracy_manifest)
    main(data=data, thresholds=thresholds, output_dir=args.output_dir)
