"""Plot task accuracy and verified Delta-ell against effective state bits."""

import argparse

from matplotlib.ticker import StrMethodFormatter

from scripts.plots.accuracy.promotion_data import load_summaries
from scripts.plots.shared.delta_ell_data import promotion_grid
from scripts.plots.shared.style import (
    FULL,
    INK_SOFT,
    PALETTE,
    figure,
    fix_box,
    parse_plot_args,
    save,
    scaled_axis,
)

MODELS = {"qwen": ("qwen3_5", "Qwen-3.5"), "nemo": ("nemotronh", "Nemotron-H")}
PREFIXES = {"GSM8K": "dce_gen_promotion", "GPQA": "gpqa_effbits", "AIME": "aime_effbits"}
BOX = 1.1


def panel(task, tag, data, thresholds, inventory, *, output_dir):
    arch, name = MODELS[tag]
    losses = promotion_grid(arch, inventory)
    _, accuracy, reference = data[name][task]
    loss_color, accuracy_color = PALETTE[:2]
    fig, ax = figure(0.345 * FULL, 1.9)
    right = ax.twinx()
    if thresholds[name][task] is not None:
        ax.axvline(thresholds[name][task], color=INK_SOFT, lw=0.8, ls="--", zorder=0)
    right.axhline(reference, color=accuracy_color, lw=0.8, ls=":", zorder=0)
    xs = sorted(x for x in losses if x in accuracy)
    ax.plot(xs, [losses[x] for x in xs], marker="o", color=loss_color)
    gx = sorted(accuracy)
    right.plot(gx, [accuracy[x] for x in gx], marker="s", color=accuracy_color)
    ax.set_ylim(0, 1.1 * max(losses[x] for x in xs))
    scaled_axis(ax, "y", color=loss_color)
    ax.set_ylabel(r"$\Delta\ell$ (nats)", color=loss_color)
    ax.tick_params(axis="y", colors=loss_color)
    lo = min(accuracy.values()) if task == "GSM8K" else min(*accuracy.values(), reference)
    hi = max(*accuracy.values(), reference)
    right.set_ylim(lo - 0.08 * (hi - lo), hi + 0.16 * (hi - lo))
    right.set_ylabel(task + r" (\%)", color=accuracy_color)
    right.tick_params(axis="y", colors=accuracy_color)
    ax.set_xlabel("effective state bits")
    ax.tick_params(axis="x", colors=INK_SOFT)
    ax.set_xticks([4.0, 5.0, 6.0, 7.0, 8.0])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    right.grid(False)
    ax.grid(axis="x", visible=False)
    for axis in (ax, right):
        axis.spines["top"].set_visible(False)
    ax.spines["left"].set_color(loss_color)
    right.spines["right"].set_color(accuracy_color)
    right.spines["right"].set_visible(True)
    right.spines["left"].set_visible(False)
    fix_box(fig, [ax, right], BOX)
    suffix = "app" if task == "GSM8K" and tag == "qwen" else tag
    save(fig, PREFIXES[task] + "_" + suffix, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", nargs="+", choices=list(PREFIXES), default=list(PREFIXES))
    parser.add_argument("--model", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--accuracy-manifest", required=True)
    parser.add_argument("--delta-ell-inventory", required=True)
    args = parse_plot_args(parser)
    data, thresholds = load_summaries(args.accuracy_manifest)
    for task in args.task:
        for model in args.model:
            panel(task, model, data, thresholds, args.delta_ell_inventory, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
