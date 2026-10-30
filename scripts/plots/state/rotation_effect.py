"""Effect of the matched within-group rotation on the accumulated error: heatmap of Δℓ(rotated) / Δℓ(plain).

Rows are models, columns INT8/INT6/INT4 for key-axis grouping + R_v and value-axis grouping + R_k, stochastic
rounding, symmetric grids (the cells of tab:sq-rot-sr-full). Colour is log10 of the ratio, annotated with the ratio.
Inputs are Δℓ inventories of ``collect_delta_ell`` (``records[].id`` = ``<model>/<bits>-<axis>-<rounding>-<rotation>``,
with ``-oracle`` appended for oracle scales, ``dce_median``); later files override earlier ones. Missing cells stay blank.

    python -m scripts.plots.state.rotation_effect --inventory INVENTORY.json [INVENTORY.json ...] --output-dir DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from scripts.plots.shared.style import HALF, INK, PALETTE, figure, save

MODELS = [
    ("mamba2_beta", "Mamba-2"),
    ("gla_beta", "GLA"),
    ("gated_deltanet", "GDN"),
    ("kda", "KDA"),
    ("qwen3_5", "Qwen-3.5"),
    ("nemotronh", "Nemotron-H"),
    ("kimi", "Kimi-Linear"),
]
BITS = ("INT8", "INT6", "INT4")
GROUPS = [("key", "rv", "key axis"), ("value", "rk", "value axis")]
CMAP = LinearSegmentedColormap.from_list("rot", [PALETTE[0], "#ffffff", PALETTE[1]])  # teal = helps, magenta = hurts
VMAX = 1.0  # log10 range shown; larger ratios saturate


def load(paths):
    """Merge inventories into {cell id: Δℓ median}; later files win."""
    cells = {}
    for p in paths:
        for r in json.load(open(p))["records"]:
            cells[r["id"]] = r["dce_median"]
    return cells


def ratios(cells, rounding="SR", suffix=""):
    """[models, 6] log10 ratios: three precisions for key + R_v, then three for value + R_k."""
    z = np.full((len(MODELS), 3 * len(GROUPS)), np.nan)
    for i, (m, _) in enumerate(MODELS):
        for g, (axis, rot, _) in enumerate(GROUPS):
            for j, b in enumerate(BITS):
                rotated, plain = (cells.get(f"{m}/{b}-{axis}-{rounding}-{r}{suffix}") for r in (rot, "none"))
                if rotated is not None and plain is not None:
                    z[i, 3 * g + j] = np.log10(rotated / plain)
    return z


def fmt(r):
    """Ratio label: two significant digits below 10, integer above."""
    return f"{r:.0f}" if r >= 10 else f"{r:.1f}" if r >= 1 else f"{r:.2f}"


def main(
    inventories,
    *,
    output_dir,
    out="sq_rotation_effect",
    rounding="SR",
    scales="loop",
    models=None,
    axes=None,
    width=None,
    colorbar=True,
    titles=True,
):
    """Draw the heatmap; ``models`` / ``axes`` restrict the rows / grouping axes (all by default)."""
    global MODELS, GROUPS
    if models:
        MODELS = [m for m in MODELS if m[0] in models]
    if axes:
        GROUPS = [g for g in GROUPS if g[0] in axes]
    z = ratios(load(inventories), rounding, "-oracle" if scales == "oracle" else "")
    gap = 0.6  # blank column between the two grouping groups
    xpos = np.concatenate([np.arange(3) + g * (3 + gap) for g in range(len(GROUPS))])
    fig, ax = figure(width or HALF * (0.55 if len(GROUPS) == 1 else 1), 0.53 * HALF, layout="constrained")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(MODELS)):
        for k, x in enumerate(xpos):
            v = z[i, k]
            if np.isnan(v):
                continue
            ax.add_patch(
                __import__("matplotlib").patches.Rectangle(
                    (x - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=CMAP(0.5 + np.clip(v, -VMAX, VMAX) / (2 * VMAX)),
                    edgecolor="white",
                    lw=0.6,
                )
            )
            ax.text(x, i, fmt(10**v), ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.6 else INK)
    ax.set_xlim(-0.5, xpos[-1] + 0.5)
    ax.set_ylim(len(MODELS) - 0.5, -0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(BITS * len(GROUPS), fontsize=7)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([lab for _, lab in MODELS], fontsize=7)
    ax.tick_params(length=0, pad=2)
    for g, (_, _, title) in enumerate(GROUPS):
        if titles:
            ax.text(xpos[3 * g : 3 * g + 3].mean(), -1.1, title, ha="center", va="center", fontsize=8, color=INK)
    if not colorbar:
        save(fig, out, preview=False, output_dir=output_dir)
        return
    sm = __import__("matplotlib").cm.ScalarMappable(cmap=CMAP, norm=__import__("matplotlib").colors.Normalize(-VMAX, VMAX))
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels(["0.1", "0.3", "1", "3", "10"], fontsize=7)
    cb.ax.tick_params(length=0)
    cb.solids.set_rasterized(False)  # keep the PGF free of raster companions
    cb.outline.set_visible(False)
    cb.set_label(r"$\Delta\ell$ rotated / plain", fontsize=7, labelpad=2)
    save(fig, out, preview=False, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", nargs="+", required=True, help="Δℓ inventory JSONs; later files override")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounding", default="SR", choices=("SR", "RTN"))
    parser.add_argument("--scales", default="loop", choices=("loop", "oracle"), help="oracle: the -oracle cells")
    parser.add_argument("--out", default="sq_rotation_effect")
    parser.add_argument("--models", nargs="*", default=None, help="restrict rows to these cell ids")
    parser.add_argument("--axes", nargs="*", default=None, choices=("key", "value"), help="restrict to these grouping axes")
    parser.add_argument("--width", type=float, default=None, help="figure width in inches")
    parser.add_argument("--no-colorbar", action="store_true", help="omit the colour bar (subfigure next to the full heatmap)")
    parser.add_argument("--no-titles", action="store_true", help="omit the grouping-axis titles (the subcaption carries them)")
    args = parser.parse_args()
    main(
        args.inventory,
        output_dir=args.output_dir,
        out=args.out,
        rounding=args.rounding,
        scales=args.scales,
        models=args.models,
        axes=args.axes,
        width=args.width,
        colorbar=not args.no_colorbar,
        titles=not args.no_titles,
    )
