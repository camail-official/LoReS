"""Plot residual excess kurtosis against unrotated W4A4 degradation for the 11 MixerZoo cells.

Input kurtosis is Pearson kurtosis; subtract three before plotting.
"""

import argparse
import json
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from lores.evaluation.output_store import source_path
from scripts.plots.shared.style import (
    INK_SOFT,
    PALETTE,
    figure,
    fix_box,
    parse_plot_args,
    save,
)


def load_statistics(paths):
    gated = {"linear_none": "linear_none_beta", "mamba2": "mamba2_beta", "gla": "gla_beta"}
    result = {}
    for path in paths:
        record = json.loads(source_path(path).read_text())
        rows = (
            {gated.get(r["cell"], r["cell"]): r["rin_kurt"] for r in record["_1.3B"]}
            if "_1.3B" in record
            else {cell: row["resid_in"]["kurt"] for cell, row in record.items()}
        )
        if result.keys() & rows.keys():
            raise ValueError(f"Duplicate residual statistics: {path}")
        result.update(rows)
    return result


def load_ladder(paths):
    result = {}
    for path in paths:
        row = json.load(open(source_path(path)))
        if "config" not in row:
            if result.keys() & row.keys():
                raise ValueError(f"Duplicate ladder results: {path}")
            result.update(row)
            continue
        cfg = row["config"]
        rot = cfg["rotation"]
        if any(rot[k] is None for k in ("residual", "r4", "o")):
            raise ValueError(f"Ladder results require explicit R1/R4/Ro settings: {path}")
        rotation = "".join(label for field, label in (("residual", "R1"), ("r4", "R4"), ("o", "Ro")) if rot[field]) or "none"
        precision = f"W{cfg['quant']['w_bits']}A{cfg['quant']['a_bits']}"
        key = f"{Path(cfg['model']['name_or_path']).name}|{rotation}|{precision}"
        if key in result:
            raise ValueError(f"Duplicate ladder result: {key}")
        result[key] = row["metrics"]["wikitext_word_perplexity"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", nargs="+", required=True, help="residual statistics for all 11 cells")
    parser.add_argument("--ladder", nargs="+", required=True, help="PTQ results or a collected ladder JSON")
    args = parse_plot_args(parser)
    kurtosis = load_statistics(args.statistics)

    CM, DASH = "$\\checkmark$", "$-$"  # math minus: centered, short -- the tables' absent marker
    # same color language as the paper's tables: decay blue, gate orange, delta vermilion;
    # colored only when the mechanism is present, dashes stay plain -- exactly like the table cells
    DCOL, GCOL, ECOL = "{\\color[RGB]{0,114,178}%s}", "{\\color[RGB]{230,159,0}%s}", "{\\color[RGB]{213,94,0}%s}"

    def lab(w, x, y, z):
        return "/".join([DCOL % w if w != DASH else DASH, CM if x else DASH, GCOL % CM if y else DASH, ECOL % CM if z else DASH])

    # Match the ladder table, relative to each cell\'s own BF16 baseline.
    _LAD = load_ladder(args.ladder)

    def deg(cell):
        b = _LAD["%s|none|W16A16" % cell]
        return 100 * (_LAD["%s|none|W4A4" % cell] - b) / b

    CELLS = [  # (legend label, pearson kurt, W4A4 none, marker)
        (lab(DASH, 0, 0, 0), kurtosis["softmax"], deg("softmax"), "o"),
        (lab(DASH, 1, 0, 0), kurtosis["softmax_qknorm"], deg("softmax_qknorm"), "s"),
        None,  # group separator: softmax above, linear attention below
        (lab(DASH, 1, 0, 0), kurtosis["linear_none"], deg("linear_none"), "^"),
        (lab(DASH, 1, 1, 0), kurtosis["linear_none_beta"], deg("linear_none_beta"), "v"),
        (lab(DASH, 1, 1, 1), kurtosis["deltanet"], deg("deltanet"), "<"),
        (lab("Scal", 1, 0, 0), kurtosis["mamba2"], deg("mamba2"), ">"),
        (lab("Scal", 1, 1, 0), kurtosis["mamba2_beta"], deg("mamba2_beta"), "D"),
        (lab("Scal", 1, 1, 1), kurtosis["gated_deltanet"], deg("gated_deltanet"), "P"),
        (lab("Diag", 1, 0, 0), kurtosis["gla"], deg("gla"), "h"),
        (lab("Diag", 1, 1, 0), kurtosis["gla_beta"], deg("gla_beta"), "X"),
        (lab("Diag", 1, 1, 1), kurtosis["kda"], deg("kda"), "*"),
    ]

    # authored at final size (no \resizebox, LaTeX sets the text at 9pt/8pt): a BOX-inch
    # coordinate box, the figure sized around it and the side legend by fix_box.
    BOX = 1.5  # inches
    fig, ax = figure(4.0, 1.6, layout="none")

    handles = []
    for item in CELLS:
        if item is None:
            handles.append(Line2D([], [], ls="", marker="", label="linear attention"))
            continue
        label, kurt, w4a4, m = item
        ms = 55 if m == "*" else 30
        ax.scatter([kurt - 3], [w4a4], s=ms, color=PALETTE[0], marker=m, zorder=3)
        handles.append(Line2D([], [], ls="", marker=m, color=PALETTE[0], ms=4 if m != "*" else 6, label=label))
    handles.insert(0, Line2D([], [], ls="", marker="", label="softmax attention"))

    ax.set_xlabel("residual kurtosis")
    ax.set_ylabel("W4A4 PPL degradation (\\%)")
    _xs = [k - 3 for _, k, _, _ in (c for c in CELLS if c)]
    _ys = [d for _, _, d, _ in (c for c in CELLS if c)]

    def _span(lo, hi, pad=0.08):
        m = (hi - lo) * pad
        return lo - m, hi + m

    # square grid cells: both axes get the same span (the larger of the two padded data
    # spans, each centred on its data) and the same tick step in the square box.
    _xl, _yl = _span(min(_xs), max(_xs)), _span(min(_ys), max(_ys))
    _S = max(_xl[1] - _xl[0], _yl[1] - _yl[0])
    ax.set_xlim(sum(_xl) / 2 - _S / 2, sum(_xl) / 2 + _S / 2)
    ax.set_ylim(sum(_yl) / 2 - _S / 2, sum(_yl) / 2 + _S / 2)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.tick_params(colors=INK_SOFT)
    # one column, taller than the box: centred on it, so the overhang splits top/bottom.
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        fontsize=8,
        labelspacing=0.32,
        handletextpad=0.25,
        borderpad=0.3,
        borderaxespad=0,
        numpoints=1,
    )
    fix_box(fig, [ax], BOX)

    save(fig, "kurtosis_law", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
