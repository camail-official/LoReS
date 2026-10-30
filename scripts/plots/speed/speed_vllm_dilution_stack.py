"""Plot weight/state decode speedups against prefill length."""

import argparse

import matplotlib.colors as mc
import numpy as np
from matplotlib.lines import Line2D

from scripts.plots.shared.style import FULL, HALF, PALETTE, figure, save, speedup_axis, square_axes
from scripts.plots.speed.timing_data import gain
from scripts.plots.speed.timing_data import load_context as load

BS = [2, 4, 8, 16, 32]
W4C, STATE = PALETTE[0], PALETTE[5]
NEMO_TS = [16, 32, 512, 1024, 4096, 8160]
QWEN_TS = [16, 32, 512, 1024, 4096, 8192]
PANELS = [("nemo", "Nemotron-H", NEMO_TS), ("qwen", "Qwen-3.5", QWEN_TS)]


def _cell(ms, *key):
    if key not in ms:
        raise ValueError(f"Missing timing cell {key}")
    return ms[key]


def fig(
    ms,
    out,
    weights="w4",
    model="both",
    width=None,
    *,
    output_dir,
    no_title=False,
    legend="full",
    aggregation="ratio-of-medians",
    nemo_ymax=None,
):
    """Plot combined weight/state gains or state-only gains."""
    panels = PANELS if model == "both" else [p for p in PANELS if p[0] == model]
    w_in = width * FULL if width else (FULL if len(panels) == 2 else HALF)
    f, axes = figure(w_in, 2.4, ncols=len(panels))
    if width:  # authored at a LaTeX slot; explicit 1.01 in boxes (main-text uniform size)
        f.set_layout_engine("none")
        L, R, GAP, TOP, BOT, PW = (
            0.42,
            0.03,
            0.28,
            (0.06 if no_title else 0.24),
            {"full": 0.66, "batchrow": 0.66, "batch": 0.84}.get(legend, 0.36),
            1.01,
        )
        n = len(panels)
        FW = L + n * PW + (n - 1) * GAP + R
        FH = PW + TOP + BOT
        f.set_size_inches(FW, FH)
        for i, ax_ in enumerate(np.atleast_1d(axes)):
            ax_.set_position([(L + i * (PW + GAP)) / FW, BOT / FH, PW / FW, PW / FH])
    axes = np.atleast_1d(axes)
    for ax, (tag, _name, TS) in zip(axes, panels):
        x = np.arange(len(TS))
        w = 0.8 / len(BS)
        for bi, b in enumerate(BS):
            xs = x + (bi - (len(BS) - 1) / 2) * w
            # shade by batch so the stack colours still read as weights-vs-state
            a = 0.45 + 0.55 * bi / (len(BS) - 1)
            if weights in ("bf16", "state_on_w4"):
                wt = "bf16" if weights == "bf16" else "w4"
                base = np.array([_cell(ms, tag, wt, "base", t, b) for t in TS])
                state_gain = gain(base, [_cell(ms, tag, wt, "int8", t, b) for t in TS], aggregation)
                ax.bar(xs, state_gain, w, color=mc.to_rgba(STATE, a), edgecolor="none")
                continue
            ref = np.array([_cell(ms, tag, "bf16", "base", t, b) for t in TS])
            wq = gain(ref, [_cell(ms, tag, weights, "base", t, b) for t in TS], aggregation)
            tot = gain(ref, [_cell(ms, tag, weights, "int8", t, b) for t in TS], aggregation)
            top = tot - wq
            ax.bar(xs, wq, w, color=mc.to_rgba(W4C, a), edgecolor="none")
            ax.bar(xs, top, w, bottom=np.where(wq * top >= 0, wq, 0), color=mc.to_rgba(STATE, a), edgecolor="none")
        ax.axhline(0.0, color="0.4", lw=0.6)

        # all six labels, abbreviated: at slot width "1024"/"4096"/"8160" collide, but
        # "1k"/"4k"/"8k" fit, so no tick has to go unlabelled
        def _tk(t):
            # round, not floor: Nemotron-H's longest prefill is 8160, which floors
            # to a wrong "7k"
            return f"{round(t / 1024)}k" if width and t >= 1024 else str(t)

        ax.set_xticks(x, [_tk(t) for t in TS])
        ax.set_xlabel("prefill length")
        # state_on_w4 divides the weight speedup out: its bars are w4_base/w4_int8, so the
        # baseline is W4, not BF16. The other two arms are measured against BF16 weights.
        # one label for the pair at slot width: there the axes box is ~1in tall, shorter than the full label, so the
        # baseline moves into the caption and the axis just says what the number is
        long = "speedup over W4 weights" if weights == "state_on_w4" else "speedup over BF16 weights"
        speedup_axis(ax, ("speedup" if width else long) if not width or ax is axes[0] else None)
        ax.grid(axis="y", alpha=0.3)
        if tag == "nemo" and nemo_ymax is not None:
            ax.set_ylim(top=nemo_ymax)
        if width and not no_title:  # one figure, no subcaptions to name the panels with
            ax.set_title(_name, fontsize=8)
    handles = (
        [Line2D([], [], color=W4C, lw=4, label=weights.upper().replace("W", "INT") + " weights")]
        if weights in ("w4", "w8")
        else []
    )
    handles += [Line2D([], [], color=STATE, lw=4, label="INT8 state")]
    # Pad the first column so batch shades occupy the second legend column.
    handles += [Line2D([], [], color="none", label=" ")] * (len(BS) - len(handles))
    handles += [
        Line2D([], [], color=mc.to_rgba("0.35", 0.45 + 0.55 * i / (len(BS) - 1)), lw=4, label=str(b)) for i, b in enumerate(BS)
    ]
    if legend == "none":
        pass
    elif legend == "batch":
        keys = [
            Line2D([], [], color=mc.to_rgba(STATE, 0.45 + 0.55 * i / (len(BS) - 1)), lw=4, label=str(b)) for i, b in enumerate(BS)
        ]
        if not width:
            f.set_layout_engine("none")
            f.subplots_adjust(bottom=0.36)
        center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
        f.legend(
            handles=keys,
            frameon=False,
            fontsize=8,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(center, 0.0),
            title="batch size",
            title_fontsize=8,
            labelspacing=0.15,
            columnspacing=0.5,
            handlelength=1.0,
            handletextpad=0.3,
            borderaxespad=0,
        )
    elif legend == "batchrow":
        pass  # Placed after the final axes geometry is known.
    elif width:  # nothing fits inside the axes at this width; put it under both panels
        f.legend(
            handles=[h for h in handles if h.get_label().strip()],
            frameon=False,
            fontsize=8,
            ncol=4,
            loc="lower center",
            bbox_to_anchor=((L + PW + GAP / 2) / FW, 0.0),
            labelspacing=0.25,
            columnspacing=0.7,
            handlelength=1.1,
            handletextpad=0.35,
            borderaxespad=0,
        )
    else:
        axes[0].legend(
            handles=handles,
            frameon=False,
            fontsize=6,
            ncol=2,
            loc="upper right",
            title="stack / batch",
            title_fontsize=7,
            labelspacing=0.2,
            columnspacing=0.8,
            handlelength=1.4,
            borderaxespad=0.1,
        )
    if width:
        square_axes(f, axes[0])
    if legend == "batchrow":
        _batchrow_legend(f, axes)
    save(f, out, output_dir=output_dir)


def _batchrow_legend(f, axes):
    """One centred row under both panels: 'batch size' + the grey shades. The row sits below the x label
    by the same gap the x label keeps from the tick labels; everything is measured on a draw."""
    keys = [
        Line2D([], [], color=mc.to_rgba("0.35", 0.45 + 0.55 * i / (len(BS) - 1)), lw=4, label=str(b)) for i, b in enumerate(BS)
    ]
    txt = f.text(0.0, 0.0, "batch size", ha="left", va="center", fontsize=8)
    leg = f.legend(
        handles=keys,
        frameon=False,
        fontsize=8,
        ncol=len(keys),
        loc="center left",
        bbox_to_anchor=(0.0, 0.0),
        columnspacing=0.8,
        handlelength=1.1,
        handletextpad=0.35,
        borderaxespad=0,
        borderpad=0,
    )  # no internal padding: the figure is trimmed to the row's glyph box
    f.canvas.draw()
    r = f.canvas.get_renderer()
    inv = f.transFigure.inverted()
    ax = axes[0]
    ticks = [t for t in ax.get_xticklabels() if t.get_text()]
    tick_bot = min(inv.transform(t.get_window_extent(r))[0][1] for t in ticks)
    xl = inv.transform(ax.xaxis.label.get_window_extent(r))
    gap = tick_bot - xl[1][1]  # tick text bottom -> x-label top
    y = xl[0][1] - gap - 0.5 * (inv.transform(leg.get_window_extent(r))[1][1] - inv.transform(leg.get_window_extent(r))[0][1])
    tw = inv.transform(txt.get_window_extent(r))
    lw = inv.transform(leg.get_window_extent(r))
    w_txt, w_leg = tw[1][0] - tw[0][0], lw[1][0] - lw[0][0]
    pad = 0.07 / f.get_size_inches()[0]  # 0.07 in between the word and the first key
    x0 = 0.5 - (w_txt + pad + w_leg) / 2  # centre the whole row on the figure
    txt.set_position((x0, y))
    leg.set_bbox_to_anchor((x0 + w_txt + pad, y))
    # trim the white space under the row: shrink the figure so the row ends 0.01 in above the bottom edge,
    # keeping every box at its size in inches (fractions are recomputed for the new height)
    FW_in, FH_in = f.get_size_inches()
    h_leg = lw[1][1] - lw[0][1]
    row_bot_in = (y - h_leg / 2) * FH_in
    shift = row_bot_in - 0.01  # 0.01 in of clearance under the glyphs
    if shift > 0:
        FH2 = FH_in - shift
        boxes = [(ax, ax.get_position()) for ax in axes]
        f.set_size_inches(FW_in, FH2)
        for ax, b in boxes:
            ax.set_position([b.x0, (b.y0 * FH_in - shift) / FH2, b.width, b.height * FH_in / FH2])
        y2 = (y * FH_in - shift) / FH2
        txt.set_position((x0, y2))
        leg.set_bbox_to_anchor((x0 + w_txt + pad, y2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+")
    ap.add_argument("--weights", default="w4", choices=["w4", "w8", "bf16", "state_on_w4"])
    ap.add_argument("--out", default="speed_vllm_dilution_stack")
    ap.add_argument("--model", default="both", choices=["both", "nemo", "qwen"])
    ap.add_argument("--width", type=float, default=None, help="author at this fraction of textwidth (the LaTeX slot)")
    ap.add_argument("--no-title", action="store_true", help="no in-figure panel title (subcaptions name the panels)")
    ap.add_argument("--legend", default="full", choices=["full", "batch", "batchrow", "none"])
    ap.add_argument("--aggregation", choices=["ratio-of-medians", "median-paired"], default="ratio-of-medians")
    ap.add_argument("--nemo-ymax", type=float)
    from scripts.plots.shared.style import parse_plot_args

    a = parse_plot_args(ap)
    fig(
        load(a.src, a.aggregation),
        a.out,
        weights=a.weights,
        model=a.model,
        width=a.width,
        no_title=a.no_title,
        aggregation=a.aggregation,
        nemo_ymax=a.nemo_ymax,
        legend=a.legend,
        output_dir=a.output_dir,
    )
