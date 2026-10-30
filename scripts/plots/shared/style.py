"""Shared PGF plot styling for LaTeX-native text.

Render at final physical size; rescaling in the paper would change font sizes.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

# Prefer the first three colors for thin lines; pale colors need markers.
PALETTE = [
    "#089099",  # 1 teal
    "#DC3977",  # 2 magenta
    "#045275",  # 3 navy
    "#7CCBA2",  # 4 green
    "#FCDE9C",  # 5 yellow
    "#F0746E",  # 6 salmon
    "#7C1D6F",  # 7 plum
    "#F7B2AD",  # 8 pink
]
INK = "#1a1a19"  # primary text
INK_SOFT = "#52514e"  # secondary text / axis furniture
GRID = "#e2e1dd"

# iclr2026_conference.sty: \textwidth 5.5in, single column.
FULL = 5.5
HALF = 0.48 * FULL  # the .48\textwidth subfigure slot

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,  # fonts come from the preamble, not matplotlib
        "pgf.preamble": r"\usepackage{times}\usepackage{amsmath}\usepackage{amssymb}\usepackage{xcolor}\usepackage{pifont}",
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        # recessive furniture: no top/right spine, hairline grid, short ticks
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    }
)


def figure(width, height, **kw):
    """Axes at an exact physical size. `width` is inches -- use FULL or HALF."""
    return plt.subplots(figsize=(width, height), **kw)


def square_axes(fig, ax):
    """Keep width fixed; bisect height for square axes despite tick-layout jumps."""

    def err(h):
        fig.set_size_inches(fig.get_size_inches()[0], h)
        fig.canvas.draw()
        pos = ax.get_position()
        w, h = fig.get_size_inches()
        return pos.width * w - pos.height * h  # >0: axes too wide, grow the figure

    h = fig.get_size_inches()[1]
    best, lo, hi = None, None, None
    for _ in range(12):
        d = err(h)
        if best is None or abs(d) < best[0]:
            best = (abs(d), h)
        if abs(d) < 0.002:
            return
        if d > 0:
            lo = h if lo is None else max(lo, h)
        else:
            hi = h if hi is None else min(hi, h)
        h = (lo + hi) / 2 if None not in (lo, hi) else h + d
    err(best[1])


def save(fig, name, preview=True, *, output_dir):
    """Write PGF and an optional PDF preview to the selected directory."""
    media = os.path.abspath(output_dir)
    os.makedirs(media, exist_ok=True)
    fig.savefig(os.path.join(media, name + ".pgf"))
    if preview:
        fig.savefig(os.path.join(media, name + ".pdf"))
    print("wrote", os.path.join(media, name + ".pgf"))


def parse_plot_args(parser):
    """Require an explicit plot destination."""
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fix_box(fig, axes, side, pad=0.02):
    """Fix the square coordinate box in inches, then fit the figure around its labels."""
    from matplotlib.transforms import Bbox

    axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    fig.set_layout_engine("none")
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        W, H = fig.get_size_inches()
        tb = fig.get_tightbbox(r)  # inches
        p0 = axes[0].get_position()
        ab = Bbox([[p0.x0 * W, p0.y0 * H], [p0.x1 * W, p0.y1 * H]])
        ml, mr = ab.x0 - tb.x0 + pad, tb.x1 - ab.x1 + pad
        mb, mt = ab.y0 - tb.y0 + pad, tb.y1 - ab.y1 + pad
        nW, nH = side + ml + mr, side + mb + mt
        fig.set_size_inches(nW, nH)
        for a in axes:
            a.set_position([ml / nW, mb / nH, side / nW, side / nH])


def scaled_axis(ax, which="y", color=None):
    """Integer ticks in units of 10^e, with one `x10^e` note at the axis end: the scale is
    chosen so the axis maximum lands in [10, 100), so every tick label is <= 2 digits and
    sibling panels keep the same label width (and hence the same box)."""
    import math

    from matplotlib.ticker import FuncFormatter, MaxNLocator

    axis = ax.yaxis if which == "y" else ax.xaxis
    lo, hi = axis.get_view_interval()
    e = int(math.floor(math.log10(hi))) - 1
    axis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], integer=False))
    ticks = [t for t in axis.get_majorticklocs() if lo <= t <= hi]
    axis.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % round(v / 10**e)))
    kw = {"color": color} if color else {}
    if which == "y":
        ax.text(0.0, 1.0, r"$\times10^{%d}$" % e, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, **kw)
    return e


def speedup_axis(ax, label=None):
    """Show a y axis of percent gains g as speedups 1 + g / 100. Ticks keep their positions and share the fewest
    decimals that write each of them exactly. `label` gets the unit appended.
    """
    from matplotlib.ticker import Formatter

    class Speedup(Formatter):
        def __call__(self, v, pos=None):
            return self.format_ticks([v])[0]

        def format_ticks(self, values):
            r = [1 + v / 100 for v in values]
            d = next((d for d in range(4) if all(abs(round(x, d) - x) < 1e-9 for x in r)), 3)
            return [f"{x:.{d}f}" for x in r]

    ax.yaxis.set_major_formatter(Speedup())
    if label:
        ax.set_ylabel(label + r" ($\times$)")
