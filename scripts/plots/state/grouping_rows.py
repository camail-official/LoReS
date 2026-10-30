"""Per-row relative readout error against the row's size within its head, value- vs key-axis grouping (INT8, RTN).

Panel renderer for ``scripts.plots.state.state_quantization`` (fig:sq-overview b), from the JSON written by
``scripts.diagnostics.row_error_vs_magnitude``.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.plots.shared.style import GRID, INK_SOFT, PALETTE

COLOR = PALETTE[2]  # one colour: the line style identifies the axis
STYLE = {"value": "-", "key": "--"}
MIN_ROWS = 50  # bins with fewer rows are not drawn


def parse(j):
    """Bin midpoints, per-bin row counts and per-arm relative errors from one row_error_vs_magnitude JSON."""
    return np.array(j["x_mid"]), np.array(j["count"]), {a: np.array(j[a]) for a in ("value", "key")}


def draw(ax, x, count, err):
    """One panel: value and key curves, the error = output line, the row-count band."""
    ok = count >= MIN_ROWS
    for arm, y in err.items():
        ax.plot(x[ok], y[ok], color=COLOR, ls=STYLE[arm], lw=1.1)
    ax.fill_between(  # where the rows are: a grey band along the bottom of the box
        x, 0, 0.8 * count / count.max(), step="mid", transform=ax.get_xaxis_transform(), color=GRID, lw=0, zorder=0
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    xlo = x[ok].min() / 10**0.05  # per panel: the lower edge of the smallest populated bin (bins are 0.1 decade wide)
    ax.set_xlim(xlo, 1)
    ax.set_ylim(1e-3, 3)
    decades = list(range(int(np.ceil(np.log10(xlo))), 1))
    ax.set_xticks([10.0**e for e in decades])
    thin = len(decades) > 4  # thin the labels rather than shrink them; explicit strings: the pgf preamble has no \mathdefault
    ax.set_xticklabels([rf"$10^{{{e}}}$" if not thin or (e - decades[-1]) % 2 == 0 else "" for e in decades])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels([rf"$10^{{{e}}}$" for e in (-3, -2, -1, 0)])
    ax.grid(axis="x", visible=False)
    ax.tick_params(colors=INK_SOFT)


def handles():
    """Legend handles: the two groupings and the row-count band."""
    return [Line2D([], [], color=COLOR, ls=STYLE[a], label=a) for a in ("value", "key")] + [
        Patch(facecolor=GRID, edgecolor="none", label="distribution")
    ]


def legend(fig, cx, y):
    """Frameless legend centred at figure coordinates (cx, y): value and key in one row, the row-count band below."""
    h = handles()
    for hs, dy in ((h[:2], 0.0), (h[2:], 0.16 / fig.get_figheight())):
        fig.legend(
            handles=hs,
            loc="upper center",
            ncol=2,
            fontsize=8,
            bbox_to_anchor=(cx, y - dy),
            labelspacing=0.25,
            columnspacing=0.5,
            handlelength=1.1,
            handletextpad=0.3,
            borderaxespad=0,
            borderpad=0,
        )
