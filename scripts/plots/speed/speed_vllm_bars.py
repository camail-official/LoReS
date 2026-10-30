"""Plot vLLM INT8-state decode speedups per batch size from completed fixed-batch timing runs.

One figure per weight setting found in ``--src``, named ``speed_vllm_bars_{bf16|w4|w8}_a3{nemo|qwen}[PIN]`` after
the model prefix and ``--pinned``.
"""

import argparse

import numpy as np

from scripts.plots.shared.style import HALF, PALETTE, figure, save, speedup_axis
from scripts.plots.speed.timing_data import BATCHES as B
from scripts.plots.speed.timing_data import gain
from scripts.plots.speed.timing_data import load_concurrency as load

STATE = PALETTE[5]


def _pct(ms, tag, arm, b, aggregation="ratio-of-medians"):
    """Speedup over the same-weights baseline."""
    base, v = ms.get((tag, "base", b)), ms.get((tag, arm, b))
    if base is None or v is None:
        raise ValueError(f"Missing timing cell {tag} {arm} c={b}")
    return gain(base, v, aggregation)


def bars(ms, tag, out, *, output_dir, bs=B, panel=None, aggregation="ratio-of-medians"):
    fig, ax = figure(*(panel or (HALF, 1.9)))
    x = np.arange(len(bs))
    speedup = np.array([_pct(ms, tag, "int8", batch, aggregation) for batch in bs])
    ax.bar(x, speedup, 0.6, color=STATE, edgecolor="none")
    ax.set_xticks(x, [str(batch) for batch in bs])
    if panel:
        ax.tick_params(labelsize=7)
    ax.set_xlabel("batch size")
    speedup_axis(ax, "speedup")
    ax.grid(axis="y", alpha=0.3)
    save(fig, out, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", nargs="+", required=True)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--model-prefix", required=True, choices=["nemo", "qwen"])
    parser.add_argument("--pinned", action="store_true", help="plot W4 with the attention block pinned")
    parser.add_argument("--panel", type=float, nargs=2, metavar=("W", "H"))
    parser.add_argument("--aggregation", choices=["ratio-of-medians", "median-paired"], default="ratio-of-medians")
    from scripts.plots.shared.style import parse_plot_args

    args = parse_plot_args(parser)
    bs = [batch for batch in B if batch <= args.max_batch]
    ms = load(args.src, args.aggregation)
    prefix = args.model_prefix + "_"
    ms = {(tag[len(prefix) :], arm, c): value for (tag, arm, c), value in ms.items() if tag.startswith(prefix)}
    if not ms:
        raise ValueError(f"No cells with prefix {prefix!r}")
    if args.pinned:
        ms = {("w4", arm, c): value for (tag, arm, c), value in ms.items() if tag == "w4p"}
        if not ms:
            raise ValueError("No pinned W4 cells found")
    suffix = f"_a3{args.model_prefix}" + ("PIN" if args.pinned else "")
    for tag in [tag for tag in ("bf16", "w4", "w8") if any(key[0] == tag for key in ms)]:
        name = f"speed_vllm_bars_{tag}{suffix}"
        bars(ms, tag, name, bs=bs, panel=args.panel, output_dir=args.output_dir, aggregation=args.aggregation)


if __name__ == "__main__":
    main()
