"""Plot integer-state decode speedups from explicit kernel runs or legacy logs."""

import argparse
import json
import re
from pathlib import Path

from lores.evaluation.output_store import source_path
from scripts.plots.shared.style import (
    FULL,
    INK_SOFT,
    PALETTE,
    figure,
    parse_plot_args,
    save,
    speedup_axis,
)

B_ALL = [1, 2, 4, 8, 16, 32, 64, 128]
CM = {"Mamba-2": PALETTE[0], "GDN": PALETTE[1], "KDA": PALETTE[2]}
MK = {"Mamba-2": "o", "GDN": "s", "KDA": "^"}
PCT = lambda r: (r - 1.0) * 100.0  # noqa: E731


def kern_blocks(log, name, pat):
    out = {}
    for m in re.finditer(rf"=== {name} 32H B=(\d+) ===(.*?)(?====|\Z)", log, re.S):
        mm = re.search(pat, m.group(2))
        if mm:
            out[int(m.group(1))] = float(mm.group(1))
    return out


def kern_m2(path):
    out = {}
    for blk in re.split(r"=== (?:mode=fast |kernel )?B=", source_path(path).read_text())[1:]:
        mm = re.search(r"int8 best.*vs fp32-best ([\d.]+)x", blk)
        if mm:
            out[int(blk.split()[0].rstrip("="))] = float(mm.group(1))
    return out


def fig_kernel(data, out, *, output_dir, b_max=None):
    fig, ax = figure(FULL, 1.9)
    fig.set_layout_engine("none")
    left, right, top, bottom, side = 0.45, 0.05, 0.20, 0.60, 1.01
    width, height = left + side + right, side + top + bottom
    fig.set_size_inches(width, height)
    ax.set_position([left / width, bottom / height, side / width, side / height])
    batches = [b for b in B_ALL if b_max is None or b <= b_max]
    positions = {b: i for i, b in enumerate(batches)}
    for name in CM:
        values = data[name]
        xs = [b for b in batches if b in values]
        ax.plot(
            [positions[b] for b in xs],
            [PCT(values[b]) for b in xs],
            marker=MK[name],
            color=CM[name],
            lw=1.2,
            ms=3.2,
            mfc=CM[name],
            mec="none",
            label=name,
        )
    ax.axhline(0.0, color=INK_SOFT, lw=0.6, ls=":")
    ax.set_xticks(range(len(batches)), [str(b) if i % 2 == 0 else "" for i, b in enumerate(batches)])
    ax.set_title("INT8", fontsize=8)
    speedup_axis(ax, "speedup")
    fig.text((left + side / 2) / width, (bottom - 0.30) / height, "batch size", ha="center", va="center")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=0.8,
        handletextpad=0.25,
        labelspacing=0.2,
        columnspacing=0.5,
        borderaxespad=0,
    )
    save(fig, out, output_dir=output_dir)


def load_runs(runs):
    data = {}
    names = {"mamba2": "Mamba-2", "gdn": "GDN", "kda": "KDA"}
    for run in runs:
        json.loads((run / "complete.json").read_text())
        config = json.loads((run / "run.json").read_text())["config"]
        if config["profile"] != "equistate" or config["state_fast"] or not config["state_sr"]:
            raise ValueError(f"Not an equal-state SR/non-FAST run: {run}")
        by_arch = data.setdefault(config["gpu_tag"], {})
        for row in json.loads((run / "results.json").read_text()):
            values = by_arch.setdefault(names[row["arch"]], {})
            if row["batch"] in values:
                raise ValueError(f"Duplicate kernel cell in {run}")
            values[row["batch"]] = row["median_us"]["fp32"] / row["median_us"]["int8"]
    return data


def load_logs(logs):
    data = {}
    patterns = {
        "gdn": ("GDN", r"int8 v-first best.*vs fp32 ([\d.]+)x"),
        "kda": ("KDA", r"int8 ROW best.*vs fp32 ([\d.]+)x"),
    }
    for gpu, arch, path in logs:
        if gpu not in ("h100", "rtx") or arch not in ("mamba2", "gdn", "kda"):
            raise ValueError("--logs expects h100|rtx mamba2|gdn|kda PATH")
        if arch == "mamba2":
            name, values = "Mamba-2", kern_m2(path)
        else:
            name, pattern = patterns[arch]
            values = kern_blocks(source_path(path).read_text(), name, pattern)
        by_arch = data.setdefault(gpu, {})
        if name in by_arch:
            raise ValueError(f"Duplicate {gpu}/{arch} kernel log")
        by_arch[name] = values
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--runs", type=Path, nargs="+", help="completed equal-state kernel run directories")
    inputs.add_argument("--logs", nargs=3, action="append", metavar=("GPU", "ARCH", "PATH"), help="explicit saved kernel logs")
    args = parse_plot_args(parser)
    data = load_runs(args.runs) if args.runs else load_logs(args.logs)
    for gpu, values in data.items():
        if set(values) != set(CM) or any(set(v) != set(B_ALL) for v in values.values()):
            raise ValueError(f"Incomplete {gpu} kernel grid")
        fig_kernel(values, f"speed_kernel_{gpu}", output_dir=args.output_dir)
        if gpu == "h100":
            fig_kernel(values, "speed_kernel_h100_int8", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
