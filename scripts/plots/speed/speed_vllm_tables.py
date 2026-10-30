"""Generate batch-size and context-length timing tables from indexed measurements.

Print LaTeX to stdout or ``--out``. Missing values are N/A.
"""

import argparse
import sys
from pathlib import Path

from scripts.plots.speed.timing_data import load_concurrency, load_context

B = [1, 2, 4, 8, 16, 32, 64]
MODELS = [("nemo", "Nemotron-H-8B"), ("qwen", "Qwen-3.5-9B")]
WBITS = {"bf16": 16, "w8": 8, "w4": 4}
SBITS = {"base": 32, "int8": 8}
GAP = r"\addlinespace[2pt]"


def was(weight, arm="base"):
    """Weight / activation / state precision; activations are BF16."""
    return f"${WBITS[weight]}/16/{SBITS[arm]}$"


def cell(value, fmt="{:.2f}"):
    """Format a measurement, keeping missing values explicit and avoiding -0.0."""
    if value is None:
        return "N/A"
    text = fmt.format(value)
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def head(ncols, first, span_label, tabcolsep="3.8pt", lead_cols="l"):
    """Open a table with grouped column headings."""
    return [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{%s}" % tabcolsep,
        r"\begin{tabular}{@{}%s%s@{}}" % (lead_cols, "r" * ncols),
        r"\toprule",
        r"%s & \multicolumn{%d}{c}{%s} \\" % (first, ncols, span_label),
        r"\cmidrule(lr){%d-%d}" % (len(lead_cols) + 1, len(lead_cols) + ncols),
    ]


def model_row(name, ncols):
    """Label a model's rows across the full table width."""
    return r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (ncols, name)


def foot(caption, label):
    """Close a table with its caption and reference label."""
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{%s}" % caption,
        r"\label{%s}" % label,
        r"\end{table}",
    ]


def tab_absolute(timings):
    """Absolute TPOT for BF16/W8/W4 weights with FP32/INT8 states."""
    lines = head(len(B), "W/A/S", "batch size")
    lines.append(r"& " + " & ".join(r"\numhead{%d}" % b for b in B) + r" \\")
    lines.append(r"\midrule")
    for mi, (tag, name) in enumerate(MODELS):
        lines.append(model_row(name, len(B) + 1))
        weights = [weight for weight in WBITS if any(key[0] == f"{tag}_{weight}" for key in timings)]
        for wi, weight in enumerate(weights):
            if wi:
                lines.append(GAP)
            for arm in SBITS:
                row = [cell(timings.get((f"{tag}_{weight}", arm, b))) for b in B]
                lines.append(f"{was(weight, arm)} & " + " & ".join(row) + r" \\")
        if mi == 0:
            lines.append(r"\midrule")
    # This table's caption precedes the tabular in the paper.
    lines[1:1] = [
        r"\caption{Absolute TPOT (ms), reported as the median of seven repetitions.}",
        r"\label{tab:vllm-absolute}",
    ]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def tab_dilution(timings):
    """Absolute TPOT by context length for BF16, W4, and W4 + INT8 state."""
    lengths = [16, 32, 512, 1024, 4096]
    batches = [2, 4, 8, 16, 32]
    lines = head(6, r"W/A/S & \numhead{batch size}", "prefill length", tabcolsep="3.5pt", lead_cols="lr")
    lines.append(r"& & " + " & ".join(r"\numhead{%d}" % t for t in lengths) + r" & \numhead{8k} \\")
    lines.append(r"\midrule")
    for mi, (tag, name) in enumerate(MODELS):
        prompts = lengths + [8160 if tag == "nemo" else 8192]
        lines.append(model_row(name, 8))
        for gi, (weight, arm) in enumerate((("bf16", "base"), ("w4", "base"), ("w4", "int8"))):
            if gi:
                lines.append(GAP)
            for i, batch in enumerate(batches):
                row = [cell(timings.get((tag, weight, arm, t, batch)), "{:.1f}") for t in prompts]
                stub = r"\multirow{%d}{*}{%s}" % (len(batches), was(weight, arm)) if i == 0 else ""
                lines.append(f"{stub} & {batch} & " + " & ".join(row) + r" \\")
        if mi == 0:
            lines.append(r"\midrule")
    return "\n".join(
        lines
        + foot(
            r"Absolute TPOT (ms, median of seven repetitions) as prefill length increases. "
            r"The 8k column uses the longest benchmarked prompt: $8160$ tokens for Nemotron-H "
            r"and $8192$ for Qwen-3.5.",
            "tab:vllm-dilution",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", required=True, nargs="+")
    parser.add_argument("--context", required=True, nargs="+")
    parser.add_argument("--out", type=Path, help="Write LaTeX here instead of stdout.")
    args = parser.parse_args()

    tables = [
        tab_absolute(load_concurrency(args.concurrency)),
        tab_dilution(load_context(args.context)),
    ]
    text = "\n\n".join(tables) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
