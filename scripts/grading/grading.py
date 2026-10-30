"""Answer extraction, grading, and reports for saved benchmark generations."""

from __future__ import annotations

import gzip
import json
import re
import statistics
from decimal import Decimal

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

from lores.evaluation.output_store import source_path

from .nemo_math_grader import extract_answer, math_equal


def _mcq_re(letters):
    choice = rf"(?P<choice>[{letters}])"
    space = r"[^\S\r\n]*"
    return (
        re.compile(
            rf"^{space}\**Answer\**{space}:{space}"
            rf"(?:(?:\\boxed\{{|\\text\{{|\\\(|\*\*|\$|\(){space})*{choice}"
            rf"(?={space}(?:\r?$|[().,;:!?}}\]*$\-–—]|\\\)))",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            rf"\\boxed\s*\{{\s*(?P<text>\\text\s*\{{\s*)?"
            rf"\(?{choice}\)?\s*(?(text)\}}\s*)\}}",
            re.IGNORECASE,
        ),
    )


MCQ_RE = {"gpqa": _mcq_re("A-D"), "mmlupro": _mcq_re("A-J")}


def mcq_extract(text, mode="gpqa"):
    """Last line-leading Answer: X choice, else last boxed choice; return (letter, source)."""
    for pattern, source in zip(MCQ_RE[mode], ("answer", "boxed")):
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group("choice").upper(), source
    return None, None


def grade_rows(rows, mode="legacy"):
    """[(task, idx, seed, correct, no_answer, capped, gen_tokens)] for one generations file."""
    out = []
    if mode not in ("legacy", "ns", "gsm8k", *MCQ_RE):
        raise ValueError(f"Unknown grading mode: {mode}")
    for r in rows:
        if mode == "gsm8k":
            result = grade_gsm8k(extract_gsm8k(r["text"]), str(r["target"]))
            pred, ok = result["candidate"], result["correct"]
        elif mode in MCQ_RE:
            pred, _ = mcq_extract(r["text"], mode)
            ok = pred is not None and pred == str(r["target"]).strip("()").upper()
        else:
            if mode == "ns":  # boxed instruction: NS default, last \boxed{} only
                pred = extract_answer(r["text"])
            else:  # bare prompt: NS relaxed path, end-anchored Answer: line, then boxed
                pred = extract_answer(r["text"].rstrip(), relaxed=True, extract_regex=r"Answer: (.+)$")
            ok = pred is not None and math_equal(str(r["target"]), pred)
        out.append((r["task"], r["idx"], r.get("seed", 0), bool(ok), pred is None, r["finish"] == "length", r["gen_tokens"]))
    return out


def load_rows(path):
    """VLLM gen_*.jsonl rows, or an HF lm-eval *_samples.json.gz dump mapped onto the same fields."""
    if not str(path).endswith("_samples.json.gz"):
        return [json.loads(line) for line in open(source_path(path))]
    j = json.load(gzip.open(source_path(path)))
    return [
        {"task": task, "idx": i, "text": s["text"], "target": s["target"], "finish": "", "gen_tokens": s["gen_toks"]}
        for task, samples in j.items()
        for i, s in {s["doc_id"]: s for s in samples}.items()
    ]


CUE = re.compile(r"(?i)\b(?:the\s+(?:final\s+)?answer\s+is|answer\**\s*:)\s*\**\s*")
SCALAR = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def extract_gsm8k(text: str) -> str:
    """Select answer text without consulting the reference."""
    boxed = extract_answer(text.replace("\x08oxed", "\\boxed"))
    if boxed is not None:
        return boxed
    cues = list(CUE.finditer(text))
    return text[cues[-1].end() :].split("\n", 1)[0] if cues else text


def _select_gsm8k_math(text: str | None) -> str | None:
    """Normalize the selected answer text and parse it with Math-Verify."""
    if text is None:
        return None
    text = text.strip()
    # Strip Markdown emphasis without turning exponentiation (2**3) into 23.
    text = re.sub(r"(?<=[\d)])\s*\*\*\s*(?=[\d(+\-])", "^", text)
    text = text.replace("**", "").replace("`", "")
    text = text.translate(str.maketrans("−–—", "---"))
    text = re.sub(r"\\?\$(?=\s*[+-]?[\d.])", "", text)
    if SCALAR.fullmatch(text):
        return text
    # Box contents may be bare LaTeX; Math-Verify expects math delimiters.
    if text.startswith("\\") and not text.startswith(("\\(", "\\[")):
        text = "$" + text + "$"
    parsed = parse(
        text, [LatexExtractionConfig(), ExprExtractionConfig()], extraction_mode="first_match", fallback_mode="first_match"
    )
    # A successful parse is followed by the selected source string. A lone
    # fallback means parsing failed; do not retry earlier numbers.
    return parsed[-1] if len(parsed) == 2 else None


def grade_gsm8k(pred: str | None, gold: str) -> dict:
    """Compare a selected GSM8K answer with its numeric reference."""
    candidate = _select_gsm8k_math(pred)
    reference = str(gold).strip().replace(",", "")
    if not SCALAR.fullmatch(reference):
        raise ValueError(f"Invalid numeric GSM8K reference: {gold!r}")
    if candidate is None:
        return dict(correct=False, candidate=None, route="no_math_answer")
    if SCALAR.fullmatch(candidate):
        return dict(correct=Decimal(candidate) == Decimal(reference), candidate=candidate, route="exact_decimal")
    return dict(correct=bool(math_equal(reference, candidate)), candidate=candidate, route="nemo_skills_math_equal")


ALIASES = {
    "gsm8k_cot": "gsm8k",
    "gpqa_diamond_cot_ns": "gpqa",
    "aime24_ns": "aime24",
    "aime25_ns": "aime25",
    "mmlu_pro": "mmlupro",
}
SIZES = {
    "gsm8k": {"gsm8k": 1319},
    "aime": {"aime24": 30, "aime25": 30},
    "gpqa": {"gpqa": 198},
    "mmlupro": {"mmlupro": 12032},
}
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpot",
)


def summarize_generations(rows, task, *, seeds=(0,), grade_mode=None, complete=True):
    """Grade one configuration; complete summaries require every question and seed."""
    groups, seen = {}, set()
    for row in rows:
        row = row | {"task": ALIASES.get(row["task"], row["task"])}
        key = (row.get("seed", 0), row["task"], row["idx"])
        if key in seen or (complete and row.get("finish") == "error"):
            raise ValueError(f"Duplicate or failed response: {key}")
        seen.add(key)
        groups.setdefault(key[0], []).append(row)
    if not groups:
        raise ValueError("No responses")
    expected = {(t, i) for t, n in SIZES[task].items() for i in range(n)}
    if complete and (
        set(groups) != set(seeds) or any({(r["task"], r["idx"]) for r in group} != expected for group in groups.values())
    ):
        raise ValueError(f"Expected full {task} coverage for seeds {list(seeds)}")
    graded, accuracy, categories = [], {}, {}
    for seed, group in sorted(groups.items()):
        g = grade_rows(group, mode=grade_mode or ("ns" if task == "aime" else task))
        accuracy[str(seed)] = 100 * sum(r[3] for r in g) / len(g)
        graded.extend(g)
        for row, score in zip(group, g):
            if "cat" in row:
                categories.setdefault(row["cat"], []).append(score[3])
    lengths = [r[6] for r in graded]
    values = list(accuracy.values())
    summary = {
        "accuracy": statistics.mean(values),
        "per_seed": accuracy,
        "n": len(graded),
        "median_tokens": statistics.median(lengths),
        "mean_tokens": statistics.mean(lengths),
        "capped": sum(r[5] for r in graded) if all(r["finish"] for r in rows) else None,
        "no_answer": sum(r[4] for r in graded),
        "standard_error": statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None,
    }
    if categories:
        summary["categories"] = {k: 100 * sum(v) / len(v) for k, v in categories.items()}
    return summary, {(t, i, s): ok for t, i, s, ok, *_ in graded}


def paired_counts(a, b):
    """A-only, B-only, both, neither over shared (task, question, seed) keys."""
    counts = [0, 0, 0, 0]
    for key in a.keys() & b.keys():
        x, y = a[key], b[key]
        counts[0 if x and not y else 1 if y and not x else 2 if x else 3] += 1
    return counts


def summarize_ruler(paths, *, complete=True):
    """Macro-average saved 32k metrics; complete runs also require all 6,500 responses."""
    scores, lengths = {}, []
    for path in paths:
        record = json.loads(path.read_text())
        samples = path.with_name(path.stem + "_samples.json.gz")
        rows = load_rows(samples) if complete else []
        for name in record["tasks"]:
            if name not in RULER_TASKS or name in scores:
                raise ValueError(f"Unexpected/duplicate RULER task: {name}")
            docs = [r for r in rows if r["task"].startswith(name)]
            if complete and (len(docs) != 500 or {r["idx"] for r in docs} != set(range(500))):
                raise ValueError(f"Incomplete RULER task: {name}")
            metrics = [v for k, v in record["metrics"].items() if k.startswith(name) and k.endswith("_32768")]
            if len(metrics) != 1:
                raise ValueError(f"Expected exactly one 32k score: {name}")
            scores[name] = 100 * metrics[0]
            lengths.extend(r["gen_tokens"] for r in docs)
    if not scores or (complete and set(scores) != set(RULER_TASKS)):
        raise ValueError("RULER requires all 13 tasks; use --partial for a progress report")
    return {
        "accuracy": statistics.mean(scores.values()),
        "tasks": scores,
        "n": len(lengths) if complete else None,
        "median_tokens": statistics.median(lengths) if lengths else None,
    }


def grading_cli(task):
    """Shared task CLI for explicit inputs, progress reports, and complete saved summaries."""
    import argparse

    from lores.evaluation.output_store import create_run, sha256, write_json_once

    parser = argparse.ArgumentParser(description=f"Grade one {task} configuration from saved outputs.")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--partial", action="store_true", help="report incomplete progress; never saves a summary")
    parser.add_argument("--model", help="with --label, save a complete summary in the output store")
    parser.add_argument("--label")
    parser.add_argument("--output-root")
    if task == "aime":
        parser.add_argument("--grade-mode", choices=("legacy", "ns"), default="legacy")
    args = parser.parse_args()
    if bool(args.model) != bool(args.label) or (args.partial and args.model):
        parser.error("saving requires both --model and --label and forbids --partial")
    paths = [source_path(p) for p in args.files]
    if task == "ruler":
        summary = summarize_ruler(paths, complete=not args.partial)
    else:
        summary, _ = summarize_generations(
            [row for path in paths for row in load_rows(path)],
            task,
            seeds=args.seeds,
            grade_mode=getattr(args, "grade_mode", None),
            complete=not args.partial,
        )
    print(json.dumps(summary, indent=2))
    if args.model:
        config = vars(args) | {"task": task, "inputs": [{"path": str(p), "sha256": sha256(p)} for p in paths]}
        run = create_run("diagnostics", args.model, args.label, config, namespace=("grading",), root=args.output_root)
        path = run / "summary.json"
        write_json_once(path, {"task": task, "model": args.model, "label": args.label, **summary})
        print(f"SUMMARY {path}")
