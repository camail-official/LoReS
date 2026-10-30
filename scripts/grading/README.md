# Offline grading

Each task has one command: `grade_aime`, `grade_gpqa`, `grade_gsm8k`,
`grade_mmlupro`, or `grade_ruler`. Shared loading, grading, aggregation, and
coverage checks live in `grading.py`. The NeMo-Skills grader retains its
upstream source and attribution in `nemo_math_grader.py`.

Pass explicit files from one configuration, never a glob across run attempts:

```bash
.venv/bin/python -m scripts.grading.grade_aime --grade-mode ns --seeds {0..31} \
  --files /path/to/run/gen_fp32_s{0..31}.jsonl
.venv/bin/python -m scripts.grading.grade_gpqa --files /path/to/run/gen_fp32_s0.jsonl
.venv/bin/python -m scripts.grading.grade_gsm8k --files /path/to/run/gsm8k_cot_samples.json.gz
.venv/bin/python -m scripts.grading.grade_mmlupro --files /path/to/shards/gen_fp32_k*_s0.jsonl
.venv/bin/python -m scripts.grading.grade_ruler --files /path/to/selected/task_metrics_*.json
```

By default, commands require all questions and requested seeds (seed 0 unless
specified), reject duplicate/failed responses, and print a JSON summary.
RULER takes 13 metric files and checks their companion response dumps.
MMLU-Pro pools all selected shards and reports category scores when available.

Add `--model NAME --label LABEL` to save a complete summary under the external
output store. `--output-root` overrides its default
`output/` under the checkout. Raw responses are never modified.

`--partial` prints incomplete progress and cannot save a complete summary.

Extraction rules:

- AIME `ns`: last boxed answer, NeMo-Skills mathematical comparison.
  `legacy` (the direct CLI default) first tries an end-anchored `Answer:` cue.
  Reproduction recipes explicitly select `ns`.
- GPQA/MMLU-Pro: last line-leading `Answer: X` choice, otherwise the last boxed
  letter (A–D/A–J). Formatting such as `Answer: (C)` and punctuation-separated
  explanations are accepted; word prefixes and `A/B/C/D` templates are rejected.
- GSM8K: last boxed answer, else last explicit answer cue, else the response.
  Math-Verify selects the answer; scalars use Decimal equality and expressions
  use NeMo-Skills comparison. No retry of earlier numbers after a failed parse.
- RULER: macro-average saved task scores at 32k.
