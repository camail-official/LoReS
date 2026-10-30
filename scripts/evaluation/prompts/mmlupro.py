r"""Export MMLU-Pro test prompts in the NeMo-Skills multiple-choice format."""

import argparse
import json

from datasets import load_dataset  # noqa: E402

INSTR = (
    "Answer the following multiple choice question. The last line of your response should be in the "
    "following format: 'Answer: A/B/C/D/E/F/G/H/I/J' (e.g. 'Answer: A').\n\n"
)
NSHARD = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    out = args.out
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", cache_dir=args.cache_dir)
    rows = []
    for i, d in enumerate(ds):
        opts = "\n".join(f"{chr(65 + j)}) {o}" for j, o in enumerate(d["options"]))  # NS get_mcq_fields
        assert d["answer"] == chr(65 + d["answer_index"]) and d["answer_index"] < len(d["options"])
        rows.append(
            {
                "task": "mmlupro",
                "idx": i,
                "qid": d["question_id"],
                "cat": d["category"],
                "target": d["answer"],
                "user": f"{INSTR}{d['question'].strip(chr(10))}\n\n{opts}",
            }
        )
    assert len(rows) == 12032
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    stem = out[: -len(".jsonl")]
    for k in range(NSHARD):
        with open(f"{stem}_k{k}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows if r["idx"] % NSHARD == k)
    print(f"{len(rows)} prompts -> {out} (+{NSHARD} shards)")


if __name__ == "__main__":
    main()
