r"""AIME 2024+2025 -> prompts.jsonl. ``bare`` is lm-eval's stock aime24/aime25 doc_to_text,
'Question: {Problem}\\nAnswer:'; ``ns`` prepends the NeMo-Skills instruction to the problem.
"""

import argparse
import json

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("style", nargs="?", choices=("bare", "ns"), default="bare")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    out = args.out
    NS = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n"
    n = 0
    with open(out, "w") as f:
        for task, path in (("aime24", "Maxwell-Jia/AIME_2024"), ("aime25", "math-ai/aime25")):
            ds = load_dataset(path, split="train" if task == "aime24" else "test", cache_dir=args.cache_dir)
            for i, d in enumerate(ds):
                prob = d.get("Problem", d.get("problem"))
                ans = d.get("Answer", d.get("answer"))
                f.write(
                    json.dumps(
                        {
                            "task": task,
                            "idx": i,
                            "target": str(ans),
                            "user": (NS + prob) if args.style == "ns" else f"Question: {prob}\nAnswer:",
                        }
                    )
                    + "\n"
                )
                n += 1
    print(f"prompts {n}")


if __name__ == "__main__":
    main()
