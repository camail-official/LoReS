r"""Export GPQA prompts with the lm-eval NeMo-Skills template and checked answer mapping."""

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from lores.evaluation.output_store import source_files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--reference-samples", help="optional saved HF samples glob for checking the answer shuffle")
    args = ap.parse_args()
    out = args.out
    random.seed(0)  # simple_evaluate(random_seed=0, numpy_random_seed=1234, torch_random_seed=1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    tm = TaskManager(include_path=str(Path(__file__).resolve().parents[3] / "lores/lmeval_tasks"))
    config = load_yaml(os.path.join(os.path.dirname(__file__), "../../../lores/lmeval_tasks/gpqa/gpqa_diamond_cot_ns.yaml"))
    config["dataset_kwargs"] = {**config.get("dataset_kwargs", {}), "cache_dir": args.cache_dir}
    task = tm.load_task_or_group([config])["gpqa_diamond_cot_ns"]
    docs = list(task.eval_docs)

    ref = {}
    for f in source_files(args.reference_samples).values() if args.reference_samples else ():
        j = json.load(gzip.open(f))
        for s in j[list(j)[0]]:
            if s.get("filter") == "strict-match":
                ref[int(s["doc_id"])] = s["target"].strip("()")
    assert len(docs) == 198, len(docs)
    if args.reference_samples:
        assert len(ref) == 198, len(ref)
        mism = [i for i, d in enumerate(docs) if task.doc_to_target(d).strip("()") != ref[i]]
        assert not mism, f"choice shuffle differs from the HF runs on {len(mism)} docs: {mism[:10]}"

    with open(out, "w") as f:
        for i, d in enumerate(docs):
            f.write(
                json.dumps({"task": "gpqa", "idx": i, "target": task.doc_to_target(d).strip("()"), "user": task.doc_to_text(d)})
                + "\n"
            )
    print(f"prompts {len(docs)} -> {out}")


if __name__ == "__main__":
    main()
