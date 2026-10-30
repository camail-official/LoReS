"""Serve and grade one AIME, GPQA, or MMLU-Pro configuration."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from pathlib import Path

import requests

from lores.backends.vllm.serving import REPO, VLLM, run_command, server, vllm_source
from lores.evaluation.output_store import (
    create_run,
    sha256,
    source_path,
    write_json_once,
)
from lores.models.registry import model_info

MODELS = ("nemo", "qwen", "kimi")
STATES = ("fp32", "sym", "asym", "rtn_sym", "rtn_asym", "recipe")
ARMS = STATES + tuple(
    f"{weight}_{state}" for weight in ("w8a16", "w8a8", "w4a8", "w4a16") for state in ("fp32", "sym", "asym", "recipe")
)


def configuration(args):
    info = model_info(args.model)
    checkpoint, prefix = info["checkpoint"], info["prefix"] + "-rot"
    state = args.arm
    tp, max_seqs = (2 if args.model == "kimi" else 1), 64
    if args.arm.startswith("w"):
        weight, state = args.arm.split("_", 1)
        checkpoint = str(args.models_root / f"{prefix}-{weight}")
        tp, max_seqs = 1, (32 if args.model == "kimi" else 64)
    if state == "recipe":
        state = "sym" if args.model == "qwen" else "asym"
    linq = {} if state == "fp32" else {"state_bits": 8}
    if state in ("sym", "asym"):
        linq["state_sr"] = True
    if state in ("asym", "rtn_asym"):
        linq["state_asym"] = True
    if linq:
        linq["state_axis"] = args.state_axis
    return {"checkpoint": checkpoint, "linq": linq, "tp": tp, "max_seqs": max_seqs, "client": _client(args)}


def _client(args):
    return {
        "system": "{'reasoning': True}" if args.model == "nemo" else "",
        "top_k": 20 if args.model == "qwen" else 0,
        "temperature": 1.0 if args.model == "kimi" else 0.6,
        "top_p": 0.95,
        "max_tokens": 65536,
        "stop": ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"] if args.task == "aime" else [],
    }


def _completion(job, *, url, model, system, sampling):
    doc, seed = job
    start = time.time()
    body = {
        "model": model,
        "seed": seed,
        **sampling,
        "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": doc["user"]}],
    }
    for attempt in range(4):
        try:
            response = requests.post(f"{url}/v1/chat/completions", json=body, timeout=43200).json()
        except requests.RequestException as error:
            response = {"error": f"{type(error).__name__}: {error}"[:300]}
        if "choices" in response:
            choice = response["choices"][0]
            return {
                **doc,
                "seed": seed,
                "text": choice["message"]["content"] or "",
                "finish": choice["finish_reason"],
                "gen_tokens": response["usage"]["completion_tokens"],
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "wall_s": round(time.time() - start, 1),
            }
        print(
            f"WARN request task={doc['task']} idx={doc['idx']} seed={seed} attempt {attempt}: {str(response)[:300]}",
            flush=True,
        )
        time.sleep(30 * (attempt + 1))
    return {
        **doc,
        "seed": seed,
        "text": "",
        "finish": "error",
        "gen_tokens": 0,
        "prompt_tokens": 0,
        "wall_s": round(time.time() - start, 1),
    }


def generate(docs, seeds, prefix, *, url, model, client):
    if any(Path(f"{prefix}_s{seed}.jsonl").exists() for seed in seeds):
        raise FileExistsError("Generation output already exists; use a fresh attempt")
    sampling = {key: value for key, value in client.items() if key != "system"}
    for key in ("top_k", "stop"):
        if not sampling[key]:
            del sampling[key]
    jobs = [(doc, seed) for seed in seeds for doc in docs]
    start = time.time()
    request = partial(_completion, url=url, model=model, system=client["system"], sampling=sampling)
    with ThreadPoolExecutor(min(len(jobs), 256)) as executor:
        results = list(executor.map(request, jobs))
    for seed in seeds:
        rows = [row for row in results if row["seed"] == seed]
        with open(f"{prefix}_s{seed}.jsonl", "x") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")
        print(
            f"seed {seed}: {len(rows)} done; capped {sum(row['finish'] == 'length' for row in rows)}; "
            f"mean gen {sum(row['gen_tokens'] for row in rows) / len(rows):.0f}"
        )
    print(f"total {len(results)} in {time.time() - start:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--task", required=True, choices=("aime", "gpqa", "mmlupro"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--shard", type=int, help="MMLU-Pro prompt shard, 0..15")
    parser.add_argument("--state-axis", choices=("value", "key"), default="value")
    parser.add_argument("--eager", action="store_true", help="serve with --enforce-eager")
    parser.add_argument("--disable-custom-all-reduce", action="store_true", help="serve with --disable-custom-all-reduce")
    parser.add_argument("--max-num-seqs", type=int, help="override the configured server batch cap")
    parser.add_argument("--models-root", type=Path, default=Path("models/hf"))
    parser.add_argument("--prompts", required=True, help="explicit prompt JSONL, prepared by scripts.evaluation.prompts")
    parser.add_argument("--grader-python", default=str(REPO / ".venv/bin/python"))
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be distinct and nonnegative")
    if args.task == "mmlupro":
        if args.shard not in range(16) or args.seeds != [0]:
            parser.error("MMLU-Pro requires --shard 0..15 and seed 0")
    elif args.shard is not None:
        parser.error("--shard is MMLU-Pro-only")
    config = configuration(args)
    if args.max_num_seqs:
        config["max_seqs"] = args.max_num_seqs
    config.update(
        task=args.task,
        model=args.model,
        arm=args.arm,
        state_axis=args.state_axis,
        seeds=args.seeds,
        shard=args.shard,
        prompts=args.prompts,
    )
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    config["vllm_source"] = vllm_source()
    prompts = source_path(args.prompts)
    docs = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    tasks = {"aime24", "aime25"} if args.task == "aime" else {args.task}
    if not docs or any(doc["task"] not in tasks for doc in docs):
        parser.error("prompts do not match --task")
    if args.shard is not None and any(doc["idx"] % 16 != args.shard for doc in docs):
        parser.error("prompts do not match --shard")
    config.update(prompts_sha256=sha256(prompts), producer_sha256=sha256(__file__))
    label = args.arm + ("-key" if args.state_axis == "key" else "")
    run = create_run("generation", args.model, label, config, namespace=(args.task,), root=args.output_root)
    print(f"OUTPUT {run}", flush=True)
    command = VLLM + [
        "serve",
        config["checkpoint"],
        "--dtype",
        "bfloat16",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--tensor-parallel-size",
        str(config["tp"]),
        "--max-model-len",
        "66560" if args.task == "aime" else "69632",
        "--max-num-seqs",
        str(config["max_seqs"]),
        "--gpu-memory-utilization",
        "0.9",
        "--additional-config",
        json.dumps({"linq": config["linq"]}),
    ]
    if args.model == "kimi":
        command += ["--trust-remote-code"]
    if args.eager:
        command += ["--enforce-eager"]
    if args.disable_custom_all_reduce:
        command += ["--disable-custom-all-reduce"]
    prefix = run / (f"gen_{args.arm}" + (f"_k{args.shard}" if args.shard is not None else ""))
    output_args = ["--output-root", args.output_root] if args.output_root else []
    with server(command, run / "server.log") as port:
        with (run / "generation.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
            generate(
                docs,
                args.seeds,
                prefix,
                url=f"http://127.0.0.1:{port}",
                model=config["checkpoint"],
                client=config["client"],
            )
    artifacts = {}
    expected = len(docs)
    for seed in args.seeds:
        path = Path(f"{prefix}_s{seed}.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != expected or any(row["finish"] == "error" for row in rows):
            raise RuntimeError(f"Incomplete generation: {path}")
        artifacts[path.name] = sha256(path)
        mode = ["--grade-mode", "ns"] if args.task == "aime" else []
        progress = ["--partial"] if args.task == "mmlupro" else []
        run_command(
            [
                args.grader_python,
                "-m",
                f"scripts.grading.grade_{args.task}",
                "--files",
                str(path),
                "--seeds",
                str(seed),
                *mode,
                *progress,
                *output_args,
            ],
            run / f"grade_s{seed}.log",
        )
    write_json_once(run / "complete.json", {"artifacts": artifacts})


if __name__ == "__main__":
    main()
