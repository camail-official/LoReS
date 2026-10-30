"""Select recurrent-kernel launches using fixed-batch full-model decode timing."""

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path

from lores.backends.vllm.serving import run_command, source_identity
from lores.evaluation.output_store import create_run, sha256, write_json_once
from scripts.benchmarks import serving


def read(path):
    return json.loads(path.read_text())


def shortlist(kernel_run, model, batches, count):
    arch, prefix = ("gdn", "gdn") if model == "qwen" else ("mamba2", "mamba")
    result, hashes = {}, {}
    for batch in batches:
        path = kernel_run / f"{arch}_b{batch}.json"
        row = read(path)
        hashes[path.name] = sha256(path)
        result[str(batch)] = {}
        for arm, source_arm in (("base", "fp32"), ("int8", "int8")):
            candidates = sorted((c for c in row["candidates"][source_arm] if "median_us" in c), key=lambda c: c["median_us"])
            incumbent = next(c for c in candidates if c["launch"] == row["selected"][source_arm])
            chosen = [incumbent]
            for candidate in candidates:
                if candidate["launch"][1] not in {c["launch"][1] for c in chosen}:
                    chosen.append(candidate)
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f"Not enough warp-count alternatives for {model}/{arm}/B{batch}")
            kind = prefix + ("_fp" if arm == "base" else "_int8")
            key = batch * row["shape"]["heads"] if kind == "mamba_fp" else batch
            result[str(batch)][arm] = dict(kind=kind, key=str(key), candidates=[dict(launch=c["launch"], isolated_us=c["median_us"]) for c in chosen])
    return result, hashes


def tables_for(shortlists, arm, choices):
    tables = {}
    for batch, arms in shortlists.items():
        item = arms[arm]
        launch = item["candidates"][choices[batch]]["launch"]
        value = dict(BLOCK_SIZE_M=launch[0], num_warps=launch[1]) if item["kind"] == "mamba_fp" else launch
        tables.setdefault(item["kind"], {})[item["key"]] = value
    return tables


def worker(path):
    from lores.backends.vllm import decode_tuning

    config = read(path)
    original = decode_tuning.override_table
    observed = {}

    def lookup(kind):
        table = original(kind)
        updates = config["launch_tables"].get(kind)
        if updates is not None:
            table = {**(table or {}), **{int(k): v for k, v in updates.items()}}
            observed[kind] = table
        return table

    decode_tuning.override_table = lookup
    try:
        serving.worker(path)
        if set(observed) != set(config["launch_tables"]):
            raise RuntimeError("The serving worker did not consume the requested launch tables")
        write_json_once(Path(config["result_dir"]) / "launch_tables.json", observed)
    finally:
        decode_tuning.override_table = original


def measure(run, phase, template, tables, repetition, candidate, args, seed, hashes):
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("Benchmark source changed during the run")
    name = f"{template['weight']}_{template['arm']}_c{candidate}_r{repetition}"
    directory = run / phase
    out = directory / name
    out.mkdir(parents=True)
    entry = copy.deepcopy(template)
    entry["linq"]["decode_tuning"] = str(run / "kernel_tuning.json")
    config = dict(entry=entry, repetition=repetition, result_dir=str(out), seed=seed,
                  batches=args.batches, warmup_batches=2, functional_check=False,
                  profile_decode=False, async_scheduling=True, launch_tables=tables)
    path = directory / f"{name}.json"
    write_json_once(path, config)
    print(f"START {phase}/{name}", flush=True)
    run_command([sys.executable, "-u", "-m", "scripts.benchmarks.serving_tuning", "--worker-config", str(path)], directory / f"{name}.log")
    engine = read(out / "engine.json")
    observed = read(out / "launch_tables.json")
    if engine["batch_queue_size"] != 2 or any(observed[k].get(b) != v for k, values in tables.items() for b, v in values.items()):
        raise RuntimeError(f"Wrong queue or launch settings in {name}")
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("Benchmark source changed during the worker")
    records = []
    for length in args.lengths:
        for batch in args.concurrencies:
            row = read(out / f"T{length}_c{batch}.json")
            if len(row["windows"]) != args.batches or any(w["completed"] != batch or w["output_tokens"] != 32 for w in row["windows"]):
                raise RuntimeError(f"Incomplete timing windows in {name}")
            mean = statistics.mean(w["tpot_ms"] for w in row["windows"])
            if not math.isfinite(mean) or not math.isclose(mean, row["mean_tpot_ms"], rel_tol=1e-12):
                raise RuntimeError(f"Invalid timing summary in {name}")
            records.append(dict(weight=entry["weight"], arm=entry["arm"], candidate=candidate, repetition=repetition,
                                length=length, batch=batch, tpot_ms=mean))
    print(f"DONE {phase}/{name}: {len(records)} cells", flush=True)
    return records, engine["gpu_properties"], config


def select(records, shortlists, args):
    choices = {arm: {} for arm in ("base", "int8")}
    scores = []
    for batch in args.concurrencies:
        for arm in choices:
            latencies = {}
            for candidate in range(args.candidates):
                latencies[candidate] = {}
                for weight in args.weights:
                    for length in args.lengths:
                        values = [r["tpot_ms"] for r in records if (r["batch"], r["arm"], r["candidate"], r["weight"], r["length"]) == (batch, arm, candidate, weight, length)]
                        if len(values) != args.repetitions:
                            raise RuntimeError("Incomplete selection grid")
                        latencies[candidate][weight, length] = statistics.median(values)
            ratios = {c: math.exp(statistics.mean(math.log(t / latencies[0][key]) for key, t in cells.items())) for c, cells in latencies.items()}
            winner = min(ratios, key=lambda c: (ratios[c], c))
            choices[arm][str(batch)] = winner
            scores.append(dict(batch=batch, arm=arm, selected=winner, candidates=[dict(
                candidate=c, launch=shortlists[str(batch)][arm]["candidates"][c]["launch"],
                latency_ratio=ratios[c], cells=[dict(weight=w, length=t, median_tpot_ms=ms) for (w, t), ms in cells.items()]
            ) for c, cells in latencies.items()]))
    return choices, scores


def merge(runs, output):
    merged, common, scopes = None, None, {}
    fields = ("model", "weights", "state_fast", "state_sr", "async_scheduling", "candidates",
              "repetitions", "batches", "warmup_batches", "kernel_tuning_sha256", "selection_rule",
              "shortlist_rule", "selection_seed", "checkpoint_sha256", "source_sha256")
    for run in runs:
        plan = read(run / "run.json")["config"]
        selected = read(run / "tuning.json")
        if read(run / "complete.json")["tuning_sha256"] != sha256(run / "tuning.json"):
            raise ValueError(f"Incomplete or modified selection: {run}")
        identity = {key: plan[key] for key in fields}
        if common is not None and identity != common:
            raise ValueError("Selection shards use different protocols or source inputs")
        if merged is None:
            common = identity
            merged = read(run / "kernel_tuning.json")
        if sha256(run / "kernel_tuning.json") != plan["kernel_tuning_sha256"]:
            raise ValueError("Changed incumbent tuning")
        choices = selected["metadata"]["full_model_selection"]["choices"]
        for batch in plan["concurrencies"]:
            if str(batch) in scopes:
                raise ValueError(f"Overlapping batch selection: {batch}")
            scopes[str(batch)] = dict(run=str(run.resolve()), lengths=plan["lengths"], sweep=plan["sweep"],
                                      tuning_sha256=sha256(run / "tuning.json"))
        for arm in ("base", "int8"):
            for kind, table in tables_for(plan["shortlists"], arm, choices[arm]).items():
                if any(selected[kind][key] != value for key, value in table.items()):
                    raise ValueError("Selected table does not match recorded choices")
                merged[kind].update(table)
    if set(scopes) != {str(b) for b in (1, 2, 4, 8, 16, 32, 64)}:
        raise ValueError("Selection must cover every serving batch exactly once")
    merged["metadata"]["full_model_selection"] = dict(protocol=common, batches=scopes)
    write_json_once(output, merged)
    print(f"MERGED {output} {sha256(output)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("nemo", "qwen"))
    parser.add_argument("--kernel-run", type=Path)
    parser.add_argument("--sweep", choices=("concurrency", "context"), default="context")
    parser.add_argument("--weights", choices=("bf16", "w4"), nargs="+", default=["bf16", "w4"])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 512, 4096])
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--validation-repetitions", type=int, default=7)
    parser.add_argument("--selection-only", action="store_true", help="freeze choices; measure them in a separate serving sweep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--state-fast", action="store_true")
    parser.add_argument("--models-root", type=Path, default=Path("models/hf"))
    parser.add_argument("--output-root")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--merge-runs", type=Path, nargs="+")
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config)
        return
    if args.merge_runs:
        if args.merged_output is None:
            parser.error("--merge-runs requires --merged-output")
        merge(args.merge_runs, args.merged_output)
        return
    if args.model is None or args.kernel_run is None:
        parser.error("--model and --kernel-run are required")
    if not 2 <= args.candidates <= 4 or min(args.repetitions, args.validation_repetitions, args.batches, *args.lengths, *args.concurrencies) < 1:
        parser.error("use 2–4 candidates and positive counts")
    if max(args.concurrencies) > 64 or any(len(v) != len(set(v)) for v in (args.weights, args.lengths, args.concurrencies)):
        parser.error("duplicate entries or unsupported concurrency")
    kernel_run = args.kernel_run.resolve()
    tuning = read(kernel_run / "tuning.json")
    metadata = tuning["metadata"]
    if not (metadata["profile"] == args.model and metadata["state_sr"] is True and metadata["state_fast"] is args.state_fast
            and metadata["gpu_tag"] == "h100" and metadata.get("state_asym") in (None, args.model == "nemo")):
        parser.error("kernel sweep must match the model, FAST mode, state grid, SR, and H100")
    if not (kernel_run / "complete.json").is_file():
        parser.error("kernel sweep is incomplete")
    if read(kernel_run / "complete.json")["tuning_sha256"] != sha256(kernel_run / "tuning.json"):
        parser.error("kernel tuning checksum mismatch")
    shortlists, input_hashes = shortlist(kernel_run, args.model, args.concurrencies, args.candidates)
    for arm in ("base", "int8"):
        for kind, table in tables_for(shortlists, arm, dict.fromkeys(shortlists, 0)).items():
            if any(tuning[kind].get(key) != value for key, value in table.items()):
                parser.error("kernel candidate rows do not match the incumbent tuning table")
    settings = argparse.Namespace(**vars(args), pinned=False, no_wa_rot=False, decode_tuning=kernel_run / "tuning.json")
    templates = {}
    for weight in args.weights:
        settings.weight = weight
        for entry in serving.configurations(settings):
            entry["weight"] = weight
            if max(args.lengths) + 32 > entry["max_model_len"]:
                parser.error("prompt plus output exceeds max_model_len")
            templates[weight, entry["arm"]] = entry
    plan = dict(model=args.model, weights=args.weights, concurrencies=args.concurrencies, lengths=args.lengths, sweep=args.sweep,
                state_fast=args.state_fast, state_sr=True, async_scheduling=True, candidates=args.candidates,
                repetitions=args.repetitions, validation_repetitions=0 if args.selection_only else args.validation_repetitions, batches=args.batches,
                warmup_batches=2, kernel_run=str(kernel_run), kernel_tuning_sha256=sha256(kernel_run / "tuning.json"),
                candidate_source_sha256=input_hashes, shortlists=shortlists, entries=list(templates.values()),
                selection_rule="minimum geometric mean of context/weight latency ratios to incumbent; each cell uses median of worker means",
                shortlist_rule="incumbent plus fastest representatives of other warp counts",
                selection_seed=args.seed, validation_seed=args.seed + 100000,
                validation_policy="separate serving sweep" if args.selection_only else "freeze first; seven paired repeats by default; never reselect on validation",
                selection_workers=len(templates) * args.candidates * args.repetitions,
                validation_workers=0 if args.selection_only else len(templates) * args.validation_repetitions)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for weight in args.weights:
        if weight == "w4":
            checkpoint = Path(templates[weight, "base"]["checkpoint"])
            if read(checkpoint / "config.json").get("linq_wa_rot") != ("r4s,ro" if args.model == "nemo" else "r4,ro"):
                raise RuntimeError("Missing W4 online rotation declaration")
            files = [checkpoint / "config.json", *sorted(checkpoint.glob("*.safetensors"))]
            if len(files) < 2:
                raise RuntimeError("Missing W4 export weights")
            plan["checkpoint_sha256"] = {str(p): sha256(p) for p in files}
    paths = [Path(__file__), Path(serving.__file__), Path("lores/backends/vllm/decode_tuning.py"), Path("lores/backends/vllm/state.py")]
    hashes = {str(p.resolve()): sha256(p) for p in paths}
    plan.update(source_identity(), source_sha256=hashes)
    run = create_run("benchmarks", "shared", f"serving-tuning-{args.model}-{'fast' if args.state_fast else 'nonfast'}-async", plan, device="h100", root=args.output_root)
    print(f"OUTPUT {run}", flush=True)
    write_json_once(run / "kernel_tuning.json", tuning)
    records, device = [], None

    def run_one(phase, template, tables, rep, candidate, seed):
        nonlocal device
        rows, gpu, config = measure(run, phase, template, tables, rep + 1, candidate, args, seed, hashes)
        if device is not None and gpu != device:
            raise RuntimeError("GPU changed within the selection/validation run")
        device = gpu
        return rows, config

    for rep in range(args.repetitions):
        ranks = list(range(args.candidates))
        ranks = ranks[rep % len(ranks):] + ranks[:rep % len(ranks)]
        for rank in ranks:
            for weight in args.weights[::1 if rep % 2 == 0 else -1]:
                for arm in ("base", "int8")[::1 if (rep + rank) % 2 == 0 else -1]:
                    tables = tables_for(shortlists, arm, dict.fromkeys(shortlists, rank))
                    rows, _ = run_one("selection", templates[weight, arm], tables, rep, rank, args.seed + rep * 1000)
                    records.extend(rows)
    choices, scores = select(records, shortlists, args)
    selected = copy.deepcopy(tuning)
    for arm in choices:
        for kind, table in tables_for(shortlists, arm, choices[arm]).items():
            selected[kind].update(table)
    selected["metadata"]["full_model_selection"] = dict(run=str(run), weights=args.weights, contexts=args.lengths,
                                                       batches=args.concurrencies, async_scheduling=True, choices=choices)
    write_json_once(run / "selection.json", scores)
    write_json_once(run / "tuning.json", selected)
    write_json_once(run / "selection_complete.json", dict(workers=plan["selection_workers"], tuning_sha256=sha256(run / "tuning.json")))
    print(f"SELECTED {json.dumps(choices, sort_keys=True)}", flush=True)
    if args.selection_only:
        write_json_once(run / "complete.json", dict(selection_workers=plan["selection_workers"], validation_workers=0,
                                                  tuning_sha256=sha256(run / "tuning.json")))
        print(f"COMPLETE {run}", flush=True)
        return
    validation = {weight: [] for weight in args.weights}
    for rep in range(args.validation_repetitions):
        for weight in args.weights[::1 if rep % 2 == 0 else -1]:
            for arm in ("base", "int8")[::1 if rep % 2 == 0 else -1]:
                _, config = run_one("validation", templates[weight, arm], tables_for(shortlists, arm, choices[arm]), rep, "selected", args.seed + 100000 + rep * 1000)
                validation[weight].append(config)
    for weight, configs in validation.items():
        destination = run / "validation" / weight
        destination.mkdir()
        serving.summarize(destination, configs)
    write_json_once(run / "complete.json", dict(selection_workers=plan["selection_workers"], validation_workers=plan["validation_workers"],
                                              tuning_sha256=sha256(run / "tuning.json"), validation_sha256={w: sha256(run / "validation" / w / "summary.json") for w in args.weights}))
    print(f"COMPLETE {run}", flush=True)


if __name__ == "__main__":
    main()
