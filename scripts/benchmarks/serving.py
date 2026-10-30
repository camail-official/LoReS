"""Matched, fixed-batch vLLM decode timing, with an explicit prefill barrier."""

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

from lores.backends.vllm.serving import REPO, run_command, source_identity, vllm_source
from lores.evaluation.output_store import create_run, sha256, write_json_once
from lores.models.registry import model_info

MODELS = {"nemo": (1040, 8160), "qwen": (528, 8192)}


def asym_grid(args):
    """Affine INT8 state grid: explicit flag, else asymmetric on Nemotron-H only."""
    return args.model == "nemo" if getattr(args, "state_asym", None) is None else args.state_asym


def configurations(args):
    info = model_info(args.model)
    pins = json.loads((REPO / "experiments/configs/checkpoints.json").read_text())
    revision = pins["nemotronh" if args.model == "nemo" else "qwen3_5"]["revision"] if args.weight == "bf16" else None
    checkpoint = info["checkpoint"] if args.weight == "bf16" else str(args.models_root / f"{info['prefix']}-rot-{args.weight}a16")
    block, longest = MODELS[args.model]
    lengths = args.lengths or ([16, 32, 512, 1024, 4096, longest] if args.sweep == "context" else [2])
    for arm in ("base", "int8"):
        bits = 0 if arm == "base" else 8
        linq = {"state_bits": bits, "decode_tuning": str(args.decode_tuning.resolve())}
        if bits:
            linq.update(state_sr=True, state_asym=asym_grid(args), state_fast=args.state_fast, state_axis="value")
        if getattr(args, "no_wa_rot", False):
            linq["wa_rot"] = ""
        yield dict(
            arm=arm,
            checkpoint=checkpoint,
            revision=revision,
            linq=linq,
            block_size=block if args.pinned else None,
            max_model_len=(8192 if args.model == "nemo" else 8224) if args.sweep == "context" else 4096,
            lengths=lengths,
            concurrencies=args.concurrencies,
        )


def fixed_batch(engine, length, concurrency, seed, profile_path=None):
    import torch
    from vllm import SamplingParams

    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    if engine.has_unfinished_requests() or scheduler.requests or core.batch_queue:
        raise RuntimeError("Previous batch did not drain")
    rng = random.Random(seed)
    sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, detokenize=False)
    for i in range(concurrency):
        engine.add_request(
            f"batch-{seed}-{i}", {"prompt_token_ids": [rng.randrange(100, 16000) for _ in range(length)]}, sampling
        )
    requests = list(scheduler.requests.values())
    prefill_steps = 0
    # Finish prefill synchronously; only decode uses the asynchronous batch queue.
    step_fn = core.step_fn
    core.step_fn = core.step
    try:
        while any(r.num_computed_tokens < length for r in requests):
            for request in requests:
                request.next_decode_eligible_step = 10**12 if request.num_computed_tokens >= length else 0
            engine.step()
            prefill_steps += 1
            if prefill_steps > concurrency * (length + 1):
                raise RuntimeError("Prefill barrier made no progress")
    finally:
        core.step_fn = step_fn
    if len(scheduler.running) != concurrency or scheduler.waiting or any(r.num_output_tokens != 1 for r in requests):
        raise RuntimeError("Not a complete fixed batch at the decode barrier")
    for request in requests:
        request.next_decode_eligible_step = 0
    torch.cuda.synchronize()
    profiler = None
    if profile_path is not None:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        profiler.start()
    engine_step_ms = []
    start = previous = time.perf_counter()
    while engine.has_unfinished_requests():
        engine.step()
        now = time.perf_counter()
        engine_step_ms.append((now - previous) * 1000)
        previous = now
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    if profiler is not None:
        profiler.stop()
        profiler.export_chrome_trace(str(profile_path))
    while core.batch_queue:
        engine.step()
    if engine.has_unfinished_requests() or any(r.num_output_tokens != 32 for r in requests):
        raise RuntimeError("Expected exactly 32 output tokens per request")
    if any(r.num_computed_tokens != length + 31 or r.num_preemptions for r in requests):
        raise RuntimeError("Decode batch was preempted or did not stay uniform")
    return dict(
        decode_ms=elapsed_ms,
        tpot_ms=elapsed_ms / 31,
        engine_step_ms=engine_step_ms,
        prefill_steps=prefill_steps,
        completed=concurrency,
        output_tokens=32,
    )


def cache_layout(engine):
    model = engine._get_driver_model_for_cleanup()
    rows = {}
    for name, module in model.named_modules():
        cache = getattr(module, "kv_cache", None)
        if isinstance(cache, (tuple, list)) and len(cache) in (2, 3):
            tensors = [x for x in cache if hasattr(x, "stride")]
            if len(tensors) == len(cache):
                rows[name] = [dict(shape=list(x.shape), stride=list(x.stride()), dtype=str(x.dtype)) for x in tensors]
    return rows


def worker(config_path):
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor.abstract import Executor

    config = json.loads(config_path.read_text())
    entry = config["entry"]
    if config["functional_check"]:
        entry["linq"].pop("decode_tuning", None)
    vllm_source()
    gpu = torch.cuda.get_device_name()
    if not config["functional_check"] and gpu != "NVIDIA H100 80GB HBM3":
        raise RuntimeError(f"Expected a dedicated H100 SXM, got {gpu}")
    kwargs = dict(
        model=entry["checkpoint"],
        revision=entry["revision"],
        dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=entry["max_model_len"],
        max_num_seqs=max(64, max(entry["concurrencies"] or [64])),
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        async_scheduling=config["async_scheduling"],
        disable_log_stats=True,
        seed=config["seed"],
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
        additional_config={"linq": entry["linq"]},
    )
    if entry["block_size"] is not None:
        kwargs["block_size"] = entry["block_size"]
    engine_config = EngineArgs(**kwargs).create_engine_config()
    engine = LLMEngine(engine_config, Executor.get_class(engine_config), log_stats=False, multiprocess_mode=False)
    out = Path(config["result_dir"])
    write_json_once(
        out / "engine.json",
        dict(
            kwargs=kwargs,
            gpu=gpu,
            gpu_properties=str(torch.cuda.get_device_properties(0)),
            torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            batch_queue_size=engine.engine_core.engine_core.batch_queue_size,
            resolved_cache_config=str(engine.vllm_config.cache_config),
            cache_layout=cache_layout(engine),
        ),
    )
    rows = []
    try:
        for length in entry["lengths"]:
            for concurrency in entry["concurrencies"]:
                for warm in range(config["warmup_batches"]):
                    fixed_batch(engine, length, concurrency, config["seed"] + warm)
                if config.get("profile_decode"):
                    fixed_batch(engine, length, concurrency, config["seed"], out / f"profile_T{length}_c{concurrency}.json")
                windows = [fixed_batch(engine, length, concurrency, config["seed"] + 100 + i) for i in range(config["batches"])]
                row = dict(
                    arm=entry["arm"],
                    repetition=config["repetition"],
                    length=length,
                    concurrency=concurrency,
                    windows=windows,
                    mean_tpot_ms=statistics.mean(w["tpot_ms"] for w in windows),
                    total_decode_ms=sum(w["decode_ms"] for w in windows),
                )
                write_json_once(out / f"T{length}_c{concurrency}.json", row)
                rows.append(row)
                print(f"CELL {entry['arm']} T={length} c={concurrency}: {row['mean_tpot_ms']:.4f} ms", flush=True)
        write_json_once(out / "complete.json", dict(cells=len(rows)))
    finally:
        engine.engine_core.shutdown()


def summarize(run, configs):
    grouped = {}
    for config in configs:
        for path in Path(config["result_dir"]).glob("T*_c*.json"):
            row = json.loads(path.read_text())
            grouped.setdefault((row["length"], row["concurrency"]), {}).setdefault(row["arm"], []).append(row)
    rows = []
    for (length, concurrency), arms in sorted(grouped.items()):
        baseline, quantized = ("norot", "rk") if "rk" in arms else ("base", "int8")
        if set(arms) != {baseline, quantized}:
            raise RuntimeError(f"Missing comparison arm at T={length}, c={concurrency}")
        means = {arm: statistics.median(x["mean_tpot_ms"] for x in cells) for arm, cells in arms.items()}
        paired = [
            a["mean_tpot_ms"] / b["mean_tpot_ms"]
            for a, b in zip(
                sorted(arms[baseline], key=lambda x: x["repetition"]),
                sorted(arms[quantized], key=lambda x: x["repetition"]),
                strict=True,
            )
        ]
        row = dict(
            length=length,
            concurrency=concurrency,
            median_tpot_ms=means,
            speedup=means[baseline] / means[quantized],
            paired_speedups=paired,
            minimum_aggregate_decode_ms=min(x["total_decode_ms"] for cells in arms.values() for x in cells),
        )
        rows.append(row)
        print(f"RESULT T={length} c={concurrency}: {means}, speedup={row['speedup']:.4f}x", flush=True)
    write_json_once(run / "summary.json", rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=("concurrency", "context"))
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--weight", choices=("bf16", "w4", "w8"))
    parser.add_argument("--models-root", type=Path, default=Path("models/hf"))
    parser.add_argument("--concurrencies", type=int, nargs="+")
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM memory budget for both arms")
    parser.add_argument("--state-asym", action=argparse.BooleanOptionalAction, help="INT8 grid; default asym on nemo")
    parser.add_argument("--state-fast", action="store_true", help="use the FAST INT8 codec and matching kernel tuning")
    parser.add_argument("--no-wa-rot", action="store_true", help="disable online W/A rotations (timing-only control)")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-tag", default="h100")
    parser.add_argument("--output-root")
    parser.add_argument("--decode-tuning", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--functional-check", action="store_true")
    parser.add_argument("--profile-decode", action="store_true", help="save a separate, untimed decode profiler trace per cell")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config)
        return
    if any(x is None for x in (args.model, args.weight, args.sweep, args.decode_tuning)):
        parser.error("--model, --weight, --sweep and --decode-tuning are required")
    if args.pinned and (args.sweep != "concurrency" or args.weight != "w4"):
        parser.error("pinned-block control is for W4A16 concurrency only")
    args.concurrencies = args.concurrencies or ([2, 4, 8, 16, 32] if args.sweep == "context" else [1, 2, 4, 8, 16, 32, 64])
    if min(args.concurrencies + (args.lengths or [1]) + [args.repetitions, args.batches, args.warmup_batches]) < 1:
        parser.error("counts and lengths must be positive")
    if any(len(values) != len(set(values)) for values in (args.concurrencies, args.lengths or [])):
        parser.error("duplicate grid cells")
    if args.functional_check and not args.pilot:
        parser.error("--functional-check requires --pilot; it is never a reported timing")
    entries = list(configurations(args))
    if any(max(entry["lengths"]) + 32 > entry["max_model_len"] for entry in entries):
        parser.error("prompt plus output exceeds the model-length setting")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("worker_config", "dry_run")}
    config.update(
        entries=entries,
        output_tokens=32,
        measured_decode_steps=31,
        async_scheduling=True,
        protocol="fixed batch, all-prefill barrier; asynchronous vLLM decode wall time, no HTTP/tokenization",
        arm_order="AB/BA alternating paired repetitions",
        prompt_token_range=[100, 16000],
    )
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    tuning = json.loads(args.decode_tuning.read_text())
    expected = "mamba" if args.model == "nemo" else "gdn"
    if not args.functional_check and not all(tuning.get(expected + suffix) for suffix in ("_fp", "_int8")):
        parser.error("tuning file does not contain both model launch tables")
    metadata = tuning.get("metadata", {})
    metadata = metadata.get("profiles", {}).get(args.model, metadata)
    if not args.functional_check and not (
        metadata.get("profile") == args.model
        and metadata.get("state_sr") is True
        and metadata.get("state_fast") is args.state_fast
        and metadata.get("gpu_tag") == "h100"
        and metadata.get("state_asym") in (None, asym_grid(args))
    ):
        parser.error(f"tuning must match {args.model}, SR, state_fast={args.state_fast}, and H100")
    for entry in entries:
        checkpoint = Path(entry["checkpoint"])
        if args.weight != "bf16":
            declaration = json.loads((checkpoint / "config.json").read_text())
            rotations = declaration.get("linq_wa_rot")
            if rotations != ("r4s,ro" if args.model == "nemo" else "r4,ro"):
                raise RuntimeError(f"Missing/wrong {args.weight.upper()} online rotations: {rotations}")
            groups = declaration["quantization_config"]["config_groups"].values()
            if not groups or any(
                group["weights"]["num_bits"] != int(args.weight[1:])
                or group["weights"]["group_size"] != 128
                or not group["weights"]["symmetric"]
                or group.get("input_activations") is not None
                or group.get("output_activations") is not None
                for group in groups
            ):
                raise RuntimeError(f"Expected symmetric group-128 {args.weight.upper()}A16 export: {checkpoint}")
            entry["export_config_sha256"] = sha256(checkpoint / "config.json")
            entry["weight_sha256"] = {p.name: sha256(p) for p in sorted(checkpoint.glob("*.safetensors"))}
            if not entry["weight_sha256"]:
                raise RuntimeError(f"No export weights: {checkpoint}")
    config.update(source_identity(), tuning_sha256=sha256(args.decode_tuning), producer_sha256=sha256(__file__))
    label = (
        f"serving-{args.sweep}-{args.model}-{args.weight}"
        + ("-fast" if args.state_fast else "-nonfast")
        + "-async"
        + ("-no-wa-rot" if args.no_wa_rot else "")
        + ("-pinned" if args.pinned else "")
        + ("-pilot" if args.pilot else "")
    )
    run = create_run("benchmarks", "shared", label, config, device=args.gpu_tag, root=args.output_root)
    print(f"OUTPUT {run}", flush=True)
    write_json_once(run / "tuning.json", tuning)
    jobs = []
    for rep in range(args.repetitions):
        for entry in entries if rep % 2 == 0 else list(reversed(entries)):
            name = f"{entry['arm']}_r{rep + 1}"
            result_dir = run / name
            result_dir.mkdir()
            entry = dict(entry, linq=dict(entry["linq"], decode_tuning=str(run / "tuning.json")))
            child = dict(
                entry=entry,
                repetition=rep + 1,
                result_dir=str(result_dir),
                seed=args.seed + rep * 1000,
                batches=args.batches,
                warmup_batches=args.warmup_batches,
                functional_check=args.functional_check,
                profile_decode=args.profile_decode,
                async_scheduling=config["async_scheduling"],
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            config_path = run / f"{name}.json"
            write_json_once(config_path, child)
            jobs.append(child)
            run_command(
                [sys.executable, "-u", "-m", "scripts.benchmarks.serving", "--worker-config", str(config_path)],
                run / f"{name}.log",
            )
    rows = summarize(run, jobs)
    if args.pilot:
        minimum = min(row["minimum_aggregate_decode_ms"] for row in rows)
        recommended = max(args.batches, math.ceil(args.batches * 2000 / minimum))
        write_json_once(run / "pilot.json", dict(recommended_batches=recommended, target_decode_ms=2000))
        print(f"PILOT recommended batches per repetition: {recommended}", flush=True)
    write_json_once(run / "complete.json", dict(cells=len(rows), summary_sha256=sha256(run / "summary.json")))
    print(f"COMPLETE {run}", flush=True)


if __name__ == "__main__":
    main()
