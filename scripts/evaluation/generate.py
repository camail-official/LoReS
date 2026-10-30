"""Run downstream lm-eval generation with decode-time state quantization, one configuration per invocation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import transformers

from lores.backends.hf.loading import load_hybrid
from lores.backends.hf.state import ROTATIONS, install_rotations, pin_fp32_state, state_installer, state_quant_mode
from lores.evaluation.generation import evaluate_generation
from lores.evaluation.state_quant import bind_mask_quantizers
from lores.models.registry import checkpoint_for


def _device():
    if not torch.cuda.is_available():
        return "cpu"
    lr = os.environ.get("LOCAL_RANK")  # set by `accelerate launch` — one model replica per GPU (DP eval)
    return f"cuda:{lr}" if lr is not None else "cuda"


def load_model(args, mode):
    device = _device()
    arch = args.model
    wa = args.w_bits, args.a_bits
    if arch == "kimi" and wa != (16, 16):
        raise ValueError("HF W/A quantization is supported for Qwen and Nemotron only")
    tok, model = load_hybrid(
        arch,
        device=device,
        device_map=device,
        attn_implementation=args.attn_impl if arch == "nemotronh" else None,
    )
    if arch == "nemotronh":
        model.config.mamba_ssm_cache_dtype = "float32"
        pin_fp32_state(model, arch, strict=True)
        eos_ids = sorted({2, tok.eos_token_id} - {None})
        model.generation_config.eos_token_id = model.config.eos_token_id = eos_ids
    elif arch == "kimi":
        model.config.max_position_embeddings = 131072
        if args.fp32_state:
            pin_fp32_state(model, arch)
    # W/A calibration precedes state quantization.
    if wa != (16, 16):
        from lores.wa_ptq.weights import install_wa

        install_wa(model, *wa, device=device, w_file=args.w_file, arch=arch)
    if arch == "nemotronh" and args.rot != "none":
        with torch.no_grad():
            model(tok("1", return_tensors="pt").input_ids.to(device))
    set_mode = state_installer(model, arch, block_size=args.block_size)
    if mode != "off":
        set_mode(mode)
    install_rotations(model, arch, args.rot)
    if arch == "qwen3_5" and args.fp32_state:
        pin_fp32_state(model, arch)
    bind_mask_quantizers(model, args, mode)
    return checkpoint_for(arch), tok, model


_TASK_CAPS = (("gsm8k", 8192), ("aime", 65536), ("gpqa", 65536), ("mmlu_pro", 65536))


def _chat_probe_tail(ckpt: str, system_instruction: str | None) -> str:
    """Render a minimal generation prompt so the caller can assert on how it ends."""
    tok = transformers.AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    msgs = ([{"role": "system", "content": system_instruction}] if system_instruction else []) + [
        {"role": "user", "content": "x"}
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# In HF chat runs, inherited question delimiters can truncate reasoning.
_UNTIL_OVERRIDE_TASKS = ("gsm8k", "aime", "mmlu_pro")
_UNTIL = "<|im_end|>"


def task_gen_kwargs(task: str, cli_gen_kwargs: str | None, *, model=None, backend="hf") -> str | None:
    """Resolve the gen_kwargs for one task under the cap + stop-string policies; hard-error on conflicts."""
    cap = next((c for pat, c in _TASK_CAPS if pat in task.lower()), None)
    if cli_gen_kwargs is None:
        cli_gen_kwargs = f"max_gen_toks={cap}" if cap is not None else None
    else:
        m = re.search(r"max_gen_toks=(\d+)", cli_gen_kwargs)
        if cap is not None:
            assert m and int(m.group(1)) == cap, f"cap policy: task {task!r} must run max_gen_toks={cap}, got {cli_gen_kwargs!r}"
        else:
            assert m is None, (
                f"cap policy: task {task!r} runs the lm-eval default — do not pass max_gen_toks ({cli_gen_kwargs!r})"
            )
    if any(pat in task.lower() for pat in _UNTIL_OVERRIDE_TASKS):
        mu = re.search(r"until=([^,]*)", cli_gen_kwargs or "")
        if mu:
            assert mu.group(1) == _UNTIL, f"stop-string policy: task {task!r} runs until={_UNTIL}, got {cli_gen_kwargs!r}"
        else:
            cli_gen_kwargs = f"{cli_gen_kwargs},until={_UNTIL}" if cli_gen_kwargs else f"until={_UNTIL}"
    # Nemotron on HF samples with top_k=50 unless gen_kwargs set top_k.
    if model == "nemotronh" and backend == "hf" and not re.search(r"(?:^|,)\s*top_k\s*=", cli_gen_kwargs or ""):
        cli_gen_kwargs = f"{cli_gen_kwargs},top_k=50" if cli_gen_kwargs else "top_k=50"
    return cli_gen_kwargs


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ROTATIONS))
    ap.add_argument("--rot", default="none", choices=sorted({r for rots in ROTATIONS.values() for r in rots}))
    ap.add_argument("--mode", default="off", help="state format/rounding: off, int8, int8_sr, int8_asym_sr, fpE4M3, ...")
    ap.add_argument(
        "--state-axis", choices=("value", "key"), default="value", help="scale grouping, independent of storage layout"
    )
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument(
        "--fp32-state", action="store_true", help="HF Qwen/Kimi: pin recurrent state to FP32 (Nemotron is always pinned)"
    )
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--tasks", default="gsm8k_cot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", default="8")
    ap.add_argument("--gen-kwargs", default=None)
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="lm-eval seed override; default uses harness seeds")
    ap.add_argument(
        "--include-path", default=None, help="extra lm-eval task dir (repo-local variants under lores/lmeval_tasks)"
    )
    ap.add_argument(
        "--mixhead", default=None, metavar="MASK_JSON", help="axis mask: true = value, false = key; overrides --state-axis"
    )
    ap.add_argument(
        "--attn-impl",
        default="eager",
        choices=("eager", "sdpa"),
        help="Nemotron attention implementation; SDPA supports longer prefills",
    )
    ap.add_argument(
        "--headbits",
        default=None,
        metavar="BITS_JSON",
        help="per-(layer, head) BIT allocation json {layer: [bits]} at --state-axis grouping; "
        "with --mixhead composes to AxisBitsQuant (the MP promotion recipe)",
    )
    ap.add_argument("--w-bits", type=int, default=16, choices=[16, 8, 4], help="Qwen/Nemotron GPTQ weight precision")
    ap.add_argument(
        "--a-bits",
        type=int,
        default=16,
        choices=[16, 8, 4],
        help="qwen/nemo: per-token linear-input fake quantization; lm_head remains unquantized",
    )
    ap.add_argument(
        "--w-file",
        default=None,
        metavar="PT",
        help="post-GPTQ state_dict for --w-bits: load if it exists, else calibrate and save. "
        "Calibrate once per (model, w-bits) and reuse the file across shards; GPTQ is not "
        "deterministic across devices (~18 GB).",
    )
    ap.add_argument(
        "--task-metadata", default=None, help="json TaskManager metadata, e.g. '{\"max_seq_lengths\":[4096,8192]}' for RULER"
    )
    ap.add_argument("--max-length", type=int, default=None, help="HFLM max_length (RULER: set >= largest seq len)")
    ap.add_argument(
        "--reasoning-off",
        action="store_true",
        help="disable reasoning: nemo system {'reasoning': False}; qwen chat-template enable_thinking=False (RULER protocol)",
    )
    ap.add_argument(
        "--backend",
        choices=["hf", "vllm"],
        default="hf",
        help="hf = the fake-quant harness (live model + decode hooks); vllm = the "
        "deployment path (lm-eval's VLLM builds its own engine, state quant "
        "from additional_config). Same task/few-shot/template/scoring either way.",
    )
    ap.add_argument("--state-fast", action="store_true", help="--backend vllm: enable reciprocal-multiply codecs.")
    ap.add_argument(
        "--vllm-ckpt",
        default=None,
        help="--backend vllm: serve this checkpoint instead of the stock id (a W/A export dir; "
        "its config.json declares the online rotations). W/A bits then live in the checkpoint.",
    )
    ap.add_argument(
        "--vllm-kwargs",
        default=None,
        help="--backend vllm: JSON dict forwarded to lm-eval's VLLM "
        '(e.g. \'{"gpu_memory_utilization": 0.85, "enforce_eager": true}\').',
    )
    ap.add_argument(
        "--stem",
        default=None,
        help="override the task part of the output stem (e.g. 'ruler' when passing all 13 subtasks explicitly)",
    )
    ap.add_argument(
        "--outdir", default=None, help="explicit output directory under output-root/new; default: fresh generation attempt"
    )
    return ap


def main():
    args = build_parser().parse_args()
    mode = state_quant_mode(args.mode, args.model, args.state_axis) if args.backend == "hf" else None
    if args.backend != "hf" and args.fp32_state:
        raise ValueError("--fp32-state requires --backend hf")
    from lores.evaluation.output_store import (
        create_run,
        source_path,
        writable_output,
    )

    if args.outdir:
        writable_output(args.outdir, root=args.output_root)
    for name in ("mixhead", "headbits"):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, str(source_path(value)))

    rots = ROTATIONS[args.model]
    assert args.rot in rots, f"{args.model}: rot must be one of {rots}"
    if args.mixhead or args.headbits:
        assert re.fullmatch(r"int[2-8](_asym)?(_sr)?", args.mode), (
            f"--mixhead/--headbits need an int<n>[_asym][_sr] mode, got {args.mode!r}"
        )
    if args.backend == "vllm":
        from lores.backends.vllm.config import LinqConfig

        assert args.rot == "none", f"--backend vllm: rotations not wired, got --rot {args.rot!r}"
        assert not (args.mixhead or args.headbits), "--backend vllm: no mixed-precision path"
        assert args.vllm_ckpt or (args.w_bits, args.a_bits) == (16, 16), (
            "--backend vllm: W/A quantization lives in the checkpoint; pass --vllm-ckpt <export dir>"
        )
        if args.state_axis == "key" and args.model != "kimi":
            raise ValueError("--backend vllm: key-axis packing is Kimi-only")
        m = re.fullmatch(r"int8(_asym)?(_sr)?", args.mode)
        if args.mode != "off" and not m:
            raise ValueError("--backend vllm: mode must be off or int8[_asym][_sr]")
        linq = {
            "state_bits": 8 if m else 0,
            "state_axis": args.state_axis if m else "value",
            "state_asym": bool(m and m[1]),
            "state_sr": bool(m and m[2]),
            "state_fast": args.state_fast,
        }
        vk = json.loads(args.vllm_kwargs) if args.vllm_kwargs else {}
        settings = vk.setdefault("additional_config", {}).setdefault("linq", {})
        settings.update(linq)
        LinqConfig.from_dict(settings)
        args.vllm_kwargs = json.dumps(vk)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # Bare AIME names are aliases for the same NeMo-Skills prompt on every model.
    tasks = [{"aime24": "aime24_ns", "aime25": "aime25_ns"}.get(t, t) for t in tasks]
    if any(t in ("aime24_ns", "aime25_ns") for t in tasks):
        task_dir = str(Path(__file__).resolve().parents[2] / "lores" / "lmeval_tasks")
        args.include_path = (
            [args.include_path, task_dir]
            if args.include_path and Path(args.include_path).resolve() != Path(task_dir)
            else task_dir
        )
    resolved_gen_kwargs = {t: task_gen_kwargs(t, args.gen_kwargs, model=args.model, backend=args.backend) for t in tasks}
    args.system_instruction = "{'reasoning': %s}" % (not args.reasoning_off) if args.model == "nemotronh" else None
    args.chat_template_args = {"enable_thinking": False} if (args.reasoning_off and args.model == "qwen3_5") else None
    bs = int(args.batch_size) if str(args.batch_size).isdigit() else args.batch_size
    print(
        f"[gen_any] model={args.model} rot={args.rot} mode={args.mode} state_axis={args.state_axis} "
        f"tasks={tasks} limit={args.limit}",
        flush=True,
    )

    # Check for an open thinking block, not a template-preclosed one.
    if args.apply_chat_template and not args.reasoning_off and args.model != "kimi":
        _probe = _chat_probe_tail(checkpoint_for(args.model), args.system_instruction)
        assert _probe.endswith("<think>\n"), (
            f"{args.model}: thinking/reasoning is NOT enabled -- generation prompt ends "
            f"{_probe[-40:]!r}. Expected an OPEN <think>."
        )

    if args.backend == "vllm":
        from lores.backends.vllm.serving import vllm_source

        vllm_source()
        name = args.vllm_ckpt or checkpoint_for(args.model)
        tok = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=args.model == "kimi")
        model = name  # a str routes evaluate_generation to the vLLM backend
    else:
        name, tok, model = load_model(args, mode)

    metrics: dict = {}
    samples: dict = {}  # {task: [per-doc record]} — correctness, generated text, gen token count
    for t in tasks:
        m, raw_samples = evaluate_generation(
            model,
            tok,
            tasks=[t],
            limit=args.limit,
            batch_size=bs,
            # Do not initialize CUDA before vLLM forks its engine process.
            device="cpu" if args.backend == "vllm" else _device(),
            record_gen_len=args.backend == "hf",  # hooks model.generate; no vLLM analogue
            gen_kwargs=resolved_gen_kwargs[t],
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.apply_chat_template,
            system_instruction=args.system_instruction,
            task_metadata=json.loads(args.task_metadata) if args.task_metadata else None,
            max_length=args.max_length,
            include_path=args.include_path,
            seed=args.seed,
            return_samples=True,
            chat_template_args=args.chat_template_args,
            vllm_kwargs=json.loads(args.vllm_kwargs) if args.vllm_kwargs else None,
        )
        for k, v in m.items():
            metrics[k if k.startswith(f"{t}_") else f"{t}_{k}"] = v
        # trim lm-eval's per-doc records to what post-hoc analysis needs: doc identity,
        # per-doc metric values (correctness), generated text, and its token count
        for st, recs in (raw_samples or {}).items():
            trimmed = []
            for s in recs:
                r = (s.get("resps") or [[]])[0]  # raw generation; filtered_resps is the
                text = r[0] if r else None  # extracted answer (e.g. gsm8k strict-match)
                keep = {k: v for k, v in s.items() if k not in ("doc", "arguments", "resps") and not k.endswith("_hash")}
                keep["text"] = text
                keep["gen_toks"] = len(tok.encode(text)) if isinstance(text, str) else None
                trimmed.append(keep)
            samples[st] = trimmed
        print(f"[gen_any] {t}: " + " ".join(f"{k}={v}" for k, v in m.items()), flush=True)

    # peak device memory: the only signal that says how much batch headroom is left
    peak_gib = max((torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())), default=0) / 2**30
    print(f"[gen_any] peak_mem_gib={peak_gib:.2f} batch_size={args.batch_size}", flush=True)

    rec = {
        "argv": sys.argv[1:],
        "peak_mem_gib": round(peak_gib, 2),
        "model": name,
        "roster_key": args.model,
        "rot": args.rot,
        "mode": args.mode,
        "state_axis": args.state_axis,
        "block_size": args.block_size,
        "limit": args.limit,
        "tasks": tasks,
        "gen_kwargs": [resolved_gen_kwargs[t] for t in tasks],
        "include_path": args.include_path,
        "system_instruction": args.system_instruction,
        "metrics": metrics,
        "w_bits": args.w_bits,
        "a_bits": args.a_bits,
        "batch_size": args.batch_size,
        "apply_chat_template": args.apply_chat_template,
    }
    from lores.fake_quant.quantizers import saturation_events

    rec["sq_saturation_events"] = saturation_events()  # saturating-guard audit (0 = clean run)
    if int(os.environ.get("RANK", 0)) != 0:
        return  # accelerate DP: only rank 0 holds the gathered results — don't race the json write
    outdir = (
        writable_output(args.outdir, root=args.output_root) / args.model / args.rot
        if args.outdir
        else create_run(
            "generation",
            args.model,
            args.stem or args.mode,
            vars(args),
            root=args.output_root,
            namespace=(tasks[0] if len(tasks) == 1 else "multi-task",),
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)
    watag = f"_w{args.w_bits}a{args.a_bits}" if (args.w_bits, args.a_bits) != (16, 16) else ""
    mixtag = f"_mixhead-{Path(args.mixhead).stem}" if args.mixhead else ""
    stem = "".join(
        [
            f"{args.stem or '-'.join(tasks)}_{args.mode}",
            (f"_{args.state_axis}" if args.mode != "off" else ""),
            mixtag,
            (f"_headbits-{Path(args.headbits).stem}" if args.headbits else ""),
            watag,
            (
                f"_n{args.limit}" if args.limit else ""
            ),  # always tag a limited run: full-test (no limit) must never share a stem with a probe
        ]
    )
    out = outdir / f"{stem}.json"
    out.write_text(json.dumps(rec, indent=2))
    print("WROTE", out, flush=True)
    import gzip

    sout = outdir / f"{stem}_samples.json.gz"
    sout.write_bytes(gzip.compress(json.dumps(samples, default=str).encode()))
    print("WROTE", sout, flush=True)


if __name__ == "__main__":
    main()
