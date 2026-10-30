"""Autoregressive evaluation and decode-length accounting for HF and vLLM."""

from __future__ import annotations

import logging
import re

log = logging.getLogger("lores")


class _GenLenRecorder:
    """Record non-padding tokens generated beyond each prompt."""

    def __init__(self, model, pad_id: int | None):
        self.model = model
        self.pad_id = pad_id
        self.lens: list[int] = []
        self._orig = model.generate

    def __enter__(self):
        rec = self

        def _wrapped(*args, **kwargs):
            inp = kwargs.get("input_ids", args[0] if args else None)
            plen = int(inp.shape[1]) if inp is not None else 0
            out = rec._orig(*args, **kwargs)
            seq = getattr(out, "sequences", out)  # GenerateOutput or plain tensor
            new = seq[:, plen:]
            for row in new:
                n = int((row != rec.pad_id).sum().item()) if rec.pad_id is not None else int(row.numel())
                rec.lens.append(n)
            return out

        self.model.generate = _wrapped
        return self

    def __exit__(self, *exc):
        self.model.generate = self._orig
        return False


def _genlen_stats(lens: list[int]) -> dict:
    if not lens:
        return {}
    s = sorted(lens)
    n = len(s)
    return {
        "genlen_n": float(n),
        "genlen_mean": sum(s) / n,
        "genlen_median": float(s[n // 2]),
        "genlen_p95": float(s[min(n - 1, int(0.95 * n))]),
        "genlen_max": float(s[-1]),
    }


def evaluate_generation(
    model,
    tokenizer,
    *,
    tasks,
    limit: int | None = None,
    batch_size: int | str = 8,
    device: str = "cuda",
    record_gen_len: bool = False,
    gen_kwargs: str | None = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    system_instruction: str | None = None,
    task_metadata: dict | None = None,
    max_length: int | None = None,
    include_path: str | None = None,
    return_samples: bool = False,
    seed: int | None = None,
    chat_template_args: dict | None = None,
    vllm_kwargs: dict | None = None,
) -> dict:
    """Evaluate explicit tasks; optionally return samples and generation-length statistics."""
    import lm_eval
    from lm_eval.tasks import TaskManager

    for noisy in ("httpx", "urllib3", "filelock", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    tasks = list(tasks)
    is_vllm = isinstance(model, str)
    if not is_vllm:
        model.eval()
    use_cache = None if is_vllm else getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = True
    log.info("generation eval: tasks=%s limit=%s batch_size=%s", tasks, limit, batch_size)
    if is_vllm:
        from lm_eval.models.vllm_causallms import VLLM

        lm = VLLM(
            pretrained=model,
            tokenizer=tokenizer.name_or_path,
            dtype="bfloat16",
            batch_size=batch_size,
            seed=0,
            **({"max_model_len": max_length} if max_length else {}) | (vllm_kwargs or {}),
        )
    else:
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=str(device),
            **({"max_length": max_length} if max_length else {}),
            **({"chat_template_args": chat_template_args} if chat_template_args else {}),
        )

    # Nemotron's template requires this exact spelling to enable reasoning.
    if system_instruction and re.search(r"reasoning\W+true", system_instruction, re.I):
        system_instruction = "{'reasoning': True}"
        probe = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_instruction}, {"role": "user", "content": "x"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "<think>" in probe, "reasoning system_instruction did not seed <think> in the rendered prompt"

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    assert not (record_gen_len and is_vllm), "record_gen_len is HF-backend only"
    recorder = _GenLenRecorder(model, pad_id) if record_gen_len else None
    if recorder is not None:
        recorder.__enter__()
    try:
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            limit=limit,
            batch_size=batch_size,
            bootstrap_iters=0,
            gen_kwargs=gen_kwargs,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            system_instruction=system_instruction,
            **({"task_manager": TaskManager(include_path=include_path)} if include_path else {}),
            **({"metadata": task_metadata} if task_metadata else {}),  # e.g. RULER max_seq_lengths
            **(
                {"random_seed": seed, "numpy_random_seed": seed, "torch_random_seed": seed, "fewshot_random_seed": seed}
                if seed is not None
                else {}
            ),
        )
    finally:
        if recorder is not None:
            recorder.__exit__()
    if use_cache is not None:
        model.config.use_cache = use_cache

    out: dict = {}
    all_results = (results or {}).get("results", {})
    harvest = (
        list(dict.fromkeys(list(tasks) + [t for t in all_results if any(t.startswith(g) or g == "ruler" for g in tasks)]))
        if any(t == "ruler" for t in tasks)
        else tasks
    )
    for task in harvest:
        r = all_results.get(task, {})
        for key, val in r.items():
            if key == "alias" or "stderr" in key or not isinstance(val, (int, float)):
                continue
            stem = key.split(",")[0]
            out[f"{task}_{stem}"] = float(val)
    if recorder is not None:
        out.update(_genlen_stats(recorder.lens))
    if return_samples:  # simple_evaluate logs samples by default (log_samples=True) — just expose them
        return out, (results or {}).get("samples", {})
    return out
