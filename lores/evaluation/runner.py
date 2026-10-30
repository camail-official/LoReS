"""Task selection and log-likelihood evaluation through lm-eval."""

from __future__ import annotations

import logging

log = logging.getLogger("lores")


_TASK_METRICS = ("acc", "acc_norm", "word_perplexity", "byte_perplexity", "bits_per_byte")


def evaluate_loglikelihood(
    model, tokenizer, *, tasks, limit: int | None = None, batch_size: int | str = 16, device: str = "cuda"
) -> dict:
    """Evaluate explicit lm-eval tasks against the loaded model."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    for noisy in ("httpx", "urllib3", "filelock", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    tasks = list(tasks)
    model.eval()
    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False
    log.info("log-likelihood eval: tasks=%s limit=%s batch_size=%s", tasks, limit, batch_size)
    lm = HFLM(
        pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=str(device).split(":")[0]
    )  # HFLM wants a bare device name ("cuda"), not "cuda:0"
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit, batch_size=batch_size, bootstrap_iters=0)
    if use_cache is not None:
        model.config.use_cache = use_cache
    return {task: {m: _lm_eval_metric(results, task, m) for m in _TASK_METRICS} for task in tasks}


def _lm_eval_metric(results, task: str, stem: str):
    """Find a metric by name, ignoring its lm-eval filter suffix."""
    r = (results or {}).get("results", {}).get(task, {})
    for k, v in r.items():
        if k == stem or k.startswith(stem + ","):
            return float(v)
    return None


def _expand_tasks(spec) -> list[str]:
    """Normalize explicit task names, preserving order and dropping duplicates."""
    if isinstance(spec, str):
        spec = [t.strip() for t in spec.split(",") if t.strip()]
    return list(dict.fromkeys(spec))


def run_eval(
    model,
    tokenizer,
    *,
    tasks="wikitext",
    batch_size: int | str = 16,
    limit: int | None = None,
    device: str = "cuda",
) -> dict:
    """Run lm-eval tasks and flatten metrics; ``wikitext`` reports word perplexity."""
    task_list = _expand_tasks(tasks)
    metrics: dict = {}
    if task_list:
        results = evaluate_loglikelihood(model, tokenizer, tasks=task_list, limit=limit, batch_size=batch_size, device=device)
        for task, scores in results.items():
            metrics[f"{task}_acc"] = scores.get("acc")  # always present, even as None
            for m, v in scores.items():
                if m != "acc" and v is not None:
                    metrics[f"{task}_{m}"] = v
    return metrics
