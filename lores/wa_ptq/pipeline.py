"""Public offline PTQ workflow: load, prepare, and evaluate."""

from __future__ import annotations

import logging

from lores.backends.hf.loading import load_ptq_model
from lores.config import PTQConfig

log = logging.getLogger("lores")


def run_ptq(cfg: PTQConfig) -> dict:
    """Load, prepare, and run lm-eval."""
    from lores.evaluation.runner import run_eval
    from lores.wa_ptq.prepare import prepare_model

    model, tokenizer = load_ptq_model(cfg)
    calib = None
    if cfg.quant.w_bits < 16:
        from lores.evaluation.data import get_wikitext2

        calib = get_wikitext2(
            tokenizer=tokenizer, nsamples=cfg.quant.calib_nsamples, seed=cfg.seed, seqlen=cfg.model.seqlen, eval_mode=False
        )
    prepare_model(cfg, model, calib_loader=calib)
    metrics = run_eval(
        model,
        tokenizer,
        tasks=cfg.eval.tasks,
        batch_size=cfg.eval.batch_size,
        limit=cfg.eval.limit,
        device=cfg.device,
    )
    log.info("LoReS PTQ done: %s", metrics)
    return metrics
