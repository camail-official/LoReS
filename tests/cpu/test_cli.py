"""Offline package import, public CLI dispatch, and evaluator wiring."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch


def test_import_lores():
    import lores  # noqa: F401
    import lores.cli  # noqa: F401
    import lores.config  # noqa: F401
    import lores.evaluation.runner  # noqa: F401
    import lores.models  # noqa: F401
    import lores.rotations.folding  # noqa: F401
    import lores.rotations.matrices  # noqa: F401
    import lores.wa_ptq.gptq  # noqa: F401
    import lores.wa_ptq.prepare  # noqa: F401


def test_cli_help_exits_zero():
    from lores.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0


def test_cli_ptq_dispatch(monkeypatch):
    captured = {}

    def fake_run_ptq(cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr("lores.wa_ptq.pipeline.run_ptq", fake_run_ptq)
    from lores.cli import main

    main(
        [
            "ptq",
            "--input_model",
            "fake/model",
            "--dtype",
            "float16",
            "--w_bits",
            "8",
            "--a_bits",
            "8",
            "--rotate_o",
            "--no-rotate_r4",  # explicit off (tri-state: an unset flag stays None until prepare_model resolves it)
            "--device",
            "cpu",
            "--eval_tasks",
            "wikitext",
            "--eval_limit",
            "2",
        ]
    )
    cfg = captured["cfg"]
    assert cfg.model.name_or_path == "fake/model"
    assert cfg.model.dtype == "float16"
    assert cfg.quant.w_bits == 8
    assert cfg.quant.a_bits == 8
    assert cfg.rotation.o is True
    assert cfg.rotation.r4 is False
    assert cfg.rotation.residual is None
    assert cfg.device == "cpu"
    assert cfg.eval.tasks == "wikitext"
    assert cfg.eval.limit == 2


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_generation_module_preserves_backend_limit_and_samples(monkeypatch, backend):
    from lores.evaluation.generation import evaluate_generation

    captured = {}
    model = torch.nn.Module()
    model.config = SimpleNamespace(use_cache=False)
    tokenizer = SimpleNamespace(name_or_path="tiny", eos_token_id=0)

    def wrapper(**kwargs):
        captured["wrapper"] = kwargs
        return "wrapped"

    def evaluate(**kwargs):
        captured["evaluate"] = kwargs
        return {"results": {"task": {"exact_match,none": 0.5}}, "samples": {"task": ["sample"]}}

    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=evaluate))
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", SimpleNamespace(TaskManager=object))
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", SimpleNamespace(HFLM=wrapper))
    monkeypatch.setitem(sys.modules, "lm_eval.models.vllm_causallms", SimpleNamespace(VLLM=wrapper))
    metrics, samples = evaluate_generation(
        "tiny-checkpoint" if backend == "vllm" else model,
        tokenizer,
        tasks=["task"],
        limit=2,
        return_samples=True,
        device="cpu",
    )
    assert metrics == {"task_exact_match": 0.5} and samples == {"task": ["sample"]}
    assert captured["evaluate"]["limit"] == 2 and "samples" not in captured["evaluate"]
    assert captured["wrapper"]["pretrained"] == ("tiny-checkpoint" if backend == "vllm" else model)
    assert model.config.use_cache is False
