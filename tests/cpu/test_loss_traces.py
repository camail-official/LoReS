"""Full FP32 paired losses and global-median reduction, with strict reference matching."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from lores.evaluation.loss_traces import (
    load_loss_pair,
    make_loss_trace,
    read_delta_ell_summary,
    save_loss_trace,
    summarize_delta_ell,
    validate_loss_trace,
)
from lores.evaluation.teacher_forcing import decode_per_position
from scripts.evaluation import delta_ell_hybrid as sq
from scripts.evaluation.delta_ell_hybrid import check_baseline_stamp


def trace(ce=None, inputs=None):
    if ce is None:
        ce = torch.tensor([[1.000001, 2.000003, 3.000005], [4.000007, 5.000011, 6.000013]])
    if inputs is None:
        inputs = torch.arange(8, dtype=torch.int64).reshape(2, 4)
    return make_loss_trace(ce, inputs, prefill=1, expected_sequences=2, expected_seqlen=4)


def test_fp32_saved_pair_preserves_sub_half_differences(tmp_path):
    q = trace()
    b = trace(q["ce"].floor())
    assert torch.equal(q["ce"].half(), b["ce"].half())
    for name, data in (("q", q), ("b", b)):
        save_loss_trace(tmp_path / f"{name}.pt", data)
    summary = summarize_delta_ell(tmp_path / "q.pt", tmp_path / "b.pt", tmp_path / "summary.json")
    assert summary["dce_median"] > 0
    assert summary["dce_median"] == (q["ce"] - b["ce"]).median().item()
    assert read_delta_ell_summary(tmp_path / "summary.json") == summary["dce_median"]
    loaded, _ = load_loss_pair(tmp_path / "q.pt", tmp_path / "b.pt")
    assert torch.equal(loaded["ce"], q["ce"])


@pytest.mark.parametrize("bad", ["half", "row", "token", "prefix", "duplicate", "reorder", "mask", "nan", "inf"])
def test_reject_lossy_or_incomplete_trace(bad):
    data = trace()
    if bad == "half":
        data["ce"] = data["ce"].half()
    elif bad == "row":
        data["ce"] = data["ce"][:1]
    elif bad == "token":
        data["ce"] = data["ce"][:, :-1]
    elif bad == "prefix":
        data["input_ids"][0, 0] += 1
    elif bad == "duplicate":
        data["sequence_ids"][1] = 0
    elif bad == "reorder":
        data["sequence_ids"] = data["sequence_ids"].flip(0)
    elif bad == "mask":
        data["valid"][0, 0] = False
    else:
        data["ce"][0, 0] = float(bad)
    with pytest.raises(ValueError):
        validate_loss_trace(data)


def test_different_prefix_is_not_a_paired_trace(tmp_path):
    q = trace()
    inputs = q["input_ids"].clone()
    inputs[:, 0] += 100
    b = trace(inputs=inputs)
    save_loss_trace(tmp_path / "q.pt", q)
    save_loss_trace(tmp_path / "b.pt", b)
    with pytest.raises(ValueError, match="alignment"):
        load_loss_pair(tmp_path / "q.pt", tmp_path / "b.pt")


def test_requested_coverage_independent_of_observed_shapes():
    with pytest.raises(ValueError, match="coverage"):
        make_loss_trace(
            torch.zeros(1, 3), torch.zeros(1, 4, dtype=torch.int64), prefill=1, expected_sequences=2, expected_seqlen=4
        )


def test_trace_publication_never_reuses_or_overwrites(tmp_path):
    path = tmp_path / "trace.pt"
    data = trace()
    save_loss_trace(path, data)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_loss_trace(path, data)
    data["ce"][0, 0] += 0.01
    with pytest.raises(FileExistsError):
        save_loss_trace(path, data)
    assert path.read_bytes() == original


@pytest.mark.parametrize("bad", ["scalar", "count", "units", "trace"])
def test_tampered_summary_or_trace_rejected(tmp_path, bad):
    save_loss_trace(tmp_path / "q.pt", trace())
    save_loss_trace(tmp_path / "b.pt", trace(torch.zeros(2, 3)))
    path = tmp_path / "summary.json"
    summary = summarize_delta_ell(tmp_path / "q.pt", tmp_path / "b.pt", path)
    if bad == "trace":
        data = trace()
        data["ce"][0, 0] += 1
        torch.save(data, tmp_path / "q.pt")
    else:
        key = {"scalar": "dce_median", "count": "n_tokens", "units": "units"}[bad]
        summary[key] = "wrong"
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        read_delta_ell_summary(path)


def test_legacy_scalar_does_not_establish_global_median(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"dce_median": 0.1}))
    with pytest.raises(ValueError, match="verified"):
        read_delta_ell_summary(path)


def median_trace(ce):
    from lores.evaluation.loss_traces import make_loss_trace

    ce = torch.tensor(ce, dtype=torch.float32)
    tokens = torch.arange(ce.numel(), dtype=torch.int64).reshape(ce.shape)
    inputs = torch.cat([torch.zeros(len(ce), 1, dtype=torch.int64), tokens], dim=1)
    return make_loss_trace(ce, inputs, prefill=1, expected_sequences=len(ce), expected_seqlen=ce.shape[1] + 1)


def test_global_median_from_saved_full_traces(tmp_path):
    q, b = median_trace([[1, 2, 100], [3, 4, 200]]), median_trace([[0, 0, 0], [0, 0, 0]])
    torch.save(q, tmp_path / "q.pt")
    torch.save(b, tmp_path / "b.pt")
    result = summarize_delta_ell(tmp_path / "q.pt", tmp_path / "b.pt", tmp_path / "summary.json")
    assert result["dce_median"] == 3  # lower middle of ALL six tokens, not sequence means/medians
    assert result["n_tokens"] == 6 and result["median_convention"] == "lower_middle"
    assert len(result["quant_trace_sha256"]) == 64
    with pytest.raises(FileExistsError):
        summarize_delta_ell(tmp_path / "q.pt", tmp_path / "b.pt", tmp_path / "summary.json")


def _stamp(**over):
    b = {
        "model": "qwen3_5",
        "seqlen": 2048,
        "prefill": 1,
        "n_seqs": 8,
        "seed": 0,
        "seqs_sha": "abc",
        "ce": torch.zeros(1),
        "am": torch.zeros(1),
        "checkpoint_identity": {"name_or_path": "test", "revision": "test-revision"},
        "state_protocol": sq.FP32_STATE_PROTOCOL,
    }
    b.update(over)
    return b


def _args(**over):
    base = {"model": "qwen3_5", "seqlen": 2048, "prefill": 1, "seed": 0}
    base.update(over)
    return SimpleNamespace(**base)


def test_baseline_stamp_accepts_exact_match():
    sq.check_baseline_stamp(_stamp(), _args(), n_seqs=8, seqs_sha="abc", checkpoint_identity=_stamp()["checkpoint_identity"])


@pytest.mark.parametrize(
    "bad",
    [
        {"model": "kimi"},
        {"seqlen": 4096},
        {"prefill": 64},
        {"n_seqs": 4},
        {"seed": 1},
        {"seqs_sha": "zzz"},
    ],
)
def test_baseline_stamp_rejects_any_protocol_drift(bad):
    b = _stamp(**{k: v for k, v in bad.items() if k in _stamp()})
    a = _args()
    n_seqs, sha = 8, "abc"
    with pytest.raises(ValueError, match="baseline/protocol mismatch"):
        sq.check_baseline_stamp(b, a, n_seqs, sha, checkpoint_identity=_stamp()["checkpoint_identity"])


@pytest.mark.parametrize("field", ["checkpoint_identity", "state_protocol"])
def test_baseline_rejects_missing_or_wrong_identity(field):
    for bad in (None, {"wrong": True}):
        b = _stamp(**{field: bad})
        with pytest.raises(ValueError, match="baseline/protocol mismatch"):
            check_baseline_stamp(b, _args(), 8, "abc", checkpoint_identity=_stamp()["checkpoint_identity"])


class TinyLM(torch.nn.Module):
    def __init__(self, fail_after=None):
        super().__init__()
        self.fail_after, self.calls = fail_after, 0

    def forward(self, input_ids, past_key_values=None, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated interruption")
        offset = 0 if past_key_values is None else past_key_values
        logits = torch.randn(*input_ids.shape, 16) + torch.arange(16) * 0.03 + offset * 0.01
        return SimpleNamespace(logits=logits, past_key_values=offset + input_ids.shape[1])


def test_interrupted_evaluation_restarts_all_sequences_without_disk_io(monkeypatch):
    def no_disk_io(*args, **kwargs):
        pytest.fail("decode must not read or save progress")

    monkeypatch.setattr(torch, "load", no_disk_io)
    monkeypatch.setattr(torch, "save", no_disk_io)
    tokens = torch.arange(15).reshape(3, 5)
    with pytest.raises(RuntimeError, match="interruption"):
        decode_per_position(TinyLM(fail_after=6), tokens, 1, 8)
    model = TinyLM()
    actual = decode_per_position(model, tokens, 1, 8)
    expected = decode_per_position(TinyLM(), tokens, 1, 8)
    assert model.calls == 15  # all three sequences, not just the previously unfinished rows
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    ce, am = actual
    assert ce.shape == am.shape == (3, 4)
    assert ce.dtype == torch.float32 and am.dtype == torch.int64
    assert torch.isfinite(ce).all()


def test_clipped_sequence_cannot_return_finite_but_invalid_losses(monkeypatch):
    count = iter([0, 1])
    monkeypatch.setattr("lores.fake_quant.quantizers.saturation_events", lambda: next(count))
    with pytest.raises(ValueError, match="saturation"):
        decode_per_position(TinyLM(), torch.arange(15).reshape(3, 5), 1, 8)
