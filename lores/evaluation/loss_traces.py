"""Complete lossless teacher-forced traces, with independently requested coverage."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .output_store import sha256, write_json_once
from .provenance import save_tensor_artifact, tensor_identity


def make_loss_trace(ce, input_ids, *, prefill, expected_sequences, expected_seqlen, metadata=None):
    """Construct a complete trace only when every requested sequence/token is present."""
    trace = dict(metadata or {})
    reserved = {
        "schema_version",
        "ce",
        "input_ids",
        "token_ids",
        "sequence_ids",
        "valid",
        "complete",
        "expected_sequences",
        "expected_seqlen",
        "input_identity",
    }
    if reserved.intersection(trace):
        raise ValueError("Trace metadata must not override coverage or tensor fields")
    trace.update(
        {
            "schema_version": 2,
            "ce": ce.detach().cpu().clone(),
            "input_ids": input_ids.detach().cpu().clone(),
            "prefill": prefill,
            "expected_sequences": expected_sequences,
            "expected_seqlen": expected_seqlen,
            "token_ids": input_ids[:, prefill:].detach().cpu().clone(),
            "sequence_ids": torch.arange(expected_sequences, dtype=torch.int64),
            "valid": torch.ones_like(ce, dtype=torch.bool, device="cpu"),
            "complete": True,
            "input_identity": tensor_identity(input_ids),
        }
    )
    validate_loss_trace(trace)
    return trace


def validate_loss_trace(trace):
    """Reject lossy, truncated, misaligned, nonfinite or selectively masked traces."""
    if trace.get("schema_version") != 2 or trace.get("complete") is not True:
        raise ValueError("Expected a complete version-2 loss trace")
    n, length, prefill = (trace.get(k) for k in ("expected_sequences", "expected_seqlen", "prefill"))
    if any(type(v) is not int for v in (n, length, prefill)) or n <= 0 or not 1 <= prefill < length:
        raise ValueError("Invalid requested trace coverage")
    ce, inputs, tokens, ids, valid = (trace[k] for k in ("ce", "input_ids", "token_ids", "sequence_ids", "valid"))
    if ce.dtype != torch.float32 or tuple(ce.shape) != (n, length - prefill):
        raise ValueError("CE must contain full requested FP32 sequence/token coverage")
    if not torch.isfinite(ce).all():
        raise ValueError("Non-finite evaluated token losses")
    if inputs.dtype != torch.int64 or tuple(inputs.shape) != (n, length):
        raise ValueError("Full input tokens must match requested sequence coverage")
    if tokens.dtype != torch.int64 or not torch.equal(tokens, inputs[:, prefill:]):
        raise ValueError("Loss targets do not match full input tokens")
    if ids.dtype != torch.int64 or not torch.equal(ids, torch.arange(n, dtype=torch.int64)):
        raise ValueError("Missing, duplicated or reordered sequence IDs")
    if valid.dtype != torch.bool or valid.shape != ce.shape or not valid.all():
        raise ValueError("Every requested token must be evaluated; no selective masking")
    if trace.get("input_identity") != tensor_identity(inputs):
        raise ValueError("Input content hash mismatch")


def save_loss_trace(path, trace):
    """Publish a complete trace atomically, without reusing or replacing existing files."""
    validate_loss_trace(trace)
    save_tensor_artifact(path, trace)


def load_loss_pair(quant_path, reference_path):
    """Load verified paired traces; prefix tokens must match, not just predicted targets."""
    q, b = [torch.load(p, map_location="cpu", weights_only=True) for p in (quant_path, reference_path)]
    for trace in (q, b):
        validate_loss_trace(trace)
    for key in ("input_ids", "token_ids", "sequence_ids", "valid"):
        if not torch.equal(q[key], b[key]):
            raise ValueError(f"Trace alignment mismatch: {key}")
    if q["prefill"] != b["prefill"] or q["ce"].shape != b["ce"].shape:
        raise ValueError("Trace coverage mismatch")
    return q, b


def read_delta_ell_summary(path):
    """Read a global-median summary after verifying that it reproduces from its saved FP32 traces."""
    path = Path(path)
    summary = json.loads(path.read_text())
    if summary.get("aggregation") != "global_token_median" or summary.get("median_convention") != "lower_middle":
        raise ValueError(f"No verified global-median trace summary: {path}")
    paths = []
    for key in ("quant", "reference"):
        trace_path = Path(summary[f"{key}_trace"])
        if not trace_path.is_absolute():
            trace_path = path.parent / trace_path
        if sha256(trace_path) != summary[f"{key}_trace_sha256"]:
            raise ValueError(f"Changed {key} trace: {trace_path}")
        paths.append(trace_path)
    q, b = load_loss_pair(*paths)
    delta = q["ce"] - b["ce"]
    if (
        not torch.isfinite(delta).all()
        or summary.get("schema_version") != 2
        or summary.get("units") != "signed_nats"
        or summary.get("shape") != list(delta.shape)
        or summary.get("n_tokens") != delta.numel()
        or summary.get("dce_median") != delta.median().item()
    ):
        raise ValueError(f"Summary does not reproduce from complete saved traces: {path}")
    return summary["dce_median"]


def summarize_delta_ell(quant_path: str | Path, reference_path: str | Path, output: str | Path) -> dict:
    """Publish the lower-middle global median of complete paired FP32 loss traces."""
    import torch

    q, b = load_loss_pair(quant_path, reference_path)
    delta = (q["ce"] - b["ce"])[q["valid"]]
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("Non-finite paired loss difference")
    result = {
        "schema_version": 2,
        "dce_median": delta.median().item(),
        "n_tokens": delta.numel(),
        "shape": list(q["ce"].shape),
        "aggregation": "global_token_median",
        "median_convention": "lower_middle",
        "units": "signed_nats",
        "quant_trace": str(Path(quant_path).absolute()),
        "reference_trace": str(Path(reference_path).absolute()),
        "quant_trace_sha256": sha256(quant_path),
        "reference_trace_sha256": sha256(reference_path),
    }
    write_json_once(Path(output), result)
    return result
