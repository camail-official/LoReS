"""Experiment identity and atomic publication of immutable tensor artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from .output_store import fingerprint, sha256, source_path


def tensor_identity(tensor) -> dict:
    """Hash tensor contents, shape and dtype, not a filename or storage address."""
    import torch

    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def file_identity(path) -> dict:
    """Resolve and hash a required input artifact."""
    path = source_path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def source_identity(repo=None) -> dict:
    """Record actual producer sources, including uncommitted edits and loaded vendor code."""
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    files = {}
    for directory in (
        "lores",
        "mixerzoo",
        "scripts/evaluation",
        "scripts/calibration",
        "scripts/diagnostics",
        "scripts/export",
        "libs/flash-linear-attention/fla",
    ):
        for path in sorted((repo / directory).rglob("*.py")):
            files[path.relative_to(repo).as_posix()] = sha256(path)
    # The loaded HF/custom implementation can differ even at the same package version.
    for name, module in list(sys.modules.items()):
        if name.startswith(("transformers.models.", "transformers_modules.", "transformers.cache_utils")):
            path = getattr(module, "__file__", None)
            if path and Path(path).is_file() and str(path).endswith(".py"):
                files["loaded:" + name] = sha256(path)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    return {"git_commit": result.stdout.strip() if result.returncode == 0 else None, "files": files, "sha256": fingerprint(files)}


def environment_identity() -> dict:
    """Record numerical library and accelerator settings that affect a replay."""
    import torch

    versions = {}
    for name in ("torch", "transformers", "triton", "flash-linear-attention", "mamba-ssm", "causal-conv1d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "versions": versions,
        "numerical_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("FLA_")
            or key
            in (
                "TRITON_F32_DEFAULT",
                "NVIDIA_TF32_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "python": sys.version,
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def local_checkpoint_identity(path) -> dict:
    """Content-address a local checkpoint, including weights and tokenizer/config files."""
    path = Path(path).resolve(strict=True)
    files = {
        p.relative_to(path).as_posix(): sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.suffix in (".safetensors", ".bin", ".pt", ".json", ".model", ".txt")
    }
    if not any(p.endswith((".safetensors", ".bin", ".pt")) for p in files):
        raise ValueError(f"No weights found in checkpoint {path}")
    return {"path": str(path), "files": files, "sha256": fingerprint(files)}


def experiment_identity(args, seqs, checkpoint, *, source=None, environment=None, reference=None) -> dict:
    """Freeze all resolved runner settings and content-address every mask/input."""
    settings = dict(vars(args)) if hasattr(args, "__dict__") else dict(args)
    for key in ("output_root", "outdir", "tag", "write_baseline"):
        settings.pop(key, None)  # output controls, not numerical settings
    mask_keys = ("headbits", "chanbits", "mixhead")
    inputs = {key: file_identity(settings[key]) for key in mask_keys if settings.get(key)}
    return {
        "version": 1,
        "settings": settings,
        "checkpoint": checkpoint,
        "tokens": tensor_identity(seqs),
        "inputs": inputs,
        "source": source if source is not None else source_identity(),
        "environment": environment if environment is not None else environment_identity(),
        "reference": file_identity(reference) if reference is not None else None,
    }


def reference_identity(seqs, prefill, checkpoint, source, environment) -> dict:
    """Common reference identity, independent of the quantized arm's mode or masks."""
    from lores.backends.hf.cache import FP32_STATE_PROTOCOL

    return {
        "version": 1,
        "tokens": tensor_identity(seqs),
        "prefill": prefill,
        "checkpoint": checkpoint,
        "source": source,
        "environment": environment,
        "state_protocol": FP32_STATE_PROTOCOL,
    }


def save_tensor_artifact(path, value):
    """Publish a complete tensor artifact atomically, without replacing an existing file."""
    import torch

    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing to write through an artifact symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".tensor-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)  # fails if a complete output already exists
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def stamp(rec: dict) -> dict:
    """Attach best-effort host, commit, and command metadata."""
    if isinstance(rec, dict) and "provenance" in rec:
        return rec
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    rec["provenance"] = {
        "host": socket.gethostname(),
        "commit": commit,
        "argv": " ".join(sys.argv[1:]),
    }
    return rec
