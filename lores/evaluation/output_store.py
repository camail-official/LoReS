"""Raw experimental outputs, independent of the repository checkout.

This module has no torch/model imports until a trace is actually summarized.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

FAMILIES = ("delta_ell", "references", "calibration", "diagnostics", "generation", "benchmarks")
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO / "output"


def output_root(root: str | Path | None = None) -> Path:
    """Return the raw-data root without creating it."""
    return Path(DEFAULT_OUTPUT_ROOT if root is None else root).absolute()


def component(value: str) -> str:
    """Validate one user-controlled directory component."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (".", ".."):
        raise ValueError(f"Unsafe path component: {value!r}")
    return value


def fingerprint(value: object) -> str:
    """Hash a complete JSON-compatible configuration deterministically."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def sha256(path: str | Path) -> str:
    """Stream a file checksum without loading large trace tensors."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json_once(path: Path, value: object) -> None:
    """Publish complete metadata atomically, rejecting any existing destination."""
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        os.unlink(temp)


def create_run(
    family: str,
    model: str,
    label: str,
    config: dict,
    *,
    root: str | Path | None = None,
    namespace: tuple[str, ...] = (),
    device: str | None = None,
) -> Path:
    """Allocate a unique attempt. Callers must include every numerical input in config.

    ``namespace`` contains one experiment (diagnostics) or task (generation).
    Benchmarks require ``device``; nothing guesses a dataset, model or device.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown output family: {family}")
    if len(namespace) != (1 if family in ("diagnostics", "generation") else 0):
        raise ValueError(f"Invalid namespace for {family}: {namespace}")
    if (family == "benchmarks") != (device is not None):
        raise ValueError("A device is required only for benchmarks")
    component(model)
    component(label)
    base = output_root(root)
    parent = base / "new" / family
    parts = (device,) if family == "benchmarks" else (*namespace, model) if family == "diagnostics" else (model, *namespace)
    for part in parts:
        parent /= component(part)
    current = base
    for part in parent.relative_to(base).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"New output directories must not redirect through aliases: {current}")
    if not parent.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Run destination escapes output root: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    stamp = fingerprint(config)
    for attempt in range(1, 1_000_000):
        dest = parent / f"{component(label)}__{stamp[:16]}__attempt-{attempt:02d}"
        try:
            dest.mkdir()
        except FileExistsError:
            continue
        write_json_once(
            dest / "run.json",
            {
                "schema_version": 1,
                "family": family,
                "model": model,
                "device": device,
                "config": config,
                "config_sha256": stamp,
                "attempt": attempt,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return dest
    raise RuntimeError("No free attempt ID")


def source_path(source: str | Path) -> Path:
    """Resolve an explicit input file; missing files are errors."""
    p = Path(os.path.abspath(source))
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def source_files(pattern: str | Path) -> dict[Path, Path]:
    """Regular files matching a glob, keyed by their absolute path."""
    matches = {}
    for name in glob.glob(str(pattern), recursive=True):
        p = Path(os.path.abspath(name))
        if p.is_file() and not p.is_symlink():
            matches[p] = p
    return dict(sorted(matches.items()))


def writable_output(path: str | Path, *, root: str | Path | None = None) -> Path:
    """Reject aliases and destinations outside the configured store."""
    p = Path(os.path.abspath(path))
    new = output_root(root) / "new"
    if not p.is_relative_to(new) or not p.resolve().is_relative_to(new.resolve()):
        raise ValueError(f"Raw outputs must be under {new}, not {p}")
    if any(parent.is_symlink() for parent in (p, *p.parents) if parent.is_relative_to(new)):
        raise ValueError(f"Raw outputs must not traverse symlinks: {p}")
    return p
