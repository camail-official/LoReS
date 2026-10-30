"""Fresh evaluation attempts in the external raw-output store."""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .output_store import component, create_run, sha256, write_json_once


def _snapshot_sources(run, source):
    """Keep the actual hashed producer files, including dirty sources, not just their hashes."""
    files = source.get("files", {})
    if not files:
        return  # generic callers/tests may have no Python-source fingerprint
    output = run / "source.zip"
    repo = Path(__file__).resolve().parents[2]
    fd, temporary = tempfile.mkstemp(prefix=".sources-", dir=run)
    try:
        with os.fdopen(fd, "wb") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, checksum in files.items():
                    if name.startswith("loaded:"):
                        module = sys.modules.get(name.removeprefix("loaded:"))
                        path = Path(module.__file__) if module is not None else None
                    else:
                        path = repo / name
                    if path is None or not path.is_file():
                        raise ValueError(f"Producer source unavailable: {name}")
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != checksum:
                        raise ValueError(f"Producer source changed since fingerprinting: {name}")
                    archive.writestr(name, data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        os.unlink(temporary)


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()


@contextlib.contextmanager
def evaluation_run(family, model, label, config, *, root=None):
    """Always allocate a fresh attempt; failed or completed attempts are never reopened."""
    if family not in ("delta_ell", "references"):
        raise ValueError("Expected a teacher-forced evaluation family")
    run = create_run(family, model, label, config, root=root)
    _snapshot_sources(run, config.get("source", {}))
    (run / "logs").mkdir()
    with (run / "logs" / "runner.log").open("x") as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            print(f"RUN {run}", flush=True)
            print(f"ARGV {sys.argv[1:]!r}", flush=True)
            yield run


def complete_run(run, artifacts):
    """Record successful completion with checksums of the run's artifacts."""
    run = Path(run)
    checksums = {}
    artifacts = list(artifacts)
    if (run / "source.zip").exists():
        artifacts.append("source.zip")
    for name in artifacts:
        path = run / component(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/noncanonical completed artifact: {path}")
        checksums[name] = sha256(path)
    write_json_once(
        run / "complete.json", {"schema_version": 1, "completed_at_utc": datetime.now(UTC).isoformat(), "artifacts": checksums}
    )
