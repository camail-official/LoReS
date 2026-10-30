"""Owned vLLM server processes shared by generation and timing runners."""

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[3]
VLLM = [sys.executable, "-m", "vllm.entrypoints.cli.main"]


PATCH = REPO / "patches/vllm.patch"


def source_identity():
    """Identify this checkout, its vLLM commit and the vLLM patch."""
    from lores.evaluation.output_store import sha256

    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "vllm_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO / "libs/vllm", text=True).strip(),
        "vllm_patch_sha256": sha256(PATCH),
    }


def vllm_source():
    """Require this checkout's vLLM with patches/vllm.patch applied."""
    spec = importlib.util.find_spec("vllm")
    expected = (REPO / "libs/vllm/vllm").resolve()
    if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(expected):
        raise RuntimeError(f"Install this checkout's vLLM: {expected.parent}")
    check = subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=REPO / "libs/vllm", capture_output=True)
    if check.returncode:
        raise RuntimeError(f"Apply {PATCH} to libs/vllm before installing the vLLM environment")
    return spec.origin


@contextmanager
def server(command, log):
    """Start an owned server on a free port and stop it on exit."""
    # Loopback requests must bypass any exported http_proxy.
    os.environ["no_proxy"] = ",".join(filter(None, [os.environ.get("no_proxy"), "127.0.0.1", "localhost"]))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with log.open("x") as output:
        process = subprocess.Popen(
            [*command, "--host", "127.0.0.1", "--port", str(port)],
            cwd=REPO,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with {process.returncode}; see {log}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(5)
            else:
                raise TimeoutError(f"Server did not become healthy; see {log}")
            yield port
        finally:
            # Signal only the process group created for this server, including workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def run_command(command, log):
    """Run a command with an exclusive output log."""
    with log.open("x") as output:
        subprocess.run(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT, check=True)
