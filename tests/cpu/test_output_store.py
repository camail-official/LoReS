"""External results are explicitly selected, content-identified, and never overwritten."""

from __future__ import annotations

import pytest
import torch

from lores.evaluation.output_store import (
    create_run,
    fingerprint,
    source_path,
    write_json_once,
)
from lores.evaluation.provenance import (
    experiment_identity,
    save_tensor_artifact,
)
from lores.evaluation.runs import evaluation_run


def test_run_identity_and_no_overwrite(tmp_path):
    config = {"mode": "int6", "seed": 0, "mask_sha256": "one", "reference": "fp32-a"}
    first = create_run("delta_ell", "gla_beta", "int6-value-rtn", config, root=tmp_path)
    second = create_run("delta_ell", "gla_beta", "int6-value-rtn", config, root=tmp_path)
    assert first != second and first.name.endswith("attempt-01") and second.name.endswith("attempt-02")
    with pytest.raises(FileExistsError):
        write_json_once(first / "run.json", {})
    for key in config:
        assert fingerprint(config) != fingerprint({**config, key: "changed"})


def test_source_path_requires_an_existing_file(tmp_path):
    path = tmp_path / "mask.json"
    write_json_once(path, {})
    assert source_path(path) == path
    with pytest.raises(FileNotFoundError):
        source_path(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        source_path(tmp_path)


def test_same_named_mask_content_changes_experiment_identity(tmp_path):
    path = tmp_path / "mask.json"
    path.write_text('{"0":[4,6]}')
    args = {"headbits": str(path), "mode": "int4_block"}
    tokens = torch.arange(15).reshape(3, 5)
    a = experiment_identity(args, tokens, {"revision": "a"}, source={}, environment={})
    path.write_text('{"0":[6,4]}')
    b = experiment_identity(args, tokens, {"revision": "a"}, source={}, environment={})
    assert fingerprint(a) != fingerprint(b)


@pytest.mark.parametrize("change", ["tokens", "seed", "prefill", "mode", "checkpoint"])
def test_numerical_inputs_change_experiment_identity(change):
    tokens = torch.arange(15).reshape(3, 5)
    args = {"seed": 8, "prefill": 1, "mode": "int4_block_sr"}
    checkpoint = {"revision": "a"}
    a = experiment_identity(args, tokens, checkpoint, source={}, environment={})
    if change == "tokens":
        tokens[0, 0] += 1
    elif change == "checkpoint":
        checkpoint = {"revision": "b"}
    else:
        args[change] = "different"
    b = experiment_identity(args, tokens, checkpoint, source={}, environment={})
    assert fingerprint(a) != fingerprint(b)


def test_tensor_publication_is_exclusive_and_rejects_symlinks(tmp_path):
    path = tmp_path / "trace.pt"
    save_tensor_artifact(path, {"ce": torch.ones(2)})
    with pytest.raises(FileExistsError):
        save_tensor_artifact(path, {"ce": torch.zeros(2)})
    alias = tmp_path / "alias.pt"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        save_tensor_artifact(alias, {})


def test_new_alias_cannot_redirect_a_writer_into_old(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (tmp_path / "new").symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="aliases"):
        with evaluation_run("delta_ell", "tiny", "int4", {}, root=tmp_path):
            pytest.fail("writer entered the archive")
    assert not list(old.iterdir())
