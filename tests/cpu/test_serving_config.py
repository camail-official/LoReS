"""Serving configuration selects valid quantization and rotation settings."""

from __future__ import annotations

import types

import pytest

from lores.backends.vllm.config import LinqConfig


class _VC:  # stand-in for VllmConfig (+ model_config.hf_config with the export's stamped linq_wa_rot)
    def __init__(self, additional_config, linq_wa_rot=None):
        self.additional_config = additional_config
        hf = types.SimpleNamespace(**({"linq_wa_rot": linq_wa_rot} if linq_wa_rot else {}))
        self.model_config = types.SimpleNamespace(hf_config=hf)


def test_defaults_off():
    c = LinqConfig.from_dict({})
    assert c.state_bits == 0 and not c.state_sr and not c.state_asym and not c.state_fast and c.wa_rot == frozenset()
    assert c.banner() == "LINQ-STATE: off"


def test_recipes_parse():
    nemo = LinqConfig.from_dict({"state_bits": 8, "state_sr": True, "state_asym": True, "wa_rot": "r4s,ro"})
    assert (
        nemo.state_asym and nemo.wa_rot == {"r4s", "ro"} and "asymmetric grid" in nemo.banner() and "exact codec" in nemo.banner()
    )
    qwen = LinqConfig.from_dict({"state_bits": "8", "state_sr": True, "wa_rot": ["r4", "ro"]})
    assert qwen.state_bits == 8 and not qwen.state_asym and qwen.wa_rot == {"r4", "ro"}


@pytest.mark.parametrize(
    "bad",
    [
        {"state_bits": 5},
        {"state_bits": 4, "state_asym": True},
        {"state_sr": True},
        {"wa_rot": "r4,r4s"},
        {"wa_rot": "r5"},
        {"bits": 8},
    ],
)
def test_rejects(bad):
    with pytest.raises(ValueError):
        LinqConfig.from_dict(bad)


def test_checkpoint_declares_wa_rot():
    c = LinqConfig.from_vllm_config(_VC({"linq": {"state_bits": 8, "state_sr": True}}, linq_wa_rot="r4s,ro"))
    assert c.wa_rot == {"r4s", "ro"} and c.state_bits == 8 and c.state_sr
    assert LinqConfig.from_vllm_config(_VC({}, linq_wa_rot="r4,ro")).wa_rot == {"r4", "ro"}


def test_additional_config_wa_rot_overrides_checkpoint():
    assert LinqConfig.from_vllm_config(_VC({"linq": {"wa_rot": "r4"}}, linq_wa_rot="r4s,ro")).wa_rot == {"r4"}
    # explicit empty = the negative-control arm (rotated export served without its rotations)
    assert LinqConfig.from_vllm_config(_VC({"linq": {"wa_rot": ""}}, linq_wa_rot="r4s,ro")).wa_rot == frozenset()


def test_checkpoint_wa_rot_is_validated():
    with pytest.raises(ValueError):
        LinqConfig.from_vllm_config(_VC({}, linq_wa_rot="r4,r4s"))


def test_state_rot_parses():
    assert LinqConfig.from_dict({"rot": "rk"}).rot == {"rk"}
    assert LinqConfig.from_dict({}).rot == frozenset()
    for bad in ("r4", "rv", "both"):
        with pytest.raises(ValueError):
            LinqConfig.from_dict({"rot": bad})


def test_backend_helpers_share_the_resolved_config_without_vllm(monkeypatch):
    """Backend helpers must share one config singleton and must not require the serving install."""
    import sys

    monkeypatch.setitem(sys.modules, "vllm", None)
    from lores.backends.vllm import config, rotations, state

    monkeypatch.setattr(config, "_RESOLVED", None)
    assert config.current() == LinqConfig()
    assert config._RESOLVED is None  # an early default must not prevent later resolution
    vc = _VC({"linq": {"state_bits": 8, "rot": "rk", "wa_rot": "r4"}})
    resolved = config.set_current(vc)
    assert config.set_current(vc) is resolved
    assert state.linq_config() is resolved
    assert rotations.linq_rot_any() and rotations.wa_rot("r4")
    with pytest.raises(RuntimeError, match="conflicting configs"):
        config.set_current(_VC({}))
