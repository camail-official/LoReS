"""HF cache hooks preserve FP32 state, prefill handoff, and per-sequence reset."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lores.backends.hf import state as sq
from lores.backends.hf.cache import (
    get_recurrent_state,
    install_state_dtype_hooks,
    install_state_quant_hook,
    set_recurrent_state,
)
from lores.fake_quant.state import quant_every


class _Inner:
    """Inner quantizer stub. Returns a NEW tensor (``S + 1``) so a pass-through step is
    distinguishable from a quantized one by value AND by object identity.
    """

    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.seen = []

    def __call__(self, S):
        self.calls += 1
        self.seen.append(S)
        return S + 1.0

    def reset(self):
        self.resets += 1


class _Mixer(nn.Module):
    """Duck-typed mixer: what ``state_quant_mixers`` and the arch hooks actually touch."""

    def __init__(self, quant=None, hooked=True, layer_idx=0):
        super().__init__()
        self._state_quant = quant
        self._sq_handle = object() if hooked else None
        self.layer_idx = layer_idx


def _qwen_cache(rs):
    return SimpleNamespace(layers=[SimpleNamespace(recurrent_states=rs)])


def _kimi_cache(rs):
    return SimpleNamespace(recurrent_states=[rs])


def _hooks():
    from lores.backends.hf.kimi_linear.integration import _kimi_state_quant_hook
    from lores.backends.hf.nemotronh.integration import _nemotronh_state_quant_hook
    from lores.backends.hf.qwen3_5.integration import _qwen3_5_state_quant_hook

    return [
        pytest.param(_qwen3_5_state_quant_hook, _qwen_cache, id="qwen3_5"),
        pytest.param(_nemotronh_state_quant_hook, _qwen_cache, id="nemotronh"),
        pytest.param(_kimi_state_quant_hook, _kimi_cache, id="kimi"),
    ]


ARCH_HOOKS = _hooks()


@pytest.mark.parametrize(("hook", "mk_cache"), ARCH_HOOKS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_arch_hook_stores_quantized_state_fp32(hook, mk_cache, dtype):
    # the quantized state is ALWAYS stored fp32 (a bf16 store double-rounds the INT
    # grid); a bf16 cache slot is promoted via set_state, an fp32 slot is written in place.
    inner = _Inner()
    mix = _Mixer(inner)
    rs = torch.arange(6, dtype=dtype).reshape(1, 1, 2, 3)
    before = rs.clone()
    cache = mk_cache(rs)
    hook(mix, (), {"hidden_states": torch.zeros(1, 1, 4), "cache_params": cache}, None)
    slot = cache.layers[0].recurrent_states if hasattr(cache, "layers") else cache.recurrent_states[0]
    assert inner.calls == 1
    assert slot.dtype == torch.float32
    assert torch.equal(slot, before.float() + 1.0)  # the quantizer's fp32 output, no re-rounding
    if dtype == torch.float32:
        assert slot is rs  # in place: tensor identity preserved for the next step's cache read


@pytest.mark.parametrize(("hook", "mk_cache"), ARCH_HOOKS)
def test_arch_hook_resets_then_quantizes_handover_on_prefill(hook, mk_cache):
    # prefill = reset per-sequence state, then quantize the prefill-final state ONCE (the
    # prefill→decode handover: an INT cache stores that state quantized)
    inner = _Inner()
    mix = _Mixer(inner)
    rs = torch.zeros(1, 1, 2, 3)
    before = rs.clone()
    cache = mk_cache(rs)
    hook(mix, (), {"hidden_states": torch.zeros(1, 8, 4), "cache_params": cache}, None)
    assert (inner.calls, inner.resets) == (1, 1)
    slot = cache.layers[0].recurrent_states if hasattr(cache, "layers") else cache.recurrent_states[0]
    assert slot.dtype == torch.float32 and torch.equal(slot, before + 1.0)


def test_install_with_none_removes_hook_for_vendor_baseline():
    # mode "off" installs q=None: any prior hook is removed and NOTHING is registered, so the
    # vendor baseline's bf16 cache slot is never touched (never promoted to fp32)
    from lores.backends.hf.cache import install_state_quant_hook

    mix = _Mixer(_Inner(), hooked=False)
    install_state_quant_hook(mix, _Inner(), lambda *a, **k: None)
    assert mix._sq_handle is not None
    install_state_quant_hook(mix, None, lambda *a, **k: None)
    assert mix._sq_handle is None and mix._state_quant is None
    assert mix not in sq.state_quant_mixers(SimpleNamespace(modules=lambda: [mix]))


@pytest.mark.parametrize(("hook", "mk_cache"), ARCH_HOOKS)
def test_arch_hook_only_resets_without_a_cache(hook, mk_cache):
    inner = _Inner()
    mix = _Mixer(inner)
    hook(mix, (), {"hidden_states": torch.zeros(1, 8, 4), "cache_params": None}, None)
    assert (inner.calls, inner.resets) == (0, 1)


@pytest.mark.parametrize(("hook", "mk_cache"), ARCH_HOOKS)
def test_resettable_quantizer_phase_restarts_on_prefill(hook, mk_cache):
    """Prefill -> C decode steps -> prefill -> C decode steps, through the arch hook's reset."""
    C, inner = 2, _Inner()
    mix = _Mixer(quant_every(inner, C))
    rs = torch.zeros(1, 1, 2, 3)
    cache = mk_cache(rs)

    def step(t):
        hook(mix, (), {"hidden_states": torch.zeros(1, t, 4), "cache_params": cache}, None)

    step(8)  # prefill: reset, then the handover quantize consumes phase 1 (C=2 -> stored fp)
    assert inner.calls == 0
    step(1)
    assert inner.calls == 1  # phase 2: chunk boundary
    step(1)
    assert inner.calls == 1
    step(8)  # next sequence: reset realigns, handover is phase 1 again
    assert inner.resets == 2
    assert inner.calls == 1
    step(1)
    assert inner.calls == 2  # phase 2: not carried over from the previous sequence


def test_install_state_quant_hook_is_idempotent():
    """Re-installing must REPLACE the hook. Stacking would quantize the state twice per decode
    step (a strictly different, lossier computation) while every shape and dtype still checks out.
    """
    mix = nn.Linear(4, 4)
    calls = []

    def hook(module, args, kwargs, output):
        calls.append(1)

    for _ in range(3):
        install_state_quant_hook(mix, object(), hook)
    mix(torch.randn(1, 4))
    assert len(calls) == 1, f"hook fired {len(calls)}x after 3 installs — handles are stacking"
    mix._sq_handle.remove()


class _DtypeMixer(torch.nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, hidden_states, cache_params=None):
        return hidden_states


class _FalseyCache(SimpleNamespace):
    def __bool__(self):
        return False  # HF cache truthiness need not indicate the presence of state.


@pytest.mark.parametrize(
    "arch,storage",
    [
        ("nemotronh", "tensor"),
        ("nemotronh", "dict"),
        ("qwen3_5", "tensor"),
        ("qwen3_5", "dict"),
        ("kimi_linear", "list"),
    ],
)
@pytest.mark.parametrize("call_style", ["kwargs", "positional"])
def test_architecture_dtype_hooks_preserve_cache_contracts(arch, storage, call_style):
    runtime = importlib.import_module(f"lores.backends.hf.{arch}.integration")
    untouched, mix = _DtypeMixer(0), _DtypeMixer(1)
    if arch == "nemotronh":
        mix.in_proj = torch.nn.Identity()
        layers = [SimpleNamespace(mixer=untouched), SimpleNamespace(mixer=mix)]
    elif arch == "qwen3_5":
        layers = [SimpleNamespace(self_attn=untouched), SimpleNamespace(linear_attn=mix)]
    else:
        layers = [SimpleNamespace(self_attn=untouched, is_linear_attn=False), SimpleNamespace(self_attn=mix, is_linear_attn=True)]
    model = SimpleNamespace(config=SimpleNamespace(model_type="nemotron_h"), model=SimpleNamespace(layers=layers))
    installer = getattr(runtime, f"install_{'kimi' if arch == 'kimi_linear' else arch}_fp32_state_cache")
    get_cache, get_state, _ = getattr(runtime, f"_{'kimi' if arch == 'kimi_linear' else arch}_cache_accessors")
    hook_calls = []
    other_hook = mix.register_forward_pre_hook(lambda *args: hook_calls.append(True))
    assert installer(model) == 1
    assert installer(model) == 1
    assert len(mix._forward_pre_hooks) == 2
    assert other_hook.id in mix._forward_pre_hooks
    assert not untouched._forward_pre_hooks
    x = torch.zeros(1, 1, 4)
    mix(x)  # no cache

    def cache_with(state):
        if storage == "list":
            return _FalseyCache(recurrent_states=[None, state])
        rs = {0: state} if storage == "dict" else state
        return _FalseyCache(layers=[SimpleNamespace(), SimpleNamespace(recurrent_states=rs)])

    def forward(cp):
        if call_style == "positional" and arch != "kimi_linear":
            return mix(x, cp)
        return mix(x, cache_params=cp)  # Kimi's vendor call is keyword-only.

    for state in (None, torch.empty(0, dtype=torch.bfloat16)):
        cp = cache_with(state)
        forward(cp)
        assert get_state(cp, 1) is state
    for _ in range(2):  # fresh sequences must each receive the pin
        original = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
        cp = cache_with(original)
        assert get_cache((x,), {"cache_params": cp}) is cp
        forward(cp)
        pinned = get_state(cp, 1)
        assert pinned.dtype == torch.float32 and torch.equal(pinned, original.float())
        forward(cp)
        assert get_state(cp, 1) is pinned
    assert hook_calls


def test_strict_pin_cannot_hide_a_low_precision_kernel_result():
    from transformers.cache_utils import LinearAttentionLayer

    class Mixer(torch.nn.Module):
        layer_idx = 0

        def forward(self, x, cache_params):
            layer = cache_params.layers[0]
            layer.update_conv_state(torch.zeros(1, 2, 4, dtype=torch.bfloat16))
            layer.update_recurrent_state(torch.ones(1, 2, 3, 4, dtype=torch.bfloat16))
            return x

    m = Mixer()
    install_state_dtype_hooks(
        [m], torch.float32, lambda a, kw: kw.get("cache_params"), get_recurrent_state, set_recurrent_state, strict=True
    )
    with pytest.raises(RuntimeError, match="Kernel returned non-FP32"):
        m(torch.zeros(1, 1, 2), cache_params=SimpleNamespace(layers=[LinearAttentionLayer()]))
