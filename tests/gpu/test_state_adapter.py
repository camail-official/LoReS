"""Exercise the serving cache adapter independently of a full vLLM engine."""

import itertools
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from lores.backends.vllm import config, state
from lores.kernels.state_int.packing.pack_state_kernel import pack_state_to_slots, unpack_state_from_slots
from lores.kernels.state_int.packing.pack_state_key_axis_naive import pack_key_axis_to_slots, unpack_key_axis_from_slots


class KDAMixer:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("axis", ["value", "key"])
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("sr", [False, True])
def test_prefill_handoff_and_continuation_preserve_slots(monkeypatch, axis, asym, sr):
    """The adapter must address the same pools, scales axis and SR sequence as the direct pack kernels."""
    monkeypatch.setattr(
        config,
        "_RESOLVED",
        config.LinqConfig(
            state_bits=8,
            state_axis=axis,
            state_asym=asym,
            state_sr=sr,
        ),
    )
    monkeypatch.setattr(state, "_PACK_SEED", itertools.count(17))
    torch.manual_seed(4)
    source = torch.randn(2, 2, 32, 64, device="cuda")
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    codes = torch.full((5, 2, 32, 64), 23, device="cuda", dtype=torch.uint8)
    rows = 64 if axis == "key" else 32
    scales = torch.ones((5, 2, rows) + ((2,) if asym else ()), device="cuda")
    mixer = KDAMixer()
    mixer.kv_cache = (torch.empty(0, device="cuda"), codes, scales)
    expected_codes, expected_scales = codes.clone(), scales.clone()
    pack, unpack = (
        (pack_key_axis_to_slots, unpack_key_axis_from_slots) if axis == "key" else (pack_state_to_slots, unpack_state_from_slots)
    )
    pack(source, expected_codes, expected_scales, indices, asym=asym, sr_seed=17 if sr else None)
    state.linq_pack_slots(mixer, indices, source)
    torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    # Whole-pool equality above also checks that unused/null slots were not touched.
    got, want = torch.empty_like(source), torch.empty_like(source)
    state.linq_unpack_slots(mixer, indices, got)
    unpack(want, expected_codes, expected_scales, indices, asym=asym)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_profiling_noop_and_reusable_fp32_scratch():
    mixer = SimpleNamespace(kv_cache=(torch.empty(0), torch.empty(0)), get_state_shape=lambda: ((3, 16), (2, 4, 8), (2, 4)))
    state.linq_pack_slots(mixer, None, None)  # profiling has no accounted pools yet
    small = state.linq_scratch(mixer, 2, "cpu")
    assert small.dtype == torch.float32 and small.shape == (2, 2, 4, 8)
    assert state.linq_scratch(mixer, 1, "cpu").data_ptr() == small.data_ptr()
    large = state.linq_scratch(mixer, 3, "cpu")
    assert large.shape == (3, 2, 4, 8) and large.data_ptr() != small.data_ptr()


def test_key_axis_accepts_any_mixer(monkeypatch):
    """Key-axis packing is not KDA-only (GDN and Mamba-2 kernels carry it); profiling stays a no-op."""
    monkeypatch.setattr(config, "_RESOLVED", config.LinqConfig(state_bits=8, state_axis="key"))
    mixer = SimpleNamespace(kv_cache=(torch.empty(0), torch.empty(0, dtype=torch.uint8), torch.empty(0)))
    assert state.linq_pack_slots(mixer, None, None) is None
