"""Tiny real Nemotron models exercise FP32 cache writes and padded generation."""

from __future__ import annotations

import pytest
import torch

from lores.backends.hf.cache import get_recurrent_state, set_recurrent_state
from lores.backends.hf.nemotronh.integration import (
    install_nemotronh_fp32_state_cache,
    install_nemotronh_state_quant,
)

_PATTERN = "M-*-"


def tiny_nemotronh(*, seed: int = 0, dtype=torch.bfloat16, **overrides):
    """A tiny CPU ``NemotronHForCausalLM`` (4 blocks: mamba / MLP / attention / MLP)."""
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    cfg_kwargs = dict(
        hidden_size=64,
        num_hidden_layers=len(_PATTERN),
        hybrid_override_pattern=_PATTERN,
        intermediate_size=128,
        vocab_size=64,
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=8,
        n_groups=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=16,
        conv_kernel=4,
    )
    cfg_kwargs.update(overrides)
    torch.manual_seed(seed)
    return NemotronHForCausalLM(NemotronHConfig(**cfg_kwargs)).to(dtype).eval()


def _mamba_layer_indices(model):
    return [i for i, b in enumerate(model.model.layers) if hasattr(b.mixer, "in_proj")]


@torch.no_grad()
def _prefill(model, n_tok=6, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (1, n_tok))
    return model(input_ids=ids, use_cache=True, return_dict=True).past_key_values


@torch.no_grad()
def _decode(model, cache, steps=1, seed=1):
    torch.manual_seed(seed)
    for _ in range(steps):
        ids = torch.randint(0, model.config.vocab_size, (1, 1))
        cache = model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True).past_key_values
    return cache


def _rs(cache, layer_idx):
    rs = cache.layers[layer_idx].recurrent_states
    # transformers >= 5.9: states are dicts keyed by state_idx; 5.8: bare tensor
    return rs.get(0) if isinstance(rs, dict) else rs


@pytest.mark.parametrize("mode", ["off", "int8_block"])
def test_state_quant_hook_does_not_regress_the_buffer_dtype(mode):
    """``_nemotronh_state_quant_hook`` writes ``q.to(rs.dtype)`` back with ``copy_``. Under the fp32 cast that
    must stay fp32 *and* keep the same tensor — an assignment-style write (``layer.recurrent_states = ...``)
    would both downcast and break the in-place contract.
    """
    model = tiny_nemotronh()
    install_nemotronh_fp32_state_cache(model)
    install_nemotronh_state_quant(model, mode=mode)
    li = _mamba_layer_indices(model)[0]

    cache = _decode(model, _prefill(model), steps=1)
    first = _rs(cache, li)
    cache = _decode(model, cache, steps=3, seed=7)
    assert _rs(cache, li).dtype is torch.float32
    assert _rs(cache, li) is first


def test_state_quant_hook_actually_quantizes_the_fp32_buffer():
    """Guards the previous test against passing vacuously (a state-quant hook that never fires would also
    leave the buffer fp32). int8 block quant must land on a coarse grid: the stored state has to equal its
    own dequantized representative, which an unquantized fp32 state generically does not.
    """
    model = tiny_nemotronh()
    install_nemotronh_fp32_state_cache(model)
    assert install_nemotronh_state_quant(model, mode="int8_block") == len(_mamba_layer_indices(model))
    li = _mamba_layer_indices(model)[0]

    quantized = _rs(_decode(model, _prefill(model), steps=3), li).clone()

    ref = tiny_nemotronh()  # same seed ⇒ same weights, same inputs, no state quant
    install_nemotronh_fp32_state_cache(ref)
    plain = _rs(_decode(ref, _prefill(ref), steps=3), li)

    assert not torch.equal(quantized, plain), "state-quant hook left the cache untouched"


@pytest.mark.parametrize("prefill", [1, 6])
def test_real_nemotron_is_fp32_from_first_store_and_every_decode(prefill):
    from lores.backends.hf.nemotronh.integration import install_nemotronh_fp32_state_cache

    model = tiny_nemotronh()
    install_nemotronh_fp32_state_cache(model, strict=True)
    cp = _prefill(model, n_tok=prefill)
    state = _rs(cp, 0)
    assert state.dtype == torch.float32
    assert not torch.equal(state, state.bfloat16().float())
    cp = _decode(model, cp, steps=3)
    assert _rs(cp, 0) is state and state.dtype == torch.float32


def test_strict_pin_rejects_already_rounded_nonzero_state():
    from lores.backends.hf.nemotronh.integration import install_nemotronh_fp32_state_cache

    model = tiny_nemotronh()
    cp = _prefill(model)
    set_recurrent_state(cp, 0, get_recurrent_state(cp, 0).bfloat16())
    install_nemotronh_fp32_state_cache(model, strict=True)
    with pytest.raises(RuntimeError, match="already-rounded"):
        _decode(model, cp)


def _left_padded_batch(pad=7):
    """(short prompt alone, the same prompt left-padded next to a longer one, its 2D mask)."""
    short = torch.tensor([[3, 5, 9, 2, 11]])
    long_ = torch.tensor([[1, 4, 6, 8, 10, 12, 14, 2, 3, 5]])
    n = long_.shape[1] - short.shape[1]
    ids = torch.cat([torch.cat([torch.full((1, n), pad), short], 1), long_], 0)
    mask = torch.cat([torch.cat([torch.zeros(1, n, dtype=torch.long), torch.ones_like(short)], 1), torch.ones_like(long_)], 0)
    return short, ids, mask


def test_decode_mask_covers_the_whole_kv_cache():
    """Direct check: after prefill the mask sizes must reflect the attention KV."""
    from transformers import DynamicCache

    model = tiny_nemotronh(dtype=torch.float32)
    _, ids, mask = _left_padded_batch()
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(ids, attention_mask=mask, past_key_values=cache, use_cache=True)

    assert cache.get_seq_length() == ids.shape[1]
    assert cache.get_mask_sizes(1, 0) == (ids.shape[1] + 1, 0)


def test_left_padding_does_not_change_the_generation():
    """End to end: a row's greedy continuation must not depend on the pads sharing its batch."""
    model = tiny_nemotronh(dtype=torch.float32)
    short, ids, mask = _left_padded_batch()
    gen = dict(max_new_tokens=12, do_sample=False, pad_token_id=0)
    with torch.no_grad():
        alone = model.generate(short, attention_mask=torch.ones_like(short), **gen)
        batched = model.generate(ids, attention_mask=mask, **gen)

    assert torch.equal(alone[0, short.shape[1] :], batched[0, ids.shape[1] :])
