"""Per-head and per-channel precision preserves grouping, rounding, and FP32 protection."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from lores.evaluation.state_quant import bind_mask_quantizers
from lores.fake_quant.quantizers import quantize_state
from lores.fake_quant.state import AxisBitsQuant, HeadBitsQuant
from scripts.calibration.allocate import build_promotion_masks
from scripts.evaluation.delta_ell_hybrid import build_parser


def _peaky_state(B=1, H=4, dk=32, dv=16, seed=0):
    """``[B, H, dk, dv]`` with one hot key-row and one hot value-column.

    ``dk != dv`` by default so a missing/extra transpose in the ``_dk`` path cannot hide behind
    a square shape.
    """
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(B, H, dk, dv, generator=g)
    S[:, :, 3, :] *= 50  # hot key-row (dk index)
    S[:, :, :, 5] *= 50  # hot value-column (dv index)
    return S


def _ref_axis_quant(x, bits, dim):
    """Independent reference: symmetric RTN with ONE scale per slice along ``dim``.

    Written straight from the spec (``scale = amax_dim / qmax``, round, clamp, rescale) rather
    than from the pad/reshape blocking machinery of ``quantize_state``, so it constrains the
    axis semantics instead of echoing them. Valid whenever the block covers the whole axis.
    Returns ``(dequantized, scale)``.
    """
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax) * scale, scale


def _ref_mixed(x, bits, mask, layout="kv"):
    """Reference routing for a per-head axis mask: value axis = dk-reduction in ``kv``."""
    val_dim, key_dim = (-2, -1) if layout == "kv" else (-1, -2)
    out = torch.empty_like(x)
    for h in range(x.shape[1]):
        dim = val_dim if bool(mask[h]) else key_dim
        out[:, h] = _ref_axis_quant(x[:, h], bits, dim)[0]
    return out


def _uniform(bits, mask, layout):
    return AxisBitsQuant(mask, torch.full((mask.numel(),), bits), layout=layout)


@pytest.mark.parametrize("bits", [4, 6, 8])
@pytest.mark.parametrize("layout", ["kv", "vk"])
def test_mixed_axis_matches_uniform_modes(bits, layout):
    """All-True mask == pure value axis; all-False == pure key axis, in BOTH layouts.

    Compared against the independent axis reference, not against ``quantize_state``'s own mode
    strings — an axis swap inside ``quantize_state`` would move both sides of that comparison
    together.
    """
    S = _peaky_state()
    H = S.shape[1]
    val_dim, key_dim = (-2, -1) if layout == "kv" else (-1, -2)
    all_val = _uniform(bits, torch.ones(H, dtype=torch.bool), layout)(S)
    all_key = _uniform(bits, torch.zeros(H, dtype=torch.bool), layout)(S)
    assert torch.equal(all_val, _ref_axis_quant(S, bits, val_dim)[0])
    assert torch.equal(all_key, _ref_axis_quant(S, bits, key_dim)[0])
    assert not torch.equal(all_val, all_key)  # the two axes must be distinguishable here


@pytest.mark.parametrize("layout", ["kv", "vk"])
def test_mixed_axis_mask_routes_per_head(layout):
    """Mixed mask: every head takes the axis its mask bit selects, and no head is left raw."""
    S = _peaky_state()
    mask = torch.tensor([True, False, True, False])
    out = _uniform(6, mask, layout)(S)
    assert torch.equal(out, _ref_mixed(S, 6, mask, layout))
    assert not torch.equal(out, S)  # a head skipped by the routing would come back unquantized


class TestAxisBitsQuant:
    """AxisBitsQuant must equal HeadBitsQuant when the axis is uniform — any drift from the
    parent is a bug in the composition, not the parent.
    """

    def _state(self, B=1, H=4, dk=16, dv=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(B, H, dk, dv, generator=g) * torch.logspace(-2, 1, H).view(1, H, 1, 1)

    def _ref_per_head(self, h, mask, bits, layout="kv", bs=128):
        """Per-head reference: quantize each head alone with the mode its (axis, bits) pair names.

        Head-at-a-time, so a routing bug cannot hide behind a shared reduction.
        """
        out = torch.empty_like(h)
        for i, (v, b) in enumerate(zip(mask.tolist(), bits.tolist())):
            val, key = ("block", "block_dk") if layout == "vk" else ("block_dk", "block")
            out[:, i : i + 1] = quantize_state(h[:, i : i + 1], f"int{b}_{val if v else key}", bs)
        return out

    def test_each_head_quantized_per_its_own_axis_and_bits(self):
        h = self._state()
        mask = torch.tensor([True, False, True, False])
        bits = torch.tensor([4, 4, 6, 6])
        q = AxisBitsQuant(mask, bits)(h)
        for i, (v, b) in enumerate(zip(mask.tolist(), bits.tolist())):
            mode = f"int{b}_block_dk" if v else f"int{b}_block"
            ref = quantize_state(h[:, i : i + 1], mode, 128)
            assert torch.equal(q[:, i : i + 1], ref), f"head {i} ({mode})"

    def test_vk_layout_flips_the_grans(self):
        h = self._state()
        mask = torch.tensor([True, False, True, False])
        bits = torch.tensor([4, 6, 4, 6])
        q = AxisBitsQuant(mask, bits, layout="vk")(h)
        for i, (v, b) in enumerate(zip(mask.tolist(), bits.tolist())):
            mode = f"int{b}_block" if v else f"int{b}_block_dk"
            ref = quantize_state(h[:, i : i + 1], mode, 128)
            assert torch.equal(q[:, i : i + 1], ref), f"head {i} ({mode})"

    def test_batch_dimension_is_fully_quantized(self):
        """Every batch row must be routed, not just row 0 — ``out[0, sel]`` for ``out[:, sel]``
        leaves the rest of the batch raw, which B=1 cannot see.
        """
        h = self._state(B=3)
        mask = torch.tensor([True, False, True, False])
        bits = torch.tensor([4, 6, 6, 4])
        q = AxisBitsQuant(mask, bits)(h)
        assert torch.equal(q, self._ref_per_head(h, mask, bits))
        assert not torch.equal(q[1:], h[1:])  # rows 1..B-1 are not passed through raw

    def test_non_contiguous_state_is_quantized_identically(self):
        """The stepper hands over whatever the recurrence produced; a ``[B, H, dv, dk]`` buffer
        viewed as ``[B, H, dk, dv]`` is non-contiguous. Result must not depend on the layout.
        """
        h = self._state(dk=32, dv=16).transpose(-1, -2)  # [1, 4, 16, 32], non-contiguous
        assert not h.is_contiguous()
        mask = torch.tensor([True, False, True, False])
        bits = torch.tensor([4, 6, 4, 6])
        q = AxisBitsQuant(mask, bits)(h)
        assert torch.equal(q, AxisBitsQuant(mask, bits)(h.contiguous()))
        assert torch.equal(q, self._ref_per_head(h, mask, bits))

    @pytest.mark.parametrize("layout", ["kv", "vk"])
    @pytest.mark.parametrize("mask_values", [[True] * 4, [False] * 4, [True, False, True, False]])
    def test_fp32_allocations_preserve_values_in_mixed_precision(self, layout, mask_values):
        h = self._state(B=2)
        # Exercise sub-FP16 precision and values outside FP16's finite range.
        h[:, 0, 0, 0] = 1.0001
        h[:, 1, 0, 0] = 100000.0
        original = h.clone()
        bits = torch.tensor([32, 32, 16, 4])
        mask = torch.tensor(mask_values)
        out = AxisBitsQuant(mask, bits, layout=layout)(h)

        assert out.dtype == torch.float32
        assert torch.equal(out[:, :2], h[:, :2])
        assert torch.isfinite(out).all()
        assert not torch.equal(h[:, :2], h[:, :2].half().float())
        assert torch.equal(out[:, 2], h[:, 2].half().float())
        val, key = ("block", "block_dk") if layout == "vk" else ("block_dk", "block")
        for i, value_axis in enumerate(mask_values):
            reference = HeadBitsQuant(val if value_axis else key, bits[i : i + 1])(h[:, i : i + 1])
            assert torch.equal(out[:, i : i + 1], reference)
        assert torch.equal(h, original)

    @pytest.mark.parametrize("layout", ["kv", "vk"])
    def test_all_fp32_allocations_are_passthrough(self, monkeypatch, layout):
        import lores.fake_quant.state as state

        def unexpected_quantization(*args, **kwargs):
            pytest.fail("FP32 allocations must not invoke an integer quantizer")

        monkeypatch.setattr(state, "quantize_state", unexpected_quantization)
        h = self._state(B=2).transpose(-1, -2)
        out = state.AxisBitsQuant(torch.tensor([True, False, True, False]), torch.full((4,), 32), layout=layout)(h)
        assert torch.equal(out, h)


def test_diagonal_ranking_promotes_channels_not_head_averages():
    scores = torch.tensor([[[-0.01, -0.9], [-0.8, -0.02]]])
    masks = build_promotion_masks(scores, 0.5, "key", "value", budget_scope="per-layer")
    assert masks["decay"]["bits"]["0"] == [[6, 4], [4, 6]]
    assert masks["decay"]["axes"]["0"] == [[True, False], [False, True]]
    assert torch.tensor(masks["random"]["bits"]["0"]).eq(6).sum() == 2


@pytest.mark.parametrize("fraction,expected", [(0, 0), (0.5, 2), (1, 4)])
@pytest.mark.parametrize("shape", [(1, 4), (1, 2, 2)])
def test_fraction_and_random_control_have_identical_budgets(shape, fraction, expected):
    scores = torch.zeros(shape)
    a = build_promotion_masks(scores, fraction, "value", "key", seed=9, budget_scope="per-layer")
    assert a == build_promotion_masks(scores, fraction, "value", "key", seed=9, budget_scope="per-layer")
    for item in a.values():
        assert torch.tensor(item["bits"]["0"]).eq(6).sum() == expected


def test_channel_specific_axis_switch_uses_each_precision_and_axis():
    torch.manual_seed(3)
    x = torch.randn(1, 2, 4, 8)
    bits = torch.tensor([[4, 6, 4, 6], [6, 4, 6, 4]])
    axes = bits == 6
    actual = AxisBitsQuant(axes, bits, mode="int4_block_asym")(x)
    expected = torch.where(axes[None, ..., None], quantize_state(x, "int6_block_dk_asym"), quantize_state(x, "int4_block_asym"))
    assert torch.equal(actual, expected)


def test_channel_mask_shape_is_checked():
    bits = torch.ones(2, 3).long() * 4
    with pytest.raises(ValueError, match="Channel mask"):
        AxisBitsQuant(bits == 4, bits)(torch.randn(1, 2, 4, 8))


@pytest.mark.parametrize("mode", ["fpE4M3_block", "off", "int4_tensor", "int4_block_unknown"])
def test_axis_allocation_rejects_unsupported_modes(mode):
    with pytest.raises(ValueError, match="integer block mode"):
        AxisBitsQuant(torch.tensor([True]), torch.tensor([4]), mode=mode)


H, DK, L = 4, 8, 3


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self._state_quant = "SENTINEL"
        self._sq_handle = object()


def _model(n=L):
    m = nn.Module()
    m.mixers = nn.ModuleList([_Mixer() for _ in range(n)])
    return m


def _write(tmp_path, name, per_layer):
    p = tmp_path / name
    p.write_text(json.dumps({str(i): per_layer for i in range(L)}))
    return str(p)


def _args(model="qwen3_5", **kw):
    a = build_parser().parse_args(["--model", model, "--mode", "int4_block", "--rot", "none"])
    for k, v in kw.items():
        assert hasattr(a, k), f"unknown flag {k!r}"
        setattr(a, k, v)
    return a


def _quants(m):
    return [mix._state_quant for mix in m.mixers]


def test_headbits_plus_mixhead_composes_to_axis_bits_quant(tmp_path):
    m = _model()
    a = _args(
        headbits=_write(tmp_path, "hb.json", [6] * H),
        mixhead=_write(tmp_path, "mh.json", [True, False] * (H // 2)),
    )
    bind_mask_quantizers(m, a, a.mode)
    assert all(isinstance(q, AxisBitsQuant) for q in _quants(m))


def test_headbits_alone_binds_head_bits_quant_at_the_modes_own_gran(tmp_path):
    m = _model()
    a = _args(mode="int4_block_dk", headbits=_write(tmp_path, "hb.json", [6] * H))
    bind_mask_quantizers(m, a, a.mode)
    qs = _quants(m)
    assert all(isinstance(q, HeadBitsQuant) and q.gran == "block_dk" for q in qs)


def test_mixhead_alone_is_rejected(tmp_path):
    m = _model()
    a = _args(mixhead=_write(tmp_path, "mh.json", [True] * H))
    with pytest.raises(ValueError, match="--mixhead requires --headbits"):
        bind_mask_quantizers(m, a, a.mode)
    assert _quants(m) == ["SENTINEL"] * L


@pytest.mark.parametrize(
    ("model", "val_gran", "key_gran"),
    [("qwen3_5", "block_dk", "block"), ("nemotronh", "block", "block_dk")],
)
def test_layout_flips_the_value_axis_for_nemotronhs_vk_state(tmp_path, model, val_gran, key_gran):
    # nemo state is [.., dv, dk] so "block" is ITS value axis; getting this backwards silently
    # swaps which heads get value-axis scales.
    m = _model()
    a = _args(
        model,
        headbits=_write(tmp_path, "hb.json", [6] * H),
        mixhead=_write(tmp_path, "mh.json", [True] * H),
    )
    bind_mask_quantizers(m, a, a.mode)
    q = _quants(m)[0]
    assert (q.val_gran, q.key_gran) == (val_gran, key_gran)


def test_mode_off_and_no_flags_are_noops(tmp_path):
    m = _model()
    a = _args(headbits=_write(tmp_path, "hb.json", [6] * H))
    bind_mask_quantizers(m, a, "off")
    assert _quants(m) == ["SENTINEL"] * L
    bind_mask_quantizers(m, _args(), "int4_block")
    assert _quants(m) == ["SENTINEL"] * L


def test_layer_count_mismatch_aborts_instead_of_binding_a_prefix(tmp_path):
    m = _model(n=L + 1)
    a = _args(headbits=_write(tmp_path, "hb.json", [6] * H))
    with pytest.raises(AssertionError, match="mixers vs"):
        bind_mask_quantizers(m, a, a.mode)


@pytest.mark.parametrize("model", ["qwen3_5", "nemotronh"])
@pytest.mark.parametrize("variant", ["_sr", "_asym", "_asym_sr"])
def test_runner_passes_actual_mode_to_composed_quantizer(tmp_path, model, variant):
    m = _model()
    a = _args(
        model,
        mode="int4_block" + variant,
        headbits=_write(tmp_path, "bits.json", [4, 6, 4, 6]),
        mixhead=_write(tmp_path, "axes.json", [True, True, False, False]),
    )
    bind_mask_quantizers(m, a, a.mode)
    assert all(q.variant == variant for q in _quants(m))
