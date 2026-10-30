"""State grids, grouping dimensions, rounding, and finite FP32 representatives."""

from __future__ import annotations

import pytest
import torch

from lores.fake_quant import quantizers
from lores.fake_quant.quantizers import (
    _fp_grid,
    quant_fp_block,
    quantize_state,
)

EPS32 = torch.finfo(torch.float32).eps


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


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_quantize_state_block_scales_over_last_axis(bits):
    """``int{b}_block`` == one symmetric scale per dk-row, taken over the dv axis. Bit-exact."""
    S = _peaky_state()
    ref, _ = _ref_axis_quant(S, bits, dim=-1)
    assert torch.equal(quantize_state(S, f"int{bits}_block"), ref)


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_quantize_state_block_dk_scales_over_second_to_last_axis(bits):
    """``int{b}_block_dk`` == one symmetric scale per dv-column, taken over the dk axis."""
    S = _peaky_state()
    ref, _ = _ref_axis_quant(S, bits, dim=-2)
    assert torch.equal(quantize_state(S, f"int{bits}_block_dk"), ref)


@pytest.mark.parametrize("mode", ["int6_block", "int6_block_dk", "int4_block", "int4_block_dk"])
def test_quantize_state_saturates_the_top_code_exactly(mode):
    """The per-group amax entry must map to exactly ``qmax`` — pins the code range (off-by-one
    in ``qmax`` survives an error-magnitude check but not this).
    """
    S = _peaky_state()
    bits = 6 if "int6" in mode else 4
    dim = -2 if mode.endswith("_dk") else -1
    _, scale = _ref_axis_quant(S, bits, dim)
    codes = (quantize_state(S, mode) / scale).round()
    assert int(codes.abs().amax()) == (1 << (bits - 1)) - 1


@pytest.mark.parametrize("mode", ["int6_block", "int6_block_dk", "int8_block", "int4_block"])
def test_quantize_state_all_zero_state_is_exact_and_finite(mode):
    """A zero state must survive the ``clamp_min(1e-12)`` scale guard as exact zeros, no NaN."""
    Z = torch.zeros(2, 3, 8, 4)
    out = quantize_state(Z, mode)
    assert torch.equal(out, Z)


def test_quantize_state_block_size_is_honoured():
    """A smaller block must give the quiet columns their own scale (behavioural, not a
    re-implementation): one loud group must not poison the rest of the row.
    """
    S = torch.full((1, 1, 4, 32), 0.01)
    S[..., :8] = 1000.0  # one loud group of 8 along the last (dv) axis
    quiet = slice(8, None)
    err_coarse = (quantize_state(S, "int6_block", 32) - S)[..., quiet].abs().amax()
    err_fine = (quantize_state(S, "int6_block", 8) - S)[..., quiet].abs().amax()
    # coarse: step = 1000/31 ~ 32; fine: step = 0.01/31 ~ 3e-4. Ratio ~1e5; assert >=1e3.
    assert err_fine * 1e3 < err_coarse


@pytest.mark.parametrize("bits", [4, 6, 8])
@pytest.mark.parametrize("dim,mode", [(-1, "int{b}_block"), (-2, "int{b}_block_dk")])
def test_rounding_error_obeys_lemma_1_bound(bits, dim, mode):
    """Lemma 1: ``|q - x| <= min(|x|, step/2)`` per entry, both grouping axes.

    Real-arithmetic proof: with ``s = amax/qmax`` no entry clamps, so ``q = round(x/s)*s``.
    If ``|x| < s/2`` then ``q == 0`` and the error is ``|x|`` exactly; otherwise it is at most
    ``s/2``. Both branches are tight, so the bound holds with equality on part of the domain
    and the only admissible slack is float32 evaluation error:

        q_hat = fl(n * fl(s)),  n = round(fl(x / fl(s))),  |n - x/s| <= 1/2 + eps
        => |q_hat - x| <= s/2 + 2*eps*|x| + eps*s

    A factor-4 safety margin over that gives ``4*eps*(|x| + s)`` — for these magnitudes ~1e-4
    of a quantization step, i.e. it cannot absorb a real violation (which would be O(s)).
    """
    S = _peaky_state()
    q = quantize_state(S, mode.format(b=bits))
    _, scale = _ref_axis_quant(S, bits, dim)
    step = scale.expand_as(S)
    slack = 4 * EPS32 * (S.abs() + step)
    bound = torch.minimum(S.abs(), step / 2)
    assert bool(((q - S).abs() <= bound + slack).all())


@pytest.mark.parametrize("e,m,bits", [(2, 1, 4), (3, 0, 4), (2, 3, 6), (3, 2, 6), (4, 3, 8), (5, 2, 8)])
def test_grid_has_one_magnitude_per_code(e, m, bits):
    """E4M3FN reserves one positive code for NaN; other research grids are all-finite."""
    g = _fp_grid(e, m)
    assert g.numel() == 2 ** (e + m) - int((e, m) == (4, 3))
    assert g[0] == 0.0
    assert torch.all(g[1:] > 0) and torch.all(g.diff() > 0), "ascending, strictly positive after 0"
    assert 1 + e + m == bits


def test_quantized_values_lie_on_the_scaled_grid():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64)
    q = quant_fp_block(x, 2, 3, block_size=64)
    grid = _fp_grid(2, 3)
    scale = x.abs().amax(dim=-1, keepdim=True) / grid[-1]
    ratios = (q / scale).abs()
    # every quantized magnitude must equal some grid point
    assert torch.isclose(ratios.unsqueeze(-1), grid.view(1, 1, 1, -1), atol=1e-5).any(-1).all()


def test_e4m3_grid_matches_every_finite_hardware_code():
    values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    expected = values[torch.isfinite(values)].abs().unique().sort().values
    assert torch.equal(_fp_grid(4, 3), expected)
    assert expected[-1] == 448


def test_e4m3_rounding_matches_hardware_including_every_midpoint():
    grid = _fp_grid(4, 3)
    mid = (grid[1:] + grid[:-1]) / 2
    x = torch.cat([grid, mid, -grid, -mid]).unsqueeze(0)
    # Including +/-448 fixes the shared scale to exactly 1.
    actual = quant_fp_block(x, 4, 3, x.shape[-1])
    assert torch.equal(actual, x.to(torch.float8_e4m3fn).float())


def test_e4m3_sr_uses_finite_neighbors_without_changing_the_grid():
    torch.manual_seed(18)
    x = torch.full((1, 4096), 400.0)
    x[0, -1] = 448
    q = quant_fp_block(x, 4, 3, 4096, stochastic=True)
    assert set(q[0, :-1].unique().tolist()) == {384.0, 416.0}
    assert abs(q[0, :-1].mean().item() - 400) < 1
    assert q[0, -1] == 448


def test_block_amax_maps_to_the_codes_largest_value():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32)
    for e, m in ((2, 1), (3, 2), (4, 3)):
        q = quant_fp_block(x, e, m, block_size=32)
        i = x.abs().argmax()
        assert torch.isclose(q.flatten()[i].abs(), x.abs().max(), rtol=1e-5), "amax must be exact"


def test_representable_input_is_a_fixed_point():
    """Values already on the (scaled) grid must survive quantization unchanged."""
    grid = _fp_grid(3, 2)
    x = torch.cat([grid, -grid]).reshape(1, -1)
    q = quant_fp_block(x, 3, 2, block_size=x.shape[-1])
    assert torch.allclose(q, x, rtol=1e-6, atol=1e-8)


def test_mode_dispatch_and_dk_variant():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 16, 64)
    direct = quant_fp_block(x, 2, 3, 64)
    assert torch.equal(quantize_state(x, "fpE2M3_block", block_size=64), direct)
    # "_dk" groups along the second-to-last axis instead (transpose-quant-transpose)
    dk = quantize_state(x, "fpE2M3_block_dk", block_size=16)
    expect = quant_fp_block(x.transpose(-1, -2).contiguous(), 2, 3, 16).transpose(-1, -2)
    assert torch.allclose(dk, expect)
    assert not torch.equal(dk, direct), "the two groupings must differ"


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("size", [8, 11])
def test_integer_variants_preserve_grid_padding_and_rng(bits, asym, stochastic, size):
    x = torch.linspace(-2, 3, 2 * size).reshape(2, size)
    # Affine ranges use real elements; padding still preserves the RNG draw shape.
    padded = torch.nn.functional.pad(x, (0, (-size) % 8)).reshape(2, -1, 8)
    if asym:
        low = torch.stack([chunk.amin(-1) for chunk in x.split(8, dim=-1)], dim=-1).unsqueeze(-1)
        high = torch.stack([chunk.amax(-1) for chunk in x.split(8, dim=-1)], dim=-1).unsqueeze(-1)
        qmin, qmax = 0, 2**bits - 1
        scale = ((high - low) / qmax).clamp_min(1e-12)
        v = (padded - low) / scale
    else:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax
        scale = padded.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
        v = padded / scale
    torch.manual_seed(71)
    codes = v.floor() + (torch.rand_like(v) < v - v.floor()).float() if stochastic else v.round()
    expected = codes.clamp(qmin, qmax) * scale
    if asym:
        expected = expected + low
    expected = expected.reshape(2, -1)[:, :size]
    expected_rng = torch.random.get_rng_state()

    name = "quant_intn_block" + ("_asym" if asym else "") + ("_sr" if stochastic else "")
    torch.manual_seed(71)
    actual = getattr(quantizers, name)(x, bits, 8)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


def test_saturating_quantizer():
    from lores.fake_quant.quantizers import (
        _sat_counts,
        quantize_state,
        saturation_events,
    )

    _sat_counts.clear()
    h = torch.randn(1, 2, 8, 128)
    h[0, 0, 0, 0] = float("inf")
    h[0, 1, 2, 3] = float("nan")
    out = quantize_state(h, "int6_block")
    assert torch.isfinite(out).all()
    assert saturation_events() == 2
    # clean input adds nothing and is numerically untouched by the guard
    h2 = torch.randn(1, 2, 8, 128)
    assert torch.isfinite(quantize_state(h2, "int6_block")).all()
    assert saturation_events() == 2


@pytest.mark.parametrize("name", ["quant_intn_block", "quant_intn_block_sr", "quant_intn_block_asym", "quant_intn_block_asym_sr"])
@pytest.mark.parametrize("bits", [1, 9, 16])
def test_integer_primitives_reject_unsupported_widths(name, bits):
    from lores.fake_quant import quantizers

    with pytest.raises(ValueError, match="2–8 bits"):
        getattr(quantizers, name)(torch.ones(1, 2, 8, 4), bits)
