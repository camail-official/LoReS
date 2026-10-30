"""Rotation matrices, signs, and inverse order preserve the intended basis."""

from __future__ import annotations

import pytest
import torch

from lores.rotations.matrices import get_orthogonal_matrix, r4_signs
from lores.utils.hadamard_utils import (
    get_had76,
    get_had228,
    get_had420,
    get_hadK,
)


@pytest.mark.parametrize("dim", [8, 16, 32, 64])
def test_orthogonal_matrix_is_a_signed_hadamard(dim: int):
    torch.manual_seed(0)
    R = get_orthogonal_matrix(dim, device="cpu")
    assert R.shape == (dim, dim)
    err = (R @ R.t() - torch.eye(dim, dtype=R.dtype)).abs().max().item()
    assert err < 1e-5, f"R not orthogonal: ‖RRᵀ-I‖∞ = {err}"
    assert torch.all((R * dim**0.5).round().abs() == 1)


def _check_hadamard(H: torch.Tensor, K: int) -> None:
    assert H.shape == (K, K), f"shape mismatch: {tuple(H.shape)} vs ({K},{K})"
    assert torch.all((H == 1) | (H == -1)), "non ±1 entries"
    err = (H @ H.t() - K * torch.eye(K)).abs().max().item()
    assert err == 0.0, f"H @ H.T != {K}·I (max abs err {err})"


def test_get_had76_is_valid_hadamard():
    _check_hadamard(get_had76(), 76)


def test_get_had228_is_valid_hadamard():
    _check_hadamard(get_had228(), 228)


def test_get_had420_is_valid_hadamard():
    _check_hadamard(get_had420(), 420)


def test_get_hadK_upstream_K_unchanged():
    """H_76/228/420 (beyond upstream SpinQuant) must not change the K selection for any flagship LLM size."""
    # (size, expected K, expected shape-or-None)
    expectations = [
        (5632, 44, (44, 44)),  # fla 1.3B GDN/KDA intermediate
        (11008, 172, (172, 172)),  # LLaMA-2-7B intermediate
        (13824, 108, (108, 108)),  # LLaMA-1-13B intermediate
        (17920, 140, (140, 140)),  # LLaMA-1-30B intermediate
        (2816, 44, (44, 44)),  # 340M paper-checkpoints intermediate
        (1024, 1, None),  # pure power of two → butterfly only, no explicit small Hadamard
        (128, 1, None),
    ]
    for size, want_K, want_shape in expectations:
        H, K = get_hadK(size)
        assert K == want_K, f"get_hadK({size}): K={K}, expected {want_K}"
        if want_shape is None:
            assert H is None, f"get_hadK({size}): expected H=None, got shape {tuple(H.shape)}"
        else:
            assert H is not None and tuple(H.shape) == want_shape


@pytest.mark.parametrize("layer", [0, 7, 23])
@pytest.mark.parametrize("width", [128, 2304])
def test_r4_signs_match_the_checkpoint_seed_without_touching_global_rng(layer, width):
    from lores.backends.vllm import rotations

    generator = torch.Generator(device="cpu").manual_seed(1000 + layer)
    expected = (torch.randint(0, 2, (width,), generator=generator, device="cpu") * 2 - 1).float()
    rng = torch.random.get_rng_state()
    assert torch.equal(r4_signs(width, layer), expected)
    assert torch.equal(rotations.r4_sign(width, f"model.layers.{layer}.mlp"), expected)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert rotations.r4_signs is r4_signs


ATOL = 1e-4


def _orthogonal(d, seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return (q * torch.sign(torch.diag(r))).to(torch.float32)


def _inputs(B=2, T=16, H=2, K=8, V=8, seed=0):
    g = torch.Generator().manual_seed(seed)

    def f(*s):
        return torch.randn(*s, generator=g, dtype=torch.float32)

    return f(B, T, H, K), f(B, T, H, K), f(B, T, H, V)  # q, k, v


def _check(op_name, run, *, qk_invariant):
    """run(q,k,v) -> o. Check R_qk and R_v behavior."""
    q, k, v = _inputs()
    M = _orthogonal(q.shape[-1], 1)
    N = _orthogonal(v.shape[-1], 2)
    o = run(q, k, v)

    # R_qk: rotate q AND k by the same M
    o_qk = run(q @ M, k @ M, v)
    qk_err = (o_qk - o).abs().max().item()
    if qk_invariant:
        assert qk_err < ATOL, f"{op_name}: R_qk should be invariant, err={qk_err:.2e}"
        qk_msg = f"INVARIANT err={qk_err:.2e}"
    else:
        assert qk_err > 1e-2, f"{op_name}: R_qk should BREAK (diagonal decay), but err={qk_err:.2e} (too small)"
        qk_msg = f"breaks (err={qk_err:.2e}) — guard justified"

    # R_v: rotate v by N -> output rotated by N -> recover with @ Nᵀ
    o_v = run(q, k, v @ N)
    v_err = (o_v @ N.t() - o).abs().max().item()
    assert v_err < ATOL, f"{op_name}: R_v covariance failed, err={v_err:.2e}"
    print(f"{op_name:18} R_qk: {qk_msg:38}  R_v: covariant err={v_err:.2e}  OK")


def test_simple_gla_scalar():  # linear/scalar (mamba2 cell) — R_qk valid
    from fla.ops.simple_gla.naive import naive_recurrent_simple_gla

    g = torch.Generator().manual_seed(3)
    glog = -torch.rand(2, 16, 2, generator=g) * 0.5  # scalar per-head log-decay
    _check(
        "simple_gla/scalar",
        lambda q, k, v: naive_recurrent_simple_gla(q, k, v, glog, output_final_state=False)[0],
        qk_invariant=True,
    )


def test_simple_gla_none():  # linear/none (linear_none cell) — R_qk valid
    from fla.ops.simple_gla.naive import naive_recurrent_simple_gla

    glog = torch.zeros(2, 16, 2)  # no decay
    _check(
        "simple_gla/none",
        lambda q, k, v: naive_recurrent_simple_gla(q, k, v, glog, output_final_state=False)[0],
        qk_invariant=True,
    )


def test_gla_diagonal():  # linear/diagonal (gla cell) — R_qk must BREAK
    from fla.ops.gla.naive import naive_recurrent_gla

    g = torch.Generator().manual_seed(4)
    gk = -torch.rand(2, 16, 2, 8, generator=g) * 0.5  # per-channel diagonal log-decay
    _check("gla/diagonal", lambda q, k, v: naive_recurrent_gla(q, k, v, gk, output_final_state=False)[0], qk_invariant=False)


def test_kda_diagonal():  # delta/diagonal (kda cell) — R_qk must BREAK; R_v valid
    from fla.ops.kda.naive import naive_recurrent_kda

    g = torch.Generator().manual_seed(5)
    gk = -torch.rand(2, 16, 2, 8, generator=g) * 0.5
    beta = torch.rand(2, 16, 2, generator=g)
    _check(
        "kda/diagonal", lambda q, k, v: naive_recurrent_kda(q, k, v, gk, beta, output_final_state=False)[0], qk_invariant=False
    )
