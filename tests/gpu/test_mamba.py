import math

import pytest
import torch

pytest.importorskip("mamba_ssm", reason="requires the HF environment")

from lores.kernels.state_int.packing.packed_state_kernel_naive import pack_state, unpack_state

# state_int kernel parity: the int8-state decode kernel must reproduce the hook-path semantics —
# dequant(codes) == unpack_state (bit), one update step == mamba_ssm reference on the dequantized
# state, and the stored codes+scales == pack_state of the reference's post-update state (bit).


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytestmark = requires_gpu


def _mamba_inputs(B=2, H=8, DIM=64, N=128, tie_hdim=True, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    state = torch.randn(B, H, DIM, N, device=dev, dtype=torch.float32) * 2
    x = torch.randn(B, H, DIM, device=dev, dtype=torch.float32)
    if tie_hdim:
        # per-head scalar dt/A broadcast as zero-stride views, the shapes the real models pass
        dt = torch.randn(B, H, device=dev).unsqueeze(-1).expand(B, H, DIM)
        A = (-torch.rand(H, device=dev)).view(H, 1, 1).expand(H, DIM, N)
        dt_bias = torch.randn(H, device=dev).unsqueeze(-1).expand(H, DIM)
    else:
        dt = torch.randn(B, H, DIM, device=dev)
        A = -torch.rand(H, DIM, N, device=dev)
        dt_bias = torch.randn(H, DIM, device=dev)
    Bm = torch.randn(B, 1, N, device=dev)
    C = torch.randn(B, 1, N, device=dev)
    D = torch.randn(H, DIM, device=dev)
    z = torch.randn(B, H, DIM, device=dev)
    return state, x, dt, A, Bm, C, D, z, dt_bias


def _codes_within_one_step(p, codes, scales, p_ref, step):
    """fp32 op-order (fma/exp) may flip a code sitting exactly on a rounding boundary by ONE step, for a
    vanishing fraction; anything more = wiring bug. Compared in value space in units of ``step``; 0.25 step
    separates a real flip from the fp32 noise in the scale itself (<= 127e-6).
    """
    dval = (unpack_state(p._replace(codes=codes, scales=scales)) - unpack_state(p_ref)).abs() / step
    n_flip = int((dval > 0.25).sum())
    assert dval.max() <= 1.0 + 1e-3, f"value diff > 1 code step ({float(dval.max()):.3f}) — wiring bug"
    assert n_flip <= max(2, dval.numel() // 10_000), f"{n_flip}/{dval.numel()} codes flipped"


@requires_gpu
@pytest.mark.parametrize("tie_hdim", [True, False])
def test_selective_state_update_int_matches_reference(tie_hdim):
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update_ref

    from lores.kernels.state_int.mamba2.selective_state_update_int import selective_state_update_int

    state, x, dt, A, Bm, C, D, z, dt_bias = _mamba_inputs(tie_hdim=tie_hdim)
    p = pack_state(state)
    codes, scales = p.codes.clone(), p.scales.clone()
    # reference: hook-path semantics — dequantized state, fp update, quantize-on-store
    st_ref = unpack_state(p).clone()
    out_ref = selective_state_update_ref(st_ref, x, dt, A, Bm, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    p_ref = pack_state(st_ref)  # st_ref updated in place by the reference
    out = selective_state_update_int(codes, scales, x, dt, A, Bm, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    torch.testing.assert_close(out, out_ref, atol=2e-4, rtol=2e-4)  # fp32 op-order only
    torch.testing.assert_close(scales, p_ref.scales, atol=0, rtol=1e-6)
    _codes_within_one_step(p, codes, scales, p_ref, scales[..., None])


@requires_gpu
@pytest.mark.parametrize("tie_hdim", [True, False])
def test_selective_state_update_int_asym_matches_reference(tie_hdim):
    """INT8 affine grid (quantizers.quant_intn_block_asym): scale + zero-point per row."""
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update_ref

    from lores.kernels.state_int.mamba2.selective_state_update_int import selective_state_update_int

    state, x, dt, A, Bm, C, D, z, dt_bias = _mamba_inputs(tie_hdim=tie_hdim)
    state = state + 0.7  # nonzero row means: where the affine grid differs from the symmetric one
    p = pack_state(state, asym=True)
    codes, scales = p.codes.clone(), p.scales.clone()
    st_ref = unpack_state(p).clone()
    out_ref = selective_state_update_ref(st_ref, x, dt, A, Bm, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    p_ref = pack_state(st_ref, asym=True)
    out = selective_state_update_int(codes, scales, x, dt, A, Bm, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True, asym=True)
    torch.testing.assert_close(out, out_ref, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(scales[..., 0], p_ref.scales[..., 0], atol=0, rtol=1e-6)
    torch.testing.assert_close(scales[..., 1], p_ref.scales[..., 1], atol=1e-6, rtol=1e-6)
    _codes_within_one_step(p, codes, scales, p_ref, scales[..., 0:1])


@requires_gpu
def test_pack_state_is_the_fake_int8_block():
    """pack_state / unpack_state == the fake modes int8_block and int8_block_asym, bit for bit."""
    from lores.fake_quant.quantizers import quantize_state

    state, *_ = _mamba_inputs()
    torch.testing.assert_close(unpack_state(pack_state(state)), quantize_state(state, "int8_block", 128), atol=0, rtol=0)
    torch.testing.assert_close(
        unpack_state(pack_state(state, asym=True)), quantize_state(state, "int8_block_asym", 128), atol=0, rtol=0
    )


T = 300


def _stall(final, init, exact):
    """Signed deficit toward init, relative to the exact expectation: 0 = unbiased, 1 = frozen."""
    return ((final - exact) * init.sign()).mean().item() / ((init - exact) * init.sign()).mean().item()


def _seed_for(mode, seed):
    """Per-row int32 seeds: advanced every step ("sr", what vLLM's seq_lens do) or constant."""
    if mode == "rtn":
        return None
    if mode == "sr":
        seed.add_(1)
    return seed  # "fixed": the same seed every step


def _run_mamba(mode, asym=False):
    from lores.kernels.state_int.mamba2.selective_state_update_int import selective_state_update_int

    dev = "cuda"
    B, H, DIM, N = 8, 16, 64, 128
    torch.manual_seed(0)
    p = pack_state(torch.randn(B, H, DIM, N, device=dev) * 2 + 0.7, asym=asym)
    codes, scales = p.codes.contiguous(), p.scales.contiguous()
    init = unpack_state(p)
    x = torch.zeros(B, H, DIM, device=dev, dtype=torch.bfloat16)
    dt = torch.full((B, H, DIM), 0.01, device=dev)
    A = torch.zeros(H, DIM, N, device=dev)
    A[..., 1:5] = -1.0  # columns 1..4 decay 0.99/step against the static row scale
    Bm = torch.zeros(B, 1, N, device=dev, dtype=torch.bfloat16)
    C = torch.randn(B, 1, N, device=dev, dtype=torch.bfloat16)
    idx = torch.arange(B, device=dev, dtype=torch.int32)
    seed = torch.arange(7, 7 + B, dtype=torch.int32, device=dev)
    for _ in range(T):
        selective_state_update_int(
            codes, scales, x, dt, A, Bm, C, state_batch_indices=idx, sr_seed=_seed_for(mode, seed), asym=asym
        )
    final = unpack_state(p._replace(codes=codes, scales=scales))
    return final[..., 1:5], init[..., 1:5], init[..., 1:5] * math.exp(-0.01 * T)


@pytest.mark.parametrize("asym", [False, True])
def test_sr_unbiased_under_decay(asym):
    s_rtn = _stall(*_run_mamba("rtn", asym))
    s_sr = _stall(*_run_mamba("sr", asym))
    s_fix = _stall(*_run_mamba("fixed", asym))
    assert s_rtn > 0.3, f"RTN should stall: {s_rtn:.3f}"
    assert abs(s_sr) < 0.03, f"SR biased: {s_sr:.3f}"
    assert s_fix > 0.05, f"constant seed should behave like a fixed dither: {s_fix:.3f}"


def test_sr_salt_decorrelates_layers():
    """Same state, seed and slots but a different per-layer salt must give different SR draws."""
    from lores.kernels.state_int.mamba2.selective_state_update_int import selective_state_update_int

    dev = "cuda"
    B, H, DIM, N = 2, 4, 64, 128
    torch.manual_seed(0)
    p = pack_state(torch.randn(B, H, DIM, N, device=dev) * 2)
    x = torch.randn(B, H, DIM, device=dev, dtype=torch.bfloat16) * 0.3
    dt = torch.full((B, H, DIM), 0.05, device=dev)
    A = torch.full((H, DIM, N), -1.0, device=dev)
    Bm = torch.randn(B, 1, N, device=dev, dtype=torch.bfloat16)
    C = torch.randn(B, 1, N, device=dev, dtype=torch.bfloat16)
    idx = torch.arange(B, device=dev, dtype=torch.int32)
    seed = torch.arange(7, 7 + B, dtype=torch.int32, device=dev)
    outs = []
    for salt in (0, 1, 2):
        codes, scales = p.codes.clone(), p.scales.clone()
        selective_state_update_int(codes, scales, x, dt, A, Bm, C, state_batch_indices=idx, sr_seed=seed, sr_salt=salt)
        outs.append(codes.clone())
    frac01 = (outs[0] != outs[1]).float().mean().item()
    frac12 = (outs[1] != outs[2]).float().mean().item()
    assert 0.05 < frac01 < 0.95 and 0.05 < frac12 < 0.95, (frac01, frac12)  # rounding directions differ per salt
