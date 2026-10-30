import pytest
import torch

pytest.importorskip("mamba_ssm", reason="requires the HF environment")

from tests.gpu.test_mamba import _mamba_inputs

# Key-axis (per state column) INT8 state on the Mamba-2 kernel: one step vs mamba_ssm's reference on the dequantized
# state. RTN exact codec: codes within one of the key-axis quantisation of the reference post-update state; SR / fast /
# asym: output and state error bounded.

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytestmark = requires_gpu


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("sr", [False, True])
def test_key_axis_matches_reference(asym, fast, sr):
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update_ref

    from lores.kernels.state_int.mamba2.selective_state_update_int import selective_state_update_int
    from lores.kernels.state_int.packing.pack_state_key_axis_naive import dequantize_key_axis, quantize_key_axis

    state, x, dt, A, Bm, C, D, z, dt_bias = _mamba_inputs()
    codes, scales = (t.contiguous() for t in quantize_key_axis(state, asym))
    st_ref = dequantize_key_axis(codes, scales, asym)
    out_ref = selective_state_update_ref(st_ref, x, dt, A, Bm, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    seed = torch.tensor([11, 22], dtype=torch.int32, device="cuda") if sr else None
    out = selective_state_update_int(
        codes, scales, x, dt, A, Bm, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True, sr_seed=seed, sr_salt=3, asym=asym,
        fast=fast, axis="key",
    )  # fmt: skip
    torch.testing.assert_close(out, out_ref, atol=2e-3, rtol=2e-3)
    st = dequantize_key_axis(codes, scales, asym)
    rel = (st - st_ref).norm() / st_ref.norm()
    assert rel < 1.5e-2, f"state rel err {rel:.4f}"
    if not sr:
        d = (codes.to(torch.int16) - quantize_key_axis(st_ref, asym)[0].to(torch.int16)).abs()
        assert d.max() <= 1, f"code diff > 1 (max {int(d.max())})"
        assert (d > 0).float().mean() < 1e-3, f"{int((d > 0).sum())} codes flipped"
