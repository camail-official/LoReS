"""Per-head Nemotron decay reduces expanded kernel inputs correctly."""

import pytest
import torch
import torch.nn.functional as F

from scripts.calibration.decay_hybrid import nemo_step_logdecay

NEMO_H, NEMO_P, NEMO_N = 8, 4, 6


def _call_site_tensors(dt_h, bias_h, a_h):
    """Expand per-head values exactly like modeling_nemotron_h's decode path."""
    dt = dt_h[None, :, None].expand(1, NEMO_H, NEMO_P)  # [B, NEMO_H, NEMO_P]
    dt_bias = bias_h[:, None].expand(NEMO_H, NEMO_P)  # [NEMO_H, NEMO_P]
    A = a_h[:, None, None].expand(NEMO_H, NEMO_P, NEMO_N)  # [NEMO_H, NEMO_P, NEMO_N]
    return dt, dt_bias, A


def test_per_head_bias_applied_exactly():
    torch.manual_seed(0)
    dt_h = torch.randn(NEMO_H)
    bias_h = torch.linspace(-15.0, 18.0, NEMO_H)  # realistic spread; distinct per head
    a_h = -torch.rand(NEMO_H) - 0.1
    dt, dt_bias, A = _call_site_tensors(dt_h, bias_h, a_h)

    got = nemo_step_logdecay(dt, A, dt_bias, dt_softplus=True)
    want = F.softplus(dt_h + bias_h) * a_h
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def test_flat_expanded_bias_refused():
    torch.manual_seed(3)
    dt_h, bias_h, a_h = torch.randn(NEMO_H), torch.arange(NEMO_H, dtype=torch.float32), -torch.ones(NEMO_H)
    dt, dt_bias, A = _call_site_tensors(dt_h, bias_h, a_h)
    with pytest.raises(AssertionError, match="one bias per head"):
        nemo_step_logdecay(dt, A, dt_bias.reshape(-1), dt_softplus=True)  # [NEMO_H*NEMO_P] layout


def test_unexpanded_bias_and_no_softplus():
    torch.manual_seed(2)
    dt_h, bias_h, a_h = torch.randn(NEMO_H), torch.randn(NEMO_H), -torch.rand(NEMO_H)
    dt, _, A = _call_site_tensors(dt_h, bias_h, a_h)

    got = nemo_step_logdecay(dt, A, bias_h, dt_softplus=False)  # bias as plain [NEMO_H]
    torch.testing.assert_close(got, (dt_h + bias_h) * a_h, rtol=0, atol=1e-6)
