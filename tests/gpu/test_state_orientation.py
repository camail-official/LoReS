"""The recurrent-state layouts the fake quantizers rely on: HF/fla [B,H,K,V], vLLM [N,HV,V,K]."""

import pytest
import torch

H, K, V, A, B = 2, 128, 128, 5, 70


def _one_hot_probe(kernel, **extra):
    """State with a single 1 at [a, b]; q = e_b lights output dim a iff the layout is [V, K]."""
    st = torch.zeros(1, H, 128, 128, device="cuda")
    st[0, 0, A, B] = 1.0
    q = torch.zeros(1, 1, H, K, device="cuda")
    q[..., B] = 1.0
    z = lambda *shape: torch.zeros(*shape, device="cuda")  # noqa: E731
    zeros = dict(k=z(1, 1, H, K), v=z(1, 1, H, V), beta=z(1, 1, H))
    o, _ = kernel(q=q, initial_state=st, use_qk_l2norm_in_kernel=False, **zeros, **extra)
    lit = (o.reshape(H, V)[0].abs() > 1e-6).nonzero().flatten().tolist()
    return "VK" if lit == [A] else "KV" if lit == [] else lit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_fla_state_is_k_major():
    from fla.ops.kda import fused_recurrent_kda

    assert _one_hot_probe(fused_recurrent_kda, g=torch.zeros(1, 1, H, K, device="cuda")) == "KV"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_vllm_state_is_v_major():
    pytest.importorskip("vllm")
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

    g_head, g_chan = torch.zeros(1, 1, H, device="cuda"), torch.zeros(1, 1, H, K, device="cuda")
    assert _one_hot_probe(fused_recurrent_gated_delta_rule, g=g_head, inplace_final_state=False) == "VK"
    assert _one_hot_probe(fused_recurrent_kda, g=g_chan, inplace_final_state=False) == "VK"
