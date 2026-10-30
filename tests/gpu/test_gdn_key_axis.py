import pytest
import torch

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytestmark = requires_gpu

pytest.importorskip("vllm", exc_type=ImportError, reason="requires the patched vLLM")

# Key-axis (per key channel) INT8 state on both GDN kernel copies (Qwen3.5 decode): one step vs vLLM's own fp32
# kernel on an fp32 slot pool. RTN exact codec: codes within one of the key-axis quantisation of the reference final
# state; SR / fast / asym: output and state error bounded; untouched slots stay untouched.

B, H, HV, K, V, POOL = 3, 2, 4, 128, 128, 8


def _setup(asym, seed=0):
    from lores.kernels.state_int.packing.pack_state_key_axis_naive import dequantize_key_axis, quantize_key_axis

    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(1, B, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, B, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, B, HV, V, device=dev, dtype=torch.bfloat16)
    a = (torch.randn(1, B, HV, device=dev) * 0.5).to(torch.bfloat16)
    b = torch.randn(1, B, HV, device=dev).to(torch.bfloat16)
    A_log = torch.log(torch.tensor([1.0, 4.0, 8.0, 16.0], device=dev))
    dt_bias = torch.full((HV,), float(torch.log(torch.expm1(torch.tensor(0.01)))), device=dev)
    pool = torch.randn(POOL, HV, V, K, device=dev) * 2
    codes, scales = (t.contiguous() for t in quantize_key_axis(pool, asym))
    deq = lambda cd, sc: dequantize_key_axis(cd, sc, asym)  # noqa: E731
    idx = torch.tensor([5, 1, 6], dtype=torch.int32, device=dev)
    return dict(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, codes=codes, scales=scales, deq=deq, idx=idx)


def _check(s, asym, sr, st_final, before):
    from lores.kernels.state_int.packing.pack_state_key_axis_naive import quantize_key_axis

    i = s["idx"].long()
    st_int = s["deq"](s["codes"][i], s["scales"][i])
    rel = (st_int - st_final).norm() / st_final.norm()
    assert rel < 1.5e-2, f"state rel err {rel:.4f}"
    if not sr:
        d = (s["codes"][i].to(torch.int16) - quantize_key_axis(st_final, asym)[0].to(torch.int16)).abs()
        assert d.max() <= 1, f"code diff > 1 (max {int(d.max())})"
    other = [j for j in range(POOL) if j not in s["idx"].tolist()]
    assert torch.equal(s["codes"][other], before[other])


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("sr", [False, True])
def test_key_axis_fused_sigmoid_matches_vllm(asym, fast, sr):
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

    from lores.kernels.state_int.gdn.fused_sigmoid_gating_int import fused_sigmoid_gating_delta_rule_update_int

    s = _setup(asym)
    cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
    pool32 = torch.zeros(POOL, HV, V, K, device="cuda")
    pool32[s["idx"].long()] = s["deq"](s["codes"], s["scales"])[s["idx"].long()]
    o_ref, _ = fused_sigmoid_gating_delta_rule_update(
        s["A_log"], s["a"], s["b"], s["dt_bias"], s["q"], s["k"], s["v"], initial_state=pool32, inplace_final_state=True,
        cu_seqlens=cu, ssm_state_indices=s["idx"], use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    before = s["codes"].clone()
    seed = torch.tensor([11, 22, 33], dtype=torch.int32, device="cuda") if sr else None
    o = fused_sigmoid_gating_delta_rule_update_int(
        s["A_log"], s["a"], s["b"], s["dt_bias"], s["q"], s["k"], s["v"], s["codes"], s["scales"], s["idx"],
        cu_seqlens=cu, use_qk_l2norm_in_kernel=True, sr_seed=seed, sr_salt=3, asym=asym, fast=fast, axis="key",
    )  # fmt: skip
    torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    _check(s, asym, sr, pool32[s["idx"].long()], before)


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("sr", [False, True])
def test_key_axis_packed_decode_matches_vllm(asym, fast, sr):
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode

    from lores.kernels.state_int.gdn.fused_sigmoid_gating_int import fused_recurrent_gated_delta_rule_packed_decode_int

    s = _setup(asym)
    mixed = torch.cat([s["q"][0].reshape(B, -1), s["k"][0].reshape(B, -1), s["v"][0].reshape(B, -1)], dim=1).contiguous()
    a2, b2 = s["a"][0].contiguous(), s["b"][0].contiguous()
    pool32 = torch.zeros(POOL, HV, V, K, device="cuda")
    pool32[s["idx"].long()] = s["deq"](s["codes"], s["scales"])[s["idx"].long()]
    o_ref = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed, a2, b2, s["A_log"], s["dt_bias"], K**-0.5, pool32, o_ref, s["idx"], use_qk_l2norm_in_kernel=True
    )
    before = s["codes"].clone()
    seed = torch.tensor([11, 22, 33], dtype=torch.int32, device="cuda") if sr else None
    o = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    fused_recurrent_gated_delta_rule_packed_decode_int(
        mixed, a2, b2, s["A_log"], s["dt_bias"], K**-0.5, s["codes"], s["scales"], o, s["idx"],
        use_qk_l2norm_in_kernel=True, sr_seed=seed, sr_salt=3, asym=asym, fast=fast, axis="key",
    )  # fmt: skip
    torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    _check(s, asym, sr, pool32[s["idx"].long()], before)
