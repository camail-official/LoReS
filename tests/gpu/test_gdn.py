import pytest
import torch

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytestmark = requires_gpu

pytest.importorskip("vllm", exc_type=ImportError, reason="requires the patched vLLM")

# fused_sigmoid_gating_delta_rule_update_int (copy of vLLM's qwen decode kernel with the packed
# int8 state) vs fla's fused_recurrent_gated_delta_rule on the same v-first state: one decode
# step on a slot pool, in-kernel gate/beta, RTN exact codec must reproduce pack_state of the
# reference final state to within one code; SR / fast / asym variants must match the output.


def _setup(asym, seed=0):
    from lores.kernels.state_int.packing.packed_state_kernel_naive import pack_state, unpack_state

    torch.manual_seed(seed)
    dev = "cuda"
    B, H, HV, K, V, POOL = 3, 2, 4, 128, 128, 8
    q = torch.randn(1, B, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, B, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, B, HV, V, device=dev, dtype=torch.bfloat16)
    a = (torch.randn(1, B, HV, device=dev) * 0.5).to(torch.bfloat16)
    b = torch.randn(1, B, HV, device=dev).to(torch.bfloat16)
    A_log = torch.log(torch.tensor([1.0, 4.0, 8.0, 16.0], device=dev))
    dt_bias = torch.full((HV,), float(torch.log(torch.expm1(torch.tensor(0.01)))), device=dev)
    pool = torch.randn(POOL, HV, V, K, device=dev) * 2
    p = pack_state(pool, asym=asym)
    idx = torch.tensor([5, 1, 6], dtype=torch.int32, device=dev)
    st0 = unpack_state(p)[idx.long()]
    g = -A_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    beta = torch.sigmoid(b.float()).to(torch.bfloat16)
    return dict(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, p=p, idx=idx, st0=st0, g=g, beta=beta, B=B, K=K)


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("sr", [False, True])
def test_vllm_int_kernel_matches_reference(asym, fast, sr):
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    from lores.kernels.state_int.gdn.fused_sigmoid_gating_int import fused_sigmoid_gating_delta_rule_update_int
    from lores.kernels.state_int.packing.packed_state_kernel_naive import pack_state, unpack_state

    s = _setup(asym)
    cu = torch.arange(0, s["B"] + 1, dtype=torch.int32, device="cuda")
    # reference 1: vLLM's own fp32 kernel (the copied code path) on an fp32 slot pool
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

    pool32 = torch.zeros(8, *s["st0"].shape[1:], device="cuda")
    pool32[s["idx"].long()] = s["st0"]
    o_ref, _ = fused_sigmoid_gating_delta_rule_update(
        s["A_log"],
        s["a"],
        s["b"],
        s["dt_bias"],
        s["q"],
        s["k"],
        s["v"],
        initial_state=pool32,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=s["idx"],
        use_qk_l2norm_in_kernel=True,
    )
    o_ref = o_ref.transpose(0, 1)
    st_final = pool32[s["idx"].long()]
    # reference 2: fla (different exp/log ops) -- the output must still agree
    tr = lambda x: x.transpose(0, 1).contiguous()  # noqa: E731
    o_fla, _ = fused_recurrent_gated_delta_rule(
        tr(s["q"]),
        tr(s["k"]),
        tr(s["v"]),
        g=tr(s["g"]),
        beta=tr(s["beta"]),
        initial_state=s["st0"].clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        state_v_first=True,
    )
    torch.testing.assert_close(o_ref.float(), o_fla.float(), atol=2e-2, rtol=2e-2)
    seed = torch.tensor([11, 22, 33], dtype=torch.int32, device="cuda") if sr else None
    o = fused_sigmoid_gating_delta_rule_update_int(
        s["A_log"],
        s["a"],
        s["b"],
        s["dt_bias"],
        s["q"],
        s["k"],
        s["v"],
        s["p"].codes,
        s["p"].scales,
        s["idx"],
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        sr_seed=seed,
        sr_salt=3,
        asym=asym,
        fast=fast,
    )
    torch.testing.assert_close(o.transpose(0, 1).float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    st_int = unpack_state(s["p"])[s["idx"].long()]
    rel = (st_int - st_final).norm() / st_final.norm()
    assert rel < 1.5e-2, f"state rel err {rel:.4f}"  # int8 grid: ~0.4% RTN, ~0.7% SR
    if not sr:
        # RTN: the kernel's pre-quantisation state equals the reference within fp32 noise, so the
        # dequantised state must be within half a code of the reference everywhere (exact
        # rounding ties may fall either way, hence the 1e-3 slack).
        p_ref = pack_state(st_final, asym=asym)
        d = (s["p"].codes[s["idx"].long()].to(torch.int16) - p_ref.codes.to(torch.int16)).abs()
        assert d.max() <= 1, f"code diff > 1 (max {int(d.max())})"
        sc = p_ref.scales.reshape(*p_ref.scales.shape[:3], -1)[..., :1]
        err = ((st_int - st_final) / sc).abs()
        i = torch.unravel_index(err.argmax(), err.shape)
        assert err.max() <= 0.5 + 1e-3, (
            f"state off by {err.max():.4f} codes at {tuple(int(x) for x in i)}: int {st_int[i].item():.6f} "
            f"ref {st_final[i].item():.6f} scale {sc[i[:3]].item():.6e}"
        )
    # untouched slots stay untouched
    other = [i for i in range(8) if i not in s["idx"].tolist()]
    assert torch.equal(s["p"].codes[other], pack_state(unpack_state(s["p"])[other], asym=asym).codes)


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("sr", [False, True])
def test_vllm_packed_int_kernel_matches_reference(asym, fast, sr):
    """The copy of vLLM's PACKED decode-only kernel vs vLLM's own fp32 packed kernel (same [q|k|v] buffer)."""
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode

    from lores.kernels.state_int.gdn.fused_sigmoid_gating_int import fused_recurrent_gated_delta_rule_packed_decode_int
    from lores.kernels.state_int.packing.packed_state_kernel_naive import pack_state, unpack_state

    s = _setup(asym)
    B, HV, V = s["B"], s["v"].shape[2], s["v"].shape[3]
    mixed = torch.cat([s["q"][0].reshape(B, -1), s["k"][0].reshape(B, -1), s["v"][0].reshape(B, -1)], dim=1).contiguous()
    a2, b2 = s["a"][0].contiguous(), s["b"][0].contiguous()
    pool32 = torch.zeros(8, *s["st0"].shape[1:], device="cuda")
    pool32[s["idx"].long()] = s["st0"]
    o_ref = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed, a2, b2, s["A_log"], s["dt_bias"], s["K"] ** -0.5, pool32, o_ref, s["idx"], use_qk_l2norm_in_kernel=True
    )
    st_final = pool32[s["idx"].long()]
    seed = torch.tensor([11, 22, 33], dtype=torch.int32, device="cuda") if sr else None
    o = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    fused_recurrent_gated_delta_rule_packed_decode_int(
        mixed,
        a2,
        b2,
        s["A_log"],
        s["dt_bias"],
        s["K"] ** -0.5,
        s["p"].codes,
        s["p"].scales,
        o,
        s["idx"],
        use_qk_l2norm_in_kernel=True,
        sr_seed=seed,
        sr_salt=3,
        asym=asym,
        fast=fast,
    )
    torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    st_int = unpack_state(s["p"])[s["idx"].long()]
    rel = (st_int - st_final).norm() / st_final.norm()
    assert rel < 1.5e-2, f"state rel err {rel:.4f}"
    if not sr:
        p_ref = pack_state(st_final, asym=asym)
        sc = p_ref.scales.reshape(*p_ref.scales.shape[:3], -1)[..., :1]
        err = ((st_int - st_final) / sc).abs()
        assert err.max() <= 0.5 + 1e-3, f"state off by {err.max():.4f} codes"
    other = [i for i in range(8) if i not in s["idx"].tolist()]
    assert torch.equal(s["p"].codes[other], pack_state(unpack_state(s["p"])[other], asym=asym).codes)
