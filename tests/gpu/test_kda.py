import pytest
import torch

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytestmark = requires_gpu

pytest.importorskip("vllm", exc_type=ImportError, reason="requires the patched vLLM")

# fused_recurrent_kda_int (copy of vLLM's Kimi-Linear / KDA decode kernel with the packed int8
# state) vs vLLM's own fused_recurrent_kda on an fp32 slot pool: one decode step, per-channel KDA
# gate, RTN exact codec must reproduce pack_state of the reference final state to within half a
# code; SR / fast / asym / launch variants must match the output.


def _setup(asym, seed=0):
    from lores.kernels.state_int.packing.packed_state_kernel_naive import pack_state, unpack_state

    torch.manual_seed(seed)
    dev = "cuda"
    B, H, K, V, POOL = 3, 4, 128, 128, 8  # KDA: H == HV, K == V (Kimi-Linear-48B: 32 heads, 128)
    q = torch.randn(1, B, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, B, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, B, H, V, device=dev, dtype=torch.bfloat16)
    # per-(token, head, key channel) log gate, as fused_kda_gate produces it: -softplus(.) * exp(A_log)
    A_log = torch.log(torch.tensor([0.5, 1.0, 2.0, 4.0], device=dev))
    g = -(torch.nn.functional.softplus(torch.randn(1, B, H, K, device=dev) * 0.5 - 2.0)) * A_log.exp()[None, None, :, None]
    beta = torch.sigmoid(torch.randn(1, B, H, device=dev)).to(torch.bfloat16)
    pool = torch.randn(POOL, H, V, K, device=dev) * 2
    p = pack_state(pool, asym=asym)
    idx = torch.tensor([5, 1, 6], dtype=torch.int32, device=dev)
    st0 = unpack_state(p)[idx.long()]
    return dict(q=q, k=k, v=v, g=g.contiguous(), beta=beta, p=p, idx=idx, st0=st0, B=B, K=K)


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("sr", [False, True])
@pytest.mark.parametrize("launch", [None, (32, 2, 2)])
def test_kda_int_kernel_matches_vllm(asym, fast, sr, launch):
    from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

    from lores.kernels.state_int.kda.fused_recurrent_kda_vllm_int import fused_recurrent_kda_int
    from lores.kernels.state_int.packing.packed_state_kernel_naive import pack_state, unpack_state

    s = _setup(asym)
    cu = torch.arange(0, s["B"] + 1, dtype=torch.int32, device="cuda")
    # reference: vLLM's own fp32 kernel (the copied code path) on an fp32 slot pool
    pool32 = torch.zeros(8, *s["st0"].shape[1:], device="cuda")
    pool32[s["idx"].long()] = s["st0"]
    o_ref, _ = fused_recurrent_kda(
        q=s["q"],
        k=s["k"],
        v=s["v"],
        g=s["g"],
        beta=s["beta"],
        initial_state=pool32,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        ssm_state_indices=s["idx"],
    )
    st_final = pool32[s["idx"].long()]
    seed = torch.tensor([11, 22, 33], dtype=torch.int32, device="cuda") if sr else None
    o = fused_recurrent_kda_int(
        s["q"],
        s["k"],
        s["v"],
        s["g"],
        s["beta"],
        None,
        s["p"].codes,
        s["p"].scales,
        s["idx"],
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        sr_seed=seed,
        sr_salt=3,
        asym=asym,
        fast=fast,
        launch=launch,
    )
    torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    st_int = unpack_state(s["p"])[s["idx"].long()]
    rel = (st_int - st_final).norm() / st_final.norm()
    assert rel < 1.5e-2, f"state rel err {rel:.4f}"
    if not sr:
        p_ref = pack_state(st_final, asym=asym)
        d = (s["p"].codes[s["idx"].long()].to(torch.int16) - p_ref.codes.to(torch.int16)).abs()
        assert d.max() <= 1, f"code diff > 1 (max {int(d.max())})"
        sc = p_ref.scales.reshape(*p_ref.scales.shape[:3], -1)[..., :1]
        err = ((st_int - st_final) / sc).abs()
        assert err.max() <= 0.5 + 1e-3, f"state off by {err.max():.4f} codes"
    other = [i for i in range(8) if i not in s["idx"].tolist()]
    assert torch.equal(s["p"].codes[other], pack_state(unpack_state(s["p"])[other], asym=asym).codes)
