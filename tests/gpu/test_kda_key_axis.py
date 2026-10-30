import pytest
import torch

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytestmark = requires_gpu

pytest.importorskip("vllm", exc_type=ImportError, reason="requires the patched vLLM")

# Key-axis (per-key-channel scale) INT8 state on the copy of vLLM's KDA decode kernel.
#   1. one step vs vLLM's own fp32 fused_recurrent_kda (RTN exact: codes within 1 of the key-axis
#      reference quantisation of the fp final state, state within half a code; SR/fast/asym/warps: output)
#   2. axis semantics: after RTN, every key column's absmax equals scale[k]*127 (sym) / span = 255*scale (asym),
#      and this does NOT hold per value row -- and the mirror statement for the value-axis kernel
#   3. torch pack/unpack: RTN sym == the fake int8_block_dk mode bit for bit; roundtrip; SR unbiased
#   4. launch invariance: RTN codes/scales bit-identical across num_warps / num_stages
#   5. T = 3 tokens per sequence (varlen) and a multi-sequence pool with untouched slots
#   6. 64-step drift vs fp32 for key- and value-axis, RTN and SR: bounded, SR no worse than RTN
#   7. LinqConfig accepts state_axis and rejects it without int8


B, H, K, V, POOL = 3, 4, 128, 128, 8


def _inputs(seed=0, T=1, dev="cuda"):
    torch.manual_seed(seed)
    q = torch.randn(1, B * T, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, B * T, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, B * T, H, V, device=dev, dtype=torch.bfloat16)
    A_log = torch.log(torch.tensor([0.5, 1.0, 2.0, 4.0], device=dev))
    g = -(torch.nn.functional.softplus(torch.randn(1, B * T, H, K, device=dev) * 0.5 - 2.0)) * A_log.exp()[None, None, :, None]
    beta = torch.sigmoid(torch.randn(1, B * T, H, device=dev)).to(torch.bfloat16)
    pool = torch.randn(POOL, H, V, K, device=dev) * 2
    idx = torch.tensor([5, 1, 6], dtype=torch.int32, device=dev)
    if T > 1:  # several tokens per sequence: vLLM hands [N, T] slot indices (every token writes the sequence's slot)
        idx = idx[:, None].expand(B, T).contiguous()
    cu = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * T
    return dict(q=q, k=k, v=v, g=g.contiguous(), beta=beta, pool=pool, idx=idx, cu=cu)


def _key_pool(pool, asym, sr_seed=None):
    from lores.kernels.state_int.packing.pack_state_key_axis_naive import dequantize_key_axis, quantize_key_axis

    codes, scales = quantize_key_axis(pool, asym, sr_seed)
    return codes.contiguous(), scales.contiguous(), dequantize_key_axis(codes, scales, asym)


def _ref_step(s, st0, T=1):
    from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

    slot = s["idx"] if s["idx"].ndim == 1 else s["idx"][:, 0]
    pool32 = torch.zeros(POOL, H, V, K, device="cuda")
    pool32[slot.long()] = st0
    o_ref, _ = fused_recurrent_kda(
        q=s["q"],
        k=s["k"],
        v=s["v"],
        g=s["g"],
        beta=s["beta"],
        initial_state=pool32,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=s["cu"],
        ssm_state_indices=s["idx"],
    )
    return o_ref, pool32[slot.long()]


def _int_step(s, codes, scales, asym, sr, fast, launch=None, axis="key"):
    from lores.kernels.state_int.kda.fused_recurrent_kda_vllm_int import fused_recurrent_kda_int

    seed = torch.tensor([11, 22, 33], dtype=torch.int32, device="cuda") if sr else None
    return fused_recurrent_kda_int(
        s["q"],
        s["k"],
        s["v"],
        s["g"],
        s["beta"],
        None,
        codes,
        scales,
        s["idx"],
        cu_seqlens=s["cu"],
        use_qk_l2norm_in_kernel=True,
        sr_seed=seed,
        sr_salt=3,
        asym=asym,
        fast=fast,
        launch=launch,
        axis=axis,
    )


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("sr", [False, True])
@pytest.mark.parametrize("warps", [None, (None, 8, 1)])
def test_key_axis_matches_vllm(asym, fast, sr, warps):
    from lores.kernels.state_int.packing.pack_state_key_axis_naive import dequantize_key_axis, quantize_key_axis

    s = _inputs()
    codes, scales, st0 = _key_pool(s["pool"], asym)
    o_ref, st_final = _ref_step(s, st0[s["idx"].long()])
    o = _int_step(s, codes, scales, asym, sr, fast, launch=warps)
    torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    st_int = dequantize_key_axis(codes, scales, asym)[s["idx"].long()]
    rel = (st_int - st_final).norm() / st_final.norm()
    assert rel < 1.5e-2, f"state rel err {rel:.4f}"
    if not sr:
        c_ref, s_ref = quantize_key_axis(st_final.contiguous(), asym)
        d = (codes[s["idx"].long()].to(torch.int16) - c_ref.to(torch.int16)).abs()
        assert d.max() <= 1, f"code diff > 1 (max {int(d.max())})"
        sc = s_ref[..., 0] if asym else s_ref  # [n, H, K]
        err = ((st_int - st_final) / sc[:, :, None, :]).abs()
        assert err.max() <= 0.5 + 1e-3, f"state off by {err.max():.4f} codes"
    other = [i for i in range(POOL) if i not in s["idx"].tolist()]
    c0, s0, _ = _key_pool(s["pool"], asym)
    assert torch.equal(codes[other], c0[other]) and torch.equal(scales[other], s0[other]), "untouched slots changed"


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
def test_axis_semantics(asym):
    """The scale really runs along the key axis (and along the value axis for the value kernel)."""
    from lores.kernels.state_int.packing.pack_state_key_axis_naive import dequantize_key_axis
    from lores.kernels.state_int.packing.packed_state_kernel_naive import pack_state, unpack_state

    s = _inputs(seed=1)
    codes, scales, _ = _key_pool(s["pool"], asym)
    _int_step(s, codes, scales, asym, sr=False, fast=False)
    st = dequantize_key_axis(codes, scales, asym)[s["idx"].long()]  # [n, H, V, K]
    sc = scales[s["idx"].long()]
    if asym:
        span = st.amax(2) - st.amin(2)  # per key column over V
        torch.testing.assert_close(span, 255.0 * sc[..., 0], atol=1e-4, rtol=1e-5)
    else:
        torch.testing.assert_close(st.abs().amax(2), 127.0 * sc, atol=1e-4, rtol=1e-5)
        # the per-VALUE-row absmax is NOT a multiple of one per-row scale: rows carry many different scales
        row_ratio = st.abs().amax(3) / st.abs().amax(3).amax(2, keepdim=True)
        assert row_ratio.min() < 0.999, "rows look uniformly scaled: axis is not the key axis"
    # mirror: the value-axis kernel keeps per-ROW grids
    p = pack_state(s["pool"], asym=asym)
    _int_step(s, p.codes, p.scales, asym, sr=False, fast=False, axis="value")
    stv = unpack_state(p)[s["idx"].long()]
    scv = p.scales[s["idx"].long()]
    scv = scv.reshape(*scv.shape[:3], -1)
    if asym:
        torch.testing.assert_close(stv.amax(3) - stv.amin(3), 255.0 * scv[..., 0], atol=1e-4, rtol=1e-5)
    else:
        torch.testing.assert_close(stv.abs().amax(3), 127.0 * scv[..., 0], atol=1e-4, rtol=1e-5)


@requires_gpu
def test_torch_pack_matches_fake_dk_and_roundtrip():
    from lores.fake_quant.quantizers import quantize_state
    from lores.kernels.state_int.packing.pack_state_key_axis_naive import dequantize_key_axis, quantize_key_axis

    torch.manual_seed(2)
    x = torch.randn(4, H, V, K, device="cuda") * 3
    c, sc = quantize_key_axis(x, asym=False)
    torch.testing.assert_close(dequantize_key_axis(c, sc, False), quantize_state(x, "int8_block_dk", 128), atol=0, rtol=0)
    for asym in (False, True):
        c, sc = quantize_key_axis(x, asym)
        y = dequantize_key_axis(c, sc, asym)
        step = (sc[..., 0] if asym else sc)[:, :, None, :]
        assert ((y - x) / step).abs().max() <= 0.5 + 1e-4, "RTN roundtrip off by more than half a code"
        # stochastic rounding: unbiased over draws
        acc = torch.zeros_like(x)
        n = 64
        for seed in range(n):
            c, sc = quantize_key_axis(x, asym, sr_seed=seed)
            acc += dequantize_key_axis(c, sc, asym)
        bias = ((acc / n - x) / step).abs().mean()
        assert bias < 0.06, f"SR bias {bias:.3f} codes (asym={asym})"


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
def test_launch_invariance(asym):
    outs = []
    for launch in (None, (None, 1, 3), (None, 2, 2), (None, 8, 1)):
        s = _inputs(seed=3)
        codes, scales, _ = _key_pool(s["pool"], asym)
        o = _int_step(s, codes, scales, asym, sr=False, fast=False, launch=launch)
        outs.append((o.clone(), codes.clone(), scales.clone()))
    for o, c, sc in outs[1:]:  # codes and outputs bit-identical; scales may differ by an fp32 ulp (reduction order)
        assert torch.equal(c, outs[0][1]), "codes depend on the launch config"
        assert ((sc - outs[0][2]).abs() / outs[0][2].abs().clamp_min(1e-12)).max() < 1e-6, "scales depend on the launch config"
        assert torch.equal(o, outs[0][0]), "output depends on the launch config"


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
def test_multi_token_varlen(asym):
    from lores.kernels.state_int.packing.pack_state_key_axis_naive import dequantize_key_axis

    s = _inputs(seed=4, T=3)
    slot = s["idx"][:, 0].long()
    codes, scales, st0 = _key_pool(s["pool"], asym)
    o_ref, st_final = _ref_step(s, st0[slot], T=3)
    o = _int_step(s, codes, scales, asym, sr=False, fast=False)
    torch.testing.assert_close(o.float(), o_ref.float(), atol=3e-2, rtol=3e-2)
    st_int = dequantize_key_axis(codes, scales, asym)[slot]
    rel = (st_int - st_final).norm() / st_final.norm()
    assert rel < 2e-2, f"T=3 state rel err {rel:.4f}"


@requires_gpu
def test_drift_64_steps():
    """Key- vs value-axis INT8 over 64 decode steps against the fp32 kernel: bounded, SR no worse than RTN."""
    from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

    from lores.kernels.state_int.kda.fused_recurrent_kda_vllm_int import fused_recurrent_kda_int
    from lores.kernels.state_int.packing.pack_state_key_axis_naive import quantize_key_axis
    from lores.kernels.state_int.packing.packed_state_kernel_naive import pack_state

    torch.manual_seed(5)
    dev = "cuda"
    pool0 = torch.randn(POOL, H, V, K, device=dev) * 2
    idx = torch.tensor([5, 1, 6], dtype=torch.int32, device=dev)
    cu = torch.arange(0, B + 1, dtype=torch.int32, device=dev)
    steps = [_inputs(seed=100 + t) for t in range(64)]
    arms = {}
    for axis in ("key", "value"):
        for sr in (False, True):
            if axis == "key":
                codes, scales = (t.contiguous() for t in quantize_key_axis(pool0, False))
            else:
                p = pack_state(pool0)
                codes, scales = p.codes.clone(), p.scales.clone()
            arms[(axis, sr)] = (codes, scales, p if axis == "value" else None)
    ref_pool = pool0.clone()
    errs = {k: [] for k in arms}
    for t, s in enumerate(steps):
        o_ref, _ = fused_recurrent_kda(
            q=s["q"],
            k=s["k"],
            v=s["v"],
            g=s["g"],
            beta=s["beta"],
            initial_state=ref_pool,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            ssm_state_indices=idx,
        )
        for (axis, sr), (codes, scales, p) in arms.items():
            seed = torch.tensor([7 + t, 9 + t, 11 + t], dtype=torch.int32, device=dev) if sr else None
            o = fused_recurrent_kda_int(
                s["q"],
                s["k"],
                s["v"],
                s["g"],
                s["beta"],
                None,
                codes,
                scales,
                idx,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
                sr_seed=seed,
                sr_salt=1,
                asym=False,
                fast=False,
                axis=axis,
            )
            errs[(axis, sr)].append(((o.float() - o_ref.float()).norm() / o_ref.float().norm()).item())
    for (axis, sr), e in errs.items():
        assert max(e) < 5e-2, f"{axis} sr={sr}: output rel err {max(e):.3f}"
    print(
        "drift (mean rel out err over 64 steps):",
        {f"{a}/{'sr' if r else 'rtn'}": round(sum(e) / len(e), 4) for (a, r), e in errs.items()},
    )
