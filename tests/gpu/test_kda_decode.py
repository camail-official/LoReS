import itertools

import pytest
import torch

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytestmark = requires_gpu

pytest.importorskip("vllm", exc_type=ImportError, reason="requires the patched vLLM")

# fused_recurrent_kda_decode_int (decode-only KDA int8 kernel) must be BIT-identical to the general
# fused_recurrent_kda_int copy it replaces for one-token steps: same o, codes, scales for every
# codec (sym/asym x RTN/SR x fast/exact), both axes, both call conventions, null slot skipped.


def _pools(axis, asym, NS, HV, V, K, dev):
    h = torch.randn(NS, HV, V, K, device=dev) * 2
    if axis == "key":
        from lores.kernels.state_int.packing.pack_state_key_axis_naive import quantize_key_axis

        codes, scales = quantize_key_axis(h, asym)
        return codes.contiguous(), scales.contiguous()
    from lores.kernels.state_int.packing.packed_state_kernel_naive import pack_state

    p = pack_state(h, asym=asym)
    return p.codes.contiguous(), p.scales.contiguous()


def _run(axis, asym, sr, fast, launch, varlen, seed=0):
    from lores.kernels.state_int.kda.fused_recurrent_kda_vllm_int import (
        fused_recurrent_kda_decode_int,
        fused_recurrent_kda_int,
    )

    torch.manual_seed(seed)
    dev = "cuda"
    B, H, K, V, NS = 5, 4, 128, 128, 9
    idx = torch.tensor([3, 0, 7, 1, 5], dtype=torch.int32, device=dev)  # slot 0 = null block
    shape = (1, B) if varlen else (B, 1)
    q = torch.randn(*shape, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(*shape, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(*shape, H, V, device=dev, dtype=torch.bfloat16)
    g = (-torch.nn.functional.softplus(torch.randn(*shape, H, K, device=dev) * 0.5 - 2.0)).contiguous()
    beta = torch.sigmoid(torch.randn(*shape, H, device=dev)).to(torch.bfloat16)
    cu = torch.arange(B + 1, dtype=torch.int32, device=dev) if varlen else None
    sr_seed = torch.randint(0, 2**31 - 1, (B,), dtype=torch.int32, device=dev) if sr else None
    c0, s0 = _pools(axis, asym, NS, H, V, K, dev)
    kw = dict(use_qk_l2norm_in_kernel=True, sr_seed=sr_seed, sr_salt=7, asym=asym, fast=fast, launch=launch, axis=axis)
    c_ref, s_ref = c0.clone(), s0.clone()
    o_ref = fused_recurrent_kda_int(q, k, v, g, beta, None, c_ref, s_ref, idx, cu_seqlens=cu, decode=False, **kw)
    c_dec, s_dec = c0.clone(), s0.clone()
    o_dec = fused_recurrent_kda_int(q, k, v, g, beta, None, c_dec, s_dec, idx, cu_seqlens=cu, **kw)  # auto-dispatch
    c_dir, s_dir = c0.clone(), s0.clone()
    o_dir = fused_recurrent_kda_decode_int(
        q.reshape(B, H, K),
        k.reshape(B, H, K),
        v.reshape(B, H, V),
        g.reshape(B, H, K),
        beta.reshape(B, H),
        None,
        c_dir,
        s_dir,
        idx,
        **kw,
    ).reshape(v.shape)
    live = (idx > 0).nonzero().flatten()
    o_ref, o_dec, o_dir = (o.reshape(B, H, V)[live] for o in (o_ref, o_dec, o_dir))
    assert torch.equal(o_ref, o_dec) and torch.equal(o_ref, o_dir), "o differs"
    assert torch.equal(c_ref, c_dec) and torch.equal(c_ref, c_dir), "codes differ"
    assert torch.equal(s_ref, s_dec) and torch.equal(s_ref, s_dir), "scales differ"
    assert torch.equal(c_ref[0], c0[0]) and torch.equal(s_ref[0], s0[0]), "null slot touched"
    assert not torch.equal(c_ref[idx[live].long()], c0[idx[live].long()]), "state not updated"


CASES = [
    (axis, asym, sr, fast, launch, varlen)
    for axis, asym, sr, fast, varlen in itertools.product(
        ["value", "key"], [False, True], [False, True], [False, True], [False, True]
    )
    for launch in ([None, (16, 1, 2), (32, 2, 1)] if axis == "value" else [None, (128, 4, 2)])
]


@requires_gpu
@pytest.mark.parametrize("axis,asym,sr,fast,launch,varlen", CASES)
def test_decode_kernel_bit_identical(axis, asym, sr, fast, launch, varlen):
    _run(axis, asym, sr, fast, launch, varlen)
