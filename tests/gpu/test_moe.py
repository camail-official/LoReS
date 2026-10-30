"""Check W4A8 expert GEMMs, routed-expert integration, and serving dispatch against reference paths."""

import pytest
import torch
import triton.language as tl

pytest.importorskip("vllm", exc_type=ImportError, reason="requires the patched vLLM")

from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import dispatch_fused_moe_kernel, fused_experts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8

from lores.backends.vllm.moe import int4_w4a8_moe_quant_config

G = 128
DEV = "cuda"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def make_experts(E, N, K, seed, scale_mul=1.0):
    """Packed int4 experts in the WNA16 post-load layout: [E, N, K/2] uint8, even k in the low
    nibble, value = nibble - 8; group scales [E, N, K/G] bf16. Returns (packed, scales, w_ref fp32).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(0, 16, (E, N, K), generator=g).to(torch.uint8)
    scale = ((torch.rand(E, N, K // G, generator=g) * 0.02 + 0.005) * scale_mul).to(torch.bfloat16)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    w_ref = (q.float() - 8) * scale.float().repeat_interleave(G, dim=2)
    return packed.to(DEV), scale.to(DEV), w_ref.to(DEV)


def routing(M, E, top_k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(M, E, generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return w.to(DEV, torch.float32), ids.to(DEV, torch.int32)


def dequant_tokens(a_q, a_s):
    return a_q.float() * a_s.float()


# ------------------------------------------------------------------ 1. stage-exact
def stage_ref(a_deq, w_ref, topk_ids, topk_w, top_k, mul_routed):
    """fp32 reference of one fused expert GEMM. A rows are tokens (top_k>1, GEMM1) or the flattened
    [M*top_k] rows (top_k=1, GEMM2).
    """
    E, N, K = w_ref.shape
    M = topk_ids.shape[0]
    k_real = topk_ids.shape[1]
    out = torch.zeros(M, k_real, N, device=DEV, dtype=torch.float32)
    for e in range(E):
        rows, slots = (topk_ids == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        a = a_deq[rows] if top_k > 1 else a_deq[rows * k_real + slots]
        y = a @ w_ref[e].T
        if mul_routed:
            y = y * topk_w[rows, slots][:, None]
        out[rows, slots] = y
    return out


@pytest.mark.parametrize("N,K,top_k", [(2048, 2304, 8), (2304, 1024, 1)])
@pytest.mark.parametrize("M", [1, 33])
@pytest.mark.parametrize("mul_routed", [False, True])
@pytest.mark.parametrize("block_k", [32, 128])
def test_stage_exact(M, top_k, N, K, mul_routed, block_k, seed=0):
    E = 8
    packed, scale, w_ref = make_experts(E, N, K, seed)
    topk_w, topk_ids = routing(M, E, top_k, seed + 1)
    rows = M if top_k > 1 else M * topk_ids.shape[1]
    a = torch.randn(rows, K, device=DEV, dtype=torch.bfloat16, generator=torch.Generator(device=DEV).manual_seed(seed + 2))
    a_q, a_s = per_token_quant_int8(a)
    assert a_q.dtype == torch.int8 and a_s.shape == (rows, 1)
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 8,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    sorted_ids, expert_ids, n_post = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E, None)
    kk = topk_ids.shape[1]
    C = torch.zeros(M, kk, N, device=DEV, dtype=torch.float32)
    dispatch_fused_moe_kernel(
        a_q,
        packed,
        C,
        a_s,
        scale,
        None,
        topk_w,
        sorted_ids,
        expert_ids,
        n_post,
        mul_routed,
        top_k,
        config,
        compute_type=tl.float32,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        block_shape=[0, G],
        B_bias=None,
        use_int4_w4a8=True,
    )
    ref = stage_ref(dequant_tokens(a_q, a_s), w_ref, topk_ids, topk_w, top_k, mul_routed)
    scale_ = ref.abs().max().item()
    err = (C - ref).abs().max().item()
    assert torch.isfinite(C).all()
    assert err <= 1e-4 * scale_, (err, scale_, M, top_k, N, K, mul_routed, block_k)
    return err / scale_


# ------------------------------------------------------------------ 2. end-to-end
def moe_ref(x, w13_ref, w2_ref, topk_w, topk_ids, a8):
    """Mirror of fused_experts_impl: A quant (per-token int8 or none), GEMM1 fp32 -> bf16 cache,
    silu(gate)*up fp32 -> bf16, requant, GEMM2 fp32 * routed weight -> bf16, fp32 sum over top_k.
    """
    M, K = x.shape
    E, N2, _ = w13_ref.shape
    inter = N2 // 2
    top_k = topk_ids.shape[1]
    cache3 = torch.zeros(M, top_k, K, device=DEV, dtype=torch.bfloat16)
    if a8:
        xq, xs = per_token_quant_int8(x)
        xd = dequant_tokens(xq, xs)
    else:
        xd = x.float()
    for e in range(E):
        rows, slots = (topk_ids == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        h = (xd[rows] @ w13_ref[e].T).to(torch.bfloat16)
        act = (torch.nn.functional.silu(h[:, :inter].float()) * h[:, inter:].float()).to(torch.bfloat16)
        if a8:
            aq, as_ = per_token_quant_int8(act)
            ad = dequant_tokens(aq, as_)
        else:
            ad = act.float()
        y = (ad @ w2_ref[e].T) * topk_w[rows, slots][:, None]
        cache3[rows, slots] = y.to(torch.bfloat16)
    return cache3.float().sum(1).to(torch.bfloat16)


def run_fused(x, packed13, s13, packed2, s2, topk_w, topk_ids, a8):
    cfg = (
        int4_w4a8_moe_quant_config(s13, s2, G)
        if a8
        else int4_w4a16_moe_quant_config(w1_scale=s13, w2_scale=s2, w1_zp=None, w2_zp=None, block_shape=[0, G])
    )
    return fused_experts(x, packed13, packed2, topk_w, topk_ids, quant_config=cfg)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("M", [1, 64])
def test_e2e(M, top_k=8, K=2304, inter=1024, E=16, seed=0):
    packed13, s13, w13_ref = make_experts(E, 2 * inter, K, seed)
    packed2, s2, w2_ref = make_experts(E, K, inter, seed + 10)
    topk_w, topk_ids = routing(M, E, top_k, seed + 1)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, generator=torch.Generator(device=DEV).manual_seed(seed + 2))
    out = run_fused(x, packed13, s13, packed2, s2, topk_w, topk_ids, a8=True)
    assert out.dtype == torch.bfloat16 and out.shape == (M, K) and torch.isfinite(out).all()
    ref = moe_ref(x, w13_ref, w2_ref, topk_w, topk_ids, a8=True)
    scale_ = ref.abs().max().item()
    d = (out.float() - ref.float()).abs()
    assert d.max().item() <= 0.02 * scale_, (d.max().item(), scale_, M)
    assert d.mean().item() <= 2e-3 * scale_, (d.mean().item(), scale_, M)
    # against the proven W4A16 path on the same packed weights: int8 activation noise only
    out16 = run_fused(x, packed13, s13, packed2, s2, topk_w, topk_ids, a8=False)
    ref16 = moe_ref(x, w13_ref, w2_ref, topk_w, topk_ids, a8=False)
    assert _rel(out16, ref16) <= 0.02, _rel(out16, ref16)  # the W4A16 path itself vs its mirror
    r = _rel(out, out16)
    assert r <= 0.03, (r, M)
    # negative control: wrong scales on the int4 path must be far off
    _, s13x, _ = make_experts(E, 2 * inter, K, seed, scale_mul=1.5)
    bad = run_fused(x, packed13, s13x, packed2, s2, topk_w, topk_ids, a8=True)
    assert _rel(bad, out16) > 0.2, _rel(bad, out16)
    return _rel(out, ref), r


# ------------------------------------------------------------------ 3. config + dispatch predicates
def test_config_flags():
    s1 = torch.ones(2, 4, 2, device=DEV, dtype=torch.bfloat16)
    cfg = int4_w4a8_moe_quant_config(s1, s1, G)
    assert cfg.use_int4_w4a8 and not cfg.use_int8_w8a8 and not cfg.use_int4_w4a16
    assert cfg.per_act_token_quant and cfg.block_shape is None and cfg.w4a8_block_shape == [0, G]
    assert cfg.quant_dtype == torch.int8 and cfg.w1_scale is s1
    cfg16 = int4_w4a16_moe_quant_config(w1_scale=s1, w2_scale=s1, block_shape=[0, G])
    assert cfg16.use_int4_w4a16 and not cfg16.use_int4_w4a8 and not cfg16.use_int8_w8a8


def test_dispatch_predicates():
    from compressed_tensors.quantization import QuantizationArgs
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

    from lores.backends.vllm.moe import (  # noqa
        LinqW4A8IntMoEMethod,
        is_linq_w4a8_int,
    )

    w = QuantizationArgs(num_bits=4, type="int", strategy="group", group_size=G, symmetric=True, dynamic=False)
    i = QuantizationArgs(num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True)
    assert CompressedTensorsConfig._is_dynamic_token_w4a8_int(w, i)
    assert not CompressedTensorsConfig._is_fp8_w4a8(w, i)  # the int-act config must not reach the fp8 kernel
    assert not CompressedTensorsConfig._is_wNa16_group_channel(w, i)
    assert is_linq_w4a8_int(CompressedTensorsConfig, w, i)
    i8 = QuantizationArgs(num_bits=8, type="float", strategy="token", symmetric=True, dynamic=True)
    assert CompressedTensorsConfig._is_fp8_w4a8(w, i8)
    assert not is_linq_w4a8_int(CompressedTensorsConfig, w, i8)  # fp8 activations stay off the int path
    assert not is_linq_w4a8_int(CompressedTensorsConfig, w, None)
    w8 = QuantizationArgs(num_bits=8, type="int", strategy="channel", symmetric=True, dynamic=False)
    assert not is_linq_w4a8_int(CompressedTensorsConfig, w8, i)  # W8A8 keeps vLLM's int8 method
    assert issubclass(LinqW4A8IntMoEMethod, object)
