"""Fused Triton conv + FWHT butterfly (vLLM prefill and decode paths).

Reference = vendor `causal_conv1d.py` output, rotated in torch. Both run in fp32 so any
discrepancy is the butterfly itself, not conv rounding.
"""

import math

import pytest
import torch

from lores.backends.vllm.rotations import make_channel_spec  # noqa: E402
from lores.rotations.matrices import sylvester_hadamard  # noqa: E402

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

vllm_conv = pytest.importorskip("vllm.model_executor.layers.mamba.ops.causal_conv1d", exc_type=ImportError)
bfly = pytest.importorskip("lores.kernels.rotations.conv_vllm")


pytestmark = requires_gpu
NULL_BLOCK_ID = vllm_conv.NULL_BLOCK_ID


def _spec(segments, dim, device):
    """(signs, log2dim, per-segment sign vectors) for the given [(off, width, head_dim)]."""
    segs, sign_vecs = [], {}
    g = torch.Generator(device="cpu").manual_seed(0)
    for off, width, hd in segments:
        sv = (torch.randint(0, 2, (hd,), generator=g, dtype=torch.int8) * 2 - 1).to(device=device, dtype=torch.float32)
        sign_vecs[off] = sv
        segs.append((off, width, hd, sv))
    signs, l2d = make_channel_spec(segs, dim, device)
    return signs, l2d, sign_vecs


def _rotate_ref(y, segments, sign_vecs, channel_dim):
    """Apply row-vector R = diag(signs) @ H / sqrt(d), with signs before the butterfly."""
    out = y.clone().float()
    for off, width, hd in segments:
        H = sylvester_hadamard(hd, dtype=torch.float32, device=y.device) / math.sqrt(hd)
        R = sign_vecs[off][:, None] * H
        sl = [slice(None)] * out.dim()
        sl[channel_dim] = slice(off, off + width)
        blk = out[tuple(sl)]
        blk = blk.movedim(channel_dim, -1).unflatten(-1, (width // hd, hd))
        blk = (blk @ R).flatten(-2).movedim(-1, channel_dim)
        out[tuple(sl)] = blk
    return out


def _segments(dim):
    """xBC-like layout: a 128-head block, a 64-head block, and an unrotated tail."""
    return [(0, dim // 2, 128), (dim // 2, dim // 4, 64)]


def _uniform_segments(dim):
    """Every channel rotated with the same head dim -- the ROT_D fast path."""
    return [(0, dim, 128)]


# ---------------------------------------------------------------- decode


def _decode_case(dim=512, batch=7, width=4, silu=True, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    slots = batch + 3
    x = torch.randn(batch, dim, device=dev, dtype=dtype)
    state = torch.randn(slots, dim, width - 1, device=dev, dtype=dtype)
    weight = torch.randn(dim, width, device=dev, dtype=dtype) * 0.2
    bias = torch.randn(dim, device=dev, dtype=dtype)
    idx = torch.randperm(slots - 1, dtype=torch.int32, device=dev)[:batch] + 1
    return x, state, weight, bias, idx, "silu" if silu else None


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("silu", [True, False])
def test_decode_matches_vendor_plus_rotation(width, silu):
    dim = 512
    x, state, weight, bias, idx, act = _decode_case(dim=dim, width=width, silu=silu)
    segs = _segments(dim)
    signs, l2d, sign_vecs = _spec(segs, dim, x.device)

    ref_state, bfly_state = state.clone(), state.clone()
    ref = vllm_conv.causal_conv1d_update(x.clone(), ref_state, weight, bias, act, conv_state_indices=idx)
    got = bfly.causal_conv1d_update(
        x.clone(), bfly_state, weight, bias, act, conv_state_indices=idx, rot_signs=signs, rot_log2dim=l2d
    )

    want = _rotate_ref(ref, segs, sign_vecs, channel_dim=1)
    torch.testing.assert_close(got.float(), want, rtol=1e-5, atol=1e-5)
    # the conv state must be untouched by the rotation: it holds raw inputs, not outputs
    torch.testing.assert_close(bfly_state, ref_state, rtol=0, atol=0)


def test_decode_identity_spec_is_vendor():
    """log2dim == 0 everywhere must reproduce the unrotated kernel bit-for-bit."""
    dim = 512
    x, state, weight, bias, idx, act = _decode_case(dim=dim)
    signs = torch.zeros(dim, dtype=torch.float32, device=x.device)
    l2d = torch.zeros(dim, dtype=torch.int8, device=x.device)
    a, b = state.clone(), state.clone()
    ref = vllm_conv.causal_conv1d_update(x.clone(), a, weight, bias, act, conv_state_indices=idx)
    got = bfly.causal_conv1d_update(x.clone(), b, weight, bias, act, conv_state_indices=idx, rot_signs=signs, rot_log2dim=l2d)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_decode_rot_none_is_vendor():
    """No spec passed at all -> the fork is a drop-in for the vendor kernel."""
    dim = 512
    x, state, weight, bias, idx, act = _decode_case(dim=dim)
    a, b = state.clone(), state.clone()
    ref = vllm_conv.causal_conv1d_update(x.clone(), a, weight, bias, act, conv_state_indices=idx)
    got = bfly.causal_conv1d_update(x.clone(), b, weight, bias, act, conv_state_indices=idx)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_decode_uniform_fastpath_matches_mixed_path():
    """ROT_D>0 drops the per-stage head-dim select; it must not change a single output."""
    dim = 512
    x, state, weight, bias, idx, act = _decode_case(dim=dim)
    segs = _uniform_segments(dim)
    signs, l2d, sign_vecs = _spec(segs, dim, x.device)
    a, b, c = state.clone(), state.clone(), state.clone()
    gen = bfly.causal_conv1d_update(x.clone(), a, weight, bias, act, conv_state_indices=idx, rot_signs=signs, rot_log2dim=l2d)
    fast = bfly.causal_conv1d_update(
        x.clone(), b, weight, bias, act, conv_state_indices=idx, rot_signs=signs, rot_log2dim=l2d, rot_d=128
    )
    # not bit-identical: dropping the select lets the compiler contract a different FMA
    torch.testing.assert_close(fast, gen, rtol=1e-6, atol=1e-6)
    plain = vllm_conv.causal_conv1d_update(x.clone(), c, weight, bias, act, conv_state_indices=idx)
    want = _rotate_ref(plain, segs, sign_vecs, channel_dim=1)
    torch.testing.assert_close(fast.float(), want, rtol=1e-5, atol=1e-5)


def test_prefill_uniform_fastpath_matches_mixed_path():
    dim = 512
    x, weight, bias, states, qsl, has_init, idx = _prefill_case(dim=dim)
    segs = _uniform_segments(dim)
    signs, l2d, sign_vecs = _spec(segs, dim, x.device)
    a, b, c = states.clone(), states.clone(), states.clone()
    kw = dict(
        query_start_loc=qsl, cache_indices=idx, has_initial_state=has_init, activation="silu", rot_signs=signs, rot_log2dim=l2d
    )
    gen = bfly.causal_conv1d_fn(x.clone(), weight, bias, conv_states=a, **kw)
    fast = bfly.causal_conv1d_fn(x.clone(), weight, bias, conv_states=b, rot_d=128, **kw)
    torch.testing.assert_close(fast, gen, rtol=1e-6, atol=1e-6)
    plain = vllm_conv.causal_conv1d_fn(
        x.clone(),
        weight,
        bias,
        conv_states=c,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=has_init,
        activation="silu",
    )
    want = _rotate_ref(plain, segs, sign_vecs, channel_dim=0)
    torch.testing.assert_close(fast.float(), want, rtol=1e-5, atol=1e-5)


def test_decode_bf16():
    dim = 512
    x, state, weight, bias, idx, act = _decode_case(dim=dim, dtype=torch.bfloat16)
    segs = _segments(dim)
    signs, l2d, sign_vecs = _spec(segs, dim, x.device)
    a, b = state.clone(), state.clone()
    ref = vllm_conv.causal_conv1d_update(x.clone(), a, weight, bias, act, conv_state_indices=idx)
    got = bfly.causal_conv1d_update(x.clone(), b, weight, bias, act, conv_state_indices=idx, rot_signs=signs, rot_log2dim=l2d)
    want = _rotate_ref(ref.float(), segs, sign_vecs, channel_dim=1)
    # the kernel rotates the fp32 accumulator, the reference rotates the rounded bf16 output
    torch.testing.assert_close(got.float(), want, rtol=3e-2, atol=3e-2)


# ---------------------------------------------------------------- prefill


def _prefill_case(dim=512, seqlens=(19, 64, 7), width=4, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    batch = len(seqlens)
    total = sum(seqlens)
    slots = batch + 3
    x = torch.randn(dim, total, device=dev, dtype=dtype)
    weight = torch.randn(dim, width, device=dev, dtype=dtype) * 0.2
    bias = torch.randn(dim, device=dev, dtype=dtype)
    states = torch.randn(slots, width - 1, dim, device=dev, dtype=dtype).transpose(1, 2)
    qsl = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=dev)
    has_init = torch.randint(0, 2, (batch,), dtype=torch.bool, device=dev)
    idx = torch.randperm(slots - 1, dtype=torch.int32, device=dev)[:batch] + 1
    return x, weight, bias, states, qsl, has_init, idx


@pytest.mark.parametrize("width", [3, 4])
def test_prefill_matches_vendor_plus_rotation(width):
    dim = 512
    x, weight, bias, states, qsl, has_init, idx = _prefill_case(dim=dim, width=width)
    segs = _segments(dim)
    signs, l2d, sign_vecs = _spec(segs, dim, x.device)

    ref_states, bfly_states = states.clone(), states.clone()
    ref = vllm_conv.causal_conv1d_fn(
        x.clone(),
        weight,
        bias,
        conv_states=ref_states,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=has_init,
        activation="silu",
    )
    got = bfly.causal_conv1d_fn(
        x.clone(),
        weight,
        bias,
        conv_states=bfly_states,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=has_init,
        activation="silu",
        rot_signs=signs,
        rot_log2dim=l2d,
    )

    want = _rotate_ref(ref, segs, sign_vecs, channel_dim=0)
    torch.testing.assert_close(got.float(), want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(bfly_states, ref_states, rtol=0, atol=0)


def test_prefill_rot_none_is_vendor():
    dim = 512
    x, weight, bias, states, qsl, has_init, idx = _prefill_case(dim=dim)
    a, b = states.clone(), states.clone()
    ref = vllm_conv.causal_conv1d_fn(
        x.clone(),
        weight,
        bias,
        conv_states=a,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=has_init,
        activation="silu",
    )
    got = bfly.causal_conv1d_fn(
        x.clone(),
        weight,
        bias,
        conv_states=b,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=has_init,
        activation="silu",
    )
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_prefill_then_decode_share_one_basis():
    """The whole point: a sequence prefilled and then decoded stays in one rotated basis."""
    dim, width = 512, 4
    segs = _segments(dim)
    x, weight, bias, states, qsl, has_init, idx = _prefill_case(dim=dim, width=width)
    signs, l2d, sign_vecs = _spec(segs, dim, x.device)
    has_init = torch.zeros_like(has_init)

    st_v, st_b = states.clone(), states.clone()
    vllm_conv.causal_conv1d_fn(
        x.clone(),
        weight,
        bias,
        conv_states=st_v,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=has_init,
        activation="silu",
    )
    bfly.causal_conv1d_fn(
        x.clone(),
        weight,
        bias,
        conv_states=st_b,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=has_init,
        activation="silu",
        rot_signs=signs,
        rot_log2dim=l2d,
    )
    # conv state carries raw inputs, so prefill hands decode an identical window either way
    torch.testing.assert_close(st_b, st_v, rtol=0, atol=0)

    xd = torch.randn(len(qsl) - 1, dim, device=x.device, dtype=x.dtype)
    ref = vllm_conv.causal_conv1d_update(xd.clone(), st_v, weight, bias, "silu", conv_state_indices=idx)
    got = bfly.causal_conv1d_update(
        xd.clone(), st_b, weight, bias, "silu", conv_state_indices=idx, rot_signs=signs, rot_log2dim=l2d
    )
    want = _rotate_ref(ref, segs, sign_vecs, channel_dim=1)
    torch.testing.assert_close(got.float(), want, rtol=1e-5, atol=1e-5)
