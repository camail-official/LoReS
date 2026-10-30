import pytest
import torch

from lores.kernels.state_int.packing.packed_state_kernel_naive import PackedState, pack_state, unpack_state

# The fused prefill pack must be bit-identical to packed_state_kernel_naive.pack_state + scatter, which is
# the torch chain it replaces.


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytestmark = requires_gpu


@requires_gpu
@pytest.mark.parametrize("N", [128, 256])
def test_matches_pack_state(N):
    from lores.kernels.state_int.packing.pack_state_kernel import pack_state_to_slots

    torch.manual_seed(0)
    dev = "cuda"
    n, H, D, NS = 5, 8, 16, 64
    states = torch.randn(n, H, D, N, device=dev, dtype=torch.float32)
    slots = torch.tensor([40, 3, 17, 63, 8], device=dev, dtype=torch.int32)

    ref = pack_state(states)
    codes = torch.zeros(NS, H, D, N, device=dev, dtype=torch.uint8)
    scales = torch.zeros(NS, H, D, device=dev, dtype=torch.float32)
    pack_state_to_slots(states, codes, scales, slots)

    # Codes may differ by one step on exact ties: the kernel rounds Triton's approximate
    # divide, exactly as the decode requant does (quant.py:requant_int8). Scales are exact.
    d = (codes[slots.long()].int() - ref.codes.int()).abs()
    assert d.max() <= 1 and (d > 0).float().mean() < 1e-3, (d.max().item(), (d > 0).sum().item())
    torch.testing.assert_close(scales[slots.long()], ref.scales, rtol=0, atol=0)


@requires_gpu
def test_saturating_guard():
    """Non-finite state must saturate, not poison the scale."""
    from lores.kernels.state_int.packing.pack_state_kernel import pack_state_to_slots

    dev = "cuda"
    n, H, D, N, NS = 1, 2, 4, 128, 8
    states = torch.randn(n, H, D, N, device=dev, dtype=torch.float32)
    states[0, 0, 0, 0] = float("inf")
    states[0, 0, 1, 0] = float("nan")
    codes = torch.zeros(NS, H, D, N, device=dev, dtype=torch.uint8)
    scales = torch.zeros(NS, H, D, device=dev, dtype=torch.float32)
    pack_state_to_slots(states, codes, scales, torch.tensor([2], device=dev, dtype=torch.int32))
    assert torch.isfinite(scales[2]).all()
    ref = pack_state(states)
    torch.testing.assert_close(codes[2], ref.codes[0], rtol=0, atol=0)


@requires_gpu
def test_fast_matches_decode_requant():
    """Packing uses the same reciprocal-multiply requantization as decode."""
    import lores.kernels.state_int.packing.pack_state_kernel as m

    torch.manual_seed(4)
    dev = "cuda"
    n, H, D, N, NS = 3, 4, 8, 128, 16
    states = torch.randn(n, H, D, N, device=dev, dtype=torch.float32)
    slots = torch.tensor([5, 11, 2], device=dev, dtype=torch.int32)
    codes = torch.zeros(NS, H, D, N, device=dev, dtype=torch.uint8)
    scales = torch.zeros(NS, H, D, device=dev, dtype=torch.float32)
    m.pack_state_to_slots(states, codes, scales, slots, fast=True)

    # reference: reciprocal-mul + round-to-nearest-even, as quant.py's requant_int8_fast does
    amax = states.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    q = torch.round(states * (127 / amax)).clamp(-127, 127)
    torch.testing.assert_close(codes[slots.long()], (q.to(torch.int16) + 127).to(torch.uint8), rtol=0, atol=0)


@requires_gpu
def test_unpack_roundtrip_matches_torch():
    """The fused unpack must equal packed_state_kernel_naive.unpack_state on the same pools."""
    from lores.kernels.state_int.packing.pack_state_kernel import pack_state_to_slots, unpack_state_from_slots

    torch.manual_seed(2)
    dev = "cuda"
    n, H, D, N, NS = 4, 8, 16, 128, 32
    states = torch.randn(n, H, D, N, device=dev, dtype=torch.float32)
    slots = torch.tensor([30, 1, 12, 7], device=dev, dtype=torch.int32)
    codes = torch.zeros(NS, H, D, N, device=dev, dtype=torch.uint8)
    scales = torch.zeros(NS, H, D, device=dev, dtype=torch.float32)
    pack_state_to_slots(states, codes, scales, slots)

    ref = unpack_state(PackedState(codes[slots.long()], scales[slots.long()], (n, H, D, N)))
    got = unpack_state_from_slots(torch.empty_like(states), codes, scales, slots)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@requires_gpu
def test_asym_matches_pack_state_and_roundtrip():
    from lores.kernels.state_int.packing.pack_state_kernel import pack_state_to_slots, unpack_state_from_slots

    dev = "cuda"
    n, H, D, N, slots_n = 3, 4, 8, 128, 7
    torch.manual_seed(0)
    states = torch.randn(n, H, D, N, device=dev) * 2 + 0.7
    codes = torch.zeros(slots_n, H, D, N, dtype=torch.uint8, device=dev)
    scales = torch.zeros(slots_n, H, D, 2, dtype=torch.float32, device=dev)
    slots = torch.tensor([5, 1, 6], device=dev, dtype=torch.int32)
    pack_state_to_slots(states, codes, scales, slots, asym=True)
    p = pack_state(states, asym=True)
    torch.testing.assert_close(scales[slots], p.scales, atol=0, rtol=0)
    assert int((codes[slots] != p.codes).sum()) <= max(2, p.codes.numel() // 10_000)
    out = torch.empty_like(states)
    unpack_state_from_slots(out, codes, scales, slots, asym=True)
    # code * scale + lo: Triton contracts it into one FMA, torch rounds twice -> 1 ulp
    torch.testing.assert_close(out, unpack_state(p._replace(codes=codes[slots], scales=scales[slots])), atol=1e-6, rtol=1e-6)


@requires_gpu
def test_asym_fast_matches_exact_within_one_code():
    """Fast versus exact asymmetric packing differs only at rounding boundaries."""
    import lores.kernels.state_int.packing.pack_state_kernel as m

    dev = "cuda"
    torch.manual_seed(2)
    n, H, D, N, NS = 3, 4, 8, 128, 9
    states = torch.randn(n, H, D, N, device=dev) * 2 + 0.7
    slots = torch.tensor([7, 0, 3], device=dev, dtype=torch.int32)
    out = {}
    for fast in ("0", "1"):
        codes = torch.zeros(NS, H, D, N, device=dev, dtype=torch.uint8)
        scales = torch.zeros(NS, H, D, 2, device=dev, dtype=torch.float32)
        m.pack_state_to_slots(states, codes, scales, slots, asym=True, fast=fast == "1")
        out[fast] = (codes[slots.long()].clone(), scales[slots.long()].clone())
    torch.testing.assert_close(out["0"][1], out["1"][1], atol=0, rtol=0)  # scale/min: same reduction
    d = (out["0"][0].int() - out["1"][0].int()).abs()
    assert int(d.max()) <= 1 and int((d > 0).sum()) <= max(2, d.numel() // 10_000), (int(d.max()), int((d > 0).sum()))


@requires_gpu
@pytest.mark.parametrize("asym", [False, True])
def test_sr_pack_is_unbiased(asym):
    """Handoff SR: averaging the dequantized pack over many seeds recovers the fp state (RTN does not)."""
    from lores.kernels.state_int.packing.pack_state_kernel import pack_state_to_slots, unpack_state_from_slots

    dev = "cuda"
    n, H, D, N = 1, 2, 4, 128
    torch.manual_seed(3)
    states = torch.randn(n, H, D, N, device=dev) * 2 + 0.7
    codes = torch.zeros(1, H, D, N, dtype=torch.uint8, device=dev)
    scales = torch.zeros((1, H, D, 2) if asym else (1, H, D), dtype=torch.float32, device=dev)
    slots = torch.zeros(1, dtype=torch.int32, device=dev)
    out = torch.empty_like(states)
    acc = torch.zeros_like(states)
    K = 400
    for seed in range(K):
        pack_state_to_slots(states, codes, scales, slots, asym=asym, sr_seed=seed + 1)
        unpack_state_from_slots(out, codes, scales, slots, asym=asym)
        acc += out
    pack_state_to_slots(states, codes, scales, slots, asym=asym)  # RTN
    unpack_state_from_slots(out, codes, scales, slots, asym=asym)
    step = (scales[..., 0] if asym else scales)[..., None].expand_as(states)
    bias_sr = ((acc / K - states) / step).abs().mean().item()
    bias_rtn = ((out - states) / step).abs().mean().item()
    assert bias_sr < 0.05, f"SR pack biased: {bias_sr:.3f} steps (RTN {bias_rtn:.3f})"
    assert bias_rtn > 0.15, f"RTN control unexpectedly small: {bias_rtn:.3f}"
