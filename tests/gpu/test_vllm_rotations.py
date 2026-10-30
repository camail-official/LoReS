"""Serving-side online Hadamards (W/A rotations): export basis, fullgraph tracing and CUDA-graph capture."""

import pytest
import torch

pytest.importorskip(
    "lores.kernels.rotations.conv_vllm",
    exc_type=ImportError,
    reason="requires the patched vLLM and its CUDA runtime",
)

DEV = "cuda"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("n", [128, 2304])
@pytest.mark.parametrize("signed", [False, True])
def test_serving_hadamard_preserves_export_basis_and_fullgraph_capture(n, signed, monkeypatch):
    from lores.backends.vllm import rotations
    from lores.backends.vllm.rotations import OnlineHadamard
    from lores.utils.hadamard_utils import matmul_hadU

    torch.manual_seed(3)
    sign = rotations.r4_sign(n, "model.layers.7.mlp") if signed else None
    h = OnlineHadamard(n, sign)
    x = torch.randn(3, n, device=DEV, dtype=torch.bfloat16)
    rotated = x if sign is None else x * sign.to(device=DEV, dtype=x.dtype)
    want = matmul_hadU(rotated.float()).to(x.dtype)
    torch.testing.assert_close(h(x), want, rtol=1e-2, atol=1e-2)
    # Fullgraph tracing catches imports/late setup accidentally moved into the tensor path.
    compiled = torch.compile(h, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), h(x), rtol=0, atol=0)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        h(x)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = h(x)
    graph.replay()
    torch.testing.assert_close(captured, h(x), rtol=0, atol=0)
    monkeypatch.setattr(rotations, "_LINQ_R4", None)
    untouched = x.clone()
    rotations.apply_expert_r4(untouched)
    torch.testing.assert_close(untouched, x, rtol=0, atol=0)
    rotations.linq_set_expert_r4(h)
    out = x.clone()
    rotations.apply_expert_r4(out)
    torch.testing.assert_close(out, h(x), rtol=0, atol=0)
    compiled_hook = torch.compile(rotations.apply_expert_r4, backend="eager", fullgraph=True)
    out = x.clone()
    compiled_hook(out)
    torch.testing.assert_close(out, h(x), rtol=0, atol=0)
    wrong_width = x[:, :17].clone()
    rotations.apply_expert_r4(wrong_width)
    torch.testing.assert_close(wrong_width, x[:, :17], rtol=0, atol=0)
