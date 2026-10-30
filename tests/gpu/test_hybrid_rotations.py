import pytest
import torch

# Fused Kronecker rotations on the nemotronh installer + the make_fused_rot funnel itself.
# Invariance oracle: online rotations are output-preserving, so rotated logits must match
# unrotated logits to fp tolerance — a wrong fused kernel breaks this.

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytestmark = requires_gpu


@requires_gpu
@pytest.mark.parametrize("shape", [(5, 128), (2, 7, 128), (2, 3, 4, 128)])
@pytest.mark.parametrize("kind", ["plain", "signed"])
def test_make_fused_rot_matches_eager(shape, kind):
    from lores.rotations.wrappers import make_fused_rot
    from lores.utils.hadamard_utils import hadamard_matrix, random_hadamard_matrix

    torch.manual_seed(0)
    d = shape[-1]
    R = (hadamard_matrix if kind == "plain" else random_hadamard_matrix)(d, "cuda").to(torch.float32)
    t = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    got = make_fused_rot(R)(t)
    ref = torch.matmul(t.to(torch.float32), R).to(t.dtype)
    assert got.shape == t.shape and got.dtype == t.dtype
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)  # fp16 H_block tensor-core dot


def _tiny_nemo():
    from transformers.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM

    torch.manual_seed(0)
    cfg = NemotronHConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        hybrid_override_pattern="M-*",
        mamba_num_heads=8,
        mamba_head_dim=32,
        ssm_state_size=128,
        n_groups=2,
        mamba_d_conv=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=64,
        vocab_size=512,
    )
    m = NemotronHForCausalLM(cfg).cuda().to(torch.float32).eval()
    _ = m.model.layers[0].mixer  # trigger lazy kernel load
    return m


@requires_gpu
@pytest.mark.parametrize("which", ["rk", "rv"])
def test_nemo_fused_invariance(which):
    pytest.importorskip("mamba_ssm", reason="requires the HF environment")
    from lores.backends.hf.nemotronh.rotations import install_nemotronh_rotations

    prompt = torch.randint(0, 512, (2, 16), device="cuda")
    steps = torch.randint(0, 512, (2, 8), device="cuda")

    def run(rotate):
        torch.manual_seed(0)
        model = _tiny_nemo()
        with torch.no_grad():
            model(prompt[:, :2])  # load the lazy kernels before installing wraps
        if rotate:
            install_nemotronh_rotations(model, rk="k" in which, rv="v" in which, seed=0)
        with torch.no_grad():
            out = model(prompt, use_cache=True)
            cache, logits = out.past_key_values, [out.logits.float()]
            for t in range(steps.shape[1]):
                o = model(steps[:, t : t + 1], past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                logits.append(o.logits.float())
        return torch.cat(logits, dim=1)

    base, rot = run(False), run(True)
    torch.testing.assert_close(rot, base, atol=5e-3, rtol=5e-3)
