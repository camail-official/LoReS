"""Check fused inverse rotation and gated normalization against eager per-head/group references."""

from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytestmark = cuda_only

# ----------------------------------------------------------------------------- helpers


def _eager_unrotate_then_norm(
    o: torch.Tensor,
    g: torch.Tensor,
    R: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    n_groups: int,
    head_dim: int,
    eps: float,
    activation: str,
) -> torch.Tensor:
    """Reference: per-head fp32 matmul against R^T, then fla's ``rms_norm_gated``.

    Mirrors :meth:`lores.rotations.wrappers.UnrotateWrapper.forward` (which expects ``o : [..., n_heads,
    head_dim]``): reshape to per-head, matmul, reshape back, then call the (Triton-fused)
    gated RMSNorm. fla's ``rms_norm_gated`` reduces over the last ``head_dim`` channels and
    applies the per-channel ``weight`` over the same axis — matches our kernel's contract.
    """
    from fla.modules.fused_norm_gate import rms_norm_gated

    # 4D reshape (last dim = head_dim) so fla's norm reduces over head_dim and the [head_dim]
    # weight matches the asserted shape (fla's `layer_norm_gated_fwd` asserts
    # `weight.shape == (x.shape[-1],)` after its internal flatten).
    oh = o.reshape(*o.shape[:-1], n_groups, head_dim)
    oh = torch.matmul(oh.float(), R.float().t()).to(o.dtype)
    gh = g.reshape(*g.shape[:-1], n_groups, head_dim)
    out = rms_norm_gated(
        oh,
        gh,
        weight if weight is not None else torch.ones(head_dim, dtype=o.dtype, device=o.device),
        None,  # bias (always None for FusedRMSNormGated)
        activation,
        residual=None,
        eps=eps,
        prenorm=False,
        residual_in_fp32=False,
    )
    return out.reshape(o.shape)


# ----------------------------------------------------------------------------- correctness


class TestUnrotateNormGdnKda:
    """Reference parity: fp32 matmul(Rᵀ) + fla.modules ``rms_norm_gated``, vs the fused kernel."""

    @cuda_only
    @pytest.mark.parametrize("activation", ["swish", "sigmoid"])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("n_groups", [1, 4])
    @pytest.mark.parametrize("seq_len", [1, 33])
    def test_matches_eager(self, activation, head_dim, n_groups, seq_len):
        from lores.kernels.rotations.unrotate_norm import unrotate_rmsnorm_gated_fwd
        from lores.rotations.matrices import factor_random_hadamard
        from lores.utils.hadamard_utils import random_hadamard_matrix

        torch.manual_seed(head_dim * 31 + n_groups * 7 + seq_len + (1 if activation == "swish" else 2))
        device, dtype = "cuda", torch.float16

        R = random_hadamard_matrix(head_dim, device=device).to(torch.float32)
        signs, H_block = factor_random_hadamard(R)
        signs, H_block = signs.to(device), H_block.to(device)

        B = 2
        o = torch.randn(B, seq_len, n_groups * head_dim, dtype=dtype, device=device) * 0.5
        g = torch.randn(B, seq_len, n_groups * head_dim, dtype=dtype, device=device) * 0.5
        weight = torch.randn(head_dim, dtype=dtype, device=device).abs() + 0.2

        out = unrotate_rmsnorm_gated_fwd(
            o,
            g,
            signs,
            H_block,
            weight,
            n_groups=n_groups,
            head_dim=head_dim,
            eps=1e-5,
            activation=activation,
        )
        ref = _eager_unrotate_then_norm(
            o,
            g,
            R,
            weight,
            n_groups=n_groups,
            head_dim=head_dim,
            eps=1e-5,
            activation=activation,
        )
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)

    @cuda_only
    def test_silu_alias(self):
        """``activation="silu"`` is an alias for ``"swish"``."""
        from lores.kernels.rotations.unrotate_norm import unrotate_rmsnorm_gated_fwd
        from lores.rotations.matrices import factor_random_hadamard
        from lores.utils.hadamard_utils import random_hadamard_matrix

        torch.manual_seed(42)
        device, dtype = "cuda", torch.float16
        head_dim, n_groups, T = 64, 2, 19
        R = random_hadamard_matrix(head_dim, device=device).to(torch.float32)
        signs, H_block = factor_random_hadamard(R)
        signs, H_block = signs.to(device), H_block.to(device)

        o = torch.randn(1, T, n_groups * head_dim, dtype=dtype, device=device) * 0.5
        g = torch.randn(1, T, n_groups * head_dim, dtype=dtype, device=device) * 0.5
        weight = torch.ones(head_dim, dtype=dtype, device=device)

        out_swish = unrotate_rmsnorm_gated_fwd(
            o, g, signs, H_block, weight, n_groups=n_groups, head_dim=head_dim, activation="swish"
        )
        out_silu = unrotate_rmsnorm_gated_fwd(
            o, g, signs, H_block, weight, n_groups=n_groups, head_dim=head_dim, activation="silu"
        )
        torch.testing.assert_close(out_silu, out_swish, atol=0, rtol=0)


class TestUnrotateNormDtypes:
    @cuda_only
    def test_bfloat16(self):
        from lores.kernels.rotations.unrotate_norm import unrotate_rmsnorm_gated_fwd
        from lores.rotations.matrices import factor_random_hadamard
        from lores.utils.hadamard_utils import random_hadamard_matrix

        torch.manual_seed(7)
        device, dtype = "cuda", torch.bfloat16
        head_dim, n_groups, T = 64, 4, 17
        R = random_hadamard_matrix(head_dim, device=device).to(torch.float32)
        signs, H_block = factor_random_hadamard(R, block_dtype=torch.bfloat16, signs_dtype=torch.bfloat16)
        signs, H_block = signs.to(device), H_block.to(device)

        o = torch.randn(1, T, n_groups * head_dim, dtype=dtype, device=device) * 0.3
        g = torch.randn(1, T, n_groups * head_dim, dtype=dtype, device=device) * 0.3
        weight = torch.randn(head_dim, dtype=dtype, device=device).abs() + 0.2

        out = unrotate_rmsnorm_gated_fwd(o, g, signs, H_block, weight, n_groups=n_groups, head_dim=head_dim, activation="swish")
        ref = _eager_unrotate_then_norm(o, g, R, weight, n_groups=n_groups, head_dim=head_dim, eps=1e-5, activation="swish")
        # bf16 tensor cores in the inner Kronecker tl.dot ⇒ wider rounding band than fp16.
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
