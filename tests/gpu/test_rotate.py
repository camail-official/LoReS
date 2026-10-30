"""Check fused Hadamard rotations against the FP32 eager reference."""

from __future__ import annotations

import math

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytestmark = cuda_only


def _eager_rotate(y: torch.Tensor, R: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """The eager rotation path: reshape to per-head, fp32 matmul, cast back."""
    yh = y.reshape(*y.shape[:-1], num_heads, head_dim)
    yh = torch.matmul(yh.float(), R.float()).to(yh.dtype)
    return yh.reshape(y.shape)


class TestSylvester:
    def test_sylvester_orthogonality(self):
        from lores.rotations.matrices import sylvester_hadamard

        for d in (2, 4, 8, 16, 32, 64, 128, 256):
            H = sylvester_hadamard(d, dtype=torch.float64)
            assert H.shape == (d, d)
            # H is symmetric and H @ H.T = d * I
            assert torch.equal(H, H.T)
            assert torch.allclose(H @ H.T, d * torch.eye(d, dtype=torch.float64), atol=0)

    def test_sylvester_pow_two_required(self):
        from lores.rotations.matrices import sylvester_hadamard

        with pytest.raises(ValueError, match="power of two"):
            sylvester_hadamard(3)
        with pytest.raises(ValueError, match="power of two"):
            sylvester_hadamard(0)


class TestFactorRandomHadamard:
    @pytest.mark.parametrize("d", [64, 128])
    def test_roundtrip(self, d):
        """``factor_random_hadamard(R)`` recovers ``(signs, H_block)`` consistent with R."""
        from lores.rotations.matrices import factor_random_hadamard, sylvester_hadamard
        from lores.utils.hadamard_utils import random_hadamard_matrix

        torch.manual_seed(123 + d)
        R = random_hadamard_matrix(d, device="cpu")  # fp64
        signs, H_block = factor_random_hadamard(R)

        assert signs.shape == (d,)
        assert H_block.shape == (d // 2, d // 2)
        assert set(signs.unique().tolist()).issubset({-1.0, 1.0})

        # Reconstruct R = diag(signs) @ H_d / √d and compare in fp32.
        H_d = sylvester_hadamard(d, dtype=torch.float64)
        R_reco = torch.diag(signs.to(torch.float64)) @ H_d / math.sqrt(d)
        torch.testing.assert_close(R_reco, R, atol=1e-6, rtol=1e-6)

        # H_block = H_{d/2} / √(d/2)
        H_half = sylvester_hadamard(d // 2, dtype=torch.float64) / math.sqrt(d // 2)
        torch.testing.assert_close(H_block.to(torch.float64), H_half, atol=1e-3, rtol=0)

    def test_non_hadamard_rejected(self):
        """A random orthogonal (non-Hadamard) matrix must be rejected, not silently mis-factored."""
        from lores.rotations.matrices import factor_random_hadamard

        torch.manual_seed(0)
        # Haar-random orthogonal — almost surely not of the form diag(±1) @ H / √D.
        A = torch.randn(64, 64, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        with pytest.raises(ValueError):
            factor_random_hadamard(Q)

    def test_non_power_of_two_rejected(self):
        from lores.rotations.matrices import factor_random_hadamard

        # A 6×6 won't even pass shape checks.
        with pytest.raises(ValueError, match="power of two"):
            factor_random_hadamard(torch.eye(6))

    def test_non_square_rejected(self):
        from lores.rotations.matrices import factor_random_hadamard

        with pytest.raises(ValueError, match="square"):
            factor_random_hadamard(torch.zeros(4, 8))


class TestRotate:
    """Per-head rotation matches the eager reference."""

    @cuda_only
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("num_heads", [1, 4])
    @pytest.mark.parametrize("seq_len", [1, 33])
    def test_matches_eager(self, head_dim, num_heads, seq_len):
        from lores.kernels.rotations.rotate import rotate_fwd
        from lores.rotations.matrices import factor_random_hadamard
        from lores.utils.hadamard_utils import random_hadamard_matrix

        torch.manual_seed(head_dim * 31 + num_heads * 7 + seq_len)
        device = "cuda"
        dtype = torch.float16

        R = random_hadamard_matrix(head_dim, device=device).to(torch.float32)
        signs, H_block = factor_random_hadamard(R)
        signs = signs.to(device)
        H_block = H_block.to(device)

        B = 2
        y = torch.randn(B, seq_len, num_heads * head_dim, dtype=dtype, device=device) * 0.5

        out = rotate_fwd(
            y,
            signs,
            H_block,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        ref = _eager_rotate(y, R, num_heads, head_dim)
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
