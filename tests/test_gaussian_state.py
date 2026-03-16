"""
Tests for GaussianState.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gaussian_state import (
    GaussianState,
    init_gaussians_random,
    quaternion_to_rotation_matrix,
)


class TestGaussianState:
    def test_random_init_shape(self):
        gs = init_gaussians_random(100, color_dim=3, device="cpu")
        assert gs.means.shape == (100, 3)
        assert gs.scales.shape == (100, 3)
        assert gs.rotations.shape == (100, 4)
        assert gs.opacities.shape == (100, 1)
        assert gs.colors.shape == (100, 3)

    def test_num_gaussians(self):
        gs = init_gaussians_random(50, device="cpu")
        assert gs.num_gaussians == 50

    def test_activated_scales_positive(self):
        gs = init_gaussians_random(10, device="cpu")
        activated = gs.get_activated_scales()
        assert (activated > 0).all()

    def test_activated_opacities_range(self):
        gs = init_gaussians_random(10, device="cpu")
        opacities = gs.get_activated_opacities()
        assert (opacities >= 0).all() and (opacities <= 1).all()

    def test_activated_colors_range(self):
        gs = init_gaussians_random(10, device="cpu")
        colors = gs.get_activated_colors()
        assert (colors >= 0).all() and (colors <= 1).all()

    def test_state_dict_roundtrip(self):
        gs = init_gaussians_random(20, device="cpu")
        sd = gs.state_dict()
        gs2 = GaussianState.from_state_dict(sd)
        assert torch.allclose(gs.means, gs2.means)
        assert torch.allclose(gs.scales, gs2.scales)

    def test_detach(self):
        gs = init_gaussians_random(10, device="cpu")
        gs_det = gs.detach()
        assert not gs_det.means.requires_grad

    def test_quaternion_identity(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity
        R = quaternion_to_rotation_matrix(q)
        assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-6)

    def test_quaternion_batch(self):
        q = torch.randn(5, 4)
        R = quaternion_to_rotation_matrix(q)
        assert R.shape == (5, 3, 3)
        # Check orthogonality: R @ R^T ≈ I
        RRT = R @ R.transpose(-1, -2)
        I = torch.eye(3).expand(5, 3, 3)
        assert torch.allclose(RRT, I, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
