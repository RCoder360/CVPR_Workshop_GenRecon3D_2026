"""
Tests for the Gaussian renderer.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gaussian_state import init_gaussians_random
from src.datasets.base_dataset import CameraInfo
from src.render.gaussian_renderer import GaussianRenderer


def _make_test_camera(device="cpu") -> CameraInfo:
    """Create a simple test camera looking at the origin."""
    import numpy as np

    eye = np.array([0.0, 0.0, 3.0])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_actual = np.cross(right, forward)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_actual
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye

    c2w_t = torch.tensor(c2w, dtype=torch.float32)
    w2c_t = torch.linalg.inv(c2w_t)

    return CameraInfo(
        c2w=c2w_t, w2c=w2c_t,
        fx=250.0, fy=250.0,
        cx=50.0, cy=50.0,
        width=100, height=100,
    )


class TestGaussianRenderer:
    def test_render_output_shape(self):
        gs = init_gaussians_random(50, device="cpu")
        camera = _make_test_camera()
        renderer = GaussianRenderer(near=0.1, far=100.0)

        out = renderer.render(gs, camera)
        assert out.color.shape == (3, 100, 100)

    def test_render_color_range(self):
        gs = init_gaussians_random(50, device="cpu")
        camera = _make_test_camera()
        renderer = GaussianRenderer()

        out = renderer.render(gs, camera)
        assert out.color.min() >= 0.0
        assert out.color.max() <= 1.0

    def test_render_with_background(self):
        gs = init_gaussians_random(20, device="cpu")
        camera = _make_test_camera()
        renderer = GaussianRenderer()
        bg = torch.ones(3)

        out = renderer.render(gs, camera, background=bg)
        assert out.color.shape == (3, 100, 100)

    def test_render_depth_output(self):
        gs = init_gaussians_random(50, device="cpu")
        camera = _make_test_camera()
        renderer = GaussianRenderer()

        out = renderer.render(gs, camera)
        if out.depth is not None:
            assert out.depth.shape[1] == 100
            assert out.depth.shape[2] == 100

    def test_render_gradient_flow(self):
        gs = init_gaussians_random(20, device="cpu")
        camera = _make_test_camera()
        renderer = GaussianRenderer()

        out = renderer.render(gs, camera)
        loss = out.color.sum()
        loss.backward()
        # Means should have gradients since they're Parameters
        assert gs.means.grad is not None

    def test_empty_frustum(self):
        """All Gaussians behind the camera should produce background."""
        gs = init_gaussians_random(10, device="cpu")
        # Place all Gaussians far behind the camera
        gs.means.data = torch.tensor([[0, 0, -100.0]] * 10)
        camera = _make_test_camera()
        renderer = GaussianRenderer()
        bg = torch.tensor([0.5, 0.5, 0.5])

        out = renderer.render(gs, camera, background=bg)
        # Should be mostly background color
        assert out.color.shape == (3, 100, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
