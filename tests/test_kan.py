"""
Tests for KAN layers and KAN network.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.kan_layers import KANLinear, MLPLinear
from src.models.kan_network import KANAppearanceNetwork, PositionalEncoding


class TestKANLinear:
    def test_forward_shape(self):
        layer = KANLinear(8, 16, grid_size=5, spline_order=3)
        x = torch.randn(4, 8)
        y = layer(x)
        assert y.shape == (4, 16)

    def test_gradient_flow(self):
        layer = KANLinear(4, 8)
        x = torch.randn(2, 4, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.spline_weight.grad is not None

    def test_regularization_loss(self):
        layer = KANLinear(4, 8)
        reg = layer.regularization_loss()
        assert reg.dim() == 0  # scalar
        assert reg.item() >= 0


class TestMLPLinear:
    def test_forward_shape(self):
        layer = MLPLinear(8, 16)
        x = torch.randn(4, 8)
        y = layer(x)
        assert y.shape == (4, 16)


class TestPositionalEncoding:
    def test_output_dim(self):
        pe = PositionalEncoding(3, num_freqs=4)
        assert pe.output_dim == 3 * (1 + 2 * 4)
        x = torch.randn(5, 3)
        y = pe(x)
        assert y.shape == (5, pe.output_dim)


class TestKANAppearanceNetwork:
    def test_forward_basic(self):
        net = KANAppearanceNetwork(
            input_dim=6, hidden_dims=[32, 32], output_dim=4,
            use_positional_encoding=True, pe_num_freqs=2,
            predict_geometry_residuals=False,
        )
        pos = torch.randn(10, 3)
        dirs = torch.randn(10, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        out = net(pos, dirs)
        assert "rgb" in out
        assert "opacity" in out
        assert out["rgb"].shape == (10, 3)
        assert out["opacity"].shape == (10, 1)
        assert (out["rgb"] >= 0).all() and (out["rgb"] <= 1).all()

    def test_geometry_residuals(self):
        net = KANAppearanceNetwork(
            input_dim=6, hidden_dims=[32], output_dim=10,
            predict_geometry_residuals=True,
        )
        pos = torch.randn(5, 3)
        dirs = torch.randn(5, 3)
        out = net(pos, dirs)
        assert "delta_mean" in out
        assert "delta_scale" in out
        assert out["delta_mean"].shape == (5, 3)
        assert out["delta_scale"].shape == (5, 3)

    def test_sparsity_loss(self):
        net = KANAppearanceNetwork(
            input_dim=6, hidden_dims=[16],
            use_positional_encoding=False,
        )
        loss = net.sparsity_loss()
        assert loss.dim() == 0

    def test_fallback_mlp(self):
        net = KANAppearanceNetwork(
            input_dim=6, hidden_dims=[32],
            use_fallback_mlp=True,
            use_positional_encoding=False,
        )
        pos = torch.randn(4, 3)
        dirs = torch.randn(4, 3)
        out = net(pos, dirs)
        assert out["rgb"].shape == (4, 3)

    def test_gradient_flow_kan(self):
        net = KANAppearanceNetwork(
            input_dim=6, hidden_dims=[16],
            use_positional_encoding=False,
        )
        pos = torch.randn(3, 3, requires_grad=True)
        dirs = torch.randn(3, 3, requires_grad=True)
        out = net(pos, dirs)
        loss = out["rgb"].sum() + out["opacity"].sum()
        loss.backward()
        assert pos.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
