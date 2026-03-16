"""
KAN Network — full appearance + geometry refinement network.

Given per-Gaussian features (position + viewing direction), predicts:
  - RGB color (3)
  - opacity adjustment (1)
  - (optional) geometry residuals: delta_mean (3) + delta_scale (3)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .kan_layers import KANLinear, MLPLinear


# ====================================================================== #
# Positional encoding
# ====================================================================== #

class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding."""

    def __init__(self, input_dim: int, num_freqs: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)  # (L,)
        self.output_dim = input_dim * (1 + 2 * num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (..., input_dim)

        Returns
        -------
        encoded : (..., output_dim)  = input_dim * (1 + 2L)
        """
        parts = [x]
        for freq in self.freqs:
            parts.append(torch.sin(freq * math.pi * x))
            parts.append(torch.cos(freq * math.pi * x))
        return torch.cat(parts, dim=-1)


# ====================================================================== #
# KAN Appearance Network
# ====================================================================== #

class KANAppearanceNetwork(nn.Module):
    """KAN-based view-dependent color + opacity predictor.

    Optionally also predicts geometry residuals (delta_mean, delta_scale).
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list[int] | None = None,
        output_dim: int = 4,
        grid_size: int = 5,
        spline_order: int = 3,
        use_positional_encoding: bool = True,
        pe_num_freqs: int = 4,
        predict_geometry_residuals: bool = False,
        use_fallback_mlp: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.predict_geometry_residuals = predict_geometry_residuals
        self.use_positional_encoding = use_positional_encoding

        # Positional encoding
        if use_positional_encoding:
            self.pe = PositionalEncoding(input_dim, pe_num_freqs)
            actual_input = self.pe.output_dim
        else:
            self.pe = None
            actual_input = input_dim

        # Choose layer type
        LayerCls = MLPLinear if use_fallback_mlp else KANLinear
        layer_kwargs = {} if use_fallback_mlp else {
            "grid_size": grid_size,
            "spline_order": spline_order,
        }

        # Build trunk
        layers = []
        prev = actual_input
        for h in hidden_dims:
            layers.append(LayerCls(prev, h, **layer_kwargs))
            prev = h
        self.trunk = nn.ModuleList(layers)

        # Output heads
        # Color + opacity head
        self.color_head = nn.Linear(prev, 4)  # rgb(3) + opacity(1)

        # Geometry residual head (optional)
        if predict_geometry_residuals:
            self.geom_head = nn.Linear(prev, 6)  # delta_mean(3) + delta_scale(3)
        else:
            self.geom_head = None

    def forward(
        self,
        positions: torch.Tensor,
        view_dirs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        positions : (N, 3) — Gaussian centers
        view_dirs : (N, 3) — normalized viewing directions

        Returns
        -------
        dict with keys:
            "rgb"       : (N, 3) in [0, 1]
            "opacity"   : (N, 1) in [0, 1]
            "delta_mean"  : (N, 3)   (if geometry residuals enabled)
            "delta_scale" : (N, 3)   (if geometry residuals enabled)
        """
        x = torch.cat([positions, view_dirs], dim=-1)  # (N, 6)

        if self.pe is not None:
            x = self.pe(x)

        for layer in self.trunk:
            x = layer(x)

        # Color + opacity
        co = self.color_head(x)  # (N, 4)
        rgb = torch.sigmoid(co[:, :3])
        opacity = torch.sigmoid(co[:, 3:4])

        out = {"rgb": rgb, "opacity": opacity}

        # Geometry residuals
        if self.geom_head is not None:
            geom = self.geom_head(x)  # (N, 6)
            out["delta_mean"] = geom[:, :3] * 0.01   # small residuals
            out["delta_scale"] = geom[:, 3:] * 0.01
        return out

    def sparsity_loss(self) -> torch.Tensor:
        """Aggregate KAN spline sparsity loss across all layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.trunk:
            if hasattr(layer, "regularization_loss"):
                total = total + layer.regularization_loss()
        return total
