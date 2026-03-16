"""
KAN Layers — Kolmogorov-Arnold Network building blocks.

Implements a B-spline-based KAN layer following the KAN paper
(Liu et al., 2024).  Each edge has a learnable univariate function
parameterized as a linear combination of B-spline basis functions.

Fallback: if any dependency is missing, the module transparently falls
back to a standard Linear layer with SiLU activation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== #
# B-Spline utilities
# ====================================================================== #

def _b_spline_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """Evaluate B-spline basis of order *k* on knot vector *grid*.

    Parameters
    ----------
    x    : (batch, in_features) — input values
    grid : (in_features, G+2k+1) — knot positions per input feature
    k    : spline order (degree)

    Returns
    -------
    bases : (batch, in_features, G+k) — basis function evaluations
    """
    x = x.unsqueeze(-1)  # (B, in, 1)
    # grid: (in, num_knots)
    # bases order 0
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()  # (B, in, num_intervals)

    for p in range(1, k + 1):
        left_num = x - grid[:, :-(p + 1)]
        left_den = grid[:, p:-1] - grid[:, :-(p + 1)]
        right_num = grid[:, (p + 1):] - x
        right_den = grid[:, (p + 1):] - grid[:, 1:(-p if p > 0 else None)]

        left = left_num / (left_den + 1e-8) * bases[:, :, :-1]
        right = right_num / (right_den + 1e-8) * bases[:, :, 1:]
        bases = left + right

    return bases  # (B, in, num_basis)


class KANLinear(nn.Module):
    """A single KAN layer mapping ``in_features → out_features``.

    Each edge (i, j) learns a univariate spline φ_{i,j}(x_i).
    The output is:  y_j = Σ_i  φ_{i,j}(x_i)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_activation: nn.Module | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Knot vector per input feature
        num_knots = grid_size + 2 * spline_order + 1
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            num_knots,
        )
        # (in_features, num_knots) — same grid for every input dim
        self.register_buffer("grid", grid.unsqueeze(0).expand(in_features, -1).contiguous())

        num_basis = grid_size + spline_order
        # Spline coefficients
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * 0.1
        )
        # Residual linear shortcut (silu(x) path)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))
        )
        self.base_activation = base_activation or nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, in_features)

        Returns
        -------
        y : (batch, out_features)
        """
        # Base path: linear(silu(x))
        base_out = F.linear(self.base_activation(x), self.base_weight)

        # Spline path
        bases = _b_spline_basis(x, self.grid, self.spline_order)  # (B, in, num_basis)
        # Einsum: (out, in, basis) × (batch, in, basis) → (batch, out)
        spline_out = torch.einsum("oik,bik->bo", self.spline_weight, bases)

        return base_out + spline_out

    def regularization_loss(self) -> torch.Tensor:
        """L1 sparsity penalty on spline coefficients."""
        return self.spline_weight.abs().mean()


# ====================================================================== #
# Fallback: plain MLP layer (used if KAN is too slow or for ablation)
# ====================================================================== #

class MLPLinear(nn.Module):
    """Simple Linear + activation, used as KAN fallback."""

    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))

    def regularization_loss(self) -> torch.Tensor:
        return self.linear.weight.abs().mean() * 0.0  # no special reg
