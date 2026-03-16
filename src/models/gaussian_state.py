"""
Gaussian State — stores all per-Gaussian parameters.

Each Gaussian is represented by:
    - mean      (N, 3)   : xyz center
    - scale     (N, 3)   : log-space scale  (sx, sy, sz)
    - rotation  (N, 4)   : quaternion        (w, x, y, z)
    - opacity   (N, 1)   : logit-space opacity
    - color     (N, C)   : SH coefficients or raw RGB (C = 3 for raw)

This module is deliberately framework-agnostic at the *logic* level;
all tensors are plain ``torch.Tensor`` with ``requires_grad=True``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GaussianState:
    """Container for all Gaussian splat parameters.

    All tensors live on the same device and have shape ``(N, ...)``.
    """
    means: torch.Tensor           # (N, 3)
    scales: torch.Tensor          # (N, 3)   — log-scale
    rotations: torch.Tensor       # (N, 4)   — quaternion wxyz
    opacities: torch.Tensor       # (N, 1)   — logit
    colors: torch.Tensor          # (N, C)   — SH or RGB
    # Optional auxiliary
    features: Optional[torch.Tensor] = None  # (N, F) extra features

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

    def get_activated_scales(self) -> torch.Tensor:
        """Exponentiate log-scales → positive scales."""
        return torch.exp(self.scales)

    def get_activated_opacities(self) -> torch.Tensor:
        """Sigmoid of logit-opacities → [0, 1]."""
        return torch.sigmoid(self.opacities)

    def get_activated_colors(self) -> torch.Tensor:
        """Sigmoid of raw colors → [0, 1] RGB."""
        return torch.sigmoid(self.colors)

    def get_rotation_matrices(self) -> torch.Tensor:
        """Convert quaternions to 3×3 rotation matrices.

        Returns (N, 3, 3).
        """
        return quaternion_to_rotation_matrix(self.rotations)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, torch.Tensor]:
        d = {
            "means": self.means,
            "scales": self.scales,
            "rotations": self.rotations,
            "opacities": self.opacities,
            "colors": self.colors,
        }
        if self.features is not None:
            d["features"] = self.features
        return d

    @classmethod
    def from_state_dict(cls, d: dict[str, torch.Tensor]) -> "GaussianState":
        return cls(
            means=d["means"],
            scales=d["scales"],
            rotations=d["rotations"],
            opacities=d["opacities"],
            colors=d["colors"],
            features=d.get("features"),
        )

    def detach(self) -> "GaussianState":
        """Return a detached copy (no grad)."""
        return GaussianState(
            means=self.means.detach(),
            scales=self.scales.detach(),
            rotations=self.rotations.detach(),
            opacities=self.opacities.detach(),
            colors=self.colors.detach(),
            features=self.features.detach() if self.features is not None else None,
        )


# ====================================================================== #
# Initialization helpers
# ====================================================================== #

def init_gaussians_random(
    num_points: int,
    color_dim: int = 3,
    init_scale: float = 0.01,
    init_opacity: float = 0.8,
    spatial_range: float = 1.0,
    device: str = "cuda",
) -> GaussianState:
    """Create a randomly initialized Gaussian state."""
    means = (torch.rand(num_points, 3, device=device) - 0.5) * 2.0 * spatial_range
    means = nn.Parameter(means)

    scales = torch.full((num_points, 3), math.log(init_scale), device=device)
    scales = nn.Parameter(scales)

    rotations = torch.zeros(num_points, 4, device=device)
    rotations[:, 0] = 1.0  # identity quaternion
    rotations = nn.Parameter(rotations)

    opacity_logit = torch.logit(torch.tensor(init_opacity)).item()
    opacities = torch.full((num_points, 1), opacity_logit, device=device)
    opacities = nn.Parameter(opacities)

    colors = torch.rand(num_points, color_dim, device=device) * 0.5
    colors = nn.Parameter(colors)

    return GaussianState(
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
    )


def init_gaussians_from_ply(
    ply_path: str,
    color_dim: int = 3,
    init_scale: float = 0.01,
    init_opacity: float = 0.8,
    device: str = "cuda",
) -> GaussianState:
    """Initialize Gaussians from a PLY point-cloud file.

    Expects vertices with (x, y, z) at minimum;
    optionally (red, green, blue) in [0, 255].
    """
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    verts = ply["vertex"]
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=-1).astype(np.float32)
    num = xyz.shape[0]

    means = nn.Parameter(torch.tensor(xyz, device=device))

    scales = nn.Parameter(
        torch.full((num, 3), math.log(init_scale), device=device)
    )

    rotations = torch.zeros(num, 4, device=device)
    rotations[:, 0] = 1.0
    rotations = nn.Parameter(rotations)

    opacity_logit = torch.logit(torch.tensor(init_opacity)).item()
    opacities = nn.Parameter(
        torch.full((num, 1), opacity_logit, device=device)
    )

    # Try to load colors from PLY
    try:
        r = np.array(verts["red"], dtype=np.float32) / 255.0
        g = np.array(verts["green"], dtype=np.float32) / 255.0
        b = np.array(verts["blue"], dtype=np.float32) / 255.0
        rgb = np.stack([r, g, b], axis=-1)
        # Store as inverse-sigmoid so sigmoid(colors) = rgb
        rgb_t = torch.tensor(rgb, device=device).clamp(1e-5, 1.0 - 1e-5)
        colors = nn.Parameter(torch.logit(rgb_t))
    except (ValueError, KeyError):
        colors = nn.Parameter(torch.rand(num, color_dim, device=device) * 0.5)

    return GaussianState(
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
    )


# ====================================================================== #
# Quaternion utilities
# ====================================================================== #

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to rotation matrix (3, 3).

    Parameters
    ----------
    q : Tensor of shape (..., 4)

    Returns
    -------
    R : Tensor of shape (..., 3, 3)
    """
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R
