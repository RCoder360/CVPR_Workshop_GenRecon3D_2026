"""
SH Baseline — Spherical Harmonics color model (no KAN).

Used as the comparison baseline.  Each Gaussian has per-vertex SH coefficients
that are optimized directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def eval_sh(deg: int, sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical harmonics up to degree *deg*.

    Parameters
    ----------
    deg : int
        Maximum SH degree (0–3 supported).
    sh_coeffs : (N, C, (deg+1)^2)
        SH coefficients per Gaussian per color channel.
    dirs : (N, 3)
        Viewing directions (unit vectors).

    Returns
    -------
    colors : (N, C) — evaluated colors (unbounded; apply sigmoid externally)
    """
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
    C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
           0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435]

    result = C0 * sh_coeffs[..., 0]

    if deg >= 1:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result + C1 * (-y * sh_coeffs[..., 1] + z * sh_coeffs[..., 2] - x * sh_coeffs[..., 3])

    if deg >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (
            C2[0] * xy * sh_coeffs[..., 4] +
            C2[1] * yz * sh_coeffs[..., 5] +
            C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[..., 6] +
            C2[3] * xz * sh_coeffs[..., 7] +
            C2[4] * (xx - yy) * sh_coeffs[..., 8]
        )

    if deg >= 3:
        result = result + (
            C3[0] * y * (3 * xx - yy) * sh_coeffs[..., 9] +
            C3[1] * xy * z * sh_coeffs[..., 10] +
            C3[2] * y * (4 * zz - xx - yy) * sh_coeffs[..., 11] +
            C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[..., 12] +
            C3[4] * x * (4 * zz - xx - yy) * sh_coeffs[..., 13] +
            C3[5] * z * (xx - yy) * sh_coeffs[..., 14] +
            C3[6] * x * (xx - 3 * yy) * sh_coeffs[..., 15]
        )

    return result


class SHColorModel(nn.Module):
    """Direct SH color baseline — no neural network, purely per-Gaussian SH."""

    def __init__(self, num_gaussians: int, sh_degree: int = 3, device: str = "cuda"):
        super().__init__()
        self.sh_degree = sh_degree
        num_coeffs = (sh_degree + 1) ** 2
        # (N, 3, num_coeffs) — 3 color channels
        self.sh_coeffs = nn.Parameter(
            torch.randn(num_gaussians, 3, num_coeffs, device=device) * 0.1
        )

    def forward(
        self,
        positions: torch.Tensor,
        view_dirs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate SH colors for given view directions.

        Parameters
        ----------
        positions : (N, 3)  — unused, kept for API compatibility
        view_dirs : (N, 3)

        Returns
        -------
        dict with "rgb": (N, 3) in [0, 1], "opacity": None
        """
        raw = eval_sh(self.sh_degree, self.sh_coeffs, view_dirs)  # (N, 3)
        rgb = torch.sigmoid(raw)
        return {"rgb": rgb, "opacity": None}

    def sparsity_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.sh_coeffs.device)
