"""
Loss functions for KAN-Refine.

Total loss:
    L = L_rgb + λ_depth * L_depth + λ_sparse * L_sparse + λ_geom * L_geom + λ_opacity * L_opacity
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== #
# SSIM (structural similarity)
# ====================================================================== #

def _gaussian_kernel_1d(size: int, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()


def _ssim_map(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """Compute per-pixel SSIM between two (B, C, H, W) images."""
    device = img1.device
    k1d = _gaussian_kernel_1d(window_size, 1.5).to(device)
    k2d = (k1d.unsqueeze(-1) @ k1d.unsqueeze(0))
    C = img1.shape[-3]
    kernel = k2d.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=C)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.conv2d(img1**2, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=C) - mu1_mu2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
    )
    return ssim


class SSIMLoss(nn.Module):
    """1 - SSIM loss."""

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        ssim = _ssim_map(pred, target, self.window_size)
        return 1.0 - ssim.mean()


# ====================================================================== #
# Photometric loss
# ====================================================================== #

class PhotometricLoss(nn.Module):
    """Combined L1 / L2 / Huber + optional SSIM."""

    def __init__(self, rgb_type: str = "l1", ssim_weight: float = 0.2):
        super().__init__()
        self.rgb_type = rgb_type
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss() if ssim_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.rgb_type == "l1":
            pixel = F.l1_loss(pred, target)
        elif self.rgb_type == "l2":
            pixel = F.mse_loss(pred, target)
        elif self.rgb_type == "huber":
            pixel = F.smooth_l1_loss(pred, target)
        else:
            pixel = F.l1_loss(pred, target)

        loss = (1.0 - self.ssim_weight) * pixel
        if self.ssim_loss is not None:
            loss = loss + self.ssim_weight * self.ssim_loss(pred, target)
        return loss


# ====================================================================== #
# Depth consistency loss
# ====================================================================== #

class DepthLoss(nn.Module):
    """L1 depth loss between rendered and ground-truth depth maps."""

    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            diff = (pred_depth - gt_depth).abs() * mask
            return diff.sum() / (mask.sum() + 1e-8)
        return F.l1_loss(pred_depth, gt_depth)


# ====================================================================== #
# Geometry regularization
# ====================================================================== #

class GeometryRegularization(nn.Module):
    """Penalize large geometry residuals (delta_mean, delta_scale)."""

    def forward(
        self,
        delta_mean: Optional[torch.Tensor] = None,
        delta_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0)
        if delta_mean is not None:
            loss = loss.to(delta_mean.device)
            loss = loss + delta_mean.pow(2).mean()
        if delta_scale is not None:
            loss = loss.to(delta_scale.device)
            loss = loss + delta_scale.pow(2).mean()
        return loss


# ====================================================================== #
# Opacity regularization
# ====================================================================== #

class OpacityRegularization(nn.Module):
    """Encourage binary opacities (close to 0 or 1)."""

    def forward(self, opacities: torch.Tensor) -> torch.Tensor:
        # Entropy-like: -o * log(o) - (1-o) * log(1-o) encourages extremes
        o = opacities.clamp(1e-5, 1 - 1e-5)
        entropy = -(o * o.log() + (1 - o) * (1 - o).log())
        return entropy.mean()


# ====================================================================== #
# Combined loss
# ====================================================================== #

class KANRefineLoss(nn.Module):
    """Full loss function with all terms.

    L = L_rgb + λ_depth * L_depth + λ_sparse * L_sparse
        + λ_geom * L_geom + λ_opacity * L_opacity
    """

    def __init__(
        self,
        rgb_weight: float = 1.0,
        rgb_type: str = "l1",
        ssim_weight: float = 0.2,
        depth_weight: float = 0.1,
        sparsity_weight: float = 0.001,
        geometry_weight: float = 0.01,
        opacity_reg_weight: float = 0.001,
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.sparsity_weight = sparsity_weight
        self.geometry_weight = geometry_weight
        self.opacity_reg_weight = opacity_reg_weight

        self.photo_loss = PhotometricLoss(rgb_type, ssim_weight)
        self.depth_loss = DepthLoss()
        self.geom_reg = GeometryRegularization()
        self.opacity_reg = OpacityRegularization()

    def forward(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        pred_depth: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        opacities: Optional[torch.Tensor] = None,
        sparsity_loss: Optional[torch.Tensor] = None,
        delta_mean: Optional[torch.Tensor] = None,
        delta_scale: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss terms and return as a dict.

        Returns
        -------
        dict with keys "total", "rgb", "depth", "sparsity", "geometry", "opacity"
        """
        device = pred_rgb.device
        losses = {}

        # RGB loss
        losses["rgb"] = self.rgb_weight * self.photo_loss(pred_rgb, gt_rgb)

        # Depth loss
        if self.depth_weight > 0 and pred_depth is not None and gt_depth is not None:
            losses["depth"] = self.depth_weight * self.depth_loss(pred_depth, gt_depth)
        else:
            losses["depth"] = torch.tensor(0.0, device=device)

        # KAN sparsity loss
        if self.sparsity_weight > 0 and sparsity_loss is not None:
            losses["sparsity"] = self.sparsity_weight * sparsity_loss
        else:
            losses["sparsity"] = torch.tensor(0.0, device=device)

        # Geometry regularization
        if self.geometry_weight > 0 and (delta_mean is not None or delta_scale is not None):
            losses["geometry"] = self.geometry_weight * self.geom_reg(delta_mean, delta_scale)
        else:
            losses["geometry"] = torch.tensor(0.0, device=device)

        # Opacity regularization
        if self.opacity_reg_weight > 0 and opacities is not None:
            losses["opacity"] = self.opacity_reg_weight * self.opacity_reg(opacities)
        else:
            losses["opacity"] = torch.tensor(0.0, device=device)

        losses["total"] = sum(losses.values())
        return losses
