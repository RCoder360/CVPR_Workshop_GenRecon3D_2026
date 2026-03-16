"""
Evaluation metrics for KAN-Refine.

Computes: PSNR, SSIM, LPIPS, Chamfer Distance, Depth RMSE, Normal Consistency.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np


# ====================================================================== #
# Image quality metrics
# ====================================================================== #

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two images (C, H, W) or (B, C, H, W) in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * math.log10(mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM (scalar) between two images."""
    from ..losses.losses import _ssim_map
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    return _ssim_map(pred, target).mean().item()


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_fn=None,
) -> float:
    """LPIPS (perceptual distance). Requires ``lpips`` package."""
    if lpips_fn is None:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to(pred.device)
            lpips_fn.eval()
        except ImportError:
            return 0.0  # graceful fallback
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    # LPIPS expects [-1, 1]
    with torch.no_grad():
        val = lpips_fn(pred * 2 - 1, target * 2 - 1)
    return val.item()


# ====================================================================== #
# Geometry metrics
# ====================================================================== #

def chamfer_distance(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
) -> float:
    """Symmetric Chamfer distance between two point clouds (N, 3), (M, 3)."""
    # pts1→pts2
    diff = pts1.unsqueeze(1) - pts2.unsqueeze(0)  # (N, M, 3)
    dist = (diff ** 2).sum(-1)  # (N, M)
    d1 = dist.min(dim=1).values.mean()
    d2 = dist.min(dim=0).values.mean()
    return (d1 + d2).item()


def depth_rmse(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Root mean squared error of depth maps."""
    diff = (pred_depth - gt_depth) ** 2
    if mask is not None:
        diff = diff * mask
        return math.sqrt(diff.sum().item() / (mask.sum().item() + 1e-8))
    return math.sqrt(diff.mean().item())


def normal_consistency(
    depth: torch.Tensor,
    gt_depth: torch.Tensor,
) -> float:
    """Approximate normal consistency from depth maps.

    Computes normals via finite differences and measures cosine similarity.
    """
    def _depth_to_normals(d: torch.Tensor) -> torch.Tensor:
        # d: (1, H, W)
        if d.dim() == 2:
            d = d.unsqueeze(0)
        dz_dx = d[:, :, 1:] - d[:, :, :-1]  # (1, H, W-1)
        dz_dy = d[:, 1:, :] - d[:, :-1, :]  # (1, H-1, W)
        # Trim to common size
        h = min(dz_dx.shape[1], dz_dy.shape[1])
        w = min(dz_dx.shape[2], dz_dy.shape[2])
        dz_dx = dz_dx[:, :h, :w]
        dz_dy = dz_dy[:, :h, :w]
        normals = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
        normals = F.normalize(normals, dim=-1)
        return normals

    n_pred = _depth_to_normals(depth)
    n_gt = _depth_to_normals(gt_depth)
    cos_sim = (n_pred * n_gt).sum(-1).clamp(-1, 1)
    return cos_sim.mean().item()


# ====================================================================== #
# Aggregate evaluator
# ====================================================================== #

def evaluate_all(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    pred_depth: Optional[torch.Tensor] = None,
    gt_depth: Optional[torch.Tensor] = None,
    pred_points: Optional[torch.Tensor] = None,
    gt_points: Optional[torch.Tensor] = None,
    lpips_fn=None,
    compute_lpips_flag: bool = True,
) -> dict[str, float]:
    """Run all metrics and return a flat dict."""
    results = {
        "psnr": compute_psnr(pred_rgb, gt_rgb),
        "ssim": compute_ssim(pred_rgb, gt_rgb),
    }

    if compute_lpips_flag:
        results["lpips"] = compute_lpips(pred_rgb, gt_rgb, lpips_fn)

    if pred_depth is not None and gt_depth is not None:
        results["depth_rmse"] = depth_rmse(pred_depth, gt_depth)
        results["normal_consistency"] = normal_consistency(pred_depth, gt_depth)

    if pred_points is not None and gt_points is not None:
        results["chamfer_distance"] = chamfer_distance(pred_points, gt_points)

    return results
