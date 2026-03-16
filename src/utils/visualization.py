"""
Visualization utilities — training curves, comparisons, depth maps.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def save_image_comparison(
    pred: torch.Tensor,
    target: torch.Tensor,
    output_path: str,
    title: str = "",
):
    """Save a side-by-side comparison of predicted vs ground truth images.

    Parameters
    ----------
    pred, target : (3, H, W) tensors in [0, 1]
    output_path : str
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pred_np = pred.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    tgt_np = target.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    diff = np.abs(pred_np - tgt_np)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tgt_np)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    axes[1].imshow(pred_np)
    axes[1].set_title("Predicted")
    axes[1].axis("off")
    axes[2].imshow(diff, vmin=0, vmax=1)
    axes[2].set_title("Abs Difference")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_depth_map(
    depth: torch.Tensor,
    output_path: str,
    title: str = "Depth",
    cmap: str = "viridis",
):
    """Save a depth map visualization.

    Parameters
    ----------
    depth : (1, H, W) or (H, W) tensor
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    d = depth.detach().cpu().squeeze().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(d, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(
    metrics_log: dict[str, list[float]],
    output_path: str,
):
    """Plot training curves from a metrics log.

    Parameters
    ----------
    metrics_log : dict mapping metric names → lists of values
    output_path : str
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_metrics = len(metrics_log)
    fig, axes = plt.subplots(1, max(n_metrics, 1), figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics_log.items()):
        ax.plot(values, linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_single_image(
    image: torch.Tensor,
    output_path: str,
):
    """Save a single (3, H, W) image tensor as PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_np = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    plt.imsave(output_path, img_np)
