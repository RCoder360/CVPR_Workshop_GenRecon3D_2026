"""
Export Gaussians to PLY format for visualization.

Exports fields: x, y, z, scale_x, scale_y, scale_z, opacity, red, green, blue
"""

from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np
import torch

from ..models.gaussian_state import GaussianState
from ..datasets.base_dataset import CameraInfo


def export_gaussians_ply(
    gaussians: GaussianState,
    output_path: str,
) -> str:
    """Export Gaussian parameters to a PLY file.

    Parameters
    ----------
    gaussians : GaussianState
    output_path : str — path to save the .ply file

    Returns
    -------
    output_path : str
    """
    from plyfile import PlyData, PlyElement

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    N = gaussians.num_gaussians
    means = gaussians.means.detach().cpu().numpy()           # (N, 3)
    scales = gaussians.get_activated_scales().detach().cpu().numpy()  # (N, 3)
    opacities = gaussians.get_activated_opacities().detach().cpu().numpy().squeeze(-1)  # (N,)
    colors = gaussians.get_activated_colors().detach().cpu().numpy()  # (N, 3)

    # Clamp colors to [0, 255] uint8
    colors_u8 = (colors * 255).clip(0, 255).astype(np.uint8)

    # Build structured array
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_x", "f4"), ("scale_y", "f4"), ("scale_z", "f4"),
        ("opacity", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    arr = np.empty(N, dtype=dtype)
    arr["x"] = means[:, 0]
    arr["y"] = means[:, 1]
    arr["z"] = means[:, 2]
    arr["scale_x"] = scales[:, 0]
    arr["scale_y"] = scales[:, 1]
    arr["scale_z"] = scales[:, 2]
    arr["opacity"] = opacities
    arr["red"] = colors_u8[:, 0]
    arr["green"] = colors_u8[:, 1]
    arr["blue"] = colors_u8[:, 2]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=True).write(output_path)
    print(f"[export] Saved {N} Gaussians → {output_path}")
    return output_path


def export_cameras_json(
    cameras: list[CameraInfo],
    output_path: str,
) -> str:
    """Export camera parameters to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames = []
    for i, cam in enumerate(cameras):
        frames.append({
            "index": i,
            "transform_matrix": cam.c2w.detach().cpu().tolist(),
            "fx": cam.fx,
            "fy": cam.fy,
            "cx": cam.cx,
            "cy": cam.cy,
            "width": cam.width,
            "height": cam.height,
            "image_path": cam.image_path,
        })

    with open(output_path, "w") as f:
        json.dump({"frames": frames}, f, indent=2)
    print(f"[export] Saved {len(cameras)} cameras → {output_path}")
    return output_path


def export_scene(
    gaussians: GaussianState,
    cameras: list[CameraInfo],
    metrics: Optional[dict] = None,
    output_dir: str = "outputs/scene_01",
) -> str:
    """Export a full scene (PLY + cameras + metrics).

    Creates:
        output_dir/
            gaussians.ply
            cameras.json
            metrics.json
    """
    os.makedirs(output_dir, exist_ok=True)

    ply_path = os.path.join(output_dir, "gaussians.ply")
    export_gaussians_ply(gaussians, ply_path)

    cam_path = os.path.join(output_dir, "cameras.json")
    export_cameras_json(cameras, cam_path)

    if metrics is not None:
        met_path = os.path.join(output_dir, "metrics.json")
        with open(met_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[export] Saved metrics → {met_path}")

    print(f"[export] Scene exported → {output_dir}")
    return output_dir
