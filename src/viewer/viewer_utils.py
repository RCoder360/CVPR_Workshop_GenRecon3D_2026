"""
Viewer utilities — PLY loading, camera orbit generation.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def load_ply_as_numpy(ply_path: str) -> dict[str, np.ndarray]:
    """Load a Gaussian PLY file into numpy arrays.

    Returns
    -------
    dict with keys: "positions", "scales", "opacities", "colors"
    """
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    v = ply["vertex"]

    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    scales = None
    try:
        scales = np.stack([v["scale_x"], v["scale_y"], v["scale_z"]], axis=-1).astype(np.float32)
    except (ValueError, KeyError):
        pass

    opacities = None
    try:
        opacities = np.array(v["opacity"], dtype=np.float32)
    except (ValueError, KeyError):
        pass

    colors = None
    try:
        r = np.array(v["red"], dtype=np.float32) / 255.0
        g = np.array(v["green"], dtype=np.float32) / 255.0
        b = np.array(v["blue"], dtype=np.float32) / 255.0
        colors = np.stack([r, g, b], axis=-1)
    except (ValueError, KeyError):
        pass

    return {
        "positions": positions,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }


def load_cameras_json(json_path: str) -> list[dict]:
    """Load cameras.json and return list of camera dicts."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("frames", [])


def generate_orbit_cameras(
    center: np.ndarray,
    radius: float = 3.0,
    num_views: int = 36,
    height: float = 0.5,
) -> list[np.ndarray]:
    """Generate orbit camera extrinsics (4×4 c2w) around a center point.

    Returns list of 4×4 camera-to-world matrices.
    """
    cameras = []
    for i in range(num_views):
        angle = 2.0 * np.pi * i / num_views
        eye = center + np.array([radius * np.cos(angle), height, radius * np.sin(angle)])
        forward = center - eye
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up_actual = np.cross(right, forward)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up_actual
        c2w[:3, 2] = -forward
        c2w[:3, 3] = eye
        cameras.append(c2w)
    return cameras
