"""
Multi-view dataset loader.

Reads a folder containing:
    images/          — RGB images
    cameras.json     — camera parameters
    depth/ (opt)     — depth maps
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base_dataset import BaseDataset, CameraInfo, ViewData


class MultiViewDataset(BaseDataset):
    """Loads multi-view images and cameras from a directory.

    Expected ``cameras.json`` schema::

        {
          "frames": [
            {
              "file_path": "images/000.png",
              "transform_matrix": [[...], ...],  // 4×4, camera-to-world
              "fx": 500, "fy": 500,
              "cx": 200, "cy": 200,
              "width": 400, "height": 400
            },
            ...
          ]
        }
    """

    def __init__(
        self,
        data_dir: str,
        image_width: int = 400,
        image_height: int = 400,
        white_background: bool = True,
        max_views: int = -1,
        device: str = "cpu",
    ):
        self.data_dir = Path(data_dir)
        self.image_width = image_width
        self.image_height = image_height
        self.white_background = white_background
        self.device = device

        # Load camera metadata
        cam_path = self.data_dir / "cameras.json"
        if not cam_path.exists():
            # Generate synthetic data for demo / debugging
            self._views: list[ViewData] = self._generate_synthetic(
                num_views=8, device=device
            )
        else:
            with open(cam_path, "r") as f:
                meta = json.load(f)
            frames = meta["frames"]
            if max_views > 0:
                frames = frames[:max_views]
            self._views = []
            for i, fr in enumerate(frames):
                cam = self._parse_camera(fr)
                img = self._load_image(fr["file_path"])
                depth = self._load_depth(fr.get("depth_path"))
                self._views.append(ViewData(camera=cam, image=img, depth=depth, index=i))

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._views)

    def __getitem__(self, idx: int) -> ViewData:
        return self._views[idx]

    def get_all_cameras(self) -> list[CameraInfo]:
        return [v.camera for v in self._views]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _parse_camera(self, frame: dict) -> CameraInfo:
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        w2c = torch.linalg.inv(c2w)
        return CameraInfo(
            c2w=c2w, w2c=w2c,
            fx=frame.get("fx", 500.0),
            fy=frame.get("fy", 500.0),
            cx=frame.get("cx", self.image_width / 2.0),
            cy=frame.get("cy", self.image_height / 2.0),
            width=frame.get("width", self.image_width),
            height=frame.get("height", self.image_height),
            image_path=frame.get("file_path"),
        )

    def _load_image(self, rel_path: str) -> torch.Tensor:
        full = self.data_dir / rel_path
        if not full.exists():
            # Fallback: random image for debugging
            return torch.rand(3, self.image_height, self.image_width)
        img = Image.open(full).convert("RGBA" if self.white_background else "RGB")
        img = img.resize((self.image_width, self.image_height), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        if self.white_background and arr.shape[-1] == 4:
            alpha = arr[..., 3:4]
            arr = arr[..., :3] * alpha + (1.0 - alpha)
        else:
            arr = arr[..., :3]
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _load_depth(self, rel_path: Optional[str]) -> Optional[torch.Tensor]:
        if rel_path is None:
            return None
        full = self.data_dir / rel_path
        if not full.exists():
            return None
        depth = np.array(Image.open(full), dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)

    # ------------------------------------------------------------------ #
    # Synthetic data fallback (for quick testing without real images)
    # ------------------------------------------------------------------ #
    def _generate_synthetic(self, num_views: int = 8, device: str = "cpu") -> list[ViewData]:
        """Create a tiny synthetic multi-view dataset of a colored cube."""
        views: list[ViewData] = []
        H, W = self.image_height, self.image_width
        fx = fy = 500.0
        cx, cy = W / 2.0, H / 2.0
        radius = 3.0

        for i in range(num_views):
            angle = 2.0 * np.pi * i / num_views
            eye = np.array([radius * np.cos(angle), 0.5, radius * np.sin(angle)])
            target = np.array([0.0, 0.0, 0.0])
            up = np.array([0.0, 1.0, 0.0])

            forward = target - eye
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-8)
            up_actual = np.cross(right, forward)

            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right
            c2w[:3, 1] = up_actual
            c2w[:3, 2] = -forward
            c2w[:3, 3] = eye

            c2w_t = torch.tensor(c2w, dtype=torch.float32)
            w2c_t = torch.linalg.inv(c2w_t)

            cam = CameraInfo(
                c2w=c2w_t, w2c=w2c_t,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=W, height=H,
            )

            # Generate a simple synthetic image (gradient pattern)
            u = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
            v = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
            r_ch = (u * 0.5 + 0.3 * (i / num_views))
            g_ch = (v * 0.5 + 0.2)
            b_ch = torch.full((H, W), 0.3 + 0.4 * (i / num_views))
            image = torch.stack([r_ch, g_ch, b_ch], dim=0).clamp(0, 1)

            views.append(ViewData(camera=cam, image=image, index=i))
        return views
