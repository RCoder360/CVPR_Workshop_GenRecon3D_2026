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
import math
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
        split: str = "train",
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
        self.split = split
        self.cameras: list[CameraInfo] = []
        self.image_paths: list[str] = []

        # Load camera metadata
        cam_path = self.data_dir / "cameras.json"
        nerf_train_path = self.data_dir / "transforms_train.json"

        if cam_path.exists():
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
                self.cameras.append(cam)
                self.image_paths.append(str(self.data_dir / fr["file_path"]))
        elif nerf_train_path.exists():
            self._views = self.load_nerf_synthetic(self.data_dir, split=split)
            if max_views > 0:
                self._views = self._views[:max_views]
                self.cameras = self.cameras[:max_views]
                self.image_paths = self.image_paths[:max_views]
        else:
            # Generate synthetic data for demo / debugging
            self._views: list[ViewData] = self._generate_synthetic(
                num_views=8, device=device
            )
            self.cameras = [v.camera for v in self._views]
            self.image_paths = ["" for _ in self._views]

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

    def load_nerf_synthetic(self, scene_path: Path, split: str) -> list[ViewData]:
        """Load NeRF-synthetic data from transforms_<split>.json.

        Expected file layout:
            scene_path/
              transforms_train.json
              transforms_test.json
              transforms_val.json
              train/*.png, test/*.png, val/*.png
        """
        split = split.lower()
        if split not in {"train", "test", "val"}:
            raise ValueError(f"Unsupported split '{split}'. Use train/test/val.")

        transforms_path = scene_path / f"transforms_{split}.json"
        if not transforms_path.exists():
            raise FileNotFoundError(f"Missing NeRF transforms file: {transforms_path}")

        with open(transforms_path, "r") as f:
            meta = json.load(f)

        camera_angle_x = float(meta["camera_angle_x"])
        frames = meta.get("frames", [])

        width = self.image_width
        height = self.image_height
        focal = 0.5 * width / math.tan(camera_angle_x / 2.0)

        views: list[ViewData] = []
        self.cameras = []
        self.image_paths = []
        camera_centers: list[np.ndarray] = []

        for i, fr in enumerate(frames):
            rel_file = fr["file_path"]
            rel_file = rel_file[2:] if rel_file.startswith("./") else rel_file
            if rel_file.lower().endswith(".png"):
                rel_png = rel_file
            else:
                rel_png = rel_file + ".png"

            image_path = scene_path / rel_png

            # NeRF provides camera-to-world in OpenGL convention.
            # Convert explicitly to world-to-camera, then flip Y/Z axes
            # for OpenCV-style camera coordinates expected by this pipeline.
            c2w_nerf = np.array(fr["transform_matrix"], dtype=np.float32)
            R_c2w = c2w_nerf[:3, :3]
            t_c2w = c2w_nerf[:3, 3]

            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            # OpenGL (+X right, +Y up, +Z backward) -> OpenCV (+X right, +Y down, +Z forward)
            R_w2c[:, 1:3] *= -1.0
            t_w2c[1:3] *= -1.0

            w2c_np = np.eye(4, dtype=np.float32)
            w2c_np[:3, :3] = R_w2c
            w2c_np[:3, 3] = t_w2c

            # Keep c2w/w2c consistent for downstream code paths.
            c2w_np = np.linalg.inv(w2c_np)
            c2w = torch.tensor(c2w_np, dtype=torch.float32)
            w2c = torch.tensor(w2c_np, dtype=torch.float32)

            # Camera center debug check: C = -R^T t
            center = -R_w2c.T @ t_w2c
            camera_centers.append(center)

            cam = CameraInfo(
                c2w=c2w,
                w2c=w2c,
                fx=float(focal),
                fy=float(focal),
                cx=float(width / 2.0),
                cy=float(height / 2.0),
                width=width,
                height=height,
                image_path=str(image_path),
            )

            img = self._load_image(rel_png)
            views.append(ViewData(camera=cam, image=img, depth=None, index=i))

            self.cameras.append(cam)
            self.image_paths.append(str(image_path))

        if camera_centers:
            centers = np.stack(camera_centers, axis=0)
            unique_centers = np.unique(np.round(centers, 5), axis=0).shape[0]
            radii = np.linalg.norm(centers - centers.mean(axis=0, keepdims=True), axis=1)
            print(
                f"[data][nerf] split={split} views={len(centers)} unique_centers={unique_centers} "
                f"radius_mean={radii.mean():.4f} radius_std={radii.std():.4f}"
            )
            for i in range(min(3, len(centers))):
                print(f"[data][nerf] center[{i}] = {centers[i].tolist()}")

        return views

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
