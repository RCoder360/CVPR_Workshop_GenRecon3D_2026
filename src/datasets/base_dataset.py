"""
Base dataset interface for KAN-Refine.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CameraInfo:
    """Stores a single camera's parameters."""
    # Extrinsics: camera-to-world 4×4
    c2w: torch.Tensor           # (4, 4)
    # Derived: world-to-camera
    w2c: torch.Tensor           # (4, 4)
    # Intrinsics
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # Optional
    image_path: Optional[str] = None


@dataclass
class ViewData:
    """A single training / evaluation sample."""
    camera: CameraInfo
    image: torch.Tensor          # (3, H, W), float [0, 1]
    depth: Optional[torch.Tensor] = None  # (1, H, W)
    mask: Optional[torch.Tensor] = None   # (1, H, W)
    index: int = 0


class BaseDataset(abc.ABC):
    """Abstract dataset interface."""

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> ViewData: ...

    @abc.abstractmethod
    def get_all_cameras(self) -> list[CameraInfo]: ...
