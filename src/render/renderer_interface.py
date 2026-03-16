"""
Renderer Interface — abstract base for swappable rendering backends.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch

from ..models.gaussian_state import GaussianState
from ..datasets.base_dataset import CameraInfo


@dataclass
class RenderOutput:
    """Output of a single render call."""
    color: torch.Tensor          # (3, H, W)
    depth: Optional[torch.Tensor] = None  # (1, H, W)
    alpha: Optional[torch.Tensor] = None  # (1, H, W)
    # For densification
    radii: Optional[torch.Tensor] = None  # (N,)
    viewspace_points: Optional[torch.Tensor] = None  # (N, 2)


class RendererInterface(abc.ABC):
    """Abstract renderer interface."""

    @abc.abstractmethod
    def render(
        self,
        gaussians: GaussianState,
        camera: CameraInfo,
        background: Optional[torch.Tensor] = None,
    ) -> RenderOutput:
        ...
