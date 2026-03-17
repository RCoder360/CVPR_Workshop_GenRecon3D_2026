"""
Gaussian Renderer — differentiable splatting in pure PyTorch.

This is a simplified "tile-based" 2D Gaussian splatting renderer suitable for
research prototyping.  It projects 3D Gaussians to 2D, sorts by depth, and
alpha-composites front-to-back.

For production quality, swap in gsplat or diff-gaussian-rasterization.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..models.gaussian_state import GaussianState
from ..datasets.base_dataset import CameraInfo
from .renderer_interface import RendererInterface, RenderOutput


class GaussianRenderer(RendererInterface):
    """Pure-PyTorch differentiable Gaussian splatting renderer."""

    def __init__(
        self,
        near: float = 0.1,
        far: float = 100.0,
        chunk_size: int = 128,
        render_devices: Optional[list[int | str]] = None,
    ):
        self.near = near
        self.far = far
        self.chunk_size = max(1, int(chunk_size))
        self.render_devices = render_devices

    def render(
        self,
        gaussians: GaussianState,
        camera: CameraInfo,
        background: Optional[torch.Tensor] = None,
    ) -> RenderOutput:
        """Render Gaussians from the given camera.

        Parameters
        ----------
        gaussians : GaussianState
        camera    : CameraInfo
        background : (3,) tensor or None (defaults to black)

        Returns
        -------
        RenderOutput
        """
        device = gaussians.device
        H, W = camera.height, camera.width

        if background is None:
            background = torch.zeros(3, device=device)
        background = background.to(device)

        # -------------------------------------------------------------- #
        # 1. World → Camera transform
        # -------------------------------------------------------------- #
        w2c = camera.w2c.to(device)  # (4, 4)
        means_h = F.pad(gaussians.means, (0, 1), value=1.0)  # (N, 4)
        means_cam = (w2c @ means_h.T).T[:, :3]  # (N, 3)

        # Depth is along -z in OpenGL convention; in our w2c it's z
        depths = means_cam[:, 2]  # (N,)

        # Filter by near/far
        valid = (depths > self.near) & (depths < self.far)
        if valid.sum() == 0:
            color = background.view(3, 1, 1).expand(3, H, W)
            return RenderOutput(color=color)

        # -------------------------------------------------------------- #
        # 2. Project to pixel coordinates
        # -------------------------------------------------------------- #
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy

        px = fx * means_cam[:, 0] / (means_cam[:, 2] + 1e-8) + cx  # (N,)
        py = fy * means_cam[:, 1] / (means_cam[:, 2] + 1e-8) + cy  # (N,)

        # -------------------------------------------------------------- #
        # 3. Compute 2D covariance (simplified: isotropic from scale)
        # -------------------------------------------------------------- #
        scales = gaussians.get_activated_scales()  # (N, 3)
        # Approximate 2D radius as max of xy scales projected
        scale_2d = (scales[:, :2].max(dim=-1).values * fx) / (depths + 1e-8)  # (N,)
        scale_2d = scale_2d.clamp(min=0.5)

        # -------------------------------------------------------------- #
        # 4. Sort by depth (front to back)
        # -------------------------------------------------------------- #
        sort_idx = depths.argsort()

        px = px[sort_idx]
        py = py[sort_idx]
        scale_2d = scale_2d[sort_idx]
        depths_sorted = depths[sort_idx]
        valid_sorted = valid[sort_idx]

        opacities = gaussians.get_activated_opacities().squeeze(-1)[sort_idx]  # (N,)
        colors = gaussians.get_activated_colors()[sort_idx]  # (N, 3)

        # -------------------------------------------------------------- #
        # 5. Rasterize via alpha compositing
        # -------------------------------------------------------------- #
        # We accumulate in a chunked fashion for memory efficiency
        color_acc = background.view(3, 1, 1).expand(3, H, W).clone()
        transmittance = torch.ones(1, H, W, device=device)

        # Process Gaussians in chunks
        chunk_size = min(self.chunk_size, px.shape[0])
        N = px.shape[0]

        depth_acc = torch.zeros(1, H, W, device=device)

        if self.render_devices and device.type == "cuda":
            chunk_devices = [torch.device(f"cuda:{int(d)}") if isinstance(d, int) else torch.device(d)
                             for d in self.render_devices]
        else:
            chunk_devices = [device]

        mesh_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

        def _meshgrid_for(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
            if dev not in mesh_cache:
                yy_t, xx_t = torch.meshgrid(
                    torch.arange(H, device=dev, dtype=torch.float32),
                    torch.arange(W, device=dev, dtype=torch.float32),
                    indexing="ij",
                )
                mesh_cache[dev] = (yy_t, xx_t)
            return mesh_cache[dev]

        for chunk_id, start in enumerate(range(0, N, chunk_size)):
            end = min(start + chunk_size, N)
            if not valid_sorted[start:end].any():
                continue

            chunk_device = chunk_devices[chunk_id % len(chunk_devices)]
            yy, xx = _meshgrid_for(chunk_device)

            px_c = px[start:end].to(chunk_device)          # (C,)
            py_c = py[start:end].to(chunk_device)
            s_c = scale_2d[start:end].to(chunk_device)     # (C,)
            o_c = opacities[start:end].to(chunk_device)    # (C,)
            col_c = colors[start:end].to(chunk_device)     # (C, 3)
            dep_c = depths_sorted[start:end].to(chunk_device)  # (C,)
            v_c = valid_sorted[start:end].to(chunk_device)

            transmittance_chunk = transmittance.to(chunk_device)
            color_chunk = torch.zeros(3, H, W, device=chunk_device)
            depth_chunk = torch.zeros(1, H, W, device=chunk_device)

            # Gaussian evaluation: exp(-0.5 * ((x-μ)/σ)^2)
            dx = xx.unsqueeze(0) - px_c.view(-1, 1, 1)   # (C, H, W)
            dy = yy.unsqueeze(0) - py_c.view(-1, 1, 1)   # (C, H, W)
            sigma = s_c.view(-1, 1, 1)

            gauss = torch.exp(-0.5 * (dx**2 + dy**2) / (sigma**2 + 1e-8))  # (C, H, W)
            alpha = (gauss * o_c.view(-1, 1, 1) * v_c.float().view(-1, 1, 1)).clamp(0, 0.99)

            for i in range(alpha.shape[0]):
                a = alpha[i:i+1]  # (1, H, W)
                weight = a * transmittance_chunk  # (1, H, W)
                color_chunk += weight * col_c[i].view(3, 1, 1)
                depth_chunk += weight * dep_c[i]
                transmittance_chunk = transmittance_chunk * (1.0 - a)

            color_acc = color_acc + color_chunk.to(device)
            depth_acc = depth_acc + depth_chunk.to(device)
            transmittance = transmittance_chunk.to(device)

        return RenderOutput(
            color=color_acc.clamp(0, 1),
            depth=depth_acc,
            alpha=1.0 - transmittance,
            radii=scale_2d,  # in original sort order via sort_idx
        )
