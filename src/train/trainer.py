"""
Trainer — main training loop for KAN-Refine.

Handles:
  - Gaussian parameter optimization
  - KAN / SH model optimization
  - Loss computation and logging
  - Periodic evaluation and checkpointing
  - Scene export
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..datasets.base_dataset import BaseDataset, ViewData
from ..models.gaussian_state import (
    GaussianState,
    init_gaussians_random,
    init_gaussians_from_ply,
)
from ..models.kan_network import KANAppearanceNetwork
from ..models.sh_baseline import SHColorModel
from ..render.gaussian_renderer import GaussianRenderer
from ..render.renderer_interface import RenderOutput
from ..losses.losses import KANRefineLoss
from ..utils.metrics import evaluate_all
from ..utils.efficiency import EfficiencyTracker, count_all_parameters
from ..utils.visualization import (
    save_image_comparison,
    save_depth_map,
    save_training_curves,
    save_single_image,
)
from ..utils.export_gaussians import export_scene
from ..utils.config import ConfigDict, save_config


class Trainer:
    """Main training loop for KAN-Refine."""

    def _appearance_module(self):
        """Return the underlying appearance model (unwrap DataParallel if needed)."""
        if isinstance(self.appearance_model, nn.DataParallel):
            return self.appearance_model.module
        return self.appearance_model

    def __init__(self, cfg: ConfigDict, dataset: BaseDataset):
        self.cfg = cfg
        self.dataset = dataset
        self.device = cfg.project.get("device", "cuda")
        if self.device == "cuda":
            self.device = "cuda:0"
        if not torch.cuda.is_available():
            self.device = "cpu"

        # Seed
        seed = cfg.project.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Output directory
        self.output_dir = Path(cfg.project.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        save_config(cfg, str(self.output_dir / "config.yaml"))

        # ------------------------------------------------------------ #
        # Initialize Gaussians
        # ------------------------------------------------------------ #
        g_cfg = cfg.gaussian
        if g_cfg.get("init_method") == "ply" and g_cfg.get("init_ply_path"):
            self.gaussians = init_gaussians_from_ply(
                g_cfg.init_ply_path,
                color_dim=3,
                init_scale=g_cfg.get("init_scale", 0.01),
                init_opacity=g_cfg.get("init_opacity", 0.8),
                device=self.device,
            )
        else:
            self.gaussians = init_gaussians_random(
                num_points=g_cfg.get("num_points", 10000),
                color_dim=3,
                init_scale=g_cfg.get("init_scale", 0.01),
                init_opacity=g_cfg.get("init_opacity", 0.8),
                device=self.device,
            )
        print(f"[init] {self.gaussians.num_gaussians} Gaussians on {self.device}")

        # ------------------------------------------------------------ #
        # Initialize appearance model (KAN or SH baseline)
        # ------------------------------------------------------------ #
        model_type = cfg.model.get("type", "kan")
        if model_type == "kan":
            kan_cfg = cfg.model.get("kan", {})
            self.appearance_model = KANAppearanceNetwork(
                input_dim=kan_cfg.get("input_dim", 6),
                hidden_dims=kan_cfg.get("hidden_dims", [64, 64]),
                output_dim=kan_cfg.get("output_dim", 4),
                grid_size=kan_cfg.get("grid_size", 5),
                spline_order=kan_cfg.get("spline_order", 3),
                use_positional_encoding=kan_cfg.get("use_positional_encoding", True),
                pe_num_freqs=kan_cfg.get("pe_num_freqs", 4),
                predict_geometry_residuals=kan_cfg.get("predict_geometry_residuals", False),
            ).to(self.device)
            self.use_kan = True
        else:
            self.appearance_model = SHColorModel(
                num_gaussians=self.gaussians.num_gaussians,
                sh_degree=cfg.model.get("sh_degree", 3),
                device=self.device,
            )
            self.use_kan = False

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.device.startswith("cuda") and self.num_gpus > 1 and cfg.project.get("use_data_parallel", True):
            print(f"[init] Using DataParallel on {self.num_gpus} GPUs for appearance model")
            self.appearance_model = nn.DataParallel(
                self.appearance_model,
                device_ids=list(range(self.num_gpus)),
            )
        print(f"[init] Model type: {model_type}")

        # ------------------------------------------------------------ #
        # Renderer
        # ------------------------------------------------------------ #
        r_cfg = cfg.get("render", {})
        render_chunk_size = r_cfg.get("chunk_size", 128)
        use_multi_gpu_render = r_cfg.get("multi_gpu", True)
        render_devices = None
        if self.device.startswith("cuda") and self.num_gpus > 1 and use_multi_gpu_render:
            render_devices = list(range(self.num_gpus))
            print(f"[init] Renderer chunk devices: {render_devices}")

        self.renderer = GaussianRenderer(
            near=r_cfg.get("near", 0.1),
            far=r_cfg.get("far", 100.0),
            chunk_size=render_chunk_size,
            render_devices=render_devices,
        )

        # ------------------------------------------------------------ #
        # Loss
        # ------------------------------------------------------------ #
        l_cfg = cfg.get("loss", {})
        self.loss_fn = KANRefineLoss(
            rgb_weight=l_cfg.get("rgb_weight", 1.0),
            rgb_type=l_cfg.get("rgb_type", "l1"),
            ssim_weight=l_cfg.get("ssim_weight", 0.2),
            depth_weight=l_cfg.get("depth_weight", 0.1),
            sparsity_weight=l_cfg.get("sparsity_weight", 0.001),
            geometry_weight=l_cfg.get("geometry_weight", 0.01),
            opacity_reg_weight=l_cfg.get("opacity_reg_weight", 0.001),
        )

        # ------------------------------------------------------------ #
        # Optimizer
        # ------------------------------------------------------------ #
        t_cfg = cfg.get("train", {})
        param_groups = [
            {"params": [self.gaussians.means], "lr": t_cfg.get("lr_means", 0.001), "name": "means"},
            {"params": [self.gaussians.scales], "lr": t_cfg.get("lr_scales", 0.005), "name": "scales"},
            {"params": [self.gaussians.rotations], "lr": t_cfg.get("lr_rotations", 0.005), "name": "rotations"},
            {"params": [self.gaussians.opacities], "lr": t_cfg.get("lr_opacities", 0.01), "name": "opacities"},
            {"params": [self.gaussians.colors], "lr": t_cfg.get("lr_colors", 0.01), "name": "colors"},
        ]
        if self.appearance_model is not None:
            param_groups.append({
                "params": self.appearance_model.parameters(),
                "lr": t_cfg.get("lr_kan", 0.001),
                "name": "appearance_model",
            })

        self.optimizer = torch.optim.Adam(param_groups)

        # LR scheduler
        decay_steps = t_cfg.get("lr_decay_steps", [3000, 4500])
        decay_rate = t_cfg.get("lr_decay_rate", 0.1)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=decay_steps, gamma=decay_rate,
        )

        # ------------------------------------------------------------ #
        # Tracking
        # ------------------------------------------------------------ #
        self.tracker = EfficiencyTracker()
        self.metrics_log: dict[str, list[float]] = {
            "loss_total": [], "loss_rgb": [], "psnr": [],
        }
        self.best_psnr = 0.0

        # Count params
        total_params = count_all_parameters(
            self.appearance_model,
            [self.gaussians.means, self.gaussians.scales,
             self.gaussians.rotations, self.gaussians.opacities,
             self.gaussians.colors],
        )
        print(f"[init] Total trainable parameters: {total_params:,}")

    # ================================================================== #
    # Training
    # ================================================================== #
    def train(self):
        """Run the full training loop."""
        t_cfg = self.cfg.get("train", {})
        num_iters = t_cfg.get("num_iterations", 5000)
        log_interval = t_cfg.get("log_interval", 100)
        save_interval = t_cfg.get("save_interval", 1000)
        eval_interval = t_cfg.get("eval_interval", 500)
        num_views = len(self.dataset)

        background = torch.ones(3, device=self.device) if self.cfg.data.get("white_background", True) else torch.zeros(3, device=self.device)

        print(f"\n{'='*60}")
        print(f"  Training KAN-Refine  |  {num_iters} iterations  |  {num_views} views")
        print(f"{'='*60}\n")

        self.tracker.reset_peak_memory()

        pbar = tqdm(range(1, num_iters + 1), desc="Training")
        for iteration in pbar:
            self.tracker.start_iteration()

            # Random view
            idx = random.randint(0, num_views - 1)
            view: ViewData = self.dataset[idx]
            gt_image = view.image.to(self.device)  # (3, H, W)

            # -------------------------------------------------------- #
            # Forward: appearance model
            # -------------------------------------------------------- #
            # Compute view direction (camera center → Gaussian means)
            cam_pos = view.camera.c2w[:3, 3].to(self.device)  # (3,)
            view_dirs = self.gaussians.means.detach() - cam_pos.unsqueeze(0)
            view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

            # Run appearance model
            app_out = self.appearance_model(self.gaussians.means, view_dirs)

            # Apply KAN outputs to Gaussian state
            if self.use_kan:
                # Override colors with KAN-predicted RGB
                kan_colors = app_out["rgb"]  # (N, 3) already in [0, 1]
                # Store as logit so get_activated_colors() works
                self.gaussians.colors.data = torch.logit(kan_colors.clamp(1e-5, 1-1e-5))

                if app_out.get("opacity") is not None:
                    kan_opacity = app_out["opacity"]
                    self.gaussians.opacities.data = torch.logit(kan_opacity.clamp(1e-5, 1-1e-5))

            # Apply geometry residuals if present
            delta_mean = app_out.get("delta_mean")
            delta_scale = app_out.get("delta_scale")
            orig_means = self.gaussians.means.clone()
            orig_scales = self.gaussians.scales.clone()

            if delta_mean is not None:
                self.gaussians.means.data = orig_means + delta_mean
            if delta_scale is not None:
                self.gaussians.scales.data = orig_scales + delta_scale

            # -------------------------------------------------------- #
            # Render
            # -------------------------------------------------------- #
            render_out: RenderOutput = self.renderer.render(
                self.gaussians, view.camera, background=background,
            )
            pred_image = render_out.color  # (3, H, W)

            # -------------------------------------------------------- #
            # Loss
            # -------------------------------------------------------- #
            app_module = self._appearance_module()
            sparsity_loss = app_module.sparsity_loss() if hasattr(app_module, 'sparsity_loss') else None

            losses = self.loss_fn(
                pred_rgb=pred_image,
                gt_rgb=gt_image,
                pred_depth=render_out.depth,
                gt_depth=view.depth.to(self.device) if view.depth is not None else None,
                opacities=self.gaussians.get_activated_opacities(),
                sparsity_loss=sparsity_loss,
                delta_mean=delta_mean,
                delta_scale=delta_scale,
            )

            total_loss = losses["total"]

            # -------------------------------------------------------- #
            # Backward + step
            # -------------------------------------------------------- #
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Restore original means/scales if we applied residuals
            if delta_mean is not None:
                self.gaussians.means.data = orig_means
            if delta_scale is not None:
                self.gaussians.scales.data = orig_scales

            self.tracker.end_iteration()

            # -------------------------------------------------------- #
            # Logging
            # -------------------------------------------------------- #
            loss_val = total_loss.item()
            self.metrics_log["loss_total"].append(loss_val)
            self.metrics_log["loss_rgb"].append(losses["rgb"].item())

            if iteration % log_interval == 0:
                with torch.no_grad():
                    psnr = -10 * torch.log10(
                        torch.nn.functional.mse_loss(pred_image, gt_image) + 1e-10
                    ).item()
                self.metrics_log["psnr"].append(psnr)
                pbar.set_postfix(loss=f"{loss_val:.4f}", psnr=f"{psnr:.2f}")

            # -------------------------------------------------------- #
            # Evaluation
            # -------------------------------------------------------- #
            if iteration % eval_interval == 0:
                self._evaluate(iteration, background)

            # -------------------------------------------------------- #
            # Checkpoint
            # -------------------------------------------------------- #
            if iteration % save_interval == 0:
                self._save_checkpoint(iteration)

        # Final evaluation & export
        print(f"\n{'='*60}")
        print("  Training complete — running final evaluation...")
        print(f"{'='*60}\n")
        self._evaluate(num_iters, background, final=True)
        self._save_checkpoint(num_iters, final=True)
        self._export(num_iters)
        self._save_logs()

    # ================================================================== #
    # Evaluation
    # ================================================================== #
    @torch.no_grad()
    def _evaluate(self, iteration: int, background: torch.Tensor, final: bool = False):
        """Run evaluation on all views."""
        psnrs, ssims = [], []
        renders_dir = self.output_dir / "renders" / f"iter_{iteration:06d}"
        renders_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(len(self.dataset)):
            view = self.dataset[idx]
            gt_image = view.image.to(self.device)

            cam_pos = view.camera.c2w[:3, 3].to(self.device)
            view_dirs = self.gaussians.means - cam_pos.unsqueeze(0)
            view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

            app_out = self.appearance_model(self.gaussians.means, view_dirs)
            if self.use_kan:
                self.gaussians.colors.data = torch.logit(app_out["rgb"].clamp(1e-5, 1-1e-5))
                if app_out.get("opacity") is not None:
                    self.gaussians.opacities.data = torch.logit(app_out["opacity"].clamp(1e-5, 1-1e-5))

            render_out = self.renderer.render(self.gaussians, view.camera, background)
            pred_image = render_out.color

            from ..utils.metrics import compute_psnr, compute_ssim
            psnrs.append(compute_psnr(pred_image, gt_image))
            ssims.append(compute_ssim(pred_image, gt_image))

            if final or idx < 4:  # Save first 4 views during training
                save_image_comparison(
                    pred_image, gt_image,
                    str(renders_dir / f"view_{idx:03d}.png"),
                    title=f"Iter {iteration} — View {idx}",
                )
                if render_out.depth is not None:
                    save_depth_map(
                        render_out.depth,
                        str(renders_dir / f"depth_{idx:03d}.png"),
                    )

        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)
        print(f"  [eval @ {iteration}] PSNR: {avg_psnr:.2f}  SSIM: {avg_ssim:.4f}")

        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self._save_checkpoint(iteration, best=True)

    # ================================================================== #
    # Checkpointing
    # ================================================================== #
    def _save_checkpoint(self, iteration: int, best: bool = False, final: bool = False):
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if best:
            path = ckpt_dir / "best.pt"
        elif final:
            path = ckpt_dir / "final.pt"
        else:
            path = ckpt_dir / f"iter_{iteration:06d}.pt"

        ckpt = {
            "iteration": iteration,
            "gaussian_state": self.gaussians.state_dict(),
            "appearance_model": self._appearance_module().state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        gs = GaussianState.from_state_dict(ckpt["gaussian_state"])
        self.gaussians = GaussianState(
            means=nn.Parameter(gs.means.to(self.device)),
            scales=nn.Parameter(gs.scales.to(self.device)),
            rotations=nn.Parameter(gs.rotations.to(self.device)),
            opacities=nn.Parameter(gs.opacities.to(self.device)),
            colors=nn.Parameter(gs.colors.to(self.device)),
        )
        self._appearance_module().load_state_dict(ckpt["appearance_model"])
        print(f"[ckpt] Loaded checkpoint from {path} (iteration {ckpt['iteration']})")

    # ================================================================== #
    # Export
    # ================================================================== #
    def _export(self, iteration: int):
        """Export the Gaussian scene."""
        scene_dir = str(self.output_dir / "scene_01")
        cameras = self.dataset.get_all_cameras()
        metrics = {
            "best_psnr": self.best_psnr,
            "total_iterations": iteration,
            **self.tracker.report(),
            "total_parameters": count_all_parameters(
                self.appearance_model,
                [self.gaussians.means, self.gaussians.scales,
                 self.gaussians.rotations, self.gaussians.opacities,
                 self.gaussians.colors],
            ),
        }
        export_scene(self.gaussians, cameras, metrics, scene_dir)

    # ================================================================== #
    # Log saving
    # ================================================================== #
    def _save_logs(self):
        """Save metrics CSV, JSON, and training curves."""
        # Training curves
        save_training_curves(
            self.metrics_log,
            str(self.output_dir / "training_curves.png"),
        )

        # Metrics CSV
        csv_path = self.output_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            keys = list(self.metrics_log.keys())
            writer.writerow(keys)
            max_len = max(len(v) for v in self.metrics_log.values())
            for i in range(max_len):
                row = []
                for k in keys:
                    vals = self.metrics_log[k]
                    row.append(vals[i] if i < len(vals) else "")
                writer.writerow(row)

        # Metrics JSON
        json_path = self.output_dir / "metrics.json"
        summary = {
            "best_psnr": self.best_psnr,
            **self.tracker.report(),
            "total_parameters": count_all_parameters(
                self.appearance_model,
                [self.gaussians.means, self.gaussians.scales,
                 self.gaussians.rotations, self.gaussians.opacities,
                 self.gaussians.colors],
            ),
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[log] Saved metrics → {csv_path}")
        print(f"[log] Saved training curves → {self.output_dir / 'training_curves.png'}")
