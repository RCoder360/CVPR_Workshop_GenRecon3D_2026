"""
evaluate.py — Evaluate a trained KAN-Refine checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt --config configs/kan_app_geom.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.config import load_config
from src.datasets.multiview_dataset import MultiViewDataset
from src.train.trainer import Trainer
from src.utils.metrics import evaluate_all
from src.utils.visualization import save_image_comparison, save_depth_map
from src.utils.export_gaussians import export_scene
from src.render.gaussian_renderer import GaussianRenderer


def main():
    parser = argparse.ArgumentParser(description="KAN-Refine Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the checkpoint .pt file")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (auto-detected from checkpoint dir if omitted)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save evaluation results")
    parser.add_argument("--export-scene", action="store_true", default=True,
                        help="Export Gaussian scene after evaluation")
    args = parser.parse_args()

    # Resolve config
    if args.config is None:
        ckpt_dir = Path(args.checkpoint).parent.parent
        config_path = ckpt_dir / "config.yaml"
        if not config_path.exists():
            config_path = os.path.join(_project_root, "configs", "default.yaml")
    else:
        config_path = args.config

    cfg = load_config(str(config_path))

    # Dataset
    data_dir = cfg.data.get("data_dir", "data/scene_01")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_project_root, data_dir)

    dataset = MultiViewDataset(
        data_dir=data_dir,
        image_width=cfg.data.get("image_width", 400),
        image_height=cfg.data.get("image_height", 400),
        white_background=cfg.data.get("white_background", True),
        max_views=cfg.data.get("num_views_eval", -1),
    )
    print(f"[eval] Loaded {len(dataset)} views")

    # Build trainer and load checkpoint
    trainer = Trainer(cfg, dataset)
    trainer.load_checkpoint(args.checkpoint)

    device = trainer.device
    renderer = trainer.renderer
    gaussians = trainer.gaussians
    appearance_model = trainer.appearance_model
    use_kan = trainer.use_kan

    background = (
        torch.ones(3, device=device)
        if cfg.data.get("white_background", True)
        else torch.zeros(3, device=device)
    )

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.checkpoint).parent.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    all_metrics = []
    lpips_fn = None

    print(f"\n{'='*60}")
    print(f"  Evaluating on {len(dataset)} views")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for idx in range(len(dataset)):
            view = dataset[idx]
            gt_image = view.image.to(device)

            cam_pos = view.camera.c2w[:3, 3].to(device)
            view_dirs = gaussians.means - cam_pos.unsqueeze(0)
            view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

            app_out = appearance_model(gaussians.means, view_dirs)
            if use_kan:
                gaussians.colors.data = torch.logit(app_out["rgb"].clamp(1e-5, 1 - 1e-5))
                if app_out.get("opacity") is not None:
                    gaussians.opacities.data = torch.logit(app_out["opacity"].clamp(1e-5, 1 - 1e-5))

            render_out = renderer.render(gaussians, view.camera, background)
            pred_image = render_out.color

            metrics = evaluate_all(
                pred_image, gt_image,
                pred_depth=render_out.depth,
                gt_depth=view.depth.to(device) if view.depth is not None else None,
                lpips_fn=lpips_fn,
                compute_lpips_flag=cfg.eval.get("compute_lpips", True),
            )
            all_metrics.append(metrics)

            # Save renders
            save_image_comparison(
                pred_image, gt_image,
                str(out_dir / f"comparison_{idx:03d}.png"),
                title=f"View {idx}  |  PSNR: {metrics['psnr']:.2f}  SSIM: {metrics['ssim']:.4f}",
            )
            if render_out.depth is not None:
                save_depth_map(render_out.depth, str(out_dir / f"depth_{idx:03d}.png"))

            print(f"  View {idx:3d}: PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}")

    # Aggregate
    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if key in m]
        agg[f"mean_{key}"] = sum(vals) / len(vals)

    print(f"\n{'='*60}")
    print(f"  Aggregate Results")
    print(f"{'='*60}")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")
    print()

    # Save metrics
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[eval] Metrics saved → {out_dir / 'eval_metrics.json'}")

    # Export scene
    if args.export_scene:
        scene_dir = str(out_dir / "scene_export")
        cameras = dataset.get_all_cameras()
        export_scene(gaussians, cameras, agg, scene_dir)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
