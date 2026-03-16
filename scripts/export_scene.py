"""
export_scene.py — Export Gaussian scene from a checkpoint.

Usage:
    python scripts/export_scene.py --checkpoint outputs/checkpoints/best.pt
    python scripts/export_scene.py --checkpoint outputs/checkpoints/best.pt --output outputs/scene_01
"""

import argparse
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
from src.utils.export_gaussians import export_scene


def main():
    parser = argparse.ArgumentParser(description="Export Gaussian Scene")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for the scene")
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

    # Dataset (for camera info)
    data_dir = cfg.data.get("data_dir", "data/scene_01")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_project_root, data_dir)

    dataset = MultiViewDataset(
        data_dir=data_dir,
        image_width=cfg.data.get("image_width", 400),
        image_height=cfg.data.get("image_height", 400),
        white_background=cfg.data.get("white_background", True),
    )

    # Build trainer and load checkpoint
    trainer = Trainer(cfg, dataset)
    trainer.load_checkpoint(args.checkpoint)

    # Export
    output_dir = args.output or str(Path(args.checkpoint).parent.parent / "scene_01")
    cameras = dataset.get_all_cameras()

    export_scene(trainer.gaussians, cameras, output_dir=output_dir)

    print(f"\n✓ Scene exported to {output_dir}")
    print(f"  View with: python src/viewer/viewer.py --scene {output_dir}/gaussians.ply")


if __name__ == "__main__":
    main()
