"""
train.py — Main training entry point for KAN-Refine.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/kan_app_geom.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on the path
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.config import load_config
from src.datasets.multiview_dataset import MultiViewDataset
from src.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="KAN-Refine Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=None,
        help="Override number of training iterations",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda / cpu)",
    )
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(_project_root, args.config) if not os.path.isabs(args.config) else args.config
    cfg = load_config(config_path)

    # Apply overrides
    if args.data_dir:
        cfg.data["data_dir"] = args.data_dir
    if args.output_dir:
        cfg.project["output_dir"] = args.output_dir
    if args.num_iterations:
        cfg.train["num_iterations"] = args.num_iterations
    if args.device:
        cfg.project["device"] = args.device

    # Resolve data directory relative to project root
    data_dir = cfg.data.get("data_dir", "data/scene_01")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_project_root, data_dir)

    # Create dataset
    dataset = MultiViewDataset(
        data_dir=data_dir,
        image_width=cfg.data.get("image_width", 400),
        image_height=cfg.data.get("image_height", 400),
        white_background=cfg.data.get("white_background", True),
        max_views=cfg.data.get("num_views_train", -1),
    )
    print(f"[data] Loaded {len(dataset)} views from {data_dir}")

    # Create trainer and start training
    trainer = Trainer(cfg, dataset)
    trainer.train()

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
