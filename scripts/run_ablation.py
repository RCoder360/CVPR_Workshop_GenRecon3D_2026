"""
run_ablation.py — Run ablation experiments comparing KAN vs SH baseline.

Trains each configuration sequentially and produces a comparison table.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --configs configs/baseline_sh.yaml configs/kan_app.yaml configs/kan_app_geom.yaml
    python scripts/run_ablation.py --num-iterations 2000
"""

import argparse
import json
import os
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.config import load_config
from src.datasets.multiview_dataset import MultiViewDataset
from src.train.trainer import Trainer


DEFAULT_CONFIGS = [
    "configs/baseline_sh.yaml",
    "configs/kan_app.yaml",
    "configs/kan_app_geom.yaml",
]


def main():
    parser = argparse.ArgumentParser(description="KAN-Refine Ablation Study")
    parser.add_argument(
        "--configs", nargs="+", default=DEFAULT_CONFIGS,
        help="List of config YAML files to compare",
    )
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/ablation")
    args = parser.parse_args()

    results = {}

    for i, config_rel in enumerate(args.configs):
        config_path = os.path.join(_project_root, config_rel)
        if not os.path.isfile(config_path):
            print(f"[warn] Config not found: {config_path}, skipping.")
            continue

        cfg = load_config(config_path)
        exp_name = cfg.project.get("name", f"experiment_{i}")

        # Override output dir
        exp_output = os.path.join(_project_root, args.output_dir, exp_name)
        cfg.project["output_dir"] = exp_output

        if args.num_iterations:
            cfg.train["num_iterations"] = args.num_iterations

        print(f"\n{'='*60}")
        print(f"  Ablation [{i+1}/{len(args.configs)}]: {exp_name}")
        print(f"  Config: {config_rel}")
        print(f"{'='*60}\n")

        # Dataset
        data_dir = cfg.data.get("data_dir", "data/scene_01")
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(_project_root, data_dir)

        dataset = MultiViewDataset(
            data_dir=data_dir,
            image_width=cfg.data.get("image_width", 400),
            image_height=cfg.data.get("image_height", 400),
            white_background=cfg.data.get("white_background", True),
        )

        # Train
        trainer = Trainer(cfg, dataset)
        trainer.train()

        # Collect results
        metrics_path = os.path.join(exp_output, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r") as f:
                results[exp_name] = json.load(f)
        else:
            results[exp_name] = {"best_psnr": trainer.best_psnr}

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<30} {'PSNR':>8} {'Params':>12} {'VRAM (MB)':>10} {'Time (s)':>10}")
    print(f"  {'-'*70}")

    for name, m in results.items():
        psnr = m.get("best_psnr", 0)
        params = m.get("total_parameters", 0)
        vram = m.get("peak_vram_mb", 0)
        time_s = m.get("total_train_time_s", 0)
        print(f"  {name:<30} {psnr:>8.2f} {params:>12,} {vram:>10.1f} {time_s:>10.1f}")

    print(f"{'='*70}\n")

    # Save to JSON
    out_path = os.path.join(_project_root, args.output_dir, "ablation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ablation] Results saved → {out_path}")


if __name__ == "__main__":
    main()
