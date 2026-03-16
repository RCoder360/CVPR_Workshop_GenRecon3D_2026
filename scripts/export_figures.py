"""
export_figures.py — Generate publication-quality figures from outputs.

Usage:
    python scripts/export_figures.py --output-dir outputs/kan_app_geom
    python scripts/export_figures.py --ablation-dir outputs/ablation
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_ablation_comparison(ablation_dir: str, output_path: str):
    """Create a bar chart comparing ablation experiments."""
    results_path = os.path.join(ablation_dir, "ablation_results.json")
    if not os.path.isfile(results_path):
        print(f"[warn] No ablation results found at {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    names = list(results.keys())
    psnrs = [results[n].get("best_psnr", 0) for n in names]
    params = [results[n].get("total_parameters", 0) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR comparison
    colors = ["#4a90d9", "#7c6aff", "#00d4aa"]
    bars1 = ax1.bar(names, psnrs, color=colors[:len(names)], edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.set_title("Reconstruction Quality", fontsize=14, fontweight="bold")
    ax1.set_ylim(bottom=max(0, min(psnrs) - 2))
    for bar, val in zip(bars1, psnrs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="x", rotation=15)

    # Parameter count comparison
    params_k = [p / 1000 for p in params]
    bars2 = ax2.bar(names, params_k, color=colors[:len(names)], edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Parameters (K)", fontsize=12)
    ax2.set_title("Model Size", fontsize=14, fontweight="bold")
    for bar, val in zip(bars2, params_k):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}K", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig] Ablation comparison → {output_path}")


def plot_training_curves_from_csv(output_dir: str, fig_path: str):
    """Plot training curves from a metrics CSV."""
    import csv

    csv_path = os.path.join(output_dir, "metrics.csv")
    if not os.path.isfile(csv_path):
        print(f"[warn] No metrics.csv found at {csv_path}")
        return

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    keys = [k for k in rows[0].keys() if k]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        vals = [float(r[key]) for r in rows if r.get(key, "")]
        ax.plot(vals, linewidth=1.5, color="#7c6aff")
        ax.set_title(key, fontsize=12, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig] Training curves → {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Export publication figures")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Single experiment output directory")
    parser.add_argument("--ablation-dir", type=str, default=None,
                        help="Ablation results directory")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures",
                        help="Where to save generated figures")
    args = parser.parse_args()

    fig_dir = os.path.join(_project_root, args.figures_dir)
    os.makedirs(fig_dir, exist_ok=True)

    if args.output_dir:
        plot_training_curves_from_csv(
            args.output_dir,
            os.path.join(fig_dir, "training_curves.png"),
        )

    if args.ablation_dir:
        plot_ablation_comparison(
            args.ablation_dir,
            os.path.join(fig_dir, "ablation_comparison.png"),
        )

    if not args.output_dir and not args.ablation_dir:
        # Default: try to find outputs
        default_dirs = [
            "outputs/kan_app_geom",
            "outputs/kan_app",
            "outputs/baseline_sh",
            "outputs",
        ]
        for d in default_dirs:
            full = os.path.join(_project_root, d)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "metrics.csv")):
                plot_training_curves_from_csv(full, os.path.join(fig_dir, f"curves_{Path(d).name}.png"))

        abl_dir = os.path.join(_project_root, "outputs", "ablation")
        if os.path.isdir(abl_dir):
            plot_ablation_comparison(abl_dir, os.path.join(fig_dir, "ablation_comparison.png"))

    print(f"\n✓ Figures exported to {fig_dir}")


if __name__ == "__main__":
    main()
