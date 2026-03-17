# KAN-Refine: Resource-Efficient Geometric Refinement using Kolmogorov-Arnold Networks

> **CVPR GenRecon3D Workshop Submission**

A geometry-aware refinement pipeline that refines coarse 3D initializations using a lightweight KAN-based attribute network inside a differentiable 3D Gaussian Splatting rendering loop.

---

## 🎯 Research Hypothesis

1. A compact **Kolmogorov-Arnold Network (KAN)** can model view-dependent color, opacity, and small residual geometric corrections more **parameter-efficiently** than standard per-Gaussian SH / MLP parameterizations.
2. With **depth-aware geometric regularization**, the method improves geometric fidelity from noisy coarse initializations while using **fewer parameters and lower memory**.

---

## 📁 Repository Structure

```
kan_refine/
├── configs/                       # Experiment configurations
│   ├── default.yaml               # Full default config
│   ├── baseline_sh.yaml           # SH baseline (no KAN)
│   ├── kan_app.yaml               # KAN appearance only
│   └── kan_app_geom.yaml          # KAN appearance + geometry residuals
├── data/                          # Dataset directory
│   └── README.md                  # Data format documentation
├── src/
│   ├── datasets/                  # Data loading
│   │   ├── base_dataset.py        # Abstract dataset interface
│   │   └── multiview_dataset.py   # Multi-view image loader
│   ├── models/                    # Core models
│   │   ├── gaussian_state.py      # Gaussian parameter state
│   │   ├── kan_layers.py          # KAN B-spline layers
│   │   ├── kan_network.py         # KAN appearance network
│   │   └── sh_baseline.py         # SH color baseline
│   ├── render/                    # Rendering
│   │   ├── renderer_interface.py  # Abstract renderer
│   │   └── gaussian_renderer.py   # PyTorch Gaussian splatting
│   ├── losses/                    # Loss functions
│   │   └── losses.py              # All loss terms
│   ├── train/                     # Training
│   │   └── trainer.py             # Main training loop
│   ├── utils/                     # Utilities
│   │   ├── config.py              # Config loader
│   │   ├── metrics.py             # PSNR, SSIM, LPIPS, etc.
│   │   ├── efficiency.py          # Parameter counting, VRAM tracking
│   │   ├── export_gaussians.py    # PLY scene export
│   │   └── visualization.py       # Image/depth/curve saving
│   └── viewer/                    # Interactive 3D viewer
│       ├── viewer.py              # Open3D + HTML viewer
│       └── viewer_utils.py        # PLY loading, camera utils
├── scripts/                       # Entry point scripts
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # Evaluation & metrics
│   ├── export_scene.py            # Export Gaussian scene
│   ├── run_ablation.py            # Run ablation experiments
│   └── export_figures.py          # Generate publication figures
├── tests/                         # Unit tests
│   ├── test_gaussian_state.py
│   ├── test_kan.py
│   └── test_renderer.py
├── outputs/                       # Training outputs (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train on Synthetic Data (no dataset needed)

```bash
python scripts/train.py --config configs/default.yaml
```

The system auto-generates synthetic multi-view data if no real dataset is found. This is perfect for verifying the pipeline works.

### 3. Train with Real Data

Place your data in `data/scene_01/` following the format in `data/README.md`, then:

```bash
python scripts/train.py --config configs/kan_app_geom.yaml --data-dir data/scene_01
```

NeRF Synthetic scenes (for example LEGO) are now supported directly with the standard folder structure:

```text
data/lego/
  train/
  test/
  val/
  transforms_train.json
  transforms_test.json
  transforms_val.json
```

Use them the same way, by pointing `--data-dir` to the scene root:

```bash
python scripts/train.py --config configs/kan_app_geom.yaml --data-dir data/lego
```

### 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
```

### 5. Export Scene

```bash
python scripts/export_scene.py --checkpoint outputs/checkpoints/best.pt
```

### 6. Interactive Viewer

```bash
# Open3D viewer (native window)
python src/viewer/viewer.py --scene outputs/scene_01/gaussians.ply

# HTML/WebGL viewer (works anywhere, including Kaggle)
python src/viewer/viewer.py --scene outputs/scene_01/gaussians.ply --mode html
```

### 7. Run Ablation Study

```bash
python scripts/run_ablation.py --num-iterations 2000
```

### 8. Export Publication Figures

```bash
python scripts/export_figures.py --ablation-dir outputs/ablation
```

---

## 🔬 Experiment Configurations

| Config | Model | Geometry Residuals | Depth Loss |
|--------|-------|--------------------|------------|
| `baseline_sh.yaml` | SH (deg 3) | ✗ | ✗ |
| `kan_app.yaml` | KAN | ✗ | ✗ |
| `kan_app_geom.yaml` | KAN | ✓ | ✓ |

---

## 📊 Metrics Tracked

| Category | Metrics |
|----------|---------|
| **Image Quality** | PSNR, SSIM, LPIPS |
| **Geometry** | Chamfer Distance, Depth RMSE, Normal Consistency |
| **Efficiency** | Parameter Count, Peak VRAM (MB), Training Time |

---

## 🖥️ Interactive Viewer

The viewer supports two modes:

### Open3D Viewer (Default)

Launches a native window with:
- **Left mouse drag** → Rotate
- **Scroll wheel** → Zoom
- **Middle mouse** → Pan
- **R** → Reset view
- **Q / Esc** → Quit

```bash
python src/viewer/viewer.py --scene outputs/scene_01/gaussians.ply
```

### HTML / WebGL Viewer

Generates a self-contained HTML file with Three.js. Works in any browser, including Kaggle/Colab.

```bash
python src/viewer/viewer.py --scene outputs/scene_01/gaussians.ply --mode html
```

---

## 📋 Data Format

The loader supports two formats and auto-detects them:

1. Project-native multiview format (`cameras.json` present)
2. NeRF Synthetic format (`transforms_train.json` present)

### Directory Structure

```
data/scene_01/
├── images/          # RGB images (*.png or *.jpg)
├── cameras.json     # Camera parameters
├── coarse_init.ply  # (Optional) Coarse point cloud
└── depth/           # (Optional) Depth maps
```

### `cameras.json` Schema

```json
{
  "frames": [
    {
      "file_path": "images/000.png",
      "transform_matrix": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 3],
        [0, 0, 0, 1]
      ],
      "fx": 500.0, "fy": 500.0,
      "cx": 200.0, "cy": 200.0,
      "width": 400, "height": 400
    }
  ]
}
```

### NeRF Synthetic Schema (`transforms_*.json`)

For NeRF Synthetic scenes, camera intrinsics are computed from `camera_angle_x`:

```text
focal = 0.5 * width / tan(camera_angle_x / 2)
fx = fy = focal
cx = width / 2
cy = height / 2
```

Each frame contributes:

- `file_path` (resolved to `<scene_root>/<file_path>.png`)
- `transform_matrix` as camera-to-world pose

Splits are loaded from:

- `train` -> `transforms_train.json`
- `test` -> `transforms_test.json`
- `val` -> `transforms_val.json`

Programmatic usage example:

```python
from src.datasets.multiview_dataset import MultiViewDataset

dataset = MultiViewDataset("data/lego", split="train")
```

---

## 📓 Kaggle Execution Workflow

### Step 1: Install Dependencies

```python
!pip install torch torchvision
!pip install lpips
!pip install open3d
!pip install plyfile
!pip install pyyaml
!pip install scikit-image
!pip install matplotlib
!pip install tqdm
```

### Step 2: Clone Repository

```python
!git clone https://github.com/YOUR_USERNAME/kan_refine.git
%cd kan_refine
```

### Step 3: Install Project Requirements

```python
!pip install -r requirements.txt
```

### Step 4: Prepare Dataset

```python
# Option A: Use Kaggle dataset
# Upload your dataset and it will be in /kaggle/input/dataset/
# Update config to point to it:

import yaml
with open('configs/kan_app_geom.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['data']['data_dir'] = '/kaggle/input/dataset'
with open('configs/kan_app_geom.yaml', 'w') as f:
    yaml.dump(cfg, f)

# Option B: Use synthetic data (no dataset needed)
# Just run training — it auto-generates synthetic data
```

### Step 5: Train Model

```python
!python scripts/train.py --config configs/kan_app_geom.yaml
```

### Step 6: Evaluate

```python
!python scripts/evaluate.py --checkpoint outputs/kan_app_geom/checkpoints/best.pt
```

### Step 7: Export Scene

```python
!python scripts/export_scene.py --checkpoint outputs/kan_app_geom/checkpoints/best.pt
```

### Step 8: Launch Viewer

```python
# For Kaggle (no desktop), use HTML mode:
!python src/viewer/viewer.py --scene outputs/kan_app_geom/scene_01/gaussians.ply --mode html

# Then display in notebook:
from IPython.display import HTML, display
with open('outputs/kan_app_geom/scene_01/viewer.html', 'r') as f:
    display(HTML(f.read()))
```

### Step 9: Run Ablation (Optional)

```python
!python scripts/run_ablation.py --num-iterations 2000
!python scripts/export_figures.py --ablation-dir outputs/ablation
```

### Step 10: Download Results

```python
# Zip outputs for download
import shutil
shutil.make_archive('/kaggle/working/kan_refine_results', 'zip', 'outputs')
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 📐 Pipeline Architecture

```
INPUT                          PROCESS                        OUTPUT
─────                          ───────                        ──────
Multi-view Images    ──►  ┌──────────────────┐         Rendered Images
Camera Parameters    ──►  │  KAN-Refine      │  ──►    Geometry Metrics
Coarse 3D Init       ──►  │  Pipeline:       │  ──►    Interactive Viewer
                          │                  │  ──►    Exported Scene
                          │  1. Gaussian     │
                          │     State Init   │
                          │  2. KAN Network  │
                          │     (color+geom) │
                          │  3. Diff. Render │
                          │  4. Loss Optim   │
                          └──────────────────┘
```

---

## 🔧 Key Design Decisions

1. **B-Spline KAN**: Each edge in the KAN layer uses learnable B-spline basis functions with a residual linear shortcut (SiLU activation path).

2. **Fallback MLP**: If KAN is too slow or for ablation, switch to `use_fallback_mlp=True` for a standard Linear+SiLU architecture.

3. **Pure PyTorch Renderer**: The splatting renderer is implemented entirely in PyTorch for maximum portability. For production, swap in `gsplat`.

4. **Dual Viewer**: Open3D for local development, HTML/Three.js for headless environments (Kaggle, Colab, SSH).

---

## 📄 Citation

```bibtex
@inproceedings{kan_refine_2025,
  title     = {KAN-Refine: Resource-Efficient Geometric Refinement 
               using Kolmogorov-Arnold Networks},
  author    = {Your Name},
  booktitle = {CVPR GenRecon3D Workshop},
  year      = {2025}
}
```

---

## 📝 License

This project is for research purposes. See LICENSE for details.
