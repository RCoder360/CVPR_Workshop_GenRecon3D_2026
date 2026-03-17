# Data Directory

Place your scene data here. The loader supports two scene formats and auto-detects them.

## Format A: Project Multiview (`cameras.json`)

Expected structure:

```
data/
  scene_01/
    images/          # Multi-view RGB images (*.png or *.jpg)
    cameras.json     # Camera intrinsics & extrinsics
    coarse_init.ply  # (Optional) Coarse initialization point cloud
    depth/           # (Optional) Depth maps for supervision
```

## `cameras.json` Format

```json
{
  "frames": [
    {
      "file_path": "images/000.png",
      "transform_matrix": [[...], [...], [...], [...]],
      "fx": 500.0,
      "fy": 500.0,
      "cx": 200.0,
      "cy": 200.0,
      "width": 400,
      "height": 400
    }
  ]
}
```

`transform_matrix` is a 4×4 camera-to-world matrix (OpenGL convention).

## Format B: NeRF Synthetic (`transforms_*.json`)

Standard NeRF synthetic structure (for example LEGO):

```
data/
  lego/
    train/
      r_0.png
      r_1.png
      ...
    test/
      r_0.png
      ...
    val/
      r_0.png
      ...
    transforms_train.json
    transforms_test.json
    transforms_val.json
```

### `transforms_<split>.json` Example

```json
{
  "camera_angle_x": 0.6911112,
  "frames": [
    {
      "file_path": "./train/r_0",
      "transform_matrix": [[...], [...], [...], [...]]
    }
  ]
}
```

For each frame:

- image path is resolved as `<scene_path>/<file_path>.png`
- `transform_matrix` is used as camera-to-world pose

Intrinsics are computed as:

```
focal = 0.5 * width / tan(camera_angle_x / 2)
fx = focal
fy = focal
cx = width / 2
cy = height / 2
```

Split mapping:

- `split="train"` -> `transforms_train.json`
- `split="test"` -> `transforms_test.json`
- `split="val"` -> `transforms_val.json`

Example:

```python
from src.datasets.multiview_dataset import MultiViewDataset

dataset = MultiViewDataset("data/lego", split="train")
```

## Kaggle Users

If running on Kaggle, place data at:

```
/kaggle/input/dataset/
  images/
  cameras.json
  coarse_init.ply
```

Then update `data_dir` in your config YAML.
