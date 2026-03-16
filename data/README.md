# Data Directory

Place your scene data here. Expected structure per scene:

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

## Kaggle Users

If running on Kaggle, place data at:

```
/kaggle/input/dataset/
  images/
  cameras.json
  coarse_init.ply
```

Then update `data_dir` in your config YAML.
