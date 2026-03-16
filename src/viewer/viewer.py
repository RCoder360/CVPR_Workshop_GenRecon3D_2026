"""
Interactive 3D Viewer for KAN-Refine Gaussian scenes.

Supports:
  - Open3D point cloud viewer (default)
  - HTML/WebGL fallback viewer (for headless / Kaggle environments)

Usage:
    python src/viewer/viewer.py --scene outputs/scene_01/gaussians.ply
    python src/viewer/viewer.py --scene outputs/scene_01/gaussians.ply --mode html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

import numpy as np

# Add project root to path
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.viewer.viewer_utils import load_ply_as_numpy, load_cameras_json


# ====================================================================== #
# Open3D Viewer
# ====================================================================== #

def launch_open3d_viewer(
    ply_path: str,
    cameras_json_path: str | None = None,
    point_size: float = 2.0,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    window_width: int = 1280,
    window_height: int = 720,
):
    """Launch an interactive Open3D viewer.

    Controls:
      - Left mouse drag  : rotate
      - Scroll            : zoom
      - Middle mouse drag : pan
      - R                 : reset view
      - Q / Esc           : quit
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[viewer] Open3D not installed. Falling back to HTML viewer.")
        launch_html_viewer(ply_path, cameras_json_path)
        return

    data = load_ply_as_numpy(ply_path)
    positions = data["positions"]
    colors = data["colors"]
    opacities = data["opacities"]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))

    if colors is not None:
        if opacities is not None:
            # Modulate color by opacity for visual effect
            alpha = opacities[:, np.newaxis].clip(0, 1)
            vis_colors = colors * alpha + np.array(bg_color) * (1.0 - alpha)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors.astype(np.float64))
        else:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    else:
        # Default white
        pcd.colors = o3d.utility.Vector3dVector(
            np.ones_like(positions, dtype=np.float64) * 0.8
        )

    # Visualize
    print(f"\n{'='*60}")
    print(f"  KAN-Refine Interactive 3D Viewer")
    print(f"  Points: {len(positions):,}")
    print(f"{'='*60}")
    print(f"  Controls:")
    print(f"    Left Mouse Drag  → Rotate")
    print(f"    Scroll Wheel     → Zoom")
    print(f"    Middle Mouse     → Pan")
    print(f"    R                → Reset View")
    print(f"    Q / Esc          → Quit")
    print(f"{'='*60}\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="KAN-Refine Gaussian Scene",
        width=window_width,
        height=window_height,
    )
    vis.add_geometry(pcd)

    # Render options
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array(bg_color)
    opt.show_coordinate_frame = True

    # Add camera frustums if available
    if cameras_json_path and os.path.isfile(cameras_json_path):
        cameras = load_cameras_json(cameras_json_path)
        for cam in cameras:
            c2w = np.array(cam["transform_matrix"], dtype=np.float64)
            cam_pos = c2w[:3, 3]
            # Draw a small sphere at camera position
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(cam_pos)
            sphere.paint_uniform_color([1.0, 0.3, 0.3])
            vis.add_geometry(sphere)

    vis.run()
    vis.destroy_window()


# ====================================================================== #
# HTML / WebGL Viewer (fallback for headless environments)
# ====================================================================== #

def launch_html_viewer(
    ply_path: str,
    cameras_json_path: str | None = None,
    output_html: str | None = None,
    max_points: int = 50000,
):
    """Generate an HTML file with a Three.js point cloud viewer and open it.

    This works in any environment with a web browser, including Kaggle.
    """
    data = load_ply_as_numpy(ply_path)
    positions = data["positions"]
    colors = data["colors"]
    opacities = data["opacities"]

    # Subsample if too many points
    N = positions.shape[0]
    if N > max_points:
        indices = np.random.choice(N, max_points, replace=False)
        positions = positions[indices]
        if colors is not None:
            colors = colors[indices]
        if opacities is not None:
            opacities = opacities[indices]

    # Prepare JSON data for embedding
    pos_list = positions.tolist()
    if colors is not None:
        col_list = colors.tolist()
    else:
        col_list = [[0.8, 0.8, 0.8]] * len(pos_list)

    if opacities is not None:
        opa_list = opacities.tolist()
    else:
        opa_list = [1.0] * len(pos_list)

    html_content = _generate_threejs_html(pos_list, col_list, opa_list)

    if output_html is None:
        output_html = str(Path(ply_path).parent / "viewer.html")

    with open(output_html, "w") as f:
        f.write(html_content)

    print(f"[viewer] HTML viewer saved → {output_html}")

    # Try to open in browser
    try:
        webbrowser.open(f"file://{os.path.abspath(output_html)}")
    except Exception:
        print(f"[viewer] Open {output_html} in your browser to view the scene.")


def _generate_threejs_html(
    positions: list,
    colors: list,
    opacities: list,
) -> str:
    """Generate a self-contained HTML page with Three.js point cloud viewer."""
    import json as _json

    pos_json = _json.dumps(positions)
    col_json = _json.dumps(colors)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KAN-Refine — Interactive Gaussian Scene Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    overflow: hidden;
  }}
  #info {{
    position: fixed; top: 16px; left: 16px; z-index: 100;
    background: rgba(15, 15, 25, 0.85);
    backdrop-filter: blur(12px);
    padding: 16px 24px;
    border-radius: 12px;
    border: 1px solid rgba(120, 100, 255, 0.2);
    max-width: 320px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5);
  }}
  #info h2 {{
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #7c6aff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  #info p {{
    font-size: 12px;
    line-height: 1.5;
    color: #aaa;
  }}
  #info .stat {{
    color: #7c6aff;
    font-weight: 600;
  }}
  canvas {{ display: block; }}
</style>
</head>
<body>
<div id="info">
  <h2>KAN-Refine Scene Viewer</h2>
  <p>Points: <span class="stat">{len(positions):,}</span></p>
  <p>Drag to rotate · Scroll to zoom · Right-drag to pan</p>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const positions = {pos_json};
const colors = {col_json};

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

// Camera
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 1000);
camera.position.set(3, 2, 3);
camera.lookAt(0, 0, 0);

// Renderer
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Controls
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.8;
controls.zoomSpeed = 1.2;

// Point cloud
const geometry = new THREE.BufferGeometry();
const posArr = new Float32Array(positions.length * 3);
const colArr = new Float32Array(colors.length * 3);
for (let i = 0; i < positions.length; i++) {{
  posArr[i*3] = positions[i][0];
  posArr[i*3+1] = positions[i][1];
  posArr[i*3+2] = positions[i][2];
  colArr[i*3] = colors[i][0];
  colArr[i*3+1] = colors[i][1];
  colArr[i*3+2] = colors[i][2];
}}
geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colArr, 3));

const material = new THREE.PointsMaterial({{
  size: 0.015,
  vertexColors: true,
  sizeAttenuation: true,
  transparent: true,
  opacity: 0.9,
}});
const points = new THREE.Points(geometry, material);
scene.add(points);

// Grid helper
const grid = new THREE.GridHelper(4, 20, 0x333355, 0x222244);
scene.add(grid);

// Axes helper
const axes = new THREE.AxesHelper(1);
scene.add(axes);

// Ambient light
scene.add(new THREE.AmbientLight(0x404040, 2));

// Animation loop
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

// Resize handler
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""


# ====================================================================== #
# CLI
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(description="KAN-Refine Interactive 3D Viewer")
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to gaussians.ply file")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Path to cameras.json (optional)")
    parser.add_argument("--mode", type=str, default="open3d",
                        choices=["open3d", "html"],
                        help="Viewer mode: 'open3d' (default) or 'html' (WebGL)")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--bg-color", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if not os.path.isfile(args.scene):
        print(f"[error] Scene file not found: {args.scene}")
        sys.exit(1)

    # Auto-detect cameras.json
    if args.cameras is None:
        scene_dir = str(Path(args.scene).parent)
        cam_path = os.path.join(scene_dir, "cameras.json")
        if os.path.isfile(cam_path):
            args.cameras = cam_path

    if args.mode == "html":
        launch_html_viewer(args.scene, args.cameras)
    else:
        launch_open3d_viewer(
            args.scene,
            args.cameras,
            point_size=args.point_size,
            bg_color=tuple(args.bg_color),
            window_width=args.width,
            window_height=args.height,
        )


if __name__ == "__main__":
    main()
