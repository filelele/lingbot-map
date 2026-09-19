"""2D Occupancy Grid generation and export for LingBot-MAP.

Transforms 3D reconstructed point clouds into standard ROS-compatible 2D occupancy
grid maps (.png + .yaml), projected onto the horizontal navigation plane (X, Z).
"""

import os
from typing import Dict, Optional, Tuple, Any
import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt


def compute_scene_bounds(points: np.ndarray) -> Dict[str, float]:
    """Compute min/max bounds and spans for X, Y, Z axes."""
    pts = np.asarray(points)
    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    z_min, z_max = float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))
    return {
        "x_min": x_min, "x_max": x_max, "x_span": x_max - x_min,
        "y_min": y_min, "y_max": y_max, "y_span": y_max - y_min,
        "z_min": z_min, "z_max": z_max, "z_span": z_max - z_min,
    }


def compute_y_histogram(
    points: np.ndarray, bins: int = 100
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute 100-bin histogram of vertical Y coordinates and estimate floor/obstacle bounds.

    Returns:
        counts: (bins,) array of point counts
        bin_edges: (bins + 1,) array of bin edge values
        defaults: dict with recommended y_min and y_max
    """
    pts = np.asarray(points)
    y_vals = pts[:, 1]
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))

    counts, bin_edges = np.histogram(y_vals, bins=bins, range=(y_min, y_max))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Find the primary peak on the positive Y side (OpenCV convention)
    # If no positive bins exist, fallback to global peak
    pos_mask = bin_centers > 0
    if np.any(pos_mask) and np.max(counts[pos_mask]) > 0:
        pos_indices = np.where(pos_mask)[0]
        peak_idx = pos_indices[np.argmax(counts[pos_mask])]
    else:
        peak_idx = int(np.argmax(counts))

    floor_y = float(bin_centers[peak_idx])

    # Default obstacle height range:
    # y_max_obs: slightly above the floor (~2-5 bins before peak) to exclude the flat floor itself
    bin_width = (y_max - y_min) / bins
    default_y_max = floor_y - 2.0 * bin_width
    # default_y_min (more negative Y)
    default_y_min = default_y_max - 10.0*bin_width
    default_y_min = max(y_min, default_y_min)

    if default_y_min >= default_y_max:
        default_y_min = y_min
        default_y_max = y_max

    defaults = {
        "floor_y_peak": floor_y,
        "default_y_min": default_y_min,
        "default_y_max": default_y_max,
        "y_min_all": y_min,
        "y_max_all": y_max,
    }
    return counts, bin_edges, defaults


def plot_y_histogram_image(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    width: int = 340,
    height: int = 180,
) -> np.ndarray:
    """Render a 100-bin Y histogram with cutoff lines as an RGB image for GUI display."""
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#252526")

    # Bar chart
    bar_width = (bin_edges[-1] - bin_edges[0]) / len(counts)
    ax.bar(bin_centers, counts, width=bar_width, color="#4ec9b0", alpha=0.8, edgecolor="none")

    # Highlight active slice
    if y_min is not None and y_max is not None and y_min < y_max:
        mask = (bin_centers >= y_min) & (bin_centers <= y_max)
        if np.any(mask):
            ax.bar(bin_centers[mask], counts[mask], width=bar_width, color="#ce9178", alpha=0.9)
        ax.axvline(y_min, color="#f44747", linestyle="--", linewidth=1.8, label=f"y_min ({y_min:.2f})")
        ax.axvline(y_max, color="#dcdcaa", linestyle="--", linewidth=1.8, label=f"y_max ({y_max:.2f})")
        ax.legend(loc="upper left", fontsize=7, facecolor="#333333", edgecolor="#555555", labelcolor="#ffffff")

    ax.tick_params(colors="#d4d4d4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555555")

    ax.set_title("Height (Y) Distribution & Obstacle Slice", color="#ffffff", fontsize=8.5, pad=3)
    ax.set_xlabel("← Up (-Y)  |  Y (units)  |  Down (+Y) →", color="#cccccc", fontsize=7)
    ax.set_ylabel("Points", color="#cccccc", fontsize=7)
    plt.tight_layout()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return rgb


def validate_grid_config(
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    resolution: float,
    min_dim: int = 5,
    max_dim: int = 2000,
) -> Tuple[bool, str, int, int]:
    """Validate that the given resolution produces a grid size within [min_dim, max_dim].

    Returns:
        valid: bool
        message: str status message
        cols: int (grid X dimension)
        rows: int (grid Z dimension)
    """
    if resolution <= 0:
        return False, "Resolution must be positive (> 0).", 0, 0

    x_span = x_max - x_min
    z_span = z_max - z_min

    if x_span <= 0 or z_span <= 0:
        return False, "Invalid point bounds: zero or negative span.", 0, 0

    cols = int(np.ceil(x_span / resolution))
    rows = int(np.ceil(z_span / resolution))

    if cols < min_dim or rows < min_dim:
        return (
            False,
            f"Grid too small: {cols}x{rows} (min allowed is {min_dim}x{min_dim}). Decrease resolution.",
            cols,
            rows,
        )

    if cols > max_dim or rows > max_dim:
        return (
            False,
            f"Grid too large: {cols}x{rows} (max allowed is {max_dim}x{max_dim}). Increase resolution.",
            cols,
            rows,
        )

    return True, f"Valid grid: {cols} cols x {rows} rows ({cols * rows:,} cells)", cols, rows


def generate_occupancy_grid(
    points: np.ndarray,
    resolution: float,
    y_min: float,
    y_max: float,
    min_points_per_cell: int = 3,
    fill_unexplored_as_occupied: bool = True,
    camera_positions: Optional[np.ndarray] = None,
    bounds: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a 2D occupancy grid from 3D points.

    Convention:
        - Free space: 255 (White)
        - Occupied: 0 (Black)
        - Bottom-left of image corresponds to (x_min, z_min) in world coordinates.
        - Row 0 (top of image) corresponds to z_max.

    Args:
        points: (N, 3) array of [X, Y, Z] coordinates.
        resolution: Unit per grid cell.
        y_min: Upper height cutoff (smaller/negative Y).
        y_max: Lower height cutoff (larger/positive Y, just above floor).
        min_points_per_cell: Minimum points in a cell to count as occupied.
        fill_unexplored_as_occupied: Fill exterior void as occupied using inside-out flood fill.
        camera_positions: Optional (M, 3) camera/rover trajectory positions.
        bounds: Optional explicit scene bounds dict.

    Returns:
        grid_img: (rows, cols) uint8 grayscale image (0 = occupied, 255 = free).
        meta: dict containing grid parameters and bounds.
    """
    pts = np.asarray(points, dtype=np.float32)
    if bounds is None:
        bounds = compute_scene_bounds(pts)

    x_min, x_max = bounds["x_min"], bounds["x_max"]
    z_min, z_max = bounds["z_min"], bounds["z_max"]

    valid, msg, cols, rows = validate_grid_config(x_min, x_max, z_min, z_max, resolution)
    if not valid:
        raise ValueError(msg)

    # Filter obstacle points in vertical band [y_min, y_max]
    obs_mask = (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
    obs_pts = pts[obs_mask]

    # Compute 2D histogram of obstacle points
    counts_obs, _, _ = np.histogram2d(
        obs_pts[:, 0],
        obs_pts[:, 2],
        bins=[cols, rows],
        range=[[x_min, x_min + cols * resolution], [z_min, z_min + rows * resolution]],
    )
    # counts_obs has shape (cols, rows); transpose to (rows, cols) where axis 0 is Z
    grid_z_x = counts_obs.T

    # Flip vertically so that row 0 corresponds to z_max (standard image top)
    # and bottom row (rows - 1) corresponds to z_min (standard ROS bottom-left origin)
    img_counts = np.flipud(grid_z_x)
    is_obstacle = (img_counts >= min_points_per_cell)

    if not fill_unexplored_as_occupied:
        # Standard 2-state: all non-obstacle cells are Free
        grid_img = np.full((rows, cols), 255, dtype=np.uint8)
        grid_img[is_obstacle] = 0
    else:
        # Inside-out flood fill (Method 2):
        # 1. Close small cracks in walls (3x3 kernel) so sparse points form solid watertight walls
        obs_u8 = is_obstacle.astype(np.uint8) * 255
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_obs = cv2.morphologyEx(obs_u8, cv2.MORPH_CLOSE, kernel_close)
        passable = (closed_obs == 0).astype(np.uint8) * 255

        # 2. Gather interior seed points (guaranteed inside the room)
        seed_mask = np.zeros((rows, cols), dtype=np.uint8)

        # Prefer camera trajectory as the ground-truth interior seeds
        if camera_positions is not None and len(camera_positions) > 0:
            cams = np.asarray(camera_positions, dtype=np.float32)
            cam_x = cams[:, 0]
            cam_z = cams[:, 2] if cams.shape[1] >= 3 else cams[:, 1]
            c_idx = np.clip(np.floor((cam_x - x_min) / resolution).astype(int), 0, cols - 1)
            r_idx = np.clip(rows - 1 - np.floor((cam_z - z_min) / resolution).astype(int), 0, rows - 1)
            for c, r in zip(c_idx, r_idx):
                cv2.circle(seed_mask, (c, r), radius=1, color=255, thickness=-1)

        # Fallback to floor center only if no valid camera seeds exist
        if np.sum(cv2.bitwise_and(seed_mask, passable)) == 0:
            floor_mask = pts[:, 1] > y_max
            if np.any(floor_mask):
                fl_x, fl_z = pts[floor_mask, 0], pts[floor_mask, 2]
                med_x, med_z = float(np.median(fl_x)), float(np.median(fl_z))
                c_c = int(np.clip(np.floor((med_x - x_min) / resolution), 0, cols - 1))
                c_r = int(np.clip(rows - 1 - np.floor((med_z - z_min) / resolution), 0, rows - 1))
                cv2.circle(seed_mask, (c_c, c_r), radius=2, color=255, thickness=-1)

        # Ensure seeds only fall on passable floor
        valid_seeds = cv2.bitwise_and(seed_mask, passable)

        # Ultimate fallback if center hit an obstacle: find passable pixel closest to center
        if np.sum(valid_seeds) == 0:
            passable_coords = np.argwhere(passable > 0)
            if len(passable_coords) > 0:
                center_r, center_c = passable_coords[len(passable_coords) // 2]
                valid_seeds[center_r, center_c] = 255

        # 3. Find connected components on passable floor
        # Use 4-connectivity so the flood fill cannot slip through diagonal 8-connected obstacle walls
        num_labels, labels = cv2.connectedComponents(passable, connectivity=4)
        interior_labels = np.unique(labels[valid_seeds > 0])
        interior_labels = interior_labels[interior_labels > 0]  # ignore 0 (obstacles)

        is_interior_floor = np.isin(labels, interior_labels)

        # 4. Build final grid: outside void is Black (0), interior floor is White (255), obstacles are Black (0)
        grid_img = np.zeros((rows, cols), dtype=np.uint8)
        grid_img[is_interior_floor] = 255
        grid_img[is_obstacle] = 0

    meta = {
        "resolution": float(resolution),
        "cols": int(cols),
        "rows": int(rows),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "origin": [float(x_min), float(z_min), 0.0],
        "min_points_per_cell": int(min_points_per_cell),
        "fill_unexplored_as_occupied": bool(fill_unexplored_as_occupied),
        "num_obstacle_points": int(len(obs_pts)),
        "num_total_points": int(len(pts)),
    }

    return grid_img, meta


def export_occupancy_grid(
    output_path: str,
    points: np.ndarray,
    resolution: float,
    y_min: float,
    y_max: float,
    min_points_per_cell: int = 3,
    fill_unexplored_as_occupied: bool = True,
    camera_positions: Optional[np.ndarray] = None,
    bounds: Optional[Dict[str, float]] = None,
) -> Tuple[str, str, Dict[str, Any], np.ndarray]:
    """Export 2D occupancy grid as ROS-standard .png and companion .yaml metadata.

    Args:
        output_path: Destination path for .png or directory (e.g. "path/to/occupancy_grid").
        points: (N, 3) point cloud array.
        resolution: Units per cell.
        y_min: Upper height cutoff.
        y_max: Lower height cutoff.
        min_points_per_cell: Minimum points to mark cell as occupied.
        fill_unexplored_as_occupied: Fill exterior void as occupied using inside-out flood fill.
        camera_positions: Optional (M, 3) camera/rover trajectory positions.
        bounds: Optional precalculated scene bounds.

    Returns:
        png_path: Absolute path to written PNG.
        yaml_path: Absolute path to written YAML.
        meta: Metadata dictionary.
        grid_img: (rows, cols) uint8 image.
    """
    output_path = os.path.abspath(output_path)
    root, ext = os.path.splitext(output_path)
    if ext.lower() in [".png", ".pgm"]:
        base_dir = os.path.dirname(output_path)
        png_path = output_path
        yaml_path = root + ".yaml"
    elif ext == "":
        # output_path is a directory name (e.g. ".../occupancy_grid")
        base_dir = output_path
        png_path = os.path.join(base_dir, "occupancy_grid.png")
        yaml_path = os.path.join(base_dir, "occupancy_grid.yaml")
    else:
        base_dir = os.path.dirname(output_path)
        png_path = root + ".png"
        yaml_path = root + ".yaml"

    os.makedirs(base_dir, exist_ok=True)

    grid_img, meta = generate_occupancy_grid(
        points=points,
        resolution=resolution,
        y_min=y_min,
        y_max=y_max,
        min_points_per_cell=min_points_per_cell,
        fill_unexplored_as_occupied=fill_unexplored_as_occupied,
        camera_positions=camera_positions,
        bounds=bounds,
    )

    # Save 8-bit PNG
    cv2.imwrite(png_path, grid_img)

    # Generate ROS Nav2 compliant YAML
    yaml_data = {
        "image": os.path.basename(png_path),
        "resolution": meta["resolution"],
        "origin": meta["origin"],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
        "mode": "trinary",
        # Extended metadata for autonomous rover
        "grid_size": [meta["cols"], meta["rows"]],
        "y_range": [meta["y_min"], meta["y_max"]],
        "min_points_per_cell": meta["min_points_per_cell"],
        "fill_unexplored_as_occupied": meta["fill_unexplored_as_occupied"],
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    return png_path, yaml_path, meta, grid_img
