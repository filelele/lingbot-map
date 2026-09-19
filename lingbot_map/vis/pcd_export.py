# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
PCD 3D export utilities for GCT predictions and point clouds.
Exports point clouds directly into standard Point Cloud Data (.pcd) format
without camera meshes or frustum geometry.
"""

import os
from typing import Optional, Union
import numpy as np

from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
from lingbot_map.vis.sky_segmentation import apply_sky_segmentation


def write_pcd(
    filename: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    binary: bool = True,
) -> str:
    """Write 3D points and optional RGB colors to a .pcd file.

    Tries to use Open3D if available for maximum performance,
    falling back to a robust built-in binary/ASCII PCD writer.

    Args:
        filename: Destination path for the .pcd file.
        points: (N, 3) float array of 3D coordinates.
        colors: Optional (N, 3) array of RGB colors (uint8 [0, 255] or float [0, 1]).
        binary: If True, write compact binary PCD; if False, write ASCII PCD.

    Returns:
        The output filename written.
    """
    points = np.asarray(points, dtype=np.float32)
    n_points = len(points)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors_u8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            colors_u8 = colors
    else:
        colors_u8 = None

    # Try Open3D first if available
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors_u8 is not None:
            colors_f = colors_u8.astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors_f)
        o3d.io.write_point_cloud(filename, pcd, write_ascii=not binary)
        return filename
    except Exception:
        pass

    # Built-in fallback PCD writer (PCL v0.7 compliant)
    if colors_u8 is not None:
        r = colors_u8[:, 0].astype(np.uint32)
        g = colors_u8[:, 1].astype(np.uint32)
        b = colors_u8[:, 2].astype(np.uint32)
        # PCL RGB float convention: (r << 16) | (g << 8) | b
        rgb_u32 = (r << 16) | (g << 8) | b
        rgb_f32 = rgb_u32.view(np.float32)

        type_str = "F F F F"
        fields_str = "x y z rgb"
        size_str = "4 4 4 4"
        count_str = "1 1 1 1"
    else:
        type_str = "F F F"
        fields_str = "x y z"
        size_str = "4 4 4"
        count_str = "1 1 1"

    data_mode = "binary" if binary else "ascii"
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields_str}\n"
        f"SIZE {size_str}\n"
        f"TYPE {type_str}\n"
        f"COUNT {count_str}\n"
        f"WIDTH {n_points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n_points}\n"
        f"DATA {data_mode}\n"
    )

    if binary:
        with open(filename, "wb") as f:
            f.write(header.encode("ascii"))
            if colors_u8 is not None:
                data = np.empty((n_points, 4), dtype=np.float32)
                data[:, :3] = points
                data[:, 3] = rgb_f32
            else:
                data = points
            f.write(data.tobytes())
    else:
        with open(filename, "w") as f:
            f.write(header)
            if colors_u8 is not None:
                for pt, rgb_val in zip(points, rgb_f32):
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {rgb_val}\n")
            else:
                for pt in points:
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

    return filename


def export_predictions_to_pcd(
    pred_dict: dict,
    output_path: str,
    vis_threshold: float = 1.5,
    downsample_factor: int = 10,
    mask_sky: bool = False,
    image_folder: Optional[str] = None,
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
    use_point_map: bool = False,
    binary: bool = True,
) -> Optional[str]:
    """Export predictions dictionary to a .pcd file.

    Args:
        pred_dict: Model prediction dictionary containing images, extrinsic, intrinsic, depth/world_points.
        output_path: File path to save the .pcd file.
        vis_threshold: Confidence threshold for filtering points.
        downsample_factor: Spatial downsample factor (1 = keep all points).
        mask_sky: Whether to apply sky segmentation to filter out sky points.
        image_folder: Folder containing source images (for sky mask caching).
        sky_mask_dir: Directory for cached sky masks.
        sky_mask_visualization_dir: Directory for sky mask visualizations.
        use_point_map: Whether to use predicted world_points directly instead of unprojecting depth.
        binary: Whether to write binary PCD (recommended) or ASCII PCD.

    Returns:
        The output path if successful, None otherwise.
    """
    images = pred_dict["images"]  # (S, 3, H, W) or (S, H, W, 3)

    depth_map = pred_dict.get("depth")
    depth_conf = pred_dict.get("depth_conf")
    extrinsics_cam = pred_dict.get("extrinsic")
    intrinsics_cam = pred_dict.get("intrinsic")

    if not use_point_map and depth_map is not None and extrinsics_cam is not None and intrinsics_cam is not None:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    elif "world_points" in pred_dict and pred_dict["world_points"] is not None:
        world_points = pred_dict["world_points"]
        conf = pred_dict.get("world_points_conf", depth_conf)
    else:
        raise ValueError("Cannot extract point cloud: neither depth map with extrinsics nor world_points provided.")

    if mask_sky:
        conf = apply_sky_segmentation(
            conf,
            image_folder=image_folder,
            images=images,
            sky_mask_dir=sky_mask_dir,
            sky_mask_visualization_dir=sky_mask_visualization_dir,
        )

    if images.ndim == 4 and images.shape[1] == 3:
        colors = images.transpose(0, 2, 3, 1)
    else:
        colors = images

    S = world_points.shape[0]
    all_points = []
    all_colors = []

    for i in range(S):
        pts = world_points[i].reshape(-1, 3)
        cols = colors[i].reshape(-1, 3)
        cf = conf[i].reshape(-1) if conf is not None else None

        # Filter NaNs and Infs
        valid = np.isfinite(pts).all(axis=1)
        if not valid.all():
            pts = pts[valid]
            cols = cols[valid]
            if cf is not None:
                cf = cf[valid]

        # Confidence filter
        if cf is not None and vis_threshold is not None:
            mask = cf > vis_threshold
            pts = pts[mask]
            cols = cols[mask]

        # Downsample
        if downsample_factor > 1 and len(pts) > 0:
            indices = np.arange(0, len(pts), downsample_factor)
            pts = pts[indices]
            cols = cols[indices]

        if len(pts) > 0:
            all_points.append(pts)
            if cols.dtype != np.uint8:
                cols = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)
            all_colors.append(cols)

    if not all_points:
        print("Warning: No points survived filtering for PCD export.")
        return None

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)

    write_pcd(output_path, merged_points, merged_colors, binary=binary)
    print(f"Point cloud saved to {output_path} ({len(merged_points):,} points)")
    return output_path
