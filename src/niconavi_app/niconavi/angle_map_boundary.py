from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

from niconavi_app.niconavi.type import RawMaps
from niconavi_app.niconavi.tools.grain_plot import detect_boundaries


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must have shape (H, W, 3).")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _to_gray_float(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_as_uint8_rgb(rgb), cv2.COLOR_RGB2GRAY).astype(np.float64)


def _theta_phi_to_rgb(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    theta, phi = np.broadcast_arrays(theta, phi)
    hsv = np.stack(
        [
            np.uint8(np.clip((np.mod(theta, 90.0) / 90.0) * 180.0, 0, 179)),
            np.full(theta.shape, 255, dtype=np.uint8),
            np.uint8(np.clip((phi / 90.0) * 255.0, 0, 255)),
        ],
        axis=-1,
    )
    rgb = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB)
    return rgb.reshape(hsv.shape)


def _orientation_distance_deg(theta1: Any, phi1: Any, theta2: Any, phi2: Any) -> Any:
    theta1 = np.asarray(theta1, dtype=np.float64)
    phi1 = np.asarray(phi1, dtype=np.float64)
    theta2 = np.asarray(theta2, dtype=np.float64)
    phi2 = np.asarray(phi2, dtype=np.float64)
    d_theta = np.abs(theta1 - theta2)
    d_theta = np.minimum(d_theta, 90.0 - d_theta)
    phi1_rad = np.deg2rad(phi1)
    phi2_rad = np.deg2rad(phi2)
    d_theta_rad = np.deg2rad(d_theta)
    cos_angle = (
        np.cos(phi1_rad) * np.cos(phi2_rad)
        + np.sin(phi1_rad) * np.sin(phi2_rad) * np.cos(d_theta_rad)
    )
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return float(angle) if angle.ndim == 0 else angle


def _orientation_vectors(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta4 = np.deg2rad(4.0 * np.asarray(theta, dtype=np.float64))
    phi_rad = np.deg2rad(np.asarray(phi, dtype=np.float64))
    sin_phi = np.sin(phi_rad)
    return sin_phi * np.cos(theta4), sin_phi * np.sin(theta4), np.cos(phi_rad)


def _orientation_from_vectors(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.hypot(x, y)
    theta = (np.degrees(np.arctan2(y, x)) % 360.0) / 4.0
    theta = np.where(xy <= 1e-12, 0.0, theta)
    phi = np.degrees(np.arctan2(xy, z))
    return theta, np.clip(phi, 0.0, 90.0)


def _region_median_color_map(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = _as_uint8_rgb(rgb)
    labels = np.asarray(labels, dtype=np.int32)
    out = np.empty_like(rgb)
    for label in np.unique(labels[labels >= 0]):
        mask = labels == label
        out[mask] = np.clip(np.median(rgb[mask], axis=0), 0, 255).astype(np.uint8)
    return out


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    out = np.full(labels.shape, -1, dtype=np.int32)
    for new_label, old_label in enumerate(np.unique(labels[labels >= 0])):
        out[labels == old_label] = new_label
    return out


def _split_connected_components(labels: np.ndarray) -> np.ndarray:
    labels = _compact_labels(labels)
    out = np.full(labels.shape, -1, dtype=np.int32)
    next_label = 0
    for label in np.unique(labels[labels >= 0]):
        n_parts, part_labels = cv2.connectedComponents(
            (labels == label).astype(np.uint8),
            connectivity=8,
        )
        for part in range(1, n_parts):
            out[part_labels == part] = next_label
            next_label += 1
    return out


def _make_dark_branch_cache(
    labels: np.ndarray,
    base_rgb: np.ndarray,
    *,
    dark_l_thresh: float,
) -> dict[str, Any]:
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    labels = _split_connected_components(_compact_labels(np.asarray(labels, dtype=np.int32)))
    base_rgb = _as_uint8_rgb(base_rgb)
    height, width = labels.shape
    lab_l = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
    dark_labels = [
        int(label)
        for label in np.unique(labels[labels >= 0])
        if float(np.median(lab_l[labels == label])) <= dark_l_thresh
    ]
    dark_mask = np.isin(labels, dark_labels)
    removal_width_threshold = np.full(labels.shape, np.inf, dtype=np.float32)

    if np.any(dark_mask):
        skeleton = skeletonize(dark_mask)
        dist = cv2.distanceTransform(dark_mask.astype(np.uint8), cv2.DIST_L2, 3)
        skeleton_width = 2.0 * dist
        n_branches, branch_labels = cv2.connectedComponents(
            skeleton.astype(np.uint8),
            connectivity=8,
        )

        for branch_id in range(1, n_branches):
            branch = branch_labels == branch_id
            if not np.any(branch):
                continue
            branch_width = float(np.median(skeleton_width[branch]))
            for y, x in zip(*np.nonzero(branch)):
                radius = max(1, int(np.ceil(skeleton_width[y, x] / 2.0)))
                y_slice = slice(max(0, y - radius), min(height, y + radius + 1))
                x_slice = slice(max(0, x - radius), min(width, x + radius + 1))
                yy, xx = np.ogrid[y_slice, x_slice]
                disk = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
                current = removal_width_threshold[y_slice, x_slice]
                current[disk] = np.minimum(current[disk], branch_width)

        removal_width_threshold[~dark_mask] = np.inf

    nearest_non_dark_labels = labels.copy()
    target_mask = ~dark_mask
    if np.any(dark_mask) and np.any(target_mask):
        _, nearest = ndi.distance_transform_edt(~target_mask, return_indices=True)
        nearest_non_dark_labels[dark_mask] = labels[
            nearest[0][dark_mask],
            nearest[1][dark_mask],
        ]

    return {
        "dark_l_thresh": float(dark_l_thresh),
        "initial_labels": labels,
        "dark_mask": dark_mask,
        "removal_width_threshold": removal_width_threshold,
        "nearest_non_dark_labels": nearest_non_dark_labels,
    }


def make_theta_phi_angle_info(raw_maps: RawMaps) -> dict[str, Any]:
    theta = np.asarray(raw_maps["extinction_angle"], dtype=np.float64)
    brightest = _as_uint8_rgb(raw_maps["R_color_map"])
    darkest = _as_uint8_rgb(raw_maps["extinction_color_map"])
    dark_gray = _to_gray_float(darkest)
    bright_gray = _to_gray_float(brightest)
    bright_ref = np.percentile(bright_gray, 99.5)
    phi = 90.0 * np.clip((bright_gray - dark_gray) / np.maximum(bright_ref - dark_gray, 1e-6), 0.0, 1.0)
    angle_map = _theta_phi_to_rgb(theta, phi)
    return {
        "theta": theta,
        "phi": phi,
        "theta_phi_angle_map": angle_map,
        "angle_map_display": angle_map,
    }


def create_shock_filter_iterator(
    theta_phi_angle_info: dict[str, Any],
    *,
    amount: float = 0.5,
    min_vividness_delta: float = 5.0,
    hue_delta_thresh_deg: float = 10.0,
    preserve_black_l_thresh: float = 18.0,
    black_source_l_thresh: float = 35.0,
    blurry_vividness_thresh: float = 60.0,
) -> Iterator[dict[str, Any]]:
    filtered = _as_uint8_rgb(theta_phi_angle_info["theta_phi_angle_map"]).astype(np.float32)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while True:
        src_u8 = np.clip(filtered, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(src_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(src_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        hue = hsv[:, :, 0] * 2.0
        vividness = hsv[:, :, 1] * (hsv[:, :, 2] / 255.0)
        l_value = lab[:, :, 0]
        blackness = 255.0 - l_value
        preserve_black = l_value <= preserve_black_l_thresh
        best_score = vividness.copy()
        best_color = filtered.copy()

        for dy, dx in neighbor_offsets:
            target_y = slice(max(0, dy), min(src_u8.shape[0], src_u8.shape[0] + dy))
            target_x = slice(max(0, dx), min(src_u8.shape[1], src_u8.shape[1] + dx))
            neighbor_y = slice(max(0, -dy), min(src_u8.shape[0], src_u8.shape[0] - dy))
            neighbor_x = slice(max(0, -dx), min(src_u8.shape[1], src_u8.shape[1] - dx))
            target = (target_y, target_x)
            neighbor = (neighbor_y, neighbor_x)
            hue_diff = np.abs(hue[target] - hue[neighbor])
            hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)
            can_color = (
                (vividness[neighbor] >= vividness[target] + min_vividness_delta)
                & (vividness[neighbor] > best_score[target])
                & (hue_diff <= hue_delta_thresh_deg)
            )
            can_black = (
                (l_value[neighbor] <= black_source_l_thresh)
                & (vividness[target] <= blurry_vividness_thresh)
                & (blackness[neighbor] > best_score[target])
            )
            target_score = best_score[target]
            target_color = best_color[target]
            target_score[can_color] = vividness[neighbor][can_color]
            target_color[can_color] = filtered[neighbor][can_color]
            target_score[can_black] = blackness[neighbor][can_black]
            target_color[can_black] = filtered[neighbor][can_black]

        update = (best_score > vividness) & (~preserve_black)
        if np.any(update):
            filtered[update] = (1.0 - float(amount)) * filtered[update] + float(amount) * best_color[update]
            filtered = np.clip(filtered, 0, 255)

        shock_angle_map = filtered.astype(np.uint8)
        yield {**theta_phi_angle_info, "shock_angle_map": shock_angle_map, "angle_map_display": shock_angle_map}


def _make_superpixel_labels(
    rgb: np.ndarray,
    slic_region_size: int,
    slic_ruler: float,
    slic_num_iterations: int,
    slic_min_element_size: int,
) -> np.ndarray:
    rgb = _as_uint8_rgb(rgb)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createSuperpixelSLIC"):
        slic = cv2.ximgproc.createSuperpixelSLIC(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            algorithm=getattr(cv2.ximgproc, "SLICO", 101),
            region_size=int(slic_region_size),
            ruler=float(slic_ruler),
        )
        slic.iterate(int(slic_num_iterations))
        slic.enforceLabelConnectivity(int(slic_min_element_size))
        return slic.getLabels().astype(np.int32)

    from skimage.segmentation import slic

    height, width = rgb.shape[:2]
    n_segments = max(1, int(round((height * width) / max(1, int(slic_region_size) ** 2))))
    return slic(
        rgb,
        n_segments=n_segments,
        compactness=float(slic_ruler),
        max_num_iter=int(slic_num_iterations),
        enforce_connectivity=True,
        min_size_factor=max(0.0, float(slic_min_element_size)) / max(1, int(slic_region_size) ** 2),
        start_label=0,
        channel_axis=-1,
    ).astype(np.int32)


def segment_angle_map(
    theta_phi_angle_info: dict[str, Any],
    *,
    delta_euler_thresh: float,
    slic_region_size: int = 30,
    slic_ruler: float = 20.0,
    slic_num_iterations: int = 10,
    slic_min_element_size: int = 25,
    dark_l_thresh: float = 35.0,
) -> dict[str, Any]:
    source = _as_uint8_rgb(theta_phi_angle_info.get("shock_angle_map", theta_phi_angle_info["theta_phi_angle_map"]))
    superpixel_labels = _make_superpixel_labels(
        source,
        slic_region_size=slic_region_size,
        slic_ruler=slic_ruler,
        slic_num_iterations=slic_num_iterations,
        slic_min_element_size=slic_min_element_size,
    )
    superpixel_median_map = _region_median_color_map(source, superpixel_labels)
    theta = np.asarray(theta_phi_angle_info["theta"], dtype=np.float64)
    phi = np.asarray(theta_phi_angle_info["phi"], dtype=np.float64)
    flat_labels = superpixel_labels.reshape(-1)
    n_labels = int(flat_labels.max()) + 1
    x, y, z = _orientation_vectors(theta.reshape(-1), phi.reshape(-1))
    counts = np.bincount(flat_labels, minlength=n_labels).astype(np.float64)
    mean_x = np.bincount(flat_labels, weights=x, minlength=n_labels) / np.maximum(counts, 1.0)
    mean_y = np.bincount(flat_labels, weights=y, minlength=n_labels) / np.maximum(counts, 1.0)
    mean_z = np.bincount(flat_labels, weights=z, minlength=n_labels) / np.maximum(counts, 1.0)
    region_theta, region_phi = _orientation_from_vectors(mean_x, mean_y, mean_z)
    parent = np.arange(n_labels, dtype=np.int32)

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = int(parent[label])
        return label

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for left, right in ((superpixel_labels[:, :-1], superpixel_labels[:, 1:]), (superpixel_labels[:-1, :], superpixel_labels[1:, :])):
        diff = left != right
        for a, b in zip(left[diff], right[diff]):
            if _orientation_distance_deg(region_theta[a], region_phi[a], region_theta[b], region_phi[b]) <= delta_euler_thresh:
                union(int(a), int(b))

    root_to_label: dict[int, int] = {}
    lookup = np.full(n_labels, -1, dtype=np.int32)
    for label in np.unique(flat_labels):
        root = find(int(label))
        root_to_label.setdefault(root, len(root_to_label))
        lookup[label] = root_to_label[root]
    merged_labels = lookup[superpixel_labels]
    dark_branch_cache = _make_dark_branch_cache(
        merged_labels,
        _region_median_color_map(source, merged_labels),
        dark_l_thresh=dark_l_thresh,
    )
    initial_labels = dark_branch_cache["initial_labels"]
    merged_superpixel_median_map = _region_median_color_map(source, initial_labels)
    return {
        **theta_phi_angle_info,
        "superpixel_labels": superpixel_labels,
        "superpixel_median_map": superpixel_median_map,
        "merged_superpixel_median_map": merged_superpixel_median_map,
        "merged_superpixel_labels": initial_labels,
        "black_artifact_removed_superpixel_labels": initial_labels,
        "dark_branch_cache": dark_branch_cache,
        "angle_map_display": merged_superpixel_median_map,
    }


def fill_dark_boundaries(
    theta_phi_angle_info: dict[str, Any],
    *,
    branch_width_thresh: float,
    dark_l_thresh: float = 15.0,
) -> dict[str, Any]:
    base_rgb = _as_uint8_rgb(
        theta_phi_angle_info.get(
            "merged_superpixel_median_map",
            theta_phi_angle_info["superpixel_median_map"],
        )
    )
    cache = theta_phi_angle_info.get("dark_branch_cache")
    if cache is None or cache.get("dark_l_thresh") != float(dark_l_thresh):
        cache = _make_dark_branch_cache(
            np.asarray(theta_phi_angle_info["merged_superpixel_labels"], dtype=np.int32),
            base_rgb,
            dark_l_thresh=dark_l_thresh,
        )

    labels = np.asarray(cache["initial_labels"], dtype=np.int32).copy()
    remove_mask = (
        np.asarray(cache["removal_width_threshold"]) <= float(branch_width_thresh)
    )

    if np.any(remove_mask):
        nearest_non_dark_labels = np.asarray(
            cache["nearest_non_dark_labels"],
            dtype=np.int32,
        )
        labels[remove_mask] = nearest_non_dark_labels[remove_mask]

    labels = _split_connected_components(_compact_labels(labels))
    cleaned_angle_map = _region_median_color_map(base_rgb, labels)
    return {
        **theta_phi_angle_info,
        "dark_branch_cache": cache,
        "black_artifact_removed_superpixel_labels": labels,
        "black_artifact_removed_angle_map": cleaned_angle_map,
        "black_artifact_removed_mask": np.where(remove_mask, 255, 0).astype(np.uint8),
        "angle_map_display": cleaned_angle_map,
    }


def grain_boundary_from_angle_labels(theta_phi_angle_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(theta_phi_angle_info["black_artifact_removed_superpixel_labels"], dtype=np.int32)
    boundary = detect_boundaries(labels)
    return labels, boundary
