from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

from niconavi_app.niconavi.type import RawMaps
from niconavi_app.niconavi.image.image import create_outside_circle_mask
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
            np.uint8(np.clip((np.mod(phi, 90.0) / 90.0) * 180.0, 0, 179)),
            np.full(theta.shape, 255, dtype=np.uint8),
            np.uint8(np.clip((theta / 90.0) * 255.0, 0, 255)),
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
    d_phi = np.abs(phi1 - phi2)
    d_phi = np.minimum(d_phi, 90.0 - d_phi)
    theta1_rad = np.deg2rad(theta1)
    theta2_rad = np.deg2rad(theta2)
    d_phi_rad = np.deg2rad(d_phi)
    cos_angle = (
        np.cos(theta1_rad) * np.cos(theta2_rad)
        + np.sin(theta1_rad) * np.sin(theta2_rad) * np.cos(d_phi_rad)
    )
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return float(angle) if angle.ndim == 0 else angle


def _orientation_vectors(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi4 = np.deg2rad(4.0 * np.asarray(phi, dtype=np.float64))
    theta_rad = np.deg2rad(np.asarray(theta, dtype=np.float64))
    sin_theta = np.sin(theta_rad)
    return sin_theta * np.cos(phi4), sin_theta * np.sin(phi4), np.cos(theta_rad)


def _orientation_from_vectors(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.hypot(x, y)
    phi = (np.degrees(np.arctan2(y, x)) % 360.0) / 4.0
    phi = np.where(xy <= 1e-12, 0.0, phi)
    theta = np.degrees(np.arctan2(xy, z))
    return np.clip(theta, 0.0, 90.0), phi


def _draw_unicode_label(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    font_size: int,
    color: tuple[int, int, int],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, fill=color, font=font)
    image[:] = np.asarray(pil_image)


def make_theta_phi_legend_image(size: int = 320) -> np.ndarray:
    size = int(size)
    margin = max(36, int(size * 0.14))
    radius = max(1, size - margin * 2)
    origin_x = margin
    origin_y = size - margin
    yy, xx = np.indices((size, size))
    dx = xx - origin_x
    dy = origin_y - yy
    rr = np.hypot(dx, dy)
    inside = (dx >= 0) & (dy >= 0) & (rr <= radius)

    legend = np.full((size, size, 3), 48, dtype=np.uint8)
    theta = (rr / radius) * 90.0
    phi = np.degrees(np.arctan2(dy, dx))
    legend[inside] = _theta_phi_to_rgb(theta[inside], phi[inside])

    grid_color = (235, 235, 235)
    label_color = (255, 255, 255)
    inner_grid_radius = int(round(radius * 10.0 / 90.0))
    for angle in range(0, 91, 10):
        rad = np.deg2rad(angle)
        start_radius = 0 if angle in (0, 90) else inner_grid_radius
        start = (
            int(round(origin_x + start_radius * np.cos(rad))),
            int(round(origin_y - start_radius * np.sin(rad))),
        )
        end = (
            int(round(origin_x + radius * np.cos(rad))),
            int(round(origin_y - radius * np.sin(rad))),
        )
        cv2.line(legend, start, end, grid_color, 1, cv2.LINE_AA)

    for theta_deg in range(10, 91, 10):
        r = int(round(radius * theta_deg / 90.0))
        cv2.ellipse(legend, (origin_x, origin_y), (r, r), 0, 270, 360, grid_color, 1, cv2.LINE_AA)
        label_x = int(round(origin_x + r))
        cv2.putText(
            legend,
            str(theta_deg),
            (label_x - 8, origin_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            label_color,
            1,
            cv2.LINE_AA,
        )

    for angle in range(0, 91, 10):
        rad = np.deg2rad(angle)
        label_r = radius + 18
        label_x = int(round(origin_x + label_r * np.cos(rad)))
        label_y = int(round(origin_y - label_r * np.sin(rad)))
        cv2.putText(
            legend,
            str(angle),
            (label_x - 8, label_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            label_color,
            1,
            cv2.LINE_AA,
        )

    _draw_unicode_label(
        legend,
        "θ",
        (origin_x + radius // 2 - 8, origin_y + 24),
        font_size=16,
        color=label_color,
    )
    _draw_unicode_label(
        legend,
        "φ",
        (origin_x + radius - 18, origin_y - radius // 2 - 54),
        font_size=16,
        color=label_color,
    )
    return legend


def _make_theta_from_dark_bright(
    darkest_rgb: np.ndarray,
    brightest_rgb: np.ndarray,
    *,
    bright_percentile: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    darkest_gray = _to_gray_float(darkest_rgb)
    brightest_gray = _to_gray_float(brightest_rgb)
    bright_ref = np.percentile(brightest_gray, bright_percentile)
    intensity_ratio = (brightest_gray - darkest_gray) / np.maximum(bright_ref - darkest_gray, eps)
    intensity_ratio = np.clip(intensity_ratio, 0.0, 1.0)

    theta_acute = 0.5 * np.degrees(np.arcsin(np.sqrt(intensity_ratio)))
    return 2.0 * theta_acute


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


def _split_labels_by_mask(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    labels = _compact_labels(labels)
    mask = np.asarray(mask, dtype=bool)
    if labels.shape != mask.shape:
        raise ValueError("labels and mask must have the same shape.")

    flat_labels = labels.reshape(-1)
    flat_mask = mask.reshape(-1)
    n_labels = int(flat_labels.max()) + 1
    counts = np.bincount(flat_labels, minlength=n_labels)
    mask_counts = np.bincount(flat_labels, weights=flat_mask.astype(np.float64), minlength=n_labels)
    crossing_labels = np.flatnonzero((mask_counts > 0) & (mask_counts < counts))

    if crossing_labels.size == 0:
        return labels

    out = labels.copy()
    next_label = n_labels
    for label in crossing_labels:
        out[(labels == label) & mask] = next_label
        next_label += 1

    return _compact_labels(out)


def _absorb_small_regions(
    labels: np.ndarray,
    *,
    max_area_px: int,
    barrier_mask: np.ndarray | None = None,
) -> np.ndarray:
    from scipy import ndimage as ndi

    labels = _split_connected_components(labels)
    if max_area_px <= 0:
        return labels

    if barrier_mask is None:
        masks = [np.ones(labels.shape, dtype=bool)]
    else:
        barrier_mask = np.asarray(barrier_mask, dtype=bool)
        if labels.shape != barrier_mask.shape:
            raise ValueError("labels and barrier_mask must have the same shape.")
        masks = [~barrier_mask, barrier_mask]

    out = labels.copy()
    for domain_mask in masks:
        domain_labels = out[domain_mask]
        domain_labels = domain_labels[domain_labels >= 0]
        if domain_labels.size == 0:
            continue

        counts = np.bincount(domain_labels)
        small_label_ids = np.flatnonzero((counts > 0) & (counts <= max_area_px))
        if small_label_ids.size == 0:
            continue

        remove_mask = domain_mask & np.isin(out, small_label_ids)
        target_mask = domain_mask & (~remove_mask) & (out >= 0)
        if not np.any(remove_mask) or not np.any(target_mask):
            continue

        _, nearest = ndi.distance_transform_edt(~target_mask, return_indices=True)
        out[remove_mask] = out[nearest[0][remove_mask], nearest[1][remove_mask]]

    return _split_connected_components(_compact_labels(out))


def make_theta_phi_angle_info(raw_maps: RawMaps) -> dict[str, Any]:
    phi = np.asarray(raw_maps["extinction_angle"], dtype=np.float64)
    brightest = _as_uint8_rgb(raw_maps["R_color_map"])
    darkest = _as_uint8_rgb(raw_maps["extinction_color_map"])
    theta = _make_theta_from_dark_bright(
        darkest,
        brightest,
    )
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
    small_region_max_area_px: int = 3,
) -> dict[str, Any]:
    source = _as_uint8_rgb(theta_phi_angle_info.get("shock_angle_map", theta_phi_angle_info["theta_phi_angle_map"]))
    outside_circle_mask = create_outside_circle_mask(source)
    superpixel_labels = _make_superpixel_labels(
        source,
        slic_region_size=slic_region_size,
        slic_ruler=slic_ruler,
        slic_num_iterations=slic_num_iterations,
        slic_min_element_size=slic_min_element_size,
    )
    superpixel_labels = _split_labels_by_mask(superpixel_labels, outside_circle_mask)
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
    outside_counts = np.bincount(
        flat_labels,
        weights=outside_circle_mask.reshape(-1).astype(np.float64),
        minlength=n_labels,
    )
    is_outside_label = outside_counts > 0
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
            if is_outside_label[a] != is_outside_label[b]:
                continue
            if _orientation_distance_deg(region_theta[a], region_phi[a], region_theta[b], region_phi[b]) <= delta_euler_thresh:
                union(int(a), int(b))

    root_to_label: dict[int, int] = {}
    lookup = np.full(n_labels, -1, dtype=np.int32)
    for label in np.unique(flat_labels):
        root = find(int(label))
        root_to_label.setdefault(root, len(root_to_label))
        lookup[label] = root_to_label[root]
    merged_labels = lookup[superpixel_labels]
    merged_labels = _split_connected_components(_compact_labels(merged_labels))
    merged_labels = _absorb_small_regions(
        merged_labels,
        max_area_px=small_region_max_area_px,
        barrier_mask=outside_circle_mask,
    )
    merged_superpixel_median_map = _region_median_color_map(source, merged_labels)
    return {
        **theta_phi_angle_info,
        "superpixel_labels": superpixel_labels,
        "superpixel_median_map": superpixel_median_map,
        "merged_superpixel_median_map": merged_superpixel_median_map,
        "merged_superpixel_labels": merged_labels,
        "black_artifact_removed_superpixel_labels": merged_labels,
        "angle_map_display": merged_superpixel_median_map,
    }


def fill_dark_boundaries(
    theta_phi_angle_info: dict[str, Any],
    *,
    branch_width_thresh: float,
    dark_l_thresh: float | None = 15.0,
    max_iterations: int = 3,
    fixed_skeleton_once: bool = False,
) -> dict[str, Any]:
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    labels = np.asarray(theta_phi_angle_info["merged_superpixel_labels"], dtype=np.int32).copy()
    base_rgb = _as_uint8_rgb(
        theta_phi_angle_info.get(
            "merged_superpixel_median_map",
            theta_phi_angle_info["superpixel_median_map"],
        )
    )
    height, width = labels.shape
    removed_mask_total = np.zeros(labels.shape, dtype=bool)
    lab_l = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]

    def label_median_l(component_labels: np.ndarray) -> dict[int, float]:
        values = {}
        for label in np.unique(component_labels[component_labels >= 0]):
            values[int(label)] = float(np.median(lab_l[component_labels == label]))
        return values

    def thin_branch_mask(region_mask: np.ndarray) -> np.ndarray:
        skeleton = skeletonize(region_mask)
        if not np.any(skeleton):
            return np.zeros_like(region_mask, dtype=bool)
        dist = cv2.distanceTransform(region_mask.astype(np.uint8), cv2.DIST_L2, 3)
        skeleton_width = 2.0 * dist
        thin_skeleton = skeleton & (skeleton_width <= float(branch_width_thresh))
        if not np.any(thin_skeleton):
            return np.zeros_like(region_mask, dtype=bool)
        n_branches, branch_labels = cv2.connectedComponents(thin_skeleton.astype(np.uint8), connectivity=8)
        mask = np.zeros_like(region_mask, dtype=bool)
        for branch_id in range(1, n_branches):
            branch = branch_labels == branch_id
            if not np.any(branch) or float(np.median(skeleton_width[branch])) > branch_width_thresh:
                continue
            for y, x in zip(*np.nonzero(branch)):
                radius = max(1, int(np.ceil(skeleton_width[y, x] / 2.0)))
                y_slice = slice(max(0, y - radius), min(height, y + radius + 1))
                x_slice = slice(max(0, x - radius), min(width, x + radius + 1))
                yy, xx = np.ogrid[y_slice, x_slice]
                mask[y_slice, x_slice] |= (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2

        mask &= region_mask
        directional_width = np.zeros_like(dist)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for y, x in zip(*np.nonzero(mask)):
            widths = []
            for dy, dx in directions:
                line_width = 1
                for sign in (-1, 1):
                    yy = y + sign * dy
                    xx = x + sign * dx
                    while 0 <= yy < height and 0 <= xx < width and region_mask[yy, xx]:
                        line_width += 1
                        yy += sign * dy
                        xx += sign * dx
                widths.append(line_width)
            directional_width[y, x] = float(np.median(widths))
        return mask & (directional_width <= float(branch_width_thresh))

    def candidate_labels(component_labels: np.ndarray) -> list[int]:
        unique_labels = np.unique(component_labels[component_labels >= 0])
        if dark_l_thresh is None:
            return [int(label) for label in unique_labels]
        return [
            label
            for label, l_value in label_median_l(component_labels).items()
            if l_value <= float(dark_l_thresh)
        ]

    def removable_mask(component_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        remove_mask = np.zeros(component_labels.shape, dtype=bool)
        candidate_mask = np.zeros(component_labels.shape, dtype=bool)
        for label in candidate_labels(component_labels):
            region_mask = component_labels == label
            candidate_mask |= region_mask
            remove_mask |= thin_branch_mask(region_mask)
        return remove_mask, candidate_mask

    fixed_remove_mask = None
    for _ in range(int(max_iterations)):
        labels = _split_connected_components(_compact_labels(labels))
        candidate_label_ids = candidate_labels(labels)
        if not candidate_label_ids:
            break

        candidate_mask = np.isin(labels, candidate_label_ids)
        if fixed_skeleton_once:
            if fixed_remove_mask is None:
                fixed_remove_mask, _ = removable_mask(labels)
            remove_mask = fixed_remove_mask & candidate_mask
        else:
            remove_mask, candidate_mask = removable_mask(labels)

        if not np.any(remove_mask):
            break

        if dark_l_thresh is None:
            next_labels = labels.copy()
            changed_any_label = False
            for label in np.unique(labels[remove_mask]):
                label_remove_mask = remove_mask & (labels == label)
                target_mask = (~remove_mask) & (labels != label)
                if not np.any(target_mask):
                    continue
                _, nearest = ndi.distance_transform_edt(~target_mask, return_indices=True)
                next_labels[label_remove_mask] = labels[
                    nearest[0][label_remove_mask],
                    nearest[1][label_remove_mask],
                ]
                changed_any_label = True
            if not changed_any_label:
                break
        else:
            target_mask = (~remove_mask) & (~candidate_mask)
            if not np.any(target_mask):
                target_mask = ~remove_mask
            if not np.any(target_mask):
                break

            _, nearest = ndi.distance_transform_edt(~target_mask, return_indices=True)
            next_labels = labels.copy()
            next_labels[remove_mask] = labels[nearest[0][remove_mask], nearest[1][remove_mask]]
        if np.array_equal(next_labels, labels):
            break

        labels = next_labels
        removed_mask_total |= remove_mask

    labels = _split_connected_components(_compact_labels(labels))
    cleaned_angle_map = _region_median_color_map(base_rgb, labels)
    return {
        **theta_phi_angle_info,
        "black_artifact_removed_superpixel_labels": labels,
        "black_artifact_removed_angle_map": cleaned_angle_map,
        "black_artifact_removed_mask": np.where(removed_mask_total, 255, 0).astype(np.uint8),
        "angle_map_display": cleaned_angle_map,
    }


def fill_dark_boundaries_by_elongation(
    theta_phi_angle_info: dict[str, Any],
    *,
    elongation_thresh: float,
    min_area: int = 10,
    neck_thresh: float = 2.0,
) -> dict[str, Any]:
    """Remove elongated dark boundaries using skeleton_length/area ratio.

    Each connected component is first split at narrow "neck" points
    (distance-transform value <= neck_thresh) so that whisker-like protrusions
    attached to large dark regions are evaluated as independent sub-components.
    Sub-components where skeleton_length/area >= elongation_thresh are removed
    and reassigned to the nearest non-removed neighbor.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    labels = np.asarray(
        theta_phi_angle_info["merged_superpixel_labels"], dtype=np.int32
    ).copy()
    base_rgb = _as_uint8_rgb(
        theta_phi_angle_info.get(
            "merged_superpixel_median_map",
            theta_phi_angle_info["superpixel_median_map"],
        )
    )

    labels = _split_connected_components(_compact_labels(labels))

    def _is_elongated(region_mask: np.ndarray) -> bool:
        area = int(np.count_nonzero(region_mask))
        if area < min_area:
            return False
        skeleton = skeletonize(region_mask)
        skeleton_length = int(np.count_nonzero(skeleton))
        if skeleton_length == 0:
            return False
        return skeleton_length / area >= elongation_thresh

    remove_mask = np.zeros(labels.shape, dtype=bool)
    for label in np.unique(labels[labels >= 0]):
        region_mask = labels == label
        if int(np.count_nonzero(region_mask)) < min_area:
            continue

        # Split at neck pixels (narrow connections between whisker and body)
        dist = cv2.distanceTransform(region_mask.astype(np.uint8), cv2.DIST_L2, 3)
        neck_mask = region_mask & (dist <= neck_thresh)
        body_mask = region_mask & ~neck_mask

        n_sub, sub_labels = cv2.connectedComponents(body_mask.astype(np.uint8), connectivity=8)
        if n_sub <= 1:
            # No neck splitting occurred; evaluate the region as a whole
            if _is_elongated(region_mask):
                remove_mask |= region_mask
            continue

        # Assign neck pixels to their nearest sub-component so every pixel is covered
        _, nearest_sub = ndi.distance_transform_edt(sub_labels == 0, return_indices=True)
        full_sub_labels = sub_labels.copy()
        neck_pixels = neck_mask & (sub_labels == 0)
        full_sub_labels[neck_pixels] = sub_labels[
            nearest_sub[0][neck_pixels], nearest_sub[1][neck_pixels]
        ]

        for sub_id in range(1, n_sub):
            sub_mask = (full_sub_labels == sub_id) & region_mask
            if _is_elongated(sub_mask):
                remove_mask |= sub_mask

    if np.any(remove_mask):
        target_mask = ~remove_mask
        _, nearest = ndi.distance_transform_edt(~target_mask, return_indices=True)
        next_labels = labels.copy()
        next_labels[remove_mask] = labels[nearest[0][remove_mask], nearest[1][remove_mask]]
        labels = _split_connected_components(_compact_labels(next_labels))

    cleaned_angle_map = _region_median_color_map(base_rgb, labels)
    return {
        **theta_phi_angle_info,
        "black_artifact_removed_superpixel_labels": labels,
        "black_artifact_removed_angle_map": cleaned_angle_map,
        "black_artifact_removed_mask": np.where(remove_mask, 255, 0).astype(np.uint8),
        "angle_map_display": cleaned_angle_map,
    }


def grain_boundary_from_angle_labels(theta_phi_angle_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(theta_phi_angle_info["black_artifact_removed_superpixel_labels"], dtype=np.int32)
    boundary = detect_boundaries(labels)
    return labels, boundary
