"""Per-grain inclination magnitude Theta (0 <= Theta < 90) from the addition
image, with a post-hoc arcsin/P95 correction and the implied thickness.

Ported from fix/ebsd_adjustment_v3/run_theta_from_color.py (+ grain_centers.py
find_grain_center_pixels / make_grain_theta_map). Uses the fixed quartz optical
constants (color_correction.OPTICAL_NO/NE) that the v3 model hard-codes, so the
result matches run_diagnostics numerically.
"""
from __future__ import annotations

import numpy as np

from niconavi_app.niconavi.color_correction import (
    OPTICAL_NO,
    OPTICAL_NE,
    LAMBDA_PLATE_FAST_AXIS_AZIMUTH_DEG,
    make_xpl_lambda_rgb_by_orientation,
    fit_alpha_huber,
    ALPHA_MIN,
    ALPHA_MAX,
)
from niconavi_app.niconavi.reconstruction import make_inscribed_circle_mask
from niconavi_app.niconavi.cpo_normalization import (
    apply_arcsin_percentile_correction,
    compute_corrected_thickness_mm,
)

# Theta LUT resolution for the add-image fit (deg).
THETA_STEP_DEG = 0.5
# Number of alpha samples in the initial (alpha, Theta) product-grid scan.
ALPHA_INIT_GRID_STEPS = 96
# Max robust-alpha <-> Theta alternations.
MAX_INIT_ALTERNATIONS = 30
# Minimum grain-center area (px) for a grain to contribute a sample.
MIN_GRAIN_CENTER_AREA_PX = 5

# Display azimuth of the lambda-plate SLOW axis: the addition position puts the
# (length-slow) c-axis parallel to it.
LAMBDA_SLOW_AXIS_AZIMUTH_DEG = (LAMBDA_PLATE_FAST_AXIS_AZIMUTH_DEG + 90.0) % 180.0


def make_addition_image(raw_maps: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel addition image from the phi_ex +/- 45 deg XPL+lambda color
    maps, selecting p45 where p45_R_map >= m45_R_map (the same rule the azimuth
    convention uses). Returns (add_image uint8 HxWx3, p45_selected bool HxW)."""
    p45_color = raw_maps.get("p45_R_color_map")
    m45_color = raw_maps.get("m45_R_color_map")
    p45_r = raw_maps.get("p45_R_map")
    m45_r = raw_maps.get("m45_R_map")
    if p45_color is None or m45_color is None:
        raise ValueError("p45/m45 XPL+lambda color maps are required.")
    p45_color = np.asarray(p45_color)
    m45_color = np.asarray(m45_color)
    if p45_r is not None and m45_r is not None:
        p45_selected = np.asarray(p45_r) >= np.asarray(m45_r)
    else:
        blueness_p = p45_color[..., 2].astype(np.int32) - p45_color[..., 0]
        blueness_m = m45_color[..., 2].astype(np.int32) - m45_color[..., 0]
        p45_selected = blueness_p >= blueness_m
    add_image = np.where(p45_selected[..., None], p45_color, m45_color)
    return add_image.astype(np.uint8), p45_selected


def find_grain_center_pixels(
    grain_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    min_area_px: int = MIN_GRAIN_CENTER_AREA_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(grain_map)
    valid = labels > 0
    valid &= make_inscribed_circle_mask(labels.shape)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    selected_labels = labels[valid]
    unique_labels, counts = np.unique(selected_labels, return_counts=True)
    center_rows: list[int] = []
    center_cols: list[int] = []
    center_labels: list[int] = []
    for label, count in zip(unique_labels, counts):
        if count < min_area_px:
            continue
        label_mask = valid & (labels == label)
        rows, cols = np.nonzero(label_mask)
        if rows.size == 0:
            continue
        centroid_row = float(np.mean(rows))
        centroid_col = float(np.mean(cols))
        nearest_index = int(
            np.argmin((rows - centroid_row) ** 2 + (cols - centroid_col) ** 2)
        )
        center_rows.append(int(rows[nearest_index]))
        center_cols.append(int(cols[nearest_index]))
        center_labels.append(int(label))

    return (
        np.asarray(center_labels, dtype=np.int32),
        np.asarray(center_rows, dtype=np.int64),
        np.asarray(center_cols, dtype=np.int64),
    )


def build_add_color_lut(*, thickness_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Ideal addition colors f_add(Theta | t) on the Theta grid (0..90 deg).
    Returns (theta_grid (K,), lut (K, 3) in 0..255)."""
    theta_grid = np.arange(0.0, 90.0, THETA_STEP_DEG, dtype=np.float64)
    colors = make_xpl_lambda_rgb_by_orientation(
        np.full(theta_grid.shape, LAMBDA_SLOW_AXIS_AZIMUTH_DEG),
        theta_grid,
        thickness_mm=thickness_mm,
    ).reshape(-1, 3)
    return theta_grid, np.clip(colors * 255.0, 0.0, 255.0)


def fit_theta_alpha_from_add_colors(
    add_lut: np.ndarray,
    add_rgb: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit y_g ~ alpha_add * f_add(Theta_g) (M=I, b=0): an alpha-grid scan with
    per-grain best Theta, then alternating robust alpha refits with exhaustive
    Theta reassignment. Returns (theta_index (G,), alpha_add)."""
    lut_sq = np.einsum("kc,kc->k", add_lut, add_lut)
    lut_dot = np.einsum("kc,gc->gk", add_lut, add_rgb)
    original_sq = np.einsum("gc,gc->g", add_rgb, add_rgb)
    alpha_grid = np.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_INIT_GRID_STEPS + 1)[1:]
    best_alpha = float(alpha_grid[0])
    best_loss = np.inf
    for alpha in alpha_grid:
        distance_sq = (
            alpha**2 * lut_sq[None, :] - 2.0 * alpha * lut_dot + original_sq[:, None]
        )
        min_distance_sq = np.maximum(np.min(distance_sq, axis=1), 0.0)
        loss = float(np.mean(np.sqrt(min_distance_sq)))
        if loss < best_loss:
            best_alpha = float(alpha)
            best_loss = loss

    def assign(alpha_value: float) -> np.ndarray:
        residual = alpha_value * add_lut[None, :, :] - add_rgb[:, None, :]
        return np.argmin(np.einsum("gkc,gkc->gk", residual, residual), axis=1)

    alpha = best_alpha
    theta_index = assign(alpha)
    for _ in range(MAX_INIT_ALTERNATIONS):
        predicted = add_lut[theta_index]
        next_alpha, _ = fit_alpha_huber(predicted, add_rgb)
        next_index = assign(next_alpha)
        converged = (
            abs(next_alpha - alpha) < 1e-6 and np.array_equal(next_index, theta_index)
        )
        alpha = next_alpha
        theta_index = next_index
        if converged:
            break
    return theta_index, alpha


def make_grain_theta_map(
    grain_map: np.ndarray,
    grain_labels: np.ndarray,
    theta_deg: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Paint each grain's estimated Theta over its pixels (NaN elsewhere)."""
    labels = np.asarray(grain_map)
    lookup = np.full(int(labels.max()) + 1, np.nan, dtype=np.float64)
    lookup[np.asarray(grain_labels, dtype=np.int64)] = np.asarray(
        theta_deg, dtype=np.float64
    )
    theta_map = np.full(labels.shape, np.nan, dtype=np.float64)
    inside = labels > 0
    theta_map[inside] = lookup[labels[inside]]
    if valid_mask is not None:
        theta_map[~np.asarray(valid_mask, dtype=bool)] = np.nan
    return theta_map


def estimate_theta_magnitude_from_add_image(
    raw_maps: dict,
    grain_map: np.ndarray,
    valid_mask: np.ndarray,
    thickness_mm: float,
    *,
    normalize_90: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Grain-constant inclination magnitude map in [0, 90] (NaN outside
    segmented grains), the arcsin/P95-corrected thickness, and Theta_error_max.

    When normalize_90 is False the arcsin/P95 correction is skipped: the map is
    the raw add-image Theta and the returned thickness equals thickness_mm.
    """
    add_image, _p45_selected = make_addition_image(raw_maps)
    labels, rows, cols = find_grain_center_pixels(grain_map, valid_mask)
    if labels.size == 0:
        raise ValueError("No valid grain-center samples were found.")
    sample_add_rgb = add_image.astype(np.float64)[rows, cols]

    theta_grid, add_lut = build_add_color_lut(thickness_mm=thickness_mm)
    theta_index, _alpha_add = fit_theta_alpha_from_add_colors(add_lut, sample_add_rgb)
    theta_raw = theta_grid[theta_index]

    if normalize_90:
        theta_used, theta_error_max = apply_arcsin_percentile_correction(theta_raw)
        corrected_thickness = compute_corrected_thickness_mm(
            theta_error_max, thickness_mm, no=OPTICAL_NO, ne=OPTICAL_NE
        )
    else:
        theta_used = theta_raw
        theta_error_max = float(np.nanpercentile(theta_raw, 95.0))
        corrected_thickness = thickness_mm

    theta_map = make_grain_theta_map(
        grain_map, labels, theta_used, valid_mask=valid_mask
    )
    return theta_map, corrected_thickness, theta_error_max
