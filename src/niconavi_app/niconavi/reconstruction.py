"""Reconstruct predicted XPL+lambda color images from azimuth/inclination
angle maps, optionally under a simulated E-down stage tilt.

Ported from fix/ebsd_adjustment_v3/reconstruction.py; the XPL+lambda color
model is color_correction.make_xpl_lambda_rgb_by_orientation (same physics),
and the E-down tilt geometry is niconavi.orientation_geometry.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from niconavi_app.niconavi import orientation_geometry
from niconavi_app.niconavi.orientation_geometry import tilt_orientation_e_down
from niconavi_app.niconavi.color_correction import (
    make_xpl_lambda_rgb_by_orientation,
)

RECONSTRUCTION_ANGLE_STEP_DEG = 2.0


def make_inscribed_circle_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    rows, cols = np.indices((height, width))
    center_row = (height - 1) / 2.0
    center_col = (width - 1) / 2.0
    radius = min(height, width) / 2.0
    return (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius**2


def make_stage0_reconstructed_image(
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    *,
    thickness_mm: float,
    no: float | None = None,
    ne: float | None = None,
    valid_mask: np.ndarray | None = None,
    angle_step_deg: float = RECONSTRUCTION_ANGLE_STEP_DEG,
) -> np.ndarray:
    azimuth = np.asarray(azimuth_deg, dtype=np.float64)
    inclination = np.asarray(inclination_deg, dtype=np.float64)
    valid = np.isfinite(azimuth) & np.isfinite(inclination)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    reconstructed = np.zeros((*azimuth.shape, 3), dtype=np.float64)
    if not np.any(valid):
        return reconstructed

    quantized_azimuth = (
        np.round(azimuth[valid] / angle_step_deg) * angle_step_deg
    ) % 180.0
    quantized_inclination = (
        np.round(inclination[valid] / angle_step_deg) * angle_step_deg
    ) % 180.0
    reconstructed[valid] = _colors_for(
        quantized_azimuth, quantized_inclination, thickness_mm, no, ne
    )
    return np.clip(reconstructed, 0.0, 1.0)


def make_xpl_lambda_reconstructed_image(
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    *,
    thickness_mm: float,
    no: float | None = None,
    ne: float | None = None,
    valid_mask: np.ndarray | None = None,
    angle_step_deg: float = RECONSTRUCTION_ANGLE_STEP_DEG,
) -> np.ndarray:
    azimuth = np.asarray(azimuth_deg, dtype=np.float64)
    inclination = np.asarray(inclination_deg, dtype=np.float64)
    valid = np.isfinite(azimuth) & np.isfinite(inclination)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    reconstructed = np.zeros((*azimuth.shape, 3), dtype=np.float64)
    if not np.any(valid):
        return reconstructed

    quantized_azimuth = (
        np.round(azimuth[valid] / angle_step_deg) * angle_step_deg
    ) % 360.0
    quantized_inclination = np.clip(
        np.round(inclination[valid] / angle_step_deg) * angle_step_deg,
        0.0,
        180.0,
    )
    reconstructed[valid] = _colors_for(
        quantized_azimuth, quantized_inclination, thickness_mm, no, ne
    )
    return np.clip(reconstructed, 0.0, 1.0)


def _colors_for(
    quantized_azimuth: np.ndarray,
    quantized_inclination: np.ndarray,
    thickness_mm: float,
    no: float | None,
    ne: float | None,
) -> np.ndarray:
    orientation_pairs = np.column_stack([quantized_azimuth, quantized_inclination])
    unique_pairs, inverse = np.unique(orientation_pairs, axis=0, return_inverse=True)
    kwargs = {}
    if no is not None:
        kwargs["no"] = no
    if ne is not None:
        kwargs["ne"] = ne
    unique_colors = make_xpl_lambda_rgb_by_orientation(
        unique_pairs[:, 0],
        unique_pairs[:, 1],
        thickness_mm=thickness_mm,
        **kwargs,
    ).reshape(-1, 3)
    return unique_colors[inverse]


def make_e_down_tilted_reconstructed_image(
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    *,
    thickness_mm: float,
    no: float | None = None,
    ne: float | None = None,
    valid_mask: np.ndarray | None = None,
    tilt_deg: float | None = None,
    theta_side: Literal["as_is", "lt90", "gt90"] = "as_is",
    stage_rotation_deg: float = 0.0,
) -> np.ndarray:
    """theta_side="lt90"/"gt90" assume inclination_deg is the pre-branch
    magnitude in [0, 90]. gt90 mirrors inclination through the horizontal
    (180 - x) at fixed azimuth - matching optical_inclination_from_theta_side_map.
    stage_rotation_deg is added to the azimuth AFTER tilting, at color lookup
    only (the +/-45 deg stage tilt acts in the sample frame)."""
    if tilt_deg is None:
        tilt_deg = orientation_geometry.E_DOWN_TILT_DEG
    if theta_side in ("as_is", "lt90"):
        branch_azimuth, branch_inclination = azimuth_deg, inclination_deg
    elif theta_side == "gt90":
        branch_azimuth = azimuth_deg
        branch_inclination = 180.0 - np.asarray(inclination_deg, dtype=np.float64)
    else:
        raise ValueError(f"Unknown theta_side: {theta_side!r}")
    tilted_azimuth, tilted_inclination = tilt_orientation_e_down(
        branch_azimuth,
        branch_inclination,
        tilt_deg=tilt_deg,
        theta_side="as_is",
    )
    return make_xpl_lambda_reconstructed_image(
        tilted_azimuth + stage_rotation_deg,
        tilted_inclination,
        thickness_mm=thickness_mm,
        no=no,
        ne=ne,
        valid_mask=valid_mask,
    )
