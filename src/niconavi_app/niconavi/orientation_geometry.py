"""Convert between (azimuth, inclination) angles and 3D display-frame unit
vectors, and apply a simulated E-down stage tilt to an orientation.

Ported verbatim from fix/ebsd_adjustment_v3/orientation_geometry.py; only the
`stereo.fold_axis_to_upper_hemisphere` import is redirected to the copy in
color_correction (same function).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from niconavi_app.niconavi.color_correction import fold_axis_to_upper_hemisphere

# E-down stage tilt used to resolve the Theta<90/Theta>90 branch (deg).
E_DOWN_TILT_DEG = 10.0


def orientation_to_display_vectors(
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
) -> np.ndarray:
    azimuth = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    inclination = np.deg2rad(np.asarray(inclination_deg, dtype=np.float64))
    sin_inclination = np.sin(inclination)
    return np.stack(
        [
            sin_inclination * np.cos(azimuth),
            sin_inclination * np.sin(azimuth),
            np.cos(inclination),
        ],
        axis=-1,
    )


def display_vectors_to_orientation(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vector_array = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vector_array, axis=-1)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    unit = vector_array / safe_norms[..., None]
    inclination = np.rad2deg(np.arccos(np.clip(unit[..., 2], -1.0, 1.0)))
    azimuth = np.rad2deg(np.arctan2(unit[..., 1], unit[..., 0])) % 360.0
    return azimuth, inclination


def force_theta_branch(
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    *,
    theta_side: Literal["as_is", "lt90", "gt90"],
) -> tuple[np.ndarray, np.ndarray]:
    if theta_side == "as_is":
        return (
            np.asarray(azimuth_deg, dtype=np.float64),
            np.asarray(inclination_deg, dtype=np.float64),
        )
    upper_azimuth, upper_inclination = fold_axis_to_upper_hemisphere(
        azimuth_deg,
        inclination_deg,
    )
    if theta_side == "lt90":
        return upper_azimuth, upper_inclination
    if theta_side == "gt90":
        return (upper_azimuth + 180.0) % 360.0, 180.0 - upper_inclination
    raise ValueError(f"Unknown theta_side: {theta_side!r}")


def tilt_orientation_e_down(
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    *,
    tilt_deg: float | None = None,
    theta_side: Literal["as_is", "lt90", "gt90"] = "as_is",
) -> tuple[np.ndarray, np.ndarray]:
    if tilt_deg is None:
        tilt_deg = E_DOWN_TILT_DEG
    branch_azimuth, branch_inclination = force_theta_branch(
        azimuth_deg,
        inclination_deg,
        theta_side=theta_side,
    )
    vectors = orientation_to_display_vectors(branch_azimuth, branch_inclination)

    # coordinate.md display axes: x=W, y=S, z=out of the image. E down means the
    # E side (negative x) moves away from the camera, i.e. a negative rotation
    # around the S axis.
    angle = np.deg2rad(-tilt_deg)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    tilted = np.empty_like(vectors)
    tilted[..., 0] = vectors[..., 0] * cos_angle + vectors[..., 2] * sin_angle
    tilted[..., 1] = vectors[..., 1]
    tilted[..., 2] = -vectors[..., 0] * sin_angle + vectors[..., 2] * cos_angle
    return display_vectors_to_orientation(tilted)
