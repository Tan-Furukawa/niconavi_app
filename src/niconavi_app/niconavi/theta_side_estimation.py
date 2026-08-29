"""Resolve the Theta<90 / Theta>90 branch of the optical inclination from a
real E-down stage tilt, by comparing the observed before/after tilt color
change against the two predicted branch color changes (cosine similarity of the
RGB change vectors).

Ported from fix/ebsd_adjustment_v3/theta_side_estimation.py, dropping the
matplotlib diagnostics: resolve_derived_inclination returns the derived,
branch-resolved inclination map in [0, 180].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from niconavi_app.niconavi import orientation_geometry
from niconavi_app.niconavi.color_correction import (
    ColorCorrectionFit,
    apply_color_correction,
)
from niconavi_app.niconavi.reconstruction import (
    make_e_down_tilted_reconstructed_image,
    make_inscribed_circle_mask,
    make_stage0_reconstructed_image,
)
from niconavi_app.niconavi.type import TiltImageResult

TiltStage = Literal["0", "45", "both"]


def normalize_rgb_image(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float64)
    if image_array.max(initial=0.0) > 1.0:
        image_array = image_array / 255.0
    return np.clip(image_array, 0.0, 1.0)


def corrected_prediction_unit_image(
    predicted_image: np.ndarray, fit: ColorCorrectionFit
) -> np.ndarray:
    return apply_color_correction(predicted_image, fit) / 255.0


def optical_inclination_from_theta_side_map(
    inclination_deg: np.ndarray,
    theta_side_map: np.ndarray,
) -> np.ndarray:
    """inclination_deg is the pre-branch magnitude in [0, 90]; theta_side_map
    picks the branch (>= 0.5 means Theta>90)."""
    inclination = np.asarray(inclination_deg, dtype=np.float64)
    theta_side = np.asarray(theta_side_map, dtype=np.float64)
    derived = np.full(inclination.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(inclination) & np.isfinite(theta_side)
    derived[valid] = np.where(
        theta_side[valid] >= 0.5, 180.0 - inclination[valid], inclination[valid]
    )
    return derived


def low_azimuth_mask(azimuth_deg: np.ndarray, half_width_deg: float) -> np.ndarray:
    azimuth = np.asarray(azimuth_deg, dtype=np.float64) % 180.0
    distance_to_axis = np.minimum(azimuth, 180.0 - azimuth)
    return np.isfinite(azimuth) & (distance_to_axis <= half_width_deg)


@dataclass
class ThetaSideFields:
    valid_mask: np.ndarray
    similarity_lt90: np.ndarray
    similarity_gt90: np.ndarray


def compute_theta_side_fields(
    *,
    tilt_result: TiltImageResult,
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    fit: ColorCorrectionFit,
    stage_rotation_deg: float = 0.0,
    baseline_predicted_image: Optional[np.ndarray] = None,
) -> ThetaSideFields:
    valid_mask = np.asarray(tilt_result["image_mask"], dtype=bool).copy()
    valid_mask &= make_inscribed_circle_mask(valid_mask.shape)
    valid_mask &= np.isfinite(azimuth_deg) & np.isfinite(inclination_deg)

    predicted_azimuth = np.asarray(azimuth_deg, dtype=np.float64) + stage_rotation_deg
    if baseline_predicted_image is None:
        baseline_predicted_image = make_stage0_reconstructed_image(
            predicted_azimuth,
            inclination_deg,
            thickness_mm=fit.thickness_mm,
            valid_mask=valid_mask,
        )

    predicted_e_down_lt90 = make_e_down_tilted_reconstructed_image(
        azimuth_deg,
        inclination_deg,
        thickness_mm=fit.thickness_mm,
        valid_mask=valid_mask,
        theta_side="lt90",
        stage_rotation_deg=stage_rotation_deg,
    )
    predicted_e_down_gt90 = make_e_down_tilted_reconstructed_image(
        azimuth_deg,
        inclination_deg,
        thickness_mm=fit.thickness_mm,
        valid_mask=valid_mask,
        theta_side="gt90",
        stage_rotation_deg=stage_rotation_deg,
    )

    corrected_stage0 = corrected_prediction_unit_image(baseline_predicted_image, fit)
    corrected_lt90 = corrected_prediction_unit_image(predicted_e_down_lt90, fit)
    corrected_gt90 = corrected_prediction_unit_image(predicted_e_down_gt90, fit)

    before_tilt = normalize_rgb_image(tilt_result["original_image"])
    after_tilt = normalize_rgb_image(tilt_result["focused_tilted_image"])
    observed_delta = after_tilt - before_tilt
    predicted_delta_lt90 = corrected_lt90 - corrected_stage0
    predicted_delta_gt90 = corrected_gt90 - corrected_stage0

    finite = (
        np.all(np.isfinite(observed_delta), axis=2)
        & np.all(np.isfinite(predicted_delta_lt90), axis=2)
        & np.all(np.isfinite(predicted_delta_gt90), axis=2)
    )
    valid_mask &= finite

    similarity_lt90 = np.full(valid_mask.shape, np.nan, dtype=np.float64)
    similarity_gt90 = np.full(valid_mask.shape, np.nan, dtype=np.float64)
    observed_vectors = observed_delta[valid_mask]
    predicted_vectors_lt90 = predicted_delta_lt90[valid_mask]
    predicted_vectors_gt90 = predicted_delta_gt90[valid_mask]
    observed_norm = np.linalg.norm(observed_vectors, axis=1)
    predicted_norm_lt90 = np.linalg.norm(predicted_vectors_lt90, axis=1)
    predicted_norm_gt90 = np.linalg.norm(predicted_vectors_gt90, axis=1)
    similarity_lt90[valid_mask] = np.sum(
        observed_vectors * predicted_vectors_lt90, axis=1
    ) / np.maximum(observed_norm * predicted_norm_lt90, 1e-12)
    similarity_gt90[valid_mask] = np.sum(
        observed_vectors * predicted_vectors_gt90, axis=1
    ) / np.maximum(observed_norm * predicted_norm_gt90, 1e-12)

    return ThetaSideFields(
        valid_mask=valid_mask,
        similarity_lt90=similarity_lt90,
        similarity_gt90=similarity_gt90,
    )


def resolve_derived_inclination(
    *,
    tilt_result: TiltImageResult,
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    stage0_predicted_image: np.ndarray,
    fit: ColorCorrectionFit,
    p45_tilt_result: Optional[TiltImageResult] = None,
    tilt_stage: TiltStage = "both",
    p45_stage_rotation_deg: float = 45.0,
    low_azimuth_half_width_deg: float = 30.0,
) -> np.ndarray:
    """Build the branch-resolved inclination map in [0, 180] from a real stage
    tilt. tilt_stage selects which tilt decides each pixel's branch:
    "0" (0 deg stage), "45" (+45 deg stage), or "both" (near-0/180 azimuth
    pixels from the +45 deg stage, the rest from the 0 deg stage)."""
    if tilt_stage not in ("0", "45", "both"):
        raise ValueError(f"Unknown tilt_stage: {tilt_stage!r}")
    if tilt_stage in ("45", "both") and p45_tilt_result is None:
        raise ValueError(
            f"tilt_stage={tilt_stage!r} needs the +45 deg stage tilt, but "
            "p45_tilt_result is None."
        )

    stage0_fields: Optional[ThetaSideFields] = None
    if tilt_stage in ("0", "both"):
        stage0_fields = compute_theta_side_fields(
            tilt_result=tilt_result,
            azimuth_deg=azimuth_deg,
            inclination_deg=inclination_deg,
            fit=fit,
            stage_rotation_deg=0.0,
            baseline_predicted_image=stage0_predicted_image,
        )

    p45_fields: Optional[ThetaSideFields] = None
    if tilt_stage in ("45", "both"):
        p45_fields = compute_theta_side_fields(
            tilt_result=p45_tilt_result,
            azimuth_deg=azimuth_deg,
            inclination_deg=inclination_deg,
            fit=fit,
            stage_rotation_deg=p45_stage_rotation_deg,
            baseline_predicted_image=None,
        )

    if tilt_stage == "0":
        similarity_lt90 = stage0_fields.similarity_lt90
        similarity_gt90 = stage0_fields.similarity_gt90
        final_valid = stage0_fields.valid_mask
    elif tilt_stage == "45":
        similarity_lt90 = p45_fields.similarity_lt90
        similarity_gt90 = p45_fields.similarity_gt90
        final_valid = p45_fields.valid_mask
    else:
        similarity_lt90 = stage0_fields.similarity_lt90.copy()
        similarity_gt90 = stage0_fields.similarity_gt90.copy()
        final_valid = stage0_fields.valid_mask.copy()
        uses_p45 = (
            low_azimuth_mask(azimuth_deg, low_azimuth_half_width_deg)
            & p45_fields.valid_mask
        )
        similarity_lt90[uses_p45] = p45_fields.similarity_lt90[uses_p45]
        similarity_gt90[uses_p45] = p45_fields.similarity_gt90[uses_p45]
        final_valid |= uses_p45

    theta_gt90 = final_valid & (similarity_gt90 > similarity_lt90)
    theta_side_map = np.full(final_valid.shape, np.nan, dtype=np.float64)
    theta_side_map[final_valid] = np.where(theta_gt90[final_valid], 1.0, 0.0)

    return optical_inclination_from_theta_side_map(inclination_deg, theta_side_map)
