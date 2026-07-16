"""Drive the run_diagnostics.py CPO orientation pipeline on a niconavi
ComputationResult, producing the addition-image-derived, arcsin/P95-corrected,
E-down-tilt-branch-resolved inclination map that the CPO stereo plots consume -
identical to fix/ebsd_adjustment_v3/run_diagnostics.py.

The result maps are written back into raw_maps the same way niconavi's original
run_all.get_inclination did (inclination = normalized magnitude,
inclination_0_to_180 = the [0, 180] derived axis, azimuth360 = normalized
azimuth), so every downstream CPO plot (grain and pixel, 90/180/360, map-COI)
uses the run_diagnostics orientation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from niconavi_app.niconavi.color_correction import (
    ColorCorrectionFit,
    fit_thickness_factor_and_alpha,
)
from niconavi_app.niconavi.optics.tools import normalize_axes
from niconavi_app.niconavi.reconstruction import make_stage0_reconstructed_image
from niconavi_app.niconavi.theta_from_add import (
    estimate_theta_magnitude_from_add_image,
    find_grain_center_pixels,
)
from niconavi_app.niconavi.theta_side_estimation import resolve_derived_inclination
from niconavi_app.niconavi.type import ComputationResult

# Match run_diagnostics.py's config block.
THETA_SIDE_TILT_STAGE = "both"
P45_STAGE_ROTATION_DEG = 45.0
LOW_AZIMUTH_HALF_WIDTH_DEG = 30.0


@dataclass
class CPOOrientationResult:
    theta_magnitude_map: np.ndarray  # [0, 90], grain-constant, NaN outside grains
    derived_inclination_0_to_180: np.ndarray  # branch-resolved [0, 180]
    optical_azimuth: np.ndarray  # raw_maps azimuth % 180
    corrected_thickness_mm: float
    theta_error_max_deg: float
    fit: Optional[ColorCorrectionFit]
    n_grain_samples: int


def estimate_cpo_orientation(
    result: ComputationResult,
    *,
    normalize_90: bool = True,
) -> CPOOrientationResult:
    """Run the add-image theta + color-correction fit + E-down side resolution
    pipeline. Requires raw_maps (with p45/m45 color+R maps), a grain map, and
    the 0 deg tilt result. Uses the +45 deg tilt too when present."""
    raw_maps = result.raw_maps
    grain_map = result.grain_map
    tilt0 = result.tilt_image_info.tilt_image0
    tilt45 = result.tilt_image_info.tilt_image45
    if raw_maps is None or grain_map is None or tilt0 is None:
        raise ValueError(
            "CPO orientation needs raw_maps, a grain map and the 0 deg tilt result."
        )

    thickness0 = result.optical_parameters.thickness
    valid_mask = np.asarray(tilt0["image_mask"], dtype=bool)
    azimuth_map = np.asarray(raw_maps["azimuth"], dtype=np.float64) % 180.0

    # 1. Inclination magnitude [0, 90] (arcsin/P95 corrected when normalize_90)
    #    + the thickness that reading implies.
    theta_magnitude_map, corrected_thickness, theta_error_max = (
        estimate_theta_magnitude_from_add_image(
            raw_maps, grain_map, valid_mask, thickness0, normalize_90=normalize_90
        )
    )

    # 2. Color-correction fit (alpha, M, b) from grain-center samples: the
    #    corrected magnitude + azimuth predict the color, compared with the
    #    observed 0 deg XPL+lambda (before-tilt) color at each grain center.
    labels, rows, cols = find_grain_center_pixels(grain_map, valid_mask)
    fit: Optional[ColorCorrectionFit] = None
    if labels.size >= 4:
        sample_azimuth = azimuth_map[rows, cols]
        sample_inclination = theta_magnitude_map[rows, cols]
        sample_rgb = np.asarray(tilt0["original_image"], dtype=np.float64)[rows, cols]
        fit, _predicted = fit_thickness_factor_and_alpha(
            azimuth_deg=sample_azimuth,
            inclination_deg=sample_inclination,
            original_rgb=sample_rgb,
            base_thickness_mm=corrected_thickness,
        )

    # 3. Resolve the Theta<90/Theta>90 branch from the real E-down tilt(s).
    if fit is None:
        # Without a color fit the branch cannot be resolved; keep the magnitude.
        derived = theta_magnitude_map.copy()
    else:
        stage0_predicted = make_stage0_reconstructed_image(
            azimuth_map,
            theta_magnitude_map,
            thickness_mm=fit.thickness_mm,
            valid_mask=valid_mask,
        )
        tilt_stage = THETA_SIDE_TILT_STAGE if tilt45 is not None else "0"
        derived = resolve_derived_inclination(
            tilt_result=tilt0,
            azimuth_deg=azimuth_map,
            inclination_deg=theta_magnitude_map,
            stage0_predicted_image=stage0_predicted,
            fit=fit,
            p45_tilt_result=tilt45,
            tilt_stage=tilt_stage,
            p45_stage_rotation_deg=P45_STAGE_ROTATION_DEG,
            low_azimuth_half_width_deg=LOW_AZIMUTH_HALF_WIDTH_DEG,
        )

    return CPOOrientationResult(
        theta_magnitude_map=theta_magnitude_map,
        derived_inclination_0_to_180=derived,
        optical_azimuth=azimuth_map,
        corrected_thickness_mm=corrected_thickness,
        theta_error_max_deg=theta_error_max,
        fit=fit,
        n_grain_samples=int(labels.size),
    )


def format_cpo_orientation_info(
    *,
    orientation: CPOOrientationResult,
    normalize_90: bool,
    displayed_minerals: Optional[list] = None,
) -> str:
    """Multi-line CPO Info-panel text: the grains/orientation source, the
    color-correction fit (M, b, alpha), and the predicted (corrected)
    thickness when 90 deg normalize is on."""
    lines: list[str] = []
    lines.append("Orientation: addition image + E-down tilt (run_diagnostics)")
    lines.append(f"Grain samples: {orientation.n_grain_samples}")
    if displayed_minerals:
        lines.append(f"Displayed minerals: {', '.join(displayed_minerals)}")

    lines.append("")
    lines.append(f"90 deg normalize: {'on' if normalize_90 else 'off'}")
    if normalize_90:
        lines.append(
            f"  Theta(P95) -> 90 deg: {orientation.theta_error_max_deg:.2f} deg"
        )
        lines.append(
            f"  Predicted thickness: {orientation.corrected_thickness_mm:.5f} mm"
        )

    lines.append("")
    fit = orientation.fit
    if fit is None:
        lines.append("Color fit (M, b, alpha): N/A (too few grain samples)")
    else:
        lines.append("Color fit  y = alpha * M @ predicted + b")
        lines.append(f"  alpha: {fit.alpha:.4f}")
        lines.append(f"  trace(M): {float(np.trace(fit.matrix)):.4f}")
        lines.append("  M:")
        for row in fit.matrix:
            lines.append("    [" + ", ".join(f"{value:7.4f}" for value in row) + "]")
        lines.append(
            "  b: [" + ", ".join(f"{value:.2f}" for value in fit.bias_rgb) + "]"
        )
        lines.append(f"  robust loss: {fit.robust_loss:.5f}")
    return "\n".join(lines)


def write_cpo_orientation_into_raw_maps(
    raw_maps: dict,
    orientation: CPOOrientationResult,
) -> None:
    """Store the derived orientation into raw_maps the way get_inclination did:
    inclination_0_to_180 = derived axis, and (inclination, azimuth360) =
    normalize_axes(derived, azimuth) so every CPO plot reads it."""
    derived = orientation.derived_inclination_0_to_180
    n_inclination, n_azimuth = normalize_axes(derived, orientation.optical_azimuth)
    raw_maps["inclination_0_to_180"] = derived
    raw_maps["inclination"] = n_inclination
    raw_maps["azimuth360"] = n_azimuth
