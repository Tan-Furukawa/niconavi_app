"""CPO "90 deg normalize": rescale a population of grain inclinations so a
chosen percentile maps to 90 deg, and derive the thickness that reading
implies.

Ported from fix/ebsd_adjustment_v3/run_theta_from_color.py
(apply_arcsin_percentile_correction / compute_corrected_thickness_mm). The
percentile is an analysis-tab input defaulting to app_config's
CPO_NORMALIZE_PERCENTILE; the refractive indices are supplied by the caller
(params.optical_parameters.no/ne) rather than hard-coded as they were in the v3
stereo module.
"""

from __future__ import annotations

import numpy as np

from niconavi_app.app_config import CPO_NORMALIZE_PERCENTILE


def apply_arcsin_percentile_correction(
    theta_deg: np.ndarray,
    *,
    percentile: float = CPO_NORMALIZE_PERCENTILE,
) -> tuple[np.ndarray, float]:
    """Theta_corrected = arcsin(sin(Theta) / sin(Theta_error_max)), with
    Theta_error_max the `percentile`-th percentile of theta_deg over the grain
    population - a robust stand-in for "the most steeply inclined grain" that
    is less sensitive to a single outlier than the max.

    This rescales sin(Theta) so the percentile grain maps to 90 deg; grains
    above that threshold have their ratio clipped to 1 (arcsin domain) and so
    also map to 90 deg. Returns (theta_corrected (same shape), theta_error_max used).
    NaNs pass through unchanged and are ignored by the percentile.
    """
    theta = np.asarray(theta_deg, dtype=np.float64)
    theta_error_max = float(np.nanpercentile(theta, percentile))
    sin_max = np.sin(np.deg2rad(theta_error_max))
    if sin_max == 0.0:
        return theta.copy(), theta_error_max
    ratio = np.clip(np.sin(np.deg2rad(theta)) / sin_max, -1.0, 1.0)
    return np.rad2deg(np.arcsin(ratio)), theta_error_max


def rescale_inclination_map_with_theta_error_max(
    theta_deg: np.ndarray,
    theta_error_max_deg: float,
) -> np.ndarray:
    """Apply the same arcsin/percentile rescale as
    apply_arcsin_percentile_correction but with an externally supplied
    theta_error_max (so a per-pixel inclination MAP can be normalized with the
    threshold derived from the selected grains). The
    0-180 hemisphere structure is preserved: a lower-hemisphere reading
    (90..180) stays lower after correction. NaNs pass through unchanged."""
    theta = np.asarray(theta_deg, dtype=np.float64)
    folded = theta % 180.0
    was_lower = folded > 90.0
    folded = np.where(was_lower, 180.0 - folded, folded)  # -> [0, 90]

    sin_max = np.sin(np.deg2rad(theta_error_max_deg))
    if sin_max == 0.0:
        return theta.copy()
    ratio = np.clip(np.sin(np.deg2rad(folded)) / sin_max, -1.0, 1.0)
    corrected = np.rad2deg(np.arcsin(ratio))  # [0, 90]
    return np.where(was_lower, 180.0 - corrected, corrected)


def compute_corrected_thickness_mm(
    theta_error_max_deg: float,
    t_false_mm: float,
    *,
    no: float,
    ne: float,
) -> float:
    """The arcsin/percentile correction implicitly assumes the P95 grain
    (Theta_error_max, uncorrected) is truly at Theta=90 deg. Both readings
    describe the same physical retardation R = t * dn(Theta), so

        t_false * dn(Theta_error_max) = t_true * dn(90 deg) = t_true * (ne - no)

    with n_eff(Theta) = (cos^2 Theta / no^2 + sin^2 Theta / ne^2)^(-1/2) and
    dn(Theta) = n_eff(Theta) - no, giving

        t_true = t_false * (n_eff(Theta_error_max) - no) / (ne - no).
    """
    theta_rad = np.deg2rad(theta_error_max_deg)
    n_eff = (np.cos(theta_rad) ** 2 / no**2 + np.sin(theta_rad) ** 2 / ne**2) ** -0.5
    return t_false_mm * (n_eff - no) / (ne - no)
