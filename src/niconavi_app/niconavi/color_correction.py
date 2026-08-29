"""Robust color correction for the CPO tab: fit a scalar alpha, a
trace-3-constrained 3x3 matrix M and a bias b that map predicted XPL+lambda
colors onto the observed grain-center colors, via iteratively-reweighted
Huber regression. The fitted M, b, alpha are reported in the CPO Info panel.

Ported from fix/ebsd_adjustment_v3/color_correction.py (+ the XPL+lambda color
prediction from that project's stereo.py / reconstruction.py). The prediction
uses niconavi's own optics primitives (optics.jones_matrix / optics.tools /
optics.uniaxial_plate), so this module has no dependency on the v3 project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

import niconavi_app.niconavi.optics.jones_matrix as jm
from niconavi_app.niconavi.optics.tools import (
    make_angle_retardation_estimation_function,
)
from niconavi_app.niconavi.optics.uniaxial_plate import get_spectral_distribution

# ---------------------------------------------------------------------------
# Optical constants (quartz + first-order red plate), matching the v3 model.
# no/ne can be overridden per call from params.optical_parameters.
# ---------------------------------------------------------------------------
OPTICAL_NO = 1.544
OPTICAL_NE = 1.553
LAMBDA_PLATE_RETARDATION_NM = 530.0
# coordinate.md: lambda-plate fast axis is NW-SE. With display azimuth measured
# counterclockwise from W, NW-SE is 135 deg.
LAMBDA_PLATE_FAST_AXIS_AZIMUTH_DEG = 135.0
# Orientation quantization for the predicted-color LUT (deg).
RECONSTRUCTION_ANGLE_STEP_DEG = 2.0

# ---------------------------------------------------------------------------
# Robust-fit hyper-parameters (copied from the v3 config block that
# run_diagnostics.py set on color_correction).
# ---------------------------------------------------------------------------
HUBER_TUNING_CONSTANT = 1.345
ALPHA_MIN = 0.0
ALPHA_MAX = 1.0
COLOR_MATRIX_REGULARIZATION = 10.0
COLOR_BIAS_REGULARIZATION = 2.0

ColorCorrectionModel = Literal["matrix_trace3_bias_fixed_thickness"]
COLOR_CORRECTION_MODEL: ColorCorrectionModel = "matrix_trace3_bias_fixed_thickness"


@dataclass(frozen=True)
class ColorCorrectionFit:
    correction_model: ColorCorrectionModel
    thickness_mm: float
    thickness_factor: float
    alpha: float
    matrix: np.ndarray
    bias_rgb: np.ndarray
    robust_loss: float


# ===========================================================================
# XPL + lambda color prediction from orientation (ported from v3 stereo.py).
# ===========================================================================
def fold_axis_to_upper_hemisphere(
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth = np.asarray(azimuth_deg, dtype=np.float64) % 360.0
    inclination = np.asarray(inclination_deg, dtype=np.float64) % 180.0
    lower = inclination > 90.0
    folded_inclination = np.where(lower, 180.0 - inclination, inclination)
    folded_azimuth = np.where(lower, azimuth + 180.0, azimuth) % 360.0
    return folded_azimuth, folded_inclination


def get_lambda_plate_plus_mineral_system(
    *,
    mineral_retardation_nm: float,
    mineral_azimuth_rad: float,
    lambda_plate_fast_axis_azimuth_deg: float,
    lambda_plate_retardation_nm: float,
    alpha: float = 1.0,
):
    mineral_rotation = jm.rotation(-mineral_azimuth_rad)
    mineral_rotation_back = jm.rotation(mineral_azimuth_rad)
    # jm.sensitive_color_plate puts the retardation phase on its local x axis,
    # i.e. that axis acts as the SLOW axis. The mineral is rotated so this axis
    # lies along the c-axis projection (quartz is length-slow); the lambda
    # plate must likewise be rotated to its own slow axis (perpendicular to the
    # given fast axis).
    plate_angle = np.deg2rad(lambda_plate_fast_axis_azimuth_deg + 90.0)
    plate_rotation = jm.rotation(-plate_angle)
    plate_rotation_back = jm.rotation(plate_angle)
    nd_filter = jm.nd_filter(alpha, alpha)
    analyzer = jm.polarizer(direction="y")

    def optical_system(wavelength: float):
        mineral = (
            mineral_rotation
            @ jm.sensitive_color_plate(mineral_retardation_nm, wavelength=wavelength)
            @ mineral_rotation_back
        )
        lambda_plate = (
            plate_rotation
            @ jm.sensitive_color_plate(
                lambda_plate_retardation_nm, wavelength=wavelength
            )
            @ plate_rotation_back
        )
        return analyzer @ lambda_plate @ nd_filter @ mineral @ np.array([1, 0])

    return optical_system


def make_xpl_lambda_rgb_by_orientation(
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    *,
    thickness_mm: float,
    no: float = OPTICAL_NO,
    ne: float = OPTICAL_NE,
    lambda_plate_fast_axis_azimuth_deg: float = LAMBDA_PLATE_FAST_AXIS_AZIMUTH_DEG,
    lambda_plate_retardation_nm: float = LAMBDA_PLATE_RETARDATION_NM,
) -> np.ndarray:
    theta_to_retardation, _ = make_angle_retardation_estimation_function(
        no=no,
        ne=ne,
        thickness=thickness_mm,
    )
    folded_azimuth_deg, folded_inclination_deg = fold_axis_to_upper_hemisphere(
        azimuth_deg,
        inclination_deg,
    )
    retardation_nm = theta_to_retardation(np.deg2rad(folded_inclination_deg))

    colors = np.empty((folded_azimuth_deg.size, 3), dtype=np.float64)
    for index, (azimuth, retardation) in enumerate(
        zip(folded_azimuth_deg.reshape(-1), retardation_nm.reshape(-1))
    ):
        colors[index] = (
            get_spectral_distribution(
                get_lambda_plate_plus_mineral_system(
                    mineral_retardation_nm=float(retardation),
                    mineral_azimuth_rad=np.deg2rad(float(azimuth)),
                    lambda_plate_fast_axis_azimuth_deg=(
                        lambda_plate_fast_axis_azimuth_deg
                    ),
                    lambda_plate_retardation_nm=lambda_plate_retardation_nm,
                    alpha=1.0,
                )
            )["rgb"].astype(np.float64)
            / 255.0
        )
    return np.clip(colors.reshape((*folded_azimuth_deg.shape, 3)), 0.0, 1.0)


def make_predicted_center_colors(
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    *,
    thickness_mm: float,
    no: float = OPTICAL_NO,
    ne: float = OPTICAL_NE,
    angle_step_deg: float = RECONSTRUCTION_ANGLE_STEP_DEG,
) -> np.ndarray:
    """Predicted XPL+lambda color (0..255) for each (azimuth, inclination),
    computed once per quantized orientation cell for speed."""
    azimuth = np.asarray(azimuth_deg, dtype=np.float64)
    inclination = np.asarray(inclination_deg, dtype=np.float64)
    quantized_azimuth = (np.round(azimuth / angle_step_deg) * angle_step_deg) % 180.0
    quantized_inclination = (
        np.round(inclination / angle_step_deg) * angle_step_deg
    ) % 180.0
    orientation_pairs = np.column_stack([quantized_azimuth, quantized_inclination])
    unique_pairs, inverse = np.unique(orientation_pairs, axis=0, return_inverse=True)
    unique_colors = make_xpl_lambda_rgb_by_orientation(
        unique_pairs[:, 0],
        unique_pairs[:, 1],
        thickness_mm=thickness_mm,
        no=no,
        ne=ne,
    ).reshape(-1, 3)
    return np.clip(unique_colors[inverse] * 255.0, 0.0, 255.0)


# ===========================================================================
# Robust color-correction fit (ported verbatim from v3 color_correction.py).
# ===========================================================================
def _robust_scale(values: np.ndarray, *, min_scale: float = 1.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return min_scale
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < min_scale:
        scale = float(np.nanstd(finite))
    return max(scale, min_scale)


def fit_alpha_huber(
    predicted_rgb: np.ndarray,
    original_rgb: np.ndarray,
    *,
    max_iterations: int = 30,
) -> tuple[float, float]:
    """Robust scalar-alpha fit (no bias, no matrix): original ~ alpha *
    predicted, via iteratively-reweighted Huber. Used by the add-image theta
    estimation (M=I, b=0)."""
    predicted = np.asarray(predicted_rgb, dtype=np.float64).reshape(-1)
    original = np.asarray(original_rgb, dtype=np.float64).reshape(-1)
    finite = np.isfinite(predicted) & np.isfinite(original)
    predicted = predicted[finite]
    original = original[finite]
    denom = float(np.dot(predicted, predicted))
    if denom <= 0:
        return ALPHA_MIN, np.inf

    alpha = float(np.clip(np.dot(predicted, original) / denom, ALPHA_MIN, ALPHA_MAX))
    for _ in range(max_iterations):
        residual = original - alpha * predicted
        delta = HUBER_TUNING_CONSTANT * _robust_scale(residual)
        abs_residual = np.abs(residual)
        weights = np.ones_like(residual)
        outside = abs_residual > delta
        weights[outside] = delta / np.maximum(abs_residual[outside], 1e-12)
        weighted_denom = float(np.sum(weights * predicted * predicted))
        if weighted_denom <= 0:
            break
        next_alpha = float(
            np.clip(
                np.sum(weights * predicted * original) / weighted_denom,
                ALPHA_MIN,
                ALPHA_MAX,
            )
        )
        if abs(next_alpha - alpha) < 1e-6:
            alpha = next_alpha
            break
        alpha = next_alpha

    residual = original - alpha * predicted
    delta = HUBER_TUNING_CONSTANT * _robust_scale(residual)
    abs_residual = np.abs(residual)
    loss = np.where(
        abs_residual <= delta,
        0.5 * residual**2,
        delta * (abs_residual - 0.5 * delta),
    )
    return alpha, float(np.mean(loss))


def fit_alpha_with_bias_huber(
    predicted_rgb: np.ndarray,
    original_rgb: np.ndarray,
    bias_rgb: np.ndarray,
    *,
    max_iterations: int = 30,
    min_scale: float = 1.0,
) -> tuple[float, float]:
    predicted = np.asarray(predicted_rgb, dtype=np.float64).reshape(-1)
    original = np.asarray(original_rgb, dtype=np.float64).reshape(-1)
    bias = np.broadcast_to(
        np.asarray(bias_rgb, dtype=np.float64),
        np.asarray(original_rgb, dtype=np.float64).shape,
    ).reshape(-1)
    finite = np.isfinite(predicted) & np.isfinite(original) & np.isfinite(bias)
    predicted = predicted[finite]
    original = original[finite]
    bias = bias[finite]
    adjusted_original = original - bias
    denom = float(np.dot(predicted, predicted))
    if denom <= 0:
        return ALPHA_MIN, np.inf

    alpha = float(
        np.clip(np.dot(predicted, adjusted_original) / denom, ALPHA_MIN, ALPHA_MAX)
    )
    for _ in range(max_iterations):
        residual = original - (alpha * predicted + bias)
        delta = HUBER_TUNING_CONSTANT * _robust_scale(residual, min_scale=min_scale)
        abs_residual = np.abs(residual)
        weights = np.ones_like(residual)
        outside = abs_residual > delta
        weights[outside] = delta / np.maximum(abs_residual[outside], 1e-12)
        weighted_denom = float(np.sum(weights * predicted * predicted))
        if weighted_denom <= 0:
            break
        next_alpha = float(
            np.clip(
                np.sum(weights * predicted * adjusted_original) / weighted_denom,
                ALPHA_MIN,
                ALPHA_MAX,
            )
        )
        if abs(next_alpha - alpha) < 1e-6:
            alpha = next_alpha
            break
        alpha = next_alpha

    residual = original - (alpha * predicted + bias)
    delta = HUBER_TUNING_CONSTANT * _robust_scale(residual, min_scale=min_scale)
    abs_residual = np.abs(residual)
    loss = np.where(
        abs_residual <= delta,
        0.5 * residual**2,
        delta * (abs_residual - 0.5 * delta),
    )
    return alpha, float(np.mean(loss))


def _make_matrix_design_matrix(
    predicted_unit_rgb: np.ndarray,
    *,
    learn_bias: bool,
) -> np.ndarray:
    predicted = np.asarray(predicted_unit_rgb, dtype=np.float64)
    row_count = predicted.shape[0]
    parameter_count = 12 if learn_bias else 9
    design = np.zeros((row_count * 3, parameter_count), dtype=np.float64)
    for channel in range(3):
        rows = np.arange(row_count) * 3 + channel
        matrix_start = channel * 3
        design[rows, matrix_start : matrix_start + 3] = predicted
        if learn_bias:
            design[rows, 9 + channel] = 1.0
    return design


def _solve_trace_constrained_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    learn_bias: bool,
) -> np.ndarray:
    identity_matrix = np.eye(3, dtype=np.float64)
    if learn_bias:
        target_parameters = np.concatenate(
            [identity_matrix.reshape(-1), np.zeros(3, dtype=np.float64)]
        )
        regularization = np.concatenate(
            [
                np.full(9, COLOR_MATRIX_REGULARIZATION, dtype=np.float64),
                np.full(3, COLOR_BIAS_REGULARIZATION, dtype=np.float64),
            ]
        )
    else:
        target_parameters = identity_matrix.reshape(-1)
        regularization = np.full(9, COLOR_MATRIX_REGULARIZATION, dtype=np.float64)
    trace_parameter_indices = [0, 4, 8]
    weighted_design = design * weights[:, None]
    normal_matrix = design.T @ weighted_design + np.diag(regularization)
    rhs = design.T @ (weights * target) + regularization * target_parameters

    parameter_count = target_parameters.size
    constraint = np.zeros((1, parameter_count), dtype=np.float64)
    constraint[0, trace_parameter_indices] = 1.0
    kkt_matrix = np.block(
        [
            [normal_matrix, constraint.T],
            [constraint, np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    kkt_rhs = np.concatenate([rhs, np.asarray([3.0], dtype=np.float64)])
    return np.linalg.solve(kkt_matrix, kkt_rhs)[:parameter_count]


def fit_matrix_trace3_bias_huber(
    predicted_rgb: np.ndarray,
    original_rgb: np.ndarray,
    *,
    learn_bias: bool = True,
    max_iterations: int = 30,
    alpha_matrix_iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    predicted_unit = np.asarray(predicted_rgb, dtype=np.float64) / 255.0
    original_unit = np.asarray(original_rgb, dtype=np.float64) / 255.0
    finite_rows = np.all(np.isfinite(predicted_unit), axis=1) & np.all(
        np.isfinite(original_unit), axis=1
    )
    predicted_unit = predicted_unit[finite_rows]
    original_unit = original_unit[finite_rows]
    if predicted_unit.shape[0] < 4:
        return (
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            ALPHA_MIN,
            np.inf,
        )

    design = _make_matrix_design_matrix(predicted_unit, learn_bias=learn_bias)
    target = original_unit.reshape(-1)
    weights = np.ones(target.shape, dtype=np.float64)

    def alpha_scaled_design(alpha_value: float) -> np.ndarray:
        scaled = design.copy()
        if learn_bias:
            scaled[:, :9] *= alpha_value
        else:
            scaled *= alpha_value
        return scaled

    def unpack_parameters(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = parameters[:9].reshape(3, 3)
        bias = parameters[9:12] if learn_bias else np.zeros(3, dtype=np.float64)
        return matrix, bias

    alpha, _ = fit_alpha_with_bias_huber(
        predicted_unit,
        original_unit,
        np.zeros(3, dtype=np.float64),
        min_scale=1.0 / 255.0,
    )
    parameters = _solve_trace_constrained_ridge(
        alpha_scaled_design(alpha),
        target,
        weights,
        learn_bias=learn_bias,
    )
    for _ in range(alpha_matrix_iterations):
        current_design = alpha_scaled_design(alpha)
        for _ in range(max_iterations):
            residual = target - current_design @ parameters
            delta = HUBER_TUNING_CONSTANT * _robust_scale(
                residual,
                min_scale=1.0 / 255.0,
            )
            abs_residual = np.abs(residual)
            next_weights = np.ones_like(residual)
            outside = abs_residual > delta
            next_weights[outside] = delta / np.maximum(abs_residual[outside], 1e-12)
            next_parameters = _solve_trace_constrained_ridge(
                current_design,
                target,
                next_weights,
                learn_bias=learn_bias,
            )
            if np.max(np.abs(next_parameters - parameters)) < 1e-6:
                parameters = next_parameters
                weights = next_weights
                break
            parameters = next_parameters
            weights = next_weights

        matrix, bias_unit = unpack_parameters(parameters)
        matrix_predicted_unit = predicted_unit @ matrix.T
        next_alpha, _ = fit_alpha_with_bias_huber(
            matrix_predicted_unit,
            original_unit,
            bias_unit,
            min_scale=1.0 / 255.0,
        )
        if abs(next_alpha - alpha) < 1e-6:
            alpha = next_alpha
            break
        alpha = next_alpha
        parameters = _solve_trace_constrained_ridge(
            alpha_scaled_design(alpha),
            target,
            weights,
            learn_bias=learn_bias,
        )

    matrix, bias_unit = unpack_parameters(parameters)
    residual = (
        original_unit - (alpha * (predicted_unit @ matrix.T) + bias_unit)
    ).reshape(-1)
    delta = HUBER_TUNING_CONSTANT * _robust_scale(residual, min_scale=1.0 / 255.0)
    abs_residual = np.abs(residual)
    huber_loss = np.where(
        abs_residual <= delta,
        0.5 * residual**2,
        delta * (abs_residual - 0.5 * delta),
    )
    regularized_loss = (
        float(np.mean(huber_loss))
        + COLOR_MATRIX_REGULARIZATION * float(np.mean((matrix - np.eye(3)) ** 2))
        + COLOR_BIAS_REGULARIZATION * float(np.mean(bias_unit**2))
    )
    return matrix, bias_unit * 255.0, alpha, regularized_loss


def apply_color_correction(
    predicted_rgb: np.ndarray,
    fit: ColorCorrectionFit,
) -> np.ndarray:
    predicted = np.asarray(predicted_rgb, dtype=np.float64)
    if predicted.max(initial=0.0) <= 1.0:
        predicted = predicted * 255.0
    corrected = fit.alpha * (predicted @ fit.matrix.T) + fit.bias_rgb
    return np.clip(corrected, 0.0, 255.0)


def fit_thickness_factor_and_alpha(
    *,
    azimuth_deg: np.ndarray,
    inclination_deg: np.ndarray,
    original_rgb: np.ndarray,
    base_thickness_mm: float,
    no: float = OPTICAL_NO,
    ne: float = OPTICAL_NE,
) -> tuple[ColorCorrectionFit, np.ndarray]:
    """Fit alpha, a trace-3 3x3 matrix M and bias b mapping the predicted
    XPL+lambda colors of each grain center (from its azimuth/inclination and
    the fixed thickness) onto its observed color. Returns the fit and the
    predicted (uncorrected) grain-center colors used."""
    predicted = make_predicted_center_colors(
        azimuth_deg,
        inclination_deg,
        thickness_mm=base_thickness_mm,
        no=no,
        ne=ne,
    )
    matrix, bias_rgb, alpha, robust_loss = fit_matrix_trace3_bias_huber(
        predicted,
        original_rgb,
        learn_bias=True,
    )
    return (
        ColorCorrectionFit(
            correction_model=COLOR_CORRECTION_MODEL,
            thickness_mm=base_thickness_mm,
            thickness_factor=1.0,
            alpha=alpha,
            matrix=matrix,
            bias_rgb=bias_rgb,
            robust_loss=robust_loss,
        ),
        predicted,
    )
