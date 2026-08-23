"""Decide, per pixel, which of the phi_ex +/- 45 deg XPL+lambda frames is the
addition one, with the same fit the addition-image inclination estimate uses.

make_R_maps used to answer this by reading both frames off the
pol_lambda retardation color chart (Lab nearest neighbour) and comparing the
two retardations. That chart is indexed by retardation and carries its own
brightness scalar - an ND filter inside the optical system, chosen on a coarse
grid by select_h_in_color_chart - so the branch decision and the inclination
fit ran on two different color models and two different alpha conventions.

Here both frames are matched against a Theta LUT built from the same
color_correction.make_xpl_lambda_rgb_by_orientation prediction the inclination
fit uses, stacked over its two branches: the mineral slow axis parallel to the
lambda slow axis (addition, compound retardation 530 + R(Theta, t)) or
perpendicular to it (subtraction, 530 - R(Theta, t)). One nearest-color
assignment returns branch and Theta together, the retardation follows as
530 +/- R(Theta, t), and the usual R+ >= R- comparison then only ever chooses
between colors a mineral of this thickness can actually show.

alpha is fitted the same way too: a scalar on the predicted color (after the
sRGB transfer function, not an ND filter inside the optics), by an alpha grid
scan followed by Huber IRLS alternating with reassignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from niconavi_app.niconavi.color_correction import (
    LAMBDA_PLATE_FAST_AXIS_AZIMUTH_DEG,
    LAMBDA_PLATE_RETARDATION_NM,
    OPTICAL_NE,
    OPTICAL_NO,
    ALPHA_MAX,
    ALPHA_MIN,
    fit_alpha_huber,
    make_xpl_lambda_rgb_by_orientation,
)
from niconavi_app.niconavi.optics.tools import (
    make_angle_retardation_estimation_function,
)

# Theta grid the LUT is sampled on: 0, step, 2*step, ... (< 90 deg).
THETA_STEP_DEG = 0.5
# Number of alpha samples in the initial (alpha, Theta) product-grid scan.
ALPHA_INIT_GRID_STEPS = 96
# Max robust-alpha <-> Theta alternations.
MAX_INIT_ALTERNATIONS = 30
# Pixels drawn (without replacement, inside the mask, both frames pooled) for
# the alpha fit. The assignment afterwards runs on every pixel; only alpha is
# fitted on a sample, as it is one number for the whole field of view.
ALPHA_FIT_SAMPLE_PIXELS = 10000
ALPHA_FIT_SEED = 1234

# Display azimuth of the lambda-plate SLOW axis: the addition position puts
# the (length-slow) c-axis parallel to it.
LAMBDA_SLOW_AXIS_AZIMUTH_DEG = (LAMBDA_PLATE_FAST_AXIS_AZIMUTH_DEG + 90.0) % 180.0


@dataclass(frozen=True)
class BranchLut:
    """Both branches of the XPL+lambda color LUT, stacked into one candidate
    list so one argmin returns branch and Theta together."""

    colors: np.ndarray  # (2K, 3), 0..255
    theta_deg: np.ndarray  # (2K,)
    compound_retardation_nm: np.ndarray  # (2K,), 530 +/- R(Theta, t)
    is_addition: np.ndarray  # (2K,) bool


@dataclass(frozen=True)
class BranchMaps:
    """Per-pixel result of the branch fit."""

    p45_retardation_nm: np.ndarray
    m45_retardation_nm: np.ndarray
    p45_theta_deg: np.ndarray
    m45_theta_deg: np.ndarray
    p45_selected: np.ndarray  # R+ >= R-, i.e. +45 deg is the addition frame
    alpha: float
    branch_agreement: float  # share of pixels whose two frames land on
    # opposite branches, as they physically must


def build_branch_luts(
    *,
    thickness_mm: float,
    no: float = OPTICAL_NO,
    ne: float = OPTICAL_NE,
    lambda_plate_retardation_nm: float = LAMBDA_PLATE_RETARDATION_NM,
) -> BranchLut:
    theta_grid = np.arange(0.0, 90.0, THETA_STEP_DEG, dtype=np.float64)
    branches = []
    for azimuth_deg in (
        LAMBDA_SLOW_AXIS_AZIMUTH_DEG,
        (LAMBDA_SLOW_AXIS_AZIMUTH_DEG + 90.0) % 180.0,
    ):
        colors = make_xpl_lambda_rgb_by_orientation(
            np.full(theta_grid.shape, azimuth_deg),
            theta_grid,
            thickness_mm=thickness_mm,
            no=no,
            ne=ne,
            lambda_plate_retardation_nm=lambda_plate_retardation_nm,
        ).reshape(-1, 3)
        branches.append(np.clip(colors * 255.0, 0.0, 255.0))
    theta_to_retardation, _ = make_angle_retardation_estimation_function(
        no=no, ne=ne, thickness=thickness_mm
    )
    mineral_nm = np.asarray(
        theta_to_retardation(np.deg2rad(theta_grid)), dtype=np.float64
    )
    return BranchLut(
        colors=np.concatenate(branches, axis=0),
        theta_deg=np.concatenate((theta_grid, theta_grid)),
        compound_retardation_nm=np.concatenate(
            (
                lambda_plate_retardation_nm + mineral_nm,
                lambda_plate_retardation_nm - mineral_nm,
            )
        ),
        is_addition=np.concatenate(
            (np.ones(theta_grid.size, bool), np.zeros(theta_grid.size, bool))
        ),
    )


def assign_lut_index(
    lut: np.ndarray,
    rgb: np.ndarray,
    alpha: float,
    *,
    chunk: int = 20000,
) -> np.ndarray:
    """Nearest alpha * lut color for every row of rgb, in RGB space.

    |alpha c_k - y|^2 expands to alpha^2 |c_k|^2 - 2 alpha c_k.y + |y|^2, so
    the assignment costs one matrix product per chunk. Chunked because the
    caller passes every pixel of an image, not just a sample.
    """
    lut = np.asarray(lut, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    lut_sq = np.einsum("kc,kc->k", lut, lut)
    index = np.empty(rgb.shape[0], dtype=np.int64)
    for start in range(0, rgb.shape[0], chunk):
        block = rgb[start : start + chunk]
        distance_sq = (
            alpha**2 * lut_sq[None, :]
            - 2.0 * alpha * (block @ lut.T)
            + np.einsum("gc,gc->g", block, block)[:, None]
        )
        index[start : start + chunk] = np.argmin(distance_sq, axis=1)
    return index


def fit_alpha_and_index(
    lut: np.ndarray,
    rgb: np.ndarray,
    *,
    stage_label: str = "branch alpha",
) -> tuple[np.ndarray, float]:
    """Fit y_g ~ alpha * c_k(g) (M=I, b=0): alpha-grid scan with per-sample
    best k, then alternate robust alpha refits with exhaustive k reassignment.

    For fixed g, k the squared distance is quadratic in alpha:
    |alpha c_k - y_g|^2 = alpha^2 |c_k|^2 - 2 alpha c_k.y_g + |y_g|^2, so the
    whole alpha grid is evaluated from three precomputed arrays.
    """
    lut = np.asarray(lut, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    lut_sq = np.einsum("kc,kc->k", lut, lut)
    lut_dot = np.einsum("kc,gc->gk", lut, rgb)
    original_sq = np.einsum("gc,gc->g", rgb, rgb)
    alpha_grid = np.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_INIT_GRID_STEPS + 1)[
        1:
    ]  # skip alpha=0 (degenerate: every k fits equally badly)
    best_alpha = float(alpha_grid[0])
    best_loss = np.inf
    for alpha in alpha_grid:
        distance_sq = (
            alpha**2 * lut_sq[None, :] - 2.0 * alpha * lut_dot + original_sq[:, None]
        )
        min_distance_sq = np.maximum(np.min(distance_sq, axis=1), 0.0)
        # Mean of L2 norms (not squared norms) so single outlier samples do
        # not dominate the alpha choice.
        loss = float(np.mean(np.sqrt(min_distance_sq)))
        if loss < best_loss:
            best_alpha = float(alpha)
            best_loss = loss

    alpha = best_alpha
    index = assign_lut_index(lut, rgb, alpha)
    for _ in range(MAX_INIT_ALTERNATIONS):
        next_alpha, _ = fit_alpha_huber(lut[index], rgb)
        next_index = assign_lut_index(lut, rgb, next_alpha)
        converged = abs(next_alpha - alpha) < 1e-6 and np.array_equal(
            next_index, index
        )
        alpha = next_alpha
        index = next_index
        if converged:
            break
    print(
        f"{stage_label}: alpha grid best={best_alpha:g} "
        f"(mean |r|={best_loss:g}), refined alpha={alpha:g}"
    )
    return index, alpha


def solve_branch_maps(
    p45_color: np.ndarray,
    m45_color: np.ndarray,
    *,
    thickness_mm: float,
    no: float = OPTICAL_NO,
    ne: float = OPTICAL_NE,
    lambda_plate_retardation_nm: float = LAMBDA_PLATE_RETARDATION_NM,
    mask: Optional[np.ndarray] = None,
    context_label: str = "addition branch",
) -> BranchMaps:
    p45_color = np.asarray(p45_color)
    m45_color = np.asarray(m45_color)
    if p45_color.shape != m45_color.shape:
        raise ValueError(
            "The +45 and -45 deg color maps must have the same shape: "
            f"{p45_color.shape} vs {m45_color.shape}"
        )
    shape = p45_color.shape[:2]
    mask = (
        np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    )

    lut = build_branch_luts(
        thickness_mm=thickness_mm,
        no=no,
        ne=ne,
        lambda_plate_retardation_nm=lambda_plate_retardation_nm,
    )
    pooled = np.concatenate(
        (p45_color[mask].reshape(-1, 3), m45_color[mask].reshape(-1, 3))
    ).astype(np.float64)
    if pooled.shape[0] == 0:
        raise ValueError(f"{context_label}: the mask selected no pixels.")
    rng = np.random.RandomState(ALPHA_FIT_SEED)
    sample = pooled[
        rng.choice(
            pooled.shape[0],
            min(ALPHA_FIT_SAMPLE_PIXELS, pooled.shape[0]),
            replace=False,
        )
    ]
    _, alpha = fit_alpha_and_index(
        lut.colors, sample, stage_label=f"{context_label}: branch alpha"
    )

    p45_index = assign_lut_index(lut.colors, p45_color.reshape(-1, 3), alpha)
    m45_index = assign_lut_index(lut.colors, m45_color.reshape(-1, 3), alpha)
    p45_retardation = lut.compound_retardation_nm[p45_index].reshape(shape)
    m45_retardation = lut.compound_retardation_nm[m45_index].reshape(shape)
    p45_selected = p45_retardation >= m45_retardation
    agreement = float(
        np.mean(
            lut.is_addition[p45_index].reshape(shape)[mask]
            != lut.is_addition[m45_index].reshape(shape)[mask]
        )
    )
    print(
        f"{context_label}: alpha={alpha:g}, reachable R "
        f"{lut.compound_retardation_nm.min():.0f}-"
        f"{lut.compound_retardation_nm.max():.0f} nm, +45 addition on "
        f"{100 * float(np.mean(p45_selected[mask])):.2f}% of masked px, "
        f"frames on opposite branches for {100 * agreement:.2f}%"
    )
    return BranchMaps(
        p45_retardation_nm=p45_retardation,
        m45_retardation_nm=m45_retardation,
        p45_theta_deg=lut.theta_deg[p45_index].reshape(shape),
        m45_theta_deg=lut.theta_deg[m45_index].reshape(shape),
        p45_selected=p45_selected,
        alpha=alpha,
        branch_agreement=agreement,
    )
