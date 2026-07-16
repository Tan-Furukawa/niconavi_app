"""Stage-tilt registration core (phase-only correlation, POC).

Registers a focus-stacked tilt capture onto the focus-stacked reference
capture of the same stage experiment, replacing the older registration inside
estimate_tilted_image / estimate_tilt_image_result. estimate_tilt_image_result_poc
is a drop-in counterpart of run_all.estimate_tilt_image_result and keeps the
exact TiltImageResult contract (resize to WORKING_WIDTH, register, optional
rotation, optional center crop, resize back), so downstream consumers are
unaffected.

Ported from fix/ebsd_adjustment_v3/tilt_registration.py, which was already
written against these niconavi modules.

Registration per pair:
1. Focus-stack the reference frames and the tilt frames (focus_stack).
2. Undo the ~STAGE_TILT_DEG stage-tilt foreshortening of the tilt capture
   (transform_stacked_image, along the sample-ascent direction phi from
   focus_stack's RANSAC plane fit).
3. Coarse: phase-only correlation (POC), computed by hand (FFT ->
   normalized cross-power spectrum -> IFFT) so the peak search can be
   restricted to |dx| <= W/3, |dy| <= H/3.
4. Fine: small center-anchored aspect-ratio adjustment (scale_x, scale_y near
   1) plus a small extra xy shift, optimized with Powell on the mean absolute
   grayscale difference inside the eroded common valid mask. All steps compose
   into one affine so the tilt image is interpolated once.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from niconavi_app.niconavi.image.image import (
    resize_array,
    resize_img,
    rotate_array,
)
from niconavi_app.niconavi.image.type import RGBPicture
from niconavi_app.niconavi.tilt_image import (
    crop_center,
    focus_stack,
    make_tilt_color_change_image,
    normalize_by_gray_scale,
    transform_stacked_image,
)
from niconavi_app.niconavi.type import (
    ComputationResult,
    TiltImageInfo,
    TiltImageResult,
)

# ============================================================================
# Physical stage tilt of the tilted capture; its cos(theta) foreshortening
# along the sample-ascent direction is undone before registration. The exact
# value is not critical: the fine aspect-ratio step absorbs the residual.
# (tilt_image_info.theta_thin_section stays 10 deg for the downstream color-
# chart tilt judgment; this constant only drives the registration.)
STAGE_TILT_DEG = 7.0

# Coarse POC peak search window: the captures are assumed to overlap within
# about a third of the frame in each direction.
COARSE_MAX_SHIFT_FRACTION = 1.0 / 3.0

# Fine-registration bounds around the coarse solution.
FINE_SCALE_BOUND = 0.03  # scale_x/scale_y in [1 -/+ this]
FINE_SHIFT_BOUND_PX = 12.0  # extra xy shift in +/- px
# Erosion of the valid-pixel mask before computing the fine metric, so border
# interpolation artifacts don't dominate the mean.
METRIC_MASK_EROSION_PX = 21

# Working width the estimate_* wrappers resize to before registering, matching
# niconavi's estimate_tilted_image.
WORKING_WIDTH = 1000
# ============================================================================


@dataclass
class TiltPairRegistration:
    ref_stacked: np.ndarray
    tilt_stacked: np.ndarray
    tilt_corrected: np.ndarray  # tilt_stacked after stage-tilt correction
    focused_index: np.ndarray
    ascent_phi: float  # sample-ascent direction from focus_stack's RANSAC fit
    coarse_shift: tuple[int, int]  # translation applied to the tilt image
    poc_peak: float
    fine_params: tuple[float, float, float, float]  # scale_x, scale_y, tx, ty
    coarse_aligned_image: np.ndarray
    coarse_aligned_mask: np.ndarray  # bool
    aligned_image: np.ndarray  # after the fine step (final)
    aligned_mask: np.ndarray  # bool
    zero_metric: float
    coarse_metric: float
    fine_metric: float


def _gray32(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def phase_only_correlation_shift(
    ref_gray: np.ndarray,
    moving_gray: np.ndarray,
    *,
    max_dx: int,
    max_dy: int,
) -> tuple[int, int, float]:
    """Bounded phase-only correlation: returns the translation (tx, ty) to
    apply to moving_gray so it aligns with ref_gray, restricted to
    |tx| <= max_dx, |ty| <= max_dy, plus the POC peak value (higher = sharper,
    more confident match)."""
    height, width = ref_gray.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    spectrum_ref = np.fft.fft2(ref_gray * window)
    spectrum_mov = np.fft.fft2(moving_gray * window)
    cross_power = spectrum_ref * np.conj(spectrum_mov)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    surface = np.fft.fftshift(np.fft.ifft2(cross_power).real)

    center_y, center_x = height // 2, width // 2
    bounded = np.full_like(surface, -np.inf)
    y0, y1 = center_y - max_dy, center_y + max_dy + 1
    x0, x1 = center_x - max_dx, center_x + max_dx + 1
    bounded[y0:y1, x0:x1] = surface[y0:y1, x0:x1]
    peak_y, peak_x = np.unravel_index(int(np.argmax(bounded)), bounded.shape)
    return peak_x - center_x, peak_y - center_y, float(surface[peak_y, peak_x])


def compose_registration_affine(
    scale_x: float,
    scale_y: float,
    fine_tx: float,
    fine_ty: float,
    coarse_tx: float,
    coarse_ty: float,
    center_xy: tuple[float, float],
) -> np.ndarray:
    """Single 2x3 affine: coarse translation, then center-anchored (scale_x,
    scale_y), then the fine translation - composed so the moving image is
    warped (and interpolated) exactly once."""
    center_x, center_y = center_xy
    return np.float32(
        [
            [scale_x, 0.0, scale_x * (coarse_tx - center_x) + center_x + fine_tx],
            [0.0, scale_y, scale_y * (coarse_ty - center_y) + center_y + fine_ty],
        ]
    )


def _warp(image: np.ndarray, matrix: np.ndarray, *, nearest: bool = False) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
    )


def register_tilt_pair(
    ref_frames: list[np.ndarray],
    tilt_frames: list[np.ndarray],
    *,
    stage_name: str = "",
    stage_tilt_deg: float = STAGE_TILT_DEG,
) -> TiltPairRegistration:
    ref_stacked, _, _, _ = focus_stack(ref_frames)
    tilt_stacked, focused_index, ascent_phi, _ = focus_stack(tilt_frames)

    # Undo the stage-tilt foreshortening (same warp for the mask, so the
    # untouched-canvas pixels can be excluded from every later metric).
    theta = float(np.deg2rad(stage_tilt_deg))
    tilt_corrected = transform_stacked_image(tilt_stacked, ascent_phi, theta)
    white = np.full_like(tilt_stacked, 255)
    tilt_mask = transform_stacked_image(white, ascent_phi, theta)[:, :, 0]
    tilt_mask[:2, :] = 0
    tilt_mask[-2:, :] = 0
    tilt_mask[:, :2] = 0
    tilt_mask[:, -2:] = 0

    ref_gray = _gray32(ref_stacked)
    moving_gray = _gray32(tilt_corrected)
    height, width = ref_gray.shape
    center_xy = ((width - 1) / 2.0, (height - 1) / 2.0)

    coarse_tx, coarse_ty, poc_peak = phase_only_correlation_shift(
        ref_gray,
        moving_gray,
        max_dx=int(width * COARSE_MAX_SHIFT_FRACTION),
        max_dy=int(height * COARSE_MAX_SHIFT_FRACTION),
    )

    erosion_kernel = np.ones(
        (METRIC_MASK_EROSION_PX, METRIC_MASK_EROSION_PX), np.uint8
    )

    def metric(params: np.ndarray) -> float:
        scale_x, scale_y, fine_tx, fine_ty = (float(v) for v in params)
        matrix = compose_registration_affine(
            scale_x, scale_y, fine_tx, fine_ty, coarse_tx, coarse_ty, center_xy
        )
        warped_gray = _warp(moving_gray, matrix)
        warped_mask = _warp(tilt_mask, matrix, nearest=True)
        valid = cv2.erode(warped_mask, erosion_kernel) > 250
        if int(valid.sum()) < 10000:
            return 1e9
        return float(np.mean(np.abs(ref_gray - warped_gray)[valid]))

    zero_metric = metric(np.array([1.0, 1.0, -coarse_tx, -coarse_ty]))  # no shift at all
    coarse_metric = metric(np.array([1.0, 1.0, 0.0, 0.0]))

    result = minimize(
        metric,
        x0=np.array([1.0, 1.0, 0.0, 0.0]),
        method="Powell",
        bounds=[
            (1.0 - FINE_SCALE_BOUND, 1.0 + FINE_SCALE_BOUND),
            (1.0 - FINE_SCALE_BOUND, 1.0 + FINE_SCALE_BOUND),
            (-FINE_SHIFT_BOUND_PX, FINE_SHIFT_BOUND_PX),
            (-FINE_SHIFT_BOUND_PX, FINE_SHIFT_BOUND_PX),
        ],
    )
    fine_params = tuple(float(v) for v in result.x)
    fine_metric = float(result.fun)

    def aligned(
        params: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = compose_registration_affine(*params, coarse_tx, coarse_ty, center_xy)
        return (
            _warp(tilt_corrected, matrix),
            _warp(tilt_mask, matrix, nearest=True) > 250,
        )

    coarse_aligned_image, coarse_aligned_mask = aligned((1.0, 1.0, 0.0, 0.0))
    aligned_image, aligned_mask = aligned(fine_params)

    label = f"{stage_name}: " if stage_name else ""
    print(
        f"{label}POC coarse shift=({coarse_tx}, {coarse_ty}) px "
        f"(peak={poc_peak:.3f}), mean|diff| {zero_metric:.2f} (none) -> "
        f"{coarse_metric:.2f} (coarse) -> {fine_metric:.2f} (fine), "
        f"fine scale=({fine_params[0]:.4f}, {fine_params[1]:.4f}), "
        f"fine shift=({fine_params[2]:.2f}, {fine_params[3]:.2f}) px"
    )

    return TiltPairRegistration(
        ref_stacked=ref_stacked,
        tilt_stacked=tilt_stacked,
        tilt_corrected=tilt_corrected,
        focused_index=focused_index,
        ascent_phi=float(ascent_phi),
        coarse_shift=(coarse_tx, coarse_ty),
        poc_peak=poc_peak,
        fine_params=fine_params,
        coarse_aligned_image=coarse_aligned_image,
        coarse_aligned_mask=coarse_aligned_mask,
        aligned_image=aligned_image,
        aligned_mask=aligned_mask,
        zero_metric=zero_metric,
        coarse_metric=coarse_metric,
        fine_metric=fine_metric,
    )


def estimate_tilted_image_poc(
    im: list[np.ndarray],
    im_tilt: list[np.ndarray],
    *,
    stage_name: str = "",
    center: tuple[int, int] | None = None,
    shape: tuple[int, int] | None = None,
    rotation: float | None = None,
) -> TiltImageResult:
    """Drop-in counterpart of niconavi's estimate_tilted_image with the
    registration replaced by register_tilt_pair (bounded POC + fine
    aspect/shift). Mirrors the original's resize-to-WORKING_WIDTH / brightness
    normalization / optional rotation / optional center crop / resize-back
    sequence and its TiltImageResult contract (including the phase-1
    color_change field)."""
    width_original = im_tilt[0].shape[1]
    height_original = im_tilt[0].shape[0]

    ref_resized = resize_array(im[0], WORKING_WIDTH)
    tilt_resized = [resize_img(frame, WORKING_WIDTH) for frame in im_tilt]

    registration = register_tilt_pair(
        [ref_resized], tilt_resized, stage_name=stage_name
    )
    ref_image = registration.ref_stacked
    aligned_image = registration.aligned_image
    mask = (registration.aligned_mask.astype(np.uint8)) * 255
    focused_index = registration.focused_index.astype(np.int32)

    aligned_image = normalize_by_gray_scale(
        ref_image, aligned_image, registration.aligned_mask
    )

    if rotation is not None and center is not None:
        aligned_image = rotate_array(aligned_image, rotation, center)
        ref_image = rotate_array(ref_image, rotation, center)
        mask = rotate_array(mask, rotation, center)
        focused_index = rotate_array(focused_index, rotation, center)

    if center is not None and shape is not None:
        aligned_image = crop_center(aligned_image, center, shape)
        ref_image = crop_center(ref_image, center, shape)
        mask = crop_center(mask, center, shape)
        focused_index = crop_center(focused_index, center, shape)
        height_original, width_original = aligned_image.shape[:2]

    resized_index = resize_array(focused_index, width_original, height_original)
    resized_mask = resize_array(mask, width_original, height_original)

    original_image = resize_img(
        RGBPicture(ref_image), width_original, height_original
    )
    focused_tilted_image = resize_img(
        RGBPicture(aligned_image), width_original, height_original
    )

    return TiltImageResult(
        original_image=original_image,
        focused_tilted_image=focused_tilted_image,
        focused_index=resized_index,
        image_mask=(resized_mask > 127),
        azimuth_thin_section=registration.ascent_phi,
        color_change=make_tilt_color_change_image(
            before=original_image, after=focused_tilted_image
        ),
    )


def estimate_tilt_image_result_poc(params: ComputationResult) -> ComputationResult:
    """Drop-in counterpart of run_all.estimate_tilt_image_result: builds
    tilt_image0 (and tilt_image45, rotated back by
    -angle_between_x_and_thin_section_axis_at_tilt like the original) from the
    raw frame lists using the POC registration above."""
    info = params.tilt_image_info
    if (
        params.raw_maps is None
        or info.image0_raw is None
        or info.tilt_image0_raw is None
        or params.center_int_x is None
        or params.center_int_y is None
    ):
        return params

    ex_angle_map = params.raw_maps["extinction_angle"]
    shape = (ex_angle_map.shape[0], ex_angle_map.shape[1])
    center = (params.center_int_x, params.center_int_y)

    tilt_image0 = estimate_tilted_image_poc(
        info.image0_raw,
        info.tilt_image0_raw,
        stage_name="stage 0 deg",
        center=center,
        shape=shape,
    )

    new_info: dict[str, object] = {**info.__dict__, "tilt_image0": tilt_image0}
    if info.image45_raw is not None and info.tilt_image45_raw is not None:
        new_info["tilt_image45"] = estimate_tilted_image_poc(
            info.image45_raw,
            info.tilt_image45_raw,
            stage_name="stage +45 deg",
            center=center,
            shape=shape,
            rotation=-params.angle_between_x_and_thin_section_axis_at_tilt,
        )

    return ComputationResult(
        **{**params.__dict__, "tilt_image_info": TiltImageInfo(**new_info)}
    )
