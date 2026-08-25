"""Gray-world white balance.

One RGB gain set is estimated from a single reference image and applied
unchanged to every other image, so relative colour and brightness comparisons
between images survive the correction - only the shared colour cast of the
illumination is removed. This is the correction the inclination pipeline in
fix/ebsd_adjustment_v3 applies before it resolves the theta / 180 - theta
branch (fix/ebsd_adjustment_v3/lib/white_balance.py, which now re-exports the
two functions below so both paths use one implementation).
"""

from __future__ import annotations

import numpy as np

from niconavi_app.niconavi.image.image import create_outside_circle_mask
from niconavi_app.niconavi.image.type import RGBPicture

# 0 disables the correction and 1 applies the full gray-world RGB gains. The
# interference colours themselves contribute to the mean colour of the field,
# so correcting all the way would eat into the colour information the maps are
# built from; half-way removes the lamp's cast without doing that.
GRAY_WORLD_CORRECTION_STRENGTH = 0.5


def gray_world_mean_gains(reference_image: np.ndarray) -> np.ndarray:
    """Full gray-world RGB gains: what makes the channel means of the reference
    image equal. Only pixels inside the largest circle that fits in the frame
    count, because the corners of a rotation frame are outside the field stop.
    """
    image = np.asarray(reference_image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Gray-world reference image must be an RGB image.")

    valid_mask = ~create_outside_circle_mask(image)
    valid_pixels = image[valid_mask]
    if valid_pixels.size == 0:
        raise ValueError("Gray-world reference image has no valid pixels.")

    channel_means = np.mean(valid_pixels, axis=0)
    if np.any(channel_means <= 0):
        raise ValueError("Gray-world reference contains an empty RGB channel.")
    target_mean = float(np.mean(channel_means))
    return target_mean / channel_means


def blend_gains(full_gains: np.ndarray, strength: float) -> np.ndarray:
    """Interpolate between no correction (strength 0) and the full gray-world
    gains (strength 1)."""
    if not 0 <= strength <= 1:
        raise ValueError("Gray-world correction strength must be between 0 and 1.")
    return 1 + strength * (np.asarray(full_gains, dtype=np.float64) - 1)


def apply_rgb_gains(image: np.ndarray, gains: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    corrected = rgb.astype(np.float64) * np.asarray(gains)[None, None, :]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def apply_rgb_gains_to_frames(
    frames: list[RGBPicture] | None, gains: np.ndarray
) -> list[RGBPicture] | None:
    if frames is None:
        return None
    return [RGBPicture(apply_rgb_gains(frame, gains)) for frame in frames]
