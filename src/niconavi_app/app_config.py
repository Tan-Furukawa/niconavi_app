from __future__ import annotations

from typing import Literal


RColorMapSource = Literal["brightest_angle", "extinction_angle", "extinction_color_map"]

# Application-wide fixed settings. These values are intentionally not exposed
# through the GUI and should be changed only by editing this module.
R_COLOR_MAP_SOURCE: RColorMapSource = "extinction_color_map"

ENABLE_FUNCTION_TAB = False

ENABLE_TILT_REF_MEDIAN_CORRECTION = False

# CPO "90 deg normalize": percentile of the selected grains' inclination that
# is rescaled to 90 deg (a robust stand-in for "the most steeply inclined
# grain"). Grains above it clip to 90 deg. See niconavi.cpo_normalization.
CPO_NORMALIZE_PERCENTILE: float = 95.0