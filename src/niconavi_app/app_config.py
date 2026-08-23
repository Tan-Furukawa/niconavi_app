from __future__ import annotations

from typing import Literal


RColorMapSource = Literal["brightest_angle", "extinction_angle", "extinction_color_map"]
AdditionBranchSource = Literal["retardation_chart", "theta_lut"]

# Application-wide fixed settings. These values are intentionally not exposed
# through the GUI and should be changed only by editing this module.
R_COLOR_MAP_SOURCE: RColorMapSource = "extinction_color_map"

# How the addition/subtraction branch is decided - which of the phi_ex +/- 45
# deg lambda-plate frames holds the c-axis on the lambda slow axis. That branch
# fixes both the azimuth of the 90 deg apart pair and which frame the addition
# image takes, so it is the one decision the azimuth and the inclination share.
# "theta_lut" matches both frames against the two-branch Theta LUT the
# addition-image inclination fit uses (niconavi.addition_branch), so the two
# share a color model, an alpha convention and a physically reachable candidate
# set. "retardation_chart" keeps the older path: read both frames off the
# pol_lambda retardation color chart, whose brightness is an ND filter chosen
# on its own coarse grid, and compare the retardations. The Theta LUT needs a
# thickness; without one that path falls back to the chart regardless.
ADDITION_BRANCH_SOURCE: AdditionBranchSource = "theta_lut"

ENABLE_FUNCTION_TAB = False

ENABLE_TILT_REF_MEDIAN_CORRECTION = False

# CPO "90 deg normalize": percentile of the selected grains' inclination that
# is rescaled to 90 deg (a robust stand-in for "the most steeply inclined
# grain"). Grains above it clip to 90 deg. See niconavi.cpo_normalization.
CPO_NORMALIZE_PERCENTILE: float = 95.0