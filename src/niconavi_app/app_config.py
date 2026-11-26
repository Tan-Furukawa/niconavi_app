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

# Show or hide the "Function" tab (the last tab, after Analysis).
#
# What it is: a page of optional helper tools that are separate from the main
# Video -> Center -> Map -> Filter -> Analysis workflow, e.g. converting
# between specimen thickness and maximum retardation and previewing the
# resulting interference color / spectral distribution.
#
# True  -> the Function tab is added to the tab bar.
# False -> the tab is not created at all; the main workflow is unaffected.
# The switch is read in components/page_tab/page_tab.py; the tab itself is
# components/page_tab/tabs/function_tab.py.
ENABLE_FUNCTION_TAB = False

# Brightness matching of the tilted image to its reference (untilted) image.
#
# What it is: before the tilted and untilted frames are compared for the
# inclination estimate, each frame's brightness (HSV V channel) can be shifted
# or scaled so that their medians inside the field of view agree. This
# compensates for an overall brightness change between the two exposures
# (lamp, exposure time, stage tilt) so that only the color change from tilting
# remains.
#
# True  -> the median brightness correction is applied.
# False -> the images are used as recorded, without brightness adjustment.
# The switch is read in niconavi/inclination.py
# (normalize_tilt_pair_brightness) and niconavi/tilt_image.py
# (normalize_by_gray_scale).
ENABLE_TILT_REF_MEDIAN_CORRECTION = False

# Show or hide the "color check" section of the images panel.
#
# What it is: after "calculate" on the analysis tab with the CPO plot
# selected, the images panel can offer two buttons under a "color check"
# heading:
#   - "RGB before correction": scatter plot of the predicted vs. observed
#     grain-centre RGB values, before the color correction is applied.
#   - "RGB after correction":  the same comparison after the color correction.
# Points close to the 1:1 line mean the color model reproduces the observed
# colors well; it is a diagnostic for checking the correction, not a result.
#
# True  -> the two scatter plots are built on calculate and the buttons appear.
# False -> the plots are not built at all (saves that computation), so the
#          buttons and the "color check" heading stay hidden. Nothing else in
#          the CPO result changes.
# The switch is read in components/page_tab/tabs/analysis_tab.py.
ENABLE_RGB_COLOR_CHECK = False

# CPO "90 deg normalize": the STARTING percentile of the selected grains'
# inclination that is rescaled to 90 deg (a robust stand-in for "the most
# steeply inclined grain"); grains above it clip to 90 deg. Unlike the rest of
# this module this one is exposed - the analysis tab's percentile field is
# seeded from it and the user edits it per run. See niconavi.cpo_normalization.
CPO_NORMALIZE_PERCENTILE: float = 95.0