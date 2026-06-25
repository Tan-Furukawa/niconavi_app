from __future__ import annotations

from typing import Literal


RColorMapSource = Literal["brightest_angle", "extinction_angle", "extinction_color_map"]

# Application-wide fixed settings. These values are intentionally not exposed
# through the GUI and should be changed only by editing this module.
R_COLOR_MAP_SOURCE: RColorMapSource = "extinction_color_map"
ENABLE_FUNCTION_TAB = False
