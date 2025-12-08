from typing import TypedDict, Literal, Optional, TypeAlias
from matplotlib.pyplot import Figure, Axes
from niconavi.tools.type import D2FloatArray
from niconavi.image.type import RGBPicture

# ---------------------------------------------
# GrainSegmentedMaps
# ---------------------------------------------

GrainSegmentedMapsAcceptedLiteral = Literal[
    "extinction_color_map",
    "R_color_map",
    "extinction_angle",
    "sd_extinction_angle_map",
    "max_retardation_map",
    "H",
    "S",
    "V",
    "eccentricity",
    "angle_deg",
    "major_axis_length",
    "minor_axis_length",
    "R_70_map",
    "R_80_map",
    "R_90_map",
    "size",
    "p45_R_color_map",
    "m45_R_color_map",
    "p45_R_map",
    "m45_R_map",
    "azimuth",
    "sd_azimuth",
]


class GrainSegmentedMaps(TypedDict):
    extinction_color_map: Optional[RGBPicture]
    R_color_map: Optional[RGBPicture]
    extinction_angle: Optional[D2FloatArray]
    sd_extinction_angle_map: Optional[D2FloatArray]
    max_retardation_map: Optional[D2FloatArray]
    H: Optional[D2FloatArray]
    S: Optional[D2FloatArray]
    V: Optional[D2FloatArray]
    eccentricity: Optional[D2FloatArray]
    R_70_map: Optional[D2FloatArray]
    R_80_map: Optional[D2FloatArray]
    R_90_map: Optional[D2FloatArray]
    size: Optional[D2FloatArray]
    p45_R_color_map: Optional[RGBPicture]
    m45_R_color_map: Optional[RGBPicture]
    p45_R_map: Optional[D2FloatArray]
    m45_R_map: Optional[D2FloatArray]
    azimuth: Optional[D2FloatArray]
    sd_azimuth: Optional[D2FloatArray]
