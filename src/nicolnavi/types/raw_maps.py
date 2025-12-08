from typing import TypedDict, Literal, Optional, TypeAlias
from niconavi.tools.type import (
    D2FloatArray,
)
from niconavi.image.type import RGBPicture

# ---------------------------------------------
# Raw maps
# ---------------------------------------------
RawMapsAcceptedTypes: TypeAlias = Literal[
    "degree_0",
    "degree_22_5",
    "degree_45",
    "degree_67_5",
    "extinction_color_map",
    "R_color_map",
    "extinction_angle",
    "max_retardation_map",
    "p45_R_color_map",
    "m45_R_color_map",
    "p45_R_map",
    "m45_R_map",
    "azimuth",
]


class RawMaps(TypedDict):
    degree_0: RGBPicture
    degree_22_5: RGBPicture
    degree_45: RGBPicture
    degree_67_5: RGBPicture
    extinction_color_map: RGBPicture
    R_color_map: RGBPicture
    extinction_angle: D2FloatArray
    max_retardation_map: D2FloatArray
    p45_R_color_map: Optional[RGBPicture]
    m45_R_color_map: Optional[RGBPicture]
    p45_R_map: Optional[D2FloatArray]
    m45_R_map: Optional[D2FloatArray]
    azimuth: Optional[D2FloatArray]
