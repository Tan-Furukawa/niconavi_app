import numpy as np
from typing import Literal, cast, overload
import numpy as np
from niconavi.tools.type import D2FloatArray, D2IntArray
from niconavi.type import Grain, GrainSegmentedMaps, GrainAcceptedLiteral

from niconavi.image.type import RGBPicture
from niconavi.tools.array import reconstruct_array_by_compressed_array
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from matplotlib.pyplot import Figure, Axes


def get_rgb_map_from_grain_list_key(
    grain_list: list[Grain],
    key: Literal[
        "R_color",
        "extinction_color",
        "p45_color",
        "m45_color",
    ],
) -> RGBPicture | None:
    plot_shape = grain_list[0]["original_shape"]

    if np.any(list(map(lambda x: x[key] is None, grain_list))):
        return None
    else:
        plotted_item = grain_list[0][key]
        if plotted_item is not None:
            if len(plotted_item) == 3:
                plotted_key_val = np.zeros(
                    plot_shape + (len(plotted_item),), dtype=np.uint8
                )
            else:
                ValueError("invalid dimension of key item.")

            grain_mask_and_target_pair = map(
                lambda g: (
                    reconstruct_array_by_compressed_array(
                        g["area_shape"], g["top_left_index"], g["original_shape"]
                    ),
                    g[key],
                ),
                grain_list,
            )

            for mask, val in grain_mask_and_target_pair:
                if val is not None:
                    plotted_key_val[mask] = val

            return cast(RGBPicture, plotted_key_val)
        else:
            raise ValueError("invalid item input.")


def get_number_map_from_grain_list_key(
    grain_list: list[Grain], key: GrainAcceptedLiteral
) -> D2FloatArray | None:
    if len(grain_list) > 0:
        plot_shape = grain_list[0]["original_shape"]

        # if np.any(list(map(lambda x: x[key] is None, grain_list))):
        #     return None
        # else:
            # check validity of key for plot ------------------------------
            # if np.ndim(plotted_item) == 0:
            # else:
            #     ValueError("invalid dimension input.")

        grain_mask_and_target_pair = map(
            lambda g: (
                reconstruct_array_by_compressed_array(
                    g["area_shape"], g["top_left_index"], g["original_shape"]
                ),
                g[key], # Noneの可能性ある
            ),
            grain_list,
        )

        plotted_key_val = np.zeros(plot_shape)
        for mask, val in grain_mask_and_target_pair:
            if (
                isinstance(val, float)
                or isinstance(val, int)
                or isinstance(val, bool)
                or isinstance(val, np.float64)
                or isinstance(val, np.uint8)
                or isinstance(val, np.int_)
                or isinstance(val, np.bool_)
            ):
                plotted_key_val[mask] = float(val)
            elif val is None: # valがNoneのときは、とくにmaskに追加しない
                continue
            else:
                continue
                # return continue

        return D2FloatArray(plotted_key_val)
    else:
        return None


def make_grain_segmented_maps_for_CIP(
    grain_list: list[Grain], segmented_map: GrainSegmentedMaps
) -> GrainSegmentedMaps:
    azimuth360 = get_number_map_from_grain_list_key(grain_list, "azimuth360")
    inclination = get_number_map_from_grain_list_key(grain_list, "inclination")

    len_in_mask_pixel = get_number_map_from_grain_list_key(
        grain_list, "len_in_mask_pixel"
    )

    return {
        **segmented_map,
        "azimuth360": azimuth360,
        "inclination": inclination,
        "len_in_mask_pixel": len_in_mask_pixel,
    }


def make_grain_segmented_maps(grain_list: list[Grain]) -> GrainSegmentedMaps:

    R_color = get_rgb_map_from_grain_list_key(grain_list, "R_color")

    extinction_color = get_rgb_map_from_grain_list_key(grain_list, "extinction_color")

    extinction_angle = get_number_map_from_grain_list_key(
        grain_list, "extinction_angle"
    )

    sd_extinction_angle = get_number_map_from_grain_list_key(
        grain_list, "sd_extinction_angle"
    )

    R = get_number_map_from_grain_list_key(grain_list, "R")

    size = get_number_map_from_grain_list_key(grain_list, "size")

    H = get_number_map_from_grain_list_key(grain_list, "H")

    S = get_number_map_from_grain_list_key(grain_list, "S")

    V = get_number_map_from_grain_list_key(grain_list, "V")

    eccentricity = get_number_map_from_grain_list_key(grain_list, "eccentricity")

    angle_deg = get_number_map_from_grain_list_key(grain_list, "angle_deg")
    major_axis_length = get_number_map_from_grain_list_key(
        grain_list, "major_axis_length"
    )
    minor_axis_length = get_number_map_from_grain_list_key(
        grain_list, "minor_axis_length"
    )

    R70 = get_number_map_from_grain_list_key(grain_list, "R70")
    R80 = get_number_map_from_grain_list_key(grain_list, "R80")
    R90 = get_number_map_from_grain_list_key(grain_list, "R90")
    m45_R_color_map = get_rgb_map_from_grain_list_key(grain_list, "m45_color")
    m45_R_map = get_number_map_from_grain_list_key(grain_list, "mR")
    p45_R_color_map = get_rgb_map_from_grain_list_key(grain_list, "p45_color")
    p45_R_map = get_number_map_from_grain_list_key(grain_list, "pR")
    azimuth = get_number_map_from_grain_list_key(grain_list, "azimuth")
    sd_azimuth = get_number_map_from_grain_list_key(grain_list, "sd_azimuth")
    quality = get_number_map_from_grain_list_key(grain_list, "exQuality")
    tilt0_plus_ratio = get_number_map_from_grain_list_key(
        grain_list, "tilt0_plus_ratio"
    )
    tilt45_plus_ratio = get_number_map_from_grain_list_key(
        grain_list, "tilt45_plus_ratio"
    )

    return {
        "R_color_map": R_color,
        "extinction_color_map": extinction_color,
        "extinction_angle": extinction_angle,
        "max_retardation_map": R,
        "size": size,
        "extinction_angle_quality": quality,
        "H": H,
        "S": S,
        "V": V,
        "eccentricity": eccentricity,
        "angle_deg": angle_deg,
        "major_axis_length": major_axis_length,
        "minor_axis_length": minor_axis_length,
        "R_70_map": R70,
        "R_80_map": R80,
        "R_90_map": R90,
        "m45_R_color_map": m45_R_color_map,
        "m45_R_map": m45_R_map,
        "p45_R_color_map": p45_R_color_map,
        "p45_R_map": p45_R_map,
        "azimuth": azimuth,
        "sd_azimuth": sd_azimuth,
        "sd_extinction_angle_map": sd_extinction_angle,
        "tilt0_plus_ratio_map": tilt0_plus_ratio,
        "tilt45_plus_ratio_map": tilt45_plus_ratio,
        "inclination": None,
        "azimuth360": None,
        "len_in_mask_pixel": None,
    }
