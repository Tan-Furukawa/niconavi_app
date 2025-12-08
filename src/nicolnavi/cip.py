# %%
from typing import Callable, Literal, TypedDict, TypeGuard, TypeVar, overload
from niconavi.tools.type import is_not_tuple_none
from niconavi.image.image import create_outside_circle_mask
from niconavi.type import Grain, CIPMapInfo, RawMaps, GrainSegmentedMaps, TiltImageInfo
from niconavi.optics.plot import make_2d_polar_map
import numpy as np
from niconavi.tools.type import D1FloatArray, D2BoolArray
from niconavi.type import GrainSelectedResult, Grain
import matplotlib.pyplot as plt


def get_inclination_and_azimuth_from_grain_list(
    grain_list: list[Grain],
    grain_classification_result: dict[str, GrainSelectedResult],
    azm_key: Literal["extinction_angle", "azimuth", "azimuth360"],
) -> dict[str, tuple[D1FloatArray, D1FloatArray]]:

    # mineral_list = grain_classification_result.keys()
    # print(list(map(lambda x: x["mineral"], grain_list)))

    mineral_list = list(grain_classification_result.keys())

    # np.unique(
    #     np.array(
    #         list(
    #             filter(lambda x: x is not None, map(lambda x: x["mineral"], grain_list))
    #         )
    #     )
    # )

    res: dict[str, tuple[D1FloatArray, D1FloatArray]] = {}

    for mineral in mineral_list:
        # print("-------------")
        # print(grain_list)
        # print("-------------")
        target_grain_list = list(filter(lambda x: x["mineral"] == mineral, grain_list))
        lis = list(
            filter(
                is_not_tuple_none,
                list(
                    map(
                        lambda x: (
                            (x["inclination"], x[azm_key])
                            if x["index"] != 0
                            else (None, None)
                        ),
                        target_grain_list,
                    )
                ),
            )
        )

        res[mineral] = (
            D1FloatArray(np.array(list(map(lambda x: x[0], lis)), dtype=np.float64)),
            D1FloatArray(np.array(list(map(lambda x: x[1], lis)), dtype=np.float64)),
        )

    return res


def make_CIP_maps(
    grain_list: list[Grain],
    raw_maps: RawMaps,
    # grain_segmented_maps: GrainSegmentedMaps,
    tilt_image_info: TiltImageInfo,
    grain_classification_result: dict[str, GrainSelectedResult],
) -> CIPMapInfo:

    # seg_azimuth90 = grain_segmented_maps["extinction_angle"]
    # seg_azimuth180 = grain_segmented_maps["azimuth"]
    # seg_azimuth360 = grain_segmented_maps["azimuth360"]
    # seg_inclination = grain_segmented_maps["inclination"]

    map_azimuth90 = raw_maps["extinction_angle"]
    map_azimuth180 = raw_maps["azimuth"]
    map_azimuth360 = raw_maps["azimuth360"]
    map_inclination = raw_maps["inclination"]

    # if seg_inclination is None or map_inclination is None:
    #     raise ValueError("seg_inclination is None or map_inclination is None")

    # if seg_azimuth90 is None or map_azimuth90 is None:
    #     raise ValueError("extinction_angle is None")

    # mask1 = tilt_image_info.tilt_image0["image_mask"]
    # if tilt_image_info.tilt_image0 is not None:
    #     mask1 = tilt_image_info.tilt_image0["image_mask"]
    # else:
    #     mask1 = D2BoolArray(np.zeros(seg_azimuth90.shape, dtype=np.bool_))

    # mask2 = tilt_image_info.tilt_image45["image_mask"]
    # if tilt_image_info.tilt_image45 is not None:
    #     mask2 = tilt_image_info.tilt_image45["image_mask"]
    # else:
    #     mask2 = D2BoolArray(np.zeros(seg_azimuth90.shape, dtype=np.bool_))

    # mask3 = ~create_outside_circle_mask(seg_azimuth90)

    # mask = D2BoolArray((mask1 & mask2) & ~mask3)
    # mask = D2BoolArray((mask1 & mask2))
    # mask = D2BoolArray(mask1 & ~mask3)

    polor90 = get_inclination_and_azimuth_from_grain_list(
        grain_list, grain_classification_result, "extinction_angle"
    )
    # gmap90, _ = make_2d_polar_map(
    #     seg_inclination, seg_azimuth90, symetry="90", color_pattern="center_white"
    # )
    gmap90 = None
    cmap90, legend90 = make_2d_polar_map(
        map_inclination, map_azimuth90, symetry="90", color_pattern="center_white"
    )

    # if seg_azimuth180 is not None and map_azimuth180 is not None:
    if map_azimuth180 is not None:
        polor180 = get_inclination_and_azimuth_from_grain_list(
            grain_list, grain_classification_result, "azimuth"
        )
        gmap180 = None
        # gmap180, _ = make_2d_polar_map(
        #     seg_inclination, seg_azimuth180, symetry="180", color_pattern="center_white"
        # )
        cmap180, legend180 = make_2d_polar_map(
            map_inclination, map_azimuth180, symetry="180", color_pattern="center_white"
        )
    else:
        polor180 = None
        cmap180 = None
        gmap180 = None
        legend180 = None

    # if seg_azimuth360 is not None and map_azimuth360 is not None:
    if map_azimuth360 is not None:
        polor360 = get_inclination_and_azimuth_from_grain_list(
            grain_list, grain_classification_result, "azimuth360"
        )
        gmap360 = None
        # gmap360, _ = make_2d_polar_map(
        #     seg_inclination,
        #     seg_azimuth360,
        #     symetry="360",
        #     color_pattern="center_white",
        #     mask=mask,
        # )
        cmap360, legend360 = make_2d_polar_map(
            map_inclination,
            map_azimuth360,
            symetry="360",
            color_pattern="center_white",
            # mask=mask,
        )
    else:
        polor360 = None
        cmap360 = None
        gmap360 = None
        legend360 = None

    # print(CIPMapInfo( polar_info180=polor180,
    #     polar_info360=polor360,
    #     polar_info90=polor90,
    #     COI180_grain=gmap180,
    #     COI360_grain=gmap360,
    #     COI90_grain=gmap90,
    #     COI180=cmap180,
    #     COI360=cmap360,
    #     COI90=cmap90,
    #     legend180=legend180,
    #     legend360=legend360,
    #     legend90=legend90,
    # ))

    return CIPMapInfo(
        polar_info180=polor180,
        polar_info360=polor360,
        polar_info90=polor90,
        COI180_grain=gmap180,
        COI360_grain=gmap360,
        COI90_grain=gmap90,
        COI180=cmap180,
        COI360=cmap360,
        COI90=cmap90,
        legend180=legend180,
        legend360=legend360,
        legend90=legend90,
    )
