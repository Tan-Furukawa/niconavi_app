from niconavi.type import ComputationResult
from typing import Callable, Literal, TypedDict, TypeGuard, TypeVar, overload
from niconavi.type import (
    ColorChart,
    OpticalParameters,
    ComputationResult,
    GrainDetectionParameters,
    QuartzWedgeNormalization,
    TiltImageInfo,
)


# -----------------------------------------------------------------------
# reset run all
# -----------------------------------------------------------------------


def remove_heavy_objects(
    params: ComputationResult,
) -> ComputationResult:

    return ComputationResult(
        **{
            **params.__dict__,
            "pics": None,
            "reta_pics": None,
            "pics_rotated": None,
            "reta_pics_rotated": None,
            # "tilt_image_info": TiltImageInfo(
            # **{
            #     **params.tilt_image_info.__dict__,
            # "image0_raw": None,
            # "image45_raw": None,
            # "tilt_image0_raw": None,
            # "tilt_image45_raw": None,
            # }
            # ),
        }
    )


def reset_load_data(
    params: ComputationResult,
) -> ComputationResult:

    tilt_img = TiltImageInfo()
    tilt_img.image0_path = None
    # tilt_img.image45_path = None
    tilt_img.tilt_image0_path = None
    # tilt_img.tilt_image45_path = None

    return ComputationResult(
        **{
            **params.__dict__,
            "pics": None,
            "reta_pics": None,
            "mask": None,
            "original_resolution": None,
            "original_reta_resolution": None,
            "resolution_height": None,
            "video_path": None,
            "reta_video_path": None,
            "tilt_image_info": tilt_img,
            "first_image": {
                **params.first_image,
                "xpl": None,
                "full_wave": None,
                "image0": None,
                "image0_tilt": None,
                # "image45": None,
                # "image45_tilt": None,
            },
        }
    )


def reset_estimate_tilt_image_result(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "tilt_image_info": TiltImageInfo(
                **{
                    **params.tilt_image_info.__dict__,
                    "tilt_image0": None,
                    # "tilt_image45": None,
                }
            ),
        }
    )


def reset_find_image_center(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "center_int_x": None,
            "center_int_y": None,
            "rotation_img": None,
        }
    )


def reset_update_superimpose_image(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(**{**params.__dict__, "rotation_img": None})


def reset_add_center_image(params: ComputationResult) -> ComputationResult:
    return ComputationResult(**{**params.__dict__, "rotation_img_with_mark": None})


def reset_convert_pics_by_resolution(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "pics": None,
            "reta_pics": None,
            "resolution_height": None,
            "center_int_x": None,
            "center_int_y": None,
        }
    )


def reset_determine_rotation_angle(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "angles": None,
            "pics_rotated": None,
            "image_rotation_direction": None,
            "reta_angles": None,
            "reta_pics_rotated": None,
            "reta_image_rotation_direction": None,
        }
    )


def reset_get_inclination(
    params: ComputationResult,
) -> ComputationResult:
    if params.raw_maps is not None:
        return ComputationResult(
            **{
                **params.__dict__,
                "raw_maps": {
                    **params.raw_maps,
                    "inclination": None,
                    "azimuth360": None,
                },
            }
        )
    else:
        return ComputationResult(
            **{
                **params.__dict__,
                "raw_maps": None,
            }
        )


def reset_make_raw_color_maps(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "raw_maps": None,
        }
    )


def reset_make_raw_R_maps(
    params: ComputationResult,
) -> ComputationResult:
    if params.raw_maps is not None:
        return ComputationResult(
            **{
                **params.__dict__,
                "raw_maps": {
                    **params.raw_maps,
                    "max_retardation_map": None,
                    "m45_R_map": None,
                    "p45_R_map": None,
                    "azimuth": None,
                },
            }
        )
    else:
        return reset_make_raw_color_maps(params)


def reset_make_grain_boundary(
    params: ComputationResult,
) -> ComputationResult:

    return ComputationResult(
        **{
            **params.__dict__,
            "grain_map": None,
            "grain_map_original": None,
            "grain_boundary": None,
            "grain_boundary_original": None,
            "grain_map_with_boundary": None,
        }
    )


def reset_analyze_grain_list(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "grain_list": None,
            "grain_segmented_maps": None,
        }
    )


def reset_analyze_grain_list_for_CIP(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "grain_segmented_maps": (
                {
                    **params.grain_segmented_maps,
                    "azimuth360": None,
                    "inclination": None,
                    "len_in_mask_pixel": None,
                }
                if params.grain_segmented_maps is not None
                else None
            ),
            "grain_list": (
                list(
                    map(
                        lambda x: {
                            **x,
                            "inclination": None,
                            "azimuth360": None,
                            "len_in_mask_pixel": None,
                        },
                        params.grain_list,
                    )
                )
                if params.grain_list is not None
                else None
            ),
        }
    )


def reset_make_CIP_map_info(params: ComputationResult) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "cip_map_info": None,
        }
    )


def reset_make_retardation_color_chart(
    params: ComputationResult,
) -> ComputationResult:

    return ComputationResult(
        **{
            **params.__dict__,
            "color_chart": ColorChart(
                **{
                    **params.color_chart.__dict__,
                    "xpl_retardation_color_chart": None,
                    "xpl_R_array": None,
                    "xpl_alpha": None,
                    "pol_lambda_retardation_color_chart": None,
                    "pol_lambda_alpha": None,
                    "pol_lambda_R_array": None,
                    "inc_retardation_color_chart": None,
                    "inc_R_array": None,
                }
            ),
        }
    )


def reset_make_retardation_color_chart_with_quartz_wedge(
    params: ComputationResult,
) -> ComputationResult:

    return ComputationResult(
        **{
            **params.__dict__,
            "quartz_wedge_normalization": QuartzWedgeNormalization(
                **{
                    **params.quartz_wedge_normalization.__dict__,
                    "quartz_wedge_path": None,
                    "raw_data": None,
                    "calibrated_data": None,
                    "retardation": None,
                    "summary_display": None,
                    "summary_rgb_display": None,
                    # "convert_retardation_to_inclination": convert_retardation_to_inclination,
                    # "max_retardation_for_inclination_search": R,
                }
            ),
        }
    )


def reset_add_retardation_analysis_to_grain_list(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "grain_list": None,
        }
    )


def reset_grain_merge(
    params: ComputationResult,
) -> ComputationResult:
    return ComputationResult(
        **{
            **params.__dict__,
            "grain_map": None,
            "grain_boundary": None,
        }
    )


def reset_grain_segmentation(
    params: ComputationResult,
) -> ComputationResult:

    return ComputationResult(
        **{
            **params.__dict__,
            "grain_classification_result": None,
            "grain_classification_image": None,
            "grain_classification_legend": None,
            "grain_list": (
                list(
                    map(
                        lambda x: {
                            **x,
                            "mineral": None,
                        },
                        params.grain_list,
                    )
                )
                if params.grain_list is not None
                else None
            ),
        }
    )
