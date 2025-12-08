# %%
from typing import Callable, Literal, TypedDict, TypeGuard, TypeVar, overload
from niconavi.custom_error import (
    InvalidRotationDirection,
    RotatedAngleError,
    UnexpectedNoneType,
    NoVideoError,
)
from niconavi.tools.type import is_not_tuple_none
from niconavi.tools.grain_plot import detect_boundaries
from niconavi.optics.uniaxial_plate import get_retardation_color_chart_with_nd_filter
from niconavi.optics.tools import normalize_axes
from niconavi.tilt_image import (
    estimate_tilted_image,
)
from copy import deepcopy
import traceback
import matplotlib.pyplot as plt
import numpy as np
from niconavi.inclination import (
    get_inclination_map,
    make_R_vs_azimuth_color_chart,
    convert_tilt_to_tilt_0_to_180,
)
from niconavi.cip import make_CIP_maps
from niconavi.find_center import (
    find_rotation_center,
    get_image_edges,
    make_superimpose_image,
    plot_center_image,
)
from niconavi.grain_analysis import (  # make_retardation_estimation_function,
    add_azimuth_to_grain_list,
    add_additional_information_to_grain_list,
    analyze_grain,
    analyze_grain_for_CIP,
)
from niconavi.grain_detection import (
    estimate_grain_map,
)
from niconavi.image.image import (
    resize_img,
    resize_image_list,
    create_outside_circle_mask,
)
from niconavi.image.type import Color, D1RGB_Array
from niconavi.make_map import make_color_maps, make_R_maps, make_retardation_color_map
from niconavi.optics.color import show_color
from niconavi.retardation_normalization import (
    select_h_in_color_chart,
    extract_center_pixels,
    normalize_retardation_plate_less_than_retardation1700,
    plot_retardation_color_chart,
    plot_rgb_in_sensitive_color_plate,
    estimate_median_alpha_of_nd_filter,
    make_R_map,
    make_true_R_color_map,
    make_theoritical_tilt_image,
)
from niconavi.rorate import fitting_by_rotation
from niconavi.tools.array import pick_element_from_array
from niconavi.tools.change_type import as_two_element_tuple
from niconavi.tools.read_data import divide_video_into_n_frame
from niconavi.tools.type import D2BoolArray, D1FloatArray
from niconavi.type import (
    ColorChart,
    OpticalParameters,
    ComputationResult,
    GrainDetectionParameters,
    QuartzWedgeNormalization,
    TiltImageInfo,
)
from niconavi.optics.uniaxial_plate import get_retardation_color_chart
from niconavi.grain_classification import create_colored_map, select_grain_by_code
from niconavi.grain_merge import merge_grain_by_code
from niconavi.grain_segmentation_map import (
    make_grain_segmented_maps,
    make_grain_segmented_maps_for_CIP,
)
from niconavi.optics.tools import (
    make_angle_retardation_estimation_function,
    get_thickness_from_max_retardation,
    get_max_retardation_from_thickness,
)
from niconavi.tools.select_code_parser import add_random_colors_to_input
import warnings

T = TypeVar("T")


def is_not_None_type(params: T | None) -> TypeGuard[T]:
    if params is not None:
        return True
    else:
        return False


def make_err_msg(params: ComputationResult, *keys: str) -> str:
    for key in keys:
        if params.__dict__[key] is None:  # type: ignore
            return f"{key} in ComputationResult type variables must not be None"
    return ""


def load_data(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:

    if is_not_None_type(params.video_path):
        try:
            video_path = params.video_path

            pics = divide_video_into_n_frame(
                video_path, params.frame_number, progress_callback
            )

            resized_pics = resize_image_list(pics, params.resolution_width)
            list(map(lambda x: resize_img(x, params.resolution_width), pics))
            h, w = as_two_element_tuple(pics[0].shape)
            original_resolution = (w, h)

            if not is_not_None_type(params.reta_video_path):
                return ComputationResult(
                    **{
                        **params.__dict__,
                        "pics": resized_pics,
                        "original_resolution": original_resolution,
                        "first_image": {
                            **params.first_image,
                            "xpl": resized_pics[0],
                        },
                    }
                )

            else:
                try:
                    reta_video_path = params.reta_video_path
                    reta_pics = divide_video_into_n_frame(
                        reta_video_path, params.frame_number, progress_callback
                    )
                    # if reta_pics[0].shape != pics[0].shape:
                    #     raise ValueError("different shape between reta_pics and pics")

                    resized_reta_pics = resize_image_list(
                        reta_pics, params.resolution_width
                    )
                    if not (
                        is_not_None_type(params.tilt_image_info.tilt_image0_path)
                        # and is_not_None_type(params.tilt_image_info.image45_path)
                        # and is_not_None_type(params.tilt_image_info.tilt_image45_path)
                    ):
                        return ComputationResult(
                            **{
                                **params.__dict__,
                                "pics": resized_pics,
                                "reta_pics": resized_reta_pics,
                                "first_image": {
                                    **params.first_image,
                                    "xpl": resized_pics[0],
                                    "full_wave": resized_reta_pics[0],
                                },
                                "original_resolution": original_resolution,
                                "original_reta_resolution": as_two_element_tuple(
                                    reta_pics[0].shape
                                ),
                                "resolution_height": h,
                            }
                        )
                    else:
                        image0 = [resized_reta_pics[0]]
                        image45 = (
                            resize_image_list(
                                divide_video_into_n_frame(
                                    params.tilt_image_info.image45_path,
                                    1,
                                    progress_callback,
                                ),
                                params.resolution_width,
                            )
                            if is_not_None_type(params.tilt_image_info.image45_path)
                            else None
                        )
                        tilt0 = resize_image_list(
                            divide_video_into_n_frame(
                                params.tilt_image_info.tilt_image0_path,
                                params.tilt_image_info.frame_num,
                                progress_callback,
                            ),
                            params.resolution_width,
                        )

                        tilt45 = (
                            resize_image_list(
                                divide_video_into_n_frame(
                                    params.tilt_image_info.tilt_image45_path,
                                    params.tilt_image_info.frame_num,
                                    progress_callback,
                                ),
                                params.resolution_width,
                            )
                            if is_not_None_type(
                                params.tilt_image_info.tilt_image45_path
                            )
                            else None
                        )

                        return ComputationResult(
                            **{
                                **params.__dict__,
                                "pics": resized_pics,
                                "reta_pics": resized_reta_pics,
                                "original_resolution": original_resolution,
                                "original_reta_resolution": as_two_element_tuple(
                                    reta_pics[0].shape
                                ),
                                "resolution_height": h,
                                "tilt_image_info": TiltImageInfo(
                                    **{
                                        **params.tilt_image_info.__dict__,
                                        "image0_raw": image0,
                                        "image45_raw": image45,
                                        "tilt_image0_raw": tilt0,
                                        "tilt_image45_raw": tilt45,
                                    }
                                ),
                                "first_image": {
                                    **params.first_image,
                                    "xpl": resized_pics[0],
                                    "full_wave": resized_reta_pics[0],
                                    "image0": image0[0],
                                    "image0_tilt": tilt0[0],
                                    "image45": (
                                        image45[0] if image45 is not None else None
                                    ),
                                    "image45_tilt": (
                                        tilt45[0] if tilt45 is not None else None
                                    ),
                                },
                            }
                        )

                except Exception as e:

                    traceback.print_exc()
                    raise ValueError(f"error in load_data(): {traceback.format_exc()}")

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in load_data(): {traceback.format_exc()}")
    else:
        raise NoVideoError()


def estimate_tilt_image_result(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:

    progress_callback(None)

    alpha = params.color_chart.pol_lambda_alpha
    im0 = params.tilt_image_info.image0_raw
    im_tilt0 = params.tilt_image_info.tilt_image0_raw

    im45 = params.tilt_image_info.image45_raw
    im_tilt45 = params.tilt_image_info.tilt_image45_raw
    angle_of_image45 = params.angle_between_x_and_thin_section_axis_at_tilt

    theta_deg = params.tilt_image_info.theta_thin_section
    center_x = params.center_int_x
    center_y = params.center_int_y

    if params.raw_maps is not None:
        ex_angle_map = params.raw_maps["extinction_angle"]
        shape = (ex_angle_map.shape[0], ex_angle_map.shape[1])
        # im0とim_tilt0は必須
        if (
            alpha is not None
            and im0 is not None
            and im_tilt0 is not None
            and center_x is not None
            and center_y is not None
            and params.raw_maps is not None
        ):
            center = (center_x, center_y)
            im_result0 = estimate_tilted_image(
                im0,
                im_tilt0,
                np.radians(theta_deg),
                center=center,
                shape=shape,
            )

            # ------------------------------
            # 45°傾き画像が存在するとき
            # ------------------------------
            if im45 is not None and im_tilt45 is not None:
                im_result45 = estimate_tilted_image(
                    im45,
                    im_tilt45,
                    np.radians(theta_deg),
                    center=center,
                    shape=shape,
                    rotation=-angle_of_image45,
                )
                return ComputationResult(
                    **{
                        **params.__dict__,
                        "tilt_image_info": TiltImageInfo(
                            **{
                                **params.tilt_image_info.__dict__,
                                "tilt_image0": im_result0,
                                "tilt_image45": im_result45,
                            }
                        ),
                    }
                )
            # ------------------------------
            # 45°傾き画像が存在しないとき
            # ------------------------------
            else:
                return ComputationResult(
                    **{
                        **params.__dict__,
                        "tilt_image_info": TiltImageInfo(
                            **{
                                **params.tilt_image_info.__dict__,
                                "tilt_image0": im_result0,
                            }
                        ),
                    }
                )
        else:
            return params

    else:
        raise ValueError("params.raw_maps is None")


def find_image_center(
    params: ComputationResult,
    progress_callback: Callable[[float], None] = lambda p: None,
) -> ComputationResult:
    if is_not_None_type(params.pics):
        try:
            rotation_img, center = find_rotation_center(params.pics, progress_callback)
            # print("----------------------")
            # print(center)
            # print("----------------------")
            return ComputationResult(
                **{
                    **params.__dict__,
                    "center_int_x": center[0],
                    "center_int_y": center[1],
                    "rotation_img": rotation_img,
                }
            )
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in find_center(): {traceback.format_exc()}")
    else:
        raise ValueError("divided frames of video is not found")


def update_superimpose_image(
    params: ComputationResult,
) -> ComputationResult:
    if is_not_None_type(params.pics):
        try:
            return ComputationResult(
                **{
                    **params.__dict__,
                    "rotation_img": make_superimpose_image(
                        get_image_edges, params.pics
                    ),
                }
            )
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in update_superimpose_image(): {e}")
    else:
        raise ValueError("divided frames of video is not found")


def add_center_image(params: ComputationResult) -> ComputationResult:
    if (
        is_not_None_type(params.rotation_img)
        and is_not_None_type(params.center_int_x)
        and is_not_None_type(params.center_int_y)
    ):
        try:
            return ComputationResult(
                **{
                    **params.__dict__,
                    "rotation_img_with_mark": plot_center_image(
                        params.rotation_img, (params.center_int_x, params.center_int_y)
                    ),
                }
            )
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in add_center_image(): {e}")
    else:
        raise ValueError("rotation center is not found")


def convert_pics_by_resolution(
    params: ComputationResult,
) -> ComputationResult:
    if (
        is_not_None_type(params.pics)
        and is_not_None_type(params.center_int_x)
        and is_not_None_type(params.center_int_y)
    ):
        try:
            center = (params.center_int_x, params.center_int_y)
            rpics = resize_img(params.pics[0], params.resolution_width)
            resolution_height = rpics.shape[0]
            # cx = int(center[0] * params.resolution_width)
            # cy = int(center[1] * resolution_height)
            cx = int(center[0])
            cy = int(center[1])

            return ComputationResult(
                **{
                    **params.__dict__,
                    "pics": list(
                        map(
                            lambda pic: resize_img(pic, params.resolution_width),
                            params.pics,
                        )
                    ),
                    "reta_pics": (
                        list(
                            map(
                                lambda pic: resize_img(pic, params.resolution_width),
                                params.reta_pics,
                            )
                        )
                        if params.reta_pics is not None
                        else None
                    ),
                    "resolution_height": resolution_height,
                    "center_int_x": cx,
                    "center_int_y": cy,
                }
            )
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in convert_pics_py_resolution(): {e}")
    else:
        raise ValueError("rotation center is not found")


def determine_rotation_angle(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:
    if (
        is_not_None_type(params.pics)
        and is_not_None_type(params.center_int_x)
        and is_not_None_type(params.center_int_y)
        and is_not_None_type(params.resolution_width)
    ):
        rrpics, angles = fitting_by_rotation(
            params.pics,
            (params.center_int_x, params.center_int_y),
            params.resolution_width,
            progress_callback,
        )

        # plt.imshow(rrpics[0])
        # plt.show()
        # plt.plot(angles)
        # plt.show()
        if angles[-1] > 0:
            direction = "clockwise"
            used_angles = angles
            raise InvalidRotationDirection("XPL movie")
        else:
            direction = "counterclockwise"
            used_angles = -angles
            if used_angles[-1] < 270 + 45:
                raise RotatedAngleError("XPL")

        if is_not_None_type(params.reta_pics):
            reta_rrpics, reta_angles = fitting_by_rotation(
                params.reta_pics,
                (params.center_int_x, params.center_int_y),
                params.resolution_width,
                progress_callback,
            )

            # print("####################")
            # print(reta_angles)
            # print("####################")

            if reta_angles[-1] > 0:
                reta_direction = "clockwise"
                used_reta_angles = reta_angles
                # raise ValueError("the clockwise rotation of XPL is not permitted")
                raise InvalidRotationDirection("XPL + λ-Plate movie")
            else:
                reta_direction = "counterclockwise"
                used_reta_angles = -reta_angles
                if used_reta_angles[-1] < 270 + 45:
                    raise RotatedAngleError("XPL+lambda")

        return ComputationResult(
            **{
                **params.__dict__,
                "angles": used_angles,
                "pics_rotated": rrpics,
                "image_rotation_direction": direction,
                "reta_angles": (
                    used_reta_angles if params.reta_pics is not None else None
                ),
                "reta_pics_rotated": (
                    reta_rrpics if params.reta_pics is not None else None
                ),
                "reta_image_rotation_direction": (
                    reta_direction if params.reta_pics is not None else None
                ),
            }
        )
    else:
        raise UnexpectedNoneType(
            "params.pics or params.center_int_x or params.center_int_y or params.resolution_width"
        )


def get_inclination(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:
    no = params.optical_parameters.no
    ne = params.optical_parameters.ne

    progress_callback(None)
    if params.tilt_image_info.estimate_inclination_by == "max R":
        if params.optical_parameters.max_R is None:
            raise ValueError("max R is None.")
        else:
            thickness = get_thickness_from_max_retardation(
                max_retardation=params.optical_parameters.max_R, no=no, ne=ne
            )

    else:
        thickness = params.optical_parameters.thickness

    max_R = get_max_retardation_from_thickness(thickness, no=no, ne=ne)

    # これ以下の議論はすべてthicknessを基準に行われる

    if params.raw_maps is not None:

        # R_map = params.raw_maps["max_retardation_map"]

        R_color_map = params.raw_maps["R_color_map"]
        xpl_color_chart = (params.color_chart.xpl_retardation_color_chart,)
        xpl_R_array = (params.color_chart.xpl_R_array,)

        # ここで再度R_mapを求める
        R_map, _, _ = make_retardation_color_map(
            R_color_map, xpl_color_chart[0], xpl_R_array[0], maxR=max_R
        )

        if R_map is not None:
            inclination_map = get_inclination_map(
                R_map, no=no, ne=ne, thickness=thickness
            )
        else:
            Warning("None type of params.raw_maps[max_retardation_map]")
            return params
    else:
        Warning("None type of params.raw_maps")
        return params

    angle_of_image45 = params.angle_between_x_and_thin_section_axis_at_tilt
    alpha = params.color_chart.pol_lambda_alpha

    # --------------------------------------------------------
    # 消光角のみしかわからないとき
    # --------------------------------------------------------
    if not (alpha is not None and params.raw_maps is not None):
        return ComputationResult(
            **{
                **params.__dict__,
                "raw_maps": {
                    **params.raw_maps,
                    "inclination": inclination_map,
                },
            }
        )

    else:
        ex_angle_map = params.raw_maps["extinction_angle"]
        azimuth_map = params.raw_maps["azimuth"]

        if azimuth_map is not None and ex_angle_map is not None and R_map is not None:
            # --------------------------------------------------------
            # azimuthが判別しているとき
            # --------------------------------------------------------
            R_vs_azimuth_color_chart = make_R_vs_azimuth_color_chart(
                alpha, progress_callback=progress_callback
            )

            im_result0 = params.tilt_image_info.tilt_image0
            im_result45 = params.tilt_image_info.tilt_image45

            # print("--------------------------------")
            # print("--------------------------------")
            # print(params.tilt_image_info.image0_path)
            # print("--------------------------------")
            # print("--------------------------------")

            if im_result0 is not None:
                # if im_result0 is not None and im_result45 is not None:
                # --------------------------------------------------------
                # 傾斜動画が入力されているとき
                # --------------------------------------------------------
                # if im_result45 is not None:  # 45°傾斜画像入力されているとき
                #     tilt_direction = estimate_tilt_direction(
                #         im_result0,
                #         R_vs_azimuth_color_chart,
                #         azimuth_map,
                #         ex_angle_map,
                #         im_result45,
                #         angle_of_image45,
                #     )
                # else:  # 45°傾斜画像の入力なし
                #     tilt_direction = estimate_tilt_direction(
                #         im_result0,
                #         R_vs_azimuth_color_chart,
                #         azimuth_map,
                #         ex_angle_map,
                #         im_result45=None,
                #         angle_of_image45=None,
                #     )

                inclination_0_to_180 = convert_tilt_to_tilt_0_to_180(
                    params,
                    thickness,
                    # im_result0["azimuth_thin_section"],
                    # azimuth_map,
                    # inclination_map,
                    # median_kernel_size=9,
                    # tilt_direction["become_red_by_tilt"],
                )

                n_inclination, n_azimuth = normalize_axes(
                    inclination_0_to_180, azimuth_map
                )

                return ComputationResult(
                    **{
                        **params.__dict__,
                        "raw_maps": {
                            **params.raw_maps,
                            "inclination": n_inclination,
                            "inclination_0_to_180": inclination_0_to_180,
                            "azimuth360": n_azimuth,
                            # "tilt_direction_estimation_result": tilt_direction,
                        },
                    }
                )

            else:
                return ComputationResult(
                    **{
                        **params.__dict__,
                        "raw_maps": {
                            **params.raw_maps,
                            "inclination": inclination_map,
                        },
                    }
                )
        else:
            return ComputationResult(
                **{
                    **params.__dict__,
                    "raw_maps": {
                        **params.raw_maps,
                        "inclination": inclination_map,
                    },
                }
            )


def make_raw_color_maps(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:
    if is_not_None_type(params.pics_rotated) and is_not_None_type(params.angles):
        try:
            return ComputationResult(
                **{
                    **params.__dict__,
                    "raw_maps": make_color_maps(
                        params.pics_rotated,
                        params.angles,
                        params.reta_pics_rotated,
                        params.reta_angles,
                        progress_callback=progress_callback,
                    ),
                }
            )
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in make_raw_maps(): {e}")
    else:
        raise Warning("params.pics_rotated or params.angles should not None")
        # raise ValueError(make_err_msg(params, "pics_rotated", "angles"))


def make_raw_R_maps(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:
    if (
        is_not_None_type(params.color_chart.xpl_R_array)
        and is_not_None_type(params.color_chart.xpl_retardation_color_chart)
        and is_not_None_type(params.raw_maps)
        and is_not_None_type(params.color_chart.xpl_retardation_color_chart)
    ):
        try:
            return ComputationResult(
                **{
                    **params.__dict__,
                    "raw_maps": make_R_maps(
                        params.raw_maps,
                        params.color_chart.xpl_retardation_color_chart,
                        params.color_chart.xpl_R_array,
                        params.color_chart.pol_lambda_retardation_color_chart,
                        params.color_chart.pol_lambda_R_array,
                        progress_callback=progress_callback,
                        full_wave_plate=params.full_wave_plate_nm,
                        # 以下パラメータは、検板を含む光学系のRetardation評価のときのみ活躍する。
                        # 検板が挿入されたときの画像のRは、min{0,530 - R_xpl} < R R_xpl + 530nmまで検査する
                        max_R=params.full_wave_plate_nm
                        + params.color_chart.xpl_max_retardation,
                        min_R=np.min(
                            params.full_wave_plate_nm
                            - params.color_chart.xpl_max_retardation,
                            0,
                        ),
                    ),
                }
            )
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in load_data(): {traceback.format_exc()}")

    else:
        traceback.print_exc()
        Warning("this line should not execute")
        return params
        # raise ValueError("failed to compute rotation map")
        # raise ValueError(make_err_msg(params, "pics_rotated", "angles"))


def make_grain_boundary(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:

    if params.raw_maps is None:
        Warning("this line should not execute")
        return params

    R_map = params.raw_maps["max_retardation_map"]
    color_map = params.raw_maps.get("R_color_map_display")

    if R_map is None:
        Warning("this line should not execute")
        return params

    angle_map = (
        params.raw_maps["extinction_angle"]
        # if params.grain_detection_parameters.angle_method == "extinction_angle"
        # else params.raw_maps["azimuth"]
    )

    if angle_map is None:
        Warning("this line should not execute")
        return params

    progress_callback(None)
    grain_map, grain_map_with_boundary = estimate_grain_map(
        color_map,
        th_about_hessian_emphasis=params.grain_detection_parameters.th_about_hessian_emphasis,
        th_about_connect_skeleton_endpoints=params.grain_detection_parameters.th_about_connect_skeleton_endpoints,
        permit_inclusion=True,
        smallest_grain_size=params.grain_detection_parameters.smallest_grain_size,
        # R_map=R_map,
        # color_map=color_map,
        # angle_map=angle_map,
        # angle_method=params.grain_detection_parameters.angle_method,
        # eval_method=params.grain_detection_parameters.eval_method,
        # color_map_median_kernel_size=params.grain_detection_parameters.color_map_median_kernel_size,
        # color_map_percentile=params.grain_detection_parameters.color_map_percentile,
        # color_map_min_R=params.grain_detection_parameters.color_map_min_R,
        # color_map_max_R=params.grain_detection_parameters.color_map_max_R,
        # color_rev_R_estimation=params.grain_detection_parameters.color_rev_estimation,
        # angle_map_median_kernel_size=params.grain_detection_parameters.angle_map_median_kernel_size,
        # angle_map_percentile=params.grain_detection_parameters.angle_map_percentile,
        # angle_map_min_R=params.grain_detection_parameters.angle_map_min_R,
        # angle_map_max_R=params.grain_detection_parameters.angle_map_max_R,
        # angle_rev_R_estimation=params.grain_detection_parameters.angle_rev_estimation,
        # smallest_grain_size=params.grain_detection_parameters.smallest_grain_size,
        # mask=params.mask,
    )

    grain_boundary = detect_boundaries(grain_map)

    # print("------------------")
    # fig, ax = plt.subplots()
    # # grain_map_with_boundary[grain_map_with_boundary == 999999] = 100
    # im = ax.imshow(grain_map_with_boundary == 0, cmap="viridis")
    # fig.colorbar(im)
    # fig.savefig("debug.pdf")
    # print("------------------")

    return ComputationResult(
        **{
            **params.__dict__,
            "grain_map": grain_map,
            "grain_map_original": deepcopy(grain_map),
            "grain_boundary": grain_boundary,
            "grain_boundary_original": deepcopy(grain_boundary),
            "grain_map_with_boundary": grain_map_with_boundary,
        }
    )


def analyze_grain_list(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:
    if is_not_None_type(params.raw_maps) and is_not_None_type(params.grain_map):
        try:
            progress_callback(None)
            grain_list = analyze_grain(
                params.raw_maps, params.grain_map, params.circ_threshold
            )
            grain_list = add_additional_information_to_grain_list(grain_list)
            segmented_map = make_grain_segmented_maps(grain_list)
            grain_list = list(filter(lambda x: x["index"] != 0, grain_list))
            return ComputationResult(
                **{
                    **params.__dict__,
                    "grain_list": grain_list,
                    "grain_segmented_maps": segmented_map,
                }
            )
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in analyze_grain_list(): {e}")
    else:
        raise ValueError("calculated maps are not found")


def analyze_grain_list_for_CIP(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:
    if (
        is_not_None_type(params.raw_maps)
        # and is_not_None_type(params.grain_segmented_maps)
        and is_not_None_type(params.grain_list)
    ):
        try:
            progress_callback(None)
            grain_list = analyze_grain_for_CIP(
                params.grain_list,
                params.raw_maps,
                params.circ_threshold,
                params.tilt_image_info,
            )
            # grain_list = add_additional_information_to_grain_list(grain_list)
            # segmented_map = make_grain_segmented_maps_for_CIP(
            #     grain_list, params.grain_segmented_maps
            # )

            return ComputationResult(
                **{
                    **params.__dict__,
                    "grain_list": grain_list,
                    # "grain_segmented_maps": segmented_map,
                }
            )
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in analyze_grain_list(): {traceback.format_exc()}")
    else:
        Warning("calculated maps are not found")
        return params


def make_CIP_map_info(params: ComputationResult) -> ComputationResult:

    if (
        # params.grain_segmented_maps is None
        params.raw_maps is None
        or params.grain_list is None
        or params.grain_classification_result is None
    ):
        raise ValueError("raw_maps or grain_maps should not None")

    return ComputationResult(
        **{
            **params.__dict__,
            "cip_map_info": make_CIP_maps(
                params.grain_list,
                params.raw_maps,
                # params.grain_segmented_maps,
                params.tilt_image_info,
                params.grain_classification_result,
            ),
        }
    )


def make_retardation_color_chart(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:
    try:
        if params.raw_maps is None:
            # warnings.warn("params.raw_maps should not None here")
            # return params
            raise ValueError("params.raw_maps should not None")

        xpl_color_map = params.raw_maps["R_color_map"]
        p45_color_map = params.raw_maps["p45_R_color_map"]
        m45_color_map = params.raw_maps["m45_R_color_map"]
        mask = ~create_outside_circle_mask(xpl_color_map)

        # cross nicol
        # --------------------------------------------
        xpl_end = params.color_chart.xpl_max_retardation
        chart_xpl = get_retardation_color_chart_with_nd_filter(
            start=0,
            end=xpl_end,
            num=100,
            nd_num=30,
            nd_filter_min=0,
            progress_callback=progress_callback,
        )

        index_xpl = select_h_in_color_chart(
            chart_xpl["color_chart"],
            chart_xpl["h"],
            D1RGB_Array(xpl_color_map[mask]),
        )

        progress_callback(None)
        # cross nicol + full wave plate
        # --------------------------------------------
        pol_lambda_end = params.color_chart.pol_lambda_max_retardation
        if (
            pol_lambda_end is not None
            and p45_color_map is not None
            and m45_color_map is not None
        ):
            chart_pol = get_retardation_color_chart_with_nd_filter(
                end=pol_lambda_end,
                num=100,
                nd_num=30,
                nd_filter_min=0,
                progress_callback=progress_callback,
            )

            mask_p = ~create_outside_circle_mask(p45_color_map)
            mask_m = ~create_outside_circle_mask(m45_color_map)

            index_pol = select_h_in_color_chart(
                chart_pol["color_chart"],
                chart_pol["h"],
                D1RGB_Array(
                    np.concatenate((p45_color_map[mask_p], m45_color_map[mask_m]))
                ),
            )
            # print("----------------")
            # print(index_pol)
            # print("----------------")
            pol_lambda_R_array = chart_pol["w"]
            pol_alpha = index_pol["best_h"]
            pol_lambda_retardation_color_chart = index_pol["color_chart_1d"]
        else:
            pol_alpha = None
            pol_lambda_R_array = None
            pol_lambda_retardation_color_chart = None

        return ComputationResult(
            **{
                **params.__dict__,
                "color_chart": ColorChart(
                    **{
                        **params.color_chart.__dict__,
                        "xpl_retardation_color_chart": index_xpl["color_chart_1d"],
                        "xpl_R_array": chart_xpl["w"],
                        "xpl_alpha": index_xpl["best_h"],
                        "pol_lambda_retardation_color_chart": pol_lambda_retardation_color_chart,
                        "pol_lambda_alpha": pol_alpha,
                        "pol_lambda_R_array": pol_lambda_R_array,
                        "inc_retardation_color_chart": None,
                        "inc_R_array": None,
                    }
                ),
            }
        )

    except Exception as e:
        raise ValueError(f"error in make_retardation_color_chart(): {e}")


def make_retardation_color_chart_with_quartz_wedge(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:

    path = params.quartz_wedge_normalization.quartz_wedge_path

    if is_not_None_type(path):
        try:

            raw_data = extract_center_pixels(path, progress_callback)

            colors_standard, retardation = get_retardation_color_chart(
                start=0,
                end=1700,
                progress_callback=progress_callback,
            )

            calibrated_data, y_new, used = (
                normalize_retardation_plate_less_than_retardation1700(
                    raw_data, colors_standard[0]
                )
            )

            retardation_new = pick_element_from_array(retardation, used)

            thin_section_thickness = (
                params.quartz_wedge_normalization.thin_section_thickness
            )

            return ComputationResult(
                **{
                    **params.__dict__,
                    "quartz_wedge_normalization": QuartzWedgeNormalization(
                        **{
                            **params.quartz_wedge_normalization.__dict__,
                            "quartz_wedge_path": path,
                            "raw_data": raw_data,
                            "calibrated_data": calibrated_data,
                            "retardation": retardation_new,
                            "summary_display": plot_retardation_color_chart(
                                calibrated_data, y_new, retardation_new
                            ),
                            "summary_rgb_display": plot_rgb_in_sensitive_color_plate(
                                calibrated_data, y_new, retardation_new
                            ),
                            # "convert_retardation_to_inclination": convert_retardation_to_inclination,
                            # "max_retardation_for_inclination_search": R,
                        }
                    ),
                }
            )

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"make_retardation_color_chart_with_quartz_wedge(): {e}")

    else:
        raise ValueError("no raw_data in QuartzWedgeNormalization")


# def estimate_optical_parameters(params: ComputationResult) -> ComputationResult:
#     if (
#         params.raw_maps is not None
#         and params.color_chart.xpl_retardation_color_chart is not None
#         and params.color_chart.xpl_retardation_array is not None
#         and params.color_chart.xpl_nd_filter_array is not None
#     ):
#         max_R_color_map = params.raw_maps["R_color_map"]
#         R_color_chart = params.color_chart.xpl_retardation_color_chart
#         R_array = params.color_chart.xpl_retardation_array
#         nd_array = params.color_chart.xpl_nd_filter_array
#     else:
#         return params

#     alpha, index, color_chart_used = estimate_median_alpha_of_nd_filter(
#         max_R_color_map, R_array, nd_array, R_color_chart
#     )

#     return ComputationResult(
#         **{
#             **params.__dict__,
#             "optical_parameters": OpticalParameters(
#                 **{
#                     **params.optical_parameters.__dict__,
#                     "alpha": alpha,
#                     "xpl_retardation_color_chart_used": color_chart_used,
#                     # "max_R": None,
#                     # "thickness": None,
#                 }
#             ),
#         }
#     )


# def add_theoritical_image(params: ComputationResult) -> ComputationResult:
#     ne = params.optical_parameters.ne
#     no = params.optical_parameters.no
#     max_R = params.optical_parameters.max_R
#     thickness = params.optical_parameters.thickness
#     tilt_deg = params.optical_parameters.tilt_deg
#     alpha = params.optical_parameters.alpha
#     color_chart_used = params.optical_parameters.xpl_retardation_color_chart_used

#     if (
#         params.raw_maps is not None
#         and params.color_chart.xpl_retardation_color_chart is not None
#         and params.color_chart.xpl_retardation_array is not None
#         and params.color_chart.xpl_nd_filter_array is not None
#         and color_chart_used is not None
#         and alpha is not None
#     ):
#         max_R_color_map = params.raw_maps["R_color_map"]
#         angle_map_0_to_90 = params.raw_maps["extinction_angle"]
#         R_array = params.color_chart.xpl_retardation_array

#         angles = D1FloatArray(np.linspace(0, np.pi / 2, num=40))
#         theoritical_image = make_theoritical_tilt_image(
#             max_R_color_map,
#             angle_map_0_to_90,
#             R_array,
#             angles,
#             no,
#             ne,
#             tilt_deg,
#             alpha,
#             color_chart_used,
#             thickness,
#             max_R,
#         )

#         return ComputationResult(
#             **{
#                 **params.__dict__,
#                 "raw_maps": {
#                     **params.raw_maps,
#                     "theoritical_image": theoritical_image,
#                 },
#             }
#         )
#     else:
#         return params


def grain_merge(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:

    code = params.grain_marge_code
    grain_list = params.grain_list
    grain_map = params.grain_map

    if grain_list is not None and code is not None and grain_map is not None:
        new_grain_map = merge_grain_by_code(grain_list, grain_map, code)
        new_grain_boundary = detect_boundaries(new_grain_map)

        return ComputationResult(
            **{
                **params.__dict__,
                "grain_map": new_grain_map,
                "grain_boundary": new_grain_boundary,
            }
        )
    else:
        raise ValueError(
            "grain_map, grain_list or grain_classification_code is not found"
        )


def add_random_colors_to_user_code(params: ComputationResult) -> ComputationResult:
    code = params.grain_classification_code
    if code is not None:
        return ComputationResult(
            **{
                **params.__dict__,
                "grain_classification_code": add_random_colors_to_input(code),
            }
        )
    else:
        return params


def grain_segmentation(
    params: ComputationResult,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ComputationResult:

    code = params.grain_classification_code
    grain_list = params.grain_list
    grain_map = params.grain_map

    if grain_list is not None and code is not None and grain_map is not None:

        index_dict, grain_list_new = select_grain_by_code(grain_list, code)

        # 色付けされたマップと凡例の生成
        segmented_result, legend = create_colored_map(grain_map, index_dict)

        return ComputationResult(
            **{
                **params.__dict__,
                "grain_classification_result": index_dict,
                "grain_classification_image": segmented_result,
                "grain_classification_legend": legend,
                "grain_list": grain_list_new,
            }
        )
    else:
        raise ValueError(
            "grain_map, grain_list or grain_classification_code is not found"
        )


if __name__ == "__main__":
    # get_retardation_color_chart_with_nd_filter(end=1000, num=100, nd_num=20)

    import pandas as pd

    r: ComputationResult = pd.read_pickle("../test/data/output/tetori_4k.pkl")
    # r = estimate_optical_parameters(r)
    # r = add_theoritical_image(r)

    r.color_chart.xpl_max_retardation = 1000
    r.color_chart.xpl_min_nd_filter = 0.2

    r = make_retardation_color_chart(r)
    r = find_image_center(r)
    # plt.imshow(r.raw_maps["theoritical_image"]["color_0"])
    # %%

    grain_detection_parameters = GrainDetectionParameters(
        smallest_grain_size=10,
        extinction_color_is_less_than=50,
        median_filter_size=3,
        shortest_contour=20,
        percentile=50,
        morphological_transformations_kernel_size=5,
        grain_boundary_logic="sobel",
    )

    r.grain_detection_parameters = grain_detection_parameters

    r = make_grain_boundary(r)
    r = analyze_grain_list(r)

    r.grain_classification_code = """
        // comment test
        quartz[gray]: R < 400 // comment
        mica[green]: R > 700 // this is comment
        grt[#991111]: R < 200 and size > 1000 // comment
        opaque[#444444]: R < 200 and size > 2000

        background [black]: index == 0
    """
    # mask = detect_boundaries()

    r = grain_segmentation(r)
    if r.grain_classification_image is not None:
        plt.imshow(r.grain_classification_image)

        r.grain_classification_legend

    r.grain_marge_code = "dist(R_1, R_2) < 10"
    r = grain_merge(r)

    # %%
