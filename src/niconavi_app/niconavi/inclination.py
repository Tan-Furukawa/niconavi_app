# %%
import numpy as np
from niconavi_app.niconavi.type import (
    TiltDirectionEstimationResult,
    ComputationResult,
    TiltImageResult,
    ColorChartInfo,
)
from niconavi_app.niconavi.tilt_image import estimate_tilted_image
from typing import Callable, Optional
from niconavi_app.niconavi.image.type import RGBPicture
from niconavi_app.niconavi.image.tools import extract_h_map
from niconavi_app.niconavi.tools.type import D2FloatArray
from niconavi_app.niconavi.optics.uniaxial_plate import calc_color_chart
from niconavi_app.niconavi.retardation_normalization import (
    make_angle_retardation_estimation_function,
    make_retardation_color_map,
    get_thickness_from_max_retardation,
)
from niconavi_app.niconavi.make_map import add_color_V_to_img
import numpy as np
import matplotlib.pyplot as plt
from niconavi_app.app_config import ENABLE_TILT_REF_MEDIAN_CORRECTION
from niconavi_app.niconavi.tools.type import (
    D2FloatArray,
    D2BoolArray,
    D1FloatArray,
)
from niconavi_app.niconavi.image.image import create_outside_circle_mask, median_filter
from niconavi_app.niconavi.optics.uniaxial_plate import (
    get_spectral_distribution,
    ColorChartInfo,
)
from niconavi_app.niconavi.optics.optical_system import (
    get_full_wave_plus_mineral_retardation_system,
)
from copy import deepcopy
import cv2


TILT_RED_DELTA_TOLERANCE = 0.0
TILT_MAJORITY_KERNEL_SIZE = 3
TILT_MAJORITY_ITERATIONS = 3


def _scale_value_channel(
    value: np.ndarray,
    *,
    source_median: float,
    target_median: float,
) -> np.ndarray:
    if source_median <= 0:
        return value
    return np.clip(value * target_median / source_median, 0, 255)


def normalize_tilt_pair_brightness(
    original_image: RGBPicture,
    tilted_image: RGBPicture,
) -> tuple[RGBPicture, RGBPicture]:
    original = np.asarray(np.clip(original_image, 0, 255), dtype=np.uint8)
    tilted = np.asarray(np.clip(tilted_image, 0, 255), dtype=np.uint8)
    if not ENABLE_TILT_REF_MEDIAN_CORRECTION:
        return RGBPicture(original), RGBPicture(tilted)

    valid_mask = ~create_outside_circle_mask(original)

    original_hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float64)
    tilted_hsv = cv2.cvtColor(tilted, cv2.COLOR_RGB2HSV).astype(np.float64)

    original_median_v = float(np.median(original_hsv[:, :, 2][valid_mask]))
    tilted_median_v = float(np.median(tilted_hsv[:, :, 2][valid_mask]))
    target_median_v = (original_median_v + tilted_median_v) / 2.0

    original_hsv[:, :, 2] = _scale_value_channel(
        original_hsv[:, :, 2],
        source_median=original_median_v,
        target_median=target_median_v,
    )
    tilted_hsv[:, :, 2] = _scale_value_channel(
        tilted_hsv[:, :, 2],
        source_median=tilted_median_v,
        target_median=target_median_v,
    )

    return (
        RGBPicture(cv2.cvtColor(np.clip(original_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)),
        RGBPicture(cv2.cvtColor(np.clip(tilted_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)),
    )


def fill_unresolved_by_nearest_confident_decision(
    decision: np.ndarray,
    filled: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    if not np.any(filled & valid_mask):
        return decision
    nearest_indices = distance_transform_edt(
        ~(filled & valid_mask),
        return_distances=False,
        return_indices=True,
    )
    unresolved = valid_mask & ~filled
    filled_decision = decision.copy()
    filled_decision[unresolved] = filled_decision[
        tuple(index[unresolved] for index in nearest_indices)
    ]
    return filled_decision


def regularize_increase_mask(
    delta_value: np.ndarray,
    *,
    tolerance: float = TILT_RED_DELTA_TOLERANCE,
    kernel_size: int = TILT_MAJORITY_KERNEL_SIZE,
    iterations: int = TILT_MAJORITY_ITERATIONS,
) -> D2BoolArray:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3.")

    valid_mask = ~create_outside_circle_mask(delta_value)
    confident = valid_mask & (np.abs(delta_value) > tolerance)
    decision = delta_value > 0
    filled = confident.copy()
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    for _ in range(iterations):
        unresolved = valid_mask & ~filled
        if not np.any(unresolved):
            break
        positive_count = cv2.filter2D(
            (filled & decision).astype(np.float32),
            ddepth=-1,
            kernel=kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        negative_count = cv2.filter2D(
            (filled & ~decision).astype(np.float32),
            ddepth=-1,
            kernel=kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        has_majority = unresolved & (positive_count != negative_count)
        if not np.any(has_majority):
            break
        decision[has_majority] = positive_count[has_majority] > negative_count[has_majority]
        filled[has_majority] = True

    if np.any(valid_mask & ~filled):
        decision = fill_unresolved_by_nearest_confident_decision(
            decision,
            filled,
            valid_mask,
        )
    return D2BoolArray(decision)


def make_red_increase_mask_from_tilt_pair(tilt_image_info: TiltImageResult) -> D2BoolArray:
    original_image, tilted_image = normalize_tilt_pair_brightness(
        tilt_image_info["original_image"],
        tilt_image_info["focused_tilted_image"],
    )
    original_red = original_image[:, :, 0].astype(np.float64)
    tilted_red = tilted_image[:, :, 0].astype(np.float64)
    return regularize_increase_mask(
        tilted_red - original_red,
        tolerance=TILT_RED_DELTA_TOLERANCE,
    )


def make_azimuth_vs_R_full_wave_color_chart(
    num_R: int,
    num_azimuth: int,
    max_R: float = 500,
    min_azimuth: float = 0,
    max_azimuth: float = 180,
    alpha: float = 1,
) -> ColorChartInfo:

    R = D1FloatArray(np.linspace(0, max_R, num_R))
    azimuth = D1FloatArray(
        np.linspace(np.radians(min_azimuth), np.radians(max_azimuth), num_azimuth)
    )

    fn = lambda w, h: get_spectral_distribution(
        get_full_wave_plus_mineral_retardation_system(R=h, azimuth=w, alpha=alpha)
    )["rgb"]

    col_chart = calc_color_chart(
        azimuth,
        R,
        fn,
    )

    col_chart["h"] = R
    col_chart["w"] = azimuth

    return col_chart


def identify_equiv(mat: D2FloatArray, a: float, b: float) -> D2FloatArray:
    """
    任意の実数を要素に含む NumPy 配列 mat の各要素 x を
    区間 [a, b) 上の代表元 ( (x - a) mod (b - a) ) + a に写す関数。

    Parameters
    ----------
    mat : np.ndarray
        任意次元の NumPy 配列（要素は実数）
    a : float
        同一視の始まりとなる実数
    b : float
        同一視の終わりとなる実数 (a < b を仮定)

    Returns
    -------
    np.ndarray
        同一視を行った結果の NumPy 配列
    """
    return D2FloatArray((mat - a) % (b - a) + a)


def get_inclination_map(
    R_map: D2FloatArray, no: float, ne: float, thickness: float
) -> D2FloatArray:
    theta_to_R, R_to_theta = make_angle_retardation_estimation_function(
        no=no, ne=ne, thickness=thickness
    )
    return D2FloatArray(np.degrees(R_to_theta(R_map)))


def make_R_map_of_full_wave_plate(
    azimuth_map: D2FloatArray,
    img: RGBPicture,
    R_vs_azimuth: ColorChartInfo,
    full_wave_plate: float = 530,
) -> D2FloatArray:

    azimuth_map_0_to_90 = np.zeros_like(azimuth_map)
    azimuth_map_0_to_90[azimuth_map <= 90] = azimuth_map[azimuth_map <= 90]
    azimuth_array_0_to_90 = R_vs_azimuth["w"][R_vs_azimuth["w"] <= 90]
    color_chart_0_to_90 = R_vs_azimuth["color_chart"][:, R_vs_azimuth["w"] <= 90]

    azimuth_map_90_to_180 = np.zeros_like(azimuth_map)
    azimuth_map_90_to_180[azimuth_map > 90] = azimuth_map[azimuth_map > 90]
    azimuth_array_90_to_180 = R_vs_azimuth["w"][R_vs_azimuth["w"] > 90]
    color_chart_90_to_180 = R_vs_azimuth["color_chart"][:, R_vs_azimuth["w"] > 90]

    im_R_0_to_90 = extract_h_map(
        img,
        azimuth_map_0_to_90,
        RGBPicture(color_chart_0_to_90),
        D1FloatArray(R_vs_azimuth["h"] + full_wave_plate),
        D1FloatArray(azimuth_array_0_to_90),
    )

    # plt.imshow(azimuth_map < 90)
    # plt.colorbar()
    # plt.show()

    im_R_90_to_180 = extract_h_map(
        img,
        azimuth_map_90_to_180,
        RGBPicture(color_chart_90_to_180),
        D1FloatArray(R_vs_azimuth["h"] + full_wave_plate),
        D1FloatArray(azimuth_array_90_to_180),
    )

    im_R = np.zeros_like(im_R_0_to_90)
    im_R[azimuth_map <= 90] = im_R_0_to_90[azimuth_map <= 90]
    im_R[azimuth_map > 90] = im_R_90_to_180[azimuth_map > 90]

    return im_R


def make_R_vs_azimuth_color_chart(
    alpha: float,
    max_R_without_full_wave: float = 500,
    full_wave_plate: float = 530,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ColorChartInfo:
    R_array = D1FloatArray(np.linspace(0, max_R_without_full_wave, 40))
    azimuth = D1FloatArray(np.linspace(0, 180, 40))

    R_vs_azimuth_color_chart = calc_color_chart(
        azimuth,
        R_array,
        lambda w, h: get_spectral_distribution(
            get_full_wave_plus_mineral_retardation_system(
                R=h, azimuth=np.radians(w), R0=full_wave_plate, alpha=alpha
            )
        )["rgb"],
        progress_callback=progress_callback,
    )
    # plt.imshow(R_vs_azimuth_color_chart["color_chart"])
    # plt.show()
    return R_vs_azimuth_color_chart


def estimate_tilt_direction(
    im_result0: TiltImageResult,
    R_vs_azimuth_color_chart: ColorChartInfo,
    azimuth_map: D2FloatArray,
    ex_angle_map: D2FloatArray,
    im_result45: Optional[TiltImageResult],
    angle_of_image45: Optional[float],
) -> TiltDirectionEstimationResult:

    # def estimate_tilt_direction(
    #     im_result0: TiltImageResult,
    #     R_vs_azimuth_color_chart: ColorChartInfo,
    #     azimuth_map: D2FloatArray,
    #     ex_angle_map: D2FloatArray,
    # ) -> TiltDirectionEstimationResult:

    R_im0 = make_R_map_of_full_wave_plate(
        azimuth_map, im_result0["original_image"], R_vs_azimuth_color_chart
    )

    R_im0_tilt = make_R_map_of_full_wave_plate(
        azimuth_map, im_result0["focused_tilted_image"], R_vs_azimuth_color_chart
    )

    if im_result45 is not None and angle_of_image45 is not None:
        R_im45 = make_R_map_of_full_wave_plate(
            identify_equiv(D2FloatArray(azimuth_map - angle_of_image45), 0, 180),
            im_result45["original_image"],
            R_vs_azimuth_color_chart,
        )
        R_im45_tilt = make_R_map_of_full_wave_plate(
            identify_equiv(D2FloatArray(azimuth_map - angle_of_image45), 0, 180),
            im_result45["focused_tilted_image"],
            R_vs_azimuth_color_chart,
        )
    else:
        R_im45 = None
        R_im45_tilt = None

    become_red = make_red_increase_mask_from_tilt_pair(im_result0)

    # res0 = R_im0_tilt - R_im0 > 0
    # res45 = R_im45_tilt - R_im45 > 0
    # inclination = np.degrees(R_to_theta(np.clip(R_map - 0, 0, 9999)))
    # plt.imshow(inclination < 27)
    # plt.colorbar()
    # plt.show()
    # tilt_direction_is_plus = np.zeros_like(res0)
    # tilt_direction_is_plus[np.bitwise_or(ex_angle_map < 22.5, ex_angle_map > 67.5)] = (
    #     res45[np.bitwise_or(ex_angle_map < 22.5, ex_angle_map > 67.5)]
    # )
    # tilt_direction_is_plus[
    #     np.bitwise_and(ex_angle_map >= 22.5, ex_angle_map <= 67.5)
    # ] = res0[np.bitwise_and(ex_angle_map >= 22.5, ex_angle_map <= 67.5)]

    return TiltDirectionEstimationResult(
        R_im0=R_im0,
        R_im0_tilt=R_im0_tilt,
        R_im45=R_im45,
        R_im45_tilt=R_im45_tilt,
        become_red_by_tilt=become_red,
    )


def convert_tilt_to_tilt_0_to_180(
    params: ComputationResult,
    thickness: float,
    # thin_section_azimuth_angle: float,
    # azimuth: D2FloatArray,
    # inclination: D2FloatArray,
    # median_kernel_size: int = 1,
    # become_red: D2BoolArray,
) -> Optional[D2FloatArray]:

    if (
        params.raw_maps is not None
        and params.color_chart.xpl_retardation_color_chart is not None
        and params.color_chart.xpl_R_array is not None
        and params.tilt_image_info.tilt_image0 is not None
        and params.raw_maps["azimuth"] is not None
    ):

        theta_to_R, R_to_theta = make_angle_retardation_estimation_function(
            no=params.optical_parameters.no,
            ne=params.optical_parameters.ne,
            thickness=thickness,
            # get_thickness_from_max_retardation(
            #     params.optical_parameters.max_R
            # ),
        )

        v = 0 #! 一旦傾きを求めるときのinclination補正は無効にする
        R_map, _, _ = make_retardation_color_map(
            img=add_color_V_to_img(params.raw_maps["R_color_map"], -v),
            color_chart=params.color_chart.xpl_retardation_color_chart,
            color_chart_reta=params.color_chart.xpl_R_array,
        )
        R_map = D2FloatArray(np.clip(R_map, 1, 9999))
        # --------------------------
        # Φ=0のとき(Φはステージの回転角)
        # --------------------------
        become_red = make_red_increase_mask_from_tilt_pair(
            params.tilt_image_info.tilt_image0
        )
        inclination = D2FloatArray(np.degrees(R_to_theta(R_map)))
        inclination_0_to_180_0 = deepcopy(inclination)
        inclination_0_to_180_0[~become_red] = (180 - inclination)[~become_red]
        # inclination_0_to_180_0[become_red] = (180 - inclination)[become_red]

        # --------------------------
        # Φ=45のとき(Φはステージの回転角)
        # --------------------------
        if params.tilt_image_info.tilt_image45 is not None:
            become_red_45 = make_red_increase_mask_from_tilt_pair(
                params.tilt_image_info.tilt_image45
            )

            inclination_0_to_180_45 = deepcopy(inclination)

            condition1 = ~become_red_45 & (params.raw_maps["azimuth"] <= 90)
            condition2 = become_red_45 & (params.raw_maps["azimuth"] >= 90)
            inclination_0_to_180_45[condition1] = (180 - inclination)[condition1]
            inclination_0_to_180_45[condition2] = (180 - inclination)[condition2]

            # condition1 = ~become_red_45 & (azimuth > 90)
            # condition2 = become_red_45 & (azimuth < 90)
            # inclination_0_to_180_45[condition1] = (180 - inclination)[condition1]
            # inclination_0_to_180_45[condition2] = (180 - inclination)[condition2]

            inclination_0_to_180 = deepcopy(inclination_0_to_180_45)
            inclination_0_to_180[
                (45 <= params.raw_maps["azimuth"]) & (params.raw_maps["azimuth"] < 135)
            ] = inclination_0_to_180_0[
                (45 <= params.raw_maps["azimuth"]) & (params.raw_maps["azimuth"] < 135)
            ]

        else:
            inclination_0_to_180 = inclination_0_to_180_0
        # inclination_0_to_180[~become_red] = (180 - inclination)[~become_red]

        return inclination_0_to_180
    else:
        return None


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    # r: ComputationResult = pd.read_pickle("../test/data/output/yamagami_class_inc.pkl")
    r: ComputationResult = pd.read_pickle(
        "../ebsd_adjustment/data_app/output_2025-05-21-17-49-00.niconavi"
    )

    alpha = r.color_chart.pol_lambda_alpha
    R_vs_azimuth_color_chart = make_R_vs_azimuth_color_chart(alpha)
