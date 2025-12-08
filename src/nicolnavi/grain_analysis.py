# %%
from scipy.spatial.distance import cdist, euclidean
import numpy as np
from numpy.typing import NDArray
from typing import (
    Literal,
    cast,
    Callable,
    Optional,
    TypeVar,
    overload,
)
import numpy as np
from niconavi.tools.type import (
    D1BoolArray,
    D2BoolArray,
    D2FloatArray,
    D1FloatArray,
    D2IntArray,
    D1IntArray,
)
from niconavi.type import (
    Grain,
    create_Grain_type,
    GrainAcceptedLiteral,
    RawMaps,
    TiltImageInfo,
)
from niconavi.tools.shape import compute_ellipse_params
from niconavi.image.image import resize_img, create_outside_circle_mask
from niconavi.image.types_operation import (
    is_MonoColorPicture,
    is_RGBPicture,
    is_RGBAPicture,
)
from niconavi.image.type import (
    Color,
    RGBPicture,
    _CommonPictureType,
    RGBAPicture,
    D1RGB_Array,
    D1RGBA_Array,
)
from niconavi.tools.change_type import as_float64, as_uint8
from niconavi.statistics.statistics import get_inscribed_circle_center
from niconavi.tools.array import (
    compress_array_by_zero_component,
    reconstruct_array_by_compressed_array,
)
from niconavi.tools.func_tools import synthesize_all_fn_in_array
from niconavi.tools.change_type import as_two_element_tuple
from niconavi.statistics.array_to_float import (
    median_with_nan,
    mean_with_nan,
    get_middle_val,
    circular_median,
    get_True_len,
    percentile_70_with_nan,
    percentile_75_with_nan,
    percentile_80_with_nan,
    percentile_90_with_nan,
    sd_extinction_angle_with_nan,
    sd_azimuth_with_nan,
)
from niconavi.image.image import get_dominant_color
from niconavi.grain_detection import assign_random_rgb
from niconavi.optics.color import show_color, convert_rgb_to_hsv

# from niconavi.retardation_normalization import make_retardation_estimation_function
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from matplotlib.pyplot import Figure, Axes


__all__ = [
    "summarize_information_from_label",
    "get_summarize_information_from_label_list",
]


def get_one_grain_area_from_label_index(
    grain_map: D2IntArray, label: int
) -> D2BoolArray:
    return cast(D2BoolArray, grain_map == label)


def get_adjacent_labels(
    labeled_array: D2IntArray,
) -> Callable[[int], D1IntArray]:
    def wrap(label: int) -> D1IntArray:
        # 対象ラベルのマスクを作成
        mask = (labeled_array == label).astype(np.uint8)

        # 3x3のカーネルを定義
        # kernel = np.ones((3, 3), np.uint8)
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

        # マスクを膨張
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # 隣接マスクを計算
        neighbor_mask = cv2.subtract(dilated_mask, mask)

        # 隣接する要素のラベルを取得
        neighbor_labels = labeled_array[neighbor_mask == 1]

        # ユニークなラベルを取得し、対象ラベルを除外
        adjacent_labels = np.unique(neighbor_labels)
        adjacent_labels = adjacent_labels[adjacent_labels != label]

        return adjacent_labels

    return wrap


def summarize_information_from_label(
    grain_map: D2IntArray,
) -> Callable[[int], Grain]:

    at_lim_labels = get_adjacent_labels(grain_map)(0)

    def wrap(label: int) -> Grain:
        one_grain_area = get_one_grain_area_from_label_index(grain_map, label)

        inscribed_radius, (centroid_y, centroid_x) = get_inscribed_circle_center(one_grain_area)
        area_size = np.sum(one_grain_area.astype(np.int_))
        area_shape, top_left_index = compress_array_by_zero_component(one_grain_area)
        contours_result = cv2.findContours(
            one_grain_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
        if len(contours) == 0:
            perimeter = 0.0
        else:
            perimeter = float(
                cv2.arcLength(max(contours, key=cv2.contourArea), True)
            )
        equivalent_radius = float(np.sqrt(area_size / (2.0 * np.pi)))

        return create_Grain_type(
            index=label,
            inscribed_radius=float(inscribed_radius),
            equivalent_radius=equivalent_radius,
            centroid=(centroid_x, centroid_y),
            size=int(area_size),
            perimeter=perimeter,
            area_shape=area_shape,
            top_left_index=top_left_index,
            original_shape=as_two_element_tuple(grain_map.shape),
            at_lim=label in set(at_lim_labels),
        )

    return wrap


def get_summarize_information_from_label_list(grain_map: D2IntArray) -> list[Grain]:
    unique_labels = np.unique(grain_map)
    grain_summary = list(
        map(summarize_information_from_label(grain_map), unique_labels)  # type: ignore
    )
    return grain_summary


def convert_non_transparent_pixel_in_img_to_1D_array(
    img: RGBPicture | RGBAPicture,
) -> D1RGB_Array:
    if is_RGBAPicture(img):
        r_img = img.reshape(-1, 4)
        f_img = filter(lambda x: x[3] != 0, r_img)
        res = np.array(list(map(lambda x: x[0:3], f_img)), dtype=np.uint8)
        return cast(D1RGB_Array, res)
    elif is_RGBPicture(img):
        r_img = img.reshape(-1, 3)
        return cast(D1RGB_Array, r_img)
    else:
        raise TypeError("img must RGBPicuture or RGBAPicture")


def reconstruct_grain_mask(grain: Grain) -> D2BoolArray:
    area_shape = grain["area_shape"]
    top_left_index = grain["top_left_index"]
    shape = grain["original_shape"]

    mask = reconstruct_array_by_compressed_array(
        area_shape, top_left_index, as_two_element_tuple(shape)
    )

    return mask


def crop_image_by_mask(rgb_image: RGBPicture, mask: D2BoolArray) -> RGBPicture:
    output = np.zeros_like(rgb_image)
    output[mask] = rgb_image[mask]

    return output


T = TypeVar("T")


@overload
def apply_grain_mask(
    applied_map: RGBAPicture,
    logic_fn: Callable[[D1RGBA_Array], T],
    key: GrainAcceptedLiteral,
    method: Literal["normal", "tiny_circ"] = "normal",
    circ_threshold: Optional[float] = None,
) -> Callable[[Grain], Grain]: ...
@overload
def apply_grain_mask(
    applied_map: RGBPicture,
    logic_fn: Callable[[D1RGB_Array], T],
    key: GrainAcceptedLiteral,
    method: Literal["normal", "tiny_circ"] = "normal",
    circ_threshold: Optional[float] = None,
) -> Callable[[Grain], Grain]: ...
@overload
def apply_grain_mask(
    applied_map: D2FloatArray,
    logic_fn: Callable[[D1FloatArray], T],
    key: GrainAcceptedLiteral,
    method: Literal["normal", "tiny_circ"] = "normal",
    circ_threshold: Optional[float] = None,
) -> Callable[[Grain], Grain]: ...
@overload
def apply_grain_mask(
    applied_map: D2IntArray,
    logic_fn: Callable[[D1IntArray], T],
    key: GrainAcceptedLiteral,
    method: Literal["normal", "tiny_circ"] = "normal",
    circ_threshold: Optional[float] = None,
) -> Callable[[Grain], Grain]: ...
@overload
def apply_grain_mask(
    applied_map: D2BoolArray,
    logic_fn: Callable[[D1BoolArray], T],
    key: GrainAcceptedLiteral,
    method: Literal["normal", "tiny_circ"] = "normal",
    circ_threshold: Optional[float] = None,
) -> Callable[[Grain], Grain]: ...


def apply_grain_mask(  # type: ignore
    applied_map,
    logic_fn,
    key: GrainAcceptedLiteral,
    method: Literal["normal", "tiny_circ"] = "normal",
    circ_threshold: Optional[float] = None,
):
    if method == "normal":

        def closure(grain: Grain):  # type: ignore
            mask = reconstruct_grain_mask(grain)

            d1_img = applied_map[mask]

            c = logic_fn(d1_img)
            grain[key] = c

            return grain

    if (
        method == "tiny_circ"
    ):  # 粒子に含まれるもっとも大きい円の半径Rに対して、circ_thereshold*rの円の内部を調査する
        circ_threshold_used = 1 if circ_threshold is None else circ_threshold

        def closure(grain: Grain):  # type: ignore

            center_idx = grain["centroid"]
            radius_value = grain.get("inscribed_radius")
            r = (radius_value if radius_value is not None else 0.0) * circ_threshold_used
            r_used = r if r > 1 else 1
            shape = applied_map.shape[:2]
            # np.ogridを使用して行番号と列番号のグリッドを生成（メモリ効率が良い）
            y, x = np.ogrid[: shape[0], : shape[1]]
            # 各画素の中心から中心との距離の二乗を計算し、radius**2以下ならTrue
            mask = ((y - center_idx[1]) ** 2 + (x - center_idx[0]) ** 2) <= r_used**2
            # plt.imshow(mask)
            # plt.show()

            d1_img = applied_map[mask]

            c = logic_fn(d1_img)
            # print(c)
            # if c is None:
            #     print(c)

            grain[key] = c

            return grain

    return closure


def add_additional_information_to_grain_list(grain_list: list[Grain]) -> list[Grain]:
    def update_grain(x: Grain) -> Grain:
        if x["R_color"] is not None and x["extinction_color"] is not None:

            (
                eccentricity,
                angle_deg,
                major_axis_length,
                minor_axis_length,
                (center_x, center_y),
            ) = compute_ellipse_params(reconstruct_grain_mask(x))
            eccentricity_mod = (
                eccentricity if 0 <= eccentricity <= 1 else 1 if eccentricity > 1 else 0
            )

            hsv_col = convert_rgb_to_hsv(x["R_color"])

            return {
                **x,
                "H": hsv_col[0],
                "S": hsv_col[1],
                "V": hsv_col[2],
                "eccentricity": eccentricity_mod,
                "angle_deg": angle_deg,
                "major_axis_length": major_axis_length,
                "minor_axis_length": minor_axis_length,
                "ellipse_center": (center_x, center_y),
                "size": x["size"],
            }

        else:
            raise ValueError("R is not exist in Grain")

    return list(map(update_grain, grain_list))


# def add_retardation_to_grain_list_v2(grain_list: list[Grain]) -> list[Grain]:

#     get_retardation_from_color = make_retardation_estimation_function()  # type: ignore

#     def update_grain(x: Grain) -> Grain:

#         if x["R_color"] is not None and x["extinction_color"] is not None:
#             return {
#                 **x,
#                 "R": get_retardation_from_color(x["R_color"]),
#                 "min_retardation": get_retardation_from_color(x["extinction_color"]),
#                 "pR": (
#                     get_retardation_from_color(x["p45_color"])
#                     if x["p45_color"] is not None
#                     else None
#                 ),
#                 "mR": (
#                     get_retardation_from_color(x["m45_color"])
#                     if x["m45_color"] is not None
#                     else None
#                 ),
#                 "max_retardation_estimated_for_inclination": (
#                     get_retardation_from_color(x["R_color"])
#                     if get_retardation_from_color is not None
#                     else None
#                 ),
#             }

#         else:
#             raise ValueError("R is not exist in Grain")

#     return list(map(update_grain, grain_list))


def add_azimuth_to_grain_list(
    grain_list: list[Grain],
    rotation_direction: Literal["clockwise", "counterclockwise"] = "clockwise",
) -> list[Grain]:
    if rotation_direction == "clockwise":

        def update_grain(x: Grain) -> Grain:
            p = x["pR"]
            m = x["mR"]
            e = x["extinction_angle"]

            if p is not None and m is not None and e is not None:
                if p > m:
                    return {**x, "azimuth": e}
                else:
                    return {**x, "azimuth": e + 90}
            else:
                return x

        return list(map(lambda x: update_grain(x), grain_list))
    elif rotation_direction == "counterclockwise":

        def update_grain(x: Grain) -> Grain:
            p = x["pR"]
            m = x["mR"]
            e = x["extinction_angle"]

            if p is not None and m is not None and e is not None:
                if p > m:
                    return {**x, "azimuth": e + 90}
                else:
                    return {**x, "azimuth": e}
            else:
                return x

        return list(map(lambda x: update_grain(x), grain_list))
    else:
        raise ValueError("rotation_direction should counterclockwise or clockwise")


def clip_as_color(arr: D1FloatArray) -> Color:
    if arr.shape != (3,):
        raise ValueError("arr must 3 dimensional vector")
    c = np.clip(arr, 0, 255)
    return cast(Color, c.astype(np.uint8))


def geometric_median(X: D1RGB_Array, eps: float = 1e-5) -> Color:
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return clip_as_color(y)
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return clip_as_color(y1)

        y = y1


def get_most_saturated_color(arr: D1RGB_Array) -> Color:
    """
    BGR値の配列 (n, 3) から、もっとも彩度(S)が高い色(B, G, R)を返す関数。
    bgr_array: shape (n, 3)、各要素は[0, 255]の範囲を想定(例: OpenCVでの色表現)。

    戻り値: (B, G, R) いずれも int, [0, 255] の範囲。
    """
    # (n, 3) -> (n, 1, 3) に reshape して、cv2.cvtColorでHSV変換できる形にする
    arr_reshaped = arr.reshape(-1, 1, 3).astype(np.uint8)

    # BGR -> HSV (H:0~179, S:0~255, V:0~255)
    hsv_reshaped = cv2.cvtColor(arr_reshaped, cv2.COLOR_RGB2HSV)

    # hsv_reshapedの形は (n, 1, 3)
    # 彩度は [H, S, V] のうちインデックス1が該当
    saturations = hsv_reshaped[:, 0, 1]

    # 彩度が最大となるインデックスを取得
    max_idx = np.argmax(saturations)

    # もとのBGR値を返す
    return arr[max_idx].astype(np.uint8)


# mapsからgrain_listの各要素を作成する関数
def analyze_grain(
    maps: RawMaps,
    grain_map: D2IntArray,
    circ_threshold: float,
) -> list[Grain]:

    if maps["max_retardation_map"] is not None:
        apply_all = synthesize_all_fn_in_array(
            [
                apply_grain_mask(
                    maps["R_color_map"],
                    geometric_median,
                    key="R_color",
                ),
                apply_grain_mask(
                    maps["cv_extinction_angle"],
                    mean_with_nan,
                    key="exQuality",
                ),
                apply_grain_mask(
                    maps["max_retardation_map"],
                    median_with_nan,
                    key="R",
                    method="tiny_circ",
                    circ_threshold=circ_threshold,
                ),
                apply_grain_mask(
                    maps["max_retardation_map"],
                    percentile_70_with_nan,
                    key="R70",
                ),
                apply_grain_mask(
                    maps["max_retardation_map"],
                    percentile_80_with_nan,
                    key="R80",
                ),
                apply_grain_mask(
                    maps["max_retardation_map"],
                    percentile_90_with_nan,
                    key="R90",
                ),
                apply_grain_mask(
                    maps["extinction_angle"],
                    circular_median(0, 90, 90),
                    # median_with_nan,
                    key="extinction_angle",
                    method="tiny_circ",
                    circ_threshold=circ_threshold,
                ),
                apply_grain_mask(
                    maps["extinction_color_map"],
                    get_dominant_color,
                    key="extinction_color",
                    method="tiny_circ",
                    circ_threshold=circ_threshold,
                ),
            ]
        )
    else:
        Warning("this line should not exec")
        apply_all = lambda x: x

    grain_list = [
        apply_all(grain)
        for grain in get_summarize_information_from_label_list(grain_map)
    ]

    if (
        maps["m45_R_color_map"] is not None
        and maps["p45_R_color_map"] is not None
        and maps["p45_R_map"] is not None
        and maps["m45_R_map"] is not None
        and maps["azimuth"] is not None
    ):

        reta_fn = synthesize_all_fn_in_array(
            [
                apply_grain_mask(
                    maps["p45_R_map"],
                    median_with_nan,
                    key="pR",
                ),
                apply_grain_mask(
                    maps["m45_R_map"],
                    median_with_nan,
                    key="mR",
                ),
                apply_grain_mask(
                    maps["p45_R_map"],
                    percentile_75_with_nan,
                    key="pR75",
                ),
                apply_grain_mask(
                    maps["azimuth"],
                    circular_median(0.0, 180.0, 180),
                    key="azimuth",
                    method="tiny_circ",
                    circ_threshold=circ_threshold,
                ),
                apply_grain_mask(
                    maps["azimuth"],
                    sd_azimuth_with_nan,
                    key="sd_azimuth",
                ),
                apply_grain_mask(
                    maps["extinction_angle"],
                    sd_extinction_angle_with_nan,
                    key="sd_extinction_angle",
                    method="tiny_circ",
                    circ_threshold=circ_threshold,
                ),
                apply_grain_mask(maps["m45_R_map"], percentile_75_with_nan, key="mR75"),
                apply_grain_mask(
                    maps["p45_R_color_map"],
                    get_dominant_color,
                    key="p45_color",
                ),
                apply_grain_mask(
                    maps["m45_R_color_map"],
                    get_dominant_color,
                    key="m45_color",
                ),
            ]
        )

        grain_list = [reta_fn(grain) for grain in grain_list]

    return grain_list


def analyze_grain_for_CIP(
    grain_list: list[Grain],
    maps: RawMaps,
    circ_threshold: float,
    tilt_image_info: Optional[TiltImageInfo] = None,
) -> list[Grain]:
    if maps["inclination"] is not None and tilt_image_info is not None:
        # if tilt_image_info.tilt_image0 is None or tilt_image_info.tilt_image45 is None:
        if tilt_image_info.tilt_image0 is None:
            mask = ~create_outside_circle_mask(maps["inclination"])
        else:
            if tilt_image_info.tilt_image45 is not None:
                mask = D2BoolArray(
                    ~(
                        tilt_image_info.tilt_image0["image_mask"]
                        & tilt_image_info.tilt_image45["image_mask"]
                    )
                )
            else:
                mask = D2BoolArray(
                    ~(
                        tilt_image_info.tilt_image0["image_mask"]
                        # & tilt_image_info.tilt_image45["image_mask"]
                    )
                )

        fn = synthesize_all_fn_in_array(
            [
                apply_grain_mask(
                    maps["inclination"],
                    median_with_nan,
                    key="inclination",
                    method="tiny_circ",
                    circ_threshold=circ_threshold,
                ),
            ]
        )
        grain_list = [fn(grain) for grain in grain_list]
        # grain_list = list(filter(lambda x: x["inclination"] is not None, grain_list))
        if maps["azimuth360"] is None:
            return grain_list
        else:
            maps["azimuth360"][mask] = np.nan

            fn2 = synthesize_all_fn_in_array(
                [
                    apply_grain_mask(
                        maps["azimuth360"],
                        circular_median(0.0, 360.0, 360),
                        key="azimuth360",
                        method="tiny_circ",
                        circ_threshold=circ_threshold,
                    ),
                    apply_grain_mask(
                        D2BoolArray(mask),
                        get_True_len,
                        key="len_in_mask_pixel",
                    ),
                ]
            )
        grain_list = [fn2(grain) for grain in grain_list]

        # print("---------")
        # print(list(map(lambda x: x["azimuth360"], grain_list)))
        # print("---------")
        return grain_list
    else:
        raise ValueError("inclination is None.")


def plot_grain_map_with_index(
    grain_map: D2IntArray, grain_list: list[Grain], fontsize: int = 4
) -> tuple[Figure, Axes]:
    # for i in np.unique(grain_map):
    #     grain_map[i]

    fig, ax = plt.subplots()
    ax.imshow(assign_random_rgb(grain_map, use_color=[(0, (0, 0, 0))]))

    # 各値に対して、その領域（同一値からなる画素群）をマスクとして重心を求め、表示
    for grain in grain_list:
        cx, cy = grain["centroid"]
        val = grain["index"]
        ax.text(
            cx, cy, str(val), color="white", ha="center", va="center", fontsize=fontsize
        )

    return fig, ax


if __name__ == "__main__":
    # (

    # ) # type: ignore

    import pandas as pd
    from niconavi.type import GrainDetectionParameters, ComputationResult
    import numpy as np
    import matplotlib.pyplot as plt
    from niconavi.grain_analysis import assign_random_rgb

    r: ComputationResult = pd.read_pickle(
        "./../test/data/output/tetori_4k_xpl_pol_til10_class_inc.pkl"
    )

    # k = analyze_grain(r.raw_maps, r.grain_map)

    r.grain_list[1]
    x = list(map(lambda x: x["centroid"][0], r.grain_list))
    y = list(map(lambda x: x["centroid"][1], r.grain_list))

    plt.imshow(assign_random_rgb(r.grain_map, use_color=[(0, (0, 0, 0))]))
    plt.scatter(x, y, s=10, marker="+")

    rgba_image_no_color = np.array(
        [
            [[255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0]],
            [[255, 255, 0, 0], [0, 255, 255, 0], [255, 0, 255, 0]],
            [[128, 128, 128, 0], [64, 64, 64, 0], [192, 192, 192, 0]],
        ],
        dtype=np.uint8,
    )
    rgba_image_full_color = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 128], [0, 0, 255, 1]],
            [[255, 255, 0, 255], [0, 255, 255, 128], [255, 0, 255, 2]],
            [[128, 128, 128, 255], [64, 64, 64, 128], [192, 192, 192, 3]],
        ],
        dtype=np.uint8,
    )
    rgba_image = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 128], [0, 0, 255, 0]],
            [[255, 255, 0, 255], [0, 255, 255, 128], [255, 0, 255, 0]],
            [[128, 128, 128, 255], [64, 64, 64, 128], [192, 192, 192, 0]],
        ],
        dtype=np.uint8,
    )

    color_chart = np.load("../data/optics/color_plate.npy")
    color_chart = color_chart[20:, :300, :]
    color_chart_retardation = np.load("../data/optics/color_plate_retardation.npy")[
        :300
    ]
    plt.imshow(np.array([color_chart[-1, :]]), aspect=50)
    plt.imshow(np.array(color_chart))

    # plt.imshow(mat)
    # plt.scatter(index[1], index[0])

    # plt.title("Center of Mass for Each Value Region")
    # plt.show()
    # add_retardation_to_grain_list(grain_map)

    # grain_map = estimate_grain_map(maps["R_color_map"], 9, 7, 40)
    # plt.imshow(maps["R_color_map"])

    # p = get_key_map_from_grain_list(grain_list, "max_retardationr")
    # plt.imshow(p)
    # plt.colorbar()

    # plt.imshow(p)
    # plt.imshow(pics[0])
    # plt.savefig("../pic/grain_detection/degree_0.pdf")
    # plt.imshow(pics[13])
    # plt.savefig("../pic/grain_detection/degree_45.pdf")
    # plt.imshow(pics[13])
