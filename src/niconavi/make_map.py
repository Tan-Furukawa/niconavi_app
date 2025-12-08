# %%

from niconavi.tools.read_data import divide_video_into_n_frame
from tqdm import tqdm
from copy import deepcopy
from niconavi.retardation_normalization import (
    make_retardation_estimation_function,
    make_retardation_color_map,
)
from niconavi.statistics.array_to_float import circular_median_data_via_grid
from typing import (
    overload,
    Optional,
    cast,
    TypeVar,
    Callable,
)
from niconavi.image.image import resize_img, create_outside_circle_mask
from niconavi.image.tools import apply_color_map, apply_2dcolor_map
import numpy as np
import matplotlib.pyplot as plt
from niconavi.tools.change_type import as_two_element_tuple
from niconavi.image.image import resize_img
from niconavi.statistics.statistics import gamma_mode_from_data
import cv2

from niconavi.tools.type import (
    D1,
    D2,
    D3,
    D1IntArray,
    D2IntArray,
    D2BoolArray,
    D1FloatArray,
    D2FloatArray,
)
from niconavi.type import RawMaps
from niconavi.image.type import RGBPicture, MonoColorPicture, D1RGB_Array
from niconavi.image.image import convert_to_gray_scale
from niconavi.tools.change_type import as_float64

__all__ = ["make_extinction_angle_map", "find_extinction_angle"]


def transpose_dat_shape(
    pics: list[MonoColorPicture],
) -> np.ndarray[D3, np.dtype[np.uint8]]:
    res = np.array(pics).transpose(1, 2, 0)
    return cast(np.ndarray[D3, np.dtype[np.uint8]], res)


K = TypeVar("K", bound=np.generic)
T = TypeVar("T")


def apply_along_axis(
    fn: Callable[[np.ndarray[D1, np.dtype[K]]], K],
    axis: int,
    mat: np.ndarray[D3, np.dtype[K]],
) -> np.ndarray[D2, np.dtype[K]]:
    return cast(np.ndarray[D2, np.dtype[K]], np.apply_along_axis(fn, 2, mat))  # type: ignore


def circular_mean_matrices(mats: list[D2FloatArray], n: float) -> D2FloatArray:
    """
    mats: list of 2D numpy arrays, all with the same shape
    n: maximum value (e.g., 360 if inputs are in degrees), 0 and n are identified
    return: a 2D numpy array with the circular mean applied elementwise
    """
    mats = np.array(mats)  # shape: (num_mats, rows, cols)

    # Convert to radians scaled to [0, 2π]
    angles_rad = (mats / n) * 2 * np.pi

    # Compute sin and cos components
    sin_sum = np.mean(np.sin(angles_rad), axis=0)
    cos_sum = np.mean(np.cos(angles_rad), axis=0)

    # Compute circular mean angle
    mean_angle_rad = np.arctan2(sin_sum, cos_sum)

    # Convert back to [0, n)
    mean_angle = (mean_angle_rad * n / (2 * np.pi)) % n

    return mean_angle


def circular_median_matrix(
    mat1: D2FloatArray, mat2: D2FloatArray, mat3: D2FloatArray, v_max: float = 360.0
) -> D2FloatArray:
    """
    Element-wise circular median of three same-shaped matrices over a [0, v_max) loop.
    """
    if mat1.shape != mat2.shape or mat1.shape != mat3.shape:
        raise ValueError("Input matrices must have the same shape")

    # normalize to [0, v_max)
    a1 = np.mod(mat1, v_max)
    a2 = np.mod(mat2, v_max)
    a3 = np.mod(mat3, v_max)

    # minimal absolute circular difference
    def _circ_diff(x: D2FloatArray, y: D2FloatArray, v_max: float) -> D2FloatArray:
        d = (x - y + v_max / 2) % v_max - v_max / 2
        return np.abs(d)

    d12 = _circ_diff(a1, a2, v_max)
    d13 = _circ_diff(a1, a3, v_max)
    d23 = _circ_diff(a2, a3, v_max)

    # sum of distances
    sum1 = d12 + d13
    sum2 = d12 + d23
    sum3 = d13 + d23

    # select minimal sum
    idx = np.argmin(np.stack((sum1, sum2, sum3)), axis=0)

    # select median
    result = np.where(idx == 0, a1, np.where(idx == 1, a2, a3))
    return result


def circular_median_matrix4(
    mat1: D2FloatArray,
    mat2: D2FloatArray,
    mat3: D2FloatArray,
    mat4: D2FloatArray,
    v_max: float = 360.0,
) -> D2FloatArray:
    """
    Element-wise circular median of four same-shaped matrices over a [0, v_max) loop.
    各要素について、4つの候補のうち“他の3つとの円環距離の合計”が最小になるものを返す。
    """
    # 形状チェック
    if not (mat1.shape == mat2.shape == mat3.shape == mat4.shape):
        raise ValueError("Input matrices must have the same shape")

    # [0, v_max)に正規化
    a1 = np.mod(mat1, v_max)
    a2 = np.mod(mat2, v_max)
    a3 = np.mod(mat3, v_max)
    a4 = np.mod(mat4, v_max)

    # 円環距離（絶対値）を返すヘルパー
    def _circ_diff(x: D2FloatArray, y: D2FloatArray, vmax: float) -> D2FloatArray:
        d = (x - y + vmax / 2) % vmax - vmax / 2
        return np.abs(d)

    # ペアごとの距離
    d12 = _circ_diff(a1, a2, v_max)
    d13 = _circ_diff(a1, a3, v_max)
    d14 = _circ_diff(a1, a4, v_max)
    d23 = _circ_diff(a2, a3, v_max)
    d24 = _circ_diff(a2, a4, v_max)
    d34 = _circ_diff(a3, a4, v_max)

    # 各候補の“他の3つとの距離和”
    sum1 = d12 + d13 + d14
    sum2 = d12 + d23 + d24
    sum3 = d13 + d23 + d34
    sum4 = d14 + d24 + d34

    # 最小のインデックスを選択
    # stackの0軸：sum1, sum2, sum3, sum4 の順
    idx = np.argmin(np.stack((sum1, sum2, sum3, sum4)), axis=0)

    # idx に応じて元の値を返す
    result = np.where(idx == 0, a1, np.where(idx == 1, a2, np.where(idx == 2, a3, a4)))

    return result


def find_extinction_angle(
    angles: D1FloatArray,
    array: D1FloatArray,
) -> Optional[float]:

    x = angles
    y = array
    if x[-1] < 0:
        raise ValueError(
            "the last element of angle should larger than 0. This error occurred by putting clockwise image"
        )

    if (
        np.std(y) > 0.1
    ):  # arrayの値の振幅(=ステージを回転させたときのあるピクセルの明暗の差)が小さい時は無視する(たとえば円周視野の外側とか)。

        # x1 = x[(45 <= x) & (x < 90 + 45)]

        x1 = x[(45 <= x) & (x < 90 + 45)]
        x2 = x[(90 + 45 <= x) & (x < 180 + 45)] - 90
        x3 = x[(180 + 45 <= x) & (x < 270 + 45)] - 180
        # x4 = x[(270 <= x) & (x < 360)] - 270

        # x1 = x[(1 <= x) & (x < 90)]
        # x2 = x[(90 <= x) & (x < 180)] - 90
        # x3 = x[(180 <= x) & (x < 270)] - 180
        # x4 = x[(270 <= x) & (x < 360)] - 270

        y1 = y[(45 <= x) & (x < 90 + 45)]
        y2 = y[(90 + 45 <= x) & (x < 180 + 45)]
        y3 = y[(180 + 45 <= x) & (x < 270 + 45)]

        # y1 = y[(45 <= x) & (x < 90 + 45)]
        # y1 = y[(1 <= x) & (x < 90)]
        # y2 = y[(90 <= x) & (x < 180)]
        # y3 = y[(180 <= x) & (x < 270)]
        # y4 = y[(270 <= x) & (x < 360)]

        min1 = cast(float, x1[np.argmin(y1)])
        min2 = cast(float, x2[np.argmin(y2)])
        min3 = cast(float, x3[np.argmin(y3)])
        # min4 = cast(float, x4[np.argmin(y4)])

        # # 奇数個に調整する必要あり。じゃないとmedianが変になる。
        # min_list = np.array([min1, min2, min3])

        # min_list2 = np.concatenate([np.concatenate([min_list, min_list + 90]), [min4]])

        # return circular_median_data_via_grid([min2, min3, min4], 0.0, 90.0, 100)

        res = circular_median_data_via_grid([min2, min3, min1], 45.0, 90.0 + 45.0, 100)
        return res - 90 if res >= 90 else res

        # plt.plot(x1, y1)
        # plt.plot(x2, y2)
        # plt.plot(x3, y3)
        # plt.plot(x4, y4)
        # plt.vlines(np.median(min_list2), 0, 100)
        # plt.show()
        # plt.plot(x,y)
        # plt.vlines(np.median(min_list2), 0, 100)
        # plt.show()

        # return np.median(min_list)
        # r = x1[np.argmax(y1)]
        # return x1[np.argmax(y1)] - 45

    else:
        return None


I = TypeVar("I", bound=np.generic)


def assign_0_to_none(val: Optional[I]) -> np.float64:
    if val is None:
        return np.float64(0.0)
    else:
        return val.astype(np.float64)


# import numpy as np
# from typing import Callable, Optional, List
# from tqdm import tqdm


def make_extinction_angle_map(
    pics: list[RGBPicture],
    angles: D1FloatArray,
    shift: float = 45,
    progress_callback: Callable[[float], None] = lambda p: None,
) -> D2FloatArray:
    """
    各画像からグレースケール変換後の画素値配列（各ピクセル毎に時間軸が並ぶ）
    に対して、logic_fn（ここでは find_extinction_angle）を適用し、結果を
    2次元配列として返す（ベクトル化処理により高速化）。
    """
    # 各 RGBPicture をグレースケールに変換
    gray_pics = list(map(convert_to_gray_scale, pics))
    # (高さ, 幅, フレーム数) の配列に変換
    t_gray_pics = as_float64(transpose_dat_shape(gray_pics))
    H, W, n_angles = t_gray_pics.shape

    # angles が全画素共通なので、最初にエラーチェック
    if angles[-1] < 0:
        raise ValueError(
            "the last element of angle should be larger than 0. This error occurred by putting clockwise image"
        )

    # --- 以下、find_extinction_angle の vectorized 処理 ---
    # 1. 各ピクセルの標準偏差を計算（低振幅の場合は None 相当＝0 とする）
    std_vals = np.std(t_gray_pics, axis=2)
    valid_mask = std_vals > 0.1  # True の箇所のみ有効とする

    # 2. 各区間に対応するブールマスク（angles は全体で共通）
    #    条件は45 <= x < 135, 135 <= x < 225, 225 <= x < 315
    mask1 = (angles >= shift) & (angles < shift + 90)
    mask2 = (angles >= shift + 90) & (angles < shift + 180)
    mask3 = (angles >= shift + 180) & (angles < shift + 270)

    # 対応する角度配列（※後で argmin した結果をこれらから抜き出す）
    # mask2, mask3 については、それぞれ 90, 180 を引いて調整
    x1 = angles[mask1]  # shape: (n1,)
    x2 = angles[mask2] - 90  # shape: (n2,)
    x3 = angles[mask3] - 180  # shape: (n3,)

    # 3. t_gray_pics の各ピクセルについて、各区間の輝度値を抽出
    #    ※t_gray_pics は (H, W, n_angles) なので、ブールインデックスで (H, W, n1) などが得られる
    y1 = t_gray_pics[..., mask1]  # shape: (H, W, n1)
    y2 = t_gray_pics[..., mask2]  # shape: (H, W, n2)
    y3 = t_gray_pics[..., mask3]  # shape: (H, W, n3)

    # 4. 各ピクセルごとに各区間での最小値を与えるインデックスを計算
    argmin1 = np.argmin(y1, axis=2)  # shape: (H, W)
    argmin2 = np.argmin(y2, axis=2)  # shape: (H, W)
    argmin3 = np.argmin(y3, axis=2)  # shape: (H, W)

    # 5. 各ピクセルにおける候補角度を、各区間の x1, x2, x3 から取得
    #    ※argmin* は各ピクセルに対して、該当する1次元インデックス（0～n-1）を返すので、x1[argmin1] 等で各画素の値が得られる
    min1 = x1[argmin1]  # shape: (H, W)
    min2 = x2[argmin2]  # shape: (H, W)
    min3 = x3[argmin3]  # shape: (H, W)

    # 6. 元の find_extinction_angle では [min2, min3, min1] の circular median を grid search で求め、
    #    得られた値が90以上なら 90 を引いていたが、入力が [45,135] の連続区間にあるため、ここでは単純な中央値と同等とみなす

    medians_adjusted = circular_median_matrix(min2, min3, min1, 90)
    # medians_adjusted = circular_mean_matrices([min2, min3, min1], 90)
    # candidates = np.stack([min2, min3, min1], axis=0)  # shape: (3, H, W)
    # medians = np.median(candidates, axis=0)  # shape: (H, W)
    # medians_adjusted = np.where(medians >= 90, medians - 90, medians)

    # 7. 標準偏差が低い（=信頼性が低い）ピクセルは、元コードでは None となり assign_0_to_none で 0 になるので、
    #    ここでもそのように扱う
    extinction_angle = np.where(valid_mask, medians_adjusted, 0.0)

    # 進捗コールバックは、ベクトル演算のため途中経過は得にくいが、完了時に 100% を通知
    progress_callback(1.0)
    return D2FloatArray(extinction_angle)


def closest_element_indices(vec: D1FloatArray, mat: D2FloatArray) -> D2IntArray:
    """
    For each element in 'mat', find the index of the closest element in 'vec'.

    Parameters:
    vec (NDArray[np.float64]): 1D array of floats.
    mat (NDArray[np.float64]): N-dimensional array of floats.

    Returns:
    NDArray[np.float64]: N-dimensional array of floats containing the indices of the closest elements in 'vec'.
    """
    diff = np.abs(mat[..., np.newaxis] - vec)
    indices = np.argmin(diff, axis=-1)
    return cast(D2IntArray, indices)


def choose_pixel_of_indices_matrix(
    pics: list[RGBPicture], indices_mat: D2IntArray
) -> RGBPicture:
    e1, e2 = as_two_element_tuple(pics[0].shape)

    if indices_mat.shape != (e1, e2):
        raise ValueError(
            "invalid input. first and second element of pics[0].shape and indices_mat.shape should be same"
        )

    i_indices = np.arange(e1)[:, np.newaxis]  # shape (10, 1)
    j_indices = np.arange(e2)  # shape (10,)
    img = np.array(pics, dtype=np.uint8)[indices_mat, i_indices, j_indices, :]
    return cast(RGBPicture, img)


def make_color_maps(
    pics: list[RGBPicture],
    angles: D1FloatArray,
    s_pics: Optional[list[RGBPicture]] = None,
    s_angles: Optional[D1FloatArray] = None,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> RawMaps:

    degree_0 = pics[np.argmin(np.abs(angles - 0.0))]
    degree_22_5 = pics[np.argmin(np.abs(angles - 22.5))]
    degree_45 = pics[np.argmin(np.abs(angles - 45.0))]
    degree_67_5 = pics[np.argmin(np.abs(angles - 67.5))]

    initial_shift = 10  # 　最初のフレームはanglesがひたすら0で偏りがあるので、動画が回転し始めたときのフレームを使用する。

    progress_callback(None)

    extinction_angle_0 = make_extinction_angle_map(pics, angles, shift=initial_shift)

    extinction_angle_225 = make_extinction_angle_map(
        pics, angles, shift=initial_shift + 22.5
    )
    extinction_angle_45 = make_extinction_angle_map(
        pics, angles, shift=initial_shift + 45
    )
    extinction_angle_675 = make_extinction_angle_map(
        pics, angles, shift=initial_shift + 67.5
    )

    cv_extinction_angle = circular_variance(
        [
            D2FloatArray(extinction_angle_0),
            D2FloatArray(extinction_angle_225),
            D2FloatArray(extinction_angle_45),
            D2FloatArray(extinction_angle_675),
        ],
        cycle=90,
    )

    extinction_angle = circular_median_matrix4(
        extinction_angle_0,
        extinction_angle_225,
        extinction_angle_45,
        extinction_angle_675,
        90,
    )

    # under cross polarized light
    max_retardation_index = closest_element_indices(
        angles, D2FloatArray(extinction_angle + 45)
    )

    min_retardation_index = closest_element_indices(angles, extinction_angle)

    R_color_map = choose_pixel_of_indices_matrix(pics, max_retardation_index)

    # plt.imshow(R_color_map)
    # plt.colorbar()
    # plt.show()

    extinction_color_map = choose_pixel_of_indices_matrix(pics, min_retardation_index)

    # plt.imshow(extinction_color_map)
    # plt.colorbar()
    # plt.show()

    mask = ~create_outside_circle_mask(extinction_color_map)
    v = get_V_from_max_R_map(extinction_color_map, mask)

    # print("---------------------")
    # print(v)
    # print("---------------------")
    # v = 20  #! いったんcolor Vで補正するのやめる
    # R_color_map_used = add_color_V_to_img(R_color_map, -v)

    hsv_R_map = cv2.cvtColor(R_color_map, cv2.COLOR_RGB2HSV)
    hsv_R_min_map = cv2.cvtColor(extinction_color_map, cv2.COLOR_RGB2HSV)

    d_hsv_R_map = deepcopy(hsv_R_map)

    d_hsv_R_map[:, :, 2] = np.clip(
        hsv_R_map[:, :, 2].astype(np.float64)
        - hsv_R_min_map[:, :, 2].astype(np.float64),
        0,
        255,
    ).astype(np.uint8)

    R_color_map_used = cv2.cvtColor(d_hsv_R_map, cv2.COLOR_HSV2RGB)

    # plt.imshow(R_color_map_used)
    # plt.show()

    extinction_color_map_used = add_color_V_to_img(extinction_color_map, -v)

    if not (s_pics is not None and s_angles is not None):
        # -------------------------------
        # full wave plateがないとき
        # -------------------------------
        return {
            "degree_0": degree_0,
            "degree_22_5": degree_22_5,
            "degree_45": degree_45,
            "degree_67_5": degree_67_5,
            "extinction_color_map": extinction_color_map,
            "R_color_map": R_color_map_used,
            "R_color_map_raw": R_color_map,
            "R_color_map_display": R_color_map,
            "extinction_angle": extinction_angle,
            "extinction_angle_0": extinction_angle_0,
            "extinction_angle_225": extinction_angle_225,
            "extinction_angle_45": extinction_angle_45,
            "extinction_angle_675": extinction_angle_675,
            "cv_extinction_angle": cv_extinction_angle,
            "inclination_0_to_180": None,
            "p45_R_color_map": None,
            "m45_R_color_map": None,
            "azimuth": None,
            "max_retardation_map": None,
            "p45_R_map": None,
            "m45_R_map": None,
            "theoritical_image": None,
            "inclination": None,
            "azimuth360": None,
            # "tilt_direction_estimation_result": None,
        }

    else:
        # -------------------------------
        # full wave plateがあるとき
        # -------------------------------
        # s_pics and pics should have same shape
        h, w = as_two_element_tuple(pics[0].shape)

        # under 1st order color plate (optional)
        s_pics = list(map(lambda x: resize_img(x, w, h), s_pics))

        plus45_angle_index = closest_element_indices(
            s_angles, D2FloatArray(extinction_angle + 45)
        )
        minus45_angle_index = closest_element_indices(
            s_angles, D2FloatArray(extinction_angle + 45 + 90)
        )
        p45_R_color_map = choose_pixel_of_indices_matrix(s_pics, plus45_angle_index)
        p45_R_color_map_used = add_color_V_to_img(p45_R_color_map, -v)
        m45_R_color_map = choose_pixel_of_indices_matrix(s_pics, minus45_angle_index)
        m45_R_color_map_used = add_color_V_to_img(m45_R_color_map, -v)

        return {
            "degree_0": degree_0,
            "degree_22_5": degree_22_5,
            "degree_45": degree_45,
            "degree_67_5": degree_67_5,
            "extinction_color_map": extinction_color_map,
            "R_color_map": R_color_map_used,
            "R_color_map_raw": R_color_map,
            "R_color_map_display": R_color_map,
            "extinction_angle": extinction_angle,
            "extinction_angle_0": extinction_angle_0,
            "extinction_angle_225": extinction_angle_225,
            "extinction_angle_45": extinction_angle_45,
            "extinction_angle_675": extinction_angle_675,
            "cv_extinction_angle": cv_extinction_angle,
            "p45_R_color_map": p45_R_color_map_used,
            "m45_R_color_map": m45_R_color_map_used,
            "inclination_0_to_180": None,
            "azimuth": None,
            "max_retardation_map": None,
            "p45_R_map": None,
            "m45_R_map": None,
            "theoritical_image": None,
            "inclination": None,
            "azimuth360": None,
            # "tilt_direction_estimation_result": None,
        }


def make_R_maps(
    raw_maps: RawMaps,
    xpl_color_chart: D1RGB_Array,
    xpl_R_array: D1FloatArray,
    pol_lambda_color_chart: Optional[D1RGB_Array | RGBPicture] = None,
    pol_lambda_R_array: Optional[D1FloatArray] = None,
    progress_callback: Callable[[float | None], None] = lambda p: None,
    full_wave_plate: float = 530,
    max_R: Optional[float] = None,
    min_R: Optional[float] = None,
) -> RawMaps:

    R_color_map = raw_maps["R_color_map"]
    extinction_angle = raw_maps["extinction_angle"]
    p45_R_color_map = raw_maps["p45_R_color_map"]
    m45_R_color_map = raw_maps["m45_R_color_map"]

    R_map, _, _ = make_retardation_color_map(
        R_color_map, xpl_color_chart, xpl_R_array, progress_callback
    )

    if not (
        p45_R_color_map is not None
        and m45_R_color_map is not None
        and pol_lambda_color_chart is not None
        and pol_lambda_R_array is not None
    ):
        return {
            **raw_maps,
            "max_retardation_map": R_map,
        }
    else:
        p45_R_map, _, _ = make_retardation_color_map(
            p45_R_color_map,
            pol_lambda_color_chart,
            pol_lambda_R_array,
            progress_callback,
            maxR=max_R,  # 検板が挿入済みのとき
            minR=min_R,  # 検板が挿入済みのとき
        )

        m45_R_map, _, _ = make_retardation_color_map(
            m45_R_color_map,
            pol_lambda_color_chart,
            pol_lambda_R_array,
            progress_callback,
            maxR=max_R,
            minR=min_R,
        )

        # R_map_plus = R_map + full_wave_plate
        # R_map_minus = np.abs(full_wave_plate - R_map)

        # chart = pol_lambda_color_chart
        # R_array = pol_lambda_R_array

        # R_plus = apply_color_map(
        #     R_map_plus,
        #     R_array[R_array > full_wave_plate],
        #     chart[R_array > full_wave_plate],
        # )
        # plt.imshow(R_plus)
        # plt.show()

        # R_minus = apply_color_map(
        #     R_map_minus,
        #     R_array[R_array <= full_wave_plate],
        #     chart[R_array <= full_wave_plate],
        # )
        # plt.imshow(R_minus)
        # plt.show()

        # lab_p45_color_map = cv2.cvtColor(p45_R_color_map, cv2.COLOR_RGB2LAB).astype(
        #     np.float64
        # )
        # lab_m45_color_map = cv2.cvtColor(m45_R_color_map, cv2.COLOR_RGB2LAB).astype(
        #     np.float64
        # )
        # lab_R_plus = cv2.cvtColor(R_plus, cv2.COLOR_RGB2LAB).astype(np.float64)
        # lab_R_minus = cv2.cvtColor(R_minus, cv2.COLOR_RGB2LAB).astype(np.float64)
        # is_plus_p45 = np.sum((lab_p45_color_map - lab_R_plus) ** 2, axis=2) < np.sum(
        #     (lab_p45_color_map - lab_R_minus) ** 2, axis=2
        # )
        # is_plus_m45 = np.sum((lab_m45_color_map - lab_R_plus) ** 2, axis=2) < np.sum(
        #     (lab_m45_color_map - lab_R_minus) ** 2, axis=2
        # )

        # p45_R_map = np.zeros_like(R_map)
        # p45_R_map[is_plus_p45] = R_map[is_plus_p45] + full_wave_plate
        # p45_R_map[~is_plus_p45] = np.abs(R_map[~is_plus_p45] - full_wave_plate)
        # m45_R_map = np.zeros_like(R_map)
        # m45_R_map[is_plus_m45] = R_map[is_plus_m45] + full_wave_plate
        # m45_R_map[~is_plus_m45] = np.abs(R_map[~is_plus_m45] - full_wave_plate)

        # def get_azimuth(p45_R_map)
        azimuth = np.zeros_like(extinction_angle)
        p_larger_than_m = p45_R_map >= m45_R_map

        #! azimuthと消光角は逆の座標系！！
        azimuth[p_larger_than_m] = 180 - extinction_angle[p_larger_than_m]
        azimuth[~p_larger_than_m] = 90 - extinction_angle[~p_larger_than_m]

        return {
            **raw_maps,
            "max_retardation_map": R_map,
            "m45_R_map": m45_R_map,
            "p45_R_map": p45_R_map,
            "azimuth": azimuth,
        }


def circular_variance(mat_list: list[D2FloatArray], cycle: float = 360) -> D2FloatArray:
    """
    mat_list: 各要素が2次元のfloat配列（角度情報を含む）のリスト
    cycle: 角度の周期（デフォルトは360、0と360は同一視）

    戻り値:
        各要素ごとの円分散（1 - 平均結果ベクトルの長さ）を格納した2次元配列
    """
    # mat_list をスタックして 3次元配列にする（形状: (n, rows, cols)）
    stacked = np.stack(mat_list, axis=0)

    # 各要素をラジアンに変換（0 と cycle が同じ意味になるように）
    angles_rad = stacked * (2 * np.pi / cycle)

    # 各角度の余弦・正弦を計算
    cos_vals = np.cos(angles_rad)
    sin_vals = np.sin(angles_rad)

    # スタック方向（=リスト内の各行列間）で平均値を計算
    mean_cos = np.mean(cos_vals, axis=0)
    mean_sin = np.mean(sin_vals, axis=0)

    # 平均結果ベクトルの長さ R の計算
    R2 = mean_cos**2 + mean_sin**2

    # 円分散は 1 - R
    dispersion = 1 - R2

    return D2FloatArray(dispersion)


def add_color_V_to_img(
    img: RGBPicture,
    dV: float,
) -> RGBPicture:
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    # print(img.astype(np.float64))
    # print(v.astype(np.float64) + (img.astype(np.float64) + np.float64(dV)))
    v2_new = np.clip(v.astype(np.float64) + dV, 0, 225).astype(np.uint8)
    im2_hsv_new = cv2.merge([h, s, v2_new])
    im2_new = cv2.cvtColor(im2_hsv_new, cv2.COLOR_HSV2RGB).astype(np.uint8)

    return RGBPicture(im2_new)


def get_V_from_ex_angle_color_map(img: RGBPicture, mask: D2BoolArray) -> float:
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    # plt.hist(v[mask], bins=200)
    # plt.show()
    return float(np.median(v[mask]))


def get_V_from_max_R_map(img: RGBPicture, mask: D2BoolArray) -> float:
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    # plt.hist(v[mask], bins=200)
    # plt.show()
    v_array = v[mask]
    # return gamma_mode_from_data(v_array)
    return np.median(v_array)


if __name__ == "__main__":

    import pandas as pd
    from niconavi.type import ComputationResult

    # r: ComputationResult = pd.read_pickle("../test/data/output/yamagami_xpl_pol.pkl")

    r: ComputationResult = pd.read_pickle(
        "../test/data/output/output_2025-06-10-18-46-22.niconavi"
        # "../test/data/output/output_2025-06-10-11-01-59.niconavi"
    )
    # r: ComputationResult = pd.read_pickle("../test/data/output/yamagami.pkl")
    r.pics
    angles = r.angles
    pics = r.pics_rotated

    # %%
    plt.imshow(r.pics_rotated[75])
    plt.colorbar()
    plt.show()
    # %%

    maps = make_color_maps(pics, angles)
    # %%

    gray_pics = list(map(convert_to_gray_scale, pics))
    # (高さ, 幅, フレーム数) の配列に変換
    t_gray_pics = as_float64(transpose_dat_shape(gray_pics))
    H, W, n_angles = t_gray_pics.shape
    # %%
    i = 150
    j = 400
    plt.imshow(r.raw_maps["extinction_angle"], cmap="hsv")
    plt.colorbar()
    plt.scatter(j, i, marker="+", color="red")
    plt.show()
    plt.imshow(t_gray_pics[:, :, 33])
    plt.colorbar()
    plt.scatter(j, i, marker="+", color="red")
    plt.show()
    plt.plot(angles, t_gray_pics[i, j, :])

    # %%

    degree_0 = pics[np.argmin(np.abs(angles - 0.0))]
    degree_22_5 = pics[np.argmin(np.abs(angles - 22.5))]
    degree_45 = pics[np.argmin(np.abs(angles - 45.0))]
    degree_67_5 = pics[np.argmin(np.abs(angles - 67.5))]

    initial_shift = 10  # 　最初のフレームはanglesがひたすら0で偏りがあるので、動画が回転し始めたときのフレームを使用する。

    extinction_angle = make_extinction_angle_map(pics, angles, shift=initial_shift)

    extinction_angle_225 = make_extinction_angle_map(
        pics, angles, shift=initial_shift + 22.5
    )
    extinction_angle_45 = make_extinction_angle_map(
        pics, angles, shift=initial_shift + 45
    )
    extinction_angle_675 = make_extinction_angle_map(
        pics, angles, shift=initial_shift + 67.5
    )

    # %%
    plt.imshow(r.raw_maps["azimuth"], cmap="hsv")
    plt.colorbar()
    plt.show()

    # %%
    plt.imshow(r.raw_maps["R_color_map"])
    plt.show()
    # %%

    #! cutoff ではなく、ストレッチ
    hsv_color_map = cv2.cvtColor(r.raw_maps["R_color_map"], cv2.COLOR_RGB2HSV)
    hsv_color_map[:, :, 2] = np.clip(
        hsv_color_map[:, :, 2].astype(np.float64) - 20, 0, 255
    ).astype(np.uint8)
    R_color_map_mod = cv2.cvtColor(hsv_color_map, cv2.COLOR_HSV2RGB)

    # plt.imshow(hsv_color_map[:,:,2])
    # plt.colorbar()
    # plt.show()

    mask = ~create_outside_circle_mask(hsv_color_map[:, :, 2])
    plt.hist(hsv_color_map[:, :, 2][mask], bins=200)
    plt.show()

    # make_R_maps(hsv_color_map)

    # mask = ~create_outside_circle_mask(r.raw_maps["extinction_color_map"])
    # v = get_V_from_ex_angle_color_map(r.raw_maps["extinction_color_map"], mask)
    # im = add_color_V_to_img(r.raw_maps["R_color_map"], -v)

# %%
