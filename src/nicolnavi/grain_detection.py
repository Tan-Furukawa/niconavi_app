# %%
from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Sequence, Optional, Literal, cast, TypeVar, TypeAlias, overload
import cv2
from niconavi.image.type import (
    RGBPicture,
    MonoColorPicture,
    BinaryPicture,
)
from niconavi.tools.type import (
    D2FloatArray,
    D2BoolArray,
    D2IntArray,
    D2UintArray,
    D2,
)
from niconavi.tools.change_type import as_uint8
from niconavi.image.types_operation import as_BinaryPicture
import skimage.morphology as skm
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from copy import deepcopy
from scipy.ndimage import median_filter
from niconavi.grain_segmentation.grain_segmentation import (
    hessian_image,
    connect_skeleton_endpoints,
    skeleton_loops_only,
    get_grain_boundary_fn,
)


__all__ = [
    "estimate_grain_map_from_pics",
    "assign_random_rgb",
]


def estimate_connected_components(
    img: D2UintArray,
    connectivity: int = 4,
) -> tuple[int, D2IntArray, D2IntArray, D2FloatArray]:
    img2 = img.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img2, connectivity=connectivity
    )
    return (
        num_labels,
        cast(D2IntArray, np.array(labels, dtype=np.int_)),
        cast(D2IntArray, np.array(stats, dtype=np.int_)),
        cast(D2FloatArray, np.array(centroids, dtype=np.int_)),
    )


T = TypeVar("T", D2UintArray, D2IntArray, D2BoolArray, D2FloatArray)


def assign_0_to_outer_circle(
    mat: T,
    r: int,
) -> T:
    # 配列の中心を計算
    center_x, center_y = mat.shape[1] // 2, mat.shape[0] // 2

    # 行列の各位置における距離を計算
    y, x = np.ogrid[: mat.shape[0], : mat.shape[1]]
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # 半径内にあるかどうかのマスクを作成
    mask = distance <= r

    # 新しい行列を作成し、円外の要素を n にする
    result = np.full_like(mat, 0)
    result[mask] = mat[mask]

    return cast(T, result)


def assign_zero_to_area_smaller_than_N(
    N: int, num_labels: int, labels: D2IntArray, stats: D2IntArray
) -> D2IntArray:
    res = labels.copy()

    for label in range(1, num_labels):  # ラベル0は背景
        size = stats[label, cv2.CC_STAT_AREA]
        if size <= N:
            # ラベルが一致する部分を0に置き換え
            res[labels == label] = 0
    return res


def remove_index_zero_to_neighbor_index(labels: D2IntArray) -> D2IntArray:
    if not np.any(labels == 0):
        Warning("labels do not contain 0 component")

    output_arr = labels.copy()
    mask_zero = labels == 0
    distance, indices = distance_transform_edt(labels == 0, return_indices=True)

    filled_values = labels[tuple(indices)]

    output_arr[mask_zero] = filled_values[mask_zero]
    return output_arr


def reindex_grain_image(
    arr: D2IntArray, ignored: Optional[list[int]] = None
) -> D2IntArray:
    """
    配列中のラベルを 0 から始まる連番に振り直します。
    ただし、ignored に含まれるラベルの画素は変更せず、
    それらのラベル番号も他のラベルの再割り当てには使いません。

    Parameters
    ----------
    arr : np.ndarray (int型想定)
        ラベルが格納された配列 (N次元でも可)。
    ignored : list[int], optional
        変更を加えずに残すラベルのリスト。指定されたラベルはそのまま。

    Returns
    -------
    out : np.ndarray
        ラベルを再割り当てした配列。
    """
    if ignored is None:
        ignored = []

    # 配列内に存在するユニークなラベルを取得
    unique_labels = np.unique(arr)

    # 再割り当ての対象(ignored に含まれないラベル)を取得してソート
    reassign_labels = [label for label in unique_labels if label not in ignored]
    reassign_labels.sort()

    # 古いラベル -> 新しいラベル への対応を格納する辞書
    label_map = {}

    # 再割り当てに使える次のラベル値を探すためのカウンタ
    next_label = 0

    # 再割り当てするラベルに対して連番を割り当て
    for old_label in reassign_labels:
        # ignored に入っている値は使えないのでスキップして次を探す
        while next_label in ignored:
            next_label += 1
        label_map[old_label] = next_label
        next_label += 1

    # ignored に含まれるラベルはマッピングを「元のラベル -> 同じラベル」にしておく
    for label in ignored:
        if label in unique_labels:
            label_map[label] = label

    # 出力用の配列をコピー
    out = arr.copy()

    # label_map に従って一括変換
    # (np.takeやベクトル化手法を使う方法もあるが、ここでは分かりやすくループで置き換える)
    for old_label, new_label in label_map.items():
        out[arr == old_label] = new_label

    return out


def assign_label_to_boundary_img(
    img: BinaryPicture,
    smallest_grain_size: int,
    mask: Optional[D2BoolArray] = None,
) -> tuple[D2IntArray, D2IntArray]:

    img2 = D2UintArray(as_uint8(~img))
    img3 = assign_0_to_outer_circle(img2, int(img2.shape[0] / 2))
    num_labels, labels, stats, centroids = estimate_connected_components(img3)

    labels_with_boundary = assign_zero_to_area_smaller_than_N(
        smallest_grain_size, num_labels, labels, stats
    )

    if mask is not None:
        labels_with_boundary[mask] = 999999

    labels_with_boundary = reindex_grain_image(labels_with_boundary, ignored=[999999])
    labels = remove_index_zero_to_neighbor_index(labels_with_boundary)
    labels = assign_0_to_outer_circle(labels, int(labels.shape[0] / 2))

    # if is_same_index is not None:
    # labels[is_same_index]
    return labels, labels_with_boundary


def remove_short_contours(
    contours: Sequence[cv2.typing.MatLike], len: float
) -> list[cv2.typing.MatLike]:
    contours_long = []
    for cnt in contours:
        length = cv2.arcLength(cnt, True)
        if length > len:
            contours_long.append(cnt)
    return contours_long


def make_grain_boundary_img_from_pic_by_logic_sobel(
    pic: RGBPicture,
    median_filter_size: int,
) -> MonoColorPicture:
    pic = median_filter(pic, median_filter_size)

    sobelx = cv2.Sobel(pic, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(pic, cv2.CV_64F, 0, 1, ksize=5)

    # 勾配強度画像を求める
    magnitude = cv2.magnitude(sobelx, sobely)

    # 正規化して可視化しやすく
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
    magnitude_norm = magnitude_norm.astype(np.uint8)
    magnitude_norm = cv2.cvtColor(magnitude_norm, cv2.COLOR_RGB2GRAY)

    return magnitude_norm


def make_grain_boundary_img_from_pic_by_logic_binary(
    pic: RGBPicture,
    median_filter_size: int,
    binary_threshold: int,
    min_contour_length: float,
) -> MonoColorPicture:

    pic = median_filter(pic, median_filter_size)
    gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

    ret, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours_long = remove_short_contours(contours, min_contour_length)

    img_blank = np.zeros_like(gray)
    img_contour_only = cv2.drawContours(
        img_blank, contours_long, -1, (255, 255, 255), 0
    )

    img_contour_only = cv2.cvtColor(img_contour_only, cv2.COLOR_BGR2RGB)

    r = cv2.cvtColor(img_contour_only, cv2.COLOR_RGB2GRAY)

    return cast(MonoColorPicture, r)


def extract_grain_boundary(
    pics: list[RGBPicture],
    binary_threshold: int,
    median_filter_size: int,
    min_contour_length: float,
    percentile: float,
    angles: Optional[D2FloatArray] = None,
    logic: Literal["sobel", "binary"] = "binary",
) -> BinaryPicture:

    pics_used: list[RGBPicture]

    if angles is not None:
        if len(angles) == len(pics):
            pics_used = cast(list[RGBPicture], pics[angles < 90])
        else:
            raise ValueError("angles and pics should same length")
    else:
        l = int(len(pics) / 4)
        pics_used = pics[2:l]

    if len(pics_used) > 0:

        grain_boundary = np.zeros_like(pics_used[0][:, :, 0]).astype(np.float64)

        for pic in pics_used:
            if logic == "binary":
                r = make_grain_boundary_img_from_pic_by_logic_binary(
                    pic, median_filter_size, binary_threshold, min_contour_length
                )
            elif logic == "sobel":
                r = make_grain_boundary_img_from_pic_by_logic_sobel(
                    pic, median_filter_size
                )
            else:
                raise ValueError("invalid variable (logic) in extract_grain_boundary")
            grain_boundary += r.astype(np.float64)

    else:
        ValueError("length of pics_used is less than 0")

    grain_boundary_bool = grain_boundary >= np.percentile(
        grain_boundary[grain_boundary != 0.0].flatten(), percentile
    )

    return grain_boundary_bool


# @overload
# def skeletonize(bool_image: BinaryPicture) -> BinaryPicture: ...


# @overload
# def skeletonize(bool_image: D2BoolArray) -> D2BoolArray: ...


# def skeletonize(bool_image):  # type: ignore
#     image = (bool_image).astype(np.uint8)
#     image[image == 1] = 255
#     out = skm.skeletonize(image)
#     # out = cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_GUOHALL)
#     return out != 0


def extend_grain_boundary(
    grain_boundary_bool: BinaryPicture, kernel_size: int
) -> BinaryPicture:
    grain_boundary_sk = skeletonize(grain_boundary_bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    grain_boundary_thick = cv2.dilate(
        np.array(grain_boundary_sk, dtype=np.uint8), kernel
    )

    out = skeletonize(as_BinaryPicture(grain_boundary_thick))

    return cast(BinaryPicture, (out | grain_boundary_bool))


def estimate_grain_map_from_pics(
    pics: list[RGBPicture],
    extinction_color_is_less_than: int = 50,
    median_filter_size: int = 5,
    shortest_contour: float = 10.0,
    percentile: float = 50,
    morphological_transformations_kernel_size: int = 5,
    smallest_grain_size: int = 1,
    grain_boundary_logic: Literal["sobel", "binary"] = "sobel",
) -> tuple[D2IntArray, D2IntArray]:

    grain_boundary_bool = extract_grain_boundary(
        pics,
        extinction_color_is_less_than,
        median_filter_size,
        shortest_contour,
        percentile,
        logic=grain_boundary_logic,
    )

    grain_boundary_bool = ~extend_grain_boundary(
        grain_boundary_bool, morphological_transformations_kernel_size
    )

    return assign_label_to_boundary_img(~grain_boundary_bool, smallest_grain_size)


def assign_random_rgb(
    input_array: D2IntArray,
    seed: Optional[int] = None,
    use_color: Optional[list[tuple[int, tuple[int, int, int]]]] = None,
) -> RGBPicture:
    if not issubclass(input_array.dtype.type, np.integer):
        raise ValueError("入力配列は整数型でなければなりません。")

    unique_values = np.unique(input_array)

    if seed is not None:
        random.seed(seed)

    random_mapping = {}

    # 指定されたIDに対して指定された色を割り当てる
    specified_ids = []
    if use_color is not None:
        for id_value, color in use_color:
            if id_value in unique_values:
                random_mapping[id_value] = color
                specified_ids.append(id_value)
            else:
                print(f"警告: ID {id_value} は入力配列に存在しません。")

    # 残りの値に対してランダムな色を割り当てる
    for value in unique_values:
        if value not in random_mapping:
            random_mapping[value] = (
                random.randint(0, 255),  # R
                random.randint(0, 255),  # G
                random.randint(0, 255),  # B
            )

    # マッピングを適用してRGB画像を作成
    vectorized_map = np.vectorize(
        lambda x: random_mapping[x], otypes=[np.uint8, np.uint8, np.uint8]
    )

    r, g, b = vectorized_map(input_array)

    output_array = np.stack((r, g, b), axis=-1)

    return output_array


def circular_gradient(
    mat: D2FloatArray, median_filter_size: int = 3, val1: float = 0, val2: float = 360
) -> D2FloatArray:
    """
    入力:
      mat: 0～359の実数値（角度画像）
      median_filter_size: 中央値フィルタの窓サイズ
      val1, val2: この2値は同一の角度として扱う（例：0と359など）
    出力:
      角度の周期性および指定した2値の同一視を考慮した勾配強度画像（単位: 度/画素）
    """
    # 入力行列をコピー
    mat_equiv = np.array(mat, copy=True)
    mat_equiv[np.isclose(mat_equiv, val2)] = val1
    mat_equiv = (mat_equiv - val1) / (val2 - val1) * 360

    # ノイズ除去のため中央値フィルタを適用
    pic = median_filter(mat_equiv, median_filter_size)

    # plt.imshow(pic)
    # plt.show()

    # 角度（度）をラジアンに変換し、sin, cos 表現に変換
    pic_rad = np.deg2rad(pic)
    sin_pic = np.sin(pic_rad)
    cos_pic = np.cos(pic_rad)

    # Sobelフィルタにより sin, cos それぞれのx, y方向の微分を計算
    sin_grad_x = cv2.Sobel(sin_pic, cv2.CV_64F, 1, 0, ksize=5)
    sin_grad_y = cv2.Sobel(sin_pic, cv2.CV_64F, 0, 1, ksize=5)
    cos_grad_x = cv2.Sobel(cos_pic, cv2.CV_64F, 1, 0, ksize=5)
    cos_grad_y = cv2.Sobel(cos_pic, cv2.CV_64F, 0, 1, ksize=5)

    # plt.imshow(sin_grad_x**2 + sin_grad_y**2 + cos_grad_x**2 + cos_grad_y**2)
    # plt.show()

    # チェーンルールにより、各方向の角度微分（ラジアン/画素）を算出
    # grad_x = cos_pic * sin_grad_x - sin_pic * cos_grad_x
    # grad_y = cos_pic * sin_grad_y - sin_pic * cos_grad_y

    # 勾配の大きさ（ラジアン/画素）を計算し、度に変換
    # magnitude_rad = np.sqrt(grad_x**2 + grad_y**2)
    magnitude_rad = np.sqrt(
        sin_grad_x**2 + sin_grad_y**2 + cos_grad_x**2 + cos_grad_y**2
    )

    return magnitude_rad


# def estimate_grain_map_from_angle(
#     angle_map: D2FloatArray,
#     method: Literal["azimuth", "extinction_angle"] = "extinction_angle",
#     median_kernel_size: int = 3,
#     percentile: float = 50,
#     smallest_grain_size: int = 1,
# ) -> tuple[D2IntArray, D2IntArray]:

#     if method == "azimuth":
#         grain_boundary = circular_gradient(angle_map, median_kernel_size, 0, 180)
#     else:
#         grain_boundary = circular_gradient(angle_map, median_kernel_size, 0, 90)

#     p = grain_boundary[grain_boundary != 0.0].flatten()
#     grain_boundary_bool = grain_boundary >= np.percentile(p, percentile)

#     return assign_label_to_boundary_img(grain_boundary_bool, smallest_grain_size)


def extract_boundary(mask: D2BoolArray) -> D2BoolArray:
    """
    2 次元の bool / 0-1 配列から、1 (True) 画素の境界線だけを True にした
    配列を返す（内部画素は False にする）。

    境界の判定は 4 近傍（上下左右）で行う
    ─ つまり「上下左右がすべて 1 なら内部、それ以外は境界」とする。
    """
    # 必ず bool 型に変換しておく
    mask = np.asarray(mask, dtype=bool)

    # 4 方向に 0 を 1 行／1 列だけ詰めてずらした配列を用意
    up = np.pad(mask, ((1, 0), (0, 0)), constant_values=False)[:-1, :]
    down = np.pad(mask, ((0, 1), (0, 0)), constant_values=False)[1:, :]
    left = np.pad(mask, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
    right = np.pad(mask, ((0, 0), (0, 1)), constant_values=False)[:, 1:]

    # 4 方向すべてが True なら「内部」ピクセル
    interior = mask & up & down & left & right

    # 境界 = 全体 − 内部
    boundary = mask & (~interior)
    return boundary


def estimate_grain_map(
    # R_map: D2FloatArray,
    color_map: RGBPicture,
    # angle_map: D2FloatArray,
    # angle_method: Literal["azimuth", "extinction_angle"] = "extinction_angle",
    # eval_method: Literal["angle map", "color map", "both"] = "both",
    th_about_hessian_emphasis: float = 0.0,
    th_about_connect_skeleton_endpoints: float = 0,
    permit_inclusion: bool = True,
    # color_map_median_kernel_size: int = 3,
    # color_map_percentile: float = 50,
    # color_map_min_R: float = 0,
    # color_map_max_R: float = 1000,
    # color_rev_R_estimation: bool = False,  # Trueのとき、minR > R or R > max_Rをしらべることになる。
    # angle_map_median_kernel_size: int = 3,
    # angle_map_percentile: float = 50,
    # angle_map_min_R: float = 100,
    # angle_map_max_R: float = 500,
    # angle_rev_R_estimation: bool = False,  # Trueのとき、minR > R or R > max_Rをしらべることになる。
    smallest_grain_size: int = 10,
    mask: Optional[D2BoolArray] = None,
) -> tuple[D2IntArray, D2IntArray]:

    fn, complexity_init = get_grain_boundary_fn(color_map)

    sk, sg = fn(1 - th_about_hessian_emphasis, th_about_connect_skeleton_endpoints)

    # ex_image = make_extinction_angle_image(ex_map)
    # hessian_color_map = hessian_image(color_map)
    # hessian_ex_image = hessian_image(ex_image)
    # im = cv2.medianBlur(color_map, ksize=11)
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # mask = (im.astype(np.float64) < 50).astype(np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask_new = cv2.erode(mask, kernel)

    # filtered_hessian_ex_image = hessian_ex_image.copy()
    # filtered_hessian_ex_image[mask_new != 0] = 0
    # h = hessian_color_map / 3 + filtered_hessian_ex_image

    # plt.imshow((hessian_color_map - filtered_hessian_ex_image)**2)

    # h = hessian_color_map / 3 + hessian_ex_image
    # h = hessian_color_map

    # sg = hessian_color_map > th_about_hessian_emphasis

    # sk = skeletonize(sg)
    # plt.imshow(sk)
    # plt.colorbar()
    # plt.show()
    # sk = connect_skeleton_endpoints(sk, th_about_connect_skeleton_endpoints)
    # sk = skeleton_loops_only(sk)
    # plt.imshow(sk)
    # plt.colorbar()

    return assign_label_to_boundary_img(sk, smallest_grain_size)[0], sg

    # if eval_method == "angle map" or eval_method == "both":
    #     if angle_method == "azimuth":
    #         angle_map_grain_boundary = circular_gradient(
    #             angle_map, angle_map_median_kernel_size, 0, 180
    #         )
    #     else:
    #         angle_map_grain_boundary = circular_gradient(
    #             angle_map, angle_map_median_kernel_size, 0, 90
    #         )

    #     angle_map_grain_boundary_bool = angle_map_grain_boundary >= np.percentile(
    #         angle_map_grain_boundary[angle_map_grain_boundary != 0.0].flatten(),
    #         angle_map_percentile,
    #     )

    # else:
    #     angle_map_grain_boundary_bool = np.zeros(angle_map.shape, dtype=np.bool_)

    # if eval_method == "color map" or eval_method == "both":
    #     # c = RGBPicture(cv2.cvtColor(color_map, cv2.COLOR_RGB2LAB).astype(np.uint8))
    #     c = color_map

    #     color_map_grain_boundary = make_grain_boundary_img_from_pic_by_logic_sobel(
    #         c, color_map_median_kernel_size
    #     )

    #     color_map_grain_boundary_bool = color_map_grain_boundary >= np.percentile(
    #         color_map_grain_boundary[color_map_grain_boundary != 0.0].flatten(),
    #         color_map_percentile,
    #     )
    # else:
    #     color_map_grain_boundary_bool = np.zeros(color_map.shape[:2], dtype=np.bool_)

    # if eval_method == "both":
    #     if not color_rev_R_estimation:
    #         color_map_condition = ~remove_line_like_components(
    #             ~(D2BoolArray((color_map_min_R <= R_map) & (R_map <= color_map_max_R)))
    #         )
    #     else:
    #         color_map_condition = ~remove_line_like_components(
    #             ~(D2BoolArray((color_map_min_R > R_map) | (R_map > color_map_max_R)))
    #         )
    #     if not angle_rev_R_estimation:
    #         angle_map_condition = ~remove_line_like_components(
    #             ~(D2BoolArray((angle_map_min_R <= R_map) & (R_map <= angle_map_max_R)))
    #         )
    #     else:
    #         angle_map_condition = ~remove_line_like_components(
    #             ~(D2BoolArray((angle_map_min_R > R_map) | (R_map > angle_map_max_R)))
    #         )
    # else:
    #     angle_map_condition = np.ones_like(angle_map_grain_boundary_bool)
    #     color_map_condition = np.ones_like(color_map_grain_boundary_bool)

    # res_bool = (color_map_grain_boundary_bool & color_map_condition) | (
    #     angle_map_grain_boundary_bool & angle_map_condition
    # )
    # if mask is None:
    #     return assign_label_to_boundary_img(res_bool, smallest_grain_size)
    # else:

    #     _, labels_with_boundary = assign_label_to_boundary_img(
    #         res_bool, smallest_grain_size
    #     )

    #     boundary = labels_with_boundary == 0

    #     mask_boundary = extract_boundary(mask)
    #     boundary[mask] = False
    #     boundary[mask_boundary] = True

    #     labels = assign_label_to_boundary_img(boundary, smallest_grain_size, mask)

    #     return labels


def remove_line_like_components(binary: D2BoolArray) -> D2BoolArray:
    # 連結成分ラベル付け
    labeled = label(binary)

    # パラメータ調整（例: ノイズ除去用の最小面積、線状と判断する際の面積/スケルトン長比の閾値）
    min_area = 0
    thickness_threshold = 2  # この値は線の太さ（=面積/スケルトン長）として調整

    # 各成分について処理
    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        # 領域マスクの作成
        region_mask = (labeled == region.label).astype(np.uint8)
        # スケルトン化
        skeleton = skeletonize(region_mask).astype(np.uint8)
        skeleton_length = np.sum(skeleton)

        # 平均的な線幅とみなせる比率を計算
        # plt.imshow(skeleton)
        # plt.show()
        thickness = region.area / (skeleton_length + 1e-5)

        # 閾値以下なら線状（曲線含む）と判断して除去
        if thickness < thickness_threshold:
            binary[labeled == region.label] = 0

    return binary != 0


if __name__ == "__main__":

    from niconavi.type import ComputationResult
    import pandas as pd

    # pics = np.load("../data/Itoshiro_2/pics_rotated_4.npy")
    # r: ComputationResult = pd.read_pickle("../test/data/output/tetori_4k_xpl_pol.pkl")
    r: ComputationResult = pd.read_pickle(
        "../test/data/output/output_2025-06-25-08-27-33.niconavi"
    )
    # angles: D1FloatArray = np.load("../data/movie_54-55/angles.npy")

    R_map = r.raw_maps["max_retardation_map"]
    color_map = r.raw_maps["R_color_map"]
    angle_map = r.raw_maps["azimuth"]
    angle_map = r.raw_maps["extinction_angle"]

    g1, g2 = estimate_grain_map(
        color_map,
        th_about_hessian_emphasis=0.01,
        th_about_connect_skeleton_endpoints=20,
        permit_inclusion=True,
        smallest_grain_size=10,
        # mask = None,
        # R_map,
        # color_map,
        # angle_map,
        # "azimuth",
        # angle_map_median_kernel_size=3,
        # color_map_median_kernel_size=3,
        # angle_map_percentile=50,
        # color_map_percentile=50,
        # angle_map_min_R=50,
        # color_map_max_R=0,
        # smallest_grain_size=10,
    )
    # %%
    plt.imshow(g2)
    # plt.imshow(g2)

    # %%
    img = np.array(R_map < 50, dtype=np.uint8)
    plt.imshow(img)
    # %%
    # 画像読み込み（すでに2値化されている前提で、値は0と1とする）
    # _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    plt.imshow(img)
    plt.show()
    binary = deepcopy(img)

# %%
