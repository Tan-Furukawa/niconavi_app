# %%
import cv2
from niconavi.image.tools import apply_color_map, apply_2dcolor_map
from niconavi.type import ComputationResult, TheoriticalImage
import niconavi.optics.optical_system as osys
from niconavi.statistics.statistics import fitted_by_bspline
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from niconavi.tools.type import D1IntArray, D1FloatArray, D2FloatArray, D2IntArray
from niconavi.image.type import Color, D1RGB_Array, RGBPicture
import scipy.signal

from typing import cast, Literal, Callable, Optional, TypedDict
from niconavi.optics.tools import (
    make_angle_retardation_estimation_function,
    get_thickness_from_max_retardation,
    get_max_retardation_from_thickness,
)
from matplotlib.pyplot import Figure, Axes, Axes
from niconavi.tools.array import pick_element_from_array
from niconavi.tools.type import D2FloatArray, D1FloatArray
from niconavi.image.type import Color, RGBPicture
from niconavi.optics.color import lab_distance_between_color
from niconavi.tools.func_tools import convert_array_to_function
from niconavi.optics.uniaxial_plate import (
    calc_color_chart,
    get_spectral_distribution,
)
from niconavi.image.image_stat.robust_regression import (
    robust_regression,
    apply_regression,
)
from niconavi.image.image_stat.normal_regression import (
    compute_color_regression,
    correct_image,
)
from copy import deepcopy


class HSelectionResInColorChart(TypedDict):
    best_h: float
    h_array: D1FloatArray
    scores: D1FloatArray
    color_chart_1d: D1RGB_Array


def select_h_in_color_chart(
    color_chart: RGBPicture,
    h_array: D1FloatArray,
    img: RGBPicture | D1RGB_Array,
    sample: int = 10000,
    seed: int = 1234,
) -> HSelectionResInColorChart:
    """
    画像 img から乱数 seed を用いて sample 個の画素をサンプリングし、
    各パレット (color_chart の各行) に対して、サンプル画素と各色の LAB 色空間での二乗距離
    の最小値を各サンプル毎に求め、その二乗和 (score) を計算します。
    得られた score が最小となるパレットに対応する h_array の値を返します。

    Parameters
    ----------
    color_chart : RGBPicture, shape=(H, W, 3)
        各行がパレット（W 色）となっている RGB 画像。
    h_array : D1FloatArray, shape=(H,)
        各パレットに対応する値の配列。
    img : RGBPicture, shape=(A, B, 3)
        サンプリング対象となる RGB 画像。
    sample : int, default=10000
        サンプリングする画素数。
    seed : int, default=1234
        サンプリングのための乱数シード。

    Returns
    -------
    float
        score が最小となるパレットに対応する h_array の値。
    """
    # color_chart

    # color_chart_used = color_chart[h_used]
    # plt.imshow(color_chart_used) 
    # plt.colorbar()
    # plt.show()


    # 画像 img を平坦化してランダムに sample 個の画素をサンプリング
    flat_img = img.reshape(-1, 3)
    np.random.seed(seed)
    if sample > flat_img.shape[0]:
        indices = np.random.choice(flat_img.shape[0], sample, replace=True)
    else:
        indices = np.random.choice(flat_img.shape[0], sample, replace=False)
    d_pixels = flat_img[indices]  # shape: (sample, 3)

    # cv2.cvtColor は入力が (H, W, 3) を前提としているため、(sample, 1, 3) にリシェイプ
    d_pixels_reshaped = d_pixels.reshape(-1, 3).astype(np.float32)
    # lab_sample = cv2.cvtColor(d_pixels_reshaped, cv2.COLOR_RGB2LAB)
    # lab_sample = d_pixels_reshaped
    # lab_sample = lab_sample.reshape(-1, 3).astype(np.float32)


    # lab_color_chart = color_chart
    # lab_color_chart = cv2.cvtColor(color_chart, cv2.COLOR_RGB2LAB).astype(np.float32)
    H = color_chart.shape[0]
    scores = np.zeros(H, dtype=np.float64)

    # 各パレット (i) について、全サンプルとの  色差（二乗距離）の最小値の和を計算
    for i in range(H):
        # パレット i の各色 (shape: (W, 3))
        palette = color_chart[i]
        # サンプル画素 (sample, 3) とパレット色 (W, 3) の差を計算 (ブロードキャストを利用)
        diff = d_pixels_reshaped[:, None, :] - palette[None, :, :]  # shape: (sample, W, 3)
        # 各差の二乗距離 (shape: (sample, W))
        dist_sq = np.sum(diff**2, axis=2)
        # 各サンプルについて、パレット内の最も近い色との差（二乗距離）の最小値
        min_diff = np.min(dist_sq, axis=1)
        # plt.imshow(np.transpose(dist_sq), aspect=1000)
        # plt.show()
        # 全サンプルの二乗距離の和を score[i] とする
        scores[i] = np.sum(min_diff)

    # score が最小となるパレットの index を求め、その index に対応する h_array の値を返す
    best_index = np.argmin(scores)

    return HSelectionResInColorChart(
        best_h=h_array[best_index],
        scores=D1FloatArray(scores),
        h_array=h_array,
        color_chart_1d=color_chart[best_index],
    )


def normalize_vec(vec: D1FloatArray) -> D1FloatArray:
    return cast(D1FloatArray, (vec - np.min(vec)) / (np.max(vec) - np.min(vec)))


def extract_center_pixels(
    video_path: str,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> D1RGB_Array:
    """
    入力: 動画ファイルパス
    出力: フレーム毎に中心画素のRGB値を格納した(n, 3)のuint8型のndarray

    1. 動画をBGR形式で読み込む
    2. 各フレームに対して中心画素のBGR値を取得し、RGB順に変換
    3. (n, 3)の配列にまとめて返す
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画ファイル {video_path} を開けませんでした。")

    center_colors = []

    while True:
        progress_callback(None)
        ret, frame = cap.read()
        if not ret:
            break
        h, w, c = frame.shape
        center_y = h // 2
        center_x = w // 2
        center_pixel_bgr = frame[center_y, center_x]  # BGR形式
        center_colors.append(center_pixel_bgr)

    progress_callback(0.0)
    cap.release()

    center_colors_uint = np.array(center_colors).astype(np.uint8)[np.newaxis]
    center_colors_rgb = cv2.cvtColor(
        np.array(center_colors_uint, dtype=np.uint8), cv2.COLOR_BGR2RGB
    )

    # リストから(n, 3)のuint8型ndarrayへ変換
    # center_colors = np.array(center_colors, dtype=np.uint8)
    return cast(D1RGB_Array, center_colors_rgb[0])


def estimate_variances(x: np.ndarray) -> np.ndarray:
    """
    x: 1次元のNDArray
    i = 0 から len(x)-1 までの各iについて、
    部分ベクトル x[:i] の分散を推定し、
    長さ len(x) のベクトルを返す関数。

    注意:
    iが小さい場合 (特にi=0,1) はサンプルが十分でないため、
    np.nan を返すようにしています。
    """
    n = len(x)
    variances = np.empty(n)

    for i in range(n):
        if i < 2:
            # 分散を計算するには少なくとも2点が必要
            variances[i] = np.nan
        else:
            # 標本分散(ddof=1)を用いる
            variances[i] = np.var(x[:i], ddof=1)

    return variances


def get_start_index(vec: D1FloatArray) -> int:
    s = np.where(np.diff(estimate_variances(vec)) > 3)[0][0]
    if s >= 1:
        return s - 1
    else:
        return s


# Create a normalized signal
def get_peak_and_valley_position(
    vec: D1FloatArray, prominence_peak: float = 0.2, prominence_valley: float = 0.2
) -> tuple[D1FloatArray, D1IntArray, D1IntArray]:
    norm_vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
    # Find peaks with a low prominence filter of 0.01
    peak_locations, _ = scipy.signal.find_peaks(norm_vec, prominence=prominence_peak)
    valley_locations, _ = scipy.signal.find_peaks(-norm_vec, prominence=prominence_peak)
    return (
        D1FloatArray(norm_vec),
        D1IntArray(peak_locations),
        D1IntArray(valley_locations),
    )


def resample_points(x: D2FloatArray, N: int) -> D2FloatArray:
    """
    x (M×3): 元の3次元点列
    base_x (N×3): 最終的なサンプリング数を決めるための点列(座標値は使用せず、点数Nのみ使用)

    返り値:
        xをN点に再サンプリングした3次元点列 (N×3)
    """
    # 元の点数
    M = x.shape[0]

    # 特別なケース: M == N の場合はそのまま返す
    if M == N:
        return x.copy()

    # 0からM-1までをパラメトリックなパラメータとし、それに対してN点分の値を線形補間
    # t_new: 0 ~ M-1 の間をN個に分割したパラメータ
    t_new = np.linspace(0, M - 1, N)

    # 線形補間を行うために、floorとceil（またはint()と+1）を使い、両端点を求める
    t_floor = np.floor(t_new).astype(int)
    t_ceil = np.clip(t_floor + 1, 0, M - 1)

    # 内挿用の係数 (t_newがfloorとceilの間でどれだけ進んでいるか)
    alpha = t_new - t_floor

    # floor点とceil点に対して線形補間
    # xは(M×3)なので、対応する点について(1 - alpha)*x_floor + alpha*x_ceilを計算
    x_resampled = (1 - alpha)[:, np.newaxis] * x[t_floor] + alpha[:, np.newaxis] * x[
        t_ceil
    ]

    return x_resampled


def normalize_retardation_plate_less_than_retardation1700(
    x: D1RGB_Array,
    y: D1RGB_Array,
    rgb_index: Literal[0, 1, 2] = 2,
    prominence_peak: float = 0.2,
    prominence_valley: float = 0.2,
) -> tuple[D1RGB_Array, D1RGB_Array, D1IntArray]:

    fx = x.astype(np.float64)
    fy = y.astype(np.float64)

    vecx = np.array(list(map(lambda x: x[rgb_index], fx)), dtype=np.float64)  # type: ignore
    vecy = np.array(list(map(lambda x: x[rgb_index], fy)), dtype=np.float64)  # type: ignore

    sx = get_start_index(vecx)  # type: ignore
    sy = get_start_index(vecy)  # type: ignore

    nvecx, px, vx = get_peak_and_valley_position(
        vecx, prominence_peak=prominence_peak, prominence_valley=prominence_valley  # type: ignore
    )
    nvecy, py, vy = get_peak_and_valley_position(
        vecy, prominence_peak=prominence_peak, prominence_valley=prominence_valley  # type: ignore
    )

    if len(px) < 3:
        raise ValueError(
            "number of peaks in x must be more than 3. Use smaller prominence_peak value."
        )

    if len(py) < 3:
        raise ValueError(
            "number of peaks in y must be more than 3. Use smaller prominence_peak value."
        )

    if len(vx) < 3:
        raise ValueError(
            "number of valleys in x must be more than 3. Use smaller prominence_valley value."
        )

    if len(vy) < 3:
        raise ValueError(
            "number of valleys in y must be more than 3. Use smaller prominence_valley value."
        )

    sections_x = np.array([sx, px[0], vx[0], px[1], vx[1], px[2], vx[2]])
    if not np.all(np.diff(sections_x) > 0):
        raise ValueError("failed to peak detection in x")

    # sections_y is break point
    sections_y = np.array([sy, py[0], vy[0], py[1], vy[1], py[2], vy[2]])
    if not np.all(np.diff(sections_y) > 0):
        raise ValueError("failed to peak detection in y")

    x_new = np.zeros_like(y)
    for i in range(len(sections_x) - 1):
        x_new[sections_y[i] : sections_y[i + 1]] = resample_points(
            fx[sections_x[i] : sections_x[i + 1]],  # type: ignore
            len(fy[sections_y[i] : sections_y[i + 1]]),
        )

    end = vy[2]

    x_new = cast(D1RGB_Array, x_new[0:end])
    y_new = y[0:end]
    r = np.array(list(map(lambda x: x[0], x_new)))
    g = np.array(list(map(lambda x: x[1], x_new)))
    b = np.array(list(map(lambda x: x[2], x_new)))

    rr = fitted_by_bspline(r, k=2, s=30000)
    rg = fitted_by_bspline(g, k=2, s=30000)
    rb = fitted_by_bspline(b, k=2, s=30000)

    # rr = fitted_by_bspline(r, k=2, s=300)
    # rg = fitted_by_bspline(g, k=2, s=300)
    # rb = fitted_by_bspline(b, k=2, s=300)

    rr = np.clip(rr, 0, 255)
    rg = np.clip(rg, 0, 255)
    rb = np.clip(rb, 0, 255)

    rx_new = np.stack([rr, rg, rb], axis=1)

    return (
        cast(D1RGB_Array, rx_new.astype(np.uint8)),
        cast(D1RGB_Array, y_new),
        cast(D1IntArray, np.arange(0, end)),
    )


def add_new_axis(vec: D1RGB_Array) -> RGBPicture:
    return cast(RGBPicture, vec[np.newaxis])


def plot_retardation_color_chart(
    x: D1RGB_Array, y: D1RGB_Array, retardation: D1FloatArray
) -> tuple[Figure, Axes, Axes]:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 3))
    ax[0].set_anchor("SE")  # type: ignore
    ax[1].set_anchor("NE")  # type: ignore
    ax[0].set_anchor("SW")  # type: ignore
    ax[1].set_anchor("NW")  # type: ignore
    ax[0].imshow(  # type: ignore
        x[np.newaxis],  # type: ignore
        extent=(retardation.min(), retardation.max(), 0, 200),  # type: ignore
    )  # type: ignore
    ax[1].imshow(  # type: ignore
        y[np.newaxis],  # type: ignore
        extent=(retardation.min(), retardation.max(), 0, 200),  # type: ignore
    )  # type: ignore
    ax[0].tick_params(left=False, labelleft=False)  # type: ignore
    ax[1].tick_params(left=False, labelleft=False)  # type: ignore
    ax[1].set_xlabel("Retardation")  # type: ignore
    fig.subplots_adjust(hspace=0.05, wspace=0)
    return fig, ax[0], ax[1]  # type: ignore


def plot_rgb_in_sensitive_color_plate(
    x: D1RGB_Array, y: D1RGB_Array, retardation: D1FloatArray
) -> tuple[Figure, Axes]:

    fig, ax = plt.subplots(figsize=(7, 2))

    vecx0 = np.array(list(map(lambda x: x[0], x)), dtype=np.float64)
    vecx1 = np.array(list(map(lambda x: x[1], x)), dtype=np.float64)
    vecx2 = np.array(list(map(lambda x: x[2], x)), dtype=np.float64)
    vecy0 = np.array(list(map(lambda x: x[0], y)), dtype=np.float64)
    vecy1 = np.array(list(map(lambda x: x[1], y)), dtype=np.float64)
    vecy2 = np.array(list(map(lambda x: x[2], y)), dtype=np.float64)

    ax.plot(retardation, normalize_vec(vecx0), c="red")  # type: ignore
    ax.plot(retardation, normalize_vec(vecx1), c="green")  # type: ignore
    ax.plot(retardation, normalize_vec(vecx2), c="blue")  # type: ignore
    ax.plot(retardation, normalize_vec(vecy0), c="red", ls="--")  # type: ignore
    ax.plot(retardation, normalize_vec(vecy1), c="green", ls="--")  # type: ignore
    ax.plot(retardation, normalize_vec(vecy2), c="blue", ls="--")  # type: ignore
    ax.tick_params(left=False, labelleft=False)
    ax.set_xlabel("Retardation")

    return fig, ax


def estimate_retardation(
    retardation: D1FloatArray, max_retardation: float, quartz_wedge_colors: D1RGB_Array
) -> Callable[[Color], float]:
    def closure(rgb: Color) -> float:
        # if np.max(retardation) < max_retardation:
        quartz_wedge_colors_used = pick_element_from_array(
            quartz_wedge_colors, retardation < max_retardation  # type: ignore
        )
        return retardation[
            np.argmin(lab_distance_between_color(rgb, quartz_wedge_colors_used))
        ]

    return closure


def get_quartz_inclination_relation_of_uniaxial_crystal(
    thin_section_thickness: float = 0.035,
    no: float = 1.544,
    ne: float = 1.553,
    progress_callback: Callable[[float | None], None] = lambda e: None,
) -> tuple[Callable[[float], float | None], tuple[float, float]]:  # 関数、定義域

    progress_callback(None)
    retardation = np.linspace(0, 1100, num=500)  # when R > 1100

    img_retardation = calc_color_chart(
        retardation,  # type: ignore
        np.linspace(0, 0, num=1),  # type: ignore
        lambda x, y: get_spectral_distribution(osys.get_retardation_system(R=x))["rgb"],
    )

    theta = np.linspace(0, 90 / 180 * np.pi, num=500)
    img_quartz = calc_color_chart(
        theta,  # type: ignore
        np.linspace(0, 0, num=1),  # type: ignore
        lambda x, y: get_spectral_distribution(
            osys.get_quartz_system(
                dz=thin_section_thickness, inclination=x, no=no, ne=ne
            )
        )["rgb"],
    )

    quartz_retardation = np.zeros(len(img_quartz["color_chart"][0]))
    for i, color in enumerate(img_quartz["color_chart"][0]):
        j = np.argmin(
            lab_distance_between_color(color, img_retardation["color_chart"][0])
            # yellow_distance_between_color(color, img_retardation["color_chart"][0])
        )
        quartz_retardation[i] = retardation[j]

    progress_callback(0)

    if not np.all(np.diff(quartz_retardation) >= -200):
        raise ValueError(
            "failed to construct the relation between crystal inclination and retardation. This may occurr when thickness of thin section is too large, or retardation of minerals are too high."
        )
    else:

        quartz_retardation = fitted_by_bspline(quartz_retardation, k=2, s=10000)

        fn = convert_array_to_function(quartz_retardation, theta * 180 / np.pi)  # type: ignore
        minq = np.min(quartz_retardation)
        maxq = np.max(quartz_retardation)
        return fn, (float(minq), float(maxq))


def build_lab_distance_computer(
    color: Color,
) -> Callable[[RGBPicture], tuple[int, int]]:
    """
    指定した RGB の色 (color) に対して、入力画像から最も近いピクセル位置を求める
    関数を返すファクトリ関数。

    Parameters
    ----------
    color : tuple (R, G, B)
        距離計算の対象となる色 (RGB形式)

    Returns
    -------
    distance_func : function
        引数に RGB 画像 (np.ndarray) を受け取り、
        color にもっとも近い画素の (row_idx, col_idx) を返す関数
    """
    # color を 1x1 ピクセル画像として Lab 変換
    color_patch = np.array([[color]], dtype=np.uint8)  # shape: (1,1,3)
    lab_color_patch = cv2.cvtColor(color_patch, cv2.COLOR_RGB2Lab)  # shape: (1,1,3)
    # lab_color を float に (後の差分計算でオーバーフローを防ぐため)
    lab_color = lab_color_patch[0, 0, :].astype(np.float64)  # shape: (3,)

    def compute_argmin_lab_distance(rgb_img: RGBPicture) -> tuple[int, int]:
        """
        入力画像 (rgb_img) の各ピクセルと、build_lab_distance_computer で
        事前に指定した color (Lab変換済) の距離を計算し、
        最も近いピクセル位置を (row_idx, col_idx) で返す。
        """
        # 画像を Lab 色空間へ変換 (float64 にしておく)
        lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab).astype(np.float64)

        # 差分を計算 (H x W x 3)
        diff = lab_img - lab_color

        # ユークリッド距離を求める (H x W)
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

        # 最小値をとるインデックスを 2次元 (row, col) で取得
        min_idx_1d = np.argmin(dist_matrix)
        min_index = np.unravel_index(min_idx_1d, dist_matrix.shape)
        return cast(tuple[int, int], min_index)

    return compute_argmin_lab_distance


def make_retardation_estimation_function(
    color_chart: RGBPicture, color_chart_retardation: D1FloatArray
) -> Callable[[Color], float]:

    def closure(c: Color) -> float:
        f = build_lab_distance_computer(c)
        index = f(color_chart)
        # mat = compute_lab_distance_map(color_chart, c)
        # index = argmin_matrix(mat)
        retardation = color_chart_retardation[index[1]]

        return retardation

    return closure


def make_retardation_color_map_v0(
    img: RGBPicture,
    color_chart: RGBPicture,
    color_chart_reta: D1FloatArray,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> D2FloatArray:

    f = make_retardation_estimation_function(color_chart, color_chart_reta)
    retardation = np.zeros(img.shape[:2])

    for i in tqdm.tqdm(range(retardation.shape[0])):
        progress_callback(i / retardation.shape[0])
        for j in range(retardation.shape[1]):
            retardation[i, j] = f(img[i, j])

    return D2FloatArray(retardation)


def make_retardation_color_map_v1(
    img: RGBPicture,
    color_chart: RGBPicture,
    color_chart_reta: D1FloatArray,
    progress_callback: Callable[[float | None], None] = lambda p: None,
    block_size: int = 1000,  # 行ごとのブロック処理（必要に応じて調整）
) -> D2FloatArray:
    """
    画像とカラーチャートを Lab 色空間に変換し、
    カラーチャート内の各色とのユークリッド距離が最小となる色に対応する retardation 値を
    入力画像の各画素ごとに割り当てた retardation マップを返す関数です。

    ここでは、一度に (H, W, chart_w, 3) のような大きな差分配列を作らず、
    カラーチャート内の各色ごとに逐次的に差分を計算することでメモリ使用量を削減します。
    """
    # 画像とカラーチャートを Lab 色空間に変換
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float64)
    lab_chart = cv2.cvtColor(color_chart, cv2.COLOR_RGB2Lab).astype(np.float64)

    H, W, _ = lab_img.shape

    # ここでは、一般的なカラーチャートが 1 行の場合を例にします。
    # カラーチャートが複数行の場合は、flat にして各色に対して処理してください。
    # plt.imshow(lab_chart)
    # plt.show()
    if lab_chart.shape[0] != 1:
        raise ValueError("この最適化コードは1行のカラーチャートを想定しています。")

    # カラーチャート内の色リスト (chart_w, 3)
    chart_colors = lab_chart[0, :, :]
    chart_w = chart_colors.shape[0]

    # 各画素について、これまでの最小距離 (初期値は無限大) と対応するカラーチャートのインデックス
    best_distance = np.full((H, W), np.inf)
    best_index = np.zeros((H, W), dtype=np.int32)

    # 画像全体をブロック単位で処理する（全体でまとめて計算するとメモリ使用量が大きくなる場合）
    retardation_map = np.empty((H, W), dtype=color_chart_reta.dtype)
    for row_start in tqdm.tqdm(range(0, H, block_size)):
        row_end = min(row_start + block_size, H)
        # 対象ブロック
        lab_img_block = lab_img[row_start:row_end, :, :]
        # 各ブロック内の最小距離と対応インデックスを初期化
        best_distance_block = np.full((row_end - row_start, W), np.inf)
        best_index_block = np.zeros((row_end - row_start, W), dtype=np.int32)

        # カラーチャートの各色について順次計算
        for j in range(chart_w):
            # ブロードキャストによる差分計算: (block_size, W, 3)
            diff = lab_img_block - chart_colors[j]
            # 各画素での二乗ユークリッド距離
            dist_sq = np.sum(diff**2, axis=2)
            # 現在の最小値より小さい画素を更新
            mask = dist_sq < best_distance_block
            best_distance_block[mask] = dist_sq[mask]
            best_index_block[mask] = j

        # カラーチャートのインデックスから retardation 値を割り当てる
        retardation_map[row_start:row_end, :] = color_chart_reta[best_index_block]

        # 進捗コールバック（ブロック処理の場合）
        progress_callback(row_end / H)

    return D2FloatArray(retardation_map)


def make_retardation_color_map_v2(
    img: RGBPicture,
    color_chart: RGBPicture,
    color_chart_reta: D1FloatArray,
    progress_callback: Callable[[float | None], None] = lambda p: None,
    block_size: int = 100,  # 行ごとのブロック処理（必要に応じて調整）
) -> D2FloatArray:
    """
    画像とカラーチャートを Lab 色空間に変換し、
    カラーチャート内の各色とのユークリッド距離が最小となる色に対応する retardation 値を
    入力画像の各画素ごとに割り当てた retardation マップを返す関数です。

    カラーチャートが複数行の場合も対応できるように、
    カラーチャートは平坦化して (総色数, 3) の配列として各色に対して処理を行います。
    """

    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float64)
    lab_chart = cv2.cvtColor(color_chart, cv2.COLOR_RGB2Lab).astype(np.float64)

    H, W, _ = lab_img.shape
    retardation = np.tile(color_chart_reta, H)

    # カラーチャートが複数行の場合も含め、常に平坦化して各色に対して処理を行う
    chart_colors = lab_chart.reshape(-1, 3)
    chart_w = chart_colors.shape[0]

    # 画像全体の各画素に対してこれまでの最小距離（初期値は無限大）と対応するカラーチャートのインデックス
    # ※（以下の best_distance, best_index はコメントアウトしているが、ブロック毎に初期化するため実際には使わない）
    best_distance = np.full((H, W), np.inf)
    best_index = np.zeros((H, W), dtype=np.int32)

    # 画像全体をブロック単位で処理する（全体でまとめて計算するとメモリ使用量が大きくなる場合）
    retardation_map = np.empty((H, W), dtype=retardation.dtype)
    for row_start in tqdm.tqdm(
        range(0, H, block_size), desc="making retardation color map"
    ):
        row_end = min(row_start + block_size, H)
        # 対象ブロック
        lab_img_block = lab_img[row_start:row_end, :, :]
        # 各ブロック内の最小距離と対応インデックスを初期化
        best_distance_block = np.full((row_end - row_start, W), np.inf)
        best_index_block = np.zeros((row_end - row_start, W), dtype=np.int32)

        # カラーチャート内の各色について順次計算
        for j in range(chart_w):
            # ブロードキャストによる差分計算: (block_size, W, 3)
            diff = lab_img_block - chart_colors[j]
            # 各画素での二乗ユークリッド距離を計算
            dist_sq = np.sum(diff**2, axis=2)
            # 現在の最小値より小さい画素のみ更新
            mask = dist_sq < best_distance_block
            best_distance_block[mask] = dist_sq[mask]
            best_index_block[mask] = j

        retardation_map[row_start:row_end, :] = retardation[best_index_block]

        # 進捗コールバック（ブロック処理の場合）
        progress_callback(row_end / H)

    return D2FloatArray(retardation_map)


def make_retardation_color_map(
    img: RGBPicture,
    color_chart: RGBPicture | D1RGB_Array,
    color_chart_reta: D1FloatArray,
    progress_callback: Callable[[float | None], None] = lambda p: None,
    block_size: int = 100,  # 行ごとのブロック処理（必要に応じて調整）
    minR: Optional[float] = None,
    maxR: Optional[float] = None,
) -> tuple[D2FloatArray, D2IntArray, D2IntArray]:
    """
    画像とカラーチャートを Lab 色空間に変換し、
    カラーチャート内の各色とのユークリッド距離が最小となる色に対応する retardation 値を
    入力画像の各画素ごとに割り当てた retardation マップと、
    その際に採用されたカラーチャート上の各色の位置（行・列のインデックス）の配列を返します。

    カラーチャートが複数行の場合も対応できるように、
    カラーチャートは平坦化して (総色数, 3) の配列として各色に対して処理を行います。

    戻り値:
      - retardation_map: 入力画像サイズ (H, W) の各画素に対する retardation 値
      - chart_indices: 入力画像サイズ (H, W, 2) の配列で、各画素に対して
                       カラーチャート内で採用された色の (行, 列) のインデックス
    """

    # color_chartがD1RGBArrayのとき
    if np.ndim(color_chart) == 2:
        color_chart = RGBPicture(np.array([color_chart], dtype=np.uint8))

    color_chart_tmp = deepcopy(color_chart)
    color_chart_reta_tmp = deepcopy(color_chart_reta)

    if minR is not None:
        color_chart = color_chart[:, color_chart_reta >= minR, :]
        color_chart_reta = color_chart_reta[color_chart_reta >= minR]
    if maxR is not None:
        color_chart = color_chart[:, color_chart_reta <= maxR, :]
        color_chart_reta = color_chart_reta[color_chart_reta <= maxR]

    # なんだかうまくいかなくてcolor chartの長さが1以下になったとき
    if len(color_chart[0]) <= 1 or len(color_chart_reta) <= 1:
        color_chart_reta = color_chart_reta_tmp
        color_chart = color_chart_tmp

    # 入力画像とカラーチャートを Lab 色空間へ変換
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float64)
    lab_chart = cv2.cvtColor(color_chart, cv2.COLOR_RGB2Lab).astype(np.float64)

    H, W, _ = lab_img.shape
    # color_chart_reta はカラーチャート内各色に対応する retardation 値の 1 次元配列とする
    # （※ np.tile の結果は 1 次元配列になりますので、各画素に対して平坦なインデックスでアクセスします）
    retardation = np.tile(color_chart_reta, H)  # 元々の配列（平坦なもの）をそのまま利用

    # カラーチャートの元の形状を取得（後で平坦なインデックスを (行, 列) に変換するため）
    chart_h, chart_w = color_chart.shape[:2]

    # plt.imshow(color_chart)
    # plt.show()

    # カラーチャートを平坦化： shape = (総色数, 3)
    chart_colors = lab_chart.reshape(-1, 3)
    num_chart_colors = chart_colors.shape[0]

    # 画像全体の各画素に対して採用されたカラーチャートの平坦なインデックスを格納する配列
    best_index = np.zeros((H, W), dtype=np.int32)
    # ※ best_distance はここでは各ブロック内で利用するのみ

    retardation_map = np.empty((H, W), dtype=retardation.dtype)

    # 画像全体をブロック単位で処理する
    for row_start in tqdm.tqdm(
        range(0, H, block_size), desc="making retardation color map"
    ):
        row_end = min(row_start + block_size, H)
        # 対象ブロック
        lab_img_block = lab_img[row_start:row_end, :, :]
        # ブロック内の各画素についての最小二乗距離と対応する平坦なカラーチャートのインデックスを初期化
        best_distance_block = np.full((row_end - row_start, W), np.inf)
        best_index_block = np.zeros((row_end - row_start, W), dtype=np.int32)

        # カラーチャート内の各色について順次計算
        for j in range(num_chart_colors):
            # ブロードキャストにより差分を計算： shape = (block_height, W, 3)
            diff = lab_img_block - chart_colors[j]
            # 二乗ユークリッド距離を計算
            dist_sq = np.sum(diff**2, axis=2)
            # 現在の最小値より小さい画素のみ更新
            mask = dist_sq < best_distance_block
            best_distance_block[mask] = dist_sq[mask]
            best_index_block[mask] = j

        # ブロック毎に retardation_map と平坦な best_index を更新
        retardation_map[row_start:row_end, :] = retardation[best_index_block]
        best_index[row_start:row_end, :] = best_index_block

        # mask = dist_sq < best_distance_block
        # best_distance_block[mask] = dist_sq[mask]
        # best_index_block[mask] = j
        # retardation_map[row_start:row_end, :] = retardation[best_index_block]

        # 進捗コールバック
        progress_callback(row_end / H)

    # 各画素ごとに、カラーチャート内の (行, 列) のインデックスに変換
    # chart_indices = np.empty((H, W, 2), dtype=np.int32)


    chart_indices_h = best_index // chart_w  # 行方向のインデックス
    chart_indices_w = best_index % chart_w  # 列方向のインデックス

    return (
        D2FloatArray(retardation_map),
        D2IntArray(chart_indices_h),
        D2IntArray(chart_indices_w),
    )


def tilt_retardation_map(
    retardation_map: D2FloatArray,
    deg: float,
    no: float,
    ne: float,
    thickness: Optional[float] = None,
    max_R: Optional[float] = None,
) -> D2FloatArray:

    if thickness is not None:
        if max_R is not None:
            max_R_used = max_R
            thickness_used = thickness
        else:
            max_R_used = get_max_retardation_from_thickness(thickness, no, ne)
            thickness_used = thickness
    else:
        if max_R is not None:
            max_R_used = max_R
            thickness_used = get_thickness_from_max_retardation(max_R)
        else:
            raise TypeError(
                "both thickness is None and max_R is None is not permitted."
            )
    theta_to_R, R_to_theta = make_angle_retardation_estimation_function(
        no=no, ne=ne, thickness=thickness_used
    )

    res = np.zeros_like(retardation_map)
    res[retardation_map > max_R_used] = retardation_map[retardation_map > max_R_used]

    res[retardation_map <= max_R_used] = theta_to_R(
        R_to_theta(retardation_map[retardation_map <= max_R_used]) + np.radians(deg)
    )
    plt.imshow(res)
    plt.show()

    # if deg < 0:
    # else:
    #     res[retardation_map > max_R_used] = retardation_map[
    #         retardation_map > max_R_used
    #     ]
    #     res[retardation_map <= max_R_used] = theta_to_R(
    #         R_to_theta(retardation_map[retardation_map <= max_R_used]) + np.radians(deg)
    #     )
    return res


def estimate_median_alpha_of_nd_filter(
    max_R_color_map: RGBPicture,
    R_array: D1FloatArray,
    nd_array: D1FloatArray,
    R_color_chart: RGBPicture,
) -> tuple[float, int, RGBPicture]:
    pic, h, w = make_retardation_color_map(
        max_R_color_map,
        R_color_chart,
        R_array,
        block_size=100,
    )

    med = np.median(h[h != 0])
    alpha = nd_array[int(med)]
    color_chart_used = R_color_chart[int(med)]
    return alpha, int(med), RGBPicture(np.array([color_chart_used]))


def make_R_map(
    max_R_color_map: RGBPicture, color_chart_1_N_img: RGBPicture, R_array: D1FloatArray
) -> D2FloatArray:
    R_map, _, _ = make_retardation_color_map(
        max_R_color_map,
        color_chart_1_N_img,
        R_array,
        block_size=100,
    )
    return R_map


def make_true_R_color_map(
    R_map: D2FloatArray, color_chart_1_N_img: RGBPicture, R_array: D1FloatArray
) -> RGBPicture:
    true_R_pic = apply_color_map(R_map, R_array, D1RGB_Array(color_chart_1_N_img[0]))
    return true_R_pic


def make_plus_deg_angle_map(
    angle_map: D2FloatArray, degree: float = 45
) -> D2FloatArray:
    angle_map_d = np.zeros_like(angle_map)
    angle_map_d_original = angle_map + np.radians(degree)
    angle_map_d[angle_map - np.radians(degree) < np.pi / 2] = angle_map_d_original[
        angle_map - np.radians(degree) < np.pi / 2
    ]
    angle_map_d[angle_map + np.radians(degree) > np.pi / 2] = (
        angle_map_d_original[angle_map + np.radians(degree) > np.pi / 2] - np.pi / 2
    )
    return angle_map_d


def make_theoritical_tilt_image(
    max_R_color_map: RGBPicture,
    angle_map_0_to_90: D2FloatArray,
    R_array: D1FloatArray,
    angles: D1FloatArray,
    no: float,
    ne: float,
    tilt_deg: float,
    alpha: float,
    color_chart_1_N: RGBPicture,
    thickness: Optional[float] = None,
    max_R: Optional[float] = None,
) -> TheoriticalImage:

    R_map = make_R_map(max_R_color_map, color_chart_1_N, R_array)

    true_R_color_map = make_true_R_color_map(R_map, color_chart_1_N, R_array)
    angle_map = D2FloatArray(angle_map_0_to_90 / 180 * np.pi)

    R_vs_inclination_chart = calc_color_chart(
        R_array,
        angles,
        lambda x, y: get_spectral_distribution(
            osys.get_retardation_system_with_nd_filter(R=x, azimuth=y, nd_filter=alpha)
        )["rgb"],
    )

    # -----------------------------------------------------
    # θ = 0
    # -----------------------------------------------------
    R_minus = tilt_retardation_map(
        R_map, -tilt_deg, no, ne, thickness=thickness, max_R=max_R
    )
    R_plus = tilt_retardation_map(
        R_map, +tilt_deg, no, ne, thickness=thickness, max_R=max_R
    )

    predicted_0_degree = apply_2dcolor_map(
        R_map,
        angle_map,  # angle mapをθ=0として使用する
        R_array,
        angles,
        R_vs_inclination_chart["color_chart"],
    )
    predicted_tilted_minus_0_degree = apply_2dcolor_map(
        R_minus,
        angle_map,
        R_array,
        angles,
        R_vs_inclination_chart["color_chart"],
    )

    predicted_tilted_0_degree = apply_2dcolor_map(
        R_plus,
        angle_map,
        R_array,
        angles,
        R_vs_inclination_chart["color_chart"],
    )

    # -----------------------------------------------------
    # θ = 22.5
    # -----------------------------------------------------
    angle_map_22_5 = make_plus_deg_angle_map(angle_map, 22.5)
    # -----------------------------------------------------
    # θ = 67.5
    # -----------------------------------------------------
    angle_map_67_5 = make_plus_deg_angle_map(angle_map, 67.5)
    # -----------------------------------------------------
    # θ = 45
    # -----------------------------------------------------
    angle_map_45 = make_plus_deg_angle_map(angle_map, 45)

    predicted_22_5_degree = apply_2dcolor_map(
        R_map,
        angle_map_22_5,
        R_array,
        angles,
        R_vs_inclination_chart["color_chart"],
    )

    predicted_67_5_degree = apply_2dcolor_map(
        R_map,
        angle_map_67_5,
        R_array,
        angles,
        R_vs_inclination_chart["color_chart"],
    )

    predicted_45_degree = apply_2dcolor_map(
        R_map,
        angle_map_45,
        R_array,
        angles,
        R_vs_inclination_chart["color_chart"],
    )

    predicted_tilted_minus_45_degree = apply_2dcolor_map(
        R_minus,
        angle_map_45,
        R_array,
        angles,
        R_vs_inclination_chart["color_chart"],
    )

    predicted_tilted_45_degree = apply_2dcolor_map(
        R_plus,
        angle_map_45,
        R_array,
        angles,
        R_vs_inclination_chart["color_chart"],
    )

    return TheoriticalImage(
        max_R_color=true_R_color_map,
        color_0=predicted_0_degree,
        color_22_5=predicted_22_5_degree,
        color_45=predicted_45_degree,
        color_67_5=predicted_67_5_degree,
        R=R_map,
        R_minus=R_minus,
        R_plus=R_plus,
        tilted_d0=predicted_tilted_0_degree,
        tilted_d45=predicted_tilted_45_degree,
        tilted_minus_d0=predicted_tilted_minus_0_degree,
        tilted_minus_d45=predicted_tilted_minus_45_degree,
    )


if __name__ == "__main__":

    import pandas as pd
    from niconavi.image.image import create_outside_circle_mask

    r: ComputationResult = pd.read_pickle(
        "../test/data/output/tetori_4k_xpl_pol_til10_grain.pkl"
    )

    gmap = r.grain_map_with_boundary
    xpl_color_chart = r.color_chart.xpl_retardation_color_chart
    xpl_R_array = r.color_chart.xpl_R_array

    plt.imshow(
        np.array([xpl_color_chart]),
        aspect=50,
        extent=(np.min(xpl_R_array), np.max(xpl_R_array), 0, 1),
    )
    plt.show()

    R_color_map = r.raw_maps["R_color_map"]
    # plt.imshow(r.grain_map)

    R_map = r.raw_maps["max_retardation_map"]
    # R_color_map[R_map > 250] = [255, 255, 255]
    plt.imshow(R_color_map)
    plt.show()

    # mask = create_outside_circle_mask(R_map)
    h = R_map[gmap != 0]
    plt.hist(R_map[gmap != 0], bins=200)
    plt.show()

    # np.sum(h > 100) / (np.prod(h.shape))
    plt.imshow(R_map < 100)
    # )
    # no: float = 1.544
    # ne: float = 1.553
    # max_R: Optional[float] = 400
    # thickness: Optional[float] = None
    # tilt_deg = 4

    # angles: D1FloatArray = D1FloatArray(np.linspace(0, np.pi / 2, num=20))

# %%
