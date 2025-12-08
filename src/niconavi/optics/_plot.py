# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
from typing import NewType
import numpy as np
from typing import Callable, Literal, TypeVar, cast, Optional, TypedDict
from niconavi.optics.tools import make_angle_retardation_estimation_function
from niconavi.optics.optical_system import (
    get_full_wave_plus_mineral_retardation_system,
    get_mineral_retardation_system,
)
from niconavi.tools.type import D1FloatArray, D2FloatArray, D3FloatArray, D2BoolArray
from niconavi.type import ColorChartInfo
from niconavi.image.type import RGBPicture, Color
from niconavi.image.image import get_color_element_by_index
from matplotlib.pyplot import Figure, Axes
from matplotlib.colorbar import Colorbar
from niconavi.image.image import create_outside_circle_mask
from matplotlib.collections import QuadMesh
import matplotlib.colors
from niconavi.type import ComputationResult
from copy import deepcopy
from scipy.stats import gaussian_kde
import cv2
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines
import matplotlib.axes as maxes


class D2PolarPlotInfo(TypedDict):
    fig: Figure
    axes: Axes
    margin_width: int


def make_2d_polar_map(
    inclination: D2FloatArray,
    azimuth: D2FloatArray,
    symetry: Literal["90", "180", "360"] = "360",
    color_pattern: Literal["center_black", "center_white"] = "center_black",
    background_color: Color = Color([0, 0, 0]),
    mask: Optional[D2BoolArray] = None,
) -> tuple[RGBPicture, RGBPicture]:

    if mask is None:
        mask = create_outside_circle_mask(inclination)

    # symetry の周期 (度) を整数に変換
    sym_val = int(symetry)

    # --- マップ画像の作成 ---
    # azimuth を周期 sym_val で折り返し、その割合を Hue とする（0～1）
    effective_azimuth = np.mod(-azimuth, sym_val)
    hue = effective_azimuth / sym_val
    # inclination は 0～90 を [0,1] に正規化
    norm_incl = inclination / 90.0

    if color_pattern == "center_black":
        # 中心は黒 (V=0) から外縁 (V=1) に変化、S は固定1
        S = np.ones_like(hue)
        V = norm_incl
    elif color_pattern == "center_white":
        # 中心は白 (S=0) から外縁 (S=1) に変化、V は固定1
        S = norm_incl
        V = np.ones_like(hue)
    else:
        raise ValueError("Invalid color_pattern")

    # OpenCV の uint8 用 HSV 表現は、
    # H: 0～179, S: 0～255, V: 0～255 となるので、各値を変換する
    H_8bit = np.uint8(np.clip(hue * 180, 0, 179))
    S_8bit = np.uint8(np.clip(S * 255, 0, 255))
    V_8bit = np.uint8(np.clip(V * 255, 0, 255))
    hsv_img = cv2.merge([H_8bit, S_8bit, V_8bit])
    # cv2.cvtColor で HSV -> RGB 変換
    rgb_map = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    rgb_map[mask] = background_color
    # if margin_right_width > 0:
    #     margin_area = np.zeros(
    #         (inclination.shape[0], margin_right_width, 3), dtype=np.uint8
    #     )
    #     margin_area[:, :, 0] = background_color[0]
    #     margin_area[:, :, 1] = background_color[1]
    #     margin_area[:, :, 2] = background_color[2]
    #     rgb_map = np.concat([rgb_map, margin_area], axis=1)

    # --- 凡例画像の作成 ---
    # 縦: inclination 0～90°、横: azimuth 0～360° (ただし同一視領域は同一色)
    H, W = 100, 100
    incl_lin = np.linspace(0, 90, H, endpoint=False)
    azim_lin = np.linspace(0, 360, W, endpoint=False)
    incl_grid, azim_grid = np.meshgrid(incl_lin, azim_lin, indexing="ij")

    effective_azim_grid = np.mod(azim_grid, sym_val)
    hue_grid = effective_azim_grid / sym_val
    norm_incl_grid = incl_grid / 90.0

    if color_pattern == "center_black":
        S_grid = np.ones_like(hue_grid)
        V_grid = norm_incl_grid
    elif color_pattern == "center_white":
        S_grid = norm_incl_grid
        V_grid = np.ones_like(hue_grid)
    else:
        raise ValueError("Invalid color_pattern")

    H_grid_8bit = np.uint8(hue_grid * 180)
    S_grid_8bit = np.uint8(S_grid * 255)
    V_grid_8bit = np.uint8(V_grid * 255)
    hsv_legend = cv2.merge([H_grid_8bit, S_grid_8bit, V_grid_8bit])
    rgb_legend = cv2.cvtColor(hsv_legend, cv2.COLOR_HSV2RGB)

    return rgb_map, rgb_legend


def get_plot_as_polar_image(
    legend: RGBPicture, background_color: Color = Color([0, 0, 0])
) -> RGBPicture:
    """
    legend: (H, W, 3) の RGBPicture。縦軸は inclination=0～90°、横軸は azimuth=0～360° に対応。
    この凡例画像を、極座標上に描画し、その結果をimshowで表示できる(H, W, 3)の配列として返す。
    """
    # 入力画像の高さと幅を取得
    H, W = legend.shape[:2]
    # 出力画像を背景色で初期化
    output = np.empty((H, W, 3), dtype=np.uint8)
    output[:] = background_color

    # 出力画像の中心と、描画可能な最大半径（画像内に内接する円）を定義
    center_y = H / 2
    center_x = W / 2
    R = min(center_x, center_y)

    # 画素座標のグリッドを作成
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    # 中心からの相対座標
    dx = j - center_x
    dy = i - center_y
    # 極座標における半径
    r = np.sqrt(dx**2 + dy**2)
    # 円内の画素のマスク（半径 R 以下/領域のみ描画）
    inside = r <= R

    # 円内の各画素について極座標 (r, θ) を計算
    # θ は arctan2 により算出し、[0,360)° に変換
    theta = np.arctan2(dy, dx)
    theta_deg = np.degrees(theta)
    theta_deg = np.mod(theta_deg, 360)

    # 半径 r を入力画像上の縦軸（inclination）の対応に変換
    # r = 0 なら 0° (中心)、r = R なら 90° (外縁)
    # 入力画像の縦軸は [0, H-1] に対応するので
    y_src = (r / R) * (H - 1)
    # θ を入力画像上の横軸（azimuth）に対応させる：
    # 0° -> 0, 360° -> W-1
    x_src = (theta_deg / 360) * (W - 1)

    # 円内領域のみで補間を行うための座標
    y_src_inside = y_src[inside]
    x_src_inside = x_src[inside]

    # 双線形補間のための上下左右の画素インデックスを求める
    y0 = np.floor(y_src_inside).astype(int)
    x0 = np.floor(x_src_inside).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)

    # 補間用の重み（小数部分）
    wy = y_src_inside - y0
    wx = x_src_inside - x0

    # 補間結果を格納する配列（3 チャンネル）
    interp_vals = np.empty((y_src_inside.shape[0], 3), dtype=np.float32)

    # 各色チャネルごとに双線形補間を実施
    for c in range(3):
        top_left = legend[y0, x0, c].astype(np.float32)
        top_right = legend[y0, x1, c].astype(np.float32)
        bottom_left = legend[y1, x0, c].astype(np.float32)
        bottom_right = legend[y1, x1, c].astype(np.float32)
        top = top_left * (1 - wx) + top_right * wx
        bottom = bottom_left * (1 - wx) + bottom_right * wx
        interp_vals[:, c] = top * (1 - wy) + bottom * wy

    # 円内の領域に補間結果を配置
    output[inside] = np.clip(interp_vals, 0, 255).astype(np.uint8)

    return RGBPicture(output)


def plot_as_polar_plot(
    legend: RGBPicture, axis: Literal["off", "all"] = "all"
) -> tuple[Figure, Axes]:
    """
    legend: (H, W, 3) の RGBPicture。縦軸は inclination=0～90°、横軸は azimuth=0～360° に対応。
    この凡例画像を、pcolormesh を用いて極座標上に描画し、Figure と Axes を返す。
    """
    # legend の shape を取得 (H: 半径方向, W: 角度方向)
    H, W, _ = legend.shape

    # 1. legend を [0,1] の float に正規化し、αチャンネルを追加して RGBA データにする
    color_data = legend.astype(np.float32) / 255.0  # (H, W, 3)
    alpha = np.ones((H, W, 1), dtype=color_data.dtype)
    color_data_rgba = np.concatenate([color_data, alpha], axis=-1)  # (H, W, 4)

    # 2. 半径方向 (inclination) と角度方向 (azimuth) の中心値を用意
    #    ※ legend の縦軸は 0～90°、横軸は 0～360° とする
    #    各画素は中心値を持つとみなす
    r_centers = np.linspace(0, 90, H, endpoint=False)  # (H,)
    theta_centers = np.linspace(0, 360, W, endpoint=False)  # (W,)

    # 3. 各中心値から「端点」を計算する関数
    def calc_edges(arr: np.ndarray) -> np.ndarray:
        if len(arr) == 1:
            return np.array([arr[0] - 0.5, arr[0] + 0.5])
        half_diff = np.diff(arr) / 2
        edges = np.concatenate(
            [[arr[0] - half_diff[0]], arr[:-1] + half_diff, [arr[-1] + half_diff[-1]]]
        )
        return edges

    r_edges = calc_edges(r_centers)  # (H+1,)
    theta_edges = calc_edges(theta_centers)  # (W+1,)

    # 4. pcolormesh では角度はラジアン単位なので、theta_edges を変換
    theta_edges_rad = np.deg2rad(theta_edges)

    # 5. theta, r の 2次元座標を作成
    #     np.meshgrid により、theta2d と r2d の shape はそれぞれ (H+1, W+1)
    theta2d, r2d = np.meshgrid(theta_edges_rad, r_edges, indexing="xy")
    # print(r_edges)

    # 6. 極座標サブプロットに pcolormesh で描画
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.pcolormesh(theta2d, r2d, color_data_rgba, shading="auto")
    if axis == "off":
        ax.grid(False)
        ax.set_ylim(0, 90)
        ax.set_rticks([90])  # type: ignore
        ax.set_yticklabels([])  # 数値ラベルは非表示
        # X, Y, Z の文字を追加
        # ax.text(np.deg2rad(0), 95, "X", ha="center", va="center", fontsize=12)
        # ax.text(np.deg2rad(90), 95, "Y", ha="center", va="center", fontsize=12)
        ax.set_thetagrids([0, 90, 180, 270], labels=["", "", "", ""])
        ax.grid(True)

    elif axis == "all":
        # 軸ラベル等の非表示（必要に応じて調整）
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax.set_ylim(0, 90)

    return fig, ax


def plot_polar_filled(
    inclination: D1FloatArray,  # 半径方向 (N,)
    azimth: D1FloatArray,  # 角度方向 (M,)
    color_chart: RGBPicture,  # (M, N, 3) のRGBデータ(0～255)
) -> tuple[Figure, Axes]:
    """
    inclination (N,) と azimth (M,) の格子で定義された色データ (color_chart)
    を pcolormesh を用いて極座標上に表示し、領域を塗りつぶす。
    """

    # -------------------------
    # 1. 入力データの確認・準備
    # -------------------------
    # ※ pcolormeshが期待する座標系に合わせるため、azimthとinclinationをソートすることを推奨
    #   （単調増加しているほうが意図した描画になる）
    azimth = np.sort(azimth)
    inclination = np.sort(inclination)

    # color_chart: (M, N, 3) → [0,1]に正規化し、αチャンネル(1.0)を追加
    # pcolormesh で扱うには shape (N, M, 4) にしておくのが無難
    # (N, M) = (半径方向, 角度方向) の順にしたいので transpose する
    color_data = color_chart.transpose((1, 0, 2))  # => (N, M, 3)
    color_data = color_data / 255.0  # type: ignore
    N, M, _ = color_data.shape
    alpha = np.ones((N, M, 1), dtype=color_data.dtype)
    color_data_rgba = np.concatenate([color_data, alpha], axis=-1)  # => (N, M, 4)

    # -------------------------
    # 2. "端点"座標の生成
    # -------------------------
    # pcolormesh では各ピクセルの四隅(端点)が必要なので、
    # 1次元配列を「端点を含む形」に拡張する。
    def calc_edges(arr: D1FloatArray) -> D1FloatArray:
        # 要素が1つしかない場合などは特別対応が必要かもしれませんが、
        # 基本的には隣接要素の中点を繋いで拡張します。
        if len(arr) == 1:
            # 例として ±0.5 の範囲を持たせるだけにする
            return D1FloatArray(np.array([arr[0] - 0.5, arr[0] + 0.5]))

        half_diff = np.diff(arr) / 2
        edges = np.concatenate(
            [[arr[0] - half_diff[0]], arr[:-1] + half_diff, [arr[-1] + half_diff[-1]]]
        )
        return D1FloatArray(edges)

    r_edges = calc_edges(inclination)  # shape: (N+1,)
    t_edges = calc_edges(azimth)  # shape: (M+1,)

    # -------------------------
    # 3. pcolormesh 用の 2次元座標 (θ, r) を作成
    # -------------------------
    # indexing='xy' を使うと 1つ目の返り値が "y方向" (行数), 2つ目が "x方向" (列数) になります
    # ここで言う y => r, x => θ として扱うと shape は (N+1, M+1) 同士になります
    theta2d, r2d = np.meshgrid(t_edges, r_edges, indexing="xy")  # (N+1, M+1)

    # -------------------------
    # 4. 極座標サブプロットに pcolormesh
    # -------------------------
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    # pcolormesh(θ2D, r2D, C, shading=...) で描画
    # Cのshape: (N, M, 4) にしているので (N+1, M+1) の格子と対応
    pc = ax.pcolormesh(theta2d, r2d, color_data_rgba, shading="auto")
    ax.set_xticklabels([])  # 角度ラベルを非表示
    ax.set_yticklabels([])  # 半径ラベルを非表示
    # ax.set_title("Polar Plot (Filled)")
    # 必要に応じて角度方向や半径方向の範囲を設定
    # ax.set_thetamin(np.degrees(t_edges[0]))
    # ax.set_thetamax(np.degrees(t_edges[-1]))
    # ax.set_rmin(r_edges[0])
    # ax.set_rmax(r_edges[-1])
    # 軸のグリッドを無効にする
    ax.grid(False)

    return fig, ax


def plot_polar(
    inclination: D1FloatArray,  # 半径方向 (N,)
    azimth: D1FloatArray,  # 角度方向 (M,)
    color_chart: RGBPicture,  # (M, N, 3) の RGB カラーデータ
) -> None:
    """
    inclination (N,) と azimth (M,) の格子上にある色データ(color_chart)を
    極座標上に散布図として可視化する。
    """

    # inclination と azimth から 2次元のグリッドを作成
    #   THETA.shape -> (M, N)
    #   R.shape     -> (M, N)
    THETA, R = np.meshgrid(azimth, inclination, indexing="ij")

    # 作成した格子点と色データを、それぞれ 1次元に平坦化
    #   THETA_flat, R_flat -> (M*N,)
    #   color_flat        -> (M*N, 3)
    THETA_flat = THETA.ravel()
    R_flat = R.ravel()
    color_flat = color_chart.reshape(-1, 3).astype(
        np.float64
    )  # または ravel() + 適宜 reshape
    color_flat = (color_flat - np.min(color_flat)) / (  # type: ignore
        np.max(color_flat) - np.min(color_flat)
    )

    # 極座標プロット用のサブプロットを作成
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    # 散布図を色付きで描画
    ax.scatter(THETA_flat, R_flat, c=color_flat, s=10, marker="s")
    plt.title("Polar Plot with RGB Color Chart")
    plt.show()


def make_polar_plot(
    xy_to_color: Callable[[float, float], Color],
    num_inc: int = 100,
    num_azimuth: int = 200,
    no: float = 1.544,
    ne: float = 1.553,
    thickness: float = 0.03,
) -> tuple[Figure, Axes]:
    theta_to_R, R_to_theta = make_angle_retardation_estimation_function(
        no=no, ne=ne, thickness=thickness
    )

    max_R = theta_to_R(np.pi / 2)
    R = D1FloatArray(np.linspace(0, max_R, num_inc))

    inclination = R_to_theta(D1FloatArray(R))
    azimuth = D1FloatArray(np.linspace(0, 2.0 * np.pi, num_azimuth))

    col_chart = calc_color_chart(
        R,
        azimuth,
        xy_to_color,
    )

    return plot_polar_filled(inclination, azimuth, col_chart["color_chart"])


def make_2d_plot(
    xy_to_color: Callable[[float, float], Color],
    num_inc: int = 100,
    num_azimuth: int = 200,
    no: float = 1.544,
    ne: float = 1.553,
    thickness: float = 0.03,
) -> tuple[Figure, Axes]:
    theta_to_R, R_to_theta = make_angle_retardation_estimation_function(
        no=no, ne=ne, thickness=thickness
    )

    max_R = theta_to_R(np.pi / 2)
    R = D1FloatArray(np.linspace(0, max_R, num_inc))

    inclination = D1FloatArray(R_to_theta(D1FloatArray(R)))
    azimuth = D1FloatArray(np.linspace(0, 2.0 * np.pi, num_azimuth))

    col_chart = calc_color_chart(
        R,
        azimuth,
        xy_to_color,
    )

    fig, ax = plt.subplots()

    ax.imshow(
        col_chart["color_chart"],
        extent=(
            np.degrees(inclination.min()),
            np.degrees(inclination.max()),
            np.degrees(azimuth.min()),
            np.degrees(azimuth.max()),
        ),
        aspect=1 / 4,
    )

    return fig, ax


def generate_color_map(
    inclination: D2FloatArray,
    azimuth: D2FloatArray,
    max_inclination: Literal[90, 180] = 180,
    max_azimuth: Literal[90, 180] = 180,
    color_pattern: Literal["center_white", "center_black"] = "center_black",
) -> tuple[RGBPicture, ColorChartInfo]:
    """
    傾斜角 (inclination) と方位角 (azimuth) からカラーマップ画像と、
    横軸が 0〜max_azimuth（方位角）、縦軸が 0〜max_inclination（傾斜角）の
    矩形のカラーチャート（legend）を生成します。

    入力:
      - inclination: 北極(0°)から南極方向の傾斜角（単位: 度）。周期は max_inclination。
      - azimuth: Eを0°として半時計回りの方位角（単位: 度）。周期は max_azimuth。
      - max_inclination: 90 または 180。
          （例: max_inclination=90 の場合、x と x+90° は同一色）
      - max_azimuth: 90 または 180。
          （例: max_azimuth=90 の場合、0° と 90° は同一色）

    仕様:
      - カラーマップは HSV 空間で、
            Hue = (azimuth mod max_azimuth) / max_azimuth,
            Saturation = sin((inclination mod max_inclination)*π/max_inclination),
            Value = 1
        として生成（HSV→RGB変換）。
      - legend（カラーチャート）は、縦軸を傾斜角 0〜max_inclination、
        横軸を方位角 0〜max_azimuth とする矩形チャートを作成し、
            Hue = (方位角/max_azimuth),
            Saturation = sin(傾斜角 * π/max_inclination),
            Value = 1
        として対応する色を描きます。
      - max_inclination=180 かつ max_azimuth=90 はエラーとします。

    出力:
      - (colormap, legend_info) のタプル。
          colormap は入力配列と同じ形状の (N, M, 3) uint8 型 RGB 画像、
          legend_info は ColorChartInfo 型の辞書（h, w, chart）です。
    """
    # --- カラーマップ画像の生成 ---
    # 入力角度を周期で折り返す
    i_mod = np.mod(inclination, max_inclination)
    a_mod = np.mod(azimuth, max_azimuth)

    # HSV値の設定
    H = a_mod / max_azimuth
    S = np.sin(i_mod * np.pi / max_inclination)
    V = np.ones_like(H)
    HSV = np.stack([H, S, V], axis=-1)
    rgb_float = matplotlib.colors.hsv_to_rgb(HSV)
    colormap = (rgb_float * 255).astype(np.uint8)

    # --- カラーチャート（legend）の生成 ---
    if max_inclination == 180 and max_azimuth == 90:
        raise ValueError(
            "max_inclination=180 かつ max_azimuth=90 の組み合わせは許されません。"
        )

    # 解像度の設定
    # 縦軸は固定解像度（例: 300 ピクセル）、
    # 横軸は角度比に合わせて決定します。
    legend_h = 300
    legend_w = int(legend_h * (max_azimuth / max_inclination))

    # 各軸の角度配列を作成（0～max_inclination, 0～max_azimuth）
    h_vals = np.linspace(0, max_inclination, legend_h, dtype=np.float64)
    w_vals = np.linspace(0, max_azimuth, legend_w, dtype=np.float64)

    # 各座標での色を計算するためのメッシュグリッド（縦:傾斜角, 横:方位角）
    H_grid, W_grid = np.meshgrid(
        h_vals, w_vals, indexing="ij"
    )  # shape: (legend_h, legend_w)
    # HSV 値の計算
    H_chart = W_grid / max_azimuth  # Hue: 0〜1
    S_chart = np.sin(H_grid * np.pi / max_inclination)  # Saturation: sin(角度)
    V_chart = np.ones_like(H_chart)  # Value: 1

    HSV_chart = np.stack([H_chart, S_chart, V_chart], axis=-1)
    rgb_chart_float = matplotlib.colors.hsv_to_rgb(HSV_chart)
    chart = (rgb_chart_float * 255).astype(np.uint8)

    legend_info = ColorChartInfo(
        what_is_h="inclination",
        what_is_w="azimuth",
        h=D1FloatArray(h_vals),
        w=D1FloatArray(w_vals),
        color_chart=RGBPicture(chart),
    )

    return RGBPicture(colormap), legend_info


# # テスト例
# print(normalize_axes(45, 10))  # (45, 10) -> すでに範囲内
# print(normalize_axes(-45, 30))  # (-45, 30) -> (45, 210)
# print(normalize_axes(100, 0))  # (100, 0) -> (80, 180)
# print(normalize_axes(200, 10))  # (200, 10) -> (20, 10)

def stereo_projection(
    inclination: D1FloatArray, azimuth: D1FloatArray
) -> tuple[D1FloatArray, D1FloatArray]:
    """
    入力:
      inclination: 1次元配列。各要素は 0～90 度（北極からの角度）
      azimuth: 1次元配列。各要素は 0～360 度
    出力:
      各 (inclination, azimuth) に対応するステレオ投影後の (x, y) 座標
      （出力は入力と同じ shape の np.ndarray）

    ステレオ投影の式（単位球の場合）:
      r = tan(inclination/2)
      x = r * sin(azimuth)
      y = r * cos(azimuth)
    （角度はラジアン単位で計算）
    """
    # 角度をラジアンに変換
    incl_rad = np.deg2rad(inclination)
    az_rad = np.deg2rad(azimuth)

    # ステレオ投影の半径 r を計算
    r = np.tan(incl_rad / 2)

    # デカルト座標 (x, y) を計算
    x = r * np.sin(az_rad)
    y = r * np.cos(az_rad)

    return x, y


def calc_edges(arr: D1FloatArray) -> D1FloatArray:
    """
    1次元配列から pcolormesh 用のエッジ配列を計算する補助関数
    """
    if len(arr) == 1:
        return D1FloatArray(np.array([arr[0] - 0.5, arr[0] + 0.5]))
    half_diff = np.diff(arr) / 2
    edges = np.concatenate(
        ([arr[0] - half_diff[0]], arr[:-1] + half_diff, [arr[-1] + half_diff[-1]])
    )
    return edges


def compute_kde(
    x: D1FloatArray, y: D1FloatArray, bandwidth: float = 0.01
) -> gaussian_kde:
    """
    与えられた x, y 座標からカーネル密度推定 (KDE) を計算する
    """
    data = np.vstack([x, y])
    kde = gaussian_kde(data, bw_method=bandwidth)
    return kde


def compute_density_on_grid(
    kde: gaussian_kde, r_vals: D1FloatArray, theta_vals: D1FloatArray
) -> tuple[D2FloatArray, D2FloatArray, D2FloatArray]:
    """
    KDE を用いて極座標グリッド上の密度を評価する
    """
    theta_grid, r_grid = np.meshgrid(theta_vals, r_vals, indexing="xy")
    # 極座標 (r, θ) → Cartesian (x, y) 変換
    x_grid = r_grid * np.sin(theta_grid)
    y_grid = r_grid * np.cos(theta_grid)
    density = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(r_grid.shape)
    return D2FloatArray(density), D2FloatArray(theta_grid), D2FloatArray(r_grid)


def create_discrete_cmap(num_colors: int = 20) -> LinearSegmentedColormap:
    """
    原色緑から原色赤への離散的なカラーマップを作成する
    """
    colors = [(0, 0, 0.8), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # 緑 → 赤
    cmap = LinearSegmentedColormap.from_list("green_red", colors, N=num_colors)
    return cmap


def plot_density_with_discrete_colors(
    ax: Axes,
    theta_edges: D1FloatArray,
    r_edges: D1FloatArray,
    density: D2FloatArray,
    num_colors: int = 20,
) -> QuadMesh:
    """
    KDE の密度データを離散的なカラーマップで描画し、離散色の境界に沿って等高線を追加する
    """
    # 密度の最小・最大値から離散レベルを決定
    vmin, vmax = density.min(), density.max()
    levels = np.linspace(vmin, vmax, num_colors + 1)
    cmap = create_discrete_cmap(num_colors)
    norm = BoundaryNorm(boundaries=levels, ncolors=num_colors)

    # pcolormesh による離散色の描画
    pcm = ax.pcolormesh(
        theta_edges,
        r_edges,
        density,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )

    # エッジから各セルの中心座標を計算（等高線描画用）
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    theta_centers_grid, r_centers_grid = np.meshgrid(
        theta_centers, r_centers, indexing="xy"
    )
    ax.contour(
        theta_centers_grid,
        r_centers_grid,
        density,
        levels=levels[1:-1],
        colors="k",
        linewidths=0.5,
    )

    return pcm



def show_img_with_direction_ticks(
    axes: maxes.Axes, img: RGBPicture, color: str = "black"
) -> None:
    """
    axes: matplotlib.axes.Axes
    img: (H, W, 3) の RGBPicture

    この関数は、img を axes.imshow(img) で表示し、
    ・枠（スパイン）や軸目盛りは表示しない
    ・描画領域の４辺の中点から、それぞれ外向きに１本の目盛り線を描画
    ・各目盛りに対して、その方向（上：N, 下：S, 右：E, 左：W）のラベルを描画
    する。
    """
    # 画像表示
    axes.imshow(img)
    H, W = img.shape[:2]

    # 軸目盛りとスパイン（枠）を非表示にする
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    # 目盛り線の長さ（画像サイズに対する割合）
    tick_length = 0.05 * min(W, H)
    offset = 2  # ラベルとの隙間（ピクセル）

    # 目盛り線は画像外にも描画できるよう clip_on=False を指定

    # 上側（上中点：N）
    top_center = (W / 2, 0)
    top_tick_end = (W / 2, -tick_length)
    line_top = mlines.Line2D(
        [top_center[0], top_tick_end[0]],
        [top_center[1], top_tick_end[1]],
        color=color,
        clip_on=False,
    )
    axes.add_line(line_top)

    # 下側（下中点：S）
    bottom_center = (W / 2, H + 1)
    bottom_tick_end = (W / 2, H + 1 + tick_length)
    line_bottom = mlines.Line2D(
        [bottom_center[0], bottom_tick_end[0]],
        [bottom_center[1], bottom_tick_end[1]],
        color=color,
        clip_on=False,
    )
    axes.add_line(line_bottom)

    # 左側（左中点：W）
    left_center = (0, H / 2)
    left_tick_end = (-tick_length, H / 2)
    line_left = mlines.Line2D(
        [left_center[0], left_tick_end[0]],
        [left_center[1], left_tick_end[1]],
        color=color,
        clip_on=False,
    )
    axes.add_line(line_left)

    # 右側（右中点：E）
    right_center = (W + 1, H / 2)
    right_tick_end = (W + 1 + tick_length, H / 2)
    line_right = mlines.Line2D(
        [right_center[0], right_tick_end[0]],
        [right_center[1], right_tick_end[1]],
        color=color,
        clip_on=False,
    )
    axes.add_line(line_right)

    # ラベルの描画（目盛り線の外側に配置）
    # 上：N（上側は tick 線の上、中央に配置）
    axes.text(
        W / 2,
        -tick_length - offset,
        "N",
        ha="center",
        va="bottom",
        color=color,
        clip_on=False,
    )
    # 下：S（下側は tick 線の下、中央に配置）
    axes.text(
        W / 2,
        H - 1 + tick_length + offset,
        "S",
        ha="center",
        va="top",
        color=color,
        clip_on=False,
    )
    # 左：W（左側は tick 線の左、中央に配置）
    axes.text(
        -tick_length - offset,
        H / 2,
        "W",
        ha="right",
        va="center",
        color=color,
        clip_on=False,
    )
    # 右：E（右側は tick 線の右、中央に配置）
    axes.text(
        W - 1 + tick_length + offset,
        H / 2,
        "E",
        ha="left",
        va="center",
        color=color,
        clip_on=False,
    )
    axes.set_facecolor("none")


def add_polar_legend(ax: Axes, legend: RGBPicture, color: str = "black") -> None:
    # fig, ax = plt.subplots()
    # ax.imshow(cmap)

    ax_inset = inset_axes(
        ax,
        width="40%",
        height="40%",
        loc="center left",
        bbox_to_anchor=(0.9, 0.65, 0.5, 0.5),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    inset_img = get_plot_as_polar_image(legend)
    show_img_with_direction_ticks(ax_inset, inset_img, color=color)

    # return ax


if __name__ == "__main__":
    from niconavi.optics.uniaxial_plate import (
        calc_color_chart,
        get_spectral_distribution,
        plot_color_chart,
    )
    import matplotlib.pyplot as plt
    from niconavi.optics.tools import make_angle_retardation_estimation_function
    import pandas as pd

    r: ComputationResult = pd.read_pickle("../../test/data/output/tetori_class_inc.pkl")

    # test/data/output/tetori_4k_xpl_pol_til.pkl_classified.pkl
    # params: ComputationResult = pd.read_pickle(
    #     "../test/data/output/tetori_4k_xpl_pol_til10.pkl"
    # )
    # fig, ax = make_polar_plot(
    #     lambda x, y: get_spectral_distribution(
    #         get_full_wave_plus_mineral_retardation_system(R=x, azimuth=y)
    #         # get_mineral_retardation_system(R=x, azimuth=y)
    #     )["rgb"],
    #     num_azimuth=100,
    #     num_inc=20,
    #     thickness=0.03,
    # )

    if r.raw_maps is not None:
        inclination = r.raw_maps["inclination"]
        azimuth = r.raw_maps["azimuth360"]

        inclination[:200, :200] = 45
        azimuth[:200, :200] = 90

        cmap, legend = make_2d_polar_map(
            inclination, azimuth, symetry="360", color_pattern="center_white"
        )
        fig, ax = plt.subplots()
        ax.imshow(cmap)

        # add_polar_legend(ax=ax, legend=legend)
        # # n_inclination, n_azimuth = normalize_axes(inclination, azimuth)

        # # ax_inset.set_title("Plot B", fontsize=8)
        # # ax_inset.legend(fontsize=6)
        # mask = r.grain_map_with_boundary != 0
        # plt.imshow(mask)
        # plt.show()
        # i, a = r.cip_map_info["polar_info360"]["quartz"]
        # i = inclination[mask]
        # a = azimuth[mask]
        # np.random.seed(100)
        # i = np.random.choice(i, 1000)
        # a = np.random.choice(a, 1000)
        # i = np.random.random(100) + 80
        # a = np.random.random(100) + 45
        # plot_as_stereo_projection(
        #     i,
        #     a,
        #     kappa=0.1,
        # )
        # plt.show()
        # # plot_as_stereo_projection(i, a)
        # plt.hist(r.cip_map_info["polar_info360"][0])
        # plt.show()

        # # plt.imshow(cmap, cmap="hsv")
        # # plt.colorbar()
        # plt.show()

# %%
