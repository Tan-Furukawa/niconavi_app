# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, Optional
from niconavi.tools.type import D2FloatArray, D2BoolArray, D1FloatArray
from niconavi.image.type import RGBPicture, Color
from niconavi.image.image import get_color_element_by_index
from matplotlib.pyplot import Figure, Axes
from matplotlib.colorbar import Colorbar
from niconavi.image.image import create_outside_circle_mask
from matplotlib.collections import QuadMesh
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
from scipy.ndimage import gaussian_filter


def canonicalize_poles(
    azimuth_deg: D1FloatArray, theta_deg: D1FloatArray
) -> tuple[D1FloatArray, D1FloatArray]:
    """
    極点データ (azimuth, theta) を
        azimuth ∈ [0, 360) かつ theta ∈ [0, 90]
    の一意表現へ変換する。

    (φ, θ) ≡ (φ + 180°, 180° − θ) の等価性を利用して
    上半球へ折り返し、方位角は 360° 周期で正規化する。

    Parameters
    ----------
    azimuth_deg, theta_deg : 1-D array-like
        φ, θ ともに 0–180° の配列（同じ長さ）。

    Returns
    -------
    phi_out, theta_out : ndarray
        φ ∈ [0, 360) と θ ∈ [0, 90] に収まる配列。
    """
    # numpy 配列化（コピー不要なら asarray）
    phi = np.asarray(azimuth_deg, dtype=float)
    th = np.asarray(theta_deg, dtype=float)

    if phi.shape != th.shape:
        raise ValueError("azimuth and theta must have the same shape")

    # --- θ > 90° を折り返して上半球へ ---
    mask = th > 90.0
    th[mask] = 180.0 - th[mask]  # θ' = 180° − θ
    phi[mask] = phi[mask] + 180.0  # φ' = φ + 180°

    # --- 方位角を 0–360° にラップ ---
    phi = np.mod(phi, 360.0)

    return D1FloatArray(phi), D1FloatArray(th)


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

    # 修正：画面上の座標系に直した
    # ラベルの描画（目盛り線の外側に配置）
    # 上：N（上側は tick 線の上、中央に配置）
    axes.text(
        W / 2,
        -tick_length - offset,
        "S",
        ha="center",
        va="bottom",
        color=color,
        clip_on=False,
    )
    # 下：S（下側は tick 線の下、中央に配置）
    axes.text(
        W / 2,
        H - 1 + tick_length + offset,
        "N",
        ha="center",
        va="top",
        color=color,
        clip_on=False,
    )
    # 左：W（左側は tick 線の左、中央に配置）
    axes.text(
        -tick_length - offset,
        H / 2,
        "E",
        ha="right",
        va="center",
        color=color,
        clip_on=False,
    )
    # 右：E（右側は tick 線の右、中央に配置）
    axes.text(
        W - 1 + tick_length + offset,
        H / 2,
        "W",
        ha="left",
        va="center",
        color=color,
        clip_on=False,
    )
    axes.set_facecolor("none")


def get_plot_as_polar_image(
    legend: RGBPicture, background_color: Color = Color(np.array([0, 0, 0]))
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


def make_legend(
    symetry: Literal["90", "180", "360"] = "360",
    color_pattern: Literal["center_black", "center_white"] = "center_black",
) -> RGBPicture:

    sym_val = int(symetry)
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

    return rgb_legend


def make_2d_polar_map(
    inclination: D2FloatArray,
    azimuth: D2FloatArray,
    symetry: Literal["90", "180", "360"] = "360",
    color_pattern: Literal["center_black", "center_white"] = "center_black",
    background_color: Color = Color(np.array([0, 0, 0], dtype=np.uint8)),
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

    rgb_legend = make_legend(
        symetry=symetry,
        color_pattern=color_pattern,
    )
    return rgb_map, rgb_legend


def plot_as_stereo_projection(
    inclination: "D1FloatArray",
    azimuth: "D1FloatArray",
    plot_points: bool = False,
    azimuth_range_max: float = 360,
    n_grid: int = 200,
    sigma: int = 6,
    cmap: str = "jet",
    levels: int = 10,
) -> tuple[Figure, Axes, Colorbar]:
    """ステレオ投影上にカーネル密度推定（等密度補正付き）を描画する。

    Parameters
    ----------
    inclination, azimuth : D1FloatArray
        極点の沈み角（°）と方位角（°）。長さは同じである必要がある。
    plot_points : bool, default False
        投影図上に個々のデータ点を重ねて表示するかどうか。
    azimuth_range_max : float, default 360
        可視化する方位角範囲の上限。90 または 180 を渡すと補助線を追加する。
    n_grid : int, default 200
        KDE 用 2‑D ヒストグラムの分割数。
    sigma : int, default 6
        ガウシアンブラーの標準偏差 (pixel)。0 以下で無効化。
    cmap : str, default "jet"
        等高線のカラーマップ。
    levels : int, default 10
        等高線数。

    Returns
    -------
    fig, ax, cbar : tuple[Figure, Axes, Colorbar]
        Matplotlib オブジェクト。
    """

    # --- 前処理 ------------------------------------------------------------
    azimuth_rad = np.radians(azimuth)
    inclination_rad = np.radians(inclination)

    # 下半球を上半球に反転（Schmidt ネット想定）
    lower = inclination_rad > (np.pi / 2)
    azimuth_reflected = np.where(lower, azimuth_rad + np.pi, azimuth_rad)
    inclination_reflected = np.where(lower, np.pi - inclination_rad, inclination_rad)

    # ステレオ投影の半径 r = tan(θ/2)
    r = np.tan(inclination_reflected / 2)
    x = r * np.cos(azimuth_reflected)
    y = r * np.sin(azimuth_reflected)

    # --- 等面積補正 --------------------------------------------------------
    theta = inclination_reflected
    with np.errstate(divide="ignore"):
        weights = 1.0 / (1.0 + np.cos(theta)) ** 2  # J⁻¹
    weights[~np.isfinite(weights)] = 0.0  # θ=0 で発散する点を 0 とする

    # --- 2‑D ヒストグラム & ガウシアン平滑化 ------------------------------
    edges = np.linspace(-1.0, 1.0, n_grid + 1)
    H, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=weights)

    if sigma > 0:
        H = gaussian_filter(H, sigma=sigma, mode="constant")

    # H /= H.sum()  # ∑=1 に正規化

    # 円外を NaN にして可視化対象外とする
    X, Y = np.meshgrid(np.linspace(-1, 1, H.shape[0]), np.linspace(-1, 1, H.shape[1]))
    H[X**2 + Y**2 >= 1] = np.nan
    H /= np.nansum(H)

    # --- 描画 --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    cf = ax.contourf(Y, X, H, levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf, label="Density")

    if plot_points:
        r_points = np.tan(inclination_rad / 2)
        ax.scatter(
            r_points * np.cos(azimuth_rad),
            r_points * np.sin(azimuth_rad),
            s=1,
            color="white",
        )

    # 方位角範囲が 90 または 180 のとき補助線を追加
    v = np.linspace(-1, 1, 128)
    if azimuth_range_max == 90:
        ax.plot(np.zeros_like(v), v, "--", color="gray")  # N-S 線
        ax.plot(v, np.zeros_like(v), "--", color="gray")  # E-W 線
    elif azimuth_range_max == 180:
        ax.plot(v, np.zeros_like(v), "--", color="gray")

    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    return fig, ax, cbar


# def plot_as_stereo_projection(
#     inclination: D1FloatArray,
#     azimuth: D1FloatArray,
#     plot_points: bool = False,
#     azimuth_range_max: float = 360,
#     n_grid: int = 200,
#     sigma: int = 6,
#     cmap: str = "jet",
#     levels: int = 10
# ) -> tuple[Figure, Axes, Colorbar]:
#     """
#     inclination, azimuth のデータをステレオ投影し、フォン・ミーゼス・フィッシャー分布による
#     カーネル密度推定の結果を描画する。
#     軸や軸ラベルを非表示にし、描画領域は NaN でないセルをぴったり含むようにリサイズする。

#     追加引数:
#       azimuth_range_max: 方位角 0〜azimuth_range_max 度を同一視して使用し、
#                          結果を 0〜azimuth_range_max の扇形として可視化する上限値（既定 360 度）。
#     """

#     # print(np.sum((azimuth > 180)) / len(azimuth))
#     # # print(np.sum((inclination)))
#     # print(np.max(inclination))
#     # print(np.min(inclination))
#     # print(np.max(azimuth))
#     # print(np.min(azimuth))
#     # print(np.sum(inclination < 0.1)/ len(inclination))
#     # print(np.sum((azimuth < 1) & (inclination > 89) ) / len(azimuth))

#     # azimuth, inclination = canonicalize_poles(azimuth, inclination)

#     azimuth = np.radians(azimuth)
#     inclination = np.radians(inclination)

#     # Reflect lower-hemisphere points into upper hemisphere
#     lower = inclination > (np.pi / 2)
#     azimuth_reflected = np.where(lower, azimuth + np.pi, azimuth)
#     inclination_reflected = np.where(lower, np.pi - inclination, inclination)

#     # radius r = tan(θ/2)  (θ: upper-hemisphere polar angle)
#     r = np.tan(inclination_reflected / 2.0)

#     # stereographic plane coordinates  X = r cosφ, Y = r sinφ
#     x = r * np.cos(azimuth_reflected)
#     y = r * np.sin(azimuth_reflected)

#     # --- 2. weights for equal-area compensation ------------------------------
#     # Jacobian J = sinθ / (1+cosθ)²   →  weight w = 1/J
#     theta = inclination_reflected
#     # weights = (1.0 + np.cos(theta)) ** 2 / (np.sin(theta))
#     weights = 1 / (1.0 + np.cos(theta)) ** 2
#     # weights = np.sin(theta/2) * np.cos(theta/2)**3
#     # weights = np.ones(theta.shape)
#     # avoid division blow-up exactly at the pole
#     weights[np.isinf(weights)] = 0.0
#     # plt.scatter(theta, azimuth)

#     # --- 3. 2-D histogram on a square grid -----------------------------------

#     edges = np.linspace(-1.0, 1.0, n_grid + 1)
#     H, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=weights, density=False)


#     # optional Gaussian blur – purely cosmetic
#     if sigma and sigma > 0:
#         H = gaussian_filter(H, sigma=sigma, mode="nearest")

#     # normalise to probability density (∑≃1)
#     H /= H.sum()

#     X, Y = np.meshgrid(np.linspace(-1, 1, H.shape[0]), np.linspace(-1, 1, H.shape[1]))
#     H[X**2 + Y**2 >= 1] = np.nan

#     # --- 4. visualisation -----------------------------------------------------
#     # fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
#     fig, ax = plt.subplots()

#     cmap = ax.contourf(Y, X, H, levels=levels, cmap=cmap)
#     cbar = fig.colorbar(cmap, label="Density")

#     # plot_points=True なら元の (inclination, azimuth) で点を重ねる

#     # 軸やラベルを非表示
#     ax.set_axis_off()
#     # カラーバーも軸を持つので、必要に応じてこちらも消す場合はコメントアウトを解除
#     # cbar.remove()

#     # ステレオ投影の NaN でない部分のみをちょうど含むように描画領域をリサイズ
#     # valid = ~np.isnan(H)
#     # x_valid = X[valid]
#     # y_valid = Y[valid]
#     # if len(x_valid) > 0 and len(y_valid) > 0:
#     #     ax.set_xlim(x_valid.min(), x_valid.max())
#     #     ax.set_ylim(y_valid.min(), y_valid.max())

#     if plot_points:
#         r = np.tan(inclination / 2)
#         x = r * np.cos(azimuth)
#         y = r * np.sin(azimuth)
#         ax.scatter(x, y, s=1, color="white")

#     if azimuth_range_max == 90:
#         v11 = np.linspace(-1, 1, 10)
#         v00 = np.linspace(0, 0, 10)
#         plt.plot(v00, v11, linestyle="--", color="gray")
#         plt.plot(v11, v00, linestyle="--", color="gray")

#     if azimuth_range_max == 180:
#         v11 = np.linspace(-1, 1, 10)
#         v00 = np.linspace(0, 0, 10)
#         plt.plot(v11, v00, linestyle="--", color="gray")
#     # 図全体を余白なくトリミング
#     # （複数ステップを組み合わせることで、極力ムダな余白を削減します）
#     # fig.tight_layout(pad=0)
#     # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

#     # アスペクト比を正方形に
#     ax.set_aspect("equal")

#     return fig, ax, cbar
#     # return ax

if __name__ == "__main__":

    inclination = np.linspace(0, 90, 100)
    azimuth = np.linspace(0, 360, 100)
    plot_as_stereo_projection(inclination, azimuth)

# %%
