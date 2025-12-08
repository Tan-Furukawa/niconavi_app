# %%
import numpy as np
from typing import Optional, TypedDict
from niconavi.type import Grain
import matplotlib.pyplot as plt


def draw_ellipses(grain_list: list[Grain], plot_area: tuple[int, int]) -> np.ndarray:
    """
    grain_list の各 Grain について、以下の条件の楕円を描画する関数です。
      - 楕円の中心は grain['ellipse_center'] (x, y) とする（x: 列, y: 行）
      - 楕円の長軸は x 軸 (正の方向) とのなす角が grain['angle_deg'] (度)
      - 楕円の長軸, 短軸の長さはそれぞれ grain['major_axis_length'], grain['minor_axis_length']

    なお、Grain の各キー (ellipse_center, angle_deg, major_axis_length, minor_axis_length)
    について、全ての Grain で値が None となっている場合は描画できる楕円が一つもないと判断し、
    ValueError を raise します。

    Parameters
    ----------
    grain_list : list[Grain]
        楕円のパラメータを含む Grain のリスト
    plot_area : tuple[int, int]
        出力画像のサイズ (H, W)

    Returns
    -------
    np.ndarray
        描画結果の画像。サイズは plot_area (H, W) で、楕円が描かれた二値画像

    Raises
    ------
    ValueError
        必要なパラメータを持つ楕円が一つもない場合
    """
    required_keys = [
        "ellipse_center",
        "angle_deg",
        "major_axis_length",
        "minor_axis_length",
    ]
    # もし、いずれかのキーについて、全ての Grain が None ならエラー
    for key in required_keys:
        if all(grain.get(key) is None for grain in grain_list):
            raise ValueError(
                f"描画できる楕円が一つもありません: 全ての Grain で '{key}' が None です。"
            )

    # 描画に必要な全パラメータが揃っている Grain のみを対象とする
    valid_grains = [
        grain
        for grain in grain_list
        if all(grain.get(key) is not None for key in required_keys)
    ]
    if not valid_grains:
        raise ValueError("描画できる楕円が一つもありません。")

    H, W = plot_area
    # 出力画像：背景0の二値画像（楕円部分を1に設定）
    image = np.zeros((H, W), dtype=np.uint8)

    for grain in valid_grains:
        center = grain["ellipse_center"]  # (center_x, center_y)
        angle_deg = grain["angle_deg"]  # x軸とのなす角 (度)
        major_axis_length = grain["major_axis_length"]
        minor_axis_length = grain["minor_axis_length"]

        # 画像上では x: 列, y: 行 なので、center は (center_x, center_y)
        center_x, center_y = center

        # 楕円の定義は半径 (a, b) を用いるため、全軸長から半分に
        a = major_axis_length / 2.0  # 半長軸
        b = minor_axis_length / 2.0  # 半短軸

        # 角度をラジアンに変換
        angle_rad = np.deg2rad(angle_deg)

        # 回転楕円のバウンディングボックスを求める
        half_width = abs(a * np.cos(angle_rad)) + abs(b * np.sin(angle_rad))
        half_height = abs(a * np.sin(angle_rad)) + abs(b * np.cos(angle_rad))

        # バウンディングボックスのインデックス (画像範囲内にトリミング)
        x_min = max(int(np.floor(center_x - half_width)), 0)
        x_max = min(int(np.ceil(center_x + half_width)), W - 1)
        y_min = max(int(np.floor(center_y - half_height)), 0)
        y_max = min(int(np.ceil(center_y + half_height)), H - 1)

        # バウンディングボックス内の座標グリッドを作成 (行: y, 列: x)
        ys_grid, xs_grid = np.meshgrid(
            np.arange(y_min, y_max + 1), np.arange(x_min, x_max + 1), indexing="ij"
        )

        # 楕円の中心からのずれ
        x_diff = xs_grid - center_x
        y_diff = ys_grid - center_y

        # 回転させた座標 (x軸方向を楕円の長軸にそろえる)
        x_rot = x_diff * np.cos(angle_rad) + y_diff * np.sin(angle_rad)
        y_rot = -x_diff * np.sin(angle_rad) + y_diff * np.cos(angle_rad)

        # 楕円の内部： (x_rot / a)^2 + (y_rot / b)^2 <= 1
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
        plt.imshow(mask)
        plt.show()

        # 対象領域に楕円を描く（該当ピクセルを 1 に）
        image[y_min : y_max + 1, x_min : x_max + 1][mask] = 1

    return image


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Optional, TypedDict
import numpy as np


def draw_ellipses_matplotlib(
    grain_list: list[Grain], plot_area: tuple[int, int], boundary_color: str = "black"
) -> tuple[plt.Figure, plt.Axes]:
    """
    grain_list の各 Grain について、以下のパラメータを用いて楕円を描画し、
    matplotlib の Figure, Axes を返す関数です。

    描画する楕円は、各 Grain のパラメータ:
      - ellipse_center: 楕円の中心 (x, y)
      - angle_deg: x軸 (正方向) と楕円の長軸とのなす角 (度)
      - major_axis_length: 楕円の長軸の全長
      - minor_axis_length: 楕円の短軸の全長
    を用いています。

    plot_area は (高さ, 幅) を表し、この範囲外はトリミングされます。

    もし、必要なキーのうちどれか一つについて、全ての Grain が None である場合は、
    描画可能な楕円が存在しないとして ValueError を raise します。

    Parameters
    ----------
    grain_list : list[Grain]
        楕円パラメータを含む Grain のリスト
    plot_area : tuple[int, int]
        出力画像のサイズ (高さ, 幅)

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        描画された matplotlib の Figure, Axes オブジェクト
    """
    required_keys = [
        "ellipse_center",
        "angle_deg",
        "major_axis_length",
        "minor_axis_length",
    ]
    # もし各キーについて全 Grain が None ならエラー
    for key in required_keys:
        if all(grain.get(key) is None for grain in grain_list):
            raise ValueError(
                f"描画できる楕円が一つもありません: 全ての Grain で '{key}' が None です。"
            )

    # 必要なパラメータを持つ Grain のみをフィルタリング
    valid_grains = [
        grain
        for grain in grain_list
        if all(grain.get(key) is not None for key in required_keys)
    ]
    if not valid_grains:
        raise ValueError("描画できる楕円が一つもありません。")

    H, W = plot_area  # plot_area: (高さ, 幅)

    # Figure, Axes の作成 (サイズは任意ですが、ここでは plot_area に合わせたサイズ感に調整)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)

    # valid_grains について楕円を描画
    for grain in valid_grains:
        center = grain["ellipse_center"]  # (center_x, center_y)
        angle_deg = 180 - grain["angle_deg"]  # x軸とのなす角 (度)
        major_axis_length = grain["major_axis_length"]
        minor_axis_length = grain["minor_axis_length"]

        # matplotlib.patches.Ellipse のパラメータ:
        #  xy: 楕円の中心, width: 横方向の全長, height: 縦方向の全長, angle: x軸からの回転角(度)
        ellipse = Ellipse(
            xy=center,
            width=major_axis_length,
            height=minor_axis_length,
            angle=angle_deg,
            edgecolor=boundary_color,
            facecolor="none",
            lw=1,
        )
        ax.add_patch(ellipse)

    # Axes の範囲を設定し、plot_area をはみ出た部分はトリミングされるようにする
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # 画像座標系 (y=0 を上部にする)
    ax.set_aspect("equal")
    # ax.set_title("Ellipses")

    return fig, ax


def draw_major_axes_matplotlib(
    grain_list: list[Grain], plot_area: tuple[int, int], line_color: str = "black"
) -> tuple[plt.Figure, plt.Axes]:
    """
    grain_list の各 Grain について、以下のパラメータを用いて長軸（major axis）の線分を描画し、
    matplotlib の Figure, Axes を返す関数です。

    各 Grain では、以下のパラメータを用います:
      - ellipse_center: 長軸の中心 (x, y)
      - angle_deg: x 軸（正方向）と長軸がなす角 (度)
      - major_axis_length: 長軸の全長

    もし、必要なキー (ellipse_center, angle_deg, major_axis_length) のいずれかが
    全ての Grain で None の場合、描画可能な楕円が存在しないとして ValueError を raise します。

    Parameters
    ----------
    grain_list : list[Grain]
        楕円パラメータを含む Grain のリスト
    plot_area : tuple[int, int]
        出力画像のサイズ (高さ, 幅)

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        描画された matplotlib の Figure, Axes オブジェクト
    """
    required_keys = ["ellipse_center", "angle_deg", "major_axis_length"]
    # 各キーについて、全 Grain で None ならエラー
    for key in required_keys:
        if all(grain.get(key) is None for grain in grain_list):
            raise ValueError(
                f"描画できる楕円が一つもありません: 全ての Grain で '{key}' が None です。"
            )

    # 必要なパラメータがすべて有効な Grain のみを対象とする
    valid_grains = [
        grain
        for grain in grain_list
        if all(grain.get(key) is not None for key in required_keys)
    ]
    if not valid_grains:
        raise ValueError("描画できる楕円が一つもありません。")

    H, W = plot_area  # plot_area は (高さ, 幅)

    # Figure, Axes の作成 (ここでは plot_area に合わせたサイズ感に調整)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)

    for grain in valid_grains:
        center = grain["ellipse_center"]  # (center_x, center_y)
        angle_deg = 180 - grain["angle_deg"]  # x軸と長軸のなす角 (度)
        major_axis_length = grain["major_axis_length"]

        center_x, center_y = center

        # 長軸は中心を起点として左右対称なので、両端は中心から major_axis_length/2 離れる
        half = major_axis_length / 2.0

        # 角度をラジアンに変換
        angle_rad = np.deg2rad(angle_deg)

        # 長軸の方向ベクトル (x 軸正方向を基準)
        dx = half * np.cos(angle_rad)
        dy = half * np.sin(angle_rad)

        # 両端の座標 (線分として描画)
        x1, y1 = center_x - dx, center_y - dy
        x2, y2 = center_x + dx, center_y + dy

        # 線分を描画 (青色, 太さ2)
        ax.plot([x1, x2], [y1, y2], color=line_color, lw=1)

    # Axes の範囲を plot_area に合わせる (x: 0〜W, y: 0〜H)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # 画像座標系に合わせる (y=0 が上)
    ax.set_aspect("equal")
    # ax.set_title("Major Axes")

    return fig, ax


if __name__ == "__main__":
    import pandas as pd
    from niconavi.type import GrainDetectionParameters, ComputationResult
    import numpy as np
    import matplotlib.pyplot as plt
    from niconavi.grain_analysis import assign_random_rgb

    r: ComputationResult = pd.read_pickle(
        "../../test/data/output/yamagami_class_inc.pkl"
    )
    plt.imshow(r.grain_segmented_maps["angle_deg"], cmap="hsv")
    plt.colorbar()
    plt.show()
    fig, ax = draw_ellipses_matplotlib(
        r.grain_list, r.raw_maps["extinction_angle"].shape
    )
    fig.show()
    fig, ax = draw_major_axes_matplotlib(
        r.grain_list, r.raw_maps["extinction_angle"].shape
    )
