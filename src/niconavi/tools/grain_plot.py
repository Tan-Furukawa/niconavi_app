# %%
import numpy as np
import matplotlib.pyplot as plt
from niconavi.tools.type import D2IntArray, D2BoolArray
from typing import cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import center_of_mass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import center_of_mass, find_objects

import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from typing import Optional
from niconavi.type import ComputationResult
import pandas as pd


def detect_boundaries(index_map: D2IntArray) -> D2BoolArray:
    """
    index_map: 2D numpy array of integers
    Returns a binary mask where boundaries are marked as True
    """
    # Shift the index_map in four directions and compare
    boundary = np.zeros_like(index_map, dtype=bool)

    # Compare with right neighbor
    boundary[:-1, :] |= index_map[:-1, :] != index_map[1:, :]
    # Compare with left neighbor
    boundary[1:, :] |= index_map[:-1, :] != index_map[1:, :]
    # Compare with down neighbor
    boundary[:, :-1] |= index_map[:, :-1] != index_map[:, 1:]
    # Compare with up neighbor
    boundary[:, 1:] |= index_map[:, :-1] != index_map[:, 1:]

    return cast(D2BoolArray, boundary)


def plot_boundaries(
    boundary_mask: D2BoolArray, figsize: tuple[float, float] = (6, 6)
) -> None:
    """
    Plots only the boundary lines.
    """
    plt.figure(figsize=figsize)
    plt.imshow(boundary_mask, cmap="gray")
    plt.title("Boundary Lines Only")
    plt.axis("off")
    plt.show()


def plot_index_map_with_boundaries(
    index_map: D2IntArray,
    boundary_mask: D2BoolArray,
    figsize: tuple[float, float] = (6, 6),
) -> None:
    """
    Plots the index_map with color coding and overlays the boundary lines.
    """
    plt.figure(figsize=figsize)
    # Display the index_map with a colormap
    plt.imshow(index_map, cmap="tab20", interpolation="none")
    # Overlay the boundaries in black
    plt.imshow(boundary_mask, cmap="gray", alpha=0.5)
    plt.title("Index Map with Boundaries")
    plt.axis("off")
    plt.show()


def plot_id_regions(array_2d: np.ndarray):
    """
    2次元のID配列をプロットし、各領域に対応するIDをテキストで描画します。

    Parameters:
    ----------
    array_2d : np.ndarray
        2次元の整数型NumPy配列。各要素は領域のIDを表します。
    """
    if array_2d.ndim != 2:
        raise ValueError("入力配列は2次元である必要があります。")
    if not issubclass(array_2d.dtype.type, np.integer):
        raise ValueError("入力配列は整数型である必要があります。")

    unique_ids = np.unique(array_2d)
    num_ids = len(unique_ids)

    # カラーマップの作成（ユニークなID数に応じて色を割り当てる）
    cmap = plt.get_cmap("tab20", num_ids)
    norm = colors.BoundaryNorm(boundaries=np.arange(num_ids + 1) - 0.5, ncolors=num_ids)

    plt.figure(figsize=(20, 20))
    plt.imshow(array_2d, cmap=cmap, norm=norm)
    # plt.colorbar(ticks=unique_ids, label="ID")

    # 各IDの重心を計算してテキストを配置
    for idx, id_val in enumerate(unique_ids):
        # IDに対応するマスクを作成
        mask = array_2d == id_val
        if not np.any(mask):
            continue  # IDが存在しない場合はスキップ

        # 重心の計算
        cy, cx = center_of_mass(mask)
        if np.isnan(cx) or np.isnan(cy):
            continue  # 重心が計算できない場合はスキップ

        plt.text(
            cx,
            cy,
            str(id_val),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
        )

    # plt.title("ID Regions with Labels")


# plt.axis("off")


# 使用例
if __name__ == "__main__":
    # ランダムなID配列を生成（例として500x500の配列を作成）
    np.random.seed(0)
    sample_array = np.random.randint(1, 101, size=(500, 500))

    r: ComputationResult = pd.read_pickle(
        "../../test/data/output/yamagami_cross_before_grain_classification.pkl"
    )

    # 関数を呼び出してプロット
    grain_map = r.grain_map
    mask = detect_boundaries(grain_map)
    # %%
    plot_id_regions(grain_map)
    plt.imshow(mask, cmap="gray")
    plt.savefig("test.pdf")
