# %%
import numpy as np
from typing import Optional, TypedDict
from niconavi.type import Grain
import matplotlib.pyplot as plt


def plot_grain_index(
    grain_list: list[Grain],
    area_shape: tuple[int, int],
    color: str = "black",
    size: float = 5,
) -> tuple[plt.Figure, plt.Axes]:
    """
    grain_list 内の各 Grain の centroid に対して、
    area_shape の範囲内であれば index を color の色で表示する関数。

    引数:
      grain_list: Grain 型の辞書リスト

    戻り値:
      (Figure, Axes) のタプル
    """
    # 図と軸を作成
    fig, ax = plt.subplots()

    # 表示領域の範囲を設定 (ここでは、最初の grain の area_shape を使用)
    if grain_list:
        height, width = area_shape
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # y 軸を反転して画像座標に合わせる

    # 各 Grain に対して、centroid が範囲内なら index をプロット
    for grain in grain_list:
        row, col = grain["centroid"]
        # centroid が area_shape の範囲内かチェック
        if 0 <= row < height and 0 <= col < width:
            ax.text(
                row,
                col,
                str(grain["index"]),
                color=color,
                fontsize=size,
                ha="center",
                va="center",
            )

    ax.set_aspect("equal")
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

    fig, ax = plot_grain_index(
        grain_list=r.grain_list,
        area_shape=r.raw_maps["extinction_angle"].shape,
        color="red",
    )
