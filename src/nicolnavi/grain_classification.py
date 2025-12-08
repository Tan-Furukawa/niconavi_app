# %%
from typing import Callable, Literal, TypedDict, TypeGuard, TypeVar, overload, Any, cast
from niconavi.tools.grain_plot import detect_boundaries
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore
import pandas as pd
from niconavi.tools.type import D1FloatArray, D2IntArray
from niconavi.image.type import RGBPicture, Color
from niconavi.image.image import rgb_float_to_int
from niconavi.type import (
    ComputationResult,
    Grain,
    GrainSelectedResult,
)
from niconavi.tools.select_code_parser import select_grain

T = TypeVar("T")

from matplotlib.colors import to_rgb
import matplotlib.cm as cm
from typing import Dict, Tuple
from numpy.typing import NDArray
from matplotlib.colors import to_rgb


def generate_color_list(num_colors: int, cmap_name: str = "tab20") -> list:
    """
    指定された数の一意な色を生成します。

    Parameters:
        num_colors (int): 必要な色の数。
        cmap_name (str): 使用するカラーマップの名前。デフォルトは 'tab20'。

    Returns:
        color_list (list): RGBカラーのリスト。
    """
    # cmap = plt.colormaps.get_cmap(cmap_name)
    cmap = plt.colormaps.get_cmap(cmap_name)
    color_list = [cmap(i)[:3] for i in range(num_colors)]  # RGBAからRGBを取得
    return color_list


def create_colored_map(
    index_map: D2IntArray,
    index_dict: Dict[str, GrainSelectedResult],
    cmap_name: str = "tab20",
) -> Tuple[RGBPicture, Dict[str, Color]]:
    """
    index_mapとindex_dictを基に色付けされた画像と凡例を作成します。
    colorが指定されているキーはその色を使用し、指定されていないキーは自動生成された色を使用します。

    Parameters:
        index_map (NDArray[np.int_]): 2次元の整数配列 (N, M)
        index_dict (Dict[str, Dict[str, Any]]): キーが文字列、値が {'color': str | None, 'index': list} の辞書
        cmap_name (str): 使用するカラーマップの名前。デフォルトは 'tab20'。

    Returns:
        result (NDArray[np.uint8]): 色付けされた画像データ (N, M, 3)
        legend (Dict[str, Tuple[float, float, float]]): キーとRGB色の対応辞書
    """
    # 色が指定されていないキーの数を数える
    auto_color_keys = [
        key
        for key, val in index_dict.items()
        if val["color"] is None and len(val["index"]) > 0
    ]
    num_auto_colors = len(auto_color_keys)

    # 自動生成色のリストを作成
    if num_auto_colors > 0:
        auto_color_list = generate_color_list(num_auto_colors, cmap_name)
    else:
        auto_color_list = []

    # 凡例の作成: 各キーに色を割り当てる
    legend = {}
    auto_color_idx = 0
    for key in index_dict:
        specified_color = index_dict[key]["color"]
        if specified_color is not None and len(index_dict[key]["index"]) > 0:
            try:
                color_rgb = to_rgb(specified_color)
            except ValueError:
                raise ValueError(f"The color '{specified_color}' is invalid")
            legend[key] = rgb_float_to_int(color_rgb)
        elif len(index_dict[key]["index"]) > 0:
            # 自動生成色を割り当て
            color_rgb = auto_color_list[auto_color_idx]
            legend[key] = rgb_float_to_int(color_rgb)
            auto_color_idx += 1
        else:
            # 'index' が空の場合は色を割り当てない（デフォルトの黒）
            continue

    # 数値から色へのマッピングを作成（後に登場するキーが優先される）
    number_to_color = {}
    for key in index_dict:
        if key not in legend:
            continue  # 'index' が空のキーはスキップ
        color = legend[key]
        for number in index_dict[key]["index"]:
            number_to_color[number] = color  # 後のキーが上書き

    # 結果用の画像配列を初期化（デフォルトは黒）
    N, M = index_map.shape
    result = np.zeros((N, M, 3), dtype=np.uint8)

    # 数値ごとにマスクを作成し、対応する色を割り当てる
    for number, color in number_to_color.items():
        mask = index_map == number
        result[mask] = np.array(color, dtype=np.uint8)

    # legend = rgb_float_to_int(legend)
    return RGBPicture(result), legend


def add_minera_name_to_grain_list(
    grain_list: list[Grain], grain_classification_result: dict[str, GrainSelectedResult]
) -> list[Grain]:
    for grain in grain_list:
        for classification_name, result in grain_classification_result.items():
            if grain["index"] in result["index"]:
                grain["mineral"] = classification_name
                break  # 最初に見つかった分類名で代入したらループを抜ける
    return grain_list


def select_grain_by_code(
    grain_list: list[Grain], code: str
) -> tuple[dict[str, GrainSelectedResult], list[Grain]]:
    grain_classification_result = select_grain(grain_list, code)
    grain_list_new = add_minera_name_to_grain_list(
        grain_list, grain_classification_result
    )
    return grain_classification_result, grain_list_new


# 使用例
if __name__ == "__main__":

    r: ComputationResult = pd.read_pickle(
        "../test/data/output/yamagami_xpl_pol_class.pkl"
    )

    grain_list = r.grain_list

    if r.raw_maps is not None:
        plt.imshow(r.raw_maps["max_retardation_map"])
        plt.colorbar()
        plt.show()

    code = """
        quartz[white]: R < 400
        mica[blue]: R > 700
        background [black]: index == 0
    """

    res, grain_list_new = select_grain_by_code(grain_list, code)
    # len(list(filter(lambda x: x["mineral"] == "mica", grain_list_new)))
    # %%
    index_map = r.grain_map_with_boundary
    index_map = r.grain_map
    # plt.imshow(detect_boundaries(index_map))
    # plt.imshow(detect_boundaries(index_map))

    if index_map is not None and grain_list is not None:
        mask = detect_boundaries(index_map)
        # index_dict = select_grain(grain_list, code)
        index_dict = r.grain_classification_result

        # color_list = generate_color_list(10)

        # 色付けされたマップと凡例の生成
        result, legend = create_colored_map(index_map, index_dict)

        # 結果の表示
        # plt.imshow(result)
        # plt.imshow(mask, cmap="gray", alpha=0.5)
        # plt.show()

    # 凡例の表示
    # print("Legend:")
    # for key, color in legend.items():
    #     print(f"{key}: {color}")

# %%

# def check_res(res):
#     a = []
#     for key in res:
#         a += res[key]["index"]
#     print("============================")
#     print(len(a) != len(set(a)))
#     print("============================")
