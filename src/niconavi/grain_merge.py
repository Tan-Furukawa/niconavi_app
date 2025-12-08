# %%
from typing import TypeVar, Callable, Optional, cast
import numpy as np
from collections import defaultdict
from niconavi.tools.type import D1FloatArray, D2IntArray, D1IntArray
from niconavi.image.type import RGBPicture, Color
from niconavi.type import Grain
from niconavi.tools.marge_code_parser import build_function


# 1. Component 型の定義
#
# class Component(TypedDict):
#     index: int
#     val1: float
#     val2: bool
#     # 必要に応じて、追加のキーを定義してください
#     # e.g., val3: str, etc.


#
# 2. Union-Find (Disjoint Set) クラス
#
class UnionFind:
    def __init__(self, elements: D1IntArray):
        # 要素ごとに parent を自分自身に初期化
        self.parent = {e: e for e in elements}
        # 各ルートのサイズ(ランク)管理用
        self.size = {e: 1 for e in elements}

    def find(self, x: int) -> int:
        # 経路圧縮
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        # x, y の根を探す
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            # union by size
            if self.size[root_x] < self.size[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]


#
# 3. merge_regions 関数
#    compare_fn には [Component, Component] -> bool の関数を与える
#    ignore_index で指定した index はどの領域とも絶対にマージされない
#
def merge_regions(
    index_map: D2IntArray,
    data_list: list[dict],
    compare_fn: Callable[[dict, dict], bool],
    ignore_index: Optional[list[int]] = None,
    use_8_neighbors: bool = False,
) -> D2IntArray:
    """
    index_map       : 2次元配列。セルに領域番号 (int) が入っている
    data_list       : [{'index': 0, 'val1': 0.0, 'val2': False}, ...] のリスト (Component のリスト)
    compare_fn      : 2つの Component を受け取り、bool を返す関数
                      True の場合に union(マージ)が実行される
    ignore_index    : マージ対象外にするインデックスのリスト
                      ここに含まれる領域番号は他とマージしない
    use_8_neighbors : True の場合は 8近傍(斜め含む)を隣接とみなす

    戻り値:
        merged_index_map : マージ結果を反映した 2次元配列 (index_map と同サイズ)
    """
    if ignore_index is None:
        ignore_index = []

    # data_list を index => Component の辞書に変換
    # 例: val_dict[1] = {"index": 1, "val1": 10.2, "val2": True}
    val_dict = {comp["index"]: comp for comp in data_list}

    # index_map に登場する全インデックス
    unique_indices = np.unique(index_map)

    # Union-Find の初期化
    uf = UnionFind(unique_indices)

    # 隣接チェックで使うオフセット (4近傍 or 8近傍)
    if use_8_neighbors:
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),  # 上下左右
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),  # 斜め4方向
        ]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    rows, cols = index_map.shape

    # -----------------------------
    # 隣接セルどうしのマージ判定
    # -----------------------------
    for r in range(rows):
        for c in range(cols):
            curr_idx = index_map[r, c]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nei_idx = index_map[nr, nc]

                    # 1) 同じインデックスならスキップ
                    if curr_idx == nei_idx:
                        continue

                    # 2) ignore_index に含まれる領域は絶対にマージしない
                    if curr_idx in ignore_index or nei_idx in ignore_index:
                        continue

                    # 3) compare_fn の結果が True なら union(マージ)
                    #    compare_fn は Component 同士を比較
                    if compare_fn(val_dict[curr_idx], val_dict[nei_idx]):
                        uf.union(curr_idx, nei_idx)

    # 経路圧縮
    for i in unique_indices:
        uf.find(i)

    # 結果を新しい index_map に反映
    merged_index_map = np.zeros_like(index_map)
    for r in range(rows):
        for c in range(cols):
            original_idx = index_map[r, c]
            root_idx = uf.find(original_idx)
            merged_index_map[r, c] = root_idx

    return merged_index_map


def merge_grain_by_code(
    grain_list: list[Grain], grain_map: D2IntArray, code: str
) -> D2IntArray:

    fn_list = [
        {"fn_name": "dist", "exec": lambda arg1, arg2: f"abs({arg1} - {arg2})"},
        {
            "fn_name": "dist90",
            "exec": lambda arg1, arg2: f"min(abs({arg1} - {arg2}), 90 - abs({arg1} - {arg2}))",
        },
        {
            "fn_name": "dist180",
            "exec": lambda arg1, arg2: f"min(abs({arg1} - {arg2}), 180 - abs({arg1} - {arg2}))",
        },
        {"fn_name": "squared", "exec": lambda arg1: f"({arg1}**2 + 50)"},
    ]

    fn = build_function(code, fn_list)
    # print(grain_list[0])
    # print(fn(grain_list[0], grain_list[1]))

    marged_map = merge_regions(
        grain_map,
        grain_list,
        compare_fn=fn,
        use_8_neighbors=False,
        ignore_index=[0],
    )

    return marged_map


# def select_grain_by_code(
#     grain_list: list[Any], code: str
# ) -> Dict[str, GrainSelectedResult]:
#     return select_grain(grain_list, code)


if __name__ == "__main__":

    from niconavi.type import ComputationResult
    import matplotlib.pyplot as plt
    import pandas as pd
    from niconavi.grain_detection import assign_random_rgb

    index_map_example = cast(
        D2IntArray,
        np.array(
            [
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 2],
                [0, 1, 1, 1, 1, 2, 2],
                [3, 3, 3, 1, 1, 1, 2],
                [3, 3, 3, 4, 1, 1, 2],
            ],
            dtype=int,
        ),
    )

    data_list_example = [
        {"index": 0, "val1": 0.0, "val2": False},
        {"index": 1, "val1": 10.2, "val2": True},
        {"index": 2, "val1": 10.5, "val2": False},
        {"index": 3, "val1": 9.2, "val2": True},
        {"index": 4, "val1": 11.1, "val2": True},
    ]

    merged_map_val1 = merge_regions(
        index_map_example,
        data_list_example,
        compare_fn=lambda x, y: abs(x["val1"] - y["val1"]) < 0.5,
        use_8_neighbors=False,
        ignore_index=[0],
    )
    print(merged_map_val1)
    # %%

    r: ComputationResult = pd.read_pickle(
        "../test/data/output/yamagami_cross_before_grain_classification.pkl"
    )
    # サンプル1: "val1" の差が 1.0 未満であればマージする例
    if r.grain_map is not None and r.grain_list is not None:
        # compare_fn の例: 2 つの数値 x, y の差が 1.0 未満かどうか

        # 実行 (4近傍)
        merged_map_val1 = merge_regions(
            r.grain_map,
            r.grain_list,
            compare_fn=lambda x, y: abs(x["R"] - y["R"]) < 300
            and x["R"] > 1000
            or (abs(x["R"] - x["R"]) < 20 and x["R"] < 50),
            use_8_neighbors=False,
            ignore_index=[0],
        )

        print("[val1] の差が 1.0 未満ならマージした結果:")
        plt.imshow(assign_random_rgb(index_map_example, use_color=[(0, (0, 0, 0))]))
        plt.show()

        plt.imshow(assign_random_rgb(merged_map_val1, use_color=[(0, (0, 0, 0))]))
        plt.show()

        if r.raw_maps is not None:
            plt.imshow(r.raw_maps["R_color_map"])
            plt.show()
