# %%
import numpy as np
from typing import overload, Optional, cast, TypeVar, Protocol, Callable, Any
from niconavi.tools.type import (
    D1,
    D2,
    D3,
    D1IntArray,
    D1BoolArray,
    D1FloatArray,
    D2BoolArray,
)
from typing import Callable, Union
import numpy as np
from numpy.typing import NDArray


T = TypeVar("T", bound=np.number)


def add_num_to_array(
    arr: np.ndarray[Any, np.dtype[T]], num: T
) -> np.ndarray[Any, np.dtype[T]]:
    return cast(np.ndarray[Any, np.dtype[T]], arr + num)


K = TypeVar("K", bound=np.ndarray)


def pick_element_from_array(arr: K, index: D1IntArray | D1BoolArray) -> K:
    return cast(K, arr[index])


def compress_array_by_zero_component(
    array: D2BoolArray,
) -> tuple[D2BoolArray, tuple[int, int]]:
    """
    非ゼロ要素を含む最小の正方形で配列を圧縮し、左上の座標を取得します。

    Parameters:
        array (np.ndarray): 2次元のNumPy配列

    Returns:
        compressed_array (np.ndarray): 圧縮された配列
        top_left (tuple[int, int]): 左上の座標 (行, 列)
    """
    # 0 0 0 1 0 0    0 0 1
    # 0 0 1 1 0 0    0 1 1
    # 0 1 1 1 0 0 -> 1 1 1
    # 0 0 0 1 0 0    0 0 1
    # 0 0 0 0 0 0

    # x - - x: top_left index
    # - - -
    # - - -
    # - - -

    # array = array.astype(np.int_)
    positions = np.argwhere(array != 0)

    if positions.size == 0:
        raise ValueError("there is no non zero component in array")

    min_row, min_col = positions.min(axis=0)
    max_row, max_col = positions.max(axis=0)

    # 高さと幅を計算
    height = max_row - min_row + 1
    width = max_col - min_col + 1

    # 正方形のサイズを決定
    size = max(height, width)

    # 行と列の開始位置と終了位置を調整して正方形にする
    row_start = min_row
    col_start = min_col

    row_end = row_start + size
    col_end = col_start + size

    # 配列の範囲を超えないように調整
    if row_end > array.shape[0]:
        row_end = array.shape[0]
        row_start = row_end - size
    if col_end > array.shape[1]:
        col_end = array.shape[1]
        col_start = col_end - size

    # 圧縮された配列を取得
    compressed_array = array[row_start:row_end, col_start:col_end]

    # 左上の座標を返す
    top_left = (row_start, col_start)

    return cast(D2BoolArray, compressed_array), top_left


def reconstruct_array_by_compressed_array(
    compressed_array: D2BoolArray,
    top_left: tuple[int, int],
    original_shape: tuple[int, int],
) -> D2BoolArray:
    """
    圧縮された配列から元の配列を復元します。

    Parameters:
        compressed_array (np.ndarray): 圧縮された配列
        top_left (Tuple[int, int]): 左上の座標 (行, 列)
        original_shape (Tuple[int, int]): 元の配列の形状

    Returns:
        original_array (np.ndarray): 復元された配列
    """
    original_array = np.zeros(original_shape, dtype=compressed_array.dtype)

    row_start, col_start = top_left
    row_end = row_start + compressed_array.shape[0]
    col_end = col_start + compressed_array.shape[1]

    original_array[row_start:row_end, col_start:col_end] = compressed_array

    return original_array


# --- 使い方の例 -------------------------------------------------------------------
# import numpy as np
# from abc import ABCMeta

# issubclass(np.dtype[np.float64], np.dtype[np.floating])

# class HasAdd(np.generic, metaclass=ABCMeta):
#     pass

# # np.generic の全てのサブクラスを調べる
# for subclass in np.generic.__subclasses__():
#     print(subclass)
#     # __add__ メソッドを持つサブクラスを登録
#     if hasattr(subclass, '__add__'):
#         HasAdd.register(subclass)

# 例として numpy の int64 型を使用
# a = np.int64(10)

# # a が HasAdd のインスタンスであることを確認
# print(isinstance(a, np.signedinteger))  # 出力: True

# # __add__ メソッドを持たない np.void 型の場合
# b = np.void(b'abc')
# print(isinstance(b, HasAdd))  # 出力: False
