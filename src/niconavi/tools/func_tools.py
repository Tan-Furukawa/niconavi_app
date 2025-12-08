# %%
import numpy as np
from typing import (
    overload,
    Optional,
    cast,
    TypeVar,
    Protocol,
    Callable,
)
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


A = TypeVar("A")


def synthesize_all_fn_in_array(fn_list: list[Callable[[A], A]]) -> Callable[[A], A]:
    def composed_function(x: A) -> A:
        for fn in fn_list:
            x = fn(x)
        return x

    return composed_function


def convert_array_to_function(
    arr1: D1FloatArray,
    arr2: D1FloatArray,
) -> Callable[[float], Union[None, float]]:
    """
    arr1, arr2 を同じ長さの配列とし、x が (min(arr1), max(arr1)) の範囲内にある場合は
    arr1 と arr2 の対応に基づいて線形補間を行い、その結果を返す関数を生成する。
    それ以外の場合は None を返す。

    Parameters
    ----------
    arr1 : NDArray[float]
        補間の基準となる x 座標の配列
    arr2 : NDArray[float]
        arr1 と対応する y 値の配列
    Returns
    -------
    Callable[[float], None | float]
        (min(arr1), max(arr1)) 内なら線形補間値、範囲外なら None を返す関数
    """
    if len(arr1) != len(arr2):
        raise ValueError("arr1 と arr2 は同じ長さである必要があります。")

    min_val = np.min(arr1)
    max_val = np.max(arr1)

    # np.interp を使う場合、arr1 は昇順である必要があるのでソートしておく
    # arr2 も同じ順番で並び替える
    sort_idx = np.argsort(arr1)
    sorted_arr1 = arr1[sort_idx]
    sorted_arr2 = arr2[sort_idx]

    def interpolation_func(x: float) -> Union[None, float]:
        # (min_val, max_val) に含まれているかチェック (あくまで開区間)
        if x < min_val or x > max_val:
            return None

        # 線形補間した結果を返す
        return float(np.interp(x, sorted_arr1, sorted_arr2))

    return interpolation_func


if __name__ == "__main__":
    # arr1, arr2 (例: 適当なデータ)
    arr1 = cast(D1FloatArray, np.array([3.0, 1.0, 5.0, 2.0]))
    arr2 = cast(D1FloatArray, np.array([9.0, 2.0, 25.0, 4.0]))

    func = convert_array_to_function(arr1, arr2)

    # min(arr1) -> 1.0, max(arr1) -> 5.0 なので、(1.0, 5.0) の開区間内を試す
    print(func(2.5))  # (1.0, 5.0) に入っているので線形補間の値が返る
    print(func(1.0))  # 下限と等しい -> None
    print(func(5.0))  # 上限と等しい -> None
    print(func(6.0))  # 範囲外      -> None
