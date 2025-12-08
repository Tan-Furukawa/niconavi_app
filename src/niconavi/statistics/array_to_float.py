import numpy as np
from typing import cast, Callable
from niconavi.tools.type import (
    D2BoolArray,
    D1BoolArray,
    D2FloatArray,
    D1FloatArray,
    D2IntArray,
    D1IntArray,
)


def circular_variance(x: D1FloatArray, max_x: float | None = None) -> float:
    """
    x: array-like
        データの配列 (0 以上 max_x 以下を想定)
        ただし max_x が None のときは x を「度数法の角度」とみなす

    max_x: float | None
        データ x の最大値 (0 と同一視する境界)。
        max_x が None の場合は x を度数法 [0,360) として扱う。
        max_x が指定されている場合は [0, max_x) を [0, 2π) に写像して扱う。

    戻り値:
        円分散 (circular variance) を返す (0 <= V <= 1)
    """

    if max_x is None:
        # 0～360 の角度データとみなしてラジアン変換
        angles_rad = np.deg2rad(x)
    else:
        # 0～max_x で循環するデータとみなし、[0, 2π) に正規化する
        angles_rad = 2 * np.pi * (x / max_x)

    # --- 2. 円分散を計算する ---
    #     (cos と sin の平均ベクトルを求め、そこから "1 - R" を計算)
    C = np.mean(np.cos(angles_rad))
    S = np.mean(np.sin(angles_rad))
    R = np.sqrt(C**2 + S**2)
    V = 1 - R  # 円分散

    return V


def circular_median_data_via_grid(
    x: list[float], a: float, b: float, n: int = 3600
) -> float:
    """
    4つのデータ x1, x2, x3, x4 が区間 [a, b] に含まれ (a, b は同一視) とき、
    グリッドサーチ(グリッド数 n)で円環上の中央値(サーキュラーメディアン)を求める。

    Parameters
    ----------
    x1, x2, x3, x4 : float
        データ点 (a <= x_i < b)
    a : float
        区間の下限 (円環のスタート)
    b : float
        区間の上限 (円環のエンド) -- ただし a と b が同一視される
    n : int, optional
        [a, b) を等間隔に n 分割して探索 (default: 3600)

    Returns
    -------
    float
        円環上での中央値 (推定値)。区間 [a, b) 内の値。
    """
    # データをまとめる
    xs = np.array(x, dtype=np.float64)

    # 円周の長さ
    circumference = b - a

    # グリッドを生成 (endpoint=False で b は含まない)
    grid = np.linspace(a, b, n, endpoint=False)  # shape: (n, )

    # ----- 以下、ベクトル化で一括計算 -----
    # grid[:, None] で shape (n,1) を作り、xs は shape (4,) -> (1,4)
    # diff.shape は (n,4) となり、全グリッド点と全データ点の差をまとめて計算
    diff = np.abs(grid[:, None] - xs[None, :])
    # 円環上の距離 = min(直線距離, circumference - 直線距離)
    dist = np.minimum(diff, circumference - diff)
    # 各グリッド点について、4点との距離の総和
    dist_sums = dist.sum(axis=1)  # shape: (n, )

    # 総和が最小となるグリッド点のインデックス
    idx_min = np.argmin(dist_sums)

    # そのグリッド点を円環中央値として返す
    circ_median = grid[idx_min]
    return circ_median


def circular_median(
    a: float = 0.0, b: float = 360.0, n: int = 3600
) -> Callable[[D1FloatArray], float]:
    """
    円環区間 [a, b] (a と b は同一視) 上のデータ x に対して、
    サーキュラーメディアンをグリッドサーチで求める関数。

    Parameters
    ----------
    x : array-like
        角度データ (a <= x[i] < b を想定, 個数は任意)
    a : float, optional
        区間の下限 (デフォルト 0)
    b : float, optional
        区間の上限 (デフォルト 360)
        ここでは a と b を同一地点とみなし、円周長 = b - a
    n : int, optional
        [a, b) を n 分割するグリッド数 (デフォルト 3600)

    Returns
    -------
    float
        推定されたサーキュラーメディアン (区間 [a, b) 内の値)
    """
    # データを numpy 配列化

    # 円周の長さ
    circumference = b - a

    # 万が一、x が a,b の範囲外にあっても mod をとって [a,b) に収める (任意)
    # x = ((x - a) % circumference) + a

    # グリッド生成 (endpoint=False により b は含まず [a,b) 区間を n 等分)
    grid = np.linspace(a, b, n, endpoint=False)  # shape: (n, )

    # -- ベクトル化で距離総和を計算して高速化 --
    # diff[i, j] = |grid[i] - x[j]|

    def closure(x: D1FloatArray) -> float | None:
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return None
        diff = np.abs(grid[:, None] - x[None, :])  # shape: (n, len(x))

        # 円環上の距離は min(直線距離, circumference - 直線距離)
        dist = np.minimum(diff, circumference - diff)  # shape: (n, len(x))

        # dist_sums[i] = Σ_j dist[i, j] (i.e. i番目グリッド点での距離総和)
        dist_sums = dist.sum(axis=1)  # shape: (n, )

        # 距離総和が最小となるグリッド点を探す
        idx_min = np.argmin(dist_sums)
        circ_median = grid[idx_min]

        return circ_median

    return closure


def get_True_len(arr: D1BoolArray) -> int | None:
    # arr = arr[~np.isnan(arr)]
    if len(arr) >= 1:
        return np.sum(arr)
    else:
        return cast(None, None)


def sd_extinction_angle_with_nan(arr: D1FloatArray) -> float | None:
    arr = arr[~np.isnan(arr)]
    if len(arr) > 2:
        return circular_variance(arr, 90)
    else:
        return cast(None, None)


def sd_azimuth_with_nan(arr: D1FloatArray) -> float | None:
    arr = arr[~np.isnan(arr)]
    if len(arr) > 2:
        return circular_variance(arr, 180)
    else:
        return cast(None, None)


def median_with_nan(arr: D1FloatArray) -> float | None:
    arr = arr[~np.isnan(arr)]
    if len(arr) > 2:
        return cast(float, np.median(arr))
    else:
        return cast(None, None)

def mean_with_nan(arr: D1FloatArray) -> float | None:
    arr = arr[~np.isnan(arr)]
    if len(arr) > 0:
        return cast(float, np.mean(arr))
    else:
        return cast(None, None)

    # return arr[int(len(arr) / 2)]


def percentile_70_with_nan(arr: D1FloatArray) -> float | None:
    # print(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) > 2:
        return cast(float, np.percentile(arr, 70))
    else:
        return cast(None, None)


def percentile_75_with_nan(arr: D1FloatArray) -> float | None:
    # print(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) > 2:
        return cast(float, np.percentile(arr, 75))
    else:
        return cast(None, None)


def percentile_80_with_nan(arr: D1FloatArray) -> float | None:
    # print(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) > 2:
        return cast(float, np.percentile(arr, 80))
    else:
        return cast(None, None)


def percentile_90_with_nan(arr: D1FloatArray) -> float | None:
    # print(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) > 2:
        return cast(float, np.percentile(arr, 90))
    else:
        return cast(None, None)


def get_ratio_with_nan(arr: D1IntArray) -> float | None:
    arr = arr[~np.isnan(arr)]
    arr = arr[np.bitwise_or(arr == -1, arr == 1)]
    if len(arr) >= 1:
        return cast(float, np.sum(arr == 1) / len(arr))
    else:
        return cast(float, 0.50)

def get_middle_val(arr: D1FloatArray) -> float | None:
    arr = arr[~np.isnan(arr)]
    if len(arr) >= 1:
        return cast(float, arr[int(len(arr) / 2)])
    else:
        return None

if __name__ == "__main__":
    # 例1: max_x=None (度数法として扱う)
    # 同じ角度が並ぶ場合は分散が 0 に近い
    angles_deg = D1FloatArray(np.array([10, 10, 10, 10]))
    print("度数法データ 同じ角度の円分散:", circular_variance(angles_deg, max_x=None))

    # 例2: max_x=None でデータが散らばっている場合
    angles_deg2 = D1FloatArray(np.array([0, 90, 180, 270]))
    print("度数法データ バラバラの円分散:", circular_variance(angles_deg2, max_x=None))

    # 例3: max_x=10 の場合 (0～10 の範囲が 0 と10 でループ)
    data_0to10 = D1FloatArray(np.array([0, 1, 9, 10]))
    print("0～10 が循環するデータの円分散:", circular_variance(data_0to10, max_x=10.0))
