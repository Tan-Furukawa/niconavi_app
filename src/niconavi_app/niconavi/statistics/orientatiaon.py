import numpy as np
from typing import Tuple


def orientation_vector(inclination: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """
    inclination, azimuth (ともに shape = (H, W)) から
    shape = (H, W, 3) の 3D単位ベクトルを返す。

    inclination: z軸(法線方向)からの傾斜角 [rad] (2D配列)
    azimuth    : x軸を基準とした方位角 [rad] (2D配列)

    戻り値: orientation_vectors (H, W, 3)
    """
    # shape = (H, W)
    x = np.sin(inclination) * np.cos(azimuth)
    y = np.sin(inclination) * np.sin(azimuth)
    z = np.cos(inclination)
    # shape = (H, W, 3) にまとめる
    return np.stack([x, y, z], axis=-1)


def misorientation_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    2つの3次元単位ベクトル vec1, vec2 (shape=(3,)) の
    ミソリエンテーション角 [rad] を arccos により計算して返す。
    """
    # 内積から arccos
    dot = np.dot(vec1, vec2)
    # 数値誤差で -1～1 を超えないようクリップ
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


def calc_kam(
    inclination: np.ndarray, azimuth: np.ndarray, kernel_radius: int = 1
) -> np.ndarray:
    """
    (1) KAM (Kernel Average Misorientation) を求める。

    各画素 (i, j) について、周囲 kernel_radius の範囲内にあるピクセルとの
    ミソリエンテーション角の平均を KAM として定義。

    例: kernel_radius = 1 であれば、(i,j) を中心とする 3x3 領域 (中心を除く8ピクセル) との
        ミソリエンテーション角の平均。

    引数:
      inclination, azimuth: 2次元配列 (H, W)
      kernel_radius       : 周辺ピクセルをどの範囲まで見るか

    戻り値:
      kam_map: shape = (H, W) の 2次元配列 (KAM [rad])
    """
    H, W = inclination.shape
    # まず 3Dベクトルに変換
    orient_vec = orientation_vector(inclination, azimuth)

    kam_map = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            # (i, j) の周囲を探索
            angles = []
            for di in range(-kernel_radius, kernel_radius + 1):
                for dj in range(-kernel_radius, kernel_radius + 1):
                    if di == 0 and dj == 0:
                        continue  # 中心自身は除外
                    ni = i + di
                    nj = j + dj
                    # 範囲内のみ
                    if 0 <= ni < H and 0 <= nj < W:
                        angle = misorientation_angle(
                            orient_vec[i, j], orient_vec[ni, nj]
                        )
                        angles.append(angle)

            if len(angles) > 0:
                kam_map[i, j] = np.mean(angles)
            else:
                kam_map[i, j] = 0.0

    return kam_map


def calc_gnd(
    inclination: np.ndarray, azimuth: np.ndarray, pixel_size: float = 1.0
) -> np.ndarray:
    """
    (3) GND (Geometrically Necessary Dislocation) 密度の近似指標を求める。

    簡易的に「隣接ピクセルとのミソリエンテーション角の勾配」から
    その画素での GND 相当量を推定する例を示す。

    ここでは、各画素 (i, j) について
      - x方向の隣 (i, j+1) とのミソリエンテーション (dx)
      - y方向の隣 (i+1, j) とのミソリエンテーション (dy)
    を計算し、それらを空間分解能 (pixel_size) で割った値の大きさを指標とする。

    引数:
      inclination, azimuth: 2次元配列 (H, W)
      pixel_size          : 画素 (i,j) と (i,j+1) 間の実空間距離 (同一)

    戻り値:
      gnd_map: shape = (H, W) の 2次元配列
    """
    H, W = inclination.shape
    orient_vec = orientation_vector(inclination, azimuth)
    gnd_map = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            # x方向勾配
            if j < W - 1:
                dx = (
                    misorientation_angle(orient_vec[i, j], orient_vec[i, j + 1])
                    / pixel_size
                )
            else:
                dx = 0.0
            # y方向勾配
            if i < H - 1:
                dy = (
                    misorientation_angle(orient_vec[i, j], orient_vec[i + 1, j])
                    / pixel_size
                )
            else:
                dy = 0.0
            # ここでは sqrt(dx^2 + dy^2) を簡易的な GND の指標とする
            gnd_map[i, j] = np.sqrt(dx * dx + dy * dy)

    return gnd_map
