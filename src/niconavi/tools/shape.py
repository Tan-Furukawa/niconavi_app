# %%
import numpy as np
from typing import TypeAlias
from niconavi.tools.type import D1FloatArray, D1IntArray, D2BoolArray

import numpy as np


def compute_ellipse_params(area: np.ndarray) -> tuple:
    """
    与えられた2次元のブール配列 area の True 領域を楕円近似したときの
    離心率, 長軸と x 軸のなす角(度, 0°は x 軸, 0〜180の範囲),
    長軸の長さ, 短軸の長さ, そして楕円の中心のインデックス (center_x, center_y)
    をタプルで返す関数。

    Parameters
    ----------
    area : np.ndarray (dtype=bool)
        True が領域を表す2次元配列

    Returns
    -------
    tuple
        (eccentricity, angle, major_axis_length, minor_axis_length, (center_x, center_y))
    """
    # True のピクセル座標を取得 (y座標: 行, x座標: 列)
    ys, xs = np.where(area)

    # 対象領域がない場合は全て0を返す
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0, (0, 0))

    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    # 重心（中心）の計算
    x_c = np.mean(xs)
    y_c = np.mean(ys)

    # 重心からのずれ
    x_diff = xs - x_c
    y_diff = ys - y_c

    # 共分散行列の成分を計算
    s_xx = np.mean(x_diff * x_diff)
    s_yy = np.mean(y_diff * y_diff)
    s_xy = np.mean(x_diff * y_diff)

    cov_matrix = np.array([[s_xx, s_xy], [s_xy, s_yy]], dtype=np.float64)

    # 固有値・固有ベクトルを計算
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues = np.abs(eigenvalues)  # 数値誤差対策

    # 最大・最小固有値を取得
    lambda_max = np.max(eigenvalues)
    lambda_min = np.min(eigenvalues)

    # 最大固有値に対応する固有ベクトル（長軸の方向）
    idx_max = np.argmax(eigenvalues)
    v_max = eigenvectors[:, idx_max]

    # 長軸と x 軸のなす角 (度に変換)
    angle_rad = np.arctan2(v_max[1], v_max[0])
    angle_deg = np.degrees(angle_rad)
    # 角度を [0, 180) に正規化
    if angle_deg < 0:
        angle_deg += 180
    elif angle_deg >= 180:
        angle_deg -= 180

    angle_deg = (180 - angle_deg) % 180
    # 点が全て同一点の場合
    if lambda_max == 0.0:
        return (0.0, angle_deg, 0.0, 0.0, (x_c, y_c))

    # 離心率の計算: e = sqrt(1 - lambda_min / lambda_max)
    eccentricity = np.sqrt(1.0 - (lambda_min / lambda_max))

    # 楕円の軸の長さの計算
    # 均一な密度の楕円では、固有値は (軸長/4)^2 に対応するため
    major_axis_length = 4 * np.sqrt(lambda_max)
    minor_axis_length = 4 * np.sqrt(lambda_min)

    return (eccentricity, angle_deg, major_axis_length, minor_axis_length, (x_c, y_c))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # テスト用の例
    # 適当な楕円形状を作る
    h, w = 100, 150
    y_center, x_center = 50, 75
    # 楕円の半径
    ry, rx = 20, 20

    Y, X = np.indices((h, w))
    ellipse_area = (
        (X - x_center) ** 2 / (rx**2) + (Y - y_center) ** 2 / (ry**2)
    ) <= 1.0
    plt.imshow(ellipse_area)

    e = compute_ellipse_params(ellipse_area)
    # print("Eccentricity:", e)
