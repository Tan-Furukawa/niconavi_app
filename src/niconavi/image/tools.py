import numpy as np
from niconavi.tools.type import D1IntArray, D1FloatArray, D2FloatArray, D2IntArray
from niconavi.image.type import Color, D1RGB_Array, RGBPicture
from niconavi.tools.type import D2FloatArray, D1FloatArray
from niconavi.type import GrainDetectionParameters, ComputationResult, ColorChartInfo
import numpy as np
import cv2


def apply_color_map(
    float_map: D2FloatArray,
    standard_val: D1FloatArray,
    color_array: D1RGB_Array,
) -> RGBPicture:
    """
    2次元の float_map 内の各実数について、その値に最も近い standard_val の要素の index を求め、
    その index に対応する color_array の RGB 値を新しいマップとして出力する関数です。

    Parameters
    ----------
    float_map : D2FloatArray
        2次元の浮動小数点数配列 (例: (H, W) の形状)。
    standard_val : D1FloatArray
        標準となる値の1次元配列 (形状: (N,))。
    color_array : RGBArray
        各標準値に対応する色の配列 (形状: (N, 3))。

    Returns
    -------
    np.ndarray
        float_map の各要素に対して色が割り当てられた新しい2次元マップ (形状: (H, W, 3))。
    """
    # float_map の各要素と standard_val の全要素との絶対差を計算するため、
    # float_map の形状を (H, W, 1) に、standard_val の形状を (1, 1, N) に拡張してブロードキャストします。
    diff = np.abs(float_map[..., np.newaxis] - standard_val)

    # 最も差が小さい（＝最も近い）標準値のインデックスを、最終軸に沿って求めます。
    indices = np.argmin(diff, axis=-1)  # shape: (H, W)

    # 得られたインデックスを用いて color_array から RGB 値を取り出すと、
    # shape (H, W, 3) の色マップが得られます。
    color_map = color_array[indices]

    return color_map


def apply_2dcolor_map(
    float_map_x: D2FloatArray,  # (H, W)
    float_map_y: D2FloatArray,  # (H, W)
    standard_x: D1FloatArray,  # (D,)
    standard_y: D1FloatArray,  # (E,)
    color_array: RGBPicture,  # (E, D, 3) と仮定（各 [y, x] に RGB 値が格納）
) -> RGBPicture:
    """
    float_map_x, float_map_y の各ピクセルの値に対して、
    最も近い標準値 standard_x, standard_y のインデックスを求め、
    そのインデックスを用いて color_array（パレット）から色を抜き出し、
    カラーマップ（画像）を作成します。

    Parameters
    ----------
    float_map_x : D2FloatArray
        各ピクセルが standard_x の値またはその近似値を持つ (H, W) の配列。
    float_map_y : D2FloatArray
        各ピクセルが standard_y の値またはその近似値を持つ (H, W) の配列。
    standard_x : D1FloatArray
        x軸方向の標準値の配列 (D,)。
    standard_y : D1FloatArray
        y軸方向の標準値の配列 (E,)。
    color_array : RGBPicture
        パレット画像。形状は (E, D, 3) と仮定し、各 [y, x] に対応する RGB 値が格納される。

    Returns
    -------
    RGBPicture
        float_map_x, float_map_y の各ピクセルに対して色を割り当てた画像 (H, W, 3)。
    """
    # --- x軸方向のインデックス計算 ---
    # float_map_x の各ピクセルの値と standard_x との絶対差を計算
    # float_map_x: (H, W) → (H, W, 1) に拡張し、standard_x: (D,) とブロードキャスト
    diff_x = np.abs(float_map_x[..., np.newaxis] - standard_x)  # shape: (H, W, D)
    indices_x = np.argmin(
        diff_x, axis=-1
    )  # 各ピクセルに対する最小差のインデックス, shape: (H, W)

    # --- y軸方向のインデックス計算 ---
    diff_y = np.abs(float_map_y[..., np.newaxis] - standard_y)  # shape: (H, W, E)
    indices_y = np.argmin(diff_y, axis=-1)  # shape: (H, W)

    # --- カラーパレットから色を抜き出す ---
    # color_array の shape は (E, D, 3) と仮定。各ピクセルで、
    # y 軸のインデックス indices_y、x 軸のインデックス indices_x を用いる。
    colored_map = color_array[indices_y, indices_x]  # 結果は (H, W, 3)

    return colored_map

def extract_h_map(
    target_map: RGBPicture,
    w_map: D2FloatArray,
    color_chart: RGBPicture,
    h_array: D1FloatArray,
    w_array: D1FloatArray
) -> D2FloatArray:
    """
    各画素について、target_mapの色とcolor_chartの候補色をLab色空間上で比較し、
    w_mapの値に最も近いw_arrayに対応する列から、最も近い色に対応するh_arrayの値を返します。

    入力:
        target_map: (A, B, 3) のRGB画像（値域0～255, dtype: uint8）
        w_map: (A, B) の浮動小数点配列
        color_chart: (H, W, 3) のRGB画像（値域0～255, dtype: uint8）。各行がh_array、各列がw_arrayに対応。
        h_array: (H,) の浮動小数点配列
        w_array: (W,) の浮動小数点配列（昇順にソート済みであることを仮定）

    処理の流れ:
      1. target_mapとcolor_chartをcv2.cvtColorを用いてLab色空間に変換します。
         （変換後は内部計算のためにfloat32にキャストします）
      2. 各画素のw_map値に対して、w_arrayから最も近い値のインデックス（列）をnp.searchsortedで高速に求めます。
      3. 同じ列インデックスごとに、対応するcolor_chartの列（すなわち、候補となる全h方向の色）のLab値と、
         対象画素のLab値とのユークリッド距離（二乗距離）をベクトル計算し、最も近い色に対応する行（h）のインデックスを求めます。
      4. そのインデックスに基づいて、h_arrayから対応する値をh_mapに設定します。

    戻り値:
        h_map: (A, B) の浮動小数点配列
    """
    # 画像は0～255のRGB（dtype: uint8）である前提
    # cv2.cvtColorは入力がuint8の場合、内部的に適切な変換を行うのでそのままでOKです。
    target_lab = cv2.cvtColor(target_map, cv2.COLOR_RGB2Lab).astype(np.float32)
    chart_lab  = cv2.cvtColor(color_chart, cv2.COLOR_RGB2Lab).astype(np.float32)
    
    A, B = w_map.shape
    H, W, _ = chart_lab.shape

    # --- ステップ1: w_mapの各画素に対し、w_arrayから最も近い値のインデックスを求める ---
    # w_arrayがソート済みであることを仮定し、np.searchsortedで高速に探索
    pos = np.searchsorted(w_array, w_map)
    left_idx = np.clip(pos - 1, 0, W - 1)
    right_idx = np.clip(pos, 0, W - 1)
    diff_left = np.abs(w_map - w_array[left_idx])
    diff_right = np.abs(w_map - w_array[right_idx])
    idx_w = np.where(diff_left <= diff_right, left_idx, right_idx)  # shape: (A, B)
    
    # --- ステップ2: 各画素ごとに、対応するcolor_chartの列（全h方向）のLab値とtarget_labとの距離を計算 ---
    # 全画素を1次元にまとめ、後で同じ列ごとにグループ処理する
    target_lab_flat = target_lab.reshape(-1, 3)   # shape: (A*B, 3)
    idx_w_flat      = idx_w.reshape(-1)             # shape: (A*B,)
    h_map_flat      = np.empty(A * B, dtype=h_array.dtype)
    
    # 同じw方向の候補列毎に処理を行う
    unique_cols = np.unique(idx_w_flat)
    for col in unique_cols:
        # この列に対応する平坦化後の画素インデックスを取得
        mask = np.where(idx_w_flat == col)[0]
        # color_chartの該当列（全行）のLab値: shape (H, 3)
        candidate_lab = chart_lab[:, int(col), :]
        # 対象画素のLab値: shape (N, 3)
        target_colors = target_lab_flat[mask, :]
        # 各候補色とのユークリッド距離（二乗距離）をベクトル計算
        diffs = target_colors[:, None, :] - candidate_lab[None, :, :]  # shape: (N, H, 3)
        dists = np.sum(diffs ** 2, axis=2)  # shape: (N, H)
        # 各画素について、最も近い候補（h_arrayのインデックス）を取得
        best_h_idx = np.argmin(dists, axis=1)
        # h_arrayから該当する値を取得して格納
        h_map_flat[mask] = h_array[best_h_idx]
    
    # 結果を元の画像サイズに整形
    h_map = h_map_flat.reshape(A, B)
    return D2FloatArray(h_map)
from typing import NewType
import numpy as np
from skimage.color import rgb2lab

# 型定義
RGBPicture = NewType("RGBPicture", np.ndarray)    # shape: (…, 3)
D2FloatArray = NewType("D2FloatArray", np.ndarray)  # 2次元浮動小数点配列
D1FloatArray = NewType("D1FloatArray", np.ndarray)  # 1次元浮動小数点配列

def extract_h_map2(
    target_map: RGBPicture,
    w_map: D2FloatArray,
    color_chart: RGBPicture,
    h_array: D1FloatArray,
    w_array: D1FloatArray
) -> D2FloatArray:
    """
    target_map: (A, B, 3) のRGB画像（値域: 0～255）
    w_map: (A, B) の浮動小数点配列
    color_chart: (H, W, 3) のRGB画像（値域: 0～255）。
                 各行がh_array、各列がw_arrayの値に対応する。
    h_array: (H,) の浮動小数点配列
    w_array: (W,) の浮動小数点配列

    各画素について、以下の処理を行います:
      1. w_mapで指定された値に最も近いw_arrayの値（＝列インデックス）を選ぶ。
      2. その列のcolor_chart内の全ての色と、target_mapの該当画素の色をLab色空間で比較し、
         最も近い色を持つ行インデックスを求める。
      3. その行インデックスに対応するh_arrayの値をh_mapの該当画素の値とする。

    戻り値:
      h_map: (A, B) の浮動小数点配列
    """
    # 入力画像は0～255なので、float32にキャストし、[0,1]に正規化してからLab変換
    target_map_norm = target_map.astype(np.float32) / 255.0
    color_chart_norm = color_chart.astype(np.float32) / 255.0

    target_lab = rgb2lab(target_map_norm)
    chart_lab = rgb2lab(color_chart_norm)

    A, B, _ = target_lab.shape
    H, W, _ = chart_lab.shape

    # 結果を格納する配列
    h_map = np.empty((A, B), dtype=h_array.dtype)

    # 各画素に対して処理
    for i in range(A):
        for j in range(B):
            # w_mapの値に最も近いw_arrayのインデックスを取得
            w_val = w_map[i, j]
            idx_w = int(np.argmin(np.abs(w_array - w_val)))
            
            # 対象のtarget_map画素のLab値
            target_color = target_lab[i, j, :]  # shape (3,)
            # color_chartの該当列 (全行) のLab色
            candidate_colors = chart_lab[:, idx_w, :]  # shape (H, 3)
            # 各候補とのEuclidean距離を計算
            distances = np.linalg.norm(candidate_colors - target_color, axis=1)
            # 最も近い色の行インデックスを取得
            idx_h = int(np.argmin(distances))
            
            # h_arrayから対応するhの値を取得
            h_map[i, j] = h_array[idx_h]

    return D2FloatArray(h_map)

