# %%
import matplotlib.pyplot as plt
from typing import cast, Callable, Optional, TypeVar, TypedDict
from niconavi.image.image import (
    rotate_array,
    resize_img,
    resize_image_list,
    median_filter,
    resize_array,
)
from niconavi.image.type import RGBPicture, MonoColorPicture
from niconavi.retardation_normalization import (
    make_retardation_color_map,
    estimate_median_alpha_of_nd_filter,
)
from niconavi.tools.type import (
    D2IntArray,
    D2FloatArray,
    D2BoolArray,
)
from niconavi.type import ComputationResult, TiltImageResult, OpticalParameters, RawMaps
from niconavi.optics.uniaxial_plate import (
    get_retardation_color_chart,
    get_spectral_distribution,
    get_retardation_color_chart_with_nd_filter,
    make_uniaxial_color_chart,
    ColorChartInfo,
)
from niconavi.optics.optical_system import get_full_wave_plus_mineral_retardation_system

from niconavi.tools.read_data import divide_video_into_n_frame

import pandas as pd
import numpy as np
import cv2


class ColorChartInfosOfTiltedImage(TypedDict):
    horizontal: ColorChartInfo
    plus_tilted: ColorChartInfo
    minus_tilted: ColorChartInfo


def get_focus_index(images: list[RGBPicture]) -> D2IntArray:
    """
    複数の画像から、各ピクセルごとにもっともピントの合った部分を抜き出し、
    1枚の合成画像 (result) と、各ピクセルにおける採用画像のインデックスマップ (selected_idx_map) を返す。

    Parameters
    ----------
    images : list of np.ndarray
        ピント位置が異なる複数の画像 (BGR, 3チャンネル)。
        全て同じサイズであることを仮定。

    Returns
    -------
    result : np.ndarray
        焦点合成された画像（BGR, 3チャンネル）
    selected_idx_map : np.ndarray
        2次元配列 (height, width)。各ピクセルで採用された画像のインデックスが格納される。
    """

    if len(images) == 0:
        raise ValueError("画像リストが空です。")

    # 各画像をグレースケール化 -> Laplacianでシャープネスを測る
    measure_stack = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        measure = np.abs(lap)
        measure_stack.append(measure)

    # (N, H, W)の3次元配列にスタック
    measure_stack = np.stack(measure_stack, axis=0)

    # 各ピクセルにおいて最大シャープネスを示す画像のインデックスを取得
    # max_idx.shape => (height, width)
    selected_idx_map = np.argmax(measure_stack, axis=0).astype(np.uint8)

    return selected_idx_map


def ransac_plane_fitting_2d(
    image: D2FloatArray,
    max_iterations: int = 200,
    distance_threshold: float = 5.0,
    min_inlier_ratio: float = 0.5,
    random_state: int = 42,
) -> tuple[D2FloatArray, float, float]:
    """
    scikit-learn を使わずに RANSAC によるロバストな平面フィッティングを行う。
    入力画像の (x, y) 座標と画素値 z について、
    z = a*x + b*y + c の形の平面を推定する。

    さらに、推定された平面を「最も急に登る方向」のベクトルについて、
    - その x–y 平面への射影が x 軸となす角 (ラジアン)
    - そのベクトルが x–y 平面となす角 (ラジアン)

    を計算し、まとめて返す。

    Parameters
    ----------
    image : np.ndarray
        2次元配列 (整数) 画像
    max_iterations : int
        RANSAC の最大反復回数
    distance_threshold : float
        インライヤー判定の閾値（予測 z と実際の z の絶対差がこれ以下ならインライヤー）
    min_inlier_ratio : float
        モデル更新時に必要なインライヤーの割合。例えば 0.5 なら、
        全データの 50% 以上がインライヤーなら、そのモデルを暫定最良解とする
    random_state : int
        乱数シード

    Returns
    -------
    fitted_image : np.ndarray
        フィッティングされた平面 z を 2次元配列にしたもの (浮動小数)
    angle_with_x_axis : float
        「平面を最も登る方向」ベクトルの x–y 平面への射影が x 軸となす角 (ラジアン, arctan2)
    angle_with_xy_plane : float
        「平面を最も登る方向」ベクトルが x–y 平面となす角 (ラジアン)
    """

    np.random.seed(random_state)

    H, W = image.shape
    # 座標を用意 (x, y)
    y_idx, x_idx = np.indices((H, W))
    # 特徴量 X: (N, 3) = [x, y, 1]
    X_all = np.stack([x_idx.ravel(), y_idx.ravel(), np.ones(H * W)], axis=1)  # (N,3)
    # 目的変数 Z
    Z_all = image.ravel()  # (N,)

    # もとのコードに書かれていたように、なぜかサンプリング元を N=300 固定にしたい場合
    # （全ピクセル数がもっと多いなら、本当に300点だけ使うのか、要確認）
    # ここでは一応「全画素数 > 300 ならサンプリング」という形にしておく
    N_full = X_all.shape[0]
    if N_full > 300:
        N_sample = 300
    else:
        N_sample = N_full

    best_inlier_count = 0
    best_params = None  # [a, b, c]

    # --- RANSAC のメインループ ---
    for _ in range(max_iterations):
        # ランダムに3点サンプリング
        sample_indices = np.random.choice(N_full, size=3, replace=False)
        X_sample = X_all[sample_indices]  # (3,3)
        Z_sample = Z_all[sample_indices]  # (3,)

        # 3点からパラメータ [a, b, c] を推定
        try:
            params = np.linalg.pinv(X_sample) @ Z_sample  # shape (3,)
            # params = [a, b, c]
        except np.linalg.LinAlgError:
            # 正則でなかった場合など
            continue

        # 全データ(ここでは N_sample 点にするのか、全画素にするのか)で評価
        # →もし本当に軽量化したいなら、ランダムに N_sample 点だけ抜き出して評価してもよい
        #   ここでは全画素を使って評価してみる
        z_pred = X_all @ params  # (N_full,)
        residuals = np.abs(Z_all - z_pred)

        inlier_mask = residuals < distance_threshold
        inlier_count = np.sum(inlier_mask)

        # ベスト更新条件
        if (
            inlier_count > best_inlier_count
            and inlier_count > min_inlier_ratio * N_full
        ):
            best_inlier_count = inlier_count
            best_params = params

    # --- RANSAC で得られたパラメータ or fallback ---
    if best_inlier_count == 0 or best_params is None:
        # 全体を通常の最小二乗でフィットする
        params_full = np.linalg.pinv(X_all) @ Z_all
        a, b, c = params_full
        z_pred_full = X_all @ params_full
        fitted_image = z_pred_full.reshape(H, W)
    else:
        # ベストモデルでインライヤー抽出し、そのみで再フィッティング
        z_pred_best = X_all @ best_params
        residuals_best = np.abs(Z_all - z_pred_best)
        inlier_mask_best = residuals_best < distance_threshold

        X_in = X_all[inlier_mask_best]
        Z_in = Z_all[inlier_mask_best]

        params_refined = np.linalg.pinv(X_in) @ Z_in
        a, b, c = params_refined
        z_pred_refined = X_all @ params_refined
        fitted_image = z_pred_refined.reshape(H, W)

    # --- 「この平面を登るベクトル」の角度計算 ---
    # 平面 z = a x + b y + c の最大傾斜方向 (a,b) を 3次元に拡張すると (a, b, a^2 + b^2) など。
    #   x–y 平面への射影 = (a, b)
    #   → x軸となす角 = atan2(b, a)
    angle_with_x_axis = np.arctan2(b, a)

    #   ベクトルが x–y 平面となす角 = arctan( z成分 / xy成分 )
    #   ただし z成分 = a^2 + b^2,  xy成分 = sqrt(a^2 + b^2)
    #   → = arctan( sqrt(a^2 + b^2) )
    xy_magnitude = np.sqrt(a**2 + b**2)
    z_component = a**2 + b**2
    angle_with_xy_plane = (
        np.arctan(z_component / xy_magnitude) if xy_magnitude != 0 else 0.0
    )

    return fitted_image, angle_with_x_axis, angle_with_xy_plane


def display_indexed_image(
    img_list: list[RGBPicture], index_matrix: D2IntArray
) -> RGBPicture:
    """
    img_list に含まれる複数の RGB 画像のうち、
    index_matrix で指定されたインデックスの画素をピクセル単位で取り出し、
    1 枚の画像として表示する関数。

    Parameters
    ----------
    img_list : list of np.ndarray (dtype=np.uint8)
        形状 (H, W, 3) の RGB 画像を格納したリスト。すべて同じ形状を仮定。
    index_matrix : np.ndarray (dtype=int)
        形状 (H, W) の整数配列で、img_list のインデックスを示す。

    Returns
    -------
    None
        生成した画像を表示するだけで、値は返さない。
    """

    # 画像がすべて同じ形状か確認（任意でチェック）
    h, w, c = img_list[0].shape
    for i, img in enumerate(img_list):
        if img.shape != (h, w, c):
            raise ValueError(f"img_list の要素 {i} が他の画像と形状が異なります。")

    # index_matrix も (H, W) か確認（任意でチェック）
    if index_matrix.shape != (h, w):
        raise ValueError("index_matrix の形状が img_list の画像と異なります。")

    # インデックスが範囲内か確認（任意でチェック）
    index_matrix = D2IntArray(np.clip(index_matrix, 0, len(img_list) - 1))
    # if index_matrix.min() < 0 or index_matrix.max() >= len(img_list):
    #     raise ValueError(
    #         "index_matrix に img_list の範囲外のインデックスが含まれています。"
    #     )

    # リストをまとめて一つの配列にスタック => 形状 (N, H, W, 3)
    stacked = np.stack(img_list, axis=0)

    # アドバンスドインデックスを使って、各 (i, j) に対し index_matrix[i, j] 番目の画像を取り出す
    # np.arange(h)[:, None] は形状 (H, 1) なので、(H, W) とブロードキャスト整合する
    out_image = stacked[index_matrix, np.arange(h)[:, None], np.arange(w), :]
    return cast(RGBPicture, out_image)


def linear_transform_image(
    image: RGBPicture, transform_matrix_2_2: D2FloatArray
) -> RGBPicture:
    """
    2x2 の行列による画像の一次変換を行う。変換は左手系

    Parameters
    ----------
    image : np.ndarray
        RGB画像(高さ×幅×3)を表す 8bit (uint8) 配列
    transform_matrix_2_2 : np.ndarray
        2x2 の変換行列

    Returns
    -------
    np.ndarray
        変換後の画像(8bit)
    """
    # 画像のサイズを取得
    rows, cols, channels = image.shape

    # warpAffine で使うために 2x2 -> 2x3 に拡張 (平行移動成分は0)
    #  [a b tx]
    #  [c d ty]
    # 今回は tx=ty=0 とする
    affine_matrix = np.array(
        [
            [transform_matrix_2_2[0, 0], transform_matrix_2_2[0, 1], 0],
            [transform_matrix_2_2[1, 0], transform_matrix_2_2[1, 1], 0],
        ],
        dtype=np.float32,
    )

    # アフィン変換を適用
    # warpAffine の第三引数 (出力サイズ) は (幅, 高さ) の順
    transformed_image = cv2.warpAffine(image, affine_matrix, (cols, rows))

    return cast(RGBPicture, transformed_image)


def focus_stack(
    img_list: list[RGBPicture],
) -> tuple[RGBPicture, D2IntArray, float, float]:

    div_n = len(img_list)
    if len(img_list) > 255:
        raise ValueError("length of img_stack should less than 255")
    index_mat = get_focus_index(img_list)
    _index_mat = resize_img(cast(MonoColorPicture, index_mat.astype(np.uint8)), 1000)
    _index_mat = median_filter(_index_mat, 21)
    _index_mat = resize_img(_index_mat, 200)

    _img, angle_with_x, angle_with_xy = ransac_plane_fitting_2d(
        cast(D2FloatArray, _index_mat.astype(np.float64))
    )

    _img = np.clip(_img, 0, div_n)  # type: ignore
    _img = cast(MonoColorPicture, _img.astype(np.uint8))  # type: ignore
    _img = resize_img(_img, height=index_mat.shape[0], width=index_mat.shape[1])  # type: ignore
    index = cast(D2IntArray, _img.astype(np.int32))
    # plt.imshow(index)
    # plt.show()
    stacked_img = display_indexed_image(img_list, index)

    return (
        cast(RGBPicture, stacked_img),
        index,
        angle_with_x,
        angle_with_xy,
    )


def transform_stacked_image(image: RGBPicture, phi: float, theta: float) -> RGBPicture:
    # 写真上で、薄片の登る方向が下のようについているとする。
    # このとき、x軸とのなす角度は 180+focus_stack(im_tilt, theta)[1]になる。(符号注意)

    # y
    # ^
    # |
    # |------------------------
    # |                        |
    # |             ↗          |
    # |           ↗            |
    # |         ↗              |
    # |                        |
    # |                        |
    # ----------------------------> x

    # イメージセンサーに写った薄片の実際の形状
    #        y
    #       /
    #      /-------------------/
    #     /                   /
    #    /√(1+tan^2θsin^2Φ)  /
    #   /                   /
    #  /gamma              /
    # /-------------------/->x
    # √(1 + tan^2θ + cos^2Φ)

    # phi_mod = (
    #     np.pi - phi_original if phi_original > 0 else np.pi + phi_original
    # )  # 右手系
    phi_mod = phi

    t1 = np.sqrt(1 + np.tan(theta) ** 2 * np.cos(phi_mod) ** 2)
    t2 = np.sqrt(1 + np.tan(theta) ** 2 * np.sin(phi_mod) ** 2)

    cos_gamma = -np.tan(theta) ** 2 * np.sin(phi_mod) * np.cos(phi_mod) / (t1 * t2)

    cos_gamma = 0
    # sin_gamma = np.sqrt(1 - cos_gamma**2)
    sin_gamma = 1

    mat = cast(D2FloatArray, np.array([[t1, 0], [t2 * cos_gamma, t2 * sin_gamma]]))

    t_image = linear_transform_image(image, mat)
    return t_image


def align_image_by_phase_correlation(
    img1: RGBPicture, img2: RGBPicture
) -> tuple[tuple[float, float], RGBPicture]:
    """
    img1 とほぼ平行移動差のみの img2 が与えられたとき、
    phaseCorrelateを用いて img2 を平行移動し、img1に位置合わせする。

    Parameters:
    -----------
    img1 : np.ndarray
        基準となる画像 (グレースケール or カラー)
    img2 : np.ndarray
        平行移動されている画像 (グレースケール or カラー)

    Returns:
    --------
    shift : (dx, dy)
        img2 を (dx, dy) だけ移動した時に、img1 と重なるような推定平行移動量
    aligned_img2 : np.ndarray
        shift 量を img2 に適用してアラインした結果画像
    """

    # 1. グレースケール化 (phaseCorrelateは1ch画像が前提)
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1

    if img2.ndim == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    # 2. float32型に変換
    f1 = np.float32(gray1)
    f2 = np.float32(gray2)

    # 3. 位相相関を用いて平行移動量 (dx, dy) を推定
    #    phaseCorrelate(ref, target) は
    #    「target をどれだけずらせば ref と重なるか」の (dx, dy) を返す。
    (dx, dy), _ = cv2.phaseCorrelate(f1, f2)  # type: ignore

    # 4. 平行移動量 (dx, dy) を用いて、img2 をアフィン変換 (warpAffine) で平行移動
    #    M は以下の通り:
    #       [1, 0, dx]
    #       [0, 1, dy]
    #    となる2x3行列。
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])  # type: ignore

    # 画像サイズ (width, height) は (列数, 行数) の順で与える
    h, w = img2.shape[:2]
    aligned_img2 = cv2.warpAffine(img2, M, (w, h), flags=cv2.INTER_LINEAR)  # type: ignore

    return (dx, dy), aligned_img2


def compare_images(im0: RGBPicture, im1: RGBPicture) -> tuple[RGBPicture, D2IntArray]:
    """
    2枚の画像 im0, im1 を比較し、各ピクセルごとに
      (1) 明度が高い画素の色
      (2) 明度が高い画像のインデックス (0: im0, 1: im1)
    を返す関数です。

    Parameters:
      im0, im1 : numpy.ndarray
          入力画像。形状は (H, W) または (H, W, 3) を想定。

    Returns:
      high_color : numpy.ndarray
          各ピクセルごとに明度が高い画像の画素値（im0またはim1）。
      high_index : numpy.ndarray
          各ピクセルごとに明度が高い画像のインデックス（0 または 1）。
    """
    # 画像がカラーの場合、明度は RGB の加重平均で計算 (ITU-R BT.601)
    if im0.ndim == 3 and im0.shape[2] == 3:
        brightness0 = 0.299 * im0[:, :, 0] + 0.587 * im0[:, :, 1] + 0.114 * im0[:, :, 2]
    else:
        brightness0 = im0.astype(np.float32)

    if im1.ndim == 3 and im1.shape[2] == 3:
        brightness1 = 0.299 * im1[:, :, 0] + 0.587 * im1[:, :, 1] + 0.114 * im1[:, :, 2]
    else:
        brightness1 = im1.astype(np.float32)

    # 各ピクセルごとに、im0 の明度が im1 の明度以上なら True、それ以外は False とする
    # 同じ明度の場合は im0 を選ぶ（True）
    mask = brightness0 >= brightness1

    # mask により、どちらの画像のピクセルを採用するか選択
    # 高い明度を持つ画像のインデックス（im0: 0, im1: 1）
    high_index = np.where(mask, 0, 1)

    # 各ピクセルごとの色の選択
    if im0.ndim == 3:
        # mask の形状を (H, W, 1) にしてブロードキャスト
        high_color = np.where(mask[:, :, None], im0, im1)
    else:
        high_color = np.where(mask, im0, im1)

    return cast(RGBPicture, high_color), cast(D2IntArray, high_index)


def get_modified_image_from_tilt_image(
    horizontal_img: RGBPicture,
    tilt_img_list: list[RGBPicture],
    theta: float,
) -> tuple[RGBPicture, D2IntArray, D2BoolArray, float]:

    # print(horizontal_img_list.shape)
    # print(tilt_img_list[0].shape)
    im_stacked, im_index, phi, theta0 = focus_stack(tilt_img_list)
    mask = np.ones_like(im_stacked)

    t_im_stacked = transform_stacked_image(im_stacked, phi, theta)

    (dx, dy), estimated_original = align_image_by_phase_correlation(
        horizontal_img, t_im_stacked
    )

    M = np.float32([[1, 0, -dx], [0, 1, -dy]])  # type: ignore
    t_mask = transform_stacked_image(mask, phi, theta)
    h, w = t_mask.shape[:2]
    mask_res = cv2.warpAffine(t_mask, M, (w, h), flags=cv2.INTER_LINEAR)  # type: ignore

    return estimated_original, im_index, mask_res[:, :, 0].astype(np.bool_), phi


def max_circ_mask_image(img: RGBPicture, center: tuple[int, int]) -> D2BoolArray:
    """
    画像 img 内に収まる、center を中心とする最大の円形マスクを作成します。

    画像の中心が (r0, c0) とした場合、上下左右の画像端までの距離の最小値を
    円の最大半径とし、その半径内の画素を True とするマスク (2次元ブール型配列) を返します。

    Parameters
    ----------
    img : np.ndarray
        入力画像。形状は (高さ, 幅, チャンネル数) または (高さ, 幅) の配列。
    center : tuple[int, int]
        円の中心座標 (行, 列)。

    Returns
    -------
    mask : np.ndarray
        img と同じ (高さ, 幅) のブール型配列。円内部が True、円外部が False となります。
    """
    height, width = img.shape[:2]
    c0, r0 = center

    # 中心から各辺までの距離の最小値を半径とする
    max_radius = min(r0, height - 1 - r0, c0, width - 1 - c0)

    # 画像サイズに合わせた座標グリッドを作成
    Y, X = np.ogrid[:height, :width]

    # (Y, X) の各画素と中心との距離の2乗が max_radius^2 以下なら円内部
    mask = (Y - r0) ** 2 + (X - c0) ** 2 < max_radius**2

    return cast(D2BoolArray, mask)


K = TypeVar("K", RGBPicture, D2BoolArray, D2FloatArray, D2IntArray)


def crop_center(img: K, center_index: tuple[int, int], shape: tuple[int, int]) -> K:
    """
    入力画像 img から，center_index を中心とする shape サイズの領域を切り抜く関数。
    切り抜く領域が img の範囲外の場合は，はみ出した部分は黒（ゼロ）で埋める。

    Parameters:
        img : NDArray[np.uint8]
            入力画像。グレースケールの場合は2次元，カラーの場合は3次元（高さ×幅×チャンネル）とする。
        center_index : tuple[int, int]
            切り抜き領域の中心位置 (row, col)。
        shape : tuple[int, int]
            切り抜き後の画像サイズ (height, width)。

    Returns:
        NDArray[np.uint8]: 切り抜かれた画像。入力画像よりも切り抜きサイズが大きい場合，
                           はみ出した部分は黒（0）で埋められる。
    """
    crop_h, crop_w = shape
    # 入力画像における切り抜き領域の左上座標を求める
    top = center_index[1] - crop_h // 2
    left = center_index[0] - crop_w // 2
    bottom = top + crop_h
    right = left + crop_w

    # 出力画像を黒（0）で初期化する
    if img.ndim == 2:
        cropped = np.zeros((crop_h, crop_w), dtype=img.dtype)
    else:
        # カラー画像の場合 (高さ, 幅, チャンネル)
        channels = img.shape[2]  # type: ignore
        cropped = np.zeros((crop_h, crop_w, channels), dtype=img.dtype)

    # 入力画像内で実際にコピー可能な領域を求める
    in_y0 = max(top, 0)
    in_y1 = min(bottom, img.shape[0])
    in_x0 = max(left, 0)
    in_x1 = min(right, img.shape[1])

    # 出力画像上の対応する領域を計算する
    out_y0 = in_y0 - top  # top < 0 の場合はオフセットが必要
    out_y1 = out_y0 + (in_y1 - in_y0)
    out_x0 = in_x0 - left
    out_x1 = out_x0 + (in_x1 - in_x0)

    # 対応する領域をコピー
    cropped[out_y0:out_y1, out_x0:out_x1] = img[in_y0:in_y1, in_x0:in_x1]

    return cast(K, cropped)


def estimate_tilted_image(
    im: list[RGBPicture],
    im_tilt: list[RGBPicture],
    # color_chart: RGBPicture,
    # retardation: D1FloatArray,
    theta: float,
    center: Optional[tuple[int, int]] = None,
    shape: Optional[tuple[int, int]] = None,
    rotation: Optional[float] = None,
) -> TiltImageResult:
    div_n = len(im_tilt)
    w_original = im_tilt[0].shape[1]
    h_original = im_tilt[0].shape[0]

    N = 1000
    rim = resize_array(im[0], N)

    mod_tilt_img, focused_index, mask, azimuth_thin_section = (
        get_modified_image_from_tilt_image(
            rim, list(map(lambda x: resize_img(x, N), im_tilt)), theta
        )
    )

    mod_tilt_img = normalize_by_gray_scale(rim, mod_tilt_img, mask)

    if rotation is not None and center is not None:
        mod_tilt_img = rotate_array(mod_tilt_img, rotation, center)
        rim = rotate_array(rim, rotation, center)
        mask = rotate_array(mask, rotation, center)
        focused_index = rotate_array(focused_index, rotation, center)

        # plt.imshow(mask)
        # plt.show()

        # plt.imshow(focused_index)
        # plt.show()

    # if 画像の形状を特定の形に強制するとき
    if center is not None and shape is not None:
        mod_tilt_img = crop_center(mod_tilt_img, center, shape)
        rim = crop_center(rim, center, shape)
        focused_index = crop_center(focused_index, center, shape)
        mask = crop_center(mask, center, shape)
        h_original, w_original = mod_tilt_img.shape[:2]

    # horizontal, _, _ = make_retardation_color_map(
    #     rim, color_chart, retardation, block_size=100
    # )

    # tilt, _, _ = make_retardation_color_map(
    #     mod_tilt_img, color_chart, retardation, block_size=100
    # )

    r_focused_index = resize_array(focused_index, w_original, h_original)

    r_mask = resize_array(mask, w_original, h_original)

    # r_original_retardation = resize_array(horizontal, w_original, h_original)

    # tilted_retardation = resize_array(tilt, w_original, h_original)

    return TiltImageResult(
        original_image=resize_img(rim, w_original, h_original),
        focused_tilted_image=resize_img(mod_tilt_img, w_original, h_original),
        focused_index=r_focused_index,
        image_mask=(r_mask == 1),
        azimuth_thin_section=azimuth_thin_section,
        # original_retardation=r_original_retardation,
        # tilted_retardation=tilted_retardation,
    )


def normalize_by_gray_scale(
    standard_img: RGBPicture,
    target_img: RGBPicture,
    mask: Optional[D2BoolArray] = None,
) -> RGBPicture:
    im1 = standard_img
    im2 = target_img
    im1_hsv = cv2.cvtColor(im1, cv2.COLOR_RGB2HSV).astype(np.float64)
    h2, s2, v2 = cv2.split(cv2.cvtColor(im2, cv2.COLOR_RGB2HSV))
    if mask is not None:
        im_med1 = np.median(im1_hsv[:, :, 2][mask])
        im_med2 = np.median(v2[mask].astype(np.float64))
    else:
        im_med1 = np.median(im1_hsv)
        im_med2 = np.median(v2.astype(np.float64))
    v2_new = np.clip(v2.astype(np.float64) + (im_med1 - im_med2), 0, 225).astype(
        np.uint8
    )
    im2_hsv_new = cv2.merge([h2, s2, v2_new])
    im2_new = cv2.cvtColor(im2_hsv_new, cv2.COLOR_HSV2RGB).astype(np.uint8)

    return RGBPicture(im2_new)


def make_inc_vs_azimuth_color_chart_with_full_wave_plate_mod(
    alpha: float,
    thickness: float,
    theta_rad: float,
    num_inc: int = 30,
    num_azimuth: int = 100,
) -> ColorChartInfosOfTiltedImage:
    color_chart = make_uniaxial_color_chart(
        lambda x, y: get_spectral_distribution(
            get_full_wave_plus_mineral_retardation_system(R=x, azimuth=y, alpha=alpha)
        )["rgb"],
        thickness=thickness,
        num_inc=num_inc,
        num_azimuth=num_azimuth,
        max_azimuth=180,
    )

    h_inclination = color_chart["h"]
    axis_index = np.argmin(np.abs(h_inclination - theta_rad))
    axis_index_used = 1 if axis_index == 0 else axis_index

    color_chart_used = color_chart["color_chart"].transpose((1, 0, 2))
    H, W = color_chart_used.shape[:2]
    h_inclination = color_chart["h"]
    w_azimuth = color_chart["w"]
    color_chart_right = color_chart_used[:, : int(W / 4), :]
    color_chart_left = color_chart_used[:, int(3 * W / 4) :, :]
    color_chart_used = np.concatenate(
        (color_chart_left, color_chart_right), axis=1
    )  # type: ignore

    w_azimuth_new = np.concatenate(
        (np.flip(w_azimuth[int(3 * W / 4) :]), w_azimuth[: int(W / 4)])
    )
    minus_color_chart = np.concatenate(
        (
            np.flip(color_chart_used[:axis_index_used, :], axis=0),
            color_chart_used[:-axis_index_used, :],
        )
    )
    plus_color_chart = np.concatenate(
        (
            color_chart_used[axis_index_used:, :],
            np.flip(color_chart_used[-axis_index_used:, :], axis=0),
        )
    )
    return ColorChartInfosOfTiltedImage(
        horizontal=ColorChartInfo(
            what_is_h="inclination",
            what_is_w="azimuth",
            color_chart=color_chart_used,
            w=w_azimuth_new,
            h=h_inclination,
        ),
        plus_tilted=ColorChartInfo(
            what_is_h="inclination",
            what_is_w="azimuth",
            color_chart=RGBPicture(plus_color_chart),
            w=w_azimuth_new,
            h=h_inclination,
        ),
        minus_tilted=ColorChartInfo(
            what_is_h="inclination",
            what_is_w="azimuth",
            color_chart=RGBPicture(minus_color_chart),
            w=w_azimuth_new,
            h=h_inclination,
        ),
    )


def make_inc_vs_azimuth_color_chart_with_full_wave_plate(
    alpha: float,
    thickness: float,
    theta_rad: float,
    num_inc: int = 30,
    num_azimuth: int = 100,
) -> ColorChartInfosOfTiltedImage:
    color_chart = make_uniaxial_color_chart(
        lambda w, h: get_spectral_distribution(
            get_full_wave_plus_mineral_retardation_system(R=h, azimuth=w, alpha=alpha)
        )["rgb"],
        thickness=thickness,
        num_inc=num_inc,
        num_azimuth=num_azimuth,
        max_azimuth=180,
    )

    h_inclination = color_chart["h"]
    axis_index = np.argmin(np.abs(h_inclination - theta_rad))
    axis_index_used = 1 if axis_index == 0 else axis_index

    color_chart_used = color_chart["color_chart"]
    H, W = color_chart_used.shape[:2]
    h_inclination = color_chart["h"]
    w_azimuth = color_chart["w"]

    minus_color_chart = np.concatenate(
        (
            np.flip(color_chart_used[:axis_index_used, :], axis=0),
            color_chart_used[:-axis_index_used, :],
        )
    )
    plus_color_chart = np.concatenate(
        (
            color_chart_used[axis_index_used:, :],
            np.flip(color_chart_used[-axis_index_used:, :], axis=0),
        )
    )
    return ColorChartInfosOfTiltedImage(
        horizontal=ColorChartInfo(
            what_is_h="inclination",
            what_is_w="azimuth",
            color_chart=color_chart_used,
            w=w_azimuth,
            h=h_inclination,
        ),
        plus_tilted=ColorChartInfo(
            what_is_h="inclination",
            what_is_w="azimuth",
            color_chart=RGBPicture(plus_color_chart),
            w=w_azimuth,
            h=h_inclination,
        ),
        minus_tilted=ColorChartInfo(
            what_is_h="inclination",
            what_is_w="azimuth",
            color_chart=RGBPicture(minus_color_chart),
            w=w_azimuth,
            h=h_inclination,
        ),
    )


def get_index_matrix_at_color_chart(
    img: RGBPicture, color_chart: ColorChartInfo
) -> tuple[D2IntArray, D2IntArray]:
    #!特殊なつかいかた
    predicted_map, index_h, index_w = make_retardation_color_map(
        img,
        color_chart["color_chart"],
        color_chart["w"],
        block_size=100,
    )
    return index_h, index_w


def judge_inclination_direction(
    color_charts: ColorChartInfosOfTiltedImage,
    im_original: RGBPicture,
    im_tilt: RGBPicture,
    index_h: D2IntArray,
    index_w: D2IntArray,
) -> D2BoolArray:

    minus_diff = color_charts["minus_tilted"]["color_chart"].astype(
        np.float64
    ) - color_charts["horizontal"]["color_chart"].astype(np.float64)
    plus_diff = color_charts["plus_tilted"]["color_chart"].astype(
        np.float64
    ) - color_charts["horizontal"]["color_chart"].astype(np.float64)

    plus_diff_img = plus_diff[index_h, index_w].astype(np.float64)
    minus_diff_img = minus_diff[index_h, index_w].astype(np.float64)

    im_plus = cv2.cvtColor(
        np.clip(plus_diff_img + im_original, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2Lab,
    ).astype(np.float64)

    im_minus = cv2.cvtColor(
        np.clip(minus_diff_img + im_original, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2Lab,
    ).astype(np.float64)

    im_tilt_lab = cv2.cvtColor(im_tilt, cv2.COLOR_RGB2Lab).astype(np.float64)
    im_original_lab = cv2.cvtColor(im_original, cv2.COLOR_RGB2Lab).astype(np.float64)

    im_diff = im_tilt_lab - im_original_lab
    im_diff_plus = im_plus - im_original_lab
    im_diff_minus = im_minus - im_original_lab

    p_plus = np.sum(im_diff * im_diff_plus, axis=2)
    p_minus = np.sum(im_diff * im_diff_minus, axis=2)

    res = np.zeros_like(p_plus, dtype=np.bool_)

    # res[np.bitwise_and(p_plus >= 0, p_minus < 0)] = 1
    # res[np.bitwise_and(p_plus < 0, p_minus >= 0)] = -1

    res[p_plus > p_minus] = True

    return D2BoolArray(res)


def make_judgment_from_TiltImageResult(
    im_result: TiltImageResult,
    color_charts: ColorChartInfosOfTiltedImage,
) -> D2BoolArray:
    im_original = im_result["original_image"]
    im_tilt = im_result["focused_tilted_image"]
    index_h, index_w = get_index_matrix_at_color_chart(
        im_original, color_charts["horizontal"]
    )
    # plt.imshow(color_charts["horizontal"]["color_chart"])
    # plt.show()
    return judge_inclination_direction(
        color_charts, im_original, im_tilt, index_h, index_w
    )


# def estimate_tilt_direction(
#     raw_maps: RawMaps,
#     im_result0: TiltImageResult,
#     # im_result45: TiltImageResult,
#     theta_deg: float,
#     alpha: float,
#     thickness: float,
# ) -> tuple[D2BoolArray, ColorChartInfosOfTiltedImage, TiltImageResult, TiltImageResult]:

#     ex_angle = raw_maps["extinction_angle"]

#     color_charts = make_inc_vs_azimuth_color_chart_with_full_wave_plate_mod(
#         alpha, thickness, theta
#     )
#     plt.imshow(color_charts["horizontal"]["color_chart"])
#     plt.show()

#     res_at_45 = make_judgment_from_TiltImageResult(im_result45, color_charts)
#     res_at_0 = make_judgment_from_TiltImageResult(im_result0, color_charts)

#     res = np.zeros_like(res_at_0)

#     res[np.bitwise_or(ex_angle <= 22.5, ex_angle > 67.5)] = res_at_45[
#         np.bitwise_or(ex_angle <= 22.5, ex_angle > 67.5)
#     ]

#     res[np.bitwise_and(ex_angle > 22.5, ex_angle <= 67.5)] = res_at_0[
#         np.bitwise_and(ex_angle > 22.5, ex_angle <= 67.5)
#     ]

#     return D2BoolArray(res), res_at_0, res_at_45, color_charts, im_result0, im_result45


if __name__ == "__main__":

    # r: ComputationResult = pd.read_pickle("../test/data/output/yamagami_cross.avi.pkl")
    r: ComputationResult = pd.read_pickle(
        # "../test/data/output/tetori_4k_xpl_pol_til.pkl_classified.pkl"
        "../test/data/output/tetori_4k_xpl_pol_til10.pkl_classified.pkl"
    )

    # color_chart = make_uniaxial_color_chart(
    #     lambda w, h: get_spectral_distribution(
    #         get_full_wave_plus_mineral_retardation_system(R=h, azimuth=w, alpha=alpha)
    #     )["rgb"],
    #     thickness=thickness,
    #     num_inc=num_inc,
    #     num_azimuth=num_azimuth,
    #     max_azimuth=180,
    # )

    thickness: float = 0.05
    width = 1000

    im0 = r.tilt_image_info.image0_raw
    im45 = r.tilt_image_info.image45_raw
    im_tilt0 = r.tilt_image_info.tilt_image0_raw
    im_tilt45 = r.tilt_image_info.tilt_image45_raw
    theta_deg = np.degrees(np.arctan(7.5 / 48))

    theta = np.radians(theta_deg)
    thickness = 0.05

    ex_angle = r.raw_maps["extinction_angle"]
    center = (r.center_int_x, r.center_int_y)
    shape = (ex_angle.shape[0], ex_angle.shape[1])

    im_result0 = estimate_tilted_image(
        im0,
        im_tilt0,
        theta,
        center=center,
        shape=shape,
    )

    im_result45 = estimate_tilted_image(
        im45,
        im_tilt45,
        theta,
        center=center,
        shape=shape,
        rotation=-35,
    )

    # up_down, res0, res45, x, y, z = estimate_tilt_direction(
    #     r.raw_maps,
    #     im_result0=im_result0,
    #     im_result45=im_result45,
    #     theta_deg=theta,
    #     alpha=r.color_chart.pol_lambda_alpha,
    #     thickness=thickness,
    # )
    # %%

    plt.imshow(im_result0["focused_index"])
    plt.show()

    plt.imshow(im_result45["focused_index"])
    plt.show()
    # plt.imshow(im_result45["focused_tilted_image"])
    # plt.imshow(im_result45["focused_tilted_image"])

    # %%
    azimuth = r.raw_maps["azimuth"]
    inclination = np.zeros_like(azimuth)
    inclination[up_down] = -azimuth[up_down]
    inclination[~up_down] = azimuth[~up_down]
    plt.imshow(inclination, cmap="hsv")
    plt.colorbar()
    plt.show()
    plt.imshow(azimuth, cmap="hsv")
    plt.colorbar()
    plt.show()

    # %%
    plt.imshow(im_result0["original_image"])
    plt.show()
    plt.imshow(im_result0["focused_tilted_image"])
    plt.show()
    # %%
    plt.imshow(im_result45["original_image"])
    plt.show()
    plt.imshow(im_result45["focused_tilted_image"])
    plt.show()
    # %%
    plt.imshow(up_down)
    plt.show()
    plt.imshow(res0)
    plt.show()
    plt.imshow(res45)
    plt.show()

    plt.imshow(azimuth, cmap="hsv")
    plt.colorbar()
    plt.show()
    # plt.imshow(up_down)
    # plt.show()
    alpha = r.color_chart.pol_lambda_alpha
    # %%

    color_chart = make_uniaxial_color_chart(
        lambda x, y: get_spectral_distribution(
            get_full_wave_plus_mineral_retardation_system(R=x, azimuth=y, alpha=alpha)
        )["rgb"],
        thickness=thickness,
        num_inc=20,
        num_azimuth=50,
        max_azimuth=180,
    )
