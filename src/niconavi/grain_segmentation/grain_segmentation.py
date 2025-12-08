# type: ignore
# -------------------------------------------------------------------
# 目的： extinction angle map のboundary推定がうまくいかないように見える
# 解決: circular_gradientを、sin_grad_x**2 + sin_grad_y**2 + cos_grad_x**2 + cos_grad_y**2
# -------------------------------------------------------------------
# %%
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from niconavi.grain_segmentation.crack import (
    crack_segmentation,
    multiscale_hessian_emphasis,
)
from niconavi.grain_segmentation.make_index import label_regions_with_skeleton
from typing import Optional
from niconavi.grain_segmentation.type import (
    BinaryImage,
    BoolMatrix,
    FloatMatrix,
    GrayScaleImage,
    IndexMap,
    RGBImage,
)
from niconavi.tools.grain_plot import detect_boundaries


def connect_skeleton_endpoints(
    skel_img: BinaryImage | BoolMatrix,
    max_distance: float,
    connectivity: int = 8,
    thickness: int = 1,
    pair_once: bool = True,
) -> BinaryImage | BoolMatrix:
    """
    スケルトン画像の端点同士のうち、ユークリッド距離が max_distance 以下のものを直線で結ぶ。

    Args:
        skel_img: 2値スケルトン画像。bool もしくは 0/255 の uint8 を想定。
        max_distance: 端点間を接続する最大距離（ピクセル）。
        connectivity: 端点判定の近傍（4 または 8）。デフォルト 8。
        thickness: 直線描画の太さ（ピクセル）。デフォルト 1。
        pair_once: True のとき、各端点は高々1回だけ接続（最近傍優先の貪欲ペアリング）。
                   False のとき、max_distance 以内の全ての相手に接続（多重接続）。

    Returns:
        入力と同じ dtype を保った 2値画像。端点同士が直線で結ばれた画像。
    """
    if skel_img.ndim != 2:
        raise ValueError("skel_img は2次元配列である必要があります。")

    # 入力の型・スケールを保持するために記録
    input_dtype = skel_img.dtype
    is_bool_input = input_dtype == np.bool_

    # bool に正規化
    skel_bool = (skel_img > 0) if skel_img.dtype != np.bool_ else skel_img
    skel_u8 = skel_bool.astype(np.uint8)

    if connectivity == 8:
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    elif connectivity == 4:
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    else:
        raise ValueError("connectivity は 4 または 8 を指定してください。")

    # 各画素の近傍に存在するスケルトン画素数（自分自身は含まない）
    neighbor_counts = cv2.filter2D(
        skel_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT
    )
    endpoints_mask = (skel_u8 == 1) & (neighbor_counts == 1)

    # 端点座標 (y, x)
    ys, xs = np.where(endpoints_mask)
    coords = (
        np.stack([ys, xs], axis=1) if ys.size > 0 else np.empty((0, 2), dtype=np.int32)
    )

    # 出力画像（0/255 の uint8）
    out = skel_bool.astype(np.uint8) * 255
    # 交差判定用に、既存スケルトン（元の線分）のマスクを保持
    skel_mask = skel_bool.copy()

    n = coords.shape[0]
    if n < 2:
        # 端点が0または1ならそのまま返す
        return (
            BoolMatrix((out > 0).astype(bool))
            if is_bool_input
            else BinaryImage(out.astype(np.uint8, copy=False))
        )

    # 距離行列を計算
    pts = coords.astype(np.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    dist2 = (diff**2).sum(axis=2)
    # 自己距離は無視
    np.fill_diagonal(dist2, np.inf)

    if pair_once:
        # i<j のペアで max_distance 以内の候補を作成し、距離の短い順に貪欲に接続
        pairs: List[Tuple[float, int, int]] = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                d2 = dist2[i, j]
                if d2 <= max_distance * max_distance:
                    pairs.append((float(d2), i, j))
        if pairs:
            pairs.sort(key=lambda t: t[0])
            used = np.zeros(n, dtype=bool)
            for _, i, j in pairs:
                if used[i] or used[j]:
                    continue
                y1, x1 = int(coords[i, 0]), int(coords[i, 1])
                y2, x2 = int(coords[j, 0]), int(coords[j, 1])
                # 交差チェック（端点は交差から除外）
                tmp = np.zeros_like(skel_mask, dtype=np.uint8)
                cv2.line(tmp, (x1, y1), (x2, y2), color=1, thickness=thickness)
                tmp_bool = tmp.astype(bool)
                conflict = tmp_bool & skel_mask
                conflict[y1, x1] = False
                conflict[y2, x2] = False
                if np.any(conflict):
                    continue  # 既存線分と交差するので接続しない
                cv2.line(out, (x1, y1), (x2, y2), color=255, thickness=thickness)
                used[i] = used[j] = True
    else:
        # max_distance 以内の全ての相手に接続（多重接続）
        distance_limit_sq = max_distance * max_distance
        for i in range(n - 1):
            for j in range(i + 1, n):
                if dist2[i, j] <= distance_limit_sq:
                    y1, x1 = int(coords[i, 0]), int(coords[i, 1])
                    y2, x2 = int(coords[j, 0]), int(coords[j, 1])
                    # 交差チェック（端点は交差から除外）
                    tmp = np.zeros_like(skel_mask, dtype=np.uint8)
                    cv2.line(tmp, (x1, y1), (x2, y2), color=1, thickness=thickness)
                    tmp_bool = tmp.astype(bool)
                    conflict = tmp_bool & skel_mask
                    conflict[y1, x1] = False
                    conflict[y2, x2] = False
                    if np.any(conflict):
                        continue  # 既存線分と交差するので接続しない
                    cv2.line(out, (x1, y1), (x2, y2), color=255, thickness=thickness)

    # 入力 dtype に戻す
    if is_bool_input:
        return BoolMatrix(out > 0)
    return BinaryImage(out.astype(input_dtype, copy=False))


def skeleton_loops_only(binary_img: BinaryImage | BoolMatrix) -> BinaryImage:
    """スケルトナイズされたループ構造のみを抽出する。"""

    bin_norm = (binary_img > 0).astype(bool)
    skel = skeletonize(bin_norm)

    graph = nx.Graph()
    rows, cols = np.where(skel)
    for y, x in zip(rows, cols):
        graph.add_node((y, x))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (
                    0 <= ny < skel.shape[0]
                    and 0 <= nx_ < skel.shape[1]
                    and skel[ny, nx_]
                ):
                    graph.add_edge((y, x), (ny, nx_))

    cycles = nx.cycle_basis(graph)
    loop_img = np.zeros_like(binary_img, dtype=np.uint8)
    for cycle in cycles:
        for y, x in cycle:
            loop_img[y, x] = 255

    return BinaryImage(loop_img)


def hessian_image(image: GrayScaleImage | RGBImage) -> FloatMatrix:
    """ヘッシアン強調により鉱物エッジを強調したスコアマップを作成する。"""

    if image.ndim == 2:
        response, _ = multiscale_hessian_emphasis(
            255 - image, sigma=0.3, s=1.3, num_iteration=6
        )
    else:
        # r, _ = multiscale_hessian_emphasis(
        #     255 - image[:,:,0], sigma=0.3, s=1.3, num_iteration=10
        # )
        # g, _ = multiscale_hessian_emphasis(
        #     255 - image[:,:,0], sigma=0.3, s=1.3, num_iteration=10
        # )
        # b, _ = multiscale_hessian_emphasis(
        #     255 - image[:,:,0], sigma=0.3, s=1.3, num_iteration=10
        # )
        magnitude_norm = 255 - cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        res, _ = multiscale_hessian_emphasis(
            cv2.equalizeHist(magnitude_norm), sigma=0.3, s=1.3, num_iteration=6
        )
        response = res

    return FloatMatrix(response)


# def make_segmented_map_from_raw_map(
#     result: ComputationResult,
#     threshold: float,
# ) -> BoolMatrix:
#     """複数角度の原画像から鉱物領域の2値マップを生成する。"""

#     degree_keys = ["degree_0", "degree_22_5", "degree_45", "degree_67_5"]
#     mineral_maps = [segmentation_minerals(result.raw_maps[key]) for key in degree_keys]
#     combined_map = np.sum(mineral_maps, axis=0, dtype=np.float64)
#     return BoolMatrix((combined_map > np.exp(threshold)).astype(bool))


def fill_small_enclosed_false_regions(
    mask: BoolMatrix,
    max_area: int,
    connectivity: int = 4,
) -> BoolMatrix:
    """
    入力:
        mask: 2次元のbool配列 (Trueがネットワーク構造、Falseがそれ以外)
        max_area: 塗りつぶす“穴”(False領域)の最大面積(ピクセル数)
        connectivity: 連結性。4 または 8 を指定可能（デフォルト4）

    出力:
        小さい“穴”(False領域)を True で潰した新しいbool配列

    動作概要:
        1) False領域を1、True領域を0とする2値画像を作成
        2) 連結成分でFalse領域をラベリング
        3) 画像の外周に接するラベルは除外（囲われていないので“穴”ではない）
        4) 面積が max_area 以下のラベル領域のみを True に変換
    """
    # 入力チェック
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask は2次元のbool配列である必要があります。")
    if max_area <= 0:
        return BoolMatrix(mask.copy())

    # False領域を1、True領域を0に（OpenCVは「非ゼロ=前景」として扱う）
    inv = (~mask).astype(np.uint8)

    # False領域の連結成分を抽出
    # labels: 各画素のラベル (0..num_labels-1), 0はinv==0の背景(Ture側)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inv, connectivity=connectivity
    )

    # 外枠(上下左右の境界)に接するラベル集合を取得
    border_labels = (
        set(np.unique(labels[0, :]))
        | set(np.unique(labels[-1, :]))
        | set(np.unique(labels[:, 0]))
        | set(np.unique(labels[:, -1]))
    )

    # 出力用にコピー
    out = mask.copy()

    # ラベル1..(num_labels-1)がFalse領域（0はinv==0の背景）
    for lbl in range(1, num_labels):
        if lbl in border_labels:
            continue  # 外部と接しているので囲まれた穴ではない
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area <= max_area:
            out[labels == lbl] = True  # 面積が閾値以下の穴を潰す

    return BoolMatrix(out)


def label_enclosed_false_regions(mask: BoolMatrix, connectivity: int = 4) -> IndexMap:
    """
    Trueで囲われた False 領域に固有のインデックスを付与する。

    入力:
        mask: (H, W) の bool 配列。True=ネットワーク(前景)、False=対象領域(後景)
        connectivity: 連結性（4 または 8）

    出力:
        labels_enclosed: (H, W) の np.int32。True領域および外部と接する False は 0。
                         True に囲まれた False 領域ごとに 1..N のインデックスを付与。
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask は (H, W) の bool 配列である必要があります。")
    if connectivity not in (4, 8):
        raise ValueError("connectivity は 4 または 8 を指定してください。")

    inv = (~mask).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(inv, connectivity=connectivity)
    labels = labels.astype(np.int32, copy=False)

    border_labels = set(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        ).tolist()
    )

    enclosed_labels = [lbl for lbl in range(1, num_labels) if lbl not in border_labels]

    lut = np.zeros(num_labels, dtype=np.int32)
    for idx, lbl in enumerate(enclosed_labels, start=1):
        lut[lbl] = idx

    out = lut[labels]

    return IndexMap(out)


def component_info_to_feature_matrix(component_info: List[Dict]) -> np.ndarray:
    """成分情報リストから特徴量行列 (N, 9) を構築する。"""

    feature_keys = [
        "logA",
        "C",
        "solidity",
        "aspect_ratio",
        "Lab_a_med",
        "Lab_b_med",
        "HSV_S_med",
        "cx",
        "cy",
    ]

    if not component_info:
        return np.zeros((0, len(feature_keys)), dtype=np.float64)

    feature_matrix = np.zeros(
        (len(component_info), len(feature_keys)), dtype=np.float64
    )
    for idx, info in enumerate(component_info):
        feature_matrix[idx] = [float(info[key]) for key in feature_keys]

    return feature_matrix


def analyze_false_components_features(
    mask: np.ndarray, pic: np.ndarray, connectivity: int = 4
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    入力:
        mask: (H, W) bool。True=ネットワーク(前景), False=対象領域(後景)
        pic : (H, W, 3) np.uint8（BGR想定）
        connectivity: 4 or 8

    出力:
        labels_int32: (H, W) np.int32。mask=False の連結成分を 1..N、mask=True は 0
        info_list   : 各成分の辞書（index昇順）
            {
              "index": int,
              "area": int, "perimeter": float,
              "logA": float, "C": float, "solidity": float, "aspect_ratio": float,
              "Lab_a_med": float, "Lab_b_med": float, "HSV_S_med": float,
              "cx": float, "cy": float   # 画素座標（x=列, y=行）
            }
        X: (N, 9) の np.float64
           列順 = [logA, C, solidity, aspect_ratio, Lab_a_med, Lab_b_med, HSV_S_med, cx, cy]
    """
    # 入力チェック
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask は (H, W) の bool である必要があります。")
    if (
        pic.ndim != 3
        or pic.shape[:2] != mask.shape
        or pic.shape[2] != 3
        or pic.dtype != np.uint8
    ):
        raise ValueError(
            "pic は (H, W, 3) の np.uint8 で、mask と同じ H, W が必要です。"
        )
    if connectivity not in (4, 8):
        raise ValueError("connectivity は 4 または 8 を指定してください。")

    H, W = mask.shape

    # False を前景(=255)に
    inv = np.where(~mask, 255, 0).astype(np.uint8)

    # 連結成分ラベリング（重心も取得）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inv, connectivity=connectivity
    )
    labels_int32 = labels.astype(np.int32)

    # 色空間は一度だけ変換
    pic_lab = cv2.cvtColor(pic, cv2.COLOR_BGR2Lab)  # (L,a,b)
    pic_hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)  # (H,S,V)

    info_list: List[Dict] = []
    rows: List[List[float]] = []

    for lbl in range(1, num_labels):  # 0 は mask=True 側（背景）
        x, y, w, h, area = (
            stats[lbl, cv2.CC_STAT_LEFT],
            stats[lbl, cv2.CC_STAT_TOP],
            stats[lbl, cv2.CC_STAT_WIDTH],
            stats[lbl, cv2.CC_STAT_HEIGHT],
            int(stats[lbl, cv2.CC_STAT_AREA]),
        )
        if area <= 0 or w <= 0 or h <= 0:
            continue

        # ---- ROI で処理 ----
        lab_roi = pic_lab[y : y + h, x : x + w]
        hsv_roi = pic_hsv[y : y + h, x : x + w]
        lbl_roi = labels[y : y + h, x : x + w]
        roi_mask = lbl_roi == lbl

        # 周長（この成分だけの2値ROIで外周のみ）
        comp_bin = np.zeros((h, w), dtype=np.uint8)
        comp_bin[roi_mask] = 255
        cnts, _ = cv2.findContours(comp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = float(sum(cv2.arcLength(c, True) for c in cnts))

        # 凸包面積
        hull_area = 0.0
        for c in cnts:
            if c.size == 0:
                continue
            hull = cv2.convexHull(c)
            hull_area += cv2.contourArea(hull)
        if hull_area <= 0:
            hull_area = float(area)

        # 形状特徴
        logA = float(np.log(area))
        C = float((perimeter * perimeter) / (4.0 * np.pi * area))  # 円形度
        solidity = float(area / hull_area)
        aspect_ratio = float(w / h)

        # 色のメディアン（a, b, S）
        a_vals = lab_roi[..., 1][roi_mask]
        b_vals = lab_roi[..., 2][roi_mask]
        s_vals = hsv_roi[..., 1][roi_mask]
        Lab_a_med = float(np.median(a_vals)) if a_vals.size else 0.0
        Lab_b_med = float(np.median(b_vals)) if b_vals.size else 0.0
        HSV_S_med = float(np.median(s_vals)) if s_vals.size else 0.0

        # 重心（connectedComponentsWithStats の centroids は (x, y)）
        cx, cy = float(centroids[lbl, 0]), float(centroids[lbl, 1])

        info_list.append(
            {
                "index": lbl,
                "area": area,
                "perimeter": perimeter,
                "logA": logA,
                "C": C,
                "solidity": solidity,
                "aspect_ratio": aspect_ratio,
                "Lab_a_med": Lab_a_med,
                "Lab_b_med": Lab_b_med,
                "HSV_S_med": HSV_S_med,
                "cx": cx,
                "cy": cy,
            }
        )

        rows.append(
            [logA, C, solidity, aspect_ratio, Lab_a_med, Lab_b_med, HSV_S_med, cx, cy]
        )

    # index 昇順に整える（rows も同期）
    info_list.sort(key=lambda d: d["index"])
    if info_list:
        X = np.array(
            [
                [
                    d["logA"],
                    d["C"],
                    d["solidity"],
                    d["aspect_ratio"],
                    d["Lab_a_med"],
                    d["Lab_b_med"],
                    d["HSV_S_med"],
                    d["cx"],
                    d["cy"],
                ]
                for d in info_list
            ],
            dtype=np.float64,
        )
    else:
        X = np.zeros((0, 9), dtype=np.float64)

    return labels_int32, info_list


def make_extinction_angle_image(extinction_map: FloatMatrix) -> GrayScaleImage:
    """消光角マップを 0-255 の疑似カラー画像に変換する。"""

    angle_rad = extinction_map / 90 * 2 * np.pi
    intensity = ((np.sin(angle_rad) + np.cos(angle_rad)) / np.sqrt(2)) ** 2 * 255
    return GrayScaleImage(intensity.astype(np.uint8))


def segment_R_color_map(color_map: RGBImage) -> BoolMatrix:
    """RGB カラーマップから鉱物領域を抽出するブールマスクを返す。"""

    equalize_hist = cv2.equalizeHist
    segmented_r, _ = crack_segmentation(
        255 - equalize_hist(color_map[:, :, 0]),
        sigma=0.3,
        iter_srm=10,
        iter_stp=10,
        iter_mhe=5,
    )
    segmented_g, _ = crack_segmentation(
        255 - equalize_hist(color_map[:, :, 1]),
        sigma=0.3,
        iter_srm=10,
        iter_stp=10,
        iter_mhe=5,
    )
    segmented_b, _ = crack_segmentation(
        255 - equalize_hist(color_map[:, :, 2]),
        sigma=0.3,
        iter_srm=10,
        iter_stp=10,
        iter_mhe=5,
    )
    return BoolMatrix(
        ((segmented_r != 0) | (segmented_g != 0) | (segmented_b != 0)).astype(bool)
    )


def segment_ex_color_map(extinction_map: GrayScaleImage) -> BoolMatrix:
    """消光角マップから鉱物領域を抽出するブールマスクを返す。"""

    segmented, _ = crack_segmentation(
        255 - extinction_map, sigma=0.3, iter_srm=10, iter_stp=10, iter_mhe=5
    )
    return BoolMatrix((segmented != 0).astype(bool))


def get_hessian_image(
    ex_map: FloatMatrix, color_map: RGBImage
) -> tuple[FloatMatrix, FloatMatrix]:
    ex_image = make_extinction_angle_image(ex_map)
    hessian_color_map = hessian_image(color_map)
    hessian_ex_image = hessian_image(ex_image)
    return FloatMatrix(hessian_color_map), FloatMatrix(hessian_ex_image)


def get_circ(color_map: RGBImage) -> BoolMatrix:

    H, W = color_map.shape

    if H != W:
        raise ValueError("the computation area is not square")

    res = np.zeros((H, H), dtype=np.bool_)
    X, Y = np.meshgrid(range(H), range(H))

    res[X**2]


def remove_short_skeleton_components(
    skeleton: np.ndarray, n: int, connectivity: int = 8
) -> np.ndarray:
    """
    スケルトン（True=線, False=非線）から、連結成分の長さが n 以下の成分を除去して返す。

    Parameters
    ----------
    skeleton : np.ndarray (bool または uint8)
        2次元のスケルトン画像（True/1 が線）。
    n : int
        除去する長さ（画素数）以下のしきい値。<= n の連結成分を削除します。
    connectivity : int
        連結性（4 または 8）。細い線の連結を正しく扱うには 8 が一般的。

    Returns
    -------
    pruned : np.ndarray (bool)
        短い連結成分を削除したスケルトン。
    """
    if skeleton.dtype != np.bool_:
        sk = skeleton.astype(bool)
    else:
        sk = skeleton
    if sk.ndim != 2:
        raise ValueError("2次元画像のみ対応しています。")

    # 前景(True)を 1 にしてラベリング
    sk_u8 = sk.astype(np.uint8)
    if connectivity not in (4, 8):
        raise ValueError("connectivity は 4 または 8 を指定してください。")
    num_labels, labels = cv2.connectedComponents(
        sk_u8, connectivity=connectivity
    )  # 0..num_labels-1

    if num_labels <= 1:
        # スケルトンが空か、1成分のみ
        return (
            np.zeros_like(sk, dtype=bool)
            if num_labels == 0
            else (sk if sk.sum() > n else np.zeros_like(sk, dtype=bool))
        )

    # 各ラベルの画素数（label 0 は背景なので除外）
    counts = np.bincount(labels.ravel())
    # keep: 画素数 > n の成分
    keep_ids = np.flatnonzero(counts > n)
    keep_ids = keep_ids[keep_ids != 0]  # 背景 0 を除く

    pruned = np.isin(labels, keep_ids)
    return pruned


# --- 端点検出（8近傍で True が1つだけの画素） ---
def _find_endpoints(skel: np.ndarray, connectivity: int = 8) -> np.ndarray:
    sk = skel.astype(np.uint8)
    if connectivity == 8:
        kernel = np.ones((3, 3), np.uint8)
    elif connectivity == 4:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    else:
        raise ValueError("connectivity は 4 または 8")
    # 近傍数（自分自身も数えるので後で -1）
    deg = (
        cv2.filter2D(
            sk, ddepth=cv2.CV_16S, kernel=kernel, borderType=cv2.BORDER_CONSTANT
        )
        - sk
    )
    endpoints = sk.astype(bool) & (deg == 1)
    return endpoints


# --- 形態学的再構成（marker をマスク mask 内で geodesic dilate し続ける） ---
def _reconstruct(
    marker: np.ndarray,
    mask: np.ndarray,
    connectivity: int = 8,
    max_iters: int | None = None,
) -> np.ndarray:
    mk = marker.astype(np.uint8)
    ms = mask.astype(np.uint8)
    if connectivity == 8:
        kernel = np.ones((3, 3), np.uint8)
    else:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

    if max_iters is None:
        h, w = mk.shape
        max_iters = int(np.ceil(np.hypot(h, w))) + 5

    prev = np.zeros_like(mk)
    cur = mk & ms
    for _ in range(max_iters):
        if np.array_equal(cur, prev):
            break
        prev = cur
        cur = cv2.dilate(cur, kernel, iterations=1)
        cur &= ms
    return cur.astype(bool)


# --- Zhang-Suen 細線化（削除禁止マスクつき） ---
def _thinning_zs_masked(binary: np.ndarray, forbid_remove: np.ndarray) -> np.ndarray:
    """
    binary: 細線化したい2値（True=前景）
    forbid_remove: True の画素は削除してはいけない（元スケルトンを渡す）
    """
    img = binary.astype(np.uint8).copy()

    changed = True
    h, w = img.shape

    while changed:
        changed = False
        for sub in (0, 1):
            to_del = np.zeros_like(img, dtype=np.uint8)

            # 8近傍を取得
            P2 = np.pad(img, 1)[0:h, 1 : w + 1]  # 上
            P3 = np.pad(img, 1)[0:h, 2 : w + 2]  # 右上
            P4 = np.pad(img, 1)[1 : h + 1, 2 : w + 2]  # 右
            P5 = np.pad(img, 1)[2 : h + 2, 2 : w + 2]  # 右下
            P6 = np.pad(img, 1)[2 : h + 2, 1 : w + 1]  # 下
            P7 = np.pad(img, 1)[2 : h + 2, 0:w]  # 左下
            P8 = np.pad(img, 1)[1 : h + 1, 0:w]  # 左
            P9 = np.pad(img, 1)[0:h, 0:w]  # 左上

            # 中心
            P1 = img

            # 条件計算（Zhang-Suen）
            # A(P1): 0->1 への遷移回数
            neighbors = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = np.zeros_like(img, dtype=np.uint8)
            for i in range(8):
                A += ((neighbors[i] == 0) & (neighbors[i + 1] == 1)).astype(np.uint8)

            # B(P1): 近傍の1の数
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

            if sub == 0:
                m1 = P2 * P4 * P6 == 0
                m2 = P4 * P6 * P8 == 0
            else:
                m1 = P2 * P4 * P8 == 0
                m2 = P2 * P6 * P8 == 0

            cond = (P1 == 1) & (A == 1) & (B >= 2) & (B <= 6) & m1 & m2

            # 既存スケルトン（削除禁止）を守る
            cond &= ~forbid_remove

            to_del[cond] = 1

            if np.any(to_del):
                img[to_del == 1] = 0
                changed = True

    return img.astype(bool)


def repair_skeleton_min_change(
    skeleton: np.ndarray, radius: int = 2, connectivity: int = 8
) -> np.ndarray:
    """
    既存のスケルトンの滑らかさを保ちながら、切れている箇所だけ最小限で補修する。
    - 元スケルトン画素は削除しない（凍結）
    - 端点近傍だけを膨張したマスクで geodesic reconstruction
    - 追加画素だけを Zhang–Suen で 1px に細線化して合成
    """
    if skeleton.dtype != np.bool_:
        sk = skeleton.astype(bool)
    else:
        sk = skeleton
    if sk.ndim != 2:
        raise ValueError("2次元のみ対応")

    # 1) 端点検出
    endpoints = _find_endpoints(sk, connectivity=connectivity)

    # 2) 端点だけを膨張して橋渡し候補マスクを作る
    if radius <= 0:
        return sk.copy()
    ksz = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    bridge_mask = cv2.dilate(endpoints.astype(np.uint8), kernel, iterations=1).astype(
        bool
    )

    # 既存スケルトンもマスクに含める（再構成が既存領域を消さないように）
    mask = bridge_mask | sk

    # 3) 形態学的再構成（マーカー=元スケルトン, マスク=mask）
    recon = _reconstruct(sk, mask, connectivity=connectivity)

    # 4) 追加画素のみ抽出し、追加部分だけ細線化（元は削除禁止）
    added = recon & ~sk
    if not added.any():
        return sk.copy()

    # 細線化対象は「元 + 追加」（連結性を見失わないため）
    target = sk | added
    thinned = _thinning_zs_masked(target, forbid_remove=sk)  # 元は削除不可

    # 5) 出力：細線化結果（= 元 + 追加の細線）
    return thinned


def max_circle_skeleton(H: int, W: int) -> np.ndarray:
    """
    H×W に収まる最大の円のスケルトン（1px幅の円周）を True/False の2値で返す。
    True が円のスケルトン、False がそれ以外。

    - 中心は画像中央 (cx=W//2, cy=H//2)
    - 半径 r は中心から各辺までの最小距離（整数）に設定
    - r >= 1 のときは1px幅の円周、r < 1 のときは中心1画素のみ True（最小ケース）
    """
    if H <= 0 or W <= 0:
        raise ValueError("H, W は正の整数である必要があります。")

    # 出力配列
    out = np.zeros((H, W), dtype=np.uint8)

    # 中心（0始まりの画素座標）
    cx, cy = W // 2, H // 2

    # 最大半径（中心から各辺までの最小距離）
    r = min(cx, cy, (W - 1) - cx, (H - 1) - cy)

    if r >= 1:
        # 1px 幅の円周を描画
        cv2.circle(out, (cx, cy), int(r), 255, thickness=1, lineType=cv2.LINE_8)
    else:
        # 極小サイズでは中心1画素のみ（半径0相当）
        if 0 <= cy < H and 0 <= cx < W:
            out[cy, cx] = 255

    return out.astype(bool)


def get_grain_boundary_fn(color_map, display: bool = False):

    gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY)
    hessian_color_map = hessian_image(gray)

    h = hessian_color_map
    h = (h - np.min(h)) / (np.max(h) - np.min(h))
    h = h ** (1 / 2)
    complexity_init = np.percentile(h, 90)

    # if display:
    #     plt.imshow(hessian_color_map, cmap="gray_r")
    #     plt.savefig("h.pdf")
    #     plt.show()

    #     plt.imshow(h, cmap="gray_r")
    #     plt.savefig("h_mod.pdf")
    #     plt.show()

    def _closure(complexity: Optional[float], connectivity: int):
        if complexity is None:
            complexity = complexity_init
        sg = h > complexity

        # if display:
        #     plt.imshow(sg, cmap="gray_r")
        #     plt.savefig("h_mod_cutoff.pdf")
        #     plt.show()

        sk = skeletonize(sg)

        # if display:
        #     plt.imshow(detect_boundaries(sk), cmap="gray_r")
        #     plt.savefig("h_mod_cutoff_sk.pdf")
        #     plt.show()

        sk = remove_short_skeleton_components(sk, 5)

        # if display:
        #     plt.imshow(detect_boundaries(sk), cmap="gray_r")
        #     plt.savefig("h_mod_cutoff_sk_remove_small.pdf")
        #     plt.show()

        sk = repair_skeleton_min_change(sk, radius=connectivity)

        # if display:
        #     plt.imshow(detect_boundaries(sk), cmap="gray_r")
        #     plt.savefig("h_mod_cutoff_sk_remove_small_connect.pdf")
        #     plt.show()

        sk = skeleton_loops_only(sk)

        # if display:
        #     plt.imshow(detect_boundaries(sk), cmap="gray_r")
        #     plt.savefig("h_mod_cutoff_sk_remove_small_connect_reskeltonize.pdf")
        #     plt.show()

        return sk, sg

    return _closure, complexity_init


# %%
if __name__ == "__main__":

    # r = pd.read_pickle("../../fix/_output/Movie_392_extinction_angle.pkl")
    # r = pd.read_pickle("../../fix/_output/Movie_368_2.pkl")
    r = pd.read_pickle("../../fix/_output/Movie_380.pkl")
    color_map = r.raw_maps["R_color_map"]
    # color_map = cv2.equalizeHist(color_map)
    ex_map = r.raw_maps["extinction_angle"]

    gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY)

    fn = get_grain_boundary_fn(color_map)
    sg, sk = fn(0.4, 10)

    # gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY)
    # gray_med = cv2.medianBlur(gray, 101)
    # diff =  gray_med.astype(np.float64) - gray.astype(np.float64)
    # diff = ((diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255).astype(np.uint8)
    # diff = cv2.medianBlur(diff, 5)
    # plt.imshow(diff)
    # plt.imshow(np.abs(gray.astype(np.float64) - gray_med.astype(np.float64))) plt.colorbar()
    # plt.show()
    # ex_image = make_extinction_angle_image(ex_map)

    plt.imshow(sg)
    plt.colorbar()
    # %%
    # %%


# sg_ex = segment_ex_color_map(ex_image)
# sg = segment_R_color_map(color_map)
# ex_image = make_extinction_angle_image(ex_map)
# sg_ex = segment_ex_color_map(ex_image)
# sg = segment_R_color_map(color_map)
# #%%
# plt.imshow(sg)
# %%
# sg = k1 > 0.07
# sk = skeletonize(sg)
# sk2 = connect_skeleton_endpoints(sk, 5)
# sk2 = skeleton_loops_only(sk2)
# plt.imshow(sk2)
# plt.colorbar()
