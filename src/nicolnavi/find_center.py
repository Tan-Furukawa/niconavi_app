# %%
from typing import Callable, cast, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm  # type: ignore
from niconavi.image.type import RGBPicture, MonoColorPicture
from niconavi.tools.type import D2FloatArray
from niconavi.tools.change_type import as_two_element_tuple
from niconavi.image.image import extract_image_edges, resize_img
from niconavi.tools.read_data import divide_video_into_n_frame
from matplotlib.figure import Figure
from tqdm import tqdm  # type: ignore
from scipy.ndimage import center_of_mass
from copy import deepcopy
import numpy as np
from skimage.transform import AffineTransform, warp

__all__ = ["find_rotation_center", "plot_center_image"]


def adjust_gamma(image: RGBPicture) -> RGBPicture:
    # ガンマ補正用のルックアップテーブルを作成
    gamma: float = 2.0
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype(
        "uint8"
    )

    # ルックアップテーブルを使って変換
    # plt.imshow(image)
    # plt.show()
    return cast(RGBPicture, cv2.LUT(image, table))
    # return cv2.


def make_superimpose_image(
    change_to_mono_color_pic: Callable[[RGBPicture], MonoColorPicture],
    pics: list[RGBPicture],
) -> MonoColorPicture:
    """
    Creates a mean image by extracting edges from input images and computing the pixel-wise maximum.

    Returns:
        NDArray[np.uint8]: The resulting mean image.
    """
    pics = list(map(adjust_gamma, pics))
    img = list(map(change_to_mono_color_pic, pics))
    stacked_img = np.max(np.array(img, dtype=np.float64), axis=0)
    # mean_image = cv2.Sobel(mean_image, cv2.CV_64F, 0, 1, ksize=3)  # x方向
    stacked_img = stacked_img.astype(np.uint8)
    stacked_img = cv2.equalizeHist(stacked_img)
    return cast(MonoColorPicture, stacked_img)


def rotate_image_180(
    img: MonoColorPicture, center: tuple[int, int]
) -> MonoColorPicture:
    """
    Rotates the image 180 degrees around the specified center point.

    Args:
        img (MatLike): Input image (grayscale).
        center (tuple[int, int]): Coordinates of the rotation center point (x0, y0).

    Returns:
        MatLike: Image after point symmetry transformation.
    """
    x0, y0 = center
    height, width = img.shape

    y_indices, x_indices = np.indices((height, width))
    x_mirror = 2 * x0 - x_indices
    y_mirror = 2 * y0 - y_indices

    valid_mask = (
        (x_mirror >= 0) & (x_mirror < width) & (y_mirror >= 0) & (y_mirror < height)
    )

    rotated = np.zeros_like(img)

    rotated[valid_mask] = img[y_mirror[valid_mask], x_mirror[valid_mask]]

    return rotated


def find_symmetry_center(
    gray_img: MonoColorPicture,
    step: int = 10,
    search_range_fraction: tuple[float, float] = (1 / 5, 4 / 5),
    progress_callback: Callable[[float], None] = lambda p: None,
) -> tuple[int, int]:
    """Detects the center point of point symmetry in the specified image.

    Args:
        gray_img (MatLike): Input image (grayscale).
        step (int, optional): Evaluation interval for target points. Defaults to 10.
        search_range_fraction (tuple[float, float], optional): Fractional range for searching the y-axis.
            Defaults to (1/5, 4/5).

    Returns:
        tuple[int, int]: Coordinates of the point symmetry center (x, y).
    """

    if search_range_fraction[0] > search_range_fraction[1]:
        raise ValueError(
            "search_range_fraction[0] should be smaller than search_range_fraction[1]"
        )

    height, width = gray_img.shape

    # Calculate search range for y-axis based on fractions
    y_min = int(height * search_range_fraction[0])
    y_max = int(height * search_range_fraction[1])

    # Initialize the best score and center point
    best_center = (0, 0)

    # Initial coarse search
    best_score, best_center = search_symmetry(
        gray_normalized=gray_img,
        y_range=(y_min, y_max),
        x_range=(0, width),
        step=step,
        best_center=best_center,
        description="   (1/2) Initial search",
        progress_callback=progress_callback,
    )

    # Refined search parameters
    search_range_step = step  # Range around the best center from the coarse search

    x_best, y_best = best_center
    y_start = max(y_best - search_range_step, y_min)
    y_end = min(y_best + search_range_step + 1, y_max)
    x_start = max(x_best - search_range_step, 0)
    x_end = min(x_best + search_range_step + 1, width)

    # Refined search for higher accuracy
    best_score, best_center = search_symmetry(
        gray_normalized=gray_img,
        y_range=(y_start, y_end),
        x_range=(x_start, x_end),
        step=1,
        best_center=best_center,
        description="   (2/2) Refined search",
        progress_callback=progress_callback,
    )

    return best_center


def search_symmetry(
    gray_normalized: MonoColorPicture,
    y_range: tuple[int, int],
    x_range: tuple[int, int],
    step: int,
    best_center: tuple[int, int],
    description: str,
    progress_callback: Callable[[float], None] = lambda p: None,
) -> tuple[float, tuple[int, int]]:
    """Searches for the symmetry center within specified y and x ranges.

    Args:
        gray_normalized (np.ndarray): Normalized grayscale image.
        y_range (tuple[int, int]): Range for y-axis (start, end).
        x_range (tuple[int, int]): Range for x-axis (start, end).
        step (int): Step size for iteration.
        best_center (Tuple[int, int]): Current best center coordinates.
        description (str): Description for the progress bar.

    Returns:
        tuple[float, tuple[int, int]]: Updated best score and best center coordinates.
    """
    best_score = -1.0  # TM_CCOEFF_NORMED ranges from -1 to 1
    search_range = range(y_range[0], y_range[1], step)
    for i, y0 in tqdm(enumerate(search_range), desc=description):
        progress_callback(i / len(search_range))
        for x0 in range(x_range[0], x_range[1], step):
            rotated = rotate_image_180(gray_normalized, (x0, y0))
            similarity = cv2.matchTemplate(
                gray_normalized, rotated, cv2.TM_CCOEFF_NORMED
            ).max()

            if similarity > best_score:
                best_score = similarity
                best_center = (x0, y0)

    return best_score, best_center


def find_low_resolution_center_point(
    superimpose_image: MonoColorPicture,
    low_resolution_width: int,
    progress_callback: Callable[[float], None] = lambda p: None,
) -> tuple[float, float]:
    """
    Finds the center of symmetry in a lower-resolution version of the mean image.

    Returns:
        tuple[float, float]: Normalized coordinates of the symmetry center.
    Raises:
        ValueError: If the center coordinates are (0, 0).
    """
    width_pixel = low_resolution_width
    rimg = resize_img(superimpose_image, width_pixel)
    height, width = rimg.shape
    print("[1/2] Find the center of a lower-resolution image:")
    center_x, center_y = find_symmetry_center(
        rimg, 10, progress_callback=progress_callback
    )
    if (center_x == 0) or (center_y == 0):
        raise ValueError(
            "trouble in center_x or center_y: rotation center should not zero"
        )

    center_0_to_1 = (
        float(center_x) / float(width),
        float(center_y) / float(height),
    )
    return center_0_to_1


def find_high_resolution_center_point(
    superimpose_image: MonoColorPicture,
    high_resolution_width: int,
    center_0_to_1: tuple[float, float],
    progress_callback: Callable[[float], None] = lambda p: None,
) -> tuple[float, float]:
    """
    Refines the center of symmetry in a higher-resolution version of the mean image.

    Returns:
        tuple[int, int]: Integer pixel coordinates of the symmetry center.
    Raises:
        ValueError: If normalized center coordinates are not set.
    """
    mimg = resize_img(superimpose_image, high_resolution_width)
    # mimg = mimg.astype(np.float64) / 255
    # plt.imshow(mimg)
    # plt.colorbar()
    # plt.show()
    height, width = mimg.shape
    cx, cy = center_0_to_1
    if cx is None or cy is None:
        raise ValueError(
            "center_0_to_1 may be (None, None). Exec find_low_resolution_center_point function before exec this function"
        )
    else:
        icx = int(width * cx)
        icy = int(height * cy)
        delta_x = 5
        delta_y = 5
        print("[2/2] Find the center of a higher-resolution image:")
        best_score, best_center = search_symmetry(
            gray_normalized=mimg,
            y_range=(icy - delta_y, icy + delta_y),
            x_range=(icx - delta_x, icx + delta_x),
            step=1,
            best_center=(icx, icy),
            description="   (1/1) Refined search",
            progress_callback=progress_callback,
        )

        center_0_to_1 = (
            float(best_center[0]) / float(width),
            float(best_center[1]) / float(height),
        )
        return center_0_to_1


def plot_center_image(
    superimpose_image: MonoColorPicture, center: tuple[int, int]
) -> Figure:
    """
    Plots the mean image with the identified center of symmetry marked.

    Raises:
        ValueError: If the mean image is not computed.
    """

    height, width = superimpose_image.shape

    img = superimpose_image

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.scatter(center[0], center[1], c="red", marker="+", s=200, linewidths=0.7)

    return fig

    # plt.imshow(img, cmap="gray")
    # plt.scatter(center01[0] * width, center01[1] * height, c="red", marker="+", s=100)


def get_image_edges(pic: RGBPicture) -> MonoColorPicture:
    # return extract_image_edges(pic, 100, 100)
    return MonoColorPicture(cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY))


def frst(
    image: MonoColorPicture,
    radii: list[float],
    xlim: Optional[tuple[int, int]] = None,
    ylim: Optional[tuple[int, int]] = None,
    alpha: int = 2,
) -> D2FloatArray:
    """
    Fast Radial Symmetry Transform (FRST) を計算する関数

    Parameters:
        image : 入力画像（NDArray、グレースケールと仮定）
        radii : 投票に用いる半径のリストまたは配列
        xlim  : 探索する x 座標の範囲を (xmin, xmax) のタプルで指定。None の場合は画像全体。
        ylim  : 探索する y 座標の範囲を (ymin, ymax) のタプルで指定。None の場合は画像全体。
        alpha : 非線形強調のパラメータ

    Returns:
        symmetry_map : 各画素の対称性スコアを示す画像
    """
    # 1. 前処理：ガウシアンフィルタによる平滑化
    smooth = cv2.GaussianBlur(image, (5, 5), sigmaX=1.5)

    # 2. エッジ検出および勾配計算
    grad_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    rows, cols = image.shape
    symmetry_map = np.zeros_like(image, dtype=np.float64)

    # 各半径 r に対して
    for r in radii:
        # 投票用の累積配列
        O = np.zeros_like(image, dtype=np.float64)  # 勾配方向の寄与回数
        M = np.zeros_like(image, dtype=np.float64)  # 勾配大きさの寄与

        # 探索範囲の設定（xlim, ylim が None でなければその範囲に制限）
        x_start = xlim[0] if xlim is not None else 0
        x_end = xlim[1] if xlim is not None else cols
        y_start = ylim[0] if ylim is not None else 0
        y_end = ylim[1] if ylim is not None else rows

        # 指定された探索範囲内の各画素に対して投票
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # 勾配が小さい画素は無視（閾値は適宜調整）
                if magnitude[y, x] < 10:
                    continue
                # 勾配方向に沿って候補中心の位置を算出
                dx = int(round(r * np.cos(angle[y, x])))
                dy = int(round(r * np.sin(angle[y, x])))
                xp = x - dx
                yp = y - dy
                if 0 <= xp < cols and 0 <= yp < rows:
                    # 勾配の大きさを重みとして票を投じる
                    M[yp, xp] += magnitude[y, x]
                    O[yp, xp] += 1

        # 局所的な非線形強調：各画素のスコア F を計算
        F = (M / (O + 1e-5)) ** alpha  # ゼロ除算防止のために微小値を加える
        # 複数の半径に対する結果を加算
        symmetry_map += F

    # 最終的な対称性マップを正規化（0～1の範囲）
    symmetry_map = cast(D2FloatArray, cv2.normalize(symmetry_map, None, 0, 1, cv2.NORM_MINMAX))  # type: ignore

    return symmetry_map


def estimate_initial_frst_peak(
    superimpose_image: MonoColorPicture,
) -> tuple[int, int]:
    oh, ow = superimpose_image.shape

    img = resize_img(superimpose_image, 1000)

    img = cv2.GaussianBlur(superimpose_image, (11, 11), sigmaX=5)  # type: ignore
    img = resize_img(img, 400)
    height, width = img.shape
    gy, gx = center_of_mass(img)
    r = (
        min(gy, (height - 1) - gy, gx, (width - 1) - gx) * 0.7
    )  # だいたい0.7倍くらいには入るでしょ
    xlim = (int(gx - r), int(gx + r))
    ylim = (int(gy - r), int(gy + r))

    img_frst = frst(img, [10, 50, 75, 100, 150, 200, 250], xlim, ylim)
    img_frst = cv2.GaussianBlur(img_frst, (101, 101), sigmaX=21)  # type: ignore

    y_indices, x_indices = np.indices(img_frst.shape)
    mask = ((y_indices - gy) ** 2 + (x_indices - gx) ** 2) <= r**2
    img_frst_masked = deepcopy(img_frst)
    img_frst_masked[~mask] = 0

    # plt.imshow(img_frst)
    # plt.show()

    cy, cx = np.unravel_index(np.argmax(img_frst_masked), img_frst_masked.shape)
    return int(cx / width * ow), int(cy / height * oh)


def estimate_precise_frst_peak(
    superimpose_image: MonoColorPicture,
    cxcy: tuple[int, int],
) -> tuple[int, int]:

    height, width = superimpose_image.shape
    cx_index = cxcy[0]
    cy_index = cxcy[1]
    r = min(cy_index, (height - 1) - cy_index, cx_index, (width - 1) - cx_index)
    xlim = (cx_index - int(r / 5), cx_index + int(r / 5))
    ylim = (cy_index - int(r / 5), cy_index + int(r / 5))

    frst_img = frst(superimpose_image, np.linspace(5, 50, 10), xlim, ylim)  # type: ignore
    # frst_img = frst(superimpose_image, [2, 4, 8, 16], xlim, ylim)  # type: ignore
    frst_img = cv2.GaussianBlur(frst_img, (51, 51), 31)  # type: ignore
    # plt.imshow(frst_img)
    # plt.show()

    cy, cx = cast(
        tuple[int, int], np.unravel_index(np.argmax(frst_img), frst_img.shape)
    )
    return int(cx), int(cy)


def rotational_symmetry_score(img: np.ndarray) -> np.ndarray:
    """
    各画素 (i, j) を「中心」とみなし、まわりを 90° ごとに 4 回回転させた
    ときの“ばらつき”を測るスコア画像を返します。

    Parameters
    ----------
    img : np.ndarray  (H × W, dtype = uint8 など)
        0–255 のグレースケール画像。

    Returns
    -------
    score : np.ndarray  (H × W, dtype = float64)
        回転対称性が高いほど小さな値になる 2-次元配列。
    """
    h, w = img.shape
    score = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            # 「中心」から上下左右端までの距離の最小値が r。
            r = min(i, j, h - 1 - i, w - 1 - j)
            if r == 0:  # 1 ピクセル幅しか取れない場所は無視
                continue

            # 重なり領域 (= はみ出さず取れる最大の正方形) を抜き出す
            patch = img[i - r : i + r + 1, j - r : j + r + 1]

            # 0°, 90°, 180°, 270° に相当する 4 枚のパッチ
            rots = [
                patch,
                np.rot90(patch, k=1),
                np.rot90(patch, k=2),
                np.rot90(patch, k=3),
            ]

            # 連続する 4 組 (img1-img2, img2-img3, img3-img4, img4-img1)
            # の二乗誤差を合計
            diff = 0.0
            for k in range(4):
                diff += np.sum((rots[k] - rots[(k + 1) % 4]) ** 2)

            # 画素数で正規化
            score[i, j] = diff / patch.size

    return score


def estimate_more_precise_frst_peak(
    superimpose_image: MonoColorPicture,
    cxcy: tuple[int, int],
) -> tuple[int, int]:
    """
    Parameters
    ----------
    superimpose_image : ndarray(H, W)
        モノクロ画像（2 次元配列）。
    cxcy : (int, int)
        画像上の注目点 (cx, cy)。

    Returns
    -------
    (int, int)
        元画像座標系でのピーク位置 (cx, cy)。
    """
    height, width = superimpose_image.shape
    cx_index, cy_index = cxcy

    # ---- 1. 切り抜き範囲の計算 --------------------------------------------- #
    r = min(cy_index, (height - 1) - cy_index, cx_index, (width - 1) - cx_index)
    half_win = int(r / 3)

    xlim = (max(cx_index - half_win, 0), min(cx_index + half_win, width - 1))
    ylim = (max(cy_index - half_win, 0), min(cy_index + half_win, height - 1))

    # Python のスライスは上限を含まないので +1 しておく
    x_start, x_end = xlim[0], xlim[1] + 1
    y_start, y_end = ylim[0], ylim[1] + 1

    cropped = superimpose_image[y_start:y_end, x_start:x_end]
    score = rotational_symmetry_score(cropped.astype(np.float64))

    # plt.imshow(cropped)
    # plt.colorbar()
    # plt.show()
    # ---- 2. スコアマップ中央部のみで最小値を探索 ----------------------------- #
    H, W = score.shape
    # 評価範囲: score[H/2:3H/2, W/2:3W/2]
    sub_score = score[H // 4 : (3 * H) // 4, W // 4 : (3 * W) // 4]

    # sub_score 内での極小点（相対座標）
    rel_cy, rel_cx = np.unravel_index(np.argmin(sub_score), sub_score.shape)

    # plt.imshow(sub_score)
    # plt.colorbar()
    # plt.show()

    # score 全体でのローカル座標へ変換
    local_cy = int(rel_cy + H // 4)
    local_cx = int(rel_cx + W // 4)

    # ---- 3. 元画像座標系へ変換 --------------------------------------------- #
    global_cx = x_start + local_cx
    global_cy = y_start + local_cy

    return global_cx, global_cy


# def estimate_more_precise_frst_peak(
#     superimpose_image: MonoColorPicture,
#     cxcy: tuple[int, int],
# ) -> tuple[int, int]:
#     """
#     superimpose_image を (xlim, ylim) で切り抜いてから FRST を実行し，
#     得られたピークの座標を元画像の座標系に戻して返す。

#     Parameters
#     ----------
#     superimpose_image : ndarray(H, W)
#         モノクロ画像（2 次元配列）。
#     cxcy : (int, int)
#         画像上の注目点 (cx, cy)。

#     Returns
#     -------
#     (int, int)
#         元画像座標系でのピーク位置 (cx, cy)。
#     """
#     height, width = superimpose_image.shape
#     cx_index, cy_index = cxcy

#     # ---- 1. 切り抜き範囲の計算 --------------------------------------------- #
#     r = min(cy_index, (height - 1) - cy_index, cx_index, (width - 1) - cx_index)
#     half_win = int(r / 4)

#     xlim = (max(cx_index - half_win, 0), min(cx_index + half_win, width - 1))
#     ylim = (max(cy_index - half_win, 0), min(cy_index + half_win, height - 1))

#     # Python のスライスは上限を含まないので +1 しておく
#     x_start, x_end = xlim[0], xlim[1] + 1
#     y_start, y_end = ylim[0], ylim[1] + 1

#     cropped = superimpose_image[y_start:y_end, x_start:x_end]
#     score = rotational_symmetry_score((cropped).astype(np.float64))

#     local_cy, local_cx = map(int, np.unravel_index(np.argmin(score), score.shape))

#     global_cx = x_start + local_cx
#     global_cy = y_start + local_cy

#     return global_cx, global_cy


# def estimate_more_precise_frst_peak(
#     superimpose_image: MonoColorPicture,
#     cxcy: tuple[int, int],
# ) -> tuple[int, int]:

#     height, width = superimpose_image.shape
#     cx_index = cxcy[0]
#     cy_index = cxcy[1]
#     r = min(cy_index, (height - 1) - cy_index, cx_index, (width - 1) - cx_index)
#     xlim = (cx_index - int(r / 5), cx_index + int(r / 5))
#     ylim = (cy_index - int(r / 5), cy_index + int(r / 5))

#     frst_img = frst(superimpose_image, np.linspace(5, 20, 3), xlim, ylim)  # type: ignore
#     frst_img = cv2.GaussianBlur(frst_img, (51, 51), 31)  # type: ignore
#     # plt.imshow(frst_img)
#     # plt.show()

#     cy, cx = cast(
#         tuple[int, int], np.unravel_index(np.argmax(frst_img), frst_img.shape)
#     )
#     return int(cx), int(cy)


def find_rotation_center(
    pics: list[RGBPicture],
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> tuple[MonoColorPicture, tuple[int, int]]:
    progress_callback(None)
    sup_img = make_superimpose_image(get_image_edges, pics)

    cxcy = estimate_initial_frst_peak(sup_img)
    # cxcy = estimate_precise_frst_peak(sup_img, cxcy)
    cxcy = estimate_more_precise_frst_peak(sup_img, cxcy)
    # cxcy = estimate_more_precise_frst_peak(sup_img, cxcy)
    # search_symmetry(sup_img, (-10, 10), (-10, 10))

    return sup_img, cxcy


if __name__ == "__main__":

    from niconavi.tools.read_data import divide_video_into_n_frame

    # pics = divide_video_into_n_frame("../test/data/yamagami_cross.avi", 100)
    pics = divide_video_into_n_frame("../test/data/tetori_cross.avi", 200)
    # n = 100
    # pics = divide_video_into_n_frame("../test/data/tetori_4k/Movie_185.avi", 100)

    pics = list(map(lambda x: resize_img(x, 1000), pics))
    sup_img = make_superimpose_image(get_image_edges, pics)
    simg, (cx, cy) = find_rotation_center(pics)
    # %%
    plt.imshow(sup_img)
    plt.scatter(cx, cy, marker="+", c="red")
    plt.show()

    # res, (cx, cy) = compute_fourfold_symmetry_map(simg, (cx, cy), 100, 100)
    # res[res == 0] = np.nan
    # plt.imshow(res)
    # plt.show()
    # plt.hist(res.flatten(), bins=1000)
    # np.unique(res.flatten())
    # plt.imshow(res)
    # plt.colorbar()
