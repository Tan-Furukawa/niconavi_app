# %%
import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Optional, cast, Literal, TypeAlias, TypeVar, overload
from niconavi.image.type import (
    RGBPicture,
    RGBAPicture,
    HSVPicture,
    _CommonPictureType,
    _PictureType,
    MonoColorPicture,
    D1RGB_Array,
    Color,
)
from niconavi.image.types_operation import is_RGBPicture, is_MonoColorPicture
from niconavi.tools.type import D2IntArray, D2FloatArray, D2BoolArray, D3BoolArray
from matplotlib.colors import to_rgba
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

T = TypeVar("T", RGBPicture, D2FloatArray, D2IntArray)


@overload
def apply_color_to_mask(
    img: RGBPicture,
    mask: D2BoolArray,
    color: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_at_2d: str = "hsv",
    is_log_norm: bool = False,
) -> RGBPicture: ...
@overload
def apply_color_to_mask(
    img: D2FloatArray,
    mask: D2BoolArray,
    color: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_at_2d: str = "hsv",
    is_log_norm: bool = False,
) -> D2FloatArray: ...
@overload
def apply_color_to_mask(
    img: D2IntArray,
    mask: D2BoolArray,
    color: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_at_2d: str = "hsv",
    is_log_norm: bool = False,
) -> D2IntArray: ...


def apply_color_to_mask(img, mask, color: str, vmin: Optional[float] = None, vmax: Optional[float] = None, cmap_at_2d: str = "hsv", is_log_norm: bool = False):  # type: ignore
    _img = deepcopy(img)

    # color を (r, g, b, a) in [0..1] に変換
    r, g, b, a = to_rgba(color)  # float値 [0..1]

    # 画像の dtype に応じて、色をスケーリング
    if np.issubdtype(_img.dtype, np.integer):
        # int 系 (uint8, int など) は [0..255] の範囲とみなす
        r, g, b, a = [int(round(x * 255)) for x in (r, g, b, a)]
    else:
        # float の場合はそのまま (0..1)
        # ※ img の実際のスケールが 0..1 ではない場合は、ここを調整してください
        pass

    # 画像が 2D か 3D かチェック
    if _img.ndim == 2:

        if vmin is None:
            vmin = _img.min()
        if vmax is None:
            vmax = _img.max()

        cmap = cm.get_cmap(cmap_at_2d)
        if not is_log_norm:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        _img2 = cmap(norm(_img))
        # plt.imshow(cmap(norm(_img)))

        color_arr = np.array([r, g, b, a], dtype=_img2.dtype)

        # print(norm.shape)
        # print(mask.shape)
        _img2[mask] = color_arr
        return _img2

    elif _img.ndim == 3:
        # 3D (N, M, C) -> RGB or RGBA
        if len(_img.shape) > 2:
            num_channels = _img.shape[2]
        else:
            raise ValueError("cannot find _img.shape[2] value")
        if num_channels == 3:
            # RGB の場合
            # (r, g, b) を代入 (alphaは無視)
            color_arr = np.array([r, g, b], dtype=_img.dtype)
            # True の画素だけ書き換え (ブールインデックスで全チャネル一括)
            _img[mask, :] = color_arr
            return _img

        elif num_channels == 4:
            # RGBA の場合
            color_arr = np.array([r, g, b, a], dtype=_img.dtype)
            # True の画素だけ書き換え
            _img[mask, :] = color_arr
            return _img
        else:
            raise ValueError("3次元画像は (N, M, 3) か (N, M, 4) を想定しています。")
    else:
        raise ValueError(
            "img は 2次元(モノクロ) または 3次元(RGB/RGBA) 配列である必要があります。"
        )

# @overload
# def reshape_grain_mask(
#     img: RGBPicture | RGBAPicture, mask: D2BoolArray
# ) -> D2BoolArray: ...
# @overload
# def reshape_grain_mask(
#     img: D2BoolArray | D2IntArray | D2FloatArray, mask: D2BoolArray
# ) -> D3BoolArray: ...
# def reshape_grain_mask(img, mask):  # type: ignore
#     # img.ndim と mask.ndim を比較
#     if img.ndim == mask.ndim:
#         # もし次元数が同じ(=2 次元同士)なら、そのまま表示可能
#         return mask
#     else:
#         # 次元数が異なる場合、img が 3次元 (N, M, 3 or 4)、mask は 2次元 (N, M)
#         # -> mask を第三軸方向に拡張する
#         if img.ndim == 3 and mask.ndim == 2:
#             # 例: img.shape = (N,M,3), mask.shape = (N,M)
#             num_channels = img.shape[-1]  # 3 or 4
#             # mask (N,M) -> (N,M,num_channels)
#             mask_modified = np.repeat(
#                 mask[:, :, np.newaxis], repeats=num_channels, axis=2
#             )

#             return cast(D3BoolArray, mask_modified)

#     # # 可視化
#     # plt.imshow(img)
#     # plt.imshow(mask_modified, cmap="gray", alpha=0.5)
#     # plt.axis("off")
#     # plt.show()


def get_dominant_color(img: D1RGB_Array) -> Color:
    """
    Determines the dominant color in the image using k-means clustering.

    Parameters:
        img (np.ndarray): The input image array.

    Returns:
        np.ndarray: An array containing the dominant color [R, G, B].
    """
    pixels = np.array(img.reshape(-1, 3), dtype=np.float32)

    if len(img) > 10:
        n_colors = 3
    else:
        n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 5, flags)  # type: ignore

    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    dominant = np.array(np.clip(dominant, 0, 255), dtype=np.uint8)

    return cast(Color, dominant)


def increase_saturation(src: RGBPicture, factor: float) -> RGBPicture:
    """
    彩度を上げる関数

    Parameters
    ----------
    src : numpy.ndarray
        入力画像 (BGR)
    factor : float
        彩度を上昇させる倍率 (1.0以上で上昇)

    Returns
    -------
    numpy.ndarray
        彩度を上げた結果画像 (BGR)
    """
    # BGR -> HSV へ変換
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # HSVをチャンネルごとに分割 (channels[0]: H, channels[1]: S, channels[2]: V)
    h, s, v = cv2.split(hsv)

    # 彩度(S)を float32 化して倍率をかける
    s = s.astype(np.float32)
    s = s * factor

    # 上限を255にクリップ
    s = np.clip(s, 0, 255)

    # 彩度(S)を 8bit に戻す
    s = s.astype(np.uint8)

    # HSVを再度合成
    hsv_modified = cv2.merge([h, s, v])

    # HSV -> BGR へ戻す
    dst = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
    return cast(RGBPicture, dst)


def increase_brightness(src: RGBPicture, factor: float) -> RGBPicture:
    """
    明度を上げる関数

    Parameters
    ----------
    src : numpy.ndarray
        入力画像 (BGR)
    factor : int
        明度を増加させる量 (正の値で上昇)

    Returns
    -------
    numpy.ndarray
        明度を上げた結果画像 (BGR)
    """
    # BGR -> HSV へ変換
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # HSVをチャンネルごとに分割 (channels[0]: H, channels[1]: S, channels[2]: V)
    h, s, v = cv2.split(hsv)

    # 明度(V)を int32 化して加算
    v = v.astype(np.int32)
    v = v * factor

    # 0～255でクリップ
    v = np.clip(v, 0, 255)

    # 明度(V)を 8bit に戻す
    v = v.astype(np.uint8)

    # HSVを再度合成
    hsv_modified = cv2.merge([h, s, v])

    # HSV -> BGR へ戻す
    dst = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
    return cast(RGBPicture, dst)


def rgb_float_to_int(rgb: tuple[float, float, float]) -> Color:
    r, g, b = rgb
    r_int = max(0, min(255, int(round(r * 255))))
    g_int = max(0, min(255, int(round(g * 255))))
    b_int = max(0, min(255, int(round(b * 255))))
    return cast(Color, [r_int, g_int, b_int])


def convert_to_gray_scale(pic: _CommonPictureType) -> MonoColorPicture:
    if is_MonoColorPicture(pic):
        return pic
    elif is_RGBPicture(pic):
        res = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        return cast(MonoColorPicture, res)
    else:
        raise ValueError("pic cannot convet to MonoColorPicture")


def convert_to_hsv_picture(color_map: RGBPicture) -> HSVPicture:
    color_map_hsv = cv2.cvtColor(color_map, cv2.COLOR_RGB2HSV_FULL)
    return cast(HSVPicture, color_map_hsv)


def median_filter(color_map: _CommonPictureType, ksize: int = 3) -> _CommonPictureType:
    c = cv2.medianBlur(color_map, ksize=ksize)
    return cast(_CommonPictureType, c)


def get_color_element_by_index(
    color_map: RGBPicture | HSVPicture, index: Literal[0, 1, 2]
) -> MonoColorPicture:
    return cast(MonoColorPicture, color_map[:, :, index])


def extract_image_edges(
    image_input: _CommonPictureType, threshold1: float = 30, threshold2: float = 30
) -> MonoColorPicture:
    """
    Extracts edges from an image using the Canny edge detection algorithm.

    Parameters:
    - image_input: numpy.ndarray
        The input image as a NumPy array. It can be a file path (str) or an image array.
    - threshold1: float, default=30
        The first threshold for the hysteresis procedure in Canny edge detection.
    - threshold2: float, default=30
        The second threshold for the hysteresis procedure in Canny edge detection.

    Returns:
    - edges: numpy.ndarray
        The edge map of the input image.

    Raises:
    - ValueError:
        If the image cannot be opened from the provided file path.
    - TypeError:
        If the image_input is neither a string nor a NumPy ndarray.
    """

    image = convert_to_gray_scale(image_input)
    # if is_RGBPicture(image_input):
    #     image = cast(RGBPicture, cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY))
    # elif is_MonoColorPicture(image_input):
    #     image = image_input
    # else:
    #     raise TypeError(
    #         "image_input must be a file path (str) or an image array (numpy.ndarray)."
    #     )

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (21, 21), 0)

    # Detect edges using the Canny algorithm
    edges = cv2.Canny(blurred, threshold1, threshold2)

    return cast(MonoColorPicture, edges)


def auto_contrast(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Automatically adjusts the contrast of a grayscale image based on standard deviation and median.

    Parameters:
    - img: numpy.ndarray of type uint8
        The input grayscale image.

    Returns:
    - img: numpy.ndarray of type uint8
        The contrast-adjusted image.
    """
    # Calculate standard deviation and median of the image
    sd = np.std(img.flatten())
    m = np.median(img.flatten())

    # Define minimum and maximum intensity values
    min_img = m - 2 * sd
    max_img = m + 2 * sd

    # Clip the image to the defined intensity range
    img[img > max_img] = max_img
    img[img < min_img] = min_img

    # Normalize the image to the range [0, 255]
    img = np.array(img, dtype=np.float64)
    img = (img - min_img) / (max_img - min_img) * 255
    img = np.array(np.ceil(img), dtype=np.uint8)

    return img


def auto_contrast_rgb(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Automatically adjusts the contrast of an RGB image by processing each channel individually.

    Parameters:
    - img: numpy.ndarray of type uint8
        The input RGB image.

    Returns:
    - numpy.ndarray of type uint8
        The contrast-adjusted RGB image.
    """
    # Split the image into Red, Green, and Blue channels
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Apply auto contrast to each channel and stack them back
    return np.stack([auto_contrast(r), auto_contrast(g), auto_contrast(b)])


def as_binary_img(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Converts a grayscale image to a binary image based on the mean intensity.

    Parameters:
    - img: numpy.ndarray of type uint8
        The input grayscale image.

    Returns:
    - bimg: numpy.ndarray of type uint8
        The binary image where pixels are either 0 or 255.
    """
    # Calculate the mean intensity of the image
    m = np.mean(img.flatten())

    # Initialize a binary image with zeros
    bimg = np.zeros_like(img, dtype=np.uint8)

    # Set pixels above the mean to 255 and others to 0
    bimg[img > m] = 255
    bimg[img <= m] = 0

    return bimg


K = TypeVar("K", D2FloatArray, D2BoolArray, D2IntArray)


def resize_2d_float_array(
    arr: np.ndarray, new_width: int, new_height: int
) -> np.ndarray:
    """
    2次元float配列を、バイリニア補間を用いて指定したサイズ (new_height, new_width) にリサイズする関数。
    ベクトル演算で最適化した実装例。

    Parameters
    ----------
    arr : np.ndarray
        リサイズ対象の2次元配列 (float型を想定)。
        shape = (height, width)
    new_height : int
        リサイズ後の高さ。
    new_width : int
        リサイズ後の幅。

    Returns
    -------
    result : np.ndarray
        リサイズ後の2次元配列。
        shape = (new_height, new_width), dtype = arr.dtype (もとのfloat型が維持される)
    """
    if arr.ndim != 2:
        raise ValueError("入力配列は2次元配列である必要があります。")

    old_height, old_width = arr.shape

    # 出力用配列（元のdtypeを保持）
    result = np.empty((new_height, new_width), dtype=arr.dtype)

    # 新しい画像での y 座標 (0 ~ new_height-1) を、元の画像の y 座標 (0 ~ old_height-1) に線形マッピング
    # new_height=1 のときも linspace は要素数1の配列が返る
    y_coords = (
        np.linspace(0, old_height - 1, new_height)
        if new_height > 1
        else np.array([0.0])
    )
    x_coords = (
        np.linspace(0, old_width - 1, new_width) if new_width > 1 else np.array([0.0])
    )

    # floor (小数点以下切り捨て) として得られる整数座標
    y0 = np.floor(y_coords).astype(int)
    x0 = np.floor(x_coords).astype(int)

    # ceil の代わりに +1 しつつ境界内にクリップ
    y1 = np.minimum(y0 + 1, old_height - 1)
    x1 = np.minimum(x0 + 1, old_width - 1)

    # 小数部分
    dy = y_coords - y0
    dx = x_coords - x0

    # ここからバイリニア補間をベクトル演算で実行
    # 2次元目を x 軸として扱うために shape を合わせる:
    #   y0, y1 は (new_height,) だが x0, x1 は (new_width,)。
    #   配列同士をブロードキャストするために [:, None] や [None, :] を使う
    #
    # arr[y0, x0], arr[y0, x1], arr[y1, x0], arr[y1, x1] の4点をまとめて取り出す。
    # それぞれ shape = (new_height, new_width) の配列になる。

    # 上辺 (y0) の左(x0), 右(x1)
    top_left = arr[y0[:, None], x0[None, :]]
    top_right = arr[y0[:, None], x1[None, :]]
    # 下辺 (y1) の左(x0), 右(x1)
    bottom_left = arr[y1[:, None], x0[None, :]]
    bottom_right = arr[y1[:, None], x1[None, :]]

    # 水平方向の補間 (dx に応じて線形補間)
    # dx は shape=(new_width,) なので [None,:] を付けてブロードキャスト (new_height, new_width) に対応させる
    top = top_left * (1 - dx[None, :]) + top_right * dx[None, :]
    bottom = bottom_left * (1 - dx[None, :]) + bottom_right * dx[None, :]

    # 垂直方向の補間 (dy に応じて線形補間)
    # dy は shape=(new_height,) なので [:,None] を付けて (new_height, 1) にし、(new_height, new_width) にブロードキャスト
    result = top * (1 - dy[:, None]) + bottom * dy[:, None]

    return result


I = TypeVar("I", RGBPicture, MonoColorPicture, D2FloatArray, D2BoolArray, D2IntArray)


def resize_array(arr: I, width: int, height: Optional[int] = None) -> I:
    """
    配列を指定した幅 (width) もしくは幅と高さ (height) でリサイズする関数。
    height が指定されていない場合はアスペクト比が維持されるように自動で高さを計算する。

    Parameters
    ----------
    arr : numpy.ndarray
        リサイズ対象の配列。
        - bool, 整数型 (int), 浮動小数点型 (float) に対応。
    width : int
        リサイズ先の幅。
    height : Optional[int], default=None
        リサイズ先の高さ。指定しない場合はアスペクト比を維持した高さを自動計算する。

    Returns
    -------
    result : numpy.ndarray
        リサイズ後の配列。元のデータ型が維持される。
    """

    if height is not None:
        if (arr.shape[0] == height) and (arr.shape[1] == width):
            return arr
    else:
        if arr.shape[1] == width:
            return arr

    aspect_ratio = arr.shape[0] / arr.shape[1]

    new_width = width
    if height is None:
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = height

    # print(arr.dtype == np.int32)
    if arr.dtype == np.bool_:
        img = resize_img(
            cast(MonoColorPicture, arr.astype(np.uint8)), width, new_height
        )
        return cast(I, img.astype(np.bool_))
    elif arr.dtype == np.uint8:
        return cast(
            I,
            resize_img(arr, width, new_height),  # type: ignore
        )
    elif arr.dtype == np.float64:
        return cast(
            I,
            resize_2d_float_array(arr, width, new_height).astype(np.float64),  # type: ignore
        )
    elif arr.dtype == np.int32:
        return cast(
            I,
            resize_2d_float_array(arr, width, new_height).astype(np.float64),  # type: ignore
        )
    else:
        raise ValueError("The arr cannot resized")


def rotate_array(arr: I, angle: float, center: tuple[float, float]) -> I:

    if arr.dtype == np.bool_:
        return cast(
            I,
            rotate_image(
                cast(MonoColorPicture, arr.astype(np.uint8)), angle, center
            ).astype(np.bool_),
        )
    elif arr.dtype == np.uint8:
        return cast(I, rotate_image(arr, angle, center))  # type: ignore
    elif arr.dtype == np.float64:
        return cast(
            I,
            rotate_2d_float_array(arr, angle, center),  # type: ignore
        )
    elif arr.dtype == np.int32:
        return cast(
            I,
            rotate_2d_float_array(arr, angle, center),  # type: ignore
        )
    else:
        raise ValueError("The arr cannot rotated")


def rotate_image(
    image: _PictureType, angle: float, center: tuple[float, float]
) -> _PictureType:
    """
    画像を指定した角度（degree単位）で回転させる関数

    Parameters:
        image (numpy.ndarray): 入力画像
        angle (float): 回転角度（正の値は反時計回り）

    Returns:
        numpy.ndarray: 回転後の画像
    """
    if angle == 0:
        return image

    # 画像のサイズを取得
    (h, w) = image.shape[:2]
    # 画像の中心座標を計算
    if center is None:
        center = (w // 2, h // 2)

    # 回転行列を取得（スケールは1.0に設定）
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # アフィン変換で画像を回転
    rotated = cv2.warpAffine(image, M, (w, h))

    return cast(_PictureType, rotated)


def rotate_2d_float_array(
    arr: D2FloatArray, degree: float, center: tuple[int, int]
) -> D2FloatArray:
    """
    2次元のfloat型 np.ndarray を、指定した中心(center)を軸にして degree 度回転させた配列を返す関数です。
    入力と出力の shape は同じで、入力画像外に写ってしまう部分は 0 にしています。

    Parameters:
        arr (np.ndarray): 入力の2次元配列 (float型)
        center (tuple[int, int]): 回転の中心 (cx, cy)。ここで cx は横方向（列）の座標、cy は縦方向（行）の座標です。
        degree (float): 回転角度（度単位）

    Returns:
        np.ndarray: 回転後の配列（入力と同じ shape）
    """
    H, W = arr.shape
    # 回転中心の座標を展開 (cx: x座標（列方向）、cy: y座標（行方向）)
    cx, cy = center
    # 度をラジアンに変換
    theta = -np.deg2rad(degree)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 出力画像上の各画素の座標 (行 i, 列 j) のグリッドを作成
    I, J = np.indices((H, W))

    # 【逆写像】
    # 出力画像の画素 (i,j) が入力画像上のどこから来たかを求めるには，
    # 以下の式を用います（座標は (x,y) = (列,行) としています）:
    #
    #   [j, i]^T = R(theta) * ([x, y]^T - [cx, cy]^T) + [cx, cy]^T
    #
    # よって逆変換は R(-theta) を用いて
    #
    #   [x, y]^T = R(-theta) * ([j, i]^T - [cx, cy]^T) + [cx, cy]^T
    #
    # R(-theta) = [[cos(theta), sin(theta)],
    #              [-sin(theta), cos(theta)]]
    #
    # よって各画素について:
    x_src = cos_theta * (J - cx) + sin_theta * (I - cy) + cx
    y_src = -sin_theta * (J - cx) + cos_theta * (I - cy) + cy

    # 出力用配列をゼロで初期化
    output = np.zeros_like(arr)

    # バイリニア補間のために，周囲の整数座標を求める
    x0 = np.floor(x_src).astype(int)
    y0 = np.floor(y_src).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # 小数部分（重み）を計算
    dx = x_src - x0
    dy = y_src - y0

    # 補間に必要な4画素すべてが画像内にある領域のみ計算する
    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    # バイリニア補間
    output[valid] = (
        arr[y0[valid], x0[valid]] * (1 - dx[valid]) * (1 - dy[valid])
        + arr[y0[valid], x1[valid]] * dx[valid] * (1 - dy[valid])
        + arr[y1[valid], x0[valid]] * (1 - dx[valid]) * dy[valid]
        + arr[y1[valid], x1[valid]] * dx[valid] * dy[valid]
    )

    return output


def resize_img(
    img: _PictureType, width: int, height: Optional[int] = None
) -> _PictureType:
    """
    Resizes an image to a specified width while maintaining the aspect ratio.

    Parameters:
    - img: numpy.ndarray of type uint8
        The input image.
    - width: int
        The desired width of the resized image.

    Returns:
    - result: numpy.ndarray of type uint8
        The resized image.
    """

    if height is not None:
        if height == img.shape[0] and width == img.shape[1]:
            return cast(_PictureType, img)
    else:
        if width == img.shape[1]:
            return cast(_PictureType, img)

    # Calculate the aspect ratio
    aspect_ratio = img.shape[0] / img.shape[1]

    # Define the new width and calculate the new height to maintain aspect ratio
    new_width = width
    if height is None:
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = height

    # Resize the image using OpenCV
    resized_image = cv2.resize(img, (new_width, new_height), fx=0, fy=0)

    # Ensure the resized image is of type uint8
    result = np.array(resized_image, dtype=np.uint8)

    return cast(_PictureType, result)


def resize_image_list(
    img_list: list[_PictureType], width: int, height: Optional[int] = None
) -> list[_PictureType]:
    return list(map(lambda x: resize_img(x, width=width, height=height), img_list))


def compute_lab_distance_map(
    rgb_img: RGBPicture,
    color: Color,
) -> D2FloatArray:
    """
    指定した色(color)とのLab色空間での距離を、画像内の各ピクセルに対して求める関数

    Parameters
    ----------
    rgb_img : np.ndarray (H x W x 3, dtype=np.uint8)
        入力画像 (RGB形式)
    color : tuple (R, G, B)
        距離を計算したい対象の色 (RGB形式)

    Returns
    -------
    dist_matrix : np.ndarray (H x W, dtype=np.float64)
        各ピクセルと color の Lab色空間での距離 (ユークリッド距離)
    """
    # dist_matrixを画像と同じ高さ・幅で初期化
    dist_matrix = np.zeros(rgb_img.shape[:2], dtype=np.float64)

    # colorを1x1ピクセル画像にしてLab色空間に変換
    color_patch = np.array(
        [[[color[0], color[1], color[2]]]], dtype=np.uint8
    )  # (1,1,3)
    lab_color_patch = cv2.cvtColor(color_patch, cv2.COLOR_RGB2Lab)
    lab_color = lab_color_patch[0, 0, :]  # (L, a, b) のみ取り出し (形状: (3,))

    # 画像全体をLab色空間に変換
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab).astype(np.float64)

    # lab_colorもfloatに
    lab_color = lab_color.astype(np.float64)

    # ピクセル単位の差分をベクトル計算し、ユークリッド距離を求める
    diff = lab_img - lab_color  # (H x W x 3)
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))  # (H x W)

    return dist_matrix


def argmin_matrix(dist_matrix: D2FloatArray) -> tuple[int, int]:
    min_idx_1d = np.argmin(dist_matrix)
    min_index = np.unravel_index(min_idx_1d, dist_matrix.shape)  # (row_idx, col_idx)
    return cast(tuple[int, int], min_index)


def cut_out_image(img: _CommonPictureType, mask: D2BoolArray) -> Optional[RGBAPicture]:
    """
    マスクがTrueの部分だけを切り取った画像を返す関数。

    Parameters:
        img (np.ndarray): 元の画像。2次元または3次元のNumPy配列。
        mask (np.ndarray): ブール型のマスク。imgと同じサイズ。

    Returns:
        np.ndarray: マスクされた部分だけのRGBA画像。
    """
    # マスクがTrueの位置のインデックスを取得
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        # マスクが空の場合、Noneを返す
        return None

    # マスクがTrueの部分を含む最小の矩形領域を計算
    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()

    # 画像とマスクを切り取り
    cropped_img = img[min_y : max_y + 1, min_x : max_x + 1]
    cropped_mask = mask[min_y : max_y + 1, min_x : max_x + 1]

    # マスクを適用し、透過部分を設定
    if is_MonoColorPicture(cropped_img):
        # グレースケール画像の場合
        # グレースケール画像をRGBに変換
        cropped_img_masked = np.where(cropped_mask, cropped_img, 0)
        cropped_img_rgb_masked = np.stack((cropped_img_masked,) * 3, axis=-1)
    elif is_RGBPicture(cropped_img):
        # カラー画像の場合
        cropped_mask_3d = cropped_mask[:, :, np.newaxis]
        cropped_img_rgb_masked = np.where(cropped_mask_3d, cropped_img, 0)

    alpha_channel = np.where(cropped_mask, 255, 0).astype(np.uint8)

    cropped_img_rgba = np.dstack((cropped_img_rgb_masked, alpha_channel))

    return cast(RGBAPicture, cropped_img_rgba)


def create_outside_circle_mask(
    image: D2BoolArray | D2FloatArray | D2IntArray | RGBPicture | RGBAPicture,
) -> D2BoolArray:
    """
    入力画像(あるいは2次元配列)の中心から、
    はみ出さない最大の円の外側を True とするマスクを作成します。

    Parameters
    ----------
    image : np.ndarray
        2次元あるいは3次元(カラー画像の場合)のndarray。
        画素値の型は np.uint8, 整数, 実数, bool のいずれも可。

    Returns
    -------
    mask : np.ndarray (bool)
        2次元のbool配列で、円領域の外側が True となります。
    """
    # 高さ(H), 幅(W)を取得（カラー画像などのときは先頭2軸だけ使う）
    H, W = image.shape[:2]

    # 画像の中心
    cy, cx = H // 2, W // 2

    # 画像からはみ出さない最大の円の半径
    radius = min(cx, cy)

    # 座標値を作成
    # y: shape (H, 1)
    # x: shape (1, W)
    y, x = np.ogrid[:H, :W]

    # 中心からの距離の二乗を計算
    dist_sq = (y - cy) ** 2 + (x - cx) ** 2

    # dist_sq > radius^2 で「円の外側」を判定
    mask = dist_sq > radius**2

    return cast(D2BoolArray, mask)


# 使い方の例
if __name__ == "__main__":
    # 適当なダミー画像を作成(高さ100, 幅150)
    dummy_image: D2FloatArray = np.zeros((100, 150), dtype=np.uint8)
    # im = resize_img(dummy_image, 150)

    # マスクを作成
    mask = create_outside_circle_mask(dummy_image)

    # 結果確認
    plt.imshow(mask)
    a = cast(
        D1RGB_Array,
        np.array([[140, 162, 166], [151, 179, 181], [139, 171, 172], [148, 177, 179]]),
    )
    print(get_dominant_color(a))

    import pandas as pd

    from niconavi.tools.grain_plot import detect_boundaries
    from niconavi.type import ComputationResult

    r: ComputationResult = pd.read_pickle(
        "../../test/data/output/yamagami_cross_before_grain_classification.pkl"
    )

    img = r.grain_map
    if r.raw_maps is not None:
        max_reta_color_map = r.raw_maps["R_color_map"]
        ex_angle = r.raw_maps["extinction_angle"]

        if img is not None:
            plt.imshow(img)
            mask = detect_boundaries(img)
            img[mask] = 0
            plt.imshow(img, cmap="gray", alpha=0.5)
            plt.show()

            plt.imshow(max_reta_color_map)
            mask = detect_boundaries(img)
            img[mask] = 0
            plt.imshow(img)
            plt.show()

            # plt.imshow(ex_angle)
            if ex_angle is not None:
                cmap = cm.get_cmap("hsv")
                norm = plt.Normalize(vmin=ex_angle.min(), vmax=ex_angle.max())
                plt.imshow(cmap(norm(ex_angle)))
                plt.show()


# ----- 使用例 -----
if __name__ == "__main__":
    # 例として、中央に白い四角がある画像を作成し、45度回転してみる
    import matplotlib.pyplot as plt

    # 100x100 の黒画像
    img = np.ones((100, 200), dtype=float)
    # 中央に 40x40 の白四角を配置
    img[30:50, 30:70] = 2.0

    # 回転中心は画像の中心 (50, 50)
    rotated = rotate_2d_float_array(cast(D2FloatArray, img), center=(50, 50), degree=45)

    # 結果を表示
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray", origin="upper")
    axes[0].set_title("Original")
    axes[1].imshow(rotated, cmap="gray", origin="upper")
    axes[1].set_title("Rotated 45°")
    plt.show()
