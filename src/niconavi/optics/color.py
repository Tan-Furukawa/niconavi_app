# %%
import cv2
import colorsys
from typing import cast, overload
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from colour import (
    plotting,
    SDS_ILLUMINANTS,
    MSDS_CMFS,
    SpectralDistribution,
    sd_to_XYZ,
    XYZ_to_sRGB,
)

from niconavi.optics.types import (
    DictColor,
    WavelengthVector,
    ReflectanceVector,
    is_WavelengthVector,
)
from niconavi.tools.type import D1IntArray, D1FloatArray, D2FloatArray
from niconavi.image.type import Color, D1RGB_Array, RGBPicture

@overload
def convert_to_Lab_color(rgb_color: Color) -> Color: ...


@overload
def convert_to_Lab_color(rgb_color: D1RGB_Array) -> D1RGB_Array: ...


def convert_to_Lab_color(rgb_color):  # type: ignore
    if np.ndim(rgb_color) == 1:
        color_bgr = np.array([[rgb_color]], dtype=np.uint8)  # 形状: (1, 1, 3)
        color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2LAB)
        lab = color_lab[0, 0, :]
        return lab
        # BGRからLab色空間へ変換
    elif np.ndim(rgb_color) == 2:
        color_bgr = np.array([rgb_color], dtype=np.uint8)  # 形状: (1, 1, 3)
        color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2LAB)
        lab = color_lab[0, :]
        return lab
    else:
        raise ValueError(
            "The dimension of rgb_color in convert_to_Lab_color should be 1 or 2"
        )

def convert_rgb_to_hsv(rgb_pixel: Color) -> Color:
    """
    1画素 (RGB) を HSV (dtype=np.uint8) に変換する関数。
    shape=(3,), dtype=np.uint8 の [R, G, B] を想定。
    Returns:
        hsv_pixel: shape=(3,), dtype=np.uint8 の [H, S, V]
                   H ∈ [0, 179], S, V ∈ [0, 255]
    """
    # まず形状を (1,1,3) に変換
    pixel_2d = rgb_pixel.reshape(1, 1, 3)  # (1,1,3)

    # cvtColorでRGB->HSVに変換
    hsv_2d = cv2.cvtColor(pixel_2d, cv2.COLOR_RGB2HSV)  # (1,1,3)

    # 再び shape=(3,) に整形して返す
    hsv_pixel = hsv_2d[0, 0]  # shape=(3,)
    return hsv_pixel


@overload
def rgb_distance_between_color(rgb_color: Color, rgb_colors: Color) -> float: ...
@overload
def rgb_distance_between_color(
    rgb_color: Color, rgb_colors: D1RGB_Array
) -> D1FloatArray: ...
def rgb_distance_between_color(rgb_color, rgb_colors):  # type: ignore
    if np.ndim(rgb_colors) == 1:
        return np.linalg.norm(
            rgb_color.astype(np.float64) - rgb_colors.astype(np.float64)
        )
    elif np.ndim(rgb_colors) == 2:
        return np.linalg.norm(
            rgb_color.astype(np.float64) - rgb_colors.astype(np.float64),
            axis=1,
        )
    else:
        raise ValueError(
            "The dimension of rgb_color in convert_to_Lab_color should be 1 or 2"
        )


@overload
def lab_distance_between_color(rgb_color: Color, rgb_colors: Color) -> float: ...
@overload
def lab_distance_between_color(
    rgb_color: Color, rgb_colors: D1RGB_Array
) -> D1FloatArray: ...
def lab_distance_between_color(rgb_color, rgb_colors):  # type: ignore
    if np.ndim(rgb_colors) == 1:
        return np.linalg.norm(
            convert_to_Lab_color(rgb_color).astype(np.float64)
            - convert_to_Lab_color(rgb_colors).astype(np.float64)
        )
    elif np.ndim(rgb_colors) == 2:
        return np.linalg.norm(
            convert_to_Lab_color(rgb_color).astype(np.float64)
            - convert_to_Lab_color(rgb_colors).astype(np.float64),
            axis=1,
        )
    else:
        raise ValueError(
            "The dimension of rgb_color in convert_to_Lab_color should be 1 or 2"
        )


def get_yellowness(color: np.ndarray) -> float:
    """
    与えられた RGB カラー (0–255 の整数値) について、
    HSV色空間に変換した上で「どのくらい黄色らしいか」を
    0.0〜1.0 で返す。

    - color: shape (3,) の NumPy 配列 [R, G, B]
             (各要素は 0〜255 の整数値)
    - return: 0.0〜1.0 の浮動小数点数
    """
    # RGB(0–255) → (0.0–1.0) に正規化
    r, g, b = color / 255.0

    # colorsys の rgb_to_hsv は それぞれ [0,1] の範囲で Hue, Saturation, Value を返す
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Hueを度数（0〜360 度）へ換算
    hue_deg = h * 360.0

    # 「60度(黄色) とどのくらい離れているか」を計算
    # Hue の円環性を考慮し、例えば hue_deg=359 と hue_deg=1 は 2度の差にできるように
    diff = abs(hue_deg - 60)
    diff = min(diff, 360 - diff)  # 色相環での距離に直す

    # diff=0 度 → 完全に 60度 → 黄色
    # diff=180 度 → 正反対（青寄りまたは赤寄り）→ 黄色味 0
    # それ以上の差については 0 とみなす
    # まず 0〜180 度を 0〜1 に正規化し、それを 1-... することで
    # 「近いほど1、大きく離れるほど0」に変換
    hue_score = max(0.0, 1.0 - diff / 180.0)

    # 最終的な「黄色らしさ」として、Hueの近さ × 彩度 × 明度 を乗じる
    yellowness = hue_score * s * v

    return yellowness


@overload
def yellow_distance_between_color(rgb_color: Color, rgb_colors: Color) -> float: ...
@overload
def yellow_distance_between_color(
    rgb_color: Color, rgb_colors: D1RGB_Array
) -> D1FloatArray: ...
def yellow_distance_between_color(rgb_color, rgb_colors):  # type: ignore
    if np.ndim(rgb_colors) == 1:
        return np.linalg.norm(
            get_yellowness(rgb_color).astype(np.float64)
            - get_yellowness(rgb_colors).astype(np.float64)
        )
    elif np.ndim(rgb_colors) == 2:
        return np.linalg.norm(
            get_yellowness(rgb_color).astype(np.float64)
            - get_yellowness(rgb_colors).astype(np.float64),
            axis=1,
        )
    else:
        raise ValueError(
            "The dimension of rgb_color in convert_to_Lab_color should be 1 or 2"
        )


# should remove this fn
def get_dominant_color(img: NDArray) -> NDArray:
    """
    Determines the dominant color in the image using k-means clustering.

    Parameters:
        img (np.ndarray): The input image array.

    Returns:
        np.ndarray: An array containing the dominant color [R, G, B].
    """
    pixels = np.array(img.reshape(-1, 3), dtype=np.float32)

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 5, flags)  # type: ignore

    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    dominant = np.array(np.clip(dominant, 0, 255), dtype=np.uint8)
    return dominant


def convert_rgb_dict_to_color_vector(dic: DictColor) -> Color:
    """
    Converts an RGB dictionary to a color vector.

    Parameters:
        dic (Dict[str, int]): A dictionary with keys 'R', 'G', 'B' and integer values.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing [R, G, B].
    """
    col = np.array(
        [
            dic["R"],
            dic["G"],
            dic["B"],
        ],
        dtype=np.uint8,
    )
    return cast(Color, col)


def convert_color_vector_to_rgb_dict(vec: Color) -> DictColor:
    """
    Converts a color vector to an RGB dictionary.

    Parameters:
        vec (np.ndarray): A NumPy array of shape (3,) representing [R, G, B].

    Returns:
        Dict[str, int]: A dictionary with keys 'R', 'G', 'B' and corresponding integer values.

    Raises:
        ValueError: If the input vector does not have shape (3,).
    """
    if vec.shape == (3,):
        return {"R": int(vec[0]), "G": int(vec[1]), "B": int(vec[2])}
    else:
        raise ValueError("Invalid input vector")


def convert_float_vector_to_uint_color(vec: NDArray) -> NDArray[np.uint8]:
    vec = np.clip(vec, 0, 255)
    return vec.astype(np.uint8)


def convert_01_vector_to_uint_color(vec: NDArray) -> NDArray[np.uint8]:
    vec = np.clip(vec * 256, 0, 255)
    return vec.astype(np.uint8)


def get_average_color(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Calculates the average color of the image.

    Parameters:
        img (np.ndarray): The input image array.

    Returns:
        np.ndarray: An array containing the average color [R, G, B].
    """
    average = img.mean(axis=0).mean(axis=0)
    average = np.array(np.clip(average, 0, 255), dtype=np.uint8)
    return average


def norm(x: NDArray, m: float = 0, sd: float = 1) -> NDArray:
    return 1 / (np.pi * 2 * sd) * np.exp(-((((x - m) / (sd * 2))) ** 2))


def custom_spectrum(type: str) -> SpectralDistribution:
    if type == "red10":
        x = np.linspace(300, 780, num=97)
        y = norm(x, 600, 10)
        illuminant = SpectralDistribution(data=dict(zip(x, y)))
        return illuminant
    else:
        raise TypeError("spectrum distribution")


def get_color_from_spectral_distribution(
    wavelengths: WavelengthVector,
    reflectance: ReflectanceVector,
    illuminant: SpectralDistribution = SDS_ILLUMINANTS["D65"],
) -> Color:

    if is_WavelengthVector(wavelengths):
        cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

        # スペクトル分布の作成
        sd_sample = SpectralDistribution(data=dict(zip(wavelengths, reflectance)))

        # print(sd_sample)

        # XYZ表色系への変換
        XYZ = sd_to_XYZ(sd_sample, cmfs=cmfs, illuminant=illuminant)

        # sRGBへの変換
        RGB = XYZ_to_sRGB(XYZ / 100)
        RGB = np.clip(RGB * 256, 0, 255).astype(np.uint8)
        return cast(Color, RGB)
    else:
        raise TypeError("invalid type wavelengths")


def show_color(rgb: NDArray) -> None:
    # 結果の表示
    plt.figure(figsize=(2, 2))
    plt.imshow([[rgb]])
    plt.axis("off")
    plt.show()


def plot_RGB_colourspaces() -> None:
    plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
        ["sRGB"], standalone=False, diagram_opacity=1.0
    )


if __name__ == "__main__":
    # 例: 純粋な黄色 [255, 255, 0]
    color_yellow = np.array([255, 255, 0], dtype=np.uint8)
    print(get_yellowness(color_yellow))  # 期待: 1.0 付近

    # 例: 灰色 [128, 128, 128]
    color_gray = np.array([128, 128, 128], dtype=np.uint8)
    print(get_yellowness(color_gray))  # 期待: 0.0 付近

    # 例: オレンジ [255, 165, 0]
    # Hueは約39度、S=1.0、V=1.0
    # 黄色との距離は21度 / 180 ≈ 0.1167 → hue_score ≈ 0.8833
    # → 0.8833 * 1.0 * 1.0 → 0.8833
    color_orange = np.array([255, 165, 0], dtype=np.uint8)
    print(get_yellowness(color_orange))  # 0.88 前後になるはず
    # 波長範囲と反射率データの定義（5nm間隔）

    def test_spectral_color_from_actual_distribution() -> None:

        # test color estimated from normal distribution with m mean and sd standard deviation.
        def estimate_expected_color(m: float, sd: float, argmax_rgb: int) -> bool:
            wavelengths = np.arange(380, 781, 5)  # from 380nm to 780nm wavelength

            reflectance = norm(wavelengths, m, sd) / np.max(norm(wavelengths, m, sd))

            rgb = get_color_from_spectral_distribution(
                wavelengths, reflectance, illuminant=SDS_ILLUMINANTS["D65"]
            )

            # plt.plot(wavelengths, reflectance)
            # show_color(rgb)

            return np.argmax(rgb) == argmax_rgb

        assert estimate_expected_color(m=660, sd=10, argmax_rgb=0)
        assert estimate_expected_color(m=500, sd=10, argmax_rgb=1)
        assert estimate_expected_color(m=440, sd=10, argmax_rgb=2)

    test_spectral_color_from_actual_distribution()
