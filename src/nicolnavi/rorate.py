# %%
from niconavi.tools.read_data import divide_video_into_n_frame
import warnings
import imageio
from copy import deepcopy
from typing import Union, Optional, Any, Literal, TypeVar, cast, Callable
from niconavi.image.image import (
    extract_image_edges,
    resize_img,
)
import numpy as np
from numpy.typing import NDArray
import cv2
from cv2.typing import MatLike
import matplotlib.pyplot as plt
from tqdm import tqdm  # type: ignore

from niconavi.image.type import RGBPicture, _CommonPictureType, MonoColorPicture
from niconavi.tools.type import D1FloatArray, D2BoolArray

__all__ = ["fitting_by_rotation"]
import concurrent.futures


def find_rotation_and_overlay(
    center: tuple[int, int],
    img1: RGBPicture | MonoColorPicture,
    img2: RGBPicture | MonoColorPicture,
    angle_range: tuple[float, float] = (0, 5),
    angle_step: float = 0.1,
) -> tuple[float, bool]:
    """
    指定した中心点を軸に第2の画像を回転させ、参照画像と最も一致する角度を探索します。

    Parameters:
        center (tuple[int, int]): 回転の中心座標。
        img1 (numpy.ndarray): 参照画像（グレースケール）。
        img2 (numpy.ndarray): 回転対象の画像（グレースケール）。
        angle_range (tuple[float, float]): 探索する角度の範囲 (start_angle, end_angle)。
        angle_step (float): 角度のステップサイズ。

    Returns:
        best_angle (float): 最も一致度が高かった回転角度。
        can_find (bool): 探索範囲内で適切な角度が見つかったかどうか。
    """

    # 参照画像のサイズ取得
    height, width = img1.shape[:2]  # type: ignore
    # print("=================")
    # print(center)
    # print(height)
    # print(width)
    # print("=================")
    angles = np.arange(angle_range[0], angle_range[1], angle_step)

    # 各角度における回転と類似度計算を行う関数
    def compute_similarity(angle: float) -> tuple[float, float]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img2 = cv2.warpAffine(img2, M, (width, height))
        # cv2.matchTemplate は内部で C++ コードが走るのでスレッドでの並列化が可能
        similarity = cv2.matchTemplate(img1, rotated_img2, cv2.TM_CCOEFF_NORMED).max()
        return angle, similarity

    # 並列処理で各角度の類似度を計算
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_similarity, angles))

    # 類似度が最大となる角度を選択
    best_angle, best_similarity = max(results, key=lambda x: x[1])

    # 探索範囲の最終角度で結果が得られている場合は、探索範囲内に十分な結果がなかったと判断
    can_find = best_angle != angles[-1]
    return best_angle, can_find


def rotate_image(
    center: tuple[int, int],
    image: _CommonPictureType,
    angle: float,
    scale: float = 1.0,
) -> _CommonPictureType:
    """
    Rotates an image around a specified center point by a given angle.

    Parameters:
        image (np.ndarray): The input image to rotate.
        angle (float): The rotation angle in degrees. Positive values mean counter-clockwise rotation.
        center (Tuple[float, float]): The (x, y) coordinates of the rotation center.
        scale (float, optional): Scaling factor. Default is 1.0.

    Returns:
        np.ndarray: The rotated image.
    """
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Get the sine and cosine components
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Compute the new bounding dimensions of the image
    h, w = image.shape[:2]
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += new_w / 2 - center[0]
    rotation_matrix[1, 2] += new_h / 2 - center[1]

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    return cast(_CommonPictureType, rotated_image)


def crop_circle(
    img: _CommonPictureType, center: tuple[int, int], r: int
) -> _CommonPictureType:
    """
    Crops the image into a circular region with a specified center and radius.

    Parameters:
    img (np.ndarray): Input image array.
    center (tuple[int, int]): Center coordinates of the circle (x, y).
    r (int): Radius of the circle.

    Returns:
    np.ndarray: The image cropped into a circular shape.
    """
    height, width = img.shape[:2]
    cx, cy = center

    # Initialize the output image
    cropped_img = np.zeros_like(img)

    # Create grids of y and x coordinates
    y, x = np.ogrid[:height, :width]

    # Calculate the distance from the center
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Create a mask (True inside the circle, False outside)
    mask = dist_from_center <= r

    # Apply the mask
    if img.ndim == 2:
        cropped_img[mask] = img[mask]
    else:
        cropped_img[mask] = img[mask]

    return cropped_img


def create_gif_of_pics(
    pics: list[RGBPicture],
    output_path: str,
    duration: float = 0.5,
    loop: int = 0,
) -> None:
    """
    Creates a GIF animation from a list of images.

    Parameters:
        images (List[Union[np.ndarray, str]]): A list of images to include in the GIF.
            Each image can be either a NumPy array or a file path to an image.
        output_path (str): The file path where the GIF will be saved.
        duration (float, optional): Duration between frames in seconds. Default is 0.5 seconds.
        loop (int, optional): Number of times the GIF should loop. 0 means infinite loop. Default is 0.

    Returns:
        None
    """
    frames: list[Any] = []
    for img in pics:
        if isinstance(img, str):
            frame = imageio.imread(img)
        elif isinstance(img, np.ndarray):
            frame = img
        else:
            raise ValueError("Each image must be a file path or a NumPy array.")
        frames.append(frame)

    imageio.mimsave(output_path, frames, duration=duration, loop=loop)


def find_best_rotation_and_overlay(
    center: tuple[int, int],
    img1: _CommonPictureType,
    img2: _CommonPictureType,
    initial_angle_range: tuple[float, float] = (-10, 10),
) -> float:

    angle, within_first_search = find_rotation_and_overlay(
        center, img1, img2, angle_range=initial_angle_range, angle_step=2
    )

    if not within_first_search:
        angle, within_2nd_search = find_rotation_and_overlay(
            center, img1, img2, angle_range=(-20, 20)
        )
        if not within_2nd_search:
            Warning(
                f"Cannot find best rotation angle between two image. If this warning appears, there is a risk of a low accuracy output. Maybe the rotation speed of your thin section movie is too fast."
            )

    d1 = 1.2
    angle, _ = find_rotation_and_overlay(
        center,
        img1,
        img2,
        angle_range=(angle - d1, angle + d1),
        angle_step=0.1,
    )

    d2 = 0.06
    angle, _ = find_rotation_and_overlay(
        center,
        img1,
        img2,
        angle_range=(angle - d2, angle + d2),
        angle_step=0.01,
    )
    return angle


def estimate_rotation_angles_of_pics(
    center: tuple[int, int],
    pics: list[_CommonPictureType],
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> tuple[D1FloatArray, float]:
    diff_angles = np.zeros(len(pics))
    n = len(pics)
    for i in tqdm(range(n), desc="rotating image and find angles"):
        progress_callback(i / n)
        if i == 0:
            continue

        img1 = pics[i - 1]
        img2 = pics[i]
        diff_angles[i] = find_best_rotation_and_overlay(center, img1, img2)

    progress_callback(None)
    angles = np.cumsum(diff_angles)

    img1 = pics[0]
    img2 = pics[-1]
    last_angle = find_best_rotation_and_overlay(
        center, img1, img2, initial_angle_range=(-90, 90)
    )

    if angles[-1] > 0:
        expected_last_angle = 360 + last_angle
    else:
        expected_last_angle = -360 + last_angle

    if np.abs(expected_last_angle - angles[-1]) > 5:
        Warning(
            "cumulated error cause the difference between diff_angles and expected angle. Maybe the rotation speed of your thin section movie is too fast."
        )

    return cast(D1FloatArray, angles), expected_last_angle


def normalize_angle_by_expected_last_angle(
    angles: D1FloatArray, expected_last_angle: float
) -> D1FloatArray:
    result = (
        angles[0] / (angles[-1] - angles[0]) * (angles[-1] - angles)
        + expected_last_angle / (angles[-1] - angles[0]) * angles
    )
    return result


def rotate_square_image(img: _CommonPictureType, angle: float) -> _CommonPictureType:
    """
    Rotates a given square image around its center pixel by a specified angle.

    Parameters:
    img (np.ndarray): Input square image array.
    angle (float): Rotation angle in degrees.

    Returns:
    np.ndarray: The rotated image.

    Raises:
    ValueError: If the image size is even or not square.
    """
    height, width = img.shape[:2]

    # Ensure the image is square and has an odd size
    if height != width or height % 2 == 0:
        raise ValueError("The image must be a square with an odd size.")

    # Coordinates of the center pixel
    center = (width // 2, height // 2)

    # Obtain the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply affine transformation to rotate the image
    rotated_img = cv2.warpAffine(
        img,
        M,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,  # Fill with colorless pixels
    )  # type: ignore

    return rotated_img


def extract_square(
    img: _CommonPictureType, center: tuple[int, int], size: int
) -> _CommonPictureType:
    """
    Extracts a square region from an image centered at a given point, with a side length of 2 * size + 1.

    Parameters:
    img (np.ndarray): Input image array.
    center (tuple[int, int]): (x, y) coordinates of the center pixel.
    size (int): Half of the side length of the square.

    Returns:
    np.ndarray: The extracted square image.
    """
    cx, cy = center  # x and y coordinates (columns and rows)
    height, width = img.shape[:2]

    # Determine the number of channels
    if img.ndim == 2:
        channels = 1
    else:
        channels = img.shape[2]  # type: ignore

    # Initialize the output image with colorless pixels
    if channels == 1:
        S = np.zeros((2 * size + 1, 2 * size + 1), dtype=img.dtype)
    else:
        S = np.zeros((2 * size + 1, 2 * size + 1, channels), dtype=img.dtype)

    # Calculate the coordinate ranges in the original image
    x_start_img = max(0, cx - size)
    x_end_img = min(width, cx + size + 1)
    y_start_img = max(0, cy - size)
    y_end_img = min(height, cy + size + 1)

    # Calculate the corresponding coordinate ranges in the output image
    x_start_S = x_start_img - (cx - size)
    x_end_S = x_end_img - (cx - size)
    y_start_S = y_start_img - (cy - size)
    y_end_S = y_end_img - (cy - size)

    # Copy the overlapping region from the original image to the output image
    S[y_start_S:y_end_S, x_start_S:x_end_S] = img[
        y_start_img:y_end_img, x_start_img:x_end_img
    ]

    return cast(_CommonPictureType, S)


def min_distance_to_edge(img: _CommonPictureType, center: tuple[int, int]) -> int:
    """
    Calculates the shortest distance from a specified center pixel to the edge of the image.

    Parameters:
    img (np.ndarray): Input image array.
    center (tuple[int, int]): (x, y) coordinates of the center pixel.

    Returns:
    int: The shortest distance from the center to the edge of the image.
    """
    cx, cy = center  # x and y coordinates (columns and rows)
    height, width = img.shape[:2]

    # Verify that the center coordinates are within the image bounds
    if not (0 <= cx < width) or not (0 <= cy < height):
        raise ValueError("The center coordinates are outside the image bounds.")

    # Calculate distances in each direction
    distance_left = cx
    distance_right = width - 1 - cx
    distance_top = cy
    distance_bottom = height - 1 - cy

    # Get the minimum distance
    min_distance = min(distance_left, distance_right, distance_top, distance_bottom)

    return min_distance


def resize_pics(pics: list[RGBPicture], width: int) -> list[RGBPicture]:
    return list(map(lambda x: resize_img(x, width), pics))


def detect_edges_of_pics(pics: list[RGBPicture]) -> list[MonoColorPicture]:
    return list(map(lambda x: extract_image_edges(x, 30, 30), list(pics)))


def determine_rotation_angles_in_pics(
    edge_pics: list[MonoColorPicture],
    center: tuple[int, int],
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> D1FloatArray:
    if edge_pics is not None:
        angles, expected_last_angle = estimate_rotation_angles_of_pics(
            center, edge_pics, progress_callback
        )
    else:
        raise ValueError(
            "There is no edge pictures. make edge pictures by prepare_fitting_image"
        )

    progress_callback(None)
    normalized_angles = normalize_angle_by_expected_last_angle(
        angles, expected_last_angle
    )

    return normalized_angles


def crop_image(
    pics: list[RGBPicture], angles: D1FloatArray, center: tuple[int, int]
) -> list[RGBPicture]:
    if angles is not None:
        for i, pic in enumerate(pics):
            angle = angles[i]

            r = min_distance_to_edge(pic, center)

            pic_tmp = extract_square(pic, center, r)
            # plt.imshow(pic_tmp)
            # plt.show()
            pic_tmp = rotate_square_image(pic_tmp, angle)
            # plt.imshow(pic_tmp)
            # plt.show()
            pic_tmp = crop_circle(pic_tmp, (r, r), r)
            # plt.imshow(pic_tmp)
            # plt.show()
            pics[i] = pic_tmp
            # if i > 2:
            #     raise ValueError("stop")
        return pics

    else:
        raise ValueError("run crop_image after self.fitting().")


def fitting_by_rotation(
    pics: list[RGBPicture],
    center: tuple[int, int],
    width: int,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> tuple[list[RGBPicture], D1FloatArray]:
    progress_callback(None)
    rpics = resize_pics(pics, width)
    edge_pics = detect_edges_of_pics(rpics)
    angels = determine_rotation_angles_in_pics(edge_pics, center, progress_callback)
    cpics = crop_image(rpics, angels, center)
    return cpics, angels


if __name__ == "__main__":

    # center = (516, 268)
    # center = (554, 293)
    # pics = np.load("../data/pic_example3.npy")

    # center = (487, 242)
    center = (521, 271)
    width = 1000

    # pics = divide_video_into_n_frame("../tmp/Movie_65.avi", 100)
    # pics = divide_video_into_n_frame("../test/data/tetori_cross.avi", 100)
    pics_raw = divide_video_into_n_frame(
        # "../test/data/input/yamagami2/Movie_245.avi", 100
        # "../test/data/input/yamagami2/Movie_245.avi", 100
        "../test/data/input/sector/Movie_303.avi",
        100,
    )

    # make_computation_result("tetori_cross.avi", "yamagami_reta.avi")

    pics, angles = fitting_by_rotation(pics_raw, center, width)
    plt.plot(angles)
#%%

    # plt.imshow(pics[0])
    # plt.show()
    # plt.imshow(pics[40])
    # plt.show()
    # plt.imshow(pics[30])
    # plt.show()

    # rotation = FittingImgByRotation(center, width, list(pics))
    # rotation.prepare_fitting_image()
    # rotation.fitting()
