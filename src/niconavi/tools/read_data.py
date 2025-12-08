import os
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import cast, Callable, Any
from tqdm import tqdm  # type: ignore
from niconavi.image.type import RGBPicture
import importlib.resources


# def divide_video_into_n_frame(
#     video_path: str,
#     n: int,
#     progress_callback: Callable[[float], None] = lambda p: None
# ) -> list[RGBPicture]:
#     """
#     動画ファイルから n 個の等間隔フレームを抽出して返す関数（最適化・修正版）。

#     Args:
#         video_path (str): 動画ファイルのパス。
#         n (int): 抽出するフレーム数。
#         progress_callback (Callable[[float], None]): 進捗更新用のコールバック関数（0～1の範囲）。

#     Returns:
#         list[RGBPicture]: 抽出したRGB画像のリスト。

#     Raises:
#         FileNotFoundError: 指定パスに動画ファイルが存在しない場合。
#         ValueError: 動画にフレームが存在しない、または特定のフレームが取得できなかった場合。
#     """
#     # 動画ファイルの存在確認
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"The specified video file does not exist: {video_path}")

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Could not open video file: {video_path}")

#     # 総フレーム数の取得
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total_frames <= 0:
#         cap.release()
#         raise ValueError("The video contains no frames.")

#     # 等間隔のインデックスを計算
#     frame_indices = np.linspace(0, total_frames - 1, n, dtype=int)
#     frames = []
#     current_frame_index = 0
#     pbar = tqdm(total=n, desc=f"Extracting {n} frames")

#     for target in frame_indices:
#         # 最初のフレームの場合は read() を使う
#         if target == 0 and current_frame_index == 0:
#             ret, frame = cap.read()
#             current_frame_index += 1
#             if not ret:
#                 cap.release()
#                 raise ValueError("Could not read frame at index 0.")
#         else:
#             # 目標フレームまで grab() でスキップ
#             while current_frame_index < target:
#                 if not cap.grab():
#                     cap.release()
#                     raise ValueError(f"Could not grab frame at index {current_frame_index}.")
#                 current_frame_index += 1
#             # grab() したフレームを retrieve() で取得
#             ret, frame = cap.retrieve()
#             current_frame_index += 1
#             if not ret:
#                 cap.release()
#                 raise ValueError(f"Could not read frame at index {current_frame_index - 1}.")

#         # BGR → RGB 変換
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)
#         pbar.update(1)
#         progress_callback(len(frames) / n)

#     pbar.close()
#     cap.release()
#     return cast(list[RGBPicture], frames)


# def divide_video_into_n_frame(
#     video_path: str, n: int, progress_callback: Callable[[float], None] = lambda p: None
# ) -> list[RGBPicture]:
#     """
#     Divides the specified video into n equally spaced frames and returns them as a NumPy array.

#     Args:
#         video_path (str): The file path to the video that needs to be divided.
#         n (int): The number of frames to extract from the video.

#     Returns:
#         np.ndarray: A NumPy array containing the extracted frames. The shape of the array is (n, height, width, channels), where height, width, and channels correspond to the dimensions of each frame.

#     Raises:
#         FileNotFoundError: If the video file does not exist at the specified video_path.
#         ValueError: If the video contains no frames or if a specific frame cannot be read.
#     """
#     # Check if the video file exists
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(
#             f"The specified video file does not exist: {video_path}"
#         )

#     # Open the video file

#     cap = cv2.VideoCapture(video_path)

#     # Get the total number of frames in the video
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total_frames == 0:
#         cap.release()
#         raise ValueError("The video contains no frames.")

#     # Calculate the indices of the frames to extract
#     frame_indices = np.linspace(0, total_frames - 1, n, dtype=int)

#     # Initialize a list to store the extracted frames
#     frames = []
#     for i, idx in enumerate(tqdm(frame_indices, desc=f"Divide video into {n} frame")):

#         progress_callback(i / n)

#         # Set the current position of the video file to the frame index
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         success, frame = cap.read()

#         if success:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)
#         else:
#             cap.release()
#             raise ValueError(f"Could not read frame at index {idx}.")

#     # Release the video capture object
#     cap.release()

#     # Convert the list of frames to a NumPy array
#     # frames_array = np.array(frames)

#     return cast(list[RGBPicture], frames)


def divide_video_into_n_frame(
    video_path: str, n: int, progress_callback: Callable[[float], None] = lambda p: None
) -> list[RGBPicture]:
    """
    動画ファイルから n 個の等間隔フレームを抽出して返す関数（set()を利用した修正版）。

    Args:
        video_path (str): 動画ファイルのパス。
        n (int): 抽出するフレーム数。
        progress_callback (Callable[[float], None]): 進捗更新用のコールバック関数（0～1の範囲）。

    Returns:
        list[RGBPicture]: 抽出したRGB画像のリスト。

    Raises:
        FileNotFoundError: 指定パスに動画ファイルが存在しない場合。
        ValueError: 動画にフレームが存在しない、または特定のフレームが取得できなかった場合。
    """
    import os
    import cv2
    import numpy as np
    from tqdm import tqdm
    from typing import Callable, cast

    # 動画ファイルの存在確認
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"The specified video file does not exist: {video_path}"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # 総フレーム数の取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("The video contains no frames.")

    # 等間隔のインデックスを計算（先頭～末尾まで）
    frame_indices = np.linspace(0, total_frames - 1, n, dtype=int)
    frames = []
    pbar = tqdm(total=n, desc=f"Extracting {n} frames")

    for target in frame_indices:
        # シークしてからフレームを取得
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Could not read frame at index {target}.")
        # BGR → RGB 変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        pbar.update(1)
        progress_callback(len(frames) / n)

    pbar.close()
    cap.release()
    return cast(list[RGBPicture], frames)


def get_first_frame_from_video(video_path: str) -> RGBPicture:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"The specified video file does not exist: {video_path}"
        )

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cast(RGBPicture, frame)
    else:
        raise ValueError("The video contains no frames.")


def load_default_data(filename: str) -> Any:
    with importlib.resources.path("niconavi.package_data", filename) as f:
        content = np.load(f)
    return content


import numpy as np
import matplotlib.pyplot as plt


def plot_3d_boolean_images(imgs: list[np.ndarray], delta_z: float) -> None:
    """
    2DのBool配列を要素とするリストimgsを、
    それぞれz軸方向にdelta_z間隔で並べて3次元にプロットする。
    Trueの画素の位置にのみ点を描画し、Falseの画素には何も描画しない。

    Parameters
    ----------
    imgs : list of np.ndarray
        各要素は2Dのbool型配列。
    delta_z : float
        z軸方向の間隔。
    """

    # x, y, z 座標を溜めるリスト
    xs = []
    ys = []
    zs = []

    # 各画像を z = i * delta_z に配置しつつ、Trueの画素だけ座標を取得
    for i, img in enumerate(imgs):
        # z 座標
        z_val = i * delta_z

        # True の位置を特定 (row_idsがy, col_idsがx)
        row_ids, col_ids = np.where(img)

        xs.extend(col_ids)
        ys.extend(row_ids)
        zs.extend([z_val] * len(row_ids))

    # 3D プロット用のFigureとAxesを作成
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # 3D空間に散布図として描画
    ax.scatter(xs, ys, zs, s=1)  # s=1で点サイズを小さくする例

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 描画
    plt.show()


# 実行例
if __name__ == "__main__":
    # サンプルとして、2つの2D Bool配列を用意
    img1 = np.array(
        [[False, True, True], [True, False, False], [True, True, False]], dtype=bool
    )

    img2 = np.array(
        [[False, False, True], [False, True, True], [True, True, True]], dtype=bool
    )

    # サンプル呼び出し
    imgs = [img1, img2]
    plot_3d_boolean_images(imgs, delta_z=2.0)
