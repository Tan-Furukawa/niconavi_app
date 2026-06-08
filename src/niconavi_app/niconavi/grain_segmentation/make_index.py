import numpy as np
import cv2
import numpy as np
import cv2

def label_regions_with_skeleton(skeleton: np.ndarray, connectivity: int = 4, max_iters: int | None = None) -> np.ndarray:
    """
    スケルトン画像（True=線, False=領域）から、
    False の連結成分に 1..N のラベルを振り、スケルトン上にも 1..N を割り当てる。
    OpenCV の形態学演算は使わず、NumPy で 3x3 最大値フィルタを実装して拡散する。
    """
    if skeleton.dtype != np.bool_:
        skeleton = skeleton.astype(bool)
    if skeleton.ndim != 2:
        raise ValueError("2次元のスケルトン画像のみ対応しています。")

    region_mask = (~skeleton).astype(np.uint8)

    if connectivity == 8:
        conn = 8
    elif connectivity == 4:
        conn = 4
    else:
        raise ValueError("connectivity は 4 または 8 を指定してください。")

    num_labels, labels = cv2.connectedComponents(region_mask, connectivity=conn)  # 0..N
    labels_all = labels.astype(np.int32)  # スケルトンは 0 のまま

    # 3x3 の近傍最大を NumPy で計算する補助関数（int32対応）
    def max_filter3x3(arr: np.ndarray) -> np.ndarray:
        # 周囲 1 ピクセルを 0 でパディング
        p = np.pad(arr, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        # 9 方向のシフト配列を作成し、要素ごとの最大をとる
        # （p[1:-1,1:-1] が arr に対応）
        candidates = [
            p[0:-2, 0:-2], p[0:-2, 1:-1], p[0:-2, 2:  ],
            p[1:-1, 0:-2], p[1:-1, 1:-1], p[1:-1, 2:  ],
            p[2:  , 0:-2], p[2:  , 1:-1], p[2:  , 2:  ],
        ]
        out = candidates[0]
        for c in candidates[1:]:
            out = np.maximum(out, c)
        return out

    if max_iters is None:
        h, w = labels_all.shape
        max_iters = int(np.ceil(np.hypot(h, w))) + 5

    for _ in range(max_iters):
        zeros = (labels_all == 0)
        if not zeros.any():
            break
        dilated_like = max_filter3x3(labels_all)   # 近傍の最大ラベル
        labels_all[zeros] = dilated_like[zeros]

    # 念のための後処理（理論上残らないはずだが、残った 0 をもう一度埋める）
    if (labels_all == 0).any():
        dilated_like = max_filter3x3(labels_all)
        zeros = (labels_all == 0)
        labels_all[zeros] = dilated_like[zeros]

    if (labels_all == 0).any():
        raise RuntimeError("一部のスケルトン画素にラベルを割り当てられませんでした。入力画像の連結を確認してください。")

    return labels_all
