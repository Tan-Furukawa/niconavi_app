#%%

import numpy as np


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
            if r == 0:               # 1 ピクセル幅しか取れない場所は無視
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


# -----------------------------
# テストデータ（64 × 64, 半径 30 の円, 中心 (20, 20)）
# -----------------------------
h = w = 64
img = np.zeros((h, w), dtype=np.uint8)

cx, cy = 20, 20          # (行, 列) ではなく (x, y) = (col, row) のつもりで置く
y, x = np.ogrid[:h, :w]
mask = (x - cx) ** 2 + (y - cy) ** 2 <= 30 ** 2
img[mask] = 255          # 円内部を白 (255)

# スコア計算
score = rotational_symmetry_score(img)

# 例: スコアを表示して確認したい場合
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()


    plt.imshow(score, cmap="hot")
    plt.title("rotational_symmetry_score")
    plt.colorbar(label="score (lower = more symmetric)")
    plt.show()
