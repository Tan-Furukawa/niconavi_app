# %%
import cv2
import numpy as np
from skimage.feature import hessian_matrix
import matplotlib.pyplot as plt
import cv2
from typing import NewType, Any, cast, TypeAlias, Optional
from numpy.typing import NDArray
from cv2.typing import MatLike, Point, Scalar, Size
from niconavi.grain_segmentation.type import (
    BinaryImage,
    RGBImage,
    LoadedData,
    GrayScaleImage,
    FloatMatrix_HxWx2x2,
    FloatMatrix,
    FloatMatrix_0to1,
    IntMatrix,
    UintMatrix_3x3,
)


def imread(filename: str) -> GrayScaleImage:
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # return img[:,:,0]
    return GrayScaleImage(img.astype(np.uint8))


def image_correction(img: GrayScaleImage) -> GrayScaleImage:
    """濃淡差や影を補正する."""
    res = cv2.GaussianBlur(img, (7, 7), 0).astype(np.uint8)
    return GrayScaleImage(res)


def calc_hessian(img: GrayScaleImage, sigma: float) -> FloatMatrix_HxWx2x2:
    Hxx, Hxy, Hyy = hessian_matrix(
        img, sigma=sigma, mode="mirror", order="xy", use_gaussian_derivatives=False
    )
    hessian = np.stack([Hxx, Hxy, Hxy, Hyy], axis=2).reshape(*img.shape, 2, 2)
    return hessian


def hessian_emphasis(
    img: GrayScaleImage, sigma: float = 1.414, alpha: float = 0.25
) -> FloatMatrix:
    # 想定: calc_hessian(img, sigma) -> shape (H, W, 2, 2)
    H = calc_hessian(img, sigma)

    # 数値誤差で非対称になるのをケア
    H = FloatMatrix_HxWx2x2(0.5 * (H + np.swapaxes(H, -1, -2)))

    # eigh は小さい順に固有値を返す: (..., 0) が小、(..., 1) が大
    eigvals = np.linalg.eigh(H)[0]  # shape = (H, W, 2)
    lam2 = eigvals[..., 0]  # 小さい方（元コードの lambda2）
    lam1 = eigvals[..., 1]  # 大きい方（元コードの lambda1）

    # 分岐をマスクで一気に評価
    cond_granular = lam1 <= 0  # 粒状
    cond_linear = (
        (~cond_granular) & (lam2 < 0) & (lam1 > 0) & (lam1 < (np.abs(lam2) / alpha))
    )  # 線状

    r = np.zeros_like(img, dtype=np.float64)
    r[cond_granular] = np.abs(lam2[cond_granular]) + lam1[cond_granular]
    r[cond_linear] = np.abs(lam2[cond_linear]) - alpha * lam1[cond_linear]

    return FloatMatrix((sigma**2) * r)


def multiscale_hessian_emphasis(
    img: GrayScaleImage,
    sigma: float = 1.414,
    alpha: float = 0.25,
    s: float = 1.414,
    num_iteration: int = 4,
) -> tuple[FloatMatrix, IntMatrix]:

    raw = img.copy()

    R = []
    for i in range(num_iteration):
        # print(".")
        r = hessian_emphasis(img, sigma=sigma * s**i, alpha=alpha)
        R.append(r)

    _R = np.stack(R, axis=0)
    del R

    out = np.max(_R, axis=0)
    out_scale = np.argmax(_R, axis=0) + 1  # scaleを1から始める

    return out, out_scale


def stochastic_relaxation_method(img: FloatMatrix, alpha: float = 1.0) -> FloatMatrix:
    """確率的弛緩法.

    Parameters
    ----------
    img : ndarray
        画素値は0-1に規格化.
        ひびの方が背景よりも輝度が高い.
    alpha : float
        補正用の重み.

    Returns
    -------
    out
        確率的弛緩法を適用後の画像.
    residual
        確率的弛緩法適用前後の2乗誤差.
    """
    P_c = np.log1p(img) / np.log1p(img.max())
    P_b = 1 - P_c

    kernels = [
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) / 3,
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) / 3,
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) / 3,
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 3,
    ]

    Q_c = []
    for kernel in kernels:
        q = cv2.filter2D(P_c, -1, kernel, borderType=cv2.BORDER_DEFAULT)
        Q_c.append(q)
    _Q_c = np.stack(Q_c, axis=0)

    del Q_c

    Q_b = 1 - _Q_c

    P_c_new = np.divide(
        alpha * P_c[None, ...] * _Q_c,
        alpha * P_c[None, ...] * _Q_c + P_b[None, ...] * Q_b,
        where=P_c[None, ...] * _Q_c + P_b[None, ...] * Q_b != 0,
    )

    out = np.max(P_c_new, axis=0)
    return out


def stepwise_threshold_processing(
    img: GrayScaleImage,
    candidate: FloatMatrix_0to1,
    iter_stp: int = 3,
    k_size: int = 21,
    element: Optional[UintMatrix_3x3] = None,
) -> FloatMatrix:

    height, width = img.shape
    if element is None:
        element = UintMatrix_3x3(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8))

    completed = set()
    t = b = r = l = int((k_size - 1) * 0.5)
    for _ in range(iter_stp):
        flag_updated = False
        # 候補領域の外周の位置インデックスを得る
        dilated = cv2.dilate(candidate, element, iterations=1)
        outline = dilated - candidate
        ys, xs = (outline == 1).nonzero()

        result = candidate.copy()
        for x, y in zip(xs, ys):
            if (x, y) in completed:
                # 判定済み領域は除く
                continue
            # 注目画素近傍で大津の二値化のしきい値を求める
            xmin = max(x - l, 0)
            xmax = min(x + r + 1, width)
            ymin = max(y - t, 0)
            ymax = min(y + b + 1, height)
            patch = GrayScaleImage(img[ymin:ymax, xmin:xmax])
            patch_candidate = FloatMatrix_0to1(candidate[ymin:ymax, xmin:xmax])

            if crack_judge(img, (x, y), patch, patch_candidate):
                # しきい値以上であれば新たな候補領域にする
                result[y, x] = 1.0
                flag_updated = True
            else:
                # しきい値未満で背景に確定する
                completed.add((x, y))

        candidate = result.copy()
        if not flag_updated:
            break
    return FloatMatrix(result)


def trans2uint(img: FloatMatrix) -> GrayScaleImage:
    _img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype("uint8")
    return GrayScaleImage(_img)


def extract_crack_mean(
    patch: GrayScaleImage, patch_candidate: FloatMatrix_0to1
) -> tuple[float, float]:
    mu_c = np.mean(patch[patch_candidate == 1])
    mu_b = np.mean(patch[patch_candidate == 0])
    return float(mu_c), float(mu_b)


def crack_judge(
    img: GrayScaleImage,
    coordinate: tuple[int, int],
    patch: GrayScaleImage,
    patch_candidate: FloatMatrix_0to1,
    beta: float = 0.9,
) -> bool:
    x, y = coordinate
    mu_c, mu_b = extract_crack_mean(patch, patch_candidate)
    res = (abs(img[y, x] - mu_c) / abs(img[y, x] - mu_b + 1e-9) * beta) <= 1.0
    return bool(res)


def binarization(img: FloatMatrix) -> GrayScaleImage:
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype("uint8")
    _, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return GrayScaleImage(res.astype(np.uint8))


def crack_segmentation(
    img: GrayScaleImage,
    iter_mhe: int = 4,
    iter_srm: int = 10,
    iter_stp: int = 10,
    sigma: float = 1.414,
) -> tuple[GrayScaleImage, GrayScaleImage]:
    # ヘッシアンの固有値別に画像を補正
    raw = img.copy()

    fimg, _ = multiscale_hessian_emphasis(img, num_iteration=iter_mhe, sigma=sigma)
    emphasis_img = trans2uint(fimg.copy())

    # res = 1 / (emphasis_img.astype(np.float64) + 1.0) * (raw.astype(np.float64) + 1.0)
    # img = res.copy()
    # emphasis_img = res.copy()

    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()

    # 確率的弛緩法でノイズを消す
    kernel = np.ones((5, 5), np.float32)
    fimg = FloatMatrix(
        cv2.morphologyEx(fimg, cv2.MORPH_CLOSE, kernel).astype(np.float64)
    )
    for _ in range(iter_srm):
        fimg = stochastic_relaxation_method(fimg, 2.0)

    # 抽出候補を二値画像として得る
    candidate = FloatMatrix_0to1(binarization(fimg).astype(np.float64) / 255)

    # 消しすぎた候補領域を段階的閾値処理で拡張する
    mask = binarization(
        stepwise_threshold_processing(
            emphasis_img, candidate, iter_stp=iter_stp, k_size=27
        )
    )

    return mask, emphasis_img


if __name__ == "__main__":

    img = GrayScaleImage(imread("../data/8o.tif")[6000:7000, 4000:5000])

    # 画像の前処理
    img = image_correction(img)
    img = GrayScaleImage(cv2.bitwise_not(img).astype(np.uint8))

    # セグメンテーション
    mask, raw = crack_segmentation(
        img,
        iter_mhe=1,
    )

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # mg = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    plt.imshow(mask)
    plt.show()

    res = 1 / (raw.astype(np.float64) + 1.0) * (img.astype(np.float64) + 1.0)
    res = np.clip(res, 0, 255).astype(np.uint8)
    # mg2 = cv2.GaussianBlur(mg, (11, 11), 31)

    imgo = img.copy()
    img[mask != 255] = 0



    # %%
    plt.imshow(imgo, cmap="gray_r")
    plt.show()
    bi = (imgo > np.percentile(img[img != 0], 90)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mg = cv2.morphologyEx(bi, cv2.MORPH_CLOSE, kernel)

    plt.imshow(mg)
    plt.show()

    binary_image = bi
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    boudary_bool = mask == 255

    dark_out = np.zeros_like(binary_image, dtype=np.uint8)
    count = np.zeros_like(binary_image, dtype=np.uint8)
    res = np.zeros_like(boudary_bool)

    for lbl in range(1, num):  # 0は背景なのでスキップ
        # if lbl % 100 == 0:
        #     print(lbl / num)
        t = labels == lbl
        online = boudary_bool[t]
        if (np.sum(online) / len(online)) < 0.7:
            res[t] = True

    plt.imshow(res)
    # plt.imshow(img)
    # plt.show()
    # %%
    # plt.colorbar()
    # plt.show()
    # img = imread("data/5.tif")[3000:5000, 2000:5000]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mg = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    plt.imshow(mg)

# %%
