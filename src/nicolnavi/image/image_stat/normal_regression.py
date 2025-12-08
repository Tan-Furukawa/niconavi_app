from niconavi.image.image import create_outside_circle_mask
import numpy as np
from typing import Optional


# def compute_color_regression(
#     img1: RGBPicture, img2: RGBPicture, num_samples: Optional[int] = 300
# ) -> np.ndarray:
#     """
#     img1 の各画素 (R, G, B) から img2 の (R', G', B') を予測する線形変換（3x3 行列とバイアス項）
#     を最小二乗法により求めます。

#     モデルは各画素に対して
#         [R', G', B'] = [R, G, B, 1] @ Theta
#     （Theta の shape は (4, 3)）となります。

#     計算時間がかかりすぎる場合、num_samples でランダムに画素を抽出して推定を行います。

#     Parameters
#     ----------
#     img1 : np.ndarray
#         入力画像。形状は (H, W, 3)。
#     img2 : np.ndarray
#         目的画像。形状は (H, W, 3)。
#     num_samples : int, optional
#         最小二乗推定に用いる画素数の上限。None の場合は全画素を使用します。

#     Returns
#     -------
#     Theta : np.ndarray
#         係数行列。shape は (4, 3) で、上3行が線形変換 A (3×3)、最終行がバイアス項 b となる。
#     """
#     H, W, C = img1.shape
#     N = H * W  # 全画素数

#     # 各画素の RGB 情報を (N, 3) に変形
#     X = img1.reshape(-1, 3).astype(np.float64)
#     Y = img2.reshape(-1, 3).astype(np.float64)

#     # 計算コスト削減のため、必要に応じてランダムサンプリングを行う
#     if num_samples is not None and num_samples < N:
#         indices = np.random.choice(N, size=num_samples, replace=False)
#         X = X[indices]
#         Y = Y[indices]

#     # バイアス項を含めるため、各入力に 1 を付加して (N, 4) の入力行列を作成
#     ones = np.ones((X.shape[0], 1))
#     X_aug = np.hstack([X, ones])  # shape: (n_samples, 4)

#     # 最小二乗法により Theta (4, 3) を求める
#     Theta, residuals, rank, s = np.linalg.lstsq(X_aug, Y, rcond=None)

#     return Theta


def compute_color_regression(
    img1: np.ndarray, img2: np.ndarray, num_samples: Optional[int] = None, dim: int = 1
) -> np.ndarray:
    """
    img1 の各画素 (R, G, B) から img2 の (R', G', B') を予測する多項式回帰モデルの
    係数 Theta を最小二乗法で求めます。

    モデルは以下のように表されます。

      (R', G', B') = feature_vector @ Theta

    ここで feature_vector は、img1 の各画素の (R, G, B) に対して、
    - dim=0 の場合は [1]
    - dim=1 の場合は [1, R, G, B]
    - dim=2 の場合は [1, R, G, B, R^2, R*G, R*B, G^2, G*B, B^2]

    のように多項式展開されたものです。

    Parameters
    ----------
    img1 : np.ndarray
        入力画像。形状は (H, W, 3) で、各画素の値は (R, G, B) となる。
    img2 : np.ndarray
        目的画像。形状は (H, W, 3) で、各画素の値は (R', G', B') となる。
    num_samples : int, optional
        最小二乗推定に用いる画素数。None の場合は全画素を使用する。
    dim : int
        多項式の次数。0, 1, 2 のいずれか。 (例: 1 なら線形回帰)

    Returns
    -------
    Theta : np.ndarray
        回帰係数。shape は (num_features, 3) となる。num_features は dim に応じて
        0次なら1, 1次なら4, 2次なら10 となる。
    """
    H, W, C = img1.shape
    # plt.imshow(img1)
    # plt.show()
    # N = H * W
    img1_mask = create_outside_circle_mask(img1)
    img2_mask = create_outside_circle_mask(img2)

    # 画像を (N, 3) に変換（float64 型にキャスト）
    X = img1[~img1_mask].astype(np.float64)
    Y = img2[~img2_mask].astype(np.float64)
    N = len(X)

    # plt.imshow(Y.reshape(-1, 239))
    # plt.show()
    # print(img2_mask.shape)

    # サンプル数が指定されている場合、ランダムサンプリングを実施
    if num_samples is not None and num_samples < N:
        indices = np.random.choice(N, size=num_samples, replace=False)
        X = X[indices]
        Y = Y[indices]

    # 多項式特徴量を作成
    X_poly = compute_poly_features(X, dim)

    # print(X_poly.shape)

    # 最小二乗法により Theta を求める
    Theta, residuals, rank, s = np.linalg.lstsq(X_poly, Y, rcond=None)

    return Theta


def compute_poly_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    入力 X (shape: (N, 3)) に対して、0次～3次の多項式特徴量を生成します。

    Parameters
    ----------
    X : np.ndarray
        入力配列。各行が (R, G, B) の3成分からなる。
    degree : int
        多項式の次数。0, 1, 2, 3 のいずれか。

    Returns
    -------
    X_poly : np.ndarray
        多項式展開後の設計行列。
          - degree=0: shape = (N, 1)
          - degree=1: shape = (N, 4)    [1, R, G, B]
          - degree=2: shape = (N, 10)   [1, R, G, B, R², R·G, R·B, G², G·B, B²]
          - degree=3: shape = (N, 20)   [1, R, G, B, R², R·G, R·B, G², G·B, B²,
                                       R³, R²·G, R²·B, R·G², R·G·B, R·B², G³, G²·B, G·B², B³]
    """
    N = X.shape[0]
    if degree == 0:
        return np.ones((N, 1))
    elif degree == 1:
        ones = np.ones((N, 1))
        return np.hstack([ones, X])
    elif degree == 2:
        ones = np.ones((N, 1))
        R = X[:, 0:1]
        G = X[:, 1:2]
        B = X[:, 2:3]
        R2 = R**2
        RG = R * G
        RB = R * B
        G2 = G**2
        GB = G * B
        B2 = B**2
        return np.hstack([R, G, B, R2, RG, RB, G2, GB, B2, ones])
    elif degree == 3:
        ones = np.ones((N, 1))

        R = X[:, 0:1]
        G = X[:, 1:2]
        B = X[:, 2:3]
        # 1次項は R, G, B  (既にX)
        # 2次項
        R2 = R**2
        RG = R * G
        RB = R * B
        G2 = G**2
        GB = G * B
        B2 = B**2
        # 3次項
        R3 = R**3
        R2G = (R**2) * G
        R2B = (R**2) * B
        RG2 = R * (G**2)
        RGB = R * G * B
        RB2 = R * (B**2)
        G3 = G**3
        G2B = (G**2) * B
        GB2 = G * (B**2)
        B3 = B**3

        return np.hstack(
            [
                R,
                G,
                B,
                R2,
                RG,
                RB,
                G2,
                GB,
                B2,
                R3,
                R2G,
                R2B,
                RG2,
                RGB,
                RB2,
                G3,
                G2B,
                GB2,
                B3,
                ones,
            ]
        )
    else:
        raise ValueError("degree must be 0, 1, 2, or 3")

    # %%


# def correct_image(img: np.ndarray, Theta: np.ndarray, dim: int = 1) -> np.ndarray:
#     """
#     得られた係数行列 Theta と入力画像 img を用いて色補正を行います。
#     各画素に対して、(R, G, B) から多項式特徴量（0次～dim次）を生成し、
#     その特徴量ベクトルと Theta の行列積により補正後の色 (R', G', B') を計算します。

#     Parameters
#     ----------
#     img : np.ndarray
#         補正対象の画像。形状は (H, W, 3)。
#     Theta : np.ndarray
#         補正用の係数行列。shape は (num_features, 3) で、
#         num_features は dim に応じて 0次なら 1、1次なら 4、2次なら 10、3次なら 20 となる。
#     dim : int, optional
#         多項式の次数。0, 1, 2, 3 のいずれか。デフォルトは 1（線形補正）。

#     Returns
#     -------
#     corrected_img : np.ndarray
#         補正後の画像。形状は (H, W, 3) で、各画素は [0, 255] にクリップされ uint8 型です。
#     """
#     # 入力画像を float 型に変換
#     img_float = img.astype(np.float64)
#     H, W, _ = img_float.shape
#     N = H * W

#     # 画像を (N, 3) に変形
#     X = img_float.reshape(-1, 3)

#     # 指定された次数に基づく多項式特徴量を作成
#     X_poly = compute_poly_features(X, dim)

#     # 特徴量行列と Theta の積により補正を適用
#     corrected_flat = X_poly @ Theta  # (N, num_features) @ (num_features, 3) → (N, 3)

#     # (N, 3) を (H, W, 3) に戻し、値を [0, 255] にクリップして uint8 に変換
#     corrected_img = np.clip(corrected_flat.reshape(H, W, 3), 0, 255).astype(np.uint8)

#     return corrected_img


# def compute_poly_features(X: np.ndarray, degree: int) -> np.ndarray:
#     """
#     入力 X (shape: (N, 3)) に対して、0次～2次の多項式特徴量を生成する関数。

#     Parameters
#     ----------
#     X : np.ndarray
#         入力配列。各行が (R, G, B) の3成分からなる。
#     degree : int
#         多項式の次数。0, 1, 2 のいずれか。

#     Returns
#     -------
#     X_poly : np.ndarray
#         多項式展開後の設計行列。degree に応じた列数となる。
#           - degree=0: shape = (N, 1)
#           - degree=1: shape = (N, 4)  [1, R, G, B]
#           - degree=2: shape = (N, 10) [1, R, G, B, R^2, R*G, R*B, G^2, G*B, B^2]
#     """

#     # ones = np.ones((X.shape[0], 1))
#     # X_aug = np.hstack([X, ones])  # shape: (n_samples, 4)

#     # # 最小二乗法により Theta (4, 3) を求める
#     # Theta, residuals, rank, s = np.linalg.lstsq(X_aug, Y, rcond=None)

#     if degree == 0:
#         return np.ones((X.shape[0], 1))
#     elif degree == 1:
#         # 1次：定数項と各変数
#         ones = np.ones((X.shape[0], 1))
#         return np.hstack([X, ones])
#     elif degree == 2:
#         # 2次：定数項、線形項、2次項（自乗および交差項）
#         ones = np.ones((X.shape[0], 1))
#         R = X[:, 0:1]
#         G = X[:, 1:2]
#         B = X[:, 2:3]
#         # 線形項は R, G, B（すでに X に含まれる）
#         # 2次項
#         R2 = R**2
#         G2 = G**2
#         B2 = B**2
#         RG = R * G
#         RB = R * B
#         GB = G * B
#         return np.hstack([R, G, B, R2, RG, RB, G2, GB, B2, ones])
#     else:
#         raise ValueError("dim (degree) must be 0, 1, or 2")


# def compute_color_regression(
#     img1: np.ndarray, img2: np.ndarray, num_samples: int = None, dim: int = 1
# ) -> np.ndarray:
#     """
#     img1 の各画素 (R, G, B) から img2 の (R', G', B') を予測する多項式回帰モデルの
#     係数 Theta を最小二乗法で求めます。

#     モデルは以下のように表されます。

#       (R', G', B') = feature_vector @ Theta

#     ここで feature_vector は、img1 の各画素の (R, G, B) に対して、
#     - dim=0 の場合は [1]
#     - dim=1 の場合は [1, R, G, B]
#     - dim=2 の場合は [1, R, G, B, R^2, R*G, R*B, G^2, G*B, B^2]

#     のように多項式展開されたものです。

#     Parameters
#     ----------
#     img1 : np.ndarray
#         入力画像。形状は (H, W, 3) で、各画素の値は (R, G, B) となる。
#     img2 : np.ndarray
#         目的画像。形状は (H, W, 3) で、各画素の値は (R', G', B') となる。
#     num_samples : int, optional
#         最小二乗推定に用いる画素数。None の場合は全画素を使用する。
#     dim : int
#         多項式の次数。0, 1, 2 のいずれか。 (例: 1 なら線形回帰)

#     Returns
#     -------
#     Theta : np.ndarray
#         回帰係数。shape は (num_features, 3) となる。num_features は dim に応じて
#         0次なら1, 1次なら4, 2次なら10 となる。
#     """
#     H, W, C = img1.shape
#     N = H * W

#     # 画像を (N, 3) に変換（float64 型にキャスト）
#     X = img1.reshape(-1, 3).astype(np.float64)
#     Y = img2.reshape(-1, 3).astype(np.float64)

#     # サンプル数が指定されている場合、ランダムサンプリングを実施
#     if num_samples is not None and num_samples < N:
#         indices = np.random.choice(N, size=num_samples, replace=False)
#         X = X[indices]
#         Y = Y[indices]

#     # 多項式特徴量を作成
#     X_poly = compute_poly_features(X, dim)

#     # 最小二乗法により Theta を求める
#     Theta, residuals, rank, s = np.linalg.lstsq(X_poly, Y, rcond=None)

#     return Theta


def correct_image(img: np.ndarray, Theta: np.ndarray, dim: int = 1) -> np.ndarray:
    """
    得られた係数行列 Theta と入力画像 img を用いて、img の色補正を行います。

    各画素に対して、まず (R, G, B) から多項式特徴量（dim 次まで）を作成し、
    その特徴量ベクトルと Theta をかけることで補正後の (R', G', B') を計算します。

    Parameters
    ----------
    img : np.ndarray
        補正対象の画像。形状は (H, W, 3)。
    Theta : np.ndarray
        補正用の係数行列。shape は (num_features, 3) で、num_features は dim に応じて
        0次なら1、1次なら4、2次なら10 となる。
    dim : int, optional
        多項式の次数。0, 1, 2 のいずれか。デフォルトは 1（線形補正）。

    Returns
    -------
    corrected_img : np.ndarray
        補正後の画像。形状は (H, W, 3) で、各画素は [0, 255] にクリップされ uint8 型です。
    """
    # 入力画像を浮動小数点型に変換
    img_float = img.astype(np.float64)
    H, W, C = img_float.shape
    N = H * W

    # (H, W, 3) の画像を (N, 3) に変形
    X = img_float.reshape(-1, 3)

    # 多項式特徴量を作成（dim の値に応じて特徴量が変わる）
    X_poly = compute_poly_features(X, dim)

    # 補正を適用: (N, num_features) @ (num_features, 3) → (N, 3)
    corrected_flat = X_poly @ Theta

    # (N, 3) を (H, W, 3) に戻す
    corrected = corrected_flat.reshape(H, W, 3)

    # 画素値を [0, 255] にクリップし、uint8 に変換
    corrected_img = np.clip(corrected, 0, 255).astype(np.uint8)

    return corrected_img


# def correct_image(img: np.ndarray, Theta: np.ndarray) -> np.ndarray:
#     """
#     得られた係数行列 Theta と入力画像 img を用いて、img の色補正を行います。

#     各画素について、補正は以下の式で行われます:
#         [R', G', B'] = [R, G, B, 1] @ Theta
#     ここで、Theta は shape (4, 3) の行列で、上3行が線形変換、最終行がバイアス項です。

#     Parameters
#     ----------
#     img : np.ndarray
#         補正対象の画像。形状は (H, W, 3)。
#     Theta : np.ndarray
#         補正用の係数行列。形状は (4, 3)。

#     Returns
#     -------
#     corrected_img : np.ndarray
#         補正後の画像。形状は (H, W, 3) で、各画素は [0, 255] にクリップされ uint8 型です。
#     """
#     # 入力画像を浮動小数点型に変換
#     img_float = img.astype(np.float64)
#     H, W, C = img_float.shape
#     N = H * W

#     # (H, W, 3) の画像を (N, 3) に変形
#     X = img_float.reshape(-1, 3)

#     # バイアス項を付加するため、各行に定数項 1 を追加 (N, 1)
#     ones = np.ones((N, 1), dtype=np.float64)
#     X_aug = np.hstack([X, ones])  # shape: (N, 4)

#     # 補正を適用 (N, 4) @ (4, 3) -> (N, 3)
#     corrected = X_aug @ Theta

#     # 結果を元の画像サイズ (H, W, 3) に戻す
#     corrected = corrected.reshape(H, W, 3)

#     # 画素値を [0, 255] にクリップし、uint8 に変換
#     corrected_img = np.clip(corrected, 0, 255).astype(np.uint8)

#     return corrected_img
