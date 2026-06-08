import numpy as np
import scipy.linalg as la


##############################################################################
# (A) 多項式特徴量生成
##############################################################################
def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    X の各行を (p次元) のベクトルとし、指定した次元 degree までの
    多項式特徴量を生成する。

    例: X.shape = (N, 3), degree=2 の場合、
        出力の形状は (N, 10) となり、[1, R, G, B, R^2, G^2, B^2, RG, RB, GB] を列に持つ。

    Parameters
    ----------
    X : np.ndarray
        入力データ。形状 (N, p)
    degree : int
        多項式の次数 (0, 1, 2, ...)

    Returns
    -------
    X_poly : np.ndarray
        多項式展開後の特徴量行列。形状 (N, n_features)
    """
    # 特徴量数を計算するためには、
    #  p=3, degree=2 なら [1, x1, x2, x3, x1^2, x2^2, x3^2, x1x2, x1x3, x2x3] の10個。
    # 一般的には下記のように生成する。
    N, p = X.shape

    # 0次元（degree=0）のときは定数1だけ返す
    if degree == 0:
        return np.ones((N, 1), dtype=X.dtype)

    # 総合的にすべての (i1, i2, ..., i_p) を考えるが、
    # ここでは itertools の combinations_with_replacement や product などを使わずに
    # シンプルに手続き的に実装してもよい。
    # ただし標準ライブラリは使用可なので itertools を使う実装例を示す。
    import itertools

    # 出力を格納するリスト
    features = []
    # まずは常に1(バイアス項)
    features.append(np.ones(N, dtype=X.dtype))

    # 次数1からdegreeまで、p次元との重複組み合わせをすべて生成
    # 例: p=3, degree=2 => (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) ... といった indices
    # ここでは (i, j, k,...) の X[:, i]*X[:, j]*X[:, k]*... が特徴量になる
    for d in range(1, degree + 1):
        # p次元から「重複あり」で d 個選ぶ組合せ
        # たとえば d=2 なら (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
        for comb in itertools.combinations_with_replacement(range(p), d):
            # comb に含まれる次元の入力をすべて掛け合わせる
            # 例: comb=(0,2) => X[:,0]*X[:,2] (R*B)
            feat = np.ones(N, dtype=X.dtype)
            for idx in comb:
                feat *= X[:, idx]
            features.append(feat)

    # リストを列方向に結合
    X_poly = np.stack(features, axis=1)  # shape: (N, n_features)
    return X_poly


##############################################################################
# (B) 通常の最小二乗解を求めるヘルパー関数
##############################################################################
def least_squares_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    最小二乗法で係数を求める: argmin_w ||Xw - y||^2
    X.shape = (N, M), y.shape = (N,)
    戻り値 w.shape = (M,)

    ここでは X^T X w = X^T y の正規方程式を用いて解くが、
    計算が不安定な場合は scipy.linalg.lstsq などを使うとよい。
    """
    # 直接 (X^T X)^{-1} X^T y を計算
    # 数値安定性を考慮するなら lstsq や pinv を使うのがベター
    # ここでは pinv を利用する例を示す：
    X_pinv = la.pinv(X)  # いわゆる擬似逆行列
    w = X_pinv @ y
    return w


##############################################################################
# (C) RANSAC の自前実装（単一出力用）
##############################################################################
def ransac_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_iterations: int = 200,
    residual_threshold: float = 5.0,
    min_inlier_ratio: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    RANSAC を用いて、外れ値にロバストな係数 w を推定する (単一出力 y を対象)。

    Parameters
    ----------
    X : np.ndarray
        入力特徴量 (N, M)
    y : np.ndarray
        ターゲット (N,)
    max_iterations : int
        RANSAC の試行回数
    residual_threshold : float
        inlier とみなす残差の閾値
    min_inlier_ratio : float
        モデルを更新するときに、inlier がこの割合未満だと採択しない
    random_state : int
        乱数シード

    Returns
    -------
    best_w : np.ndarray
        推定された最良モデルの係数 (M,)
    """
    rng = np.random.default_rng(random_state)
    N, M = X.shape

    best_w = None
    best_inlier_count = 0
    best_error_sum = 1e15  # inlier の誤差合計(あるいは平均)が最小のもの

    # 必要最小限のサンプル数は M (特徴量次元) とする（Xがフルランクであるために）
    subset_size = M

    for _ in range(max_iterations):
        # 1) ランダムに M 個の点をサンプリング
        subset_indices = rng.choice(N, size=subset_size, replace=False)
        X_subset = X[subset_indices, :]
        y_subset = y[subset_indices]

        # 2) subset で最小二乗解を求める
        w_candidate = least_squares_fit(X_subset, y_subset)

        # 3) 全データに対して残差を評価
        residuals = np.abs(X @ w_candidate - y)

        # 4) inlier を判定
        inlier_mask = residuals < residual_threshold
        inlier_count = np.sum(inlier_mask)

        # 5) 十分数の inlier が得られたら（しきい値を超えたら）、
        #    その inlier だけで再度フィットし直してモデルを更新
        if inlier_count >= min_inlier_ratio * N:
            X_in = X[inlier_mask, :]
            y_in = y[inlier_mask]
            w_refined = least_squares_fit(X_in, y_in)

            # inlier に対するエラー合計を計算
            residuals_in = np.abs(X_in @ w_refined - y_in)
            error_sum_in = residuals_in.sum()

            # inlier 数が多い・または誤差が小さいモデルを更新
            if inlier_count > best_inlier_count or (
                inlier_count == best_inlier_count and error_sum_in < best_error_sum
            ):
                best_w = w_refined
                best_inlier_count = inlier_count
                best_error_sum = error_sum_in

    # inlier がほとんど見つからなかった場合などで best_w が None の可能性あり
    if best_w is None:
        # fallback: 全データで最小二乗
        best_w = least_squares_fit(X, y)

    return best_w


##############################################################################
# (D) ロバスト回帰 (RANSAC + 多項式特徴量) を使って、RGB変換パラメータを学習
##############################################################################
def robust_regression(
    img1: np.ndarray,
    img2: np.ndarray,
    dim: int = 2,
    max_iterations: int = 200,
    residual_threshold: float = 5.0,
    min_inlier_ratio: float = 0.1,
    random_state: int = 42,
):
    """
    外れ値にロバストな多項式回帰 (RANSAC) により、
    img1 (R,G,B) -> img2 (R',G',B') の変換を学習する。

    Parameters
    ----------
    img1 : np.ndarray
        変換元の RGB画像 (h, w, 3)
    img2 : np.ndarray
        変換先の RGB画像 (h, w, 3)
    dim : int, default=2
        多項式の次数 (0,1,2,...)
    max_iterations : int
        RANSAC の最大試行回数
    residual_threshold : float
        RANSAC で inlier とみなす残差の閾値
    min_inlier_ratio : float
        RANSAC で inlier として採択する割合の閾値
    random_state : int
        RANSAC の乱数シード

    Returns
    -------
    Theta : np.ndarray
        shape = (3, n_features)
        R, G, B 各チャンネルの回帰係数をまとめたもの。
        例: Theta[0,:] が Rチャンネルを得るための係数ベクトル
    X_poly : np.ndarray
        学習時に作成した (N, n_features) の多項式特徴量行列 (メモリの都合上、不要なら返さなくても可)
    """
    # 画像を (N, 3) にフラット化
    h, w, c = img1.shape
    assert c == 3, "img1, img2 は RGB (3チャンネル) である必要があります。"
    X_rgb = img1.reshape(-1, 3).astype(np.float64)  # 入力 (R,G,B)
    Y_rgb = img2.reshape(-1, 3).astype(np.float64)  # 出力 (R',G',B')

    # 多項式特徴量を作成
    X_poly = polynomial_features(X_rgb, degree=dim)  # (N, n_features)

    # R, G, B 各チャンネルを独立に RANSAC で学習
    Theta_list = []
    for ch in range(3):
        y = Y_rgb[:, ch]
        w_best = ransac_fit(
            X_poly,
            y,
            max_iterations=max_iterations,
            residual_threshold=residual_threshold,
            min_inlier_ratio=min_inlier_ratio,
            random_state=random_state + ch,  # 多少シードをずらす
        )
        Theta_list.append(w_best)

    # (3, n_features) にまとめる
    Theta = np.vstack(Theta_list)
    return Theta, X_poly


##############################################################################
# (E) 学習済みパラメータ (Theta) で新しい画像を変換
##############################################################################
def apply_regression(Theta: np.ndarray, img: np.ndarray, dim: int) -> np.ndarray:
    """
    学習した回帰パラメータ Theta を使って、入力 img (RGB) の各画素を変換する。

    Parameters
    ----------
    Theta : np.ndarray
        形状 (3, n_features) の回帰係数
    img : np.ndarray
        変換する画像 (h, w, 3)
    dim : int
        多項式次元 (robust_regression() と同じもの)

    Returns
    -------
    img_est : np.ndarray
        変換後の推定画像 (h, w, 3), 値は一旦 float64 で計算後、0~255 にクリップして uint8 化
    """
    h, w, c = img.shape
    assert c == 3, "入力 img は RGB (3チャンネル) を想定します。"

    # 画像を (N, 3) にフラット化
    X_rgb = img.reshape(-1, 3).astype(np.float64)

    # 多項式特徴量に変換
    X_poly = polynomial_features(X_rgb, degree=dim)  # (N, n_features)

    # 予測: Y_est = X_poly @ Theta^T  => shape: (N, 3)
    # Theta.shape = (3, n_features) なので転置して行列積
    Y_est = X_poly @ Theta.T

    # 画像形状に戻す
    img_est = Y_est.reshape(h, w, c)

    # 画素値として扱うなら 0~255 でクリップして uint8 化
    img_est = np.clip(img_est, 0, 255).astype(np.uint8)
    return img_est


# 例: img1 から img2 への変換規則をロバスト回帰で学習し、
#     新たな画像 img_new を変換する。
