# %%
import numpy as np
from typing import Dict, Any, Tuple, List, Callable, cast
from numpy.typing import NDArray
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from niconavi.tools.type import D1FloatArray, D1IntArray, D2BoolArray
from niconavi.image.type import D1RGB_Array
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from typing import NewType, Tuple
import numpy as np
import scipy.ndimage as nd


def gamma_mode_from_data(data: D1FloatArray) -> float:
    """
    入力データをガンマ分布とみなして、その最頻度値（mode）を推定する関数

    ガンマ分布のパラメータは、方法-of-moments により以下のように推定します:
      - 平均: μ = α・θ
      - 分散: σ² = α・θ²
    よって、α = μ² / σ², θ = σ² / μ と推定できます。
    なお、α > 1 の場合は mode = (α - 1) * θ, α <= 1 の場合は mode = 0 とします。

    Parameters:
        data (list または numpy.array): 一次元の数値データ

    Returns:
        float: 推定されたガンマ分布の最頻度値。データが空の場合は None を返します。
    """
    data = np.array(data)
    if data.size == 0:
        return None

    # サンプル平均と不偏分散（ddof=1）を計算
    mean = data.mean()
    variance = data.var(ddof=1)

    # 分散が 0 の場合は、全て同一の値なのでその値を返す
    if variance == 0:
        return data[0]

    # 方法-of-moments によるパラメータ推定
    alpha = mean**2 / variance
    theta = variance / mean

    # ガンマ分布の最頻度値の計算
    if alpha > 1:
        mode = (alpha - 1) * theta
    else:
        mode = 0
    return mode


# 使用例
if __name__ == "__main__":
    sample_data = [1.2, 2.3, 3.1, 2.2, 3.8, 2.9, 3.0]
    estimated_mode = gamma_mode_from_data(sample_data)
    print("入力データ:", sample_data)
    print("推定されたガンマ分布の最頻度値:", estimated_mode)


def find_rightmost_kde_peak(data: D1FloatArray) -> float:
    """
    与えられた一次元実数配列に対してカーネル密度推定を行い、
    KDEのピークの中で最も右側（xが最大）のピークのxの値を返します。

    Parameters:
    ----------
    data : np.ndarray
        入力の一次元実数配列。

    Returns:
    -------
    float
        KDEの最も右側のピーク位置のxの値。

    Raises:
    -------
    TypeError:
        入力がnumpy.ndarrayでない場合。
    ValueError:
        入力配列が一次元でない、または空の場合。
    RuntimeError:
        ピークが検出できなかった場合。
    """
    # 入力の検証
    if not isinstance(data, np.ndarray):
        raise TypeError("入力はnumpy.ndarrayでなければなりません。")
    if data.ndim != 1:
        raise ValueError("入力配列は一次元でなければなりません。")
    if len(data) == 0:
        raise ValueError("入力配列は空であってはなりません。")

    # カーネル密度推定を作成
    kde = gaussian_kde(data)

    # データの範囲を拡張して評価範囲を設定
    data_min, data_max = data.min(), data.max()
    padding = (data_max - data_min) * 0.1  # 10%のパディング
    x_eval = np.linspace(data_min - padding, data_max + padding, 1000)

    # KDEを評価
    kde_values = kde(x_eval)

    # plt.plot(x_eval, kde_values)
    # ピークの検出
    peaks, _ = find_peaks(kde_values)

    if len(peaks) == 0:
        raise RuntimeError("ピークが検出できませんでした。データを確認してください。")

    # ピークのx値を取得
    peak_x_values = x_eval[peaks]

    # 最も右側のピークを選択
    rightmost_peak_x = peak_x_values[
        -1
    ]  # x_evalは昇順にソートされているため最後のピークが最も右側

    return float(rightmost_peak_x)


def find_kde_peak(data: D1FloatArray) -> float:
    """
    与えられた一次元実数配列に対してカーネル密度推定を行い、
    最も高いピークのときのxの値を返します。

    Parameters:
    ----------
    data : np.ndarray
        入力の一次元実数配列。

    Returns:
    -------
    float
        KDEのピーク位置のxの値。
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("入力はnumpy.ndarrayでなければなりません。")
    if data.ndim != 1:
        raise ValueError("入力配列は一次元でなければなりません。")
    if len(data) == 0:
        raise ValueError("入力配列は空であってはなりません。")

    # カーネル密度推定を作成
    kde = gaussian_kde(data)

    # データの範囲を拡張して評価範囲を設定
    data_min, data_max = data.min(), data.max()
    padding = (data_max - data_min) * 0.1  # 10%のパディング
    x_eval = np.linspace(data_min - padding, data_max + padding, 1000)

    # KDEを評価
    kde_values = kde(x_eval)

    # 最大密度のインデックスを取得
    peak_index = np.argmax(kde_values)

    # ピーク位置のxを返す
    peak_x = x_eval[peak_index]

    return float(peak_x)


def multiple_linear_regression(X: NDArray, Y: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Perform multiple linear regression to find A and B in Y = AX + B.

    Parameters:
    X (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).
    Y (numpy.ndarray): Output target matrix of shape (n_samples, n_outputs).

    Returns:
    A (numpy.ndarray): Coefficient matrix of shape (n_features, n_outputs).
    B (numpy.ndarray): Intercept vector of shape (n_outputs,).
    """
    # Ensure X and Y have the correct dimensions
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-dimensional arrays.")

    n_samples_X, n_features = X.shape
    n_samples_Y, n_outputs = Y.shape

    if n_samples_X != n_samples_Y:
        raise ValueError("The number of samples in X and Y must be equal.")

    # Add a column of ones to X to account for the intercept B
    X_augmented = np.hstack([X, np.zeros((n_samples_X, 1))])

    # Compute the least squares solution to find W = [A; B]
    W, residuals, rank, s = np.linalg.lstsq(X_augmented, Y, rcond=None)

    # Extract A and B from W
    A = W[:-1, :]  # Coefficient matrix A
    B = W[-1, :]  # Intercept vector B

    return A, B


def generate_polynomial_features(
    X: NDArray[np.float64], degree: int
) -> NDArray[np.float64]:
    N, D = X.shape
    exponents = []

    def generate_exponents(
        current_exp: list, remaining_degree: int, index: int
    ) -> None:
        if index == D:
            if remaining_degree >= 0:
                exponents.append(current_exp.copy())
            return
        for i in range(remaining_degree + 1):
            current_exp.append(i)
            generate_exponents(current_exp, remaining_degree - i, index + 1)
            current_exp.pop()

    generate_exponents([], degree, 0)

    Phi = np.ones((N, len(exponents)))
    for i, exp in enumerate(exponents):
        for d in range(D):
            Phi[:, i] *= X[:, d] ** exp[d]

    return Phi


# def multiple_polynomial_regression(
#     X: NDArray[np.float64], Y: NDArray[np.float64], degree: int
# ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
#     # モデルの構築と学習
#     Phi = generate_polynomial_features(X, degree)
#     B = np.linalg.lstsq(Phi, Y, rcond=None)[0]

#     # 予測関数
#     def predict(X_new: NDArray[np.float64]) -> NDArray[np.float64]:
#         Phi_new = generate_polynomial_features(X_new, degree)
#         Y_pred = Phi_new @ B
#         return Y_pred

#     return predict


def multiple_polynomial_regression(
    X: NDArray[np.float64], Y: NDArray[np.float64], degree: int
) -> Tuple[
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
]:
    """
    Perform multiple polynomial regression and return both predict and inverse_predict functions.

    Parameters:
    X (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).
    Y (numpy.ndarray): Output target matrix of shape (n_samples, n_outputs).
    degree (int): The degree of the polynomial features.

    Returns:
    predict (Callable): Function to predict Y from X_new.
    inverse_predict (Callable): Function to estimate X from Y_new.
    """
    # モデルの構築と学習
    Phi = generate_polynomial_features(X, degree)
    B = np.linalg.lstsq(Phi, Y, rcond=None)[0]

    # 予測関数
    def predict(X_new: NDArray[np.float64]) -> NDArray[np.float64]:
        Phi_new = generate_polynomial_features(X_new, degree)
        Y_pred = Phi_new @ B
        return Y_pred

    # 逆予測関数
    def inverse_predict(Y_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Estimate X from Y using optimization.

        Parameters:
        Y_new (numpy.ndarray): Target output matrix of shape (n_outputs,).

        Returns:
        X_estimated (numpy.ndarray): Estimated input features of shape (n_features,).
        """
        n_outputs = Y_new.shape[-1]
        n_features = X.shape[1]

        def objective(X_candidate: NDArray, Y_target: NDArray) -> float:
            Phi_candidate = generate_polynomial_features(
                X_candidate.reshape(1, -1), degree
            )
            Y_pred = Phi_candidate @ B
            return np.sum((Y_pred - Y_target) ** 2)

        X_estimated = []
        for y in Y_new:
            res = minimize(objective, x0=np.zeros(n_features), args=(y,), method="BFGS")
            if res.success:
                X_estimated.append(res.x)
            else:
                raise ValueError("Inverse prediction optimization failed.")

        return np.array(X_estimated)

    return predict, inverse_predict


def fixed_dim3_regression(
    x: D1FloatArray, y: D1FloatArray, index_start: int, index_end: int
) -> Callable[[float], float]:
    """
    index_start と index_end の (x, y) を厳密に通るように固定したうえで、
    三次多項式 p(x) = a x^3 + b x^2 + c x + d が区間 [index_start, index_end] 内の
    他の点を最小二乗法で近似するようにフィットする。

    Parameters
    ----------
    x : np.ndarray
        x のデータ (1次元)
    y : np.ndarray
        y のデータ (1次元)
    index_start : int
        フィット区間の開始インデックス
    index_end : int
        フィット区間の終了インデックス

    Returns
    -------
    Callable[[float], float]
        フィットされた三次多項式を入力 x に対して評価する関数
    """

    # フィット区間の端点 (固定したい点)
    x_s, y_s = x[index_start], y[index_start]
    x_e, y_e = x[index_end], y[index_end]

    # フィットに使用する区間内データ (端点は除いて最小二乗にかける)
    if index_end - index_start > 1:
        x_sub = x[index_start + 1 : index_end]
        y_sub = y[index_start + 1 : index_end]
    else:
        # 区間内にデータ点がない場合
        x_sub = np.array([], dtype=float)
        y_sub = np.array([], dtype=float)

    # --------------------------------
    #  c, d を a, b から導出する関数
    # --------------------------------
    def get_cd(a: float, b, x_s, y_s, x_e, y_e):
        #  p(x_s) = y_s,  p(x_e) = y_e を満たすように c, d を解く
        denom = x_s - x_e
        # 例外: x_s == x_e のケースは通常の問題設定外なので割愛
        c = ((y_s - y_e) - a * (x_s**3 - x_e**3) - b * (x_s**2 - x_e**2)) / denom
        d = y_s - (a * x_s**3 + b * x_s**2 + c * x_s)
        return c, d

    # --------------------------------
    #  残差を返す関数 (least_squares 用)
    # --------------------------------
    def residual(a_b, x_data, y_data):
        a, b = a_b
        c, d = get_cd(a, b, x_s, y_s, x_e, y_e)
        return a * x_data**3 + b * x_data**2 + c * x_data + d - y_data

    # --------------------------------
    #  最適化実行
    # --------------------------------
    if len(x_sub) > 0:
        res = least_squares(
            fun=residual,
            x0=[0.0, 0.0],  # (a, b)の初期値
            args=(x_sub, y_sub),
        )
        a_opt, b_opt = res.x
    else:
        # 区間に点が無い場合, a,b=0 として端点条件で c,d を決める
        a_opt, b_opt = 0.0, 0.0

    # (a_opt, b_opt) が決まったので c, d を復元
    c_opt, d_opt = get_cd(a_opt, b_opt, x_s, y_s, x_e, y_e)

    # --------------------------------
    #  三次関数 p(x) を返す
    # --------------------------------
    def fitted_function(xx: float) -> float:
        return a_opt * xx**3 + b_opt * xx**2 + c_opt * xx + d_opt

    return fitted_function


def regression_at_fixed_point(
    x: D1FloatArray,
    y: D1FloatArray,
    break_points: D1IntArray,
) -> Callable[[float], float]:
    """
    x, y のデータ列を指定し、break_points で指定される区間ごとに
    fixed_3_dim_regression を適用して得られる「区分的三次関数」を返す。

    たとえば break_points = [i0, i1, i2, ...] が与えられた場合、
      - [i0, i1], [i1, i2], [i2, i3], ...
    の各区間において端点を固定した三次関数をフィットし、
    その区間内 ( x[i_k] <= X < x[i_{k+1}] ) に対しては当該三次関数を使うような
    関数 p(X) を構築する。

    Parameters
    ----------
    x : np.ndarray
        x のデータ (1次元)
    y : np.ndarray
        y のデータ (1次元)
    break_points : np.ndarray
        区間区切りとなるインデックス群 (整数の 1 次元配列)
        要素数が N+1 個なら N 区間として扱う

    Returns
    -------
    Callable[[float], float]
        区間ごとにフィットした三次多項式からなる「区分三次関数」を返す関数
    """

    # まず、各区間 [break_points[i], break_points[i+1]] に対して
    # 三次多項式をフィットし、その「区間の x の最小値、最大値、三次関数」を
    # リストにまとめておく。
    segments: List[Tuple[float, float, Callable[[float], float]]] = []

    for i in range(len(break_points) - 1):
        idx_start = break_points[i]
        idx_end = break_points[i + 1]
        # 三次関数をフィット
        cubic_func = fixed_dim3_regression(x, y, idx_start, idx_end)
        # 区間の x の実際の範囲 (実数)
        x_min = x[idx_start]
        x_max = x[idx_end]
        if x_min <= x_max:
            segments.append((x_min, x_max, cubic_func))
        else:
            # 仮に x[idx_start] > x[idx_end] という場合が来たらエラーでもよい
            # ここでは x_min, x_max を入れ替えて格納する例
            segments.append((x_max, x_min, cubic_func))

    def piecewise_cubic(X: float) -> float:
        """
        与えられた X に対して、どの区間に属するかを調べ、
        適切な三次関数で値を計算する区分三次関数。
        """
        # 区間が無い場合は 0 を返すかエラーにするかはお好みで
        if not segments:
            return 0.0

        # もし X が segments[0] の左端より小さい場合は、その先頭区間の外挿とする
        x_min_first, x_max_first, f_first = segments[0]
        if X < x_min_first:
            return f_first(X)

        # 最後の区間より大きい場合は、最後の区間の外挿とする
        x_min_last, x_max_last, f_last = segments[-1]
        if X > x_max_last:
            return f_last(X)

        # 中間の区間に属するかチェック
        for x_min_seg, x_max_seg, f_seg in segments:
            if x_min_seg <= X <= x_max_seg:
                return f_seg(X)

        # 万一すべての条件を外れた場合は、最後の区間を返すなど
        return f_last(X)

    return piecewise_cubic


def fitted_by_bspline(y: np.ndarray, k: int = 3, s: float = 0.0) -> np.ndarray:
    """
    Bスプラインを用いて1次元データをスムーズにフィッティングする関数

    Parameters
    ----------
    y : np.ndarray
        フィット対象の1次元配列データ
    k : int, optional
        スプライン曲線の次数(デフォルト=3で三次スプライン)
    s : float, optional
        スムージング係数。大きくするほどフィットが滑らかになる（デフォルト=0は補間寄り）

    Returns
    -------
    y_fitted : np.ndarray
        Bスプラインによるフィット結果(入力 y と同じ長さ)
    """
    # x軸をデータのインデックスとする
    x = np.arange(len(y))

    # Bスプラインのパラメータを求める
    # tckは (t:ノットベクトル, c:スプライン係数, k:次数) からなるタプル
    tck = splrep(x, y, k=k, s=s)

    # 得られたスプライン関数を使ってフィット値を生成
    y_fitted = splev(x, tck)
    return y_fitted


def compute_centroid_in_2D_area(
    array: D2BoolArray,
) -> tuple[float, float]:
    # if not issubclass(array.dtype.type, np.integer):
    #     raise ValueError("入力配列は整数型でなければなりません。")

    # 1の要素のインデックスを取得
    positions = np.argwhere(array)

    if positions.size == 0:
        raise ValueError("配列に1の要素が含まれていません。")

    # 重心を計算
    centroid_y = positions[:, 0].mean()
    centroid_x = positions[:, 1].mean()
    centroid = (centroid_y, centroid_x)

    return centroid


def get_inscribed_circle_center(
    grain_shape: D2BoolArray,
) -> tuple[int, tuple[int, int]]:
    """
    与えられた2次元のbool型配列（grain_shape）に対して、内接円の中心とその半径を求める関数。

    パラメータ:
        grain_shape: D2BoolArray (N, M)
            Trueが図形内部、Falseが図形外部を示す2次元のブール配列。

    戻り値:
        tuple[int, tuple[int, int]]
            内接円の半径（整数）と、その中心のインデックス (row, column) のタプル。
    """
    # grain_shape内のTrue領域から外部（False）までの距離を計算
    distance_map = nd.distance_transform_edt(grain_shape)

    # 最大の距離（内接円の半径に対応）を持つ画素のインデックスを取得
    center_idx = np.unravel_index(np.argmax(distance_map), distance_map.shape)

    # 内接円の半径（整数値に変換）
    radius = int(distance_map[center_idx])

    return radius, center_idx


if __name__ == "__main__":

    # 例2: ランダムに生成した角度データ
    np.random.seed(2)
    data_rand = np.random.uniform(0, 90, size=100)
    med_rand = circular_median(0, 90, 3600)(data_rand)
    print(f"Random data: {data_rand}")
    print(f"Median for random data: {med_rand:.3f}")
    # %%


# ----- 簡単な動作確認 -----

# 使い方例
if __name__ == "__main__":
    # サンプルデータ (正弦波 + ノイズ)
    x_data = np.linspace(0, 2 * np.pi, 50)
    y_data = np.sin(x_data) + 0.1 * np.random.randn(*x_data.shape)

    # Bスプラインフィット
    y_fit = fitted_by_bspline(y_data, k=3, s=1.0)

    # 結果の表示例
    import matplotlib.pyplot as plt

    plt.plot(x_data, y_data, "o", label="Original data")
    plt.plot(x_data, y_fit, "-", label="B-Spline fitted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # --- 動作確認用サンプル ---

    import matplotlib.pyplot as plt

    # ダミーデータ (ノイズ入り)
    np.random.seed(0)
    xx = np.linspace(0, 10, 50)

    def true_func(t):
        return t**3 - 2 * t**2 + 3 * t + 1

    yy = true_func(xx) + np.random.normal(0, 50, size=xx.size)

    # break_points を設定 (インデックス)
    # たとえば [0, 10, 20, 30, 40, 49] のように複数区間に分ける
    bp = np.array([0, 10, 20, 30, 40, 49])

    # 区間ごとに三次関数フィット -> 区分三次関数としてまとめる
    piecewise_func = regression_at_fixed_point(xx, yy, bp)

    # 可視化用に piecewise_func を評価
    xx_fine = np.linspace(0, 10, 200)
    yy_piecewise = [piecewise_func(xv) for xv in xx_fine]

    # 描画
    plt.figure()
    plt.scatter(xx, yy, color="gray", label="data")
    plt.plot(xx_fine, yy_piecewise, color="red", label="piecewise cubic")

    # 各 break_points の「x 軸上の位置」を線で示す
    for b in bp:
        plt.axvline(xx[b], color="blue", linestyle="--", alpha=0.3)

    plt.legend()
    plt.show()

    # サンプル実行

    # 適当なテストデータを用意 (例: 元の関数が y = x^3 - 2x^2 + 3x + 1 + ノイズ)

    np.random.seed(0)
    xx = np.linspace(0, 10, 50)
    true_func = lambda t: t**3 - 2 * t**2 + 3 * t + 1
    yy = true_func(xx) + np.random.normal(0, 50, size=xx.size)  # ノイズを入れる

    # index_start=5, index_end=40 の区間だけフィット
    #  さらに三次関数は (x[5], y[5]) と (x[40], y[40]) を厳密に通る
    cubic_func = fixed_dim3_regression(xx, yy, 5, 40)

    # fitted_functionを使って予測値を作成
    yy_fit = np.array([cubic_func(xi) for xi in xx])

    # 結果表示
    plt.scatter(xx, yy, color="gray", label="data")
    plt.scatter([xx[5], xx[40]], [yy[5], yy[40]], color="red", label="fixed points")
    plt.plot(xx, yy_fit, color="blue", label="fitted cubic")
    plt.legend()
    plt.show()
