# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
from niconavi.tools.type import D1FloatArray, D2FloatArray, D3FloatArray
from matplotlib.pyplot import Figure, Axes
from matplotlib.colorbar import Colorbar
import matplotlib.pyplot as plt


def spherical_to_cartesian(
    inclination_deg: D1FloatArray, azimuth_deg: D1FloatArray
) -> D3FloatArray:
    """
    入力: inclination_deg: 0〜180度（極角; 0:北極, 90:赤道, 180:南極）
           azimuth_deg: 0〜360度
    出力: 単位ベクトル (N x 3)
    """
    theta = inclination_deg  # 極角
    phi = azimuth_deg
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return D3FloatArray(np.column_stack((x, y, z)))


def compute_vmf_normalization(kappa: float, d: int = 3) -> float:
    """
    d次元（実際は S^(d-1) 上の分布、ここでは d=3）のフォン・ミーゼス・フィッシャー分布の
    正規化定数 C_d(kappa) = kappa^(d/2-1) / ((2pi)^(d/2)* I_(d/2-1)(kappa))
    """
    nu = d / 2 - 1  # for d=3, nu = 0.5
    return (kappa**nu) / ((2 * np.pi) ** (d / 2) * iv(nu, kappa))


def stereographic_inverse(u: D2FloatArray, v: D2FloatArray) -> D3FloatArray:
    """
    ステレオ投影（南極から射影）による上半球から球面上への逆変換
    (u,v): 平面上の座標
    返り値: 対応する単位ベクトルの (x,y,z)（N個の点）

    逆変換の式:
      X = 2u/(1+u^2+v^2)
      Y = 2v/(1+u^2+v^2)
      Z = (1 - u^2 - v^2)/(1+u^2+v^2)
    """
    denom = 1 + u**2 + v**2
    X = 2 * u / denom
    Y = 2 * v / denom
    Z = (1 - u**2 - v**2) / denom
    return D3FloatArray(np.column_stack((X, Y, Z)))


def kde_vmf(
    inclination: D1FloatArray,
    azimuth: D1FloatArray,
    bandwidth: float,
    grid_res: int = 200,
) -> tuple[D2FloatArray, D2FloatArray, D2FloatArray]:
    """
    入力:
      inclination: D1FloatArray, 上半球の傾斜角（0～90°）
      azimuth: D1FloatArray, 方位角（0～360°）
      bandwidth: float, vMF分布の集中度パラメータ (kappa)
      grid_res: int, ステレオ投影のメッシュ解像度（例: 200×200）
    処理:
      ・上球面データの antipodal point (下球面) を計算して結合
      ・各データを 3次元単位ベクトルに変換
      ・フォン・ミーゼス・フィッシャーのカーネル密度推定を実施
      ・上半球のステレオ投影座標上の評価結果を計算して返す
    返り値:
      X, Y: メッシュグリッド（ステレオ投影平面上の座標）
      density: 各メッシュ点でのKDE値
    """
    # 上球面のデータは、入力そのまま
    top_incl = D1FloatArray(np.array(inclination))
    top_azi = D1FloatArray(np.array(azimuth))

    # 下球面の点は上球面の各点の原点対称
    # 球面の原点対称は、単位ベクトルで -1 倍
    top_vectors = spherical_to_cartesian(top_incl, top_azi)
    bottom_vectors = -top_vectors  # antipodal

    # 全データ（2N個）
    data_vectors = np.concatenate((top_vectors, bottom_vectors), axis=0)
    N_total = data_vectors.shape[0]

    # vMFの正規化定数（d=3）
    C = compute_vmf_normalization(bandwidth, d=3)

    # メッシュ作成（ステレオ投影平面: u,v）
    u = np.linspace(-1, 1, grid_res)
    v = np.linspace(-1, 1, grid_res)
    U, V = np.meshgrid(u, v)
    # 上半球のステレオ投影は単位円内 (u^2+v^2 <= 1)
    mask = U**2 + V**2 <= 1

    # メッシュ上の (u,v) を (x,y,z) に変換（逆ステレオ投影）
    U_flat = D2FloatArray(U[mask].ravel())
    V_flat = D2FloatArray(V[mask].ravel())
    grid_vectors = stereographic_inverse(U_flat, V_flat)  # shape (M,3)

    # KDEの評価
    # 各評価点 grid_vectors[j] と各サンプル data_vectors[i] との内積
    dots = grid_vectors @ data_vectors.T  # shape (n_grid, N_total)
    # カーネル値: C * exp(bandwidth * (x dot xi))
    kernel_vals = C * np.exp(bandwidth * dots)
    # 平均を取ってKDE値とする（密度推定）
    density_vals = np.sum(kernel_vals, axis=1) / N_total

    # 結果をグリッドサイズに戻す（unit circle 内の点のみ）
    density_grid = np.full(U.shape, np.nan)
    density_grid[mask] = density_vals

    return D2FloatArray(U), D2FloatArray(V), D2FloatArray(density_grid)


def sample_sphere_points(num: int) -> tuple[D1FloatArray, D1FloatArray]:
    """
    球面上の点をランダムサンプリングする関数

    引数:
      num: サンプリングする点の数

    返り値:
      inclination: 傾斜角（θ）の配列（[0, π]）
      azimuth: 方位角（φ）の配列（[0, 2π)）
    """
    # 方位角 φ は 0 から 2π まで一様にサンプリング
    azimuth = 2 * np.pi * np.random.rand(num)

    # 傾斜角 θ は cos(θ) を -1〜1 の一様乱数としてサンプリングし、
    # θ = arccos( cos(θ) ) により求める
    cos_inclination = 2 * np.random.rand(num) - 1
    inclination = np.arccos(cos_inclination)
    return D1FloatArray(inclination), D1FloatArray(azimuth)


#     return fig, ax, cbar
def plot_as_stereo_projection(
    inclination: D1FloatArray,
    azimuth: D1FloatArray,
    kappa: float = 0.1,
    plot_points: bool = False,
    azimuth_range_max: float = 360,
) -> tuple[Figure, Axes, Colorbar]:
    """
    inclination, azimuth のデータをステレオ投影し、フォン・ミーゼス・フィッシャー分布による
    カーネル密度推定の結果を描画する。
    軸や軸ラベルを非表示にし、描画領域は NaN でないセルをぴったり含むようにリサイズする。

    追加引数:
      azimuth_range_max: 方位角 0〜azimuth_range_max 度を同一視して使用し、
                         結果を 0〜azimuth_range_max の扇形として可視化する上限値（既定 360 度）。
    """

    # --- azimuth を 0〜azimuth_range_max 度で同一視した上で 0〜2π に正規化 ---
    azimuth_mod_deg = azimuth % azimuth_range_max
    azimuth_mod_rad = azimuth_mod_deg / azimuth_range_max * (2.0 * np.pi)
    # inclination はそのまま radian に変換
    inclination_rad = np.radians(inclination)

    # --- KDEを計算 ---
    U, V, density = kde_vmf(
        inclination_rad,
        azimuth_mod_rad,
        kappa,
        grid_res=200,
    )

    fig, ax = plt.subplots()
    cmap = ax.contourf(U, V, density, levels=10, cmap="jet")
    cbar = fig.colorbar(cmap, label="Density")
    plt.show()
    # U = transform_matrix(U, np.radians(azimuth_range_max))
    # V = transform_matrix(V, np.radians(azimuth_range_max))
    density = transform_matrix(density, np.radians(azimuth_range_max))

    # --- ステレオ投影平面 (U,V) を極座標に変換して扇形領域を決定 ---
    alpha = np.arctan2(V, U)  # -π〜π
    alpha[alpha < 0.0] += 2.0 * np.pi  # 0〜2π に揃える
    wedge_max_rad = azimuth_range_max * np.pi / 180.0

    # 扇形外の領域を NaN に置き換え
    density_wedge = np.copy(density)
    density_wedge[alpha > wedge_max_rad] = np.nan

    # --- 描画 ---
    fig, ax = plt.subplots()
    # カラー等高線図
    cmap = ax.contourf(U, V, density_wedge, levels=10, cmap="jet")
    cbar = fig.colorbar(cmap, label="Density")

    # plot_points=True なら元の (inclination, azimuth) で点を重ねる
    if plot_points:
        r = np.tan(inclination_rad / 2.0)
        x = r * np.cos(np.radians(azimuth))
        y = r * np.sin(np.radians(azimuth))
        ax.scatter(x, y, s=1, color="white")

    # 軸やラベルを非表示
    ax.set_axis_off()
    # カラーバーも軸を持つので、必要に応じてこちらも消す場合はコメントアウトを解除
    # cbar.remove()

    # ステレオ投影の NaN でない部分のみをちょうど含むように描画領域をリサイズ
    valid = ~np.isnan(density_wedge)
    x_valid = U[valid]
    y_valid = V[valid]
    if len(x_valid) > 0 and len(y_valid) > 0:
        ax.set_xlim(x_valid.min(), x_valid.max())
        ax.set_ylim(y_valid.min(), y_valid.max())

    # 図全体を余白なくトリミング
    # （複数ステップを組み合わせることで、極力ムダな余白を削減します）
    # fig.tight_layout(pad=0)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # アスペクト比を正方形に
    ax.set_aspect("equal")

    return fig, ax, cbar


def transform_matrix(mat: np.ndarray, azimuth_max: float) -> np.ndarray:
    """
    mat の内接円内にある要素 (R, θ) を，角度 θ/(2π)*azimuth_max に写像した位置へ
    移動した新しい 2 次元配列を作成して返します．
    内接円外の領域および移動後に埋まらない領域は np.nan にします。

    Parameters
    ----------
    mat : np.ndarray (2次元)
        変換元の実数配列
    azimuth_max : float
        [0, 2π) の実数値

    Returns
    -------
    transformed : np.ndarray
        同じ大きさの 2 次元実数配列．円の外側や要素が移動してこない領域は np.nan．
    """

    # 行列の高さ(H)と幅(W)
    H, W = mat.shape

    # 中心座標 (cy, cx) を計算
    cy = H / 2.0
    cx = W / 2.0

    # 内接円の半径 (行列の縦横いずれか短い方の半分)
    radius = min(H, W) / 2.0

    # 結果を格納する配列（初期値は np.nan）
    transformed = np.full((H, W), np.nan, dtype=mat.dtype)

    # 全要素をループ
    for r in range(H):
        for c in range(W):
            # 中心からの位置(dx, dy)
            dx = c - cx
            dy = r - cy

            # 半径R
            R = np.hypot(dx, dy)  # sqrt(dx^2 + dy^2)

            # 円の内部か判定
            if R <= radius:
                # 角度θ (範囲を [0, 2π) にする)
                theta = np.arctan2(dy, dx)
                if theta < 0:
                    theta += 2.0 * np.pi

                # 角度の変換: θ → θ' = (θ / 2π) * azimuth_max
                theta_new = (theta / (2.0 * np.pi)) * azimuth_max

                # 新しい座標 (row2, col2)
                #   x軸が右向き, y軸が下向きであることに注意
                new_x = cx + R * np.cos(theta_new)
                new_y = cy + R * np.sin(theta_new)

                # 画素に対応させるために最近傍整数へ
                col2 = int(round(new_x))
                row2 = int(round(new_y))

                # 範囲内であれば値を割り当てる
                if 0 <= row2 < H and 0 <= col2 < W:
                    transformed[row2, col2] = mat[r, c]

    return transformed


if __name__ == "__main__":
    import pandas as pd
    from niconavi.type import ComputationResult

    r: ComputationResult = pd.read_pickle("../../test/data/output/tetori_class_inc.pkl")

    # test/data/output/tetori_4k_xpl_pol_til.pkl_classified.pkl

    # params: ComputationResult = pd.read_pickle(
    #     "../test/data/output/tetori_4k_xpl_pol_til10.pkl"
    # )
    # fig, ax = make_polar_plot(
    #     lambda x, y: get_spectral_distribution(
    #         get_full_wave_plus_mineral_retardation_system(R=x, azimuth=y)
    #         # get_mineral_retardation_system(R=x, azimuth=y)
    #     )["rgb"],
    #     num_azimuth=100,
    #     num_inc=20,
    #     thickness=0.03,
    # )

    if r.raw_maps is not None:
        i, a = r.cip_map_info["polar_info90"]["quartz"]
        # i = np.array([40.0])
        # a = np.array([200.0])

        plot_as_stereo_projection(i, a, 50, azimuth_range_max=90, plot_points=True)
        # i = inclination[mask]
