# %%
import numpy as np
from typing import cast, Literal, Callable, TypeVar
from niconavi.tools.type import D1FloatArray, D2FloatArray
import matplotlib.pyplot as plt
from copy import deepcopy

T = TypeVar("T", D1FloatArray, D2FloatArray)
T0 = TypeVar("T0", float, D1FloatArray, D2FloatArray)
T1 = TypeVar("T1", float, D1FloatArray, D2FloatArray)

def normalize_axes(inclination: T, azimuth: T) -> tuple[T, T]:
    """
    任意の (inclination, azimuth) を、軸の同一視
      (i, a) ~ (-i, a+180) および (i, a) ~ (i+180, a)
    を用いて、0 <= inclination < 90, 0 <= azimuth < 360 の形式に正規化する関数です。
    """
    # azimuth を 0 <= azimuth < 360 に正規化
    azimuth_used = azimuth % 360
    # inclination は軸の同一視から周期が 180 なので、まず 0 <= inclination < 180 に
    inclination_used = inclination % 180

    # inclination が 90 以上なら、(180 - inclination, azimuth + 180) に変換

    res_inclination = deepcopy(inclination_used)
    res_azimuth = deepcopy(azimuth_used)

    res_inclination[inclination_used >= 90] = (180 - inclination_used)[
        inclination_used >= 90
    ]
    res_azimuth[inclination_used >= 90] = ((azimuth_used + 180) % 360)[
        inclination_used >= 90
    ]

    return cast(T, res_inclination), cast(T, res_azimuth)


def make_angle_retardation_estimation_function(
    no: float, ne: float, thickness: float
) -> tuple[Callable[[T0], T0], Callable[[T0], T0]]:
    """
    パラメータ no, ne, thickness（単位：mm 等）から，
    ・θ -> retardation の関数
    ・retardation -> θ の関数
    を返す関数を生成する．

    ここで，
      - n_o = no
      - n_e(θ) = no * ne / sqrt(no^2*sin^2θ + ne^2*cos^2θ)
      - 位相差 δ = thickness*1e6 * |n_o - n_e(θ)|
        （位相差の単位は，例えばマイクロメートル単位など．）

    仮定：
      - θ は [0, π/2] の範囲
      - 位相差は δ ∈ [0, thickness*1e6*(no-ne)] （no > ne と仮定）

    Returns:
        tuple:
            - theta_to_retardation: Callable[[float], float]
                角度 θ を与えると位相差 δ を返す関数
            - retardation_to_theta: Callable[[float], float]
                位相差 δ を与えると角度 θ を返す関数
    """
    T = thickness * 1e6  # 定数 T の定義

    def theta_to_retardation(theta: T1) -> T1:
        """
        θ から位相差 δ を求める関数．
        """
        n_o = no
        # n_e(θ) の計算
        denominator = np.sqrt(no**2 * np.sin(theta) ** 2 + ne**2 * np.cos(theta) ** 2)
        n_e_theta = no * ne / denominator
        # ここでは 0 ≤ θ ≤ π/2 であれば n_o ≥ n_e(θ) となると仮定している
        delta = T * np.abs(n_o - n_e_theta)
        return delta

    def retardation_to_theta(delta: T1) -> T1:
        """
        位相差 δ から θ を求める関数．
        逆関数は以下の式から求める:
            r = δ / T
            sin²θ = [ne² (2 no r − r²)] / [(no² − ne²)(no − r)²]
        ただし r = δ/T であり，数値計算上の誤差対策として [0,1] にクランプする．
        """
        r = delta / T
        # δ の最大値は (no - ne)*T となるはず．それを超える場合はエラー
        # if r > (no - ne):
        #     raise ValueError(f"delta (= {delta}) is too large; maximum is {T*(no-ne)}")
        # # (no - r) が正であることを確認
        # if (no - r) <= 0:
        #     raise ValueError("Invalid value: no - (delta/T) must be positive.")
        # sin^2(theta) の計算
        # sin2_theta = (ne**2 * (2 * no * r - r**2)) / ((no**2 - ne**2) * (no - r) ** 2)
        if no > ne:
            sin2_theta = (-(ne**2) + (no * ne) ** 2 / (no - r) ** 2) / (no**2 - ne**2)
        else:
            sin2_theta = (-(ne**2) + (no * ne) ** 2 / (no + r) ** 2) / (no**2 - ne**2)

        sin2_theta = np.clip(sin2_theta, 0, 1)
        theta = np.arcsin(np.sqrt(sin2_theta))
        return theta

    return theta_to_retardation, retardation_to_theta


def get_thickness_from_max_retardation(
    max_retardation: float,
    no: float = 1.544,
    ne: float = 1.553,
) -> float:
    return max_retardation / (1e6 * np.abs(no - ne))


def get_max_retardation_from_thickness(
    thin_section_thickness: float = 0.05,
    no: float = 1.544,
    ne: float = 1.553,
) -> float:
    return thin_section_thickness * 1e6 * np.abs(no - ne)


# K = TypeVar("K", float, int)
# def f(x: K) -> K:
#     return x

if __name__ == "__main__":
    theta_to_R, R_to_theta = make_angle_retardation_estimation_function(
        1.544, 1.553, 0.04
    )

    x = np.linspace(0, 360, 500)
    y = np.degrees(R_to_theta(x))
    plt.plot(x, y)
    plt.scatter(x[5], y[5])
    plt.show()

# %%
