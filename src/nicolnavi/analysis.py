# %%
import scipy.stats as stats
import matplotlib.colors as mcolors
import colorsys
from niconavi.grain_analysis import reconstruct_grain_mask
from typing import Callable, Literal, TypeVar, Optional, Any, overload
from niconavi.tools.grain_plot import detect_boundaries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from niconavi.tools.type import (
    is_not_None_list,
    D2BoolArray,
    D2FloatArray,
    D2IntArray,
    D1FloatArray,
    is_not_none,
)
from niconavi.type import (
    Grain,
    GrainNumLiteral,
    GrainAcceptedLiteral,
    RawMapsNumLiteral,
    ComputationResult,
    GrainSelectedResult,
    is_GrainNumLiteral,
    is_RawMapsNumLiteral,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
import traceback


def rose_diagram(
    angles: list[float] | list[int],
    max_angle: float | None = 360,
    num_bins: int = 36,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    *,
    flip: bool = False,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    角度(度数法)のリスト angles を用いてローズダイヤグラムを作成する。
    max_angle に応じて 0～max_angle の範囲を半円または円等として描画する。

    Parameters
    ----------
    angles : list of float
        度数法(°)で与えられる角度のリスト
    max_angle : float, optional
        描画範囲の最大角度(度数法)
        例: 360 -> 全円、180 -> 半円
        デフォルトは 360
    num_bins : int, optional
        ビンの数(ヒストグラムの分割数)
        デフォルトは 36

    Returns
    -------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        極座標系の Axes
    fig : matplotlib.figure.Figure
        図(Figure)オブジェクト
    """

    if max_angle is None:
        max_angle = 360.0
    angles_np = np.asarray(angles, dtype=float)
    angles_rad = np.deg2rad(angles_np)

    theta_max = np.deg2rad(max_angle)

    # ヒストグラム(度数)を計算
    # range=(0,theta_max) で 0～theta_max までを num_bins 分割
    counts, bin_edges = np.histogram(angles_rad, bins=num_bins, range=(0, theta_max))

    # bin_edges は [0, ..., theta_max] の長さ(num_bins+1)、中心角はその中間点
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # プロット用に figure, ax を作成 (subplot_kw={"projection": "polar"} がポイント)
    if fig is None or ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # 極座標プロット(バーで表現)
    # 幅は各ビンの幅 = (theta_max / num_bins)
    width = bin_edges[1] - bin_edges[0]
    bars = ax.bar(
        bin_centers,
        counts,
        width=width,
        bottom=0,
        align="center",
        edgecolor="black",
        **kwargs,
    )

    if flip and np.isclose(max_angle, 180.0):
        mirror_centers = (bin_centers + np.pi) % (2 * np.pi)
        ax.bar(
            mirror_centers,
            counts,
            width=width,
            bottom=0,
            align="center",
            edgecolor="black",
            **kwargs,
        )
        theta_limit = 2 * np.pi
    else:
        theta_limit = theta_max

    # 極座標の目盛り(0°～max_angle°)の設定
    # matplotlib は 0 ～ 2π を一周とするため、max_angle=180(半円)なら 0～π、360(全円)なら 0～2π
    ax.set_thetalim(0, theta_limit)  # type: ignore

    # 0度位置を上にしたり、他の位置にする場合には以下を適宜変更
    # ax.set_theta_zero_location("N")      # 0度を上(北方向)に
    # ax.set_theta_direction(-1)          # 角度の増加方向を時計回りに変更

    # 目盛りのラベルを読みやすいように度数法で表示する
    # x軸(角度方向)のティックは自動では 0~2π の分割になるので自前で設定
    # 例えば 0, 30, 60, ... max_angle でラベルを付ける例
    tick_limit = 360.0 if flip and np.isclose(max_angle, 180.0) else max_angle
    tick_degs = np.linspace(0, tick_limit, num=7)
    ax.set_xticks(np.deg2rad(tick_degs))
    ax.set_xticklabels([f"{int(deg)}°" for deg in tick_degs])

    # y軸(半径方向)は自動でも問題なければそのままでもよい
    # 例えば最大値を超えた範囲まで見たければ ylim を設定
    # ax.set_ylim(top=counts.max() * 1.1)
    # タイトルなど

    return fig, ax


def is_in_index_dict(
    mineral: str, index_dict: dict[str, GrainSelectedResult]
) -> Callable[[Grain], bool]:
    def closure(grain: Grain) -> bool:
        try:
            return grain["index"] in index_dict[mineral]["index"]  # type: ignore
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"error in is_in_index_dict: {e}")

    return closure


def get_mineral_list_from_grain_list(
    params: ComputationResult, target: GrainNumLiteral, mineral: str
) -> list[float | int | None]:
    index_dict = params.grain_classification_result
    grain_list = params.grain_list
    if index_dict is not None and grain_list is not None:
        grain_list_target = list(
            filter(is_in_index_dict(mineral, index_dict), grain_list)
        )

        return list(map(lambda x: x[target], grain_list_target))
    else:
        raise ValueError("index_dict and grain_list should not None.")


def get_value_from_grain_list_of_key_mineral(
    params: ComputationResult,
    target_at_grain_list: str,
    d2map: D2FloatArray,
    mineral: str,
) -> list[float | int | None]:
    index_dict = params.grain_classification_result
    grain_list = params.grain_list
    if index_dict is not None and grain_list is not None:
        grain_list_target = list(
            filter(is_in_index_dict(mineral, index_dict), grain_list)
        )
        res = []
        for grain in grain_list_target:
            res += list(d2map[reconstruct_grain_mask(grain)])
        return res
    else:
        raise ValueError("index_dict and grain_list should not None.")


# def plot_1d_function_grain_base(
#     params: ComputationResult,
#     target: GrainNumLiteral,
#     mineral: str,
#     ax_plot_fn: Callable[[ComputationResult, list[float]], tuple[Figure, Axes]],
# ) -> tuple[Figure, Axes]:

#     l = get_mineral_list_from_grain_list(params, target, mineral)
#     if is_not_None_list(l):
#         return ax_plot_fn(params, l)
#     else:
#         raise ValueError("None is included in grain[target]")


# def plot_1d_function_map_base(
#     params: ComputationResult,
#     target: RawMapsNumLiteral,
#     mineral: str,
#     ax_plot_fn: Callable[[ComputationResult, list[float]], tuple[Figure, Axes]],
# ) -> tuple[Figure, Axes]:

#     maps = params.raw_maps

#     grain_map = params.grain_map

#     grain_selection_result = params.grain_classification_result[mineral]["index"]  # type: ignore

#     if (
#         maps is not None
#         and grain_map is not None
#         and grain_selection_result is not None
#     ):

#         mask_grain = np.isin(grain_map, grain_selection_result)

#         m = maps[target]
#         if m is not None:
#             out_mask = create_outside_circle_mask(m)
#             mo = m[~out_mask & mask_grain]

#             tt = list(map(lambda x: float(x), mo))

#             return ax_plot_fn(params, tt)
#         else:
#             raise ValueError("raw_maps[target] is None")

#     else:
#         raise ValueError("raw_maps is None")


def plot_2d_function_grain_base(
    params: ComputationResult,
    target1: GrainNumLiteral,
    target2: GrainNumLiteral,
    mineral: str,
    ax_plot_fn: Callable[
        [ComputationResult, list[float], list[float]],
        tuple[Figure, Axes],
    ],
) -> tuple[Figure, Axes]:

    index_dict = params.grain_classification_result
    grain_list = params.grain_list

    if index_dict is not None and grain_list is not None:
        grain_list_target = list(
            filter(is_in_index_dict(mineral, index_dict), grain_list)
        )

        t1 = list(map(lambda x: x[target1], grain_list_target))
        t2 = list(map(lambda x: x[target2], grain_list_target))

        if is_not_None_list(t1) and is_not_None_list(t2):
            return ax_plot_fn(params, t1, t2)
        else:
            raise ValueError("None is included in grain[target]")
    else:
        raise ValueError("index_dict is None.")


def plot_rose_diagram(
    params: ComputationResult,
    target: Literal["azimuth", "extinction_angle", "angle_deg"],
    method: Literal["grain", "pixel"],
    mineral: str,
    color: str,
    max_angle: float = 360,
    bins: int = 20,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:

    # def ax_plot_fn(
    #     params: ComputationResult, t: list[int] | list[float]
    # ) -> tuple[Figure, Axes]:
    #     max_angle = 180.0 if (target == "azimuth") or (target == "angle_deg") else 90.0
    #     bins = (
    #         params.plot_parameters.rose_diagram180_bins
    #         if target == "azimuth"
    #         else params.plot_parameters.rose_diagram90_bins
    #     )
    #     return rose_diagram(t, max_angle, bins, fig, ax, color=color)

    vec = get_mineral_list_from_grain_list(params, target, mineral)
    vec = list(filter(lambda x: x is not None, vec))
    return rose_diagram(vec, max_angle, bins, fig, ax, color=color)

    # if method == "grain":
    # return plot_1d_function_grain_base(params, target, mineral, ax_plot_fn)

    # if is_not_None_list(l):
    #     return ax_plot_fn(params, l)
    # else:
    #     raise ValueError("None is included in grain[target]")

    # elif method == "pixel":
    #     # print(target)
    #     return plot_1d_function_map_base(params, target, mineral, ax_plot_fn)
    # else:
    #     raise ValueError("method is grain or map")


def plot_histogram(
    params: ComputationResult,
    target: GrainNumLiteral | RawMapsNumLiteral,
    method: Literal["grain", "pixel"],
    mineral: str,
    color: str,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    bins: int = 20,
    xlim: Optional[tuple[float, float]] = None,
    log_x: bool = False,
    alpha: float = 1.0,
) -> tuple[Figure, Axes]:
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    def ax_plot_fn(
        params: ComputationResult, t: list[int] | list[float]
    ) -> tuple[Figure, Axes]:
        data = [float(v) for v in t if v is not None]
        if len(data) == 0:
            return fig, ax

        if log_x:
            data = [v for v in data if v > 0]
            if len(data) == 0:
                return fig, ax

            if xlim is not None:
                min_val, max_val = xlim
            else:
                min_val = min(data)
                max_val = max(data)

            if min_val <= 0 or max_val <= min_val:
                return fig, ax

            bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins + 1)
            ax.hist(data, bins=bin_edges, color=color, alpha=alpha)
            ax.set_xscale("log")
            ax.set_xlim(min_val, max_val)
        else:
            if xlim is None:
                ax.hist(data, bins, color=color, alpha=alpha)
            else:
                ax.hist(data, bins, color=color, range=xlim, alpha=alpha)
                ax.set_xlim(xlim)
        return fig, ax

    # if method == "grain":
    if is_GrainNumLiteral(target):
        # return plot_1d_function_grain_base(params, target, mineral, ax_plot_fn)
        l = get_mineral_list_from_grain_list(params, target, mineral)
        if is_not_None_list(l):
            return ax_plot_fn(params, l)
        else:
            raise ValueError("None is included in grain[target]")
    else:
        raise ValueError(
            "invalid Literal pair: grain method with RawMapsNumLiteral"
            )
    # elif method == "pixel":
    #     if is_RawMapsNumLiteral(target):
    #         return plot_1d_function_map_base(params, target, mineral, ax_plot_fn)
    #     else:
    #         raise ValueError("invalid Literal pair: map method with GrainNumLiteral")
    # else:
    #     raise ValueError("method should grain or map")


# def get_log_ok_float_list(
#     vec1: list[float],
#     vec2: list[float],
#     log_x: bool = False,
#     log_y: bool = False,
# ) -> tuple[list[float], list[float]]:
#     if log_x:
#         used = list(filter(lambda x: x > 0, range(len(vec1))))
#         vec1 = list(map(lambda i: float(vec1[i]), used))
#         vec2 = list(map(lambda i: float(vec2[i]), used))
#         vec1 = list(map(lambda x: np.log(x), vec1))
#     if log_y:
#         used = list(filter(lambda x: x > 0, range(len(vec2))))
#         vec1 = list(map(lambda i: float(vec1[i]), used))
#         vec2 = list(map(lambda i: float(vec2[i]), used))
#         vec2 = list(map(lambda x: np.log(x), vec2))
#     return vec1, vec2


def plot_grain_scatter(
    params: ComputationResult,
    target1: GrainNumLiteral,
    target2: GrainNumLiteral,
    mineral: str,
    color: str,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[Figure, Axes]:

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    def ax_plot_fn(
        params: ComputationResult, t1: list[float], t2: list[float]
    ) -> tuple[Figure, Axes]:

        paired = [
            (float(v1), float(v2))
            for v1, v2 in zip(t1, t2)
            if v1 is not None and v2 is not None
        ]
        if log_x:
            paired = [p for p in paired if p[0] > 0]
        if log_y:
            paired = [p for p in paired if p[1] > 0]

        if len(paired) == 0:
            return fig, ax

        xs, ys = zip(*paired)

        ax.scatter(xs, ys, color=color)

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        if xlab is not None:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)

        return fig, ax

    return plot_2d_function_grain_base(params, target1, target2, mineral, ax_plot_fn)


def make_displayed_grain_classification_result(
    params: ComputationResult,
) -> dict[str, GrainSelectedResult]:
    r = params.grain_classification_result
    if r is not None:
        return {k: v for k, v in r.items() if v["display"] is True}
    else:
        return {}


def filter_displayed_grain_classification_result(
    grain_classification_result: dict[str, GrainSelectedResult],
) -> dict[str, GrainSelectedResult]:
    return {
        k: v for k, v in grain_classification_result.items() if v["display"] is True
    }


def rose_diagram_for_all_minerals(
    params: ComputationResult,
    target_grain_list: Literal["azimuth", "extinction_angle", "angle_deg"],
    target_raw_map: Literal["azimuth", "extinction_angle"],
    method: Literal["grain", "pixel"],
    alpha: float = 1.0,
    flip: bool = False,
) -> tuple[Figure, Axes]:
    r = params.grain_classification_result
    if r is not None:
        filtered_gcr = filter_displayed_grain_classification_result(r)
    else:
        filtered_gcr = {}

    if filtered_gcr is not None:

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        if target_grain_list == "azimuth":
            ax.set_thetalim(0, np.radians(180))  # type: ignore
        elif target_grain_list == "extinction_angle":
            ax.set_thetalim(0, np.radians(90))  # type: ignore
        elif target_grain_list == "angle_deg":
            ax.set_thetalim(0, np.radians(180))  # type: ignore
        else:
            raise ValueError(
                "unexpected target_grain_list. target_grain_list should azimuth or extinction_angle"
            )

        classification_res = list(filtered_gcr.keys())

        for key in classification_res:

            vec = get_mineral_list_from_grain_list(params, target_grain_list, key)
            vec = list(filter(lambda x: x is not None, vec))

            max_angle = (
                180.0
                if (target_grain_list == "azimuth")
                or (target_grain_list == "angle_deg")
                else 90.0
            )
            bins = params.plot_parameters.rose_diagram_bins
            color = filtered_gcr[key]["color"]
            rose_diagram(vec, max_angle, bins, fig, ax, color=color, alpha=alpha, flip=flip)

        return fig, ax
    else:
        raise ValueError("r.grain_classification_result is None")


def make_grain_mask(
    grain_classification_result: dict[str, GrainSelectedResult], grain_map: D2IntArray
) -> D2BoolArray:

    r = grain_classification_result
    filtered_gcr = filter_displayed_grain_classification_result(r)
    # print("-----------")
    # print(filtered_gcr)
    # print("-----------")

    res = np.zeros(grain_map.shape, dtype=np.bool_)
    plotted_value_list: list[int] = []
    for mineral in filtered_gcr.keys():
        v = filtered_gcr[mineral]["index"]
        # print(v)
        plotted_value_list += list(filter(lambda x: x != 0, v))
    # print(plotted_value_list)
    for index in plotted_value_list:
        res[grain_map == index] = True
    return D2BoolArray(~res)


def histogram_for_all_minerals(
    params: ComputationResult,
    target: GrainNumLiteral | RawMapsNumLiteral,
    method: Literal["grain", "pixel"],
    log_x: bool = False,
    alpha: float = 1.0,
) -> tuple[Figure, Axes]:

    r = params.grain_classification_result
    if r is not None:
        filtered_gcr = filter_displayed_grain_classification_result(r)
    else:
        filtered_gcr = {}

    if filtered_gcr is not None:

        xlim: Optional[tuple[float, float]] = None

        if is_GrainNumLiteral(target):
            plotted_value_list: list[float | int | None] = []
            for mineral in filtered_gcr.keys():
                plotted_value_list += get_mineral_list_from_grain_list(
                    params, target, mineral
                )
            if len(plotted_value_list) > 0:
                values = [float(v) for v in plotted_value_list if v is not None]
                if log_x:
                    values = [v for v in values if v > 0]
                if len(values) > 0:
                    max_val = max(values)
                    min_val = min(values)
                    if log_x and min_val <= 0:
                        min_candidates = [v for v in values if v > 0]
                        if len(min_candidates) > 0:
                            min_val = min(min_candidates)
                    if max_val > min_val > 0:
                        xlim = (min_val, max_val)

        fig, ax = plt.subplots()
        classification_res = list(filtered_gcr.keys())

        bins = params.plot_parameters.histogram_bins

        for key in classification_res:
            plot_histogram(
                params,
                target,
                method,
                key,
                filtered_gcr[key]["color"],
                fig,
                ax,
                bins,
                xlim,
                log_x=log_x,
                alpha=alpha,
            )

        return fig, ax
    else:
        raise ValueError("r.grain_classification_result is None")


# 色を明るくする補助関数
def lighten_color(color: str, amount: float = 0.5) -> tuple[float, float, float]:
    """
    与えられた色を明るくする（amountが大きいほど明るくなる）
    """
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(*c)
    # l: 輝度（0〜1）; 明るくするために輝度を補正
    l = 1 - amount * (1 - l)
    return colorsys.hls_to_rgb(h, l, s)


def add_regression_line(
    fig: Figure,
    ax: Axes,
    vec1: D1FloatArray,
    vec2: D1FloatArray,
    color: str = "green",
    origin: bool = False,
    text_color: str = "black",
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[Figure, Axes, str]:
    """
    fig, ax: matplotlibのFigure, Axesオブジェクト
    vec1, vec2: NewTypeで定義された1次元のfloat配列（例: numpy.arrayまたはlist）
    color: 回帰直線の色（信頼区間はこの色を薄くした色で描画）
    origin: Trueの場合、回帰直線は原点を通るものとする

    グラフ上に回帰直線、回帰式（y = aaax + bbb）および95%信頼区間を描画する。
    """
    # 入力をnumpy配列に変換
    x = np.array(vec1, dtype=np.float64)
    y = np.array(vec2, dtype=np.float64)

    if log_x and np.any(x <= 0):
        raise ValueError("log scale requires positive x values")
    if log_y and np.any(y <= 0):
        raise ValueError("log scale requires positive y values")

    x_trans = np.log10(x) if log_x else x
    y_trans = np.log10(y) if log_y else y

    # 描画用のx軸範囲を用意（データ範囲全体をカバーするように）
    x_range_trans = np.linspace(x_trans.min(), x_trans.max(), 100)

    if not origin:
        # 一般の線形回帰（切片あり）の場合
        a, b = np.polyfit(x_trans, y_trans, 1)
        y_pred_line_trans = a * x_range_trans + b

        # 残差から平均二乗誤差（MSE）を計算
        n = len(x_trans)
        y_hat = a * x_trans + b
        residuals = y_trans - y_hat
        dof = n - 2  # 自由度
        mse = np.sum(residuals**2) / dof
        mean_x = np.mean(x_trans)
        Sxx = np.sum((x_trans - mean_x) ** 2)

        # 各xについて予測平均の標準誤差
        SE = np.sqrt(mse * (1 / n + (x_range_trans - mean_x) ** 2 / Sxx))
        t_val = stats.t.ppf(0.975, dof)
        ci_upper_trans = y_pred_line_trans + t_val * SE
        ci_lower_trans = y_pred_line_trans - t_val * SE

        # 回帰直線の描画
        x_range_plot = 10**x_range_trans if log_x else x_range_trans
        y_pred_plot = 10**y_pred_line_trans if log_y else y_pred_line_trans
        ax.plot(x_range_plot, y_pred_plot, color=color)
        # 95%信頼区間の描画（薄い色を使用）
        # lighter = lighten_color(color, amount=0.5)
        # ax.fill_between(
        #     x_range_plot,
        #     10**ci_lower_trans if log_y else ci_lower_trans,
        #     10**ci_upper_trans if log_y else ci_upper_trans,
        #     color=lighter,
        #     alpha=0.5,
        #     label="95% CI",
        # )

        # 回帰式をグラフ上に表示
        if log_x or log_y:
            eq_text = f"log(y) = {a:.3f} log(x) + {b:.3f}" if log_x and log_y else (
                f"log(y) = {a:.3f} x + {b:.3f}" if log_y else f"y = {a:.3f} log(x) + {b:.3f}"
            )
        else:
            eq_text = f"y = {a:.3f}x + {b:.3f}"
        # ax.text(
        #     0.05,
        #     0.95,
        #     eq_text,
        #     transform=ax.transAxes,
        #     fontsize=12,
        #     verticalalignment="top",
        #     color=text_color,
        # )
        # ax.set_title(eq_text)
    else:
        # 原点を通る回帰（切片 = 0 の制約付き）の場合
        a = np.sum(x_trans * y_trans) / np.sum(x_trans**2)
        b = 0.0
        y_pred_line_trans = a * x_range_trans

        n = len(x_trans)
        y_hat = a * x_trans
        residuals = y_trans - y_hat
        dof = n - 1  # 原点固定の場合はパラメータは1つ
        mse = np.sum(residuals**2) / dof
        Sxx = np.sum(x_trans**2)

        # 予測値の標準誤差（xの依存性あり）
        SE = np.sqrt(mse * (x_range_trans**2 / Sxx))
        t_val = stats.t.ppf(0.975, dof)
        ci_upper_trans = y_pred_line_trans + t_val * SE
        ci_lower_trans = y_pred_line_trans - t_val * SE

        x_range_plot = 10**x_range_trans if log_x else x_range_trans
        y_pred_plot = 10**y_pred_line_trans if log_y else y_pred_line_trans
        ax.plot(x_range_plot, y_pred_plot, color=color)
        # lighter = lighten_color(color, amount=0.5)
        # ax.fill_between(
        #     x_range_plot,
        #     10**ci_lower_trans if log_y else ci_lower_trans,
        #     10**ci_upper_trans if log_y else ci_upper_trans,
        #     color=lighter,
        #     alpha=0.5,
        #     label="95% CI",
        # )
        if log_x or log_y:
            eq_text = (
                f"log(y) = {a:.3f} log(x)"
                if log_x and log_y
                else ("log(y) = {a:.3f} x" if log_y else f"y = {a:.3f} log(x)")
            ).format(a=a)
        else:
            eq_text = f"y = {a:.3f}x"  # 切片は0なので表示省略
        # ax.text(
        #     0.05,
        #     0.95,
        #     eq_text,
        #     transform=ax.transAxes,
        #     fontsize=12,
        #     verticalalignment="top",
        #     color=text_color,
        # )
        # ax.set_title(eq_text)

    return fig, ax, eq_text


def scatter_for_all_minerals(
    params: ComputationResult,
    target1: GrainNumLiteral,
    target2: GrainNumLiteral,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    text_color: str = "black",
    origin: bool = True,
    display_regression: bool = True,
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[Figure, Axes]:

    r = params.grain_classification_result
    if r is not None:
        filtered_gcr = filter_displayed_grain_classification_result(r)
    else:
        filtered_gcr = {}

    if filtered_gcr is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        classification_res = list(filtered_gcr)

        for key in classification_res:
            fig, ax = plot_grain_scatter(
                params,
                target1,
                target2,
                key,
                filtered_gcr[key]["color"],
                fig,
                ax,
                xlab,
                ylab,
                log_x,
                log_y,
            )

        grain_list = params.grain_list
        if grain_list is not None and display_regression:
            pairs = [
                (g[target1], g[target2])
                for g in grain_list
                if g["mineral"] in classification_res
            ]

            filtered_pairs = [
                (float(x), float(y))
                for x, y in pairs
                if is_not_none(x) and is_not_none(y)
            ]

            if log_x:
                filtered_pairs = [p for p in filtered_pairs if p[0] > 0]
            if log_y:
                filtered_pairs = [p for p in filtered_pairs if p[1] > 0]

            if len(filtered_pairs) < 2:
                return fig, ax
            else:
                xx = [p[0] for p in filtered_pairs]
                yy = [p[1] for p in filtered_pairs]

                fig, ax, regression = add_regression_line(
                    fig,
                    ax,
                    D1FloatArray(np.array(xx)),
                    D1FloatArray(np.array(yy)),
                    text_color=text_color,
                    origin=origin,
                    color="magenta",
                    log_x=log_x,
                    log_y=log_y,
                )
                ax.set_title(regression)
                return fig, ax
        else:
            return fig, ax
    else:
        raise ValueError("r.grain_classification_result is None")


if __name__ == "__main__":

    # r: ComputationResult = pd.read_pickle(
    #     "../test/data/output/yamagami_cross_before_grain_classification.pkl"
    # )
    from niconavi.run_all import grain_segmentation

    r: ComputationResult = pd.read_pickle(
        "../test/data/output/tetori_cross.avi.pkl_classified.pkl"
    )

    r.grain_classification_code = """
    quartz[red]:  index != 0 and V > 50
    grt[green]: R < 100

        // background[black]: index == 0
        // grt[red]: R < 50
        // quartz[white]: R > 200 and size > 100
    """

    # r = grain_segmentation(r)
    # r.plot_parameters.rose_diagram180_bins = 20
    # r.plot_parameters.rose_diagram90_bins = 20
    r.grain_classification_result["quartz"]["display"] = True

    plt.imshow(
        make_grain_mask(r.grain_classification_result, r.grain_map_with_boundary)
    )
    # %%

    # r.grain_classification_result
    # k = filter_displayed_grain_classification_result(r)

    rose_diagram_for_all_minerals(r, "azimuth", "pixel")
    rose_diagram_for_all_minerals(r, "azimuth", "grain")
    rose_diagram_for_all_minerals(r, "extinction_angle", "pixel")
    rose_diagram_for_all_minerals(r, "extinction_angle", "grain")
    histogram_for_all_minerals(r, "eccentricity", "grain")
    histogram_for_all_minerals(r, "max_retardation_map", "pixel")
    histogram_for_all_minerals(r, "extinction_angle", "pixel")
    histogram_for_all_minerals(r, "extinction_angle", "grain")
    scatter_for_all_minerals(r, "azimuth", "extinction_angle")
    scatter_for_all_minerals(r, "R", "sd_azimuth")

    # histogram_for_all_minerals(r, "eccentricity", "map")
