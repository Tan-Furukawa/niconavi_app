from stores import Stores
import numpy as np

from niconavi.optics.plot import canonicalize_poles

from niconavi.tools.type import D2FloatArray, D2IntArray, D2BoolArray, D1FloatArray

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from niconavi.image.type import RGBPicture
from niconavi.image.image import apply_color_to_mask, create_outside_circle_mask


from niconavi.analysis import make_grain_mask


from typing import Optional, Any, Literal

import matplotlib
from niconavi.optics.plot import add_polar_legend, plot_as_stereo_projection
from numpy.typing import NDArray


from copy import deepcopy
from tools.no_image import get_no_image
from components.view.spatial_units import apply_micrometer_axis


# matplotlib.use("Agg")  # 非対話型バックエンドに切り替え
# matplotlib.use("svg")  # 非対話型バックエンドに切り替え


def make_vmin_vmax(img: D2FloatArray | D2IntArray) -> tuple[float, float]:
    mask = create_outside_circle_mask(img)
    vmax = np.max(img[~mask])
    vmin = np.min(img[~mask])

    return float(vmin), float(vmax)


def plot_RGBpicture(
    stores: Stores, img: RGBPicture | None, mask: Optional[D2BoolArray] = None
) -> Figure:


    if img is not None:
        fig, ax = plt.subplots()
        imshow_with_grain_mask(stores, ax, img, mask=mask)
        apply_micrometer_axis(ax, stores)
        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig, ax)
        return fig
    return get_no_image()


def imshow_with_grain_mask(
    stores: Stores,
    ax: Axes,
    img: RGBPicture | D2FloatArray | D2IntArray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_at_2d: str = "hsv",
    is_log_norm: bool = False,
    mask: Optional[D2BoolArray] = None,
    **kwargs: Any,
) -> AxesImage:


    grain_map = stores.computation_result.grain_map.get()
    grain_boundary = stores.computation_result.grain_boundary.get()
    circ_mask = create_outside_circle_mask(img)
    mask_mode_mask = stores.computation_result.mask.get()

    if mask_mode_mask is None:
        mask_mode_mask = np.ones_like(circ_mask)

    if mask is not None and stores.ui.apply_mask.get():
        circ_mask_used = D2BoolArray(mask | mask_mode_mask)
    elif mask is not None:
        circ_mask_used = D2BoolArray(mask)
    elif mask is None and stores.ui.apply_mask.get():  # mask is None
        circ_mask_used = mask_mode_mask
    else:  # mask is None
        # circ_mask_used = circ_mask & (~circ_mask & mask_mode_mask)
        circ_mask_used = np.zeros_like(circ_mask)

    # !
    background_color = stores.ui.image_viewer.background_color.get()
    boundary_color = stores.ui.image_viewer.grain_boundary_color.get()

    if grain_map is not None and grain_boundary is not None:
        if stores.ui.display_grain_boundary.get():

            gmask = grain_boundary

            _img = apply_color_to_mask(
                img,
                gmask,
                stores.ui.image_viewer.grain_boundary_color.get(),
                vmin,
                vmax,
                cmap_at_2d,
                is_log_norm=is_log_norm,
            )

            _img = apply_color_to_mask(
                _img,
                circ_mask_used,
                stores.ui.image_viewer.background_color.get(),
                vmin,
                vmax,
                cmap_at_2d,
            )
            return ax.imshow(_img, cmap=cmap_at_2d, vmin=vmin, vmax=vmax, **kwargs)
        else:
            _img = apply_color_to_mask(
                img,
                circ_mask_used,
                stores.ui.image_viewer.background_color.get(),
                vmin,
                vmax,
                cmap_at_2d,
            )

            return ax.imshow(_img, **kwargs, vmax=vmax, vmin=vmin, cmap=cmap_at_2d)
    else:

        _img = apply_color_to_mask(
            img,
            circ_mask_used,
            stores.ui.image_viewer.background_color.get(),
            vmin,
            vmax,
            cmap_at_2d,
        )

        return ax.imshow(_img, **kwargs, vmax=vmax, vmin=vmin, cmap=cmap_at_2d)


def apply_mask(stores: Stores, mat: NDArray) -> NDArray:
    if np.ndim(mat) == 2 or np.ndim(mat) == 3:
        grain_map = stores.computation_result.grain_map.get()
        grain_boundary = stores.computation_result.grain_boundary.get()

        if stores.ui.apply_mask.get() and (
            stores.computation_result.mask.get() is not None
        ):
            mask_used_mask = stores.computation_result.mask.get()
        else:
            mask_used_mask = np.zeros(grain_map.shape, dtype=np.bool_)

        if np.ndim(mat) == 2:
            circ_mask = create_outside_circle_mask(mat)  # type: ignore
        else:
            circ_mask = create_outside_circle_mask(mat[:, :, 0])  # type: ignore

        mat_used = deepcopy(mat)

        print(mask_used_mask)

        if grain_map is not None and grain_boundary is not None:

            grain_classification_result = (
                stores.computation_result.grain_classification_result.get()
            )
            if grain_classification_result is not None:
                grain_mask = make_grain_mask(grain_classification_result, grain_map)
            else:
                grain_mask = None

            if grain_mask is not None:
                circ_mask_used = D2BoolArray(circ_mask | grain_mask)
            else:
                circ_mask_used = D2BoolArray(circ_mask)

            mat_used[circ_mask_used] = np.nan
            if mask_used_mask is not None:
                mat_used[mask_used_mask] = np.nan
            return mat_used
        else:
            mat_used[circ_mask] = np.nan
            if mask_used_mask is not None:
                mat_used[mask_used_mask] = np.nan
            return mat_used
    else:
        raise ValueError("the dimension of mat should 2 or 3")


def set_default_figure_style(
    fig: Figure,
    ax: Optional[Axes] = None,
    method: Literal[
        "rose diagram", "scatter", "histogram", "map_only_right_bottom", "polar plot"
    ] = "scatter",
    line_color: Optional[str] = None,
) -> None:

    fig.patch.set_facecolor("none")
    # fig.subplots_adjust(left=-0.2)

    if ax is not None:
        pass
    else:
        ax = fig.axes[0]

    ax.set_facecolor("none")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    if method == "rose diagram":
        ax.grid(color="white", alpha=0.3)
        ax.spines["polar"].set_color("white")
        ax.spines["start"].set_color("white")
        ax.spines["end"].set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for patch in ax.patches:
            patch.set_edgecolor("white")

    elif method == "polar plot":

        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.spines["top"].set_color("white")

        font_size = 7
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        fig.axes[0].xaxis.label.set_fontsize(font_size)
        fig.axes[0].yaxis.label.set_fontsize(font_size)

        # ax.grid(color="white", alpha=0.3)
        # ax.spines["polar"].set_color("white")
        # ax.spines["start"].set_color("white")
        # ax.spines["end"].set_color("white")
        # ax.xaxis.label.set_color("white")
        # ax.yaxis.label.set_color("white")
        # for patch in ax.patches:
        #     patch.set_edgecolor("white")
        # # font_size = 7
        # # fig.axes[0].xaxis.label.set_fontsize(font_size)
        # # fig.axes[0].yaxis.label.set_fontsize(font_size)
        # # ax.title.set_color("white")

    elif method == "histogram":
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        for patch in ax.patches:
            patch.set_edgecolor("white")

    elif method == "scatter":
        # 軸、メモリ、文字などを白色に設定
        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.spines["top"].set_color("white")

        font_size = 12
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        fig.axes[0].xaxis.label.set_fontsize(font_size)
        fig.axes[0].yaxis.label.set_fontsize(font_size)

        for col in ax.collections:
            ...
            # if isinstance(col, mc.PathCollection):
            # col.set_edgecolors("white" if line_color is None else line_color)  # type: ignore
            # col.set_colors("white")
            # col.set_linewidths(1.0)  # type: ignore

        ax.title.set_color("white")

    elif method == "map_only_right_bottom":
        # 軸、メモリ、文字などを白色に設定
        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")

        font_size = 7
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        fig.axes[0].xaxis.label.set_fontsize(font_size)
        fig.axes[0].yaxis.label.set_fontsize(font_size)
        ax.title.set_color("white")


def set_colorbar_style(cbar: Colorbar, label_name: str = "") -> None:
    font_size = 7
    # cbar.outline.set_edgecolor("white")  # 例: 赤色
    cbar.ax.yaxis.set_tick_params(
        color="white", labelcolor="white", labelsize=font_size
    )  # 例: 青色
    cbar.set_label(label_name, color="white")


def add_colorbar_at_correct_position(
    fig: Figure,
    img_ax: Axes,
    axes_image: AxesImage,
    is_log_norm: bool = False,
    cmap_log: Optional[str] = None,
    vmin_log: Optional[float] = None,
    vmax_log: Optional[float] = None,
) -> Colorbar:

    pos = img_ax.get_position()

    if is_log_norm and cmap_log is not None:
        cmap = cm.get_cmap(cmap_log)
        norm = LogNorm(vmin=vmin_log, vmax=vmax_log)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=img_ax)
    else:
        cbar = fig.colorbar(axes_image, ax=img_ax)

    pos_img = img_ax.get_position()
    pos_cbar = cbar.ax.get_position()  # [x0, y0, width, height] 形式
    # 例: カラーバーを画像の右側に少し余白をあけて配置
    # 今回は pos_img.x1 （画像軸右端）から 0.02 離して配置するイメージ
    new_x0 = pos_img.x1 + 0.1
    new_width = 0.1  # カラーバーの幅
    new_pos_cbar = (new_x0, pos_img.y0, new_width, pos_img.height)
    cbar.ax.set_position(new_pos_cbar)
    # 軸の位置を元に戻す
    img_ax.set_position(pos)
    return cbar


def plot_polar_map(
    stores: Stores,
    cmap: RGBPicture | None,
    legend: RGBPicture | None,
    mask: Optional[D2BoolArray] = None,
) -> Figure:
    if cmap is not None and legend is not None:
        fig, ax = plt.subplots()
        imshow_with_grain_mask(stores, ax, cmap, mask=mask)
        apply_micrometer_axis(ax, stores)
        add_polar_legend(ax, legend, color="white")
        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig, ax, method="map_only_right_bottom")
        return fig
    else:
        return get_no_image()


def plot_float_map(
    stores: Stores,
    img: D2FloatArray | D2IntArray | None,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    label: str,
    is_log_norm: bool = False,
    mask: Optional[D2BoolArray] = None,
    display_color_bar: bool = True,
    **kwargs: Any,
) -> Figure:

    if img is not None:
        if vmin is None or vmax is None:
            vmin_vmax_used = make_vmin_vmax(img)
            if vmin is None:
                vmin_used = vmin_vmax_used[0]
            else:
                vmin_used = vmin
            if vmax is None:
                vmax_used = vmin_vmax_used[1]
            else:
                vmax_used = vmax
        else:
            vmin_used = vmin
            vmax_used = vmax

        fig, ax = plt.subplots()

        im = imshow_with_grain_mask(
            stores,
            ax,
            img,
            cmap_at_2d=cmap,
            vmin=vmin_used,
            vmax=vmax_used,
            is_log_norm=is_log_norm,
            mask=mask,
            **kwargs,
        )
        apply_micrometer_axis(ax, stores)

        if display_color_bar:
            cbar = add_colorbar_at_correct_position(
                fig, ax, im, is_log_norm, cmap, vmin_used, vmax_used
            )

            stores.ui.displayed_fig.set(deepcopy(fig))
            set_colorbar_style(cbar, label)
        set_default_figure_style(fig, ax)
        return fig
    else:
        return get_no_image()


def plot_polar_distribution(
    stores: Stores,
    angles: dict[str, tuple[D1FloatArray, D1FloatArray]],
    azimuth_range: tuple[float, float] = (0, 360),
    mask: Optional[D2BoolArray] = None,
    pixel_azimuth_map: Optional[D2FloatArray] = None,
    pixel_inclination_map: Optional[D2FloatArray] = None,
) -> Figure:
    grain_classification_result = (
        stores.computation_result.grain_classification_result.get()
    )

    if grain_classification_result is not None:
        # ---------------------------------
        # polar plotをgrain単位でプロットするとき
        # ---------------------------------
        if stores.ui.analysis_tab.computation_unit.get() == "grain":
            inclination = D1FloatArray(np.array([], dtype=np.float64))
            azimuth = D1FloatArray(np.array([], dtype=np.float64))

            for mineral in grain_classification_result.keys():
                # print(angles)
                # print("mineral")
                if grain_classification_result[mineral]["display"]:
                    # print("--------------------")
                    # # print(grain_classification_result)
                    # print(angles[mineral][0])
                    # print(angles[mineral][1])
                    # print("--------------------")
                    inclination = D1FloatArray(
                        np.concatenate([inclination, angles[mineral][0]])
                    )

                    azimuth = D1FloatArray(np.concatenate([azimuth, angles[mineral][1]]))
                    # print(np.sum(angles[mineral][1] == 0))
                    # print(np.sum(azimuth == 0))
                else:
                    continue

        # ---------------------------------
        # polar plotをpixel単位でプロットするとき
        # ---------------------------------
        else:

            grain_map = stores.computation_result.grain_map.get()
            grain_boundary = stores.computation_result.grain_boundary.get()
            if (
                grain_map is not None
                and grain_boundary is not None
                and pixel_inclination_map is not None
                and pixel_azimuth_map is not None
            ):
                # circ_mask = create_outside_circle_mask(pixel_inclination_map)
                # if mask is not None:
                #     circ_mask_used = D2BoolArray(circ_mask | mask)
                # else:
                #     circ_mask_used = D2BoolArray(circ_mask)

                tmp_azimuth = apply_mask(stores, pixel_azimuth_map).flatten()
                tmp_inc = apply_mask(stores, pixel_inclination_map).flatten()

                if len(tmp_inc) > 2000 and len(tmp_azimuth) > 2000:
                    sample = np.random.choice(len(tmp_inc), 2000)
                    azimuth, inclination = canonicalize_poles(
                        tmp_azimuth[sample], tmp_inc[sample]
                    )

                else:
                    return get_no_image()

            else:
                return get_no_image()

        bandwidth = stores.ui.analysis_tab.cip_bandwidth.get()

        p = stores.ui.analysis_tab.cip_points_noise_size_percent.get() * 0.01

        np.random.seed(1234)
        # inclination.
        if len(inclination) < 2 or len(azimuth) < 2:
            return get_no_image()
        max_inc = np.nanmax(inclination)
        min_inc = np.nanmin(inclination)
        dinc = np.random.normal(0, (max_inc - min_inc) * p, len(inclination))
        max_azm = np.nanmax(azimuth)
        min_azm = np.nanmin(azimuth)
        dazm = np.random.normal(0, (max_azm - min_azm) * p, len(azimuth))

        fig, ax, cbar = plot_as_stereo_projection(
            inclination=D1FloatArray(np.clip(inclination + dinc, min_inc, max_inc)),
            azimuth=D1FloatArray(np.clip(azimuth + dazm, min_azm, max_azm)),
            azimuth_range_max=azimuth_range[1],
            sigma=bandwidth,
            plot_points=stores.ui.analysis_tab.cip_display_points.get(),
            levels=stores.ui.analysis_tab.cip_contour.get(),
            cmap=stores.ui.analysis_tab.cip_theme.get(),
        )
        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig, ax, method="polar plot")

        if cbar is not None:
            set_colorbar_style(cbar, "density")

        return fig
    else:
        return get_no_image()
