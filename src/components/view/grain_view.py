from __future__ import annotations

from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure

from stores import Stores
from niconavi.analysis import make_grain_mask
from niconavi.grain_detection import assign_random_rgb
from niconavi.tools.type import D2BoolArray, D2IntArray

from components.view.plot import plot_RGBpicture, plot_float_map, imshow_with_grain_mask
from components.view.style import set_default_figure_style
from components.view.spatial_units import apply_micrometer_axis
from tools.no_image import get_no_image


def plot_grain_map(
    stores: Stores, grain_map: D2IntArray | None, show_grain_boundary: bool
) -> Figure:
    if grain_map is None:
        return get_no_image()

    fig, ax = plt.subplots()

    if show_grain_boundary:
        imshow_with_grain_mask(
            stores,
            ax,
            assign_random_rgb(
                grain_map,
                use_color=[(0, (0, 0, 0)), (999999, (0, 0, 0))],
            ),
        )
    else:
        ax.imshow(
            assign_random_rgb(
                grain_map,
                use_color=[(0, (0, 0, 0)), (999999, (0, 0, 0))],
            )
        )

    apply_micrometer_axis(ax, stores)
    stores.ui.displayed_fig.set(deepcopy(fig))
    set_default_figure_style(fig, ax)
    return fig


def at_grain_tab(stores: Stores) -> Figure:
    grain_map = stores.computation_result.grain_map.get()
    grain_classification_result = (
        stores.computation_result.grain_classification_result.get()
    )

    if grain_map is not None and grain_classification_result is not None:
        grain_mask: Optional[D2BoolArray] = make_grain_mask(
            grain_classification_result, grain_map
        )
    else:
        grain_mask = None

    raw_map = stores.computation_result.raw_maps.get()

    tilt0 = stores.computation_result.tilt_image_info.tilt_image0.get()
    tilt45 = stores.computation_result.tilt_image_info.tilt_image45.get()

    if raw_map is None:
        return get_no_image()

    selected = stores.ui.selected_button_at_grain_tab.get()

    match selected:

        case 0:
            return plot_RGBpicture(stores, raw_map.get("degree_0"), mask=grain_mask)
        case 1:
            return plot_RGBpicture(stores, raw_map.get("degree_22_5"), mask=grain_mask)
        case 2:
            return plot_RGBpicture(stores, raw_map.get("degree_45"), mask=grain_mask)
        case 3:
            return plot_RGBpicture(stores, raw_map.get("degree_67_5"), mask=grain_mask)
        case 4:
            return plot_float_map(
                stores,
                raw_map.get("extinction_angle"),
                "hsv",
                0,
                90,
                "extinction angle",
                mask=grain_mask,
            )
        case 5:
            base_map = raw_map.get("R_color_map_display")
            return plot_RGBpicture(stores, base_map, mask=grain_mask)
        case 6:
            return plot_RGBpicture(
                stores, raw_map.get("extinction_color_map"), mask=grain_mask
            )
        case 7:
            return plot_grain_map(stores, grain_map, True)
        case 8:
            grain_map_with_boundary = (
                stores.computation_result.grain_map_with_boundary.get()
            )
            return plot_float_map(
                stores,
                grain_map_with_boundary,
                "gray_r",
                0,
                None,
                "grain map",
                mask=grain_mask,
                display_color_bar=False,
            )
        case 9:
            return plot_float_map(
                stores,
                raw_map.get("max_retardation_map"),
                "plasma",
                0,
                None,
                "Retardation",
                mask=grain_mask,
            )
        case 10:
            return plot_RGBpicture(
                stores, raw_map.get("p45_R_color_map"), mask=grain_mask
            )
        case 11:
            return plot_RGBpicture(
                stores, raw_map.get("m45_R_color_map"), mask=grain_mask
            )
        case 12:
            return plot_float_map(
                stores,
                raw_map.get("p45_R_map"),
                "plasma",
                0,
                None,
                "Retardation",
                mask=grain_mask,
            )
        case 13:
            return plot_float_map(
                stores,
                raw_map.get("m45_R_map"),
                "plasma",
                0,
                None,
                "Retardation",
                mask=grain_mask,
            )
        case 14:
            return plot_float_map(
                stores,
                raw_map.get("azimuth"),
                "hsv",
                0,
                180,
                "azimuth",
                mask=grain_mask,
            )
        case 15:
            focused = tilt0["focused_tilted_image"] if tilt0 is not None else None
            return plot_RGBpicture(stores, focused, mask=grain_mask)
        case 16:
            focused = tilt45["focused_tilted_image"] if tilt45 is not None else None
            return plot_RGBpicture(stores, focused, mask=grain_mask)
        case 17:
            original = tilt0["original_image"] if tilt0 is not None else None
            return plot_RGBpicture(stores, original, mask=grain_mask)
        case 18:
            original = tilt45["original_image"] if tilt45 is not None else None
            return plot_RGBpicture(stores, original, mask=grain_mask)
        case 19:
            return plot_float_map(
                stores,
                raw_map.get("cv_extinction_angle"),
                "plasma",
                0,
                1,
                "quality of φ",
                mask=grain_mask,
            )
        case 20:
            return plot_float_map(
                stores,
                stores.computation_result.mask.get(),
                "gray_r",
                0,
                1,
                "quality of φ",
                mask=grain_mask,
                display_color_bar=False,
            )
        case _:
            return get_no_image()
