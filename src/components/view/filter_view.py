from __future__ import annotations

from copy import deepcopy

from matplotlib.pyplot import Figure

from stores import Stores
from state import State
from niconavi.analysis import make_grain_mask
from niconavi.plot.index_plot import plot_grain_index

from components.view.plot import plot_float_map, plot_RGBpicture
from components.view.style import set_default_figure_style
from components.view.spatial_units import apply_micrometer_axis
from tools.no_image import get_no_image


def at_filter_tab(stores: Stores, tab_selected_state: State) -> Figure:
    grain_map = stores.computation_result.grain_map.get()
    grain_classification_result = (
        stores.computation_result.grain_classification_result.get()
    )

    if grain_map is not None and grain_classification_result is not None:
        grain_mask = make_grain_mask(grain_classification_result, grain_map)
    else:
        grain_mask = None

    selection_result = stores.computation_result.grain_classification_image.get()
    grain_list = stores.computation_result.grain_list.get()

    segmented_maps = stores.computation_result.grain_segmented_maps.get()
    raw_maps = stores.computation_result.raw_maps.get()

    if grain_list is None or segmented_maps is None or raw_maps is None:
        return get_no_image()

    index = tab_selected_state.get()

    if index == 0:
        return plot_RGBpicture(stores, selection_result, mask=grain_mask)
    if index == 1:
        return plot_float_map(
            stores,
            segmented_maps.get("max_retardation_map"),
            "plasma",
            0,
            None,
            "Retardation",
            mask=grain_mask,
        )
    if index == 2:
        return plot_float_map(stores, segmented_maps.get("H"), "hsv", 0, 179, "Hue", mask=grain_mask)
    if index == 3:
        return plot_float_map(
            stores,
            segmented_maps.get("S"),
            "gray",
            0,
            255,
            "Saturation",
            mask=grain_mask,
        )
    if index == 4:
        return plot_float_map(
            stores,
            segmented_maps.get("V"),
            "gray",
            0,
            255,
            "Value",
            mask=grain_mask,
        )
    if index == 5:
        return plot_RGBpicture(stores, segmented_maps.get("R_color_map_raw"), mask=grain_mask)
    if index == 6:
        return plot_float_map(
            stores,
            segmented_maps.get("eccentricity"),
            "viridis",
            0,
            1,
            "eccentricity",
            mask=grain_mask,
        )
    if index == 7:
        return plot_float_map(
            stores,
            segmented_maps.get("R_70_map"),
            "plasma",
            0,
            None,
            "Retardation",
            mask=grain_mask,
        )
    if index == 8:
        return plot_float_map(
            stores,
            segmented_maps.get("size"),
            "viridis",
            None,
            None,
            "grain size",
            is_log_norm=True,
            mask=grain_mask,
        )
    if index == 9:
        return plot_float_map(
            stores,
            segmented_maps.get("extinction_angle"),
            "hsv",
            0,
            90,
            "extinction angle",
            mask=grain_mask,
        )
    if index == 10:
        return plot_float_map(
            stores,
            segmented_maps.get("azimuth"),
            "hsv",
            0,
            180,
            "azimuth",
            mask=grain_mask,
        )
    if index == 11:
        return plot_float_map(
            stores,
            segmented_maps.get("sd_azimuth"),
            "binary",
            0,
            1,
            "sd(azimuth)",
            mask=grain_mask,
        )
    if index == 12:
        return plot_float_map(
            stores,
            segmented_maps.get("sd_extinction_angle_map"),
            "binary",
            0,
            1,
            "sd(extinction angle)",
        )
    if index == 13:
        return plot_float_map(
            stores,
            segmented_maps.get("R_90_map"),
            "plasma",
            0,
            None,
            "Retardation",
            mask=grain_mask,
        )
    if index == 14:
        return plot_float_map(
            stores,
            segmented_maps.get("R_80_map"),
            "plasma",
            0,
            None,
            "Retardation",
            mask=grain_mask,
        )
    if index == 16:
        return plot_float_map(
            stores,
            segmented_maps.get("extinction_angle_quality"),
            "gray",
            0,
            1,
            "quality of φ",
            mask=grain_mask,
        )
    if index == 17:
        return plot_float_map(
            stores,
            segmented_maps.get("angle_deg"),
            "hsv",
            0,
            180,
            "SPO",
            mask=grain_mask,
        )
    if index == 18:
        fig, ax = plot_grain_index(
            grain_list=grain_list,
            area_shape=raw_maps["extinction_angle"].shape,
            color="white",
            size=4,
        )
        apply_micrometer_axis(ax, stores)
        if stores.ui.display_grain_boundary.get():
            ax.imshow(
                stores.computation_result.grain_boundary.get(),
                alpha=0.5,
                cmap="gray",
            )
        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig, ax)
        return fig

    return get_no_image()
