from typing import Optional

from stores import Stores, as_ComputationResult
from matplotlib.pyplot import Figure, Axes
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from niconavi.image.image import apply_color_to_mask, create_outside_circle_mask
from niconavi.analysis import (
    make_grain_mask,
    scatter_for_all_minerals,
    histogram_for_all_minerals,
    rose_diagram_for_all_minerals,
    filter_displayed_grain_classification_result,
    get_mineral_list_from_grain_list,
)
import matplotlib
import numpy as np
from niconavi.type import ComputationResult

# from niconavi.optics.polar_plot import plot_as_stereo_projection
from match_used_name import (
    to_grain_display,
)
from copy import deepcopy
from tools.no_image import get_no_image

from components.view.style import set_default_figure_style
from components.view.spatial_units import (
    apply_micrometer_axis,
    convert_grain_units_for_targets,
    format_quantity_label,
)
from components.view.plot import plot_polar_map, plot_float_map, plot_polar_distribution
from niconavi.statistics.array_to_float import circular_variance


HISTOGRAM_STATS_DEFAULT = "Mean: -\nStd Dev: -\nMin: -\nMax: -\n95th percentile: -\nMode: -\nCount: -\nIntegral ratio: -\nCount ratio: -"
ROSE_STATS_DEFAULT = "Mean orientation: -\nCircular variance: -"


def _format_metric_line(
    title: str,
    bulk_value: Optional[float],
    mineral_values: dict[str, Optional[float]],
    fmt: str = "{:.3f}",
    suffix: str = "",
) -> str:
    parts: list[str] = []
    if bulk_value is None:
        parts.append("bulk: -")
    else:
        parts.append(f"bulk: {fmt.format(bulk_value)}{suffix}")
    for mineral in sorted(mineral_values.keys()):
        value = mineral_values[mineral]
        if value is None:
            parts.append(f"{mineral}: -")
        else:
            parts.append(f"{mineral}: {fmt.format(value)}{suffix}")
    return f"{title}: " + ", ".join(parts)


def _format_ratio_line(title: str, mineral_values: dict[str, float]) -> str:
    if not mineral_values:
        return f"{title}: -"
    total = sum(mineral_values.values())
    if total <= 0:
        return f"{title}: -"
    parts = []
    for mineral in sorted(mineral_values.keys()):
        ratio = 100.0 * mineral_values[mineral] / total
        parts.append(f"{mineral}: {ratio:.2f}%")
    return f"{title}: " + ", ".join(parts)


def _update_histogram_stats(
    stores: Stores, target: str, method: str, params: ComputationResult
) -> None:
    # if method != "grain":
    #     stores.ui.analysis_tab.histogram_stats_text.set(HISTOGRAM_STATS_DEFAULT)
    #     return

    classification = params.grain_classification_result
    if classification is None:
        stores.ui.analysis_tab.histogram_stats_text.set(HISTOGRAM_STATS_DEFAULT)
        return

    filtered = filter_displayed_grain_classification_result(classification)
    if not filtered:
        stores.ui.analysis_tab.histogram_stats_text.set(HISTOGRAM_STATS_DEFAULT)
        return

    all_values: list[np.ndarray] = []
    integral_values: dict[str, float] = {}
    count_values: dict[str, float] = {}
    mean_values: dict[str, Optional[float]] = {}
    std_values: dict[str, Optional[float]] = {}
    min_values: dict[str, Optional[float]] = {}
    max_values: dict[str, Optional[float]] = {}
    percentile_95_values: dict[str, Optional[float]] = {}
    mode_values: dict[str, Optional[float]] = {}
    count_numbers: dict[str, float] = {}

    for mineral in filtered.keys():
        values = [
            v
            for v in get_mineral_list_from_grain_list(params, target, mineral)
            if v is not None
        ]
        if not values:
            integral_values[mineral] = 0.0
            count_values[mineral] = 0.0
            mean_values[mineral] = None
            std_values[mineral] = None
            min_values[mineral] = None
            max_values[mineral] = None
            percentile_95_values[mineral] = None
            mode_values[mineral] = None
            count_numbers[mineral] = 0.0
            continue
        arr = np.asarray(values, dtype=float)
        all_values.append(arr)
        integral_values[mineral] = float(arr.sum())
        count_values[mineral] = float(len(arr))
        mean_values[mineral] = float(np.mean(arr))
        std_values[mineral] = float(np.std(arr))
        min_values[mineral] = float(np.min(arr))
        max_values[mineral] = float(np.max(arr))
        percentile_95_values[mineral] = float(np.percentile(arr, 95))
        unique, counts = np.unique(arr, return_counts=True)
        mode_values[mineral] = float(unique[np.argmax(counts)])
        count_numbers[mineral] = float(len(arr))

    if not all_values:
        stores.ui.analysis_tab.histogram_stats_text.set(HISTOGRAM_STATS_DEFAULT)
        return

    concatenated = np.concatenate(all_values)
    mean_val = float(np.mean(concatenated))
    std_val = float(np.std(concatenated))
    min_val = float(np.min(concatenated))
    max_val = float(np.max(concatenated))
    unique_all, counts_all = np.unique(concatenated, return_counts=True)
    percentile_95_val = float(np.percentile(concatenated, 95))
    mode_val = float(unique_all[np.argmax(counts_all)])
    total_count = float(len(concatenated))

    integral_text = _format_ratio_line("Integral ratio", integral_values)
    count_ratio_text = _format_ratio_line("Count ratio", count_values)

    mean_line = _format_metric_line("Mean", mean_val, mean_values)
    std_line = _format_metric_line("Std Dev", std_val, std_values)
    min_line = _format_metric_line("Min", min_val, min_values)
    max_line = _format_metric_line("Max", max_val, max_values)
    percentile_95_line = _format_metric_line(
        "95th percentile", percentile_95_val, percentile_95_values
    )
    mode_line = _format_metric_line("Mode", mode_val, mode_values)
    count_line = _format_metric_line("Count", total_count, count_numbers, fmt="{:.0f}")

    stats_text = (
        f"{mean_line}\n"
        f"{std_line}\n"
        f"{min_line}\n"
        f"{max_line}\n"
        f"{percentile_95_line}\n"
        f"{mode_line}\n"
        f"{count_line}\n"
        f"{integral_text}\n"
        f"{count_ratio_text}"
    )
    stores.ui.analysis_tab.histogram_stats_text.set(stats_text)


def _circular_mean(values: np.ndarray, cycle: float) -> Optional[float]:
    if values.size == 0:
        return None
    angles_rad = 2.0 * np.pi * (values / cycle)
    C = np.mean(np.cos(angles_rad))
    S = np.mean(np.sin(angles_rad))
    if np.isclose(C, 0.0) and np.isclose(S, 0.0):
        return None
    mean_angle = np.arctan2(S, C)
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return float((mean_angle / (2 * np.pi)) * cycle)


def _circular_variance(values: np.ndarray, cycle: float) -> Optional[float]:
    if values.size == 0:
        return None
    if np.isclose(cycle, 360.0):
        return float(circular_variance(values.astype(float), None))
    return float(circular_variance(values.astype(float), cycle))


def _update_rose_stats(
    stores: Stores,
    target: str,
    method: str,
) -> None:
    if method != "grain":
        stores.ui.analysis_tab.rose_stats_text.set(ROSE_STATS_DEFAULT)
        return

    params = as_ComputationResult(stores.computation_result)
    classification = params.grain_classification_result
    if classification is None:
        stores.ui.analysis_tab.rose_stats_text.set(ROSE_STATS_DEFAULT)
        return

    filtered = filter_displayed_grain_classification_result(classification)
    if not filtered:
        stores.ui.analysis_tab.rose_stats_text.set(ROSE_STATS_DEFAULT)
        return

    cycle = 90.0 if target == "extinction_angle" else 180.0

    mean_values: dict[str, Optional[float]] = {}
    variance_values: dict[str, Optional[float]] = {}
    all_values: list[float] = []

    for mineral in filtered.keys():
        raw_values = [
            v
            for v in get_mineral_list_from_grain_list(params, target, mineral)
            if v is not None
        ]
        if not raw_values:
            mean_values[mineral] = None
            variance_values[mineral] = None
            continue
        arr = np.asarray(raw_values, dtype=float)
        all_values.extend(arr.tolist())
        mean_values[mineral] = _circular_mean(arr, cycle)
        variance_values[mineral] = _circular_variance(arr, cycle)

    if not all_values:
        stores.ui.analysis_tab.rose_stats_text.set(ROSE_STATS_DEFAULT)
        return

    bulk_arr = np.asarray(all_values, dtype=float)
    bulk_mean = _circular_mean(bulk_arr, cycle)
    bulk_var = _circular_variance(bulk_arr, cycle)

    mean_line = _format_metric_line(
        "Mean orientation", bulk_mean, mean_values, fmt="{:.3f}°"
    )
    variance_line = _format_metric_line("Circular variance", bulk_var, variance_values)

    stores.ui.analysis_tab.rose_stats_text.set(f"{mean_line}\n{variance_line}")


# matplotlib.use("Agg")  # 非対話型バックエンドに切り替え
# matplotlib.use("svg")  # 非対話型バックエンドに切り替え


def at_analysis_tab(stores: Stores) -> Figure:
    stores.ui.analysis_tab.histogram_stats_text.set(HISTOGRAM_STATS_DEFAULT)
    stores.ui.analysis_tab.rose_stats_text.set(ROSE_STATS_DEFAULT)
    params = as_ComputationResult(stores.computation_result)
    if stores.ui.analysis_tab.plot_option.get() == "rose diagram":
        print("-- computation_unit is grain")
        t1 = stores.ui.analysis_tab.grain_rose_diagram_target.get()
        fig, ax = rose_diagram_for_all_minerals(
            params,
            t1,
            t1,  # raw_mapsには、shape_orientaion(angle_deg)がないので、これを選択しても表示されない
            stores.ui.analysis_tab.computation_unit.get(),
            alpha=stores.ui.analysis_tab.rose_alpha.get(),
            flip=stores.ui.analysis_tab.rose_flip.get(),
        )
        _update_rose_stats(
            stores,
            t1,
            stores.ui.analysis_tab.computation_unit.get(),
        )
        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig, method="rose diagram")
        return fig

    elif stores.ui.analysis_tab.plot_option.get() == "histogram":
        t1 = stores.ui.analysis_tab.grain_histogram_target.get()
        # hist_bins = max(1, stores.ui.analysis_tab.histogram_bins.get() or 1)

        params_scaled, unit_map = convert_grain_units_for_targets(
            stores, params, {t1}
        )
        fig, ax = histogram_for_all_minerals(
            params_scaled,
            t1,
            stores.ui.analysis_tab.computation_unit.get(),
            log_x=stores.ui.analysis_tab.histogram_log_x.get(),
            alpha=stores.ui.analysis_tab.histogram_alpha.get(),
        )

        _update_histogram_stats(
            stores,
            t1,
            stores.ui.analysis_tab.computation_unit.get(),
            params_scaled,
        )

        label = format_quantity_label(
            to_grain_display(t1), unit_map.get(t1)
        )
        if label is not None:
            ax.set_xlabel(label)

        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig, method="histogram")
        return fig

    elif stores.ui.analysis_tab.plot_option.get() == "scatter":
        # if stores.ui.analysis_tab.computation_unit == "grain":
        t1 = stores.ui.analysis_tab.scatter_target_x.get()
        t2 = stores.ui.analysis_tab.scatter_target_y.get()
        params_scaled, unit_map = convert_grain_units_for_targets(
            stores, params, {t1, t2}
        )
        x_label = format_quantity_label(to_grain_display(t1), unit_map.get(t1))
        y_label = format_quantity_label(to_grain_display(t2), unit_map.get(t2))
        fig, ax = scatter_for_all_minerals(
            params_scaled,
            t1,
            t2,
            xlab=x_label,
            ylab=y_label,
            text_color="white",
            origin=stores.ui.analysis_tab.scatter_regression_origin.get(),
            display_regression=stores.ui.analysis_tab.scatter_show_regression.get(),
            log_x=stores.ui.analysis_tab.scatter_log_x.get(),
            log_y=stores.ui.analysis_tab.scatter_log_y.get(),
        )

        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig, method="scatter")
        return fig

    elif stores.ui.analysis_tab.plot_option.get() == "SPO":
        raw_map = stores.computation_result.raw_maps.get()
        grain_map = stores.computation_result.grain_map.get()
        grain_classification_result = (
            stores.computation_result.grain_classification_result.get()
        )
        if grain_map is not None and grain_classification_result is not None:
            grain_mask = make_grain_mask(grain_classification_result, grain_map)
        else:
            return get_no_image()
        segmented_map = stores.computation_result.grain_segmented_maps.get()
        grain_list = stores.computation_result.grain_list.get()

        if segmented_map is not None:
            if stores.ui.selected_button_at_analysis_tab.get() == 0:
                return plot_float_map(
                    stores,
                    segmented_map["angle_deg"],
                    "hsv",
                    0,
                    180,
                    "SPO",
                    mask=grain_mask,
                )
            elif stores.ui.selected_button_at_analysis_tab.get() == 1:
                fig, ax = rose_diagram_for_all_minerals(
                    params,
                    "angle_deg",
                    "grain",
                    alpha=stores.ui.analysis_tab.rose_alpha.get(),
                    flip=stores.ui.analysis_tab.rose_flip.get(),
                )
                stores.ui.displayed_fig.set(deepcopy(fig))
                set_default_figure_style(fig, method="rose diagram")
                return fig
            elif stores.ui.selected_button_at_analysis_tab.get() == 2:
                classification_result = (
                    stores.computation_result.grain_classification_result.get()
                )
                if not grain_list or classification_result is None:
                    return get_no_image()

                filtered = filter_displayed_grain_classification_result(
                    classification_result
                )
                if len(filtered) == 0:
                    return get_no_image()

                required_keys = ["ellipse_center", "angle_deg", "major_axis_length"]
                plot_area = raw_map["extinction_angle"].shape
                height, width = plot_area
                fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
                plotted = False

                for mineral, info in filtered.items():
                    color = info["color"]
                    for grain in grain_list:
                        if grain.get("mineral") != mineral:
                            continue
                        if not all(grain.get(key) is not None for key in required_keys):
                            continue
                        center_x, center_y = grain["ellipse_center"]
                        angle_deg = 180 - grain["angle_deg"]
                        major_axis_length = grain["major_axis_length"]

                        half = major_axis_length / 2.0
                        angle_rad = np.deg2rad(angle_deg)
                        dx = half * np.cos(angle_rad)
                        dy = half * np.sin(angle_rad)

                        x1, y1 = center_x - dx, center_y - dy
                        x2, y2 = center_x + dx, center_y + dy
                        ax.plot([x1, x2], [y1, y2], color=color, lw=1)
                        plotted = True

                if not plotted:
                    plt.close(fig)
                    return get_no_image()

                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_aspect("equal")
                apply_micrometer_axis(ax, stores)
                stores.ui.displayed_fig.set(deepcopy(fig))
                set_default_figure_style(fig, method="scatter")
                return fig
            elif stores.ui.selected_button_at_analysis_tab.get() == 3:
                classification_result = (
                    stores.computation_result.grain_classification_result.get()
                )
                if not grain_list or classification_result is None:
                    return get_no_image()

                filtered = filter_displayed_grain_classification_result(
                    classification_result
                )
                if len(filtered) == 0:
                    return get_no_image()

                required_keys = [
                    "ellipse_center",
                    "angle_deg",
                    "major_axis_length",
                    "minor_axis_length",
                ]
                plot_area = raw_map["extinction_angle"].shape
                height, width = plot_area
                fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
                plotted = False

                for mineral, info in filtered.items():
                    color = info["color"]
                    for grain in grain_list:
                        if grain.get("mineral") != mineral:
                            continue
                        if not all(grain.get(key) is not None for key in required_keys):
                            continue
                        center = grain["ellipse_center"]
                        angle_deg = 180 - grain["angle_deg"]
                        major_axis_length = grain["major_axis_length"]
                        minor_axis_length = grain["minor_axis_length"]

                        ellipse = Ellipse(
                            xy=center,
                            width=major_axis_length,
                            height=minor_axis_length,
                            angle=angle_deg,
                            edgecolor=color,
                            facecolor="none",
                            lw=1,
                        )
                        ax.add_patch(ellipse)
                        plotted = True

                if not plotted:
                    plt.close(fig)
                    return get_no_image()

                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_aspect("equal")
                apply_micrometer_axis(ax, stores)
                stores.ui.displayed_fig.set(deepcopy(fig))
                set_default_figure_style(fig, method="scatter")
                return fig
        else:
            return get_no_image()

        return get_no_image()

    elif stores.ui.analysis_tab.plot_option.get() == "CPO":
        raw_map = stores.computation_result.raw_maps.get()
        grain_map = stores.computation_result.grain_map.get()
        grain_classification_result = (
            stores.computation_result.grain_classification_result.get()
        )
        if grain_map is not None and grain_classification_result is not None:
            grain_mask = make_grain_mask(grain_classification_result, grain_map)
        else:
            return get_no_image()

        if raw_map is not None:

            if (
                stores.computation_result.tilt_image_info.tilt_image0.get() is not None
                and stores.computation_result.tilt_image_info.tilt_image0.get()
                is not None
            ):
                mask0 = stores.computation_result.tilt_image_info.tilt_image0.get()
                mask45 = stores.computation_result.tilt_image_info.tilt_image45.get()

                if mask0 is not None and mask45 is not None:
                    mask = ~(mask0["image_mask"] & mask45["image_mask"])
                elif mask0 is not None and mask45 is None:
                    mask = ~(mask0["image_mask"])
                    mask = ~create_outside_circle_mask(raw_map["extinction_angle"])
                else:
                    mask = ~create_outside_circle_mask(raw_map["extinction_angle"])
            else:
                mask = ~create_outside_circle_mask(raw_map["extinction_angle"])

            if stores.ui.selected_button_at_analysis_tab.get() == 0:
                return plot_float_map(
                    stores,
                    raw_map["extinction_angle"],
                    "hsv",
                    0,
                    90,
                    "extinction angle",
                    mask=grain_mask,
                )
            elif stores.ui.selected_button_at_analysis_tab.get() == 1:
                return plot_float_map(
                    stores,
                    raw_map["azimuth"],
                    "hsv",
                    0,
                    180,
                    "azimuth",
                    mask=grain_mask,
                )
            elif stores.ui.selected_button_at_analysis_tab.get() == 2:
                return plot_float_map(
                    stores,
                    raw_map["inclination"],
                    "viridis",
                    0,
                    90,
                    "inclination",
                    mask=grain_mask,
                )
            elif stores.ui.selected_button_at_analysis_tab.get() == 3:
                return plot_float_map(
                    stores,
                    raw_map["inclination_0_to_180"],
                    "hsv",
                    0,
                    180,
                    "inclination",
                    mask=mask | grain_mask,
                )
            else:
                segmented_map = stores.computation_result.grain_segmented_maps.get()
                if segmented_map is not None:
                    if stores.ui.selected_button_at_analysis_tab.get() == 4:
                        return plot_float_map(
                            stores,
                            segmented_map["extinction_angle"],
                            "hsv",
                            0,
                            90,
                            "extinction angle",
                            mask=grain_mask,
                        )
                    elif stores.ui.selected_button_at_analysis_tab.get() == 5:
                        return plot_float_map(
                            stores,
                            segmented_map["azimuth"],
                            "hsv",
                            0,
                            180,
                            "azimuth",
                            mask=grain_mask,
                        )
                    elif stores.ui.selected_button_at_analysis_tab.get() == 6:
                        return plot_float_map(
                            stores,
                            segmented_map["inclination"],
                            "viridis",
                            0,
                            90,
                            "inclination",
                            mask=grain_mask,
                        )
                    elif stores.ui.selected_button_at_analysis_tab.get() == 7:
                        # print("-------------------------")
                        # print(segmented_map["azimuth360"][300,300])
                        # print(segmented_map["azimuth360"][200,200])
                        # print("-------------------------")
                        return plot_float_map(
                            stores,
                            segmented_map["azimuth360"],
                            "hsv",
                            0,
                            360,
                            "azimuth",
                            mask=mask | grain_mask,
                        )
                    else:
                        cip_map_info = stores.computation_result.cip_map_info.get()

                        if cip_map_info is not None:
                            if stores.ui.selected_button_at_analysis_tab.get() == 8:
                                return plot_polar_map(
                                    stores,
                                    cip_map_info["COI90"],
                                    cip_map_info["legend90"],
                                    mask=grain_mask,
                                )
                            elif stores.ui.selected_button_at_analysis_tab.get() == 9:
                                return plot_polar_map(
                                    stores,
                                    cip_map_info["COI180"],
                                    cip_map_info["legend180"],
                                    mask=grain_mask,
                                )
                            elif stores.ui.selected_button_at_analysis_tab.get() == 10:
                                # print(cip_map_info["COI360"])
                                return plot_polar_map(
                                    stores,
                                    cip_map_info["COI360"],
                                    cip_map_info["legend360"],
                                    mask=mask | grain_mask,
                                )
                            elif stores.ui.selected_button_at_analysis_tab.get() == 11:
                                return plot_polar_map(
                                    stores,
                                    cip_map_info["COI90_grain"],
                                    cip_map_info["legend90"],
                                    mask=grain_mask,
                                )
                            elif stores.ui.selected_button_at_analysis_tab.get() == 12:
                                return plot_polar_map(
                                    stores,
                                    cip_map_info["COI180_grain"],
                                    cip_map_info["legend180"],
                                    mask=grain_mask,
                                )
                            elif stores.ui.selected_button_at_analysis_tab.get() == 13:
                                return plot_polar_map(
                                    stores,
                                    cip_map_info["COI360_grain"],
                                    cip_map_info["legend360"],
                                    mask=mask | grain_mask,
                                )
                            elif stores.ui.selected_button_at_analysis_tab.get() == 14:
                                return plot_polar_distribution(
                                    stores,
                                    cip_map_info["polar_info90"],
                                    (0, 90),
                                    mask=grain_mask,
                                    pixel_azimuth_map=raw_map[
                                        "extinction_angle"
                                    ],  #! extinction angleだけ、正の向きがちがう
                                    pixel_inclination_map=raw_map["inclination"],
                                )
                            elif stores.ui.selected_button_at_analysis_tab.get() == 15:
                                return plot_polar_distribution(
                                    stores,
                                    cip_map_info["polar_info180"],
                                    (0, 180),
                                    mask=grain_mask,
                                    pixel_azimuth_map=raw_map["azimuth"],
                                    pixel_inclination_map=raw_map["inclination"],
                                )
                            elif stores.ui.selected_button_at_analysis_tab.get() == 16:
                                return plot_polar_distribution(
                                    stores,
                                    cip_map_info["polar_info360"],
                                    (0, 360),
                                    mask=grain_mask,
                                    pixel_azimuth_map=raw_map["azimuth"],
                                    pixel_inclination_map=raw_map[
                                        "inclination_0_to_180"
                                    ],
                                )
                            else:
                                return get_no_image()
                        else:
                            return get_no_image()
                else:
                    return get_no_image()
        else:
            return get_no_image(stores)

    else:
        return get_no_image()
