from reactive_state import ReactiveElevatedButton
from stores import Stores
from state import State
from components.selector.tools import (
    make_elevated_button,
    exist_in_segmented_maps,
)


def make_filter_button_visible_state(stores: Stores, selected_index: int) -> tuple[
    # ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
]:

    classification = make_elevated_button(
        stores,
        "result",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        0,
        [stores.computation_result.grain_segmented_maps],
    )

    retardation = make_elevated_button(
        stores,
        "R (median)",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        1,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "max_retardation_map"),
    )

    H = make_elevated_button(
        stores,
        "color (H)",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        2,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "H"),
    )

    S = make_elevated_button(
        stores,
        "color (S)",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        3,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "S"),
    )

    V = make_elevated_button(
        stores,
        "color (V)",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        4,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "V"),
    )

    R_color = make_elevated_button(
        stores,
        "color",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        5,
        [
            stores.computation_result.grain_segmented_maps,
            stores.computation_result.raw_maps,
        ],
    )

    eccentricity = make_elevated_button(
        stores,
        "eccentricity",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        6,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "eccentricity"),
    )

    R70 = make_elevated_button(
        stores,
        "R (70 %til.)",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        7,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "R_70_map"),
    )

    size = make_elevated_button(
        stores,
        "grain size",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        8,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "size"),
    )

    ex_angle = make_elevated_button(
        stores,
        "extinction angle",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        9,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "extinction_angle"),
    )

    azimuth = make_elevated_button(
        stores,
        "azimuth",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        10,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "azimuth"),
    )

    sd_azimuth = make_elevated_button(
        stores,
        "sd(azimuth)",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        11,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "sd_azimuth"),
    )

    sd_ex_angle = make_elevated_button(
        stores,
        "sd(extinction angle)",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        12,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "sd_extinction_angle_map"),
    )

    R90 = make_elevated_button(
        stores,
        "R (90 %til.)",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        13,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "R_90_map"),
    )

    R80 = make_elevated_button(
        stores,
        "R (80 %til.)",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        14,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "R_80_map"),
    )

    # R_color_raw = make_elevated_button(
    #     stores,
    #     "color",
    #     stores.ui.selected_button_at_filter_tab,
    #     selected_index,
    #     15,
    #     [stores.computation_result.raw_maps],
    # )

    # quality = make_elevated_button(
    #     stores,
    #     "quality of φ",
    #     stores.ui.selected_button_at_filter_tab,
    #     selected_index,
    #     16,
    #     [stores.computation_result.grain_segmented_maps],
    #     lambda: exist_in_segmented_maps(stores, "extinction_angle_quality"),
    # )

    spo = make_elevated_button(
        stores,
        "SPO",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        17,
        [stores.computation_result.grain_segmented_maps],
        lambda: exist_in_segmented_maps(stores, "angle_deg"),
    )

    index = make_elevated_button(
        stores,
        "index",
        stores.ui.selected_button_at_filter_tab,
        selected_index,
        18,
        [stores.computation_result.grain_list],
        lambda: stores.computation_result.grain_list.get() is not None,
    )

    return (
        classification,
        retardation,
        H,
        S,
        V,
        R_color,
        eccentricity,
        R70,
        size,
        ex_angle,
        azimuth,
        sd_azimuth,
        sd_ex_angle,
        R90,
        R80,
        # quality,
        spo,
        index,
        # R_color_raw,
    )
