from reactive_state import ReactiveElevatedButton
from stores import Stores
from components.selector.tools import (
    make_elevated_button,
    exist_in_raw_maps,
    exist_in_segmented_grain_map,
)


def make_cip_button_visible_state(
    stores: Stores,
    selected_index: int,
) -> tuple[
    # ReactiveElevatedButton,
    # ReactiveElevatedButton,
    # ReactiveElevatedButton,
    # ReactiveElevatedButton,
    # ReactiveElevatedButton,
    # ReactiveElevatedButton,
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
]:

    extinction_angle = make_elevated_button(
        stores,
        "extinction angle (Φ)",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        0,
        [stores.computation_result.raw_maps, stores.ui.analysis_tab.plot_option],
        lambda: stores.ui.analysis_tab.plot_option.get() == "CPO",
    )

    azimuth = make_elevated_button(
        stores,
        "azimuth (0°-180°)",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        1,
        [stores.computation_result.raw_maps, stores.ui.analysis_tab.plot_option],
        lambda: exist_in_raw_maps(stores, "azimuth")
        and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    )

    inclination = make_elevated_button(
        stores,
        "inclination",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        2,
        [stores.computation_result.raw_maps, stores.ui.analysis_tab.plot_option],
        lambda: exist_in_raw_maps(stores, "inclination")
        and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    )

    azimuth360 = make_elevated_button(
        stores,
        # "azimuth (0°-360°)",
        "inclination (0°-180°)",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        3,
        [stores.computation_result.raw_maps, stores.ui.analysis_tab.plot_option],
        lambda: exist_in_raw_maps(stores, "inclination_0_to_180")
        and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    )

    # seg_extinction_angle = make_elevated_button(
    #     stores,
    #     "extinction angle",
    #     stores.ui.selected_button_at_analysis_tab,
    #     selected_index,
    #     4,
    #     [
    #         stores.computation_result.grain_segmented_maps,
    #         stores.ui.analysis_tab.plot_option,
    #     ],
    #     lambda: exist_in_segmented_grain_map(stores, "extinction_angle")
    #     and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    # )

    # seg_azimuth = make_elevated_button(
    #     stores,
    #     "azimuth (0°-180°)",
    #     stores.ui.selected_button_at_analysis_tab,
    #     selected_index,
    #     5,
    #     [
    #         stores.computation_result.grain_segmented_maps,
    #         stores.ui.analysis_tab.plot_option,
    #     ],
    #     lambda: exist_in_segmented_grain_map(stores, "azimuth")
    #     and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    # )

    # seg_inclination = make_elevated_button(
    #     stores,
    #     "inclination",
    #     stores.ui.selected_button_at_analysis_tab,
    #     selected_index,
    #     6,
    #     [
    #         stores.computation_result.grain_segmented_maps,
    #         stores.ui.analysis_tab.plot_option,
    #     ],
    #     lambda: exist_in_segmented_grain_map(stores, "inclination")
    #     and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    # )

    # seg_azimuth360 = make_elevated_button(
    #     stores,
    #     "azimuth (0°-360°)",
    #     stores.ui.selected_button_at_analysis_tab,
    #     selected_index,
    #     7,
    #     [
    #         stores.computation_result.grain_segmented_maps,
    #         stores.ui.analysis_tab.plot_option,
    #     ],
    #     lambda: exist_in_segmented_grain_map(stores, "azimuth360")
    #     and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    # )

    coi90_map = make_elevated_button(
        stores,
        "φ=0°-90°",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        8,
        [
            stores.computation_result.cip_map_info,
            stores.ui.analysis_tab.plot_option,
        ],
        lambda: stores.computation_result.cip_map_info.get()["COI90"] is not None
        and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    )

    coi180_map = make_elevated_button(
        stores,
        "φ=0°-180°",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        9,
        [
            stores.computation_result.cip_map_info,
            stores.ui.analysis_tab.plot_option,
        ],
        lambda: stores.computation_result.cip_map_info.get()["COI180"] is not None
        and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    )

    coi360_map = make_elevated_button(
        stores,
        "φ=0°-360°",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        10,
        [
            stores.computation_result.cip_map_info,
            stores.ui.analysis_tab.plot_option,
        ],
        lambda: stores.computation_result.cip_map_info.get()["COI360"] is not None
        and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    )

    # coi90_grain = make_elevated_button(
    #     stores,
    #     "φ=0°-90°",
    #     stores.ui.selected_button_at_analysis_tab,
    #     selected_index,
    #     11,
    #     [
    #         stores.computation_result.cip_map_info,
    #         stores.ui.analysis_tab.plot_option,
    #     ],
    #     lambda: stores.computation_result.cip_map_info.get()["COI90_grain"] is not None
    #     and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    # )

    # coi180_grain = make_elevated_button(
    #     stores,
    #     "φ=0°-180°",
    #     stores.ui.selected_button_at_analysis_tab,
    #     selected_index,
    #     12,
    #     [
    #         stores.computation_result.cip_map_info,
    #         stores.ui.analysis_tab.plot_option,
    #     ],
    #     lambda: stores.computation_result.cip_map_info.get()["COI180_grain"] is not None
    #     and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    # )

    # coi360_grain = make_elevated_button(
    #     stores,
    #     "φ=0°-360°",
    #     stores.ui.selected_button_at_analysis_tab,
    #     selected_index,
    #     13,
    #     [
    #         stores.computation_result.cip_map_info,
    #         stores.ui.analysis_tab.plot_option,
    #     ],
    #     lambda: stores.computation_result.cip_map_info.get()["COI360_grain"] is not None
    #     and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    # )

    polar90 = make_elevated_button(
        stores,
        "φ=0°-90°",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        14,
        [
            stores.computation_result.cip_map_info,
            stores.ui.analysis_tab.plot_option,
        ],
        lambda: stores.computation_result.cip_map_info.get()["polar_info90"] is not None
        and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    )

    polar180 = make_elevated_button(
        stores,
        "φ=0°-180°",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        15,
        [
            stores.computation_result.cip_map_info,
            stores.ui.analysis_tab.plot_option,
        ],
        lambda: stores.computation_result.cip_map_info.get()["polar_info180"]
        is not None
        and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    )

    polar360 = make_elevated_button(
        stores,
        "φ=0°-360°",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        16,
        [
            stores.computation_result.cip_map_info,
            stores.ui.analysis_tab.plot_option,
        ],
        lambda: stores.computation_result.cip_map_info.get()["polar_info360"]
        is not None
        and (stores.ui.analysis_tab.plot_option.get() == "CPO"),
    )

    return (
        extinction_angle,
        azimuth,
        inclination,
        azimuth360,
        # seg_extinction_angle,
        # seg_azimuth,
        # seg_inclination,
        # seg_azimuth360,
        coi90_map,
        coi180_map,
        coi360_map,
        # coi90_grain,
        # coi180_grain,
        # coi360_grain,
        polar90,
        polar180,
        polar360,
    )
