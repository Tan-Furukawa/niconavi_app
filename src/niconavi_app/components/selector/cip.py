from niconavi_app.reactive_state import ReactiveElevatedButton
from niconavi_app.stores import Stores
from niconavi_app.components.selector.tools import (
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
        help_text="Circular median extinction angle φex calculated from the rotating XPL movie (0°–90°).",
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
        help_text="Uses the λ-plate retardations: if R(φex+45°) ≥ R(φex−45°), azimuth = 180°−φex; otherwise azimuth = 90°−φex.",
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
        help_text="Inclination magnitude estimated from the addition image, section thickness, and refractive indices (0°–90°).",
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
        help_text="Resolves inclination as θ or 180°−θ from the colour change in the registered tilted images.",
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
        help_text="Colour orientation map combining pixel inclination with extinction angle φex (0°–90° symmetry).",
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
        help_text="Colour orientation map combining pixel inclination with λ-plate-derived azimuth (0°–180°).",
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
        help_text="Colour orientation map using the tilt-resolved 0°–360° axis direction.",
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
        help_text="Polar distribution of classified-grain inclination and extinction angle φex (0°–90° symmetry).",
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
        help_text="Polar distribution of classified-grain inclination and λ-plate-derived azimuth (0°–180°).",
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
        help_text="Polar distribution of classified-grain orientations after resolving the 0°–360° axis direction from tilted images.",
    )

    # Grain-center RGB regression figures, produced by "calculate": one button
    # per image. Shown only while CPO is selected and the figure exists (the
    # figure being non-None is enforced by make_elevated_button's not-None
    # condition list).
    rgb_before = make_elevated_button(
        stores,
        "RGB before correction",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        17,
        [
            stores.ui.analysis_tab.cip_regression_before_figure,
            stores.ui.analysis_tab.plot_option,
        ],
        lambda: stores.ui.analysis_tab.plot_option.get() == "CPO",
        help_text="Regression of predicted versus observed grain-centre RGB values before colour correction.",
    )

    rgb_after = make_elevated_button(
        stores,
        "RGB after correction",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        18,
        [
            stores.ui.analysis_tab.cip_regression_after_figure,
            stores.ui.analysis_tab.plot_option,
        ],
        lambda: stores.ui.analysis_tab.plot_option.get() == "CPO",
        help_text="Regression of colour-corrected predicted versus observed grain-centre RGB values.",
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
        rgb_before,
        rgb_after,
    )
