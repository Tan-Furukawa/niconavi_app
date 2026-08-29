from niconavi_app.stores import Stores
from niconavi_app.components.selector.tools import (
    make_elevated_button,
    make_check_prompt_button,
    exist_in_raw_maps,
    ReactiveElevatedButtonInSelector,
)


def make_grain_classification_button_visible_state(
    stores: Stores,
    selected_index: int,
) -> tuple[
    # ReactiveElevatedButtonInSelector,
    # ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
    ReactiveElevatedButtonInSelector,
]:

    degree_0 = make_elevated_button(
        stores,
        "φ=0°",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        0,
        [stores.computation_result.raw_maps],
    )

    degree_22_5 = make_elevated_button(
        stores,
        "φ=22.5°",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        1,
        [stores.computation_result.raw_maps],
    )

    degree_45 = make_elevated_button(
        stores,
        "φ=45°",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        2,
        [stores.computation_result.raw_maps],
    )

    degree_67_5 = make_elevated_button(
        stores,
        "φ=67.5°",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        3,
        [stores.computation_result.raw_maps],
    )

    extinction_angle = make_elevated_button(
        stores,
        "extinction angle (φex)",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        4,
        [stores.computation_result.raw_maps],
    )

    R_color = make_elevated_button(
        stores,
        "color",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        5,
        [stores.computation_result.raw_maps],
    )

    angle_map = make_elevated_button(
        stores,
        "Angle map",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        21,
        [stores.ui.map_tab.angle_map_display],
        lambda: stores.ui.map_tab.angle_map_display.get() is not None,
    )

    extinction_color = make_check_prompt_button(
        stores,
        "extinction color",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        6,
        [stores.computation_result.raw_maps],
    )

    grain_map = make_elevated_button(
        stores,
        "grains",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        7,
        [stores.computation_result.grain_map],
    )

    # grain_map_with_boundary = make_elevated_button(
    #     stores,
    #     "grains with boundary",
    #     stores.ui.selected_button_at_grain_tab,
    #     selected_index,
    #     8,
    #     [stores.computation_result.grain_map_with_boundary],
    # )

    # "R" (idx 9), "φex+45°" (idx 12) and "φex-45°" (idx 13) buttons were
    # retired (see move_v3.md); the additive image supersedes them. They are
    # not created at all - a created-but-unplaced reactive button would crash
    # on selected_index change (update() on a control not added to the page).

    p45_R_color_map = make_elevated_button(
        stores,
        "color at φex+45°",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        10,
        [stores.computation_result.raw_maps],
        lambda: exist_in_raw_maps(stores, "p45_R_color_map"),
    )

    m45_R_color_map = make_elevated_button(
        stores,
        "color at φex-45°",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        11,
        [stores.computation_result.raw_maps],
        lambda: exist_in_raw_maps(stores, "m45_R_color_map"),
    )

    azimuth = make_elevated_button(
        stores,
        "azimuth",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        14,
        [stores.computation_result.raw_maps],
        lambda: exist_in_raw_maps(stores, "azimuth"),
    )

    tilt0 = make_elevated_button(
        stores,
        "φ=0° (tilt)",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        15,
        [stores.computation_result.tilt_image_info.tilt_image0],
        lambda: stores.computation_result.tilt_image_info.tilt_image0.get() is not None,
    )

    tilt45 = make_elevated_button(
        stores,
        "φ=45° (tilt)",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        16,
        [stores.computation_result.tilt_image_info.tilt_image45],
        lambda: stores.computation_result.tilt_image_info.tilt_image45.get()
        is not None,
    )

    horiz0 = make_elevated_button(
        stores,
        "φ=0° (horiz.)",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        17,
        [stores.computation_result.tilt_image_info.tilt_image0],
        lambda: stores.computation_result.tilt_image_info.tilt_image0.get() is not None,
    )

    horiz45 = make_elevated_button(
        stores,
        "φ=45° (horiz.)",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        18,
        [stores.computation_result.tilt_image_info.tilt_image45],
        lambda: stores.computation_result.tilt_image_info.tilt_image45.get()
        is not None,
    )

    # quality = make_elevated_button(
    #     stores,
    #     "quality of φ",
    #     stores.ui.selected_button_at_grain_tab,
    #     selected_index,
    #     19,
    #     [stores.computation_result.raw_maps],
    #     lambda: exist_in_raw_maps(stores, "cv_extinction_angle"),
    # )

    mask = make_elevated_button(
        stores,
        "mask",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        20,
        [stores.computation_result.mask],
        lambda: stores.computation_result.mask.get() is not None,
    )

    add_image = make_elevated_button(
        stores,
        "additive image",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        22,
        [stores.computation_result.raw_maps],
        lambda: exist_in_raw_maps(stores, "add_image"),
    )

    color_change0 = make_check_prompt_button(
        stores,
        "original color change (0°)",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        23,
        [stores.computation_result.tilt_image_info.tilt_image0],
        lambda: stores.computation_result.tilt_image_info.tilt_image0.get() is not None,
    )

    color_change45 = make_check_prompt_button(
        stores,
        "original color change (45°)",
        stores.ui.selected_button_at_grain_tab,
        selected_index,
        24,
        [stores.computation_result.tilt_image_info.tilt_image45],
        lambda: stores.computation_result.tilt_image_info.tilt_image45.get()
        is not None,
    )


    return (
        degree_0,
        degree_22_5,
        degree_45,
        degree_67_5,
        angle_map,
        R_color,
        extinction_color,
        extinction_angle,
        grain_map,
        # grain_map_with_boundary,
        p45_R_color_map,
        m45_R_color_map,
        azimuth,
        tilt0,
        tilt45,
        horiz0,
        horiz45,
        # quality,
        mask,
        add_image,
        color_change0,
        color_change45,
        # delta_R_tilt_0,
        # delta_R_tilt_45,
    )
