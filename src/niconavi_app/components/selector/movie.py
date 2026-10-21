from niconavi_app.reactive_state import ReactiveElevatedButton
from niconavi_app.stores import Stores
from niconavi_app.components.selector.tools import make_elevated_button


def make_movie_selection_button_visible_state(
    stores: Stores,
) -> tuple[
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
]:

    xpl = make_elevated_button(
        stores,
        "xpl",
        stores.ui.selected_button_at_movie_tab,
        0,
        0,
        [stores.computation_result.video_path],
        help_text="First frame of the rotating XPL movie, with the selected white-balance correction.",
    )

    full_wave = make_elevated_button(
        stores,
        "xpl+λ",
        stores.ui.selected_button_at_movie_tab,
        0,
        1,
        [stores.computation_result.reta_video_path],
        help_text="First frame of the rotating XPL + λ-plate movie.",
    )

    tilt0 = make_elevated_button(
        stores,
        # "xpl+λ(0°;Tilt)",
        "xpl+λ(Tilt)",
        stores.ui.selected_button_at_movie_tab,
        0,
        2,
        [stores.computation_result.tilt_image_info.tilt_image0_path],
        help_text="First frame of the 0° tilted XPL + λ-plate movie.",
    )

    tilt45 = make_elevated_button(
        stores,
        "xpl+λ(45°;Tilt)",
        stores.ui.selected_button_at_movie_tab,
        0,
        3,
        [stores.computation_result.tilt_image_info.tilt_image45_path],
        help_text="First frame of the 45° tilted XPL + λ-plate movie.",
    )

    # horiz0 = make_elevated_button(
    #     stores,
    #     "xpl+λ(0°;Horiz.)",
    #     stores.ui.selected_button_at_movie_tab,
    #     0,
    #     4,
    #     [stores.computation_result.tilt_image_info.image0_path],
    # )

    horiz45 = make_elevated_button(
        stores,
        "xpl+λ(45°;Horiz)",
        stores.ui.selected_button_at_movie_tab,
        0,
        5,
        [stores.computation_result.tilt_image_info.image45_path],
        help_text="First frame of the horizontal 45° reference movie used for tilt alignment.",
    )

    # return xpl, full_wave, tilt0
    return xpl, full_wave, tilt0, tilt45, horiz45
