from reactive_state import ReactiveElevatedButton
from stores import Stores
from components.selector.tools import (
    make_elevated_button,
    exist_in_segmented_maps,
    exist_in_segmented_grain_map,
)


def make_spo_button_visible_state(
    stores: Stores,
    selected_index: int,
) -> tuple[
    ReactiveElevatedButton,
    ReactiveElevatedButton,
    ReactiveElevatedButton,
]:

    spo = make_elevated_button(
        stores,
        "SPO",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        0,
        [
            stores.computation_result.grain_segmented_maps,
            stores.ui.analysis_tab.plot_option,
        ],
        # lambda: exist_in_segmented_maps(stores, "angle_deg")
        # and stores.ui.analysis_tab.plot_option.get() == "SPO",
        lambda: stores.ui.analysis_tab.plot_option.get() == "SPO",
    )

    # rose_cpo = make_elevated_button(
    #     stores,
    #     "SPO (rose diagram)",
    #     stores.ui.selected_button_at_analysis_tab,
    #     selected_index,
    #     1,
    #     [stores.computation_result.grain_list, stores.ui.analysis_tab.plot_option],
    #     # lambda: stores.computation_result.grain_list.get() is not None
    #     lambda: stores.ui.analysis_tab.plot_option.get() == "SPO",
    # )

    ellipse = make_elevated_button(
        stores,
        "ellipse (major axis)",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        2,
        [stores.computation_result.grain_list, stores.ui.analysis_tab.plot_option],
        # lambda: stores.computation_result.grain_list.get() is not None
        lambda: stores.ui.analysis_tab.plot_option.get() == "SPO",
    )

    major_axis = make_elevated_button(
        stores,
        "ellipse",
        stores.ui.selected_button_at_analysis_tab,
        selected_index,
        3,
        [stores.computation_result.grain_list, stores.ui.analysis_tab.plot_option],
        # lambda: stores.computation_result.grain_list.get() is not None
        # and stores.ui.analysis_tab.plot_option.get() == "SPO",
        lambda: stores.ui.analysis_tab.plot_option.get() == "SPO",
    )

    return (
        spo,
        # rose_cpo,
        ellipse,
        major_axis,
    )
