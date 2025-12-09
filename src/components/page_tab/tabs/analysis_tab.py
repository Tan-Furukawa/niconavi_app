from stores import (
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
)
from reactive_state import (
    ReactiveRow,
    ReactiveColumn,
    ReactiveSlider,
)
from state import ReactiveState
from components.log_view import update_logs
from components.progress_bar import update_progress_bar
from components.common_component import (
    CustomReactiveCheckbox,
    CustomRadio,
    CustomReactiveText,
    CustomExecuteButton,
    make_ADD_counter_button,
    make_REMOVE_counter_button,
    make_reactive_float_text_filed,
    ReactiveCustomDropDown,
    CustomText,
    ReactiveFloatTextField,
)
from tools.tools import switch_tab_index, force_update_image_view
from components.log_view import update_logs
from match_used_name import (
    to_grain_display,
    to_raw_map_display,
    to_rose_display,
    inv_grain_display,
    inv_raw_maps_display,
    inv_rose_display,
    RoseDiagramUsedInPlot,
    RoseDiagramUsedInPixel,
    GrainNumListUsedInPlot,
    RawMapsNumListUsedInPlot,
)
from components.view.spatial_units import format_quantity_label, get_grain_unit_label
from typing import Callable, Optional
import flet as ft
from flet import Page
from logging import getLogger, Logger
import niconavi.run_all as po
import traceback
from niconavi.tools.str_parser import (
    parse_larger_than_0,
    parse_int,
)
from tools.validation import validation_larger_than_0_float
from components.page_tab.tabs.reset_onclick import reset_onclick_cip_computation_button


def onclick_cip_start_button(
    stores: Stores, e: ft.FilePickerResultEvent, *, logger: Logger
) -> None:
    if (
        stores.computation_result.optical_parameters.max_R.get() is None
        and stores.computation_result.tilt_image_info.estimate_inclination_by.get()
        == "max R"
    ):
        update_logs(stores, ("Please provide the max R value.", "err"))
        return None

    try:
        update_progress_bar(None, stores)
        r = as_ComputationResult(stores.computation_result)
        r = reset_onclick_cip_computation_button(r)
        save_in_ComputationResultState(r, stores)

        r = po.get_inclination(
            r, progress_callback=lambda p: update_progress_bar(p, stores)
        )
        update_progress_bar(None, stores)
        r = po.analyze_grain_list_for_CIP(r)
        r = po.make_CIP_map_info(r)
        update_progress_bar(0, stores)
        save_in_ComputationResultState(r, stores)
        update_logs(stores, ("Inclination estimation completed.", "ok"))

    except Exception as e:
        update_logs(stores, (str(e), "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def on_change_checkbox(
    stores: Stores, get_key_fn: Callable[[], Optional[str]]
) -> Callable[[ft.ControlEvent], None]:

    def closure(e: ft.ControlEvent) -> None:
        key = get_key_fn()
        if key is not None:
            gc_state = stores.computation_result.grain_classification_result
            gc = gc_state.get()
            if gc is not None:
                gc[key]["display"] = e.control.value
                # stores.ui.analysis_tab.computation_unit.set("grain")
                gc_state.set(gc)
                force_update_image_view(stores)  # ! けっこうむりやりupdateしてる。
            else:
                raise ValueError(
                    "stores.computation_result.grain_classification_result should not None"
                )
        else:
            # do nothing
            ...

    return closure


def checkbox_reactive_state(
    stores: Stores, get_key_fn: Callable[[], Optional[str]]
) -> Callable[[], bool]:
    def closure() -> bool:
        key = get_key_fn()
        if key is not None:
            gc = stores.computation_result.grain_classification_result.get()
            if gc is not None:
                return gc[key]["display"]
            else:
                return False
        else:
            return False

    return closure


def make_mineral_list(stores: Stores) -> list[CustomReactiveCheckbox]:

    def get_elem_from_keys(i: int) -> Optional[str]:
        ll = stores.computation_result.grain_classification_result.get()
        if ll is None:
            return None
        llkeys = list(ll.keys())
        if i < len(llkeys):
            return llkeys[i]
        else:
            return None

    def get_visible_state(i: int) -> bool:
        ll = stores.computation_result.grain_classification_result.get()
        if ll is None:
            return False
        llkeys = list(ll.keys())
        # llkeysは、["quartz", "garnet", ...]のような配列
        if i < len(llkeys):
            if (
                llkeys[i] == "mask"
            ):  # maskはgrain classification codeの予約語であり、maskは表示しない。
                return False
            else:
                return True
        else:
            return False

    return list(
        map(
            lambda i: CustomReactiveCheckbox(
                value=ReactiveState(
                    checkbox_reactive_state(
                        stores, lambda idx=i: get_elem_from_keys(idx)
                    ),
                    [stores.computation_result.grain_classification_result],
                ),
                label=ReactiveState(
                    lambda idx=i: get_elem_from_keys(idx),
                    [stores.computation_result.grain_classification_result],
                ),
                visible=ReactiveState(
                    lambda idx=i: get_visible_state(idx),
                    [stores.computation_result.grain_classification_result],
                ),
                on_change=on_change_checkbox(
                    stores, lambda idx=i: get_elem_from_keys(idx)
                ),
            ),
            range(20),
        )
    )


def make_drop_rose_diagram(stores: Stores) -> ReactiveCustomDropDown:
    return ReactiveCustomDropDown(
        hint_text=to_grain_display(
            stores.ui.analysis_tab.grain_rose_diagram_target.get()
        ),
        width=200,
        options=list(
            map(
                lambda x: ft.dropdown.Option(to_rose_display(x)),
                RoseDiagramUsedInPlot,
            )
        ),
        on_change=lambda e: stores.ui.analysis_tab.grain_rose_diagram_target.set(
            inv_rose_display(e.control.value)
        ),
    )


#! depevoping
def make_drop_rose_diagram_at_pixel(stores: Stores) -> ReactiveCustomDropDown:
    return ReactiveCustomDropDown(
        hint_text=to_grain_display(
            stores.ui.analysis_tab.grain_rose_diagram_target.get()
        ),
        width=200,
        options=list(
            map(
                lambda x: ft.dropdown.Option(to_rose_display(x)),
                RoseDiagramUsedInPixel,
            )
        ),
        on_change=lambda e: stores.ui.analysis_tab.grain_rose_diagram_target.set(
            inv_rose_display(e.control.value)
        ),
    )


def make_mineral_list_Row(stores: Stores) -> ReactiveRow:
    # return ReactiveRow(
    #     controls=ReactiveState(
    #         lambda: make_mineral_list(stores),
    #         [
    #             stores.computation_result.grain_classification_legend,
    #             stores.computation_result.grain_classification_result,
    #         ],
    #     ),
    #     scroll=True,
    # )

    return ft.Row(
        controls=make_mineral_list(stores),
        scroll=True,
    )


def _format_grain_dropdown_text(stores: Stores, key: str) -> str:
    base = to_grain_display(key)
    formatted = format_quantity_label(base, get_grain_unit_label(stores, key))
    return formatted if formatted is not None else base


def _make_grain_option(stores: Stores, key: str) -> ft.dropdown.Option:
    text = _format_grain_dropdown_text(stores, key)
    value = to_grain_display(key)
    return ft.dropdown.Option(text=text, key=value)


def make_drop_histogram_at_grain(stores: Stores) -> ReactiveCustomDropDown:
    return ReactiveCustomDropDown(
        hint_text=_format_grain_dropdown_text(
            stores, stores.ui.analysis_tab.grain_histogram_target.get()
        ),
        width=200,
        options=list(
            map(lambda x: _make_grain_option(stores, x), GrainNumListUsedInPlot)
        ),
        on_change=lambda e: stores.ui.analysis_tab.grain_histogram_target.set(
            inv_grain_display(e.control.value)
        ),
    )


def make_drop_scatter_target_x(stores: Stores) -> ReactiveCustomDropDown:
    return ReactiveCustomDropDown(
        hint_text=_format_grain_dropdown_text(
            stores, stores.ui.analysis_tab.scatter_target_x.get()
        ),
        width=200,
        options=list(
            map(lambda x: _make_grain_option(stores, x), GrainNumListUsedInPlot)
        ),
        on_change=lambda e: stores.ui.analysis_tab.scatter_target_x.set(
            inv_grain_display(e.control.value)
        ),
    )


def make_drop_scatter_target_y(stores: Stores) -> ReactiveCustomDropDown:
    return ReactiveCustomDropDown(
        hint_text=_format_grain_dropdown_text(
            stores, stores.ui.analysis_tab.scatter_target_y.get()
        ),
        width=200,
        options=list(
            map(lambda x: _make_grain_option(stores, x), GrainNumListUsedInPlot)
        ),
        on_change=lambda e: stores.ui.analysis_tab.scatter_target_y.set(
            inv_grain_display(e.control.value)
        ),
    )


# def make_CIP_no_and_ne_input(stores: Stores) -> tuple[ReactiveFloatTextField, ReactiveFloatTextField]:
#     no = ReactiveFloatTextField(
#         value=stores.computation_result.optical_parameters.no,
#         on_change=lambda e: stores.computation_result.optical_parameters.no.set(
#             e.control.value
#         ),
#     )
#     ne = ReactiveFloatTextField(
#         value=stores.computation_result.optical_parameters.ne,
#         on_change=lambda e: stores.computation_result.optical_parameters.ne.set(
#             e.control.value
#         ),
#     )
#     return no, ne


def make_pixel_or_grain_radio_button(stores: Stores) -> ft.RadioGroup:
    return ft.RadioGroup(
        content=ft.Row(
            [
                CustomRadio(value="grain", label="Grain"),
                CustomRadio(value="pixel", label="Pixel"),
            ]
        ),
        value=stores.ui.analysis_tab.computation_unit.get(),
        on_change=lambda e: stores.ui.analysis_tab.computation_unit.set(
            e.control.value
        ),
    )


def make_max_R_or_thickness_radio_button(stores: Stores) -> ft.RadioGroup:
    return ft.RadioGroup(
        content=ft.Row(
            [
                CustomRadio(value="max R", label="Max R"),
                CustomRadio(value="thickness", label="Thickness"),
            ]
        ),
        value="max R",
        on_change=lambda e: stores.computation_result.tilt_image_info.estimate_inclination_by.set(
            e.control.value
        ),
    )


def make_cip_thickness_input(
    stores: Stores,
) -> ft.Row:

    visible = ReactiveState(
        lambda: stores.computation_result.tilt_image_info.estimate_inclination_by.get()
        == "thickness",
        [stores.computation_result.tilt_image_info.estimate_inclination_by],
    )

    input = make_reactive_float_text_filed(
        stores,
        stores.computation_result.optical_parameters.thickness,
        parse_larger_than_0,
        accept_None=False,
    )

    return ReactiveRow([input, CustomText("mm")], visible=visible)


def make_cip_bandwidth_input(
    stores: Stores,
) -> ft.Row:

    input = make_reactive_float_text_filed(
        stores,
        stores.ui.analysis_tab.cip_bandwidth,
        parse_larger_than_0,
        accept_None=False,
    )
    return ft.Row(
        [
            CustomText("Polar plot bandwidth"),
            make_REMOVE_counter_button(stores, stores.ui.analysis_tab.cip_bandwidth),
            input,
            make_ADD_counter_button(stores, stores.ui.analysis_tab.cip_bandwidth),
        ]
    )


def make_cip_contour_num(
    stores: Stores,
) -> ft.Row:

    input = make_reactive_float_text_filed(
        stores,
        stores.ui.analysis_tab.cip_contour,
        parse_int,
        accept_None=False,
    )
    return ft.Row(
        [
            CustomText("Number of contours"),
            # make_REMOVE_counter_button(stores, stores.ui.analysis_tab.cip_contour),
            input,
            # make_ADD_counter_button(stores, stores.ui.analysis_tab.cip_contour),
        ]
    )


def make_cip_max_R_input(
    stores: Stores,
) -> ft.Row:

    visible = ReactiveState(
        lambda: stores.computation_result.tilt_image_info.estimate_inclination_by.get()
        == "max R",
        [stores.computation_result.tilt_image_info.estimate_inclination_by],
    )

    input = make_reactive_float_text_filed(
        stores,
        stores.computation_result.optical_parameters.max_R,
        parse_larger_than_0,
        accept_None=True,
    )

    return ReactiveRow([input, CustomText("nm")], visible=visible)


def make_cip_noise_size_pint(stores: Stores) -> ft.Row:

    input = make_reactive_float_text_filed(
        stores,
        stores.ui.analysis_tab.cip_points_noise_size_percent,
        parse_larger_than_0,
        accept_None=True,
    )

    return ReactiveRow(
        [
            CustomText("Point noise"),
            input,
            CustomText("%"),
        ],
        visible=stores.ui.analysis_tab.cip_display_points,
    )


def make_cip_start_button(stores: Stores, *, logger: Logger) -> CustomExecuteButton:
    return CustomExecuteButton(
        "Start CPO computation",
        on_click=lambda e: onclick_cip_start_button(stores, e, logger=logger),
        visible=ReactiveState(
            lambda: stores.ui.computing_is_stop.get(),
            [stores.ui.computing_is_stop],
        ),
    )

    # return ft.ElevatedButton(
    #     "start CIP computation",
    #     on_click=lambda e: onclick_cip_start_button(stores, e, logger=logger),
    # )


def make_CIP_no_and_ne_input(
    stores: Stores,
) -> tuple[ReactiveFloatTextField, ReactiveFloatTextField]:
    no = make_reactive_float_text_filed(
        stores,
        stores.computation_result.optical_parameters.no,
        parse_larger_than_0,
        accept_None=False,
    )
    ne = make_reactive_float_text_filed(
        stores,
        stores.computation_result.optical_parameters.ne,
        parse_larger_than_0,
        accept_None=False,
    )
    return no, ne


def make_scatter_origin_checkbox(stores: Stores) -> CustomReactiveCheckbox:
    return CustomReactiveCheckbox(
        label="origin",
        visible=stores.ui.analysis_tab.scatter_show_regression,
        value=stores.ui.analysis_tab.scatter_regression_origin,
        on_change=lambda e: stores.ui.analysis_tab.scatter_regression_origin.set(
            e.control.value
        ),
    )


def make_scatter_regression_checkbox(stores: Stores) -> CustomReactiveCheckbox:
    return CustomReactiveCheckbox(
        label="regression",
        value=stores.ui.analysis_tab.scatter_show_regression,
        on_change=lambda e: stores.ui.analysis_tab.scatter_show_regression.set(
            e.control.value
        ),
    )


def make_histogram_log_checkbox(stores: Stores) -> CustomReactiveCheckbox:
    return CustomReactiveCheckbox(
        label="log(x)",
        value=stores.ui.analysis_tab.histogram_log_x,
        on_change=lambda e: stores.ui.analysis_tab.histogram_log_x.set(e.control.value),
    )


def make_scatter_log_x_checkbox(stores: Stores) -> CustomReactiveCheckbox:
    return CustomReactiveCheckbox(
        label="log(x)",
        value=stores.ui.analysis_tab.scatter_log_x,
        on_change=lambda e: stores.ui.analysis_tab.scatter_log_x.set(e.control.value),
    )


def make_scatter_log_y_checkbox(stores: Stores) -> CustomReactiveCheckbox:
    return CustomReactiveCheckbox(
        label="log(y)",
        value=stores.ui.analysis_tab.scatter_log_y,
        on_change=lambda e: stores.ui.analysis_tab.scatter_log_y.set(e.control.value),
    )


def make_cip_theme_input(stores: Stores) -> ReactiveCustomDropDown:
    d = ReactiveCustomDropDown(
        hint_text="jet",
        options=[
            ft.dropdown.Option("jet"),
            ft.dropdown.Option("gray_r"),
            ft.dropdown.Option("gray"),
            ft.dropdown.Option("viridis"),
            ft.dropdown.Option("plasma"),
        ],
        on_change=lambda e: stores.ui.analysis_tab.cip_theme.set(e.control.value),
    )
    d.width = 100
    d.content_padding = 5
    return d


def make_cip_display_points_input(stores: Stores) -> CustomReactiveCheckbox:

    return CustomReactiveCheckbox(
        label="display points",
        value=stores.ui.analysis_tab.cip_display_points,
        on_change=lambda e: stores.ui.analysis_tab.cip_display_points.set(
            e.control.value
        ),
    )


class AnalysisTab(ft.Container):
    def __init__(
        self,
        page: Page,
        stores: Stores,
    ):

        super().__init__()
        self.padding = stores.appearance.tab_padding

        logger = getLogger("niconavi").getChild(__name__)

        analysis_tab = stores.ui.analysis_tab

        visible_rose_diagram = ReactiveState(
            (lambda: analysis_tab.plot_option.get() == "rose diagram"),
            [analysis_tab.plot_option, analysis_tab.computation_unit],
        )

        visible_histogram = ReactiveState(
            lambda: (analysis_tab.plot_option.get() == "histogram"),
            [analysis_tab.plot_option, analysis_tab.computation_unit],
        )

        visible_scatter = ReactiveState(
            lambda: analysis_tab.plot_option.get() == "scatter",
            [analysis_tab.plot_option],
        )

        visible_CIP = ReactiveState(
            lambda: analysis_tab.plot_option.get() == "CPO",
            [analysis_tab.plot_option],
        )

        visible_CIP_Polar = ReactiveState(
            lambda: (
                stores.ui.selected_button_at_analysis_tab.get() == 14
                or stores.ui.selected_button_at_analysis_tab.get() == 15
                or stores.ui.selected_button_at_analysis_tab.get() == 16
            )
            and analysis_tab.plot_option.get() == "CPO",
            [stores.ui.selected_button_at_analysis_tab, analysis_tab.plot_option],
        )

        # selection_is_grain = ReactiveState(
        #     lambda: stores.ui.analysis_tab.computation_unit.get() == "grain",
        #     [stores.ui.analysis_tab.computation_unit],
        # )
        # selection_is_pixel = ReactiveState(
        #     lambda: stores.ui.analysis_tab.computation_unit.get() == "pixel",
        #     [stores.ui.analysis_tab.computation_unit],
        # )

        plot_option = ReactiveCustomDropDown(
            hint_text="rose diagram",
            width=100,
            options=[
                ft.dropdown.Option("rose diagram"),
                ft.dropdown.Option("histogram"),
                ft.dropdown.Option("scatter"),
                ft.dropdown.Option("SPO"),
                ft.dropdown.Option("CPO"),
            ],
            on_change=lambda e: analysis_tab.plot_option.set(e.control.value),
        )

        mineral_list = make_mineral_list_Row(stores)

        drop_rose_diagram = make_drop_rose_diagram(stores)
        # drop_rose_diagram_at_pixel = make_drop_rose_diagram_at_pixel(stores)
        drop_histogram = make_drop_histogram_at_grain(stores)
        histogram_log_x_checkbox = make_histogram_log_checkbox(stores)

        def _on_histogram_alpha_change(value: float) -> None:
            stores.ui.analysis_tab.histogram_alpha.set(value)
            force_update_image_view(stores)

        def _on_rose_alpha_change(value: float) -> None:
            stores.ui.analysis_tab.rose_alpha.set(value)
            force_update_image_view(stores)

        histogram_alpha_slider = ReactiveSlider(
            value=stores.ui.analysis_tab.histogram_alpha,
            min=0.0,
            max=1.0,
            divisions=20,
            on_change=lambda e: _on_histogram_alpha_change(float(e.control.value)),
        )
        drop_scatter_target_x = make_drop_scatter_target_x(stores)
        drop_scatter_target_y = make_drop_scatter_target_y(stores)
        no, ne = make_CIP_no_and_ne_input(stores)
        cip_radio = make_max_R_or_thickness_radio_button(stores)
        pixel_or_grain_radio = make_pixel_or_grain_radio_button(stores)
        cip_thickness = make_cip_thickness_input(stores)
        cip_max_R = make_cip_max_R_input(stores)
        cip_start_button = make_cip_start_button(stores, logger=logger)
        cip_bandwidth = make_cip_bandwidth_input(stores)
        cip_theme = make_cip_theme_input(stores)
        cip_display_points = make_cip_display_points_input(stores)
        scatter_regression = make_scatter_regression_checkbox(stores)
        scatter_origin = make_scatter_origin_checkbox(stores)
        scatter_log_x = make_scatter_log_x_checkbox(stores)
        scatter_log_y = make_scatter_log_y_checkbox(stores)
        cip_point_noise_input = make_cip_noise_size_pint(stores)
        rose_alpha_slider = ReactiveSlider(
            value=stores.ui.analysis_tab.rose_alpha,
            min=0.0,
            max=1.0,
            divisions=20,
            on_change=lambda e: _on_rose_alpha_change(float(e.control.value)),
        )

        histogram_bins_input = make_reactive_float_text_filed(
            stores,
            stores.computation_result.plot_parameters.histogram_bins,
            parse_int,
            accept_None=False,
        )

        rose_bins_input = make_reactive_float_text_filed(
            stores,
            stores.computation_result.plot_parameters.rose_diagram_bins,
            parse_int,
            accept_None=False,
        )

        # one_pixel = make_reactive_float_text_filed(
        #     stores,
        #     stores.ui.one_pixel,
        #     parse_larger_than_0,
        #     accept_None=True,
        # )

        # ft.Row(
        #     [
        #         CustomText("1 px ="),
        #         one_pixel,
        #         CustomText("μm"),
        #     ]
        # ),

        content = ft.Column(
            [
                ReactiveColumn(
                    [
                        mineral_list,
                        plot_option,
                        ReactiveColumn(
                            [
                                drop_rose_diagram,
                                ft.Row(
                                    [
                                        CustomText("Bins"),
                                        make_REMOVE_counter_button(
                                            stores,
                                            stores.computation_result.plot_parameters.rose_diagram_bins,
                                            step=1,
                                            min_value=1,
                                            value_type=int,
                                        ),
                                        rose_bins_input,
                                        make_ADD_counter_button(
                                            stores,
                                            stores.computation_result.plot_parameters.rose_diagram_bins,
                                            step=1,
                                            value_type=int,
                                        ),
                                    ]
                                ),
                                ft.Row(
                                    [
                                        CustomText("Opacity"),
                                        rose_alpha_slider,
                                    ]
                                ),
                                CustomReactiveCheckbox(
                                    label="flip",
                                    value=stores.ui.analysis_tab.rose_flip,
                                    on_change=lambda e: (
                                        stores.ui.analysis_tab.rose_flip.set(
                                            e.control.value
                                        ),
                                        force_update_image_view(stores),
                                    ),
                                ),
                                CustomText("information"),
                                ft.SelectionArea(
                                    CustomReactiveText(
                                        stores.ui.analysis_tab.rose_stats_text
                                    )
                                ),
                                # ReactiveColumn( [drop_rose_diagram], visible=selection_is_grain
                                # ),
                                # ReactiveColumn(
                                #     [drop_rose_diagram_at_pixel],
                                #     visible=selection_is_pixel,
                                # ),
                            ],
                            visible=visible_rose_diagram,
                        ),
                        ReactiveColumn(
                            [
                                # CustomReactiveText("histogram:"),
                                drop_histogram,
                                histogram_log_x_checkbox,
                                ft.Row(
                                    [
                                        CustomText("Bins"),
                                        make_REMOVE_counter_button(
                                            stores,
                                            stores.computation_result.plot_parameters.histogram_bins,
                                            step=1,
                                            min_value=1,
                                            value_type=int,
                                        ),
                                        histogram_bins_input,
                                        make_ADD_counter_button(
                                            stores,
                                            stores.computation_result.plot_parameters.histogram_bins,
                                            step=1,
                                            value_type=int,
                                        ),
                                    ]
                                ),
                                ft.Row(
                                    [
                                        CustomText("Opacity"),
                                        histogram_alpha_slider,
                                    ]
                                ),
                                CustomText("information"),
                                ft.SelectionArea(
                                    CustomReactiveText(
                                        stores.ui.analysis_tab.histogram_stats_text
                                    )
                                ),
                            ],
                            visible=visible_histogram,
                        ),
                        ReactiveColumn(
                            [
                                CustomReactiveText(
                                    "scatter plotted by",
                                    visible=visible_scatter,
                                ),
                                ft.Row(
                                    [
                                        CustomText("x:"),
                                        drop_scatter_target_x,
                                    ]
                                ),
                                ft.Row(
                                    [
                                        CustomText("y:"),
                                        drop_scatter_target_y,
                                    ]
                                ),
                                ft.Row([scatter_regression, scatter_origin]),
                                ft.Row([scatter_log_x, scatter_log_y]),
                            ],
                            visible=visible_scatter,
                        ),
                        ReactiveColumn(
                            [
                                CustomText("refractive indices (default: quartz)"),
                                ft.Row([CustomText("ω ="), no, CustomText(" ε ="), ne]),
                                CustomText("thickness(mm) or max retardation (nm)"),
                                ft.Row([cip_radio, cip_max_R, cip_thickness]),
                                cip_start_button,
                            ],
                            visible=visible_CIP,
                        ),
                    ],
                    visible=ReactiveState(
                        lambda: stores.computation_result.grain_classification_result.get()
                        is not None,
                        [stores.computation_result.grain_classification_result],
                    ),
                ),
                ft.Divider(),
                ReactiveColumn(
                    [
                        ft.Row(
                            [
                                CustomText("Plots for each"),
                                pixel_or_grain_radio,
                            ]
                        ),
                        cip_bandwidth,
                        make_cip_contour_num(stores),
                        ft.Row(
                            [
                                CustomText("Color theme:"),
                                cip_theme,
                            ]
                        ),
                        cip_display_points,
                        cip_point_noise_input,
                    ],
                    visible=visible_CIP_Polar,
                ),
            ],
            scroll=ft.ScrollMode.ADAPTIVE,
        )

        self.content = content
