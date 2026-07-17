from niconavi_app.stores import (
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
)
from niconavi_app.reactive_state import (
    ReactiveRow,
    ReactiveColumn,
    ReactiveSlider,
)
from niconavi_app.state import ReactiveState
from niconavi_app.components.log_view import update_logs
from niconavi_app.components.progress_bar import update_progress_bar
from niconavi_app.components.common_component import (
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
from niconavi_app.tools.tools import switch_tab_index, force_update_image_view
from niconavi_app.components.log_view import update_logs
from niconavi_app.match_used_name import (
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
from niconavi_app.components.view.spatial_units import format_quantity_label, get_grain_unit_label
from typing import Callable, Optional
import flet as ft
from flet import Page
from logging import getLogger, Logger
import niconavi_app.niconavi.run_all as po
import traceback
from niconavi_app.niconavi.tools.str_parser import (
    parse_larger_than_0,
    parse_int,
)
from niconavi_app.tools.validation import validation_larger_than_0_float
from niconavi_app.components.page_tab.tabs.reset_onclick import reset_onclick_cip_computation_button
from niconavi_app.niconavi.analysis import grain_stat_method_is_supported
from niconavi_app.niconavi.cpo_pipeline import (
    estimate_cpo_orientation,
    write_cpo_orientation_into_raw_maps,
    format_cpo_orientation_info,
    make_cpo_regression_figure,
    CPOOrientationResult,
)


def onclick_cip_start_button(
    stores: Stores, e: ft.FilePickerResultEvent, page: Optional[Page], *, logger: Logger
) -> None:
    # CPO input is thickness only now (Max R retired, see move_v3.md).
    if stores.computation_result.optical_parameters.thickness.get() is None:
        update_logs(stores, ("Please provide the thickness value.", "err"))
        return None

    def log(message: str, level: str = "msg") -> None:
        update_logs(stores, (message, level), logger=logger)

    try:
        update_progress_bar(None, stores)
        log("Starting CPO computation...")
        # Drop any previous RGB comparison figure (image-list button hides).
        stores.ui.analysis_tab.cip_regression_figure.set(None)
        r = as_ComputationResult(stores.computation_result)
        # Force the thickness-based inclination path (the Max R radio is gone).
        r.tilt_image_info.estimate_inclination_by = "thickness"
        r = reset_onclick_cip_computation_button(r)
        save_in_ComputationResultState(r, stores)

        update_progress_bar(0.1, stores)
        log("Estimating base inclination from the retardation map...")
        r = po.get_inclination(
            r, progress_callback=lambda p: update_progress_bar(p, stores)
        )

        # Replace the retardation-based inclination with the addition-image
        # theta + color-correction fit + E-down-tilt branch resolution - the
        # exact run_diagnostics.py pipeline (verified bit-identical). This
        # overrides raw_maps inclination / inclination_0_to_180 / azimuth360
        # so every CPO plot (grain & pixel, 90/180/360, map-COI) matches.
        normalize_90 = stores.ui.analysis_tab.cip_normalize_90.get()
        info_text, orientation = _run_cpo_orientation_pipeline(
            r,
            normalize_90=normalize_90,
            progress_callback=lambda p: update_progress_bar(p, stores),
            log=log,
        )

        update_progress_bar(0.8, stores)
        log("Analyzing grains for CPO...")
        r = po.analyze_grain_list_for_CIP(r)

        update_progress_bar(0.9, stores)
        log("Building the CPO stereo / COI maps...")
        r = po.make_CIP_map_info(r)
        save_in_ComputationResultState(r, stores)
        stores.ui.analysis_tab.cip_stats_text.set(info_text)

        # Build the before/after RGB regression figure and expose it as the
        # image-list "RGB comparison" button (shown while CPO is selected).
        figure = (
            make_cpo_regression_figure(orientation)
            if orientation is not None
            else None
        )
        stores.ui.analysis_tab.cip_regression_figure.set(figure)
        if figure is not None:
            log("RGB comparison ready (see the image list).")

        update_progress_bar(0, stores)
        log("CPO computation completed.", "ok")

    except Exception as e:
        update_logs(stores, (str(e), "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def _run_cpo_orientation_pipeline(
    r,
    *,
    normalize_90: bool,
    progress_callback: Callable[[float | None], None] = lambda p: None,
    log: Callable[..., None] = lambda *a, **k: None,
) -> tuple[str, Optional[CPOOrientationResult]]:
    """Run the run_diagnostics.py CPO orientation pipeline on r (mutating
    r.raw_maps inclination maps in place). Returns (info text, orientation);
    orientation is None when the required inputs are missing."""
    if (
        r.raw_maps is None
        or r.grain_map is None
        or r.tilt_image_info.tilt_image0 is None
    ):
        return (
            "CPO computed, but the addition-image orientation needs the\n"
            "retardation-plate maps and the 0° tilt image.",
            None,
        )

    orientation = estimate_cpo_orientation(
        r,
        normalize_90=normalize_90,
        progress_callback=progress_callback,
        log_callback=lambda m: log(m),
    )
    write_cpo_orientation_into_raw_maps(r.raw_maps, orientation)

    displayed_minerals = None
    if r.grain_classification_result is not None:
        displayed_minerals = sorted(
            mineral
            for mineral, selection in r.grain_classification_result.items()
            if mineral != "mask" and selection.get("display")
        )
    info_text = format_cpo_orientation_info(
        orientation=orientation,
        normalize_90=normalize_90,
        displayed_minerals=displayed_minerals,
    )
    return info_text, orientation


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


def _make_grain_options(stores: Stores) -> list[ft.dropdown.Option]:
    return [_make_grain_option(stores, key) for key in GrainNumListUsedInPlot]


def _make_grain_stat_dropdown(stores: Stores, target_state) -> ReactiveCustomDropDown:
    dropdown = ReactiveCustomDropDown(
        hint_text=_format_grain_dropdown_text(stores, target_state.get()),
        value=to_grain_display(target_state.get()),
        width=200,
        options=_make_grain_options(stores),
    )

    def refresh_dropdown_labels() -> None:
        dropdown.options = _make_grain_options(stores)
        dropdown.hint_text = _format_grain_dropdown_text(stores, target_state.get())
        dropdown.value = to_grain_display(target_state.get())
        dropdown.update()

    stores.ui.one_pixel.bind(refresh_dropdown_labels)
    target_state.bind(refresh_dropdown_labels)
    stores.computation_result.grain_list.bind(refresh_dropdown_labels)
    return dropdown


def make_drop_histogram_at_grain(stores: Stores) -> ReactiveCustomDropDown:
    dropdown = _make_grain_stat_dropdown(
        stores, stores.ui.analysis_tab.grain_histogram_target
    )
    dropdown.on_change = lambda e: stores.ui.analysis_tab.grain_histogram_target.set(
        inv_grain_display(e.control.value)
    )
    return dropdown


def make_drop_scatter_target_x(stores: Stores) -> ReactiveCustomDropDown:
    dropdown = _make_grain_stat_dropdown(
        stores, stores.ui.analysis_tab.scatter_target_x
    )
    dropdown.on_change = lambda e: stores.ui.analysis_tab.scatter_target_x.set(
        inv_grain_display(e.control.value)
    )
    return dropdown


def make_drop_scatter_target_y(stores: Stores) -> ReactiveCustomDropDown:
    dropdown = _make_grain_stat_dropdown(
        stores, stores.ui.analysis_tab.scatter_target_y
    )
    dropdown.on_change = lambda e: stores.ui.analysis_tab.scatter_target_y.set(
        inv_grain_display(e.control.value)
    )
    return dropdown


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


class ReactiveDisabledRadioGroup(ft.RadioGroup):
    def __init__(
        self,
        disabled: ReactiveState[bool],
        visible: ReactiveState[bool] | bool = True,
        active_value_state=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._disabled = disabled
        self._visible = visible
        self._active_value_state = active_value_state
        self.disabled = disabled.get()
        self.visible = visible.get() if isinstance(visible, ReactiveState) else visible
        if self.disabled:
            self.value = None
        disabled.bind(lambda: self._update_reactive_props())
        if isinstance(visible, ReactiveState):
            visible.bind(lambda: self._update_reactive_props())
        if active_value_state is not None:
            active_value_state.bind(lambda: self._update_reactive_props())

    def _update_reactive_props(self) -> None:
        self.disabled = self._disabled.get()
        self.visible = (
            self._visible.get()
            if isinstance(self._visible, ReactiveState)
            else self._visible
        )
        if self.disabled:
            self.value = None
        elif self._active_value_state is not None:
            self.value = self._active_value_state.get()
        self.update()


def make_grain_stat_method_radio(
    stores: Stores,
    *,
    label: str,
    target_state,
    method_state,
    visible: ReactiveState[bool],
) -> ft.Row:
    supported = ReactiveState(
        lambda: grain_stat_method_is_supported(target_state.get()),
        [target_state],
    )
    disabled = ReactiveState(lambda: not supported.get(), [supported])

    def on_change(e: ft.ControlEvent) -> None:
        if supported.get():
            method_state.set(e.control.value)
            force_update_image_view(stores)

    radio = ReactiveDisabledRadioGroup(
        disabled=disabled,
        visible=visible,
        active_value_state=method_state,
        content=ft.Row(
            [
                CustomRadio(value="median", label="median"),
                CustomRadio(value="mean", label="mean"),
            ],
            spacing=6,
        ),
        value=method_state.get(),
        on_change=on_change,
    )

    return ReactiveRow(
        [CustomText(label), radio],
        visible=visible,
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def make_cip_thickness_input(
    stores: Stores,
) -> ft.Row:
    # Thickness is the only CPO inclination input now (Max R retired), so it is
    # always shown - no longer gated on the removed estimate_inclination_by radio.
    input = make_reactive_float_text_filed(
        stores,
        stores.computation_result.optical_parameters.thickness,
        parse_larger_than_0,
        accept_None=False,
    )

    return ft.Row([input, CustomText("mm")])


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


def make_cip_start_button(
    stores: Stores, page: Optional[Page], *, logger: Logger
) -> CustomExecuteButton:
    return CustomExecuteButton(
        "calculate",
        on_click=lambda e: onclick_cip_start_button(stores, e, page, logger=logger),
        enabled=ReactiveState(
            lambda: stores.ui.computing_is_stop.get(),
            [stores.ui.computing_is_stop],
        ),
    )


def make_cip_normalize_90_checkbox(stores: Stores) -> CustomReactiveCheckbox:
    return CustomReactiveCheckbox(
        label="90° normalize",
        value=stores.ui.analysis_tab.cip_normalize_90,
        on_change=lambda e: stores.ui.analysis_tab.cip_normalize_90.set(
            e.control.value
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
        rose_stat_method = make_grain_stat_method_radio(
            stores,
            label="Grain value",
            target_state=stores.ui.analysis_tab.grain_rose_diagram_target,
            method_state=stores.ui.analysis_tab.rose_stat_method,
            visible=visible_rose_diagram,
        )
        histogram_stat_method = make_grain_stat_method_radio(
            stores,
            label="Grain value",
            target_state=stores.ui.analysis_tab.grain_histogram_target,
            method_state=stores.ui.analysis_tab.histogram_stat_method,
            visible=visible_histogram,
        )

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
        scatter_x_stat_method = make_grain_stat_method_radio(
            stores,
            label="x value",
            target_state=stores.ui.analysis_tab.scatter_target_x,
            method_state=stores.ui.analysis_tab.scatter_x_stat_method,
            visible=visible_scatter,
        )
        scatter_y_stat_method = make_grain_stat_method_radio(
            stores,
            label="y value",
            target_state=stores.ui.analysis_tab.scatter_target_y,
            method_state=stores.ui.analysis_tab.scatter_y_stat_method,
            visible=visible_scatter,
        )
        no, ne = make_CIP_no_and_ne_input(stores)
        pixel_or_grain_radio = make_pixel_or_grain_radio_button(stores)
        cip_thickness = make_cip_thickness_input(stores)
        cip_normalize_90 = make_cip_normalize_90_checkbox(stores)
        cip_start_button = make_cip_start_button(stores, page, logger=logger)
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
                                rose_stat_method,
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
                                histogram_stat_method,
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
                                scatter_x_stat_method,
                                ft.Row(
                                    [
                                        CustomText("y:"),
                                        drop_scatter_target_y,
                                    ]
                                ),
                                scatter_y_stat_method,
                                ft.Row([scatter_regression, scatter_origin]),
                                ft.Row([scatter_log_x, scatter_log_y]),
                            ],
                            visible=visible_scatter,
                        ),
                        ReactiveColumn(
                            [
                                CustomText("refractive indices (default: quartz)"),
                                ft.Row([CustomText("ω ="), no, CustomText(" ε ="), ne]),
                                CustomText("thickness (mm)"),
                                cip_thickness,
                                cip_normalize_90,
                                cip_start_button,
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Container(
                                                content=CustomText("Information"),
                                                padding=ft.padding.only(
                                                    left=8, top=4, bottom=4
                                                ),
                                                bgcolor=ft.Colors.BLACK26,
                                            ),
                                            ft.Container(
                                                content=ft.SelectionArea(
                                                    CustomReactiveText(
                                                        stores.ui.analysis_tab.cip_stats_text
                                                    )
                                                ),
                                                padding=8,
                                            ),
                                        ],
                                        spacing=0,
                                    ),
                                    border=ft.border.all(1, ft.Colors.WHITE24),
                                    border_radius=6,
                                    margin=ft.margin.only(top=8),
                                ),
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
