from stores import (
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
)
from reactive_state import (
    ReactiveText,
    ReactiveExpansionTile,
    ReactiveRadioGroup,
    ReactiveDivider,
    ReactiveRow,
    ReactiveColumn,
    ReactiveElevatedButton,
    ReactiveTextField,
    ReactiveCheckbox,
    ReactiveSlider,
)
from state import State, ReactiveState, is_not_None_state
from components.log_view import update_logs
from components.progress_bar import update_progress_bar
from components.common_component import (
    make_reactive_float_text_filed,
    make_solidable_checkbox,
    make_REMOVE_counter_button,
    make_ADD_counter_button,
    CustomText,
    customDivider,
    CustomExecuteButton,
    CustomRadio,
    CustomReactiveCheckbox,
)
from tools.tools import switch_tab_index
from components.log_view import update_logs
from components.page_tab.tabs.reset_onclick import (
    reset_onclick_grain_boundary_button,
    reset_onclick_grain_analyze_button,
)

from components.labeling_app.labeling_controller import LabelingController
from typing import Callable, overload, Optional, cast
import flet as ft
from flet import Page
from result import Ok, Err, Result, is_ok, is_err
from logging import getLogger, Logger
import numpy as np
from niconavi.image.type import RGBPicture
import cv2

# from gui.src.components.labeling_app.reset import reset_filter_tab

from components.page_tab.tabs.merge_component import (
    make_code_input,
    make_continue_button,
    make_merge_button,
    make_reset_button,
)
import niconavi.run_all as po
from niconavi.tools.str_parser import (
    parse_int,
    parse_larger_than_0,
    parse_larger_than_1,
)
from components.labeling_app.reset import reset_filter_tab
import traceback


def _apply_brightness_contrast(
    image: Optional[RGBPicture],
    brightness: float,
    contrast: float,
) -> Optional[RGBPicture]:
    if image is None:
        return None

    img = image.astype(np.float32)
    adjusted = (img - 127.5) * contrast + 127.5
    adjusted = adjusted * brightness
    return cast(RGBPicture, np.clip(adjusted, 0, 255).astype(np.uint8))


def _snap_to_odd(value: float, minimum: int = 1, maximum: int = 21) -> int:
    snapped = int(round(value))
    snapped = max(minimum, min(maximum, snapped))
    if snapped % 2 == 0:
        snapped = snapped + 1 if snapped < maximum else snapped - 1
    return snapped


def update_r_color_map_display(stores: Stores) -> None:
    raw_maps = stores.computation_result.raw_maps.get()
    if raw_maps is None:
        return

    base_map = raw_maps.get("R_color_map_raw")
    if base_map is None:
        return

    base_map_used: Optional[RGBPicture] = cast(Optional[RGBPicture], base_map)

    if stores.ui.grain_tab.brightness_correction.get():
        extinction_map = raw_maps.get("extinction_color_map")
        if extinction_map is not None:
            hsv_R_map = cv2.cvtColor(base_map, cv2.COLOR_RGB2HSV)
            hsv_R_min_map = cv2.cvtColor(extinction_map, cv2.COLOR_RGB2HSV)
            d_hsv_R_map = hsv_R_map.copy()
            d_hsv_R_map[:, :, 2] = np.clip(
                hsv_R_map[:, :, 2].astype(np.float64)
                - hsv_R_min_map[:, :, 2].astype(np.float64),
                0,
                255,
            ).astype(np.uint8)
            base_map_used = cv2.cvtColor(d_hsv_R_map, cv2.COLOR_HSV2RGB)

    adjusted = _apply_brightness_contrast(
        base_map_used,
        stores.ui.grain_tab.slider_brightness.get()
        if stores.ui.grain_tab.use_brightness
        else 1.0,
        stores.ui.grain_tab.slider_contrast.get()
        if stores.ui.grain_tab.use_contrast
        else 1.0,
    )

    if adjusted is None:
        return

    kernel_size = int(stores.ui.grain_tab.slider_median_kernel.get())
    if kernel_size > 1:
        adjusted = cv2.medianBlur(adjusted, kernel_size)

    stores.computation_result.raw_maps.set(
        {
            **raw_maps,
            "R_color_map_display": adjusted,
        }
    )


def edit_button_click(stores: Stores, e: ft.ControlEvent, *, logger: Logger) -> None:
    try:

        update_progress_bar(None, stores)
        r = as_ComputationResult(stores.computation_result)

        res = po.analyze_grain_list(
            r,
            progress_callback=lambda p: update_progress_bar(p, stores),
        )

        update_progress_bar(0.0, stores)
        update_logs(stores, ("Grain analysis completed.", "ok"))
        save_in_ComputationResultState(res, stores)
        switch_tab_index(stores, 2)

    except Exception as e:
        update_logs(stores, (str(e), "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def continue_button_click(
    stores: Stores, e: ft.ControlEvent, *, logger: Logger
) -> None:
    try:
        reset_filter_tab(stores)
        controller = LabelingController(stores=stores)
        controller.reset_application()
        controller.on_load_clicked()

        update_progress_bar(None, stores)
        r = as_ComputationResult(stores.computation_result)
        r = reset_onclick_grain_analyze_button(r)
        save_in_ComputationResultState(r, stores)

        update_progress_bar(0.0, stores)
        update_logs(stores, ("Grain analysis completed.", "ok"))
        # save_in_ComputationResultState(res, stores)

        stores.ui.progress.set(3)
        switch_tab_index(stores, 3)

    except Exception as e:
        update_logs(stores, (str(e), "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def execute_grain_boundary_calc_button_click(
    stores: Stores,
    e: ft.ControlEvent,
    *,
    logger: Logger,
) -> None:
    try:


        update_progress_bar(None, stores)
        r = as_ComputationResult(stores.computation_result)
        r = reset_onclick_grain_boundary_button(r)
        r = po.make_grain_boundary(r)
        # r = reset_onclick_grain_analyze_button(r)
        save_in_ComputationResultState(r, stores)
        update_progress_bar(0.0, stores)
        update_logs(stores, ("Grain segmentation completed.", "ok"))
        # switch_tab_index(stores, 3)

        reset_filter_tab(stores)
        controller = LabelingController(stores=stores)
        controller.reset_application()
        controller.on_load_clicked()

    except Exception as e:
        update_logs(stores, (str(e), "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())
    # analyze_grain_list(stores, logger=logger)


def make_execute_grain_boundary_calc_button(
    stores: Stores, *, logger: Logger
) -> ReactiveElevatedButton:
    execute_grain_boundary_calc_button = CustomExecuteButton(
        "Calculate grain boundaries",
        on_click=lambda e: execute_grain_boundary_calc_button_click(
            stores, e, logger=logger
        ),
        visible=ReactiveState(
            lambda: (
                stores.ui.computing_is_stop.get()
                and stores.computation_result.raw_maps.get() is not None
            ),
            [stores.ui.computing_is_stop, stores.computation_result.raw_maps],
        ),
    )
    return execute_grain_boundary_calc_button


def make_edit_button_visible(
    stores: Stores, *, logger: Logger
) -> ReactiveElevatedButton:
    edit_button_visible = ReactiveState(
        lambda: stores.computation_result.grain_map.get() is not None,
        [stores.computation_result.grain_map],
    )
    edit_button = ReactiveElevatedButton(
        "Edit",
        on_click=lambda e: edit_button_click(stores, e, logger=logger),
        visible=ReactiveState(
            lambda: (stores.ui.computing_is_stop.get() and edit_button_visible.get()),
            [stores.ui.computing_is_stop, edit_button_visible],
        ),
    )
    return edit_button


def make_continue_button_visible(
    stores: Stores, *, logger: Logger
) -> ReactiveElevatedButton:

    continue_button_visible = ReactiveState(
        lambda: stores.computation_result.grain_map.get() is not None
        and stores.ui.computing_is_stop.get(),
        [stores.computation_result.grain_map, stores.ui.computing_is_stop],
    )

    continue_button = CustomExecuteButton(
        "▶ Continue",
        visible=continue_button_visible,
        on_click=lambda e: continue_button_click(stores, e, logger=logger),
    )
    return continue_button


def is_exist_azimuth(stores: Stores) -> bool:
    raw_maps = stores.computation_result.raw_maps.get()
    if raw_maps is not None:
        if raw_maps["azimuth"] is not None:
            return True
        else:
            return False
    else:
        return False


def make_color_R_rev_estimation_text(stores: Stores) -> ReactiveText:
    return ReactiveText(
        ReactiveState(
            lambda: (
                "<= R <="
                if not stores.computation_result.grain_detection_parameters.color_rev_estimation.get()
                else "> R, R >"
            ),
            [stores.computation_result.grain_detection_parameters.color_rev_estimation],
        ),
        color=ft.Colors.WHITE,
    )


class GrainTab(ft.Container):
    def __init__(
        self,
        page: Page,
        stores: Stores,
    ):

        super().__init__()

        logger = getLogger("niconavi").getChild(__name__)

        self.padding = stores.appearance.tab_padding

        smallest_grain_size = make_reactive_float_text_filed(
            stores,
            stores.computation_result.grain_detection_parameters.smallest_grain_size,
            parse_larger_than_1,
        )

        th_about_connect_skeleton_endpoints = make_reactive_float_text_filed(
            stores,
            stores.computation_result.grain_detection_parameters.th_about_connect_skeleton_endpoints,
            parse_int,
            accept_None=False,
        )

        th_about_hessian_emphasis = make_reactive_float_text_filed(
            stores,
            stores.computation_result.grain_detection_parameters.th_about_hessian_emphasis,
            parse_larger_than_0,
            accept_None=False,
        )

        execute_grain_boundary_calc_button = make_execute_grain_boundary_calc_button(
            stores, logger=logger
        )

        continue_button = make_continue_button_visible(stores, logger=logger)

        # azimuth_use_checkbox = make_azimuth_ex_angle_radio(stores)

        marge_panel = ReactiveExpansionTile(
            # visible=visible_marge_panel,
            visible=False, #! つかわないから隠す
            title=CustomText("Merge grains by code"),
            affinity=ft.TileAffinity.PLATFORM,
            maintain_state=False,
            # bgcolor=ft.Colors.BLACK87,
            # collapsed_bgcolor=ft.Colors.BLACK87,
            controls=[
                ft.Column(
                    [
                        make_code_input(stores),
                        ft.Row(
                            [
                                make_merge_button(stores, logger=logger),
                                make_reset_button(stores, logger=logger),
                            ]
                        ),
                        make_continue_button(stores, logger=logger),
                    ]
                ),
            ],
        )

        def _on_brightness_change(value: float) -> None:
            stores.ui.grain_tab.slider_brightness.set(value)
            update_r_color_map_display(stores)

        def _on_contrast_change(value: float) -> None:
            stores.ui.grain_tab.slider_contrast.set(value)
            update_r_color_map_display(stores)

        def _on_brightness_correction_change(value: bool) -> None:
            stores.ui.grain_tab.brightness_correction.set(value)
            update_r_color_map_display(stores)

        def _on_median_change(value: float) -> None:
            kernel = _snap_to_odd(value)
            stores.ui.grain_tab.slider_median_kernel.set(kernel)
            update_r_color_map_display(stores)

        content = ft.Column(
            [
                ft.Column(
                    [
                        ft.Column(
                            [
                                ft.Row(
                                    [
                                        CustomText("Brightness"),
                                        ReactiveSlider(
                                            value=stores.ui.grain_tab.slider_brightness,
                                            min=0.2,
                                            max=2.0,
                                            divisions=90,
                                            on_change=lambda e: _on_brightness_change(
                                                float(e.control.value)
                                            ),
                                        ),
                                    ],
                                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                    visible=stores.ui.grain_tab.use_brightness,
                                ),
                                ft.Row(
                                    [
                                        CustomText("Contrast"),
                                        ReactiveSlider(
                                            value=stores.ui.grain_tab.slider_contrast,
                                            min=0.2,
                                            max=2.0,
                                            divisions=90,
                                            on_change=lambda e: _on_contrast_change(
                                                float(e.control.value)
                                            ),
                                        ),
                                    ],
                                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                    visible=stores.ui.grain_tab.use_contrast,
                                ),
                                ft.Row(
                                    [
                                        CustomText("Median filter kernel"),
                                        ReactiveSlider(
                                            value=stores.ui.grain_tab.slider_median_kernel,
                                            min=1,
                                            max=21,
                                            divisions=10,
                                            on_change=lambda e: _on_median_change(
                                                float(e.control.value)
                                            ),
                                        ),
                                    ],
                                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                ),
                                ft.Row(
                                    [
                                        CustomReactiveCheckbox(
                                            value=stores.ui.grain_tab.brightness_correction,
                                            label="Brightness correction",
                                            on_change=lambda e: _on_brightness_correction_change(
                                                bool(e.control.value)
                                            ),
                                        )
                                    ],
                                    alignment=ft.MainAxisAlignment.START,
                                ),
                                ft.Divider(),
                            ]
                        ),

                        ft.Row(
                            [
                                CustomText("Boundary detection sensitivity"),
                                make_REMOVE_counter_button(
                                    stores,
                                    stores.computation_result.grain_detection_parameters.th_about_hessian_emphasis,
                                    step=0.05,
                                    min_value=0.0,
                                    max_value=1.0,
                                    precision=2,
                                    value_type=float,
                                ),
                                th_about_hessian_emphasis,
                                make_ADD_counter_button(
                                    stores,
                                    stores.computation_result.grain_detection_parameters.th_about_hessian_emphasis,
                                    step=0.05,
                                    min_value=0.0,
                                    max_value=1.0,
                                    precision=2,
                                    value_type=float,
                                ),
                            ]
                        ),
                        ft.Row(
                            [
                                CustomText("Boundary connectivity"),
                                make_REMOVE_counter_button(
                                    stores,
                                    stores.computation_result.grain_detection_parameters.th_about_connect_skeleton_endpoints,
                                    step=5,
                                    min_value=0,
                                    max_value=100,
                                    value_type=int,
                                ),
                                th_about_connect_skeleton_endpoints,
                                make_ADD_counter_button(
                                    stores,
                                    stores.computation_result.grain_detection_parameters.th_about_connect_skeleton_endpoints,
                                    step=5,
                                    min_value=0,
                                    max_value=100,
                                    value_type=int,
                                ),
                            ]
                        ),
                        ft.Row(
                            [
                                CustomText("Smallest grain size"),
                                make_REMOVE_counter_button(
                                    stores,
                                    stores.computation_result.grain_detection_parameters.smallest_grain_size,
                                    step=10,
                                    min_value=1,
                                    max_value=200,
                                    value_type=int,
                                ),
                                smallest_grain_size,
                                make_ADD_counter_button(
                                    stores,
                                    stores.computation_result.grain_detection_parameters.smallest_grain_size,
                                    step=10,
                                    min_value=0,
                                    max_value=200,
                                    value_type=int,
                                ),
                            ]
                        ),
                    ],
                ),
                customDivider(),
                execute_grain_boundary_calc_button,
                # grain_boundary_checkbox,
                ft.Row(
                    [
                        continue_button,
                        # edit_button,
                    ]
                ),
                # parameter_setting,
                marge_panel,
            ],
            scroll=ft.ScrollMode.ADAPTIVE,
        )

        self.content = content
