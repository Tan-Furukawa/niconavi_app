from niconavi_app.stores import (
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
)
from niconavi_app.reactive_state import (
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
from niconavi_app.state import State, ReactiveState, is_not_None_state
from niconavi_app.components.log_view import update_logs
from niconavi_app.components.progress_bar import update_progress_bar
from niconavi_app.components.common_component import (
    make_reactive_float_text_filed,
    make_solidable_checkbox,
    make_REMOVE_counter_button,
    make_ADD_counter_button,
    CustomText,
    customDivider,
    CustomExecuteButton,
    CustomReactiveText,
    CustomRadio,
    CustomReactiveCheckbox,
)
from niconavi_app.tools.tools import switch_tab_index
from niconavi_app.components.log_view import update_logs
from niconavi_app.components.page_tab.tabs.reset_onclick import (
    reset_onclick_grain_boundary_button,
    reset_onclick_grain_analyze_button,
)

from niconavi_app.components.labeling_app.labeling_controller import LabelingController
from typing import Callable, overload, Optional, cast
import flet as ft
from flet import Page
from result import Ok, Err, Result, is_ok, is_err
from logging import getLogger, Logger
import numpy as np
from niconavi_app.niconavi.image.type import RGBPicture
import cv2

# from gui.src.components.labeling_app.reset import reset_filter_tab

from niconavi_app.components.page_tab.tabs.merge_component import (
    make_code_input,
    make_continue_button,
    make_merge_button,
    make_reset_button,
)
import niconavi_app.niconavi.run_all as po
from niconavi_app.niconavi.angle_map_boundary import (
    create_shock_filter_iterator,
    fill_dark_boundaries,
    grain_boundary_from_angle_labels,
    make_theta_phi_angle_info,
    segment_angle_map,
)
from niconavi_app.niconavi.tools.str_parser import (
    parse_int,
    parse_larger_than_0,
    parse_larger_than_1,
)
from niconavi_app.components.labeling_app.reset import reset_filter_tab
import traceback


def select_grain_tab_image(stores: Stores, button_index: int) -> None:
    stores.ui.display_common_image_view.set(False)
    stores.ui.selected_button_at_grain_tab.set(button_index)


def reset_angle_map_workflow(stores: Stores) -> None:
    raw_maps = stores.computation_result.raw_maps.get()
    if raw_maps is None:
        return

    angle_map_info = make_theta_phi_angle_info(raw_maps)
    stores.ui.map_tab.angle_map_info.set(angle_map_info)
    stores.ui.map_tab.angle_map_display.set(angle_map_info["angle_map_display"])
    stores.ui.map_tab.shock_filter_iterator.set(create_shock_filter_iterator(angle_map_info))
    stores.ui.map_tab.cleaning_count.set(0)
    stores.ui.map_tab.fill_boundary_count.set(0.0)
    stores.ui.map_tab.segmentation_angle.set(10)
    stores.ui.map_tab.segmentation_done.set(False)
    stores.ui.map_tab.fill_boundary_started.set(False)
    stores.ui.map_tab.boundary_registered.set(False)
    stores.ui.display_grain_boundary.set(False)
    stores.computation_result.grain_map.set(None)
    stores.computation_result.grain_map_original.set(None)
    stores.computation_result.grain_boundary.set(None)
    stores.computation_result.grain_boundary_original.set(None)
    stores.computation_result.grain_map_with_boundary.set(None)
    select_grain_tab_image(stores, 21)


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


def cleaning_button_click(stores: Stores, e: ft.ControlEvent, *, logger: Logger) -> None:
    try:
        update_progress_bar(None, stores)
        iterator = stores.ui.map_tab.shock_filter_iterator.get()
        if iterator is None:
            reset_angle_map_workflow(stores)
            iterator = stores.ui.map_tab.shock_filter_iterator.get()
        if iterator is None:
            update_progress_bar(0.0, stores)
            return

        angle_map_info = next(iterator)
        stores.ui.map_tab.angle_map_info.set(angle_map_info)
        stores.ui.map_tab.angle_map_display.set(angle_map_info["angle_map_display"])
        stores.ui.map_tab.cleaning_count.set(stores.ui.map_tab.cleaning_count.get() + 1)
        select_grain_tab_image(stores, 21)
        update_progress_bar(0.0, stores)
    except Exception:
        update_logs(stores, ("Failed to clean angle map.", "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def segmentation_button_click(stores: Stores, e: ft.ControlEvent, *, logger: Logger) -> None:
    try:
        update_progress_bar(None, stores)
        angle_map_info = stores.ui.map_tab.angle_map_info.get()
        if angle_map_info is None:
            reset_angle_map_workflow(stores)
            angle_map_info = stores.ui.map_tab.angle_map_info.get()
        if angle_map_info is None:
            update_progress_bar(0.0, stores)
            return

        angle_map_info = segment_angle_map(
            angle_map_info,
            delta_euler_thresh=stores.ui.map_tab.segmentation_angle.get(),
        )
        stores.ui.map_tab.angle_map_info.set(angle_map_info)
        stores.ui.map_tab.angle_map_display.set(angle_map_info["angle_map_display"])
        stores.ui.map_tab.segmentation_done.set(True)
        stores.ui.map_tab.fill_boundary_started.set(False)
        select_grain_tab_image(stores, 21)
        update_progress_bar(0.0, stores)
    except Exception:
        update_logs(stores, ("Failed to segment angle map.", "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def fill_boundary_button_click(stores: Stores, e: ft.ControlEvent, *, logger: Logger) -> None:
    try:
        update_progress_bar(None, stores)
        angle_map_info = stores.ui.map_tab.angle_map_info.get()
        if angle_map_info is None:
            update_progress_bar(0.0, stores)
            return

        next_count = stores.ui.map_tab.fill_boundary_count.get() + 0.5
        angle_map_info = fill_dark_boundaries(
            angle_map_info,
            dark_l_thresh=15,
            branch_width_thresh=next_count,
            max_iterations=3,
            fixed_skeleton_once=False,
        )
        stores.ui.map_tab.angle_map_info.set(angle_map_info)
        stores.ui.map_tab.angle_map_display.set(angle_map_info["angle_map_display"])
        stores.ui.map_tab.fill_boundary_count.set(next_count)
        stores.ui.map_tab.fill_boundary_started.set(True)
        select_grain_tab_image(stores, 21)
        update_progress_bar(0.0, stores)
    except Exception:
        update_logs(stores, ("Failed to fill dark boundaries.", "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def ok_button_click(stores: Stores, e: ft.ControlEvent, *, logger: Logger) -> None:
    try:
        update_progress_bar(None, stores)
        angle_map_info = stores.ui.map_tab.angle_map_info.get()
        if angle_map_info is None:
            update_progress_bar(0.0, stores)
            return

        grain_map, grain_boundary = grain_boundary_from_angle_labels(angle_map_info)
        grain_map = grain_map.astype(np.int32) + 1
        grain_map_with_boundary = grain_map.copy()
        grain_map_with_boundary[grain_boundary] = 0

        stores.computation_result.grain_map.set(grain_map)
        stores.computation_result.grain_map_original.set(grain_map.copy())
        stores.computation_result.grain_boundary.set(grain_boundary)
        stores.computation_result.grain_boundary_original.set(grain_boundary.copy())
        stores.computation_result.grain_map_with_boundary.set(grain_map_with_boundary)
        stores.ui.display_grain_boundary.set(True)
        stores.ui.map_tab.boundary_registered.set(True)
        select_grain_tab_image(stores, 7)
        update_logs(stores, ("Angle-map grain boundaries registered.", "ok"))
        update_progress_bar(0.0, stores)
    except Exception:
        update_logs(stores, ("Failed to register grain boundaries.", "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def reset_angle_map_button_click(
    stores: Stores,
    e: ft.ControlEvent,
    *,
    logger: Logger,
) -> None:
    try:
        update_progress_bar(None, stores)
        reset_angle_map_workflow(stores)
        update_progress_bar(0.0, stores)
    except Exception:
        update_logs(stores, ("Failed to reset angle map.", "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def make_map_action_button(
    text: str,
    *,
    visible: ReactiveState[bool],
    enabled: ReactiveState[bool],
    on_click: Callable[[ft.ControlEvent], None],
) -> ReactiveElevatedButton:
    button = ReactiveElevatedButton(
        text,
        visible=visible,
        bgcolor=ReactiveState(
            lambda: (
                ft.Colors.LIGHT_GREEN_700
                if enabled.get()
                else ft.Colors.BLUE_GREY_700
            ),
            [enabled],
        ),
        on_click=lambda e: on_click(e) if enabled.get() else None,
    )
    button.height = 30
    button.content_padding = ft.padding.only(left=10, top=3, bottom=3)
    button.color = ft.Colors.WHITE
    button.style = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=5))
    return button


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

        segmentation_angle = make_reactive_float_text_filed(
            stores,
            stores.ui.map_tab.segmentation_angle,
            parse_int,
            accept_None=False,
        )

        has_angle_map_source = ReactiveState(
            lambda: stores.computation_result.raw_maps.get() is not None,
            [stores.computation_result.raw_maps],
        )
        can_use_angle_map = ReactiveState(
            lambda: (
                stores.ui.computing_is_stop.get()
                and stores.computation_result.raw_maps.get() is not None
            ),
            [stores.ui.computing_is_stop, stores.computation_result.raw_maps],
        )
        cleaning_enabled = ReactiveState(
            lambda: can_use_angle_map.get()
            and not stores.ui.map_tab.segmentation_done.get(),
            [can_use_angle_map, stores.ui.map_tab.segmentation_done],
        )
        segmentation_enabled = ReactiveState(
            lambda: can_use_angle_map.get()
            and not stores.ui.map_tab.boundary_registered.get()
            and not stores.ui.map_tab.fill_boundary_started.get(),
            [
                can_use_angle_map,
                stores.ui.map_tab.boundary_registered,
                stores.ui.map_tab.fill_boundary_started,
            ],
        )
        fill_enabled = ReactiveState(
            lambda: can_use_angle_map.get()
            and stores.ui.map_tab.segmentation_done.get()
            and not stores.ui.map_tab.boundary_registered.get(),
            [
                can_use_angle_map,
                stores.ui.map_tab.segmentation_done,
                stores.ui.map_tab.boundary_registered,
            ],
        )
        ok_enabled = ReactiveState(
            lambda: can_use_angle_map.get()
            and stores.ui.map_tab.segmentation_done.get()
            and not stores.ui.map_tab.boundary_registered.get(),
            [
                can_use_angle_map,
                stores.ui.map_tab.segmentation_done,
                stores.ui.map_tab.boundary_registered,
            ],
        )
        continue_enabled = ReactiveState(
            lambda: stores.ui.computing_is_stop.get()
            and stores.ui.map_tab.boundary_registered.get(),
            [stores.ui.computing_is_stop, stores.ui.map_tab.boundary_registered],
        )
        reset_enabled = ReactiveState(
            lambda: can_use_angle_map.get(),
            [can_use_angle_map],
        )

        content = ft.Column(
            [
                ft.Row(
                    [
                        make_map_action_button(
                            "Cleaning",
                            on_click=lambda e: cleaning_button_click(
                                stores, e, logger=logger
                            ),
                            visible=has_angle_map_source,
                            enabled=cleaning_enabled,
                        ),
                        CustomReactiveText(
                            ReactiveState(
                                lambda: str(stores.ui.map_tab.cleaning_count.get()),
                                [stores.ui.map_tab.cleaning_count],
                            ),
                            visible=has_angle_map_source,
                        ),
                    ]
                ),
                ft.Row(
                    [
                        make_map_action_button(
                            "Segmentation",
                            on_click=lambda e: segmentation_button_click(
                                stores, e, logger=logger
                            ),
                            visible=has_angle_map_source,
                            enabled=segmentation_enabled,
                        ),
                        CustomText("angle:"),
                        segmentation_angle,
                    ]
                ),
                ft.Row(
                    [
                        make_map_action_button(
                            "Fill boundary",
                            on_click=lambda e: fill_boundary_button_click(
                                stores, e, logger=logger
                            ),
                            visible=has_angle_map_source,
                            enabled=fill_enabled,
                        ),
                        CustomReactiveText(
                            ReactiveState(
                                lambda: f"{stores.ui.map_tab.fill_boundary_count.get():g}",
                                [stores.ui.map_tab.fill_boundary_count],
                            ),
                            visible=has_angle_map_source,
                        ),
                    ]
                ),
                ft.Row(
                    [
                        make_map_action_button(
                            "OK",
                            on_click=lambda e: ok_button_click(stores, e, logger=logger),
                            visible=has_angle_map_source,
                            enabled=ok_enabled,
                        ),
                        make_map_action_button(
                            "Reset",
                            on_click=lambda e: reset_angle_map_button_click(
                                stores, e, logger=logger
                            ),
                            visible=has_angle_map_source,
                            enabled=reset_enabled,
                        ),
                    ]
                ),
                ft.Row(
                    [
                        make_map_action_button(
                            "▶ Continue",
                            on_click=lambda e: continue_button_click(
                                stores, e, logger=logger
                            ),
                            visible=has_angle_map_source,
                            enabled=continue_enabled,
                        ),
                    ]
                ),
            ],
            scroll=ft.ScrollMode.ADAPTIVE,
        )

        self.content = content
