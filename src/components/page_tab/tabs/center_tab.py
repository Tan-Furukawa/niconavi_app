from stores import (
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
)
from components.page_tab.tabs.grain_tab import update_r_color_map_display
from tools.tools import switch_tab_index
from tools.parser import parse_float_larger_than_0, to_str
from reactive_state import ReactiveElevatedButton
from state import ReactiveState, StateProperty
from components.progress_bar import update_progress_bar
from components.common_component import (
    make_REMOVE_counter_button,
    make_ADD_counter_button,
    ReactiveFloatTextField,
    CustomExecuteButton,
    CustomText,
    make_reactive_float_text_filed,
)

from components.log_view import update_logs
from components.page_tab.tabs.reset_onclick import reset_onclick_center_button

from niconavi.tools.str_parser import (
    parse_int,
)

import flet as ft
from flet import Page

from logging import getLogger, Logger
from tools.error import exec_at_error
from niconavi.custom_error import (
    InvalidRotationDirection,
    RotatedAngleError,
    UnexpectedNoneType,
)
import niconavi.run_all as po
import numpy as np
from typing import Optional
from niconavi.image.type import RGBPicture


def _normalize_adjustment_value(value: float, minimum: float = 0.2, maximum: float = 2.0) -> float:
    return float(max(minimum, min(maximum, value)))


def _estimate_brightness_contrast(image: Optional[RGBPicture]) -> tuple[float, float]:
    if image is None:
        return 1.0, 1.0

    img = image.astype(np.float32)
    brightness = float(np.mean(img) / 128.0)
    contrast = float(np.std(img) / 64.0)
    return (
        _normalize_adjustment_value(brightness),
        _normalize_adjustment_value(contrast),
    )


def save_grain_tab_adjustments(stores: Stores, image: Optional[RGBPicture]) -> None:
    brightness, contrast = _estimate_brightness_contrast(image)
    stores.ui.grain_tab.slider_brightness.force_set(brightness)
    stores.ui.grain_tab.slider_contrast.force_set(contrast)
    update_r_color_map_display(stores)


def on_click_center_button(stores: Stores, *, logger: Logger) -> None:


    if stores.computation_result.pics.get() is None:
        exec_at_error(1004, stores, logger=logger)
        return
    try:
        update_progress_bar(None, stores)
        r = as_ComputationResult(stores.computation_result)
        r = reset_onclick_center_button(r)
        save_in_ComputationResultState(r, stores)

        update_logs(
            stores,
            ("Calculating rotation angles between images...", "msg"),
            logger=logger,
        )

        r = po.determine_rotation_angle(
            r, progress_callback=lambda p: update_progress_bar(p, stores)
        )

        try:
            update_logs(
                stores,
                    (
                    "Estimating extinction angles and colors at extinction angle + 45°...",
                    "msg",
                ),
                logger=logger,
            )

            r = po.make_raw_color_maps(
                r, progress_callback=lambda p: update_progress_bar(p, stores)
            )

            color_map: Optional[RGBPicture] = None
            if r.raw_maps is not None:
                color_map = r.raw_maps.get("R_color_map_display")
            save_grain_tab_adjustments(stores, color_map)

            update_logs(
                stores, ("Simulating retardation colors...", "msg"), logger=logger
            )
            r = po.make_retardation_color_chart(
                r, progress_callback=lambda p: update_progress_bar(p, stores)
            )

            update_logs(
                stores,
                ("Estimating retardation at extinction angle + 45°...", "msg"),
                logger=logger,
            )
            r = po.make_raw_R_maps(
                r, progress_callback=lambda p: update_progress_bar(p, stores)
            )

            im_tilt0 = r.tilt_image_info.tilt_image0_raw
            # im_tilt45 = r.tilt_image_info.tilt_image45_raw
            # if im_tilt0 is not None and im_tilt45 is not None:
            if im_tilt0 is not None:
                update_logs(
                    stores,
                    (
                        "Stacking focused frames from the tilted thin-section point-shift movie...",
                        "msg",
                    ),
                    logger=logger,
                )
                r = po.estimate_tilt_image_result(
                    r, progress_callback=lambda p: update_progress_bar(p, stores)
                )

            update_progress_bar(None, stores)
            update_logs(stores, ("Image processing completed.", "ok"), logger=logger)
            save_in_ComputationResultState(r, stores)
            update_r_color_map_display(stores)

            update_progress_bar(0.0, stores)
            stores.ui.progress.set(2)
            switch_tab_index(stores, 2, logger=logger)

        except Exception as e:
            exec_at_error(9004, stores, logger=logger)

    except RotatedAngleError as e:
        if str(e) == "XPL":
            exec_at_error(8004, stores, logger=logger)
        elif str(e) == "XPL+lambda":
            exec_at_error(8005, stores, logger=logger)
        else:
            exec_at_error(9999, stores, logger=logger)

    except InvalidRotationDirection as e:
        if str(e) == "Incorrect rotation direction of XPL movie.":
            exec_at_error(8002, stores, logger=logger)
        elif str(e) == "Incorrect rotation direction of XPL + λ-Plate movie.":
            exec_at_error(8003, stores, logger=logger)
        else:
            exec_at_error(9999, stores, logger=logger)

    except Exception as e:
        exec_at_error(9003, stores, logger=logger)


class CenterTab(ft.Container):
    def __init__(
        self,
        page: Page,
        stores: Stores,
    ):
        super().__init__()

        logger = getLogger("niconavi").getChild(__name__)

        self.padding = stores.appearance.tab_padding

        execute_button = CustomExecuteButton(
            "▶ continue",
            on_click=lambda e: on_click_center_button(stores, logger=logger),
            visible=ReactiveState(
                lambda: (
                    stores.ui.computing_is_stop.get()
                    and stores.computation_result.rotation_img.get() is not None
                    # and not stores.ui.once_start.get()
                ),
                [
                    stores.ui.computing_is_stop,
                    stores.computation_result.rotation_img,
                    stores.ui.once_start,
                ],
            ),
        )

        input_cx = make_reactive_float_text_filed(
            stores,
            stores.computation_result.center_int_x,
            parse_int,
            accept_None=False,
        )

        input_cy = make_reactive_float_text_filed(
            stores,
            stores.computation_result.center_int_y,
            parse_int,
            accept_None=False,
        )

        content = ft.Column(
            [
                ft.Column(
                    [
                        ft.Row(
                            [
                                CustomText("cx:"),
                                make_REMOVE_counter_button(
                                    stores, stores.computation_result.center_int_x
                                ),
                                input_cx,
                                make_ADD_counter_button(
                                    stores, stores.computation_result.center_int_x
                                ),
                            ]
                        ),
                        # ReactiveFloatTextField(
                        #     value=cx,
                        #     on_change=lambda e: on_change_cx(stores, e, logger=logger),
                        # ),
                        ft.Row(
                            [
                                CustomText("cy:"),
                                make_REMOVE_counter_button(
                                    stores, stores.computation_result.center_int_y
                                ),
                                input_cy,
                                make_ADD_counter_button(
                                    stores, stores.computation_result.center_int_y
                                ),
                            ]
                        ),
                    ],
                ),
                ft.Divider(),
                execute_button,
            ],
            scroll=ft.ScrollMode.ADAPTIVE,
        )

        self.content = content
