from stores import (
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
)
from reactive_state import (
    ReactiveElevatedButton,
    ReactiveCheckbox,
)
from state import State, ReactiveState, is_not_None_state
from components.log_view import update_logs
from components.progress_bar import update_progress_bar
from components.common_component import ReactiveCodeTextInput, CustomText
from tools.tools import switch_tab_index
from components.log_view import update_logs

from typing import Callable, overload
import flet as ft
from flet import Page
from result import Ok, Err, Result, is_ok, is_err

from logging import getLogger, Logger

import niconavi.run_all as po
import traceback
from copy import deepcopy
import traceback


def continue_button_click(
    stores: Stores, e: ft.ControlEvent, *, logger: Logger
) -> None:
    try:

        update_progress_bar(None, stores)
        r = as_ComputationResult(stores.computation_result)
        res = po.analyze_grain_list(
            r,
            progress_callback=lambda p: update_progress_bar(p, stores),
        )

        update_progress_bar(0.0, stores)
        update_logs(stores, ("Merge completed.", "ok"))
        save_in_ComputationResultState(res, stores)
        switch_tab_index(stores, 4)

    except Exception as e:
        update_logs(stores, (str(e), "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def reset_button_click(stores: Stores, e: ft.ControlEvent, *, logger: Logger) -> None:
    try:

        r = as_ComputationResult(stores.computation_result)
        r.grain_boundary = deepcopy(r.grain_boundary_original)
        r.grain_map = deepcopy(r.grain_map_original)
        save_in_ComputationResultState(r, stores)

    except Exception as e:
        update_logs(stores, ("Failed to reset the merge state.", "err"))
        update_logs(stores, (str(e), "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def merge_button_click(stores: Stores, e: ft.ControlEvent, *, logger: Logger) -> None:
    try:

        update_progress_bar(None, stores)
        r = as_ComputationResult(stores.computation_result)
        # once reset result
        r.grain_boundary = deepcopy(r.grain_boundary_original)
        r.grain_map = deepcopy(r.grain_map_original)
        res = po.grain_merge(r)
        update_progress_bar(0.0, stores)
        update_logs(stores, ("Merge completed.", "ok"))
        save_in_ComputationResultState(res, stores)

    except Exception as e:
        update_logs(stores, ("Merge failed.", "err"))
        update_logs(stores, (str(e), "err"))
        update_progress_bar(0.0, stores)
        traceback.print_exc()
        logger.error(traceback.format_exc())


def get_merge_code(stores: Stores) -> str:
    code_state = stores.computation_result.grain_marge_code
    code = code_state.get()
    return "" if code is None else code


def make_code_input(stores: Stores) -> ReactiveCodeTextInput:
    code_input = ReactiveState(
        lambda: get_merge_code(stores),
        [stores.computation_result.grain_classification_code],
    )

    marge_code = ReactiveCodeTextInput(
        value=code_input,
        on_change=lambda e: stores.computation_result.grain_marge_code.set(
            e.control.value
        ),
    )

    return marge_code


def make_merge_button(stores: Stores, *, logger: Logger) -> ReactiveElevatedButton:
    merge_button = ReactiveElevatedButton(
        "Merge",
        on_click=lambda e: merge_button_click(stores, e, logger=logger),
    )
    return merge_button


def make_continue_button(stores: Stores, *, logger: Logger) -> ReactiveElevatedButton:
    continue_button = ReactiveElevatedButton(
        "Continue",
        visible=True,
        on_click=lambda e: continue_button_click(stores, e, logger=logger),
    )
    return continue_button


def make_reset_button(stores: Stores, *, logger: Logger) -> ReactiveElevatedButton:
    reset_button = ReactiveElevatedButton(
        "Reset",
        on_click=lambda e: reset_button_click(stores, e, logger=logger),
    )
    return reset_button


class MergeTab(ft.Container):
    def __init__(
        self,
        page: Page,
        stores: Stores,
    ):

        super().__init__()

        logger = getLogger("niconavi").getChild(__name__)

        self.padding = stores.appearance.tab_padding

        marge_code = make_code_input(stores)
        merge_button = make_merge_button(stores, logger=logger)
        continue_button = make_continue_button(stores, logger=logger)
        reset_button = make_reset_button(stores, logger=logger)

        checkbox_visible = ReactiveState(
            lambda: stores.computation_result.grain_map.get() is not None,
            [stores.computation_result.grain_map],
        )

        grain_boundary_checkbox = ReactiveCheckbox(
            label="Show grain boundaries",
            value=stores.ui.display_grain_boundary,
            visible=checkbox_visible,
            on_change=lambda e: stores.ui.display_grain_boundary.set(e.control.value),
        )

        content = ft.Column(
            [
                ft.Column(
                    [
                        CustomText("Merge code"),
                        marge_code,
                    ]
                ),
                ft.Divider(),
                grain_boundary_checkbox,
                ft.Row([merge_button, reset_button]),
                continue_button,
            ],
            scroll=ft.ScrollMode.ADAPTIVE,
        )

        self.content = content
