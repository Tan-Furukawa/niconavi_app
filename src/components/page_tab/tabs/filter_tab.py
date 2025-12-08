from stores import (
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
)
from reactive_state import (
    ReactiveElevatedButton,
)
from state import ReactiveState
from components.log_view import update_logs
from components.progress_bar import update_progress_bar
from components.common_component import (
    ReactiveCodeTextInput,
    CustomExecuteButton,
    CustomText,
)
from components.log_view import update_logs
from scipy.ndimage import binary_erosion
import flet as ft
from flet import Page
from typing import Optional

from logging import getLogger, Logger
import traceback
import numpy as np

import niconavi.run_all as po
from niconavi.tools.str_parser import parse_percent, parse_int, parse_odd_int
from components.page_tab.tabs.reset_onclick import reset_onclick_classify_button
from components.labeling_app.label_controls import LabelSelectionPane
from components.labeling_app.labeling_controller import LabelingController
from components.labeling_app.labeling_left_view import create_labeling_left_container
from components.labeling_app.labeling_right_view import create_labeling_right_container
from stores import Stores




def classify_button_click(
    stores: Stores, e: ft.ControlEvent, *, logger: Logger
) -> None:
    try:

        update_progress_bar(None, stores)
        r = as_ComputationResult(stores.computation_result)
        r = reset_onclick_classify_button(r)
        save_in_ComputationResultState(r, stores)

        r = po.add_random_colors_to_user_code(r)
        save_in_ComputationResultState(r, stores)

        r = po.grain_segmentation(r)
        update_progress_bar(0.0, stores)
        update_logs(stores, ("Grain segmentation completed.", "ok"))
        save_in_ComputationResultState(r, stores)

    except Exception as e:
        traceback.print_exc()
        update_logs(stores, ("Grain classification failed.", "err"))
        update_logs(stores, (str(e), "err"))
        update_progress_bar(0.0, stores)
        logger.error(str(e))


class FilterTab(ft.Container):
    def __init__(
        self,
        page: Page,
        stores: Stores,
    ):

        super().__init__()

        logger = getLogger("niconavi").getChild(__name__)

        self.padding = stores.appearance.tab_padding


        controller = LabelingController(stores=stores)

        label_selection = LabelSelectionPane(
            on_add_label=controller.handle_label_added,
            on_remove_label=controller.handle_label_removed,
            on_select_label=controller.handle_label_selected,
            on_color_changed=controller.handle_label_color_changed,
        )

        controller.attach_label_selection(label_selection)

        right_container = create_labeling_right_container(
            stores=stores,
            controller=controller,
            label_selection=label_selection,
            page=page,
            image_width = 800,
        )

        content = ft.Column(
            [
                ft.Column([right_container]),
                ft.Divider(),
                # grain_boundary_checkbox,
                # classify_button,
            ],
            scroll=ft.ScrollMode.ADAPTIVE,
        )

        self.content = content
