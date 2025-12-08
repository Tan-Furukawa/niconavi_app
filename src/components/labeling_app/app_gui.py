from __future__ import annotations

import flet as ft

from components.labeling_app.label_controls import LabelSelectionPane
from components.labeling_app.labeling_controller import LabelingController
from components.labeling_app.labeling_left_view import create_labeling_left_container
from components.labeling_app.labeling_right_view import create_labeling_right_container
from stores import Stores


DEFAULT_LEFT_IMAGE_WIDTH = 1000


def configure_page(page: ft.Page) -> None:
    page.title = "Thin Section Labeling"
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO


def main(page: ft.Page) -> None:
    configure_page(page)

    stores = Stores()

    # 1
    # ==================================
    controller1 = LabelingController(stores=stores)

    label_selection = LabelSelectionPane(
        on_add_label=controller1.handle_label_added,
        on_remove_label=controller1.handle_label_removed,
        on_select_label=controller1.handle_label_selected,
        on_color_changed=controller1.handle_label_color_changed,
    )

    controller1.attach_label_selection(label_selection)

    left_container = create_labeling_left_container(
        stores=stores, controller=controller1, page=page
    )

    # 2
    # =============================--
    controller2 = LabelingController(stores=stores)

    label_selection2 = LabelSelectionPane(
        on_add_label=controller2.handle_label_added,
        on_remove_label=controller2.handle_label_removed,
        on_select_label=controller2.handle_label_selected,
        on_color_changed=controller2.handle_label_color_changed,
    )

    controller2.attach_label_selection(label_selection2)

    right_container = create_labeling_right_container(
        stores=stores,
        controller=controller2,
        label_selection=label_selection2,
        image_width=DEFAULT_LEFT_IMAGE_WIDTH,
        page=page,
    )

    layout = ft.Row(
        controls=[left_container, right_container],
        spacing=16,
        vertical_alignment=ft.CrossAxisAlignment.START,
        expand=True,
    )

    page.add(layout)


if __name__ == "__main__":
    ft.app(target=main)
