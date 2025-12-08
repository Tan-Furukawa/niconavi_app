from __future__ import annotations

from typing import Optional

import flet as ft

from components.labeling_app.labeling_controller import LabelingController
from stores import Stores
from components.labeling_app.ui_theme import BACKGROUND_COLOR, BUTTON_STYLE, TEXT_COLOR, apply_border


def _maybe_update(control: ft.Control) -> None:
    if getattr(control, "page", None) is not None:
        control.update()


def create_labeling_left_container(
    stores: Stores, controller: LabelingController, page: Optional[ft.Page] = None
) -> ft.Container:
    labeling = stores.labeling

    load_status_text = ft.Text(
        value=labeling.load_status_text.get(),
        color=TEXT_COLOR,
    )

    def sync_load_status() -> None:
        load_status_text.value = labeling.load_status_text.get()
        _maybe_update(load_status_text)

    labeling.load_status_text.bind(sync_load_status)

    # load_button = ft.ElevatedButton(
    #     text="Load Data",
    #     icon=ft.Icons.DOWNLOAD,
    #     on_click=controller.on_load_clicked,
    #     disabled=labeling.load_button_disabled.get(),
    #     icon_color=TEXT_COLOR,
    #     style=BUTTON_STYLE,
    # )

    # reset_button = ft.ElevatedButton(
    #     text="Reset",
    #     icon=ft.Icons.REPLAY,
    #     on_click=controller.reset_application,
    #     disabled=not labeling._loaded.get(),
    #     icon_color=TEXT_COLOR,
    #     style=BUTTON_STYLE,
    # )

    # def sync_load_button() -> None:
    #     load_button.disabled = labeling.load_button_disabled.get()
    #     _maybe_update(load_button)

    # labeling.load_button_disabled.bind(sync_load_button)

    # def sync_reset_button() -> None:
    #     reset_button.disabled = not labeling._loaded.get()
    #     _maybe_update(reset_button)

    # labeling._loaded.bind(sync_reset_button)

    image = ft.Image(
        src_base64=labeling.image_src_base64.get() or "",
        width=labeling.image_display_width.get() or None,
        height=labeling.image_display_height.get() or None,
        fit=ft.ImageFit.CONTAIN,
    )

    image_holder = ft.Container(
        content=image,
        width=labeling.image_display_width.get() or None,
        height=labeling.image_display_height.get() or None,
        alignment=ft.alignment.center,
        bgcolor=BACKGROUND_COLOR,
    )

    cursor = getattr(ft.MouseCursor, "CROSSHAIR", ft.MouseCursor.PRECISE)
    gesture = ft.GestureDetector(
        mouse_cursor=cursor,
        on_tap_down=controller.on_image_tap,
        content=image_holder,
    )

    image_container = apply_border(
        ft.Container(
            content=gesture,
            alignment=ft.alignment.center,
            width=labeling.image_display_width.get() or None,
            height=labeling.image_display_height.get() or None,
            bgcolor=BACKGROUND_COLOR,
        )
    )

    horizontal_scroll = ft.Row(
        controls=[image_container],
        alignment=ft.MainAxisAlignment.CENTER,
        scroll=ft.ScrollMode.AUTO,
    )

    scroll_wrapper = ft.Column(
        controls=[horizontal_scroll],
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    def sync_image() -> None:
        image.src_base64 = labeling.image_src_base64.get() or ""
        display_width = labeling.image_display_width.get()
        display_height = labeling.image_display_height.get()
        image.width = display_width or None
        image.height = display_height or None
        image_holder.width = display_width or None
        image_holder.height = display_height or None
        image_container.width = display_width or None
        image_container.height = display_height or None
        record_rendered_dimensions(display_width, display_height)
        _maybe_update(image)
        _maybe_update(image_holder)
        _maybe_update(gesture)
        _maybe_update(image_container)
        _maybe_update(horizontal_scroll)
        _maybe_update(scroll_wrapper)

    labeling.image_src_base64.bind(sync_image)
    labeling.image_display_width.bind(sync_image)
    labeling.image_display_height.bind(sync_image)

    def record_rendered_dimensions(width: float | None, height: float | None) -> None:
        ui_state = stores.labeling_computation_result.setdefault("ui_state", {})
        if width is not None:
            ui_state["rendered_image_width"] = float(width)
        if height is not None:
            ui_state["rendered_image_height"] = float(height)

    def handle_image_resize(event) -> None:
        record_rendered_dimensions(getattr(event, "width", None), getattr(event, "height", None))

    # image_container.on_resize = handle_image_resize
    image_container.on_resize = handle_image_resize
    record_rendered_dimensions(image_container.width, image_container.height)

    allow_scale_up: bool = False

    def compute_target_width() -> Optional[int]:
        nonlocal allow_scale_up
        ui_state = stores.labeling_computation_result.get("ui_state", {})
        requested_width = ui_state.get("requested_left_image_width")
        try:
            requested_width_value = float(requested_width) if requested_width else None
        except (TypeError, ValueError):
            requested_width_value = None

        if requested_width_value is not None and requested_width_value > 0:
            allow_scale_up = True
            return max(int(requested_width_value), 1)

        allow_scale_up = False
        existing_width = labeling.image_display_width.get()
        if existing_width:
            return int(existing_width)
        return None

    def update_display_dimensions() -> None:
        if not labeling._loaded.get():
            return
        max_width = compute_target_width()
        if max_width is None or max_width <= 0:
            return
        controller.update_display_dimensions(
            max_width=max_width, allow_scale_up=allow_scale_up
        )

    def sync_visibility() -> None:
        is_loaded = labeling._loaded.get()
        image_container.visible = is_loaded
        horizontal_scroll.visible = is_loaded
        scroll_wrapper.visible = is_loaded
        if is_loaded:
            update_display_dimensions()
        _maybe_update(image_container)
        _maybe_update(horizontal_scroll)
        _maybe_update(scroll_wrapper)

    labeling._loaded.bind(sync_visibility)

    app_area = ft.Container(
        expand=True,
        alignment=ft.alignment.center,
        bgcolor=BACKGROUND_COLOR,
    )

    def sync_app_area() -> None:
        app_area.content = scroll_wrapper if labeling._loaded.get() else None
        _maybe_update(app_area)

    labeling._loaded.bind(sync_app_area)

    if page is not None:
        existing_resize_handler = page.on_resized

        def handle_resize(event) -> None:
            if callable(existing_resize_handler):
                existing_resize_handler(event)
            update_display_dimensions()

        page.on_resized = handle_resize

    sync_load_status()
    # sync_load_button()
    sync_image()
    sync_visibility()
    sync_app_area()

    if page is not None:
        gesture.data = {"controller": controller}

    shared_controls = stores.labeling_computation_result.setdefault("shared_controls", {})
    click_registry = shared_controls.setdefault("click_area", {})
    click_registry["labeling_image"] = image_container

    # action_row = ft.Row(controls=[load_button, reset_button], spacing=8)
    # action_row = ft.Row(controls=[load_button], spacing=8)

    # load_controls = ft.Column(
    #     controls=[load_status_text, action_row],
    #     spacing=12,
    #     horizontal_alignment=ft.CrossAxisAlignment.START,
    # )

    root_column = ft.Column(
        controls=[app_area],
        spacing=12,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    # root_column = ft.Column(
    #     controls=[load_controls, app_area],
    #     spacing=12,
    #     horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    # )

    container = apply_border(
        ft.Container(content=root_column, expand=True, padding=12, bgcolor=BACKGROUND_COLOR)
    )
    return container
