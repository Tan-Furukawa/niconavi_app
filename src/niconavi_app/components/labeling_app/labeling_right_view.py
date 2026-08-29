from __future__ import annotations

from typing import Optional

import flet as ft

from niconavi_app.components.labeling_app.label_controls import LabelSelectionPane
from niconavi_app.components.labeling_app.labeling_controller import LabelingController
from niconavi_app.stores import Stores
from niconavi_app.components.labeling_app.ui_theme import (
    BACKGROUND_COLOR,
    TEXT_COLOR,
    apply_border,
)
from niconavi_app.components.common_component import (
    CustomExecuteButton,
    confirm_discard_downstream,
    FILTER_TAB_STAGE,
)
from niconavi_app.components.log_view import update_logs


def _maybe_update(control: ft.Control) -> None:
    if getattr(control, "page", None) is not None:
        control.update()


def create_labeling_right_container(
    stores: Stores,
    controller: LabelingController,
    label_selection: LabelSelectionPane,
    image_width: Optional[int] = None,
    page: Optional[ft.Page] = None,
) -> ft.Container:
    labeling = stores.labeling

    def section_title(text: str) -> ft.Text:
        return ft.Text(text, color=TEXT_COLOR, weight=ft.FontWeight.W_900)

    status_text = ft.Text(value=labeling.status_text.get(), color=TEXT_COLOR, size=12)
    last_action_text = ft.Text(value=labeling.last_action_text.get(), color=TEXT_COLOR, size=12)
    labeled_stats_text = ft.Text(
        value=labeling.labeled_stats_text.get(), color=TEXT_COLOR, size=12
    )
    prediction_stats_text = ft.Text(
        value=labeling.prediction_stats_text.get(),
        color=TEXT_COLOR,
        size=12,
    )

    def sync_status() -> None:
        status_text.value = labeling.status_text.get()
        _maybe_update(status_text)

    def sync_last_action() -> None:
        last_action_text.value = labeling.last_action_text.get()
        _maybe_update(last_action_text)

    def sync_labeled_stats() -> None:
        labeled_stats_text.value = labeling.labeled_stats_text.get()
        _maybe_update(labeled_stats_text)

    def sync_prediction_stats() -> None:
        prediction_stats_text.value = labeling.prediction_stats_text.get()
        _maybe_update(prediction_stats_text)

    labeling.status_text.bind(sync_status)
    labeling.last_action_text.bind(sync_last_action)
    labeling.labeled_stats_text.bind(sync_labeled_stats)
    labeling.prediction_stats_text.bind(sync_prediction_stats)

    boundary_checkbox = ft.Checkbox(
        label=ft.Text("Boundaries", color=ft.Colors.WHITE),
        value=labeling.show_boundaries.get(),
        on_change=controller.handle_show_boundaries_change,
        fill_color=ft.Colors.BLUE_100,
        check_color=ft.Colors.BLACK,
    )
    boundary_checkbox.dense = True

    def sync_boundary_checkbox() -> None:
        boundary_checkbox.value = labeling.show_boundaries.get()
        _maybe_update(boundary_checkbox)

    def sync_boundary_checkbox_enabled() -> None:
        boundary_checkbox.disabled = not labeling._loaded.get()
        _maybe_update(boundary_checkbox)

    labeling.show_boundaries.bind(sync_boundary_checkbox)
    labeling._loaded.bind(sync_boundary_checkbox_enabled)

    training_checkbox = ft.Checkbox(
        label=ft.Text("Training boxes", color=ft.Colors.WHITE),
        value=labeling.show_training_boxes.get(),
        on_change=controller.handle_show_training_boxes_change,
        label_style=ft.TextStyle(color=TEXT_COLOR),
        fill_color=ft.Colors.BLUE_100,
        check_color=ft.Colors.BLACK,
    )
    training_checkbox.dense = True

    def sync_training_checkbox() -> None:
        training_checkbox.value = labeling.show_training_boxes.get()
        _maybe_update(training_checkbox)

    def sync_training_checkbox_enabled() -> None:
        training_checkbox.disabled = not labeling._loaded.get()
        _maybe_update(training_checkbox)

    labeling.show_training_boxes.bind(sync_training_checkbox)
    labeling._loaded.bind(sync_training_checkbox_enabled)

    def make_feature_checkbox(feature_name: str, label: str) -> ft.Checkbox:
        state = getattr(labeling, f"use_{feature_name}_features")
        checkbox = ft.Checkbox(
            label=ft.Text(label, color=ft.Colors.WHITE),
            value=state.get(),
            on_change=lambda e: controller.handle_feature_checkbox_change(
                feature_name, e
            ),
            fill_color=ft.Colors.BLUE_100,
            check_color=ft.Colors.BLACK,
        )
        checkbox.dense = True

        def sync_checkbox() -> None:
            checkbox.value = state.get()
            _maybe_update(checkbox)

        def sync_checkbox_enabled() -> None:
            checkbox.disabled = not labeling._loaded.get()
            _maybe_update(checkbox)

        state.bind(sync_checkbox)
        labeling._loaded.bind(sync_checkbox_enabled)
        sync_checkbox()
        sync_checkbox_enabled()
        return checkbox

    feature_checkbox_row = ft.Row(
        controls=[
            make_feature_checkbox("color", "Color"),
            make_feature_checkbox("shape", "Shape"),
            make_feature_checkbox("position", "Position"),
        ],
        wrap=False,
        spacing=0,
        run_spacing=0,
        visible=False,
    )
    feature_toggle_icon = ft.IconButton(
        icon=ft.Icons.ADD,
        icon_color=ft.Colors.BLUE_100,
        tooltip="Feature settings",
    )

    def toggle_feature_controls(_e) -> None:
        feature_checkbox_row.visible = not feature_checkbox_row.visible
        feature_toggle_icon.icon = (
            ft.Icons.REMOVE if feature_checkbox_row.visible else ft.Icons.ADD
        )
        _maybe_update(feature_checkbox_row)
        _maybe_update(feature_toggle_icon)

    feature_toggle_icon.on_click = toggle_feature_controls
    feature_controls = ft.Column(
        controls=[
            ft.Row(
                controls=[
                    section_title("Learning features"),
                    feature_toggle_icon,
                ],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            feature_checkbox_row,
        ],
        spacing=0,
    )

    overlay_value_text = ft.Text(
        value=f"Opacity: {labeling.overlay_alpha.get():.2f}",
        color=TEXT_COLOR,
    )
    overlay_slider = ft.Slider(
        min=0.0,
        max=1.0,
        divisions=20,
        value=labeling.overlay_alpha.get(),
        on_change=controller.handle_overlay_opacity_change,
        disabled=not labeling._loaded.get(),
    )
    overlay_slider.expand = True

    def sync_overlay_controls() -> None:
        value = float(labeling.overlay_alpha.get())
        overlay_value_text.value = f"Opacity: {value:.2f}"
        overlay_slider.value = value
        _maybe_update(overlay_value_text)
        _maybe_update(overlay_slider)

    def sync_overlay_enabled() -> None:
        overlay_slider.disabled = not labeling._loaded.get()
        _maybe_update(overlay_slider)

    labeling.overlay_alpha.bind(sync_overlay_controls)
    labeling._loaded.bind(sync_overlay_enabled)

    overlay_controls = ft.Column(
        controls=[
            ft.Row(
                controls=[section_title("Display"), overlay_value_text],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            ft.Row(
                controls=[boundary_checkbox, training_checkbox],
                wrap=False,
                spacing=0,
            ),
            overlay_slider,
        ],
        spacing=4,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    label_selection_container = ft.Container(
        content=label_selection.as_container(),
        bgcolor=BACKGROUND_COLOR,
    )

    shared_controls = stores.labeling_computation_result.setdefault(
        "shared_controls", {}
    )
    label_registry = shared_controls.setdefault("label_selection", {})
    label_registry["labeling_selection"] = label_selection_container

    # has_labels = ReactiveState(
    #     lambda: any(stores.labeling.labels.get().values()),
    #     [stores.labeling.labels],
    # )

    def on_done_clicked(_e) -> None:
        stats = labeling.labeled_stats_text.get()
        if stats is not None and stats.startswith("Labeled samples:"):
            try:
                count = int(stats.split(":")[1].split("(")[0].strip())
            except Exception:
                count = 0
        else:
            count = 0
        if count <= 0:
            update_logs(stores, ("No labeled samples to save.", "err"))
            return

        confirm_discard_downstream(
            page,
            stores,
            FILTER_TAB_STAGE,
            "Save filter labels",
            controller.finish_labeling,
        )

    done_button = CustomExecuteButton(
        text="Done",
        icon=ft.Icons.CHECK,
        on_click=on_done_clicked,
    )

    def sync_done_button_enabled() -> None:
        stats = labeling.labeled_stats_text.get()
        if stats is not None and stats.startswith("Labeled samples:"):
            try:
                count = int(stats.split(":")[1].split("(")[0].strip())
            except Exception:
                count = 0
        else:
            count = 0
        # done_button.disabled = count <= 0
        _maybe_update(done_button)

    labeling.labeled_stats_text.bind(sync_done_button_enabled)

    sync_status()
    sync_last_action()
    sync_labeled_stats()
    sync_prediction_stats()
    sync_boundary_checkbox()
    sync_boundary_checkbox_enabled()
    sync_training_checkbox()
    sync_training_checkbox_enabled()
    sync_overlay_controls()
    sync_overlay_enabled()
    sync_done_button_enabled()

    controls_column = ft.Column(
        controls=[
            status_text,
            ft.Divider(),
            overlay_controls,
            feature_controls,
            ft.Divider(),
            section_title("Labels"),
            label_selection_container,
            ft.Divider(),
            section_title("Result"),
            last_action_text,
            labeled_stats_text,
            prediction_stats_text,
            ft.Row([done_button], alignment=ft.MainAxisAlignment.START),
        ],
        spacing=8,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    container = apply_border(
        ft.Container(
            content=controls_column,
            width=320,
            padding=12,
            bgcolor=BACKGROUND_COLOR,
        )
    )
    ui_state = stores.labeling_computation_result.setdefault("ui_state", {})
    ui_state["right_panel_width"] = container.width or 0
    if image_width is not None:
        ui_state["requested_left_image_width"] = int(image_width)
    else:
        ui_state.pop("requested_left_image_width", None)
    ui_state.setdefault("row_spacing", 16)
    return container
