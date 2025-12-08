from __future__ import annotations

from typing import Callable, Dict, Optional

import flet as ft

LabelAddedCallback = Callable[[str], Optional[int]]
LabelRemovedCallback = Callable[[int], tuple[bool, Optional[int]]]
LabelSelectedCallback = Callable[[Optional[int]], None]
LabelColorChangedCallback = Callable[[int, str], None]

from components.labeling_app.ui_theme import (
    ICON_BUTTON_STYLE,
    TEXT_COLOR,
    TEXT_FIELD_STYLE,
    apply_border,
)
from components.common_component import CustomTextField


class LabelSelectionPane:
    """Right pane controls for managing dynamic label names and colors."""

    def __init__(
        self,
        on_add_label: LabelAddedCallback,
        on_remove_label: LabelRemovedCallback,
        on_select_label: LabelSelectedCallback,
        on_color_changed: LabelColorChangedCallback,
    ) -> None:
        self.on_add_label = on_add_label
        self.on_remove_label = on_remove_label
        self.on_select_label = on_select_label
        self.on_color_changed = on_color_changed

        self._rows: Dict[int, ft.Container] = {}
        self._radios: Dict[int, ft.Radio] = {}
        self._color_swatches: Dict[int, ft.Container] = {}
        self._label_colors: Dict[int, str] = {}
        self._color_picklists: Dict[int, ft.Container] = {}
        self._color_option_controls: Dict[int, list[ft.Container]] = {}
        self._active_color_class_id: Optional[int] = None

        self._color_columns = [
            ["#b71c1c", "#d32f2f", "#ef5350", "#ffcdd2"],
            ["#e65100", "#f57c00", "#ffb74d", "#ffe0b2"],
            ["#f9a825", "#fdd835", "#fff176", "#fff9c4"],
            ["#1b5e20", "#388e3c", "#81c784", "#c8e6c9"],
            ["#004d40", "#00796b", "#4db6ac", "#b2dfdb"],
            ["#0d47a1", "#1976d2", "#64b5f6", "#bbdefb"],
            ["#4a148c", "#6a1b9a", "#ab47bc", "#d1c4e9"],
            ["#000000", "#424242", "#757575", "#bdbdbd"],
        ]

        self.new_label_field = CustomTextField (
            hint_text="Label name",
            dense=True,
            expand=True,
            # **TEXT_FIELD_STYLE,
        )
        self.add_button = ft.IconButton(
            icon=ft.Icons.ADD_CIRCLE,
            tooltip="Add label",
            on_click=self._handle_add_clicked,
            icon_color=ft.Colors.BLUE_100,
            # icon_color=TEXT_COLOR,
            # style=ICON_BUTTON_STYLE,
        )
        self._input_row = ft.Row(
            controls=[self.new_label_field, self.add_button],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.END,
            spacing=8,
        )

        self._radio_column = ft.Column(spacing=6, tight=True)
        self.radio_group = ft.RadioGroup(
            value=None,
            on_change=self._handle_selection_changed,
            content=self._radio_column,
        )
        self.radio_group.visible = False

        self._layout = ft.Column(
            controls=[
                self.radio_group,
                self._input_row,
            ],
            spacing=12,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        self.container = apply_border(ft.Container(content=self._layout, padding=12))

    def as_container(self) -> ft.Container:
        return self.container

    def add_label(self, class_id: int, label_name: str, select: bool = True) -> None:
        if class_id in self._rows:
            self._update_label_text(class_id, label_name)
            if select:
                self._set_selected(class_id)
            return

        radio = ft.Radio(
            value=str(class_id),
            label=label_name,
            fill_color=TEXT_COLOR,
            active_color=TEXT_COLOR,
            label_style=ft.TextStyle(color=TEXT_COLOR),
        )
        color_value = self._label_colors.get(class_id, self._default_color())
        border = ft.border.all(width=1, color=TEXT_COLOR) if TEXT_COLOR else None
        color_swatch = ft.Container(
            width=14,
            height=14,
            bgcolor=color_value,
            border_radius=4,
            border=border,
            ink=True,
            data=class_id,
            on_click=self._handle_color_clicked,
        )
        remove_button = ft.IconButton(
            icon=ft.Icons.REMOVE_CIRCLE,
            tooltip=f"Remove {label_name}",
            data=class_id,
            on_click=self._handle_remove_clicked,
            icon_color=ft.Colors.BLUE_100,
            # style=ICON_BUTTON_STYLE,
        )
        label_with_color = ft.Row(
            controls=[color_swatch, radio],
            spacing=8,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        header_row = ft.Row(
            controls=[label_with_color, remove_button],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        color_picker = self._build_color_picker(class_id)
        row = ft.Column(
            controls=[header_row, color_picker],
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.START,
        )
        container = ft.Container(content=row)

        self._rows[class_id] = container
        self._radios[class_id] = radio
        self._color_swatches[class_id] = color_swatch
        self._radio_column.controls.append(container)
        self.radio_group.visible = True

        if select:
            self._set_selected(class_id)

        self._update_controls()

    def remove_label(self, class_id: int, next_selection: Optional[int]) -> None:
        self._hide_all_color_pickers()
        container = self._rows.pop(class_id, None)
        self._radios.pop(class_id, None)
        self._color_swatches.pop(class_id, None)
        self._color_picklists.pop(class_id, None)
        self._color_option_controls.pop(class_id, None)
        if self._active_color_class_id == class_id:
            self._active_color_class_id = None
        if container is not None and container in self._radio_column.controls:
            self._radio_column.controls.remove(container)

        if not self._radio_column.controls:
            self.radio_group.visible = False
            self.radio_group.value = None
        else:
            self._set_selected(next_selection)

        self._update_controls()

    def clear(self) -> None:
        self._hide_all_color_pickers()
        self._rows.clear()
        self._radios.clear()
        self._color_swatches.clear()
        self._label_colors.clear()
        self._color_picklists.clear()
        self._color_option_controls.clear()
        self._radio_column.controls.clear()
        self.radio_group.value = None
        self.radio_group.visible = False
        self._active_color_class_id = None
        self._update_controls()

    def _set_selected(self, class_id: Optional[int]) -> None:
        if class_id is None or class_id not in self._radios:
            self.radio_group.value = None
        else:
            self.radio_group.value = str(class_id)
        self._update_controls()

    def _update_controls(self) -> None:
        if getattr(self.container, "page", None) is not None:
            self.container.update()

    def _update_label_text(self, class_id: int, label_name: str) -> None:
        radio = self._radios.get(class_id)
        if radio is None:
            return
        radio.label = label_name
        if getattr(radio, "page", None) is not None:
            radio.update()

    def update_colors(self, colors: Dict[int, str]) -> None:
        self._label_colors = dict(colors)
        for class_id, swatch in self._color_swatches.items():
            self._apply_color(class_id, swatch)
        self._highlight_selected_option()
        self._update_controls()

    def _apply_color(self, class_id: int, swatch: ft.Container) -> None:
        color = self._label_colors.get(class_id, self._default_color())
        swatch.bgcolor = color
        if getattr(swatch, "page", None) is not None:
            swatch.update()

    def _build_color_picker(self, class_id: int) -> ft.Container:
        option_controls: list[ft.Container] = []
        rows: list[ft.Row] = []
        n_cols = len(self._color_columns)
        n_rows = max((len(col) for col in self._color_columns), default=0)
        for row_idx in range(n_rows):
            row_items: list[ft.Control] = []
            for col_idx in range(n_cols):
                column = self._color_columns[col_idx]
                if row_idx >= len(column):
                    continue
                color = column[row_idx]
                option = ft.Container(
                    width=28,
                    height=28,
                    bgcolor=color,
                    border_radius=6,
                    ink=True,
                    data={"class_id": class_id, "color": color},
                    on_click=self._handle_color_option_clicked,
                )
                option_controls.append(option)
                row_items.append(option)
            if row_items:
                rows.append(ft.Row(controls=row_items, spacing=8))

        column = ft.Column(controls=rows, spacing=6)
        picker = ft.Container(content=column, padding=8, visible=False)

        self._color_option_controls[class_id] = option_controls
        self._color_picklists[class_id] = picker
        return picker

    def _show_color_picker(self, class_id: int) -> None:
        for cid, picker in self._color_picklists.items():
            picker.visible = cid == class_id
            if getattr(picker, "page", None) is not None:
                picker.update()

    def _hide_all_color_pickers(self) -> None:
        for picker in self._color_picklists.values():
            picker.visible = False
            if getattr(picker, "page", None) is not None:
                picker.update()

    def _handle_color_clicked(self, e) -> None:
        class_id = e.control.data
        if class_id is None:
            return
        class_id = int(class_id)
        if class_id not in self._color_picklists:
            return
        picker = self._color_picklists[class_id]
        if picker.visible:
            self._hide_all_color_pickers()
            self._active_color_class_id = None
            self._update_controls()
            return
        self._active_color_class_id = class_id
        self._show_color_picker(class_id)
        self._highlight_selected_option()
        self._update_controls()

    def _handle_color_option_clicked(self, e) -> None:
        data = e.control.data
        if not isinstance(data, dict):
            return
        class_id = data.get("class_id")
        color = data.get("color")
        if class_id is None or color is None:
            return
        class_id = int(class_id)
        color_str = str(color)
        if class_id not in self._color_swatches:
            return
        self._active_color_class_id = class_id
        self._label_colors[class_id] = color_str
        swatch = self._color_swatches.get(class_id)
        if swatch is not None:
            self._apply_color(class_id, swatch)
        self.on_color_changed(class_id, color_str)
        self._highlight_selected_option()
        self._hide_all_color_pickers()
        self._active_color_class_id = None
        self._update_controls()

    def _highlight_selected_option(self) -> None:
        highlight_color = TEXT_COLOR or "#ffffff"
        for class_id, options in self._color_option_controls.items():
            selected_color = str(self._label_colors.get(class_id, self._default_color())).lower()
            for option in options:
                data = option.data if isinstance(option.data, dict) else {}
                option_color = str(data.get("color", "")).lower()
                border_color = (
                    highlight_color if option_color == selected_color else ft.Colors.TRANSPARENT
                )
                option.border = ft.border.all(width=2, color=border_color)
                if getattr(option, "page", None) is not None:
                    option.update()

    @staticmethod
    def _default_color() -> str:
        return "#9e9e9e"

    def _handle_add_clicked(self, _e) -> None:
        name = self.new_label_field.value.strip()
        if not name:
            return
        if len(self._rows) >= 10:
            self.new_label_field.error_text = "Maximum of 10 minerals reached."
            if getattr(self.new_label_field, "page", None) is not None:
                self.new_label_field.update()
            return
        class_id = self.on_add_label(name)
        if class_id is None:
            return
        self.add_label(class_id, name, select=True)
        self.new_label_field.value = ""
        self.new_label_field.error_text = None
        if getattr(self.new_label_field, "page", None) is not None:
            self.new_label_field.update()

    def _handle_remove_clicked(self, e) -> None:
        class_id = e.control.data
        if class_id is None:
            return
        removed, next_selection = self.on_remove_label(int(class_id))
        if not removed:
            return
        self.remove_label(int(class_id), next_selection)
        self.on_select_label(next_selection)

    def _handle_selection_changed(self, e) -> None:
        value = e.control.value
        class_id = int(value) if value not in (None, "") else None
        self.on_select_label(class_id)
        if class_id is None:
            self._hide_all_color_pickers()
