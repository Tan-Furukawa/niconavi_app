from __future__ import annotations

import flet as ft
from types import SimpleNamespace

from niconavi_app.components.context_help import (
    ContextHelpController,
    resolve_help_topic,
)


class FakePage:
    def __init__(self) -> None:
        self.opened: list[ft.AlertDialog] = []
        self.closed: list[ft.AlertDialog] = []
        self.update_count = 0

    def open(self, dialog: ft.AlertDialog) -> None:
        self.opened.append(dialog)

    def close(self, dialog: ft.AlertDialog) -> None:
        self.closed.append(dialog)

    def update(self) -> None:
        self.update_count += 1


def test_help_mode_intercepts_and_restores_button() -> None:
    calls: list[str] = []
    button = ft.ElevatedButton(
        "Segment",
        disabled=True,
        on_click=lambda _: calls.append("segment"),
    )
    page = FakePage()
    controller = ContextHelpController(page)  # type: ignore[arg-type]
    help_button = controller.create_button()
    controller.add_scope(ft.Column([button]), area="controls")

    controller.activate()

    assert controller.active
    assert help_button.icon_color == ft.Colors.ORANGE_400
    assert button.disabled is False
    button.on_click(None)
    assert calls == []
    assert page.opened[-1].title is None
    assert "Groups neighbouring regions" in page.opened[-1].content.content.value

    close_button = page.opened[-1].actions[0]
    close_button.on_click(None)

    assert controller.active is False
    assert help_button.icon_color == ft.Colors.WHITE
    assert button.disabled is True
    button.disabled = False
    button.on_click(None)
    assert calls == ["segment"]


def test_help_mode_restores_changed_control_value() -> None:
    changes: list[bool] = []
    checkbox = ft.Checkbox(
        label="Apply mask",
        value=False,
        on_change=lambda event: changes.append(bool(event.control.value)),
    )
    page = FakePage()
    controller = ContextHelpController(page)  # type: ignore[arg-type]
    controller.add_scope(checkbox, area="controls")
    controller.activate()

    checkbox.value = True
    checkbox.on_change(None)

    assert checkbox.value is False
    assert changes == []
    controller.deactivate()
    assert checkbox.value is False


def test_text_field_is_temporarily_read_only() -> None:
    field = ft.TextField(value="30", read_only=False)
    page = FakePage()
    controller = ContextHelpController(page)  # type: ignore[arg-type]
    controller.add_scope(field, area="controls")

    controller.activate()
    assert field.read_only is True
    assert field.always_call_on_tap is True

    controller.deactivate()
    assert field.read_only is False
    assert field.always_call_on_tap is False


def test_hidden_controls_are_not_intercepted() -> None:
    original = lambda _: None
    hidden = ft.ElevatedButton("Hidden", visible=False, on_click=original)
    page = FakePage()
    controller = ContextHelpController(page)  # type: ignore[arg-type]
    controller.add_scope(ft.Column([hidden]), area="controls")

    controller.activate()

    assert hidden.on_click is original


def test_specific_and_image_help_are_user_facing() -> None:
    slic = resolve_help_topic(
        ft.TextField(),
        area="controls",
        context=("SLIC superpixel size", "² px²"),
    )
    image = resolve_help_topic(
        ft.ElevatedButton("Original color change (0°)"),
        area="images",
    )

    assert "Larger values create coarser initial regions" in slic.body
    assert "does not recalculate" in image.body


def test_workflow_tab_help_uses_clicked_tab() -> None:
    tabs = ft.Tabs(tabs=[], selected_index=0)

    topic = resolve_help_topic(
        tabs,
        area="controls",
        event=SimpleNamespace(data="1"),
    )

    assert topic.body.startswith("Center tab:")


def test_custom_help_text_only_opens_in_help_mode() -> None:
    page = FakePage()
    controller = ContextHelpController(page)  # type: ignore[arg-type]

    controller.show_text("Video tab: select the required inputs.")
    assert page.opened == []

    controller.activate()
    controller.show_text("Video tab: select the required inputs.")

    assert page.opened[-1].title is None
    assert page.opened[-1].content.content.value.startswith("Video tab:")
