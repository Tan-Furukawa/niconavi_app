from __future__ import annotations

import flet as ft

from niconavi_app.components.context_help import (
    HELP_WAIT_MS,
    TAB_HELP,
    attach_context_help,
    make_help_tooltip,
    set_context_help,
    suppress_context_help,
)


def test_tooltip_waits_two_seconds_and_wraps_long_text() -> None:
    tooltip = make_help_tooltip(
        "Sets the section thickness in millimetres. A larger value allows a "
        "larger retardation range in later estimates."
    )

    assert tooltip.wait_duration == HELP_WAIT_MS == 2000
    assert "\n" in tooltip.message
    assert max(map(len, tooltip.message.splitlines())) <= 58
    assert tooltip.bgcolor == "#fff7f7f7"
    assert tooltip.text_style.color == ft.Colors.BLACK87


def test_explicit_help_is_attached_without_changing_event_handler() -> None:
    calls: list[str] = []
    button = ft.ElevatedButton("Calculate", on_click=lambda _: calls.append("run"))
    original_handler = button.on_click

    set_context_help(button, "Calculates CPO from the current grain labels.")

    assert isinstance(button.tooltip, ft.Tooltip)
    assert button.tooltip.wait_duration == 2000
    assert button.on_click is original_handler


def test_tree_help_uses_parameter_specific_text() -> None:
    field = ft.TextField(value="30")
    root = ft.Row([ft.Text("SLIC superpixel size"), field, ft.Text("² px²")])

    attach_context_help(root, area="controls")

    assert isinstance(field.tooltip, ft.Tooltip)
    assert (
        "Larger values create coarser initial regions"
        in field.tooltip.message.replace("\n", " ")
    )


def test_image_button_uses_explicit_calculation_help() -> None:
    button = ft.ElevatedButton("Original color change (0°)", on_click=lambda _: None)
    set_context_help(
        button,
        "Signed RGB difference between the aligned tilted and horizontal images.",
    )

    attach_context_help(button, area="images")

    assert isinstance(button.tooltip, ft.Tooltip)
    assert button.tooltip.message.replace("\n", " ") == (
        "Signed RGB difference between the aligned tilted and horizontal images."
    )


def test_image_button_has_no_repetitive_generic_help() -> None:
    button = ft.ElevatedButton("Unannotated image", on_click=lambda _: None)

    attach_context_help(button, area="images")

    assert button.tooltip is None


def test_rotation_center_field_and_buttons_have_the_same_help() -> None:
    decrease = ft.IconButton(icon=ft.Icons.REMOVE_CIRCLE, on_click=lambda _: None)
    field = ft.TextField(value="100")
    increase = ft.IconButton(icon=ft.Icons.ADD_CIRCLE, on_click=lambda _: None)
    root = ft.Column(
        [
            ft.Text("Rotation center"),
            ft.Row([ft.Text("X"), decrease, field, increase]),
        ]
    )

    attach_context_help(root, area="controls")

    assert isinstance(decrease.tooltip, ft.Tooltip)
    assert isinstance(field.tooltip, ft.Tooltip)
    assert isinstance(increase.tooltip, ft.Tooltip)
    assert decrease.tooltip.message == field.tooltip.message == increase.tooltip.message


def test_suppressed_obvious_control_has_no_help() -> None:
    button = ft.IconButton(
        icon=ft.Icons.ADD,
        tooltip="Advanced settings",
        on_click=lambda _: None,
    )

    suppress_context_help(button)
    attach_context_help(button, area="controls")

    assert button.tooltip is None


def test_each_workflow_tab_has_specific_help() -> None:
    assert TAB_HELP[0].startswith("Video tab:")
    assert TAB_HELP[1].startswith("Center tab:")
    assert TAB_HELP[2].startswith("Map tab:")
    assert TAB_HELP[3].startswith("Filter tab:")
    assert TAB_HELP[4].startswith("Analysis tab:")
