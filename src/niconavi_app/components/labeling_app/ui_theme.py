from __future__ import annotations

import flet as ft


BACKGROUND_COLOR: str | None = None
TEXT_COLOR: str | None = ft.Colors.WHITE
BORDER_COLOR: str | None = ft.Colors.LIGHT_GREEN_700


BUTTON_STYLE: ft.ButtonStyle | None = None


ICON_BUTTON_STYLE: ft.ButtonStyle = ft.ButtonStyle(shape=ft.CircleBorder(), color="black", padding=2, bgcolor=ft.Colors.BLUE_100, shadow_color=ft.Colors.BLUE_400)


TEXT_FIELD_STYLE: dict = {}


def apply_border(container: ft.Container, padding: float | None = None) -> ft.Container:
    """Return the container untouched except for optional padding."""

    if padding is not None:
        container.padding = padding
    return container


def themed_text(value: str, weight: ft.FontWeight | None = None) -> ft.Text:
    """Create a Text control without forcing custom colors."""

    return ft.Text(value=value, weight=weight)
