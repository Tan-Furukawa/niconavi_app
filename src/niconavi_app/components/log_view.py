from niconavi_app.stores import Stores

import flet as ft
from typing import Callable, Literal, Optional
from logging import Logger
from niconavi_app.components.common_component import (
    CustomText,
)


# Icon and text colour per log level. "msg" marks work in progress ("Uploading
# xxx ...") and carries an icon like the rest, so a running step reads as a step
# rather than as a stray line of text. Its text stays white: only the levels
# that need attention colour the message itself.
LOG_LEVEL_STYLE: dict[str, tuple[str, str, Optional[str]]] = {
    "ok": (ft.Icons.CHECK_CIRCLE, ft.Colors.GREEN, None),
    "err": (ft.Icons.CANCEL, ft.Colors.RED, ft.Colors.RED),
    "warn": (ft.Icons.WARNING, ft.Colors.AMBER, ft.Colors.AMBER),
    "msg": (ft.Icons.PENDING, ft.Colors.BLUE_200, None),
}


def create_column(stores: Stores) -> ft.Column:
    data = stores.ui.log_view.log_contents.get()

    controls = []
    for text_val, status in data:
        icon, icon_color, text_color = LOG_LEVEL_STYLE.get(
            status, LOG_LEVEL_STYLE["msg"]
        )
        controls.append(
            ft.Row(
                controls=[
                    ft.Icon(icon, color=icon_color, size=20),
                    ft.Container(
                        CustomText(text_val, color=text_color)
                        if text_color is not None
                        else CustomText(text_val),
                        # icon + paddingの分を引く
                        width=stores.appearance.tabs_width - 45,
                    ),
                ]
            )
        )

    return ft.Column(controls=controls, scroll=ft.ScrollMode.ALWAYS, spacing=10)


class LogView(ft.Container):
    def __init__(self, stores: Stores, column: ft.Column) -> None:
        super().__init__()
        self.content = column

        self.height = 200
        self.padding = 10
        self.width = stores.appearance.tabs_width
        # self.border = ft.border.all(color=ft.Colors.WHITE)
        self.bgcolor = ft.Colors.BLACK12


def update_logs(
    stores: Stores,
    entry: tuple[str, Literal["ok", "err", "msg", "warn"]],
    logger: Optional[Logger] = None,
) -> None:
    """
    Append a log message to the UI log store and mirror it to the optional logger.
    """
    message, level = entry

    if logger is not None:
        log_fn = {
            "err": logger.error,
            "warn": logger.warning,
        }.get(level, logger.info)
        log_fn(message)

    existing = stores.ui.log_view.log_contents.get()
    stores.ui.log_view.log_contents.set([*existing, entry])
