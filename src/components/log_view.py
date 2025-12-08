from stores import Stores

import flet as ft
from typing import Callable, Literal, Optional
from logging import Logger
from components.common_component import (
    CustomText,
)


def create_column(stores: Stores) -> ft.Column:

    # 状態毎のアイコンマップ（適宜変更可能）
    data = stores.ui.log_view.log_contents.get()

    icon_map = {
        "ok": ft.Icons.CHECK_CIRCLE,
        "err": ft.Icons.CANCEL,
        "msg": ft.Icons.MESSAGE,
        "warn": ft.Icons.WARNING,
    }

    controls = []
    for text_val, status in data:
        # "msg"の場合はアイコンなしでテキストのみ表示しても良い
        if status == "ok":
            controls.append(
                ft.Row(
                    [
                        ft.Icon(icon_map["ok"], color=ft.Colors.GREEN, size=20),
                        ft.Container(
                            CustomText(text_val),
                            width=stores.appearance.tabs_width
                            - 45,  # icon + paddingの分を引く
                        ),
                    ]
                )
            )

        elif status == "err":
            controls.append(
                ft.Row(
                    controls=[
                        ft.Icon(icon_map["err"], color=ft.Colors.RED, size=20),
                        ft.Container(
                            CustomText(text_val, color=ft.Colors.RED),
                            width=stores.appearance.tabs_width - 45,
                        ),
                    ]
                )
            )

        elif status == "warn":
            controls.append(
                ft.Row(
                    controls=[
                        ft.Icon(icon_map["warn"], color=ft.Colors.AMBER, size=20),
                        ft.Container(
                            CustomText(text_val, color=ft.Colors.AMBER),
                            width=stores.appearance.tabs_width - 45,
                        ),
                    ]
                )
            )

        else:  # "msg"
            controls.append(
                ft.Row(
                    controls=[
                        ft.Container(
                            CustomText(text_val), width=stores.appearance.tabs_width - 45
                        ),
                    ],
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
