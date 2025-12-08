from niconavi.tools.read_data import divide_video_into_n_frame
from reactive_state import (
    ReactiveText,
    ReactiveDivider,
    ReactiveElevatedButton,
)
from stores import Stores
from state import ReactiveState, State
from state import StateProperty  # type: ignore
import flet as ft
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from typing import Any, Callable, Optional, Literal


class ReactiveElevatedButtonInSelector(ReactiveElevatedButton):
    def __init__(
        self,
        text: StateProperty[str],
        visible: StateProperty[bool] = True,
        bgcolor: StateProperty[Optional[str]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(text, visible, bgcolor, **kwargs)
        # self.border_radius = 0
        # self.bgcolor = bgcolor
        # self.height = 30
        self.color = "white"
        self.width = 280
        self.style = ft.ButtonStyle(
            shape=ft.ContinuousRectangleBorder(radius=3),
        )


def make_image_button_color(
    stores: Stores,
    selected_index: int,
    button_index: int,
    tab_selection_state: State[int],
) -> ReactiveState:
    return ReactiveState(
        lambda: (
            stores.appearance.button_active_color
            if tab_selection_state.get() == button_index
            and stores.ui.selected_index.get() == selected_index
            and not stores.ui.display_common_image_view.get()
            else stores.appearance.button_inactive_color
        ),
        [
            tab_selection_state,
            stores.ui.selected_index,
            stores.ui.display_common_image_view,
        ],
    )


def make_elevated_button(
    stores: Stores,
    button_name: str,
    tab_selection_state: State[int],
    selected_index: int,  # 選択されているタブのindex
    button_index: int,  # ボタンの固有index
    additional_condition: list[State] = [],
    additional_formula: Callable[[], bool] = lambda: True,
) -> ReactiveElevatedButtonInSelector:

    def click_button() -> None:
        stores.ui.display_common_image_view.set(False)  # 次の行よりこっちが先!
        tab_selection_state.set(button_index)

    visibility = ReactiveState(
        lambda: all(list(map(lambda x: x.get() is not None, additional_condition)))
        and additional_formula()
        and stores.ui.selected_index.get() == selected_index,
        # and not stores.ui.computing.get(),
        additional_condition + [stores.ui.selected_index],
    )

    button = ReactiveElevatedButtonInSelector(
        button_name,
        visible=visibility,
        bgcolor=make_image_button_color(
            stores, selected_index, button_index, tab_selection_state
        ),
        on_click=lambda e: click_button(),
    )

    return button


# タブの種類に関係なくボタンを表示したい場合はこれを使用する
def make_always_display_elevated_button(
    stores: Stores,
    button_name: str,
    button_index: int,
    visible_condition: Callable[[], bool] = lambda: True,
    visible_reliance_state: list[State] = [],
) -> ReactiveElevatedButtonInSelector:

    def click_common_image_button() -> None:
        stores.ui.display_common_image_view.set(
            True
        )  # 次の行よりこっちが先!(class ImageViewで、stores.ui.display_common_image_viewを含めると、2回更新されてしまう)
        stores.ui.selected_button_at_common_image_view.set(button_index)

    return ReactiveElevatedButtonInSelector(
        button_name,
        visible=ReactiveState(lambda: visible_condition(), visible_reliance_state),
        bgcolor=ReactiveState(
            lambda: (
                stores.appearance.button_active_color
                if stores.ui.display_common_image_view.get()  # display_common_image_viewがTrueのとき、image_viewで最優先される
                and stores.ui.selected_button_at_common_image_view.get() == button_index
                else stores.appearance.button_inactive_color
            ),
            [
                stores.ui.display_common_image_view,
                stores.ui.selected_button_at_common_image_view,
            ],
        ),
        on_click=lambda e: click_common_image_button(),
    )


def exist_in_segmented_grain_map(
    stores: Stores,
    key: Literal[
        "p45_R_color_map",
        "m45_R_color_map",
        "p45_R_map",
        "m45_R_map",
        "azimuth",
        "azimuth360",
        "inclination",
    ],
) -> bool:
    raw_map = stores.computation_result.raw_maps.get()
    if raw_map is not None:
        if raw_map[key] is not None:
            return True
        else:
            return False
    else:
        return False


def exist_in_raw_maps(
    stores: Stores,
    key: Literal[
        "p45_R_color_map",
        "m45_R_color_map",
        "p45_R_map",
        "m45_R_map",
        "azimuth360",
        "inclination",
        "cv_extinction_angle",
        "inclination_0_to_180",
    ],
) -> bool:
    raw_map = stores.computation_result.raw_maps.get()
    # print("------------------------------------")
    # print(raw_map)
    # print("------------------------------------")
    if raw_map is not None:
        if raw_map[key] is not None:
            return True
        else:
            return False
    else:
        return False


def exist_in_segmented_maps(
    stores: Stores,
    key: Literal[
        "extinction_color_map",
        "R_color_map",
        "extinction_angle",
        "max_retardation_map",
        "H",
        "S",
        "V",
        "eccentricity",
        "angle_deg",
        "major_axis_length",
        "minor_axis_length",
        "R_70_map",
        "R_80_map",
        "R_90_map",
        "size",
        "p45_R_color_map",
        "m45_R_color_map",
        "p45_R_map",
        "m45_R_map",
        "azimuth",
        "sd_azimuth",
        "sd_extinction_angle_map",
        "extinction_angle_quality",
    ],
) -> bool:
    maps = stores.computation_result.grain_segmented_maps.get()
    if maps is not None:
        if maps[key] is not None:
            return True
        else:
            return False
    else:
        return False


def make_reactive_text(
    stores: Stores,
    text: str,
    selected_index: int,
    additional_formula: Callable[[], bool] = lambda: True,
    additional_condition: list[State] = [],
) -> ft.Column:

    visible = ReactiveState(
        lambda: stores.ui.selected_index.get() == selected_index
        and additional_formula(),
        [stores.ui.selected_index] + additional_condition,
    )

    return ReactiveText(
        text=text,
        visible=visible,
        color=ft.Colors.BLUE_100,
        weight=ft.FontWeight.BOLD,
    )
