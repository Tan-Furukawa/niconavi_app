from stores import Stores
from reactive_state import (
    ReactiveTextField,
    ReactiveDropDown,
    ReactiveElevatedButton,
    ReactiveCheckbox,
    ReactiveText,
    ReactiveRow,
)
from state import StateProperty, ReactiveState, State
import threading


import flet as ft
from typing import Any, Optional, overload, Callable


class customDivider(ft.Divider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.color = ft.Colors.WHITE60


class CustomText(ft.Text):
    def __init__(self, value: str, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self.color = ft.Colors.WHITE


class CustomReactiveText(ReactiveText):
    def __init__(
        self,
        text: StateProperty[str],
        visible: StateProperty[bool] = True,
        **kwargs: Any
    ) -> None:
        super().__init__(text=text, visible=visible, **kwargs)
        self.color = ft.Colors.WHITE


class CustomReactiveCheckbox(ReactiveCheckbox):
    def __init__(
        self,
        value: StateProperty[bool],
        label: StateProperty[str],
        visible: StateProperty[bool] = True,
        **kwargs: Any
    ) -> None:
        super().__init__(value, label=label, visible=visible, **kwargs)
        self.label_style = ft.TextStyle(color=ft.Colors.WHITE)
        self.fill_color = ft.Colors.BLUE_100
        self.check_color = ft.Colors.BLACK
        # self.check_color = ft.Colors.WHITE


class CustomRadio(ft.Radio):
    def __init__(
        self, value: Optional[str] = None, label: Optional[str] = None, **kwargs: Any
    ) -> None:
        super().__init__(value=value, label=label, **kwargs)
        self.fill_color = ft.Colors.WHITE
        self.label_style = ft.TextStyle(color=ft.Colors.WHITE)

class IntTextField(ft.TextField):
    def __init__(
        self, stores: Stores, value: Optional[int] = None, **kwargs: Any
    ) -> None:
        super().__init__(value=value, **kwargs)
        self.width = 100
        self.height = 40
        self.content_padding = ft.padding.only(left=10, top=3, bottom=3)
        self.border_color = ft.Colors.WHITE60


class ReactiveIntTextField(ReactiveTextField):
    def __init__(self, value: StateProperty[str], **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self.width = 100
        self.height = 40
        self.content_padding = ft.padding.only(left=10, top=3, bottom=3)
        self.border_color = ft.Colors.WHITE60


class ReactiveCodeTextInput(ReactiveTextField):
    def __init__(self, value: StateProperty[str], **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self.multiline = True
        # self.text_style = ft.Text.font_family("Consolas")
        self.min_lines = 10
        self.max_lines = 10
        # self.content_padding = ft.padding.only(left=10, top=3, bottom=3)
        self.border_color = ft.Colors.WHITE60


class CustomTextField(ft.TextField):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # self.width = 70
        # self.height = 30
        # self.content_padding = ft.padding.only(left=10, top=3, bottom=3)
        self.border_color = ft.Colors.WHITE60
        self.hint_style = ft.TextStyle(color=ft.Colors.WHITE60)
        self.color = ft.Colors.WHITE


class ReactiveFloatTextField(ReactiveTextField):
    def __init__(
        self,
        value: StateProperty[str],
        visible: StateProperty[bool] = True,
        read_only: StateProperty[bool] = False,
        **kwargs: Any
    ) -> None:
        super().__init__(value, visible, read_only, **kwargs)
        self.width = 70
        self.height = 30
        self.content_padding = ft.padding.only(left=10, top=3, bottom=3)
        self.border_color = ft.Colors.WHITE60


class CustomDropDown(ft.Dropdown):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.width = 400
        self.height = 40
        self.content_padding = ft.padding.only(left=10, top=3, bottom=3)
        self.border_color = ft.Colors.WHITE60
        self.text_style = ft.TextStyle(color=ft.Colors.WHITE)
        self.label_style = ft.TextStyle(color=ft.Colors.WHITE)
        self.color = ft.Colors.WHITE


class ReactiveCustomDropDown(ReactiveDropDown):
    def __init__(self, visible: StateProperty[bool] = True, **kwargs: Any) -> None:
        super().__init__(visible, **kwargs)
        self.width = 350
        self.height = 40
        self.content_padding = ft.padding.only(left=10, top=3, bottom=3)
        self.border_color = ft.Colors.WHITE60
        self.color = ft.Colors.WHITE
        self.bgcolor = ft.Colors.GREY_600
        self.hint_style = ft.TextStyle(color=ft.Colors.WHITE)


class CustomExecuteButton(ReactiveElevatedButton):
    def __init__(
        self,
        text: StateProperty[str],
        visible: StateProperty[bool] = True,
        bgcolor: str = ft.Colors.LIGHT_GREEN_700,
        **kwargs: Any
    ) -> None:
        super().__init__(visible=visible, text=text, **kwargs)
        self.height = 30
        self.content_padding = ft.padding.only(left=10, top=3, bottom=3)
        self.bgcolor = bgcolor
        self.color = ft.Colors.WHITE
        self.style = ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=5),
        )


class CustomSelectFileButton(ReactiveElevatedButton):
    def __init__(
        self,
        text: StateProperty[str],
        visible: StateProperty[bool] = True,
        **kwargs: Any
    ) -> None:
        super().__init__(visible=visible, text=text, **kwargs)
        # self.width = 100
        self.height = 30
        self.content_padding = ft.padding.only(left=10, top=3, bottom=3)
        self.bgcolor = ft.Colors.BLACK
        self.color = ft.Colors.BLUE_100
        self.style = ft.ButtonStyle(
            bgcolor= ft.Colors.BLACK,
            overlay_color= ft.Colors.BLACK54
        )



@overload
def make_reactive_float_text_filed(
    stores: Stores,
    param_stores: State[int] | State[int | None],
    parser: Callable[[str], int | None],
    visible: StateProperty[bool] = True,
    accept_None: bool = True,
) -> ReactiveTextField: ...


@overload
def make_reactive_float_text_filed(
    stores: Stores,
    param_stores: State[float] | State[float | None],
    parser: Callable[[str], float | None],
    visible: StateProperty[bool] = True,
    accept_None: bool = True,
) -> ReactiveTextField: ...


def make_reactive_float_text_filed(
    stores: Stores,
    param_stores: State[float] | State[float | None] | State[int] | State[int | None],
    parser: Callable[[str], float | None] | Callable[[str], int | None],
    visible: StateProperty[bool] = True,
    accept_None: bool = True,
) -> ReactiveFloatTextField:  #! 変な実装注意！！

    def get_val() -> str:
        if stores.ui.dummy_str.get() is not None:  # 避難させた文字列があるとき
            v = stores.ui.dummy_str.get()
            stores.ui.dummy_str.set(None)
            return v
        else:
            return str("" if param_stores.get() is None else param_stores.get())

    state_val = ReactiveState(
        lambda: get_val(),
        [param_stores],
    )

    def delay_update() -> (
        None
    ):  # 変な値が入れられたときの修正機能(直前の正しい値を1秒後に代入)
        v = param_stores.get()
        # print("000000000000000")
        # print(v)
        # print("000000000000000")
        # param_stores.set(
        #     -999
        # )  # 一旦ちがう値を入れて値を変えたことにする（じゃないsetしたときに反応しない）
        if v is not None:
            v_new = int(v) if int(v) == v else v  # 0.0表示を防止する
        else:
            v_new = None
        param_stores.set(v_new)
        state_val.force_update()

    def on_change(stores: Stores, e: ft.ControlEvent) -> None:
        stores.ui.dummy_str.set(None)  # 念の為、変更のたびに毎回初期化させる
        val = e.control.value
        try:
            fval = parser(val)

            if (
                fval == param_stores.get()
            ):  # parseされた値が保存されている値と本質的に何も変わらないとき、なにもしない。
                pass
            else:  # parseの値が変更されたとき
                if fval is not None:  # parseに成功したとき
                    stores.ui.dummy_str.set(val)  # parseする前の文字列を避難させる
                    param_stores.set(fval)  # parseした値を収める
                else:
                    if accept_None:
                        param_stores.set(None)
                    else:
                        timer = threading.Timer(1, delay_update)
                        timer.start()
        except Exception as e:
            timer = threading.Timer(1, delay_update)
            timer.start()

    return ReactiveFloatTextField(
        value=state_val,
        visible=visible,
        color=ft.Colors.WHITE,
        read_only=ReactiveState(
            lambda: not stores.ui.computing_is_stop.get(), [stores.ui.computing_is_stop]
        ),
        on_change=lambda e: on_change(stores, e),
    )


def make_solidable_checkbox(
    label: str,
    state_val: State[bool],
    can_edit_reactive_state_formula: Callable,
    can_edit_reliance_states: list[State],
) -> ft.Row:
    return ft.Row(
        [
            CustomReactiveCheckbox(
                label=label,
                value=state_val,
                on_change=lambda e: state_val.set(e.control.value),
                visible=ReactiveState(
                    lambda: can_edit_reactive_state_formula(), can_edit_reliance_states
                ),
            ),
            ReactiveRow(
                [
                    ReactiveRow(
                        [
                            ft.Icon(ft.Icons.CHECK_BOX),
                            CustomText(value=label),
                        ],
                        visible=ReactiveState(lambda: state_val.get(), [state_val]),
                    ),
                    ReactiveRow(
                        [
                            ft.Icon(ft.Icons.CHECK_BOX_OUTLINE_BLANK),
                            CustomText(value=label),
                        ],
                        visible=ReactiveState(lambda: not state_val.get(), [state_val]),
                    ),
                ],
                visible=ReactiveState(
                    lambda: not can_edit_reactive_state_formula(),
                    can_edit_reliance_states,
                ),
            ),
        ]
    )


# ---------------------------------------
# カウンター
# ---------------------------------------
def make_REMOVE_counter_button(
    stores: Stores,
    target_state: StateProperty[int | None] | StateProperty[float | None],
    *,
    step: float = 1,
    min_value: float | None = 1,
    max_value: float | None = None,
    precision: int | None = None,
    value_type: type[int] | type[float] | None = None,
) -> ReactiveRow:

    def cx_minus_click() -> None:
        current = target_state.get()
        if current is None:
            return
        new_value = float(current) - step
        if precision is not None:
            new_value = round(new_value, precision)
        if min_value is not None:
            new_value = max(min_value, new_value)
        if max_value is not None:
            new_value = min(max_value, new_value)
        cast_type = value_type or type(current)
        if cast_type is int:
            target_state.set(int(new_value))
        elif cast_type is float:
            target_state.set(float(new_value))
        else:
            target_state.set(new_value)

    return ReactiveRow(
        [
            ft.IconButton(
                ft.Icons.REMOVE_CIRCLE,
                on_click=lambda e: cx_minus_click(),
                icon_color=ft.Colors.BLUE_100,
            )
        ],
        visible=stores.ui.computing_is_stop,
    )


def make_ADD_counter_button(
    stores: Stores,
    target_state: StateProperty[int | None] | StateProperty[float | None],
    *,
    step: float = 1,
    min_value: float | None = None,
    max_value: float | None = None,
    precision: int | None = None,
    value_type: type[int] | type[float] | None = None,
) -> ReactiveRow:
    def cx_plus_click() -> None:
        current = target_state.get()
        if current is None:
            return
        new_value = float(current) + step

        cast_type = value_type or type(current)

        if cast_type is int and current == 1 and step > 1:
            new_value = 10.0
        if precision is not None:
            new_value = round(new_value, precision)
        if min_value is not None:
            new_value = max(min_value, new_value)
        if max_value is not None:
            new_value = min(max_value, new_value)
        cast_type = value_type or type(current)
        if cast_type is int:
            target_state.set(int(new_value))
        elif cast_type is float:
            target_state.set(float(new_value))
        else:
            target_state.set(new_value)

    return ReactiveRow(
        [
            ft.IconButton(
                ft.Icons.ADD_CIRCLE,
                on_click=lambda e: cx_plus_click(),
                icon_color=ft.Colors.BLUE_100,
            )
        ],
        visible=stores.ui.computing_is_stop,
    )
