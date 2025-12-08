from state import bind_props, get_prop_value, StateProperty  # type: ignore
from typing import TypeVar, Optional, Any
import flet as ft
from flet import Text, Tab, Tabs, ProgressRing, ElevatedButton
from flet.matplotlib_chart import MatplotlibChart
from matplotlib.pyplot import Figure
from matplotlib.collections import QuadMesh
from matplotlib.backends.backend_agg import FigureCanvasAgg
import base64
import io


T = TypeVar("T")


class ReactiveDivider(ft.Divider):
    def __init__(self, visible: StateProperty[bool] = True, **kwargs: Any):
        super().__init__(**kwargs)
        self._visible = visible

        self.set_props()
        bind_props(
            [self._visible], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.visible = get_prop_value(self._visible)

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveRadioGroup(ft.RadioGroup):
    def __init__(self, visible: StateProperty[bool] = True, **kwargs: Any):
        super().__init__(**kwargs)
        self._visible = visible

        self.set_props()
        bind_props([self._visible], lambda: self.content_update())

    def set_props(self) -> None:
        self.visible = get_prop_value(self._visible)

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveText(Text):
    def __init__(
        self,
        text: StateProperty[str],
        visible: StateProperty[bool] = True,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._text = text
        self._visible = visible

        self.set_props()
        bind_props(
            [self._text, self._visible], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.value = get_prop_value(self._text)
        self.visible = get_prop_value(self._visible)

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveTextField(ft.TextField):
    def __init__(
        self,
        value: StateProperty[str],
        visible: StateProperty[bool] = True,
        read_only: StateProperty[bool] = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._value = value
        self._visible = visible
        self._read_only = read_only

        self.set_props()
        bind_props(
            [self._value, self._visible, self._read_only], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.value = get_prop_value(self._value)
        self.visible = get_prop_value(self._visible)
        self.read_only = get_prop_value(self._read_only)

    def content_update(self) -> None:
        self.set_props()
        self.update()


# ft.Image(src_)


def _figure_contains_raster(fig: Optional[Figure]) -> bool:
    if fig is None:
        return False
    for ax in fig.axes:
        if ax.images:
            return True
        for collection in ax.collections:
            if isinstance(collection, QuadMesh):
                return True
    return False


def _figure_to_png_base64(fig: Figure) -> str:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    return base64.b64encode(buffer.getvalue()).decode()


class ReactiveMatplotlibChart(ft.Container):
    def __init__(self, figure: StateProperty[Figure], **kwargs: Any):
        super().__init__(**kwargs)
        self._figure = figure
        self._chart = MatplotlibChart()
        self._image = ft.Image(fit=ft.ImageFit.CONTAIN)
        self._mode: Optional[str] = None
        self.content = self._chart
        if self.expand is not None:
            self._chart.expand = self.expand
            self._image.expand = self.expand

        self.set_props()
        bind_props(
            [self._figure], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        fig = get_prop_value(self._figure)

        if _figure_contains_raster(fig):
            png_base64 = _figure_to_png_base64(fig)
            if self._mode != "image":
                self.content = self._image
                self._mode = "image"
            self._image.src_base64 = png_base64
        else:
            if self._mode != "chart":
                self.content = self._chart
                self._mode = "chart"
            self._chart.figure = fig

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveImage(ft.Image):
    def __init__(self, src_base64: StateProperty[str], **kwargs: Any):
        super().__init__(**kwargs)
        self._src_base64 = src_base64

        self.set_props()
        bind_props(
            [self._src_base64], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.src_base64 = get_prop_value(
            self._src_base64
        )  # 通常の変数かStateかを判断して値を取得

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveContainer(ft.Container):
    def __init__(self, content: StateProperty[ft.Control], **kwargs: Any):
        super().__init__(**kwargs)
        self._content = content

        self.set_props()
        bind_props(
            [self._content], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.content = get_prop_value(
            self._content
        )  # 通常の変数かStateかを判断して値を取得

    def content_update(self) -> None:
        self.set_props()
        self.update()


# ft.Checkbox(label="Unchecked by default checkbox", value=False)
class ReactiveDropDown(ft.Dropdown):
    def __init__(self, visible: StateProperty[bool] = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._visible = visible

        self.set_props()
        bind_props(
            [self._visible], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.visible = get_prop_value(self._visible)

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveCheckbox(ft.Checkbox):
    def __init__(
        self,
        value: StateProperty[bool],
        label: StateProperty[str],
        visible: StateProperty[bool] = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._value = value
        self._label = label
        self._visible = visible

        self.set_props()
        bind_props(
            [self._value, self._label, self._visible], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.value = get_prop_value(self._value)
        self.label = get_prop_value(self._label)
        self.visible = get_prop_value(self._visible)

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveRow(ft.Row):
    def __init__(
        self,
        controls: StateProperty[list],
        visible: StateProperty[bool] = True,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        self._controls = controls
        self._visible = visible

        self.set_props()
        bind_props([self._controls, self._visible], lambda: self.content_update())

    def set_props(self) -> None:
        self.controls = get_prop_value(self._controls)
        self.visible = get_prop_value(self._visible)

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveColumn(ft.Column):
    def __init__(
        self,
        controls: StateProperty[list],
        visible: StateProperty[bool] = True,
        scroll_offset_update: Optional[float] = None,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._controls = controls
        self._visible = visible
        self.scroll_offset_update = scroll_offset_update

        self.set_props()
        bind_props([self._controls, self._visible], lambda: self.content_update())

    def set_props(self) -> None:
        self.controls = get_prop_value(self._controls)
        self.visible = get_prop_value(self._visible)

    def content_update(self) -> None:
        if self.scroll_offset_update is not None:
            self.scroll_to(offset=-1, duration=0)
        self.set_props()
        self.update()


class ReactiveElevatedButton(ElevatedButton):
    def __init__(
        self,
        text: StateProperty[str],
        visible: StateProperty[bool] = True,
        bgcolor: StateProperty[Optional[str]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self._text = text
        self._visible = visible
        self._bgcolor = bgcolor
        self.set_props()

        bind_props(
            [self._text, self._visible, self._bgcolor], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.text = get_prop_value(self._text)
        self.visible = get_prop_value(self._visible)
        self.bgcolor = get_prop_value(self._bgcolor)

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveProgressBar(ft.ProgressBar):
    def __init__(
        self,
        value: StateProperty[float | None],
        visible: StateProperty[bool],
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._value = value
        self._visible = visible

        self.set_props()
        bind_props(
            [self._value, self._visible], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.value = get_prop_value(
            self._value
        )  # 通常の変数かStateかを判断して値を取得
        self.visible = get_prop_value(
            self._visible
        )  # 通常の変数かStateかを判断して値を取得

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveProgressRing(ProgressRing):
    def __init__(
        self,
        value: StateProperty[float | None],
        visible: StateProperty[bool],
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._value = value
        self._visible = visible

        self.set_props()
        bind_props(
            [self._value, self._visible], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.value = get_prop_value(
            self._value
        )  # 通常の変数かStateかを判断して値を取得
        self.visible = get_prop_value(
            self._visible
        )  # 通常の変数かStateかを判断して値を取得

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveTabs(Tabs):
    def __init__(
        self,
        selected_index: StateProperty[int] = 0,
        **kwargs: Any
        # animation_duration: Optional[int] = None,
        # on_change: Optional[Callable[[Tabs], None]] = None,
        # expand: Optional[bool] = None,
        # scrollable: bool = True,
        # tabs: list[Tab] = [],
    ):
        super().__init__(
            **kwargs
            # animation_duration=animation_duration,
            # expand=expand,
            # tabs=tabs,
            # scrollable=scrollable,
        )
        # self.on_change = on_change
        self._selected_index = selected_index
        # self.on_change = lambda e: self._selected_index.update()
        self.set_props()
        bind_props(
            [self._selected_index], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.selected_index = get_prop_value(
            self._selected_index
        )  # 通常の変数かStateかを判断して値を取得
        # self.size = get_prop_value(self.size)

    def content_update(self) -> None:
        self.set_props()
        self.update()

    # def build(self):
    #     return self.control


class ReactiveExpansionTile(ft.ExpansionTile):
    def __init__(self, visible: StateProperty[bool], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._visible = visible

        self.set_props()
        bind_props(
            [self._visible], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.visible = get_prop_value(
            self._visible
        )  # 通常の変数かStateかを判断して値を取得

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveTab(ft.Tab):
    def __init__(self, visible: StateProperty[bool] = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._visible = visible

        self.set_props()
        bind_props(
            [self._visible], lambda: self.content_update()
        )  # 自動でデータバインディング

    def set_props(self) -> None:
        self.visible = get_prop_value(
            self._visible
        )  # 通常の変数かStateかを判断して値を取得

    def content_update(self) -> None:
        self.set_props()
        self.update()


class ReactiveSlider(ft.Slider):
    def __init__(
        self,
        value: StateProperty[float],
        min: StateProperty[float] = 0.0,
        max: StateProperty[float] = 1.0,
        divisions: StateProperty[int | None] = None,
        label: StateProperty[str | None] = None,
        visible: StateProperty[bool] = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._value = value
        self._min = min
        self._max = max
        self._divisions = divisions
        self._label = label
        self._visible = visible

        self.set_props()
        bind_props(
            [self._value, self._min, self._max, self._divisions, self._label, self._visible],
            lambda: self.content_update(),
        )

    def set_props(self) -> None:
        self.value = get_prop_value(self._value)
        self.min = get_prop_value(self._min)
        self.max = get_prop_value(self._max)
        self.divisions = get_prop_value(self._divisions)
        self.label = get_prop_value(self._label)
        self.visible = get_prop_value(self._visible)

    def content_update(self) -> None:
        self.set_props()
        self.update()


ReactiveSlide = ReactiveSlider
