import flet as ft
import flet.canvas as cv
import numpy as np
from niconavi_app.niconavi.optics.uniaxial_plate import get_retardation_color_chart
from niconavi_app.niconavi.image.type import D1RGB_Array
from niconavi_app.niconavi.tools.type import D1FloatArray

from niconavi_app.components.common_component import (
    CustomText,
)


def make_color_bar(
    start: float,
    middle: float,
    end: float,
    height: float = 10,
    col_start: str = ft.Colors.AMBER,
    col_middle: str = ft.Colors.BLUE,
    col_end: str = ft.Colors.RED,
) -> ft.Container:
    return ft.Container(
        content=ft.Row(
            controls=[
                ft.Container(
                    expand=int(start * 100),
                    bgcolor=col_start,
                    height=height,
                    margin=0,
                    padding=0,
                ),
                ft.Container(
                    expand=int((middle - start) * 100),
                    bgcolor=col_middle,
                    height=height,
                    margin=0,
                    padding=0,
                ),
                ft.Container(
                    expand=int((end - middle) * 100),
                    bgcolor=col_end,
                    height=height,
                    margin=0,
                    padding=0,
                ),
                ft.Container(
                    expand=int((1 - end) * 100),
                    bgcolor="black",
                    height=height,
                    padding=0,
                    # margin=ft.margin.only(left=10),
                    margin=0,
                ),
            ],
            spacing=0,
        ),
        height=height,
        margin=ft.margin.only(bottom=2),
    )


def rgb_array_to_hex_colors(color_array: D1RGB_Array) -> list[str]:
    """
    (N, 3) のRGB配列を #RRGGBB 形式の文字列リストに変換する。
    """
    return [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b) in color_array]


_MINOR_TICK_COUNT = 100
_MAJOR_TICK_COUNT = 10
_MIN_LABEL_SPACING = 50


def _label_stride(width: float) -> int:
    """Return a readable major-tick interval for the available width."""
    major_tick_spacing = max(width, 0) / _MAJOR_TICK_COUNT
    for stride in (1, 2, 5, 10):
        if major_tick_spacing * stride >= _MIN_LABEL_SPACING:
            return stride
    return 10


def _build_scale_shapes(width: float, tick_height: float) -> list[cv.Line | cv.Text]:
    """Draw ticks and labels in one coordinate system so they stay aligned."""
    width = max(width, 0)
    white_paint = ft.Paint(color=ft.Colors.WHITE, stroke_width=1)
    label_style = ft.TextStyle(size=12, color=ft.Colors.WHITE)
    shapes: list[cv.Line | cv.Text] = []

    for index in range(_MINOR_TICK_COUNT + 1):
        x = width * index / _MINOR_TICK_COUNT
        is_major = index % (_MINOR_TICK_COUNT // _MAJOR_TICK_COUNT) == 0
        shapes.append(
            cv.Line(
                x1=x,
                y1=0,
                x2=x,
                y2=tick_height if is_major else tick_height / 2,
                paint=white_paint,
            )
        )

    stride = _label_stride(width)
    for index in range(0, _MAJOR_TICK_COUNT + 1, stride):
        x = width * index / _MAJOR_TICK_COUNT
        if index == 0:
            alignment = ft.alignment.top_left
        elif index == _MAJOR_TICK_COUNT:
            alignment = ft.alignment.top_right
        else:
            alignment = ft.alignment.top_center
        shapes.append(
            cv.Text(
                x=x,
                y=tick_height + 2,
                text=str(index * 200),
                style=label_style,
                alignment=alignment,
            )
        )

    return shapes


class ResponsiveScale(cv.Canvas):
    def __init__(self, tick_height: float = 10) -> None:
        self.tick_height = tick_height
        super().__init__(
            height=tick_height + 20,
            resize_interval=50,
            on_resize=self._handle_resize,
        )

    def _handle_resize(self, event: cv.CanvasResizeEvent) -> None:
        self.shapes = _build_scale_shapes(event.width, self.tick_height)
        self.update()


def build_mark_row(
    mark_positions: D1FloatArray, tick_width: int, tick_height: int
) -> ft.Row:
    """
    mark_positions (0～1範囲の値) に対応する位置に目盛り線を配置する Row を作成する。
    各目盛りは背景の等間隔分布に対して、対象位置のみ強調表示します。
    """
    mark_positions_background = np.linspace(0, 1, 100)
    index_list = np.abs(mark_positions[:, None] - mark_positions_background).argmin(
        axis=1
    )
    children = []
    for i in range(len(mark_positions_background)):
        if i in index_list:
            children.append(
                ft.Container(
                    width=tick_width, height=tick_height, bgcolor=ft.Colors.AMBER
                )
            )
        else:
            children.append(
                ft.Container(
                    width=tick_width, height=tick_height, bgcolor=ft.Colors.TRANSPARENT
                )
            )
    return ft.Row(
        controls=children,
        height=tick_height,
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )


def build_mark_row_label(
    mark_positions: D1FloatArray, tick_width: int, tick_height: int
) -> ft.Row:
    """
    mark_positions (0～1範囲の値) に対応する位置に目盛りラベルを配置する Row を作成する。
    """
    mark_positions_background = np.linspace(0, 1, 100)
    index_list = np.abs(mark_positions[:, None] - mark_positions_background).argmin(
        axis=1
    )
    children = []
    for i in range(len(mark_positions_background)):
        if i in index_list:
            children.append(
                ft.Container(
                    content=CustomText("▼", size=10),
                    width=tick_width,
                    alignment=ft.alignment.center,
                )
            )
        else:
            children.append(ft.Container(width=0))
    return ft.Row(
        controls=children,
        height=20,
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )


def build_histogram(
    hist_vals: np.ndarray, bins: int, hist_height: float, max_val: float
) -> ft.Row:
    """
    hist_vals から np.histogram を用いてヒストグラムを作成し、
    各ビンの頻度に応じた高さで棒グラフを ft.Row で表示します。
    """
    hist_vals = hist_vals[hist_vals < max_val]
    hist_vals_new = np.concatenate([[0], hist_vals, [max_val]])

    counts, bin_edges = np.histogram(hist_vals_new, bins=bins)
    counts[0] -= counts[0]
    counts[-1] -= counts[-1]
    max_count = counts.max() if counts.max() > 0 else 1
    bars = []
    for count in counts:
        # 各ビンの高さを、最大値に対する割合で計算
        bar_h = (count / max_count) * hist_height
        bars.append(
            ft.Container(
                height=bar_h,
                bgcolor=ft.Colors.WHITE,
                expand=1,
            )
        )
    return ft.Row(
        controls=bars,
        height=hist_height,
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.END,
    )


class ColorChartBar(ft.Column):
    def __init__(self) -> None:
        super().__init__()
        # サイズ設定
        hist_height = 50  # ヒストグラムの高さ
        gradient_height = 20  # カラーバーの高さ
        tick_height = 10  # 目盛り線の高さ
        max_val = 2000

        # ヒストグラム用のデータ（例として np.random.random を使用）
        hist_vals = np.random.random(100) * 500
        bins = 200  # ビンの個数を指定

        # ヒストグラムRowを作成（カラーバーの上に配置）
        histogram_row = build_histogram(hist_vals, bins, hist_height, max_val)

        # グラデーション用のRGB配列（赤→黄→緑→水→青）
        chart, _ = get_retardation_color_chart(0, max_val, 30)
        color_bar = chart[0]
        gradient_bar = ft.Container(
            height=gradient_height,
            gradient=ft.LinearGradient(
                begin=ft.alignment.center_left,  # 左から
                end=ft.alignment.center_right,  # 右へ
                colors=rgb_array_to_hex_colors(color_bar),
            ),
        )

        scale = ResponsiveScale(tick_height)

        bar1 = make_color_bar(0.2, 0.7, 0.9)

        # ヒストグラム、補助バー、グラデーションバー、目盛り、ラベルを縦に配置
        self.controls = [
            # histogram_row,  # ヒストグラム（カラーバーの上部）
            # bar1,
            gradient_bar,  # グラデーションカラーバー
            scale,  # カラーバー下部の目盛りとラベル
        ]
        self.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        self.spacing = 0


if __name__ == "__main__":

    def main(page: ft.Page):
        # ページ下部に配置する例
        colorbar = ColorChartBar()
        page.vertical_alignment = ft.MainAxisAlignment.END
        page.add(colorbar)

    ft.app(target=main)
