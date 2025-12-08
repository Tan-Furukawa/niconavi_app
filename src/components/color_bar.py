import flet as ft
import numpy as np
from typing import NewType
from numpy.typing import NDArray
from niconavi.optics.uniaxial_plate import get_retardation_color_chart
from niconavi.image.type import D1RGB_Array
from niconavi.tools.type import D1FloatArray
import math
import base64

from components.common_component import (
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


def build_ticks_row(
    tick_values: D1FloatArray, tick_width: int, tick_height: int
) -> ft.Row:
    """
    tick_values の各値に対応する位置に目盛り線を配置する Row を作成する。
    Row の alignment を SPACE_BETWEEN にすることで、各目盛りが均等配置されます。
    """
    children = []
    for i, _ in enumerate(tick_values):
        if i % 10 == 0:
            children.append(
                ft.Container(width=tick_width, height=tick_height, bgcolor="white")
            )
        else:
            children.append(
                ft.Container(width=tick_width, height=tick_height / 2, bgcolor="white")
            )
    return ft.Row(
        controls=children,
        height=tick_height,
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


def build_labels_row(tick_values: D1FloatArray, label_width: int) -> ft.Row:
    """
    各目盛りに対して、ラベル（テキスト）を均等配置する Row を作成する。
    最初と最後のラベルは左右に半分の幅を割り当てています。
    """
    children = []
    for i, tick in enumerate(tick_values):
        if i == 0:
            label = ft.Container(
                content=CustomText(f"{tick:.0f}", size=12),
                width=label_width // 2,
                alignment=ft.alignment.bottom_left,
            )
        elif i == len(tick_values) - 1:
            label = ft.Container(
                content=CustomText("", size=12),
                width=label_width // 2,
                alignment=ft.alignment.bottom_right,
            )
        else:
            label = ft.Container(
                content=CustomText(f"{tick:.0f}", size=12),
                width=label_width,
                alignment=ft.alignment.center,
            )
        children.append(label)
    return ft.Row(controls=children, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)


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
        tick_width = 2  # 目盛り線の幅
        label_width = 30  # 各目盛りラベルの幅
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

        # 目盛り・ラベルRowの作成
        tick_values_ticks = D1FloatArray(np.linspace(0, 200, 101))
        ticks_row = build_ticks_row(tick_values_ticks, tick_width, tick_height)
        tick_values_labels = D1FloatArray(np.linspace(0, 2000, 11))
        labels_row = build_labels_row(tick_values_labels, label_width)

        bar1 = make_color_bar(0.2, 0.7, 0.9)

        # ヒストグラム、補助バー、グラデーションバー、目盛り、ラベルを縦に配置
        self.controls = [
            # histogram_row,  # ヒストグラム（カラーバーの上部）
            # bar1,
            gradient_bar,  # グラデーションカラーバー
            ticks_row,  # カラーバー下部目盛り
            labels_row,  # ラベル
        ]
        self.horizontal_alignment = "center"
        self.spacing = 0


if __name__ == "__main__":

    def main(page: ft.Page):
        # ページ下部に配置する例
        colorbar = ColorChartBar()
        page.vertical_alignment = ft.MainAxisAlignment.END
        page.add(colorbar)

    ft.app(target=main)
