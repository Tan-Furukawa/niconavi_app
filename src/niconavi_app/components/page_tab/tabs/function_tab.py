from __future__ import annotations

import flet as ft
import numpy as np
from matplotlib.figure import Figure

from niconavi_app.components.common_component import (
    CustomDropDown,
    CustomExecuteButton,
    CustomRadio,
    CustomText,
    CustomTextField,
)
from niconavi_app.niconavi.optics.tools import (
    get_max_retardation_from_thickness,
    get_thickness_from_max_retardation,
)
import niconavi_app.niconavi.optics.optical_system as optical_system
from niconavi_app.niconavi.optics.uniaxial_plate import get_spectral_distribution
from niconavi_app.stores import Stores


QUARTZ_OMEGA_REFRACTIVE_INDEX = 1.544
QUARTZ_EPSILON_REFRACTIVE_INDEX = 1.553
DEFAULT_THICKNESS_MM = 0.03
DEFAULT_MAX_RETARDATION_NM = get_max_retardation_from_thickness(
    DEFAULT_THICKNESS_MM,
    no=QUARTZ_OMEGA_REFRACTIVE_INDEX,
    ne=QUARTZ_EPSILON_REFRACTIVE_INDEX,
)


def _make_figure(
    *,
    thickness_mm: float,
    max_retardation_nm: float,
    no: float,
    ne: float,
) -> Figure:
    x_max = max(thickness_mm * 1.2, 0.001)
    y_max = max(max_retardation_nm * 1.2, 1.0)
    thickness_values = np.linspace(0.0, x_max, 200)
    retardation_values = get_max_retardation_from_thickness(
        thickness_values,
        no=no,
        ne=ne,
    )

    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot(thickness_values, retardation_values, color="black", linewidth=1.5)
    ax.scatter(
        [thickness_mm],
        [max_retardation_nm],
        color="black",
        s=28,
        zorder=3,
    )
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Thickness [mm]")
    ax.set_ylabel("Maximum retardation [nm]")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _parse_positive_float(value: str, field_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be larger than 0.")
    return parsed


def _retardation_interference_color(retardation_nm: float) -> tuple[int, int, int]:
    rgb = get_spectral_distribution(
        optical_system.get_retardation_system(R=retardation_nm, alpha=1)
    )["rgb"]
    red, green, blue = [int(value) for value in np.asarray(rgb, dtype=np.uint8)]
    return red, green, blue


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


class FunctionTab(ft.Container):
    def __init__(self, page: ft.Page, stores: Stores):
        super().__init__()
        self.padding = stores.appearance.tab_padding
        self.expand = True
        self.stores = stores

        self.mode = "thickness"
        self.last_thickness_mm = DEFAULT_THICKNESS_MM
        self.last_max_retardation_nm = DEFAULT_MAX_RETARDATION_NM

        self.feature_dropdown = CustomDropDown(
            value="thickness vs retardation",
            options=[ft.dropdown.Option("thickness vs retardation")],
        )
        self.feature_dropdown.width = 300

        self.no_input = CustomTextField(
            value=f"{QUARTZ_OMEGA_REFRACTIVE_INDEX:g}",
            width=70,
            height=30,
            content_padding=ft.padding.only(left=10, top=3, bottom=3),
        )
        self.ne_input = CustomTextField(
            value=f"{QUARTZ_EPSILON_REFRACTIVE_INDEX:g}",
            width=70,
            height=30,
            content_padding=ft.padding.only(left=10, top=3, bottom=3),
        )

        self.thickness_input = CustomTextField(
            value=f"{DEFAULT_THICKNESS_MM:g}",
            width=90,
            height=30,
            content_padding=ft.padding.only(left=10, top=3, bottom=3),
        )
        self.retardation_input = CustomTextField(
            value=f"{DEFAULT_MAX_RETARDATION_NM:.3g}",
            width=90,
            height=30,
            content_padding=ft.padding.only(left=10, top=3, bottom=3),
        )
        self.thickness_row = ft.Row(
            [self.thickness_input, CustomText("mm")],
            spacing=8,
        )
        self.retardation_row = ft.Row(
            [self.retardation_input, CustomText("nm")],
            spacing=8,
            visible=False,
        )

        self.result_text = CustomText(
            f"Maximum retardation: {DEFAULT_MAX_RETARDATION_NM:.3f} nm"
        )
        self.status_text = CustomText("")
        self.status_text.color = ft.Colors.RED_200
        self.interference_color = _retardation_interference_color(
            DEFAULT_MAX_RETARDATION_NM
        )
        self.color_swatch = ft.Container(
            width=34,
            height=24,
            bgcolor=_rgb_to_hex(self.interference_color),
            border=ft.border.all(1, ft.Colors.WHITE60),
        )
        self.color_text = CustomText(
            f"Interference color RGB: {self.interference_color}"
        )

        self.mode_radio = ft.RadioGroup(
            value=self.mode,
            content=ft.Column(
                [
                    CustomRadio(value="thickness", label="Thickness"),
                    CustomRadio(value="max_retardation", label="Maximum retardation"),
                ],
                spacing=0,
            ),
            on_change=self._on_mode_change,
        )

        self.calculate_button = CustomExecuteButton(
            "Calculate",
            on_click=self._on_calculate,
        )

        self.content = ft.Column(
            [
                CustomText("Function"),
                self.feature_dropdown,
                CustomText("Refractive index"),
                ft.Row(
                    [
                        CustomText("ω"),
                        self.no_input,
                        CustomText("ε"),
                        self.ne_input,
                    ],
                    spacing=8,
                    wrap=False,
                ),
                CustomText("Input"),
                self.mode_radio,
                self.thickness_row,
                self.retardation_row,
                self.calculate_button,
                self.result_text,
                ft.Row([self.color_swatch, self.color_text], spacing=8),
                self.status_text,
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        )

    def _on_mode_change(self, e: ft.ControlEvent) -> None:
        self.mode = str(e.control.value)
        self.thickness_row.visible = self.mode == "thickness"
        self.retardation_row.visible = self.mode == "max_retardation"
        self.status_text.value = ""
        self.update()

    def _on_calculate(self, _: ft.ControlEvent) -> None:
        try:
            no = _parse_positive_float(str(self.no_input.value), "ω")
            ne = _parse_positive_float(str(self.ne_input.value), "ε")
            if no == ne:
                raise ValueError("ω and ε must be different.")

            if self.mode == "thickness":
                thickness_mm = _parse_positive_float(
                    str(self.thickness_input.value),
                    "Thickness",
                )
                max_retardation_nm = get_max_retardation_from_thickness(
                    thickness_mm,
                    no=no,
                    ne=ne,
                )
                self.retardation_input.value = f"{max_retardation_nm:.6g}"
                self.result_text.value = (
                    f"Maximum retardation: {max_retardation_nm:.3f} nm"
                )
            else:
                max_retardation_nm = _parse_positive_float(
                    str(self.retardation_input.value),
                    "Maximum retardation",
                )
                thickness_mm = get_thickness_from_max_retardation(
                    max_retardation_nm,
                    no=no,
                    ne=ne,
                )
                self.thickness_input.value = f"{thickness_mm:.6g}"
                self.result_text.value = f"Thickness: {thickness_mm:.6f} mm"

            self.last_thickness_mm = float(thickness_mm)
            self.last_max_retardation_nm = float(max_retardation_nm)
            self.stores.ui.function_tab.figure.set(
                _make_figure(
                    thickness_mm=self.last_thickness_mm,
                    max_retardation_nm=self.last_max_retardation_nm,
                    no=no,
                    ne=ne,
                )
            )
            self.interference_color = _retardation_interference_color(
                self.last_max_retardation_nm
            )
            self.color_swatch.bgcolor = _rgb_to_hex(self.interference_color)
            self.color_text.value = f"Interference color RGB: {self.interference_color}"
            self.status_text.value = ""
        except ValueError as exc:
            self.status_text.value = str(exc)

        self.update()


def at_function_tab(stores: Stores) -> Figure:
    figure = stores.ui.function_tab.figure.get()
    if figure is not None:
        return figure
    return _make_figure(
        thickness_mm=DEFAULT_THICKNESS_MM,
        max_retardation_nm=DEFAULT_MAX_RETARDATION_NM,
        no=QUARTZ_OMEGA_REFRACTIVE_INDEX,
        ne=QUARTZ_EPSILON_REFRACTIVE_INDEX,
    )
