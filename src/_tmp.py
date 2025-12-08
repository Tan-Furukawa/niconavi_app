from state import State

from typing import TypedDict, Optional, Literal, cast
from niconavi.image.type import RGBPicture
from niconavi.type import (
    GrainDetectionParameters,
    ComputationResult,
    PlotParameters,
    TiltImageResult,
    ColorChart,
    TiltImageInfo,
    OpticalParameters,
    GrainNumLiteral,
    RawMapsNumLiteral,
)
import numpy as np
from niconavi.tools.type import (
    D2BoolArray,
    D2FloatArray,
    D2IntArray,
    D1FloatArray,
)
from niconavi.image.type import Color, RGBPicture, MonoColorPicture
from niconavi.type import Grain, RawMaps, GrainSegmentedMaps, GrainSelectedResult
from matplotlib.figure import Figure
import flet as ft

class GrainDetectionParametersState:
    def __init__(self) -> None:
        self.median_kernel_size: State[int] = State(3)
        self.diff_threshold: State[int] = State(15)
        self.smallest_grain_size: State[int] = State(10)
        self.extinction_color_is_less_than: State[int] = State(50)
        self.median_filter_size: State[int] = State(3)
        self.shortest_contour: State[float] = State(10.0)
        self.percentile: State[float] = State(50.0)
        self.morphological_transformations_kernel_size: State[int] = State(5)
        self.grain_boundary_logic: State[Literal["sobel", "binary", "color_map"]] = (
            State("color_map")
        )


class PlotParametersState:
    def __init__(self) -> None:
        self.rose_diagram90_bins: State[int] = State(50)
        self.rose_diagram180_bins: State[int] = State(100)
        self.histogram_bins: State[int] = State(100)


class OpticalParametersState:
    def __init__(self) -> None:
        self.no: State[float] = State(1.544)
        self.ne: State[float] = State(1.553)
        self.tilt_deg: State[float] = State(4.0)
        self.alpha: State[Optional[float]] = State(None)
        self.reta_alpha: State[Optional[float]] = State(None)
        self.max_R: State[Optional[float]] = State(None)
        self.thickness: State[Optional[float]] = State(None)

        self.xpl_retardation_color_chart_used: State[Optional[RGBPicture]] = State(None)


class TiltImageInfoState:
    def __init__(self) -> None:
        self.theta_thin_section: State[float] = State(4.0)
        self.frame_number: State[int] = State(20)
        self.image0_path: State[Optional[str]] = State(None)
        self.image45_path: State[Optional[str]] = State(None)
        self.tilt_image0_path: State[Optional[str]] = State(None)
        self.tilt_image45_path: State[Optional[str]] = State(None)
        self.tilt_image0: State[Optional[TiltImageResult]] = State(None)
        self.tilt_image45: State[Optional[TiltImageResult]] = State(None)


class ColorChartState:
    def __init__(self) -> None:
        self.alpha: State[Optional[float]] = State(None)
        self.xpl_alpha: State[Optional[float]] = State(None)
        self.inc_alpha: State[Optional[float]] = State(None)
        self.xpl_max_retardation: State[float] = State(1000.0)
        self.pol_lambda_max_retardation: State[float] = State(1500.0)
        self.inc_max_retardation: State[Optional[float]] = State(400)
        self.xpl_retardation_color_chart: State[Optional[RGBPicture]] = State(None)
        self.pol_lambda_retardation_color_chart: State[Optional[RGBPicture]] = State(
            None
        )
        self.inc_retardation_color_chart: State[Optional[RGBPicture]] = State(None)
        self.xpl_retardation_array: State[Optional[D1FloatArray]] = State(None)
        self.xpl_nd_filter_array: State[Optional[D1FloatArray]] = State(None)


class ComputationResultState:
    def __init__(self) -> None:

        self.video_path: State[str | None] = State(None)
        self.original_resolution: State[Optional[tuple[int, int]]] = State(None)
        self.original_reta_resolution: State[Optional[tuple[int, int]]] = State(None)
        self.reta_video_path: State[str | None] = State(None)
        self.xpl_max_retardation: State[float] = State(1500)
        self.xpl_min_nd_filter: State[float] = State(0.2)
        self.xpl_retardation_color_chart: State[Optional[RGBPicture]] = State(None)
        self.xpl_retardation_array: State[Optional[D1FloatArray]] = State(None)
        self.xpl_nd_filter_array: State[Optional[D1FloatArray]] = State(None)
        self.resolution_width: State[int] = State(1000)
        self.resolution_height: State[Optional[int]] = State(None)
        self.frame_number: State[int] = State(100)
        self.pics: State[Optional[list[RGBPicture]]] = State(None)
        self.reta_pics: State[Optional[list[RGBPicture]]] = State(None)
        self.pics_rotated: State[Optional[list[RGBPicture]]] = State(None)
        self.reta_pics_rotated: State[Optional[list[RGBPicture]]] = State(None)
        self.angles: State[Optional[D1FloatArray]] = State(None)
        self.reta_angles: State[Optional[D1FloatArray]] = State(None)
        # self.center_float: State[Optional[tuple[float, float]]] = State(None)
        self.center_int: State[Optional[tuple[int, int]]] = State(None)
        self.rotation_img: State[Optional[MonoColorPicture]] = State(None)
        self.rotation_img_with_mark: State[Optional[Figure]] = State(None)
        self.image_rotation_direction: State[
            Optional[Literal["clockwise", "counterclockwise"]]
        ] = State(None)
        self.grain_list: State[Optional[list[Grain]]] = State(None)
        self.grain_detection_parameters: GrainDetectionParametersState = (
            GrainDetectionParametersState()
        )
        self.grain_map: State[Optional[D2IntArray]] = State(None)
        self.grain_map_original: State[Optional[D2IntArray]] = State(None)
        self.grain_map_with_boundary: State[Optional[D2IntArray]] = State(None)
        self.grain_boundary: State[Optional[D2BoolArray]] = State(None)
        self.grain_boundary_original: State[Optional[D2BoolArray]] = State(None)
        self.raw_maps: State[Optional[RawMaps]] = State(None)
        self.grain_classification_result: State[
            Optional[dict[str, GrainSelectedResult]]
        ] = State(None)
        self.grain_classification_image: State[Optional[RGBPicture]] = State(None)
        self.grain_classification_legend: State[Optional[dict[str, Color]]] = State(
            None
        )
        self.grain_segmented_maps: State[Optional[GrainSegmentedMaps]] = State(None)
        self.grain_marge_code: State[Optional[str]] = State("")
        self.grain_classification_code: State[Optional[str]] = State(
            """// this is comment
// put classification code here

background[black]: index == 0
"""
        )

        self.plot_parameters: PlotParametersState = PlotParametersState()
        self.optical_parameters: OpticalParametersState = OpticalParametersState()
        self.tilt_image_info: TiltImageInfoState = TiltImageInfoState()
        self.color_chart: ColorChartState = ColorChartState()


def as_ColorChart(param: ColorChartState) -> ColorChart:
    res_dict = {}
    for key in param.__dict__:
        if isinstance(param.__dict__[key], State):
            res_dict[key] = param.__dict__[key].get()
    return ColorChart(**res_dict)


def as_TiltImageInfo(param: TiltImageInfoState) -> TiltImageInfo:
    res_dict = {}
    for key in param.__dict__:
        if isinstance(param.__dict__[key], State):
            res_dict[key] = param.__dict__[key].get()
    return TiltImageInfo(**res_dict)


def as_OpticalParameters(
    param: OpticalParametersState,
) -> OpticalParameters:
    res_dict = {}
    for key in param.__dict__:
        if isinstance(param.__dict__[key], State):
            res_dict[key] = param.__dict__[key].get()
    return OpticalParameters(**res_dict)


def as_PlotParameters(
    param: PlotParametersState,
) -> PlotParameters:
    res_dict = {}
    for key in param.__dict__:
        if isinstance(param.__dict__[key], State):
            # print(param.__dict__[key].get())
            res_dict[key] = param.__dict__[key].get()
    return PlotParameters(**res_dict)


def as_GrainDetectionParameters(
    param: GrainDetectionParametersState,
) -> GrainDetectionParameters:
    res_dict = {}
    for key in param.__dict__:
        if isinstance(param.__dict__[key], State):
            res_dict[key] = param.__dict__[key].get()
    return GrainDetectionParameters(**res_dict)


def as_ComputationResult(param: ComputationResultState) -> ComputationResult:
    res_dict = {}
    for key in param.__dict__:
        if isinstance(param.__dict__[key], State):
            res_dict[key] = param.__dict__[key].get()
        elif isinstance(param.__dict__[key], GrainDetectionParametersState):
            res_dict[key] = as_GrainDetectionParameters(param.__dict__[key])
        elif isinstance(param.__dict__[key], PlotParametersState):
            res_dict[key] = as_PlotParameters(param.__dict__[key])
        elif isinstance(param.__dict__[key], OpticalParametersState):
            res_dict[key] = as_OpticalParameters(param.__dict__[key])
        elif isinstance(param.__dict__[key], TiltImageInfoState):
            res_dict[key] = as_TiltImageInfo(param.__dict__[key])
        elif isinstance(param.__dict__[key], ColorChartState):
            res_dict[key] = as_ColorChart(param.__dict__[key])
        else:
            raise ValueError("unexpected type occurred in ComputationResultState")

    return ComputationResult(**res_dict)


def save_in_ColorChartState(param: ColorChartState, stores: Stores) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.color_chart.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])


def save_in_TiltImageInfoState(param: TiltImageInfoState, stores: Stores) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.tilt_image_info.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])


def save_in_OpticalParametersState(
    param: OpticalParametersState, stores: Stores
) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.optical_parameters.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])


def save_in_PlotParametersState(param: PlotParametersState, stores: Stores) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.plot_parameters.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])


def save_in_GrainDetectionParametersState(
    param: GrainDetectionParameters, stores: Stores
) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.grain_detection_parameters.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])


def save_in_ComputationResultState(param: ComputationResult, stores: Stores) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])
        elif isinstance(d[key], GrainDetectionParametersState):
            save_in_GrainDetectionParametersState(p_dict[key], stores)
        elif isinstance(d[key], PlotParametersState):
            save_in_PlotParametersState(p_dict[key], stores)
        elif isinstance(d[key], OpticalParametersState):
            save_in_OpticalParametersState(p_dict[key], stores)
        elif isinstance(d[key], TiltImageInfoState):
            save_in_TiltImageInfoState(p_dict[key], stores)
        elif isinstance(d[key], ColorChartState):
            save_in_ColorChartState(p_dict[key], stores)
        else:
            raise ValueError("unexpected type occurred in ComputationResult")


if __name__ == "__main__":
    r = Stores()
    k = as_ComputationResult(r.computation_result)
    print(k)
# %%
