
#!------------------------------------------------------
#! This code is automatically generated.
#! このコードは自動生成されています。手動で編集をしないでください。
#!------------------------------------------------------

from state import State
from niconavi.image.type import Color, RGBPicture, MonoColorPicture, D1RGB_Array
from typing import TypedDict, Optional, Literal, cast, Any
from niconavi.image.type import RGBPicture
from niconavi.type import (
    QuartzWedgeNormalization,
    GrainDetectionParameters,
    ComputationResult,
    PlotParameters,
    TiltImageResult,
    ColorChart,
    TiltImageInfo,
    OpticalParameters,
    FirstVideoImage,
    GrainNumLiteral,
    # RawMapsNumLiteral,
)
import numpy as np
from niconavi.tools.type import (
    D2BoolArray,
    D2FloatArray,
    D2IntArray,
    D1FloatArray,
    D1IntArray
)
from niconavi.image.type import Color, RGBPicture, MonoColorPicture
from niconavi.type import (
    Grain,
    GrainFeaturesForClassification,
    RawMaps,
    GrainSegmentedMaps,
    GrainSelectedResult,
    CIPMapInfo,
)
from matplotlib.pyplot import Figure, Axes
import flet as ft
from typing import Any, Dict, Optional, TYPE_CHECKING


# ------------------------------------------------------------
# Appearance
# ------------------------------------------------------------
class Appearance:
    def __init__(self) -> None:
        self.niconavi_version = "NicoNavi v0.2.1"
        self.tabs_width = 400
        self.tab_padding = 20
        self.progress_bar_height = 50
        self.log_area_height = 100
        self.button_active_color = "#222222"
        self.button_inactive_color = "#444444"


# ----------------------------------------------
# |                               |            |
# |                               |            |
# |                               |            |
# |           page                |     tab    |
# |                               |            |
# |                               |            |
# |                               |------------|
# |                               |     log    |
# ----------------------------------------------

# # ------------------------------------------------------------
# # Labeling
# # ------------------------------------------------------------


if TYPE_CHECKING:
    from components.labeling_app.label_controls import LabelSelectionPane


# ------------------------------------------------------------
# Labeling
# ------------------------------------------------------------
class LabelingMap:
    def __init__(self) -> None:
        self.index_map: State[Optional[Any]] = State(None)
        self.boundary_mask: State[Optional[Any]] = State(None)
        self.background_image: State[Optional[Any]] = State(None)
        self.predictions: State[Optional[Any]] = State(None)
        self.probabilities: State[Optional[Any]] = State(None)
        self.features: State[Optional[Any]] = State(None)




class Labeling:
    def __init__(self) -> None:
        self.labels: State[dict[int, str]] = State({})
        self._next_class_id: State[int] = State(1)
        self._reusable_class_ids: State[list[int]] = State([])
        self.current_class: State[Optional[int]] = State(None)
        self.palette: State[list] = State([])
        self.user_clicked: State[bool] = State(False)
        self.background_mode: State[str] = State("boundary_photo")
        self._loaded: State[bool] = State(False)
        self.labeling_param: State[Optional[LabelingMap]] = State(None)
        self.display_predictions: State[Optional[np.ndarray]] = State(None)
        self._clicked_indices_cache: State[Optional[Any]] = State(None)
        self.results: State[Optional[Dict[str, Any]]] = State(None)
        self.load_status_text: State[str] = State(
            "No data is loaded yet. Click Load to continue."
        )
        self.load_button_disabled: State[bool] = State(False)
        self.status_text: State[str] = State("Add a label, then click on the image.")
        self.last_action_text: State[str] = State("")
        self.labeled_stats_text: State[str] = State("")
        self.prediction_stats_text: State[str] = State("")
        self.background_toggle_text: State[str] = State("Background: Boundaries and Photo")
        self.legend_entries: State[list[dict[str, str]]] = State([])
        self.show_boundaries: State[bool] = State(True)
        self.show_training_boxes: State[bool] = State(True)
        self.custom_colors: State[dict[int, str]] = State({})
        self.overlay_alpha: State[float] = State(0.65)
        self.image_src_base64: State[str] = State("")
        self.image_width: State[int] = State(0)
        self.image_height: State[int] = State(0)
        self.image_display_width: State[int] = State(0)
        self.image_display_height: State[int] = State(0)
        self.custom_colors: State[dict[int, str]] = State({})
        self.overlay_alpha: State[float] = State(0.65)

class LabelingShared:
    def __init__(self) -> None:
        self.labeling_map = LabelingMap()
        self.clf: State[Optional[Any]] = State(None)
        self._label_selection_panes: list["LabelSelectionPane"] = []

    def register_label_selection(self, pane: "LabelSelectionPane") -> None:
        if pane not in self._label_selection_panes:
            self._label_selection_panes.append(pane)

    def unregister_label_selection(self, pane: "LabelSelectionPane") -> None:
        if pane in self._label_selection_panes:
            self._label_selection_panes.remove(pane)

    def clear_label_selections(self) -> None:
        for pane in list(self._label_selection_panes):
            pane.clear()

    def update_label_colors(self, colors: Dict[int, str]) -> None:
        for pane in list(self._label_selection_panes):
            pane.update_colors(colors)

    def populate_labels(self, labels: Dict[int, str], colors: Dict[int, str]) -> None:
        for pane in list(self._label_selection_panes):
            pane.clear()
            pane.update_colors(colors)
            for class_id, label_name in labels.items():
                pane.add_label(class_id, label_name, select=False)
            pane._set_selected(next(iter(labels), None))


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
class ProgressBar:
    def __init__(self) -> None:
        self.progress: State[float | None] = State(0.0)
        self.message = State("")
        self.visible = State(True)


class ImageViewer:
    def __init__(self) -> None:
        self.noImage: RGBPicture = cast(
            RGBPicture, np.zeros((10, 10, 3), dtype=np.uint8)
        )
        # self.first_video_frame: State[Optional[RGBPicture]] = State(None)
        # self.first_reta_video_frame: State[Optional[RGBPicture]] = State(None)
        self.grain_boundary_color: State[str] = State("white")
        self.background_color: State[str] = State("black")


class MovieTabObj:
    def __init__(self) -> None:
        # self.initial_cross_image: State[Optional[RGBPicture]] = State(None)
        # self.initial_reta_image: State[Optional[RGBPicture]] = State(None)
        pass


class LogView:
    def __init__(self) -> None:
        self.log_contents: State[
            list[tuple[str, Literal["ok", "err", "msg", "warn"]]]
        ] = State([])


class AnalysisTab:
    def __init__(self) -> None:
        self.computation_unit: State[Literal["pixel", "grain"]] = State("grain")
        self.selected_keys: Optional[str] = None
        self.plot_option: State[
            Literal["rose diagram", "histogram", "scatter", "CPO", "SPO"]
        ] = State("rose diagram")
        self.grain_rose_diagram_target: State[
            Literal["azimuth", "extinction_angle", "angle_deg"]
            # Literal["azimuth", "extinction_angle"]
        ] = State("extinction_angle")
        # self.map_rose_diagram_target: State[Literal["azimuth", "extinction_angle"]] = (
        #     State("extinction_angle")
        # )
        self.grain_histogram_target: State[GrainNumLiteral] = State("R")
        # self.map_histogram_target: State[RawMapsNumLiteral] = State(
        #     "max_retardation_map"
        # )
        self.scatter_target_x: State[GrainNumLiteral] = State("R")
        self.scatter_target_y: State[GrainNumLiteral] = State("extinction_angle")
        self.scatter_show_regression: State[bool] = State(True)
        self.scatter_regression_origin: State[bool] = State(True)
        self.scatter_log_x: State[bool] = State(False)
        self.scatter_log_y: State[bool] = State(False)
        self.histogram_log_x: State[bool] = State(False)
        self.histogram_alpha: State[float] = State(0.5)
        self.rose_alpha: State[float] = State(0.5)
        self.rose_flip: State[bool] = State(True)
        self.cip_bandwidth: State[float] = State(6.)
        self.cip_contour: State[int] = State(10)
        self.cip_theme: State[str] = State("jet")
        self.cip_display_points: State[bool] = State(False)
        self.cip_points_noise_size_percent: State[float] = State(0.5)
        self.histogram_stats_text: State[str] = State(
            "Mean: -\nStd Dev: -\nMin: -\nMax: -\n95th percentile: -\nMode: -\nCount: -\nIntegral ratio: -\nCount ratio: -"
        )
        self.rose_stats_text: State[str] = State(
            "Mean orientation: -\nCircular variance: -"
        )


class GrainTab:
    def __init__(self) -> None:
        self.slider_brightness: State[float] = State(1.0)
        self.slider_contrast: State[float] = State(1.0)
        self.slider_median_kernel: State[int] = State(1)
        self.brightness_correction: State[bool] = State(True)
        self.use_brightness: bool = False #! True: under develop
        self.use_contrast: bool = False #! True: under develop


class UI:
    def __init__(self) -> None:
        self.selected_index = State(0)
        self.one_pixel = State(None) # Noneのとき、px表示で、それ以外のときはμm表示する。
        self.image_button_index = State(0)
        self.progress_bar = ProgressBar()
        self.movie_tab = MovieTabObj()
        self.grain_tab = GrainTab()
        self.log_view = LogView()
        self.image_viewer = ImageViewer()
        self.force_update_image_view = State(0)
        self.analysis_tab = AnalysisTab()
        self.display_grain_boundary = State(True)
        self.apply_mask = State(False)
        self.selected_button_at_common_image_view = State(0)
        self.display_common_image_view = State(False)
        # self.once_click_start_button_at_movie_tab = State(False) 
        self.selected_button_at_movie_tab = State(0)  # image viewの上のボタン
        self.selected_button_at_grain_tab = State(5)  # image viewの上のボタン
        self.selected_button_at_filter_tab = State(0)  # image viewの上のボタン
        self.selected_button_at_merge_tab = State(15)  # image viewの上のボタン
        self.selected_button_at_analysis_tab = State(0)  # image viewの上のボタン
        self.dummy_str: State[Optional[str]] = State(
            None
        )  #! 文字自動修正を行うコードに使用する。変な使い方してる。
        self.computing_is_stop: State[bool] = State(True)
        self.once_start: State[bool] = State(False)
        self.displayed_fig: State[Optional[Figure]] = State(None)


# ------------------------------------------------------------
# Stores
# ------------------------------------------------------------


class Stores:
    def __init__(self) -> None:
        self.ui = UI()
        self.appearance = Appearance()
        # self.computation_parameters = ComputationResult()
        self.computation_result = ComputationResultState()  # type: ignore
        self.labeling = Labeling()

        self.labeling_computation_result: Dict[str, Any] = {
            "ui_state": {},
            "shared_controls": {},
        }
        self.labeling = Labeling()
        self.labeling_shared = LabelingShared()
        self.labeling.labeling_param.force_set(self.labeling_shared.labeling_map)


        # self.labeling_computation_result: dict[str, Any] = {
        #     "ui_state": {},
        #     "shared_controls": {},
        # }
        # self.labeling = Labeling()


def reset_labeling_stores(stores: Stores) -> None:
    stores.labeling = Labeling()
    stores.labeling_computation_result = {
        "ui_state": {},
        "shared_controls": {},
    }
    stores.labeling = Labeling()
    stores.labeling_shared = LabelingShared()
    stores.labeling.labeling_param.force_set(stores.labeling_shared.labeling_map)


#! ----------------------------------------------
#! 以下ComputationResultStateの定義が自動生成されるので、なにも記入しないこと
#! ----------------------------------------------
# {{{from make stores}}}

class GrainDetectionParametersState:
    def __init__(self) -> None:
        self.smallest_grain_size: State[int] = State(10)
        self.th_about_hessian_emphasis: State[float] = State(0.7)
        self.th_about_connect_skeleton_endpoints: State[float] = State(10)
        self.permit_inclusion: State[bool] = State(True)


def as_GrainDetectionParameters(param: GrainDetectionParametersState) -> GrainDetectionParameters:
    res_dict = {}
    for key, value in param.__dict__.items():
        if isinstance(value, State):
            res_dict[key] = value.get()
    return GrainDetectionParameters(**res_dict)

class QuartzWedgeNormalizationState:
    def __init__(self) -> None:
        self.no: State[float] = State(1.544)
        self.ne: State[float] = State(1.553)
        self.thin_section_thickness: State[Optional[float]] = State(0.03)
        self.quartz_wedge_path: State[Optional[str]] = State(None)
        self.raw_data: State[Optional[D1RGB_Array]] = State(None)
        self.calibrated_data: State[Optional[D1RGB_Array]] = State(None)
        self.retardation: State[Optional[D1FloatArray]] = State(None)
        self.R: State[Optional[float]] = State(1000.0)
        self.max_retardation_for_inclination_search: State[Optional[float]] = State(None)
        self.max_retardation_for_1st_order_color_plate: State[float] = State(1700.0)


def as_QuartzWedgeNormalization(param: QuartzWedgeNormalizationState) -> QuartzWedgeNormalization:
    res_dict = {}
    for key, value in param.__dict__.items():
        if isinstance(value, State):
            res_dict[key] = value.get()
    return QuartzWedgeNormalization(**res_dict)

class TiltImageInfoState:
    def __init__(self) -> None:
        self.estimate_inclination_by: State[Literal['thickness', 'max R']] = State('max R')
        self.theta_thin_section: State[float] = State(10.0)
        self.frame_num: State[int] = State(20)
        self.image0_path: State[Optional[str]] = State(None)
        self.tilt_image0_path: State[Optional[str]] = State(None)
        self.image0_raw: State[Optional[list[RGBPicture]]] = State(None)
        self.tilt_image0_raw: State[Optional[list[RGBPicture]]] = State(None)
        self.tilt_image0: State[Optional[TiltImageResult]] = State(None)
        self.image45_raw: State[Optional[list[RGBPicture]]] = State(None)
        self.image45_path: State[Optional[str]] = State(None)
        self.tilt_image45_path: State[Optional[str]] = State(None)
        self.tilt_image45_raw: State[Optional[list[RGBPicture]]] = State(None)
        self.tilt_image45: State[Optional[TiltImageResult]] = State(None)


def as_TiltImageInfo(param: TiltImageInfoState) -> TiltImageInfo:
    res_dict = {}
    for key, value in param.__dict__.items():
        if isinstance(value, State):
            res_dict[key] = value.get()
    return TiltImageInfo(**res_dict)

class PlotParametersState:
    def __init__(self) -> None:
        self.rose_diagram90_bins: State[int] = State(30)
        self.rose_diagram180_bins: State[int] = State(30)
        self.histogram_bins: State[int] = State(30)
        self.rose_diagram_bins: State[int] = State(30)


def as_PlotParameters(param: PlotParametersState) -> PlotParameters:
    res_dict = {}
    for key, value in param.__dict__.items():
        if isinstance(value, State):
            res_dict[key] = value.get()
    return PlotParameters(**res_dict)

class OpticalParametersState:
    def __init__(self) -> None:
        self.no: State[float] = State(1.544)
        self.ne: State[float] = State(1.553)
        self.tilt_deg: State[float] = State(4.0)
        self.max_R: State[Optional[float]] = State(None)
        self.thickness: State[float] = State(0.03)
        self.xpl_retardation_color_chart_used: State[Optional[RGBPicture]] = State(None)


def as_OpticalParameters(param: OpticalParametersState) -> OpticalParameters:
    res_dict = {}
    for key, value in param.__dict__.items():
        if isinstance(value, State):
            res_dict[key] = value.get()
    return OpticalParameters(**res_dict)

class ColorChartState:
    def __init__(self) -> None:
        self.xpl_alpha: State[Optional[float]] = State(None)
        self.pol_lambda_alpha: State[Optional[float]] = State(None)
        self.inc_alpha: State[Optional[float]] = State(None)
        self.xpl_max_retardation: State[float] = State(300)
        self.pol_lambda_max_retardation: State[Optional[float]] = State(1500.0)
        self.inc_max_retardation: State[Optional[float]] = State(300 + 530)
        self.xpl_retardation_color_chart: State[Optional[D1RGB_Array]] = State(None)
        self.pol_lambda_retardation_color_chart: State[Optional[D1RGB_Array]] = State(None)
        self.inc_retardation_color_chart: State[Optional[D1RGB_Array]] = State(None)
        self.xpl_R_array: State[Optional[D1FloatArray]] = State(None)
        self.pol_lambda_R_array: State[Optional[D1FloatArray]] = State(None)
        self.inc_R_array: State[Optional[D1FloatArray]] = State(None)


def as_ColorChart(param: ColorChartState) -> ColorChart:
    res_dict = {}
    for key, value in param.__dict__.items():
        if isinstance(value, State):
            res_dict[key] = value.get()
    return ColorChart(**res_dict)

class ComputationResultState:
    def __init__(self) -> None:
        self.video_path: State[Optional[str]] = State(None)
        self.original_resolution: State[Optional[tuple[int, int]]] = State(None)
        self.original_reta_resolution: State[Optional[tuple[int, int]]] = State(None)
        self.reta_video_path: State[Optional[str]] = State(None)
        self.mask: State[Optional[D2BoolArray]] = State(None)
        self.resolution_width: State[int] = State(1000)
        self.full_wave_plate_nm: State[float] = State(530.0)
        self.circ_threshold: State[float] = State(0.5)
        self.angle_between_x_and_thin_section_axis_at_tilt: State[float] = State(45)
        self.use_raw_in_grain_boundary_detection: State[bool] = State(False)
        self.quartz_wedge_normalization: State[QuartzWedgeNormalization] = State(QuartzWedgeNormalization())
        self.resolution_height: State[Optional[int]] = State(None)
        self.frame_number: State[int] = State(100)
        self.pics: State[Optional[list[RGBPicture]]] = State(None)
        self.pics_rotated: State[Optional[list[RGBPicture]]] = State(None)
        self.reta_pics: State[Optional[list[RGBPicture]]] = State(None)
        self.reta_pics_rotated: State[Optional[list[RGBPicture]]] = State(None)
        self.angles: State[Optional[D1FloatArray]] = State(None)
        self.reta_angles: State[Optional[D1FloatArray]] = State(None)
        self.center_int_x: State[Optional[int]] = State(None)
        self.center_int_y: State[Optional[int]] = State(None)
        self.rotation_img: State[Optional[MonoColorPicture]] = State(None)
        self.image_rotation_direction: State[Optional[Literal['clockwise', 'counterclockwise']]] = State(None)
        self.reta_image_rotation_direction: State[Optional[Literal['clockwise', 'counterclockwise']]] = State(None)
        self.grain_list: State[Optional[list[Grain]]] = State(None)
        self.grain_features_list_for_classification: State[Optional[list[GrainFeaturesForClassification]]] = State(None)
        self.grain_detection_parameters: GrainDetectionParametersState = GrainDetectionParametersState()
        self.grain_map: State[Optional[D2IntArray]] = State(None)
        self.grain_map_original: State[Optional[D2IntArray]] = State(None)
        self.grain_map_with_boundary: State[Optional[D2IntArray]] = State(None)
        self.grain_boundary: State[Optional[D2BoolArray]] = State(None)
        self.grain_boundary_original: State[Optional[D2BoolArray]] = State(None)
        self.raw_maps: State[Optional[RawMaps]] = State(None)
        self.first_image: State[FirstVideoImage] = State(FirstVideoImage(xpl=None, full_wave=None, image0=None, image0_tilt=None, image45=None, image45_tilt=None))
        self.cip_map_info: State[CIPMapInfo] = State(CIPMapInfo(polar_info180=None, polar_info360=None, polar_info90=None, COI180_grain=None, COI360_grain=None, COI90_grain=None, COI180=None, COI360=None, COI90=None, legend180=None, legend360=None, legend90=None))
        self.grain_segmented_maps: State[Optional[GrainSegmentedMaps]] = State(None)
        self.grain_marge_code: State[Optional[str]] = State(None)
        self.grain_classification_result: State[Optional[dict[str, GrainSelectedResult]]] = State(None)
        self.grain_classification_code: State[Optional[str]] = State(None)
        self.grain_classification_image: State[Optional[RGBPicture]] = State(None)
        self.grain_classification_legend: State[Optional[dict[str, Color]]] = State(None)
        self.plot_parameters: PlotParametersState = PlotParametersState()
        self.optical_parameters: OpticalParametersState = OpticalParametersState()
        self.tilt_image_info: TiltImageInfoState = TiltImageInfoState()
        self.color_chart: ColorChartState = ColorChartState()





def save_in_GrainDetectionParametersState(param: GrainDetectionParameters, stores: Stores) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.grain_detection_parameters.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])

def save_in_PlotParametersState(param: PlotParameters, stores: Stores) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.plot_parameters.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])

def save_in_OpticalParametersState(param: OpticalParameters, stores: Stores) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.optical_parameters.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])

def save_in_TiltImageInfoState(param: TiltImageInfo, stores: Stores) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.tilt_image_info.__dict__
    for key in d:
        if isinstance(d[key], State):
            d[key].set(p_dict[key])

def save_in_ColorChartState(param: ColorChart, stores: Stores) -> None:
    p_dict = param.__dict__
    d = stores.computation_result.color_chart.__dict__
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
