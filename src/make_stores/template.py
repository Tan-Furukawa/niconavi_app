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
        self.niconavi_version = "NicoNavi v0.2.2"
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
        self.one_pixel = State(0) # 0のとき、未定で、>0のとき、その値にする。
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
        self.progress = State(0)


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
