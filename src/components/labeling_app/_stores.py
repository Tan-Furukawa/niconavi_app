from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from state import State


if TYPE_CHECKING:
    from label_controls import LabelSelectionPane


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
        self.current_class: State[Optional[int]] = State(None)
        self.palette: State[list] = State([])
        self.background_mode: State[str] = State("boundary_photo")
        self.user_clicked: State[bool] = State(False)
        self._loaded: State[bool] = State(False)
        self.labeling_param: State[Optional[LabelingMap]] = State(None)
        self.display_predictions: State[Optional[np.ndarray]] = State(None)
        self._clicked_indices_cache: State[Optional[Any]] = State(None)
        self.results: State[Optional[Dict[str, Any]]] = State(None)
        self.load_status_text: State[str] = State(
            "Data has not been loaded yet. Press the button to load."
        )
        self.load_button_disabled: State[bool] = State(False)
        self.status_text: State[str] = State("Add a label and click on the image.")
        # self.last_action_text: State[str] = State("")
        self.labeled_stats_text: State[str] = State("")
        self.prediction_stats_text: State[str] = State("")
        self.background_toggle_text: State[str] = State("Background: Boundaries + Photo")
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


class Stores:
    def __init__(self) -> None:
        self.labeling_computation_result: Dict[str, Any] = {
            "ui_state": {},
            "shared_controls": {},
        }
        self.labeling = Labeling()
        self.labeling_shared = LabelingShared()
        self.labeling.labeling_param.force_set(self.labeling_shared.labeling_map)
