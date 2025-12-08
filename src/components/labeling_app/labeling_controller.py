from __future__ import annotations

from components.log_view import update_logs
from typing import Dict, List, Optional, cast
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import binary_erosion

from components.labeling_app.label_propagation import InteractiveLabelPropagation
from components.labeling_app.visualization import render_overlay_base64
from components.labeling_app.label_controls import LabelSelectionPane
from components.labeling_app.reset import reset_filter_tab
# from components.labeling_app.reset import reset_filter_tab
from stores import LabelingMap, Stores, as_ComputationResult, save_in_ComputationResultState
from components.page_tab.tabs.reset_onclick import reset_onclick_classify_button
from niconavi.grain_segmentation.grain_segmentation import (
    analyze_false_components_features,
    component_info_to_feature_matrix,
)

from niconavi.run_all import analyze_grain_list
from components.progress_bar import update_progress_bar
from components.log_view import update_logs
from niconavi.type import (
    Grain,
    GrainSelectedResult,
)

INITIAL_LABELS: Dict[int, int] = {}


def add_minera_name_to_grain_list(
    grain_list: list[Grain], grain_classification_result: dict[str, GrainSelectedResult]
) -> list[Grain]:
    for grain in grain_list:
        for classification_name, result in grain_classification_result.items():
            if grain["index"] in result["index"]:
                grain["mineral"] = classification_name
                break  # 最初に見つかった分類名で代入したらループを抜ける
    return grain_list

class LabelingController:
    def __init__(self, stores: Stores) -> None:
        self.stores = stores
        self.labeling = stores.labeling
        self.label_selection: Optional[LabelSelectionPane] = None
        self.labeling_map = self._ensure_labeling_map()

    @property
    def clf(self) -> Optional[InteractiveLabelPropagation]:
        return cast(
            Optional[InteractiveLabelPropagation],
            self.stores.labeling_shared.clf.get(),
        )

    @clf.setter
    def clf(self, value: Optional[InteractiveLabelPropagation]) -> None:
        self.stores.labeling_shared.clf.force_set(value)

    # ------------------------------------------------------------------
    # Wiring helpers
    # ------------------------------------------------------------------
    def attach_label_selection(self, pane: LabelSelectionPane) -> None:
        self.label_selection = pane
        self.stores.labeling_shared.register_label_selection(pane)

    # ------------------------------------------------------------------
    # Public handlers (invoked by UI controls)
    # ------------------------------------------------------------------
    def on_load_clicked(self, _e=None) -> None:

        if self.labeling._loaded.get():
            return
        self.labeling.load_status_text.set(
            ""
            # "Loading..."
        )
        self.labeling.load_button_disabled.set(True)

        self._initialize_app()
        self.labeling._loaded.set(True)
        self.labeling.load_status_text.set(
            ""
            # "Data loaded."
        )

        self.refresh_visuals(update_stats=True)


    # reset_filter_tab(self.stores)
    def reset_application(self, _e=None) -> None:
        if self.labeling.load_button_disabled.get() and not self.labeling._loaded.get():
            self.labeling.load_button_disabled.set(False)

        self.labeling._loaded.set(False)
        self.labeling.load_status_text.set(
            ""
        )
        self.labeling.load_button_disabled.set(False)
        self.labeling.user_clicked.set(False)
        self.labeling.labels.set({})
        self.labeling._next_class_id.set(1)
        self.labeling._reusable_class_ids.set([])
        self.labeling.current_class.set(None)
        self.labeling.palette.set([])
        self.labeling._clicked_indices_cache.set(None)
        self.labeling.show_boundaries.set(True)
        self.labeling.show_training_boxes.set(True)
        self.labeling.background_toggle_text.set("Background: Boundaries + Photo")
        # self.labeling.status_text.set("Add a label and click on the image.")
        # self.labeling.last_action_text.set("")
        self.labeling.labeled_stats_text.set("")
        self.labeling.prediction_stats_text.set("")
        self.labeling.legend_entries.set([])
        self.labeling.custom_colors.set({})
        self.labeling.overlay_alpha.set(0.65)
        self.labeling.display_predictions.set(None)
        self.labeling.results.set(None)

        self.labeling.image_src_base64.set("")
        self.labeling.image_width.set(0)
        self.labeling.image_height.set(0)
        self.labeling.image_display_width.set(0)
        self.labeling.image_display_height.set(0)

        for attr in (
            "index_map",
            "boundary_mask",
            "background_image",
            "features",
            "predictions",
            "probabilities",
        ):
            getattr(self.labeling_map, attr).set(None)

        self.clf = None

        ui_state = self.stores.labeling_computation_result.get("ui_state", {})
        shared_controls = self.stores.labeling_computation_result.get("shared_controls", {})
        self.stores.labeling_computation_result.clear()
        self.stores.labeling_computation_result.update(
            {"ui_state": ui_state, "shared_controls": shared_controls}
        )

        self.stores.labeling_shared.clear_label_selections()
        self.stores.labeling_shared.update_label_colors({})







    def handle_label_added(self, label_name: str) -> Optional[int]:
        if not self.labeling._loaded.get():
            return None

        labels = self.labeling.labels.get()
        if any(existing == label_name for existing in labels.values()):
            # self.labeling.last_action_text.set(f"Label '{label_name}' already exists.")
            update_logs(self.stores, (f"Label '{label_name}' already exists.", "warn"))

            return None

        class_id = self.labeling._next_class_id.get()
        self.labeling._next_class_id.set(class_id + 1)

        updated_labels = dict(labels)
        updated_labels[class_id] = label_name
        self.labeling.labels.set(updated_labels)
        self.labeling.current_class.set(class_id)
        self._update_status_text()
        self._push_label_color_update()
        return class_id

    def handle_label_removed(self, class_id: int) -> tuple[bool, Optional[int]]:
        if not self.labeling._loaded.get():
            return False, self.labeling.current_class.get()
        labels = dict(self.labeling.labels.get())
        if class_id not in labels:
            return False, self.labeling.current_class.get()

        labels.pop(class_id, None)
        self.labeling.labels.set(labels)
        self._remove_custom_color(class_id)
        self._clear_class_assignments(class_id)

        next_selection = self.labeling.current_class.get()
        if next_selection == class_id:
            next_selection = self._first_label_id(labels)

        if not labels:
            self.labeling.user_clicked.set(False)

        self.labeling.current_class.set(next_selection)
        self._update_status_text()
        self.refresh_visuals(update_stats=True)
        self._push_label_color_update()
        return True, next_selection

    def handle_label_selected(self, class_id: Optional[int]) -> None:
        if not self.labeling._loaded.get():
            return
        labels = self.labeling.labels.get()
        if class_id is not None and class_id not in labels:
            return
        self.labeling.current_class.set(class_id)
        self._update_status_text()

    def handle_overlay_opacity_change(self, e) -> None:
        if not self.labeling._loaded.get():
            return
        value = getattr(e.control, "value", None)
        try:
            alpha = float(value)
        except (TypeError, ValueError):
            return
        alpha = max(0.0, min(1.0, alpha))
        if abs(self.labeling.overlay_alpha.get() - alpha) < 1e-6:
            return
        self.labeling.overlay_alpha.set(alpha)
        self.refresh_visuals(update_stats=False)

    def handle_label_color_changed(self, class_id: int, color_hex: str) -> None:
        if not self.labeling._loaded.get():
            return
        labels = self.labeling.labels.get()
        if class_id not in labels:
            return
        try:
            normalized = mcolors.to_hex(color_hex)
        except ValueError:
            return
        current_colors = dict(self.labeling.custom_colors.get() or {})
        if current_colors.get(class_id) == normalized:
            return
        current_colors[class_id] = normalized
        self.labeling.custom_colors.set(current_colors)
        self.refresh_visuals(update_stats=False)
        self._push_label_color_update()

    def handle_show_boundaries_change(self, e) -> None:
        value = getattr(e.control, "value", None)
        show = bool(value)
        if show == bool(self.labeling.show_boundaries.get()):
            if not self.labeling._loaded.get():
                self.labeling.show_boundaries.set(show)
            return
        self.labeling.show_boundaries.set(show)
        if self.labeling._loaded.get():
            self.refresh_visuals(update_stats=False)

    def handle_show_training_boxes_change(self, e) -> None:
        value = getattr(e.control, "value", None)
        show = bool(value)
        if show == bool(self.labeling.show_training_boxes.get()):
            if not self.labeling._loaded.get():
                self.labeling.show_training_boxes.set(show)
            return
        self.labeling.show_training_boxes.set(show)
        if self.labeling._loaded.get():
            self.refresh_visuals(update_stats=False)

    def on_image_tap(self, e) -> None:
        if not self.labeling._loaded.get():
            return
        index_map = self.labeling_map.index_map.get()
        predictions = self.labeling_map.predictions.get()
        if index_map is None or predictions is None or self.clf is None:
            return
        class_id = self.labeling.current_class.get()
        labels = self.labeling.labels.get()
        if class_id is None or class_id not in labels:
            # self.labeling.last_action_text.set("Add or select a label.")
            update_logs(self.stores, ("Add or select a label.", "warn"))
            return

        original_width = int(index_map.shape[1])
        original_height = int(index_map.shape[0])
        control = getattr(e, "control", None)
        ui_state = self.stores.labeling_computation_result.get("ui_state", {})
        stored_width = (
            ui_state.get("rendered_image_width")
            or self.labeling.image_display_width.get()
        )
        stored_height = (
            ui_state.get("rendered_image_height")
            or self.labeling.image_display_height.get()
        )
        display_width = self._effective_dimension(
            control, "width", stored_width, default=original_width
        )
        display_height = self._effective_dimension(
            control, "height", stored_height, default=original_height
        )
        scale_x = original_width / display_width if display_width else 1.0
        scale_y = original_height / display_height if display_height else 1.0
        x = int(round(e.local_x * scale_x))
        y = int(round(e.local_y * scale_y))
        if x < 0:
            x = 0
        elif x >= original_width:
            x = original_width - 1
        if y < 0:
            y = 0
        elif y >= original_height:
            y = original_height - 1

        region_index = int(index_map[y, x])
        if region_index == 0:
            # self.labeling.last_action_text.set(
            #     "Clicked a boundary region (not labelable)."
            # )
            return
        if region_index >= predictions.size or region_index < 0:
            # self.labeling.last_action_text.set(
            #     f"Clicked invalid region (index={region_index})."
            # )
            return

        self.clf.set_label(region_index, class_id=class_id)
        self.clf.propagate()
        self._set_labeling_param("predictions", self.clf.current_predictions())
        self._set_labeling_param("probabilities", self.clf.current_probabilities())
        self.labeling.user_clicked.set(True)
        self.labeling._clicked_indices_cache.set(None)
        self.labeling.display_predictions.set(self._compute_display_predictions())
        self.refresh_visuals(update_stats=True)
        label_name = labels.get(class_id, f"Class {class_id}")
        # self.labeling.last_action_text.set(
        #     f"Set region index={region_index} to '{label_name}'."
        # )




    def finish_labeling(self, _e=None) -> None:
        if not self.labeling._loaded.get():
            return

        predictions = self.labeling_map.predictions.get()
        if predictions is None:
            return

        predictions = np.asarray(predictions).astype(np.int32, copy=True)

        if self.clf is not None and np.any(predictions == 0):
            self.clf.propagate()
            refreshed = self.clf.current_predictions()
            if refreshed is not None:
                predictions = np.asarray(refreshed).astype(np.int32, copy=True)
            if np.any(predictions == 0):
                probabilities = self.clf.current_probabilities()
                classes = self.clf.lp_.classes_ if self.clf.lp_ is not None else None
                if probabilities is not None and classes is not None:
                    classes = np.asarray(classes)
                    nonzero_mask = classes != 0
                    if np.any(nonzero_mask):
                        nonzero_classes = classes[nonzero_mask]
                        nonzero_probs = probabilities[:, nonzero_mask]
                        if nonzero_probs.size != 0:
                            best_idx = nonzero_probs.argmax(axis=1)
                            fill_mask = predictions == 0
                            if np.any(fill_mask):
                                predictions[fill_mask] = nonzero_classes[best_idx[fill_mask]]

        if np.any(predictions == 0):
            nonzero_vals = predictions[predictions != 0]
            if nonzero_vals.size:
                fallback = np.bincount(nonzero_vals).argmax()
                predictions[predictions == 0] = fallback

        self.labeling_map.predictions.set(predictions)

        # print("-------------")
        # print(np.unique(predictions))
        # print("-------------")

        # reset existing results
        update_progress_bar(None, self.stores)
        r = as_ComputationResult(self.stores.computation_result)
        r = reset_onclick_classify_button(r)
        save_in_ComputationResultState(r, self.stores)
        update_progress_bar(0, self.stores)


        labels = self.labeling.labels.get()
        palette = self.labeling.palette.get() or []
        color_map = self._current_color_map()
        results: Dict[str, Dict] = {}

        label_ids = [cid for cid, name in labels.items() if cid != 0 and name]
        assigned_classes = sorted(set(predictions[(predictions != 0) & np.isin(predictions, label_ids)]))

        for class_id in assigned_classes:
            label_name = labels.get(class_id)
            if not label_name:
                continue
            indices = np.flatnonzero(predictions == class_id).astype(np.int32)
            if indices.size == 0:
                continue
            color_hex = color_map.get(class_id)
            if color_hex is None and class_id < len(palette):
                color_hex = mcolors.to_hex(palette[class_id])
            if color_hex is None:
                color_hex = "#000000"
            results[label_name] = {
                "color": color_hex,
                "index": indices,
                "display": True,
            }
        self.labeling.results.set(results)
        self.stores.labeling_computation_result["grain_classification_result"] = results
        self.stores.computation_result.grain_classification_result.set(results)

        # self.labeling.last_action_text.set("Saved classification results.")
        update_logs(self.stores, (f"Saved classification results.", "ok"))


        r = as_ComputationResult(self.stores.computation_result)

        r = analyze_grain_list(
            r,
            progress_callback=lambda p: update_progress_bar(p, self.stores),
        )


        # print("--------------")
        # print(r.grain_list[0])
        # print(results)
        # print("--------------")

        new_grain_list = add_minera_name_to_grain_list(r.grain_list, results)
        r.grain_list = new_grain_list


        update_progress_bar(0.0, self.stores)
        update_logs(self.stores, ("Grain analysis completed.", "ok"))
        save_in_ComputationResultState(r, self.stores)
        # if res.mask is not None:
        #     if res.grain_classification_code is not None:
        #         if "mask[purple]: index == 999999" not in res.grain_classification_code:
        #             res.grain_classification_code = (
        #                 res.grain_classification_code
        #                 + "\n"
        #                 + "mask[purple]: index == 999999"
        #             )
        #     else:
        #         res.grain_classification_code = (
        #             "" + "\n" + "mask[purple]: index == 999999"
        #         )


        self.stores.ui.selected_index.set(4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_app(self) -> None:
        comp = self.stores.labeling_computation_result
        required_keys = {"index_map", "boundary_mask", "background_image", "features"}
        if required_keys.issubset(comp.keys()):
            index_map = comp["index_map"]
            boundary_mask = comp["boundary_mask"]
            background_image = comp["background_image"]
            features = comp["features"]
        else:

            R_color_map = self.stores.computation_result.raw_maps.get()["R_color_map_raw"]
            index_map = self.stores.computation_result.grain_map.get()
            grain_boundary = self.stores.computation_result.grain_boundary.get()
            boundary_mask = self.stores.computation_result.grain_boundary.get()

            mat, info = analyze_false_components_features(
                grain_boundary != 0, R_color_map, connectivity=4
            )
            features = component_info_to_feature_matrix(info)

            background_image = R_color_map

            # index_map = np.load("data/index_map.npy")
            # boundary_mask = np.load("data/data.npy").astype(bool)
            # background_image = np.load("data/pic.npy")
            # features = np.load("data/features.npy")

            comp.update(
                {
                    "index_map": index_map,
                    "boundary_mask": boundary_mask,
                    "background_image": background_image,
                    "features": features,
                }
            )

        self._set_labeling_param("index_map", index_map)
        self._set_labeling_param("boundary_mask", boundary_mask)
        self._set_labeling_param("background_image", background_image)
        self._set_labeling_param("features", features)

        self.clf = InteractiveLabelPropagation(
            n_neighbors=5,
            kernel="knn",
            reject_threshold=0.55,
        )
        if features is None:
            raise RuntimeError("features failed to load")
        self.clf.fit_features(features)
        for idx, cls in INITIAL_LABELS.items():
            self.clf.set_label(idx, class_id=cls)
        self.clf.propagate()
        self._set_labeling_param("predictions", self.clf.current_predictions())
        self._set_labeling_param("probabilities", self.clf.current_probabilities())

        self.labeling.user_clicked.set(False)
        self.labeling.display_predictions.set(self._compute_display_predictions())
        self.labeling.labels.set({})
        self.labeling._next_class_id.set(1)
        self.labeling.current_class.set(None)
        self.labeling.palette.set([])
        self.labeling._clicked_indices_cache.set(None)
        self.labeling.show_boundaries.set(True)
        self.labeling.show_training_boxes.set(True)
        self.labeling.background_mode.set("boundary_photo")
        self.labeling.background_toggle_text.set("Background: Boundaries + Photo")
        # self.labeling.status_text.set("Add a label and click on the image.")
        self.labeling.last_action_text.set("")
        self.labeling.labeled_stats_text.set("")
        self.labeling.prediction_stats_text.set("")
        self.labeling.legend_entries.set([])
        self.labeling.custom_colors.set({})
        self.labeling.overlay_alpha.set(0.65)

        self.stores.labeling_shared.clear_label_selections()
        self.stores.labeling_shared.update_label_colors({})

        index_map = self.labeling_map.index_map.get()
        if index_map is None:
            raise RuntimeError("index_map failed to load")
        width = int(index_map.shape[1])
        height = int(index_map.shape[0])
        self.labeling.image_width.set(width)
        self.labeling.image_height.set(height)
        self.labeling.image_src_base64.set("")
        self.update_display_dimensions()

    def refresh_visuals(self, update_stats: bool = False) -> None:
        if not self.labeling._loaded.get():
            return
        index_map = self.labeling_map.index_map.get()
        boundary_mask = self.labeling_map.boundary_mask.get()
        background_image = self.labeling_map.background_image.get()
        if index_map is None or boundary_mask is None or background_image is None:
            return

        display_predictions = self._compute_display_predictions()
        if display_predictions.size == 0:
            return
        self.labeling.display_predictions.set(display_predictions.copy())

        border_mask = (
            self._build_clicked_border_mask()
            if self.labeling.show_training_boxes.get()
            else None
        )
        overlay_alpha = float(max(0.0, min(1.0, self.labeling.overlay_alpha.get())))
        custom_colors = self.labeling.custom_colors.get() or {}
        show_boundaries = bool(self.labeling.show_boundaries.get())
        image_base64, palette = render_overlay_base64(
            index_map,
            display_predictions,
            overlay_alpha=overlay_alpha,
            boundary_mask=boundary_mask,
            background_image=background_image,
            show_boundaries=show_boundaries,
            highlight_border_mask=border_mask,
            custom_colors=custom_colors,
        )
        self.labeling.palette.set(list(palette))
        color_map = self._build_label_color_map(palette)
        self.stores.labeling_shared.update_label_colors(color_map)
        self.labeling.legend_entries.set(self._build_legend_entries(palette))
        self.labeling.image_src_base64.set(image_base64)

        if update_stats:
            self._update_stats()

    def _build_label_color_map(self, palette: List) -> Dict[int, str]:
        color_map: Dict[int, str] = {}
        labels = self.labeling.labels.get()
        custom_colors = self.labeling.custom_colors.get() or {}
        for class_id, color in enumerate(palette):
            if class_id == 0:
                continue
            label_name = labels.get(class_id)
            if not label_name:
                continue
            override = custom_colors.get(class_id)
            if override is not None:
                color_map[class_id] = override
            else:
                color_map[class_id] = mcolors.to_hex(color)
        return color_map

    def _build_legend_entries(self, palette: List) -> list[dict[str, str | int]]:
        entries: list[dict[str, str | int]] = []
        labels = self.labeling.labels.get()
        custom_colors = self.labeling.custom_colors.get() or {}
        for class_id, color in enumerate(palette):
            if class_id == 0:
                continue
            label_name = labels.get(class_id)
            if not label_name:
                continue
            hex_color = custom_colors.get(class_id) or mcolors.to_hex(color)
            entries.append({"color": hex_color, "label": label_name, "class_id": class_id})
        if not entries:
            entries.append({"color": "#cccccc", "label": "Background only", "class_id": 0})
        return entries

    def _push_label_color_update(self) -> None:
        color_map = self._current_color_map()
        self.stores.labeling_shared.update_label_colors(color_map)

    def _current_color_map(self) -> Dict[int, str]:
        labels = self.labeling.labels.get()
        palette = self.labeling.palette.get() or []
        custom_colors = self.labeling.custom_colors.get() or {}
        color_map: Dict[int, str] = {}
        for class_id, label_name in labels.items():
            if class_id == 0 or not label_name:
                continue
            color = custom_colors.get(class_id)
            if color is None and class_id < len(palette):
                try:
                    color = mcolors.to_hex(palette[class_id])
                except ValueError:
                    color = None
            if color is not None:
                color_map[class_id] = color
        return color_map

    def _remove_custom_color(self, class_id: int) -> None:
        current_colors = dict(self.labeling.custom_colors.get() or {})
        if current_colors.pop(class_id, None) is not None:
            self.labeling.custom_colors.set(current_colors)

    def _update_stats(self) -> None:
        if self.clf is None:
            return
        predictions = self.labeling.display_predictions.get()
        if predictions is None:
            return
        labeled_mask = self.clf.labeled_mask()
        labeled_total = int(np.count_nonzero(labeled_mask))
        labeled_counts = np.bincount(
            self.clf.y_user_[labeled_mask],
            minlength=int(self.clf.y_user_.max() + 1) if labeled_total else 1,
        )
        labels = self.labeling.labels.get()
        labeled_parts: List[str] = []
        for cls_idx, count in enumerate(labeled_counts.tolist()):
            if cls_idx == 0 or count == 0:
                continue
            labeled_parts.append(f"{labels.get(cls_idx, f'Class {cls_idx}')}:{count}")
        if not labeled_parts:
            labeled_parts = ["None"]
        self.labeling.labeled_stats_text.set(
            f"Labeled samples: {labeled_total} ({' / '.join(labeled_parts)})"
        )

        unique, counts = np.unique(predictions, return_counts=True)
        predicted_parts: List[str] = []
        for cls, count in zip(unique.tolist(), counts.tolist()):
            if cls == 0 or count <= 0:
                continue
            predicted_parts.append(f"{labels.get(cls, f'Class {cls}')}:{count}")
        prediction_text = "Predicted class distribution: " + (
            " / ".join(predicted_parts) if predicted_parts else "Background only"
        )
        self.labeling.prediction_stats_text.set(prediction_text)

    def _update_status_text(self) -> None:
        labels = self.labeling.labels.get()
        current = self.labeling.current_class.get()
        if not labels:
            text = "Add a label and click on the image."
        elif current is None:
            text = "Select a label."
        else:
            name = labels.get(current, f"Class {current}")
            text = f"Selected label: {name}"
        # self.labeling.status_text.set(text)

    def _compute_display_predictions(self) -> np.ndarray:
        predictions = self.labeling_map.predictions.get()
        if predictions is None:
            return np.array([], dtype=int)
        if self.clf is None:
            return predictions
        if not self.labeling.user_clicked.get():
            return predictions
        probabilities = self.labeling_map.probabilities.get()
        # print("---------")
        # print("probabilities")
        # print(probabilities)
        # print("---------")
        if probabilities is None:
            return predictions
        classes = self.clf.lp_.classes_ if self.clf.lp_ is not None else None
        if classes is None:
            return predictions
        nonzero_mask = classes != 0
        if not np.any(nonzero_mask):
            return predictions
        nonzero_classes = classes[nonzero_mask]
        nonzero_probs = probabilities[:, nonzero_mask]
        if nonzero_probs.size == 0:
            return predictions
        best_idx = nonzero_probs.argmax(axis=1)
        return nonzero_classes[best_idx]

    def _clear_class_assignments(self, class_id: int) -> None:
        if self.clf is None or class_id == 0:
            return
        self.clf.clear_class(class_id)
        self.clf.propagate()
        self._set_labeling_param("predictions", self.clf.current_predictions())
        self._set_labeling_param("probabilities", self.clf.current_probabilities())
        self.labeling.display_predictions.set(self._compute_display_predictions())
        self.labeling._clicked_indices_cache.set(None)
        self._remove_custom_color(class_id)
        self._push_label_color_update()

    @staticmethod
    def _effective_dimension(
        control, attr: str, stored: Optional[int], default: int
    ) -> float:
        candidates = [stored, default]
        if control is not None:
            candidates.insert(0, getattr(control, attr, None))
            content = getattr(control, "content", None)
            if content is not None:
                candidates.insert(0, getattr(content, attr, None))
                inner = getattr(content, "content", None)
                if inner is not None:
                    candidates.insert(0, getattr(inner, attr, None))
        for candidate in candidates:
            if isinstance(candidate, (int, float)) and candidate > 0:
                return float(candidate)
        return float(default)

    def _build_clicked_border_mask(self) -> Optional[np.ndarray]:
        if not self.labeling._loaded.get() or self.clf is None:
            return None
        index_map = self.labeling_map.index_map.get()
        if index_map is None:
            return None
        clicked_indices = self._cached_clicked_indices()
        if clicked_indices.size == 0:
            return None
        region_mask = np.isin(index_map, clicked_indices)
        if not np.any(region_mask):
            return None
        eroded = binary_erosion(
            region_mask, structure=np.ones((3, 3), dtype=bool), border_value=0
        )
        border_mask = region_mask & ~eroded
        return border_mask

    def _cached_clicked_indices(self) -> np.ndarray:
        cached = self.labeling._clicked_indices_cache.get()
        if cached is not None:
            return cached
        if self.clf is None:
            indices = np.array([], dtype=np.int32)
        else:
            labeled_mask = self.clf.labeled_mask()
            if labeled_mask is None or labeled_mask.size == 0:
                indices = np.array([], dtype=np.int32)
            else:
                indices = np.flatnonzero(labeled_mask).astype(np.int32, copy=False)
                indices = indices[indices != 0]
        self.labeling._clicked_indices_cache.set(indices)
        return indices

    def update_display_dimensions(
        self,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        allow_scale_up: bool = False,
    ) -> None:
        index_map = self.labeling_map.index_map.get()
        if index_map is None:
            return
        original_width = int(index_map.shape[1])
        original_height = int(index_map.shape[0])
        scale_candidates: list[float] = []
        if max_width is not None and max_width > 0:
            scale_candidates.append(max_width / original_width)
        if max_height is not None and max_height > 0:
            scale_candidates.append(max_height / original_height)
        if scale_candidates:
            scale = min(scale_candidates)
        else:
            scale = 1.0
        if not allow_scale_up:
            scale = min(scale, 1.0)
        width = max(int(round(original_width * scale)), 1)
        height = max(int(round(original_height * scale)), 1)
        self.labeling.image_display_width.set(width)
        self.labeling.image_display_height.set(height)

    def _set_labeling_param(self, attr: str, value: Optional[np.ndarray]) -> None:
        state = getattr(self.labeling_map, attr)
        state.set(self._copy_optional_array(value))

    def _ensure_labeling_map(self) -> LabelingMap:
        labeling_map = self.stores.labeling_shared.labeling_map
        self.labeling.labeling_param.force_set(labeling_map)
        return labeling_map

    @staticmethod
    def _copy_optional_array(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if array is None:
            return None
        return array.copy()

    @staticmethod
    def _first_label_id(labels: Dict[int, str]) -> Optional[int]:
        if not labels:
            return None
        return sorted(labels.keys())[0]
