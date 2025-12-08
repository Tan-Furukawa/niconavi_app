from __future__ import annotations

from stores import Stores
# from components.labeling_app.labeling_controller import LabelingController

_LABELING_MAP_FIELDS: tuple[str, ...] = (
    "index_map",
    "boundary_mask",
    "background_image",
    "features",
    "predictions",
    "probabilities",
)


def _has_labeling_payload(stores: Stores) -> bool:
    labeling = getattr(stores, "labeling", None)
    if labeling is None:
        return False
    if labeling._loaded.get():
        return True

    if labeling.labels.get():
        return True
    if labeling.image_src_base64.get():
        return True
    if labeling.display_predictions.get() is not None:
        return True

    labeling_shared = getattr(stores, "labeling_shared", None)
    labeling_map = getattr(labeling_shared, "labeling_map", None)
    if labeling_map is None:
        return False
    for field in _LABELING_MAP_FIELDS:
        state = getattr(labeling_map, field, None)
        if state is not None and state.get() is not None:
            return True

    return False


def reset_filter_tab(stores: Stores) -> None:
    labeling = stores.labeling

    if labeling.load_button_disabled.get() and not labeling._loaded.get():
        labeling.load_button_disabled.set(False)

    if not _has_labeling_payload(stores):
        return

    labeling._loaded.set(False)
    labeling.load_status_text.set("")
    labeling.load_button_disabled.set(False)
    labeling.user_clicked.set(False)
    labeling.labels.set({})
    labeling._next_class_id.set(1)
    labeling._reusable_class_ids.set([])
    labeling.current_class.set(None)
    labeling.palette.set([])
    labeling._clicked_indices_cache.set(None)
    labeling.show_boundaries.set(True)
    labeling.show_training_boxes.set(True)
    labeling.background_toggle_text.set("Background: Boundaries + Photo")
    labeling.status_text.set("Add a label and click on the image.")
    labeling.last_action_text.set("")
    labeling.labeled_stats_text.set("")
    labeling.prediction_stats_text.set("")
    labeling.legend_entries.set([])
    labeling.custom_colors.set({})
    labeling.overlay_alpha.set(0.65)
    labeling.display_predictions.set(None)
    labeling.results.set(None)

    labeling.image_src_base64.set("")
    labeling.image_width.set(0)
    labeling.image_height.set(0)
    labeling.image_display_width.set(0)
    labeling.image_display_height.set(0)

    labeling_shared = stores.labeling_shared
    labeling_map = labeling_shared.labeling_map
    for field in _LABELING_MAP_FIELDS:
        getattr(labeling_map, field).set(None)

    labeling_shared.clf.force_set(None)

    ui_state = stores.labeling_computation_result.get("ui_state", {})
    shared_controls = stores.labeling_computation_result.get("shared_controls", {})
    stores.labeling_computation_result.clear()
    stores.labeling_computation_result.update(
        {"ui_state": ui_state, "shared_controls": shared_controls}
    )

    labeling_shared.clear_label_selections()
    labeling_shared.update_label_colors({})

    # controller = LabelingController(stores=stores)
    # controller.reset_application()
    # controller.on_load_clicked()
