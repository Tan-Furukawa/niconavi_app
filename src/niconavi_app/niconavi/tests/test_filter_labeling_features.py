import cv2
import numpy as np
from types import SimpleNamespace

from niconavi_app.components.labeling_app.label_propagation import (
    InteractiveLabelPropagation,
)
import niconavi_app.components.labeling_app.labeling_controller as controller_module
from niconavi_app.components.labeling_app.labeling_controller import LabelingController
from niconavi_app.niconavi.grain_segmentation.grain_segmentation import (
    COMPONENT_FEATURE_COLUMNS,
    COMPONENT_FEATURE_KEYS,
    analyze_indexed_components_features,
    component_info_to_feature_matrix,
)
from niconavi_app.stores import Stores


class _ClassifierStub:
    def __init__(self) -> None:
        self.labels: list[tuple[int, int]] = []

    def set_label(self, region_index: int, *, class_id: int) -> None:
        self.labels.append((region_index, class_id))

    def propagate(self) -> None:
        pass

    def current_predictions(self) -> np.ndarray:
        return np.array([0, 1], dtype=np.int32)

    def current_probabilities(self) -> np.ndarray:
        return np.ones((2, 1), dtype=np.float64)


def test_default_labeling_features_include_size_but_not_position() -> None:
    stores = Stores()
    labeling = stores.labeling

    assert labeling.use_color_features.get() is True
    assert labeling.use_shape_features.get() is True
    assert labeling.use_size_features.get() is True
    assert labeling.use_position_features.get() is False

    weights = LabelingController(stores)._feature_weights(
        np.zeros((1, len(COMPONENT_FEATURE_KEYS)), dtype=np.float64)
    )
    assert np.all(weights[list(COMPONENT_FEATURE_COLUMNS["color"])] > 0)
    assert np.all(weights[list(COMPONENT_FEATURE_COLUMNS["shape"])] > 0)
    assert np.all(weights[list(COMPONENT_FEATURE_COLUMNS["size"])] > 0)
    assert np.all(weights[list(COMPONENT_FEATURE_COLUMNS["position"])] == 0)


def test_component_features_preserve_grain_ids_and_rgb_color() -> None:
    index_map = np.zeros((4, 6), dtype=np.int32)
    index_map[:, :3] = 2
    index_map[:, 3:] = 5
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :3] = (180, 30, 10)
    image[:, 3:] = (20, 80, 220)

    info = analyze_indexed_components_features(index_map, image)
    features = component_info_to_feature_matrix(info)
    expected_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    assert [item["index"] for item in info] == [2, 5]
    assert features.shape == (6, len(COMPONENT_FEATURE_KEYS))
    assert np.all(features[[0, 1, 3, 4]] == 0)
    assert features[2, 1] == np.log(12)
    assert features[2, 2] == np.median(expected_lab[:, :3, 0]) * 100.0 / 255.0
    assert features[2, 3] == np.median(expected_lab[:, :3, 1]) - 128.0
    assert features[2, 4] == np.median(expected_lab[:, :3, 2]) - 128.0
    assert features[5, 2] == np.median(expected_lab[:, 3:, 0]) * 100.0 / 255.0


def test_color_features_learn_dark_and_bright_grains() -> None:
    gray_levels = np.array([20, 45, 70, 95, 160, 185, 210, 235], dtype=np.uint8)
    index_map = np.tile(np.arange(1, 9, dtype=np.int32), (4, 1))
    image = np.repeat(gray_levels[None, :, None], 4, axis=0)
    image = np.repeat(image, 3, axis=2)
    features = component_info_to_feature_matrix(
        analyze_indexed_components_features(index_map, image)
    )

    weights = np.zeros(features.shape[1], dtype=np.float64)
    weights[list(COMPONENT_FEATURE_COLUMNS["color"])] = 0.02
    classifier = InteractiveLabelPropagation(
        n_neighbors=3,
        kernel="knn",
        reject_threshold=0.0,
        prototype_weight=1.0,
        scale_features=False,
    )
    classifier.fit_features(features, feature_weights=weights)
    classifier.set_label(1, class_id=1)
    classifier.set_label(2, class_id=1)
    classifier.set_label(7, class_id=2)
    classifier.set_label(8, class_id=2)
    classifier.propagate()

    predictions = classifier.current_predictions()
    assert np.all(predictions[1:5] == 1)
    assert np.all(predictions[5:9] == 2)


def test_shape_feature_is_rotation_invariant_elongation() -> None:
    index_map = np.zeros((24, 24), dtype=np.int32)
    index_map[2:5, 2:14] = 1
    index_map[8:20, 18:21] = 2
    index_map[12:18, 6:12] = 3
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    features = component_info_to_feature_matrix(
        analyze_indexed_components_features(index_map, image)
    )

    assert np.isclose(features[1, 0], features[2, 0])
    assert features[1, 0] > features[3, 0]


def test_size_feature_uses_logarithmic_component_area() -> None:
    index_map = np.zeros((16, 30), dtype=np.int32)
    index_map[1:3, 1:3] = 1
    index_map[1:4, 6:9] = 2
    index_map[1:5, 12:16] = 3
    index_map[1:7, 20:26] = 4
    image = np.zeros((16, 30, 3), dtype=np.uint8)
    features = component_info_to_feature_matrix(
        analyze_indexed_components_features(index_map, image)
    )

    size_column = COMPONENT_FEATURE_COLUMNS["size"][0]
    assert np.allclose(
        features[1:5, size_column],
        np.log(np.array([4, 9, 16, 36], dtype=np.float64)),
    )

    weights = np.zeros(features.shape[1], dtype=np.float64)
    weights[size_column] = 1.0
    classifier = InteractiveLabelPropagation(
        n_neighbors=3,
        kernel="knn",
        reject_threshold=0.0,
        prototype_weight=1.0,
        scale_features=False,
    )
    classifier.fit_features(features, feature_weights=weights)
    classifier.set_label(1, class_id=1)
    classifier.set_label(4, class_id=2)
    classifier.propagate()

    assert classifier.current_predictions()[1:5].tolist() == [1, 1, 2, 2]


def test_image_tap_waits_for_confirmation_when_analysis_exists(monkeypatch) -> None:
    stores = Stores()
    stores.ui.progress.set(4)
    stores.labeling._loaded.set(True)
    stores.labeling.labels.set({1: "Quartz"})
    stores.labeling.current_class.set(1)
    stores.labeling.labeling_param.get().index_map.set(
        np.ones((4, 4), dtype=np.int32)
    )
    stores.labeling.labeling_param.get().predictions.set(
        np.zeros(2, dtype=np.int32)
    )

    controller = LabelingController(stores)
    classifier = _ClassifierStub()
    controller.clf = classifier
    monkeypatch.setattr(
        controller, "_compute_display_predictions", lambda: np.array([0, 1])
    )
    monkeypatch.setattr(controller, "refresh_visuals", lambda **_kwargs: None)

    reset_calls: list[bool] = []
    monkeypatch.setattr(
        controller,
        "_reset_analysis_for_label_change",
        lambda: reset_calls.append(True),
    )
    pending_confirmation: dict[str, object] = {}

    def capture_confirmation(page, stores, stage, action, on_confirm) -> None:
        pending_confirmation.update(
            page=page,
            stores=stores,
            stage=stage,
            action=action,
            on_confirm=on_confirm,
        )

    monkeypatch.setattr(
        controller_module, "confirm_discard_downstream", capture_confirmation
    )
    page = object()
    event = SimpleNamespace(
        local_x=2.0,
        local_y=2.0,
        control=SimpleNamespace(width=4.0, height=4.0),
    )

    controller.on_image_tap(event, page=page)

    assert classifier.labels == []
    assert reset_calls == []
    assert pending_confirmation["page"] is page
    assert pending_confirmation["action"] == "Change mineral label"

    pending_confirmation["on_confirm"]()

    assert reset_calls == [True]
    assert classifier.labels == [(1, 1)]


def test_label_change_reset_discards_saved_analysis() -> None:
    stores = Stores()
    stores.ui.progress.set(4)
    stores.ui.selected_index.set(4)
    stores.ui.selected_button_at_analysis_tab.set(7)
    stores.labeling.results.set({"Quartz": {"index": np.array([1])}})
    stores.labeling_computation_result["grain_classification_result"] = {
        "Quartz": {"index": np.array([1])}
    }
    stores.computation_result.grain_classification_result.set(
        {"Quartz": {"index": np.array([1])}}
    )

    LabelingController(stores)._reset_analysis_for_label_change()

    assert stores.computation_result.grain_classification_result.get() is None
    assert stores.labeling.results.get() is None
    assert "grain_classification_result" not in stores.labeling_computation_result
    assert stores.ui.progress.get() == 3
    assert stores.ui.selected_index.get() == 3
    assert stores.ui.selected_button_at_analysis_tab.get() == 0
    assert stores.ui.computing_is_stop.get() is True
