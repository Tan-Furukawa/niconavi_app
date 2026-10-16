import cv2
import numpy as np

from niconavi_app.components.labeling_app.label_propagation import (
    InteractiveLabelPropagation,
)
from niconavi_app.niconavi.grain_segmentation.grain_segmentation import (
    COMPONENT_FEATURE_COLUMNS,
    COMPONENT_FEATURE_KEYS,
    analyze_indexed_components_features,
    component_info_to_feature_matrix,
)


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
    assert features[2, 1] == np.median(expected_lab[:, :3, 0]) * 100.0 / 255.0
    assert features[2, 2] == np.median(expected_lab[:, :3, 1]) - 128.0
    assert features[2, 3] == np.median(expected_lab[:, :3, 2]) - 128.0
    assert features[5, 1] == np.median(expected_lab[:, 3:, 0]) * 100.0 / 255.0


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
