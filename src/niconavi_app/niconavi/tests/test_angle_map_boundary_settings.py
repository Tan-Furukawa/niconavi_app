from __future__ import annotations

import inspect
import logging

import numpy as np

from niconavi_app.components import app_bar
from niconavi_app.components.page_tab.tabs import grain_tab
from niconavi_app.niconavi.angle_map_boundary import (
    _absorb_small_regions,
    grain_boundary_from_angle_labels,
    segment_angle_map,
)
from niconavi_app.save import collect_project_ui_state
from niconavi_app.stores import Stores


def test_slic_setting_defaults_to_30_with_minimum_5() -> None:
    stores = Stores()

    assert stores.ui.map_tab.smallest_grain_linear_size.get() == 3
    assert stores.ui.map_tab.slic_superpixel_size.get() == 30
    assert grain_tab.parse_smallest_grain_size("1") == 1
    assert grain_tab.parse_smallest_grain_size("0") is None
    assert grain_tab.parse_slic_superpixel_size("5") == 5
    assert grain_tab.parse_slic_superpixel_size("30") == 30
    assert grain_tab.parse_slic_superpixel_size("4") is None
    assert inspect.signature(segment_angle_map).parameters["slic_ruler"].default == 20.0


def test_segment_uses_linear_slic_size(
    monkeypatch,
) -> None:
    stores = Stores()
    stores.ui.map_tab.smallest_grain_linear_size.set(3)
    stores.ui.map_tab.slic_superpixel_size.set(30)
    stores.ui.map_tab.angle_map_info.set({"input": True})
    received: dict[str, object] = {}

    def fake_segment(info, **kwargs):
        received.update(kwargs)
        return {
            **info,
            "angle_map_display": np.zeros((2, 2, 3), dtype=np.uint8),
            "merged_superpixel_labels": np.zeros((2, 2), dtype=np.int32),
        }

    monkeypatch.setattr(grain_tab, "segment_angle_map", fake_segment)
    grain_tab.segmentation_button_click(
        stores,
        None,
        logger=logging.getLogger("test-segmentation-settings"),
    )

    assert received["slic_region_size"] == 30
    assert received["small_region_min_area_px"] == 9


def test_small_regions_are_absorbed_only_below_configured_area() -> None:
    one_pixel = np.zeros((5, 5), dtype=np.int32)
    one_pixel[2, 2] = 1
    assert np.unique(_absorb_small_regions(one_pixel, min_area_px=4)).size == 1

    four_pixels = np.zeros((5, 5), dtype=np.int32)
    four_pixels[:2, :2] = 1
    assert np.unique(_absorb_small_regions(four_pixels, min_area_px=4)).size == 2


def test_apply_does_not_absorb_small_regions() -> None:
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 1

    result, _boundary = grain_boundary_from_angle_labels(
        {"black_artifact_removed_superpixel_labels": labels}
    )

    assert np.array_equal(result, labels)


def test_slic_setting_round_trips_through_project_state() -> None:
    stores = Stores()
    stores.ui.map_tab.smallest_grain_linear_size.set(6)
    stores.ui.map_tab.slic_superpixel_size.set(12)
    ui_state = collect_project_ui_state(stores)

    restored = Stores()
    app_bar.restore_map_tab_state(restored, ui_state)

    assert restored.ui.map_tab.smallest_grain_linear_size.get() == 6
    assert restored.ui.map_tab.slic_superpixel_size.get() == 12

    ui_state["map_tab"]["slic_superpixel_size"] = 3
    app_bar.restore_map_tab_state(restored, ui_state)
    assert restored.ui.map_tab.slic_superpixel_size.get() == 5
