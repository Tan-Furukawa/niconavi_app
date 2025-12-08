from __future__ import annotations

from typing import Optional, Iterable, Sequence, Dict
from copy import copy, deepcopy

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from stores import Stores
from niconavi.type import ComputationResult, Grain
from match_used_name import (
    GRAIN_MEASUREMENT_DIMENSION_MAP,
    GrainMeasurementDimension,
    get_grain_measurement_dimension,
)

def get_pixel_to_micrometer_scale(stores: Stores) -> Optional[float]:
    """
    Returns the micrometer length that corresponds to a single pixel.
    When no valid user input is provided, None is returned so plotters
    can keep displaying pixel units.
    """
    value = stores.ui.one_pixel.get()
    if value is None:
        return None

    try:
        scale = float(value)
    except (TypeError, ValueError):
        return None

    if scale <= 0:
        return None

    return scale


def _format_micrometer_tick(value: float, scale: float) -> str:
    micrometer = value * scale
    if abs(micrometer) >= 1000:
        text = f"{micrometer:,.0f}"
    elif abs(micrometer) >= 10:
        text = f"{micrometer:.1f}"
    else:
        text = f"{micrometer:.2f}"

    if text.endswith(".0"):
        return text[:-2]
    return text


def apply_micrometer_axis(ax: Axes, stores: Stores) -> None:
    """
    Re-labels the axes in micrometers if the pixel size is known.
    """
    scale = get_pixel_to_micrometer_scale(stores)
    if scale is None:
        return

    formatter = FuncFormatter(lambda value, pos: _format_micrometer_tick(value, scale))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("μm")
    ax.set_ylabel("μm")


def _collect_grain_values(grains: list[Grain], key: str) -> list[float]:
    values: list[float] = []
    for grain in grains:
        value = grain.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _base_unit_for_dimension(
    dimension: GrainMeasurementDimension, scale_available: bool
) -> str:
    if dimension == "length":
        return "μm" if scale_available else "px"
    return "μm²" if scale_available else "px²"


def _compute_multiplier_and_unit(
    values: Sequence[float],
    dimension: GrainMeasurementDimension,
    scale_um: Optional[float],
) -> tuple[float, str]:
    exponent = 1 if dimension == "length" else 2
    if scale_um is None:
        return 1.0, _base_unit_for_dimension(dimension, False)

    base_multiplier = scale_um**exponent
    base_unit = _base_unit_for_dimension(dimension, True)

    return base_multiplier, base_unit


def convert_grain_units_for_targets(
    stores: Stores,
    params: ComputationResult,
    target_keys: Iterable[str],
) -> tuple[ComputationResult, Dict[str, str]]:
    scale = get_pixel_to_micrometer_scale(stores)
    dimension_map: Dict[str, GrainMeasurementDimension] = {}
    for key in set(target_keys):
        dimension = get_grain_measurement_dimension(key)
        if dimension is not None:
            dimension_map[key] = dimension

    if not dimension_map:
        return params, {}

    if params.grain_list is None:
        unit_map = {}
        for key, dimension in dimension_map.items():
            _, unit = _compute_multiplier_and_unit([], dimension, scale)
            unit_map[key] = unit
        return params, unit_map

    px_values = {
        key: _collect_grain_values(params.grain_list, key) for key in dimension_map
    }

    multiplier_map: Dict[str, float] = {}
    unit_map: Dict[str, str] = {}
    conversion_needed = False

    for key, dimension in dimension_map.items():
        multiplier, unit = _compute_multiplier_and_unit(
            px_values[key], dimension, scale
        )
        multiplier_map[key] = multiplier
        unit_map[key] = unit
        if multiplier != 1.0 and len(px_values[key]) > 0:
            conversion_needed = True

    if not conversion_needed:
        return params, unit_map

    converted = copy(params)
    converted.grain_list = deepcopy(params.grain_list)
    assert converted.grain_list is not None

    for grain in converted.grain_list:
        for key, multiplier in multiplier_map.items():
            if multiplier == 1.0:
                continue
            value = grain.get(key)
            if value is None:
                continue
            grain[key] = float(value) * multiplier

    return converted, unit_map


def get_grain_unit_label(stores: Stores, key: str) -> Optional[str]:
    dimension = get_grain_measurement_dimension(key)
    if dimension is None:
        return None

    grain_state = stores.computation_result.grain_list
    grain_list = grain_state.get()
    values = _collect_grain_values(grain_list, key) if grain_list else []
    _, unit = _compute_multiplier_and_unit(
        values, dimension, get_pixel_to_micrometer_scale(stores)
    )
    return unit


def format_quantity_label(
    base_label: Optional[str], unit_label: Optional[str]
) -> Optional[str]:
    if base_label is None:
        return None
    if unit_label:
        return f"{base_label} ({unit_label})"
    return base_label
