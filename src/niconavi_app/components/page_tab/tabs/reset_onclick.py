import niconavi_app.niconavi.reset_run_all as rpo
from niconavi_app.niconavi.type import ComputationResult
from niconavi_app.stores import Stores, CIP_STATS_DEFAULT

# -----------------------------------------------------------------------
# onclick
# -----------------------------------------------------------------------
#
# Every cascade below ends at reset_onclick_cip_computation_button, so `stores`
# is threaded through all of them: the CPO panel text and the RGB comparison
# figure live in the UI state rather than in ComputationResult, and resetting
# the computation without clearing them leaves the Analysis tab reporting a run
# whose results are already gone.


def reset_onclick_load_data(r: ComputationResult, stores: Stores) -> ComputationResult:
    r = rpo.reset_load_data(r)
    r = rpo.reset_find_image_center(r)
    r = rpo.reset_make_retardation_color_chart(r)
    r = reset_onclick_center_button(r, stores)
    return r


def reset_onclick_center_button(
    r: ComputationResult, stores: Stores
) -> ComputationResult:
    r = rpo.reset_determine_rotation_angle(r)
    r = rpo.reset_make_raw_color_maps(r)
    r = rpo.reset_make_retardation_color_chart(r)
    r = rpo.reset_make_raw_R_maps(r)
    r = rpo.reset_estimate_tilt_image_result(r)
    r = reset_onclick_grain_boundary_button(r, stores)
    return r


def reset_onclick_recalculate_button(
    r: ComputationResult, stores: Stores
) -> ComputationResult:
    r = rpo.reset_make_retardation_color_chart(r)
    r = rpo.reset_make_raw_R_maps(r)
    r = rpo.reset_estimate_tilt_image_result(r)
    r = reset_onclick_grain_boundary_button(r, stores)
    return r


def reset_onclick_grain_boundary_button(
    r: ComputationResult, stores: Stores
) -> ComputationResult:
    r = rpo.reset_make_grain_boundary(r)
    r = reset_onclick_grain_analyze_button(r, stores)
    return r


def reset_onclick_grain_analyze_button(
    r: ComputationResult, stores: Stores
) -> ComputationResult:
    r = rpo.reset_analyze_grain_list(r)
    r = reset_onclick_classify_button(r, stores)
    return r


def reset_onclick_classify_button(
    r: ComputationResult, stores: Stores
) -> ComputationResult:
    r = rpo.reset_grain_segmentation(r)
    r = reset_onclick_cip_computation_button(r, stores)
    return r


def reset_onclick_cip_computation_button(
    r: ComputationResult, stores: Stores
) -> ComputationResult:
    r = rpo.reset_get_inclination(r)
    r = rpo.reset_analyze_grain_list_for_CIP(r)
    r = rpo.reset_make_CIP_map_info(r)
    reset_cip_ui_outputs(stores)
    return r


def reset_cip_ui_outputs(stores: Stores) -> None:
    """Drop what the last CPO run left on the Analysis tab.

    The Info panel hides itself once its text is back to the placeholder, and
    the color check buttons disappear with their figures.
    """
    stores.ui.analysis_tab.cip_stats_text.set(CIP_STATS_DEFAULT)
    stores.ui.analysis_tab.cip_regression_before_figure.set(None)
    stores.ui.analysis_tab.cip_regression_after_figure.set(None)
