import niconavi.reset_run_all as rpo
from niconavi.type import ComputationResult

# -----------------------------------------------------------------------
# onclick
# -----------------------------------------------------------------------


def reset_onclick_load_data(r: ComputationResult) -> ComputationResult:
    r = rpo.reset_load_data(r)
    r = rpo.reset_find_image_center(r)
    r = rpo.reset_make_retardation_color_chart(r)
    r = reset_onclick_center_button(r)
    return r


def reset_onclick_center_button(r: ComputationResult) -> ComputationResult:
    r = rpo.reset_determine_rotation_angle(r)
    r = rpo.reset_make_raw_color_maps(r)
    r = rpo.reset_make_retardation_color_chart(r)
    r = rpo.reset_make_raw_R_maps(r)
    r = rpo.reset_estimate_tilt_image_result(r)
    r = reset_onclick_grain_boundary_button(r)
    return r

def reset_onclick_recalculate_button(r: ComputationResult) -> ComputationResult:
    r = rpo.reset_make_retardation_color_chart(r)
    r = rpo.reset_make_raw_R_maps(r)
    r = rpo.reset_estimate_tilt_image_result(r)
    r = reset_onclick_grain_boundary_button(r)
    return r


def reset_onclick_grain_boundary_button(r: ComputationResult) -> ComputationResult:
    r = rpo.reset_make_grain_boundary(r)
    r = reset_onclick_grain_analyze_button(r)
    return r


def reset_onclick_grain_analyze_button(r: ComputationResult) -> ComputationResult:
    r = rpo.reset_analyze_grain_list(r)
    r = reset_onclick_classify_button(r)
    return r


def reset_onclick_classify_button(r: ComputationResult) -> ComputationResult:
    r = rpo.reset_grain_segmentation(r)
    r = reset_onclick_cip_computation_button(r)
    return r


def reset_onclick_cip_computation_button(r: ComputationResult) -> ComputationResult:
    r = rpo.reset_get_inclination(r)
    r = rpo.reset_analyze_grain_list_for_CIP(r)
    r = rpo.reset_make_CIP_map_info(r)
    return r
