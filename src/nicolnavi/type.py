# %%
from typing import TypedDict, Literal, Optional, TypeAlias, TypeGuard
from matplotlib.pyplot import Figure, Axes
from niconavi.tools.type import (
    D2BoolArray,
    D2FloatArray,
    D2IntArray,
    D1FloatArray,
    D1IntArray,
)
import numpy as np
from niconavi.image.type import Color, RGBPicture, MonoColorPicture, D1RGB_Array

RawMapsAcceptedLiteral: TypeAlias = Literal[
    "degree_0",
    "degree_22_5",
    "degree_45",
    "degree_67_5",
    "extinction_color_map",
    "R_color_map",
    "R_color_map_raw",
    "R_color_map_display",
    "extinction_angle",
    "max_retardation_map",
    "p45_R_color_map",
    "m45_R_color_map",
    "p45_R_map",
    "m45_R_map",
    "azimuth",
    "inclination",
    "inclination_0_to_180",
    "azimuth360",
]


RawMapsNumLiteral: TypeAlias = Literal[
    "extinction_angle",
    "max_retardation_map",
    "p45_R_map",
    "m45_R_map",
    "azimuth",
]


class ColorChartInfo(TypedDict):
    what_is_h: Optional[str]
    what_is_w: Optional[str]
    h: D1FloatArray
    w: D1FloatArray
    color_chart: RGBPicture


class TheoriticalImage(TypedDict):
    max_R_color: RGBPicture
    color_0: RGBPicture
    color_22_5: RGBPicture
    color_45: RGBPicture
    color_67_5: RGBPicture
    R: D2FloatArray
    R_minus: D2FloatArray
    R_plus: D2FloatArray
    tilted_d0: RGBPicture
    tilted_d45: RGBPicture
    tilted_minus_d0: RGBPicture
    tilted_minus_d45: RGBPicture


class TiltDirectionEstimationResult(TypedDict):
    R_im0: D2FloatArray
    R_im0_tilt: D2FloatArray
    become_red_by_tilt: D2BoolArray
    R_im45: Optional[D2FloatArray]
    R_im45_tilt: Optional[D2FloatArray]


class RawMaps(TypedDict):  # make_AcceptedLiteral
    degree_0: RGBPicture
    degree_22_5: RGBPicture
    degree_45: RGBPicture
    degree_67_5: RGBPicture
    extinction_color_map: RGBPicture
    R_color_map: RGBPicture
    R_color_map_raw: RGBPicture
    R_color_map_display: RGBPicture
    extinction_angle: D2FloatArray
    extinction_angle_0: D2FloatArray
    extinction_angle_225: D2FloatArray
    extinction_angle_45: D2FloatArray
    extinction_angle_675: D2FloatArray
    cv_extinction_angle: D2FloatArray
    p45_R_color_map: Optional[RGBPicture]
    m45_R_color_map: Optional[RGBPicture]
    azimuth: Optional[D2FloatArray]
    max_retardation_map: Optional[D2FloatArray]
    p45_R_map: Optional[D2FloatArray]
    m45_R_map: Optional[D2FloatArray]
    theoritical_image: Optional[TheoriticalImage]
    inclination: Optional[D2FloatArray]
    inclination_0_to_180: Optional[D2FloatArray]
    azimuth360: Optional[D2FloatArray]
    # tilt_direction_estimation_result: Optional[TiltDirectionEstimationResult]


class CIPMapInfo(TypedDict):
    polar_info90: Optional[
        dict[str, tuple[D1FloatArray, D1FloatArray]]
    ]  # inclination, azimuth
    polar_info180: Optional[dict[str, tuple[D1FloatArray, D1FloatArray]]]
    polar_info360: Optional[dict[str, tuple[D1FloatArray, D1FloatArray]]]
    legend90: Optional[RGBPicture]
    legend180: Optional[RGBPicture]
    legend360: Optional[RGBPicture]
    COI90: Optional[RGBPicture]
    COI180: Optional[RGBPicture]
    COI360: Optional[RGBPicture]
    COI90_grain: Optional[RGBPicture]
    COI180_grain: Optional[RGBPicture]
    COI360_grain: Optional[RGBPicture]


GrainSegmentedMapsAcceptedLiteral = Literal[
    "extinction_color_map",
    "R_color_map",
    "extinction_angle",
    "sd_extinction_angle_map",
    "max_retardation_map",
    "H",
    "S",
    "V",
    "R_70_map",
    "R_80_map",
    "R_90_map",
    "size",
    "p45_R_color_map",
    "extinction_angle_quality",
    "m45_R_color_map",
    "p45_R_map",
    "m45_R_map",
    "azimuth",
    "inclination",
    "sd_azimuth",
    "eccentricity",
    "angle_deg",
    "major_axis_length",
    "minor_axis_length",
    "tilt0_plus_ratio_map",
    "tilt45_plus_ratio_map",
    "azimuth360",
    "len_in_mask_pixel",
]


class GrainSegmentedMaps(TypedDict):
    extinction_color_map: Optional[RGBPicture]
    R_color_map: Optional[RGBPicture]
    extinction_angle: Optional[D2FloatArray]
    sd_extinction_angle_map: Optional[D2FloatArray]
    max_retardation_map: Optional[D2FloatArray]
    H: Optional[D2FloatArray]
    S: Optional[D2FloatArray]
    V: Optional[D2FloatArray]
    R_70_map: Optional[D2FloatArray]
    R_80_map: Optional[D2FloatArray]
    R_90_map: Optional[D2FloatArray]
    size: Optional[D2FloatArray]
    extinction_angle_quality: Optional[D2FloatArray]
    p45_R_color_map: Optional[RGBPicture]
    m45_R_color_map: Optional[RGBPicture]
    p45_R_map: Optional[D2FloatArray]
    m45_R_map: Optional[D2FloatArray]
    azimuth: Optional[D2FloatArray]
    sd_azimuth: Optional[D2FloatArray]
    eccentricity: Optional[D2FloatArray]
    angle_deg: Optional[D2FloatArray]
    major_axis_length: Optional[D2FloatArray]
    minor_axis_length: Optional[D2FloatArray]
    tilt0_plus_ratio_map: Optional[D2FloatArray]
    tilt45_plus_ratio_map: Optional[D2FloatArray]
    inclination: Optional[D2FloatArray]
    azimuth360: Optional[D2FloatArray]
    len_in_mask_pixel: Optional[D2FloatArray]


class Grain(TypedDict):
    index: int
    centroid: tuple[float, float]
    inscribed_radius: Optional[float]
    equivalent_radius: Optional[float]
    exQuality: Optional[float]
    size: int
    perimeter: Optional[float]
    area_shape: D2BoolArray
    top_left_index: tuple[int, int]
    at_lim: bool
    mineral: Optional[str]
    original_shape: tuple[int, int]
    R_color: Optional[Color]
    extinction_color: Optional[Color]
    inclination: Optional[float]
    extinction_angle: Optional[float]  # clockwise positive
    sd_extinction_angle: Optional[float]  # clockwise positive
    p45_color: Optional[Color]
    m45_color: Optional[Color]
    R: Optional[float]
    max_retardation_estimated_for_inclination: Optional[float]
    min_retardation: Optional[float]
    pR: Optional[float]
    mR: Optional[float]
    R70: Optional[float]
    R80: Optional[float]
    R90: Optional[float]
    pR75: Optional[float]
    mR75: Optional[float]
    azimuth: Optional[float]  # clockwise negative
    sd_azimuth: Optional[float]  # clockwise negative
    H: Optional[float]
    V: Optional[float]
    S: Optional[float]
    eccentricity: Optional[float]
    angle_deg: Optional[float]
    major_axis_length: Optional[float]
    minor_axis_length: Optional[float]
    ellipse_center: Optional[tuple[int, int]]
    tilt0_plus_ratio: Optional[float]
    tilt45_plus_ratio: Optional[float]
    azimuth360: Optional[float]
    len_in_mask_pixel: Optional[int]


class GrainFeaturesForClassification(TypedDict):
    logA: float
    C: float
    solidity: float
    aspect_ratio: float
    Lab_a_med: float
    Lab_b_med: float
    HSV_S_med: float
    cx: float
    cy: float


class GrainDetectionParameters:
    def __init__(
        self,
        smallest_grain_size: int = 10,
        th_about_hessian_emphasis: float = 0.7,
        th_about_connect_skeleton_endpoints: float = 10,
        permit_inclusion: bool = True,
        # color_map_median_kernel_size: int = 3,
        # color_map_percentile: float = 50.0,
        # angle_map_median_kernel_size: int = 3,
        # angle_map_percentile: float = 50.0,
        # color_map_min_R: float = 0,
        # color_map_max_R: float = 9999,
        # color_rev_estimation: bool = False,  # Trueのとき、minR > R or R > max_Rをしらべることになる。
        # angle_map_min_R: float = 50,
        # angle_map_max_R: float = 530,
        # angle_rev_estimation: bool = False,  # Trueのとき、minR > R or R > max_Rをしらべることになる。
        # angle_method: Literal["azimuth", "extinction_angle"] = "extinction_angle",
        # eval_method: Literal["angle map", "color map", "both"] = "both",
    ) -> None:
        self.smallest_grain_size = smallest_grain_size
        self.th_about_connect_skeleton_endpoints = th_about_connect_skeleton_endpoints
        self.th_about_hessian_emphasis = th_about_hessian_emphasis
        self.permit_inclusion = permit_inclusion
        # self.color_map_median_kernel_size = color_map_median_kernel_size
        # self.color_map_percentile = color_map_percentile
        # self.angle_map_median_kernel_size = angle_map_median_kernel_size
        # self.angle_map_percentile = angle_map_percentile
        # # self.use_angle = use_angle
        # self.color_map_min_R = color_map_min_R
        # self.color_rev_estimation = color_rev_estimation
        # self.angle_rev_estimation = angle_rev_estimation
        # self.color_map_max_R = color_map_max_R
        # self.angle_map_min_R = angle_map_min_R
        # self.angle_map_max_R = angle_map_max_R
        # self.angle_method = angle_method
        # self.eval_method = eval_method


class QuartzWedgeNormalization:
    def __init__(
        self,
        no: float = 1.544,
        ne: float = 1.553,
        thin_section_thickness: Optional[float] = 0.03,
        quartz_wedge_path: Optional[str] = None,
        raw_data: Optional[D1RGB_Array] = None,
        calibrated_data: Optional[D1RGB_Array] = None,
        retardation: Optional[D1FloatArray] = None,
        # summary_display: Optional[tuple[Figure, Axes, Axes]] = None,
        # summary_rgb_display: Optional[tuple[Figure, Axes]] = None,
        R: Optional[float] = 1000.0,
        max_retardation_for_inclination_search: Optional[float] = None,
        max_retardation_for_1st_order_color_plate: float = 1700.0,  # more than 1700 (this value should not change)
        # convert_retardation_to_inclination: Optional[
        #     Callable[[float], float | None]
        # ] = None,
    ) -> None:
        self.quartz_wedge_path = quartz_wedge_path
        self.raw_data = raw_data
        self.calibrated_data = calibrated_data
        self.retardation = retardation
        # self.summary_display = summary_display
        # self.summary_rgb_display = summary_rgb_display
        self.R = R
        self.max_retardation_for_1st_order_color_plate = (
            max_retardation_for_1st_order_color_plate
        )
        self.max_retardation_for_inclination_search = (
            max_retardation_for_inclination_search
        )
        self.thin_section_thickness = thin_section_thickness
        self.no = no
        self.ne = ne
        # self.convert_retardation_to_inclination = convert_retardation_to_inclination


class GrainSelectedResult(TypedDict):
    color: str
    index: D1IntArray
    display: bool


class TiltImageResult(TypedDict):
    original_image: RGBPicture
    focused_tilted_image: RGBPicture
    focused_index: D2IntArray
    image_mask: D2BoolArray
    azimuth_thin_section: float
    # original_retardation: D2FloatArray
    # tilted_retardation: D2FloatArray


class TiltImageInfo:
    def __init__(
        self,
        estimate_inclination_by: Literal["thickness", "max R"] = "max R",
        theta_thin_section: float = 10.0,
        frame_num: int = 20,
        image0_path: Optional[str] = None,
        tilt_image0_path: Optional[str] = None,
        image0_raw: Optional[list[RGBPicture]] = None,
        tilt_image0_raw: Optional[list[RGBPicture]] = None,
        tilt_image0: Optional[TiltImageResult] = None,
        image45_raw: Optional[list[RGBPicture]] = None,
        image45_path: Optional[str] = None,
        tilt_image45_path: Optional[str] = None,
        tilt_image45_raw: Optional[list[RGBPicture]] = None,
        tilt_image45: Optional[TiltImageResult] = None,
    ) -> None:
        self.estimate_inclination_by = estimate_inclination_by
        self.theta_thin_section = theta_thin_section
        self.frame_num = frame_num
        self.image0_path = image0_path
        self.tilt_image0_path = tilt_image0_path
        self.image0_raw = image0_raw
        self.tilt_image0_raw = tilt_image0_raw
        self.tilt_image0 = tilt_image0
        self.image45_raw = image45_raw
        self.image45_path = image45_path
        self.tilt_image45_path = tilt_image45_path
        self.tilt_image45_raw = tilt_image45_raw
        self.tilt_image45 = tilt_image45


class PlotParameters:
    def __init__(
        self,
        rose_diagram90_bins: int = 30,
        rose_diagram180_bins: int = 30,
        histogram_bins: int = 30,
        rose_diagram_bins: int = 30,
    ) -> None:
        self.rose_diagram90_bins = rose_diagram90_bins
        self.rose_diagram180_bins = rose_diagram180_bins
        self.histogram_bins = histogram_bins
        self.rose_diagram_bins = rose_diagram_bins


class OpticalParameters:
    def __init__(
        self,
        no: float = 1.544,
        ne: float = 1.553,
        tilt_deg: float = 4.0,
        max_R: Optional[float] = None,
        thickness: float = 0.03,
        xpl_retardation_color_chart_used: Optional[RGBPicture] = None,
    ) -> None:
        self.no = no
        self.ne = ne
        self.tilt_deg = tilt_deg
        self.max_R = max_R
        self.thickness = thickness
        self.xpl_retardation_color_chart_used = xpl_retardation_color_chart_used


class ColorChart:
    def __init__(
        self,
        xpl_alpha: Optional[float] = None,
        pol_lambda_alpha: Optional[float] = None,
        inc_alpha: Optional[float] = None,
        xpl_max_retardation: float = 300,
        pol_lambda_max_retardation: Optional[float] = 1500.0,
        inc_max_retardation: Optional[float] = 300 + 530,
        xpl_retardation_color_chart: Optional[D1RGB_Array] = None,
        pol_lambda_retardation_color_chart: Optional[D1RGB_Array] = None,
        inc_retardation_color_chart: Optional[D1RGB_Array] = None,
        xpl_R_array: Optional[D1FloatArray] = None,
        pol_lambda_R_array: Optional[D1FloatArray] = None,
        inc_R_array: Optional[D1FloatArray] = None,
    ):
        self.xpl_alpha = xpl_alpha
        self.pol_lambda_alpha = pol_lambda_alpha
        self.inc_alpha = inc_alpha
        self.xpl_max_retardation = xpl_max_retardation
        self.pol_lambda_max_retardation = pol_lambda_max_retardation
        self.inc_max_retardation = inc_max_retardation
        self.xpl_retardation_color_chart = xpl_retardation_color_chart
        self.pol_lambda_retardation_color_chart = pol_lambda_retardation_color_chart
        self.inc_retardation_color_chart = inc_retardation_color_chart
        self.xpl_R_array = xpl_R_array
        self.pol_lambda_R_array = pol_lambda_R_array
        self.inc_R_array = inc_R_array


class FirstVideoImage(TypedDict):
    xpl: Optional[RGBPicture]
    full_wave: Optional[RGBPicture]
    image0: Optional[RGBPicture]
    image0_tilt: Optional[RGBPicture]
    image45: Optional[RGBPicture]
    image45_tilt: Optional[RGBPicture]


class ComputationResult:
    def __init__(
        self,
        video_path: Optional[str] = None,
        original_resolution: Optional[tuple[int, int]] = None,
        original_reta_resolution: Optional[tuple[int, int]] = None,
        reta_video_path: Optional[str] = None,
        mask: Optional[D2BoolArray] = None,
        resolution_width: int = 1000,
        full_wave_plate_nm: float = 530.0,
        circ_threshold: float = 0.5,
        angle_between_x_and_thin_section_axis_at_tilt: float = 45,
        use_raw_in_grain_boundary_detection: bool = False,
        quartz_wedge_normalization: QuartzWedgeNormalization = QuartzWedgeNormalization(),
        resolution_height: Optional[int] = None,
        frame_number: int = 100,
        pics: Optional[list[RGBPicture]] = None,
        pics_rotated: Optional[list[RGBPicture]] = None,
        reta_pics: Optional[list[RGBPicture]] = None,
        reta_pics_rotated: Optional[list[RGBPicture]] = None,
        angles: Optional[D1FloatArray] = None,
        reta_angles: Optional[D1FloatArray] = None,
        # center_float: Optional[tuple[float, float]] = None,
        center_int_x: Optional[int] = None,
        center_int_y: Optional[int] = None,
        rotation_img: Optional[MonoColorPicture] = None,
        # rotation_img_with_mark: Optional[Figure] = None,
        image_rotation_direction: Optional[
            Literal["clockwise", "counterclockwise"]
        ] = None,
        reta_image_rotation_direction: Optional[
            Literal["clockwise", "counterclockwise"]
        ] = None,
        grain_list: Optional[list[Grain]] = None,
        grain_features_list_for_classification: Optional[
            list[GrainFeaturesForClassification]
        ] = None,
        grain_detection_parameters: GrainDetectionParameters = GrainDetectionParameters(),
        grain_map: Optional[D2IntArray] = None,
        grain_map_original: Optional[D2IntArray] = None,
        grain_map_with_boundary: Optional[D2IntArray] = None,
        # use_angle_for_grain_boundary_estimation: Literal[
        #     "azimuth", "extinction_angle"
        # ] = "extinction_angle",
        grain_boundary: Optional[D2BoolArray] = None,
        grain_boundary_original: Optional[D2BoolArray] = None,
        raw_maps: Optional[RawMaps] = None,
        first_image: FirstVideoImage = FirstVideoImage(
            xpl=None,
            full_wave=None,
            image0=None,
            image0_tilt=None,
            image45=None,
            image45_tilt=None,
        ),
        cip_map_info: CIPMapInfo = CIPMapInfo(
            polar_info180=None,
            polar_info360=None,
            polar_info90=None,
            COI180_grain=None,
            COI360_grain=None,
            COI90_grain=None,
            COI180=None,
            COI360=None,
            COI90=None,
            legend180=None,
            legend360=None,
            legend90=None,
        ),
        grain_segmented_maps: Optional[GrainSegmentedMaps] = None,
        grain_marge_code: Optional[str] = None,
        grain_classification_result: Optional[dict[str, GrainSelectedResult]] = None,
        grain_classification_code: Optional[str] = None,
        grain_classification_image: Optional[RGBPicture] = None,
        grain_classification_legend: Optional[dict[str, Color]] = None,
        plot_parameters: PlotParameters = PlotParameters(),
        optical_parameters: OpticalParameters = OpticalParameters(),
        tilt_image_info: TiltImageInfo = TiltImageInfo(),
        color_chart: ColorChart = ColorChart(),
    ):
        self.video_path = video_path
        self.reta_video_path = reta_video_path
        self.mask = mask
        self.original_resolution = original_resolution
        self.original_reta_resolution = original_reta_resolution
        self.quartz_wedge_normalization = quartz_wedge_normalization
        self.resolution_width = resolution_width
        self.circ_threshold = circ_threshold
        self.full_wave_plate_nm = full_wave_plate_nm
        self.angle_between_x_and_thin_section_axis_at_tilt = (
            angle_between_x_and_thin_section_axis_at_tilt
        )
        self.use_raw_in_grain_boundary_detection = use_raw_in_grain_boundary_detection
        self.resolution_height = resolution_height
        self.frame_number = frame_number
        self.pics = pics
        self.pics_rotated = pics_rotated
        self.reta_pics = reta_pics
        self.reta_pics_rotated = reta_pics_rotated
        self.angles = angles
        self.reta_angles = reta_angles
        self.center_int_x = center_int_x
        self.center_int_y = center_int_y
        self.rotation_img = rotation_img
        self.reta_image_rotation_direction = reta_image_rotation_direction
        self.image_rotation_direction = image_rotation_direction
        self.grain_list = grain_list
        self.grain_features_list_for_classification = (
            grain_features_list_for_classification
        )
        self.grain_detection_parameters = grain_detection_parameters
        self.grain_map = grain_map
        self.grain_map_original = grain_map_original
        self.grain_map_with_boundary = grain_map_with_boundary
        # self.use_angle_for_grain_boundary_estimation = (
        #     use_angle_for_grain_boundary_estimation
        # )
        self.grain_boundary = grain_boundary
        self.grain_boundary_original = grain_boundary_original
        self.raw_maps = raw_maps
        self.first_image = first_image
        self.grain_segmented_maps = grain_segmented_maps
        self.cip_map_info = cip_map_info
        self.grain_classification_code = grain_classification_code
        self.grain_marge_code = grain_marge_code
        self.grain_classification_result = grain_classification_result
        self.grain_classification_image = grain_classification_image
        self.grain_classification_legend = grain_classification_legend
        self.plot_parameters = plot_parameters
        self.optical_parameters = optical_parameters
        self.tilt_image_info = tilt_image_info
        self.color_chart = color_chart


def create_Grain_type(
    index: int,
    centroid: tuple[float, float],
    inscribed_radius: Optional[float],
    equivalent_radius: Optional[float],
    size: int,
    perimeter: Optional[float],
    area_shape: D2BoolArray,
    top_left_index: tuple[int, int],
    at_lim: bool,
    original_shape: tuple[int, int],
    exQuality: Optional[float] = None,
    mineral: Optional[str] = None,
    R_color: Optional[Color] = None,
    extinction_color: Optional[Color] = None,
    extinction_angle: Optional[float] = None,
    sd_extinction_angle: Optional[float] = None,
    p45_color: Optional[Color] = None,
    m45_color: Optional[Color] = None,
    R: Optional[float] = None,
    R70: Optional[float] = None,
    R80: Optional[float] = None,
    R90: Optional[float] = None,
    max_retardation_estimated_for_inclination: Optional[float] = None,
    min_retardation: Optional[float] = None,
    pR: Optional[float] = None,
    mR: Optional[float] = None,
    inclination: Optional[float] = None,
    pR75: Optional[float] = None,
    mR75: Optional[float] = None,
    azimuth: Optional[float] = None,
    # azimuth360: Optional[float] = None,
    sd_azimuth: Optional[float] = None,
    H: Optional[float] = None,
    V: Optional[float] = None,
    S: Optional[float] = None,
    eccentricity: Optional[float] = None,
    angle_deg: Optional[float] = None,
    major_axis_length: Optional[float] = None,
    minor_axis_length: Optional[float] = None,
    ellipse_center: Optional[tuple[int, int]] = None,
    tilt0_plus_ratio: Optional[float] = None,
    tilt45_plus_ratio: Optional[float] = None,
    azimuth360: Optional[float] = None,
    len_in_mask_pixel: Optional[int] = None,
) -> Grain:
    return Grain(
        index=index,
        centroid=centroid,
        inscribed_radius=inscribed_radius,
        equivalent_radius=equivalent_radius,
        size=size,
        perimeter=perimeter,
        exQuality=exQuality,
        area_shape=area_shape,
        top_left_index=top_left_index,
        at_lim=at_lim,
        original_shape=original_shape,
        mineral=mineral,
        R_color=R_color,
        extinction_color=extinction_color,
        extinction_angle=extinction_angle,
        sd_extinction_angle=sd_extinction_angle,
        p45_color=p45_color,
        m45_color=m45_color,
        R=R,
        R70=R70,
        R80=R80,
        R90=R90,
        max_retardation_estimated_for_inclination=max_retardation_estimated_for_inclination,
        min_retardation=min_retardation,
        pR=pR,
        mR=mR,
        inclination=inclination,
        pR75=pR75,
        mR75=mR75,
        azimuth=azimuth,
        sd_azimuth=sd_azimuth,
        H=H,
        V=V,
        S=S,
        eccentricity=eccentricity,
        angle_deg=angle_deg,
        major_axis_length=major_axis_length,
        minor_axis_length=minor_axis_length,
        ellipse_center=ellipse_center,
        tilt0_plus_ratio=tilt0_plus_ratio,
        tilt45_plus_ratio=tilt45_plus_ratio,
        azimuth360=azimuth360,
        len_in_mask_pixel=len_in_mask_pixel,
    )


GrainAcceptedLiteral = Literal[
    "index",
    "centroid",
    "inscribed_radius",
    "equivalent_radius",
    "size",
    "perimeter",
    "exQuality",
    "area_shape",
    "top_left_index",
    "at_lim",
    "original_shape",
    "R_color",
    "extinction_color",
    "inclination",
    "extinction_angle",
    "sd_extinction_angle",
    "p45_color",
    "m45_color",
    "R",
    "max_retardation_estimated_for_inclination",
    "min_retardation",
    "pR",
    "mR",
    "R70",
    "R80",
    "R90",
    "pR75",
    "mR75",
    "azimuth",
    "sd_azimuth",
    "H",
    "V",
    "S",
    "eccentricity",
    "angle_deg",
    "major_axis_length",
    "minor_axis_length",
    "ellipse_center" "tilt0_plus_ratio",
    "tilt45_plus_ratio",
    "azimuth360",
    "len_in_mask_pixel",
]


GrainNumLiteral = Literal[
    "index",
    "size",
    "perimeter",
    "inscribed_radius",
    "equivalent_radius",
    "inclination",
    "extinction_angle",
    "sd_extinction_angle",
    "R",
    "max_retardation_estimated_for_inclination",
    "min_retardation",
    "pR",
    "mR",
    "R70",
    "R80",
    "R90",
    "pR75",
    "mR75",
    "azimuth",
    "sd_azimuth",
    "H",
    "V",
    "S",
    "eccentricity",
    "angle_deg",
    "major_axis_length",
    "minor_axis_length",
    "tilt0_plus_ratio",
    "tilt45_plus_ratio",
    "azimuth360",
    "len_in_mask_pixel",
]

RawMapsNumList = list(RawMapsNumLiteral.__dict__["__args__"])
GrainNumList = list(GrainNumLiteral.__dict__["__args__"])
GrainAcceptedList = list(GrainAcceptedLiteral.__dict__["__args__"])
RawMapsAcceptedList = list(RawMapsAcceptedLiteral.__dict__["__args__"])


def is_GrainNumLiteral(name: str) -> TypeGuard[GrainNumLiteral]:
    return name in GrainNumList


def is_RawMapsNumLiteral(name: str) -> TypeGuard[RawMapsNumLiteral]:
    return name in RawMapsNumList


def is_GrainAcceptedLiteral(name: str) -> TypeGuard[GrainAcceptedLiteral]:
    return name in GrainAcceptedList


def is_RawMapsAcceptedLiteral(name: str) -> TypeGuard[RawMapsAcceptedLiteral]:
    return name in RawMapsAcceptedList


# %%
if __name__ == "__main__":
    x = TiltImageInfo()
    x.__dict__
