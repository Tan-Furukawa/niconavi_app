from stores import Stores
from components.selector.tools import (
    make_reactive_text,
    exist_in_raw_maps,
)
from components.selector.movie import (
    make_movie_selection_button_visible_state,
)
from components.selector.grain import (
    make_grain_classification_button_visible_state,
)
from components.selector.filter import (
    make_filter_button_visible_state,
)
from components.selector.cip import (
    make_cip_button_visible_state,
)
from components.selector.spo import make_spo_button_visible_state


import flet as ft


class ImageSelector(ft.Container):
    def __init__(self, stores: Stores, page: ft.Page):
        super().__init__()

        self.expand = True
        self.width = 200
        self.margin = 0
        self.padding = 10

        # mask = make_always_visible_state(stores)

        mv_xpl, mv_full_wave, mv_tilt0, mv_tilt45, mv_horiz45 = (
        # mv_xpl, mv_full_wave, mv_tilt0 =(
            make_movie_selection_button_visible_state(stores)
        )

        (
            gr_degree_0,
            gr_degree_22_5,
            gr_degree_45,
            gr_degree_67_5,
            gr_R_color,
            gr_min_retardation_color,
            gr_retardation,
            gr_extinction_angle,
            gr_grain_map,
            # gr_grain_map_with_boundary,
            gr_p45_R_color_map,
            gr_m45_R_color_map,
            gr_p45_R_map,
            gr_m45_R_map,
            gr_azimuth,
            gr_tilt0,
            gr_tilt45,
            gr_horiz0,
            gr_horiz45,
            # gr_quality,
            gr_mask,
            # gr_delta_R_tilt_0,
            # gr_delta_R_tilt_45,
        ) = make_grain_classification_button_visible_state(stores, 2)

        # (
        #     fil_classification,
        #     fil_retardation,
        #     fil_max_R_red,
        #     fil_max_R_green,
        #     fil_max_R_blue,
        #     fil_max_retardation_color,
        #     fil_eccentricity,
        #     fil_max_R_70p,
        #     fil_size,
        #     fil_ex_angle,
        #     fil_azimuth,
        #     fil_sd_azimuth,
        #     fil_sd_ex_angle,
        #     fil_max_R_90p,
        #     fil_max_R_80p,
        #     fil_quality,
        #     fil_spo,
        #     fil_index,
        #     # fil_R_color_raw,
        # ) = make_filter_button_visible_state(stores, 3)

        (
            cip_extinction_angle,
            cip_azimuth,
            cip_inclination,
            cip_azimuth360,
            # cip_seg_extinction_angle,
            # cip_seg_azimuth,
            # cip_seg_inclination,
            # cip_seg_azimuth360,
            cip_coi90_map,
            cip_coi180_map,
            cip_coi360_map,
            # cip_coi90_grain,
            # cip_coi180_grain,
            # cip_coi360_grain,
            cip_polar90,
            cip_polar180,
            cip_polar360,
        ) = make_cip_button_visible_state(stores, 4)

        spo_spo, spo_ellipse, spo_major_axis = (
            make_spo_button_visible_state(stores, 4)
        )

        self.content = ft.Column(
            [
                # tab
                # mask,
                mv_xpl,
                mv_full_wave,
                mv_tilt0,
                mv_tilt45,
                # mv_horiz0,
                mv_horiz45,
                # tab
                gr_mask,
                make_reactive_text(stores, "XRL", 2),
                gr_R_color,
                gr_retardation,
                # gr_quality,
                make_reactive_text(stores, "grain map", 2),
                gr_grain_map,
                # gr_grain_map_with_boundary,
                make_reactive_text(stores, "extinction", 2),
                gr_min_retardation_color,
                gr_extinction_angle,
                gr_azimuth,
                make_reactive_text(stores, "raw image", 2),
                gr_degree_0,
                gr_degree_22_5,
                gr_degree_45,
                gr_degree_67_5,
                make_reactive_text(
                    stores,
                    "xpl + λ",
                    2,
                    lambda: (
                        stores.computation_result.tilt_image_info.tilt_image0.get()
                        is not None
                    )
                    # and (
                    #     stores.computation_result.tilt_image_info.tilt_image45.get()
                    #     is not None
                    # )
                    ,
                    [
                        stores.computation_result.tilt_image_info.tilt_image0,
                        # stores.computation_result.tilt_image_info.tilt_image45,
                    ],
                ),
                gr_horiz0,
                gr_tilt0,
                # gr_delta_R_tilt_0,
                gr_horiz45,
                gr_tilt45,
                # gr_delta_R_tilt_45,
                make_reactive_text(
                    stores,
                    "XRL with retardation plate",
                    2,
                    lambda: exist_in_raw_maps(stores, "p45_R_color_map"),
                ),
                gr_p45_R_color_map,
                gr_m45_R_color_map,
                gr_p45_R_map,
                gr_m45_R_map,
                # make_reactive_text(stores, "classification", 3),
                # fil_classification,
                # fil_index,
                # make_reactive_text(stores, "XPL", 3),
                # fil_max_retardation_color,
                # fil_max_R_red,
                # fil_max_R_green,
                # fil_max_R_blue,
                # fil_retardation,
                # fil_quality,
                # fil_max_R_70p,
                # fil_max_R_80p,
                # fil_max_R_90p,
                # make_reactive_text(stores, "parameters", 3),
                # fil_eccentricity,
                # fil_size,
                # fil_ex_angle,
                # fil_azimuth,
                # fil_sd_ex_angle,
                # fil_sd_azimuth,
                # fil_spo,
                # SPO
                spo_spo,
                # spo_rose_cpo,
                spo_ellipse,
                spo_major_axis,
                make_reactive_text(
                    stores,
                    "map",
                    4,
                    lambda: stores.ui.analysis_tab.plot_option.get() == "CPO",
                    [stores.ui.analysis_tab.plot_option],
                ),
                # CIP
                cip_extinction_angle,
                cip_azimuth,
                cip_azimuth360,
                cip_inclination,
                #! grain系は非表示にする
                # make_reactive_text(
                #     stores,
                #     "grain",
                #     4,
                #     lambda: stores.ui.analysis_tab.plot_option.get() == "CPO",
                #     [stores.ui.analysis_tab.plot_option],
                # ),
                # cip_seg_extinction_angle,
                # cip_seg_azimuth,
                # cip_seg_azimuth360,
                # cip_seg_inclination,
                make_reactive_text(
                    stores,
                    "map coi",
                    4,
                    lambda: stores.ui.analysis_tab.plot_option.get() == "CPO",
                    [stores.ui.analysis_tab.plot_option],
                ),
                cip_coi90_map,
                cip_coi180_map,
                cip_coi360_map,
                #! grain系は非表示にする
                # make_reactive_text(
                #     stores,
                #     "grain coi",
                #     4,
                #     lambda: stores.ui.analysis_tab.plot_option.get() == "CPO",
                #     [stores.ui.analysis_tab.plot_option],
                # ),
                # cip_coi90_grain,
                # cip_coi180_grain,
                # cip_coi360_grain,
                make_reactive_text(
                    stores,
                    "polar plot",
                    4,
                    lambda: stores.ui.analysis_tab.plot_option.get() == "CPO",
                    [stores.ui.analysis_tab.plot_option],
                ),
                cip_polar90,
                cip_polar180,
                cip_polar360,
            ],
            scroll=ft.ScrollMode.ADAPTIVE,
        )
