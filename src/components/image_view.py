from reactive_state import ReactiveMatplotlibChart, ReactiveColumn
from stores import Stores
from state import ReactiveState
import flet as ft
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure

from components.labeling_app.label_controls import LabelSelectionPane
from components.labeling_app.labeling_controller import LabelingController
from components.labeling_app.labeling_left_view import create_labeling_left_container
from components.view.analysis_view import at_analysis_tab
from components.view.movie_view import at_movie_tab
from components.view.center_view import at_center_tab
from components.view.grain_view import at_grain_tab
from components.view.filter_view import at_filter_tab
from components.view.spatial_units import get_pixel_to_micrometer_scale
from tools.no_image import get_no_image


def which_tab_opened(stores: Stores) -> Figure:
    plt.close("all")  # 描画する前に、メモリ上のプロットたちを開放させる

    pixel_scale_um = get_pixel_to_micrometer_scale(stores)
    if pixel_scale_um is None:
        stores.ui.image_viewer.pixel_size_um = None  # type: ignore[attr-defined]
        stores.ui.image_viewer.display_unit = "px"  # type: ignore[attr-defined]
    else:
        stores.ui.image_viewer.pixel_size_um = pixel_scale_um  # type: ignore[attr-defined]
        stores.ui.image_viewer.display_unit = "μm"  # type: ignore[attr-defined]

    match stores.ui.selected_index.get():

        case 0:
            print("-----event at movie tab-----")
            return at_movie_tab(stores)
        case 1:
            print("-----event at center tab-----")
            return at_center_tab(stores)
        case 2:
            print("-----event at grain tab-----")
            return at_grain_tab(stores)
        # case 3:
        #     print("-----event at merge tab-----")
        #     return at_filter_tab(stores, stores.ui.selected_button_at_merge_tab)
        case 3:
            print("-----event at filter tab-----")
            return at_filter_tab(stores, stores.ui.selected_button_at_filter_tab)
        case 4:
            print("-----event at analysis tab-----")
            return at_analysis_tab(stores)
        case _:
            print("other tab open")
            return get_no_image()


class ImageView(ft.Container):
    def __init__(self, stores: Stores, page: ft.Page):
        super().__init__()
        self.expand = True

        at_change_stores_list = [
            stores.ui.selected_index,
            stores.ui.force_update_image_view,
            # stores.ui.selected_button_at_common_image_view,
            stores.ui.selected_button_at_movie_tab,
            stores.ui.selected_button_at_grain_tab,
            stores.ui.selected_button_at_filter_tab,
            stores.ui.selected_button_at_merge_tab,
            stores.ui.selected_button_at_analysis_tab,
            stores.ui.display_grain_boundary,
            stores.ui.analysis_tab.cip_bandwidth,
            stores.computation_result.first_image,
            stores.computation_result.rotation_img,
            stores.computation_result.center_int_x,
            stores.computation_result.center_int_y,
            stores.computation_result.raw_maps,
            stores.computation_result.grain_map,
            stores.computation_result.grain_map_with_boundary,
            stores.computation_result.grain_boundary,
            stores.computation_result.grain_classification_image,
            stores.computation_result.grain_segmented_maps,
            stores.computation_result.grain_classification_result,
            stores.ui.apply_mask,
            stores.ui.analysis_tab.plot_option,
            stores.ui.analysis_tab.computation_unit,
            stores.ui.analysis_tab.cip_contour,
            stores.ui.analysis_tab.cip_theme,
            stores.ui.analysis_tab.cip_display_points,
            stores.ui.analysis_tab.cip_points_noise_size_percent,
            stores.ui.analysis_tab.grain_rose_diagram_target,
            stores.ui.analysis_tab.grain_histogram_target,
            stores.ui.analysis_tab.scatter_target_x,
            stores.ui.analysis_tab.scatter_target_y,
            stores.ui.analysis_tab.histogram_alpha,
            stores.ui.analysis_tab.rose_alpha,
            stores.ui.analysis_tab.rose_flip,
            stores.computation_result.tilt_image_info.tilt_image0,
            stores.computation_result.tilt_image_info.tilt_image45,
            stores.ui.analysis_tab.scatter_regression_origin,
            stores.ui.analysis_tab.scatter_show_regression,
            stores.ui.analysis_tab.histogram_log_x,
            stores.ui.analysis_tab.scatter_log_x,
            stores.ui.analysis_tab.scatter_log_y,
            stores.ui.grain_tab.slider_contrast,
            stores.ui.grain_tab.slider_brightness,
            stores.ui.grain_tab.slider_median_kernel,
            stores.ui.grain_tab.brightness_correction,
            stores.computation_result.plot_parameters.histogram_bins,
            stores.computation_result.plot_parameters.rose_diagram_bins,
            stores.computation_result.plot_parameters.rose_diagram180_bins,
            stores.computation_result.plot_parameters.rose_diagram90_bins,
            stores.ui.one_pixel,
        ]

        image = ReactiveState(lambda: which_tab_opened(stores), at_change_stores_list)

        controller = LabelingController(stores=stores)

        label_selection = LabelSelectionPane(
            on_add_label=controller.handle_label_added,
            on_remove_label=controller.handle_label_removed,
            on_select_label=controller.handle_label_selected,
            on_color_changed=controller.handle_label_color_changed,
        )

        controller.attach_label_selection(label_selection)

        left_container = create_labeling_left_container(
            stores=stores, controller=controller, page=page
        )

        self.content = ft.Container(
            content=ReactiveColumn(
                [
                    ReactiveColumn(
                        [
                            ft.Container(
                                content=ReactiveMatplotlibChart(image),
                                height=700,
                            )
                        ],
                        visible=ReactiveState(
                            lambda: stores.ui.selected_index.get() != 3,
                            [stores.ui.selected_index],
                        ),
                    ),
                    ReactiveColumn(
                        [left_container],
                        visible=ReactiveState(
                            lambda: stores.ui.selected_index.get() == 3,
                            [stores.ui.selected_index],
                        ),
                    ),
                ]
            ),
            # alignment=ft.alignment.top_left,
            alignment=ft.alignment.center,
        )
