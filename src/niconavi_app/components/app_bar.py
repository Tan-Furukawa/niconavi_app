import flet as ft
from typing import Callable, Any
from pathlib import Path
import numpy as np
from niconavi_app.stores import (
    ComputationResult,
    Stores,
    save_in_ComputationResultState,
)
from niconavi_app.niconavi.tools.str_parser import parse_larger_than_0
# from niconavi_app.components.selector.always import make_always_visible_state
from logging import getLogger, Logger
from niconavi_app.reactive_state import ReactiveProgressRing, ReactiveText, ReactiveCheckbox
from niconavi_app.state import ReactiveState, State
from niconavi_app.save import (
    collect_project_ui_state,
    save_file_result,
    save_image_as_pdf,
    save_grain_information,
    export_image_as_pdf_bytes,
    export_project_bytes,
    export_grain_information_zip_bytes,
)
from flet import (
    TextAlign,
    Container,
    Page,
    Text,
    AppBar,
    PopupMenuButton,
    PopupMenuItem,
    margin,
)
from flet import (
    ElevatedButton,
    FilePicker,
    FilePickerResultEvent,
    FilePickerFileType,
    Page,
)
from niconavi_app.components.page_tab.tabs.movie_tab import (
    make_simple_file_handler,
    make_upload_file_handler,
)

from niconavi_app.components.labeling_app.labeling_controller import LabelingController
from niconavi_app.components.common_component import (
    make_reactive_float_text_filed,
    CustomText,
    CustomReactiveCheckbox,
    make_confirmation_dialog,
)
from niconavi_app.components.context_help import ContextHelpController
from niconavi_app.components.log_view import update_logs
from niconavi_app.download_manager import register_download
from niconavi_app.project_io import load_project_archive


from datetime import datetime

from niconavi_app.niconavi.analysis import (
    make_grain_mask,
)
from niconavi_app.tools.tools import switch_tab_index, force_update_image_view
from niconavi_app.niconavi.grain_segmentation.grain_segmentation import (
    analyze_indexed_components_features,
    component_info_to_feature_matrix,
)
from niconavi_app.components.labeling_app.visualization import render_overlay_base64
from niconavi_app.niconavi.angle_map_boundary import (
    create_shock_filter_iterator,
    make_theta_phi_angle_info,
)


def image_width_micrometers(
    pixel_size_micrometers: float | int | None,
    resolution_width: int,
) -> float | None:
    if pixel_size_micrometers is None or resolution_width <= 0:
        return None
    return float(pixel_size_micrometers) * resolution_width


def pixel_size_micrometers(
    image_width_micrometers: float | int | None,
    resolution_width: int,
) -> float | None:
    if image_width_micrometers is None or resolution_width <= 0:
        return None
    return float(image_width_micrometers) / resolution_width


def make_image_width_input(stores: Stores) -> ft.TextField:
    """Edit physical image width while retaining the internal µm/px scale."""

    def current_image_width() -> float | None:
        return image_width_micrometers(
            stores.ui.one_pixel.get(),
            stores.computation_result.resolution_width.get(),
        )

    displayed_width = State(current_image_width())

    def update_pixel_size() -> None:
        stores.ui.one_pixel.set(
            pixel_size_micrometers(
                displayed_width.get(),
                stores.computation_result.resolution_width.get(),
            )
        )

    def update_displayed_width() -> None:
        displayed_width.set(current_image_width())

    displayed_width.bind(update_pixel_size)
    stores.ui.one_pixel.bind(update_displayed_width)
    stores.computation_result.resolution_width.bind(update_displayed_width)

    return make_reactive_float_text_filed(
        stores,
        displayed_width,
        parse_larger_than_0,
        accept_None=True,
    )


def get_current_datetime_str() -> str:
    """
    現在の日付と時刻(秒まで)を 'yyyy-mm-dd-HH-MM-SS' の形で文字列として返す。
    例: '2025-01-22-14-05-59'
    """
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def download_image_pdf(
    page: Page,
    stores: Stores,
    filename: str,
    *,
    logger: Logger,
) -> None:
    try:
        pdf_bytes, export_target = export_image_as_pdf_bytes(stores)
    except ValueError as exc:
        update_logs(stores, (str(exc), "err"), logger=logger)
        return

    token = register_download(pdf_bytes, filename, "application/pdf")
    page.launch_url(f"/api/download/{token}")

    # if export_target == "labeling view":
    #     stores.labeling.last_action_text.set(
    #         f"Prepared labeling view PDF download ({filename})."
    #     )

    update_logs(
        stores,
        (f"Prepared {export_target} PDF download ({filename}).", "ok"),
        logger=logger,
    )


def download_project_file(
    page: Page,
    stores: Stores,
    filename: str,
    *,
    logger: Logger,
) -> None:
    try:
        project_bytes = export_project_bytes(stores)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to export project: %s", exc)
        update_logs(
            stores,
            ("Could not prepare the project download.", "err"),
            logger=logger,
        )
        return

    token = register_download(project_bytes, filename, "application/octet-stream")
    page.launch_url(f"/api/download/{token}")

    update_logs(
        stores,
        (f"Prepared project download ({filename}).", "ok"),
        logger=logger,
    )


def download_grain_information(
    page: Page,
    stores: Stores,
    filename: str,
    *,
    logger: Logger,
) -> None:
    try:
        archive_bytes = export_grain_information_zip_bytes(stores)
    except ValueError as exc:
        update_logs(stores, (str(exc), "err"), logger=logger)
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to export grain information: %s", exc)
        update_logs(
            stores,
            ("Could not prepare the grain information download.", "err"),
            logger=logger,
        )
        return

    token = register_download(archive_bytes, filename, "application/zip")
    page.launch_url(f"/api/download/{token}")

    update_logs(
        stores,
        (f"Prepared grain information download ({filename}).", "ok"),
        logger=logger,
    )


def make_grain_boundary_checkbox(stores: Stores) -> ReactiveCheckbox:
    checkbox_visible = ReactiveState(
        lambda: (stores.computation_result.grain_map.get() is not None)
        and (stores.ui.selected_index.get() != 3),
        [stores.computation_result.grain_map, stores.ui.selected_index],
    )

    controller = LabelingController(stores=stores)

    def on_change(e):
        stores.ui.display_grain_boundary.set(e.control.value)

    return CustomReactiveCheckbox(
        label="GBs",
        value=stores.ui.display_grain_boundary,
        visible=checkbox_visible,
        # on_change=lambda e: stores.ui.display_grain_boundary.set(e.control.value),
        on_change=on_change,
    )


def make_mask_checkbox(stores: Stores) -> ReactiveCheckbox:
    checkbox_visible = ReactiveState(
        lambda: stores.computation_result.mask.get() is not None,
        [stores.computation_result.mask],
    )

    return CustomReactiveCheckbox(
        label="Apply mask",
        value=stores.ui.apply_mask,
        visible=checkbox_visible,
        on_change=lambda e: stores.ui.apply_mask.set(e.control.value),
    )


def load_existing_project(
    stores: Stores, file_path: str, *, logger: Logger | None = None
) -> None:
    active_logger = logger or getLogger("niconavi").getChild(__name__)
    update_logs(
        stores,
        (f"Loading project from {file_path}...", "msg"),
        logger=active_logger,
    )
    project = load_project_archive(Path(file_path))
    r = project.computation_result
    save_in_ComputationResultState(r, stores)
    stores.ui.current_project_path.set(file_path)
    stores.ui.once_start.set(True)
    restored_progress = infer_project_progress(r)
    stores.ui.progress.set(restored_progress)
    restore_map_tab_state(stores, project.ui_state)
    restore_labeling_feature_state(stores, project.ui_state)
    switch_tab_index(stores, restored_progress)
    restore_filter_tab_view(stores)
    restore_ui_selection(stores, project.ui_state, restored_progress)
    update_logs(
        stores,
        (f"Loaded project from {file_path}.", "ok"),
        logger=active_logger,
    )

    # if stores.computation_result.grain_classification_result.get() is not None:
    #     switch_tab_index(stores, 3)
    # force_update_image_view(stores)


def infer_project_progress(result: ComputationResult) -> int:
    if result.grain_classification_result is not None:
        return 4
    if result.grain_map is not None:
        return 3
    if result.raw_maps is not None:
        return 2
    if result.rotation_img is not None or result.center_int_x is not None:
        return 1
    if result.video_path is not None:
        return 0
    return 0


def restore_map_tab_state(stores: Stores, ui_state: dict[str, Any]) -> None:
    map_state = ui_state.get("map_tab", {}) if isinstance(ui_state, dict) else {}
    angle_map_info = map_state.get("angle_map_info")

    if angle_map_info is None:
        raw_maps = stores.computation_result.raw_maps.get()
        if raw_maps is not None:
            angle_map_info = make_theta_phi_angle_info(raw_maps)

    if angle_map_info is not None:
        angle_map_display = map_state.get("angle_map_display")
        if angle_map_display is None:
            angle_map_display = angle_map_info.get("angle_map_display")
        stores.ui.map_tab.angle_map_info.set(angle_map_info)
        stores.ui.map_tab.angle_map_display.set(angle_map_display)
        stores.ui.map_tab.shock_filter_iterator.set(
            create_shock_filter_iterator(angle_map_info)
        )
    else:
        stores.ui.map_tab.angle_map_info.set(None)
        stores.ui.map_tab.angle_map_display.set(None)
        stores.ui.map_tab.shock_filter_iterator.set(None)

    stores.ui.map_tab.cleaning_count.set(int(map_state.get("cleaning_count", 0)))
    stores.ui.map_tab.segmentation_angle.set(int(map_state.get("segmentation_angle", 5)))
    smallest_grain_size = max(
        1, int(map_state.get("smallest_grain_linear_size", 3))
    )
    slic_superpixel_size = max(5, int(map_state.get("slic_superpixel_size", 30)))
    stores.ui.map_tab.smallest_grain_linear_size.set(smallest_grain_size)
    stores.ui.map_tab.slic_superpixel_size.set(slic_superpixel_size)
    stores.ui.map_tab.fill_boundary_count.set(
        float(map_state.get("fill_boundary_count", 0.0))
    )
    stores.ui.map_tab.segmentation_done.set(
        bool(map_state.get("segmentation_done", False))
    )
    stores.ui.map_tab.fill_boundary_started.set(
        bool(map_state.get("fill_boundary_started", False))
    )
    stores.ui.map_tab.boundary_registered.set(
        bool(map_state.get("boundary_registered", False))
    )


def restore_ui_selection(
    stores: Stores,
    ui_state: dict[str, Any],
    fallback_progress: int,
) -> None:
    if not isinstance(ui_state, dict):
        return
    stored_pixel_size = ui_state.get("one_pixel", 0)
    if stored_pixel_size is None:
        stores.ui.one_pixel.set(None)
    else:
        try:
            parsed_pixel_size = float(stored_pixel_size)
        except (TypeError, ValueError):
            parsed_pixel_size = 0.0
        stores.ui.one_pixel.set(parsed_pixel_size if parsed_pixel_size > 0 else 0)
    selected_index = int(ui_state.get("selected_index", fallback_progress))
    if selected_index <= fallback_progress:
        switch_tab_index(stores, selected_index)
    stores.ui.selected_button_at_filter_tab.set(
        int(ui_state.get("selected_button_at_filter_tab", 0))
    )
    stores.ui.selected_button_at_grain_tab.set(
        int(ui_state.get("selected_button_at_grain_tab", 0))
    )


def restore_labeling_feature_state(stores: Stores, ui_state: dict[str, Any]) -> None:
    labeling_state = ui_state.get("labeling", {}) if isinstance(ui_state, dict) else {}
    stores.labeling.use_color_features.set(
        bool(labeling_state.get("use_color_features", True))
    )
    stores.labeling.use_shape_features.set(
        bool(labeling_state.get("use_shape_features", True))
    )
    stores.labeling.use_size_features.set(
        bool(labeling_state.get("use_size_features", True))
    )
    stores.labeling.use_position_features.set(
        bool(labeling_state.get("use_position_features", False))
    )


def on_result_load_project_file(
    stores: Stores,
    page: Page,
    resolve_file: Callable[[Any, Callable[[str], None]], None],
    *,
    logger: Logger,
) -> Callable[[ft.FilePickerResultEvent], None]:

    def proceed(resolved_path: str) -> None:
        if stores.ui.once_start.get():

            def handle_no(e1: ft.ControlEvent) -> None:
                page.close(dlg_modal)

            def handle_yes(e1: ft.ControlEvent) -> None:
                load_existing_project(stores, resolved_path, logger=logger)
                page.close(dlg_modal)

            dlg_modal = make_confirmation_dialog(
                "Load project?",
                "Unsaved changes will be discarded.",
                [("Cancel", handle_no), ("Load", handle_yes)],
            )

            page.open(dlg_modal)
        else:
            load_existing_project(stores, resolved_path, logger=logger)

    def closure(e: ft.FilePickerResultEvent) -> None:
        if not e.files:
            return
        try:
            resolve_file(e.files[0], proceed)
        except FileNotFoundError:
            update_logs(
                stores,
                ("Could not load the project file: file not found.", "err"),
                logger=logger,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process project file: %s", exc)
            update_logs(
                stores,
                ("Could not load the project file.", "err"),
                logger=logger,
            )

    return closure


def make_apply_mask_button(stores: Stores, *, logger: Logger) -> CustomReactiveCheckbox:

    def get_number_of_classification_phase(stores: Stores) -> int:
        ll = stores.computation_result.grain_classification_result.get()
        if ll is None:
            return 0
        else:
            # llkeysは、["quartz", "garnet", ...]のような配列
            llkeys = list(ll.keys())
            return len(llkeys)


def onlick_apply_mask_button(
    stores: Stores, e: ft.FilePickerResultEvent, *, logger: Logger
) -> None:
    grain_map = stores.computation_result.grain_map.get()
    grain_classification_result = (
        stores.computation_result.grain_classification_result.get()
    )
    if grain_map is not None and grain_classification_result is not None:

        grain_mask = make_grain_mask(grain_classification_result, grain_map)
        stores.computation_result.mask.set(grain_mask)

        switch_tab_index(stores, 0, logger=logger)
    else:
        grain_mask = None

    # return CustomExecuteButton(
    #     "save mask",
    #     on_click=lambda e: onlick_apply_mask_button(stores, e, logger=logger),
    #     visible=ReactiveState(
    #         lambda: get_number_of_classification_phase(stores) >= 2,
    #         [stores.computation_result.grain_classification_result],
    #     ),
    # )


class niconaviAppBar:
    def __init__(
        self,
        page: Page,
        stores: Stores,
        *,
        attach_to_page: bool = True,
        help_controller: ContextHelpController | None = None,
    ) -> None:

        logger = getLogger("niconavi").getChild(__name__)

        file_picker_load_project: FilePicker | None = None
        if not page.web:
            file_picker_load_project = FilePicker(on_result=save_file_result(stores))

        file_picker_load_project_file = FilePicker()
        if page.web:
            load_project_resolver = make_upload_file_handler(
                page=page,
                file_picker=file_picker_load_project_file,
                stores=stores,
                logger=logger,
                storage_key="load_project",
            )
        else:
            load_project_resolver = make_simple_file_handler()

        file_picker_load_project_file.on_result = on_result_load_project_file(
            stores,
            page,
            load_project_resolver,
            logger=logger,
        )

        file_picker_save_as_pdf: FilePicker | None = None
        if not page.web:
            file_picker_save_as_pdf = FilePicker(on_result=save_image_as_pdf(stores))
        file_picker_save_grain: FilePicker | None = None
        if not page.web:
            file_picker_save_grain = FilePicker(
                on_result=save_grain_information(stores)
            )

        page.overlay.append(file_picker_load_project_file)
        if file_picker_load_project is not None:
            page.overlay.append(file_picker_load_project)
        if file_picker_save_as_pdf is not None:
            page.overlay.append(file_picker_save_as_pdf)
        if file_picker_save_grain is not None:
            page.overlay.append(file_picker_save_grain)

        self.page = page

        def handle_save_project(_: ft.ControlEvent) -> None:
            if page.web:
                filename = f"output_{get_current_datetime_str()}.niconavi"
                download_project_file(page, stores, filename, logger=logger)
                return

            handle_save_project_as(_)

        def handle_save_project_as(_: ft.ControlEvent) -> None:
            filename = f"output_{get_current_datetime_str()}.niconavi"
            if page.web:
                download_project_file(page, stores, filename, logger=logger)
            elif file_picker_load_project is not None:
                file_picker_load_project.save_file(
                    file_name=filename,
                    file_type=FilePickerFileType.CUSTOM,
                    allowed_extensions=["niconavi"],
                )
            else:
                update_logs(
                    stores,
                    ("Project saving is unavailable.", "err"),
                    logger=logger,
                )

        def handle_save_image_as_pdf(_: ft.ControlEvent) -> None:
            filename = f"image_{get_current_datetime_str()}.pdf"
            if page.web:
                download_image_pdf(page, stores, filename, logger=logger)
            elif file_picker_save_as_pdf is not None:
                file_picker_save_as_pdf.save_file(
                    file_name=filename,
                    file_type=FilePickerFileType.CUSTOM,
                    allowed_extensions=["pdf"],
                )
            else:
                update_logs(
                    stores,
                    ("PDF export is unavailable.", "err"),
                    logger=logger,
                )

        def handle_save_grain(_: ft.ControlEvent) -> None:
            if page.web:
                filename = f"grain_data_{get_current_datetime_str()}.zip"
                download_grain_information(page, stores, filename, logger=logger)
            elif file_picker_save_grain is not None:
                file_picker_save_grain.save_file(
                    file_name=f"grain_data_{get_current_datetime_str()}",
                    file_type=FilePickerFileType.CUSTOM,
                    allowed_extensions=["csv"],
                )
            else:
                update_logs(
                    stores,
                    ("Grain information export is unavailable.", "err"),
                    logger=logger,
                )

        def make_file_menu_item(
            text: str,
            on_click: Callable[[ft.ControlEvent], None],
            icon: Any = None,
        ) -> PopupMenuItem:
            item = PopupMenuItem(
                text=text,
                icon=icon,
                disabled=not stores.ui.computing_is_stop.get(),
                on_click=lambda e: on_click(e)
                if stores.ui.computing_is_stop.get()
                else None,
            )

            def sync_disabled() -> None:
                item.disabled = not stores.ui.computing_is_stop.get()
                try:
                    item.update()
                except AssertionError:
                    pass

            stores.ui.computing_is_stop.bind(sync_disabled)
            return item

        def build_menu_items() -> list[PopupMenuItem]:
            return [
                make_file_menu_item(
                    text="Save project",
                    icon=ft.Icons.SAVE,
                    on_click=handle_save_project,
                ),
                make_file_menu_item(
                    text="Load project",
                    icon=ft.Icons.FOLDER_OPEN,
                    on_click=lambda _: file_picker_load_project_file.pick_files(
                        allowed_extensions=["niconavi"]
                    ),
                ),
                make_file_menu_item(
                    text="Export image (PDF)", on_click=handle_save_image_as_pdf
                ),
                make_file_menu_item(
                    text="Export grain data", on_click=handle_save_grain
                ),
            ]

        def build_toolbar_controls() -> list[ft.Control]:
            return [
                ft.Row(
                    [
                        CustomText("Image width:"),
                        make_image_width_input(stores),
                        CustomText("μm"),
                    ]
                ),

                make_mask_checkbox(stores),
                make_grain_boundary_checkbox(stores),
            ]

        def build_menu_control() -> Container:
            return Container(
                content=PopupMenuButton(
                    items=build_menu_items(),
                    icon=ft.Icons.MENU,
                    icon_color=ft.Colors.WHITE,
                ),
                margin=margin.only(left=10, right=15),
            )

        def build_end_controls() -> list[ft.Control]:
            controls: list[ft.Control] = []
            if help_controller is not None:
                controls.append(help_controller.create_button())
            controls.append(build_menu_control())
            return controls

        self.appbar = None
        if attach_to_page:
            self.appbar = AppBar(
                title=CustomText(
                    stores.appearance.niconavi_version,
                    size=15,
                    text_align=TextAlign.CENTER,
                ),
                center_title=True,
                toolbar_height=50,
                bgcolor="#ff333333",
                actions=[*build_toolbar_controls(), *build_end_controls()],
            )
            self.page.appbar = self.appbar
            self.page.update()

        self.toolbar = ft.Container(
            content=ft.Row(
                [
                    ft.Row(
                        build_toolbar_controls(),
                        alignment=ft.MainAxisAlignment.END,
                        scroll=ft.ScrollMode.HIDDEN,
                        expand=True,
                    ),
                    ft.VerticalDivider(),
                    *build_end_controls(),
                ],
                spacing=0,
            ),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            bgcolor=ft.Colors.BLACK26,
        )


def restore_filter_tab_view(stores: Stores) -> None:

    # controller = LabelingController(stores=stores)
    # controller.reset_application()

    stores.labeling.image_view_pan_x.set(0.0)
    stores.labeling.image_view_pan_y.set(0.0)
    stores.labeling.image_view_zoom.set(1.0)

    grain_map = stores.computation_result.grain_map.get()
    raw_maps = stores.computation_result.raw_maps.get()
    grain_boundary = stores.computation_result.grain_boundary.get()

    if grain_map is None or raw_maps is None:
        return

    background_image = (
        raw_maps.get("R_color_map_raw") if isinstance(raw_maps, dict) else None
    )
    if background_image is None and isinstance(raw_maps, dict):
        background_image = raw_maps.get("R_color_map")
    if background_image is None:
        return

    boundary_mask = None
    if grain_boundary is not None:
        boundary_mask = grain_boundary.astype(bool)
    else:
        boundary_mask = np.zeros_like(grain_map, dtype=bool)

    stores.labeling.image_width.set(grain_map.shape[1])
    stores.labeling.image_height.set(grain_map.shape[0])

    try:
        info = analyze_indexed_components_features(grain_map, background_image)
        features = component_info_to_feature_matrix(info)
    except Exception:
        features = None

    stores.labeling_computation_result["index_map"] = grain_map
    stores.labeling_computation_result["boundary_mask"] = boundary_mask
    stores.labeling_computation_result["background_image"] = background_image
    if features is not None:
        stores.labeling_computation_result["features"] = features

    labeling_map = stores.labeling_shared.labeling_map
    labeling_map.index_map.set(grain_map)
    labeling_map.boundary_mask.set(boundary_mask)
    labeling_map.background_image.set(background_image)
    if features is not None:
        labeling_map.features.set(features)
        controller = LabelingController(stores=stores)
        controller.rebuild_classifier_from_current_features()
    else:
        stores.labeling_shared.clf.force_set(None)

    results = stores.computation_result.grain_classification_result.get()
    if results:
        ordered_items = list(results.items())
        label_dict: dict[int, str] = {}
        custom_colors: dict[int, str] = {}
        grain_map_max_index = int(np.max(grain_map)) if grain_map.size else 0
        valid_grain_ids = np.unique(grain_map)
        valid_grain_ids = valid_grain_ids[valid_grain_ids > 0]
        predictions_length = grain_map_max_index + 1
        predictions = np.zeros(predictions_length, dtype=np.int32)
        for class_id, (label_name, info) in enumerate(ordered_items, start=1):
            label_dict[class_id] = label_name
            color = info.get("color") or "#9e9e9e"
            custom_colors[class_id] = color
            indices = np.asarray(info.get("index", []), dtype=np.int32)
            if indices.size > 0:
                valid = indices[np.isin(indices, valid_grain_ids)]
                predictions[valid] = class_id

        labeling_map.predictions.set(predictions)
        stores.labeling.labels.set(label_dict)
        stores.labeling._next_class_id.set(len(label_dict) + 1)
        stores.labeling._reusable_class_ids.set([])
        stores.labeling.current_class.set(next(iter(label_dict), None))
        stores.labeling.custom_colors.set(custom_colors)
        legend_entries = [
            {"color": custom_colors[cid], "label": label_dict[cid], "class_id": cid}
            for cid in label_dict
        ]
        stores.labeling.legend_entries.set(legend_entries)
        stores.labeling.results.set(results)
        stores.labeling.display_predictions.set(predictions)
        stores.labeling._clicked_indices_cache.set(None)
        stores.labeling.user_clicked.set(True)
        stores.labeling._loaded.set(True)
        stores.labeling.status_text.set("Select a label.")
        # stores.labeling.last_action_text.set("Project loaded.")
        # update_logs(stores, ("", "warn"))
        total_labeled = int(np.count_nonzero(predictions))
        stores.labeling.labeled_stats_text.set(f"Labeled regions: {total_labeled}")
        stores.labeling.prediction_stats_text.set("")
        stores.labeling.image_width.set(grain_map.shape[1])
        stores.labeling.image_height.set(grain_map.shape[0])

        overlay_alpha = float(max(0.0, min(1.0, stores.labeling.overlay_alpha.get())))
        image_base64, palette = render_overlay_base64(
            grain_map,
            predictions,
            overlay_alpha=overlay_alpha,
            boundary_mask=boundary_mask,
            background_image=background_image,
            show_boundaries=stores.labeling.show_boundaries.get(),
            custom_colors=custom_colors,
        )
        stores.labeling.image_src_base64.set(image_base64)
        stores.labeling.palette.set(list(palette))

        stores.labeling_shared.update_label_colors(custom_colors)
        stores.labeling_shared.populate_labels(label_dict, custom_colors)
        stores.ui.selected_button_at_filter_tab.set(0)
    else:
        stores.labeling.labels.set({})
        stores.labeling._next_class_id.set(1)
        stores.labeling._reusable_class_ids.set([])
        stores.labeling.current_class.set(None)
        stores.labeling.custom_colors.set({})
        stores.labeling.legend_entries.set([])
        predictions_length = int(np.max(grain_map)) + 1 if grain_map.size else 1
        empty = np.zeros(predictions_length, dtype=np.int32)
        stores.labeling.display_predictions.set(empty)
        labeling_map.predictions.set(empty)
        image_base64, palette = render_overlay_base64(
            grain_map,
            empty,
            overlay_alpha=float(max(0.0, min(1.0, stores.labeling.overlay_alpha.get()))),
            boundary_mask=boundary_mask,
            background_image=background_image,
            show_boundaries=stores.labeling.show_boundaries.get(),
        )
        stores.labeling.image_src_base64.set(image_base64)
        stores.labeling.palette.set(list(palette))
        stores.labeling.image_width.set(grain_map.shape[1])
        stores.labeling.image_height.set(grain_map.shape[0])
        stores.labeling._loaded.set(True)
        stores.labeling.user_clicked.set(False)
        stores.labeling.status_text.set("Add a label and click on the image.")
        # stores.labeling.last_action_text.set("Project loaded.")
        stores.ui.selected_button_at_filter_tab.set(0)
