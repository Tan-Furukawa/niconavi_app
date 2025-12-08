import flet as ft
from typing import Callable, Any
import sys
import pandas as pd
import numpy as np
from stores import (
    ComputationResult,
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
)

# from components.selector.always import make_always_visible_state
from logging import getLogger, Logger
from reactive_state import ReactiveProgressRing, ReactiveText, ReactiveCheckbox
from state import ReactiveState
from save import (
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
from components.page_tab.tabs.movie_tab import (
    make_simple_file_handler,
    make_upload_file_handler,
)

from components.labeling_app.labeling_controller import LabelingController
from components.common_component import (
    CustomText,
    CustomReactiveCheckbox,
)
from components.log_view import update_logs
from download_manager import register_download


from datetime import datetime

from niconavi.analysis import (
    make_grain_mask,
)
from tools.tools import switch_tab_index, force_update_image_view
from niconavi.grain_segmentation.grain_segmentation import (
    analyze_false_components_features,
    component_info_to_feature_matrix,
)
from components.labeling_app.visualization import render_overlay_base64


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
        label="Show grain boundaries",
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


def load_existing_project(stores: Stores, file_path: str) -> None:
    try:
        r: ComputationResult = pd.read_pickle(file_path)
    except ModuleNotFoundError as exc:
        if "numpy._core" in str(exc):
            ensure_numpy_core_alias()
            r = pd.read_pickle(file_path)
        else:
            raise
    save_in_ComputationResultState(r, stores)
    stores.ui.once_start.set(True)
    restore_filter_tab_view(stores)

    # if stores.computation_result.grain_classification_result.get() is not None:
    #     switch_tab_index(stores, 3)
    # force_update_image_view(stores)


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
                load_existing_project(stores, resolved_path)
                page.close(dlg_modal)

            dlg_modal = ft.AlertDialog(
                modal=True,
                title=CustomText("Please confirm"),
                content=CustomText(
                    "Loading a project will discard the current session. Continue?"
                ),
                actions=[
                    ft.TextButton("Yes", on_click=handle_yes),
                    ft.TextButton("No", on_click=handle_no),
                ],
                actions_alignment=ft.MainAxisAlignment.END,
            )

            page.open(dlg_modal)
        else:
            load_existing_project(stores, resolved_path)

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
        self, page: Page, stores: Stores, *, attach_to_page: bool = True
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

        def build_menu_items() -> list[PopupMenuItem]:
            return [
                PopupMenuItem(text="Save project", on_click=handle_save_project),
                PopupMenuItem(
                    text="Load project",
                    on_click=lambda _: file_picker_load_project_file.pick_files(
                        allowed_extensions=["niconavi", "pkl"]
                    ),
                ),
                PopupMenuItem(
                    text="Save image as PDF", on_click=handle_save_image_as_pdf
                ),
                PopupMenuItem(
                    text="Save grain information as CSV", on_click=handle_save_grain
                ),
            ]

        def build_action_controls() -> list[ft.Control]:
            return [
                make_mask_checkbox(stores),
                make_grain_boundary_checkbox(stores),
                ft.VerticalDivider(),
                Container(
                    content=PopupMenuButton(
                        items=build_menu_items(),
                        icon=ft.Icons.MENU,
                        icon_color=ft.Colors.WHITE,
                    ),
                    margin=margin.only(left=10, right=25),
                ),
            ]

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
                actions=build_action_controls(),
            )
            self.page.appbar = self.appbar
            self.page.update()

        self.toolbar = ft.Container(
            content=ft.Row(
                build_action_controls(),
                alignment=ft.MainAxisAlignment.END,
            ),
            width=stores.appearance.tabs_width,
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            bgcolor=ft.Colors.BLACK26,
        )


def restore_filter_tab_view(stores: Stores) -> None:

    # controller = LabelingController(stores=stores)
    # controller.reset_application()

    grain_map = stores.computation_result.grain_map.get()
    raw_maps = stores.computation_result.raw_maps.get()
    grain_boundary = stores.computation_result.grain_boundary.get()

    if grain_map is None or raw_maps is None:
        return

    background_image = (
        raw_maps.get("R_color_map") if isinstance(raw_maps, dict) else None
    )
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
        _, info = analyze_false_components_features(
            boundary_mask, background_image, connectivity=4
        )
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

    results = stores.computation_result.grain_classification_result.get()
    if results:
        ordered_items = list(results.items())
        label_dict: dict[int, str] = {}
        custom_colors: dict[int, str] = {}
        max_index = 0
        for _, info in ordered_items:
            indices = info.get("index")
            if indices is not None and len(indices) > 0:
                max_index = max(max_index, int(np.max(indices)))
        predictions_length = max_index + 1 if max_index >= 0 else 1
        predictions = np.zeros(predictions_length, dtype=np.int32)
        for class_id, (label_name, info) in enumerate(ordered_items, start=1):
            label_dict[class_id] = label_name
            color = info.get("color") or "#9e9e9e"
            custom_colors[class_id] = color
            indices = np.asarray(info.get("index", []), dtype=np.int32)
            if indices.size > 0:
                valid = indices[(indices >= 0) & (indices < predictions_length)]
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
        total_labeled = sum(len(info.get("index", [])) for info in results.values())
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
        empty = np.array([], dtype=np.int32)
        stores.labeling.display_predictions.set(empty)
        labeling_map.predictions.set(empty)
        stores.labeling.image_src_base64.set("")
        stores.labeling.palette.set(["#bdbdbd"])
        stores.labeling._loaded.set(False)
        stores.labeling.user_clicked.set(False)
        stores.labeling.status_text.set("Add a label and click on the image.")
        # stores.labeling.last_action_text.set("Project loaded.")
        stores.ui.selected_button_at_filter_tab.set(0)


def ensure_numpy_core_alias() -> None:
    try:
        import numpy as np
    except Exception:
        return

    core_module = getattr(np, "core", None)
    if core_module is None:
        return

    sys.modules.setdefault("numpy._core", core_module)
    numeric_module = getattr(core_module, "numeric", None)
    if numeric_module is not None:
        sys.modules.setdefault("numpy._core.numeric", numeric_module)
