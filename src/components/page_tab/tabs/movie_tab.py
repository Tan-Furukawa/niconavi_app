from typing import Callable, Dict, Any
import flet as ft
from flet import Page
import os
import shutil
import string
from pathlib import Path
import uuid

from stores import (
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
    reset_labeling_stores,
)


from components.common_component import (
    make_reactive_float_text_filed,
    ReactiveFloatTextField,
    CustomSelectFileButton,
    CustomExecuteButton,
    CustomText,
    CustomReactiveText,
)

from components.labeling_app.labeling_controller import LabelingController
from state import State, ReactiveState, is_not_None_state
from reactive_state import ReactiveRow
from components.log_view import update_logs
from components.progress_bar import update_progress_bar
from tools.tools import switch_tab_index
from niconavi.tools.str_parser import parse_larger_than_0, parse_int
from tools.error import exec_at_error

from niconavi.custom_error import NoVideoError
from niconavi.image.type import RGBPicture
from niconavi.image.image import resize_img
from niconavi.tools.read_data import (
    get_first_frame_from_video,
    divide_video_into_n_frame,
)
from components.page_tab.tabs.reset_onclick import (
    reset_onclick_load_data,
    reset_onclick_recalculate_button,
)

from logging import getLogger, Logger
import niconavi.run_all as po

from copy import deepcopy


def load_data_clicked(stores: Stores, *, logger: Logger) -> None:

    n_frames = stores.computation_result.frame_number.get()
    try:
        logger.info("Attempting to load data.")
        r = as_ComputationResult(stores.computation_result)

        update_logs(
            stores, (f"Dividing video into {n_frames} frames...", "msg"), logger=logger
        )

        r = po.load_data(r, progress_callback=lambda p: update_progress_bar(p, stores))

        update_logs(
            stores,
            (f"Successfully divided video into {n_frames} frames.", "ok"),
            logger=logger,
        )

        logger.info(f"Divide video into {n_frames} frames")

        try:
            update_logs(
                stores, ("Finding stage rotation center...", "msg"), logger=logger
            )

            r = po.find_image_center(
                r, progress_callback=lambda p: update_progress_bar(p, stores)
            )

            update_logs(
                stores,
                ("Stage rotation center determination completed.", "ok"),
                logger=logger,
            )

            update_progress_bar(0.0, stores)

            save_in_ComputationResultState(r, stores)

            switch_tab_index(stores, 1, logger)

            stores.ui.once_start.set(True)
            print(stores.ui.computing_is_stop.get())

        except Exception as e:
            exec_at_error(9002, stores, logger=logger)
    except NoVideoError as e:
        exec_at_error(1002, stores, logger=logger)
    except Exception as e:
        exec_at_error(9001, stores, logger=logger)


def reset_button_click(stores: Stores, *, logger: Logger) -> None:
    logger.info("On click reset button")

    controller = LabelingController(stores=stores)
    controller.reset_application()

    r = as_ComputationResult(stores.computation_result)
    r = reset_onclick_load_data(r)
    save_in_ComputationResultState(r, stores)
    stores.ui.once_start.set(False)


def recalculate_maps_click(stores: Stores, *, logger: Logger) -> None:

    try:
        update_progress_bar(None, stores)

        r = as_ComputationResult(stores.computation_result)

        r = reset_onclick_recalculate_button(r)

        save_in_ComputationResultState(r, stores)

        update_logs(stores, ("Starting recalculation...", "msg"), logger=logger)
        update_logs(stores, ("Simulating retardation colors...", "msg"), logger=logger)

        r = po.make_retardation_color_chart(
            r, progress_callback=lambda p: update_progress_bar(p, stores)
        )

        update_logs(
            stores,
            ("Estimating retardation at extinction angle + 45°...", "msg"),
            logger=logger,
        )

        r = po.make_raw_R_maps(
            r, progress_callback=lambda p: update_progress_bar(p, stores)
        )

        im_tilt0 = r.tilt_image_info.tilt_image0_raw
        # im_tilt45 = r.tilt_image_info.tilt_image45_raw
        # if im_tilt0 is not None and im_tilt45 is not None:

        if im_tilt0 is not None:
            update_logs(
                stores,
                (
                    "Stacking focused frames from the tilted thin-section point-shift video...",
                    "msg",
                ),
                logger=logger,
            )
            r = po.estimate_tilt_image_result(
                r, progress_callback=lambda p: update_progress_bar(p, stores)
            )

        update_progress_bar(None, stores)
        update_logs(stores, ("Image processing completed.", "ok"), logger=logger)
        save_in_ComputationResultState(r, stores)

        update_progress_bar(0.0, stores)
        switch_tab_index(stores, 2, logger=logger)

    except Exception as e:
        exec_at_error(9004, stores, logger=logger)


def pick_files_result(
    setted_state: State[str | None],
    # set_picture_state: State[RGBPicture | None],
    first_image_name: str,
    # set_original_resolution_info_state: State[tuple[int, int] | None],
    image_selector_index: int,
    stores: Stores,
    *,
    logger: Logger,
    handle_file: Callable[[ft.FilePicker, Callable[[str], None]], None],
) -> Callable[[ft.FilePickerResultEvent], None]:
    def closure(e: ft.FilePickerResultEvent) -> None:
        if e.files:

            def process_selected_video(resolved_path: str) -> None:
                resolution_width = stores.computation_result.resolution_width.get()

                initial_frame = get_first_frame_from_video(resolved_path)
                initial_frame_resized = resize_img(initial_frame, resolution_width)

                initial_frame_dir = deepcopy(
                    stores.computation_result.first_image.get()
                )
                initial_frame_dir[first_image_name] = initial_frame_resized

                stores.computation_result.first_image.set(initial_frame_dir)

                update_logs(
                    stores,
                    (
                        f"Imported video file {resolved_path}. Original resolution: {initial_frame.shape[1]}x{initial_frame.shape[0]} px.",
                        "ok",
                    ),
                    logger=logger,
                )
                setted_state.set(resolved_path)
                stores.ui.selected_button_at_movie_tab.set(image_selector_index)

            try:
                handle_file(e.files[0], process_selected_video)
            except FileNotFoundError:
                exec_at_error(1002, stores, logger=logger)
            except Exception as exc:
                logger.exception("Failed to process selected video: %s", exc)
                exec_at_error(9001, stores, logger=logger)
        else:
            setted_state.set(None)
            exec_at_error(1003, stores, logger=logger)

    return closure


def _extract_display_name(path: str | None, *, empty_placeholder: str) -> str:
    if path is None:
        return empty_placeholder
    name = os.path.basename(path)
    prefix, sep, rest = name.partition("_")
    if (
        sep
        and len(prefix) == 32
        and all(ch in string.hexdigits for ch in prefix)
        and rest
    ):
        return rest
    return name


def make_simple_file_handler() -> Callable[[Any, Callable[[str], None]], None]:
    def handler(file_info: Any, process: Callable[[str], None]) -> None:
        if not file_info.path:
            raise FileNotFoundError("File path is not available from FilePicker.")
        process(file_info.path)

    return handler


def make_upload_file_handler(
    *,
    page: Page,
    file_picker: ft.FilePicker,
    stores: Stores,
    logger: Logger,
    storage_key: str,
) -> Callable[[Any, Callable[[str], None]], None]:

    upload_root = Path(os.getenv("FLET_UPLOAD_DIR", Path.cwd() / "uploads")).resolve()
    upload_root.mkdir(parents=True, exist_ok=True)

    pending_uploads: Dict[str, tuple[Path, Callable[[str], None]]] = {}

    def handle_upload_event(event: ft.FilePickerUploadEvent) -> None:
        entry = pending_uploads.get(event.file_name)
        if entry is None:
            return

        dest_path, process_callback = entry

        if event.error:
            pending_uploads.pop(event.file_name, None)
            logger.error("Upload failed for %s: %s", event.file_name, event.error)
            exec_at_error(9001, stores, logger=logger)
            try:
                if dest_path.exists():
                    dest_path.unlink()
            except OSError as cleanup_error:
                logger.warning(
                    "Failed to remove temporary upload file %s: %s",
                    dest_path,
                    cleanup_error,
                )
            return

        if event.progress is None or event.progress < 1:
            return

        pending_uploads.pop(event.file_name, None)
        update_logs(
            stores,
            (f"Upload completed for {event.file_name}.", "ok"),
            logger=logger,
        )

        if not dest_path.exists():
            logger.error("Uploaded file not found at %s", dest_path)
            exec_at_error(9001, stores, logger=logger)
            return

        try:
            process_callback(str(dest_path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process uploaded file %s: %s", dest_path, exc)
            exec_at_error(9001, stores, logger=logger)
            try:
                if dest_path.exists():
                    dest_path.unlink()
            except OSError as cleanup_error:
                logger.warning(
                    "Failed to remove uploaded file after error %s: %s",
                    dest_path,
                    cleanup_error,
                )

    file_picker.on_upload = handle_upload_event

    def handler(file_info: Any, process: Callable[[str], None]) -> None:
        file_path = file_info.path
        if file_path and os.path.exists(file_path):
            process(file_path)
            return

        dest_relative = (
            Path("movie_tab")
            / storage_key
            / f"{uuid.uuid4().hex}_{Path(file_info.name).name}"
        )
        dest_path = upload_root / dest_relative
        target_dir = upload_root / "movie_tab" / storage_key
        if target_dir.exists():
            for child in target_dir.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        pending_uploads[file_info.name] = (dest_path, process)

        try:
            upload_url = page.get_upload_url(dest_relative.as_posix(), 600)
        except Exception as exc:  # noqa: BLE001
            pending_uploads.pop(file_info.name, None)
            logger.exception("Failed to request upload URL: %s", exc)
            exec_at_error(9001, stores, logger=logger)
            return

        try:
            file_picker.upload(
                [
                    ft.FilePickerUploadFile(
                        name=file_info.name,
                        upload_url=upload_url,
                    )
                ]
            )
            update_logs(
                stores,
                (f"Uploading {file_info.name}...", "msg"),
                logger=logger,
            )
        except Exception as exc:  # noqa: BLE001
            pending_uploads.pop(file_info.name, None)
            logger.exception("Failed to start upload: %s", exc)
            exec_at_error(9001, stores, logger=logger)
            return

    return handler


# def __init__(self, page: Page, saving_path_in_stores: State[str | None], on_load: Callable):


class FilePickButton(ft.Container):
    def __init__(
        self,
        page: Page,
        saving_path_in_stores: State[str | None],
        # set_picture_state: State[RGBPicture | None],
        first_image_name: str,
        image_selector_index: int,
        # set_original_resolution_info_state: State[tuple[int, int] | None],
        stores: Stores,
        *,
        logger: Logger,
        enable_web_upload: bool = False,
    ):
        super().__init__()

        file_picker: ft.FilePicker = ft.FilePicker()

        if enable_web_upload:
            handle_file = make_upload_file_handler(
                page=page,
                file_picker=file_picker,
                stores=stores,
                logger=logger,
                storage_key=f"{first_image_name}_{image_selector_index}",
            )
        else:
            handle_file = make_simple_file_handler()

        file_picker.on_result = pick_files_result(
            saving_path_in_stores,
            first_image_name,
            image_selector_index,
            stores,
            logger=logger,
            handle_file=handle_file,
        )

        page.overlay.append(file_picker)
        button_display = ReactiveState(
            lambda: _extract_display_name(
                saving_path_in_stores.get(), empty_placeholder="Select"
            ),
            [saving_path_in_stores],
        )
        pick_files_button = CustomSelectFileButton(
            text=button_display,
            on_click=lambda _: file_picker.pick_files(
                allowed_extensions=["mp4", "avi", "mov"]
            ),
        )
        self.content = pick_files_button


def make_file_pick_button(
    page: Page,
    saving_path_in_stores: State[str | None],
    first_image_name: str,
    stores: Stores,
    image_selector_index: int,
    *,
    logger: Logger,
    enable_web_upload: bool = False,
) -> ft.Row:

    button = FilePickButton(
        page=page,
        saving_path_in_stores=saving_path_in_stores,
        first_image_name=first_image_name,
        image_selector_index=image_selector_index,
        stores=stores,
        logger=logger,
        enable_web_upload=enable_web_upload,
    )

    return ft.Row(
        [
            ReactiveRow(
                [button],
                visible=ReactiveState(
                    lambda: not stores.ui.once_start.get(), [stores.ui.once_start]
                ),
            ),
            CustomReactiveText(
                text=ReactiveState(
                    lambda: _extract_display_name(
                        saving_path_in_stores.get(), empty_placeholder="Not selected"
                    ),
                    [saving_path_in_stores],
                ),
                visible=ReactiveState(
                    lambda: stores.ui.once_start.get(), [stores.ui.once_start]
                ),
            ),
        ]
    )


class MovieTab(ft.Container):
    def __init__(
        self,
        page: Page,
        stores: Stores,
    ):
        super().__init__()

        logger = getLogger("niconavi").getChild(__name__)

        reset_button = CustomExecuteButton(
            "reset all",
            on_click=lambda e: reset_button_click(stores, logger=logger),
            visible=ReactiveState(
                lambda: stores.ui.computing_is_stop.get()
                and stores.ui.once_start.get(),
                [stores.ui.computing_is_stop, stores.ui.once_start],
            ),
            bgcolor=ft.Colors.RED_ACCENT,
        )

        recalculate_maps = CustomExecuteButton(
            "▶ recalulate",
            on_click=lambda e: recalculate_maps_click(stores, logger=logger),
            visible=ReactiveState(
                lambda: stores.ui.computing_is_stop.get()
                and stores.ui.once_start.get()
                and (
                    (stores.computation_result.center_int_x.get() is not None)
                    and (stores.computation_result.center_int_y.get() is not None)
                )
                and stores.computation_result.raw_maps.get() is not None,
                [stores.ui.computing_is_stop, stores.ui.once_start],
            ),
        )

        execute_button = CustomExecuteButton(
            "▶ start",
            on_click=lambda e: load_data_clicked(stores, logger=logger),
            visible=ReactiveState(
                lambda: stores.ui.computing_is_stop.get()
                and not stores.ui.once_start.get(),
                [stores.ui.computing_is_stop, stores.ui.once_start],
            ),
        )

        cross_polarized_pick_files_button = make_file_pick_button(
            page,
            saving_path_in_stores=stores.computation_result.video_path,
            first_image_name="xpl",
            image_selector_index=0,
            # set_original_resolution_info_state=stores.computation_result.original_resolution,
            stores=stores,
            logger=logger,
            enable_web_upload=page.web,
        )

        retardation_plate_pick_files_button = make_file_pick_button(
            page,
            saving_path_in_stores=stores.computation_result.reta_video_path,
            first_image_name="full_wave",
            image_selector_index=1,
            # set_original_resolution_info_state=stores.computation_result.original_reta_resolution,
            stores=stores,
            logger=logger,
            enable_web_upload=page.web,
        )

        image45_files_button = make_file_pick_button(
            page,
            saving_path_in_stores=stores.computation_result.tilt_image_info.image45_path,
            first_image_name="image45",
            image_selector_index=5,
            # set_original_resolution_info_state=stores.computation_result.original_reta_resolution,
            stores=stores,
            logger=logger,
            enable_web_upload=page.web,
        )

        tilt_image45_files_button = make_file_pick_button(
            page,
            saving_path_in_stores=stores.computation_result.tilt_image_info.tilt_image45_path,
            first_image_name="image45_tilt",
            image_selector_index=3,
            # set_original_resolution_info_state=stores.computation_result.original_reta_resolution,
            stores=stores,
            logger=logger,
            enable_web_upload=page.web,
        )

        tilt_image0_files_button = make_file_pick_button(
            page,
            saving_path_in_stores=stores.computation_result.tilt_image_info.tilt_image0_path,
            first_image_name="image0_tilt",
            image_selector_index=2,
            # set_original_resolution_info_state=stores.computation_result.original_reta_resolution,
            stores=stores,
            logger=logger,
            enable_web_upload=page.web,
        )

        xpl_max_R_input = make_reactive_float_text_filed(
            stores,
            stores.computation_result.color_chart.xpl_max_retardation,
            parse_larger_than_0,
            accept_None=False,
        )

        xpl_pol_max_R = CustomReactiveText(
            ReactiveState(
                lambda: f"{stores.computation_result.color_chart.xpl_max_retardation.get() + stores.computation_result.full_wave_plate_nm.get()}",
                [
                    stores.computation_result.color_chart.xpl_max_retardation,
                    stores.computation_result.full_wave_plate_nm,
                ],
            )
        )

        full_wave_plate_nm_input = make_reactive_float_text_filed(
            stores,
            stores.computation_result.full_wave_plate_nm,
            parse_larger_than_0,
            accept_None=False,
        )

        #! 設定を一時的にOFFにする
        #! ----------------------------------------------------------
        # frame_num_input = make_reactive_float_text_filed(
        #     stores,
        #     stores.computation_result.frame_number,
        #     parse_int,
        #     accept_None=False,
        # )

        # resolution_width = make_reactive_float_text_filed(
        #     stores,
        #     stores.computation_result.resolution_width,
        #     parse_int,
        #     accept_None=False,
        # )

        one_pixel = make_reactive_float_text_filed(
            stores,
            stores.ui.one_pixel,
            parse_larger_than_0,
            accept_None=True,
        )


        #! ----------------------------------------------------------

        self.padding = stores.appearance.tab_padding
        self.content = ft.Column(
            [
                ft.Row(
                    [
                        CustomText("XPL Rotated", weight=ft.FontWeight.W_900),
                        CustomText("*Required", italic=True, size=12),
                        cross_polarized_pick_files_button,
                    ]
                ),
                ft.Row(
                    [
                        CustomText("max retardation ="),
                        xpl_max_R_input,
                        CustomText("nm"),
                    ]
                ),
                # resolution_info_text,
                ft.Divider(),
                ft.Row(
                    [
                        CustomText("XPL + λ-Plate Rotated", weight=ft.FontWeight.W_900),
                        CustomText("*Optional", italic=True, size=12),
                        retardation_plate_pick_files_button,
                    ]
                ),
                ft.Row(
                    [
                        CustomText("max retardation ="),
                        xpl_pol_max_R,
                        CustomText("nm"),
                    ]
                ),
                ft.Row(
                    [
                        CustomText("λ ="),
                        full_wave_plate_nm_input,
                        CustomText("nm"),
                    ]
                ),
                ft.Divider(),
                ft.Row(
                    [
                        CustomText("XPL + λ-Plate 0°/45°", weight=ft.FontWeight.W_900),
                        CustomText("*Optional; requires XPL+λ", italic=True, size=12),
                    ]
                ),
                ft.Row(
                    [
                        CustomText("0° tilt"),
                        tilt_image0_files_button,
                    ]
                ),
                ft.Row(
                    [
                        CustomText("45° tilt:"),
                        tilt_image45_files_button,
                    ]
                ),
                ft.Row(
                    [
                        CustomText("45° not tilt:"),
                        image45_files_button,
                    ]
                ),
                ft.Divider(),
                execute_button,
                recalculate_maps,
                reset_button,
                ft.Divider(),

                ft.Row(
                    [
                        ft.Icon(ft.Icons.SETTINGS),
                        CustomText("Setting"),
                    ]
                ),
                ft.Row(
                    [
                        CustomText("1 px ="),
                        one_pixel,
                        CustomText("μm"),
                    ]
                ),
                # ft.Row(
                #     [
                #         CustomText("image width resolution ="),
                #         resolution_width,
                #         CustomText("px"),
                #     ]
                # ),
                # ft.Row(
                #     [
                #         CustomText("number of frames"),
                #         frame_num_input,
                #         CustomText("frames"),
                #     ]
                # ),
                # ft.Divider(),
            ],
            scroll=ft.ScrollMode.ADAPTIVE,
        )
