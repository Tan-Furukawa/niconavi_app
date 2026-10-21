import os
from pathlib import Path
from contextlib import asynccontextmanager
import signal
import threading
import time

from niconavi_app.state import ReactiveState
from niconavi_app.stores import (
    Stores,
    save_in_ComputationResultState,
    as_ComputationResult,
)

from niconavi_app.niconavi.type import GrainDetectionParameters, ComputationResult
from niconavi_app.components.progress_bar import update_progress_bar

from niconavi_app.components.app_bar import (
    load_existing_project,
    niconaviAppBar,
    restore_filter_tab_view,
)
from niconavi_app.components.page_tab.page_tab import PageTabs
from niconavi_app.components.log_view import create_column, update_logs, LogView
from niconavi_app.components.image_view import ImageView
from niconavi_app.components.image_selector import ImageSelector
from niconavi_app.components.progress_bar import ProgressBar
from niconavi_app.components.color_bar import ColorChartBar
from niconavi_app.components.context_help import attach_context_help
from niconavi_app.reactive_state import ReactiveColumn

import flet as ft
import flet.fastapi as ffast
from logging import getLogger
import pandas as pd
from niconavi_app.niconavi.log import set_logger
from niconavi_app.niconavi.reset_run_all import remove_heavy_objects
from niconavi_app.components.common_component import (
    CustomText,
)
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import RedirectResponse
from niconavi_app.download_manager import pop_download

import matplotlib

matplotlib.use("svg")  # 非対話型バックエンドに切り替え

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Where flet writes what the user uploads. In a checkout that is
# <parent of this repo>/uploads, where it has always been. An installed build
# has no such layout - parents[3] of a site-packages path is somewhere inside
# the runtime, which may not even be writable - so FLET_UPLOAD_DIR wins when
# the caller sets it. packaging/build_linux.sh's launcher does exactly that.
_configured_upload_dir = os.environ.get("FLET_UPLOAD_DIR")
UPLOAD_ROOT = (
    Path(_configured_upload_dir).expanduser().resolve()
    if _configured_upload_dir
    else (PROJECT_ROOT / "uploads").resolve()
)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FLET_SECRET_KEY", "niconavi-dev-secret")
os.environ["FLET_UPLOAD_DIR"] = str(UPLOAD_ROOT)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
# Drop a .niconavi file in here and the newest one loads on startup. It lives
# inside this repo rather than the parent one: reaching out to a sibling of
# PROJECT_ROOT made the app unusable from a plain niconavi_app checkout.
APP_ROOT = Path(__file__).resolve().parents[2]
DEBUG_START_PROJECT_DIR = APP_ROOT / "for-debug" / "start-from-project"
MIN_SIDE_PANEL_WIDTH = 320.0
MAX_SIDE_PANEL_WIDTH = 720.0


def resized_side_panel_width(current_width: float, delta_x: float) -> float:
    """Resize a right-anchored panel while keeping it within usable limits."""
    return max(
        MIN_SIDE_PANEL_WIDTH,
        min(MAX_SIDE_PANEL_WIDTH, current_width - delta_x),
    )


def find_debug_start_project() -> Path | None:
    if not DEBUG_START_PROJECT_DIR.exists():
        return None
    project_files = [
        path
        for path in DEBUG_START_PROJECT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".niconavi"
    ]
    if not project_files:
        return None
    return max(project_files, key=lambda path: path.stat().st_mtime)


def window_icon_path() -> str:
    """Return the native window icon configured by a packaged launcher."""
    return os.environ.get("NICONAVI_WINDOW_ICON", str(ASSETS_DIR / "favicon.png"))


def main(page: ft.Page) -> None:

    logger = getLogger("niconavi")
    set_logger()

    logger.info("==================User made new project=====================")

    stores = Stores()

    page.padding = 0
    page.title = stores.appearance.niconavi_version
    page.bgcolor = "#ff444444"
    page.window.icon = window_icon_path()

    # ! ===================
    def terminate_process() -> None:
        time.sleep(0.2)
        try:
            parent_pid = os.getppid()
            if parent_pid and parent_pid != 1:
                os.kill(parent_pid, signal.SIGTERM)
        except Exception:
            pass
        os.kill(os.getpid(), signal.SIGINT)

    def handle_disconnect(_: object) -> None:
        logger.info("Page disconnected; shutting down server.")
        threading.Thread(target=terminate_process, daemon=True).start()

    # A native Flet window makes ft.app() return normally when it is closed.
    # Killing its parent here could also terminate the user's shell because
    # packaged launchers replace themselves with this Python process.
    if page.web:
        page.on_disconnect = handle_disconnect
    # ! ===================

    log_column_controls = ReactiveState(
        lambda: create_column(stores).controls, [stores.ui.log_view.log_contents]
    )

    column = ReactiveColumn(
        controls=log_column_controls,
        scroll=ft.ScrollMode.ALWAYS,
        spacing=10,
        scroll_offset_update=-1,
    )

    log_view = LogView(stores, column)

    log_view_header = ft.Container(
        content=CustomText("Output"),
        height=25,
        padding=ft.padding.only(left=20),
        alignment=ft.alignment.center_left,
        bgcolor=ft.Colors.BLACK26,
    )

    image_selector_header = ft.Container(
        content=CustomText("Images"),
        height=25,
        width=200,
        margin=0,
        # padding=ft.padding.only(left=20),
        alignment=ft.alignment.center,
        bgcolor=ft.Colors.BLACK26,
    )

    image_selector = ft.Container(
        content=ft.Column(
            [image_selector_header, ImageSelector(stores, page)],
            # scroll=True
        ),
        margin=0,
        padding=ft.padding.only(top=0),
        width=200,
        bgcolor=ft.Colors.BLACK12,
        # margin=ft.margin.only(top=0, bottom=0),
        # padding=0,
        # bgcolor="white",
        # border_radius = 0,
        # padding=ft.padding.only(left=10, right=10),
    )

    container1 = ft.Container(
        content=ft.Column(
            [
                ImageView(stores, page),
                ColorChartBar(),
            ]
        ),
        margin=0,
        padding=10,
        expand=True,
        bgcolor=ft.Colors.BLACK,
        border_radius=0,
    )

    progress_bar = ProgressBar(stores)

    app_bar_controls = niconaviAppBar(page, stores, attach_to_page=False)
    page_tabs = PageTabs(page, stores)

    container2 = ft.Container(
        ft.Column(
            [
                app_bar_controls.toolbar,
                page_tabs,
                ft.Column(
                    [progress_bar, log_view_header, log_view],
                    spacing=0,
                    horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ],
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
        margin=0,
        width=stores.appearance.tabs_width,
    )

    attach_context_help(image_selector, area="images")
    attach_context_help(container2, area="controls")

    def resize_side_panel(event: ft.DragUpdateEvent) -> None:
        current_width = float(container2.width or stores.appearance.tabs_width)
        new_width = resized_side_panel_width(current_width, event.delta_x)
        if abs(new_width - current_width) < 0.5:
            return
        container2.width = new_width
        stores.appearance.tabs_width = int(round(new_width))
        container2.update()

    resize_handle = ft.GestureDetector(
        mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT,
        drag_interval=10,
        on_horizontal_drag_update=resize_side_panel,
        content=ft.Container(
            width=10,
            bgcolor=ft.Colors.BLACK12,
            content=ft.VerticalDivider(width=1, color=ft.Colors.WHITE24),
        ),
    )

    main_row = ft.Row(
        [
            container1,
            image_selector,
            resize_handle,
            container2,
        ],
        spacing=0,
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    page.add(
        ft.Container(
            main_row,
            expand=True,
        )
    )

    page.update()

    logger.info("Page layout and elements have been updated.")

    debug_project_path = find_debug_start_project()
    if debug_project_path is not None:
        try:
            load_existing_project(stores, str(debug_project_path))
            update_logs(
                stores,
                (f"Loaded debug start project: {debug_project_path}", "ok"),
                logger=logger,
            )
            page.update()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load debug start project: %s", exc)
            update_logs(
                stores,
                (f"Failed to load debug start project: {debug_project_path}", "err"),
                logger=logger,
            )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await ffast.app_manager.start()
    try:
        yield
    finally:
        await ffast.app_manager.shutdown()


app = FastAPI(lifespan=lifespan)

flet_app = ffast.app(
    main,
    route_url_strategy="path",
    upload_dir=str(UPLOAD_ROOT),
    assets_dir=str(ASSETS_DIR),
    secret_key=os.environ.get("FLET_SECRET_KEY"),
    # view=ft.WEB_BROWSER
)

app.mount("/app", flet_app)


@app.get("/")
async def redirect_root() -> RedirectResponse:
    return RedirectResponse(url="/app", status_code=307)


@app.get("/api/download/{token}")
async def download(token: str) -> Response:
    item = pop_download(token)
    if item is None:
        raise HTTPException(status_code=404, detail="Download token not found.")

    headers = {
        "Content-Disposition": f'attachment; filename="{item.filename}"',
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    return Response(content=item.content, media_type=item.mime_type, headers=headers)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run NicoNavi GUI application.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--web",
        action="store_true",
        help="Serve the application in a web browser instead of opening a desktop window.",
    )
    mode.add_argument(
        "--desktop",
        action="store_true",
        help="Launch as a desktop application (the default).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host interface for the FastAPI server (--web only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8551,
        help="Port for the FastAPI server (--web only).",
    )
    args = parser.parse_args()

    if not args.web:
        ft.app(
            target=main,
            view=ft.AppView.FLET_APP,
            upload_dir=str(UPLOAD_ROOT),
            assets_dir=str(ASSETS_DIR),
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port)
