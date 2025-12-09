import os
from pathlib import Path
from contextlib import asynccontextmanager
import signal
import threading
import time

from state import ReactiveState
from stores import (
    Stores,
    save_in_ComputationResultState,
    as_ComputationResult,
)

from niconavi.type import GrainDetectionParameters, ComputationResult
from components.progress_bar import update_progress_bar

from components.app_bar import niconaviAppBar, restore_filter_tab_view
from components.page_tab.page_tab import PageTabs
from components.log_view import create_column, update_logs, LogView
from components.image_view import ImageView
from components.image_selector import ImageSelector
from components.progress_bar import ProgressBar
from components.color_bar import ColorChartBar
from reactive_state import ReactiveColumn

import flet as ft
import flet.fastapi as ffast
from logging import getLogger
import pandas as pd
from niconavi.log import set_logger
from niconavi.reset_run_all import remove_heavy_objects
from components.common_component import (
    CustomText,
)
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import RedirectResponse
from download_manager import pop_download

import matplotlib

matplotlib.use("svg")  # 非対話型バックエンドに切り替え

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = (PROJECT_ROOT / "uploads").resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FLET_SECRET_KEY", "niconavi-dev-secret")
os.environ.setdefault("FLET_UPLOAD_DIR", str(UPLOAD_ROOT))

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def main(page: ft.Page) -> None:

    logger = getLogger("niconavi")
    set_logger()

    logger.info("==================User made new project=====================")

    stores = Stores()

    page.padding = 0
    page.title = stores.appearance.niconavi_version
    page.bgcolor = "#ff444444"

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
        content=CustomText("output"),
        height=25,
        padding=ft.padding.only(left=20),
        width=stores.appearance.tabs_width,
        bgcolor=ft.Colors.BLACK26,
    )

    image_selector_header = ft.Container(
        content=CustomText("image list"),
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

    container2 = ft.Container(
        ft.Column(
            [
                app_bar_controls.toolbar,
                PageTabs(page, stores),
                ft.Column([progress_bar, log_view_header, log_view], spacing=0),
            ],
            spacing=0,
        ),
        margin=0,
    )

    page.add(
        ft.Container(
            ft.Row(
                [
                    container1,
                    image_selector,
                    container2,
                ],
            ),
            expand=True,
        )
    )

    page.update()

    logger.info("Page layout and elements have been updated.")


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
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="Launch as a desktop application instead of serving via web.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host interface for the FastAPI server (web mode only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8551,
        help="Port for the FastAPI server (web mode only).",
    )
    args = parser.parse_args()

    if args.desktop:
        ft.app(
            target=main,
            view=ft.AppView.FLET_APP,
            upload_dir=str(UPLOAD_ROOT),
            assets_dir=str(ASSETS_DIR),
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port)
