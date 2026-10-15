"""Launcher shared by the Windows and AppImage distributions.

The packaged Python runtime and third-party modules live next to this file.
Keeping this bootstrap outside ``niconavi_app`` lets both package formats use
the same relocatable directory layout without installing anything globally.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any, TextIO


APP_NAME = "NicoNavi"
PACKAGE_ROOT = Path(__file__).resolve().parent
LIB_DIR = PACKAGE_ROOT / "lib"
WINDOWS_APP_ID = "io.github.tan_furukawa.NicoNavi"


def _user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / APP_NAME

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home).expanduser() if xdg_data_home else Path.home() / ".local" / "share"
    return base / "niconavi"


def _write_diagnostic(log_dir: Path, filename: str, message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with (log_dir / filename).open("a", encoding="utf-8") as output:
        output.write(f"[{timestamp}] {message.rstrip()}\n")


def _capture_python_output(log_dir: Path) -> TextIO:
    output = (log_dir / "launcher.log").open(
        "a", encoding="utf-8", buffering=1
    )
    # pythonw.exe normally sets these streams to None, which previously hid
    # exceptions caught inside Flet's session worker.
    if sys.stdout is None:
        sys.stdout = output
    if sys.stderr is None:
        sys.stderr = output
    return output


def _find_assets_dir() -> Path:
    """Locate packaged assets without importing the heavy application module."""
    packaged_assets = LIB_DIR / "niconavi_app" / "assets"
    if packaged_assets.is_dir():
        return packaged_assets

    package_spec = find_spec("niconavi_app")
    if package_spec and package_spec.submodule_search_locations:
        assets_dir = Path(next(iter(package_spec.submodule_search_locations))) / "assets"
        if assets_dir.is_dir():
            return assets_dir.resolve()

    raise FileNotFoundError("Could not locate the NicoNavi assets directory.")


def _show_startup_screen(page: Any, ft: Any, assets_dir: Path) -> Any:
    """Display a client-side animation while Python imports the application."""
    page.padding = 0
    page.title = APP_NAME
    page.bgcolor = "#ff444444"
    page.window.icon = os.environ.get(
        "NICONAVI_WINDOW_ICON", str(assets_dir / "favicon.png")
    )

    startup_screen = ft.Container(
        content=ft.Column(
            [
                ft.Image(
                    src="favicon.png",
                    width=120,
                    height=120,
                    fit=ft.ImageFit.CONTAIN,
                ),
                ft.ProgressRing(
                    width=36,
                    height=36,
                    stroke_width=4,
                    color=ft.Colors.LIGHT_BLUE_200,
                ),
                ft.Text(
                    "Starting NicoNavi...",
                    color=ft.Colors.WHITE,
                    size=16,
                ),
                ft.Text(
                    "The first launch may take about 5 minutes.",
                    color=ft.Colors.WHITE70,
                    size=13,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
            tight=True,
        ),
        alignment=ft.alignment.center,
        bgcolor="#ff444444",
        left=0,
        top=0,
        right=0,
        bottom=0,
    )
    page.overlay.append(startup_screen)
    page.update()
    return startup_screen


def _configure_windows_identity() -> None:
    if sys.platform != "win32":
        return

    from windows_taskbar import patch_flet_desktop

    pythonw = PACKAGE_ROOT / "runtime" / "pythonw.exe"
    launcher = PACKAGE_ROOT / "launcher.pyw"
    icon = PACKAGE_ROOT / "niconavi.ico"
    relaunch_command = f'"{pythonw}" "{launcher}"'

    # Flet 0.28 only applies the Windows title-bar icon when it receives an
    # absolute path to an .ico file.  Pass the installer-generated icon to the
    # application before its first page update instead of falling back to the
    # Flet fish icon.
    os.environ["NICONAVI_WINDOW_ICON"] = str(icon)

    patch_flet_desktop(
        app_id=WINDOWS_APP_ID,
        relaunch_command=relaunch_command,
        display_name=APP_NAME,
        icon_path=str(icon),
    )


def main() -> None:
    data_dir = _user_data_dir()
    upload_dir = data_dir / "uploads"
    log_dir = data_dir / "logs"
    cache_dir = data_dir / "cache"

    for directory in (upload_dir, log_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    captured_output = _capture_python_output(log_dir)
    _write_diagnostic(log_dir, "launcher.log", "Starting NicoNavi launcher")

    # main.py reads FLET_UPLOAD_DIR while it is imported.  It therefore has to
    # be set before importing the application package.
    os.environ.setdefault("NICONAVI_DATA_DIR", str(data_dir))
    os.environ.setdefault("FLET_UPLOAD_DIR", str(upload_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))

    # A normal ``python launcher.pyw`` invocation adds the script directory to
    # sys.path, but the packaging smoke test executes it through runpy.  Add it
    # explicitly so bundled launcher helpers resolve the same way in both
    # cases.
    sys.path.insert(0, str(PACKAGE_ROOT))
    sys.path.insert(0, str(LIB_DIR))

    # The current logger writes niconavi.log relative to the working directory.
    # Keep all mutable files out of Program Files and the read-only AppImage.
    os.chdir(log_dir)

    import flet as ft

    _configure_windows_identity()
    assets_dir = _find_assets_dir()

    def run_session(page: ft.Page) -> None:
        _write_diagnostic(log_dir, "launcher.log", "Flet session connected")
        startup_screen = _show_startup_screen(page, ft, assets_dir)
        _write_diagnostic(log_dir, "launcher.log", "Startup screen displayed")
        try:
            # Importing the application pulls in the scientific Python stack.
            # Keep it deferred until the client can animate the startup screen.
            from niconavi_app.main import main as niconavi_main

            _write_diagnostic(log_dir, "launcher.log", "Application modules loaded")
            niconavi_main(page)
            page.overlay.remove(startup_screen)
            page.update()
        except BaseException:  # noqa: BLE001 - keep pythonw failures observable
            details = traceback.format_exc()
            _write_diagnostic(log_dir, "startup-error.log", details)
            try:
                if startup_screen in page.overlay:
                    page.overlay.remove(startup_screen)
                page.clean()
                page.title = f"{APP_NAME} - startup error"
                page.add(
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text(
                                    "NicoNavi could not start.",
                                    size=24,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                ft.Text(
                                    "Details were saved to:\n"
                                    f"{log_dir / 'startup-error.log'}"
                                ),
                            ],
                            tight=True,
                        ),
                        padding=30,
                    )
                )
                page.update()
            except BaseException:  # noqa: BLE001
                _write_diagnostic(
                    log_dir,
                    "startup-error.log",
                    "Failed to render the startup error page:\n"
                    + traceback.format_exc(),
                )
            return
        _write_diagnostic(log_dir, "launcher.log", "NicoNavi UI initialized")

    ft.app(
        target=run_session,
        name=APP_NAME,
        view=ft.AppView.FLET_APP,
        upload_dir=str(upload_dir),
        assets_dir=str(assets_dir),
    )
    captured_output.flush()


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - pythonw.exe has no stderr to show
        try:
            failure_dir = _user_data_dir() / "logs"
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "startup-error.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        finally:
            raise
