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
from pathlib import Path
from typing import TextIO


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

    from niconavi_app.main import ASSETS_DIR, UPLOAD_ROOT
    from niconavi_app.main import main as niconavi_main

    def run_session(page: ft.Page) -> None:
        _write_diagnostic(log_dir, "launcher.log", "Flet session connected")
        try:
            niconavi_main(page)
        except BaseException:  # noqa: BLE001 - keep pythonw failures observable
            details = traceback.format_exc()
            _write_diagnostic(log_dir, "startup-error.log", details)
            try:
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
        upload_dir=str(UPLOAD_ROOT),
        assets_dir=str(ASSETS_DIR),
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
