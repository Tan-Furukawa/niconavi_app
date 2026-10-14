"""Launcher shared by the Windows and AppImage distributions.

The packaged Python runtime and third-party modules live next to this file.
Keeping this bootstrap outside ``niconavi_app`` lets both package formats use
the same relocatable directory layout without installing anything globally.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


APP_NAME = "NicoNavi"
PACKAGE_ROOT = Path(__file__).resolve().parent
LIB_DIR = PACKAGE_ROOT / "lib"


def _user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / APP_NAME

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home).expanduser() if xdg_data_home else Path.home() / ".local" / "share"
    return base / "niconavi"


def main() -> None:
    data_dir = _user_data_dir()
    upload_dir = data_dir / "uploads"
    log_dir = data_dir / "logs"
    cache_dir = data_dir / "cache"

    for directory in (upload_dir, log_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # main.py reads FLET_UPLOAD_DIR while it is imported.  It therefore has to
    # be set before importing the application package.
    os.environ.setdefault("NICONAVI_DATA_DIR", str(data_dir))
    os.environ.setdefault("FLET_UPLOAD_DIR", str(upload_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))

    sys.path.insert(0, str(LIB_DIR))

    # The current logger writes niconavi.log relative to the working directory.
    # Keep all mutable files out of Program Files and the read-only AppImage.
    os.chdir(log_dir)

    import flet as ft

    from niconavi_app.main import ASSETS_DIR, UPLOAD_ROOT
    from niconavi_app.main import main as niconavi_main

    ft.app(
        target=niconavi_main,
        view=ft.AppView.FLET_APP,
        upload_dir=str(UPLOAD_ROOT),
        assets_dir=str(ASSETS_DIR),
    )


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
