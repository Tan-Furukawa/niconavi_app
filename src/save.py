from __future__ import annotations

import base64
import binascii
from io import BytesIO, StringIO
import json
from pathlib import Path
from typing import Callable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from pandas import to_pickle
import zipfile

from components.log_view import update_logs

from flet import FilePickerResultEvent

from stores import Stores, as_ComputationResult
from niconavi.reset_run_all import remove_heavy_objects


# ---------------------------------------------------------------------------
# Configuration for grain exports
# ---------------------------------------------------------------------------
# If not None, only keys listed here will be exported from grain_list entries.
GRAIN_LIST_INCLUDE_KEYS: set[str] | None = None

# Keys listed here will be skipped when exporting grain_list entries.
GRAIN_LIST_EXCLUDE_KEYS: set[str] = {"area_shape", "exQuality"}


def save_fn(path: Path, stores: Stores) -> None:
    res = as_ComputationResult(stores.computation_result)
    res = remove_heavy_objects(res)
    to_pickle(res, path)


def export_project_bytes(stores: Stores) -> bytes:
    res = as_ComputationResult(stores.computation_result)
    res = remove_heavy_objects(res)
    buffer = BytesIO()
    to_pickle(res, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def save_file_result(stores: Stores) -> Callable[[FilePickerResultEvent], None]:
    def closure(e: FilePickerResultEvent) -> None:
        if not e.path:
            raise ValueError("failed to get file path")
        path = Path(e.path)
        if path.suffix.lower() != ".niconavi":
            path = path.with_suffix(".niconavi")
        save_fn(path, stores)

    return closure


def _figure_pdf_bytes(stores: Stores) -> bytes:
    fig = stores.ui.displayed_fig.get()
    if fig is None:
        raise ValueError("No figure available to save.")
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf")
    buffer.seek(0)
    return buffer.getvalue()


def save_figure_as_pdf(path: Path, stores: Stores) -> None:
    pdf_bytes = _figure_pdf_bytes(stores)
    path.write_bytes(pdf_bytes)
    update_logs(stores, (f"Saved figure to {path}.", "ok"))


def _labeling_view_pdf_bytes(stores: Stores) -> bytes:
    image_base64 = stores.labeling.image_src_base64.get()
    if not image_base64:
        raise ValueError("No labeling image available to save.")
    try:
        image_bytes = base64.b64decode(image_base64)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Failed to decode labeling image.") from exc
    image = mpimg.imread(BytesIO(image_bytes), format="png")
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def save_labeling_view_as_pdf(path: Path, stores: Stores) -> None:
    pdf_bytes = _labeling_view_pdf_bytes(stores)
    path.write_bytes(pdf_bytes)
    message = f"Saved labeling view to {path}."
    # stores.labeling.last_action_text.set(message)
    update_logs(stores, (message, "ok"))


def export_image_as_pdf_bytes(stores: Stores) -> tuple[bytes, str]:
    if stores.ui.selected_index.get() == 3:
        return _labeling_view_pdf_bytes(stores), "labeling view"
    return _figure_pdf_bytes(stores), "figure"


def save_image_as_pdf(stores: Stores) -> Callable[[FilePickerResultEvent], None]:
    def closure(e: FilePickerResultEvent) -> None:
        try:
            pdf_bytes, export_target = export_image_as_pdf_bytes(stores)
        except ValueError as exc:
            update_logs(stores, (str(exc), "err"))
            return

        if not e.path:
            update_logs(
                stores,
                ("Failed to save PDF: no destination path provided.", "err"),
            )
            return
        path = Path(e.path)
        if path.suffix.lower() != ".pdf":
            path = path.with_suffix(".pdf")

        path.write_bytes(pdf_bytes)
        if export_target == "labeling view":
            message = f"Saved labeling view to {path}."
            # stores.labeling.last_action_text.set(message)
        else:
            message = f"Saved figure to {path}."
        update_logs(stores, (message, "ok"))

    return closure


COLOR_KEYS = {"R_color", "extinction_color"}


def _as_hex_color(value) -> str | None:
    if isinstance(value, str):
        if value.startswith("#") and len(value) in {4, 7}:
            hex_part = value[1:]
            if len(hex_part) == 3:
                return "".join(ch * 2 for ch in hex_part).upper()
            return hex_part.upper()
        if len(value) == 6 and all(ch in "0123456789ABCDEFabcdef" for ch in value):
            return value.upper()
        return None

    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None

    if arr.size < 3:
        return None

    rgb = arr.flat[:3]
    if np.max(rgb) <= 1.0:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255).astype(int)
    return "{:02X}{:02X}{:02X}".format(*rgb.tolist())


def _serialize_generic(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)
    if isinstance(value, dict):
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)
    return value


def _serialize_value(key: str, value):
    if key in COLOR_KEYS:
        hex_color = _as_hex_color(value)
        if hex_color is not None:
            return hex_color
    if isinstance(value, str) and value.startswith("#"):
        return value[1:].upper()
    return _serialize_generic(value)


def _build_grain_information_files(stores: Stores) -> dict[str, bytes]:
    grain_list = stores.computation_result.grain_list.get()
    grain_map = stores.computation_result.grain_map.get()

    if grain_list is None:
        raise ValueError("No grain list available to export.")
    if grain_map is None:
        raise ValueError("No grain map available to export.")

    def include_key(key: str) -> bool:
        if GRAIN_LIST_INCLUDE_KEYS is not None and key not in GRAIN_LIST_INCLUDE_KEYS:
            return False
        if key in GRAIN_LIST_EXCLUDE_KEYS:
            return False
        return True

    def flatten_record(grain: dict) -> dict[str, object]:
        record: dict[str, object] = {}

        for key, value in grain.items():
            if not include_key(key):
                continue

            if isinstance(value, (tuple, list)) and value:
                simple = all(isinstance(v, (int, float, np.generic)) for v in value)
                if simple:
                    labels = []
                    if len(value) == 2:
                        labels = ["_x", "_y"]
                    elif len(value) == 3:
                        labels = ["_x", "_y", "_z"]
                    else:
                        labels = [f"_{idx}" for idx in range(len(value))]

                    for suffix, item in zip(labels, value):
                        record[f"{key}{suffix}"] = _serialize_value(f"{key}{suffix}", item)
                    continue

            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    record[f"{key}_{sub_key}"] = _serialize_value(f"{key}_{sub_key}", sub_val)
                continue

            record[key] = _serialize_value(key, value)

        return record

    rows = [flatten_record(grain) for grain in grain_list]

    df = pd.DataFrame(rows)
    grain_list_bytes = df.to_csv(index=False).encode("utf-8")

    np_grain_map = np.asarray(grain_map, dtype=np.int64)
    map_buffer = StringIO()
    np.savetxt(map_buffer, np_grain_map, fmt="%d", delimiter=",")
    grain_map_bytes = map_buffer.getvalue().encode("utf-8")

    return {
        "grain_list.csv": grain_list_bytes,
        "grain_map.csv": grain_map_bytes,
    }


def export_grain_information(folder: Path, stores: Stores) -> None:
    files = _build_grain_information_files(stores)

    folder.mkdir(parents=True, exist_ok=True)

    for name, data in files.items():
        (folder / name).write_bytes(data)


def export_grain_information_zip_bytes(stores: Stores) -> bytes:
    files = _build_grain_information_files(stores)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    buffer.seek(0)
    return buffer.getvalue()


def save_grain_information(stores: Stores) -> Callable[[FilePickerResultEvent], None]:
    def closure(e: FilePickerResultEvent) -> None:
        if not e.path:
            update_logs(stores, ("File selection canceled.", "msg"))
            return
        folder = Path(e.path)
        print("----------")
        print(folder)
        print("----------")
        if folder.suffix:
            folder = folder.with_suffix("")
        try:
            export_grain_information(folder, stores)
            update_logs(stores, (f"Saved grain information to {folder}.", "ok"))
        except ValueError as exc:
            update_logs(stores, (str(exc), "err"))

    return closure
