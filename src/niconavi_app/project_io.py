from __future__ import annotations

import inspect
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np

logger = logging.getLogger("niconavi")

from niconavi_app.niconavi.reset_run_all import remove_heavy_objects
from niconavi_app.niconavi.type import (
    ColorChart,
    ComputationResult,
    GrainDetectionParameters,
    OpticalParameters,
    PlotParameters,
    QuartzWedgeNormalization,
    TiltImageInfo,
)

SCHEMA_VERSION = 1

PROJECT_CLASSES = {
    "ColorChart": ColorChart,
    "ComputationResult": ComputationResult,
    "GrainDetectionParameters": GrainDetectionParameters,
    "OpticalParameters": OpticalParameters,
    "PlotParameters": PlotParameters,
    "QuartzWedgeNormalization": QuartzWedgeNormalization,
    "TiltImageInfo": TiltImageInfo,
}


class ProjectArchiveWriter:
    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        self.array_index = 0

    def add_array(self, array: np.ndarray) -> dict[str, Any]:
        name = f"arrays/array_{self.array_index:08d}.npy"
        self.array_index += 1
        buffer = BytesIO()
        np.save(buffer, array, allow_pickle=False)
        self.archive.writestr(name, buffer.getvalue())
        return {"__niconavi_type__": "ndarray", "path": name}


class ProjectArchive:
    def __init__(
        self,
        computation_result: ComputationResult,
        ui_state: dict[str, Any] | None = None,
    ) -> None:
        self.computation_result = computation_result
        self.ui_state = ui_state or {}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _serialize_value(value: Any, writer: ProjectArchiveWriter) -> Any:
    value = _json_scalar(value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.ndarray):
        return writer.add_array(value)

    if isinstance(value, Path):
        return {"__niconavi_type__": "path", "value": str(value)}

    if isinstance(value, tuple):
        return {
            "__niconavi_type__": "tuple",
            "items": [_serialize_value(item, writer) for item in value],
        }

    if isinstance(value, list):
        return [_serialize_value(item, writer) for item in value]

    if isinstance(value, dict):
        return {
            "__niconavi_type__": "dict",
            "items": [
                [_serialize_value(key, writer), _serialize_value(item, writer)]
                for key, item in value.items()
            ],
        }

    class_name = value.__class__.__name__
    if class_name in PROJECT_CLASSES and hasattr(value, "__dict__"):
        return {
            "__niconavi_type__": "object",
            "class": class_name,
            "fields": {
                key: _serialize_value(item, writer)
                for key, item in value.__dict__.items()
            },
        }

    raise TypeError(f"Unsupported project value: {type(value).__name__}")


def _read_array(archive: zipfile.ZipFile, path: str) -> np.ndarray:
    with archive.open(path, "r") as file:
        return np.load(file, allow_pickle=False)


def _deserialize_value(value: Any, archive: zipfile.ZipFile) -> Any:
    if isinstance(value, list):
        return [_deserialize_value(item, archive) for item in value]

    if not isinstance(value, dict) or "__niconavi_type__" not in value:
        return value

    value_type = value["__niconavi_type__"]

    if value_type == "ndarray":
        return _read_array(archive, value["path"])

    if value_type == "path":
        return value["value"]

    if value_type == "tuple":
        return tuple(_deserialize_value(item, archive) for item in value["items"])

    if value_type == "dict":
        return {
            _deserialize_value(key, archive): _deserialize_value(item, archive)
            for key, item in value["items"]
        }

    if value_type == "object":
        class_name = value["class"]
        cls = PROJECT_CLASSES.get(class_name)
        if cls is None:
            raise ValueError(f"Unsupported project class: {class_name}")
        fields = {
            key: _deserialize_value(item, archive)
            for key, item in value["fields"].items()
        }
        accepted_params = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in accepted_params.values()
        )
        if not accepts_kwargs:
            unknown_keys = [key for key in fields if key not in accepted_params]
            for key in unknown_keys:
                logger.warning(
                    "Ignoring unknown field '%s' while loading %s from project file "
                    "(saved by an older/newer version of the app).",
                    key,
                    class_name,
                )
                del fields[key]
        return cls(**fields)

    raise ValueError(f"Unsupported project value type: {value_type}")


def export_project_bytes(
    result: ComputationResult,
    ui_state: dict[str, Any] | None = None,
) -> bytes:
    result = remove_heavy_objects(result)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest = {
            "format": "niconavi-project",
            "schema_version": SCHEMA_VERSION,
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        writer = ProjectArchiveWriter(archive)
        payload = _serialize_value(result, writer)
        archive.writestr("computation_result.json", json.dumps(payload))
        if ui_state is not None:
            ui_payload = _serialize_value(ui_state, writer)
            archive.writestr("ui_state.json", json.dumps(ui_payload))
    buffer.seek(0)
    return buffer.getvalue()


def load_project_archive(path: Path) -> ProjectArchive:
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != "niconavi-project":
            raise ValueError("Unsupported project file.")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported project schema version: {manifest.get('schema_version')}"
            )
        payload = json.loads(archive.read("computation_result.json").decode("utf-8"))
        result = _deserialize_value(payload, archive)
        if not isinstance(result, ComputationResult):
            raise ValueError("Project file does not contain a ComputationResult.")
        ui_state = {}
        if "ui_state.json" in archive.namelist():
            ui_payload = json.loads(archive.read("ui_state.json").decode("utf-8"))
            ui_state = _deserialize_value(ui_payload, archive)
            if not isinstance(ui_state, dict):
                ui_state = {}
        return ProjectArchive(result, ui_state)


def load_project(path: Path) -> ComputationResult:
    return load_project_archive(path).computation_result


def save_project_atomic(
    path: Path,
    result: ComputationResult,
    ui_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = export_project_bytes(result, ui_state=ui_state)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as file:
        tmp_path = Path(file.name)
        file.write(data)

    try:
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
