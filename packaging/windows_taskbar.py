"""Give the Flet client the host application's Windows taskbar identity.

Flet 0.28 runs the visible window in its bundled ``flet.exe`` process.  Without
explicit shell properties Windows labels that window "Flet description", and
pinning or relaunching it starts the bare (and therefore blank) Flet client.

This is a small compatibility backport of the taskbar handling added upstream
to newer Flet releases.  It can be removed after the application migrates away
from Flet 0.28.
"""

# Adapted from flet_desktop/win_taskbar.py in the Flet project.
# Copyright Appveyor Systems Inc.; licensed under the Apache License 2.0.

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
import time
from ctypes import wintypes


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, value: str | None = None) -> None:
        super().__init__()
        if value:
            _ole32.CLSIDFromString(value, ctypes.byref(self))


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", ctypes.c_uint32)]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_uint16),
        ("reserved1", ctypes.c_uint16),
        ("reserved2", ctypes.c_uint16),
        ("reserved3", ctypes.c_uint16),
        ("data", ctypes.c_void_p),
        ("data2", ctypes.c_void_p),
    ]


_ole32 = ctypes.OleDLL("ole32")
_shell32 = ctypes.OleDLL("shell32")
_shlwapi = ctypes.OleDLL("shlwapi")
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_ole32.CLSIDFromString.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p]
_ole32.PropVariantClear.argtypes = [ctypes.c_void_p]
_shlwapi.SHStrDupW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]

_PROPERTY_STORE_SET_VALUE = ctypes.WINFUNCTYPE(
    ctypes.HRESULT,
    ctypes.c_void_p,
    ctypes.POINTER(PROPERTYKEY),
    ctypes.POINTER(PROPVARIANT),
)
_PROPERTY_STORE_COMMIT = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)
_PROPERTY_STORE_RELEASE = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)


class _PropertyStoreVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", ctypes.c_void_p),
        ("AddRef", ctypes.c_void_p),
        ("Release", _PROPERTY_STORE_RELEASE),
        ("GetCount", ctypes.c_void_p),
        ("GetAt", ctypes.c_void_p),
        ("GetValue", ctypes.c_void_p),
        ("SetValue", _PROPERTY_STORE_SET_VALUE),
        ("Commit", _PROPERTY_STORE_COMMIT),
    ]


class _PropertyStore(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_PropertyStoreVTable))]


_shell32.SHGetPropertyStoreForWindow.argtypes = [
    wintypes.HWND,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND,
    ctypes.POINTER(wintypes.DWORD),
]
_user32.GetClassNameW.argtypes = [wintypes.HWND, ctypes.c_wchar_p, ctypes.c_int]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

_ENUM_WINDOWS_CALLBACK = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
)
_IID_PROPERTY_STORE = "{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}"
_APP_USER_MODEL_FMTID = "{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}"
_PID_RELAUNCH_COMMAND = 2
_PID_RELAUNCH_ICON = 3
_PID_RELAUNCH_DISPLAY_NAME = 4
_PID_APP_USER_MODEL_ID = 5
_VT_LPWSTR = 31
_GET_WINDOW_OWNER = 4
_SYNCHRONIZE = 0x00100000
_WAIT_TIMEOUT = 0x00000102
_FLUTTER_WINDOW_CLASS = "FLUTTER_RUNNER_WIN32_WINDOW"


def _find_flutter_window(process_id: int) -> int | None:
    found: list[int] = []

    def callback(window: int, _: int) -> bool:
        owner_process_id = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(window, ctypes.byref(owner_process_id))
        if owner_process_id.value != process_id:
            return True
        if _user32.GetWindow(window, _GET_WINDOW_OWNER):
            return True
        class_name = ctypes.create_unicode_buffer(64)
        _user32.GetClassNameW(window, class_name, len(class_name))
        if class_name.value == _FLUTTER_WINDOW_CLASS:
            found.append(window)
        return True

    _user32.EnumWindows(_ENUM_WINDOWS_CALLBACK(callback), 0)
    return found[0] if found else None


def _string_variant(value: str) -> PROPVARIANT:
    buffer = ctypes.c_void_p()
    _shlwapi.SHStrDupW(value, ctypes.byref(buffer))
    variant = PROPVARIANT()
    variant.vt = _VT_LPWSTR
    variant.data = buffer
    return variant


def _set_window_properties(window: int, properties: list[tuple[int, str]]) -> None:
    interface_id = GUID(_IID_PROPERTY_STORE)
    store_pointer = ctypes.c_void_p()
    _shell32.SHGetPropertyStoreForWindow(
        window, ctypes.byref(interface_id), ctypes.byref(store_pointer)
    )
    store = ctypes.cast(store_pointer, ctypes.POINTER(_PropertyStore))
    vtable = store.contents.lpVtbl.contents
    try:
        for property_id, value in properties:
            key = PROPERTYKEY(GUID(_APP_USER_MODEL_FMTID), property_id)
            variant = _string_variant(value)
            try:
                vtable.SetValue(
                    store_pointer, ctypes.byref(key), ctypes.byref(variant)
                )
            finally:
                _ole32.PropVariantClear(ctypes.byref(variant))
        vtable.Commit(store_pointer)
    finally:
        vtable.Release(store_pointer)


def apply_taskbar_identity_async(
    process_id: int,
    *,
    app_id: str,
    relaunch_command: str,
    display_name: str,
    icon_path: str,
) -> None:
    """Apply NicoNavi identity to a newly started Flet client window."""

    if os.name != "nt":
        return

    properties = [
        (_PID_APP_USER_MODEL_ID, app_id),
        (_PID_RELAUNCH_COMMAND, relaunch_command),
        (_PID_RELAUNCH_DISPLAY_NAME, display_name),
        (_PID_RELAUNCH_ICON, f"{icon_path},0"),
    ]

    def worker() -> None:
        with contextlib.suppress(OSError):
            _ole32.CoInitialize(None)
        process_handle = _kernel32.OpenProcess(
            _SYNCHRONIZE, False, process_id
        )
        try:
            while True:
                window = _find_flutter_window(process_id)
                if window:
                    _set_window_properties(window, properties)
                    return
                if (
                    not process_handle
                    or _kernel32.WaitForSingleObject(process_handle, 0)
                    != _WAIT_TIMEOUT
                ):
                    return
                time.sleep(0.2)
        finally:
            if process_handle:
                _kernel32.CloseHandle(process_handle)

    threading.Thread(target=worker, name="niconavi-taskbar", daemon=True).start()


def patch_flet_desktop(
    *, app_id: str, relaunch_command: str, display_name: str, icon_path: str
) -> None:
    """Patch Flet 0.28 so each desktop client receives taskbar properties."""

    import flet_desktop

    original = flet_desktop.open_flet_view_async

    async def open_flet_view_async(page_url, assets_dir, hidden):
        process, pid_file = await original(page_url, assets_dir, hidden)
        apply_taskbar_identity_async(
            process.pid,
            app_id=app_id,
            relaunch_command=relaunch_command,
            display_name=display_name,
            icon_path=icon_path,
        )
        return process, pid_file

    flet_desktop.open_flet_view_async = open_flet_view_async
