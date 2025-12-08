import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

_TTL_SECONDS = 600.0


@dataclass
class DownloadItem:
    content: bytes
    filename: str
    mime_type: str
    expires_at: float


_registry: Dict[str, DownloadItem] = {}
_lock = threading.Lock()


def _cleanup_locked(current_time: float) -> None:
    expired_tokens = [
        token for token, item in _registry.items() if item.expires_at <= current_time
    ]
    for token in expired_tokens:
        del _registry[token]


def register_download(content: bytes, filename: str, mime_type: str) -> str:
    token = uuid.uuid4().hex
    expires_at = time.time() + _TTL_SECONDS
    with _lock:
        _cleanup_locked(time.time())
        _registry[token] = DownloadItem(
            content=content,
            filename=filename,
            mime_type=mime_type,
            expires_at=expires_at,
        )
    return token


def pop_download(token: str) -> Optional[DownloadItem]:
    with _lock:
        item = _registry.pop(token, None)
    return item
