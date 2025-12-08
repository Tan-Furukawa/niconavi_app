from typing import Any, Callable, TypeVar, Optional
from stores import Stores
from components.log_view import update_logs
import traceback


T = TypeVar("T", float, int)


def _message(base: str, prefix: Optional[str]) -> str:
    return f"{prefix} {base}".strip() if prefix else base


def validation_when_button_click(
    stores: Stores,
    val: Any,
    parse: Callable[[str], Optional[T]],
    base_message: str,
    prefix: Optional[str] = None,
) -> Optional[T]:
    try:
        parsed_value = parse(val)
        if parsed_value is None:
            update_logs(stores, (_message(base_message, prefix), "err"))
            return None
        return parsed_value
    except Exception:
        traceback.print_exc()
        update_logs(stores, (_message(base_message, prefix), "err"))
        return None


def parse_odd_int(v: str) -> Optional[int]:
    value = int(v)
    return value if value % 2 == 1 else None


def parse_larger_than_zero_int(v: str) -> Optional[int]:
    value = int(v)
    return value if value >= 0 else None


def parse_larger_than_zero_float(v: str) -> Optional[float]:
    value = float(v)
    return value if value >= 0 else None


def parse_percent(v: str) -> Optional[float]:
    value = float(v)
    return value if 0 <= value <= 100 else None


def validation_larger_than_0_float(
    stores: Stores, val: Any, msg: Optional[str] = None
) -> Optional[float]:
    return validation_when_button_click(
        stores,
        val,
        parse_larger_than_zero_float,
        "Enter a positive value.",
        msg,
    )


def validation_larger_than_0_int(
    stores: Stores, val: Any, msg: Optional[str] = None
) -> Optional[int]:
    return validation_when_button_click(
        stores,
        val,
        parse_larger_than_zero_int,
        "Enter a positive integer.",
        msg,
    )


def validation_odd_int(
    stores: Stores, val: Any, msg: Optional[str] = None
) -> Optional[int]:
    return validation_when_button_click(
        stores,
        val,
        parse_odd_int,
        "Enter an odd number.",
        msg,
    )


def validation_percent(
    stores: Stores, val: Any, msg: Optional[str] = None
) -> Optional[float]:
    return validation_when_button_click(
        stores,
        val,
        parse_percent,
        "Enter a number between 0 and 100.",
        msg,
    )
