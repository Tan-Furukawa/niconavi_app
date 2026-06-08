from typing import Optional, TypeVar
# check the type is not None and if not, return the type
T = TypeVar("T")
def check_not_None(value: Optional[T]) -> T:
    if value is None:
        raise ValueError("Value cannot be None")
    return value
