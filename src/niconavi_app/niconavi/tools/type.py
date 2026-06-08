import numpy as np
from typing import TypeAlias, TypeGuard, TypeVar, NewType
from numpy.typing import NDArray

D1: TypeAlias = tuple[int]
D2: TypeAlias = tuple[int, int]
D3: TypeAlias = tuple[int, int, int]

# D1FloatArray: TypeAlias = np.ndarray[D1, np.dtype[np.float64]]
# D2FloatArray: TypeAlias = np.ndarray[D2, np.dtype[np.float64]]
# D1IntArray: TypeAlias = np.ndarray[D1, np.dtype[np.int_]]
# D2IntArray: TypeAlias = np.ndarray[D2, np.dtype[np.int_]]
# D1BoolArray: TypeAlias = np.ndarray[D1, np.dtype[np.bool_]]
# D2BoolArray: TypeAlias = np.ndarray[D2, np.dtype[np.bool_]]
# D3BoolArray: TypeAlias = np.ndarray[D3, np.dtype[np.bool_]]
# D2UintArray: TypeAlias = np.ndarray[D2, np.dtype[np.uint8]]

# D1FloatArray = NewType("D1FloatArray", np.ndarray[D1, np.dtype[np.float64]])
# D2FloatArray = NewType("D2FloatArray", np.ndarray[D2, np.dtype[np.float64]])
# D1IntArray = NewType("D1IntArray", np.ndarray[D1, np.dtype[np.int_]])
# D2IntArray = NewType("D2IntArray", np.ndarray[D2, np.dtype[np.int_]])
# D1BoolArray = NewType("D1BoolArray", np.ndarray[D1, np.dtype[np.bool_]])
# D2BoolArray = NewType("D2BoolArray", np.ndarray[D2, np.dtype[np.bool_]])
# D3BoolArray = NewType("D3BoolArray", np.ndarray[D3, np.dtype[np.bool_]])
# D2UintArray = NewType("D2UintArray", np.ndarray[D2, np.dtype[np.uint8]])

D1FloatArray = NewType("D1FloatArray", NDArray[np.float64])
D2FloatArray = NewType("D2FloatArray", NDArray[np.float64])
D3FloatArray = NewType("D3FloatArray", NDArray[np.float64])
D1IntArray = NewType("D1IntArray", NDArray[np.int32])
D2IntArray = NewType("D2IntArray", NDArray[np.int32])
D1BoolArray = NewType("D1BoolArray", NDArray[np.bool_])
D2BoolArray = NewType("D2BoolArray", NDArray[np.bool_])
D3BoolArray = NewType("D3BoolArray", NDArray[np.bool_])
D2UintArray = NewType("D2UintArray", NDArray[np.uint8])


T = TypeVar("T")


def is_not_none(x: float | None) -> TypeGuard[float]:
    return x is not None


def is_not_tuple_none(
    x: tuple[float | None, float | None]
) -> TypeGuard[tuple[float, float]]:
    return x[0] is not None and x[1] is not None


def is_not_None_list(arr: list[T | None]) -> TypeGuard[list[T]]:
    return bool(np.all(arr is not None))
