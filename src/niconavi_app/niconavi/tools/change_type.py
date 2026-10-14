import numpy as np
from typing import TypeVar

K = TypeVar("K", bound=np.generic)
S = TypeVar("S")


def as_float64(mat: np.ndarray[S, np.dtype[K]]) -> np.ndarray[S, np.dtype[np.float64]]:
    return mat.astype(np.float64)


def as_uint8(mat: np.ndarray[S, np.dtype[K]]) -> np.ndarray[S, np.dtype[np.uint8]]:
    return mat.astype(np.uint8)


T = TypeVar("T")


def as_two_element_tuple(tuple_input: tuple[T, ...]) -> tuple[T, T]:
    if len(tuple_input) == 1:
        raise ValueError("cannot convert (n,) tuple into two element tuple")
    return (tuple_input[0], tuple_input[1])
