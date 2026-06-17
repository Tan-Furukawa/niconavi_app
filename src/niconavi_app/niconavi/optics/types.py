import numpy as np
from logging import getLogger
from numpy.typing import NDArray
from typing import (
    NewType,
    Literal,
    TypeGuard,
    TypedDict,
)


logger = getLogger("niconavi").getChild(__name__)


class DictColor(TypedDict):
    R: int
    G: int
    B: int


JonesMatrix = NewType(
    "JonesMatrix", np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]]
)

JonesVector = NewType(
    "JonesVector", np.ndarray[tuple[Literal[2]], np.dtype[np.complex128]]
)

WavelengthVector = NewType(
    "WavelengthVector", np.ndarray[tuple[int], np.dtype[np.int_]]
)

ReflectanceVector = NewType(
    "ReflectanceVector", np.ndarray[tuple[int], np.dtype[np.float64]]
)


def is_WavelengthVector(vec: NDArray) -> TypeGuard[WavelengthVector]:
    if vec.dtype == np.dtype(np.float64):
        if np.ndim(vec) == 1:
            dvec = np.diff(vec)  # from 380nm to 780nm wavelength
            if np.all(dvec == 1):
                return True
            elif np.all(dvec == 5):
                if vec[0] % 5 == 0:
                    return True
                else:
                    return False
            else:
                return False
        else:
            logger.debug("Wavelength vector check failed: expected 1D array, got ndim=%s.", np.ndim(vec))
            return False
    else:
        logger.debug("Wavelength vector check failed: expected float64 dtype, got %s.", vec.dtype)
        return False
