import numpy as np
from numpy.typing import NDArray
from typing import TypeGuard, cast
from niconavi.image.type import RGBPicture, MonoColorPicture, RGBAPicture, BinaryPicture
from cv2.typing import MatLike


def is_RGBAPicture(pic: NDArray[np.uint8]) -> TypeGuard[RGBAPicture]:
    s = pic.shape
    if (len(s) == 3) and (s[2] == 4):
        return True
    else:
        return False


def is_RGBPicture(pic: NDArray[np.uint8]) -> TypeGuard[RGBPicture]:
    s = pic.shape
    if (len(s) == 3) and (s[2] == 3):
        return True
    else:
        return False


def is_MonoColorPicture(pic: NDArray[np.uint8]) -> TypeGuard[MonoColorPicture]:
    s = pic.shape
    if len(s) == 2:
        return True
    else:
        return False

def as_BinaryPicture(arr: NDArray | MatLike) -> BinaryPicture:
    if np.ndim(arr) != 2:
        raise ValueError("invalid dimension of arr.")
    return cast(BinaryPicture, arr != 0)