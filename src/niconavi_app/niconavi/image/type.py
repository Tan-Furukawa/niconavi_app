# %% 
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TypeAlias, TypeVar, NewType

D1: TypeAlias = tuple[int]
D2: TypeAlias = tuple[int, int]
D3: TypeAlias = tuple[int, int, int]

D1RGB: TypeAlias = tuple[int, Literal[3]]
D1RGBA: TypeAlias = tuple[int, Literal[3]]
D2RGB: TypeAlias = tuple[int, int, Literal[3]]
D2RGBA: TypeAlias = tuple[int, int, Literal[3]]

Color = NewType("Color", NDArray[np.uint8])

# class Color(np.ndarray[tuple[Literal[3]], np.dtype[np.uint8]]): pass

# D1RGB_Array = NewType("D1RGB_Array", np.ndarray[D1RGB, np.dtype[np.uint8]])
D1RGB_Array = NewType("D1RGB_Array", NDArray[np.uint8])
# D1RGB_Array: TypeAlias = np.ndarray[D1RGB, np.dtype[np.uint8]]

D1RGBA_Array = NewType("D1RGBA_Array", np.ndarray[D1RGBA, np.dtype[np.uint8]])

HSVPicture = NewType("HSVPicture", np.ndarray[D2RGB, np.dtype[np.uint8]])

# RGBPicture = NewType("RGBPicture", np.ndarray[D2RGB, np.dtype[np.uint8]])
RGBPicture = NewType("RGBPicture", NDArray[np.uint8])
# RGBPicture: TypeAlias = np.ndarray[D2RGB, np.dtype[np.uint8]]

BinaryPicture = NewType("BinaryPicture", np.ndarray[D2, np.dtype[np.bool_]])

RGBAPicture = NewType("RGBAPicture", np.ndarray[D2RGBA, np.dtype[np.uint8]])

MonoColorPicture = NewType("MonoColorPicture", np.ndarray[D2, np.dtype[np.uint8]])

# MonoColorPicturesAranged: TypeAlias = np.ndarray[D3, np.dtype[np.uint8]]

_CommonPictureType = TypeVar("_CommonPictureType", RGBPicture, MonoColorPicture)

_PictureType = TypeVar("_PictureType", RGBPicture, HSVPicture, MonoColorPicture)


# k: D1RGB_Array = np.array([])

# kk = f(k)
