import numpy as np
from typing import Literal, NewType, Optional, TypedDict
from numpy.typing import NDArray

RGBImage = NewType("RGBImage", NDArray[np.uint8])
BinaryImage = NewType("BinaryImage", NDArray[np.uint8])
GrayScaleImage = NewType("GrayScaleImage", NDArray[np.uint8])
HessianImage = NewType("HessianImage", NDArray[np.float64])
FloatMatrix = NewType("FloatMatrix", NDArray[np.float64])
FloatMatrix_0to1 = NewType("FloatMatrix_0to1", NDArray[np.float64])
FloatMatrix_HxWx2x2 = NewType("FloatMatrix_HxWx2x2", NDArray[np.float64])
IntMatrix = NewType("IntMatrix", NDArray[np.int32])
UintMatrix_3x3 = NewType("UintMatrix_3x3", NDArray[np.uint8])
BoolMatrix = NewType("BoolMatrix", NDArray[np.bool_])
IntVector = NewType("IntVector", NDArray[np.int32])
FloatVector = NewType("FloatVector", NDArray[np.float64])
ProbabilityMatrix = NewType("ProbabilityMatrix", NDArray[np.float64])
IndexMap = NewType("IndexMap", NDArray[np.int32])


class SpinelClassificationParams(TypedDict):
    min_clip: int
    spinel_brightness_threshold: int
    spinel_min_size: int
    morph_close_size_after_sobel: int
    morph_close_size_after_find_spinel: int
    min_rotrect_aspect: float
    min_circularity: float
    min_solidity: float
    max_complexity: float
    method: Literal["open only", "cross only", "both"]
    remove_index: list[int]


class LoadedData(TypedDict):
    open: str
    cross: str
    o_shift_h: int
    o_shift_w: int
    th_flood_mask: int
    ignore_flood: Optional[list[int]]
    spinel_classification_params: SpinelClassificationParams
    olivin_th: float
    olivin_boundary_th: float
    olivin_morph_expand: int


class SegmentedData(TypedDict):
    imo: GrayScaleImage
    imc: GrayScaleImage
    mask: BinaryImage
    spinel: BinaryImage
    olivin: BinaryImage
    olivin_filled: BinaryImage
    frames: FloatMatrix
