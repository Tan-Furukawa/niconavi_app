from typing import Callable, Optional, Literal
from niconavi.type import RawMapsNumList

RoseDiagramUsedInPlot = [
    "extinction_angle",
    "azimuth",
    "angle_deg",
]

RoseDiagramDisplayInPlot = ["extinction angle (Φ)", "azimuth", "shape preferred orientation"]

RoseDiagramUsedInPixel = [
    "extinction_angle",
    "azimuth",
]

RoseDiagramDisplayInPixel = ["extinction angle", "azimuth"]


RawMapsNumListUsedInPlot = [
    "extinction_angle",
    "max_retardation_map",
    "p45_R_map",
    "m45_R_map",
    "azimuth",
]


RawMapsNumListDisplayInPlot = [
    "extinction angle (Φ)",
    "retardation",
    "XPL+λ Φ+45",
    "XPL+λ Φ-45",
    "azimuth",
]


GrainNumListUsedInPlot = [
    "size", # mu**2/px**2
    "perimeter", # mu/px
    "inscribed_radius", # mu/px
    "equivalent_radius", # mu/px
    "extinction_angle",
    # "sd_extinction_angle",
    "R",
    # "min_retardation",
    "azimuth",
    "inclination",
    # "sd_azimuth",
    "V",
    # "S",
    "eccentricity",
    "angle_deg",
    "major_axis_length", # mu/px
    "minor_axis_length", # mu/px
]

GrainNumListDisplayInPlot = [
    "size", # mu**2/px**2
    "perimeter", # mu/px
    "inscribed radius", # mu/px
    "area-equivalent radius", # mu/px
    "extinction angle",
    # "cv(extinction_angle)",
    "retardation",
    # "min retardation",
    "azimuth",
    "inclination",
    # "cv(azimuth)",
    "brightness",
    # "Color (Saturation)",
    "eccentricity",
    "shape preferred orientation",
    "ellipse major axis length", # mu/px
    "ellipse minor axis length", # mu/px
]

GrainMeasurementDimension = Literal["length", "area"]

GRAIN_MEASUREMENT_DIMENSION_MAP: dict[str, GrainMeasurementDimension] = {
    "size": "area",
    "perimeter": "length",
    "inscribed_radius": "length",
    "equivalent_radius": "length",
    "major_axis_length": "length",
    "minor_axis_length": "length",
}


def get_grain_measurement_dimension(name: str) -> Optional[GrainMeasurementDimension]:
    return GRAIN_MEASUREMENT_DIMENSION_MAP.get(name)


def convert_A_to_B(
    AB_list: list[tuple[str, str]],
) -> tuple[Callable[[str], Optional[str]], Callable[[str], Optional[str]]]:
    """
    A, B のペアのリストを受け取り、以下を返す関数:
      fn1(a): A を与えると対応する B を返す関数 (未登録なら None)
      fn2(b): B を与えると対応する A を返す関数 (未登録なら None)

    ただし、A と B が全単射 (1対1対応) でない場合は ValueError を投げる。
    """
    dictAtoB = {}  # type: ignore
    dictBtoA = {}  # type: ignore

    for A, B in AB_list:
        # A->B の重複チェック (同じAが異なるBに対応していないか)
        if A in dictAtoB and dictAtoB[A] != B:
            raise ValueError(
                f"A '{A}' に複数の異なる B が割り当てられています: "
                f"{dictAtoB[A]} と {B}"
            )
        # B->A の重複チェック (同じBが異なるAに対応していないか)
        if B in dictBtoA and dictBtoA[B] != A:
            raise ValueError(
                f"B '{B}' に複数の異なる A が割り当てられています: "
                f"{dictBtoA[B]} と {A}"
            )

        dictAtoB[A] = B
        dictBtoA[B] = A

    # A->B, B->A それぞれの辞書のサイズ(キー数)が同じかどうかチェック
    # （全単射であれば一致するはず）
    if len(dictAtoB) != len(dictBtoA):
        raise ValueError("AとBのペアが全単射ではありません。")

    # fn1: A -> B (未登録なら None)
    def fn1(a: str) -> Optional[str]:
        return dictAtoB.get(a)

    # fn2: B -> A (未登録なら None)
    def fn2(b: str) -> Optional[str]:
        return dictBtoA.get(b)

    return fn1, fn2


inv_raw_maps_display, to_raw_map_display = convert_A_to_B(
    list(zip(RawMapsNumListDisplayInPlot, RawMapsNumListUsedInPlot))
)

inv_grain_display, to_grain_display = convert_A_to_B(
    list(zip(GrainNumListDisplayInPlot, GrainNumListUsedInPlot))
)

inv_rose_display, to_rose_display = convert_A_to_B(
    list(zip(RoseDiagramDisplayInPlot, RoseDiagramUsedInPlot))
)
