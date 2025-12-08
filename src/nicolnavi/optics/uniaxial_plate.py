# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, cast, TypedDict, Optional
from numpy.typing import NDArray
from niconavi.optics.tools import make_angle_retardation_estimation_function
import niconavi.optics.jones_matrix as jm
import niconavi.optics.optical_system as osys
from niconavi.optics.types import (
    JonesVector,
    is_WavelengthVector,
)
from niconavi.tools.type import D2FloatArray, D1FloatArray
from niconavi.image.type import Color, RGBPicture

import matplotlib.pyplot as plt
from colour import (
    SDS_ILLUMINANTS,
    MSDS_CMFS,
    SpectralDistribution,
    sd_to_XYZ,
    XYZ_to_sRGB,
)
from niconavi.type import ColorChartInfo


def get_jones_vector_s0(v: JonesVector) -> float:
    ss = np.real(np.conjugate(v[0]) * v[0] + np.conjugate(v[1]) * v[1])
    return cast(float, ss)


class SpectralDistributionSummary(TypedDict):
    wavelengths: D1FloatArray
    spectrum: D1FloatArray
    rgb: Color


def get_spectral_distribution(
    optical_system: Callable[[float], JonesVector],
    wavelengths: D1FloatArray = D1FloatArray(
        np.arange(380, 781, 5, dtype=np.float64)
    ),  # from 380nm to 780nm wavelength
) -> SpectralDistributionSummary:

    # 0 th Stock parameter
    S0 = np.zeros_like(wavelengths)
    for i, l in enumerate(wavelengths):

        v = optical_system(l)
        S0[i] = get_jones_vector_s0(v)

    rgb = get_color_from_spectral_distribution(wavelengths, S0)

    return {
        "wavelengths": wavelengths,
        "spectrum": S0,
        "rgb": rgb,
    }


def plot_color_chart(
    color_chart_info: ColorChartInfo,
    xlab: str = "x",
    ylab: str = "y",
    size: tuple[float, float] = (5, 5),
) -> None:

    plt.imshow(
        color_chart_info["color_chart"],
        origin="lower",
        extent=(
            color_chart_info["h"].min(),
            color_chart_info["h"].max(),
            color_chart_info["w"].min(),
            color_chart_info["w"].max(),
        ),
        aspect="auto",
    )
    plt.gcf().set_size_inches(size[0], size[1])
    plt.xlabel(xlab)
    plt.ylabel(ylab)


class JonesVectorPlot:
    def __init__(self, jones_vector: NDArray[np.complex64]) -> None:
        self.jones_vector = jones_vector
        pass

    def plot_jones_vector(self) -> None:
        theta = np.linspace(0, 2 * np.pi, dtype=np.complex64, num=100)

        polarization_state_x = np.real(self.jones_vector[0] * np.exp(1j * theta))
        polarization_state_y = np.real(self.jones_vector[1] * np.exp(1j * theta))

        plt.plot(polarization_state_x, polarization_state_y)
        rmax = np.max([polarization_state_x, polarization_state_y])
        rmin = np.min([polarization_state_x, polarization_state_y])

        plt.gcf().set_size_inches(5, 5)
        plt.xlim((rmax, rmin))
        plt.ylim((rmax, rmin))


def get_quartz_with_sensitive_color_plate_system(
    dz: float,
    inclination: float,
    azimuth: float = np.pi * 1 / 4,
) -> Callable[[float], JonesVector]:

    def optical_system(wavelength: float) -> JonesVector:
        r = jm.rotation(np.pi / 4)
        rm = jm.rotation(-np.pi / 4)
        s = r @ jm.sensitive_color_plate(wavelength=wavelength) @ rm

        # s = jm.identity()

        quartz = jm.uniaxial_crystal(
            inclination=inclination, azimuth=azimuth, wavelength=wavelength, dz=dz
        )

        polarizer = jm.polarizer(direction="y")

        v = polarizer @ s @ quartz @ np.array([1, 0])
        return v

    return optical_system


def get_color_from_spectral_distribution(
    wavelengths: D1FloatArray,
    reflectance: D1FloatArray,
    illuminant: SpectralDistribution = SDS_ILLUMINANTS["D65"],
) -> Color:

    if is_WavelengthVector(wavelengths):

        cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

        sd_sample = SpectralDistribution(data=dict(zip(wavelengths, reflectance)))

        # XYZ表色系への変換
        XYZ = sd_to_XYZ(sd_sample, cmfs=cmfs, illuminant=illuminant)

        # sRGBへの変換
        RGB = XYZ_to_sRGB(XYZ / 100)
        RGB = np.clip(RGB * 256, 0, 255).astype(np.uint8)
        return cast(Color, RGB)
    else:
        raise TypeError("invalid type wavelengths")


def calc_color_chart(
    w: D1FloatArray,
    h: D1FloatArray,
    xy_to_color: Callable[[float, float], Color],
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ColorChartInfo:

    color_chart = np.zeros((len(h), len(w), 3), dtype=np.uint8)

    for i, xi in tqdm(enumerate(w), total=len(w), desc="make retardation color chart"):
        progress_callback((i + 1) / len(w))
        for j, yj in enumerate(h):
            color_chart[j, i] = xy_to_color(xi, yj)

    return {
        "what_is_h": None,
        "what_is_w": None,
        "h": h,
        "w": w,
        "color_chart": cast(RGBPicture, color_chart),
    }


def get_retardation_color_chart_with_nd_filter(
    start: float = 0,
    end: float = 1000,
    num: int = 400,
    nd_num: int = 100,
    nd_filter_min: float = 0,
    nd_filter_max: float = 1,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> ColorChartInfo:

    x = cast(D1FloatArray, np.linspace(start, end, num=num))
    y = cast(D1FloatArray, np.linspace(nd_filter_min, nd_filter_max, num=nd_num))

    img = calc_color_chart(
        x,
        y,
        lambda x, y: get_spectral_distribution(
            osys.get_retardation_system_with_nd_filter(R=x, nd_filter=y)
        )["rgb"],
        progress_callback,
    )
    return ColorChartInfo(
        what_is_h="retardation",
        what_is_w="nd_filter",
        color_chart=img["color_chart"],
        w=cast(D1FloatArray, x),
        h=cast(D1FloatArray, y),
    )


def get_retardation_color_chart(
    start: float = 0,
    end: float = 1000,
    num: int = 400,
    alpha: float = 1,
    progress_callback: Callable[[float | None], None] = lambda p: None,
) -> tuple[RGBPicture, D1FloatArray]:

    x = cast(D1FloatArray, np.linspace(start, end, num=num))
    y = cast(D1FloatArray, np.linspace(0, 0, num=1))

    img = calc_color_chart(
        x,
        y,
        lambda x, y: get_spectral_distribution(
            osys.get_retardation_system(R=x, alpha=alpha)
        )["rgb"],
        progress_callback,
    )
    return img["color_chart"], cast(D1FloatArray, x)


def make_uniaxial_color_chart(
    R_inclination_to_color: Callable[[float, float], Color],
    num_inc: int,
    num_azimuth: int,
    no: float = 1.544,
    ne: float = 1.553,
    thickness: float = 0.03,
    max_azimuth: float = 360,
) -> ColorChartInfo:
    theta_to_R, R_to_theta = make_angle_retardation_estimation_function(
        no=no, ne=ne, thickness=thickness
    )

    inclination = D1FloatArray(np.linspace(0, np.pi / 2, num_inc))
    R = D1FloatArray(theta_to_R(inclination))
    azimuth = D1FloatArray(np.linspace(0, np.radians(max_azimuth), num_azimuth))

    col_chart = calc_color_chart(
        azimuth,
        R,
        R_inclination_to_color,
    )

    col_chart["h"] = inclination
    col_chart["w"] = azimuth

    return col_chart


if __name__ == "__main__":

    from niconavi.optics.optical_system import (
        get_full_wave_plus_mineral_retardation_system,
    )

    chart = make_uniaxial_color_chart(
        lambda w, h: get_spectral_distribution(
            get_full_wave_plus_mineral_retardation_system(R=h, azimuth=w, alpha=1)
        )["rgb"],
        thickness=0.04,
        num_inc=20,
        num_azimuth=30,
        max_azimuth=180,
        no=1.544,
        ne=1.553,
    )

    plt.imshow(chart["color_chart"])

    # %%
    r = chart["color_chart"][:, :, 0]
    plt.show()
    plt.imshow(r)
    plt.show()
