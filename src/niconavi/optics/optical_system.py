# %%
import numpy as np
from typing import Callable
import niconavi.optics.jones_matrix as jm
from niconavi.optics.tools import make_angle_retardation_estimation_function
from niconavi.tools.type import D1FloatArray
from niconavi.image.type import RGBPicture
from niconavi.image.image import get_color_element_by_index
from niconavi.optics.types import JonesVector

def get_quartz_system(
    dz: float,
    inclination: float,
    azimuth: float = 45 / 180 * np.pi,
    no: float = 1.544,
    ne: float = 1.553,
) -> Callable[[float], JonesVector]:

    def optical_system(wavelength: float) -> JonesVector:

        s = jm.identity()

        quartz = jm.uniaxial_crystal(
            inclination=inclination,
            azimuth=azimuth,
            wavelength=wavelength,
            dz=dz,
            no=no,
            ne=ne,
        )

        polarizer = jm.polarizer(direction="y")

        v = polarizer @ s @ quartz @ np.array([1, 0])
        return v

    return optical_system

def get_quartz_plus_full_wave_system(
    dz: float,
    inclination: float,
    azimuth: float = 45 / 180 * np.pi,
    no: float = 1.544,
    ne: float = 1.553,
) -> Callable[[float], JonesVector]:

    def optical_system(wavelength: float) -> JonesVector:

        r = jm.rotation(np.pi / 4)
        rm = jm.rotation(-np.pi / 4)
        s = rm @ jm.sensitive_color_plate(R=530, wavelength=wavelength) @ r

        quartz = jm.uniaxial_crystal(
            inclination=inclination,
            azimuth=azimuth,
            wavelength=wavelength,
            dz=dz,
            no=no,
            ne=ne,
        )

        polarizer = jm.polarizer(direction="y")

        v = polarizer @ s @ quartz @ np.array([1, 0])
        return v

    return optical_system


def get_full_wave_plate_system(
    thickness: float,
    R: float,
) -> Callable[[float], JonesVector]:

    def optical_system(wavelength: float) -> JonesVector:

        r = jm.rotation(np.pi / 4)
        rm = jm.rotation(-np.pi / 4)
        s = r @ jm.sensitive_color_plate(R, wavelength=wavelength) @ rm
        polarizer = jm.polarizer(direction="y")

        return polarizer @ s @ np.array([1, 0])

    return optical_system


def get_retardation_system_with_nd_filter(
    R: float,
    nd_filter: float,
    azimuth: float = np.pi / 4,
) -> Callable[[float], JonesVector]:
    def optical_system(wavelength: float) -> JonesVector:

        f = jm.nd_filter(nd_filter, nd_filter)
        r = jm.rotation(azimuth)
        rm = jm.rotation(-azimuth)
        s = r @ jm.sensitive_color_plate(R, wavelength=wavelength) @ f @ rm
        polarizer = jm.polarizer(direction="y")

        return polarizer @ s @ np.array([1, 0])

    return optical_system


def get_retardation_system(
    R: float,
    azimuth: float = np.pi / 4,
    alpha: float = 1.0,
) -> Callable[[float], JonesVector]:
    def optical_system(wavelength: float) -> JonesVector:

        r = jm.rotation(azimuth)
        rm = jm.rotation(-azimuth)
        s = r @ jm.sensitive_color_plate(R, wavelength=wavelength) @ rm
        a = jm.nd_filter(alpha, alpha)
        polarizer = jm.polarizer(direction="y")

        return polarizer @ a @ s @ np.array([1, 0])

    return optical_system


def get_polar_retardation_system(
    R: float,
    azimuth: float = np.pi / 4,
) -> Callable[[float], JonesVector]:
    def optical_system(wavelength: float) -> JonesVector:

        r = jm.rotation(azimuth)
        rm = jm.rotation(-azimuth)
        r4 = jm.rotation(np.pi / 4)
        rm4 = jm.rotation(-np.pi / 4)
        R0 = 570
        s = r4 @ jm.sensitive_color_plate(R0 / 4, wavelength=wavelength) @ rm4
        p = r @ jm.sensitive_color_plate(R, wavelength=wavelength) @ rm
        s_rev = rm4 @ jm.sensitive_color_plate(R0 / 4, wavelength=wavelength) @ r4
        polarizer = jm.polarizer(direction="y")

        return polarizer @ s_rev @ p @ s @ np.array([1, 0])

    return optical_system


def get_mineral_retardation_system(
    R: float,
    azimuth: float = np.pi / 4,
) -> Callable[[float], JonesVector]:
    def optical_system(wavelength: float) -> JonesVector:

        r = jm.rotation(azimuth)
        rm = jm.rotation(-azimuth)
        p = r @ jm.sensitive_color_plate(R, wavelength=wavelength) @ rm
        polarizer = jm.polarizer(direction="y")

        return polarizer @ p @ np.array([1, 0])

    return optical_system


def get_full_wave_plus_mineral_retardation_system(
    R: float,
    azimuth: float = np.pi / 4,
    R0: float = 530,
    alpha: float = 1,
) -> Callable[[float], JonesVector]:
    def optical_system(wavelength: float) -> JonesVector:

        nd_filter = jm.nd_filter(alpha, alpha)
        r = jm.rotation(-azimuth)
        rm = jm.rotation(azimuth)
        r4 = jm.rotation(np.pi / 4)
        rm4 = jm.rotation(-np.pi / 4)
        p = r @ jm.sensitive_color_plate(R, wavelength=wavelength) @ rm
        s = rm4 @ jm.sensitive_color_plate(R0, wavelength=wavelength) @ r4
        polarizer = jm.polarizer(direction="y")

        return polarizer @ s @ nd_filter @ p @ np.array([1, 0])

    return optical_system
