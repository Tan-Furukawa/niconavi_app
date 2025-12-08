import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, cast
from numpy.typing import NDArray

from niconavi.optics.types import JonesMatrix

# class JonesMatrix:
# def __init__(self) -> None:
# pass


def identity() -> JonesMatrix:
    return cast(JonesMatrix, np.array([[1.0, 0], [0, 1.0]], dtype=np.complex128))


def nd_filter(p: float, q: float) -> JonesMatrix:
    return cast(JonesMatrix, np.array([[p, 0], [0, q]], dtype=np.complex64))


def polarizer(direction: Literal["x", "y"] = "x") -> JonesMatrix:

    match direction:
        case "x":
            return cast(
                JonesMatrix, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)
            )
        case "y":
            return cast(
                JonesMatrix, np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex64)
            )
        case _:
            raise TypeError("plate is x or y")


def rotation(theta: float) -> JonesMatrix:
    return cast(
        JonesMatrix,
        np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.complex64,
        ),
    )


def sensitive_color_plate(R: float = 530, wavelength: float = 147.3 * 4) -> JonesMatrix:
    return cast(
        JonesMatrix,
        np.array(
            [[np.exp(1j * 2 * np.pi * R / wavelength), 0], [0, 1]],
            dtype=np.complex64,
        ),
    )


def uniaxial_crystal(
    n1: float = 1.0,
    n2: float = 1.0,
    no: float = 1.544,
    ne: float = 1.553,
    dz: float = 0.03,  # mm
    wavelength: float = 600,  # nm
    azimuth: float = 30 * np.pi / 180,
    inclination: float = 30 * np.pi / 180,
) -> JonesMatrix:

    dzm = dz * 10**-3
    lamnm = wavelength * 10**-9

    gamma = np.cos(inclination)

    ee = ne**2
    eo = no**2

    ke = ne * no / np.sqrt(eo + gamma**2 * (ee - eo)) * 2 * np.pi / lamnm
    ko = no * 2 * np.pi / lamnm
    k1 = n1 * 2 * np.pi / lamnm
    k2 = n2 * 2 * np.pi / lamnm

    to = (
        2
        * k1
        * ko
        / (
            ko * (k1 + k2) * np.cos(ko * dzm)
            - 1j * (ko**2 + k1 * k2) * np.sin(ko * dzm)
        )
    )
    te = (
        2
        * k1
        * ke
        / (
            ke * (k1 + k2) * np.cos(ke * dzm)
            - 1j * (ke**2 + k1 * k2) * np.sin(ke * dzm)
        )
    )

    res = np.array(
        [
            [
                to * np.cos(azimuth) ** 2 + te * np.sin(azimuth) ** 2,
                (to - te) * np.cos(azimuth) * np.sin(azimuth),
            ],
            [
                (to - te) * np.cos(azimuth) * np.sin(azimuth),
                to * np.sin(azimuth) ** 2 + te * np.cos(azimuth) ** 2,
            ],
        ]
    )
    return cast(JonesMatrix, res)
