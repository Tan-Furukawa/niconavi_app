# %%
import os
import yaml  # type: ignore
from typing import Dict, Any, Tuple, List, Literal, TypedDict
import numpy as np
from numpy.typing import NDArray
from niconavi.tools.read_data import get_first_frame_from_video

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from optics.color import show_color
from niconavi.image.type import Color
from niconavi.optics.types import DictColor

from niconavi.optics.uniaxial_plate import get_spectral_distribution
from niconavi.optics.optical_system import get_quartz_system

from tools.draw import plot_grid, draw_rectangle, extract_rectangle_region
from optics.color import (
    get_average_color,
    get_dominant_color,
    convert_01_vector_to_uint_color,
    convert_float_vector_to_uint_color,
    convert_color_vector_to_rgb_dict,
    convert_rgb_dict_to_color_vector,
)
from tools.file_operation import (
    extract_numbers_from_filename,
    make_dir,
    read_parameter_yaml,
)
from niconavi.statistics.statistics import multiple_polynomial_regression


class Center(TypedDict):
    x: int
    y: int


class CalibrationParameters(TypedDict):
    center: Center
    file_name: str
    radius: float
    inclination: float
    thickness: float
    true_color: DictColor
    dominant_color: DictColor
    mean_color: DictColor


def get_quartz_retardation_color(
    thickness: float, inclination: float
) -> NDArray[np.uint8]:

    return get_spectral_distribution(get_quartz_system(thickness * 0.001, inclination))[
        "rgb"
    ]

    # p = CalcPolarization(
    #     optical_system=get_quartz_system(thickness * 0.001, inclination)
    # )
    # _, rgb = p.get_spectral_distribution()
    # return rgb


def make_new_parameters(
    file_name: str,
    center: Tuple[int, int],
    radius: float,
    root_path: str,
) -> CalibrationParameters:
    """
    Adds calibration information to the parameters dictionary and saves it to the YAML file.

    Parameters:
        file_name (str): The name of the calibration file.
        center (Tuple[int, int]): The (x, y) coordinates of the center point.
        radius (float): The radius used for extracting the rectangle region.

    Returns:
        None
    """
    img = get_first_frame_from_video(os.path.join(root_path, file_name))
    img = extract_rectangle_region(img, center, radius)
    dominant_color = get_dominant_color(img)
    mean_color = get_average_color(img)

    inclination, thickness = extract_numbers_from_filename(file_name)

    true_color = convert_01_vector_to_uint_color(
        get_quartz_retardation_color(thickness, inclination / 180 * np.pi)
    )
    return {
        "center": {"x": center[0], "y": center[1]},
        "file_name": file_name,
        "radius": radius,
        "inclination": inclination,
        "thickness": thickness,
        "true_color": convert_color_vector_to_rgb_dict(true_color),
        "dominant_color": convert_color_vector_to_rgb_dict(dominant_color),
        "mean_color": convert_color_vector_to_rgb_dict(mean_color),
    }


def save_parameter_as_yaml(
    parameter_dir_path: str,
    parameters: CalibrationParameters,
    file_name: str = "_parameter.yaml",
) -> None:
    """
    Saves the parameters dictionary to the parameter YAML file.

    Parameters:
        file_name (str): The name of the YAML file to save.

    Returns:
        None
    """
    file_path = os.path.join(parameter_dir_path, file_name)
    with open(file_path, "w") as f:
        yaml.safe_dump(parameters, f, default_flow_style=False, allow_unicode=True)


def read_parameter_from_yaml_file(
    parameter_dir_path: str,
) -> list[CalibrationParameters]:
    return read_parameter_yaml(parameter_dir_path)


def add_new_parameter_to_parameter_list(
    parameter: CalibrationParameters, parameter_list: list[CalibrationParameters]
) -> list[CalibrationParameters]:
    return parameter_list + [parameter]


def record_parameter_and_update(
    parameter_dir_path: str,
    parameter: CalibrationParameters,
    parameter_list: list[CalibrationParameters],
) -> list[CalibrationParameters]:
    param_list = read_parameter_from_yaml_file(parameter_dir_path)
    save_parameter_as_yaml(parameter_dir_path, parameter)
    return add_new_parameter_to_parameter_list(parameter, param_list)


class GetCalibrationColorFromImg:
    def __init__(
        self, root_path: str = "calibration", parameter_dirname: str = "parameters"
    ) -> None:
        """
        Initializes the GetCalibrationColorFromImg class.

        Parameters:
            root_path (str): The root directory path where calibration files are stored.
            parameter_dirname (str): The name of the directory where parameter files are stored.

        Returns:
            None
        """
        self.root_path = root_path
        self.parameter_dir_path = os.path.join(self.root_path, parameter_dirname)
        self.parameters: Dict[str, Any] = read_parameter_yaml(self.parameter_dir_path)
        make_dir(self.root_path, parameter_dirname)

    def update_parameters(self) -> None:
        """
        Updates the parameters by reading the parameter YAML file.
        """
        self.parameters = read_parameter_yaml(self.parameter_dir_path)

        save_parameter_as_yaml(self.parameter_dir_path, self.parameters)

    def compare_estimated_and_true_color(self, file_name: str) -> None:
        if file_name not in self.parameters:
            raise ValueError(f"File name '{file_name}' not found in parameters.")

        true_color = self.parameters[file_name]["true_color"]
        dominant_color = self.parameters[file_name]["dominant_color"]
        mean_color = self.parameters[file_name]["mean_color"]

        true_color = convert_rgb_dict_to_color_vector(true_color)
        dominant_color = convert_rgb_dict_to_color_vector(dominant_color)
        mean_color = convert_rgb_dict_to_color_vector(mean_color)

        print("true_color")
        show_color(true_color)
        print("dominant_color")
        show_color(dominant_color)
        print("mean_color")
        show_color(mean_color)

    def plot_calibrate_range(self, file_name: str) -> Tuple[Figure, Axes]:
        """
        Plots the calibration range by drawing a rectangle on the image.

        Parameters:
            file_name (str): The name of the calibration file.

        Returns:
            Tuple[Figure, Axes]: The matplotlib Figure and Axes objects.

        Raises:
            ValueError: If the file_name is not found in parameters.
        """
        if file_name not in self.parameters:
            raise ValueError(f"File name '{file_name}' not found in parameters.")

        center_x = self.parameters[file_name]["center"]["x"]
        center_y = self.parameters[file_name]["center"]["y"]
        radius = self.parameters[file_name]["radius"]
        center = (center_x, center_y)

        img = get_first_frame_from_video(os.path.join(self.root_path, file_name))
        img = draw_rectangle(img, center, radius)
        return plot_grid(img)

    def get_actual_color_and_true_color(
        self, color_type: Literal["dominant_color", "mean_color"]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Extracts specific fields from the parameters and returns a list of dictionaries.

        Each dictionary contains:
            - 'inclination' (int)
            - 'thickness' (float)
            - 'mean_color' (np.ndarray): NumPy array of [R, G, B]
            - 'dominant_color' (np.ndarray): NumPy array of [R, G, B]

        Returns:
            Tuple[NDArray[np.uint8], NDArray[np.uint8]]: actual_color, true_color
        """
        # Read the parameters from the YAML file
        data = read_parameter_yaml(self.parameter_dir_path)
        result = []

        for key, value in data.items():
            try:

                entry = {
                    "inclination": value["inclination"],
                    "thickness": value["thickness"],
                    "mean_color": convert_rgb_dict_to_color_vector(value["mean_color"]),
                    "dominant_color": convert_rgb_dict_to_color_vector(
                        value["dominant_color"]
                    ),
                    "true_color": convert_rgb_dict_to_color_vector(value["true_color"]),
                }

                result.append(entry)
            except KeyError as e:
                print(f"Missing key {e} in entry {key}")

        actual_color = np.array(
            list(map(lambda x: x[color_type], result)), dtype=np.uint8
        )

        true_color = np.array(
            list(map(lambda x: x["true_color"], result)), dtype=np.uint8
        )

        return actual_color, true_color


if __name__ == "__main__":
    center = (950, 550)
    radius = 20
    file_name = "q_90_40.avi"

    calibrator = GetCalibrationColorFromImg("../tmp/calibration")
    calibrator.add_information_of_parameters(file_name, center, radius)
    fig, ax = calibrator.plot_calibrate_range(file_name)
    calibrator.compare_estimated_and_true_color(file_name)

    X, Y = calibrator.get_actual_color_and_true_color("dominant_color")
    i = 7
    predict = multiple_polynomial_regression(
        X.astype(np.float64), Y.astype(np.float64), 1
    )

    y = predict(np.array([X[i]]))
    y = convert_float_vector_to_uint_color(y)
    y = y[0]

    show_color(Y[i])
    show_color(y)


# %%
