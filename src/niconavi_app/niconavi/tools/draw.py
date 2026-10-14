import cv2
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_grid(v: NDArray[np.uint8]) -> Tuple[Figure, Axes]:
    """
    Plots an image array with a grid overlay.

    Parameters:
        v (np.ndarray): The image array to display.

    Returns:
        Tuple[Figure, Axes]: The matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots()
    ax.imshow(v)
    ax.grid(color="gray", linestyle="dotted", linewidth=1)
    return fig, ax


def draw_rectangle(
    img: NDArray[np.uint8], center: Tuple[int, int], radius: float
) -> NDArray[np.uint8]:
    """
    Draws a rectangle on the image centered at the given point with the specified radius.

    Parameters:
        img (np.ndarray): The input image array.
        center (Tuple[int, int]): The (x, y) coordinates of the center point.
        radius (float): The radius to determine the size of the rectangle.

    Returns:
        np.ndarray: The image with the rectangle drawn on it.
    """
    # Copy the original image to avoid modifying it
    output_img = img.copy()
    # Calculate the top-left and bottom-right points of the rectangle
    point1 = [int(center[0] - radius), int(center[1] - radius)]
    point2 = [int(center[0] + radius), int(center[1] + radius)]
    # Draw the rectangle on the image
    cv2.rectangle(output_img, point1, point2, (255, 0, 0), thickness=3)
    return output_img


def extract_rectangle_region(
    img: NDArray, center: Tuple[int, int], radius: float
) -> NDArray:
    """
    Extracts a rectangular region from the image centered at the given point with the specified radius.

    Parameters:
        img (np.ndarray): The input image array.
        center (Tuple[int, int]): The (x, y) coordinates of the center point.
        radius (float): The radius to determine the size of the rectangle.

    Returns:
        np.ndarray: The extracted rectangular region of the image.
    """
    # Get the absolute value of the radius
    radius = abs(radius)

    # Calculate the bounding box coordinates
    x_min = int(max(center[0] - radius, 0))
    x_max = int(min(center[0] + radius, img.shape[1] - 1))
    y_min = int(max(center[1] - radius, 0))
    y_max = int(min(center[1] + radius, img.shape[0] - 1))

    # Crop the image
    cropped_img = img[y_min : y_max + 1, x_min : x_max + 1]

    # Create a mask
    mask_height = y_max - y_min + 1
    mask_width = x_max - x_min + 1
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Adjust the center coordinates for the cropped image
    cropped_center = (int(center[0] - x_min), int(center[1] - y_min))
    # Draw a filled rectangle on the mask
    cv2.rectangle(
        mask,
        [int(cropped_center[0] - radius), int(cropped_center[1] - radius)],
        [int(cropped_center[0] + radius), int(cropped_center[1] + radius)],
        (255, 255, 255),
        thickness=-1,
    )

    # Apply the mask to extract the rectangular region
    cropped_img_masked = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

    return cropped_img_masked
