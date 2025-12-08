from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import numpy as np

from stores import Stores
from tools.no_image import get_no_image
from components.view.style import set_default_figure_style
from components.view.spatial_units import apply_micrometer_axis


_MOVIE_KEY_MAP = {
    0: "xpl",
    1: "full_wave",
    2: "image0_tilt",
    3: "image45_tilt",
    4: "image0",
    5: "image45",
}


def at_movie_tab(stores: Stores) -> Figure:

    first_images = stores.computation_result.first_image.get()
    if first_images is None:
        return get_no_image()

    selected = stores.ui.selected_button_at_movie_tab.get()
    key = _MOVIE_KEY_MAP.get(selected)
    if key is None:
        raise ValueError("unexpected value of stores.ui.selected_button_at_movie_tab")

    first_frame = first_images.get(key)

    if first_frame is not None:
        fig, ax = plt.subplots()

        ax.imshow(first_frame)
        apply_micrometer_axis(ax, stores)
        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig, ax)

        return fig

    return get_no_image()
