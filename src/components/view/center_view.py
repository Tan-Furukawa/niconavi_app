from __future__ import annotations

from copy import deepcopy

from matplotlib.pyplot import Figure

from stores import Stores
from niconavi.find_center import plot_center_image
from components.view.style import set_default_figure_style
from components.view.spatial_units import apply_micrometer_axis
from tools.no_image import get_no_image


def at_center_tab(stores: Stores) -> Figure:
    image = stores.computation_result.rotation_img.get()
    cx = stores.computation_result.center_int_x.get()
    cy = stores.computation_result.center_int_y.get()

    if image is not None and cx is not None and cy is not None:


        fig = plot_center_image(image, (cx, cy))
        if fig.axes:
            apply_micrometer_axis(fig.axes[0], stores)
        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig)
        return fig

    return get_no_image()
