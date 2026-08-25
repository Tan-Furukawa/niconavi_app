from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import numpy as np

from niconavi_app.niconavi.image.white_balance import (
    GRAY_WORLD_CORRECTION_STRENGTH,
    apply_rgb_gains,
    blend_gains,
    gray_world_mean_gains,
)
from niconavi_app.niconavi.type import FirstVideoImage
from niconavi_app.stores import Stores
from niconavi_app.tools.no_image import get_no_image
from niconavi_app.components.view.style import set_default_figure_style
from niconavi_app.components.view.spatial_units import apply_micrometer_axis


_MOVIE_KEY_MAP = {
    0: "xpl",
    1: "full_wave",
    2: "image0_tilt",
    3: "image45_tilt",
    4: "image0",
    5: "image45",
}


def preview_white_balance(
    stores: Stores, first_images: FirstVideoImage, frame: np.ndarray
) -> np.ndarray:
    """Show the frame as the run will use it, whatever the checkbox says now.

    Before "start" the frames in first_image are still the raw ones read at
    file-pick time, so the checkbox only changes what the preview should look
    like - the same gains load_data will use are computed here from the first
    XPL frame. Once load_data has run it has already baked the correction into
    the frames and recorded the gains it used, so the frame is shown as it is
    and never corrected twice.
    """
    if not stores.computation_result.apply_white_balance.get():
        return frame
    if stores.computation_result.white_balance_gains.get() is not None:
        return frame

    reference_image = first_images.get("xpl")
    if reference_image is None:
        return frame

    try:
        gains = blend_gains(
            gray_world_mean_gains(reference_image), GRAY_WORLD_CORRECTION_STRENGTH
        )
    except ValueError:
        # A reference frame the gray-world gains cannot be estimated from -
        # leave the preview alone and let "start" report the problem.
        return frame
    return apply_rgb_gains(frame, gains)


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

        ax.imshow(preview_white_balance(stores, first_images, first_frame))
        apply_micrometer_axis(ax, stores)
        stores.ui.displayed_fig.set(deepcopy(fig))
        set_default_figure_style(fig, ax)

        return fig

    return get_no_image()
