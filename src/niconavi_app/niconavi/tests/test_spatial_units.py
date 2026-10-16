from matplotlib.figure import Figure

from niconavi_app.components.view.spatial_units import apply_micrometer_axis
from niconavi_app.stores import Stores


def test_image_axes_use_pixels_when_image_width_is_not_calibrated() -> None:
    stores = Stores()
    stores.ui.one_pixel.set(0.0)
    figure = Figure()
    axis = figure.subplots()

    apply_micrometer_axis(axis, stores)

    assert axis.get_xlabel() == "px"
    assert axis.get_ylabel() == "px"


def test_image_axes_use_micrometers_when_image_width_is_calibrated() -> None:
    stores = Stores()
    stores.ui.one_pixel.set(0.5)
    figure = Figure()
    axis = figure.subplots()

    apply_micrometer_axis(axis, stores)

    assert axis.get_xlabel() == "μm"
    assert axis.get_ylabel() == "μm"
