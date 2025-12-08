# %%
from niconavi.optics.uniaxial_plate import get_retardation_color_chart_with_nd_filter

import numpy as np
import matplotlib.pyplot as plt
import niconavi.optics.optical_system as osys
from niconavi.optics.types import (
    WavelengthVector,
)


# color_chart_1500_100_20_3, retardation_1500_100_20_3 = (
#     get_retardation_color_chart_with_nd_filter(end=1500, num=100, nd_num=20)
# )

# plt.imshow(color_chart_1500_100_20_3[2:])
# np.save("color_chart_1500_100_20_3.npy", color_chart_1500_100_20_3[3:,:])
# np.save("retardation_1500_100_20_3.npy", retardation_1500_100_20_3)
