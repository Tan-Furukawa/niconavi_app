from stores import Stores

from niconavi.image.type import RGBPicture
from PIL import Image
import io
import flet as ft
import base64
from typing import overload, Optional
from logging import getLogger, Logger


@overload
def convert_RGBPicture_to_src_base64(img: None) -> None: ...
@overload
def convert_RGBPicture_to_src_base64(img: RGBPicture) -> str: ...


def convert_RGBPicture_to_src_base64(img: RGBPicture | None) -> str | None:
    if img is None:
        return None
    else:
        pil_image: Image.Image = Image.fromarray(img)
        buffer: io.BytesIO = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str: str = base64.b64encode(buffer.getvalue()).decode()
        # img_control: ft.Image = ft.Image(src_base64=img_str)
        return img_str


def switch_tab_index(stores: Stores, index: int, logger: Optional[Logger] = None) -> None:
    if logger is not None:
        logger.info(f"switch to {index} tab")
    stores.ui.selected_index.set(index)


def force_update_image_view(stores: Stores) -> None:
    stores.ui.force_update_image_view.set(stores.ui.force_update_image_view.get() + 1)


# import numpy as np
# np.array([1, 2]) == np.array([1,2])
