from reactive_state import ReactiveElevatedButton
from stores import Stores
from components.selector.tools import make_always_display_elevated_button


#! developping
def make_always_visible_state(
    stores: Stores,
) -> tuple[ReactiveElevatedButton,]:
    mask = make_always_display_elevated_button(
        stores,
        "mask",
        0,
        visible_condition=lambda: stores.computation_result.mask.get() is not None,
        visible_reliance_state=[stores.computation_result.mask],
    )

    return mask
