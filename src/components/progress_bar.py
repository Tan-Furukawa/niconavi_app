from reactive_state import ReactiveProgressBar
from stores import Stores
import flet as ft


class ProgressBar(ReactiveProgressBar):
    def __init__(self, stores: Stores):
        super().__init__(
            value=stores.ui.progress_bar.progress,
            visible=stores.ui.progress_bar.visible,
        )
        self.width = stores.appearance.tabs_width
        self.height = 5
        self.color = ft.Colors.AMBER


def update_progress_bar(progress: float | None, stores: Stores) -> None:
    # progress bar が動いているときは計算中とする。計算中は入力などがブロックされる。
    if progress == 0:
        stores.ui.computing_is_stop.set(True)
    else:
        stores.ui.computing_is_stop.set(False)

    stores.ui.progress_bar.progress.set(progress)
