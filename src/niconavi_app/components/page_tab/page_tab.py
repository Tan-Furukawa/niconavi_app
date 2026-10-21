from niconavi_app.reactive_state import ReactiveTabs, ReactiveTab
from niconavi_app.state import State, ReactiveState, StateProperty
from niconavi_app.components.app_bar import niconaviAppBar
from niconavi_app.stores import Stores
from niconavi_app.components.page_tab.tabs.movie_tab import MovieTab
from niconavi_app.components.page_tab.tabs.center_tab import CenterTab
from niconavi_app.components.page_tab.tabs.grain_tab import GrainTab
from niconavi_app.components.page_tab.tabs.analysis_tab import AnalysisTab
from niconavi_app.components.page_tab.tabs.function_tab import FunctionTab
from niconavi_app.tools.tools import switch_tab_index
from niconavi_app.components.page_tab.tabs.filter_tab import FilterTab
from niconavi_app.components.page_tab.tabs.merge_component import MergeTab
from niconavi_app.app_config import ENABLE_FUNCTION_TAB
from niconavi_app.components.context_help import TAB_HELP, set_context_help

import flet as ft
from logging import Logger, getLogger


class PageTabs(ft.Container):
    def __init__(
        self,
        page: ft.Page,
        stores: Stores,
    ):
        super().__init__()

        logger = getLogger("niconavi").getChild(__name__)

        self.margin = 0
        self.padding = 0
        self.border_radius = 0
        self.bgcolor = ft.Colors.BLACK12
        # self.width = stores.appearance.tabs_width
        # self.height = page.window.height
        # self.height = page.window.height - stores.appearance.log_area_height
        self.tab_selected_index = stores.ui.selected_index

        def tab_label(text: str, index: int) -> ft.Control:
            return set_context_help(ft.Container(content=ft.Text(text)), TAB_HELP[index])

        self.expand = True
        function_tabs = []
        if ENABLE_FUNCTION_TAB:
            function_tabs.append(
                ReactiveTab(
                    tab_content=tab_label("Function", 5),
                    content=FunctionTab(page, stores),
                )
            )

        self.content = ft.Column(
            [
                ReactiveTabs(
                    on_change=lambda e: switch_tab_index(
                        stores, e.control.selected_index, logger=logger
                    ),
                    scrollable=True,
                    expand=True,
                    selected_index=self.tab_selected_index,
                    animation_duration=0,
                    label_color=ft.Colors.BLUE_100,
                    unselected_label_color=ft.Colors.WHITE,
                    tabs=[
                        ReactiveTab(
                            tab_content=tab_label("Video", 0),
                            content=MovieTab(page, stores),
                        ),
                        ReactiveTab(
                            tab_content=tab_label("Center", 1),
                            # tab_content=ft.Icon(ft.Icons.SEARCH),
                            content=CenterTab(page, stores),
                            # icon=ft.Icons.ADJUST,
                            visible=ReactiveState(
                                lambda: stores.ui.progress.get() >= 1,
                                [stores.ui.progress],
                            ),
                        ),
                        ReactiveTab(
                            tab_content=tab_label("Map", 2),
                            content=GrainTab(page, stores),
                            visible=ReactiveState(
                                lambda: stores.ui.progress.get() >= 2,
                                [stores.ui.progress],
                            ),
                            # visible=ReactiveState(
                            #     lambda: stores.computation_result.raw_maps.get() is not None,
                            #     [stores.computation_result.raw_maps],
                            # ),
                        ),
                        # ft.Tab(
                        #     text="merge",
                        #     content=MergeTab(page, stores),
                        #     # icon=ft.Icons.PUBLIC,
                        #     # icon=ft.Icons.SETTINGS,
                        # ),
                        ReactiveTab(
                            tab_content=tab_label("Filter", 3),
                            content=FilterTab(page, stores),
                            visible=ReactiveState(
                                lambda: stores.ui.progress.get() >= 3,
                                [stores.ui.progress],
                            ),
                            # icon=ft.Icons.FILTER_ALT,
                            # visible=ReactiveState(
                            #     lambda: stores.computation_result.grain_segmented_maps.get()
                            #     is not None
                            #     and stores.computation_result.grain_list.get() is not None,
                            #     [
                            #         stores.computation_result.grain_segmented_maps,
                            #         stores.computation_result.grain_list,
                            #     ],
                            # ),
                        ),
                        ReactiveTab(
                            tab_content=tab_label("Analysis", 4),
                            content=AnalysisTab(page, stores),
                            visible=ReactiveState(
                                lambda: stores.ui.progress.get() >= 4,
                                [stores.ui.progress],
                            ),
                            # visible=ReactiveState(
                            #     lambda: stores.computation_result.grain_classification_result.get()
                            #     is not None,
                            #     [stores.computation_result.grain_classification_result],
                            # ),
                            # icon=ft.Icons.BAR_CHART,
                        ),
                        *function_tabs,
                    ],
                ),
            ]
        )
