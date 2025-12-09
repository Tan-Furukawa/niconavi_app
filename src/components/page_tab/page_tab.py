from reactive_state import ReactiveTabs, ReactiveTab
from state import State, ReactiveState, StateProperty
from components.app_bar import niconaviAppBar
from stores import Stores
from components.page_tab.tabs.movie_tab import MovieTab
from components.page_tab.tabs.center_tab import CenterTab
from components.page_tab.tabs.grain_tab import GrainTab
from components.page_tab.tabs.analysis_tab import AnalysisTab
from tools.tools import switch_tab_index
from components.page_tab.tabs.filter_tab import FilterTab
from components.page_tab.tabs.merge_component import MergeTab

import flet as ft
from logging import Logger, getLogger


class PageTabs(ft.Container):
    def __init__(self, page: ft.Page, stores: Stores):
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

        self.width = stores.appearance.tabs_width
        self.expand = True

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
                            text="video",
                            content=MovieTab(page, stores),
                        ),
                        ReactiveTab(
                            text="center",
                            # tab_content=ft.Icon(ft.Icons.SEARCH),
                            content=CenterTab(page, stores),
                            # icon=ft.Icons.ADJUST,
                            visible=ReactiveState(
                                lambda: stores.ui.progress.get() >= 1,
                                [stores.ui.progress],
                            ),
                        ),
                        ReactiveTab(
                            text="map",
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
                            text="filter",
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
                            text="analysis",
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
                    ],
                ),
            ]
        )
