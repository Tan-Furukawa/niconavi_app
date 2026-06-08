from matplotlib.pyplot import Figure, Axes
from typing import Optional, Any, Literal
from numpy.typing import NDArray


def set_default_figure_style(
    fig: Figure,
    ax: Optional[Axes] = None,
    method: Literal[
        "rose diagram", "scatter", "histogram", "map_only_right_bottom", "polar plot"
    ] = "scatter",
    line_color: Optional[str] = None,
) -> None:

    fig.patch.set_facecolor("none")
    # fig.subplots_adjust(left=-0.2)

    if ax is not None:
        pass
    else:
        ax = fig.axes[0]

    ax.set_facecolor("none")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    if method == "rose diagram":
        ax.grid(color="white", alpha=0.3)
        ax.spines["polar"].set_color("white")
        ax.spines["start"].set_color("white")
        ax.spines["end"].set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for patch in ax.patches:
            patch.set_edgecolor("white")

    elif method == "polar plot":

        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.spines["top"].set_color("white")

        font_size = 7
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        fig.axes[0].xaxis.label.set_fontsize(font_size)
        fig.axes[0].yaxis.label.set_fontsize(font_size)

        # ax.grid(color="white", alpha=0.3)
        # ax.spines["polar"].set_color("white")
        # ax.spines["start"].set_color("white")
        # ax.spines["end"].set_color("white")
        # ax.xaxis.label.set_color("white")
        # ax.yaxis.label.set_color("white")
        # for patch in ax.patches:
        #     patch.set_edgecolor("white")
        # # font_size = 7
        # # fig.axes[0].xaxis.label.set_fontsize(font_size)
        # # fig.axes[0].yaxis.label.set_fontsize(font_size)
        # # ax.title.set_color("white")

    elif method == "histogram":
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        for patch in ax.patches:
            patch.set_edgecolor("white")

    elif method == "scatter":
        # 軸、メモリ、文字などを白色に設定
        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.spines["top"].set_color("white")

        font_size = 12
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        fig.axes[0].xaxis.label.set_fontsize(font_size)
        fig.axes[0].yaxis.label.set_fontsize(font_size)

        for col in ax.collections:
            ...
            # if isinstance(col, mc.PathCollection):
            # col.set_edgecolors("white" if line_color is None else line_color)  # type: ignore
            # col.set_colors("white")
            # col.set_linewidths(1.0)  # type: ignore

        ax.title.set_color("white")

    elif method == "map_only_right_bottom":
        # 軸、メモリ、文字などを白色に設定
        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")

        font_size = 7
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        fig.axes[0].xaxis.label.set_fontsize(font_size)
        fig.axes[0].yaxis.label.set_fontsize(font_size)
        ax.title.set_color("white")
