from io import BytesIO
import base64

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from typing import Dict, Tuple


def _compute_class_map(index_map: np.ndarray, class_ids: np.ndarray):
    if class_ids.ndim != 1:
        raise ValueError("class_ids must be 1D array.")

    index_map = np.asarray(index_map)
    class_ids = np.asarray(class_ids)

    # Only allow positive region ids; 0 and negatives represent non-clickable areas.
    valid_mask = (index_map > 0) & (index_map < class_ids.size)
    class_map = np.zeros_like(index_map, dtype=int)
    if np.any(valid_mask):
        region_indices = index_map[valid_mask].astype(int, copy=False)
        class_map[valid_mask] = class_ids[region_indices]

    max_class_id = int(class_map.max()) if class_map.size else 0
    n_classes = max_class_id + 1
    return class_map, n_classes


def _build_palette(n_classes: int):
    if n_classes <= 1:
        return ["#4d4d4d"]

    tab20 = plt.colormaps["tab20"]
    base_colors = list(getattr(tab20, "colors", [])) or [tab20(i / max(tab20.N - 1, 1)) for i in range(tab20.N)]
    palette = ["#bdbdbd"]
    while len(palette) < n_classes:
        palette.append(base_colors[(len(palette) - 1) % len(base_colors)])
    return palette[:n_classes]


def _apply_custom_palette(palette: list, custom_colors: Dict[int, str] | None) -> None:
    if not custom_colors:
        return
    for class_id, color in custom_colors.items():
        if not isinstance(class_id, int) or class_id < 0:
            continue
        if class_id >= len(palette):
            continue
        try:
            palette[class_id] = mcolors.to_hex(color)
        except ValueError:
            continue


def plot_index_map_predictions(index_map: np.ndarray, class_ids: np.ndarray):
    class_map, n_classes = _compute_class_map(index_map, class_ids)
    palette = _build_palette(n_classes)
    cmap = ListedColormap(palette)

    norm = BoundaryNorm(np.arange(n_classes + 1), n_classes)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(class_map, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_classes))
    cbar.set_label("Predicted class")
    ax.set_title("Predicted classes per index map region")
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_overlay_base64(
    index_map: np.ndarray,
    class_ids: np.ndarray,
    overlay_alpha: float = 0.65,
    boundary_mask: np.ndarray | None = None,
    background_image: np.ndarray | None = None,
    background_mode: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 100,
    highlight_border_mask: np.ndarray | None = None,
    custom_colors: Dict[int, str] | None = None,
    show_boundaries: bool | None = None,
) -> Tuple[str, list]:
    class_map, n_classes = _compute_class_map(index_map, class_ids)
    palette = _build_palette(n_classes)
    _apply_custom_palette(palette, custom_colors)

    rgba_palette = []
    for class_id, color in enumerate(palette):
        rgba = list(mcolors.to_rgba(color))
        rgba[3] = 0.0 if class_id == 0 else overlay_alpha
        rgba_palette.append(tuple(rgba))

    cmap = ListedColormap(rgba_palette)
    norm = BoundaryNorm(np.arange(n_classes + 1), n_classes)

    if figsize is None:
        h, w = class_map.shape
        figsize = (w / dpi, h / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if show_boundaries is None:
        show_boundaries = True

    use_background_image = background_image is not None
    effective_boundary_mask = boundary_mask
    if background_mode is not None:
        use_background_image = background_mode != "boundary"

    if use_background_image and background_image is not None:
        ax.imshow(background_image, interpolation="nearest")
    else:
        if effective_boundary_mask is None:
            effective_boundary_mask = (np.asarray(index_map) == 0)
        ax.imshow(effective_boundary_mask, cmap="gray", interpolation="nearest")

    ax.imshow(class_map, cmap=cmap, norm=norm, interpolation="nearest")

    if show_boundaries:
        if effective_boundary_mask is None:
            effective_boundary_mask = (np.asarray(index_map) == 0)
        boundary_overlay = np.zeros((*effective_boundary_mask.shape, 4), dtype=float)
        boundary_overlay[..., :3] = 1.0
        boundary_overlay[..., 3] = np.asarray(effective_boundary_mask, dtype=float) * 0.8
        ax.imshow(boundary_overlay, interpolation="nearest")

    if highlight_border_mask is not None:
        border = np.asarray(highlight_border_mask, dtype=bool)
        if border.shape != class_map.shape:
            raise ValueError("highlight_border_mask must match index_map shape")
        overlay = np.zeros((*border.shape, 4), dtype=float)
        overlay[..., 0] = 1.0
        overlay[..., 3] = border.astype(float)
        ax.imshow(overlay, interpolation="nearest")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return encoded, palette
