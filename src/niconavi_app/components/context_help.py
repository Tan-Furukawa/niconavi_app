"""Delayed, concise hover help for NicoNavi controls."""

from __future__ import annotations

import re
import textwrap
from typing import Any

import flet as ft


HELP_WAIT_MS = 2000
HELP_SHOW_MS = 15000
HELP_WRAP_COLUMNS = 58


TAB_HELP = (
    "Video tab: select the required rotated XPL movie, optional supporting "
    "movies, specimen thickness, and acquisition settings.",
    "Center tab: set the rotation centre of the XPL video, then calculate "
    "the aligned optical maps.",
    "Map tab: clean the orientation map, segment grains, fill boundary "
    "regions, and confirm the grain boundaries.",
    "Filter tab: create labels and mark example grains so the software can "
    "classify the remaining grains.",
    "Analysis tab: choose a plot or CPO result and adjust how the processed "
    "grain data are displayed.",
    "Function tab: run optional processing tools that are enabled for this build.",
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, ft.Text):
        return str(value.value or "").strip()
    return ""


def _direct_labels(control: ft.Control) -> list[str]:
    return [
        value
        for child in control._get_children()
        if isinstance(child, ft.Text)
        if (value := _text_value(child))
    ]


def _control_title(control: ft.Control, context: tuple[str, ...]) -> str:
    label = _text_value(getattr(control, "label", None))
    text = _text_value(getattr(control, "text", None))
    tooltip = getattr(control, "tooltip", None)
    tooltip_text = _text_value(tooltip)
    hint = _text_value(getattr(control, "hint_text", None))
    value = _text_value(getattr(control, "value", None))
    context_text = " ".join(context).strip()

    if isinstance(control, ft.PopupMenuButton):
        return "File menu"
    if isinstance(control, ft.IconButton):
        if tooltip_text:
            return tooltip_text
        icon = str(control.icon or "").lower()
        if "remove" in icon:
            return f"Decrease {context_text}".strip()
        if "add" in icon:
            return f"Increase {context_text}".strip()
    if control.__class__.__name__ == "CustomSelectFileButton":
        return context_text or text or "Select input file"
    if label:
        return label
    if text and _normalise(text) not in {"select", "not selected"}:
        return text
    if isinstance(control, ft.Dropdown):
        return context_text or hint or value or "this selection"
    if isinstance(control, ft.TextField):
        return context_text or hint or "this parameter"
    if isinstance(control, ft.RadioGroup):
        return context_text or "this selection"
    if isinstance(control, ft.Slider):
        return context_text or "this value"
    return tooltip_text or context_text or text or control.__class__.__name__


def _specific_body(title: str) -> str | None:
    name = _normalise(title)
    rules: tuple[tuple[tuple[str, ...], str], ...] = (
        (
            ("image width",),
            "Sets the physical width of the displayed field in micrometres.",
        ),
        (
            ("apply mask",),
            "Shows or hides the current mask. When enabled, masked areas are excluded from the displayed result.",
        ),
        (
            ("calculate grain boundaries",),
            "Creates grain boundaries from the current angle map and segmentation settings.",
        ),
        (
            ("gbs",),
            "Shows or hides the grain-boundary overlay without recalculating the grains.",
        ),
        (
            ("rotated xpl + λ plate",),
            "Selects the optional rotating XPL movie recorded with the retardation plate. Adding it enables plate-based colour and orientation results.",
        ),
        (
            ("rotated xpl",),
            "Selects the required rotating XPL movie used to create the main optical maps.",
        ),
        (
            ("0° tilted image",),
            "Selects the optional 0° tilted XPL + λ movie used for the 0° original-colour-change result.",
        ),
        (
            ("45° tilted image",),
            "Selects the optional 45° tilted XPL + λ movie used for the 45° original-colour-change result.",
        ),
        (
            ("45° reference image",),
            "Selects the 45° reference movie used to align the corresponding tilted movie. Poor alignment produces false colour changes.",
        ),
        (
            ("thin section thickness",),
            "Sets the section thickness in millimetres. A larger value allows a larger retardation range in later estimates.",
        ),
        (
            ("λ =", "lambda"),
            "Sets the retardation of the inserted plate in nanometres. Use the value specified for the plate used during recording.",
        ),
        (
            ("gray-world white balance",),
            "Balances colour using the first XPL frame. Enable it when the illumination has a noticeable colour cast; it is applied when processing starts.",
        ),
        (
            ("rotation center x", "decrease x", "increase x"),
            "Sets the horizontal rotation centre. Adjust it until the rotation centre of the specimen pattern matches the centre marker in the preview.",
        ),
        (
            ("rotation center y", "decrease y", "increase y"),
            "Sets the vertical rotation centre. Adjust it until the rotation centre of the specimen pattern matches the centre marker in the preview.",
        ),
        (
            ("smallest grain size",),
            "Sets the minimum retained grain size as the square of the entered pixel length. Larger values absorb more small regions into neighbouring grains.",
        ),
        (
            ("slic superpixel size",),
            "Sets the approximate SLIC superpixel area as the square of the entered pixel length. Larger values create coarser initial regions and usually fewer boundaries.",
        ),
        (
            ("clean",),
            "Removes small local irregularities from the angle map. Repeating this step smooths more areas and may remove fine details.",
        ),
        (
            ("segment",),
            "Groups neighbouring regions with similar orientations. A larger Angle value permits more merging and usually produces fewer, larger grains.",
        ),
        (
            ("angle deg", "angle"),
            "Sets the maximum orientation difference used for grain segmentation. Larger values merge more neighbouring regions.",
        ),
        (
            ("fill",),
            "Assigns narrow unfilled boundary areas to nearby grains. Repeating it fills farther into the remaining boundary regions.",
        ),
        (
            ("apply",),
            "Accepts the currently displayed grain boundaries for the next analysis step.",
        ),
        (
            ("edit",),
            "Opens the grain-label editing view so grain assignments can be corrected manually.",
        ),
        (
            ("recalculate",),
            "Runs the map calculations again with the current files and parameters. Results from later steps are replaced.",
        ),
        (
            ("start",),
            "Loads the selected inputs and starts the first processing step. Check the required XPL movie and parameters before continuing.",
        ),
        (
            ("reset",),
            "Returns the current workflow section to its initial state and removes results that depend on it.",
        ),
        (
            ("continue",),
            "Finishes this step and moves to the next tab.",
        ),
        (
            ("file menu",),
            "Opens commands for saving or loading a project and exporting results.",
        ),
        (
            ("save project",),
            "Saves the input paths, settings, and calculated results in a NicoNavi project file.",
        ),
        (
            ("load project",),
            "Loads a saved NicoNavi project and restores its available workflow results.",
        ),
        (
            ("export image",),
            "Exports the image currently shown in the main view as a PDF.",
        ),
        (
            ("export grain data",),
            "Exports calculated grain measurements and related tables for use outside NicoNavi.",
        ),
        (
            ("calculate",),
            "Calculates CPO and inclination results from the selected grain labels and refractive indices. Existing CPO results are replaced.",
        ),
        (
            ("save",),
            "Saves the current filter labels and uses them to classify the remaining grains. Existing analysis results based on older labels are replaced.",
        ),
        (
            ("add label",),
            "Adds a new filter label using the name entered on the left. Select the new label before marking example grains.",
        ),
        (
            ("label name",),
            "Names a new grain class for filtering. Use a short mineral or category name, then select the + button to add it.",
        ),
        (
            ("move image up",),
            "Moves the filter image upward so a different area can be labelled. This changes the view only, not the grain data.",
        ),
        (
            ("move image down",),
            "Moves the filter image downward so a different area can be labelled. This changes the view only, not the grain data.",
        ),
        (
            ("move image left",),
            "Moves the filter image left so a different area can be labelled. This changes the view only, not the grain data.",
        ),
        (
            ("move image right",),
            "Moves the filter image right so a different area can be labelled. This changes the view only, not the grain data.",
        ),
        (
            ("zoom in",),
            "Magnifies the filter image to make small grains easier to label. This changes the view only.",
        ),
        (
            ("zoom out",),
            "Shows a wider area of the filter image. This changes the view only.",
        ),
        (
            ("boundaries",),
            "Shows or hides grain boundaries in the filter view. Hiding them can make colour easier to inspect; it does not change the segmentation.",
        ),
        (
            ("training boxes",),
            "Shows or hides the marked training examples used for filtering. This changes the overlay only.",
        ),
        (
            ("color theme",),
            "Chooses the colour palette used for the polar plot. This changes appearance only.",
        ),
        (
            ("color mode",),
            "Chooses stepped or smoothly varying colours for the polar plot. This changes appearance only.",
        ),
        (
            ("color",),
            "Uses grain colour when predicting filter labels. Disable it if colour varies strongly within the same class.",
        ),
        (
            ("elongation",),
            "Uses grain elongation when predicting filter labels. Enable it when classes have visibly different shapes.",
        ),
        (
            ("size",),
            "Uses grain size when predicting filter labels. Enable it when classes occupy different size ranges.",
        ),
        (
            ("position",),
            "Uses image position when predicting filter labels. Enable it only when class occurrence genuinely depends on location.",
        ),
        (
            ("bins", "decrease bins", "increase bins"),
            "Sets how many groups are shown in the histogram or rose diagram. Larger values show finer detail but can make the plot noisier.",
        ),
        (
            ("opacity",),
            "Controls plot transparency. Larger values make the plotted bars or sectors more solid.",
        ),
        (
            ("flip",),
            "Mirrors the rose diagram orientation without changing the underlying measurements.",
        ),
        (
            ("grain value",),
            "Chooses whether each grain is represented by its median or mean value. Median is less affected by unusual pixels.",
        ),
        (("regression",), "Shows or hides a fitted trend line on the scatter plot."),
        (
            ("origin",),
            "Forces the scatter-plot regression line through zero when regression is enabled.",
        ),
        (
            ("log(x)",),
            "Uses a logarithmic horizontal axis. This can make values spanning a wide range easier to compare.",
        ),
        (
            ("log(y)",),
            "Uses a logarithmic vertical axis. This can make values spanning a wide range easier to compare.",
        ),
        (
            ("polar plot bandwidth",),
            "Controls smoothing in the polar density plot. Larger values produce a smoother, less detailed distribution.",
        ),
        (
            ("number of contours",),
            "Sets the number of colour bands in the polar plot. Larger values show smaller changes in density.",
        ),
        (
            ("point noise",),
            "Adds a small visual offset to overlapping points. Larger percentages separate points more, without changing their measured values.",
        ),
        (
            ("normalize to 90°",),
            "Rescales the selected inclination distribution so its chosen percentile reaches 90°. Use this only when relative inclination is more useful than the absolute estimate.",
        ),
        (
            ("percentile",),
            "Selects which upper percentile is mapped to 90° during normalization. Higher values are influenced more by the most inclined grains.",
        ),
        (
            ("display points",),
            "Shows or hides individual measurements on top of the polar density plot.",
        ),
        (
            ("group by",),
            "Chooses whether the analysis represents individual pixels or whole grains.",
        ),
        (
            ("ω =", "ε =", "refractive indices"),
            "Sets the ordinary and extraordinary refractive indices used for inclination estimation. Keep the quartz defaults unless analysing another material.",
        ),
    )
    for phrases, body in rules:
        if any(phrase in name for phrase in phrases):
            return body
    return None


def _wrap_help(body: str) -> str:
    paragraphs = body.splitlines() or [body]
    return "\n".join(
        textwrap.fill(
            paragraph,
            width=HELP_WRAP_COLUMNS,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for paragraph in paragraphs
    )


def make_help_tooltip(body: str) -> ft.Tooltip:
    return ft.Tooltip(
        message=_wrap_help(body),
        wait_duration=HELP_WAIT_MS,
        show_duration=HELP_SHOW_MS,
        padding=12,
        bgcolor="#fff7f7f7",
        border_radius=6,
        text_style=ft.TextStyle(color=ft.Colors.BLACK87, size=13),
        text_align=ft.TextAlign.LEFT,
        enable_tap_to_dismiss=True,
    )


def set_context_help(control: ft.Control, body: str) -> ft.Control:
    """Attach purpose-written delayed hover help to a control."""
    setattr(control, "_context_help_body", body)
    control.tooltip = make_help_tooltip(body)
    return control


def suppress_context_help(control: ft.Control) -> ft.Control:
    """Remove hover help from a self-explanatory control."""
    setattr(control, "_context_help_suppress", True)
    control.tooltip = None
    return control


def _is_interactive(control: ft.Control) -> bool:
    if isinstance(
        control,
        (
            ft.TextField,
            ft.Dropdown,
            ft.RadioGroup,
            ft.Slider,
            ft.Checkbox,
            ft.Switch,
            ft.PopupMenuButton,
        ),
    ):
        return True
    return any(
        hasattr(control, event_name) and getattr(control, event_name) is not None
        for event_name in ("on_click", "on_change", "on_tap")
    )


def _help_body(
    control: ft.Control,
    *,
    area: str,
    context: tuple[str, ...],
) -> str | None:
    explicit = getattr(control, "_context_help_body", None)
    if isinstance(explicit, str):
        return explicit

    title = _control_title(control, context)
    if area == "images":
        # Image selectors carry calculation-specific help at their definition.
        # Do not replace a missing description with repetitive display text.
        return None

    specific = _specific_body(title)
    if specific is not None:
        return specific
    if isinstance(control, ft.TextField):
        return (
            f"Sets {title}. The new value is used the next time the related "
            "calculation or plot is updated."
        )
    if isinstance(control, (ft.Dropdown, ft.RadioGroup)):
        return (
            f"Chooses {title}. The current plot or display updates to use the "
            "selected option."
        )
    if isinstance(control, (ft.Checkbox, ft.Switch)):
        return f"Turns {title} on or off for the current display or calculation."
    if isinstance(control, ft.Slider):
        return f"Adjusts {title}. Moving right increases the value."
    return None


def attach_context_help(root: ft.Control, *, area: str) -> None:
    """Attach hover help throughout one layout subtree."""
    seen: set[int] = set()

    def visit(control: ft.Control, context: tuple[str, ...]) -> None:
        if id(control) in seen:
            return
        seen.add(id(control))

        labels = tuple(_direct_labels(control))
        child_context = (*context, *labels) if labels else context
        if not getattr(control, "_context_help_suppress", False):
            body = _help_body(control, area=area, context=context)
            if body is not None and _is_interactive(control):
                set_context_help(control, body)

        for child in control._get_children():
            visit(child, child_context)

    visit(root, ())
