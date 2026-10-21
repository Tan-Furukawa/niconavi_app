"""Context-sensitive help mode for the Images and settings panels.

The controller temporarily intercepts events from the controls that are visible
when help mode starts.  It restores both handlers and displayed values when help
mode ends, so asking for help can never run an action or alter a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Iterable

import flet as ft

from niconavi_app.components.common_component import CustomText


@dataclass(frozen=True)
class HelpTopic:
    title: str
    body: str


@dataclass
class _InterceptedControl:
    control: ft.Control
    handlers: dict[str, Callable[..., Any] | None] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)


def set_context_help(control: ft.Control, body: str) -> ft.Control:
    """Attach purpose-written help text while keeping the original control type."""
    setattr(control, "_context_help_body", body)
    return control


def suppress_context_help(control: ft.Control) -> ft.Control:
    """Keep an obvious control inert in help mode without opening a dialog."""
    setattr(control, "_context_help_suppress", True)
    return control


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, ft.Text):
        return str(value.value or "").strip()
    return ""


def _direct_labels(control: ft.Control) -> list[str]:
    labels: list[str] = []
    for child in control._get_children():
        if isinstance(child, ft.Text):
            value = _text_value(child)
            if value:
                labels.append(value)
    return labels


def _control_title(control: ft.Control, context: tuple[str, ...]) -> str:
    label = _text_value(getattr(control, "label", None))
    text = _text_value(getattr(control, "text", None))
    tooltip = _text_value(getattr(control, "tooltip", None))
    hint = _text_value(getattr(control, "hint_text", None))
    value = _text_value(getattr(control, "value", None))
    context_text = " ".join(context).strip()

    if isinstance(control, ft.Tabs):
        return "Workflow tabs"
    if isinstance(control, ft.PopupMenuButton):
        return "File menu"
    if isinstance(control, ft.IconButton):
        icon = str(control.icon or "").lower()
        if tooltip:
            return tooltip
        if "remove" in icon:
            return f"Decrease {context_text}".strip()
        if "add" in icon:
            return f"Increase {context_text}".strip()
        return tooltip or context_text or "Icon button"
    if control.__class__.__name__ == "CustomSelectFileButton":
        return context_text or text or "Select input file"
    if label:
        return label
    if text and _normalise(text) not in {"select", "not selected"}:
        return text
    if isinstance(control, ft.Dropdown):
        return context_text or hint or value or "Selection"
    if isinstance(control, ft.TextField):
        return context_text or label or hint or "Parameter"
    if isinstance(control, ft.RadioGroup):
        return context_text or "Selection"
    if isinstance(control, ft.Slider):
        return context_text or "Slider"
    return tooltip or context_text or text or control.__class__.__name__


def _specific_body(title: str) -> str | None:
    name = _normalise(title)

    rules: tuple[tuple[tuple[str, ...], str], ...] = (
        (
            ("image width",),
            "Sets the physical width of the displayed field in micrometres. A larger value increases reported physical grain sizes; it does not change the image pixels.",
        ),
        (
            ("apply mask",),
            "Shows or hides the current mask. When enabled, masked areas are excluded from the displayed result.",
        ),
        (
            ("calculate grain boundaries",),
            "Creates grain boundaries from the current map and segmentation settings.",
        ),
        (
            ("gbs", "grain boundaries"),
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
            "Runs the next workflow step using the current result. Later results may be replaced when an earlier step is run again.",
        ),
        (
            ("workflow tabs",),
            "Moves between the available workflow stages. Complete the tabs from left to right; later tabs appear as results become available.",
        ),
        (
            ("file menu",),
            "Opens commands for saving or loading a project and exporting results.",
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
            "Adds a new filter label using the name entered in the Label name box. Select the new label before marking example grains.",
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
            ("color theme",),
            "Chooses the colour palette used for the polar plot. This changes appearance only.",
        ),
        (
            ("color mode",),
            "Chooses stepped or smoothly varying colours for the polar plot. This changes appearance only.",
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


TAB_HELP = (
    "Video tab: select the required rotated XPL movie, optional supporting movies, specimen thickness, and acquisition settings.",
    "Center tab: set the rotation centre of the XPL video, then calculate the aligned optical maps.",
    "Map tab: clean the orientation map, segment grains, fill boundary regions, and confirm the grain boundaries.",
    "Filter tab: create labels and mark example grains so the software can classify the remaining grains.",
    "Analysis tab: choose a plot or CPO result and adjust how the processed grain data are displayed.",
    "Function tab: run optional processing tools that are enabled for this build.",
)


def _clicked_tab_index(control: ft.Tabs, event: Any | None) -> int:
    data = getattr(event, "data", None)
    if isinstance(data, str):
        try:
            decoded = json.loads(data)
            if isinstance(decoded, dict):
                data = decoded.get("index", decoded.get("selected_index"))
            else:
                data = decoded
        except (json.JSONDecodeError, TypeError):
            pass
    try:
        return int(data)
    except (TypeError, ValueError):
        return int(control.selected_index or 0)


def resolve_help_topic(
    control: ft.Control,
    *,
    area: str,
    context: tuple[str, ...] = (),
    event: Any | None = None,
) -> HelpTopic:
    """Return concise user-facing help for a control."""
    title = _control_title(control, context)
    explicit_body = getattr(control, "_context_help_body", None)
    if isinstance(explicit_body, str):
        return HelpTopic(title=title, body=explicit_body)
    if isinstance(control, ft.Tabs):
        index = _clicked_tab_index(control, event)
        body = TAB_HELP[index] if 0 <= index < len(TAB_HELP) else TAB_HELP[0]
        return HelpTopic(title=title, body=body)
    if area == "images":
        return HelpTopic(
            title=title,
            body=(
                f"Displays {title} in the main image view. "
                "Changing the displayed image does not recalculate the results."
            ),
        )
    specific = _specific_body(title)
    if specific is not None:
        return HelpTopic(title=title, body=specific)
    if isinstance(control, ft.TextField):
        body = (
            "Sets this parameter for the next calculation. Enter a valid value; "
            "larger values increase the setting and smaller values reduce it."
        )
    elif isinstance(control, (ft.Dropdown, ft.RadioGroup)):
        body = (
            "Chooses how the current result is calculated or displayed. "
            "Select the option that matches the result you want to inspect."
        )
    elif isinstance(control, (ft.Checkbox, ft.Switch)):
        body = (
            "Turns this option on or off. The enabled state applies the option "
            "to the current display or the next calculation."
        )
    elif isinstance(control, ft.Slider):
        body = (
            "Adjusts this display setting. Moving right increases the value; "
            "moving left decreases it."
        )
    else:
        body = f"Selects {title} for the current operation or display."
    return HelpTopic(title=title, body=body)


class ContextHelpController:
    """Own the help-mode lifecycle and restore intercepted controls exactly."""

    ACTIVE_COLOR = ft.Colors.ORANGE_400
    INACTIVE_COLOR = ft.Colors.WHITE

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.active = False
        self._scopes: list[tuple[ft.Control, str]] = []
        self._intercepted: list[_InterceptedControl] = []
        self._buttons: list[ft.IconButton] = []
        self._dialog: ft.AlertDialog | None = None

    def add_scope(self, root: ft.Control, *, area: str) -> None:
        """Register one layout subtree as a source of contextual help."""
        self._scopes.append((root, area))

    def create_button(self) -> ft.IconButton:
        button = ft.IconButton(
            icon=ft.Icons.HELP_OUTLINE,
            icon_color=self.INACTIVE_COLOR,
            tooltip="Context help",
            on_click=lambda _: self.toggle(),
        )
        setattr(button, "_context_help_excluded", True)
        self._buttons.append(button)
        return button

    def toggle(self) -> None:
        if self.active:
            self.deactivate()
        else:
            self.activate()

    def show_text(self, body: str) -> None:
        """Open purpose-written help from a control with custom click handling."""
        if self.active:
            self._open_help(HelpTopic(title="", body=body))

    def activate(self) -> None:
        if self.active:
            return
        self.active = True
        seen: set[int] = set()
        for root, area in self._scopes:
            self._intercept_tree(root, area=area, context=(), seen=seen)
        self._sync_button_colors()
        self._safe_page_update()

    def deactivate(self, *, close_dialog: bool = True) -> None:
        if close_dialog and self._dialog is not None:
            self.page.close(self._dialog)
        self._dialog = None
        for record in reversed(self._intercepted):
            for event_name, handler in record.handlers.items():
                setattr(record.control, event_name, handler)
            for property_name, value in record.properties.items():
                setattr(record.control, property_name, value)
        self._intercepted.clear()
        self.active = False
        self._sync_button_colors()
        self._safe_page_update()

    def _intercept_tree(
        self,
        control: ft.Control,
        *,
        area: str,
        context: tuple[str, ...],
        seen: set[int],
    ) -> None:
        if id(control) in seen or getattr(control, "_context_help_excluded", False):
            return
        seen.add(id(control))
        if control.visible is False:
            return

        local_labels = tuple(_direct_labels(control))
        child_context = local_labels or context
        event_names = self._help_event_names(control)
        if event_names:
            self._intercept(
                control,
                event_names,
                area=area,
                context=context,
                suppress=bool(getattr(control, "_context_help_suppress", False)),
            )

        for child in control._get_children():
            self._intercept_tree(
                child,
                area=area,
                context=child_context,
                seen=seen,
            )

    @staticmethod
    def _help_event_names(control: ft.Control) -> tuple[str, ...]:
        if isinstance(control, ft.TextField):
            return ("on_click", "on_change", "on_submit")
        if isinstance(control, ft.Tabs):
            return ("on_click", "on_change")
        if isinstance(control, ft.PopupMenuButton):
            return ("on_open",)

        names: list[str] = []
        for event_name in ("on_click", "on_change", "on_tap"):
            if (
                hasattr(control, event_name)
                and getattr(control, event_name) is not None
            ):
                names.append(event_name)
        return tuple(names)

    def _intercept(
        self,
        control: ft.Control,
        event_names: Iterable[str],
        *,
        area: str,
        context: tuple[str, ...],
        suppress: bool,
    ) -> None:
        record = _InterceptedControl(control=control)
        property_names = ["disabled", "value", "selected_index"]
        if isinstance(control, ft.TextField):
            property_names.extend(("read_only", "always_call_on_tap"))
        for property_name in property_names:
            if hasattr(control, property_name):
                record.properties[property_name] = getattr(control, property_name)
        if "disabled" in record.properties:
            control.disabled = False
        if isinstance(control, ft.TextField):
            # Editable fields use the first tap to request focus. Read-only mode
            # lets that same tap reach on_click immediately while help is active.
            control.read_only = True
            control.always_call_on_tap = True

        def handle_help_event(event: Any) -> None:
            topic = resolve_help_topic(
                control,
                area=area,
                context=context,
                event=event,
            )
            for property_name, value in record.properties.items():
                if property_name != "disabled":
                    setattr(control, property_name, value)
            if not suppress:
                self._open_help(topic)

        for event_name in event_names:
            record.handlers[event_name] = getattr(control, event_name)
            setattr(control, event_name, handle_help_event)
        self._intercepted.append(record)

    def _open_help(self, topic: HelpTopic) -> None:
        if not self.active or self._dialog is not None:
            return

        def close_help(_: ft.ControlEvent) -> None:
            dialog = self._dialog
            self._dialog = None
            if dialog is not None:
                self.page.close(dialog)
            self.deactivate(close_dialog=False)

        self._dialog = ft.AlertDialog(
            modal=True,
            bgcolor="#ff333333",
            surface_tint_color=ft.Colors.TRANSPARENT,
            content=ft.SelectionArea(CustomText(topic.body)),
            actions=[
                ft.TextButton(
                    "Close",
                    style=ft.ButtonStyle(color=ft.Colors.LIGHT_BLUE_200),
                    on_click=close_help,
                )
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.open(self._dialog)

    def _sync_button_colors(self) -> None:
        color = self.ACTIVE_COLOR if self.active else self.INACTIVE_COLOR
        for button in self._buttons:
            button.icon_color = color

    def _safe_page_update(self) -> None:
        try:
            self.page.update()
        except AssertionError:
            # Unit tests and controls built before mounting do not have a page.
            pass
