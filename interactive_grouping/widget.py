"""Interactive grouping widget (controller).

:class:`InteractiveBinning` wires the model (:class:`~.session.BinningSession`), the
figure backend (:mod:`.plotting`) and the presentation helpers (:mod:`.ui_components`)
together. The layout follows SAS Enterprise Miner "Interactive Grouping": a variable
selector, a *Groupings* tab with the fine and group charts, the grouping table
(Fine / Coarse detail level) and the variable statistics, and a *Variables* tab.

Editing happens at the *Fine* detail level only; the *Coarse* level is a read-only
overview of the groups. Edits never change the applied grouping directly: they
produce a *new* (draft) grouping per variable that is committed with **Apply**.
"""

from __future__ import annotations

import functools
import html
import json
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from . import ui_components as ui
from .exceptions import BinningEditError
from .operations import parse_cutoffs, split_fine_bin, suggest_fine_split
from .plotting import FigureBackend, get_backend, shorten
from .session import BinningSession, VariableRole
from .strategies import NEW_GROUP, NEW_GROUP_BELOW, FineLevel
from .variable import TREND_OPTIONS, FitResult, VariableBinning

__all__ = ["InteractiveBinning"]

FINE = "Fine"
COARSE = "Coarse"


@dataclass
class _VariableState:
    """Per-variable UI state that is not part of the model."""

    fine_bins: int
    draft: FitResult | None = None
    extra_cutoffs: set[float] = field(default_factory=set)


def _action(level: str | None = None) -> Callable:
    """Decorate a UI callback.

    * ignores calls caused by programmatic widget updates (see ``_updating``);
    * optionally requires the grouping table to be at detail ``level``;
    * shows :class:`BinningEditError` (and unexpected errors) in the message area -
      exceptions raised in ipywidgets callbacks would otherwise be lost in the
      browser console.
    """
    def decorate(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self: InteractiveBinning, *args: Any) -> Any:
            if self._updating:
                return None
            try:
                if level is not None and self.detail != level:
                    raise BinningEditError(f"Switch Detail Level to '{level}' for this action.")
                return method(self, *args)
            except BinningEditError as exc:
                self._show_message("err", html.escape(str(exc)))
            except Exception as exc:  # report anything, never fail silently
                details = html.escape(traceback.format_exc(limit=4))
                self._show_message("err", f"Unexpected error: {html.escape(repr(exc))}"
                                          f"<pre style='font-size:11px'>{details}</pre>")
            return None
        return wrapper
    return decorate


class InteractiveBinning:
    """Jupyter widget to inspect and manually edit optbinning groupings.

    Parameters
    ----------
    source : BinningProcess, BinningSession or dict of str to OptimalBinning
        Binning to start from. A ``BinningProcess`` is fitted on ``X, y`` if needed.
    X : pandas.DataFrame, optional
        Data of the binning (required unless ``source`` is a ``BinningSession``).
    y : array-like, optional
        Binary target (required unless ``source`` is a ``BinningSession``).
    backend : {"plotly", "matplotlib"}
        Interactive plotly figures (default) or static matplotlib images.
    plotly_js : {"cdn", "inline"}
        How plotly.js is loaded (plotly backend only); ``"inline"`` works offline.
    fine_bins : int
        Default number of fine bins of numerical variables.
    figure_height : int
        Height of the charts in pixels.
    advanced : bool, default False
        Show the *Advanced* block: number of fine bins, typing all group cutoffs,
        the optbinning optimizer and the full optbinning binning table.

        .. warning::
           Experimental and **untested** in real modelling workflows: it is
           recommended to check the resulting groupings (e.g. in the Coarse view and
           the Variable Statistics) before using them.

    Examples
    --------
    >>> w = InteractiveBinning(binning_process, X, y)  # doctest: +SKIP
    >>> w                                              # doctest: +SKIP
    >>> w.to_binning_process()                         # doctest: +SKIP
    """

    def __init__(self, source: Any, X: pd.DataFrame | None = None, y: Any = None,
                 backend: str = "plotly", plotly_js: str = "cdn", fine_bins: int = 20,
                 figure_height: int = 380, advanced: bool = False) -> None:
        self.session = self._make_session(source, X, y)
        self.backend_name = backend
        self.backend: FigureBackend = get_backend(
            backend, **({"plotly_js": plotly_js} if backend == "plotly" else {}))
        self.figure_height = figure_height
        self.advanced = advanced
        self._states = {name: _VariableState(fine_bins=fine_bins) for name in self.session}
        self._fine_level: FineLevel | None = None
        self._updating = False
        self._build_ui()
        self._refresh_all()

    @staticmethod
    def _make_session(source: Any, X: pd.DataFrame | None, y: Any) -> BinningSession:
        if isinstance(source, BinningSession):
            return source
        if X is None or y is None:
            raise ValueError("X and y are required.")
        if isinstance(source, dict):
            return BinningSession.from_optbinnings(source, X, y)
        return BinningSession.from_binning_process(source, X, y)

    @classmethod
    def from_data(cls, X: pd.DataFrame, y: Any, variable_names: list[str] | None = None,
                  categorical_variables: list[str] | None = None, *,
                  backend: str = "plotly", plotly_js: str = "cdn", fine_bins: int = 20,
                  advanced: bool = False,
                  **binning_process_kwargs: Any) -> InteractiveBinning:
        """Fit an optbinning ``BinningProcess`` on ``X, y`` and open the widget."""
        session = BinningSession.fit(X, y, variable_names=variable_names,
                                     categorical_variables=categorical_variables,
                                     **binning_process_kwargs)
        return cls(session, backend=backend, plotly_js=plotly_js, fine_bins=fine_bins,
                   advanced=advanced)

    # ================================================================== public API
    def summary(self) -> pd.DataFrame:
        """Summary of all variables (see :meth:`BinningSession.summary`)."""
        return self.session.summary()

    def splits(self) -> dict[str, list]:
        """Return the applied splits of every variable."""
        return self.session.splits()

    def binning_fit_params(self, only_edited: bool = False) -> dict[str, dict]:
        """``binning_fit_params`` reproducing the applied groupings."""
        return self.session.binning_fit_params(only_edited=only_edited)

    def to_binning_process(self, fit: bool = True, include_rejected: bool = False):
        """Optbinning ``BinningProcess`` with the applied groupings."""
        return self.session.to_binning_process(fit=fit, include_rejected=include_rejected)

    def transform(self, X: pd.DataFrame, metric: str = "woe", include_rejected: bool = False,
                  **kwargs: Any) -> pd.DataFrame:
        """Transform ``X`` with the applied groupings (WoE by default)."""
        return self.session.transform(X, metric=metric, include_rejected=include_rejected,
                                      **kwargs)

    def save_json(self, path: str) -> None:
        """Save the applied groupings and roles to a JSON file."""
        self.session.save_json(path)

    def load_json(self, path: str) -> list[str]:
        """Load groupings saved by :meth:`save_json`; unapplied edits are discarded."""
        problems = self.session.load_json(path)
        for state in self._states.values():
            state.draft = None
        self._refresh_all()
        return problems

    @property
    def var(self) -> VariableBinning:
        """The selected variable."""
        return self.session[self.var_select.value]

    @property
    def state(self) -> _VariableState:
        """UI state of the selected variable."""
        return self._states[self.var_select.value]

    @property
    def draft(self) -> FitResult:
        """New (not yet applied) grouping of the selected variable, else the applied one."""
        return self.state.draft or self.var.current

    @property
    def has_draft(self) -> bool:
        """Whether the selected variable has an unapplied grouping."""
        return self.state.draft is not None

    @property
    def detail(self) -> str:
        """Detail level of the grouping table: ``"Fine"`` or ``"Coarse"``."""
        return self.detail_toggle.value

    @property
    def fine_level(self) -> FineLevel:
        """Fine-level view currently shown."""
        return self._fine_level

    def show(self) -> None:
        """Display the widget."""
        display(self.ui)

    def _ipython_display_(self) -> None:
        display(self.ui)

    # ================================================================== layout
    def _build_ui(self) -> None:
        self.tabs = widgets.Tab(children=[self._build_variables_tab(),
                                          self._build_groupings_tab()])
        self.tabs.set_title(0, "Variables")
        self.tabs.set_title(1, "Groupings")
        self.tabs.selected_index = 1
        title = widgets.HTML("Interactive Grouping · optbinning")
        title.add_class("ig-title")
        self.ui = widgets.VBox([widgets.HTML(ui.CSS), title, self._build_selector(), self.tabs],
                               layout=widgets.Layout(width="100%"))
        self.ui.add_class("ig-root")
        self._bind_events()

    def _build_selector(self) -> widgets.Widget:
        self.var_select = widgets.Dropdown(options=self.session.names,
                                           layout=widgets.Layout(flex="1 1 auto", width="auto"))
        self.btn_prev = ui.button("◀ Previous", tooltip="Previous variable")
        self.btn_next = ui.button("Next ▶", tooltip="Next variable")
        label = widgets.HTML("<b>Selected Variable</b>", layout=widgets.Layout(width="140px"))
        return ui.card("Variable Selection", widgets.HBox(
            [label, self.var_select, self.btn_prev, self.btn_next],
            layout=widgets.Layout(width="100%")))

    def _build_groupings_tab(self) -> widgets.Widget:
        reset_all_row = widgets.HBox([self._build_reset_all()], layout=widgets.Layout(
            justify_content="flex-end", margin="10px 0 0 0"))
        # The advanced widgets are always created (the refresh code updates them), but
        # only shown with ``advanced=True``. EXPERIMENTAL / UNTESTED in real workflows:
        # check the groupings they produce.
        self.advanced_panel = self._build_advanced()
        children = [self._build_charts(), self._build_editor(),
                    *([self.advanced_panel] if self.advanced else []), reset_all_row]
        return widgets.VBox(children, layout=widgets.Layout(width="100%"))

    def _build_charts(self) -> widgets.Widget:
        self.group_header = widgets.HTML()
        self.fig_fine = self.backend.create_view()
        self.fig_groups = self.backend.create_view()
        # Side by side on wide screens, stacked on narrow ones (flex-wrap).
        fine_box = widgets.Box([self.fig_fine],
                               layout=widgets.Layout(flex="3 1 520px", min_width="0"))
        groups_box = widgets.Box([self.fig_groups],
                                 layout=widgets.Layout(flex="2 1 380px", min_width="0"))
        charts = widgets.HBox([fine_box, groups_box],
                              layout=widgets.Layout(width="100%", flex_flow="row wrap"))
        return ui.card(self.group_header, charts)

    def _build_editor(self) -> widgets.Widget:
        """Build the grouping table with its toolbars and the statistics panel."""
        self.rows_select = widgets.SelectMultiple(options=[], rows=18,
                                                  layout=widgets.Layout(width="100%"))
        self.rows_select.add_class("ig-mono")

        # Fine level: move the selected fine bins to another / a new group.
        self.assign_group = widgets.Dropdown(options=[], layout=widgets.Layout(width="340px"))
        self.btn_assign = ui.button("Assign", "info", "Move the selected fine bins / categories "
                                                      "to the chosen group")
        self.assign_hint = widgets.HTML()
        self.assign_tools = widgets.HBox([widgets.HTML("<b>Assign selected to group</b>&nbsp;"),
                                          self.assign_group, self.btn_assign, self.assign_hint])
        # Fine level, numerical variables: split one fine bin (the groups do not change).
        self.split_value = widgets.FloatText(description="at",
                                             layout=widgets.Layout(width="170px"))
        self.btn_split_fine = ui.button("Split fine bin", "", "Split the selected fine bin in "
                                        "two; both parts stay in the same group")
        self.split_tools = widgets.HBox([widgets.HTML("<b>Split selected fine bin</b>&nbsp;"),
                                         self.split_value, self.btn_split_fine])
        for toolbar in (self.assign_tools, self.split_tools):
            toolbar.add_class("ig-toolbar")
        # Coarse level: read-only overview.
        self.coarse_note = widgets.HTML(ui.message(
            "info", "Coarse view is read-only: switch Detail Level to <b>Fine</b> to edit."))

        self.message = widgets.HTML()
        table_card = ui.card("Grouping table — select rows with Ctrl/Shift+click",
                             self.rows_select, self.assign_tools, self.split_tools,
                             self.coarse_note, self.message,
                             layout=widgets.Layout(flex="2 1 560px", min_width="0",
                                                   margin="0 12px 16px 0"))
        return widgets.HBox([table_card, self._build_statistics_card()],
                            layout=widgets.Layout(width="100%", flex_flow="row wrap",
                                                  align_items="flex-start"))

    def _build_statistics_card(self) -> widgets.Widget:
        self.stats_html = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.detail_toggle = widgets.ToggleButtons(options=[FINE, COARSE], value=FINE)
        self.detail_toggle.style.button_width = "90px"
        self.role_select = widgets.Dropdown(
            options=[("Input", VariableRole.INPUT), ("Rejected", VariableRole.REJECTED)],
            layout=widgets.Layout(width="160px"))
        self.btn_apply = ui.button("✓ Apply", "success", "Recompute the variable with optbinning "
                                   "using the new grouping and apply it", width="48%")
        self.btn_discard = ui.button("✗ Discard", "warning", "Drop the new grouping", width="48%")
        self.btn_undo = ui.button("↶ Undo", "", "Undo the last applied change", width="48%")
        self.btn_redo = ui.button("↷ Redo", "", "Redo", width="48%")
        self.btn_reset = ui.button("⟲ Reset variable", "danger",
                                   "Back to the original optbinning grouping (undoable)",
                                   width="98%")
        row = widgets.Layout(width="100%")
        return ui.card(
            "Variable Statistics", self.stats_html,
            ui.panel_title("Detail Level"), self.detail_toggle,
            ui.panel_title("Variable Role"), self.role_select,
            ui.panel_title("Changes"),
            widgets.HBox([self.btn_apply, self.btn_discard], layout=row),
            widgets.HBox([self.btn_undo, self.btn_redo], layout=row),
            self.btn_reset,
            layout=widgets.Layout(flex="1 1 330px", min_width="0"))

    def _build_advanced(self) -> widgets.Widget:
        self.fine_bins_input = widgets.BoundedIntText(min=2, max=200, description="Fine bins",
                                                      layout=widgets.Layout(width="170px"))
        self.btn_fine_bins = ui.button("Rebuild fine bins", "", "Number of ~equal-frequency fine "
                                       "bins (group boundaries are always kept)")
        self.cutoffs_text = widgets.Text(description="Cutoffs", placeholder="e.g. 25, 35.5, 60",
                                         layout=widgets.Layout(width="70%"))
        self.btn_set_cutoffs = ui.button("Set group cutoffs", "", "Replace all group boundaries")
        self.opt_trend = widgets.Dropdown(options=[(str(t) if t else "none", t)
                                                   for t in TREND_OPTIONS],
                                          value="auto", description="Trend",
                                          layout=widgets.Layout(width="240px"))
        self.opt_max_bins = widgets.BoundedIntText(value=0, min=0, max=100,
                                                   description="Max groups",
                                                   layout=widgets.Layout(width="170px"))
        self.opt_min_size = widgets.BoundedFloatText(value=0.0, min=0, max=0.5, step=0.01,
                                                     description="Min group %",
                                                     layout=widgets.Layout(width="170px"))
        self.opt_only_merge = widgets.Checkbox(value=True, indent=False,
                                               description="only merge current groups")
        self.btn_optimize = ui.button("Run optimizer → new grouping", "primary",
                                      "Run optbinning's optimizer; the result becomes the new "
                                      "grouping")
        self.binning_table = widgets.HTML(layout=widgets.Layout(width="100%"))
        options = widgets.VBox([
            widgets.HBox([self.fine_bins_input, self.btn_fine_bins]),
            widgets.HBox([self.cutoffs_text, self.btn_set_cutoffs],
                         layout=widgets.Layout(width="100%")),
            widgets.HTML("<div class='ig-sub'>optbinning optimizer ('Max groups' 0 = no limit, "
                         "'Min group %' as a fraction, e.g. 0.05)</div>"),
            widgets.HBox([self.opt_trend, self.opt_max_bins, self.opt_min_size],
                         layout=widgets.Layout(flex_flow="row wrap")),
            widgets.HBox([self.opt_only_merge, self.btn_optimize]),
        ])
        accordion = widgets.Accordion(children=[options, self.binning_table],
                                      selected_index=None)
        accordion.set_title(0, "Advanced: fine bins, cutoffs, optimizer")
        accordion.set_title(1, "optbinning binning table (new grouping)")
        return accordion

    def _build_reset_all(self) -> widgets.Widget:
        self.btn_reset_all = ui.button("Reset All Changes", "danger",
                                       "Reset every variable to its original optbinning grouping")
        return self.btn_reset_all

    def _build_variables_tab(self) -> widgets.Widget:
        self.summary_html = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.path_text = widgets.Text(value="binning_splits.json", description="File",
                                      layout=widgets.Layout(width="420px"))
        self.btn_save = ui.button("Save JSON", "info")
        self.btn_load = ui.button("Load JSON", "")
        self.export_message = widgets.HTML()
        self.export_text = widgets.Textarea(layout=widgets.Layout(width="100%", height="200px"))
        help_text = ui.message(
            "info", "In Python: <code>w.to_binning_process()</code> → fitted "
            "<code>optbinning.BinningProcess</code> with your groupings (variables with role "
            "<i>Rejected</i> are left out; ready for <code>Scorecard</code>); "
            "<code>w.transform(X)</code> → WoE; <code>w.binning_fit_params()</code>; "
            "<code>w.save_json(path)</code> / <code>w.load_json(path)</code>.")
        export = ui.card("Export", widgets.HTML(help_text),
                         widgets.HBox([self.path_text, self.btn_save, self.btn_load]),
                         self.export_message,
                         widgets.HTML("<div class='ig-sub'>Applied groupings (JSON)</div>"),
                         self.export_text)
        return widgets.VBox([ui.card("Variables", self.summary_html), export],
                            layout=widgets.Layout(width="100%"))

    def _bind_events(self) -> None:
        """Connect widget events to the callbacks (one table instead of many calls)."""
        observers = {
            self.var_select: self._on_variable_change,
            self.detail_toggle: self._on_detail_change,
            self.role_select: self._on_role_change,
            self.rows_select: self._on_rows_change,
            self.tabs: self._on_tab_change,
        }
        for widget, handler in observers.items():
            trait = "selected_index" if widget is self.tabs else "value"
            widget.observe(handler, names=trait)
        clicks = {
            self.btn_prev: lambda _: self._step_variable(-1),
            self.btn_next: lambda _: self._step_variable(+1),
            self.btn_assign: self._on_assign,
            self.btn_split_fine: self._on_split_fine,
            self.btn_fine_bins: self._on_fine_bins,
            self.btn_set_cutoffs: self._on_set_cutoffs,
            self.btn_optimize: self._on_optimize,
            self.btn_apply: self._on_apply,
            self.btn_discard: self._on_discard,
            self.btn_undo: self._on_undo,
            self.btn_redo: self._on_redo,
            self.btn_reset: self._on_reset,
            self.btn_reset_all: self._on_reset_all,
            self.btn_save: self._on_save,
            self.btn_load: self._on_load,
        }
        for btn, handler in clicks.items():
            btn.on_click(handler)

    # ================================================================== refresh
    @contextmanager
    def _updating_widgets(self):
        """Suppress callbacks while widget values are changed programmatically."""
        previous, self._updating = self._updating, True
        try:
            yield
        finally:
            self._updating = previous

    def _show_message(self, kind: str, text: str) -> None:
        self.message.value = ui.message(kind, text) if text else ""

    def _refresh_all(self) -> None:
        self._refresh_selector()
        self._refresh_variable()
        if self.tabs.selected_index == 0:
            self._refresh_summary()

    def _variable_label(self, position: int, name: str) -> str:
        var = self.session[name]
        parts = [f"{position + 1} : {name}", var.dtype, f"IV {var.current.iv:.3f}",
                 f"Groups {var.current.n_bins}"]
        if self.session.roles[name] == VariableRole.REJECTED:
            parts.append("rejected")
        if var.is_edited:
            parts.append("✎ edited")
        if self._states[name].draft is not None:
            parts.append("● new grouping not applied")
        return " | ".join(parts)

    def _refresh_selector(self) -> None:
        with self._updating_widgets():
            selected = self.var_select.value
            self.var_select.options = [(self._variable_label(i, n), n)
                                       for i, n in enumerate(self.session.names)]
            self.var_select.value = selected

    def _refresh_variable(self) -> None:
        """Redraw everything that depends on the selected variable."""
        var, state, draft = self.var, self.state, self.draft
        self._fine_level = var.fine_level(draft.splits, state.fine_bins, state.extra_cutoffs)
        self._refresh_header()
        self._refresh_charts()
        self._refresh_table()
        self._refresh_controls()
        self.stats_html.value = ui.statistics_html(var.original, var.current, draft,
                                                   var.strategy.ordered)
        self.binning_table.value = ui.binning_table_html(draft.table, draft.iv)

    def _refresh_header(self) -> None:
        name = self.var.name
        badges = ""
        if self.has_draft:
            badges += ui.badge("pending", "new grouping — not applied")
        if self.session.roles[name] == VariableRole.REJECTED:
            badges += ui.badge("rejected", "rejected")
        self.group_header.value = (f"{html.escape(name)} <span style='opacity:.8;font-size:13px'>"
                                   f"({self.var.dtype})</span>{badges}")

    def _refresh_charts(self) -> None:
        draft, fine = self.draft, self._fine_level
        group_woe = dict(enumerate(draft.regular["woe"].astype(float)))
        self.backend.render(self.fig_fine, self.backend.fine_figure(
            fine.table, group_woe, self.var.strategy.ordered,
            title=f"Fine bins ({len(fine.regular)}) colored by group",
            height=self.figure_height))
        self.backend.render(self.fig_groups, self.backend.groups_figure(
            draft.table, title=f"Groups ({draft.n_bins})" + (" — new" if self.has_draft else ""),
            height=self.figure_height))

    def _refresh_table(self) -> None:
        options = (ui.fine_options(self._fine_level.table) if self.detail == FINE
                   else ui.coarse_options(self.draft.table))
        with self._updating_widgets():
            self.rows_select.options = options
            self.rows_select.value = ()
            self.role_select.value = self.session.roles[self.var.name]
            self.fine_bins_input.value = self.state.fine_bins
        self._refresh_targets()

    def _refresh_controls(self) -> None:
        """Show the toolbars that fit the detail level and the variable type."""
        var, numerical = self.var, self.var.strategy.ordered
        fine = self.detail == FINE
        _set_visible(self.assign_tools, fine)
        _set_visible(self.split_tools, fine and numerical)
        _set_visible(self.coarse_note, not fine)
        for widget in (self.fine_bins_input, self.btn_fine_bins, self.cutoffs_text,
                       self.btn_set_cutoffs):
            widget.disabled = not numerical
        if numerical:
            self.cutoffs_text.value = ", ".join(f"{s:.10g}" for s in self.draft.splits)
        self.btn_apply.disabled = self.btn_discard.disabled = not self.has_draft
        self.btn_undo.disabled = not var.can_undo
        self.btn_redo.disabled = not var.can_redo
        self.btn_reset.disabled = not (var.is_edited or self.has_draft)

    def _refresh_targets(self) -> None:
        """Fill "Assign selected to group" with the valid targets of the selection.

        Numerical variables: the previous / next group and a new group above / below
        the selection (see :func:`~.operations.move_to_neighbour` and
        :func:`~.operations.split_group_at`); categorical variables: any other group or
        a new one. Only targets that change the grouping are offered.
        """
        options: list[tuple[str, int]] = []
        hint = ""
        if self.detail == FINE:
            try:
                rows = self._selected_rows()
            except BinningEditError as exc:
                hint = str(exc)
            else:
                options = self._target_options(rows)
                hint = (self._target_hint(rows) if options
                        else "no valid target for this selection")
        with self._updating_widgets():
            self.assign_group.options = options
            self.assign_group.value = options[0][1] if options else None
        self.assign_group.disabled = self.btn_assign.disabled = not options
        self.assign_hint.value = (f"<span class='ig-note'>&nbsp;{html.escape(hint)}</span>"
                                  if hint else "")

    def _target_options(self, rows: list[int]) -> list[tuple[str, int]]:
        draft, ordered = self.draft, self.var.strategy.ordered
        labels = draft.regular["bin"]
        selected_group = int(self._fine_level.table.loc[rows[0], "group"])
        new_number = draft.n_bins + 1
        options = []
        for target in self.var.strategy.fine_targets(draft.splits, self._fine_level, rows):
            if target == NEW_GROUP:
                text = (f"{new_number}: new group — this and higher fine bins" if ordered
                        else f"{new_number}: new group")
            elif target == NEW_GROUP_BELOW:
                text = f"{new_number}: new group — this and lower fine bins"
            else:
                side = ""
                if ordered:
                    side = " (previous)" if target < selected_group else " (next)"
                text = f"{target + 1}{side}: {shorten(labels[target], 24)}"
            options.append((text, target))
        return options

    def _target_hint(self, rows: list[int]) -> str:
        if not self.var.strategy.ordered:
            return "any group or a new one"
        return ("previous / next group also takes the fine bins up to the group edge; "
                "a new group splits the group here")

    def _refresh_summary(self) -> None:
        pending = {name for name, state in self._states.items() if state.draft is not None}
        self.summary_html.value = ui.variables_html(self.session.summary(), pending)
        self.export_text.value = json.dumps(self.session.to_dict(), ensure_ascii=False,
                                            indent=1, default=str)

    # ================================================================== edit helpers
    def _selected_rows(self) -> list[int]:
        """Return the selected regular rows of the grouping table (header ignored)."""
        selection = [v for v in self.rows_select.value if v != ui.HEADER_VALUE]
        if any(isinstance(v, str) for v in selection):
            raise BinningEditError("Missing / special bins cannot be regrouped: optbinning "
                                   "always keeps them as separate bins.")
        if not selection:
            raise BinningEditError("Select rows in the table first (Ctrl/Shift+click).")
        return sorted(int(v) for v in selection)

    def _propose(self, splits: list, message: str) -> None:
        """Fit ``splits`` with optbinning and show them as the new grouping."""
        self._set_draft(self.var.fit_splits(splits), message)

    def _set_draft(self, result: FitResult, message: str) -> None:
        if self.var.strategy.equal(result.splits, self.var.current.splits):
            self.state.draft = None
            message += " (same as the applied grouping)"
        else:
            self.state.draft = result
        self._refresh_selector()
        self._refresh_variable()
        self._show_message("ok", message)

    def _require_numerical(self) -> None:
        if not self.var.strategy.ordered:
            raise BinningEditError("This action is for numerical variables only.")

    def _group_of_label(self, label: str) -> int | None:
        """1-based group of the fine row with ``label`` in the shown fine level."""
        rows = self._fine_level.regular
        match = rows.index[rows["label"] == label]
        return int(rows.loc[match[0], "group"]) + 1 if len(match) else None

    # ================================================================== callbacks
    @_action()
    def _on_variable_change(self, _change: dict) -> None:
        self._show_message("", "")
        self._refresh_variable()

    @_action()
    def _step_variable(self, step: int) -> None:
        names = self.session.names
        self.var_select.value = names[(names.index(self.var_select.value) + step) % len(names)]

    @_action()
    def _on_tab_change(self, change: dict) -> None:
        if change["new"] == 0:
            self._refresh_summary()

    @_action()
    def _on_detail_change(self, _change: dict) -> None:
        self._refresh_table()
        self._refresh_controls()

    @_action()
    def _on_role_change(self, change: dict) -> None:
        role = change["new"]
        self.session.set_role(self.var.name, role)
        self._refresh_selector()
        self._refresh_header()
        rejected = role == VariableRole.REJECTED
        note = " – the variable is left out of the export." if rejected else "."
        self._show_message("ok", f"Role set to <b>{role}</b>{note}")

    @_action()
    def _on_rows_change(self, _change: dict) -> None:
        self._refresh_targets()
        self._suggest_fine_split()

    def _suggest_fine_split(self) -> None:
        """Pre-fill the split value with the median of a single selected fine bin."""
        selection = [v for v in self.rows_select.value if isinstance(v, int)]
        if self.detail != FINE or not self.var.strategy.ordered or len(selection) != 1:
            return
        row = self._fine_level.table.loc[selection[0]]
        suggestion = suggest_fine_split(self.var.x_clean, row["lo"], row["hi"])
        if suggestion is not None:
            self.split_value.value = suggestion

    @_action(level=FINE)
    def _on_assign(self, _button: widgets.Button) -> None:
        rows = self._selected_rows()
        target = self.assign_group.value
        if target is None:
            raise BinningEditError("Choose a target group.")
        draft = self.draft
        label = self._fine_level.table.loc[rows[0], "label"]
        new = self.var.strategy.fine_assign(draft.splits, self._fine_level, rows, int(target))
        is_new = target in (NEW_GROUP, NEW_GROUP_BELOW)
        what = f"fine bin {rows[0] + 1}" if len(rows) == 1 else f"{len(rows)} fine bins"
        where = f"a new group ({draft.n_bins + 1})" if is_new else f"group {target + 1}"
        merged = ("" if is_new or self.var.strategy.n_groups(new) >= draft.n_bins
                  else " The source group became empty, so the groups merged.")
        self._propose(new, f"Assigned {what} to {where}.{merged}")
        # Groups are renumbered after the recomputation (numerical: by value order,
        # categorical: optbinning sorts them by event rate) - report the final number.
        final = self._group_of_label(label)
        if final is not None and (is_new or final != target + 1):
            self._show_message("ok", f"Assigned {what} to {where} — after renumbering it is "
                                     f"group {final}.{merged}")

    @_action(level=FINE)
    def _on_split_fine(self, _button: widgets.Button) -> None:
        self._require_numerical()
        rows = self._selected_rows()
        if len(rows) != 1:
            raise BinningEditError("Select exactly one fine bin to split.")
        row = self._fine_level.table.loc[rows[0]]
        value = split_fine_bin(self.var.x_clean, row["lo"], row["hi"], self.split_value.value)
        # Only the fine level changes: the value is kept as a fine-bin boundary.
        self.state.extra_cutoffs.add(value)
        self._refresh_variable()
        self._show_message("ok", f"Fine bin {rows[0] + 1} split at {value:g}; "
                                 "the groups did not change.")

    @_action()
    def _on_fine_bins(self, _button: widgets.Button) -> None:
        self.state.fine_bins = int(self.fine_bins_input.value)
        self._refresh_variable()
        self._show_message("ok", f"Rebuilt with ~{self.state.fine_bins} fine bins.")

    @_action()
    def _on_set_cutoffs(self, _button: widgets.Button) -> None:
        self._require_numerical()
        self._propose(parse_cutoffs(self.cutoffs_text.value, self.var.x_range),
                      "Group cutoffs replaced.")

    @_action()
    def _on_optimize(self, _button: widgets.Button) -> None:
        overrides: dict[str, Any] = {"monotonic_trend": self.opt_trend.value}
        if self.opt_max_bins.value:
            overrides["max_n_bins"] = int(self.opt_max_bins.value)
        if self.opt_min_size.value:
            overrides["min_bin_size"] = float(self.opt_min_size.value)
        start = self.draft.splits if self.opt_only_merge.value else None
        self._set_draft(self.var.optimize(start_splits=start, **overrides),
                        "Optimizer result loaded as the new grouping.")

    @_action()
    def _on_apply(self, _button: widgets.Button) -> None:
        if not self.has_draft:
            raise BinningEditError("Nothing to apply.")
        var, old_iv = self.var, self.var.current.iv
        var.commit(self.state.draft)
        self.state.draft = None
        self._refresh_all()
        self._show_message("ok", f"Applied. {var.name}: IV {old_iv:.4f} → {var.current.iv:.4f}, "
                                 f"{var.current.n_bins} groups.")

    @_action()
    def _on_discard(self, _button: widgets.Button) -> None:
        self.state.draft = None
        self._refresh_selector()
        self._refresh_variable()
        self._show_message("ok", "New grouping discarded.")

    def _history_step(self, step: Callable[[], bool], done: str, nothing: str) -> None:
        """Run an undo / redo / reset step; unapplied edits of the variable are dropped."""
        self.state.draft = None
        changed = step()
        self._refresh_all()
        self._show_message("ok" if changed else "info", done if changed else nothing)

    @_action()
    def _on_undo(self, _button: widgets.Button) -> None:
        self._history_step(self.var.undo, "Undone.", "Nothing to undo.")

    @_action()
    def _on_redo(self, _button: widgets.Button) -> None:
        self._history_step(self.var.redo, "Redone.", "Nothing to redo.")

    @_action()
    def _on_reset(self, _button: widgets.Button) -> None:
        self._history_step(self.var.reset, "Reset to the original optbinning grouping.",
                           "Already the original grouping.")

    @_action()
    def _on_reset_all(self, _button: widgets.Button) -> None:
        edited = [var for var in self.session.variables.values() if var.is_edited]
        for var in edited:
            var.reset()
        for state in self._states.values():
            state.draft = None
        self._refresh_all()
        self._show_message("ok", f"All changes reset ({len(edited)} variable(s); each can be "
                                 "undone with Undo).")

    @_action()
    def _on_save(self, _button: widgets.Button) -> None:
        path = self.path_text.value
        try:
            self.save_json(path)
        except OSError as exc:
            self.export_message.value = ui.message("err", html.escape(str(exc)))
            return
        self.export_message.value = ui.message("ok", f"Saved to {html.escape(path)}")

    @_action()
    def _on_load(self, _button: widgets.Button) -> None:
        path = self.path_text.value
        try:
            problems = self.load_json(path)
        except (OSError, ValueError) as exc:
            self.export_message.value = ui.message("err", html.escape(str(exc)))
            return
        self._refresh_summary()
        text = "<br>".join([f"Loaded {html.escape(path)}.", *map(html.escape, problems)])
        self.export_message.value = ui.message("warn" if problems else "ok", text)


def _set_visible(widget: widgets.Widget, visible: bool) -> None:
    widget.layout.display = "" if visible else "none"

