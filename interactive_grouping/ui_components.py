"""Presentation building blocks of the widget.

Stateless helpers only: the stylesheet, factories for cards and buttons, HTML tables
and the fixed-width rows of the grouping table. Keeping them apart from the widget
controller makes both easier to read and to test.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import ipywidgets as widgets
import numpy as np
import pandas as pd

from ._utils import RowKind
from .metrics import iv_strength

__all__ = [
    "CSS",
    "HEADER_VALUE",
    "badge",
    "binning_table_html",
    "button",
    "card",
    "coarse_options",
    "fine_options",
    "message",
    "panel_title",
    "statistics_html",
    "variables_html",
]

NBSP = " "
HEADER_VALUE = "__header__"
"""Value of the header row of the grouping table (never an edit target)."""

CSS = """
<style>
.ig-root {background:#cfd8e0; padding:18px; border-radius:6px; color:#0b0b0b;
          font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;}
.ig-root, .ig-root * {box-sizing:border-box;}
.ig-root .widget-vbox, .ig-root .widget-hbox {overflow-x:hidden;}
.ig-root .widget-html, .ig-root .widget-select-multiple {max-width:100%;}
.ig-root select, .ig-root input, .ig-root textarea {background:#fff !important;
          color:#0b0b0b !important;}
.ig-root .widget-label, .ig-root .widget-html-content {color:#0b0b0b;}
.ig-root .ig-title .widget-html-content {font-size:20px; font-weight:600; color:#1f3d4a;}
.ig-root .ig-card {background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.18); margin:0 0 16px 0;
          border-radius:3px; overflow:visible;}
.ig-root .ig-card-header {background:#4a8aa0; padding:10px 18px; margin:0; line-height:1.4;}
.ig-root .ig-card-header .widget-html-content {color:#fff; font-size:16px;}
.ig-root .ig-card-body {padding:12px 18px; overflow-x:hidden;}
.ig-root .ig-panel-title .widget-html-content {font-size:13px; font-weight:600;
          color:#1f3d4a; margin-top:6px;}
.ig-root .ig-toolbar {flex-wrap:wrap; gap:4px; align-items:center;}
.ig-root .ig-mono select {font-family:Consolas,Menlo,'DejaVu Sans Mono',monospace;
          font-size:12px;}
.ig-root .ig-mono select option:first-child {font-weight:700; color:#1f3d4a;
          background:#eef2f5;}
.ig-root .ig-sub {font-size:12px; color:#52514e; font-weight:600; margin-top:4px;}
.ig-badge {padding:1px 8px; border-radius:10px; font-size:12px; margin-left:8px;}
.ig-badge.pending {background:#fdecc8; color:#5a3d00;}
.ig-badge.rejected {background:#fbe3e3; color:#8c1d1d;}
.ig-msg {padding:6px 10px; border-radius:4px; font-size:13px; margin:4px 0;}
.ig-msg.ok {background:#e3f4e3; color:#1d5e1d;} .ig-msg.err {background:#fbe3e3; color:#8c1d1d;}
.ig-msg.info {background:#eef2f5; color:#33424a;} .ig-msg.warn {background:#fdf1d6; color:#6b4a00;}
.ig-note {font-size:11.5px; color:#52514e; margin-top:6px;}
table.ig-table {border-collapse:collapse; font-size:12.5px; width:100%; color:#0b0b0b;}
table.ig-table th {text-align:right; padding:5px 8px; border-bottom:2px solid #c9c8c3;
          color:#52514e; font-weight:600; white-space:nowrap; background:#fafaf9;}
table.ig-table td {text-align:right; padding:4px 8px; border-bottom:1px solid #eceae6;
          white-space:nowrap;}
table.ig-table .l {text-align:left; white-space:normal; max-width:420px;}
table.ig-table tr.special td {color:#52514e; font-style:italic;}
table.ig-table tr.total td {font-weight:600; border-top:2px solid #c9c8c3;}
table.ig-table tr.rejected td {opacity:.55;}
table.ig-table td.new, table.ig-table th.new {background:#eef4fb; font-weight:600;}
.ig-bar {display:inline-block; height:8px; background:#86b6ef; border-radius:2px;
         vertical-align:middle; margin-right:6px;}
</style>
"""


# --------------------------------------------------------------------------- #
# Widget factories
# --------------------------------------------------------------------------- #
def card(header: str | widgets.Widget, *children: widgets.Widget,
         layout: widgets.Layout | None = None) -> widgets.VBox:
    """Create a white card with a coloured header (interpretML / SAS look)."""
    header_widget = header if isinstance(header, widgets.Widget) else widgets.HTML(header)
    header_widget.add_class("ig-card-header")
    body = widgets.VBox(list(children))
    body.add_class("ig-card-body")
    box = widgets.VBox([header_widget, body], layout=layout or widgets.Layout(width="100%"))
    box.add_class("ig-card")
    return box


def button(description: str, style: str = "", tooltip: str = "",
           width: str = "auto") -> widgets.Button:
    """Create a button with consistent sizing."""
    return widgets.Button(description=description, button_style=style,
                          tooltip=tooltip or description,
                          layout=widgets.Layout(width=width, min_width="110px"))


def panel_title(text: str) -> widgets.HTML:
    """Create a small section title used inside cards."""
    title = widgets.HTML(text)
    title.add_class("ig-panel-title")
    return title


def message(kind: str, text: str) -> str:
    """HTML of a status message; ``kind`` is ``ok``, ``err``, ``warn`` or ``info``."""
    return f'<div class="ig-msg {kind}">{text}</div>'


def badge(kind: str, text: str) -> str:
    """HTML of a small badge shown in card headers (``pending`` / ``rejected``)."""
    return f"<span class='ig-badge {kind}'>{html.escape(text)}</span>"


def _fmt(value: Any, spec: str) -> str:
    """Format a number, ``-`` for missing / non-finite values."""
    try:
        return format(value, spec) if np.isfinite(float(value)) else "-"
    except (TypeError, ValueError):
        return "-"


def _bar(fraction: float, width: int = 60) -> str:
    return f'<span class="ig-bar" style="width:{width * fraction:.0f}px"></span>'


# --------------------------------------------------------------------------- #
# HTML tables
# --------------------------------------------------------------------------- #
def binning_table_html(table: pd.DataFrame, iv: float) -> str:
    """Render a normalized binning table with totals."""
    max_abs_woe = float(np.nanmax(np.abs(table["woe"]))) if len(table) else 1.0
    max_pct = float(table["count_pct"].max()) or 1.0
    head = ("<tr><th>#</th><th class='l'>Bin</th><th>Count</th><th class='l'>Count %</th>"
            "<th>Non-event</th><th>Event</th><th>Event rate</th><th>WoE</th><th>IV</th></tr>")
    rows = []
    for i, row in table.iterrows():
        regular = row["kind"] == RowKind.REGULAR
        rows.append(
            f"<tr class='{'' if regular else 'special'}'><td>{i + 1 if regular else ''}</td>"
            f"<td class='l'>{html.escape(str(row['bin']))}</td><td>{row['count']:,}</td>"
            f"<td class='l'>{_bar(row['count_pct'] / max_pct)}{row['count_pct']:.2%}</td>"
            f"<td>{row['non_event']:,}</td><td>{row['event']:,}</td>"
            f"<td>{row['event_rate']:.2%}</td>"
            f"<td style='{_woe_background(row['woe'], max_abs_woe)}'>{row['woe']:+.4f}</td>"
            f"<td>{row['iv']:.4f}</td></tr>")
    total, events = int(table["count"].sum()), int(table["event"].sum())
    rows.append(f"<tr class='total'><td></td><td class='l'>Totals</td><td>{total:,}</td>"
                f"<td class='l'>100.00%</td><td>{total - events:,}</td><td>{events:,}</td>"
                f"<td>{(events / total if total else 0):.2%}</td><td></td><td>{iv:.4f}</td></tr>")
    return f"<table class='ig-table'>{head}{''.join(rows)}</table>"


def _woe_background(woe: float, max_abs: float) -> str:
    """Diverging tint: blue for lower risk (WoE > 0), red for higher risk."""
    if not np.isfinite(woe) or max_abs <= 0:
        return ""
    alpha = min(abs(woe) / max_abs, 1.0) * 0.35
    rgb = "42,120,214" if woe >= 0 else "227,73,72"
    return f"background: rgba({rgb},{alpha:.2f});"


def statistics_html(original, applied, new, ordered: bool) -> str:
    """Compare three :class:`~interactive_grouping.variable.FitResult` side by side."""
    results = (original, applied, new)
    rows = [("Groups", [r.n_bins for r in results], "d"),
            ("Information Value", [r.iv for r in results], ".4f"),
            ("Gini", [r.gini for r in results], ".4f")]
    if ordered:
        rows.append(("Trend", [r.trend for r in results], ""))
    rows.append(("IV strength", [iv_strength(r.iv) for r in results], ""))

    body = "".join(
        f"<tr><td class='l'>{name}</td>"
        + "".join(f"<td{' class=new' if k == 2 else ''}>{_fmt(v, spec) if spec else v}</td>"
                  for k, v in enumerate(values))
        + "</tr>" for name, values, spec in rows)
    notes = "".join(message("warn", f"⚠ {html.escape(m)}") for m in new.messages)
    return ("<table class='ig-table'><tr><th class='l'></th><th>Original</th><th>Applied</th>"
            f"<th class='new'>New</th></tr>{body}</table>"
            "<div class='ig-note'>Original = optbinning, Applied = committed, "
            "New = current edit (press Apply to commit)</div>" + notes)


def variables_html(summary: pd.DataFrame, pending: set[str]) -> str:
    """Render the Variables tab table (``pending`` = names with unapplied edits)."""
    max_iv = max(float(summary["iv"].max()), 1e-9)
    head = ("<tr><th>#</th><th class='l'>Variable</th><th class='l'>Role</th>"
            "<th class='l'>Type</th><th>Original groups</th><th>Groups</th><th>Original IV</th>"
            "<th class='l'>IV</th><th>Original Gini</th><th>Gini</th><th class='l'>Strength</th>"
            "<th class='l'>Trend</th><th>Edited</th><th>Not applied</th></tr>")
    rows = []
    for i, row in summary.iterrows():
        rows.append(
            f"<tr class='{row['role']}'><td>{i + 1}</td>"
            f"<td class='l'><b>{html.escape(row['name'])}</b></td>"
            f"<td class='l'>{row['role'].capitalize()}</td><td class='l'>{row['dtype']}</td>"
            f"<td>{row['original_n_bins']}</td><td>{row['n_bins']}</td>"
            f"<td>{row['original_iv']:.4f}</td>"
            f"<td class='l'>{_bar(row['iv'] / max_iv)}{row['iv']:.4f}</td>"
            f"<td>{row['original_gini']:.4f}</td><td>{row['gini']:.4f}</td>"
            f"<td class='l'>{row['strength']}</td>"
            f"<td class='l'>{html.escape(str(row['trend']))}</td>"
            f"<td>{'✎' if row['edited'] else ''}</td>"
            f"<td>{'●' if row['name'] in pending else ''}</td></tr>")
    return (f"<table class='ig-table'>{head}{''.join(rows)}</table>"
            "<div class='ig-note'>IV strength: &lt;0.02 useless · 0.02–0.1 weak · "
            "0.1–0.3 medium · 0.3–0.5 strong · &gt;0.5 suspicious (check for leakage). "
            "Change the role on the Groupings tab.</div>")


# --------------------------------------------------------------------------- #
# Fixed-width grouping table (options of a SelectMultiple)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Column:
    """A column of the fixed-width grouping table."""

    title: str
    width: int
    value: Callable[[int, pd.Series], str]
    align: str = ">"

    def cell(self, text: str) -> str:
        """Pad or truncate ``text`` to the column width."""
        if len(text) > self.width:
            text = text[: self.width - 1] + "…"
        return f"{text:{self.align}{self.width}}"


def _render_options(columns: Sequence[Column], table: pd.DataFrame,
                    key: Callable[[int, pd.Series], Hashable]) -> list[tuple[str, Hashable]]:
    """Render rows as ``(label, value)`` options, preceded by the header row.

    Spaces become non-breaking spaces: browsers collapse ordinary ones inside
    ``<option>``. The header is an option too, which guarantees that it is aligned
    with the rows (a separate HTML header is rendered with a different font metric).
    """
    def line(cells: Sequence[str]) -> str:
        return "  ".join(cells).replace(" ", NBSP)

    options = [(line([c.cell(c.title) for c in columns]), HEADER_VALUE)]
    for i, row in table.iterrows():
        options.append((line([c.cell(c.value(i, row)) for c in columns]), key(i, row)))
    return options


def _value_width(labels: pd.Series, limit: int) -> int:
    return int(min(limit, max([5, *(len(str(label)) for label in labels)])))


def _non_regular_key(i: int, row: pd.Series) -> str:
    return f"{row['kind']}:{i}"


def _count_columns(with_share: bool = False) -> list[Column]:
    """Count / rate / WoE columns shared by the fine and coarse tables."""
    share = [Column("Count%", 7, lambda i, r: f"{r['count_pct']:.1%}")] if with_share else []
    return [
        Column("Count", 8, lambda i, r: f"{r['count']:,}"),
        *share,
        Column("Non-event", 9, lambda i, r: f"{r['non_event']:,}"),
        Column("Event", 7, lambda i, r: f"{r['event']:,}"),
        Column("Ev.rate", 8, lambda i, r: _fmt(r["event_rate"], ".2%")),
        Column("WoE", 8, lambda i, r: _fmt(r["woe"], "+.3f")),
    ]


def fine_options(fine_table: pd.DataFrame) -> list[tuple[str, Hashable]]:
    """Options of the grouping table at the fine level.

    Values: the fine-row index for regular rows, ``"<kind>:<index>"`` otherwise.
    """
    def regular(row: pd.Series) -> bool:
        return row["kind"] == RowKind.REGULAR

    columns = [
        Column("#", 3, lambda i, r: str(i + 1) if regular(r) else ""),
        Column("Value", _value_width(fine_table["label"], 30), lambda i, r: str(r["label"]),
               "<"),
        Column("Group", 5, lambda i, r: (str(r["group"] + 1) if r["group"] >= 0 else "-")
               if regular(r) else str(r["kind"])[:4]),
        *_count_columns(),
    ]
    return _render_options(columns, fine_table,
                           lambda i, r: int(i) if regular(r) else _non_regular_key(i, r))


def coarse_options(table: pd.DataFrame) -> list[tuple[str, Hashable]]:
    """Options of the grouping table at the group level.

    Values: the 0-based group index for regular rows, ``"<kind>:<index>"`` otherwise.
    """
    shown = table[(table["kind"] == RowKind.REGULAR) | (table["count"] > 0)]

    def regular(row: pd.Series) -> bool:
        return row["kind"] == RowKind.REGULAR

    columns = [
        Column("Group", 5, lambda i, r: str(i + 1) if regular(r) else str(r["kind"])[:4]),
        Column("Value", _value_width(shown["bin"], 34), lambda i, r: str(r["bin"]), "<"),
        *_count_columns(with_share=True),
        Column("IV", 7, lambda i, r: _fmt(r["iv"], ".4f")),
    ]
    return _render_options(columns, shown,
                           lambda i, r: int(i) if regular(r) else _non_regular_key(i, r))
