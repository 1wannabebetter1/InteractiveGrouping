"""Figures of the widget and the backends that draw and display them.

Two charts are drawn:

* the *groups* chart - stacked non-event / event counts per group (left axis) and
  the WoE line (right axis), like optbinning's ``binning_table.plot()``;
* the *fine* chart - fine bins coloured by group with group bands, the fine WoE
  line and the group WoE segments, like SAS Interactive Grouping.

The data for both charts is prepared once (:func:`_groups_data`, :func:`_fine_data`)
and drawn by a :class:`FigureBackend`. The widget only depends on that interface
(dependency inversion), so a new backend needs no change in the widget.

Figures are never displayed through ``ipywidgets.Output``: updating an output widget
from callbacks is unreliable outside JupyterLab (PyCharm, VS Code). Instead a backend
sets a plain widget trait - an ``<iframe srcdoc>`` for plotly, PNG bytes for
matplotlib.
"""

from __future__ import annotations

import contextlib
import html
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import ipywidgets as widgets
import numpy as np
import pandas as pd

from ._utils import RowKind

__all__ = ["FigureBackend", "MatplotlibBackend", "PlotlyBackend", "get_backend"]

# Validated reference palette (light surface).
BLUE = "#2a78d6"
RED = "#e34948"
LIGHT_BLUE = "#86b6ef"
ORANGE = "#eb6834"
GREY = "#9e9d98"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948")

NON_EVENT_LABEL = "Non-event (y=0)"
EVENT_LABEL = "Event (y=1)"


def shorten(text: Any, length: int = 28) -> str:
    """Truncate ``text`` to ``length`` characters with an ellipsis."""
    text = str(text)
    return text if len(text) <= length else text[: length - 1] + "…"


def group_color(group: int) -> str:
    """Colour of a 0-based group (the palette repeats; -1 = not grouped)."""
    return CATEGORICAL[group % len(CATEGORICAL)] if group >= 0 else GREY


# --------------------------------------------------------------------------- #
# Backend-independent chart data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _GroupsData:
    rows: pd.DataFrame        # regular groups + non-empty missing / special rows
    x: np.ndarray
    labels: list[str]
    regular: np.ndarray       # boolean mask of regular rows


def _groups_data(table: pd.DataFrame) -> _GroupsData:
    rows = table[(table["kind"] == RowKind.REGULAR) | (table["count"] > 0)]
    rows = rows.reset_index(drop=True)
    regular = (rows["kind"] == RowKind.REGULAR).to_numpy()
    labels = [f"{i + 1}: {shorten(b, 24)}" if r else str(b)
              for i, (b, r) in enumerate(zip(rows["bin"], regular))]
    return _GroupsData(rows, np.arange(len(rows)), labels, regular)


@dataclass(frozen=True)
class _FineData:
    rows: pd.DataFrame        # regular fine rows
    x: np.ndarray
    groups: np.ndarray
    spans: list[tuple[int, int, int]]   # (group, first x, last x) of contiguous blocks
    group_woe: dict[int, float]
    tick_positions: list[float]
    tick_labels: list[str]
    x_title: str


def _fine_data(fine_table: pd.DataFrame, group_woe: dict[int, float],
               ordered: bool) -> _FineData:
    rows = fine_table[fine_table["kind"] == RowKind.REGULAR].reset_index(drop=True)
    groups = rows["group"].to_numpy()
    spans: list[tuple[int, int, int]] = []
    for i, group in enumerate(groups):
        if spans and spans[-1][0] == group:
            spans[-1] = (int(group), spans[-1][1], i)
        else:
            spans.append((int(group), i, i))
    if ordered:  # ticks at group boundaries, labelled with the cutoff value
        positions = [first - 0.5 for _, first, _ in spans[1:]]
        labels = [f"{rows['lo'][first]:.4g}" for _, first, _ in spans[1:]]
        x_title = "Cutoff"
    else:
        positions = list(range(len(rows)))
        labels = [shorten(label, 16) for label in rows["label"]]
        x_title = "Category"
    return _FineData(rows, np.arange(len(rows)), groups, spans, group_woe,
                     positions, labels, x_title)


def _hover_rows(rows: pd.DataFrame) -> np.ndarray:
    """Values shown in the tooltips (shared by both charts)."""
    return np.column_stack([rows.get("bin", rows.get("label")), rows["count"],
                            rows["event_rate"] * 100, rows["woe"]])


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #
class FigureBackend(ABC):
    """Draws the charts and shows them in a widget."""

    @abstractmethod
    def groups_figure(self, table: pd.DataFrame, title: str, height: int) -> Any:
        """Draw counts per group and the WoE line from a normalized binning table."""

    @abstractmethod
    def fine_figure(self, fine_table: pd.DataFrame, group_woe: dict[int, float],
                    ordered: bool, title: str, height: int) -> Any:
        """Draw fine bins coloured by group from a fine-level table."""

    @abstractmethod
    def create_view(self) -> widgets.Widget:
        """Create the widget that displays one figure."""

    @abstractmethod
    def render(self, view: widgets.Widget, figure: Any) -> None:
        """Show ``figure`` in ``view`` (``None`` clears it)."""


# --------------------------------------------------------------------------- #
# Plotly
# --------------------------------------------------------------------------- #
class PlotlyBackend(FigureBackend):
    """Interactive plotly figures rendered in an ``<iframe srcdoc>``.

    Parameters
    ----------
    plotly_js : {"cdn", "inline"}
        How plotly.js is included: from the CDN (needs internet) or embedded into
        every figure (works offline, ~3.5 MB per figure).
    """

    _HOVER = ("<b>%{customdata[0]}</b><br>Count: %{customdata[1]:,}"
              "<br>Event rate: %{customdata[2]:.2f}%<br>WoE: %{customdata[3]:.4f}"
              "<extra></extra>")

    def __init__(self, plotly_js: str = "cdn") -> None:
        if plotly_js not in ("cdn", "inline"):
            raise ValueError("plotly_js must be 'cdn' or 'inline'")
        self.plotly_js = True if plotly_js == "inline" else "cdn"

    @staticmethod
    def _new_figure():
        from plotly.subplots import make_subplots

        return make_subplots(specs=[[{"secondary_y": True}]])

    @staticmethod
    def _style(fig, title: str, height: int, count_title: str, x_title: str):
        fig.update_layout(
            template="plotly_white", height=height, barmode="stack", bargap=0.12,
            title={"text": title, "x": 0.01, "font": {"size": 15}},
            margin={"l": 60, "r": 20, "t": 80, "b": 40},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                    "xanchor": "right", "x": 1, "traceorder": "normal"},
            hoverlabel={"bgcolor": "white", "font_size": 12},
            font={"color": INK_SECONDARY},
        )
        with contextlib.suppress(ValueError):  # rounded bar ends need plotly >= 5.19
            fig.update_layout(barcornerradius=3)
        fig.update_xaxes(showgrid=False, title_text=x_title)
        fig.update_yaxes(title_text=count_title, secondary_y=False, rangemode="tozero",
                         gridcolor=GRID)
        fig.update_yaxes(title_text="WoE", secondary_y=True, showgrid=False, zeroline=False,
                         tickmode="auto", nticks=6, tickformat=".2~f")
        # WoE = 0 reference line on the right axis.
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=0, y1=0,
                      line={"color": INK_SECONDARY, "width": 1, "dash": "dot"})
        return fig

    def groups_figure(self, table: pd.DataFrame, title: str, height: int):
        """See :meth:`FigureBackend.groups_figure`."""
        import plotly.graph_objects as go

        data = _groups_data(table)
        rows, hover = data.rows, _hover_rows(data.rows)
        fig = self._new_figure()
        for column, name, color in (("non_event", NON_EVENT_LABEL, LIGHT_BLUE),
                                    ("event", EVENT_LABEL, ORANGE)):
            fig.add_trace(go.Bar(x=data.x, y=rows[column], name=name, marker_color=color,
                                 customdata=hover, hovertemplate=self._HOVER),
                          secondary_y=False)
        marker = {"color": INK, "size": 9, "line": {"color": "white", "width": 1.5}}
        fig.add_trace(go.Scatter(x=data.x[data.regular], y=rows["woe"][data.regular],
                                 name="WoE", mode="lines+markers",
                                 line={"color": INK, "width": 2}, marker=marker,
                                 customdata=hover[data.regular], hovertemplate=self._HOVER),
                      secondary_y=True)
        if (~data.regular).any():
            fig.add_trace(go.Scatter(x=data.x[~data.regular], y=rows["woe"][~data.regular],
                                     name="WoE (special / missing)", mode="markers",
                                     marker={**marker, "symbol": "diamond"},
                                     customdata=hover[~data.regular],
                                     hovertemplate=self._HOVER),
                          secondary_y=True)
        fig.update_xaxes(tickvals=list(data.x), ticktext=data.labels,
                         tickangle=-30 if len(rows) > 5 else 0)
        return self._style(fig, title, height, "Bin count", "Group")

    def fine_figure(self, fine_table: pd.DataFrame, group_woe: dict[int, float],
                    ordered: bool, title: str, height: int):
        """See :meth:`FigureBackend.fine_figure`."""
        import plotly.graph_objects as go

        data = _fine_data(fine_table, group_woe, ordered)
        rows, hover = data.rows, _hover_rows(data.rows)
        fig = self._new_figure()
        for group, first, last in data.spans:
            fig.add_vrect(x0=first - 0.5, x1=last + 0.5, fillcolor=group_color(group),
                          opacity=0.10, line_width=0, layer="below")
            if first:
                fig.add_vline(x=first - 0.5, line_dash="dash", line_color=INK_SECONDARY,
                              line_width=1)
        for group in sorted(set(data.groups.tolist())):
            mask = data.groups == group
            fig.add_trace(go.Bar(x=data.x[mask], y=rows["count"][mask],
                                 marker_color=group_color(group), name=f"Group {group + 1}",
                                 showlegend=False, customdata=hover[mask],
                                 hovertemplate=self._HOVER), secondary_y=False)
        fig.add_trace(go.Scatter(x=data.x, y=rows["woe"], mode="lines+markers", name="Fine WoE",
                                 line={"color": INK_SECONDARY, "width": 1},
                                 marker={"size": 5, "color": INK_SECONDARY},
                                 customdata=hover, hovertemplate=self._HOVER), secondary_y=True)
        seg_x, seg_y = [], []
        for group, first, last in data.spans:
            seg_x += [first - 0.45, last + 0.45, None]
            seg_y += [group_woe.get(group, np.nan)] * 2 + [None]
        fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", name="Group WoE",
                                 line={"color": RED, "width": 3}, hoverinfo="skip"),
                      secondary_y=True)
        fig.update_xaxes(tickvals=data.tick_positions, ticktext=data.tick_labels,
                         tickangle=0 if ordered else -35, range=[-0.6, len(rows) - 0.4])
        fig = self._style(fig, title, height, "Count", data.x_title)
        for group, first, last in data.spans:  # group numbers above the bands
            fig.add_annotation(x=(first + last) / 2, y=1.0, xref="x", yref="paper",
                               yanchor="bottom", showarrow=False,
                               text=f"<b>{group + 1}</b>" if group >= 0 else "-",
                               font={"size": 11, "color": group_color(group)})
        fig.update_layout(bargap=0.08)
        return fig

    def create_view(self) -> widgets.Widget:
        """See :meth:`FigureBackend.create_view`."""
        return widgets.HTML(layout=widgets.Layout(width="100%"))

    def render(self, view: widgets.Widget, figure: Any) -> None:
        """See :meth:`FigureBackend.render`."""
        if figure is None:
            view.value = ""
            return
        page = figure.to_html(full_html=True, include_plotlyjs=self.plotly_js,
                              config={"responsive": True, "displaylogo": False})
        page = page.replace("<head>", "<head><style>html,body{margin:0;padding:0;"
                                      "background:#fff;overflow:hidden}</style>", 1)
        height = int(figure.layout.height or 450) + 8
        view.value = (f'<iframe srcdoc="{html.escape(page, quote=True)}" '
                      f'style="width:100%;height:{height}px;border:0;display:block" '
                      f'sandbox="allow-scripts allow-downloads"></iframe>')


# --------------------------------------------------------------------------- #
# Matplotlib
# --------------------------------------------------------------------------- #
class MatplotlibBackend(FigureBackend):
    """Static matplotlib figures rendered as PNG images (no internet needed).

    Parameters
    ----------
    dpi : int
        Resolution of the PNG images.
    """

    _PX_PER_INCH = 90  # converts the requested pixel height into figure inches

    def __init__(self, dpi: int = 110) -> None:
        self.dpi = dpi

    def _new_axes(self, title: str, height: int):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, height / self._PX_PER_INCH))
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=18)
        twin = ax.twinx()
        for axis in (ax, twin):
            axis.spines[["top"]].set_visible(False)
            axis.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        twin.axhline(0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
        twin.set_ylabel("WoE", fontsize=9)
        return fig, ax, twin

    @staticmethod
    def _legend(ax, twin) -> None:
        handles, labels = ax.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        ax.legend(handles + twin_handles, labels + twin_labels, frameon=False, fontsize=8,
                  ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0))

    def groups_figure(self, table: pd.DataFrame, title: str, height: int):
        """See :meth:`FigureBackend.groups_figure`."""
        data = _groups_data(table)
        rows = data.rows
        fig, ax, twin = self._new_axes(title, height)
        ax.bar(data.x, rows["non_event"], color=LIGHT_BLUE, width=0.8, label=NON_EVENT_LABEL)
        ax.bar(data.x, rows["event"], bottom=rows["non_event"], color=ORANGE, width=0.8,
               label=EVENT_LABEL)
        ax.set_ylabel("Bin count", fontsize=9)
        rotate = len(rows) > 5
        ax.set_xticks(data.x, data.labels, rotation=30 if rotate else 0,
                      ha="right" if rotate else "center")
        style = {"color": INK, "markersize": 6, "markeredgecolor": "white"}
        twin.plot(data.x[data.regular], rows["woe"][data.regular], "-o", linewidth=2,
                  label="WoE", **style)
        if (~data.regular).any():
            twin.plot(data.x[~data.regular], rows["woe"][~data.regular], "D",
                      label="WoE (special / missing)", **style)
        self._legend(ax, twin)
        fig.tight_layout()
        return fig

    def fine_figure(self, fine_table: pd.DataFrame, group_woe: dict[int, float],
                    ordered: bool, title: str, height: int):
        """See :meth:`FigureBackend.fine_figure`."""
        data = _fine_data(fine_table, group_woe, ordered)
        fig, ax, twin = self._new_axes(title, height)
        for group, first, last in data.spans:
            color = group_color(group)
            ax.axvspan(first - 0.5, last + 0.5, color=color, alpha=0.10, lw=0)
            ax.text((first + last) / 2, 1.0, str(group + 1), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)
            if first:
                ax.axvline(first - 0.5, color=INK_SECONDARY, linestyle="--", linewidth=0.8)
        ax.bar(data.x, data.rows["count"], width=0.9,
               color=[group_color(g) for g in data.groups])
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlabel(data.x_title, fontsize=9)
        ax.set_xticks(data.tick_positions, data.tick_labels,
                      rotation=0 if ordered else 35, ha="center" if ordered else "right")
        ax.set_xlim(-0.6, len(data.rows) - 0.4)
        twin.plot(data.x, data.rows["woe"], "-o", color=INK_SECONDARY, linewidth=1,
                  markersize=3, label="Fine WoE")
        for k, (group, first, last) in enumerate(data.spans):
            woe = group_woe.get(group, np.nan)
            twin.plot([first - 0.45, last + 0.45], [woe, woe], color=RED, linewidth=3,
                      label="Group WoE" if k == 0 else None)
        self._legend(ax, twin)
        fig.tight_layout()
        return fig

    def create_view(self) -> widgets.Widget:
        """See :meth:`FigureBackend.create_view`."""
        return widgets.Image(format="png", layout=widgets.Layout(width="100%", height="auto",
                                                                 display="none"))

    def render(self, view: widgets.Widget, figure: Any) -> None:
        """See :meth:`FigureBackend.render`."""
        import matplotlib.pyplot as plt

        if figure is None:
            view.layout.display = "none"
            return
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=self.dpi, bbox_inches="tight",
                       facecolor="white")
        plt.close(figure)  # free memory; the image is all we need
        view.value = buffer.getvalue()
        view.layout.display = ""


_BACKENDS = {"plotly": PlotlyBackend, "matplotlib": MatplotlibBackend}


def get_backend(name: str, **options: Any) -> FigureBackend:
    """Create a figure backend by name (``"plotly"`` or ``"matplotlib"``)."""
    try:
        return _BACKENDS[name](**options)
    except KeyError:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}") from None
