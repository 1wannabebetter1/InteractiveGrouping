"""Grouping state of a single variable.

Every recomputation goes through :class:`optbinning.OptimalBinning` with
``user_splits`` and ``user_splits_fixed=True``; WoE, IV and the binning table are
therefore always produced by optbinning itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from ._utils import RowKind, flatten_special_codes, quietly
from .exceptions import BinningEditError
from .history import History
from .metrics import detect_trend, weight_of_evidence
from .operations import fine_cutoffs_numerical
from .strategies import FineLevel, GroupingStrategy, get_strategy

__all__ = ["RELAXED_PARAMS", "TREND_OPTIONS", "FitResult", "VariableBinning"]

RELAXED_PARAMS: dict[str, Any] = {
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "min_bin_n_nonevent": None,
    "max_bin_n_nonevent": None,
    "min_bin_n_event": None,
    "max_bin_n_event": None,
    "monotonic_trend": None,
    "min_event_rate_diff": 0,
    "max_pvalue": None,
    "gamma": 0,
}
"""optbinning constraints switched off for manual groupings.

They could make the optimisation infeasible together with fixed user splits.
"""

TREND_OPTIONS: tuple = ("auto", "auto_heuristic", "auto_asc_desc", "ascending", "descending",
                        "concave", "convex", "peak", "valley", "peak_heuristic",
                        "valley_heuristic", None)
"""Values accepted by optbinning's ``monotonic_trend``."""

_TABLE_COLUMNS = {"Count": "count", "Count (%)": "count_pct", "Non-event": "non_event",
                  "Event": "event", "Event rate": "event_rate", "WoE": "woe", "IV": "iv"}


@dataclass(frozen=True)
class FitResult:
    """A grouping fitted by optbinning.

    Attributes
    ----------
    optb : optbinning.OptimalBinning
        The fitted estimator (used for ``transform``).
    splits : list
        Normalized splits.
    table : pandas.DataFrame
        Binning table without totals; columns ``bin``, ``kind``, ``count``,
        ``count_pct``, ``non_event``, ``event``, ``event_rate``, ``woe``, ``iv``.
    iv, gini : float
        Information Value and Gini of the grouping.
    messages : tuple of str
        Notes on how optbinning changed the requested grouping.
    """

    optb: OptimalBinning
    splits: list
    table: pd.DataFrame
    iv: float
    gini: float
    messages: tuple[str, ...] = ()

    @property
    def regular(self) -> pd.DataFrame:
        """Regular bins only (without missing / special), re-indexed from 0."""
        return self.table[self.table["kind"] == RowKind.REGULAR].reset_index(drop=True)

    @property
    def n_bins(self) -> int:
        """Number of regular groups."""
        return len(self.regular)

    @property
    def trend(self) -> str:
        """Shape of the event rate across the regular groups."""
        return detect_trend(self.regular["event_rate"].to_numpy())


class VariableBinning:
    """Grouping of one variable: original result, edit history and helpers.

    Parameters
    ----------
    name : str
        Variable name.
    x, y : array-like
        Raw values of the variable and the binary target.
    optb : optbinning.OptimalBinning, optional
        Binning to start from (fitted or not). Its parameters are reused for every
        recomputation.
    params : dict, optional
        ``OptimalBinning`` parameters used when ``optb`` is not given.
    """

    def __init__(self, name: str, x: Any, y: Any, optb: OptimalBinning | None = None,
                 params: dict | None = None) -> None:
        base = dict(optb.get_params() if optb is not None else params or {})
        for key in ("user_splits", "user_splits_fixed"):
            base.pop(key, None)
        base["name"] = name
        base.setdefault("dtype", "numerical")

        self.name = name
        self.strategy: GroupingStrategy = get_strategy(base["dtype"])
        self.base_params: dict[str, Any] = base
        self.x = np.asarray(x, dtype=self.strategy.x_dtype)
        self.y = np.asarray(y)
        self._clean = self._clean_mask()

        if optb is None or not getattr(optb, "_is_fitted", False):
            optb = self._fit(OptimalBinning(**base))
        self.original: FitResult = self._result(optb)
        self._history: History[FitResult] = History(self.original)

    # ------------------------------------------------------------------ data
    @property
    def dtype(self) -> str:
        """The optbinning ``dtype`` of the variable."""
        return self.strategy.dtype

    def _clean_mask(self) -> np.ndarray:
        """Mask of observations that are neither missing nor special codes."""
        mask = ~pd.isna(self.x)
        codes = flatten_special_codes(self.base_params.get("special_codes"))
        if codes:
            mask &= ~pd.Series(self.x).isin(codes).to_numpy()
        return mask

    @property
    def x_clean(self) -> np.ndarray:
        """Values without missing and special codes."""
        return self.x[self._clean]

    @property
    def y_clean(self) -> np.ndarray:
        """Target aligned with :attr:`x_clean`."""
        return self.y[self._clean]

    @property
    def x_range(self) -> tuple[float, float]:
        """Minimum and maximum of the clean values (numerical variables)."""
        values = self.x_clean.astype(float)
        return float(values.min()), float(values.max())

    # ------------------------------------------------------------------ fitting
    def _fit(self, optb: OptimalBinning) -> OptimalBinning:
        """Fit ``optb`` and translate optbinning errors into user-facing messages."""
        try:
            quietly(optb.fit, self.x, self.y)
        except (ValueError, TypeError) as exc:
            raise BinningEditError(f"optbinning: {_explain_fit_error(exc)}") from exc
        return optb

    def _result(self, optb: OptimalBinning, requested: list | None = None) -> FitResult:
        """Wrap a fitted estimator into a :class:`FitResult`."""
        splits = self.strategy.normalize(optb.splits)
        table, iv, gini = _binning_table(optb, self.strategy, splits)
        notes = () if requested is None else tuple(self.strategy.changes_note(requested, splits))
        return FitResult(optb=optb, splits=splits, table=table, iv=iv, gini=gini, messages=notes)

    def fit_splits(self, splits: list) -> FitResult:
        """Recompute the statistics with optbinning for exactly these splits."""
        splits = self.strategy.normalize(splits)
        self.strategy.validate(splits)
        optb = OptimalBinning(**{**self.base_params, **RELAXED_PARAMS,
                                 "user_splits": self.strategy.to_optbinning(splits),
                                 "user_splits_fixed": [True] * len(splits)})
        return self._result(self._fit(optb), requested=splits)

    def optimize(self, start_splits: list | None = None, **overrides: Any) -> FitResult:
        """Run optbinning's optimizer with the original parameters and ``overrides``.

        Parameters
        ----------
        start_splits : list, optional
            If given, they are used as (non-fixed) pre-bins, so the optimizer can only
            merge the current groups.
        **overrides
            ``OptimalBinning`` parameters, e.g. ``monotonic_trend`` or ``max_n_bins``.
        """
        params = {**self.base_params, **overrides}
        if start_splits is not None:
            params["user_splits"] = self.strategy.to_optbinning(
                self.strategy.normalize(start_splits))
        result = self._result(self._fit(OptimalBinning(**params)))
        if result.n_bins < 2:
            raise BinningEditError("The optimizer returned a single group with these "
                                   "constraints - try weaker constraints.")
        return result

    # ------------------------------------------------------------------ fine level
    def fine_level(self, splits: list, n_bins: int = 20,
                   extra_cutoffs: Iterable[float] = ()) -> FineLevel:
        """Build the fine-level view of ``splits``.

        Parameters
        ----------
        splits : list
            Grouping to show (e.g. a not yet applied one).
        n_bins : int
            Approximate number of fine bins (numerical variables).
        extra_cutoffs : iterable of float
            User-added cutoffs that must stay fine-bin boundaries.
        """
        cutoffs: tuple[float, ...] = ()
        if self.strategy.ordered:
            required = {*self.original.splits, *self.current.splits, *splits, *extra_cutoffs}
            cutoffs = tuple(fine_cutoffs_numerical(self.x_clean, n_bins, required))
        rows = self.strategy.fine_rows(self.x_clean, self.y_clean.astype(float), splits, cutoffs)
        rows["kind"] = RowKind.REGULAR
        table = pd.concat([rows, self._non_regular_rows()], ignore_index=True)
        return FineLevel(table=self._with_rates(table), cutoffs=cutoffs)

    def _non_regular_rows(self) -> pd.DataFrame:
        """Missing / special rows; optbinning never regroups them."""
        table = self.current.table
        rows = table[(table["kind"] != RowKind.REGULAR) & (table["count"] > 0)]
        return pd.DataFrame({"label": rows["bin"], "kind": rows["kind"], "group": -1,
                             "count": rows["count"], "event": rows["event"]})

    def _with_rates(self, table: pd.DataFrame) -> pd.DataFrame:
        """Add non-event counts, event rate, WoE and share of observations."""
        table = table.astype({"count": int, "event": int, "group": int})
        table["non_event"] = table["count"] - table["event"]
        total_event = float(self.y.sum())
        total_non_event = float(len(self.y)) - total_event
        with np.errstate(invalid="ignore", divide="ignore"):
            table["event_rate"] = np.where(table["count"] > 0,
                                           table["event"] / table["count"], np.nan)
        table["woe"] = weight_of_evidence(table["non_event"].to_numpy(float),
                                          table["event"].to_numpy(float),
                                          total_non_event, total_event)
        table["count_pct"] = table["count"] / len(self.y)
        return table

    # ------------------------------------------------------------------ history
    @property
    def current(self) -> FitResult:
        """The applied (committed) grouping."""
        return self._history.current

    @property
    def splits(self) -> list:
        """Splits of the applied grouping."""
        return self.current.splits

    @property
    def optb(self) -> OptimalBinning:
        """Fitted optbinning estimator of the applied grouping."""
        return self.current.optb

    @property
    def is_edited(self) -> bool:
        """Whether the applied grouping differs from the original optbinning result."""
        return not self.strategy.equal(self.current.splits, self.original.splits)

    @property
    def can_undo(self) -> bool:
        """Whether an applied change can be undone."""
        return self._history.can_undo

    @property
    def can_redo(self) -> bool:
        """Whether an undone change can be redone."""
        return self._history.can_redo

    def commit(self, result: FitResult) -> None:
        """Apply ``result`` (no-op if it equals the current grouping)."""
        if not self.strategy.equal(result.splits, self.current.splits):
            self._history.push(result)

    def set_splits(self, splits: list) -> FitResult:
        """Recompute and apply ``splits``."""
        result = self.fit_splits(splits)
        self.commit(result)
        return result

    def undo(self) -> bool:
        """Undo the last applied change."""
        return self._history.undo()

    def redo(self) -> bool:
        """Redo the last undone change."""
        return self._history.redo()

    def reset(self) -> bool:
        """Apply the original optbinning grouping (undoable); return whether it changed."""
        changed = self.is_edited
        self.commit(self.original)
        return changed

    # ------------------------------------------------------------------ export
    def fit_params(self) -> dict[str, Any]:
        """``OptimalBinning`` parameters that reproduce the applied grouping."""
        params = {k: v for k, v in self.base_params.items() if k != "name"}
        params.update(RELAXED_PARAMS)
        params["user_splits"] = self.strategy.to_optbinning(self.splits)
        params["user_splits_fixed"] = [True] * len(self.splits)
        return params


def _binning_table(optb: OptimalBinning, strategy: GroupingStrategy,
                   splits: list) -> tuple[pd.DataFrame, float, float]:
    """Normalize optbinning's binning table; return it with IV and Gini."""
    binning_table = optb.binning_table
    raw = binning_table.build(add_totals=False).reset_index(drop=True)
    n_regular = strategy.n_groups(splits)

    table = raw[list(_TABLE_COLUMNS)].rename(columns=_TABLE_COLUMNS)
    table = table.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    table = table.astype({"count": int, "non_event": int, "event": int})
    labels, kinds = [], []
    for i, raw_bin in enumerate(raw["Bin"]):
        if i < n_regular:
            labels.append(strategy.bin_label(raw_bin))
            kinds.append(RowKind.REGULAR)
        else:
            labels.append(str(raw_bin))
            kinds.append(RowKind.MISSING if str(raw_bin).lower().startswith("missing")
                         else RowKind.SPECIAL)
    table.insert(0, "kind", kinds)
    table.insert(0, "bin", labels)
    return table, float(binning_table.iv), float(binning_table.gini)


def _explain_fit_error(exc: Exception) -> str:
    """Rephrase known optbinning errors for the end user."""
    text = str(exc)
    if "pure prebins" in text:
        cutoffs = text.split("Fixed user_splits")[-1].split("are removed")[0].strip()
        return (f"cutoff(s) {cutoffs} would create a 'pure' group (only events or only "
                "non-events, or empty) where WoE is undefined. Choose another value or merge "
                "that group with a neighbour.")
    return text
