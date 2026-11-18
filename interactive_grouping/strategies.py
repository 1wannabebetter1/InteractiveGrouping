"""Variable-type strategies (Strategy pattern).

Everything that differs between numerical and categorical variables - how splits are
represented, compared, merged and broken down into fine bins - lives behind the
:class:`GroupingStrategy` interface. The rest of the package works with the
abstraction only, so supporting a new variable type means adding one class
(open/closed principle) without touching the callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ._utils import RowKind, format_number, to_python_scalar
from .exceptions import BinningEditError
from .operations import fine_labels, move_categories, move_to_neighbour, split_group_at

__all__ = [
    "NEW_GROUP",
    "NEW_GROUP_BELOW",
    "CategoricalStrategy",
    "FineLevel",
    "GroupingStrategy",
    "NumericalStrategy",
    "get_strategy",
]

NEW_GROUP = -1
"""Target of :meth:`GroupingStrategy.fine_assign`: a new group.

For numerical variables the new group is *above* the selection: the selected and all
following fine bins of their group.
"""
NEW_GROUP_BELOW = -2
"""Target of :meth:`GroupingStrategy.fine_assign` (numerical variables only): a new
group made of the selected and all preceding fine bins of their group."""

_ATOL = 1e-12


@dataclass(frozen=True)
class FineLevel:
    """Fine-level view of a variable.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per fine bin (numerical) or category (categorical) followed by the
        missing / special rows. Columns: ``label``, ``kind``, ``group`` (0-based, -1 for
        missing / special), ``count``, ``event``, ``non_event``, ``event_rate``,
        ``woe``, ``count_pct`` and ``lo`` / ``hi`` (numerical) or ``category``.
    cutoffs : tuple of float
        Fine cutoffs (numerical variables only).
    """

    table: pd.DataFrame
    cutoffs: tuple[float, ...] = ()

    @property
    def regular(self) -> pd.DataFrame:
        """Rows that can be regrouped (no missing / special rows)."""
        return self.table[self.table["kind"] == RowKind.REGULAR]


class GroupingStrategy(ABC):
    """Operations whose behaviour depends on the variable type."""

    dtype: ClassVar[str]
    """optbinning ``dtype`` handled by the strategy."""
    ordered: ClassVar[bool]
    """Whether groups are ordered intervals (enables cutoffs and the trend)."""
    new_group_targets: ClassVar[tuple[int, ...]]
    """Targets of :meth:`fine_assign` that create a new group."""
    x_dtype: ClassVar[type]
    """numpy dtype the raw values are stored with."""

    # -- representation ----------------------------------------------------
    @abstractmethod
    def normalize(self, splits: Any) -> list:
        """Convert optbinning splits to plain, JSON-serialisable Python lists."""

    @abstractmethod
    def to_optbinning(self, splits: list) -> Any:
        """Convert normalized splits to optbinning's ``user_splits`` format."""

    @abstractmethod
    def equal(self, a: list, b: list) -> bool:
        """Check whether two normalized splits define the same grouping."""

    @abstractmethod
    def n_groups(self, splits: list) -> int:
        """Return the number of regular groups defined by ``splits``."""

    @abstractmethod
    def bin_label(self, raw_bin: Any) -> str:
        """Return a readable label of a regular bin of optbinning's binning table."""

    @abstractmethod
    def changes_note(self, requested: list, fitted: list) -> list[str]:
        """Explain how optbinning's result differs from the requested splits."""

    def validate(self, splits: list) -> None:
        """Raise :class:`BinningEditError` if ``splits`` define fewer than 2 groups."""
        if self.n_groups(splits) < 2:
            raise BinningEditError("At least 2 groups are required.")

    # -- fine level --------------------------------------------------------
    @abstractmethod
    def fine_rows(self, x: np.ndarray, y: np.ndarray, splits: list,
                  cutoffs: Sequence[float]) -> pd.DataFrame:
        """Aggregate clean values into fine rows.

        Returns a frame with at least ``label``, ``group``, ``count`` and ``event``.
        """

    @abstractmethod
    def fine_assign(self, splits: list, fine: FineLevel, rows: Sequence[int],
                    target: int) -> list:
        """Move fine ``rows`` to group ``target`` (0-based) or to a new group.

        ``target`` is a group index or one of :attr:`new_group_targets`. If the source
        group becomes empty, it disappears (the groups merge).
        """

    def fine_targets(self, splits: list, fine: FineLevel, rows: Sequence[int]) -> list[int]:
        """List the valid targets for ``rows`` that change the grouping.

        Existing groups come first, then the new-group targets.
        """
        targets = []
        for target in [*range(self.n_groups(splits)), *self.new_group_targets]:
            try:
                self.fine_assign(splits, fine, rows, target)
            except BinningEditError:
                continue
            targets.append(target)
        return targets


class NumericalStrategy(GroupingStrategy):
    """Numerical variables: groups are contiguous intervals separated by cutoffs."""

    dtype = "numerical"
    ordered = True
    x_dtype = float
    new_group_targets = (NEW_GROUP, NEW_GROUP_BELOW)

    def normalize(self, splits: Any) -> list[float]:
        if splits is None:
            return []
        return [float(s) for s in np.asarray(splits, dtype=float).ravel()]

    def to_optbinning(self, splits: list) -> np.ndarray:
        return np.asarray(splits, dtype=float)

    def equal(self, a: list, b: list) -> bool:
        return len(a) == len(b) and bool(np.allclose(a, b, rtol=0, atol=_ATOL))

    def n_groups(self, splits: list) -> int:
        return len(splits) + 1

    def bin_label(self, raw_bin: Any) -> str:
        return str(raw_bin)

    def changes_note(self, requested: list, fitted: list) -> list[str]:
        dropped = [s for s in requested
                   if not any(np.isclose(s, f, rtol=0, atol=_ATOL) for f in fitted)]
        if not dropped:
            return []
        return ["optbinning removed cutoff(s): " + ", ".join(map(format_number, dropped))]

    def fine_rows(self, x: np.ndarray, y: np.ndarray, splits: list,
                  cutoffs: Sequence[float]) -> pd.DataFrame:
        cutoffs = list(cutoffs)
        n_fine = len(cutoffs) + 1
        index = np.searchsorted(cutoffs, x, side="right")
        edges = [-np.inf, *cutoffs, np.inf]
        return pd.DataFrame({
            "label": [f"{'(' if i == 0 else '['}{format_number(edges[i])}, "
                      f"{format_number(edges[i + 1])})" for i in range(n_fine)],
            "lo": edges[:-1],
            "hi": edges[1:],
            "group": fine_labels(cutoffs, splits),
            "count": np.bincount(index, minlength=n_fine),
            "event": np.bincount(index, weights=y, minlength=n_fine),
        })

    def fine_assign(self, splits: list, fine: FineLevel, rows: Sequence[int],
                    target: int) -> list:
        # Only the boundary to a neighbouring group can move (see move_to_neighbour);
        # a new group starts at the selection and extends to the group edge.
        if target in (NEW_GROUP, NEW_GROUP_BELOW):
            direction = 1 if target == NEW_GROUP else -1
            return split_group_at(fine.cutoffs, splits, rows, direction)
        group = int(fine_labels(fine.cutoffs, splits)[min(rows)]) if rows else -1
        if abs(int(target) - group) != 1:
            raise BinningEditError("Numerical fine bins can join a neighbouring group only.")
        return move_to_neighbour(fine.cutoffs, splits, rows, int(target) - group)


class CategoricalStrategy(GroupingStrategy):
    """Categorical variables: groups are arbitrary sets of categories."""

    dtype = "categorical"
    ordered = False
    x_dtype = object
    new_group_targets = (NEW_GROUP,)

    def normalize(self, splits: Any) -> list[list]:
        if splits is None:
            return []
        return [[to_python_scalar(c) for c in list(group)] for group in splits]

    def to_optbinning(self, splits: list) -> list[np.ndarray]:
        return [np.asarray(group, dtype=object) for group in splits]

    def equal(self, a: list, b: list) -> bool:
        return {frozenset(g) for g in a} == {frozenset(g) for g in b}

    def n_groups(self, splits: list) -> int:
        return len(splits)

    def bin_label(self, raw_bin: Any) -> str:
        categories = [raw_bin] if isinstance(raw_bin, str) else list(raw_bin)
        return ", ".join(map(str, categories))

    def changes_note(self, requested: list, fitted: list) -> list[str]:
        return [] if self.equal(requested, fitted) else [
            "optbinning adjusted the category groups."]

    def fine_rows(self, x: np.ndarray, y: np.ndarray, splits: list,
                  cutoffs: Sequence[float]) -> pd.DataFrame:
        stats = (pd.DataFrame({"category": x, "y": y})
                 .groupby("category", sort=False)["y"].agg(["count", "sum"]).reset_index())
        stats["category"] = [to_python_scalar(c) for c in stats["category"]]
        group_of = {c: i for i, group in enumerate(splits) for c in group}
        stats["group"] = [group_of.get(c, -1) for c in stats["category"]]
        stats["rate"] = stats["sum"] / stats["count"]
        # Order rows by group, then by event rate: groups appear as contiguous blocks.
        stats = stats.sort_values(["group", "rate", "count"], ascending=[True, True, False])
        return pd.DataFrame({
            "label": [str(c) for c in stats["category"]],
            "category": list(stats["category"]),
            "group": stats["group"].to_numpy(),
            "count": stats["count"].to_numpy(),
            "event": stats["sum"].to_numpy(),
        })

    def fine_assign(self, splits: list, fine: FineLevel, rows: Sequence[int],
                    target: int) -> list:
        categories = [fine.table.loc[r, "category"] for r in rows]
        return move_categories(splits, categories, None if target == NEW_GROUP else int(target))


_STRATEGIES: dict[str, GroupingStrategy] = {
    s.dtype: s for s in (NumericalStrategy(), CategoricalStrategy())}


def get_strategy(dtype: str) -> GroupingStrategy:
    """Return the strategy for an optbinning ``dtype``."""
    try:
        return _STRATEGIES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype {dtype!r}; "
                         f"expected one of {sorted(_STRATEGIES)}.") from None
