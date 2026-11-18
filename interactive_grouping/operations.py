"""Pure operations on groupings.

Every function takes the current splits and returns *new* splits (inputs are never
mutated) or raises :class:`~interactive_grouping.exceptions.BinningEditError` with a
user-facing message.

Representation of splits (the same as optbinning's ``user_splits``):

* numerical variables: sorted list of cutoffs; ``k`` cutoffs define ``k + 1`` groups,
  group ``i`` is ``[cutoff[i - 1], cutoff[i])``;
* categorical variables: list of groups, each a list of categories.

Fine level (SAS-style "fine bins -> groups"): a numerical variable is also cut by
finer *fine cutoffs* which always contain the group splits, so every fine bin belongs
to exactly one group.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np

from ._utils import round_significant
from .exceptions import BinningEditError

__all__ = [
    "add_numerical_split",
    "fine_cutoffs_numerical",
    "fine_labels",
    "merge_groups",
    "merge_numerical",
    "move_categories",
    "move_to_neighbour",
    "parse_cutoffs",
    "quantile_splits_in_bin",
    "split_fine_bin",
    "split_group_at",
    "splits_from_labels",
    "suggest_fine_split",
]

_ATOL = 1e-12  # tolerance for "the same cutoff"


def _contains(values: Iterable[float], value: float) -> bool:
    return any(np.isclose(value, v, rtol=0, atol=_ATOL) for v in values)


def _sorted_indices(indices: Iterable[int], minimum: int, what: str) -> list[int]:
    idx = sorted({int(i) for i in indices})
    if len(idx) < minimum:
        raise BinningEditError(f"Select at least {'two' if minimum == 2 else 'one'} {what}.")
    return idx


def _require_two_groups(n_groups: int) -> None:
    if n_groups < 2:
        raise BinningEditError("At least 2 groups must remain.")


# --------------------------------------------------------------------------- #
# Numerical variables: group level
# --------------------------------------------------------------------------- #
def merge_numerical(splits: list[float], group_indices: Iterable[int]) -> list[float]:
    """Merge groups of a numerical variable.

    A non-contiguous selection merges the whole range between the outermost groups,
    because numerical groups must stay contiguous.
    """
    idx = _sorted_indices(group_indices, 2, "groups to merge")
    if idx[0] < 0 or idx[-1] > len(splits):
        raise BinningEditError("Group index out of range.")
    new = splits[:idx[0]] + splits[idx[-1]:]
    _require_two_groups(len(new) + 1)
    return new


def add_numerical_split(splits: list[float], value: float,
                        x_min: float | None = None, x_max: float | None = None) -> list[float]:
    """Add a cutoff at ``value``.

    The cutoff must lie in ``(x_min, x_max]`` so that both new groups can be non-empty.
    """
    value = float(value)
    if not np.isfinite(value):
        raise BinningEditError("Cutoff must be a finite number.")
    if x_min is not None and value <= x_min:
        raise BinningEditError(f"Cutoff {value:g} must be > min value ({x_min:g}).")
    if x_max is not None and value > x_max:
        raise BinningEditError(f"Cutoff {value:g} must be <= max value ({x_max:g}).")
    if _contains(splits, value):
        raise BinningEditError(f"Cutoff {value:g} already exists.")
    return sorted([*splits, value])


def quantile_splits_in_bin(splits: list[float], x: np.ndarray, group_index: int,
                           n_parts: int) -> list[float]:
    """Split one group into ``n_parts`` approximately equal-frequency parts."""
    if n_parts < 2:
        raise BinningEditError("Number of parts must be >= 2.")
    edges = [-np.inf, *splits, np.inf]
    if not 0 <= group_index < len(edges) - 1:
        raise BinningEditError("Group index out of range.")
    lo, hi = edges[group_index], edges[group_index + 1]
    values = x[(x >= lo) & (x < hi)]
    if len(values) == 0:
        raise BinningEditError("The selected group is empty.")
    quantiles = np.quantile(values, np.linspace(0, 1, n_parts + 1)[1:-1])
    new_cuts = [q for q in round_significant(np.unique(quantiles))
                if lo < q < hi and q > values.min() and not _contains(splits, q)]
    if not new_cuts:
        raise BinningEditError("Could not find distinct quantile cutoffs "
                               "(too few unique values in this group).")
    return sorted([*splits, *new_cuts])


def parse_cutoffs(text: str, x_range: tuple[float, float]) -> list[float]:
    """Parse a comma / semicolon separated list of cutoffs typed by the user."""
    try:
        values = sorted({float(token) for token in text.replace(";", ",").split(",")
                         if token.strip()})
    except ValueError:
        raise BinningEditError("Cutoffs must be numbers separated by commas.") from None
    lo, hi = x_range
    outside = [v for v in values if not lo < v <= hi]
    if outside:
        raise BinningEditError(f"Cutoffs outside the data range ({lo:g}, {hi:g}]: {outside}")
    return values


# --------------------------------------------------------------------------- #
# Categorical variables: group level
# --------------------------------------------------------------------------- #
def merge_groups(groups: list[list], group_indices: Iterable[int]) -> list[list]:
    """Merge category groups; the merged group is appended at the end."""
    idx = _sorted_indices(group_indices, 2, "groups to merge")
    if idx[0] < 0 or idx[-1] >= len(groups):
        raise BinningEditError("Group index out of range.")
    merged = [category for i in idx for category in groups[i]]
    new = [g for i, g in enumerate(groups) if i not in idx] + [merged]
    _require_two_groups(len(new))
    return new


def move_categories(groups: list[list], categories: Iterable,
                    target: int | None = None) -> list[list]:
    """Move ``categories`` into group ``target`` or, if ``target`` is None, a new group."""
    moved = list(categories)
    if not moved:
        raise BinningEditError("Select at least one category.")
    moved_set = set(moved)
    unknown = moved_set - {c for g in groups for c in g}
    if unknown:
        raise BinningEditError(f"Unknown categories: {sorted(map(str, unknown))}")
    if target is not None and not 0 <= target < len(groups):
        raise BinningEditError("Target group index out of range.")

    remaining = [[c for c in g if c not in moved_set] for g in groups]
    if target is None:
        remaining.append(moved)
    else:
        remaining[target] = remaining[target] + moved
    new = [g for g in remaining if g]
    _require_two_groups(len(new))
    if {frozenset(g) for g in new} == {frozenset(g) for g in groups}:
        raise BinningEditError("Nothing changed.")
    return new


# --------------------------------------------------------------------------- #
# Numerical variables: fine level
# --------------------------------------------------------------------------- #
def fine_cutoffs_numerical(x: np.ndarray, n_bins: int = 20,
                           required: Iterable[float] = ()) -> list[float]:
    """Build fine-bin cutoffs.

    Parameters
    ----------
    x : numpy.ndarray
        Clean (non-missing, non-special) values.
    n_bins : int
        Target number of approximately equal-frequency fine bins.
    required : iterable of float
        Cutoffs that must be kept (the group splits).

    Returns
    -------
    list of float
        Sorted cutoffs. Low-cardinality data is cut between unique values; integer
        data is cut at ``k - 0.5``. Optional cutoffs creating empty fine bins are
        dropped.
    """
    x = np.asarray(x, dtype=float)
    required_set = {float(r) for r in required}
    unique = np.unique(x)
    if len(unique) <= n_bins:
        candidates = list((unique[:-1] + unique[1:]) / 2)
    else:
        quantiles = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
        if np.all(unique == np.round(unique)):
            quantiles = np.ceil(quantiles) - 0.5
        candidates = [q for q in round_significant(np.unique(quantiles), digits=4)
                      if unique[0] < q <= unique[-1]]
    cutoffs = sorted(set(candidates) | required_set)
    return _drop_empty_fine_bins(x, cutoffs, required_set)


def _drop_empty_fine_bins(x: np.ndarray, cutoffs: list[float],
                          required: set[float]) -> list[float]:
    """Remove optional cutoffs adjacent to empty fine bins (right edge first)."""
    while True:
        counts = np.bincount(np.searchsorted(cutoffs, x, side="right"),
                             minlength=len(cutoffs) + 1)
        removable = next((j for i in np.flatnonzero(counts == 0) for j in (i, i - 1)
                          if 0 <= j < len(cutoffs) and cutoffs[j] not in required), None)
        if removable is None:
            return cutoffs
        cutoffs.pop(removable)


def fine_labels(cutoffs: Sequence[float], splits: Sequence[float]) -> np.ndarray:
    """Return the 0-based group of every fine bin (``splits`` must be within ``cutoffs``)."""
    lefts = np.array([-np.inf, *cutoffs])
    return np.searchsorted(np.asarray(splits, dtype=float), lefts, side="right")


def splits_from_labels(cutoffs: Sequence[float], labels: Sequence) -> list[float]:
    """Return the group splits: fine cutoffs where the group label changes."""
    return [float(cutoffs[i - 1]) for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


class _Selection(NamedTuple):
    """Fine bins selected inside one group."""

    labels: np.ndarray   # group of every fine bin
    group: int           # group of the selection
    first: int           # first / last selected fine bin
    last: int
    group_first: int     # first / last fine bin of the group
    group_last: int


def _selection(cutoffs: Sequence[float], splits: Sequence[float],
               rows: Iterable[int]) -> _Selection:
    """Validate that ``rows`` lie in one group and describe them."""
    labels = fine_labels(cutoffs, splits)
    idx = _sorted_indices(rows, 1, "fine bin")
    if idx[0] < 0 or idx[-1] >= len(labels):
        raise BinningEditError("Fine bin index out of range.")
    group = int(labels[idx[0]])
    if labels[idx[-1]] != group:
        raise BinningEditError("Select fine bins of one group only.")
    members = np.flatnonzero(labels == group)
    return _Selection(labels, group, idx[0], idx[-1], int(members[0]), int(members[-1]))


def _check_direction(direction: int) -> None:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 (lower) or +1 (higher)")


def _splits_or_error(cutoffs: Sequence[float], labels: np.ndarray, old: Sequence[float],
                     unchanged_message: str) -> list[float]:
    new = splits_from_labels(cutoffs, labels)
    _require_two_groups(len(new) + 1)
    if len(new) == len(old) and np.allclose(new, old, rtol=0, atol=_ATOL):
        raise BinningEditError(unchanged_message)
    return new


def move_to_neighbour(cutoffs: Sequence[float], splits: Sequence[float],
                      rows: Iterable[int], direction: int) -> list[float]:
    """Move fine bins to the previous (``direction=-1``) or next (``+1``) group.

    Numerical groups must stay contiguous, so the boundary between the two groups
    moves: joining the *next* group also moves every following fine bin of the
    group (from the first selected one to the group end); joining the *previous*
    group also moves every preceding one (from the group start to the last selected
    one). If the whole group moves, it simply merges with its neighbour.
    """
    _check_direction(direction)
    sel = _selection(cutoffs, splits, rows)
    target = sel.group + direction
    if not 0 <= target <= int(sel.labels.max()):
        raise BinningEditError(f"There is no {'next' if direction > 0 else 'previous'} group.")
    labels = sel.labels.copy()
    if direction > 0:
        labels[sel.first:sel.group_last + 1] = target
    else:
        labels[sel.group_first:sel.last + 1] = target
    return _splits_or_error(cutoffs, labels, splits, "Nothing changed.")


def split_group_at(cutoffs: Sequence[float], splits: Sequence[float],
                   rows: Iterable[int], direction: int) -> list[float]:
    """Start a new group at the selected fine bins without breaking the order.

    ``direction=+1``: the selected and all *following* fine bins of the group form a
    new, higher group (a cutoff at the lower edge of the selection).
    ``direction=-1``: the selected and all *preceding* fine bins of the group form a
    new, lower group (a cutoff at the upper edge of the selection).
    """
    _check_direction(direction)
    sel = _selection(cutoffs, splits, rows)
    labels = sel.labels.copy()
    if direction > 0:
        labels[sel.first:sel.group_last + 1] = -1
        unchanged = "The selection already starts its group."
    else:
        labels[sel.group_first:sel.last + 1] = -1
        unchanged = "The selection already ends its group."
    return _splits_or_error(cutoffs, labels, splits, unchanged)


def split_fine_bin(x: np.ndarray, lo: float, hi: float, value: float) -> float:
    """Validate a cutoff that splits the fine bin ``[lo, hi)`` in two.

    Splitting a fine bin refines the fine level only; the groups do not change.

    Returns
    -------
    float
        The validated cutoff.
    """
    value = float(value)
    if not lo < value < hi:
        raise BinningEditError(f"The value must lie inside the fine bin ({lo:g}, {hi:g}).")
    inside = x[(x >= lo) & (x < hi)]
    if not ((inside < value).any() and (inside >= value).any()):
        raise BinningEditError("Both parts of the fine bin must contain observations.")
    return value


def suggest_fine_split(x: np.ndarray, lo: float, hi: float) -> float | None:
    """Suggest a cutoff near the median of the fine bin ``[lo, hi)``.

    Returns ``None`` when the fine bin holds a single unique value.
    """
    unique = np.unique(x[(x >= lo) & (x < hi)])
    if len(unique) < 2:
        return None
    middle = len(unique) // 2
    return round_significant([(unique[middle - 1] + unique[middle]) / 2])[0]
