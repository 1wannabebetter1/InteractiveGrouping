"""Tests of the pure split operations."""

import numpy as np
import pytest

from interactive_grouping.exceptions import BinningEditError
from interactive_grouping.operations import (
    add_numerical_split,
    fine_cutoffs_numerical,
    fine_labels,
    merge_groups,
    merge_numerical,
    move_categories,
    move_to_neighbour,
    parse_cutoffs,
    quantile_splits_in_bin,
    split_fine_bin,
    split_group_at,
    splits_from_labels,
    suggest_fine_split,
)

CUTOFFS = [1.0, 2.0, 3.0, 4.0, 5.0]  # 6 fine bins


def _same_groups(a, b):
    return {frozenset(g) for g in a} == {frozenset(g) for g in b}


def test_merge_numerical():
    assert merge_numerical([1.0, 2.0, 3.0], [1, 2]) == [1.0, 3.0]
    assert merge_numerical([1.0, 2.0, 3.0], [0, 2]) == [3.0]  # whole range
    with pytest.raises(BinningEditError):
        merge_numerical([1.0, 2.0], [1])
    with pytest.raises(BinningEditError):
        merge_numerical([1.0], [0, 1])  # a single group would remain


def test_add_numerical_split():
    assert add_numerical_split([1.0, 3.0], 2.0, 0, 10) == [1.0, 2.0, 3.0]
    for bad in (3.0, 0.0, 11.0, float("nan")):
        with pytest.raises(BinningEditError):
            add_numerical_split([1.0, 3.0], bad, 0, 10)


def test_quantile_splits_in_bin():
    new = quantile_splits_in_bin([50.0], np.arange(100, dtype=float), 0, 2)
    assert len(new) == 2
    assert 0 < new[0] < 50


def test_parse_cutoffs():
    assert parse_cutoffs("3; 1, 2", (0, 10)) == [1.0, 2.0, 3.0]
    with pytest.raises(BinningEditError):
        parse_cutoffs("abc", (0, 10))
    with pytest.raises(BinningEditError, match="outside"):
        parse_cutoffs("0, 5", (0, 10))


def test_categorical_operations():
    groups = [["a", "b"], ["c"], ["d"]]
    assert _same_groups(merge_groups(groups, [1, 2]), [["a", "b"], ["c", "d"]])
    assert _same_groups(move_categories(groups, ["b"]), [["a"], ["c"], ["d"], ["b"]])
    assert _same_groups(move_categories(groups, ["b"], target=2), [["a"], ["c"], ["d", "b"]])
    with pytest.raises(BinningEditError):
        move_categories([["a"], ["b"]], ["a"], target=1)  # one group would remain
    with pytest.raises(BinningEditError):
        move_categories(groups, ["zzz"])
    with pytest.raises(BinningEditError, match="Nothing changed"):
        move_categories(groups, ["c"])


def test_fine_labels_roundtrip():
    labels = fine_labels(CUTOFFS, [2.0, 4.0])
    assert list(labels) == [0, 0, 1, 1, 2, 2]
    assert splits_from_labels(CUTOFFS, labels) == [2.0, 4.0]


# Fine bins 0..5 with SPLITS -> groups [0, 0, 1, 1, 2, 2].
SPLITS = [2.0, 4.0]


@pytest.mark.parametrize(("rows", "direction", "expected"), [
    ([3], +1, [2.0, 3.0]),   # last fine bin of group 1 joins group 2
    ([2], -1, [3.0, 4.0]),   # first fine bin of group 1 joins group 0
    ([2], +1, [2.0]),        # from the group start: the whole group joins -> merge
    ([3], -1, [4.0]),        # up to the group end: the whole group joins -> merge
])
def test_move_to_neighbour_takes_bins_up_to_the_group_edge(rows, direction, expected):
    assert move_to_neighbour(CUTOFFS, SPLITS, rows, direction) == expected


def test_move_to_neighbour_errors():
    with pytest.raises(BinningEditError, match="no previous"):
        move_to_neighbour(CUTOFFS, SPLITS, [0], -1)
    with pytest.raises(BinningEditError, match="no next"):
        move_to_neighbour(CUTOFFS, SPLITS, [5], +1)
    with pytest.raises(BinningEditError, match="one group"):
        move_to_neighbour(CUTOFFS, SPLITS, [1, 2], +1)
    with pytest.raises(BinningEditError, match="At least 2 groups"):
        move_to_neighbour([1.0], [1.0], [1], -1)  # the only other group disappears


def test_split_group_at():
    assert split_group_at(CUTOFFS, SPLITS, [3], +1) == [2.0, 3.0, 4.0]  # this and higher
    assert split_group_at(CUTOFFS, SPLITS, [2], -1) == [2.0, 3.0, 4.0]  # this and lower
    with pytest.raises(BinningEditError, match="starts its group"):
        split_group_at(CUTOFFS, SPLITS, [2], +1)
    with pytest.raises(BinningEditError, match="ends its group"):
        split_group_at(CUTOFFS, SPLITS, [3], -1)


def test_split_fine_bin():
    x = np.arange(10.0)
    assert split_fine_bin(x, 2.0, 6.0, 4.0) == 4.0
    with pytest.raises(BinningEditError, match="inside"):
        split_fine_bin(x, 2.0, 6.0, 7.0)
    with pytest.raises(BinningEditError, match="Both parts"):
        split_fine_bin(x, 2.0, 6.0, 5.5)  # no observation in [5.5, 6)
    assert suggest_fine_split(x, 2.0, 6.0) == 3.5
    assert suggest_fine_split(np.ones(5), 0.0, 2.0) is None


def test_fine_cutoffs_keep_required_and_have_no_empty_bins():
    x = np.repeat(np.arange(10.0), 50)  # integer data
    cutoffs = fine_cutoffs_numerical(x, 5, required=[2.5, 7.25])
    assert {2.5, 7.25} <= set(cutoffs)
    counts = np.bincount(np.searchsorted(cutoffs, x, side="right"), minlength=len(cutoffs) + 1)
    assert counts.min() > 0
