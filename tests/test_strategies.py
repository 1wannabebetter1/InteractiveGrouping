"""Tests of the variable-type strategies."""

import pandas as pd
import pytest

from interactive_grouping.exceptions import BinningEditError
from interactive_grouping.strategies import (
    NEW_GROUP,
    NEW_GROUP_BELOW,
    CategoricalStrategy,
    FineLevel,
    NumericalStrategy,
    get_strategy,
)


def test_registry():
    assert isinstance(get_strategy("numerical"), NumericalStrategy)
    assert isinstance(get_strategy("categorical"), CategoricalStrategy)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        get_strategy("multiclass")


def test_numerical_fine_targets_are_neighbours_and_new_groups():
    strategy = NumericalStrategy()
    fine = FineLevel(pd.DataFrame({"kind": ["regular"] * 6}), cutoffs=(1.0, 2.0, 3.0, 4.0, 5.0))
    splits = [2.0, 4.0]  # fine groups [0, 0, 1, 1, 2, 2]
    assert strategy.fine_targets(splits, fine, [2]) == [0, 2, NEW_GROUP_BELOW]  # group start
    assert strategy.fine_targets(splits, fine, [3]) == [0, 2, NEW_GROUP]  # group end
    assert strategy.fine_targets(splits, fine, [2, 3]) == [0, 2]  # the whole group
    assert strategy.fine_assign(splits, fine, [3], 2) == [2.0, 3.0]
    with pytest.raises(BinningEditError, match="neighbouring"):
        strategy.fine_assign([1.0, 2.0, 3.0, 4.0], fine, [0], 3)


def test_categorical_fine_targets_are_any_group():
    strategy = CategoricalStrategy()
    fine = FineLevel(pd.DataFrame({"kind": ["regular"] * 4, "category": list("abcd")}))
    groups = [["a", "b"], ["c"], ["d"]]
    assert strategy.fine_targets(groups, fine, [0]) == [1, 2, NEW_GROUP]
    assert strategy.fine_targets(groups, fine, [2]) == [0, 2]  # "c" is already alone


@pytest.mark.parametrize(("strategy", "a", "b", "equal"), [
    (NumericalStrategy(), [1.0, 2.0], [1.0, 2.0 + 1e-15], True),
    (NumericalStrategy(), [1.0], [1.0, 2.0], False),
    (CategoricalStrategy(), [["a", "b"], ["c"]], [["c"], ["b", "a"]], True),
    (CategoricalStrategy(), [["a"], ["b", "c"]], [["a", "b"], ["c"]], False),
])
def test_equal(strategy, a, b, equal):
    assert strategy.equal(a, b) is equal
