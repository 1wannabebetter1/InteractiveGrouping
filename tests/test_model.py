"""Tests of the model: VariableBinning, BinningSession, History and metrics."""

import numpy as np
import pytest

from interactive_grouping import BinningEditError, BinningSession, VariableRole
from interactive_grouping.history import History
from interactive_grouping.metrics import detect_trend, iv_strength
from interactive_grouping.operations import merge_groups, merge_numerical


def test_history():
    history = History(0)
    assert not history.undo()
    history.push(1)
    history.push(2)
    assert history.undo()
    assert history.current == 1
    assert history.redo()
    assert history.current == 2
    history.undo()
    history.push(3)  # a new change discards the redo branch
    assert not history.can_redo
    assert history.current == 3


@pytest.mark.parametrize(("rates", "trend"), [
    ([1, 2, 3], "ascending"), ([3, 2, 1], "descending"), ([1, 3, 2], "peak"),
    ([3, 1, 2], "valley"), ([1], "single bin"), ([1, 1], "flat"),
])
def test_detect_trend(rates, trend):
    assert detect_trend(rates) == trend


def test_iv_strength():
    assert [iv_strength(v) for v in (0.01, 0.05, 0.2, 0.4, 0.8)] == [
        "useless", "weak", "medium", "strong", "suspicious"]


def test_fit_splits_is_exact(session):
    var = session["age"]
    result = var.fit_splits([30.0, 45.0, 60.0])
    assert result.splits == [30.0, 45.0, 60.0]
    assert result.n_bins == 4
    assert result.table["count"].sum() == len(var.x)


def test_commit_undo_redo_reset(session):
    var = session["age"]
    original = list(var.splits)
    var.commit(var.fit_splits(merge_numerical(var.splits, [0, 1])))
    assert var.is_edited
    assert var.can_undo
    assert var.undo()
    assert var.splits == original
    assert not var.is_edited
    assert var.redo()
    assert var.is_edited
    assert var.reset()
    assert not var.is_edited
    assert var.can_undo
    assert not var.reset()  # already the original grouping


def test_categorical_refit(session):
    var = session["purpose"]
    new = merge_groups(var.splits, [0, 1])
    assert var.strategy.equal(var.fit_splits(new).splits, new)


def test_invalid_groupings_are_rejected(session):
    var = session["age"]
    low, _ = var.x_range
    with pytest.raises(BinningEditError, match="pure"):
        var.fit_splits([low + 0.5, 50.0])  # tiny first group without events
    with pytest.raises(BinningEditError):
        session["purpose"].fit_splits([])


def test_optimize_only_merges(session):
    var = session["income"]
    result = var.optimize(start_splits=var.splits, monotonic_trend="descending", max_n_bins=4)
    assert result.n_bins <= 4
    assert set(result.splits) <= set(var.splits)


def test_fine_level_is_consistent_with_groups(session):
    var = session["age"]
    fine = var.fine_level(var.splits)
    assert set(var.splits) <= set(fine.cutoffs)
    by_group = fine.regular.groupby("group")["count"].sum().to_numpy()
    np.testing.assert_array_equal(by_group, var.current.regular["count"].to_numpy())
    assert fine.table["count"].sum() == len(var.x)

    cat = session["purpose"]
    categories = set(cat.fine_level(cat.splits).regular["category"])
    assert categories == {c for group in cat.splits for c in group}


def test_export_roundtrip(session, data, tmp_path):
    X, y = data
    session["age"].set_splits([35.0, 50.0, 65.0])
    home = session["home_ownership"]
    home.set_splits(merge_groups(home.splits, [0, 1]))

    process = session.to_binning_process()
    for name in session:
        var = session[name]
        fitted = var.strategy.normalize(process.get_binned_variable(name).splits)
        assert var.strategy.equal(fitted, var.splits), name
    np.testing.assert_allclose(session.transform(X).to_numpy(float),
                               process.transform(X, metric="woe").to_numpy(float))

    path = tmp_path / "splits.json"
    session.save_json(str(path))
    restored = BinningSession.fit(X, y, special_codes=[-1])
    assert restored.load_json(str(path)) == []
    assert restored["age"].splits == [35.0, 50.0, 65.0]
    assert restored["home_ownership"].is_edited


def test_roles_and_summary(session):
    session.set_role("region", VariableRole.REJECTED)
    assert "region" not in session.to_binning_process().variable_names
    assert "region" in session.to_binning_process(include_rejected=True).variable_names
    summary = session.summary().set_index("name")
    assert summary.loc["region", "role"] == VariableRole.REJECTED
    assert {"iv", "gini", "n_bins", "edited", "original_iv"} <= set(summary.columns)
    with pytest.raises(ValueError, match="role must be one of"):
        session.set_role("region", "unknown")
