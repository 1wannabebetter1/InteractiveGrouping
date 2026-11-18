"""Headless UI tests: drive the widget by setting values and clicking buttons."""

import pytest

from interactive_grouping import BinningSession, InteractiveBinning, make_credit_data
from interactive_grouping import ui_components as ui
from interactive_grouping.strategies import NEW_GROUP, NEW_GROUP_BELOW


def _make(backend="plotly", **kwargs):
    X, y = make_credit_data(n=6000, seed=3)
    return InteractiveBinning(BinningSession.fit(X, y, special_codes=[-1]), backend=backend,
                              **kwargs)


@pytest.fixture(params=["plotly", "matplotlib"])
def widget(request):
    return _make(request.param)


@pytest.fixture
def advanced_widget():
    return _make(advanced=True)


def _ok(w):
    assert "ig-msg err" not in w.message.value, w.message.value


def _err(w, text=""):
    assert "ig-msg err" in w.message.value, w.message.value
    assert text in w.message.value, w.message.value


def _groups(w):
    return w.fine_level.regular["group"].to_numpy()


def _targets(w):
    return [value for _, value in w.assign_group.options]


def _assign(w, rows, target):
    w.rows_select.value = tuple(rows)
    w.assign_group.value = target
    w.btn_assign.click()


# --------------------------------------------------------------------------- layout
def test_table_options_are_aligned(widget):
    labels = [label for label, _ in ui.fine_options(widget.fine_level.table)]
    assert len({len(label) for label in labels}) == 1  # fixed width => aligned columns
    assert "Group" in labels[0]
    assert widget.rows_select.options[0][1] == ui.HEADER_VALUE


def test_starts_on_groupings_fine_view(widget):
    w = widget
    assert w.tabs.selected_index == 1
    assert w.detail == "Fine"
    assert len(w.rows_select.options) > w.var.current.n_bins + 1
    assert w.fig_fine.value
    assert w.fig_groups.value


def test_advanced_block_is_hidden_by_default(widget, advanced_widget):
    groupings_tab = widget.tabs.children[1]
    assert not widget.advanced
    assert widget.advanced_panel not in groupings_tab.children
    assert advanced_widget.advanced_panel in advanced_widget.tabs.children[1].children


def test_coarse_view_is_read_only(widget):
    w = widget
    w.detail_toggle.value = "Coarse"
    assert w.assign_tools.layout.display == "none"
    assert w.split_tools.layout.display == "none"
    assert w.coarse_note.layout.display == ""
    w.rows_select.value = (0, 1)
    assert not w.assign_group.options
    w.btn_assign.click()
    _err(w, "Fine")
    assert not w.has_draft


# --------------------------------------------------------------------------- numerical
def test_numerical_targets(widget):
    w = widget
    w.var_select.value = "age"
    assert w.btn_assign.disabled  # nothing selected
    groups = _groups(w)
    last = int(groups.max())
    first_of_last = int(w.fine_level.regular.index[groups == last][0])
    w.rows_select.value = (first_of_last,)
    expected = [last - 1, NEW_GROUP_BELOW] if (groups == last).sum() > 1 else [last - 1]
    assert _targets(w) == expected
    # an inner fine bin: both neighbours and a new group in both directions
    inner = next(i for i in range(1, len(groups) - 1)
                 if groups[i - 1] == groups[i] == groups[i + 1] and 0 < groups[i] < last)
    w.rows_select.value = (inner,)
    g = int(groups[inner])
    assert _targets(w) == [g - 1, g + 1, NEW_GROUP, NEW_GROUP_BELOW]


def test_join_next_group_takes_the_following_fine_bins(widget):
    w = widget
    w.var_select.value = "age"
    groups = _groups(w)
    # a group with >= 3 fine bins that has a next group; take its second fine bin
    g = next(int(k) for k in sorted(set(groups)) if (groups == k).sum() >= 3
             and k < groups.max())
    members = [int(i) for i in w.fine_level.regular.index[groups == g]]
    _assign(w, [members[1]], g + 1)
    _ok(w)
    new_groups = _groups(w)
    assert new_groups[members[0]] == g  # the first fine bin stays
    assert all(new_groups[i] == g + 1 for i in members[1:])  # the rest joins the next group


def test_join_previous_group_takes_the_preceding_fine_bins(widget):
    w = widget
    w.var_select.value = "age"
    groups = _groups(w)
    g = next(int(k) for k in sorted(set(groups)) if (groups == k).sum() >= 3 and k > 0)
    members = [int(i) for i in w.fine_level.regular.index[groups == g]]
    _assign(w, [members[-2]], g - 1)
    _ok(w)
    new_groups = _groups(w)
    assert new_groups[members[-1]] == g  # the last fine bin stays
    assert all(new_groups[i] == g - 1 for i in members[:-1])


def test_moving_a_whole_group_merges(widget):
    w = widget
    w.var_select.value = "age"
    n0 = w.var.current.n_bins
    groups = _groups(w)
    first_of_group_1 = int(w.fine_level.regular.index[groups == 1][0])
    _assign(w, [first_of_group_1], 0 if (groups == 1).sum() == 1 else 2)
    if (groups == 1).sum() > 1:  # the whole group must move: use the group start
        w.btn_discard.click()
        last_of_group_1 = int(w.fine_level.regular.index[groups == 1][-1])
        _assign(w, [last_of_group_1], 0)
    _ok(w)
    assert w.draft.n_bins == n0 - 1
    assert "merged" in w.message.value


def test_new_group_above_and_below_then_apply(widget):
    w = widget
    w.var_select.value = "age"
    n0 = w.var.current.n_bins
    groups = _groups(w)
    inner = next(i for i in range(1, len(groups) - 1)
                 if groups[i - 1] == groups[i] == groups[i + 1])
    _assign(w, [inner], NEW_GROUP)
    _ok(w)
    assert w.draft.n_bins == n0 + 1
    assert _groups(w)[inner] != _groups(w)[inner - 1]  # boundary at the lower edge
    _assign(w, [inner], NEW_GROUP_BELOW)  # and the upper edge -> a one-bin group
    _ok(w)
    assert w.draft.n_bins == n0 + 2
    w.btn_apply.click()
    _ok(w)
    assert w.var.is_edited
    assert "edited" in w.var_select.label
    w.btn_undo.click()
    assert not w.var.is_edited
    w.btn_redo.click()
    assert w.var.is_edited
    w.btn_reset.click()
    assert not w.var.is_edited


def test_split_fine_bin_keeps_the_groups(widget):
    w = widget
    w.var_select.value = "income"
    n_fine = len(w.fine_level.regular)
    splits = list(w.draft.splits)
    w.rows_select.value = (0,)
    assert w.split_value.value != 0  # median suggested
    w.btn_split_fine.click()
    _ok(w)
    assert len(w.fine_level.regular) == n_fine + 1
    assert w.draft.splits == splits
    assert not w.has_draft


def test_missing_rows_are_not_editable(widget):
    w = widget
    w.var_select.value = "income"  # has special and missing rows
    special = [v for _, v in w.rows_select.options
               if isinstance(v, str) and v.startswith(("missing", "special"))]
    assert special
    w.rows_select.value = (special[0],)
    assert w.btn_assign.disabled
    assert "Missing" in w.assign_hint.value
    w.btn_assign.click()
    _err(w, "Missing / special")


# --------------------------------------------------------------------------- categorical
def test_categorical_any_group_or_new(widget):
    w = widget
    w.var_select.value = "purpose"
    n_groups = len(w.var.splits)
    regular = w.fine_level.regular
    sizes = regular["group"].value_counts()
    row = int(regular.index[regular["group"] == sizes[sizes > 1].index[0]][0])
    category = w.fine_level.table.loc[row, "category"]
    w.rows_select.value = (row,)
    assert NEW_GROUP_BELOW not in _targets(w)
    assert len([t for t in _targets(w) if t != NEW_GROUP]) == n_groups - 1
    _assign(w, [row], NEW_GROUP)
    _ok(w)
    assert len(w.draft.splits) == n_groups + 1
    assert [category] in w.draft.splits
    # the singleton joins another group -> its group disappears (merge)
    table = w.fine_level.table
    row = int(table.index[table["category"] == category][0])
    target = next(int(g) for g in table["group"] if g >= 0
                  and category not in w.draft.splits[int(g)])
    _assign(w, [row], target)
    _ok(w)
    assert len(w.draft.splits) == n_groups
    w.btn_apply.click()
    _ok(w)


# --------------------------------------------------------------------------- other features
def test_drafts_survive_switching_and_variables_tab(widget):
    w = widget
    w.var_select.value = "age"
    groups = _groups(w)
    inner = next(i for i in range(1, len(groups) - 1)
                 if groups[i - 1] == groups[i] == groups[i + 1])
    _assign(w, [inner], NEW_GROUP)
    w.btn_next.click()
    w.btn_prev.click()
    assert w.var_select.value == "age"
    assert w.has_draft
    w.tabs.selected_index = 0
    assert "●" in w.summary_html.value


def test_rejected_role_is_excluded_from_export(widget):
    w = widget
    w.var_select.value = "region"
    w.role_select.value = "rejected"
    _ok(w)
    assert "rejected" in w.var_select.label
    assert "region" not in w.to_binning_process().variable_names
    assert "region" not in w.transform(w.session.X).columns


def test_reset_all(widget):
    w = widget
    for name in ("age", "purpose"):
        w.var_select.value = name
        groups = _groups(w)
        row = int(w.fine_level.regular.index[groups == 0][-1])  # last bin of group 1
        w.rows_select.value = (row,)
        _assign(w, [row], _targets(w)[0])
        w.btn_apply.click()
        _ok(w)
    assert w.session["age"].is_edited
    assert w.session["purpose"].is_edited
    w.btn_reset_all.click()
    assert not any(var.is_edited for var in w.session.variables.values())


def test_save_and_load(widget, tmp_path):
    w = widget
    w.path_text.value = str(tmp_path / "splits.json")
    w.btn_save.click()
    assert "Saved" in w.export_message.value
    w.btn_load.click()
    assert "Loaded" in w.export_message.value


def test_switching_variable_updates_figures(widget):
    """Regression: figures are widget traits, so a variable change must change them."""
    w = widget
    w.var_select.value = "age"
    fine_before, groups_before = w.fig_fine.value, w.fig_groups.value
    w.var_select.value = "income"
    assert w.fig_fine.value != fine_before
    assert w.fig_groups.value != groups_before
    if w.backend_name == "plotly":
        assert "<iframe" in w.fig_fine.value


# --------------------------------------------------------------------------- advanced
def test_advanced_cutoffs_fine_bins_and_optimizer(advanced_widget):
    w = advanced_widget
    w.var_select.value = "utilization"
    w.cutoffs_text.value = "0.2, 0.5, 0.8"
    w.btn_set_cutoffs.click()
    _ok(w)
    assert w.draft.splits == [0.2, 0.5, 0.8]
    w.cutoffs_text.value = "abc"
    w.btn_set_cutoffs.click()
    _err(w)
    n_before = len(w.fine_level.cutoffs)
    w.fine_bins_input.value = 40
    w.btn_fine_bins.click()
    _ok(w)
    assert len(w.fine_level.cutoffs) > n_before
    w.var_select.value = "income"
    w.opt_trend.value = "descending"
    w.opt_max_bins.value = 4
    w.btn_optimize.click()
    _ok(w)
    assert w.draft.n_bins <= 4
