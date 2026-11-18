"""Shared fixtures."""

import matplotlib
import pytest

from interactive_grouping import BinningSession, make_credit_data

matplotlib.use("Agg")  # never open GUI windows during tests


@pytest.fixture(scope="session")
def data():
    """Synthetic data shared by all tests (read-only)."""
    return make_credit_data(n=8000, seed=1)


@pytest.fixture
def session(data):
    """A fresh session, so tests can edit it freely."""
    X, y = data
    return BinningSession.fit(X, y, special_codes=[-1])
