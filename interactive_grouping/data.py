"""Synthetic credit-scoring data set for demos (no downloads needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_credit_data"]


def make_credit_data(n: int = 20_000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic credit-scoring data set.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Seed of the random generator (the data is reproducible).

    Returns
    -------
    X : pandas.DataFrame
        Six numerical and three categorical features. ``income`` contains missing
        values and the special code ``-1`` ("not provided"); ``months_employed`` and
        ``region`` contain missing values.
    y : pandas.Series
        Binary target ``default`` (1 = default). The age effect is U-shaped on
        purpose, so that the optimal monotonic binning is worth editing.
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.gamma(5, 5.5, n) + 18, 18, 85).round()
    income = np.round(np.exp(rng.normal(10.3, 0.6, n)), -2)
    months_employed = np.clip(rng.exponential(60, n), 0, 480).round()
    utilization = np.clip(rng.beta(2, 3, n) * 1.2, 0, 1.2).round(3)
    late_payments = rng.poisson(rng.choice([0.1, 1.5], n, p=[0.8, 0.2]))
    debt_to_income = np.clip(rng.normal(0.35, 0.15, n), 0, 1.5).round(3)

    home = rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n, p=[0.42, 0.18, 0.37, 0.03])
    purpose = rng.choice(
        ["car", "house", "education", "business", "medical", "vacation", "debt_consolidation",
         "wedding", "other"], n,
        p=[0.18, 0.15, 0.08, 0.07, 0.06, 0.05, 0.3, 0.02, 0.09])
    region = rng.choice(["North", "South", "East", "West", "Center"], n,
                        p=[0.2, 0.25, 0.15, 0.15, 0.25]).astype(object)

    # true log-odds: non-monotonic (U-shaped) age effect, etc.
    logit = (-2.2
             + 0.0016 * (age - 42) ** 2 - 0.012 * (age - 42)
             - 0.9 * (np.log(income) - 10.3)
             - 0.004 * months_employed
             + 2.0 * (utilization - 0.5)
             + 0.45 * np.minimum(late_payments, 5)
             + 1.5 * (debt_to_income - 0.35)
             + pd.Series(home).map({"RENT": 0.35, "OWN": -0.4, "MORTGAGE": -0.15,
                                    "OTHER": 0.5}).to_numpy()
             + pd.Series(purpose).map({"car": -0.1, "house": -0.3, "education": 0.2,
                                       "business": 0.45, "medical": 0.3, "vacation": 0.25,
                                       "debt_consolidation": 0.1, "wedding": 0.0,
                                       "other": 0.15}).to_numpy())
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # missing values and special codes
    income[rng.random(n) < 0.04] = -1          # special code: not provided
    income[rng.random(n) < 0.03] = np.nan
    months_employed[rng.random(n) < 0.05] = np.nan
    region[rng.random(n) < 0.06] = None

    X = pd.DataFrame({
        "age": age,
        "income": income,
        "months_employed": months_employed,
        "utilization": utilization,
        "late_payments": late_payments.astype(float),
        "debt_to_income": debt_to_income,
        "home_ownership": pd.Series(home, dtype=object),
        "purpose": pd.Series(purpose, dtype=object),
        "region": pd.Series(region, dtype=object),
    })
    return X, pd.Series(y, name="default")
